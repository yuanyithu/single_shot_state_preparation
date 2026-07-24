"""Exact algebra and transition tests for truth-free dressed logical XOR moves."""

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_center_preserving import (
    _select_rank_first,
    choose_dressed_candidates,
    xor_log_acceptance,
    xor_metropolis_step,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import state_label
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, generator in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generator
    return states


def test_dressed_candidate_rule_uses_lower_weight_then_packed_tie_break():
    base = np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    signatures = np.asarray([1, 2, 3], dtype=np.uint64)
    code_moves = np.packbits(np.asarray([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8), axis=1, bitorder="little")
    decoded_states = np.asarray([
        [0, 1, 0, 0, 0, 0, 0, 0],  # lower than base xor code move
        [0, 0, 1, 1, 0, 0, 0, 0],  # equal weight, smaller packed bytes
        [1, 1, 0, 0, 0, 0, 0, 0],  # invalid and ignored
    ], dtype=np.uint8)
    decoded = np.packbits(decoded_states, axis=1, bitorder="little")
    selected = choose_dressed_candidates(
        base, signatures, code_moves, np.asarray([1, 1, 0], dtype=np.uint8),
        np.asarray([1, 2, -1], dtype=np.int32), decoded,
        logical_dimension=2, max_moves=3,
    )
    by_index = {
        int(index): int(source)
        for index, source in zip(selected["selected_indices"], selected["source_kind"])
    }
    assert by_index == {0: 1, 1: 1, 2: 0}
    assert np.array_equal(selected["candidate_source_counts"], [1, 2])


def test_rank_first_selection_does_not_confuse_low_weight_with_full_span():
    signatures = np.asarray([1, 3, 2, 4], dtype=np.uint64)
    weights = np.asarray([1, 1, 100, 5], dtype=np.int32)
    packed = np.asarray([[1], [2], [3], [4]], dtype=np.uint8)
    selected, roles = _select_rank_first(
        signatures, weights, weights, packed, logical_dimension=3, max_moves=3,
    )
    assert set(signatures[selected].tolist()) == {1, 3, 4}
    assert np.array_equal(roles, np.ones(3, dtype=np.uint8))


@pytest.mark.parametrize("classical", [
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_symmetric_logical_xor_transition_has_exact_stationarity_and_balance(
        classical, nonzero, p):
    model, frame = build_model(classical)
    error = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        error[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ error.astype(np.int64) % 2
    ).astype(np.uint8)
    states = _hard_coset_states(model, syndrome)
    packed_to_index = {
        np.packbits(state, bitorder="little").tobytes(): index
        for index, state in enumerate(states)
    }
    moves = np.asarray(model.logical_move_basis, dtype=np.uint8)
    transition = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
    for source, state in enumerate(states):
        for move in moves:
            proposal = state ^ move
            target = packed_to_index[np.packbits(proposal, bitorder="little").tobytes()]
            log_ratio = xor_log_acceptance(int(state.sum()), int(proposal.sum()), p)
            acceptance = min(1.0, float(np.exp(log_ratio)))
            transition[source, target] += acceptance / moves.shape[0]
            transition[source, source] += (1.0 - acceptance) / moves.shape[0]
    target = (p / (1.0 - p)) ** states.sum(axis=1)
    target /= target.sum()
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-15
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 1e-13
    assert np.max(np.abs(target @ transition - target)) <= 1e-13

    state, accepted, log_ratio = xor_metropolis_step(
        states[0], moves[0], p, np.nextafter(1.0, 0.0),
    )
    assert log_ratio == xor_log_acceptance(
        int(states[0].sum()), int((states[0] ^ moves[0]).sum()), p,
    )
    assert np.array_equal(state, (states[0] ^ moves[0]) if accepted else states[0])


def test_uint64_bit63_signature_is_not_converted_through_int64():
    signatures = np.asarray([
        np.uint64(1), np.uint64(1) << np.uint64(63),
    ], dtype=np.uint64)
    weights = np.asarray([1, 2], dtype=np.int32)
    packed = np.asarray([[1], [2]], dtype=np.uint8)
    selected, _ = _select_rank_first(
        signatures, weights, weights, packed, logical_dimension=2, max_moves=2,
    )
    assert set(int(value) for value in signatures[selected]) == {1, 1 << 63}


def test_xor_step_rejects_invalid_uniform_and_preserves_input_on_rejection():
    state = np.asarray([0, 0, 0, 0], dtype=np.uint8)
    move = np.asarray([1, 1, 1, 1], dtype=np.uint8)
    original = state.copy()
    result, accepted, _ = xor_metropolis_step(state, move, 0.04, 0.999)
    assert not accepted
    assert np.array_equal(result, original)
    assert np.array_equal(state, original)
    with pytest.raises(ValueError, match="uniform"):
        xor_metropolis_step(state, move, 0.04, 1.0)
