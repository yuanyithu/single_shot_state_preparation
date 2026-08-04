"""Exact-oracle tests for auxiliary A-row stabilizer heatbaths."""

import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer import (
    auxiliary_stabilizer_conditional_probabilities,
    auxiliary_stabilizer_delta,
    auxiliary_stabilizer_row_heatbath,
    auxiliary_stabilizer_sweep,
    brute_force_auxiliary_stabilizer_conditional,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    hgp_syndrome_matrix,
    join_hgp_state,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


class _DeterministicRng:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, generator in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generator
    packed = np.packbits(states, axis=1, bitorder="little")
    assert len({row.tobytes() for row in packed}) == states.shape[0]
    return states


def _posterior(states, p):
    weights = (p / (1.0 - p)) ** states.sum(axis=1)
    return weights / weights.sum(dtype=np.float64)


def _syndrome(model, nonzero):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_auxiliary_conditionals_match_brute_force(H, p, nonzero_syndrome):
    model, _ = build_model(H)
    syndrome = _syndrome(model, nonzero_syndrome)
    for state in _hard_coset_states(model, syndrome):
        A, B = split_hgp_state(state, H)
        for a_row in range(H.shape[1]):
            eliminated, _ = auxiliary_stabilizer_conditional_probabilities(H, A, B, a_row, p)
            brute, _ = brute_force_auxiliary_stabilizer_conditional(H, A, B, a_row, p)
            assert np.max(np.abs(eliminated - brute)) <= 4e-14


@pytest.mark.parametrize("H", SMALL_H)
def test_auxiliary_deltas_are_hard_coset_stabilizers(H):
    model, frame = build_model(H)
    rows, columns = H.shape
    for a_row in range(columns):
        for assignment in range(1 << rows):
            delta_a, delta_b = auxiliary_stabilizer_delta(H, a_row, assignment)
            assert not hgp_syndrome_matrix(delta_a, delta_b, H).any()
            delta = join_hgp_state(delta_a, delta_b)
            assert not (model.H_check.astype(np.int64) @ delta.astype(np.int64) % 2).any()
            assert not (frame.W_basis.astype(np.int64) @ delta.astype(np.int64) % 2).any()


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_auxiliary_blocks_have_detailed_balance_and_sweep_stationarity(H, p, nonzero_syndrome):
    model, _ = build_model(H)
    syndrome = _syndrome(model, nonzero_syndrome)
    states = _hard_coset_states(model, syndrome)
    posterior = _posterior(states, p)
    packed_index = {np.packbits(state, bitorder="little").tobytes(): index for index, state in enumerate(states)}
    transitions = []
    for a_row in range(H.shape[1]):
        transition = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
        for source, state in enumerate(states):
            A, B = split_hgp_state(state, H)
            probabilities, _ = auxiliary_stabilizer_conditional_probabilities(H, A, B, a_row, p)
            for assignment, probability in enumerate(probabilities):
                delta_a, delta_b = auxiliary_stabilizer_delta(H, a_row, assignment)
                target = join_hgp_state(A ^ delta_a, B ^ delta_b)
                index = packed_index[np.packbits(target, bitorder="little").tobytes()]
                transition[source, index] += probability
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 5e-15
        flow = posterior[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 8e-14
        transitions.append(transition)
    sweep = transitions[0]
    for transition in transitions[1:]:
        sweep = sweep @ transition
    assert np.max(np.abs(posterior @ sweep - posterior)) <= 1e-13


def test_auxiliary_heatbath_and_sweep_are_replayable_and_preserve_syndrome():
    H = SMALL_H[1]
    model, _ = build_model(H)
    syndrome = _syndrome(model, True)
    initial = _hard_coset_states(model, syndrome)[7]
    A, B = split_hgp_state(initial, H)
    first = auxiliary_stabilizer_row_heatbath(H, A, B, 1, 0.10, _DeterministicRng([0.2] * 8))
    second = auxiliary_stabilizer_row_heatbath(H, A, B, 1, 0.10, _DeterministicRng([0.2] * 8))
    for left, right in zip(first, second):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    final_a, final_b, assignments = auxiliary_stabilizer_sweep(
        H, A, B, 0.10, _DeterministicRng([0.3] * 32), row_order=(2, 0, 1),
    )
    assert assignments.shape == (3,)
    final = join_hgp_state(final_a, final_b)
    residual = (
        model.H_check.astype(np.int64) @ final.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    assert not residual.any()
