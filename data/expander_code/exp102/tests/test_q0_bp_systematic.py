"""Algebra and proposal-density tests for BP-guided systematic IID proposals."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    BpSystematicError,
    build_bp_systematic_proposal,
    build_systematic_hard_coset_coordinates,
    xor_sum_product_marginals,
)


class _Model:
    def __init__(self, H_check):
        self.H_check = np.asarray(H_check, dtype=np.uint8)
        self.num_qubits = int(self.H_check.shape[1])


def _coset_states(H_check, syndrome):
    H_check = np.asarray(H_check, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    return np.asarray([
        state for state in itertools.product((0, 1), repeat=H_check.shape[1])
        if np.array_equal(
            (H_check.astype(np.int64) @ np.asarray(state, dtype=np.int64) % 2).astype(np.uint8),
            syndrome,
        )
    ], dtype=np.uint8)


def test_xor_sum_product_matches_exact_marginals_on_a_tree_check():
    H_check = np.asarray([[1, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1], dtype=np.uint8)
    p = 0.20
    diagnostics = xor_sum_product_marginals(
        H_check, syndrome, p, iterations=4, damping=0.0, llr_cap=30.0,
    )
    states = _coset_states(H_check, syndrome)
    weights = np.asarray([p ** int(state.sum()) * (1.0 - p) ** (3 - int(state.sum()))
                          for state in states])
    exact = (weights[:, None] * states).sum(axis=0) / weights.sum()
    assert diagnostics.final_max_message_delta == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(diagnostics.marginal_probability_one, exact, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("column_order", [np.asarray([0, 1, 2]), np.asarray([2, 1, 0])])
def test_systematic_coordinates_are_a_hard_coset_bijection(column_order):
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    coordinates = build_systematic_hard_coset_coordinates(
        H_check, syndrome, column_order=column_order,
    )
    expected = _coset_states(H_check, syndrome)
    actual = np.asarray([
        coordinates.state_from_coordinates(np.asarray(bits, dtype=np.uint8))
        for bits in itertools.product((0, 1), repeat=coordinates.dimension)
    ], dtype=np.uint8)
    assert {
        tuple(row) for row in actual
    } == {
        tuple(row) for row in expected
    }
    for state in actual:
        coordinate = coordinates.coordinates_of_state(state)
        assert np.array_equal(coordinates.state_from_coordinates(coordinate), state)
        assert np.array_equal(coordinate, state[coordinates.free_columns])


def test_bp_systematic_proposal_is_normalized_and_supports_exact_importance_identity():
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    model = _Model(H_check)
    p = 0.20
    proposal = build_bp_systematic_proposal(
        model, syndrome, p, column_order=np.asarray([0, 1, 2]),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )
    states = _coset_states(H_check, syndrome)
    q = np.asarray([math.exp(proposal.log_probability_state(state)) for state in states])
    assert q.sum() == pytest.approx(1.0, abs=2e-14)
    assert np.all(q > 0.0)
    target = np.asarray([(p / (1.0 - p)) ** int(state.sum()) for state in states])
    # Exact proposal normalization makes the one-copy importance identity exact.
    assert float(np.dot(q, target / q)) == pytest.approx(float(target.sum()), abs=2e-13)
    for state in states:
        coordinate = proposal.coordinates.coordinates_of_state(state)
        assert proposal.log_probability_state(state) == pytest.approx(
            proposal.log_probability_coordinates(coordinate), abs=1e-14,
        )


def test_bp_systematic_draw_replays_its_state_and_density():
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    model = _Model(H_check)
    proposal = build_bp_systematic_proposal(
        model, syndrome, 0.20, column_order=np.asarray([2, 1, 0]),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )
    draw = proposal.sample(np.random.default_rng(1234))
    assert 0 <= draw["component_index"] < proposal.num_components
    assert np.array_equal(
        proposal.coordinates.state_from_coordinates(draw["coordinate"]), draw["state"],
    )
    assert draw["log_q"] == pytest.approx(proposal.log_probability_state(draw["state"]), abs=1e-14)
    residual = (H_check.astype(np.int64) @ draw["state"].astype(np.int64) % 2).astype(np.uint8)
    assert np.array_equal(residual, syndrome)


def test_systematic_coordinates_reject_nonpermutation_column_order():
    with pytest.raises(BpSystematicError, match="permutation"):
        build_systematic_hard_coset_coordinates(
            np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
            np.asarray([1, 0], dtype=np.uint8), column_order=np.asarray([0, 0, 2]),
        )
