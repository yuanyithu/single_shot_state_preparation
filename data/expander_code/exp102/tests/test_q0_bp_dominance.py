"""Exact small-HGP checks for conservative BP-dominance witnesses."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_bp_dominance import (
    canonical_rank_complete_logical_witnesses,
    deterministic_witness_panel,
    posterior_to_proposal_lower,
    proposal_probability_upper,
)
from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


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


def _proposal(H_check, syndrome, order):
    model = _Model(H_check)
    return build_bp_systematic_proposal(
        model, syndrome, 0.20, column_order=np.asarray(order, dtype=np.int32),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )


def test_witness_lower_bound_is_below_exact_posterior_to_proposal_ratio():
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    p = 0.20
    proposal = _proposal(H_check, syndrome, [0, 1, 2])
    states = _coset_states(H_check, syndrome)
    odds = p / (1.0 - p)
    target = odds ** states.sum(axis=1)
    normalizer = float(target.sum())
    q = np.asarray([math.exp(proposal.log_probability_state(state)) for state in states])
    assert q.sum() == pytest.approx(1.0, abs=2e-14)
    assert normalizer * (1.0 - p) ** H_check.shape[1] <= 1.0 + 2e-15
    for state, target_weight, density in zip(states, target, q, strict=True):
        lower, q_upper = posterior_to_proposal_lower(state, proposal, p)
        exact_ratio = float(target_weight / normalizer / density)
        assert float(q_upper) >= density * (1.0 - 2e-13)
        assert float(lower) <= exact_ratio * (1.0 + 2e-12)


def test_outward_probability_bound_is_above_direct_density():
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    proposal = _proposal(H_check, syndrome, [2, 1, 0])
    for state in _coset_states(H_check, syndrome):
        coordinate = proposal.coordinates.coordinates_of_state(state)
        upper = proposal_probability_upper(coordinate, proposal)
        direct = math.exp(proposal.log_probability_state(state))
        assert float(upper) >= direct * (1.0 - 2e-13)


def _mask_rank(values):
    pivots = {}
    for raw in values:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = value
                break
            value ^= previous
    return len(pivots)


def test_canonical_witnesses_span_logicals_and_panel_stays_in_hard_coset():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    logical = canonical_rank_complete_logical_witnesses(
        model, frame, planted, candidate_orders=(1, 2, 3),
    )
    assert len(logical) == model.k
    assert _mask_rank([record["signature"] for record in logical]) == model.k
    for record in logical:
        residual = (model.H_check.astype(np.int64) @ record["state"].astype(np.int64) % 2)
        assert not residual.any()

    forward = build_bp_systematic_proposal(
        model, syndrome, 0.20, column_order=np.arange(model.num_qubits, dtype=np.int32),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )
    reverse = build_bp_systematic_proposal(
        model, syndrome, 0.20,
        column_order=np.arange(model.num_qubits - 1, -1, -1, dtype=np.int32),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )
    panel = deterministic_witness_panel(
        model, frame, planted, {"forward": forward, "reverse": reverse},
        candidate_orders=(1, 2, 3),
    )
    assert panel
    assert any("planted" in record["origins"] for record in panel)
    assert len(panel) <= 1 + model.k + 2 * (model.num_qubits - model.num_checks)
    for record in panel:
        residual = (model.H_check.astype(np.int64) @ record["state"].astype(np.int64) % 2)
        assert not residual.any()
