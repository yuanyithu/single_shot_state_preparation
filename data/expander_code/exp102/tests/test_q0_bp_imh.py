"""Exactness and replay tests for BP-systematic independence MH."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_bp_imh import (
    BpImhError,
    acceptance_decision,
    combine_bp_proposals,
    log_acceptance_ratio,
    replay_bp_imh_trajectory,
    run_bp_imh_trajectory,
    validate_bp_imh_transcript,
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


def _proposal(model, syndrome, p, order):
    return build_bp_systematic_proposal(
        model, syndrome, p, column_order=np.asarray(order, dtype=np.int32),
        bp_iterations=8, bp_damping=0.25, bp_llr_cap=30.0,
        min_probability=1e-5, component_weights=(0.90, 0.09, 0.01),
    )


def test_combined_proposal_normalization_and_complete_imh_stationarity():
    H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    model = _Model(H_check)
    p = 0.20
    forward = _proposal(model, syndrome, p, [0, 1, 2])
    reverse = _proposal(model, syndrome, p, [2, 1, 0])
    proposal = combine_bp_proposals((forward, reverse), (0.5, 0.5))
    states = _coset_states(H_check, syndrome)
    q = np.asarray([math.exp(proposal.log_probability_state(state)) for state in states])
    assert q.sum() == pytest.approx(1.0, abs=2e-14)
    target = (p / (1.0 - p)) ** states.sum(axis=1)
    posterior = target / target.sum()
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    for source, state in enumerate(states):
        current_log_q = proposal.log_probability_state(state)
        for destination, proposed in enumerate(states):
            ratio = log_acceptance_ratio(
                int(state.sum()), current_log_q, int(proposed.sum()),
                proposal.log_probability_state(proposed), p,
            )
            probability = min(1.0, math.exp(min(0.0, ratio)))
            if source == destination:
                transition[source, source] += q[destination]
            else:
                transition[source, destination] += q[destination] * probability
                transition[source, source] += q[destination] * (1.0 - probability)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-14
    flow = posterior[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 2e-14
    assert np.max(np.abs(posterior @ transition - posterior)) <= 2e-14


def test_acceptance_decision_keeps_self_loop_probability_explicit():
    accepted, clipped = acceptance_decision(-math.log(4.0), 0.24)
    assert accepted and clipped == pytest.approx(-math.log(4.0))
    accepted, clipped = acceptance_decision(-math.log(4.0), 0.25)
    assert not accepted and clipped == pytest.approx(-math.log(4.0))
    assert acceptance_decision(10.0, 0.999999)[0]


def test_small_hgp_trajectory_seed_replay_and_raw_transition_audit():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    initial[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ initial.astype(np.int64) % 2
    ).astype(np.uint8)
    p = 0.20
    forward = _proposal(model, syndrome, p, np.arange(model.num_qubits))
    reverse = _proposal(model, syndrome, p, np.arange(model.num_qubits - 1, -1, -1))
    proposal = combine_bp_proposals((forward, reverse), (0.5, 0.5))
    raw = run_bp_imh_trajectory(
        model, frame, syndrome, p, proposal, initial, 123456789,
        burn_steps=16, measurement_steps=64,
    )
    assert replay_bp_imh_trajectory(
        model, frame, syndrome, p, proposal, initial, 123456789, raw,
        burn_steps=16, measurement_steps=64,
    )
    assert validate_bp_imh_transcript(
        model, frame, syndrome, p, proposal, raw,
        burn_steps=16, measurement_steps=64,
    )
    states = np.unpackbits(
        raw["measurement_states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    )
    residual = model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    assert np.array_equal(residual.T.astype(np.uint8), np.repeat(
        syndrome[None, :], states.shape[0], axis=0,
    ))


def test_raw_audit_rejects_changed_acceptance_decision():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    p = 0.20
    forward = _proposal(model, syndrome, p, np.arange(model.num_qubits))
    reverse = _proposal(model, syndrome, p, np.arange(model.num_qubits - 1, -1, -1))
    proposal = combine_bp_proposals((forward, reverse), (0.5, 0.5))
    raw = run_bp_imh_trajectory(
        model, frame, syndrome, p, proposal, initial, 987654321,
        burn_steps=8, measurement_steps=16,
    )
    tampered = {key: np.array(value, copy=True) for key, value in raw.items()}
    tampered["measurement_accepted"][0] ^= np.uint8(1)
    with pytest.raises(BpImhError, match="transcript changed"):
        validate_bp_imh_transcript(
            model, frame, syndrome, p, proposal, tampered,
            burn_steps=8, measurement_steps=16,
        )
