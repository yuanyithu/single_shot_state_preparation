"""Exact checks for the uniform-anchored collapsed-B replica transition."""

import math

import numpy as np

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    build_full_row_elimination_plan,
    full_row_conditional_probabilities,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_uniform_anchor_pt import (
    UniformAnchorReplicaExchangeConfig,
    UniformAnchorReplicaExchangeSeedIdentity,
    collapsed_complete_score,
    full_energy_swap_log_acceptance,
    run_uniform_anchor_replica_exchange_trajectory,
    uniform_anchor_lambda_schedule,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)


def _all_b_columns(rank):
    values = np.zeros((1 << (rank * rank), rank), dtype=np.uint32)
    for integer in range(values.shape[0]):
        for column in range(rank):
            for row in range(rank):
                if (integer >> (column * rank + row)) & 1:
                    values[integer, column] |= np.uint32(1 << row)
    return values


def _b_index(columns, rank):
    result = 0
    for column, value in enumerate(columns):
        for row in range(rank):
            result |= ((int(value) >> row) & 1) << (column * rank + row)
    return result


def _a_syndromes(syndrome, H, b_columns):
    rows, columns = H.shape
    syndrome_matrix = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    values = np.asarray(
        [_bits_to_mask(syndrome_matrix[:, column]) for column in range(columns)],
        dtype=np.uint32,
    )
    for column in range(columns):
        for variable in np.flatnonzero(H[:, column]):
            values[column] ^= b_columns[int(variable)]
    return values


def _replace_b_row(columns, row, assignment):
    result = np.asarray(columns, dtype=np.uint32).copy()
    mask = np.uint32(1 << row)
    clear = np.uint32(~int(mask) & 0xFFFFFFFF)
    for variable in range(result.size):
        result[variable] &= clear
        if (assignment >> variable) & 1:
            result[variable] |= mask
    return result


def _collapsed_data(H, syndrome, p):
    rows = int(H.shape[0])
    b_states = _all_b_columns(rows)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    a_states = [_a_syndromes(syndrome, H, b_columns) for b_columns in b_states]
    scores = np.asarray([
        collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds)
        for b_columns, a_syndromes in zip(b_states, a_states)
    ], dtype=np.float64)
    return b_states, a_states, log_mass, log_odds, scores


def _normalized_power(scores, power):
    values = np.exp(float(power) * (scores - scores.max()))
    return values / values.sum(dtype=np.float64)


def _row_kernel(H, b_states, a_states, log_mass, log_odds, row):
    plan = build_full_row_elimination_plan(H)
    transition = np.zeros((b_states.shape[0], b_states.shape[0]), dtype=np.float64)
    for source, (b_columns, a_syndromes) in enumerate(zip(b_states, a_states)):
        probabilities, _ = full_row_conditional_probabilities(
            H, plan, b_columns, a_syndromes, row, log_mass, log_odds,
        )
        for assignment, probability in enumerate(probabilities):
            target = _b_index(_replace_b_row(b_columns, row, assignment), H.shape[0])
            transition[source, target] += probability
    return transition


def test_uniform_anchor_ladder_has_exact_endpoints_and_frozen_shape():
    values = uniform_anchor_lambda_schedule(8)
    assert values.dtype == np.float64
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(values[1:] > values[:-1])
    assert np.allclose(values, 1.0 - values[::-1], rtol=0.0, atol=2e-16)
    config = UniformAnchorReplicaExchangeConfig(
        p=0.04, burn_rounds=16, measurement_rounds=32, num_replicas=8,
    )
    assert config.method_id == "UARE08-R1"
    assert config.as_dict()["hot_endpoint"] == "exact_uniform_B_refresh"
    assert config.as_dict()["tempered_term"] == "complete_collapsed_log_density"


def test_complete_energy_swap_satisfies_pairwise_detailed_balance():
    H = SMALL_H
    model, _ = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _, _, _, _, scores = _collapsed_data(H, syndrome, 0.10)
    lower, upper = 0.23, 0.79
    pi_lower = _normalized_power(scores, lower)
    pi_upper = _normalized_power(scores, upper)
    for left in range(scores.size):
        for right in range(scores.size):
            forward_log = full_energy_swap_log_acceptance(
                lower, upper, scores[left], scores[right],
            )
            reverse_log = full_energy_swap_log_acceptance(
                lower, upper, scores[right], scores[left],
            )
            forward = min(1.0, math.exp(forward_log))
            reverse = min(1.0, math.exp(reverse_log))
            assert abs(
                pi_lower[left] * pi_upper[right] * forward
                - pi_lower[right] * pi_upper[left] * reverse
            ) <= 2e-14


def test_two_rung_composite_transition_preserves_product_target():
    """Enumerate the r=2 two-replica chain, including refresh and swap."""
    H = SMALL_H
    model, _ = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    b_states, a_states, log_mass, log_odds, scores = _collapsed_data(H, syndrome, 0.10)
    states = b_states.shape[0]
    cold = _normalized_power(scores, 1.0)
    hot = np.full(states, 1.0 / states, dtype=np.float64)
    # One exact row heatbath is enough for invariance even though it is not a
    # complete cold sweep; that is exactly the state-independent row clock
    # used by UARE at a positive rung.
    cold_row = _row_kernel(H, b_states, a_states, log_mass, log_odds, row=0)
    assert np.max(np.abs(cold @ cold_row - cold)) <= 7e-14
    local = np.zeros((states * states, states * states), dtype=np.float64)
    for hot_source in range(states):
        for cold_source in range(states):
            source = hot_source * states + cold_source
            for hot_target in range(states):
                for cold_target in range(states):
                    target = hot_target * states + cold_target
                    local[source, target] = hot[hot_target] * cold_row[cold_source, cold_target]
    swap = np.zeros_like(local)
    for hot_source in range(states):
        for cold_source in range(states):
            source = hot_source * states + cold_source
            log_acceptance = full_energy_swap_log_acceptance(
                0.0, 1.0, scores[hot_source], scores[cold_source],
            )
            acceptance = min(1.0, math.exp(log_acceptance))
            unchanged = source
            exchanged = cold_source * states + hot_source
            swap[source, unchanged] += 1.0 - acceptance
            swap[source, exchanged] += acceptance
    transition = local @ swap
    product = np.outer(hot, cold).reshape(-1)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 3e-15
    assert np.max(np.abs(product @ transition - product)) <= 9e-14


def test_reference_trajectory_replays_hard_coset_and_complete_score_trace():
    H = SMALL_H
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = UniformAnchorReplicaExchangeConfig(
        p=0.10, burn_rounds=8, measurement_rounds=16, num_replicas=4,
    )
    seed = UniformAnchorReplicaExchangeSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        resource_tier="test",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace="exp102.q0_hgp_uniform_anchor_pt.test",
    )
    raw = run_uniform_anchor_replica_exchange_trajectory(
        model, frame, H, syndrome, config, seed, epsilon,
    )
    unpacked = np.unpackbits(
        raw["measurement_states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    residual = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    mass = build_classical_coset_mass(H, config.p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(config.p / (1.0 - config.p))
    expected_scores = np.asarray([
        collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds)
        for b_columns, a_syndromes in zip(
            raw["measurement_b_columns"], raw["measurement_a_syndromes"],
        )
    ])
    assert not residual.any()
    assert raw["raw_version"] == "exp102.q0_hgp_uniform_anchor_pt.raw.v0"
    assert raw["lambda_values"][0] == 0.0
    assert raw["lambda_values"][-1] == 1.0
    assert np.array_equal(raw["measurement_residual_weights"], np.zeros(16, dtype=np.int32))
    assert np.max(np.abs(raw["measurement_complete_scores"] - expected_scores)) <= 2e-13
    assert raw["measurement_hot_refresh_changed_bits"].shape == (16,)
    assert raw["measurement_row_counters"].shape == (4, 3)


def test_reference_and_numba_trajectories_are_bit_identical():
    H = SMALL_H
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = UniformAnchorReplicaExchangeConfig(
        p=0.10, burn_rounds=8, measurement_rounds=16, num_replicas=4,
    )
    seed = UniformAnchorReplicaExchangeSeedIdentity(
        source_commit="5" * 40,
        config_sha256="6" * 64,
        registry_sha256="7" * 64,
        cell_fingerprint="8" * 64,
        method_id=config.method_id,
        resource_tier="test",
        init_family="U",
        trajectory_index=3,
        trajectory_namespace="exp102.q0_hgp_uniform_anchor_pt.test",
    )
    reference = run_uniform_anchor_replica_exchange_trajectory(
        model, frame, H, syndrome, config, seed, epsilon, engine="reference",
    )
    accelerated = run_uniform_anchor_replica_exchange_trajectory(
        model, frame, H, syndrome, config, seed, epsilon, engine="numba",
    )
    exact_fields = (
        "initial_state_packed", "burn_state_packed", "final_state_packed",
        "measurement_states_packed", "measurement_b_columns",
        "measurement_a_syndromes", "burn_labels", "measurement_labels",
        "measurement_weights", "measurement_residual_weights",
        "measurement_block", "burn_b_weights", "measurement_b_weights",
        "burn_row_counters", "measurement_row_counters",
        "burn_hot_refresh_changed_bits", "measurement_hot_refresh_changed_bits",
        "burn_swap_attempts", "burn_swap_accepts", "measurement_swap_attempts",
        "measurement_swap_accepts", "burn_cold_a_column_draws",
        "measurement_cold_a_column_draws",
    )
    for field in exact_fields:
        assert np.array_equal(reference[field], accelerated[field]), field
    for field in ("burn_complete_scores", "measurement_complete_scores"):
        assert np.array_equal(reference[field], accelerated[field]), field
