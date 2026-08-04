"""Exact checks for UASRE: UARE transport plus a cold stabilizer block."""

import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer import (
    auxiliary_stabilizer_conditional_probabilities,
    auxiliary_stabilizer_delta,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer_pt import (
    AuxiliaryStabilizerReplicaExchangeConfig,
    AuxiliaryStabilizerReplicaExchangeSeedIdentity,
    run_auxiliary_stabilizer_replica_exchange_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    build_classical_coset_mass,
    join_hgp_state,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    build_full_row_elimination_plan,
    full_row_conditional_probabilities,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_uniform_anchor_pt import (
    collapsed_complete_score,
    full_energy_swap_log_acceptance,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)


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


def _a_syndromes(syndrome, matrix, b_columns):
    rows, columns = matrix.shape
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    result = np.asarray([_bits_to_mask(Y[:, column]) for column in range(columns)], dtype=np.uint32)
    for column in range(columns):
        for variable in np.flatnonzero(matrix[:, column]):
            result[column] ^= b_columns[int(variable)]
    return result


def _replace_b_row(columns, row, assignment):
    result = np.asarray(columns, dtype=np.uint32).copy()
    mask = np.uint32(1 << row)
    clear = np.uint32(~int(mask) & 0xFFFFFFFF)
    for variable in range(result.size):
        result[variable] &= clear
        if (assignment >> variable) & 1:
            result[variable] |= mask
    return result


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


def _data(p=0.10):
    model, _ = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    b_states = _all_b_columns(H.shape[0])
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    a_states = [_a_syndromes(syndrome, H, b_columns) for b_columns in b_states]
    scores = np.asarray([
        collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds)
        for b_columns, a_syndromes in zip(b_states, a_states)
    ], dtype=np.float64)
    b_posterior = np.exp(scores - scores.max())
    b_posterior /= b_posterior.sum(dtype=np.float64)
    return model, syndrome, b_states, a_states, log_mass, log_odds, scores, b_posterior


def _cold_row_kernel(b_states, a_states, log_mass, log_odds):
    plan = build_full_row_elimination_plan(H)
    transition = np.zeros((b_states.shape[0], b_states.shape[0]), dtype=np.float64)
    for source, (b_columns, a_syndromes) in enumerate(zip(b_states, a_states)):
        probabilities, _ = full_row_conditional_probabilities(
            H, plan, b_columns, a_syndromes, 0, log_mass, log_odds,
        )
        for assignment, probability in enumerate(probabilities):
            target = _b_index(_replace_b_row(b_columns, 0, assignment), H.shape[0])
            transition[source, target] += probability
    return transition


def _auxiliary_marginal_kernel(model, syndrome, p):
    states = _hard_coset_states(model, syndrome)
    posterior = (p / (1.0 - p)) ** states.sum(axis=1)
    posterior /= posterior.sum(dtype=np.float64)
    groups = {index: [] for index in range(1 << (H.shape[0] ** 2))}
    for index, state in enumerate(states):
        _, B = split_hgp_state(state, H)
        columns = np.asarray([_bits_to_mask(B[:, column]) for column in range(B.shape[1])], dtype=np.uint32)
        groups[_b_index(columns, H.shape[0])].append(index)
    b_mass = np.asarray([posterior[groups[index]].sum(dtype=np.float64) for index in range(len(groups))])
    transition = np.zeros((len(groups), len(groups)), dtype=np.float64)
    for source, members in groups.items():
        for state_index in members:
            A, B = split_hgp_state(states[state_index], H)
            lift_probability = posterior[state_index] / b_mass[source]
            probabilities, _ = auxiliary_stabilizer_conditional_probabilities(H, A, B, 0, p)
            for assignment, probability in enumerate(probabilities):
                delta_a, delta_b = auxiliary_stabilizer_delta(H, 0, assignment)
                _, target_B = split_hgp_state(join_hgp_state(A ^ delta_a, B ^ delta_b), H)
                columns = np.asarray([
                    _bits_to_mask(target_B[:, column]) for column in range(target_B.shape[1])
                ], dtype=np.uint32)
                transition[source, _b_index(columns, H.shape[0])] += lift_probability * probability
    return b_mass, transition


def test_uasre_config_has_frozen_kernel_and_identity():
    config = AuxiliaryStabilizerReplicaExchangeConfig(
        p=0.04, burn_rounds=16, measurement_rounds=32, num_replicas=8,
        positive_row_updates_per_round=1, cold_auxiliary_rows_per_round=1,
    )
    assert config.method_id == "UASRE08-R1-A1"
    assert config.as_dict()["cold_block_kernel"] == "exact_auxiliary_A_row_stabilizer_heatbath.v1"
    assert config.lambda_values[0] == 0.0
    assert config.lambda_values[-1] == 1.0


def test_post_swap_auxiliary_lift_preserves_two_rung_product_target():
    p = 0.10
    model, syndrome, b_states, a_states, log_mass, log_odds, scores, cold = _data(p)
    auxiliary_cold, auxiliary = _auxiliary_marginal_kernel(model, syndrome, p)
    assert np.max(np.abs(auxiliary_cold - cold)) <= 8e-14
    assert np.max(np.abs(auxiliary.sum(axis=1) - 1.0)) <= 7e-15
    assert np.max(np.abs(cold @ auxiliary - cold)) <= 9e-14
    assert np.max(np.abs(cold[:, None] * auxiliary - (cold[:, None] * auxiliary).T)) <= 9e-14
    cold_row = _cold_row_kernel(b_states, a_states, log_mass, log_odds)
    count = b_states.shape[0]
    hot = np.full(count, 1.0 / count, dtype=np.float64)
    local = np.zeros((count * count, count * count), dtype=np.float64)
    for hot_source in range(count):
        for cold_source in range(count):
            source = hot_source * count + cold_source
            for hot_target in range(count):
                for cold_target in range(count):
                    target = hot_target * count + cold_target
                    local[source, target] = hot[hot_target] * cold_row[cold_source, cold_target]
    swap = np.zeros_like(local)
    for hot_source in range(count):
        for cold_source in range(count):
            source = hot_source * count + cold_source
            acceptance = min(1.0, math.exp(full_energy_swap_log_acceptance(
                0.0, 1.0, scores[hot_source], scores[cold_source],
            )))
            swap[source, source] += 1.0 - acceptance
            swap[source, cold_source * count + hot_source] += acceptance
    post_auxiliary = np.zeros_like(local)
    for hot_source in range(count):
        for cold_source in range(count):
            source = hot_source * count + cold_source
            for cold_target in range(count):
                post_auxiliary[source, hot_source * count + cold_target] += auxiliary[cold_source, cold_target]
    transition = local @ swap @ post_auxiliary
    product = np.outer(hot, cold).reshape(-1)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 5e-15
    assert np.max(np.abs(product @ transition - product)) <= 1e-13


def _seed(config):
    return AuxiliaryStabilizerReplicaExchangeSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        resource_tier="test",
        init_family="U",
        trajectory_index=2,
        trajectory_namespace="exp102.q0_hgp_aux_stabilizer_pt.test",
    )


def test_uasre_reference_trajectory_replays_and_stays_in_hard_coset():
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = AuxiliaryStabilizerReplicaExchangeConfig(0.10, 8, 16, 4, 1, 1)
    first = run_auxiliary_stabilizer_replica_exchange_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon, engine="reference",
    )
    second = run_auxiliary_stabilizer_replica_exchange_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon, engine="reference",
    )
    for field in first:
        if field != "engine":
            assert np.array_equal(np.asarray(first[field]), np.asarray(second[field])), field
    assert first["raw_version"] == "exp102.q0_hgp_aux_stabilizer_pt.raw.v0"
    assert first["measurement_auxiliary_assignments"].shape == (16, 1)
    assert first["burn_auxiliary_counters"][0] == 8
    assert first["measurement_auxiliary_counters"][0] == 16
    assert not first["measurement_residual_weights"].any()


def test_uasre_reference_numba_are_bit_identical():
    pytest.importorskip("numba")
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = AuxiliaryStabilizerReplicaExchangeConfig(0.10, 8, 16, 4, 1, 1)
    reference = run_auxiliary_stabilizer_replica_exchange_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon, engine="reference",
    )
    accelerated = run_auxiliary_stabilizer_replica_exchange_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon, engine="numba",
    )
    for field in reference:
        if field != "engine":
            assert np.array_equal(np.asarray(reference[field]), np.asarray(accelerated[field])), field
