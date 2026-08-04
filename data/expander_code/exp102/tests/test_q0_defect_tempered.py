"""Exact and transcript checks for the diagnostic defect-tempered kernel."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_defect_tempered import (
    DefectTemperedConfig,
    DefectTemperedSeedIdentity,
    defect_tempered_swap_log_acceptance,
    run_defect_tempered_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import unpack_states
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _all_states(num_qubits):
    values = np.arange(1 << num_qubits, dtype=np.uint64)
    return ((values[:, None] >> np.arange(num_qubits, dtype=np.uint64)) & 1).astype(np.uint8)


def _syndrome_weights(model, states, syndrome):
    residual = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    return residual.sum(axis=1).astype(np.int32)


def _target(model, states, syndrome, p, kq):
    kp = math.log((1.0 - p) / p)
    values = np.exp(-kp * states.sum(axis=1) - kq * _syndrome_weights(model, states, syndrome))
    return values / values.sum()


def _nonzero_syndrome(model):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[0] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    assert syndrome.any()
    return epsilon, syndrome


def _seed(method="DTC-test", family="P", trajectory=0):
    return DefectTemperedSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=method,
        resource_tier="test",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace="q0_defect_tempered_test",
    )


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
def test_finite_penalty_d0_conditional_is_exact_hard_posterior(classical, p, nonzero):
    model, _ = _model(classical)
    states = _all_states(model.num_qubits)
    if nonzero:
        _, syndrome = _nonzero_syndrome(model)
    else:
        syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    defects = _syndrome_weights(model, states, syndrome)
    soft = _target(model, states, syndrome, p, kq=3.75)
    conditioned = soft[defects == 0]
    conditioned /= conditioned.sum()
    kp = math.log((1.0 - p) / p)
    hard = np.exp(-kp * states[defects == 0].sum(axis=1))
    hard /= hard.sum()
    assert np.max(np.abs(conditioned - hard)) <= 2e-14


def _single_bit_heatbath_matrix(model, states, syndrome, p, kq, qubit):
    lookup = {
        np.packbits(row, bitorder="little").tobytes(): index
        for index, row in enumerate(states)
    }
    target = _target(model, states, syndrome, p, kq)
    matrix = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
    for source, state in enumerate(states):
        flipped = state.copy()
        flipped[qubit] ^= 1
        target_index = lookup[np.packbits(flipped, bitorder="little").tobytes()]
        probability = target[target_index] / (target[source] + target[target_index])
        matrix[source, target_index] = probability
        matrix[source, source] = 1.0 - probability
    return target, matrix


def test_local_heatbath_hot_refresh_and_swap_are_exact_on_small_hgp():
    model, _ = _model([[1, 1, 1]])
    _, syndrome = _nonzero_syndrome(model)
    states = _all_states(model.num_qubits)
    p = 0.10
    kq_cold, kq_hot = 3.0, 0.0
    cold = _target(model, states, syndrome, p, kq_cold)
    hot = _target(model, states, syndrome, p, kq_hot)
    for qubit in range(model.num_qubits):
        target, matrix = _single_bit_heatbath_matrix(
            model, states, syndrome, p, kq_cold, qubit,
        )
        assert np.max(np.abs(target @ matrix - target)) <= 1e-14
        flow = target[:, None] * matrix
        assert np.max(np.abs(flow - flow.T)) <= 1e-14
    # The iid endpoint is a rank-one reversible kernel for the Kq=0 target.
    refresh = np.repeat(hot[None, :], states.shape[0], axis=0)
    assert np.max(np.abs(hot @ refresh - hot)) <= 1e-14
    assert np.max(np.abs(hot[:, None] * refresh - (hot[:, None] * refresh).T)) <= 1e-14
    defects = _syndrome_weights(model, states, syndrome)
    for left, right in itertools.product(range(states.shape[0]), repeat=2):
        log_ratio = defect_tempered_swap_log_acceptance(
            kq_cold, kq_hot, defects[left], defects[right],
        )
        forward = cold[left] * hot[right] * min(1.0, math.exp(log_ratio))
        backward = cold[right] * hot[left] * min(1.0, math.exp(-log_ratio))
        assert abs(forward - backward) <= 2e-15


def test_reference_trajectory_is_replayable_and_filters_only_hard_coset_clocks():
    model, frame = _model([[1, 1, 1]])
    initial, syndrome = _nonzero_syndrome(model)
    config = DefectTemperedConfig(
        method_id="DTC-test",
        p=0.10,
        kq_values=(4.0, 2.0, 0.5, 0.0),
        burn_rounds=4,
        measurement_rounds=16,
    )
    outputs = [
        run_defect_tempered_trajectory(model, frame, syndrome, config, _seed(), initial)
        for _ in range(2)
    ]
    for name in outputs[0]:
        assert np.array_equal(np.asarray(outputs[0][name]), np.asarray(outputs[1][name])), name
    result = outputs[0]
    assert result["ladder_kq"].tolist() == [4.0, 2.0, 0.5, 0.0]
    assert result["measurement_d0_mask"].dtype == np.uint8
    assert np.array_equal(
        result["measurement_d0_mask"],
        (result["measurement_defects"] == 0).astype(np.uint8),
    )
    states = unpack_states(result["measurement_states_packed"], model.num_qubits)
    residuals = _syndrome_weights(model, states, syndrome)
    assert np.array_equal(residuals, result["measurement_residual_weights"])
    if result["measurement_d0_mask"].any():
        assert not residuals[result["measurement_d0_mask"].astype(bool)].any()
    assert result["measurement_hot_refresh_bit_changes"] >= 0
    assert result["measurement_swap_attempts"].sum() > 0


def test_reference_and_numba_transcripts_match_field_by_field():
    model, frame = _model([[1, 1, 0], [0, 1, 1]])
    initial, syndrome = _nonzero_syndrome(model)
    config = DefectTemperedConfig(
        method_id="DTC-test",
        p=0.25,
        kq_values=(4.0, 2.0, 0.5, 0.0),
        burn_rounds=3,
        measurement_rounds=16,
    )
    outputs = [
        run_defect_tempered_trajectory(
            model, frame, syndrome, config, _seed(), initial, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    for name in outputs[0]:
        if name != "engine":
            assert np.array_equal(
                np.asarray(outputs[0][name]), np.asarray(outputs[1][name]),
            ), name


def test_m8_bit63_reference_numba_transcript_has_no_int64_conversion():
    _, _, H = load_frozen_code(
        "data/expander_code/exp102/registry/registry.json", "m08_c06",
    )
    model, frame = build_model(H)
    # The canonical pairing makes the 64th logical generator a direct bit-63 probe.
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[0] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    initial = epsilon ^ model.logical_move_basis[63]
    config = DefectTemperedConfig(
        method_id="DTC-test",
        p=0.04,
        kq_values=(4.0, 1.0, 0.0),
        burn_rounds=1,
        measurement_rounds=8,
    )
    results = [
        run_defect_tempered_trajectory(
            model, frame, syndrome, config, _seed(), initial, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    assert (int(results[0]["initial_label"]) >> 63) & 1
    assert results[0]["measurement_labels"].dtype == np.uint64
    for name in results[0]:
        if name != "engine":
            assert np.array_equal(
                np.asarray(results[0][name]), np.asarray(results[1][name]),
            ), name


def test_config_rejects_nonfinite_or_noncanonical_ladders():
    with pytest.raises(ValueError, match="exact Kq=0"):
        DefectTemperedConfig("DTC", 0.1, (4.0, 1.0), 1, 8)
    with pytest.raises(ValueError, match="strictly descending"):
        DefectTemperedConfig("DTC", 0.1, (4.0, 4.0, 0.0), 1, 8)
    with pytest.raises(ValueError, match="positive integer"):
        DefectTemperedConfig("DTC", 0.1, (4.0, 0.0), True, 8)
