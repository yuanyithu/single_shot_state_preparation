"""Exact small-HGP checks for full collapsed-B-column Gibbs updates."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    FullColumnGibbsConfig,
    build_full_column_candidate_cache,
    build_full_column_workspace,
    collapsed_a_syndromes,
    full_column_conditional_probabilities,
    full_column_gibbs_update,
    run_full_column_gibbs_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


class _DeterministicRng:
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


def _syndrome(model, nonzero):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    return epsilon, syndrome


def _b_columns(value, rows):
    result = np.zeros(rows, dtype=np.uint32)
    for column in range(rows):
        for row in range(rows):
            result[column] |= np.uint32(((value >> (column * rows + row)) & 1) << row)
    return result


def _b_index(columns, rows):
    value = 0
    for column in range(rows):
        for row in range(rows):
            value |= ((int(columns[column]) >> row) & 1) << (column * rows + row)
    return value


def _direct_column_probabilities(H, b_columns, a_syndromes, column, log_mass, p):
    rows, _ = H.shape
    old = int(b_columns[column])
    neighbors = np.flatnonzero(H[column])
    log_weights = np.empty(1 << rows, dtype=np.float64)
    for candidate in range(log_weights.size):
        value = candidate.bit_count() * math.log(p / (1.0 - p))
        for factor in neighbors:
            value += float(log_mass[int(a_syndromes[int(factor)]) ^ old ^ candidate])
        log_weights[candidate] = value
    log_weights -= log_weights.max()
    probabilities = np.exp(log_weights)
    return probabilities / probabilities.sum(dtype=np.float64)


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_full_column_conditional_matches_direct_enumeration(H, p, nonzero_syndrome):
    model, _ = build_model(H)
    epsilon, syndrome = _syndrome(model, nonzero_syndrome)
    b_columns, a_syndromes, _ = _initial_collapsed_masks(epsilon, syndrome, H)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    cache = build_full_column_candidate_cache(H.shape[0], p)
    workspace = build_full_column_workspace(cache)
    for column in range(H.shape[0]):
        actual = full_column_conditional_probabilities(
            H, syndrome.reshape(H.shape), b_columns, a_syndromes, column,
            log_mass, cache, workspace,
        )
        expected = _direct_column_probabilities(H, b_columns, a_syndromes, column, log_mass, p)
        assert np.max(np.abs(actual - expected)) <= 2e-13


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_single_column_detailed_balance_and_full_sweep_stationarity(H, p, nonzero_syndrome):
    model, _ = build_model(H)
    _, syndrome_flat = _syndrome(model, nonzero_syndrome)
    syndrome = syndrome_flat.reshape(H.shape)
    rows, _ = H.shape
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    cache = build_full_column_candidate_cache(rows, p)
    states = [_b_columns(value, rows) for value in range(1 << (rows * rows))]
    target = np.empty(len(states), dtype=np.float64)
    kernels = []
    for column in range(rows):
        kernel = np.zeros((len(states), len(states)), dtype=np.float64)
        for source, b_columns in enumerate(states):
            a_syndromes = collapsed_a_syndromes(H, syndrome, b_columns)
            target[source] = (
                (p / (1.0 - p)) ** sum(int(value).bit_count() for value in b_columns)
                * np.prod([mass[int(value)] for value in a_syndromes])
            )
            probabilities = full_column_conditional_probabilities(
                H, syndrome, b_columns, a_syndromes, column, log_mass, cache,
                build_full_column_workspace(cache),
            )
            for candidate, probability in enumerate(probabilities):
                destination_columns = b_columns.copy()
                destination_columns[column] = np.uint32(candidate)
                kernel[source, _b_index(destination_columns, rows)] += probability
        assert np.max(np.abs(kernel.sum(axis=1) - 1.0)) <= 2e-13
        detailed_balance = target[:, None] * kernel - target[None, :] * kernel.T
        assert np.max(np.abs(detailed_balance)) <= 3e-13
        kernels.append(kernel)
    target /= target.sum(dtype=np.float64)
    sweep = kernels[0]
    for kernel in kernels[1:]:
        sweep = sweep @ kernel
    assert np.max(np.abs(target @ sweep - target)) <= 4e-13


def test_full_column_update_preserves_cached_syndromes():
    H = SMALL_H[1]
    p = 0.04
    model, _ = build_model(H)
    epsilon, syndrome_flat = _syndrome(model, True)
    syndrome = syndrome_flat.reshape(H.shape)
    b_columns, a_syndromes, _ = _initial_collapsed_masks(epsilon, syndrome_flat, H)
    cache = build_full_column_candidate_cache(H.shape[0], p)
    workspace = build_full_column_workspace(cache)
    mass = build_classical_coset_mass(H, p, engine="reference")
    full_column_gibbs_update(
        b_columns, a_syndromes, H, syndrome, 0, np.log(mass), cache, workspace,
        _DeterministicRng(0.5),
    )
    assert np.array_equal(a_syndromes, collapsed_a_syndromes(H, syndrome, b_columns))


def test_full_column_trajectory_replays_exactly():
    H = SMALL_H[1]
    p = 0.10
    model, frame = build_model(H)
    epsilon, syndrome = _syndrome(model, True)
    config = FullColumnGibbsConfig(p, burn_sweeps=8, measurement_sweeps=16)
    cache = build_full_column_candidate_cache(H.shape[0], p)
    first = run_full_column_gibbs_trajectory(
        model, frame, H, syndrome, config, epsilon, 112233, 445566,
        cache=cache, workspace=build_full_column_workspace(cache),
    )
    second = run_full_column_gibbs_trajectory(
        model, frame, H, syndrome, config, epsilon, 112233, 445566,
        cache=cache, workspace=build_full_column_workspace(cache),
    )
    for key in (
        "initial_b_columns", "burn_b_columns", "final_b_columns",
        "measurement_b_columns", "measurement_labels", "measurement_weights",
        "measurement_blocks", "counters",
    ):
        assert np.array_equal(first[key], second[key])
    assert first["counters"][0] == (config.burn_sweeps + config.measurement_sweeps) * H.shape[0]
    assert first["counters"][3] == config.measurement_sweeps
