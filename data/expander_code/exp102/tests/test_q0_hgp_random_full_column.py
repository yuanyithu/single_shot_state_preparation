"""Exact tests for one-column-per-clock collapsed-HGP random scan."""

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    build_full_column_candidate_cache,
    build_full_column_workspace,
    full_column_conditional_probabilities,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column import (
    RandomFullColumnConfig,
    replay_random_full_column_trajectory,
    run_random_full_column_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, generator in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generator
    return states


def _syndrome(model, nonzero):
    error = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        error[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ error.astype(np.int64) % 2
    ).astype(np.uint8)


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
def test_random_scan_mixture_of_full_column_conditionals_is_stationary(H, p, nonzero):
    model, _ = build_model(H)
    syndrome = _syndrome(model, nonzero)
    states = _hard_coset_states(model, syndrome)
    packed_to_index = {
        np.packbits(state, bitorder="little").tobytes(): index
        for index, state in enumerate(states)
    }
    rows = H.shape[0]
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    cache = build_full_column_candidate_cache(rows, p)
    workspace = build_full_column_workspace(cache)
    transition = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
    for source, state in enumerate(states):
        A, B = split_hgp_state(state, H)
        del A
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
        for column in range(rows):
            probabilities = full_column_conditional_probabilities(
                H, syndrome.reshape(H.shape), b_columns, a_syndromes,
                column, log_mass, cache=cache, workspace=workspace,
            )
            for candidate, probability in enumerate(probabilities):
                target_B = B.copy()
                target_B[:, column] = np.asarray(
                    [(candidate >> bit) & 1 for bit in range(rows)], dtype=np.uint8,
                )
                # Sum over every compatible A state because the collapsed
                # transition integrates A out before the observation redraw.
                matching = []
                matching_mass = []
                for target, target_state in enumerate(states):
                    target_A, observed_B = split_hgp_state(target_state, H)
                    if np.array_equal(observed_B, target_B):
                        matching.append(target)
                        matching_mass.append((p / (1.0 - p)) ** int(target_A.sum()))
                normalizer = sum(matching_mass)
                for target, value in zip(matching, matching_mass):
                    transition[source, target] += (
                        probability * value / normalizer / rows
                    )
    target = (p / (1.0 - p)) ** states.sum(axis=1)
    target /= target.sum()
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-13
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 2e-13
    assert np.max(np.abs(target @ transition - target)) <= 2e-13


@pytest.mark.parametrize("H", SMALL_H)
def test_random_full_column_raw_replays_bit_exactly(H):
    model, frame = build_model(H)
    syndrome = _syndrome(model, True)
    initial = model.logical_sector_section.apply(syndrome, strict=True)
    config = RandomFullColumnConfig(
        p=0.10, burn_updates=3, measurement_updates=8,
    )
    raw = run_random_full_column_trajectory(
        model, frame, H, syndrome, config, initial, 11, 12, 13,
    )
    assert replay_random_full_column_trajectory(
        model, frame, H, syndrome, config, initial, 11, 12, 13, raw,
    )
    assert int(raw["burn__counters"][0]) == 3
    assert int(raw["measurement__counters"][0]) == 8
    assert int(raw["measurement__counters"][3]) == 8
    assert raw["measurement__states_packed"].shape[0] == 8


def test_random_full_column_config_rejects_sweep_ambiguity():
    with pytest.raises(ValueError, match="eight blocks"):
        RandomFullColumnConfig(p=0.04, burn_updates=1, measurement_updates=7)
    with pytest.raises(ValueError, match="schedule"):
        RandomFullColumnConfig(
            p=0.04, burn_updates=1, measurement_updates=8,
            schedule="all_columns_per_sweep",
        )
