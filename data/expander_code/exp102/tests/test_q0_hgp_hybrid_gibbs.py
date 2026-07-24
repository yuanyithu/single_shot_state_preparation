import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    FullColumnGibbsConflictError,
    build_full_column_direct_block_cache,
    build_full_column_direct_block_workspace,
    collapsed_a_syndromes,
    full_column_direct_block_conditional_probabilities,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs import (
    build_full_row_elimination_plan,
    full_row_conditional_log_weights,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_hybrid_gibbs import (
    HYBRID_COUNTERS,
    HybridGibbsConfig,
    hybrid_gibbs_clock,
    replay_hybrid_gibbs_trajectory,
    run_hybrid_gibbs_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    return PortablePrng(seed)


def _b_matrices(rows):
    for bits in itertools.product((0, 1), repeat=rows * rows):
        yield np.asarray(bits, dtype=np.uint8).reshape(rows, rows)


def _b_columns(B):
    return np.asarray([
        _bits_to_mask(B[:, column]) for column in range(B.shape[1])
    ], dtype=np.uint32)


def _a_syndromes(H, Y, B):
    matrix = Y ^ (
        B.astype(np.int64) @ H.astype(np.int64) % 2
    ).astype(np.uint8)
    return np.asarray([
        _bits_to_mask(matrix[:, column]) for column in range(H.shape[1])
    ], dtype=np.uint32)


def _target(H, Y, p, matrices):
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    scores = []
    for B in matrices:
        a_syndromes = _a_syndromes(H, Y, B)
        scores.append(
            int(B.sum()) * log_odds
            + sum(float(log_mass[int(value)]) for value in a_syndromes)
        )
    scores = np.asarray(scores, dtype=np.float64)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    return probabilities, mass, log_mass, log_odds


@pytest.mark.parametrize("H", [
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
])
@pytest.mark.parametrize("p", [0.04, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_complete_hybrid_clock_preserves_exact_collapsed_target(
        H, p, nonzero_syndrome):
    rows, columns = H.shape
    Y = np.zeros((rows, columns), dtype=np.uint8)
    if nonzero_syndrome:
        Y[0, 0] = 1
    matrices = list(_b_matrices(rows))
    keys = {matrix.tobytes(): index for index, matrix in enumerate(matrices)}
    target, mass, log_mass, log_odds = _target(H, Y, p, matrices)
    cache = build_full_column_direct_block_cache(rows, p, mass)
    kernel = np.zeros((len(matrices), len(matrices)), dtype=np.float64)
    for source_index, B in enumerate(matrices):
        b_columns = _b_columns(B)
        a_syndromes = _a_syndromes(H, Y, B)
        for selected_column in range(rows):
            column_probabilities = (
                full_column_direct_block_conditional_probabilities(
                    H, Y, b_columns, a_syndromes, selected_column, mass,
                    cache=cache,
                )
            )
            for new_column, column_probability in enumerate(
                    column_probabilities):
                after_column = B.copy()
                after_column[:, selected_column] = [
                    (new_column >> row) & 1 for row in range(rows)
                ]
                after_column_columns = _b_columns(after_column)
                after_column_a = _a_syndromes(H, Y, after_column)
                for selected_row in range(rows):
                    row_log_weights = full_row_conditional_log_weights(
                        H, after_column_columns, after_column_a, selected_row,
                        log_mass, log_odds,
                    )
                    row_probabilities = np.exp(
                        row_log_weights - row_log_weights.max()
                    )
                    row_probabilities /= row_probabilities.sum()
                    for new_row, row_probability in enumerate(
                            row_probabilities):
                        proposed = after_column.copy()
                        proposed[selected_row, :] = [
                            (new_row >> column) & 1
                            for column in range(rows)
                        ]
                        kernel[source_index, keys[proposed.tobytes()]] += (
                            column_probability * row_probability / (rows * rows)
                        )
    assert np.max(np.abs(kernel.sum(axis=1) - 1.0)) <= 4e-15
    assert np.max(np.abs(target @ kernel - target)) <= 5e-15


def test_hybrid_clock_replays_portable_stream_and_rejects_tampered_cache():
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    planted[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    mass = build_classical_coset_mass(H, 0.10, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(0.10 / 0.90)
    cache = build_full_column_direct_block_cache(H.shape[0], 0.10, mass)
    plan = build_full_row_elimination_plan(H)
    initial_b, initial_a, _ = _initial_collapsed_masks(planted, syndrome, H)
    transcripts = []
    for _ in range(2):
        b_columns = initial_b.copy()
        a_syndromes = initial_a.copy()
        step = hybrid_gibbs_clock(
            b_columns, a_syndromes, H, syndrome.reshape(H.shape), mass,
            log_mass, log_odds, cache,
            build_full_column_direct_block_workspace(cache), plan, _rng(1234),
            column_engine="reference",
        )
        transcripts.append((step, b_columns.copy(), a_syndromes.copy()))
    assert transcripts[0][0] == transcripts[1][0]
    assert np.array_equal(transcripts[0][1], transcripts[1][1])
    assert np.array_equal(transcripts[0][2], transcripts[1][2])
    assert np.array_equal(
        transcripts[0][2],
        collapsed_a_syndromes(H, syndrome.reshape(H.shape), transcripts[0][1]),
    )
    bad = initial_a.copy()
    bad[0] ^= np.uint32(1)
    with pytest.raises(FullColumnGibbsConflictError, match="cached A syndromes"):
        hybrid_gibbs_clock(
            initial_b.copy(), bad, H, syndrome.reshape(H.shape), mass,
            log_mass, log_odds, cache,
            build_full_column_direct_block_workspace(cache), plan, _rng(1),
            column_engine="reference",
        )


def test_hybrid_trajectory_full_raw_replays_and_stays_in_hard_coset():
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    initial[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ initial.astype(np.int64) % 2
    ).astype(np.uint8)
    config = HybridGibbsConfig(
        p=0.10, burn_clocks=8, measurement_clocks=16,
    )
    mass = build_classical_coset_mass(H, config.p, engine="reference")
    cache = build_full_column_direct_block_cache(H.shape[0], config.p, mass)
    plan = build_full_row_elimination_plan(H)
    raw = run_hybrid_gibbs_trajectory(
        model, frame, H, syndrome, config, initial, 11, 22, 33, mass=mass,
        column_cache=cache,
        column_workspace=build_full_column_direct_block_workspace(cache),
        row_plan=plan, column_engine="reference",
    )
    assert replay_hybrid_gibbs_trajectory(
        model, frame, H, syndrome, config, initial, 11, 22, 33, raw,
        mass=mass, column_cache=cache,
        column_workspace=build_full_column_direct_block_workspace(cache),
        row_plan=plan, column_engine="reference",
    )
    assert tuple(raw["measurement__counters"][[0, 1, 4, 7]]) == (16, 16, 16, 16)
    states = np.unpackbits(
        raw["measurement__states_packed"], axis=1,
        count=model.num_qubits, bitorder="little",
    ).astype(np.uint8)
    residuals = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    assert np.array_equal(
        residuals, np.repeat(syndrome[None, :], states.shape[0], axis=0),
    )
    assert len(HYBRID_COUNTERS) == raw["measurement__counters"].size


def test_hybrid_config_rejects_clock_and_identity_changes():
    with pytest.raises(ValueError, match="divide into eight"):
        HybridGibbsConfig(p=0.10, burn_clocks=8, measurement_clocks=15)
    with pytest.raises(ValueError, match="method changed"):
        HybridGibbsConfig(
            p=0.10, burn_clocks=8, measurement_clocks=16,
            method_id="changed",
        )
