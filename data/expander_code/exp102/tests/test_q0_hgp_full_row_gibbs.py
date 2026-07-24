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
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs import (
    _assignment_log_probability,
    build_full_row_elimination_plan,
    eliminate_full_row_conditional,
    full_row_conditional_statistics,
    full_row_conditional_log_weights,
    full_row_elimination_gibbs_update,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng
    return PortablePrng(seed)


def _b_matrices(rows):
    for value in range(1 << (rows * rows)):
        bits = np.asarray([
            (value >> bit) & 1 for bit in range(rows * rows)
        ], dtype=np.uint8)
        yield bits.reshape(rows, rows)


def _b_columns(B):
    return np.asarray([
        _bits_to_mask(B[:, column]) for column in range(B.shape[1])
    ], dtype=np.uint32)


def _collapsed_scores(H, syndrome, p):
    rows, columns = H.shape
    Y = syndrome.reshape(rows, columns)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    matrices = list(_b_matrices(rows))
    scores = np.empty(len(matrices), dtype=np.float64)
    for index, B in enumerate(matrices):
        syndromes = Y ^ (
            B.astype(np.int64) @ H.astype(np.int64) % 2
        ).astype(np.uint8)
        scores[index] = int(B.sum()) * log_odds + sum(
            float(log_mass[int(_bits_to_mask(syndromes[:, column]))])
            for column in range(columns)
        )
    return matrices, scores, log_mass, log_odds


@pytest.mark.parametrize("H", [
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_elimination_distribution_matches_complete_row_enumeration(
        H, p, nonzero_syndrome):
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        planted[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    b_columns, a_syndromes, _ = _initial_collapsed_masks(planted, syndrome, H)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    plan = build_full_row_elimination_plan(H)
    for row in range(H.shape[0]):
        direct = full_row_conditional_log_weights(
            H, b_columns, a_syndromes, row, log_mass, log_odds,
        )
        log_z, records = eliminate_full_row_conditional(
            H, b_columns, a_syndromes, row, log_mass, log_odds, plan=plan,
        )
        maximum = float(direct.max())
        expected_log_z = maximum + math.log(float(np.exp(direct - maximum).sum()))
        assert abs(log_z - expected_log_z) <= 3e-13
        for candidate in range(1 << H.shape[0]):
            assignment = tuple(
                (candidate >> variable) & 1 for variable in range(H.shape[0])
            )
            actual = _assignment_log_probability(records, assignment)
            assert abs(actual - (direct[candidate] - expected_log_z)) <= 3e-13
        probabilities = np.exp(direct - expected_log_z)
        old_row = sum(
            ((int(b_columns[column]) >> row) & 1) << column
            for column in range(H.shape[0])
        )
        statistics = full_row_conditional_statistics(
            H, b_columns, a_syndromes, row, log_mass, log_odds, plan=plan,
        )
        expected_entropy = -float(np.sum(probabilities * np.log(probabilities)))
        expected_change = sum(
            probabilities[candidate] * int(candidate ^ old_row).bit_count()
            for candidate in range(probabilities.size)
        )
        expected_weight = sum(
            probabilities[candidate] * candidate.bit_count()
            for candidate in range(probabilities.size)
        )
        assert abs(statistics["entropy_nats"] - expected_entropy) <= 3e-13
        assert abs(
            statistics["expected_hamming_change"] - expected_change
        ) <= 3e-13
        assert abs(statistics["expected_row_weight"] - expected_weight) <= 3e-13
        assert abs(
            statistics["self_probability"] - probabilities[old_row]
        ) <= 3e-15


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_single_row_detailed_balance_and_complete_row_sweep_stationarity(
        p, nonzero_syndrome):
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        planted[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    matrices, scores, log_mass, log_odds = _collapsed_scores(H, syndrome, p)
    target = np.exp(scores - scores.max())
    target /= target.sum()
    keys = {matrix.tobytes(): index for index, matrix in enumerate(matrices)}
    row_kernels = []
    for selected_row in range(H.shape[0]):
        kernel = np.zeros((len(matrices), len(matrices)), dtype=np.float64)
        for source_index, B in enumerate(matrices):
            state = np.concatenate((
                np.zeros(H.shape[1] ** 2, dtype=np.uint8), B.reshape(-1),
            ))
            # Only B is needed to recover the cached collapsed syndromes.
            Y = syndrome.reshape(H.shape)
            a_matrix = Y ^ (
                B.astype(np.int64) @ H.astype(np.int64) % 2
            ).astype(np.uint8)
            a_syndromes = np.asarray([
                _bits_to_mask(a_matrix[:, column]) for column in range(H.shape[1])
            ], dtype=np.uint32)
            columns = _b_columns(B)
            direct = full_row_conditional_log_weights(
                H, columns, a_syndromes, selected_row, log_mass, log_odds,
            )
            probabilities = np.exp(direct - direct.max())
            probabilities /= probabilities.sum()
            for candidate, probability in enumerate(probabilities):
                proposed = B.copy()
                proposed[selected_row, :] = [
                    (candidate >> column) & 1 for column in range(H.shape[0])
                ]
                kernel[source_index, keys[proposed.tobytes()]] += probability
        assert np.max(np.abs(kernel.sum(axis=1) - 1.0)) <= 2e-15
        flow = target[:, None] * kernel
        assert np.max(np.abs(flow - flow.T)) <= 3e-15
        row_kernels.append(kernel)
    sweep = row_kernels[0] @ row_kernels[1]
    assert np.max(np.abs(target @ sweep - target)) <= 3e-15


def test_update_replays_portable_draw_and_rejects_tampered_cache():
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    planted[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    p = 0.10
    log_mass = np.log(build_classical_coset_mass(H, p, engine="reference"))
    log_odds = math.log(p / (1.0 - p))
    initial_b, initial_syndromes, _ = _initial_collapsed_masks(
        planted, syndrome, H,
    )
    transcripts = []
    for _ in range(2):
        b_columns = initial_b.copy()
        a_syndromes = initial_syndromes.copy()
        transcripts.append(full_row_elimination_gibbs_update(
            b_columns, a_syndromes, H, syndrome.reshape(H.shape), 1,
            log_mass, log_odds, _rng(12345),
        ) + (b_columns.copy(), a_syndromes.copy()))
    for left, right in zip(transcripts[0], transcripts[1]):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    bad = initial_syndromes.copy()
    bad[0] ^= np.uint32(1)
    with pytest.raises(FullColumnGibbsConflictError, match="cached syndromes"):
        full_row_elimination_gibbs_update(
            initial_b.copy(), bad, H, syndrome.reshape(H.shape), 0,
            log_mass, log_odds, _rng(1),
        )


def test_frozen_m8_plan_has_small_width_and_expected_identity():
    registry = "data/expander_code/exp102/registry/registry.json"
    _, _, H = load_frozen_code(registry, "m08_c06")
    plan = build_full_row_elimination_plan(H)
    assert plan.induced_width == 12
    assert plan.largest_factor_entries == 8192
    assert plan.elimination_order == (
        3, 16, 6, 23, 9, 15, 4, 13, 12, 8, 22, 0,
        1, 2, 5, 7, 10, 11, 14, 17, 18, 19, 20, 21,
    )
    altered = H.copy()
    altered[0, 0] ^= 1
    with pytest.raises(FullColumnGibbsConflictError, match="plan binding"):
        plan.validate(altered)
