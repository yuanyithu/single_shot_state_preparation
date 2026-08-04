import itertools

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed_ais import (
    COLLAPSED_AIS_KERNEL,
    CollapsedAisConfig,
    CollapsedAisSeedIdentity,
    a_syndromes_from_b,
    collapsed_log_likelihood,
    collapsed_log_target,
    normalized_log_weights,
    quadratic_lambda_schedule,
    run_collapsed_ais_population,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _model(H):
    H = np.asarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    return H, model, frame


def _seed(config, family="column_major", population=0):
    return CollapsedAisSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        base_family=family,
        population_index=population,
        trajectory_namespace="q0_hgp_collapsed_ais_test",
    )


def _syndrome_from_epsilon(model, indices):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[list(indices)] = 1
    return (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)


def _all_b_columns(rank):
    result = np.zeros((1 << (rank * rank), rank), dtype=np.uint32)
    for integer in range(result.shape[0]):
        for column in range(rank):
            for row in range(rank):
                if (integer >> (column * rank + row)) & 1:
                    result[integer, column] |= np.uint32(1) << np.uint32(row)
    return result


def _b_index(row, rank):
    result = 0
    for column, value in enumerate(row):
        for bit in range(rank):
            result |= ((int(value) >> bit) & 1) << (column * rank + bit)
    return result


def _normalized(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.exp(values - values.max())
    return result / result.sum()


def _random_block_heatbath_matrix(H, syndrome, p, power):
    """Exact random-scan kernel matching the reversible AIS mutation step."""
    rank = H.shape[0]
    b_states = _all_b_columns(rank)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    a_syndromes = a_syndromes_from_b(syndrome, H, b_states)
    target = _normalized(collapsed_log_target(b_states, a_syndromes, p, log_mass, power))
    blocks_per_column = (rank + 7) // 8
    transition = np.zeros((b_states.shape[0], b_states.shape[0]), dtype=np.float64)
    for source, source_b in enumerate(b_states):
        for column in range(rank):
            for block in range(blocks_per_column):
                start = block * 8
                width = min(8, rank - start)
                candidates = []
                for category in range(1 << width):
                    candidate = source_b.copy()
                    selected = np.uint32(((1 << width) - 1) << start)
                    candidate[column] = (candidate[column] & ~selected) | np.uint32(category << start)
                    candidates.append(_b_index(candidate, rank))
                probabilities = target[candidates]
                probabilities /= probabilities.sum()
                for target_index, probability in zip(candidates, probabilities):
                    transition[source, target_index] += probability / (rank * blocks_per_column)
    return target, transition, log_mass


@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_reversible_random_block_mutation_obeys_detailed_balance(H, p):
    H, model, _ = _model(H)
    syndrome = _syndrome_from_epsilon(model, (0,))
    for power in (0.25, 0.75, 1.0):
        target, transition, _ = _random_block_heatbath_matrix(H, syndrome, p, power)
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 3e-15
        flow = target[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 4e-14


@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_ais_path_weight_recovers_exact_cold_bridge_measure(H):
    H, model, _ = _model(H)
    syndrome = _syndrome_from_epsilon(model, (0,))
    p = 0.10
    b_states = _all_b_columns(H.shape[0])
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    a_syndromes = a_syndromes_from_b(syndrome, H, b_states)
    likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
    measure = _normalized(collapsed_log_target(b_states, a_syndromes, p, log_mass, 0.0))
    for previous, current in zip((0.0, 0.25, 0.75), (0.25, 0.75, 1.0)):
        measure = measure * np.exp((current - previous) * likelihood)
        _, transition, _ = _random_block_heatbath_matrix(H, syndrome, p, current)
        measure = measure @ transition
    cold = _normalized(collapsed_log_target(b_states, a_syndromes, p, log_mass, 1.0))
    measure /= measure.sum()
    assert np.max(np.abs(measure - cold)) <= 7e-14


def test_ais_config_and_reference_numba_transcript_identity():
    H, model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome_from_epsilon(model, (0, 4))
    config = CollapsedAisConfig(0.10, 12, quadratic_lambda_schedule(5))
    reference = run_collapsed_ais_population(
        model, frame, H, syndrome, config, _seed(config), engine="reference",
    )
    accelerated = run_collapsed_ais_population(
        model, frame, H, syndrome, config, _seed(config), engine="numba",
    )
    for name in reference:
        if name != "engine":
            assert np.array_equal(
                np.asarray(reference[name]), np.asarray(accelerated[name]),
            ), name
    assert config.as_dict()["kernel"] == COLLAPSED_AIS_KERNEL
    assert np.isclose(reference["final_normalized_weights"].sum(), 1.0)
    assert np.all(reference["mutation_attempts"] > 0)


def test_ais_small_population_recovers_simple_exact_b_marginal():
    H, model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome_from_epsilon(model, (0,))
    config = CollapsedAisConfig(0.10, 2048, quadratic_lambda_schedule(8))
    result = run_collapsed_ais_population(
        model, frame, H, syndrome, config, _seed(config, "row_major"), engine="numba",
    )
    b_states = _all_b_columns(1)
    mass = build_classical_coset_mass(H, config.p, engine="reference")
    a_syndromes = a_syndromes_from_b(syndrome, H, b_states)
    exact = _normalized(collapsed_log_target(b_states, a_syndromes, config.p, np.log(mass)))
    final_bits = (result["b_columns_by_stage"][-1, :, 0] & np.uint32(1)) != 0
    estimate = float(np.dot(result["final_normalized_weights"], final_bits.astype(np.float64)))
    assert abs(estimate - float(exact[1])) <= 0.05
    normal, ess, maximum, log_mean = normalized_log_weights(result["cumulative_log_weights"][-1])
    assert np.array_equal(normal, result["final_normalized_weights"])
    assert ess == float(result["final_importance_ess"])
    assert maximum == float(result["final_max_normalized_weight"])
    assert log_mean == float(result["final_log_mean_weight"])
