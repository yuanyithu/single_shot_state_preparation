import itertools

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import split_hgp_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed_smc import (
    COLLAPSED_SMC_KERNEL,
    CollapsedSmcConfig,
    CollapsedSmcSeedIdentity,
    a_syndromes_from_b,
    collapsed_lambda_schedule,
    collapsed_log_likelihood,
    collapsed_log_target,
    run_collapsed_smc_population,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _model(H):
    H = np.asarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    return H, model, frame


def _seed(config, family="column_major", population=0):
    return CollapsedSmcSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        base_family=family,
        population_index=population,
        trajectory_namespace="q0_hgp_collapsed_smc_test",
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


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    result = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(result.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                result[coefficient] ^= row
    return result


def _normalized_log_weights(values):
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(values - values.max())
    return weights / weights.sum()


@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_collapsed_smc_target_matches_enumerated_full_hard_coset(H, p):
    H, model, _ = _model(H)
    syndrome = _syndrome_from_epsilon(model, (0,))
    from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
        build_classical_coset_mass,
    )

    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    b_columns = _all_b_columns(H.shape[0])
    a_syndromes = a_syndromes_from_b(syndrome, H, b_columns)
    collapsed = _normalized_log_weights(
        collapsed_log_target(b_columns, a_syndromes, p, log_mass)
    )

    states = _hard_coset_states(model, syndrome)
    full_weights = (p / (1.0 - p)) ** states.sum(axis=1)
    full_weights /= full_weights.sum()
    by_b = {}
    for state, weight in zip(states, full_weights):
        _, B = split_hgp_state(state, H)
        key = np.packbits(B.reshape(-1), bitorder="little").tobytes()
        by_b[key] = by_b.get(key, 0.0) + float(weight)
    enumerated = []
    for row in b_columns:
        B = np.zeros((H.shape[0], H.shape[0]), dtype=np.uint8)
        for column, value in enumerate(row):
            for bit in range(H.shape[0]):
                B[bit, column] = (int(value) >> bit) & 1
        enumerated.append(by_b[np.packbits(B.reshape(-1), bitorder="little").tobytes()])
    assert np.max(np.abs(collapsed - np.asarray(enumerated))) <= 3e-14


def test_quadratic_schedule_and_config_are_frozen():
    values = collapsed_lambda_schedule(8)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(left < right for left, right in zip(values, values[1:]))
    config = CollapsedSmcConfig(0.10, 16, values)
    assert config.method_id == "CSMC08-B8-S1-N16"
    assert config.as_dict()["kernel"] == COLLAPSED_SMC_KERNEL
    with pytest.raises(ValueError, match="strictly increasing"):
        CollapsedSmcConfig(0.10, 16, (0.0, 0.5, 0.5, 1.0))


def test_collapsed_smc_reference_numba_transcript_identity():
    H, model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome_from_epsilon(model, (0, 4))
    config = CollapsedSmcConfig(0.10, 12, collapsed_lambda_schedule(5))
    reference = run_collapsed_smc_population(
        model, frame, H, syndrome, config, _seed(config), engine="reference",
    )
    accelerated = run_collapsed_smc_population(
        model, frame, H, syndrome, config, _seed(config), engine="numba",
    )
    for name in reference:
        if name != "engine":
            assert np.array_equal(
                np.asarray(reference[name]), np.asarray(accelerated[name]),
            ), name
    assert np.allclose(reference["normalized_incremental_weights"].sum(axis=1), 1.0)
    for stage, parents in enumerate(reference["parent_indices"], start=1):
        assert np.array_equal(
            reference["roots_by_stage"][stage],
            reference["roots_by_stage"][stage - 1][parents],
        )


def test_small_smcpopulation_recovers_simple_exact_b_marginal():
    H, model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome_from_epsilon(model, (0,))
    config = CollapsedSmcConfig(0.10, 1024, collapsed_lambda_schedule(8))
    result = run_collapsed_smc_population(
        model, frame, H, syndrome, config, _seed(config, "row_major"), engine="numba",
    )
    from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
        build_classical_coset_mass,
    )

    mass = build_classical_coset_mass(H, config.p, engine="reference")
    b_states = _all_b_columns(1)
    a_syndromes = a_syndromes_from_b(syndrome, H, b_states)
    exact = _normalized_log_weights(
        collapsed_log_target(b_states, a_syndromes, config.p, np.log(mass))
    )
    observed = np.mean((result["b_columns_by_stage"][-1, :, 0] & np.uint32(1)) != 0)
    assert abs(float(observed) - float(exact[1])) <= 0.07
    replay_likelihood = collapsed_log_likelihood(
        a_syndromes_from_b(syndrome, H, result["b_columns_by_stage"][-1]), np.log(mass),
    )
    assert np.array_equal(replay_likelihood, result["log_likelihood_by_stage"][-1])
