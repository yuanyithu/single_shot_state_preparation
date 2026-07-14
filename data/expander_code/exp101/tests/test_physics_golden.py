"""Small-CSS golden tests for the exp101.physics.v2 reduced posterior."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.enumerate_exact import exact_reference
from src.gf2 import gf2_matmul, gf2_rank
from src.model import (
    DisorderRealization,
    assemble_sector_model,
    coupling_from_probability,
    wire_ensemble,
)
from src.observables import build_observable_frame


def toy_css_model():
    # rank(H_Z)=1, one X stabilizer, and one logical qubit.
    H_Z = np.array([[1, 1, 0]], dtype=np.uint8)
    H_X = np.array([[0, 0, 1]], dtype=np.uint8)
    logicals = SimpleNamespace(
        logical_X=np.array([[1, 1, 0]], dtype=np.uint8),
        logical_Z=np.array([[0, 1, 0]], dtype=np.uint8),
    )
    model = assemble_sector_model(
        H_X, H_Z, logicals, sector="x_error"
    )
    return model, build_observable_frame(model)


def test_incomplete_q_zero_move_set_is_rejected_at_model_assembly():
    """A missing kernel generator previously split exact and MCMC support."""
    H_Z = np.array([[1, 1, 0]], dtype=np.uint8)
    incomplete_H_X = np.zeros((0, 3), dtype=np.uint8)
    logicals = SimpleNamespace(
        logical_X=np.array([[1, 1, 0]], dtype=np.uint8),
        logical_Z=np.array([[0, 1, 0]], dtype=np.uint8),
    )
    missing_kernel_move = np.array([0, 0, 1], dtype=np.uint8)
    assert not gf2_matmul(H_Z, missing_kernel_move[:, None]).any()
    assert gf2_rank(logicals.logical_X) == 1
    assert H_Z.shape[1] - gf2_rank(H_Z) == 2

    # Exact q=0 enumeration would include the missing direction, while an
    # S+L MCMC built from these inputs could never traverse it.
    with pytest.raises(AssertionError, match=r"do not span ker\(H_check\)"):
        assemble_sector_model(
            incomplete_H_X, H_Z, logicals, sector="x_error"
        )


def test_logical_observable_must_annihilate_stabilizers():
    H_Z = np.array([[1, 1, 0]], dtype=np.uint8)
    H_X = np.array([[0, 0, 1]], dtype=np.uint8)
    malformed_logicals = SimpleNamespace(
        logical_X=np.array([[1, 1, 0]], dtype=np.uint8),
        logical_Z=np.array([[0, 1, 1]], dtype=np.uint8),
    )
    assert np.array_equal(
        gf2_matmul(
            malformed_logicals.logical_X,
            malformed_logicals.logical_Z.T,
        ),
        np.eye(1, dtype=np.uint8),
    )
    with pytest.raises(AssertionError, match="annihilate stabilizers"):
        assemble_sector_model(H_X, H_Z, malformed_logicals, sector="x_error")


def bits(value, width):
    return np.array(
        [(value >> index) & 1 for index in range(width)], dtype=np.uint8
    )


def bernoulli_probability(error_bits, probability):
    weight = int(np.sum(error_bits))
    return probability**weight * (1.0 - probability) ** (
        len(error_bits) - weight
    )


def test_raw_preparation_variables_equal_reduced_posterior_pointwise():
    """Enumerate sigma_prep, readout error, data truth, and every candidate."""
    model, frame = toy_css_model()
    p = 0.17
    q = 0.23
    K_p = coupling_from_probability(p)
    K_q = coupling_from_probability(q)

    for sigma_value in range(2):
        sigma_prep = bits(sigma_value, model.num_checks)
        c_prep = model.logical_sector_section.apply(sigma_prep)
        for measurement_value in range(2):
            measurement_error = bits(
                measurement_value, model.num_checks
            )
            s_prep = sigma_prep ^ measurement_error
            for epsilon_value in range(1 << model.num_qubits):
                epsilon_data_true = bits(
                    epsilon_value, model.num_qubits
                )
                F_total = c_prep ^ epsilon_data_true
                sigma_final = gf2_matmul(
                    model.H_check, F_total[:, None]
                )[:, 0]
                effective_syndrome = s_prep ^ sigma_final
                expected_effective = (
                    gf2_matmul(
                        model.H_check, epsilon_data_true[:, None]
                    )[:, 0]
                    ^ measurement_error
                )
                assert np.array_equal(
                    effective_syndrome, expected_effective
                )

                raw_sector_Z = np.zeros(1 << model.k)
                reduced_sector_Z = np.zeros(1 << model.k)
                for a_value in range(1 << model.num_qubits):
                    a = bits(a_value, model.num_qubits)
                    e = a ^ F_total
                    raw_data_weight = int(np.sum(a ^ F_total))
                    raw_syndrome_weight = int(np.sum(
                        gf2_matmul(model.H_check, a[:, None])[:, 0]
                        ^ s_prep
                    ))
                    reduced_data_weight = int(np.sum(e))
                    reduced_syndrome_weight = int(np.sum(
                        gf2_matmul(model.H_check, e[:, None])[:, 0]
                        ^ effective_syndrome
                    ))
                    raw_weight = np.exp(
                        -K_p * raw_data_weight
                        - K_q * raw_syndrome_weight
                    )
                    reduced_weight = np.exp(
                        -K_p * reduced_data_weight
                        - K_q * reduced_syndrome_weight
                    )
                    assert raw_weight == reduced_weight
                    raw_probability = bernoulli_probability(
                        a ^ F_total, p
                    ) * bernoulli_probability(
                        gf2_matmul(model.H_check, a[:, None])[:, 0]
                        ^ s_prep,
                        q,
                    )
                    reduced_probability = bernoulli_probability(
                        e, p
                    ) * bernoulli_probability(
                        gf2_matmul(model.H_check, e[:, None])[:, 0]
                        ^ effective_syndrome,
                        q,
                    )
                    assert raw_probability == reduced_probability
                    logical_class = int(frame.label_of(e)[0])
                    raw_sector_Z[logical_class] += raw_probability
                    reduced_sector_Z[logical_class] += reduced_probability

                assert np.array_equal(raw_sector_Z, reduced_sector_Z)
                assert np.sum(raw_sector_Z) == np.sum(reduced_sector_Z)
                raw_weights = raw_sector_Z / np.sum(raw_sector_Z)
                reduced_weights = reduced_sector_Z / np.sum(
                    reduced_sector_Z
                )
                assert np.array_equal(raw_weights, reduced_weights)
                raw_purity = float(np.sum(raw_weights**2))
                reduced_purity = float(np.sum(reduced_weights**2))
                assert 2 * raw_purity - 1 == 2 * reduced_purity - 1
                assert np.max(raw_weights) == np.max(reduced_weights)


def test_q_zero_true_is_quenched_coset_legacy_is_clean_kernel():
    model, frame = toy_css_model()
    epsilon_data_true = np.array([1, 0, 0], dtype=np.uint8)
    measurement_error = np.zeros(model.num_checks, dtype=np.uint8)
    effective_syndrome = gf2_matmul(
        model.H_check, epsilon_data_true[:, None]
    )[:, 0]
    disorder = DisorderRealization(
        epsilon_data_true=epsilon_data_true,
        measurement_error=measurement_error,
        effective_syndrome=effective_syndrome,
        p=0.13,
        q=0.0,
    )
    true = wire_ensemble(model, disorder, "true_posterior", frame)
    legacy = wire_ensemble(
        model, disorder, "legacy_delta_only", frame
    )
    assert np.array_equal(
        true.gibbs_syndrome_argument, effective_syndrome
    )
    assert not legacy.gibbs_syndrome_argument.any()

    true_result = exact_reference(model, frame, true)
    legacy_result = exact_reference(model, frame, legacy)
    assert true_result["table"].kind == "coset"
    assert legacy_result["table"].kind == "coset"
    assert not np.array_equal(
        true_result["table"].table, legacy_result["table"].table
    )


def test_exact_oracle_reports_distinct_v2_statistics_and_frames():
    model, frame = toy_css_model()
    epsilon_data_true = np.array([0, 1, 0], dtype=np.uint8)
    measurement_error = np.array([1], dtype=np.uint8)
    effective_syndrome = (
        gf2_matmul(model.H_check, epsilon_data_true[:, None])[:, 0]
        ^ measurement_error
    )
    disorder = DisorderRealization(
        epsilon_data_true=epsilon_data_true,
        measurement_error=measurement_error,
        effective_syndrome=effective_syndrome,
        p=0.19,
        q=0.11,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    result = exact_reference(model, frame, wiring)

    sign = -1.0 if wiring.planted_logical_class[0] else 1.0
    assert np.allclose(
        result["characters_relative"],
        sign * result["characters_absolute"],
    )
    assert result["q_top_absolute"] == result["q_top_relative"]
    assert np.isclose(
        result["posterior_mass_on_planted_class"],
        result["weights_absolute"][int(wiring.planted_logical_class[0])],
    )
    assert result["map_success_probability"] == np.max(
        result["weights_absolute"]
    )
    assert result["posterior_purity"] <= result[
        "map_success_probability"
    ]
    assert result["map_success_probability"] <= result[
        "map_success_algebraic_upper_bound"
    ]
    assert result["map_success_algebraic_lower_bound"] == result[
        "posterior_purity"
    ]
    assert result["map_success_estimated_lower_bound"] is None
    assert result["map_success_estimated_upper_bound"] is None
    assert result["map_success_bound_kind"] == "exact_posterior_algebraic"
    assert result["map_success_bound_has_confidence_coverage"] is False
    assert result["weights_are_exact_sector_posterior"] is True
    assert "w0" not in result
