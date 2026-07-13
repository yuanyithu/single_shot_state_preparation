"""Golden checks for the paper preparation variables and reduced posterior."""

import itertools

import numpy as np

from src.enumerate_exact import exact_reference
from src.gf2 import gf2_matmul, gf2_nullspace
from src.graphs import repetition_parity_check_matrix
from src.hgp import hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import (
    DisorderRealization,
    STATE_PREP_PROTOCOL,
    assemble_sector_model,
    coupling_from_probability,
    wire_ensemble,
)
from src.observables import build_observable_frame, characters_from_sector_weights
from src.reference_mcmc import (
    McmcState,
    single_bit_log_acceptance,
    support_move_log_acceptance,
)


def _bits(value, length):
    return np.array([(value >> bit) & 1 for bit in range(length)], dtype=np.uint8)


def _bernoulli_mass(bits, probability):
    weight = int(np.asarray(bits, dtype=np.uint8).sum())
    length = int(np.asarray(bits).size)
    return probability**weight * (1.0 - probability) ** (length - weight)


def _label_bits(frame, vector):
    return sum(
        int(value) << bit for bit, value in enumerate(frame.label_of(vector))
    )


def _setup():
    H_Z, H_X = hgp_from_H(repetition_parity_check_matrix(2))
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return H_Z, H_X, logicals, model, build_observable_frame(model)


def _physical_syndromes(model):
    syndromes = {
        tuple(gf2_matmul(model.H_check, _bits(value, model.num_qubits)[:, None])[:, 0])
        for value in range(1 << model.num_qubits)
    }
    return [np.asarray(syndrome, dtype=np.uint8) for syndrome in sorted(syndromes)]


def _q_top(weights):
    purity = float(np.sum(np.asarray(weights) ** 2))
    count = len(weights)
    return (count * purity - 1.0) / (count - 1.0)


def test_raw_paper_and_reduced_posterior_match_for_every_physical_variable():
    """Enumerate preparation syndromes, all representatives, noise, and a."""
    _, _, _, model, frame = _setup()
    p, q = 0.17, 0.13
    max_pointwise_error = 0.0
    max_partition_error = 0.0
    max_sector_error = 0.0
    max_q_top_error = 0.0
    max_map_error = 0.0

    kernel = gf2_nullspace(model.H_check)
    for sigma_prep in _physical_syndromes(model):
        base_chain = model.logical_sector_section.apply(sigma_prep)
        for kernel_value in range(1 << kernel.shape[0]):
            preparation_chain = base_chain.copy()
            for bit, generator in enumerate(kernel):
                if (kernel_value >> bit) & 1:
                    preparation_chain ^= generator
            assert np.array_equal(
                gf2_matmul(model.H_check, preparation_chain[:, None])[:, 0],
                sigma_prep,
            )
            for delta_value, epsilon_value in itertools.product(
                range(1 << model.num_checks), range(1 << model.num_qubits)
            ):
                measurement_error = _bits(delta_value, model.num_checks)
                epsilon_data_true = _bits(epsilon_value, model.num_qubits)
                s_prep = sigma_prep ^ measurement_error
                F_total = preparation_chain ^ epsilon_data_true
                sigma_final = gf2_matmul(model.H_check, F_total[:, None])[:, 0]
                effective_syndrome = s_prep ^ sigma_final
                assert np.array_equal(
                    effective_syndrome,
                    gf2_matmul(
                        model.H_check, epsilon_data_true[:, None]
                    )[:, 0] ^ measurement_error,
                )

                raw_sector = np.zeros(1 << model.k, dtype=np.float64)
                reduced_sector = np.zeros_like(raw_sector)
                raw_partition = 0.0
                reduced_partition = 0.0
                for a_value in range(1 << model.num_qubits):
                    a = _bits(a_value, model.num_qubits)
                    e = a ^ F_total
                    raw_readout_residual = (
                        gf2_matmul(model.H_check, a[:, None])[:, 0] ^ s_prep
                    )
                    reduced_readout_residual = (
                        gf2_matmul(model.H_check, e[:, None])[:, 0]
                        ^ effective_syndrome
                    )
                    raw_weight = _bernoulli_mass(
                        raw_readout_residual, q
                    ) * _bernoulli_mass(a ^ F_total, p)
                    reduced_weight = _bernoulli_mass(e, p) * _bernoulli_mass(
                        reduced_readout_residual, q
                    )
                    max_pointwise_error = max(
                        max_pointwise_error, abs(raw_weight - reduced_weight)
                    )
                    label = _label_bits(frame, e)
                    raw_sector[label] += raw_weight
                    reduced_sector[label] += reduced_weight
                    raw_partition += raw_weight
                    reduced_partition += reduced_weight

                max_partition_error = max(
                    max_partition_error, abs(raw_partition - reduced_partition)
                )
                max_sector_error = max(
                    max_sector_error,
                    float(np.max(np.abs(raw_sector - reduced_sector))),
                )
                raw_probability = raw_sector / raw_partition
                reduced_probability = reduced_sector / reduced_partition
                max_q_top_error = max(
                    max_q_top_error,
                    abs(_q_top(raw_probability) - _q_top(reduced_probability)),
                )
                max_map_error = max(
                    max_map_error,
                    abs(
                        float(raw_probability.max())
                        - float(reduced_probability.max())
                    ),
                )

    assert max_pointwise_error < 1e-15
    assert max_partition_error < 1e-14
    assert max_sector_error < 1e-14
    assert max_q_top_error < 1e-14
    assert max_map_error < 1e-14


def test_q_zero_true_is_quenched_coset_and_legacy_is_clean_kernel():
    _, _, _, model, frame = _setup()
    epsilon_data_true = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon_data_true[0] = 1
    measurement_error = np.zeros(model.num_checks, dtype=np.uint8)
    effective_syndrome = gf2_matmul(
        model.H_check, epsilon_data_true[:, None]
    )[:, 0]
    assert effective_syndrome.any()
    disorder = DisorderRealization(
        epsilon_data_true=epsilon_data_true,
        measurement_error=measurement_error,
        effective_syndrome=effective_syndrome,
        p=0.17,
        q=0.0,
    )
    true_wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    legacy_wiring = wire_ensemble(
        model, disorder, "legacy_delta_only", frame
    )
    zero = np.zeros(model.num_qubits, dtype=np.uint8)

    assert np.array_equal(
        true_wiring.gibbs_syndrome_argument, effective_syndrome
    )
    assert np.array_equal(
        legacy_wiring.gibbs_syndrome_argument, measurement_error
    )
    assert np.isfinite(true_wiring.total_energy(model, epsilon_data_true))
    assert np.isfinite(legacy_wiring.total_energy(model, zero))
    with np.testing.assert_raises_regex(ValueError, "hard constraint"):
        true_wiring.total_energy(model, zero)
    with np.testing.assert_raises_regex(ValueError, "hard constraint"):
        legacy_wiring.total_energy(model, epsilon_data_true)

    legacy_result = exact_reference(
        model, frame, legacy_wiring, force_python=True
    )
    assert legacy_result["posterior_purity"] is None
    assert legacy_result["posterior_mass_on_planted_class"] is None
    assert legacy_result["map_success_probability"] is None
    assert np.isfinite(legacy_result["formal_sector_purity"])
    assert np.isfinite(legacy_result["largest_sector_mass"])


def test_x_error_matrix_wiring_is_the_logical_plus_preparation_convention():
    H_Z, H_X, logicals, model, _ = _setup()
    assert STATE_PREP_PROTOCOL == "plus_Zcheck_X"
    assert model.sector == "x_error"
    assert np.array_equal(model.H_check, H_Z)
    assert np.array_equal(model.stabilizer_rows, H_X)
    assert np.array_equal(model.logical_move_basis, logicals.logical_X)
    assert np.array_equal(model.logical_obs_basis, logicals.logical_Z)


def test_z_error_matrix_wiring_is_the_hadamard_dual_convention():
    H_Z, H_X, logicals, _, _ = _setup()
    model = assemble_sector_model(H_X, H_Z, logicals, sector="z_error")
    assert model.sector == "z_error"
    assert np.array_equal(model.H_check, H_X)
    assert np.array_equal(model.stabilizer_rows, H_Z)
    assert np.array_equal(model.logical_move_basis, logicals.logical_Z)
    assert np.array_equal(model.logical_obs_basis, logicals.logical_X)
    assert np.array_equal(
        gf2_matmul(model.logical_move_basis, model.logical_obs_basis.T),
        np.eye(model.k, dtype=np.uint8),
    )


def test_shifted_coordinate_energy_identity_is_pointwise_exact():
    """The paper variable x=e xor epsilon reduces to the canonical e."""
    _, _, _, model, _ = _setup()
    K_p = coupling_from_probability(0.19)
    K_q = coupling_from_probability(0.11)
    for epsilon_value, measurement_value, e_value in itertools.product(
        range(1 << model.num_qubits),
        range(1 << model.num_checks),
        range(1 << model.num_qubits),
    ):
        epsilon_data_true = _bits(epsilon_value, model.num_qubits)
        measurement_error = _bits(measurement_value, model.num_checks)
        e = _bits(e_value, model.num_qubits)
        x = e ^ epsilon_data_true
        effective_syndrome = (
            gf2_matmul(model.H_check, epsilon_data_true[:, None])[:, 0]
            ^ measurement_error
        )
        raw_energy = (
            K_p * int(np.sum(x ^ epsilon_data_true))
            + K_q * int(np.sum(
                gf2_matmul(model.H_check, x[:, None])[:, 0]
                ^ measurement_error
            ))
        )
        reduced_energy = (
            K_p * int(np.sum(e))
            + K_q * int(np.sum(
                gf2_matmul(model.H_check, e[:, None])[:, 0]
                ^ effective_syndrome
            ))
        )
        assert raw_energy == reduced_energy


def test_fixed_effective_syndrome_is_truth_independent_off_kernel():
    """Changing H epsilon while compensating measurement leaves Gibbs inputs."""
    _, _, _, model, frame = _setup()
    effective_syndrome = _bits(2, model.num_checks)
    epsilon_a = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon_b = epsilon_a.copy()
    supported_qubit = int(np.flatnonzero(model.H_check.any(axis=0))[0])
    epsilon_b[supported_qubit] = 1
    H_epsilon_b = gf2_matmul(model.H_check, epsilon_b[:, None])[:, 0]
    assert H_epsilon_b.any()
    measurement_a = effective_syndrome.copy()
    measurement_b = effective_syndrome ^ H_epsilon_b
    disorder_a = DisorderRealization(
        epsilon_data_true=epsilon_a,
        measurement_error=measurement_a,
        effective_syndrome=effective_syndrome,
        p=0.19,
        q=0.11,
    )
    disorder_b = DisorderRealization(
        epsilon_data_true=epsilon_b,
        measurement_error=measurement_b,
        effective_syndrome=effective_syndrome,
        p=0.19,
        q=0.11,
    )
    wiring_a = wire_ensemble(model, disorder_a, "true_posterior", frame)
    wiring_b = wire_ensemble(model, disorder_b, "true_posterior", frame)
    assert not np.array_equal(measurement_a, measurement_b)
    assert np.array_equal(
        wiring_a.gibbs_syndrome_argument,
        wiring_b.gibbs_syndrome_argument,
    )
    for value in range(1 << model.num_qubits):
        e = _bits(value, model.num_qubits)
        assert wiring_a.total_energy(model, e) == wiring_b.total_energy(model, e)
        syndrome_term = (
            gf2_matmul(model.H_check, e[:, None])[:, 0]
            ^ effective_syndrome
        ).astype(np.uint8)
        state = McmcState(
            v=e,
            syndrome_term=syndrome_term,
            data_weight=int(e.sum()),
            syndrome_weight=int(syndrome_term.sum()),
        )
        for qubit in range(model.num_qubits):
            assert single_bit_log_acceptance(
                model, wiring_a, state, qubit
            ) == single_bit_log_acceptance(model, wiring_b, state, qubit)
        for row in np.vstack((model.stabilizer_rows, model.logical_move_basis)):
            support = np.flatnonzero(row)
            assert support_move_log_acceptance(
                wiring_a, state, support
            ) == support_move_log_acceptance(wiring_b, state, support)


def test_canonical_energy_matches_probability_weight_up_to_one_constant():
    """The energy reads y_eff and e, while truth affects only the Mattis label."""
    _, _, _, model, frame = _setup()
    epsilon_data_true = _bits(3, model.num_qubits)
    measurement_error = _bits(1, model.num_checks)
    effective_syndrome = (
        gf2_matmul(model.H_check, epsilon_data_true[:, None])[:, 0]
        ^ measurement_error
    )
    p, q = 0.19, 0.11
    disorder = DisorderRealization(
        epsilon_data_true=epsilon_data_true,
        measurement_error=measurement_error,
        effective_syndrome=effective_syndrome,
        p=p,
        q=q,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    normalizing_constant = (1.0 - p) ** model.num_qubits * (
        1.0 - q
    ) ** model.num_checks
    for value in range(1 << model.num_qubits):
        e = _bits(value, model.num_qubits)
        residual = gf2_matmul(model.H_check, e[:, None])[:, 0] ^ effective_syndrome
        probability_weight = _bernoulli_mass(e, p) * _bernoulli_mass(residual, q)
        energy_weight = np.exp(-wiring.total_energy(model, e))
        assert np.isclose(
            probability_weight, normalizing_constant * energy_weight, atol=1e-15
        )
    assert wiring.K_p == coupling_from_probability(p)
    assert wiring.K_q == coupling_from_probability(q)


def test_character_table_preserves_q_top_under_planted_translation():
    weights_absolute = np.array([0.05, 0.15, 0.10, 0.70])
    planted_class = 2
    weights_relative = np.array(
        [weights_absolute[label ^ planted_class] for label in range(4)]
    )
    absolute = characters_from_sector_weights(weights_absolute)
    relative = characters_from_sector_weights(weights_relative)
    signs = np.array([
        -1.0 if (mask & planted_class).bit_count() & 1 else 1.0
        for mask in range(1, 4)
    ])
    assert np.allclose(relative, signs * absolute)
    assert np.isclose(np.mean(relative**2), np.mean(absolute**2))
    assert weights_absolute.max() == weights_relative.max()


def test_exact_oracle_exposes_complete_v2_posterior_statistics():
    _, _, _, model, frame = _setup()
    epsilon_data_true = _bits(7, model.num_qubits)
    measurement_error = _bits(2, model.num_checks)
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
    result = exact_reference(model, frame, wiring, force_python=True)

    required = {
        "weights_absolute",
        "weights_relative",
        "characters_absolute",
        "characters_relative",
        "posterior_purity",
        "posterior_mass_on_planted_class",
        "map_success_probability",
        "map_success_lower_bound",
        "map_success_upper_bound",
        "q_top",
        "q_top_absolute",
        "q_top_relative",
    }
    assert required <= result.keys()
    assert "w0" not in result
    assert np.isclose(result["weights_absolute"].sum(), 1.0)
    assert np.isclose(result["weights_relative"].sum(), 1.0)
    assert np.isclose(
        result["posterior_purity"],
        np.sum(result["weights_absolute"] ** 2),
    )
    assert result["posterior_purity"] <= result["map_success_probability"]
    assert (
        result["map_success_probability"]
        <= result["map_success_upper_bound"]
    )
    assert np.isclose(result["q_top_absolute"], result["q_top_relative"])
