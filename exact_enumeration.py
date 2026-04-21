import numpy as np

from build_toric_code_examples import (
    build_2d_toric_code,
    build_2d_toric_zero_syndrome_move_data,
    build_3d_toric_code,
    build_3d_toric_zero_syndrome_move_data,
)
from linear_section import build_linear_section
from main import (
    _build_kernel_basis_from_linear_section,
    _build_q0_initial_chain_bits_per_start,
    _run_single_disorder_measurement,
)
from mcmc import draw_disorder_sample
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


def _build_all_chain_bits(num_qubits):
    """
    构造 shape (2**n, n) 的全体 c ∈ F_2^n。
    """
    index_dtype = np.uint32 if num_qubits <= 32 else np.uint64
    configuration_indices = np.arange(1 << num_qubits, dtype=index_dtype)
    bit_positions = np.arange(num_qubits, dtype=index_dtype)
    all_chain_bits = (
        (configuration_indices[:, None] >> bit_positions[None, :]) & 1
    ).astype(bool)
    return all_chain_bits


def _logsumexp(log_values):
    """
    仅用 NumPy 实现的 logsumexp。
    """
    max_log_value = np.max(log_values)
    if not np.isfinite(max_log_value):
        return max_log_value
    return float(
        max_log_value + np.log(np.sum(np.exp(log_values - max_log_value)))
    )


def _compute_log_bernoulli_product(weights, length, probability):
    """
    计算 r^w (1-r)^(L-w) 的对数，边界点按 0^0 = 1 处理。
    """
    weights = np.asarray(weights)

    if probability == 0.0:
        return np.where(weights == 0, 0.0, -np.inf)
    if probability == 1.0:
        return np.where(weights == length, 0.0, -np.inf)

    return (
        weights.astype(np.float64) * np.log(probability)
        + (length - weights).astype(np.float64) * np.log1p(-probability)
    )


def compute_exact_logical_observable_means(
        parity_check_matrix,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability,
        logical_observable_masks):
    """
    枚举所有 2^n 个 c，精确计算 logical observable 的期望 m_u。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    if num_qubits > 22:
        raise ValueError("exact enumeration only supported for n ≤ 22")

    all_chain_bits = _build_all_chain_bits(num_qubits)
    all_chain_bits_uint8 = all_chain_bits.astype(np.uint8)
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    logical_observable_masks_uint8 = logical_observable_masks.astype(np.uint8)

    syndrome_term_bits = (
        all_chain_bits_uint8 @ parity_check_matrix_uint8.T
    ) % 2
    syndrome_term_bits = syndrome_term_bits.astype(bool)
    syndrome_term_bits ^= observed_syndrome_bits[None, :]

    data_term_bits = all_chain_bits ^ disorder_data_error_bits[None, :]

    syndrome_weights = np.count_nonzero(syndrome_term_bits, axis=1)
    data_weights = np.count_nonzero(data_term_bits, axis=1)

    log_weights = (
        _compute_log_bernoulli_product(
            weights=syndrome_weights,
            length=num_checks,
            probability=syndrome_error_probability,
        )
        + _compute_log_bernoulli_product(
            weights=data_weights,
            length=num_qubits,
            probability=data_error_probability,
        )
    )

    log_partition_function = _logsumexp(log_weights)
    if not np.isfinite(log_partition_function):
        raise ValueError(
            "posterior has zero total weight for the provided disorder sample"
        )

    normalized_probabilities = np.exp(log_weights - log_partition_function)

    logical_parity_bits = (
        all_chain_bits_uint8 @ logical_observable_masks_uint8.T
    ) % 2
    logical_observable_values = 1.0 - 2.0 * logical_parity_bits.astype(
        np.float64
    )

    return (
        normalized_probabilities @ logical_observable_values
    ).astype(np.float64, copy=False)


def _run_validation_case(
        test_name,
        parity_check_matrix,
        logical_observable_masks,
        checks_touching_each_qubit,
        kernel_basis,
        zero_syndrome_move_data,
        syndrome_error_probability,
        data_error_probability,
        seed,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        tolerance):
    """
    运行单组 exact-vs-MCMC 对比并打印结果。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    rng = np.random.default_rng(seed)

    (
        observed_syndrome_bits,
        disorder_data_error_bits,
    ) = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        rng=rng,
    )

    exact_m_u = compute_exact_logical_observable_means(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        logical_observable_masks=logical_observable_masks,
    )
    mcmc_m_u, _ = _run_single_disorder_measurement(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        rng=rng,
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
    )

    abs_diff = np.abs(exact_m_u - mcmc_m_u)
    pass_flags = abs_diff < tolerance

    for observable_index in range(logical_observable_masks.shape[0]):
        print(
            f"{test_name:<10} "
            f"{observable_index:>3d} "
            f"{exact_m_u[observable_index]:+10.4f} "
            f"{mcmc_m_u[observable_index]:+10.4f} "
            f"{abs_diff[observable_index]:10.4f} "
            f"{tolerance:10.4f} "
            f"{'PASS' if pass_flags[observable_index] else 'FAIL':>6}"
        )

    max_abs_diff = float(np.max(abs_diff))
    assert max_abs_diff < tolerance, (
        f"{test_name} failed: max(|exact - mcmc|) = {max_abs_diff:.6f} "
        f">= {tolerance:.6f}"
    )


def _run_zero_syndrome_move_structure_test(
        parity_check_matrix,
        zero_syndrome_move_data,
        lattice_size,
        expected_contractible_weight,
        expected_winding_weight):
    all_moves = np.concatenate(
        (
            zero_syndrome_move_data["contractible_moves"],
            zero_syndrome_move_data["winding_moves"],
        ),
        axis=0,
    )
    syndrome_bits = (
        parity_check_matrix.astype(np.uint8) @ all_moves.T.astype(np.uint8)
    ) % 2
    assert not np.any(syndrome_bits), (
        "q=0 move structure test failed: some moves are not in ker(H_Z)"
    )
    assert np.all(
        zero_syndrome_move_data["contractible_moves"].sum(axis=1)
        == expected_contractible_weight
    ), (
        "q=0 move structure test failed: local moves must have weight "
        f"{expected_contractible_weight}"
    )
    assert np.all(
        zero_syndrome_move_data["winding_moves"].sum(axis=1)
        == expected_winding_weight
    ), (
        "q=0 move structure test failed: winding moves must have weight "
        f"{expected_winding_weight}"
    )


def _run_3d_q0_sector_distinguishability_test():
    lattice_size = 2
    parity_check_matrix, dual_logical_z_basis = build_3d_toric_code(
        lattice_size=lattice_size
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )
    _run_zero_syndrome_move_structure_test(
        parity_check_matrix=parity_check_matrix,
        zero_syndrome_move_data=zero_syndrome_move_data,
        lattice_size=lattice_size,
        expected_contractible_weight=6,
        expected_winding_weight=lattice_size * lattice_size,
    )
    initial_chain_bits_per_start, start_sector_labels = (
        _build_q0_initial_chain_bits_per_start(
            observed_syndrome_bits=np.zeros(
                parity_check_matrix.shape[0],
                dtype=bool,
            ),
            linear_section_data=linear_section_data,
            zero_syndrome_move_data=zero_syndrome_move_data,
            q0_num_start_chains=8,
        )
    )
    assert len(np.unique(start_sector_labels)) == 8, (
        "3D q=0 start sector labels must enumerate 8 distinct sectors"
    )

    logical_signature_list = []
    for chain_bits in initial_chain_bits_per_start:
        logical_parity_bits = np.bitwise_xor.reduce(
            logical_observable_masks & chain_bits[None, :],
            axis=1,
        ).astype(np.int8, copy=False)
        logical_signature_list.append(tuple(logical_parity_bits.tolist()))

    assert len(set(logical_signature_list)) == 8, (
        "3D q=0 start sectors must induce distinct logical signatures"
    )


def _run_q0_multi_start_validation_case(
        test_name,
        parity_check_matrix,
        logical_observable_masks,
        checks_touching_each_qubit,
        linear_section_data,
        zero_syndrome_move_data,
        data_error_probability,
        seed,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        tolerance):
    """
    q=0 下对合法初态分别做 exact-vs-MCMC 回归。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    rng = np.random.default_rng(seed)

    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=0.0,
        data_error_probability=data_error_probability,
        rng=rng,
    )
    exact_m_u = compute_exact_logical_observable_means(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=0.0,
        data_error_probability=data_error_probability,
        logical_observable_masks=logical_observable_masks,
    )
    initial_chain_bits_per_start, start_sector_labels = (
        _build_q0_initial_chain_bits_per_start(
            observed_syndrome_bits=observed_syndrome_bits,
            linear_section_data=linear_section_data,
            zero_syndrome_move_data=zero_syndrome_move_data,
            q0_num_start_chains=(
                1
                << zero_syndrome_move_data["start_sector_generators"].shape[0]
            ),
        )
    )

    for start_index, start_sector_label in enumerate(start_sector_labels):
        start_rng = np.random.default_rng(seed + 100 + start_index)
        mcmc_m_u, _ = _run_single_disorder_measurement(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=0.0,
            data_error_probability=data_error_probability,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=num_burn_in_sweeps,
            num_measurements_per_disorder=num_measurements_per_disorder,
            num_sweeps_between_measurements=num_sweeps_between_measurements,
            rng=start_rng,
            zero_syndrome_move_data=zero_syndrome_move_data,
            initial_chain_bits=initial_chain_bits_per_start[start_index],
        )

        abs_diff = np.abs(exact_m_u - mcmc_m_u)
        pass_flags = abs_diff < tolerance
        for observable_index in range(logical_observable_masks.shape[0]):
            print(
                f"{test_name}:{start_sector_label:<2} "
                f"{observable_index:>3d} "
                f"{exact_m_u[observable_index]:+10.4f} "
                f"{mcmc_m_u[observable_index]:+10.4f} "
                f"{abs_diff[observable_index]:10.4f} "
                f"{tolerance:10.4f} "
                f"{'PASS' if pass_flags[observable_index] else 'FAIL':>6}"
            )

        max_abs_diff = float(np.max(abs_diff))
        assert max_abs_diff < tolerance, (
            f"{test_name}:{start_sector_label} failed: "
            f"max(|exact - mcmc|) = {max_abs_diff:.6f} "
            f">= {tolerance:.6f}"
        )


if __name__ == "__main__":
    lattice_size = 3
    num_burn_in_sweeps = 1000
    num_measurements_per_disorder = 5000
    num_sweeps_between_measurements = 2
    tolerance = 4.0 / np.sqrt(num_measurements_per_disorder)

    parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
        lattice_size=lattice_size
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    kernel_basis = _build_kernel_basis_from_linear_section(
        parity_check_matrix=parity_check_matrix,
        linear_section_data=linear_section_data,
    )
    zero_syndrome_move_data = build_2d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )
    _run_zero_syndrome_move_structure_test(
        parity_check_matrix=parity_check_matrix,
        zero_syndrome_move_data=zero_syndrome_move_data,
        lattice_size=lattice_size,
        expected_contractible_weight=4,
        expected_winding_weight=lattice_size,
    )
    print("Test 0 passed: q=0 move structure")
    _run_3d_q0_sector_distinguishability_test()
    print("Test 0b passed: 3D q=0 sector distinguishability")

    print(
        f"{'test_name':<10} "
        f"{'u':>3} "
        f"{'exact_m_u':>10} "
        f"{'mcmc_m_u':>10} "
        f"{'abs_diff':>10} "
        f"{'tolerance':>10} "
        f"{'pass':>6}"
    )

    _run_validation_case(
        test_name="Test 1",
        parity_check_matrix=parity_check_matrix,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        kernel_basis=kernel_basis,
        zero_syndrome_move_data=None,
        syndrome_error_probability=0.05,
        data_error_probability=0.08,
        seed=20240701,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        tolerance=tolerance,
    )
    _run_validation_case(
        test_name="Test 2",
        parity_check_matrix=parity_check_matrix,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        kernel_basis=kernel_basis,
        zero_syndrome_move_data=zero_syndrome_move_data,
        syndrome_error_probability=0.0,
        data_error_probability=0.08,
        seed=20240702,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        tolerance=tolerance,
    )
    _run_validation_case(
        test_name="Test 3",
        parity_check_matrix=parity_check_matrix,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        kernel_basis=kernel_basis,
        zero_syndrome_move_data=None,
        syndrome_error_probability=0.02,
        data_error_probability=0.02,
        seed=20240703,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        tolerance=tolerance,
    )
    _run_q0_multi_start_validation_case(
        test_name="Test 4",
        parity_check_matrix=parity_check_matrix,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        linear_section_data=linear_section_data,
        zero_syndrome_move_data=zero_syndrome_move_data,
        data_error_probability=0.08,
        seed=20240704,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        tolerance=tolerance,
    )

    print("All exact-enumeration validations passed.")
