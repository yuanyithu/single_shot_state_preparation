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
    run_disorder_average_simulation,
    run_q_positive_single_bit_acceptance_bruteforce_test,
)
from mcmc_diagnostics import equal_log_odds_ladder
from mcmc import draw_disorder_sample
from mcmc_parallel_tempering import run_parallel_tempering_measurement
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


DEFAULT_EXACT_ENUMERATION_CHUNK_SIZE = 1 << 18


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


def _iter_chain_bit_chunks(num_qubits, chunk_size):
    """
    分块生成全体 c ∈ F_2^n，避免一次性分配 2^n × n 矩阵。
    """
    index_dtype = np.uint32 if num_qubits <= 32 else np.uint64
    total_num_configurations = 1 << num_qubits
    bit_positions = np.arange(num_qubits, dtype=index_dtype)

    for chunk_start in range(0, total_num_configurations, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, total_num_configurations)
        configuration_indices = np.arange(
            chunk_start,
            chunk_stop,
            dtype=index_dtype,
        )
        yield (
            (configuration_indices[:, None] >> bit_positions[None, :]) & 1
        ).astype(bool)


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
        logical_observable_masks,
        chunk_size=DEFAULT_EXACT_ENUMERATION_CHUNK_SIZE):
    """
    分块枚举所有 2^n 个 c，精确计算 m_u、q_top 与 posterior normalization。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    logical_observable_masks_uint8 = logical_observable_masks.astype(np.uint8)

    max_log_weight = -np.inf
    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, chunk_size):
        chain_bits_chunk_uint8 = chain_bits_chunk.astype(np.uint8)
        syndrome_term_bits = (
            chain_bits_chunk_uint8 @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_term_bits = syndrome_term_bits.astype(bool)
        syndrome_term_bits ^= observed_syndrome_bits[None, :]

        data_term_bits = chain_bits_chunk ^ disorder_data_error_bits[None, :]
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
        max_log_weight = max(max_log_weight, float(np.max(log_weights)))

    if not np.isfinite(max_log_weight):
        raise ValueError(
            "posterior has zero total weight for the provided disorder sample"
        )

    weighted_observable_sum = np.zeros(
        logical_observable_masks.shape[0],
        dtype=np.float64,
    )
    normalized_weight_sum = 0.0

    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, chunk_size):
        chain_bits_chunk_uint8 = chain_bits_chunk.astype(np.uint8)
        syndrome_term_bits = (
            chain_bits_chunk_uint8 @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_term_bits = syndrome_term_bits.astype(bool)
        syndrome_term_bits ^= observed_syndrome_bits[None, :]

        data_term_bits = chain_bits_chunk ^ disorder_data_error_bits[None, :]
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
        scaled_weights = np.exp(log_weights - max_log_weight)
        logical_parity_bits = (
            chain_bits_chunk_uint8 @ logical_observable_masks_uint8.T
        ) % 2
        logical_observable_values = (
            1.0 - 2.0 * logical_parity_bits.astype(np.float64)
        )
        weighted_observable_sum += scaled_weights @ logical_observable_values
        normalized_weight_sum += float(np.sum(scaled_weights))

    m_u_values = (
        weighted_observable_sum / normalized_weight_sum
    ).astype(np.float64, copy=False)
    log_partition_function = float(
        max_log_weight + np.log(normalized_weight_sum)
    )
    normalized_probability_sum = float(
        np.exp(np.log(normalized_weight_sum) + max_log_weight
               - log_partition_function)
    )
    return {
        "m_u_values": m_u_values,
        "q_top_value": float(np.mean(m_u_values ** 2)),
        "log_partition_function": log_partition_function,
        "normalized_probability_sum": normalized_probability_sum,
    }


def compute_exact_logical_sector_weights(
        parity_check_matrix,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability,
        logical_observable_masks,
        num_primitive_logical_qubits,
        chunk_size=DEFAULT_EXACT_ENUMERATION_CHUNK_SIZE):
    """
    分块枚举所有 2^n 个 c，精确计算 primitive logical signature 的后验权重。
    """
    if num_primitive_logical_qubits < 1:
        raise ValueError("num_primitive_logical_qubits must be >= 1")
    if logical_observable_masks.shape[0] < num_primitive_logical_qubits:
        raise ValueError(
            "logical_observable_masks does not contain enough primitive masks"
        )

    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    primitive_masks_uint8 = logical_observable_masks[
        :num_primitive_logical_qubits
    ].astype(np.uint8)
    num_signatures = 1 << num_primitive_logical_qubits

    max_log_weight = -np.inf
    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, chunk_size):
        chain_bits_chunk_uint8 = chain_bits_chunk.astype(np.uint8)
        syndrome_term_bits = (
            chain_bits_chunk_uint8 @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_term_bits = syndrome_term_bits.astype(bool)
        syndrome_term_bits ^= observed_syndrome_bits[None, :]

        data_term_bits = chain_bits_chunk ^ disorder_data_error_bits[None, :]
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
        max_log_weight = max(max_log_weight, float(np.max(log_weights)))

    if not np.isfinite(max_log_weight):
        raise ValueError(
            "posterior has zero total weight for the provided disorder sample"
        )

    signature_weight_sums = np.zeros(num_signatures, dtype=np.float64)
    normalized_weight_sum = 0.0

    signature_bit_weights = (1 << np.arange(
        num_primitive_logical_qubits,
        dtype=np.int64,
    ))
    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, chunk_size):
        chain_bits_chunk_uint8 = chain_bits_chunk.astype(np.uint8)
        syndrome_term_bits = (
            chain_bits_chunk_uint8 @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_term_bits = syndrome_term_bits.astype(bool)
        syndrome_term_bits ^= observed_syndrome_bits[None, :]

        data_term_bits = chain_bits_chunk ^ disorder_data_error_bits[None, :]
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
        scaled_weights = np.exp(log_weights - max_log_weight)
        primitive_logical_parity_bits = (
            chain_bits_chunk_uint8 @ primitive_masks_uint8.T
        ) % 2
        signature_indices = primitive_logical_parity_bits.astype(
            np.int64
        ) @ signature_bit_weights
        np.add.at(signature_weight_sums, signature_indices, scaled_weights)
        normalized_weight_sum += float(np.sum(scaled_weights))

    signature_probabilities = (
        signature_weight_sums / normalized_weight_sum
    ).astype(np.float64, copy=False)
    return {
        "signature_probabilities": signature_probabilities,
        "signature_labels": np.array(
            [
                "".join(
                    "1" if (signature_index >> bit_index) & 1 else "0"
                    for bit_index in range(num_primitive_logical_qubits)
                )
                for signature_index in range(num_signatures)
            ]
        ),
        "normalized_probability_sum": float(
            np.sum(signature_probabilities)
        ),
    }


def _print_exact_vs_mcmc_table(
        test_name,
        exact_m_u,
        mcmc_m_u,
        tolerance):
    abs_diff = np.abs(exact_m_u - mcmc_m_u)
    pass_flags = abs_diff < tolerance

    for observable_index in range(exact_m_u.shape[0]):
        print(
            f"{test_name:<18} "
            f"{observable_index:>3d} "
            f"{exact_m_u[observable_index]:+10.4f} "
            f"{mcmc_m_u[observable_index]:+10.4f} "
            f"{abs_diff[observable_index]:10.4f} "
            f"{tolerance:10.4f} "
            f"{'PASS' if pass_flags[observable_index] else 'FAIL':>6}"
        )


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

    exact_result = compute_exact_logical_observable_means(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        logical_observable_masks=logical_observable_masks,
    )
    exact_m_u = exact_result["m_u_values"]
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

    _print_exact_vs_mcmc_table(
        test_name=test_name,
        exact_m_u=exact_m_u,
        mcmc_m_u=mcmc_m_u,
        tolerance=tolerance,
    )

    max_abs_diff = float(np.max(np.abs(exact_m_u - mcmc_m_u)))
    assert max_abs_diff < tolerance, (
        f"{test_name} failed: max(|exact - mcmc|) = {max_abs_diff:.6f} "
        f">= {tolerance:.6f}"
    )


def _run_zero_syndrome_move_structure_test(
        parity_check_matrix,
        zero_syndrome_move_data,
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
    )["m_u_values"]
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

        _print_exact_vs_mcmc_table(
            test_name=f"{test_name}:{start_sector_label}",
            exact_m_u=exact_m_u,
            mcmc_m_u=mcmc_m_u,
            tolerance=tolerance,
        )

        max_abs_diff = float(np.max(np.abs(exact_m_u - mcmc_m_u)))
        assert max_abs_diff < tolerance, (
            f"{test_name}:{start_sector_label} failed: "
            f"max(|exact - mcmc|) = {max_abs_diff:.6f} "
            f">= {tolerance:.6f}"
        )


def _run_3d_q_positive_exact_vs_mcmc_regression():
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
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )

    num_burn_in_sweeps = 2000
    num_measurements_per_disorder = 12000
    num_sweeps_between_measurements = 2
    tolerance = 5.0 / np.sqrt(num_measurements_per_disorder)
    base_seed = 20260422

    # Original grid, kept for backward-compat coverage
    base_grid = [
        (q, p)
        for q in (1e-3, 1e-2, 4e-2)
        for p in (0.05, 0.15, 0.25)
    ]
    # Extension (plan B1): cover the problematic physics ballpark p∈[0.20, 0.28]
    # at q=5e-3 — this is where the L>=3 non-PT scans are unreliable, so we want
    # a hard exact anchor at L=2.
    extension_grid = [
        (5e-3, p) for p in (0.20, 0.22, 0.24, 0.26, 0.28)
    ]
    regression_grid = base_grid + extension_grid

    print("\n3D q>0 exact-vs-MCMC regression")
    for point_index, (
            syndrome_error_probability,
            data_error_probability) in enumerate(regression_grid):
        seed = base_seed + point_index
        disorder_rng = np.random.default_rng(seed)
        observed_syndrome_bits, disorder_data_error_bits = (
            draw_disorder_sample(
                num_checks=parity_check_matrix.shape[0],
                num_qubits=parity_check_matrix.shape[1],
                syndrome_error_probability=syndrome_error_probability,
                data_error_probability=data_error_probability,
                rng=disorder_rng,
            )
        )
        exact_result = compute_exact_logical_observable_means(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability=data_error_probability,
            logical_observable_masks=logical_observable_masks,
        )
        assert np.isclose(
            exact_result["normalized_probability_sum"],
            1.0,
            atol=1e-12,
            rtol=0.0,
        ), (
            "posterior normalization failed for 3D q>0 regression: "
            f"L={lattice_size}, p={data_error_probability}, "
            f"q={syndrome_error_probability}, seed={seed}"
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
            num_sweeps_between_measurements=(
                num_sweeps_between_measurements
            ),
            rng=np.random.default_rng(seed),
            zero_syndrome_move_data=zero_syndrome_move_data,
        )

        exact_m_u = exact_result["m_u_values"]
        _print_exact_vs_mcmc_table(
            test_name="3D-q>0-regression",
            exact_m_u=exact_m_u,
            mcmc_m_u=mcmc_m_u,
            tolerance=tolerance,
        )
        print(
            "3D-q>0-summary: "
            f"L={lattice_size}, p={data_error_probability:.4g}, "
            f"q={syndrome_error_probability:.4g}, seed={seed}, "
            f"logZ={exact_result['log_partition_function']:.10f}, "
            f"norm_sum={exact_result['normalized_probability_sum']:.10f}, "
            f"q_top_exact={exact_result['q_top_value']:.6f}, "
            f"q_top_mcmc={float(np.mean(mcmc_m_u ** 2)):.6f}"
        )
        max_abs_diff = float(np.max(np.abs(exact_m_u - mcmc_m_u)))
        q_top_abs_diff = abs(
            exact_result["q_top_value"] - float(np.mean(mcmc_m_u ** 2))
        )
        assert max_abs_diff < tolerance, (
            "3D q>0 exact-vs-MCMC m_u regression failed: "
            f"L={lattice_size}, p={data_error_probability}, "
            f"q={syndrome_error_probability}, seed={seed}, "
            f"max_abs_diff={max_abs_diff:.6f}, tol={tolerance:.6f}"
        )
        assert q_top_abs_diff < tolerance, (
            "3D q>0 exact-vs-MCMC q_top regression failed: "
            f"L={lattice_size}, p={data_error_probability}, "
            f"q={syndrome_error_probability}, seed={seed}, "
            f"q_top_abs_diff={q_top_abs_diff:.6f}, tol={tolerance:.6f}"
        )


def _run_3d_q_positive_exact_vs_pt_regression():
    """
    在 L=2 上验证 parallel tempering 的冷链 marginal 与 exact posterior 一致。
    直接验证 [src/mcmc_parallel_tempering.py](src/mcmc_parallel_tempering.py) 的
    swap 公式与 accumulation 实现正确性。

    只覆盖几个代表性 (q, p_cold) 点。每个点用 equal-log-odds 9-温度 ladder，
    冷端为 p_cold，热端 p_hot=0.44（physics 目标 ladder 与 c2 suite 一致）。
    """
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
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )

    num_burn_in_sweeps = 2000
    num_measurements = 8000
    num_sweeps_between_measurements = 2
    num_temperatures = 9
    p_hot = 0.44
    tolerance = 5.0 / np.sqrt(num_measurements)
    base_seed = 20260423

    regression_grid = [
        (5e-3, 0.22),
        (5e-3, 0.26),
        (1e-3, 0.10),
    ]

    print("\n3D q>0 exact-vs-PT regression")
    for point_index, (
            syndrome_error_probability,
            p_cold) in enumerate(regression_grid):
        seed = base_seed + point_index
        disorder_rng = np.random.default_rng(seed)
        observed_syndrome_bits, disorder_data_error_bits = (
            draw_disorder_sample(
                num_checks=parity_check_matrix.shape[0],
                num_qubits=parity_check_matrix.shape[1],
                syndrome_error_probability=syndrome_error_probability,
                data_error_probability=p_cold,
                rng=disorder_rng,
            )
        )
        ladder = equal_log_odds_ladder(
            p_cold=p_cold,
            p_hot=p_hot,
            num_temperatures=num_temperatures,
        )

        pt_result = run_parallel_tempering_measurement(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability_ladder=ladder,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=num_burn_in_sweeps,
            num_measurements=num_measurements,
            num_sweeps_between_measurements=(
                num_sweeps_between_measurements
            ),
            rng=np.random.default_rng(seed),
            zero_syndrome_move_data=zero_syndrome_move_data,
        )

        for t_index, p_value in enumerate(ladder):
            exact_result = compute_exact_logical_observable_means(
                parity_check_matrix=parity_check_matrix,
                observed_syndrome_bits=observed_syndrome_bits,
                disorder_data_error_bits=disorder_data_error_bits,
                syndrome_error_probability=syndrome_error_probability,
                data_error_probability=float(p_value),
                logical_observable_masks=logical_observable_masks,
            )
            exact_m_u = exact_result["m_u_values"]
            pt_m_u = pt_result["m_u_values_per_temperature"][t_index]
            pt_q_top = float(
                pt_result["q_top_value_per_temperature"][t_index]
            )
            max_abs_diff = float(np.max(np.abs(exact_m_u - pt_m_u)))
            q_top_abs_diff = abs(exact_result["q_top_value"] - pt_q_top)
            tag = "COLD" if t_index == 0 else (
                "HOT " if t_index == len(ladder) - 1 else "    "
            )
            print(
                f"  {tag} T={t_index} p={float(p_value):.4f}  "
                f"|Δm_u|max={max_abs_diff:.4f}  "
                f"|Δq_top|={q_top_abs_diff:.4f}  "
                f"(tol={tolerance:.4f})"
            )
            assert max_abs_diff < tolerance, (
                "3D q>0 exact-vs-PT m_u regression failed: "
                f"L={lattice_size}, p_cold={p_cold}, p_T={float(p_value)}, "
                f"q={syndrome_error_probability}, t_index={t_index}, seed={seed}, "
                f"max_abs_diff={max_abs_diff:.6f}, tol={tolerance:.6f}"
            )
            assert q_top_abs_diff < tolerance, (
                "3D q>0 exact-vs-PT q_top regression failed: "
                f"L={lattice_size}, p_cold={p_cold}, p_T={float(p_value)}, "
                f"q={syndrome_error_probability}, t_index={t_index}, seed={seed}, "
                f"q_top_abs_diff={q_top_abs_diff:.6f}, tol={tolerance:.6f}"
            )
        swap_rates = np.array(pt_result["swap_acceptance_rates"])
        print(
            f"  grid point {point_index}: p_cold={p_cold}, q={syndrome_error_probability}, "
            f"swap_acc={np.round(swap_rates, 3).tolist()}, "
            f"cold_winding_acc="
            f"{float(pt_result['winding_acceptance_rate_per_temperature'][0]):.3e}"
        )


def _run_3d_q_positive_production_path_regression():
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
    num_checks, num_qubits = parity_check_matrix.shape
    zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )

    syndrome_error_probability = 5e-3
    data_error_probability = 0.22
    num_burn_in_sweeps = 2000
    num_measurements_per_disorder = 8000
    num_sweeps_between_measurements = 2
    num_start_chains = 4
    num_replicas_per_start = 2
    tolerance = 6.0 / np.sqrt(num_measurements_per_disorder)
    seed = 20260424

    disorder_rng = np.random.default_rng(seed)
    syndrome_uniform_values = disorder_rng.random((1, num_checks))
    data_uniform_values = disorder_rng.random((1, num_qubits))
    observed_syndrome_bits = (
        syndrome_uniform_values[0] < syndrome_error_probability
    )
    disorder_data_error_bits = (
        data_uniform_values[0] < data_error_probability
    )
    exact_result = compute_exact_logical_observable_means(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        logical_observable_masks=logical_observable_masks,
    )

    print("\n3D q>0 production-path regression")
    for label, extra_kwargs in (
            (
                "multistart",
                {
                    "num_start_chains": num_start_chains,
                    "num_replicas_per_start": num_replicas_per_start,
                },
            ),
            (
                "multistart+pt",
                {
                    "num_start_chains": num_start_chains,
                    "num_replicas_per_start": num_replicas_per_start,
                    "pt_p_hot": 0.44,
                    "pt_num_temperatures": 9,
                },
            )):
        simulation_result = run_disorder_average_simulation(
            parity_check_matrix=parity_check_matrix,
            dual_logical_z_basis=dual_logical_z_basis,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability=data_error_probability,
            num_disorder_samples=1,
            num_burn_in_sweeps=num_burn_in_sweeps,
            num_sweeps_between_measurements=num_sweeps_between_measurements,
            num_measurements_per_disorder=num_measurements_per_disorder,
            seed=seed,
            zero_syndrome_move_data=zero_syndrome_move_data,
            precomputed_syndrome_uniform_values_per_disorder=(
                syndrome_uniform_values
            ),
            precomputed_data_uniform_values_per_disorder=(
                data_uniform_values
            ),
            **extra_kwargs,
        )
        production_m_u = simulation_result[
            "logical_observable_mean_values_per_disorder"
        ][0]
        production_q_top = float(
            simulation_result["disorder_q_top_values"][0]
        )
        max_abs_diff = float(np.max(np.abs(
            exact_result["m_u_values"] - production_m_u
        )))
        q_top_abs_diff = abs(
            exact_result["q_top_value"] - production_q_top
        )
        print(
            f"  {label:<14} |Δm_u|max={max_abs_diff:.4f} "
            f"|Δq_top|={q_top_abs_diff:.4f} "
            f"(tol={tolerance:.4f})"
        )
        assert max_abs_diff < tolerance, (
            "3D q>0 production-path m_u regression failed: "
            f"label={label}, max_abs_diff={max_abs_diff:.6f}, "
            f"tol={tolerance:.6f}"
        )
        assert q_top_abs_diff < tolerance, (
            "3D q>0 production-path q_top regression failed: "
            f"label={label}, q_top_abs_diff={q_top_abs_diff:.6f}, "
            f"tol={tolerance:.6f}"
        )


def _run_q_positive_single_bit_bruteforce_regression():
    parity_check_matrix, _ = build_3d_toric_code(lattice_size=2)
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    print("\nq>0 single-bit brute-force ratio test")
    for syndrome_error_probability, data_error_probability in (
            (1e-3, 0.05),
            (1e-3, 0.25),
            (1e-2, 0.15),
            (4e-2, 0.25)):
        result = run_q_positive_single_bit_acceptance_bruteforce_test(
            parity_check_matrix=parity_check_matrix,
            checks_touching_each_qubit=checks_touching_each_qubit,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability=data_error_probability,
            rng=np.random.default_rng(
                20260422 + int(1000 * syndrome_error_probability)
            ),
            num_random_cases=512,
        )
        print(
            "single-bit-ratio: "
            f"p={data_error_probability:.4g}, "
            f"q={syndrome_error_probability:.4g}, "
            f"cases={result['num_random_cases']}, "
            "delta_syndrome_weight=0 covered"
        )


def _run_q_to_zero_continuity_diagnostic():
    lattice_size = 2
    data_error_probability = 0.15
    q_values = (0.0, 1e-6, 1e-4, 1e-3)
    seed = 2026042217

    parity_check_matrix, dual_logical_z_basis = build_3d_toric_code(
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
    zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(
        lattice_size=lattice_size
    )
    num_checks, num_qubits = parity_check_matrix.shape
    disorder_rng = np.random.default_rng(seed)
    syndrome_uniform_values = disorder_rng.random(num_checks)
    data_uniform_values = disorder_rng.random(num_qubits)

    print("\nq->0 continuity diagnostic")
    q0_q_top_value = None
    for syndrome_error_probability in q_values:
        observed_syndrome_bits = (
            syndrome_uniform_values < syndrome_error_probability
        )
        disorder_data_error_bits = (
            data_uniform_values < data_error_probability
        )
        m_u_values, acceptance_rate = _run_single_disorder_measurement(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability=data_error_probability,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=2000,
            num_measurements_per_disorder=6000,
            num_sweeps_between_measurements=2,
            rng=np.random.default_rng(seed),
            zero_syndrome_move_data=zero_syndrome_move_data,
        )
        q_top_value = float(np.mean(m_u_values ** 2))
        if q0_q_top_value is None:
            q0_q_top_value = q_top_value
        elif syndrome_error_probability == 1e-6:
            assert abs(q_top_value - q0_q_top_value) < 0.1, (
                "q->0 continuity diagnostic failed: "
                f"q=1e-6 differs too much from q=0 "
                f"(q_top diff={abs(q_top_value - q0_q_top_value):.6f})"
            )
        print(
            f"continuity: q={syndrome_error_probability:.1e}, "
            f"q_top={q_top_value:.6f}, "
            f"acceptance={acceptance_rate:.6f}, "
            f"m_u={np.array2string(m_u_values, precision=6)}"
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
        expected_contractible_weight=4,
        expected_winding_weight=lattice_size,
    )
    print("Test 0 passed: q=0 move structure")
    _run_3d_q0_sector_distinguishability_test()
    print("Test 0b passed: 3D q=0 sector distinguishability")

    print(
        f"{'test_name':<18} "
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
    _run_q_positive_single_bit_bruteforce_regression()
    _run_3d_q_positive_exact_vs_mcmc_regression()
    _run_3d_q_positive_exact_vs_pt_regression()
    _run_3d_q_positive_production_path_regression()
    _run_q_to_zero_continuity_diagnostic()

    print("All exact-enumeration validations passed.")
