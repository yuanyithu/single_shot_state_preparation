import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from build_toric_code_examples import (
    build_2d_toric_code,
    build_2d_toric_zero_syndrome_move_data,
)
from linear_section import apply_linear_section, build_linear_section
from mcmc import (
    accumulate_logical_observables,
    draw_disorder_sample,
    draw_disorder_sample_from_uniform_values,
    initialize_mcmc_state,
)
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def _ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def _compute_log_odds(probability):
    """
    计算边界安全的 log(probability / (1 - probability)).
    """
    if probability == 0.0:
        return -np.inf
    if probability == 0.5:
        return 0.0
    if probability == 1.0:
        return np.inf
    return float(np.log(probability / (1.0 - probability)))


def _attempt_single_bit_metropolis_update_safe(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        qubit_index,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng):
    """
    边界安全版本的单比特 Metropolis 更新。

    关键差异：当某一项的 delta_weight == 0 时，直接跳过该项，避免
    0 * (+/-inf) 产生 nan。
    """
    touched_checks = checks_touching_each_qubit[qubit_index]

    if current_data_term_bits[qubit_index]:
        delta_data_weight = -1
    else:
        delta_data_weight = +1

    delta_syndrome_weight = 0
    for check_index in touched_checks:
        if current_syndrome_term_bits[check_index]:
            delta_syndrome_weight -= 1
        else:
            delta_syndrome_weight += 1

    log_acceptance = 0.0
    if delta_data_weight != 0:
        log_acceptance += delta_data_weight * log_odds_data
    if delta_syndrome_weight != 0:
        log_acceptance += delta_syndrome_weight * log_odds_syndrome

    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))

    if not accepted:
        return False

    current_chain_bits[qubit_index] ^= True
    current_data_term_bits[qubit_index] ^= True
    current_syndrome_term_bits[touched_checks] ^= True
    return True


def _run_one_sweep_safe(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng):
    """
    边界安全版本的 sweep。
    """
    num_qubits = current_chain_bits.shape[0]
    accepted_count = 0

    for qubit_index in rng.permutation(num_qubits):
        accepted_count += int(
            _attempt_single_bit_metropolis_update_safe(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                qubit_index=int(qubit_index),
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
                rng=rng,
            )
        )

    return accepted_count


def _build_kernel_basis_from_linear_section(
        parity_check_matrix,
        linear_section_data):
    """
    从 E H_Z Π = [[I, A], [0, 0]] 重建 ker(H_Z) 的一组基。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    rank = linear_section_data["rank"]
    column_permutation = linear_section_data["column_permutation"]
    row_xor_steps = linear_section_data["row_xor_steps"]

    transformed_matrix = parity_check_matrix.copy().astype(bool)
    transformed_matrix = transformed_matrix[:, column_permutation]

    for operation_name, first_index, second_index in row_xor_steps:
        if operation_name == "swap_rows":
            transformed_matrix[[first_index, second_index]] = (
                transformed_matrix[[second_index, first_index]]
            )
        else:
            transformed_matrix[first_index] ^= transformed_matrix[second_index]

    nullity = num_qubits - rank
    kernel_basis = np.zeros((nullity, num_qubits), dtype=bool)

    for free_column_offset in range(nullity):
        free_column_index = rank + free_column_offset
        basis_vector_in_reduced_coordinates = np.zeros(
            num_qubits,
            dtype=bool,
        )
        basis_vector_in_reduced_coordinates[:rank] = transformed_matrix[
            :rank,
            free_column_index,
        ]
        basis_vector_in_reduced_coordinates[free_column_index] = True

        basis_vector = np.zeros(num_qubits, dtype=bool)
        basis_vector[column_permutation] = basis_vector_in_reduced_coordinates
        kernel_basis[free_column_offset] = basis_vector

    if num_checks > 0:
        syndrome_bits = (
            parity_check_matrix.astype(np.uint8)
            @ kernel_basis.T.astype(np.uint8)
        ) % 2
        assert not np.any(syndrome_bits), "kernel basis construction failed"

    return kernel_basis


def _sample_random_kernel_move_bits(
        kernel_basis,
        rng,
        max_subset_size=3):
    """
    从 ker(H_Z) 基向量中随机抽取 1 到 max_subset_size 个向量并 XOR，
    生成一次全局闭环提议。
    """
    num_generators = kernel_basis.shape[0]
    if num_generators == 0:
        return np.zeros(kernel_basis.shape[1], dtype=bool)

    effective_max_subset_size = min(max_subset_size, num_generators)
    subset_size = int(rng.integers(1, effective_max_subset_size + 1))
    selected_generator_indices = rng.choice(
        num_generators,
        size=subset_size,
        replace=False,
    )
    generator_bits = np.bitwise_xor.reduce(
        kernel_basis[selected_generator_indices],
        axis=0,
    )
    return generator_bits


def _attempt_zero_syndrome_move_update(
        current_chain_bits,
        current_data_term_bits,
        move_bits,
        log_odds_data,
        rng):
    """
    在 q == 0 且 syndrome 固定时，沿 ker(H_Z) 方向做 Metropolis 更新。
    """
    move_support = move_bits
    support_size = int(np.count_nonzero(move_support))
    if support_size == 0:
        return False

    current_ones_on_support = int(
        np.count_nonzero(current_data_term_bits[move_support])
    )
    delta_data_weight = support_size - 2 * current_ones_on_support
    log_acceptance = delta_data_weight * log_odds_data

    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))

    if not accepted:
        return False

    current_chain_bits[move_support] ^= True
    current_data_term_bits[move_support] ^= True
    return True


def _run_one_kernel_sweep_zero_syndrome(
        current_chain_bits,
        current_data_term_bits,
        kernel_basis,
        log_odds_data,
        rng):
    """
    对 ker(H_Z) 的随机闭环提议做一次 sweep，保持 H_Z c 不变。
    """
    accepted_count = 0
    num_proposals = kernel_basis.shape[0]
    if num_proposals == 0:
        return 0

    for _ in range(num_proposals):
        generator_bits = _sample_random_kernel_move_bits(
            kernel_basis=kernel_basis,
            rng=rng,
            max_subset_size=3,
        )
        accepted_count += int(
            _attempt_zero_syndrome_move_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                move_bits=generator_bits,
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )

    return accepted_count


def _run_one_geometric_sweep_zero_syndrome(
        current_chain_bits,
        current_data_term_bits,
        zero_syndrome_move_data,
        log_odds_data,
        rng):
    """
    对预计算的局部闭环和 winding loop 各做一次随机顺序 sweep。
    """
    accepted_count = 0
    contractible_moves = zero_syndrome_move_data["contractible_moves"]
    winding_moves = zero_syndrome_move_data["winding_moves"]

    for move_index in rng.permutation(contractible_moves.shape[0]):
        accepted_count += int(
            _attempt_zero_syndrome_move_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                move_bits=contractible_moves[int(move_index)],
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )

    for move_index in rng.permutation(winding_moves.shape[0]):
        accepted_count += int(
            _attempt_zero_syndrome_move_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                move_bits=winding_moves[int(move_index)],
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )

    return accepted_count


def _count_zero_syndrome_proposals(
        zero_syndrome_move_data=None,
        kernel_basis=None):
    if zero_syndrome_move_data is not None:
        return (
            zero_syndrome_move_data["contractible_moves"].shape[0]
            + zero_syndrome_move_data["winding_moves"].shape[0]
        )
    if kernel_basis is None:
        return 0
    return kernel_basis.shape[0]


def _run_one_zero_syndrome_sweep(
        current_chain_bits,
        current_data_term_bits,
        log_odds_data,
        rng,
        zero_syndrome_move_data=None,
        kernel_basis=None):
    if zero_syndrome_move_data is not None:
        return _run_one_geometric_sweep_zero_syndrome(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            zero_syndrome_move_data=zero_syndrome_move_data,
            log_odds_data=log_odds_data,
            rng=rng,
        )
    return _run_one_kernel_sweep_zero_syndrome(
        current_chain_bits=current_chain_bits,
        current_data_term_bits=current_data_term_bits,
        kernel_basis=kernel_basis,
        log_odds_data=log_odds_data,
        rng=rng,
    )


def _build_q0_start_sector_labels(q0_num_start_chains):
    if q0_num_start_chains < 1 or q0_num_start_chains > 4:
        raise ValueError("q0_num_start_chains must be in [1, 4]")
    all_labels = np.array(["00", "10", "01", "11"])
    return all_labels[:q0_num_start_chains]


def _build_q0_initial_chain_bits_per_start(
        observed_syndrome_bits,
        linear_section_data,
        zero_syndrome_move_data,
        q0_num_start_chains):
    """
    用 section representative 加上两条独立 kernel winding loop 生成 q=0 初态。
    """
    if zero_syndrome_move_data is None:
        raise ValueError(
            "zero_syndrome_move_data is required for q=0 multi-start chains"
        )

    start_sector_labels = _build_q0_start_sector_labels(q0_num_start_chains)
    start_sector_generators = zero_syndrome_move_data[
        "start_sector_generators"
    ]
    section_representative = apply_linear_section(
        observed_syndrome_bits,
        linear_section_data,
    )
    initial_chain_bits_per_start = np.repeat(
        section_representative[None, :],
        q0_num_start_chains,
        axis=0,
    )

    for start_index, label in enumerate(start_sector_labels):
        if label[0] == "1":
            initial_chain_bits_per_start[start_index] ^= (
                start_sector_generators[0]
            )
        if label[1] == "1":
            initial_chain_bits_per_start[start_index] ^= (
                start_sector_generators[1]
            )

    return initial_chain_bits_per_start, start_sector_labels


def _compute_q0_diagnostic_spreads(
        q0_m_u_values_per_start,
        q0_q_top_values_per_start):
    pairwise_m_u_diff = np.abs(
        q0_m_u_values_per_start[:, None, :]
        - q0_m_u_values_per_start[None, :, :]
    )
    q0_m_u_spread_linf = float(np.max(pairwise_m_u_diff))
    q0_q_top_spread = float(
        np.max(q0_q_top_values_per_start)
        - np.min(q0_q_top_values_per_start)
    )
    return q0_m_u_spread_linf, q0_q_top_spread


def _run_single_disorder_measurement(
        parity_check_matrix,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability,
        logical_observable_masks,
        checks_touching_each_qubit,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        rng,
        zero_syndrome_move_data=None,
        kernel_basis=None,
        log_odds_data=None,
        log_odds_syndrome=None,
        initial_chain_bits=None):
    """
    对固定的 (s, eta) 运行一次 MCMC，返回 (m_u_values, acceptance_rate)。
    """
    num_qubits = parity_check_matrix.shape[1]

    if log_odds_data is None:
        log_odds_data = _compute_log_odds(data_error_probability)
    if log_odds_syndrome is None:
        log_odds_syndrome = _compute_log_odds(syndrome_error_probability)
    if (
            syndrome_error_probability == 0.0
            and zero_syndrome_move_data is None
            and kernel_basis is None):
        linear_section_data = build_linear_section(parity_check_matrix)
        kernel_basis = _build_kernel_basis_from_linear_section(
            parity_check_matrix=parity_check_matrix,
            linear_section_data=linear_section_data,
        )
    if syndrome_error_probability == 0.0 and initial_chain_bits is not None:
        initial_syndrome_bits = (
            parity_check_matrix.astype(np.uint8)
            @ np.asarray(initial_chain_bits, dtype=np.uint8)
        ) % 2
        if not np.array_equal(
                initial_syndrome_bits.astype(bool),
                observed_syndrome_bits):
            raise ValueError(
                "q=0 initial_chain_bits must satisfy H_Z c = observed_syndrome_bits"
            )

    (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    ) = initialize_mcmc_state(
        num_qubits=num_qubits,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        parity_check_matrix=parity_check_matrix,
        rng=rng,
        initial_chain_bits=initial_chain_bits,
    )
    num_zero_syndrome_proposals = _count_zero_syndrome_proposals(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
    )

    for _ in range(num_burn_in_sweeps):
        if syndrome_error_probability == 0.0:
            _run_one_zero_syndrome_sweep(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                log_odds_data=log_odds_data,
                rng=rng,
                zero_syndrome_move_data=zero_syndrome_move_data,
                kernel_basis=kernel_basis,
            )
        else:
            _run_one_sweep_safe(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
                rng=rng,
            )

    num_masks = logical_observable_masks.shape[0]
    logical_observable_sum_values = np.zeros(num_masks, dtype=np.int64)
    total_accepted_count = 0
    total_attempted_count = 0

    for _ in range(num_measurements_per_disorder):
        for _ in range(num_sweeps_between_measurements):
            if syndrome_error_probability == 0.0:
                total_accepted_count += _run_one_zero_syndrome_sweep(
                    current_chain_bits=current_chain_bits,
                    current_data_term_bits=current_data_term_bits,
                    log_odds_data=log_odds_data,
                    rng=rng,
                    zero_syndrome_move_data=zero_syndrome_move_data,
                    kernel_basis=kernel_basis,
                )
                total_attempted_count += num_zero_syndrome_proposals
            else:
                total_accepted_count += _run_one_sweep_safe(
                    current_chain_bits=current_chain_bits,
                    current_data_term_bits=current_data_term_bits,
                    current_syndrome_term_bits=current_syndrome_term_bits,
                    checks_touching_each_qubit=checks_touching_each_qubit,
                    log_odds_data=log_odds_data,
                    log_odds_syndrome=log_odds_syndrome,
                    rng=rng,
                )
                total_attempted_count += num_qubits

        accumulate_logical_observables(
            current_chain_bits=current_chain_bits,
            logical_observable_masks=logical_observable_masks,
            logical_observable_sum_values=logical_observable_sum_values,
        )

    m_u_values = (
        logical_observable_sum_values / num_measurements_per_disorder
    ).astype(np.float64, copy=False)
    acceptance_rate = total_accepted_count / total_attempted_count
    return m_u_values, acceptance_rate


def run_disorder_average_simulation(
        parity_check_matrix,
        dual_logical_z_basis,
        syndrome_error_probability,
        data_error_probability,
        num_disorder_samples,
        num_burn_in_sweeps,
        num_sweeps_between_measurements,
        num_measurements_per_disorder,
        seed,
        zero_syndrome_move_data=None,
        q0_num_start_chains=4,
        precomputed_syndrome_uniform_values_per_disorder=None,
        precomputed_data_uniform_values_per_disorder=None):
    rng = np.random.default_rng(seed)

    num_checks, num_qubits = parity_check_matrix.shape

    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    kernel_basis = None
    if syndrome_error_probability == 0.0 and zero_syndrome_move_data is None:
        kernel_basis = _build_kernel_basis_from_linear_section(
            parity_check_matrix=parity_check_matrix,
            linear_section_data=linear_section_data,
        )

    log_odds_data = _compute_log_odds(data_error_probability)
    log_odds_syndrome = _compute_log_odds(syndrome_error_probability)

    num_masks = logical_observable_masks.shape[0]
    disorder_q_top_values = np.empty(num_disorder_samples, dtype=np.float64)
    logical_observable_mean_values_per_disorder = np.empty(
        (num_disorder_samples, num_masks),
        dtype=np.float64,
    )
    average_acceptance_rate_per_disorder = np.empty(
        num_disorder_samples,
        dtype=np.float64,
    )
    use_precomputed_disorder_uniforms = (
        precomputed_syndrome_uniform_values_per_disorder is not None
        or precomputed_data_uniform_values_per_disorder is not None
    )
    if (
            precomputed_syndrome_uniform_values_per_disorder is None
            ) != (
                precomputed_data_uniform_values_per_disorder is None):
        raise ValueError(
            "precomputed_syndrome_uniform_values_per_disorder and "
            "precomputed_data_uniform_values_per_disorder must be provided "
            "together"
        )
    if use_precomputed_disorder_uniforms:
        precomputed_syndrome_uniform_values_per_disorder = np.asarray(
            precomputed_syndrome_uniform_values_per_disorder,
            dtype=np.float64,
        )
        precomputed_data_uniform_values_per_disorder = np.asarray(
            precomputed_data_uniform_values_per_disorder,
            dtype=np.float64,
        )
        expected_syndrome_shape = (num_disorder_samples, num_checks)
        expected_data_shape = (num_disorder_samples, num_qubits)
        if (
                precomputed_syndrome_uniform_values_per_disorder.shape
                != expected_syndrome_shape):
            raise ValueError(
                "precomputed_syndrome_uniform_values_per_disorder must have "
                f"shape {expected_syndrome_shape}"
            )
        if (
                precomputed_data_uniform_values_per_disorder.shape
                != expected_data_shape):
            raise ValueError(
                "precomputed_data_uniform_values_per_disorder must have "
                f"shape {expected_data_shape}"
            )

    q0_start_sector_labels = None
    if (
            syndrome_error_probability == 0.0
            and zero_syndrome_move_data is not None):
        q0_start_sector_labels = _build_q0_start_sector_labels(
            q0_num_start_chains
        )
        q0_logical_observable_mean_values_per_disorder_per_start = np.empty(
            (num_disorder_samples, q0_num_start_chains, num_masks),
            dtype=np.float64,
        )
        q0_q_top_values_per_disorder_per_start = np.empty(
            (num_disorder_samples, q0_num_start_chains),
            dtype=np.float64,
        )
        q0_m_u_spread_linf_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )
        q0_q_top_spread_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )

    for disorder_index in range(num_disorder_samples):
        if use_precomputed_disorder_uniforms:
            (
                observed_syndrome_bits,
                disorder_data_error_bits,
            ) = draw_disorder_sample_from_uniform_values(
                syndrome_uniform_values=(
                    precomputed_syndrome_uniform_values_per_disorder[
                        disorder_index
                    ]
                ),
                data_uniform_values=(
                    precomputed_data_uniform_values_per_disorder[
                        disorder_index
                    ]
                ),
                syndrome_error_probability=syndrome_error_probability,
                data_error_probability=data_error_probability,
            )
        else:
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
        if q0_start_sector_labels is not None:
            (
                initial_chain_bits_per_start,
                _,
            ) = _build_q0_initial_chain_bits_per_start(
                observed_syndrome_bits=observed_syndrome_bits,
                linear_section_data=linear_section_data,
                zero_syndrome_move_data=zero_syndrome_move_data,
                q0_num_start_chains=q0_num_start_chains,
            )
            q0_m_u_values_per_start = np.empty(
                (q0_num_start_chains, num_masks),
                dtype=np.float64,
            )
            q0_q_top_values_per_start = np.empty(
                q0_num_start_chains,
                dtype=np.float64,
            )
            q0_acceptance_rates_per_start = np.empty(
                q0_num_start_chains,
                dtype=np.float64,
            )

            for start_index in range(q0_num_start_chains):
                start_seed = int(
                    rng.integers(
                        0,
                        np.iinfo(np.uint64).max,
                        dtype=np.uint64,
                    )
                )
                start_rng = np.random.default_rng(start_seed)
                (
                    q0_m_u_values_per_start[start_index],
                    q0_acceptance_rates_per_start[start_index],
                ) = _run_single_disorder_measurement(
                    parity_check_matrix=parity_check_matrix,
                    observed_syndrome_bits=observed_syndrome_bits,
                    disorder_data_error_bits=disorder_data_error_bits,
                    syndrome_error_probability=syndrome_error_probability,
                    data_error_probability=data_error_probability,
                    logical_observable_masks=logical_observable_masks,
                    checks_touching_each_qubit=checks_touching_each_qubit,
                    num_burn_in_sweeps=num_burn_in_sweeps,
                    num_measurements_per_disorder=(
                        num_measurements_per_disorder
                    ),
                    num_sweeps_between_measurements=(
                        num_sweeps_between_measurements
                    ),
                    rng=start_rng,
                    zero_syndrome_move_data=zero_syndrome_move_data,
                    kernel_basis=kernel_basis,
                    log_odds_data=log_odds_data,
                    log_odds_syndrome=log_odds_syndrome,
                    initial_chain_bits=initial_chain_bits_per_start[
                        start_index
                    ],
                )
                q0_q_top_values_per_start[start_index] = float(
                    np.mean(q0_m_u_values_per_start[start_index] ** 2)
                )

            m_u_values = np.mean(q0_m_u_values_per_start, axis=0)
            acceptance_rate = float(np.mean(q0_acceptance_rates_per_start))
            q_top_value = float(np.mean(m_u_values ** 2))
            (
                q0_m_u_spread_linf_per_disorder[disorder_index],
                q0_q_top_spread_per_disorder[disorder_index],
            ) = _compute_q0_diagnostic_spreads(
                q0_m_u_values_per_start=q0_m_u_values_per_start,
                q0_q_top_values_per_start=q0_q_top_values_per_start,
            )
            q0_logical_observable_mean_values_per_disorder_per_start[
                disorder_index
            ] = q0_m_u_values_per_start
            q0_q_top_values_per_disorder_per_start[disorder_index] = (
                q0_q_top_values_per_start
            )
        else:
            m_u_values, acceptance_rate = _run_single_disorder_measurement(
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
                rng=rng,
                zero_syndrome_move_data=zero_syndrome_move_data,
                kernel_basis=kernel_basis,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
            )
            q_top_value = float(np.mean(m_u_values ** 2))

        logical_observable_mean_values_per_disorder[disorder_index] = m_u_values
        disorder_q_top_values[disorder_index] = q_top_value
        average_acceptance_rate_per_disorder[disorder_index] = acceptance_rate

    disorder_average_q_top = float(np.mean(disorder_q_top_values))

    if q0_start_sector_labels is not None:
        return {
            "disorder_q_top_values": disorder_q_top_values,
            "disorder_average_q_top": disorder_average_q_top,
            "logical_observable_mean_values_per_disorder": (
                logical_observable_mean_values_per_disorder
            ),
            "average_acceptance_rate_per_disorder": (
                average_acceptance_rate_per_disorder
            ),
            "q0_start_sector_labels": q0_start_sector_labels,
            "q0_logical_observable_mean_values_per_disorder_per_start": (
                q0_logical_observable_mean_values_per_disorder_per_start
            ),
            "q0_q_top_values_per_disorder_per_start": (
                q0_q_top_values_per_disorder_per_start
            ),
            "q0_m_u_spread_linf_per_disorder": (
                q0_m_u_spread_linf_per_disorder
            ),
            "q0_q_top_spread_per_disorder": q0_q_top_spread_per_disorder,
        }

    return {
        "disorder_q_top_values": disorder_q_top_values,
        "disorder_average_q_top": disorder_average_q_top,
        "logical_observable_mean_values_per_disorder": (
            logical_observable_mean_values_per_disorder
        ),
        "average_acceptance_rate_per_disorder": (
            average_acceptance_rate_per_disorder
        ),
    }


def scan_data_error_probability(
        parity_check_matrix,
        dual_logical_z_basis,
        syndrome_error_probability,
        data_error_probability_list,
        num_disorder_samples,
        num_burn_in_sweeps,
        num_sweeps_between_measurements,
        num_measurements_per_disorder,
        seed,
        zero_syndrome_move_data=None,
        q0_num_start_chains=4):
    data_error_probability_array = np.asarray(
        data_error_probability_list,
        dtype=np.float64,
    )
    num_points = data_error_probability_array.shape[0]

    q_top_curve = np.empty(num_points, dtype=np.float64)
    q_top_std_curve = np.empty(num_points, dtype=np.float64)
    average_acceptance_rate_curve = np.empty(num_points, dtype=np.float64)
    q0_mean_m_u_spread_linf_curve = None
    q0_mean_q_top_spread_curve = None

    if syndrome_error_probability == 0.0 and zero_syndrome_move_data is not None:
        q0_mean_m_u_spread_linf_curve = np.empty(
            num_points,
            dtype=np.float64,
        )
        q0_mean_q_top_spread_curve = np.empty(
            num_points,
            dtype=np.float64,
        )

    for point_index, data_error_probability in enumerate(
            data_error_probability_array):
        result = run_disorder_average_simulation(
            parity_check_matrix=parity_check_matrix,
            dual_logical_z_basis=dual_logical_z_basis,
            syndrome_error_probability=syndrome_error_probability,
            data_error_probability=float(data_error_probability),
            num_disorder_samples=num_disorder_samples,
            num_burn_in_sweeps=num_burn_in_sweeps,
            num_sweeps_between_measurements=(
                num_sweeps_between_measurements
            ),
            num_measurements_per_disorder=num_measurements_per_disorder,
            seed=seed + point_index,
            zero_syndrome_move_data=zero_syndrome_move_data,
            q0_num_start_chains=q0_num_start_chains,
        )

        disorder_q_top_values = result["disorder_q_top_values"]
        q_top_curve[point_index] = result["disorder_average_q_top"]
        if num_disorder_samples == 1:
            q_top_std_curve[point_index] = 0.0
        else:
            q_top_std_curve[point_index] = np.std(
                disorder_q_top_values,
                ddof=1,
            )
        average_acceptance_rate_curve[point_index] = float(
            np.mean(result["average_acceptance_rate_per_disorder"])
        )
        if q0_mean_m_u_spread_linf_curve is not None:
            q0_mean_m_u_spread_linf_curve[point_index] = float(
                np.mean(result["q0_m_u_spread_linf_per_disorder"])
            )
            q0_mean_q_top_spread_curve[point_index] = float(
                np.mean(result["q0_q_top_spread_per_disorder"])
            )

    scan_result = {
        "data_error_probability_list": data_error_probability_array,
        "q_top_curve": q_top_curve,
        "q_top_std_curve": q_top_std_curve,
        "average_acceptance_rate_curve": average_acceptance_rate_curve,
    }
    if q0_mean_m_u_spread_linf_curve is not None:
        scan_result["q0_start_sector_labels"] = _build_q0_start_sector_labels(
            q0_num_start_chains
        )
        scan_result["q0_mean_m_u_spread_linf_curve"] = (
            q0_mean_m_u_spread_linf_curve
        )
        scan_result["q0_mean_q_top_spread_curve"] = (
            q0_mean_q_top_spread_curve
        )
    return scan_result


def _run_single_scan_point_task(task_data):
    """
    并行 worker：对单个 (L, p) 点完成一次 disorder 平均。
    """
    lattice_index = task_data["lattice_index"]
    point_index = task_data["point_index"]
    lattice_size = task_data["lattice_size"]
    syndrome_error_probability = task_data["syndrome_error_probability"]
    data_error_probability = task_data["data_error_probability"]
    num_disorder_samples = task_data["num_disorder_samples"]
    num_burn_in_sweeps = task_data["num_burn_in_sweeps"]
    num_sweeps_between_measurements = (
        task_data["num_sweeps_between_measurements"]
    )
    num_measurements_per_disorder = (
        task_data["num_measurements_per_disorder"]
    )
    q0_num_start_chains = task_data["q0_num_start_chains"]
    seed = task_data["seed"]

    parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
        lattice_size=lattice_size
    )
    zero_syndrome_move_data = None
    if syndrome_error_probability == 0.0:
        zero_syndrome_move_data = build_2d_toric_zero_syndrome_move_data(
            lattice_size=lattice_size
        )
    simulation_result = run_disorder_average_simulation(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        num_disorder_samples=num_disorder_samples,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        num_measurements_per_disorder=num_measurements_per_disorder,
        seed=seed,
        zero_syndrome_move_data=zero_syndrome_move_data,
        q0_num_start_chains=q0_num_start_chains,
    )

    disorder_q_top_values = simulation_result["disorder_q_top_values"]
    if num_disorder_samples == 1:
        q_top_std = 0.0
    else:
        q_top_std = float(
            np.std(disorder_q_top_values, ddof=1)
        )

    task_result = {
        "lattice_index": lattice_index,
        "point_index": point_index,
        "q_top": simulation_result["disorder_average_q_top"],
        "q_top_std": q_top_std,
        "average_acceptance_rate": float(
            np.mean(simulation_result["average_acceptance_rate_per_disorder"])
        ),
    }
    if "q0_start_sector_labels" in simulation_result:
        task_result["q0_mean_m_u_spread_linf"] = float(
            np.mean(simulation_result["q0_m_u_spread_linf_per_disorder"])
        )
        task_result["q0_mean_q_top_spread"] = float(
            np.mean(simulation_result["q0_q_top_spread_per_disorder"])
        )
        task_result["q0_logical_observable_mean_values_per_disorder_per_start"] = (
            simulation_result[
                "q0_logical_observable_mean_values_per_disorder_per_start"
            ]
        )
        task_result["q0_q_top_values_per_disorder_per_start"] = (
            simulation_result["q0_q_top_values_per_disorder_per_start"]
        )
        task_result["q0_m_u_spread_linf_per_disorder"] = simulation_result[
            "q0_m_u_spread_linf_per_disorder"
        ]
        task_result["q0_q_top_spread_per_disorder"] = simulation_result[
            "q0_q_top_spread_per_disorder"
        ]
    return task_result


def _compute_parallel_worker_count(num_tasks):
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(num_tasks, cpu_count))


def _build_multiprocessing_context():
    """
    优先使用 fork 以减少 worker 启动与数据复制开销；不可用时回退到 spawn。
    """
    available_start_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_start_methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _store_scan_point_result(
        q_top_curve_matrix,
        q_top_std_curve_matrix,
        average_acceptance_rate_curve_matrix,
        q0_mean_m_u_spread_linf_curve_matrix,
        q0_mean_q_top_spread_curve_matrix,
        q0_logical_observable_mean_values_per_disorder_per_start_tensor,
        q0_q_top_values_per_disorder_per_start_tensor,
        q0_m_u_spread_linf_per_disorder_tensor,
        q0_q_top_spread_per_disorder_tensor,
        task_result):
    lattice_index = task_result["lattice_index"]
    point_index = task_result["point_index"]
    q_top_curve_matrix[lattice_index, point_index] = task_result["q_top"]
    q_top_std_curve_matrix[lattice_index, point_index] = (
        task_result["q_top_std"]
    )
    average_acceptance_rate_curve_matrix[
        lattice_index,
        point_index,
    ] = task_result["average_acceptance_rate"]
    if (
            q0_mean_m_u_spread_linf_curve_matrix is not None
            and "q0_mean_m_u_spread_linf" in task_result):
        q0_mean_m_u_spread_linf_curve_matrix[
            lattice_index,
            point_index,
        ] = task_result["q0_mean_m_u_spread_linf"]
        q0_mean_q_top_spread_curve_matrix[
            lattice_index,
            point_index,
        ] = task_result["q0_mean_q_top_spread"]
        q0_logical_observable_mean_values_per_disorder_per_start_tensor[
            lattice_index,
            point_index,
        ] = task_result[
            "q0_logical_observable_mean_values_per_disorder_per_start"
        ]
        q0_q_top_values_per_disorder_per_start_tensor[
            lattice_index,
            point_index,
        ] = task_result["q0_q_top_values_per_disorder_per_start"]
        q0_m_u_spread_linf_per_disorder_tensor[
            lattice_index,
            point_index,
        ] = task_result["q0_m_u_spread_linf_per_disorder"]
        q0_q_top_spread_per_disorder_tensor[
            lattice_index,
            point_index,
        ] = task_result["q0_q_top_spread_per_disorder"]


def scan_multiple_code_sizes(
        lattice_size_list,
        syndrome_error_probability,
        data_error_probability_list,
        num_disorder_samples,
        num_burn_in_sweeps,
        num_sweeps_between_measurements,
        num_measurements_per_disorder,
        seed_base,
        q0_num_start_chains=4):
    """
    扫描多个 2D toric code 尺寸，并在内部对 burn-in 做线性放大。
    """
    burn_in_scaling_reference_num_qubits = 18
    lattice_size_array = np.asarray(lattice_size_list, dtype=np.int64)
    data_error_probability_array = np.asarray(
        data_error_probability_list,
        dtype=np.float64,
    )

    num_sizes = lattice_size_array.shape[0]
    num_points = data_error_probability_array.shape[0]

    num_qubits_list = np.empty(num_sizes, dtype=np.int64)
    num_logical_qubits_list = np.empty(num_sizes, dtype=np.int64)
    effective_num_burn_in_sweeps_list = np.empty(num_sizes, dtype=np.int64)
    q_top_curve_matrix = np.empty((num_sizes, num_points), dtype=np.float64)
    q_top_std_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    average_acceptance_rate_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    num_masks = None
    task_data_list = []

    for lattice_index, lattice_size in enumerate(lattice_size_array):
        parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
            lattice_size=int(lattice_size)
        )
        num_qubits = parity_check_matrix.shape[1]
        num_logical_qubits = dual_logical_z_basis.shape[0]
        if num_masks is None:
            num_masks = (1 << num_logical_qubits) - 1
        effective_num_burn_in_sweeps = int(np.ceil(
            num_burn_in_sweeps
            * (num_qubits / burn_in_scaling_reference_num_qubits)
        ))

        num_qubits_list[lattice_index] = num_qubits
        num_logical_qubits_list[lattice_index] = num_logical_qubits
        effective_num_burn_in_sweeps_list[lattice_index] = (
            effective_num_burn_in_sweeps
        )
        for point_index, data_error_probability in enumerate(
                data_error_probability_array):
            task_data_list.append({
                "lattice_index": lattice_index,
                "point_index": point_index,
                "lattice_size": int(lattice_size),
                "syndrome_error_probability": (
                    syndrome_error_probability
                ),
                "data_error_probability": float(data_error_probability),
                "num_disorder_samples": num_disorder_samples,
                "num_burn_in_sweeps": effective_num_burn_in_sweeps,
                "num_sweeps_between_measurements": (
                    num_sweeps_between_measurements
                ),
                "num_measurements_per_disorder": (
                    num_measurements_per_disorder
                ),
                "q0_num_start_chains": q0_num_start_chains,
                "seed": (
                    seed_base + lattice_index * num_points + point_index
                ),
            })

    q0_start_sector_labels = None
    q0_mean_m_u_spread_linf_curve_matrix = None
    q0_mean_q_top_spread_curve_matrix = None
    q0_logical_observable_mean_values_per_disorder_per_start_tensor = None
    q0_q_top_values_per_disorder_per_start_tensor = None
    q0_m_u_spread_linf_per_disorder_tensor = None
    q0_q_top_spread_per_disorder_tensor = None
    if syndrome_error_probability == 0.0:
        q0_start_sector_labels = _build_q0_start_sector_labels(
            q0_num_start_chains
        )
        q0_mean_m_u_spread_linf_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        q0_mean_q_top_spread_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        q0_logical_observable_mean_values_per_disorder_per_start_tensor = (
            np.empty(
                (
                    num_sizes,
                    num_points,
                    num_disorder_samples,
                    q0_num_start_chains,
                    num_masks,
                ),
                dtype=np.float64,
            )
        )
        q0_q_top_values_per_disorder_per_start_tensor = np.empty(
            (
                num_sizes,
                num_points,
                num_disorder_samples,
                q0_num_start_chains,
            ),
            dtype=np.float64,
        )
        q0_m_u_spread_linf_per_disorder_tensor = np.empty(
            (num_sizes, num_points, num_disorder_samples),
            dtype=np.float64,
        )
        q0_q_top_spread_per_disorder_tensor = np.empty(
            (num_sizes, num_points, num_disorder_samples),
            dtype=np.float64,
        )

    num_tasks = len(task_data_list)
    num_workers = _compute_parallel_worker_count(num_tasks)
    multiprocessing_context = _build_multiprocessing_context()

    if num_workers == 1:
        task_results = map(_run_single_scan_point_task, task_data_list)
        for task_result in task_results:
            _store_scan_point_result(
                q_top_curve_matrix=q_top_curve_matrix,
                q_top_std_curve_matrix=q_top_std_curve_matrix,
                average_acceptance_rate_curve_matrix=(
                    average_acceptance_rate_curve_matrix
                ),
                q0_mean_m_u_spread_linf_curve_matrix=(
                    q0_mean_m_u_spread_linf_curve_matrix
                ),
                q0_mean_q_top_spread_curve_matrix=(
                    q0_mean_q_top_spread_curve_matrix
                ),
                q0_logical_observable_mean_values_per_disorder_per_start_tensor=(
                    q0_logical_observable_mean_values_per_disorder_per_start_tensor
                ),
                q0_q_top_values_per_disorder_per_start_tensor=(
                    q0_q_top_values_per_disorder_per_start_tensor
                ),
                q0_m_u_spread_linf_per_disorder_tensor=(
                    q0_m_u_spread_linf_per_disorder_tensor
                ),
                q0_q_top_spread_per_disorder_tensor=(
                    q0_q_top_spread_per_disorder_tensor
                ),
                task_result=task_result,
            )
    else:
        try:
            with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=multiprocessing_context) as executor:
                future_list = [
                    executor.submit(_run_single_scan_point_task, task_data)
                    for task_data in task_data_list
                ]
                for future in as_completed(future_list):
                    task_result = future.result()
                    _store_scan_point_result(
                        q_top_curve_matrix=q_top_curve_matrix,
                        q_top_std_curve_matrix=(
                            q_top_std_curve_matrix
                        ),
                        average_acceptance_rate_curve_matrix=(
                            average_acceptance_rate_curve_matrix
                        ),
                        q0_mean_m_u_spread_linf_curve_matrix=(
                            q0_mean_m_u_spread_linf_curve_matrix
                        ),
                        q0_mean_q_top_spread_curve_matrix=(
                            q0_mean_q_top_spread_curve_matrix
                        ),
                        q0_logical_observable_mean_values_per_disorder_per_start_tensor=(
                            q0_logical_observable_mean_values_per_disorder_per_start_tensor
                        ),
                        q0_q_top_values_per_disorder_per_start_tensor=(
                            q0_q_top_values_per_disorder_per_start_tensor
                        ),
                        q0_m_u_spread_linf_per_disorder_tensor=(
                            q0_m_u_spread_linf_per_disorder_tensor
                        ),
                        q0_q_top_spread_per_disorder_tensor=(
                            q0_q_top_spread_per_disorder_tensor
                        ),
                        task_result=task_result,
                    )
        except PermissionError:
            task_results = map(_run_single_scan_point_task, task_data_list)
            for task_result in task_results:
                _store_scan_point_result(
                    q_top_curve_matrix=q_top_curve_matrix,
                    q_top_std_curve_matrix=(
                        q_top_std_curve_matrix
                    ),
                    average_acceptance_rate_curve_matrix=(
                        average_acceptance_rate_curve_matrix
                    ),
                    q0_mean_m_u_spread_linf_curve_matrix=(
                        q0_mean_m_u_spread_linf_curve_matrix
                    ),
                    q0_mean_q_top_spread_curve_matrix=(
                        q0_mean_q_top_spread_curve_matrix
                    ),
                    q0_logical_observable_mean_values_per_disorder_per_start_tensor=(
                        q0_logical_observable_mean_values_per_disorder_per_start_tensor
                    ),
                    q0_q_top_values_per_disorder_per_start_tensor=(
                        q0_q_top_values_per_disorder_per_start_tensor
                    ),
                    q0_m_u_spread_linf_per_disorder_tensor=(
                        q0_m_u_spread_linf_per_disorder_tensor
                    ),
                    q0_q_top_spread_per_disorder_tensor=(
                        q0_q_top_spread_per_disorder_tensor
                    ),
                    task_result=task_result,
                )

    scan_result = {
        "lattice_size_list": lattice_size_array,
        "num_qubits_list": num_qubits_list,
        "num_logical_qubits_list": num_logical_qubits_list,
        "effective_num_burn_in_sweeps_list": (
            effective_num_burn_in_sweeps_list
        ),
        "data_error_probability_list": data_error_probability_array,
        "q_top_curve_matrix": q_top_curve_matrix,
        "q_top_std_curve_matrix": q_top_std_curve_matrix,
        "average_acceptance_rate_curve_matrix": (
            average_acceptance_rate_curve_matrix
        ),
    }
    if q0_start_sector_labels is not None:
        scan_result["q0_start_sector_labels"] = q0_start_sector_labels
        scan_result["q0_mean_m_u_spread_linf_curve_matrix"] = (
            q0_mean_m_u_spread_linf_curve_matrix
        )
        scan_result["q0_mean_q_top_spread_curve_matrix"] = (
            q0_mean_q_top_spread_curve_matrix
        )
        scan_result[
            "q0_logical_observable_mean_values_per_disorder_per_start_tensor"
        ] = q0_logical_observable_mean_values_per_disorder_per_start_tensor
        scan_result["q0_q_top_values_per_disorder_per_start_tensor"] = (
            q0_q_top_values_per_disorder_per_start_tensor
        )
        scan_result["q0_m_u_spread_linf_per_disorder_tensor"] = (
            q0_m_u_spread_linf_per_disorder_tensor
        )
        scan_result["q0_q_top_spread_per_disorder_tensor"] = (
            q0_q_top_spread_per_disorder_tensor
        )
    return scan_result


if __name__ == "__main__":
    # 历史复现：原 L=3 单尺寸扫描主入口
    # lattice_size = 3
    # syndrome_error_probability = 0.0
    # data_error_probability_list = np.linspace(0.02, 0.20, 10)
    # num_disorder_samples = 40
    # num_burn_in_sweeps = 500
    # num_sweeps_between_measurements = 5
    # num_measurements_per_disorder = 400
    # seed = 20240601
    #
    # parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
    #     lattice_size=lattice_size
    # )
    # num_logical_qubits = dual_logical_z_basis.shape[0]
    #
    # scan_result = scan_data_error_probability(
    #     parity_check_matrix=parity_check_matrix,
    #     dual_logical_z_basis=dual_logical_z_basis,
    #     syndrome_error_probability=syndrome_error_probability,
    #     data_error_probability_list=data_error_probability_list,
    #     num_disorder_samples=num_disorder_samples,
    #     num_burn_in_sweeps=num_burn_in_sweeps,
    #     num_sweeps_between_measurements=num_sweeps_between_measurements,
    #     num_measurements_per_disorder=num_measurements_per_disorder,
    #     seed=seed,
    # )
    #
    # print(
    #     f"{'p':>8} {'q_top':>12} {'std_dev':>12} "
    #     f"{'avg_acceptance_rate':>20}"
    # )
    # for point_index, data_error_probability in enumerate(
    #         scan_result["data_error_probability_list"]):
    #     print(
    #         f"{data_error_probability:8.4f} "
    #         f"{scan_result['q_top_curve'][point_index]:12.6f} "
    #         f"{scan_result['q_top_std_curve'][point_index]:12.6f} "
    #         f"{scan_result['average_acceptance_rate_curve'][point_index]:20.6f}"
    #     )
    #
    # np.savez(
    #     "scan_result.npz",
    #     **scan_result,
    #     lattice_size=np.int64(lattice_size),
    #     num_logical_qubits=np.int64(num_logical_qubits),
    #     code_type=np.array("2d_toric"),
    #     syndrome_error_probability=np.float64(syndrome_error_probability),
    #     num_disorder_samples=np.int64(num_disorder_samples),
    #     num_burn_in_sweeps=np.int64(num_burn_in_sweeps),
    #     num_sweeps_between_measurements=np.int64(
    #         num_sweeps_between_measurements
    #     ),
    #     num_measurements_per_disorder=np.int64(
    #         num_measurements_per_disorder
    #     ),
    #     seed=np.int64(seed),
    # )

    lattice_size_list = [3, 5, 7]
    syndrome_error_probability = 0.0
    data_error_probability_list = np.array(
        [
            0.080,
            0.085,
            0.090,
            0.095,
            0.100,
            0.105,
            0.110,
            0.115,
            0.120,
            0.125,
            0.130,
        ],
        dtype=np.float64,
    )
    num_disorder_samples = 100
    num_burn_in_sweeps = 1000
    num_sweeps_between_measurements = 5
    num_measurements_per_disorder = 300
    q0_num_start_chains = 4
    seed_base = 20240801
    burn_in_scaling_reference_num_qubits = 18

    scan_result_multi = scan_multiple_code_sizes(
        lattice_size_list=lattice_size_list,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability_list=data_error_probability_list,
        num_disorder_samples=num_disorder_samples,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        num_measurements_per_disorder=num_measurements_per_disorder,
        seed_base=seed_base,
        q0_num_start_chains=q0_num_start_chains,
    )

    for lattice_index, lattice_size in enumerate(
            scan_result_multi["lattice_size_list"]):
        num_qubits = scan_result_multi["num_qubits_list"][lattice_index]
        effective_num_burn_in_sweeps = (
            scan_result_multi["effective_num_burn_in_sweeps_list"][
                lattice_index
            ]
        )
        print(
            f"L={int(lattice_size)} "
            f"(num_qubits={int(num_qubits)}, "
            f"effective_num_burn_in_sweeps="
            f"{int(effective_num_burn_in_sweeps)})"
        )
        print(
            f"{'p':>8} {'q_top':>12} {'std_dev':>12} "
            f"{'m_u_spread':>12} {'q_top_spread':>14} "
            f"{'acceptance_rate':>18}"
        )
        for point_index, data_error_probability in enumerate(
                scan_result_multi["data_error_probability_list"]):
            q0_m_u_spread_value = np.nan
            q0_q_top_spread_value = np.nan
            if "q0_mean_m_u_spread_linf_curve_matrix" in scan_result_multi:
                q0_m_u_spread_value = (
                    scan_result_multi[
                        "q0_mean_m_u_spread_linf_curve_matrix"
                    ][lattice_index, point_index]
                )
                q0_q_top_spread_value = (
                    scan_result_multi["q0_mean_q_top_spread_curve_matrix"][
                        lattice_index,
                        point_index,
                    ]
                )
            print(
                f"{data_error_probability:8.4f} "
                f"{scan_result_multi['q_top_curve_matrix'][lattice_index, point_index]:12.6f} "
                f"{scan_result_multi['q_top_std_curve_matrix'][lattice_index, point_index]:12.6f} "
                f"{q0_m_u_spread_value:12.6f} "
                f"{q0_q_top_spread_value:14.6f} "
                f"{scan_result_multi['average_acceptance_rate_curve_matrix'][lattice_index, point_index]:18.6f}"
            )
        print()

    crossing_header = f"{'p':>8}"
    for lattice_size in scan_result_multi["lattice_size_list"]:
        crossing_header += f" {f'L={int(lattice_size)}':>10}"
    print(crossing_header)
    for point_index, data_error_probability in enumerate(
            scan_result_multi["data_error_probability_list"]):
        row = f"{data_error_probability:8.4f}"
        for lattice_index in range(
                scan_result_multi["lattice_size_list"].shape[0]):
            row += (
                f" "
                f"{scan_result_multi['q_top_curve_matrix'][lattice_index, point_index]:10.6f}"
            )
        print(row)

    output_dir = _ensure_data_dir()
    output_path = output_dir / "scan_result_multi_L_q0_geometric_multistart.npz"
    np.savez(
        output_path,
        **scan_result_multi,
        code_type=np.array("2d_toric"),
        syndrome_error_probability=np.float64(syndrome_error_probability),
        num_disorder_samples=np.int64(num_disorder_samples),
        num_burn_in_sweeps=np.int64(num_burn_in_sweeps),
        num_sweeps_between_measurements=np.int64(
            num_sweeps_between_measurements
        ),
        num_measurements_per_disorder=np.int64(
            num_measurements_per_disorder
        ),
        seed_base=np.int64(seed_base),
        q0_num_start_chains=np.int64(q0_num_start_chains),
        burn_in_scaling_reference_num_qubits=np.int64(
            burn_in_scaling_reference_num_qubits
        ),
    )
    print(f"Saved scan result to {output_path}")
