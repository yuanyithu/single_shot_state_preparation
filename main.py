import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from build_toric_code_examples import build_2d_toric_code
from linear_section import build_linear_section
from mcmc import (
    accumulate_logical_observables,
    draw_disorder_sample,
    initialize_mcmc_state,
)
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


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


def _attempt_kernel_move_update(
        current_chain_bits,
        current_data_term_bits,
        generator_bits,
        log_odds_data,
        rng):
    """
    在 q == 0 且 syndrome 固定时，沿 ker(H_Z) 方向做 Metropolis 更新。
    """
    generator_support = generator_bits
    support_size = int(np.count_nonzero(generator_support))
    if support_size == 0:
        return False

    current_ones_on_support = int(
        np.count_nonzero(current_data_term_bits[generator_support])
    )
    delta_data_weight = support_size - 2 * current_ones_on_support
    log_acceptance = delta_data_weight * log_odds_data

    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))

    if not accepted:
        return False

    current_chain_bits[generator_support] ^= True
    current_data_term_bits[generator_support] ^= True
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
            _attempt_kernel_move_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                generator_bits=generator_bits,
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )

    return accepted_count


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
        kernel_basis=None,
        log_odds_data=None,
        log_odds_syndrome=None):
    """
    对固定的 (s, eta) 运行一次 MCMC，返回 (m_u_values, acceptance_rate)。
    """
    num_qubits = parity_check_matrix.shape[1]

    if log_odds_data is None:
        log_odds_data = _compute_log_odds(data_error_probability)
    if log_odds_syndrome is None:
        log_odds_syndrome = _compute_log_odds(syndrome_error_probability)
    if syndrome_error_probability == 0.0 and kernel_basis is None:
        linear_section_data = build_linear_section(parity_check_matrix)
        kernel_basis = _build_kernel_basis_from_linear_section(
            parity_check_matrix=parity_check_matrix,
            linear_section_data=linear_section_data,
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
    )

    for _ in range(num_burn_in_sweeps):
        if syndrome_error_probability == 0.0:
            _run_one_kernel_sweep_zero_syndrome(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                kernel_basis=kernel_basis,
                log_odds_data=log_odds_data,
                rng=rng,
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
                total_accepted_count += _run_one_kernel_sweep_zero_syndrome(
                    current_chain_bits=current_chain_bits,
                    current_data_term_bits=current_data_term_bits,
                    kernel_basis=kernel_basis,
                    log_odds_data=log_odds_data,
                    rng=rng,
                )
                total_attempted_count += kernel_basis.shape[0]
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
        seed):
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
    if syndrome_error_probability == 0.0:
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

    for disorder_index in range(num_disorder_samples):
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
            num_sweeps_between_measurements=num_sweeps_between_measurements,
            rng=rng,
            kernel_basis=kernel_basis,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
        )
        q_top_value = float(np.mean(m_u_values ** 2))

        logical_observable_mean_values_per_disorder[disorder_index] = m_u_values
        disorder_q_top_values[disorder_index] = q_top_value
        average_acceptance_rate_per_disorder[disorder_index] = acceptance_rate

    disorder_average_q_top = float(np.mean(disorder_q_top_values))

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
        seed):
    data_error_probability_array = np.asarray(
        data_error_probability_list,
        dtype=np.float64,
    )
    num_points = data_error_probability_array.shape[0]

    q_top_curve = np.empty(num_points, dtype=np.float64)
    q_top_std_error_curve = np.empty(num_points, dtype=np.float64)
    average_acceptance_rate_curve = np.empty(num_points, dtype=np.float64)

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
        )

        disorder_q_top_values = result["disorder_q_top_values"]
        q_top_curve[point_index] = result["disorder_average_q_top"]
        if num_disorder_samples == 1:
            q_top_std_error_curve[point_index] = 0.0
        else:
            q_top_std_error_curve[point_index] = (
                np.std(disorder_q_top_values, ddof=1)
                / np.sqrt(num_disorder_samples)
            )
        average_acceptance_rate_curve[point_index] = float(
            np.mean(result["average_acceptance_rate_per_disorder"])
        )

    return {
        "data_error_probability_list": data_error_probability_array,
        "q_top_curve": q_top_curve,
        "q_top_std_error_curve": q_top_std_error_curve,
        "average_acceptance_rate_curve": average_acceptance_rate_curve,
    }


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
    seed = task_data["seed"]

    parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
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
    )

    disorder_q_top_values = simulation_result["disorder_q_top_values"]
    if num_disorder_samples == 1:
        q_top_std_error = 0.0
    else:
        q_top_std_error = float(
            np.std(disorder_q_top_values, ddof=1)
            / np.sqrt(num_disorder_samples)
        )

    return {
        "lattice_index": lattice_index,
        "point_index": point_index,
        "q_top": simulation_result["disorder_average_q_top"],
        "q_top_std_error": q_top_std_error,
        "average_acceptance_rate": float(
            np.mean(simulation_result["average_acceptance_rate_per_disorder"])
        ),
    }


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
        q_top_std_error_curve_matrix,
        average_acceptance_rate_curve_matrix,
        task_result):
    lattice_index = task_result["lattice_index"]
    point_index = task_result["point_index"]
    q_top_curve_matrix[lattice_index, point_index] = task_result["q_top"]
    q_top_std_error_curve_matrix[lattice_index, point_index] = (
        task_result["q_top_std_error"]
    )
    average_acceptance_rate_curve_matrix[
        lattice_index,
        point_index,
    ] = task_result["average_acceptance_rate"]


def scan_multiple_code_sizes(
        lattice_size_list,
        syndrome_error_probability,
        data_error_probability_list,
        num_disorder_samples,
        num_burn_in_sweeps,
        num_sweeps_between_measurements,
        num_measurements_per_disorder,
        seed_base):
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
    q_top_std_error_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    average_acceptance_rate_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    task_data_list = []

    for lattice_index, lattice_size in enumerate(lattice_size_array):
        parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
            lattice_size=int(lattice_size)
        )
        num_qubits = parity_check_matrix.shape[1]
        num_logical_qubits = dual_logical_z_basis.shape[0]
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
                "seed": (
                    seed_base + lattice_index * num_points + point_index
                ),
            })

    num_tasks = len(task_data_list)
    num_workers = _compute_parallel_worker_count(num_tasks)
    multiprocessing_context = _build_multiprocessing_context()

    if num_workers == 1:
        task_results = map(_run_single_scan_point_task, task_data_list)
        for task_result in task_results:
            _store_scan_point_result(
                q_top_curve_matrix=q_top_curve_matrix,
                q_top_std_error_curve_matrix=q_top_std_error_curve_matrix,
                average_acceptance_rate_curve_matrix=(
                    average_acceptance_rate_curve_matrix
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
                        q_top_std_error_curve_matrix=(
                            q_top_std_error_curve_matrix
                        ),
                        average_acceptance_rate_curve_matrix=(
                            average_acceptance_rate_curve_matrix
                        ),
                        task_result=task_result,
                    )
        except PermissionError:
            task_results = map(_run_single_scan_point_task, task_data_list)
            for task_result in task_results:
                _store_scan_point_result(
                    q_top_curve_matrix=q_top_curve_matrix,
                    q_top_std_error_curve_matrix=(
                        q_top_std_error_curve_matrix
                    ),
                    average_acceptance_rate_curve_matrix=(
                        average_acceptance_rate_curve_matrix
                    ),
                    task_result=task_result,
                )

    return {
        "lattice_size_list": lattice_size_array,
        "num_qubits_list": num_qubits_list,
        "num_logical_qubits_list": num_logical_qubits_list,
        "effective_num_burn_in_sweeps_list": (
            effective_num_burn_in_sweeps_list
        ),
        "data_error_probability_list": data_error_probability_array,
        "q_top_curve_matrix": q_top_curve_matrix,
        "q_top_std_error_curve_matrix": q_top_std_error_curve_matrix,
        "average_acceptance_rate_curve_matrix": (
            average_acceptance_rate_curve_matrix
        ),
    }


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
    #     f"{'p':>8} {'q_top':>12} {'std_error':>12} "
    #     f"{'avg_acceptance_rate':>20}"
    # )
    # for point_index, data_error_probability in enumerate(
    #         scan_result["data_error_probability_list"]):
    #     print(
    #         f"{data_error_probability:8.4f} "
    #         f"{scan_result['q_top_curve'][point_index]:12.6f} "
    #         f"{scan_result['q_top_std_error_curve'][point_index]:12.6f} "
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
            f"{'p':>8} {'q_top':>12} {'std_error':>12} "
            f"{'acceptance_rate':>18}"
        )
        for point_index, data_error_probability in enumerate(
                scan_result_multi["data_error_probability_list"]):
            print(
                f"{data_error_probability:8.4f} "
                f"{scan_result_multi['q_top_curve_matrix'][lattice_index, point_index]:12.6f} "
                f"{scan_result_multi['q_top_std_error_curve_matrix'][lattice_index, point_index]:12.6f} "
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

    np.savez(
        "scan_result_multi_L_kernel_mix.npz",
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
        burn_in_scaling_reference_num_qubits=np.int64(
            burn_in_scaling_reference_num_qubits
        ),
    )
