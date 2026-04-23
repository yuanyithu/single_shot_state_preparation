import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from build_toric_code_examples import (
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from linear_section import apply_linear_section, build_linear_section
from mcmc import (
    accumulate_logical_observables,
    draw_disorder_sample,
    draw_disorder_sample_from_uniform_values,
    initialize_mcmc_state,
)
from mcmc_diagnostics import (
    analyze_chain_diagnostics,
    equal_log_odds_ladder,
    summarize_multi_chain_convergence,
)
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_RUNS_DIR = (
    DATA_DIR / "2d_toric_code" / "without_measurement_noise"
)


def _ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def _ensure_local_run_dir(run_family):
    _ensure_data_dir()
    output_dir = LOCAL_RUNS_DIR / run_family
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    log_acceptance = _compute_single_bit_log_acceptance(
        current_data_term_bits=current_data_term_bits,
        current_syndrome_term_bits=current_syndrome_term_bits,
        qubit_index=qubit_index,
        checks_touching_each_qubit=checks_touching_each_qubit,
        log_odds_data=log_odds_data,
        log_odds_syndrome=log_odds_syndrome,
    )

    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))

    if not accepted:
        return False

    touched_checks = checks_touching_each_qubit[qubit_index]
    current_chain_bits[qubit_index] ^= True
    current_data_term_bits[qubit_index] ^= True
    current_syndrome_term_bits[touched_checks] ^= True
    return True


def _compute_single_bit_log_acceptance(
        current_data_term_bits,
        current_syndrome_term_bits,
        qubit_index,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome):
    """
    计算翻转单个 qubit 的 Metropolis 对数接受率。
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
    return float(log_acceptance)


def _compute_total_log_weight(
        current_chain_bits,
        disorder_data_error_bits,
        observed_syndrome_bits,
        parity_check_matrix,
        log_odds_data,
        log_odds_syndrome):
    """
    用完整权重重算当前链的非规范化对数后验，供诊断测试对拍。
    """
    data_weight = int(
        np.count_nonzero(current_chain_bits ^ disorder_data_error_bits)
    )
    syndrome_bits = (
        parity_check_matrix.astype(np.uint8)
        @ current_chain_bits.astype(np.uint8)
    ) % 2
    syndrome_weight = int(
        np.count_nonzero(syndrome_bits.astype(bool) ^ observed_syndrome_bits)
    )
    return float(
        data_weight * log_odds_data
        + syndrome_weight * log_odds_syndrome
    )


def run_q_positive_single_bit_acceptance_bruteforce_test(
        parity_check_matrix,
        checks_touching_each_qubit,
        syndrome_error_probability,
        data_error_probability,
        rng,
        num_random_cases=256,
        atol=1e-12):
    """
    对拍 q>0 单比特更新的 log_acceptance 与暴力总权重差值。
    """
    if syndrome_error_probability <= 0.0:
        raise ValueError(
            "run_q_positive_single_bit_acceptance_bruteforce_test requires q>0"
        )
    if data_error_probability <= 0.0:
        raise ValueError(
            "run_q_positive_single_bit_acceptance_bruteforce_test requires p>0"
        )

    num_checks, num_qubits = parity_check_matrix.shape
    log_odds_data = _compute_log_odds(data_error_probability)
    log_odds_syndrome = _compute_log_odds(syndrome_error_probability)
    matched_zero_delta_syndrome_case = False

    for case_index in range(num_random_cases):
        current_chain_bits = rng.integers(0, 2, size=num_qubits).astype(bool)
        disorder_data_error_bits = (
            rng.random(num_qubits) < data_error_probability
        )
        observed_syndrome_bits = (
            rng.random(num_checks) < syndrome_error_probability
        )
        current_data_term_bits = (
            current_chain_bits ^ disorder_data_error_bits
        )
        current_syndrome_term_bits = (
            (
                parity_check_matrix.astype(np.uint8)
                @ current_chain_bits.astype(np.uint8)
            ) % 2
        ).astype(bool) ^ observed_syndrome_bits
        qubit_index = int(rng.integers(0, num_qubits))

        touched_checks = checks_touching_each_qubit[qubit_index]
        delta_syndrome_weight = 0
        for check_index in touched_checks:
            if current_syndrome_term_bits[check_index]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
        if delta_syndrome_weight == 0:
            matched_zero_delta_syndrome_case = True

        implementation_log_acceptance = _compute_single_bit_log_acceptance(
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            qubit_index=qubit_index,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
        )
        flipped_chain_bits = current_chain_bits.copy()
        flipped_chain_bits[qubit_index] ^= True
        brute_force_log_acceptance = (
            _compute_total_log_weight(
                current_chain_bits=flipped_chain_bits,
                disorder_data_error_bits=disorder_data_error_bits,
                observed_syndrome_bits=observed_syndrome_bits,
                parity_check_matrix=parity_check_matrix,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
            )
            - _compute_total_log_weight(
                current_chain_bits=current_chain_bits,
                disorder_data_error_bits=disorder_data_error_bits,
                observed_syndrome_bits=observed_syndrome_bits,
                parity_check_matrix=parity_check_matrix,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
            )
        )
        if not np.isclose(
                implementation_log_acceptance,
                brute_force_log_acceptance,
                atol=atol,
                rtol=0.0):
            raise AssertionError(
                "q>0 single-bit log_acceptance mismatch: "
                f"case_index={case_index}, qubit_index={qubit_index}, "
                f"delta_syndrome_weight={delta_syndrome_weight}, "
                f"implementation={implementation_log_acceptance:.16e}, "
                f"bruteforce={brute_force_log_acceptance:.16e}"
            )

    if not matched_zero_delta_syndrome_case:
        raise AssertionError(
            "q>0 single-bit brute-force test did not hit delta_syndrome_weight=0"
        )

    return {
        "num_random_cases": int(num_random_cases),
        "syndrome_error_probability": float(syndrome_error_probability),
        "data_error_probability": float(data_error_probability),
        "matched_zero_delta_syndrome_case": (
            matched_zero_delta_syndrome_case
        ),
    }


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
    kernel 路径不区分 contractible 与 winding，统一归入 contractible 字段。
    """
    accepted_count = 0
    num_proposals = kernel_basis.shape[0]
    if num_proposals == 0:
        return {"contractible_accepted": 0, "winding_accepted": 0}

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

    return {"contractible_accepted": accepted_count, "winding_accepted": 0}


def _run_one_geometric_sweep_zero_syndrome(
        current_chain_bits,
        current_data_term_bits,
        zero_syndrome_move_data,
        log_odds_data,
        rng,
        winding_repeat_factor=1):
    """
    对预计算的局部闭环和 winding loop 各做一次随机顺序 sweep。
    返回按 contractible / winding 拆分的接受计数。
    """
    if winding_repeat_factor < 1:
        raise ValueError("winding_repeat_factor must be >= 1")
    contractible_accepted_count = 0
    winding_accepted_count = 0
    contractible_moves = zero_syndrome_move_data["contractible_moves"]
    winding_moves = zero_syndrome_move_data["winding_moves"]

    for move_index in rng.permutation(contractible_moves.shape[0]):
        contractible_accepted_count += int(
            _attempt_zero_syndrome_move_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                move_bits=contractible_moves[int(move_index)],
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )

    for _ in range(winding_repeat_factor):
        for move_index in rng.permutation(winding_moves.shape[0]):
            winding_accepted_count += int(
                _attempt_zero_syndrome_move_update(
                    current_chain_bits=current_chain_bits,
                    current_data_term_bits=current_data_term_bits,
                    move_bits=winding_moves[int(move_index)],
                    log_odds_data=log_odds_data,
                    rng=rng,
                )
            )

    return {
        "contractible_accepted": contractible_accepted_count,
        "winding_accepted": winding_accepted_count,
    }


def _count_zero_syndrome_proposals(
        zero_syndrome_move_data=None,
        kernel_basis=None,
        winding_repeat_factor=1):
    contractible, winding = _count_zero_syndrome_proposals_split(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=winding_repeat_factor,
    )
    return contractible + winding


def _count_zero_syndrome_proposals_split(
        zero_syndrome_move_data=None,
        kernel_basis=None,
        winding_repeat_factor=1):
    """
    返回 (contractible_proposals_per_sweep, winding_proposals_per_sweep)。
    kernel 路径把所有 proposal 归入 contractible。
    """
    if winding_repeat_factor < 1:
        raise ValueError("winding_repeat_factor must be >= 1")
    if zero_syndrome_move_data is not None:
        contractible = int(
            zero_syndrome_move_data["contractible_moves"].shape[0]
        )
        winding = int(
            winding_repeat_factor
            * zero_syndrome_move_data["winding_moves"].shape[0]
        )
        return contractible, winding
    if kernel_basis is None:
        return 0, 0
    return int(kernel_basis.shape[0]), 0


def _has_zero_syndrome_proposals(
        zero_syndrome_move_data=None,
        kernel_basis=None,
        winding_repeat_factor=1):
    return _count_zero_syndrome_proposals(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=winding_repeat_factor,
    ) > 0


def _run_one_zero_syndrome_sweep(
        current_chain_bits,
        current_data_term_bits,
        log_odds_data,
        rng,
        zero_syndrome_move_data=None,
        kernel_basis=None,
        winding_repeat_factor=1):
    if zero_syndrome_move_data is not None:
        return _run_one_geometric_sweep_zero_syndrome(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            zero_syndrome_move_data=zero_syndrome_move_data,
            log_odds_data=log_odds_data,
            rng=rng,
            winding_repeat_factor=winding_repeat_factor,
        )
    return _run_one_kernel_sweep_zero_syndrome(
        current_chain_bits=current_chain_bits,
        current_data_term_bits=current_data_term_bits,
        kernel_basis=kernel_basis,
        log_odds_data=log_odds_data,
        rng=rng,
    )


def _compute_logical_observable_values(
        current_chain_bits,
        logical_observable_masks):
    masked_bits = logical_observable_masks & current_chain_bits
    parity_bits = np.bitwise_xor.reduce(masked_bits, axis=1)
    return (1 - 2 * parity_bits.astype(np.int8)).astype(
        np.int8,
        copy=False,
    )


def _build_measurement_diagnostic_config(
        num_zero_syndrome_sweeps_per_cycle=1,
        winding_repeat_factor=1,
        record_measurement_trajectories=False):
    if num_zero_syndrome_sweeps_per_cycle < 1:
        raise ValueError("num_zero_syndrome_sweeps_per_cycle must be >= 1")
    if winding_repeat_factor < 1:
        raise ValueError("winding_repeat_factor must be >= 1")
    return {
        "num_zero_syndrome_sweeps_per_cycle": int(
            num_zero_syndrome_sweeps_per_cycle
        ),
        "winding_repeat_factor": int(winding_repeat_factor),
        "record_measurement_trajectories": bool(
            record_measurement_trajectories
        ),
    }


def _run_measurement_update_cycle(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        syndrome_error_probability,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng,
        num_qubits,
        num_zero_syndrome_proposals,
        use_hybrid_zero_syndrome_sweeps,
        zero_syndrome_move_data=None,
        kernel_basis=None,
        num_zero_syndrome_sweeps_per_cycle=1,
        winding_repeat_factor=1):
    single_bit_accepted_count = 0
    single_bit_attempted_count = 0
    contractible_accepted_count = 0
    winding_accepted_count = 0
    contractible_attempted_count = 0
    winding_attempted_count = 0

    (
        contractible_per_sweep,
        winding_per_sweep,
    ) = _count_zero_syndrome_proposals_split(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=winding_repeat_factor,
    )

    def _apply_zero_syndrome_sweeps():
        nonlocal contractible_accepted_count
        nonlocal winding_accepted_count
        nonlocal contractible_attempted_count
        nonlocal winding_attempted_count
        for _ in range(num_zero_syndrome_sweeps_per_cycle):
            sweep_result = _run_one_zero_syndrome_sweep(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                log_odds_data=log_odds_data,
                rng=rng,
                zero_syndrome_move_data=zero_syndrome_move_data,
                kernel_basis=kernel_basis,
                winding_repeat_factor=winding_repeat_factor,
            )
            contractible_accepted_count += sweep_result["contractible_accepted"]
            winding_accepted_count += sweep_result["winding_accepted"]
            contractible_attempted_count += contractible_per_sweep
            winding_attempted_count += winding_per_sweep

    if syndrome_error_probability == 0.0:
        _apply_zero_syndrome_sweeps()
        return {
            "single_bit_accepted_count": single_bit_accepted_count,
            "single_bit_attempted_count": single_bit_attempted_count,
            "contractible_accepted_count": contractible_accepted_count,
            "winding_accepted_count": winding_accepted_count,
            "contractible_attempted_count": contractible_attempted_count,
            "winding_attempted_count": winding_attempted_count,
        }

    single_bit_accepted_count += _run_one_sweep_safe(
        current_chain_bits=current_chain_bits,
        current_data_term_bits=current_data_term_bits,
        current_syndrome_term_bits=current_syndrome_term_bits,
        checks_touching_each_qubit=checks_touching_each_qubit,
        log_odds_data=log_odds_data,
        log_odds_syndrome=log_odds_syndrome,
        rng=rng,
    )
    single_bit_attempted_count += num_qubits

    if use_hybrid_zero_syndrome_sweeps:
        _apply_zero_syndrome_sweeps()

    return {
        "single_bit_accepted_count": single_bit_accepted_count,
        "single_bit_attempted_count": single_bit_attempted_count,
        "contractible_accepted_count": contractible_accepted_count,
        "winding_accepted_count": winding_accepted_count,
        "contractible_attempted_count": contractible_attempted_count,
        "winding_attempted_count": winding_attempted_count,
    }


def _build_q0_start_sector_labels(
        q0_num_start_chains,
        num_start_sector_generators):
    max_num_start_chains = 1 << num_start_sector_generators
    if q0_num_start_chains < 1 or q0_num_start_chains > max_num_start_chains:
        raise ValueError(
            "q0_num_start_chains must be in "
            f"[1, {max_num_start_chains}] for "
            f"{num_start_sector_generators} start generators"
        )
    all_labels = np.array(
        [
            "".join(
                "1" if (sector_index >> generator_index) & 1 else "0"
                for generator_index in range(num_start_sector_generators)
            )
            for sector_index in range(max_num_start_chains)
        ]
    )
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

    start_sector_generators = zero_syndrome_move_data[
        "start_sector_generators"
    ]
    num_start_sector_generators = start_sector_generators.shape[0]
    start_sector_labels = _build_q0_start_sector_labels(
        q0_num_start_chains=q0_num_start_chains,
        num_start_sector_generators=num_start_sector_generators,
    )
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
        for generator_index, bit in enumerate(label):
            if bit == "1":
                initial_chain_bits_per_start[start_index] ^= (
                    start_sector_generators[generator_index]
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


def _resolve_num_start_chains(q0_num_start_chains, num_start_chains):
    if num_start_chains is None:
        resolved_num_start_chains = int(q0_num_start_chains)
    else:
        resolved_num_start_chains = int(num_start_chains)
    if resolved_num_start_chains < 1:
        raise ValueError("num_start_chains must be >= 1")
    return resolved_num_start_chains


def _compute_total_acceptance_rate_from_counts(
        single_bit_accepted_count,
        single_bit_attempted_count,
        contractible_accepted_count,
        contractible_attempted_count,
        winding_accepted_count,
        winding_attempted_count):
    total_accepted_count = (
        int(single_bit_accepted_count)
        + int(contractible_accepted_count)
        + int(winding_accepted_count)
    )
    total_attempted_count = (
        int(single_bit_attempted_count)
        + int(contractible_attempted_count)
        + int(winding_attempted_count)
    )
    if total_attempted_count == 0:
        return 0.0
    return float(total_accepted_count / total_attempted_count)


def _run_parallel_tempering_single_chain(
        parity_check_matrix,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability_ladder,
        logical_observable_masks,
        checks_touching_each_qubit,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        rng,
        zero_syndrome_move_data=None,
        initial_chain_bits=None,
        num_zero_syndrome_sweeps_per_cycle=1,
        winding_repeat_factor=1,
        pt_swap_attempt_every_num_sweeps=1):
    from mcmc_parallel_tempering import run_parallel_tempering_measurement

    num_temperatures = int(len(data_error_probability_ladder))
    if initial_chain_bits is None:
        initial_chain_bits_per_temperature = None
    else:
        initial_chain_bits_per_temperature = np.broadcast_to(
            np.asarray(initial_chain_bits, dtype=bool),
            (num_temperatures, initial_chain_bits.shape[0]),
        ).copy()

    pt_result = run_parallel_tempering_measurement(
        parity_check_matrix=parity_check_matrix,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability_ladder=data_error_probability_ladder,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        rng=rng,
        zero_syndrome_move_data=zero_syndrome_move_data,
        initial_chain_bits_per_temperature=initial_chain_bits_per_temperature,
        num_zero_syndrome_sweeps_per_cycle=(
            num_zero_syndrome_sweeps_per_cycle
        ),
        winding_repeat_factor=winding_repeat_factor,
        swap_attempt_every_num_sweeps=pt_swap_attempt_every_num_sweeps,
        return_diagnostics=True,
    )
    cold_index = 0
    acceptance_rate = _compute_total_acceptance_rate_from_counts(
        single_bit_accepted_count=(
            pt_result["single_bit_accepted_count_per_temperature"][
                cold_index
            ]
        ),
        single_bit_attempted_count=(
            pt_result["single_bit_attempted_count_per_temperature"][
                cold_index
            ]
        ),
        contractible_accepted_count=(
            pt_result["contractible_accepted_count_per_temperature"][
                cold_index
            ]
        ),
        contractible_attempted_count=(
            pt_result["contractible_attempted_count_per_temperature"][
                cold_index
            ]
        ),
        winding_accepted_count=(
            pt_result["winding_accepted_count_per_temperature"][
                cold_index
            ]
        ),
        winding_attempted_count=(
            pt_result["winding_attempted_count_per_temperature"][
                cold_index
            ]
        ),
    )
    return {
        "m_u_values": pt_result["m_u_values_per_temperature"][cold_index],
        "q_top_value": float(
            pt_result["q_top_value_per_temperature"][cold_index]
        ),
        "acceptance_rate": acceptance_rate,
        "single_bit_acceptance_rate": float(
            pt_result["single_bit_acceptance_rate_per_temperature"][
                cold_index
            ]
        ),
        "contractible_acceptance_rate": float(
            pt_result["contractible_acceptance_rate_per_temperature"][
                cold_index
            ]
        ),
        "winding_acceptance_rate": float(
            pt_result["winding_acceptance_rate_per_temperature"][
                cold_index
            ]
        ),
        "contractible_attempted_count": np.int64(
            pt_result["contractible_attempted_count_per_temperature"][
                cold_index
            ]
        ),
        "winding_attempted_count": np.int64(
            pt_result["winding_attempted_count_per_temperature"][
                cold_index
            ]
        ),
        "logical_observable_values_per_measurement": pt_result[
            "logical_observable_values_per_measurement_per_temperature"
        ][cold_index],
        "pt_ladder": pt_result["data_error_probability_ladder"],
        "pt_q_top_value_per_temperature": (
            pt_result["q_top_value_per_temperature"]
        ),
        "pt_winding_acceptance_rate_per_temperature": (
            pt_result["winding_acceptance_rate_per_temperature"]
        ),
        "pt_swap_acceptance_rates": pt_result["swap_acceptance_rates"],
    }


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
        initial_chain_bits=None,
        num_zero_syndrome_sweeps_per_cycle=1,
        winding_repeat_factor=1,
        return_diagnostics=False):
    """
    对固定的 (s, eta) 运行一次 MCMC，返回 (m_u_values, acceptance_rate)。
    """
    num_qubits = parity_check_matrix.shape[1]
    diagnostic_config = _build_measurement_diagnostic_config(
        num_zero_syndrome_sweeps_per_cycle=(
            num_zero_syndrome_sweeps_per_cycle
        ),
        winding_repeat_factor=winding_repeat_factor,
        record_measurement_trajectories=return_diagnostics,
    )

    if log_odds_data is None:
        log_odds_data = _compute_log_odds(data_error_probability)
    if log_odds_syndrome is None:
        log_odds_syndrome = _compute_log_odds(syndrome_error_probability)
    if zero_syndrome_move_data is None and kernel_basis is None:
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
        winding_repeat_factor=diagnostic_config["winding_repeat_factor"],
    )
    use_hybrid_zero_syndrome_sweeps = _has_zero_syndrome_proposals(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=diagnostic_config["winding_repeat_factor"],
    )

    for _ in range(num_burn_in_sweeps):
        _run_measurement_update_cycle(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            syndrome_error_probability=syndrome_error_probability,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=rng,
            num_qubits=num_qubits,
            num_zero_syndrome_proposals=num_zero_syndrome_proposals,
            use_hybrid_zero_syndrome_sweeps=(
                use_hybrid_zero_syndrome_sweeps
            ),
            zero_syndrome_move_data=zero_syndrome_move_data,
            kernel_basis=kernel_basis,
            num_zero_syndrome_sweeps_per_cycle=diagnostic_config[
                "num_zero_syndrome_sweeps_per_cycle"
            ],
            winding_repeat_factor=diagnostic_config[
                "winding_repeat_factor"
            ],
        )

    num_masks = logical_observable_masks.shape[0]
    logical_observable_sum_values = np.zeros(num_masks, dtype=np.int64)
    total_single_bit_accepted_count = 0
    total_single_bit_attempted_count = 0
    total_contractible_accepted_count = 0
    total_contractible_attempted_count = 0
    total_winding_accepted_count = 0
    total_winding_attempted_count = 0
    if diagnostic_config["record_measurement_trajectories"]:
        logical_observable_values_per_measurement = np.empty(
            (num_measurements_per_disorder, num_masks),
            dtype=np.int8,
        )
        cumulative_m_u_trajectory = np.empty(
            (num_measurements_per_disorder, num_masks),
            dtype=np.float64,
        )
        cumulative_q_top_trajectory = np.empty(
            num_measurements_per_disorder,
            dtype=np.float64,
        )
        single_bit_acceptance_rate_per_measurement = np.empty(
            num_measurements_per_disorder,
            dtype=np.float64,
        )
        zero_syndrome_acceptance_rate_per_measurement = np.empty(
            num_measurements_per_disorder,
            dtype=np.float64,
        )
        contractible_acceptance_rate_per_measurement = np.empty(
            num_measurements_per_disorder,
            dtype=np.float64,
        )
        winding_acceptance_rate_per_measurement = np.empty(
            num_measurements_per_disorder,
            dtype=np.float64,
        )

    for measurement_index in range(num_measurements_per_disorder):
        measurement_single_bit_accepted_count = 0
        measurement_single_bit_attempted_count = 0
        measurement_contractible_accepted_count = 0
        measurement_contractible_attempted_count = 0
        measurement_winding_accepted_count = 0
        measurement_winding_attempted_count = 0
        for _ in range(num_sweeps_between_measurements):
            cycle_result = _run_measurement_update_cycle(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                syndrome_error_probability=syndrome_error_probability,
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
                rng=rng,
                num_qubits=num_qubits,
                num_zero_syndrome_proposals=num_zero_syndrome_proposals,
                use_hybrid_zero_syndrome_sweeps=(
                    use_hybrid_zero_syndrome_sweeps
                ),
                zero_syndrome_move_data=zero_syndrome_move_data,
                kernel_basis=kernel_basis,
                num_zero_syndrome_sweeps_per_cycle=diagnostic_config[
                    "num_zero_syndrome_sweeps_per_cycle"
                ],
                winding_repeat_factor=diagnostic_config[
                    "winding_repeat_factor"
                ],
            )
            measurement_single_bit_accepted_count += cycle_result[
                "single_bit_accepted_count"
            ]
            measurement_single_bit_attempted_count += cycle_result[
                "single_bit_attempted_count"
            ]
            measurement_contractible_accepted_count += cycle_result[
                "contractible_accepted_count"
            ]
            measurement_contractible_attempted_count += cycle_result[
                "contractible_attempted_count"
            ]
            measurement_winding_accepted_count += cycle_result[
                "winding_accepted_count"
            ]
            measurement_winding_attempted_count += cycle_result[
                "winding_attempted_count"
            ]

        accumulate_logical_observables(
            current_chain_bits=current_chain_bits,
            logical_observable_masks=logical_observable_masks,
            logical_observable_sum_values=logical_observable_sum_values,
        )
        total_single_bit_accepted_count += (
            measurement_single_bit_accepted_count
        )
        total_single_bit_attempted_count += (
            measurement_single_bit_attempted_count
        )
        total_contractible_accepted_count += (
            measurement_contractible_accepted_count
        )
        total_contractible_attempted_count += (
            measurement_contractible_attempted_count
        )
        total_winding_accepted_count += (
            measurement_winding_accepted_count
        )
        total_winding_attempted_count += (
            measurement_winding_attempted_count
        )
        if diagnostic_config["record_measurement_trajectories"]:
            logical_observable_values = _compute_logical_observable_values(
                current_chain_bits=current_chain_bits,
                logical_observable_masks=logical_observable_masks,
            )
            logical_observable_values_per_measurement[
                measurement_index
            ] = logical_observable_values
            cumulative_m_u_values = (
                logical_observable_sum_values / (measurement_index + 1)
            ).astype(np.float64, copy=False)
            cumulative_m_u_trajectory[measurement_index] = (
                cumulative_m_u_values
            )
            cumulative_q_top_trajectory[measurement_index] = float(
                np.mean(cumulative_m_u_values ** 2)
            )
            if measurement_single_bit_attempted_count == 0:
                single_bit_acceptance_rate_per_measurement[
                    measurement_index
                ] = 0.0
            else:
                single_bit_acceptance_rate_per_measurement[
                    measurement_index
                ] = (
                    measurement_single_bit_accepted_count
                    / measurement_single_bit_attempted_count
                )
            measurement_zero_syndrome_accepted_count = (
                measurement_contractible_accepted_count
                + measurement_winding_accepted_count
            )
            measurement_zero_syndrome_attempted_count = (
                measurement_contractible_attempted_count
                + measurement_winding_attempted_count
            )
            if measurement_zero_syndrome_attempted_count == 0:
                zero_syndrome_acceptance_rate_per_measurement[
                    measurement_index
                ] = 0.0
            else:
                zero_syndrome_acceptance_rate_per_measurement[
                    measurement_index
                ] = (
                    measurement_zero_syndrome_accepted_count
                    / measurement_zero_syndrome_attempted_count
                )
            if measurement_contractible_attempted_count == 0:
                contractible_acceptance_rate_per_measurement[
                    measurement_index
                ] = 0.0
            else:
                contractible_acceptance_rate_per_measurement[
                    measurement_index
                ] = (
                    measurement_contractible_accepted_count
                    / measurement_contractible_attempted_count
                )
            if measurement_winding_attempted_count == 0:
                winding_acceptance_rate_per_measurement[
                    measurement_index
                ] = 0.0
            else:
                winding_acceptance_rate_per_measurement[
                    measurement_index
                ] = (
                    measurement_winding_accepted_count
                    / measurement_winding_attempted_count
                )

    m_u_values = (
        logical_observable_sum_values / num_measurements_per_disorder
    ).astype(np.float64, copy=False)
    total_zero_syndrome_accepted_count = (
        total_contractible_accepted_count + total_winding_accepted_count
    )
    total_zero_syndrome_attempted_count = (
        total_contractible_attempted_count + total_winding_attempted_count
    )
    total_accepted_count = (
        total_single_bit_accepted_count
        + total_zero_syndrome_accepted_count
    )
    total_attempted_count = (
        total_single_bit_attempted_count
        + total_zero_syndrome_attempted_count
    )
    if total_attempted_count == 0:
        acceptance_rate = 0.0
    else:
        acceptance_rate = total_accepted_count / total_attempted_count
    if not return_diagnostics:
        return m_u_values, acceptance_rate
    if total_single_bit_attempted_count == 0:
        single_bit_acceptance_rate = 0.0
    else:
        single_bit_acceptance_rate = (
            total_single_bit_accepted_count
            / total_single_bit_attempted_count
        )
    if total_zero_syndrome_attempted_count == 0:
        zero_syndrome_acceptance_rate = 0.0
    else:
        zero_syndrome_acceptance_rate = (
            total_zero_syndrome_accepted_count
            / total_zero_syndrome_attempted_count
        )
    if total_contractible_attempted_count == 0:
        contractible_acceptance_rate = 0.0
    else:
        contractible_acceptance_rate = (
            total_contractible_accepted_count
            / total_contractible_attempted_count
        )
    if total_winding_attempted_count == 0:
        winding_acceptance_rate = 0.0
    else:
        winding_acceptance_rate = (
            total_winding_accepted_count
            / total_winding_attempted_count
        )
    return {
        "m_u_values": m_u_values,
        "q_top_value": float(np.mean(m_u_values ** 2)),
        "acceptance_rate": float(acceptance_rate),
        "single_bit_acceptance_rate": float(single_bit_acceptance_rate),
        "zero_syndrome_acceptance_rate": float(
            zero_syndrome_acceptance_rate
        ),
        "contractible_acceptance_rate": float(contractible_acceptance_rate),
        "winding_acceptance_rate": float(winding_acceptance_rate),
        "contractible_attempted_count": np.int64(
            total_contractible_attempted_count
        ),
        "winding_attempted_count": np.int64(
            total_winding_attempted_count
        ),
        "logical_observable_values_per_measurement": (
            logical_observable_values_per_measurement
        ),
        "cumulative_m_u_trajectory": cumulative_m_u_trajectory,
        "cumulative_q_top_trajectory": cumulative_q_top_trajectory,
        "single_bit_acceptance_rate_per_measurement": (
            single_bit_acceptance_rate_per_measurement
        ),
        "zero_syndrome_acceptance_rate_per_measurement": (
            zero_syndrome_acceptance_rate_per_measurement
        ),
        "contractible_acceptance_rate_per_measurement": (
            contractible_acceptance_rate_per_measurement
        ),
        "winding_acceptance_rate_per_measurement": (
            winding_acceptance_rate_per_measurement
        ),
        "num_zero_syndrome_sweeps_per_cycle": np.int64(
            diagnostic_config["num_zero_syndrome_sweeps_per_cycle"]
        ),
        "winding_repeat_factor": np.int64(
            diagnostic_config["winding_repeat_factor"]
        ),
    }


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
        num_start_chains=None,
        num_replicas_per_start=1,
        pt_p_hot=None,
        pt_num_temperatures=None,
        pt_swap_attempt_every_num_sweeps=1,
        precomputed_syndrome_uniform_values_per_disorder=None,
        precomputed_data_uniform_values_per_disorder=None):
    rng = np.random.default_rng(seed)

    num_checks, num_qubits = parity_check_matrix.shape
    num_logical_qubits = int(dual_logical_z_basis.shape[0])
    resolved_num_start_chains = _resolve_num_start_chains(
        q0_num_start_chains=q0_num_start_chains,
        num_start_chains=num_start_chains,
    )
    num_replicas_per_start = int(num_replicas_per_start)
    if num_replicas_per_start < 1:
        raise ValueError("num_replicas_per_start must be >= 1")
    use_parallel_tempering = (
        pt_p_hot is not None or pt_num_temperatures is not None
    )
    if use_parallel_tempering:
        if pt_p_hot is None or pt_num_temperatures is None:
            raise ValueError(
                "pt_p_hot and pt_num_temperatures must be provided together"
            )
        if syndrome_error_probability == 0.0:
            raise ValueError("parallel tempering is only supported for q>0")
        if float(pt_p_hot) <= float(data_error_probability):
            raise ValueError("pt_p_hot must be greater than cold p")
        if int(pt_num_temperatures) < 2:
            raise ValueError("pt_num_temperatures must be >= 2")

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
    if zero_syndrome_move_data is None:
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
        q0_start_sector_labels = _build_q0_initial_chain_bits_per_start(
            observed_syndrome_bits=np.zeros(num_checks, dtype=bool),
            linear_section_data=linear_section_data,
            zero_syndrome_move_data=zero_syndrome_move_data,
            q0_num_start_chains=resolved_num_start_chains,
        )[1]
        q0_logical_observable_mean_values_per_disorder_per_start = np.empty(
            (num_disorder_samples, resolved_num_start_chains, num_masks),
            dtype=np.float64,
        )
        q0_q_top_values_per_disorder_per_start = np.empty(
            (num_disorder_samples, resolved_num_start_chains),
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
    q_positive_start_sector_labels = None
    chain_logical_observable_mean_values_per_disorder_per_start_replica = None
    chain_q_top_values_per_disorder_per_start_replica = None
    chain_average_acceptance_rate_per_disorder_per_start_replica = None
    chain_contractible_acceptance_rate_per_disorder_per_start_replica = None
    chain_winding_acceptance_rate_per_disorder_per_start_replica = None
    q_top_spread_per_disorder = None
    m_u_spread_linf_per_disorder = None
    max_r_hat_per_disorder = None
    min_effective_sample_size_per_disorder = None
    num_chains_that_never_flipped_sector_per_disorder = None
    pt_min_swap_acceptance_rate_per_disorder = None
    pt_mean_swap_acceptance_rate_per_disorder = None
    if syndrome_error_probability > 0.0:
        if zero_syndrome_move_data is None and resolved_num_start_chains > 1:
            raise ValueError(
                "zero_syndrome_move_data is required when num_start_chains > 1"
            )
        if zero_syndrome_move_data is None:
            q_positive_start_sector_labels = np.array(["0"])
        else:
            q_positive_start_sector_labels = (
                _build_q0_initial_chain_bits_per_start(
                    observed_syndrome_bits=np.zeros(num_checks, dtype=bool),
                    linear_section_data=linear_section_data,
                    zero_syndrome_move_data=zero_syndrome_move_data,
                    q0_num_start_chains=resolved_num_start_chains,
                )[1]
            )
        chain_shape = (
            num_disorder_samples,
            resolved_num_start_chains,
            num_replicas_per_start,
        )
        chain_logical_observable_mean_values_per_disorder_per_start_replica = (
            np.empty(chain_shape + (num_masks,), dtype=np.float64)
        )
        chain_q_top_values_per_disorder_per_start_replica = np.empty(
            chain_shape,
            dtype=np.float64,
        )
        chain_average_acceptance_rate_per_disorder_per_start_replica = (
            np.empty(chain_shape, dtype=np.float64)
        )
        chain_contractible_acceptance_rate_per_disorder_per_start_replica = (
            np.empty(chain_shape, dtype=np.float64)
        )
        chain_winding_acceptance_rate_per_disorder_per_start_replica = (
            np.empty(chain_shape, dtype=np.float64)
        )
        q_top_spread_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )
        m_u_spread_linf_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )
        max_r_hat_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )
        min_effective_sample_size_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.float64,
        )
        num_chains_that_never_flipped_sector_per_disorder = np.empty(
            num_disorder_samples,
            dtype=np.int64,
        )
        if use_parallel_tempering:
            pt_min_swap_acceptance_rate_per_disorder = np.empty(
                num_disorder_samples,
                dtype=np.float64,
            )
            pt_mean_swap_acceptance_rate_per_disorder = np.empty(
                num_disorder_samples,
                dtype=np.float64,
            )
            data_error_probability_ladder = equal_log_odds_ladder(
                p_cold=float(data_error_probability),
                p_hot=float(pt_p_hot),
                num_temperatures=int(pt_num_temperatures),
            )
        else:
            data_error_probability_ladder = None

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
                q0_num_start_chains=resolved_num_start_chains,
            )
            q0_m_u_values_per_start = np.empty(
                (resolved_num_start_chains, num_masks),
                dtype=np.float64,
            )
            q0_q_top_values_per_start = np.empty(
                resolved_num_start_chains,
                dtype=np.float64,
            )
            q0_acceptance_rates_per_start = np.empty(
                resolved_num_start_chains,
                dtype=np.float64,
            )

            for start_index in range(resolved_num_start_chains):
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
            if zero_syndrome_move_data is None:
                section_representative = apply_linear_section(
                    observed_syndrome_bits,
                    linear_section_data,
                )
                initial_chain_bits_per_start = section_representative[
                    None, :
                ]
            else:
                (
                    initial_chain_bits_per_start,
                    _,
                ) = _build_q0_initial_chain_bits_per_start(
                    observed_syndrome_bits=observed_syndrome_bits,
                    linear_section_data=linear_section_data,
                    zero_syndrome_move_data=zero_syndrome_move_data,
                    q0_num_start_chains=resolved_num_start_chains,
                )

            chain_m_u_values_per_start_replica = np.empty(
                (
                    resolved_num_start_chains,
                    num_replicas_per_start,
                    num_masks,
                ),
                dtype=np.float64,
            )
            chain_q_top_values_per_start_replica = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.float64,
            )
            chain_average_acceptance_rates_per_start_replica = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.float64,
            )
            chain_contractible_acceptance_rates_per_start_replica = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.float64,
            )
            chain_winding_acceptance_rates_per_start_replica = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.float64,
            )
            chain_effective_sample_size_values = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.float64,
            )
            chain_first_signature_change_index_values = np.empty(
                (resolved_num_start_chains, num_replicas_per_start),
                dtype=np.int64,
            )
            logical_observable_values_tensor = np.empty(
                (
                    resolved_num_start_chains * num_replicas_per_start,
                    num_measurements_per_disorder,
                    num_masks,
                ),
                dtype=np.int8,
            )
            if use_parallel_tempering:
                pt_min_swap_acceptance_rates_per_start_replica = np.empty(
                    (resolved_num_start_chains, num_replicas_per_start),
                    dtype=np.float64,
                )
                pt_mean_swap_acceptance_rates_per_start_replica = np.empty(
                    (resolved_num_start_chains, num_replicas_per_start),
                    dtype=np.float64,
                )

            flattened_chain_index = 0
            for start_index in range(resolved_num_start_chains):
                for replica_index in range(num_replicas_per_start):
                    chain_seed = int(
                        rng.integers(
                            0,
                            np.iinfo(np.uint64).max,
                            dtype=np.uint64,
                        )
                    )
                    chain_rng = np.random.default_rng(chain_seed)
                    if use_parallel_tempering:
                        measurement_result = (
                            _run_parallel_tempering_single_chain(
                                parity_check_matrix=parity_check_matrix,
                                observed_syndrome_bits=(
                                    observed_syndrome_bits
                                ),
                                disorder_data_error_bits=(
                                    disorder_data_error_bits
                                ),
                                syndrome_error_probability=(
                                    syndrome_error_probability
                                ),
                                data_error_probability_ladder=(
                                    data_error_probability_ladder
                                ),
                                logical_observable_masks=(
                                    logical_observable_masks
                                ),
                                checks_touching_each_qubit=(
                                    checks_touching_each_qubit
                                ),
                                num_burn_in_sweeps=num_burn_in_sweeps,
                                num_measurements_per_disorder=(
                                    num_measurements_per_disorder
                                ),
                                num_sweeps_between_measurements=(
                                    num_sweeps_between_measurements
                                ),
                                rng=chain_rng,
                                zero_syndrome_move_data=(
                                    zero_syndrome_move_data
                                ),
                                initial_chain_bits=(
                                    initial_chain_bits_per_start[
                                        start_index
                                    ]
                                ),
                                pt_swap_attempt_every_num_sweeps=(
                                    pt_swap_attempt_every_num_sweeps
                                ),
                            )
                        )
                        pt_swap_acceptance_rates = np.asarray(
                            measurement_result["pt_swap_acceptance_rates"],
                            dtype=np.float64,
                        )
                        if pt_swap_acceptance_rates.size == 0:
                            pt_min_swap_acceptance_rates_per_start_replica[
                                start_index,
                                replica_index,
                            ] = 0.0
                            pt_mean_swap_acceptance_rates_per_start_replica[
                                start_index,
                                replica_index,
                            ] = 0.0
                        else:
                            pt_min_swap_acceptance_rates_per_start_replica[
                                start_index,
                                replica_index,
                            ] = float(np.min(pt_swap_acceptance_rates))
                            pt_mean_swap_acceptance_rates_per_start_replica[
                                start_index,
                                replica_index,
                            ] = float(np.mean(pt_swap_acceptance_rates))
                    else:
                        measurement_result = _run_single_disorder_measurement(
                            parity_check_matrix=parity_check_matrix,
                            observed_syndrome_bits=observed_syndrome_bits,
                            disorder_data_error_bits=(
                                disorder_data_error_bits
                            ),
                            syndrome_error_probability=(
                                syndrome_error_probability
                            ),
                            data_error_probability=data_error_probability,
                            logical_observable_masks=(
                                logical_observable_masks
                            ),
                            checks_touching_each_qubit=(
                                checks_touching_each_qubit
                            ),
                            num_burn_in_sweeps=num_burn_in_sweeps,
                            num_measurements_per_disorder=(
                                num_measurements_per_disorder
                            ),
                            num_sweeps_between_measurements=(
                                num_sweeps_between_measurements
                            ),
                            rng=chain_rng,
                            zero_syndrome_move_data=(
                                zero_syndrome_move_data
                            ),
                            kernel_basis=kernel_basis,
                            log_odds_data=log_odds_data,
                            log_odds_syndrome=log_odds_syndrome,
                            initial_chain_bits=(
                                initial_chain_bits_per_start[start_index]
                            ),
                            return_diagnostics=True,
                        )
                    chain_analysis = analyze_chain_diagnostics(
                        logical_observable_values_per_measurement=(
                            measurement_result[
                                "logical_observable_values_per_measurement"
                            ]
                        ),
                        num_logical_qubits=num_logical_qubits,
                    )
                    chain_m_u_values_per_start_replica[
                        start_index,
                        replica_index,
                    ] = measurement_result["m_u_values"]
                    chain_q_top_values_per_start_replica[
                        start_index,
                        replica_index,
                    ] = float(measurement_result["q_top_value"])
                    chain_average_acceptance_rates_per_start_replica[
                        start_index,
                        replica_index,
                    ] = float(measurement_result["acceptance_rate"])
                    chain_contractible_acceptance_rates_per_start_replica[
                        start_index,
                        replica_index,
                    ] = float(
                        measurement_result.get(
                            "contractible_acceptance_rate",
                            0.0,
                        )
                    )
                    chain_winding_acceptance_rates_per_start_replica[
                        start_index,
                        replica_index,
                    ] = float(
                        measurement_result.get(
                            "winding_acceptance_rate",
                            0.0,
                        )
                    )
                    chain_effective_sample_size_values[
                        start_index,
                        replica_index,
                    ] = float(chain_analysis["effective_sample_size"])
                    chain_first_signature_change_index_values[
                        start_index,
                        replica_index,
                    ] = int(
                        chain_analysis["first_signature_change_index"]
                    )
                    logical_observable_values_tensor[
                        flattened_chain_index
                    ] = measurement_result[
                        "logical_observable_values_per_measurement"
                    ]
                    flattened_chain_index += 1

            flattened_chain_m_u_values = (
                chain_m_u_values_per_start_replica.reshape(
                    resolved_num_start_chains * num_replicas_per_start,
                    num_masks,
                )
            )
            flattened_chain_q_top_values = (
                chain_q_top_values_per_start_replica.reshape(-1)
            )
            flattened_chain_effective_sample_size_values = (
                chain_effective_sample_size_values.reshape(-1)
            )
            flattened_chain_first_signature_change_index_values = (
                chain_first_signature_change_index_values.reshape(-1)
            )
            convergence_summary = summarize_multi_chain_convergence(
                chain_m_u_values=flattened_chain_m_u_values,
                chain_q_top_values=flattened_chain_q_top_values,
                chain_effective_sample_size_values=(
                    flattened_chain_effective_sample_size_values
                ),
                chain_first_signature_change_index_values=(
                    flattened_chain_first_signature_change_index_values
                ),
                logical_observable_values_tensor=(
                    logical_observable_values_tensor
                ),
            )

            m_u_values = np.mean(flattened_chain_m_u_values, axis=0)
            acceptance_rate = float(np.mean(
                chain_average_acceptance_rates_per_start_replica
            ))
            q_top_value = float(np.mean(m_u_values ** 2))
            chain_logical_observable_mean_values_per_disorder_per_start_replica[
                disorder_index
            ] = chain_m_u_values_per_start_replica
            chain_q_top_values_per_disorder_per_start_replica[
                disorder_index
            ] = chain_q_top_values_per_start_replica
            chain_average_acceptance_rate_per_disorder_per_start_replica[
                disorder_index
            ] = chain_average_acceptance_rates_per_start_replica
            chain_contractible_acceptance_rate_per_disorder_per_start_replica[
                disorder_index
            ] = chain_contractible_acceptance_rates_per_start_replica
            chain_winding_acceptance_rate_per_disorder_per_start_replica[
                disorder_index
            ] = chain_winding_acceptance_rates_per_start_replica
            q_top_spread_per_disorder[disorder_index] = (
                convergence_summary["q_top_spread"]
            )
            m_u_spread_linf_per_disorder[disorder_index] = (
                convergence_summary["m_u_spread_linf"]
            )
            max_r_hat_per_disorder[disorder_index] = (
                convergence_summary["max_r_hat"]
            )
            min_effective_sample_size_per_disorder[disorder_index] = (
                convergence_summary["min_effective_sample_size"]
            )
            num_chains_that_never_flipped_sector_per_disorder[
                disorder_index
            ] = convergence_summary[
                "num_chains_that_never_flipped_sector"
            ]
            if use_parallel_tempering:
                pt_min_swap_acceptance_rate_per_disorder[
                    disorder_index
                ] = float(np.min(
                    pt_min_swap_acceptance_rates_per_start_replica
                ))
                pt_mean_swap_acceptance_rate_per_disorder[
                    disorder_index
                ] = float(np.mean(
                    pt_mean_swap_acceptance_rates_per_start_replica
                ))

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

    result = {
        "disorder_q_top_values": disorder_q_top_values,
        "disorder_average_q_top": disorder_average_q_top,
        "logical_observable_mean_values_per_disorder": (
            logical_observable_mean_values_per_disorder
        ),
        "average_acceptance_rate_per_disorder": (
            average_acceptance_rate_per_disorder
        ),
    }
    if q_positive_start_sector_labels is not None:
        result["start_sector_labels"] = q_positive_start_sector_labels
        result["num_start_chains"] = np.int64(resolved_num_start_chains)
        result["num_replicas_per_start"] = np.int64(
            num_replicas_per_start
        )
        result[
            "chain_logical_observable_mean_values_per_disorder_per_start_replica"
        ] = chain_logical_observable_mean_values_per_disorder_per_start_replica
        result["chain_q_top_values_per_disorder_per_start_replica"] = (
            chain_q_top_values_per_disorder_per_start_replica
        )
        result[
            "chain_average_acceptance_rate_per_disorder_per_start_replica"
        ] = chain_average_acceptance_rate_per_disorder_per_start_replica
        result[
            "chain_contractible_acceptance_rate_per_disorder_per_start_replica"
        ] = chain_contractible_acceptance_rate_per_disorder_per_start_replica
        result[
            "chain_winding_acceptance_rate_per_disorder_per_start_replica"
        ] = chain_winding_acceptance_rate_per_disorder_per_start_replica
        result["q_top_spread_per_disorder"] = q_top_spread_per_disorder
        result["m_u_spread_linf_per_disorder"] = (
            m_u_spread_linf_per_disorder
        )
        result["max_r_hat_per_disorder"] = max_r_hat_per_disorder
        result["min_effective_sample_size_per_disorder"] = (
            min_effective_sample_size_per_disorder
        )
        result["num_chains_that_never_flipped_sector_per_disorder"] = (
            num_chains_that_never_flipped_sector_per_disorder
        )
        result["pt_enabled"] = np.bool_(use_parallel_tempering)
        if use_parallel_tempering:
            result["pt_p_hot"] = np.float64(pt_p_hot)
            result["pt_num_temperatures"] = np.int64(pt_num_temperatures)
            result["pt_min_swap_acceptance_rate_per_disorder"] = (
                pt_min_swap_acceptance_rate_per_disorder
            )
            result["pt_mean_swap_acceptance_rate_per_disorder"] = (
                pt_mean_swap_acceptance_rate_per_disorder
            )
    return result


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
        q0_num_start_chains=4,
        num_start_chains=None,
        num_replicas_per_start=1,
        pt_p_hot=None,
        pt_num_temperatures=None,
        pt_swap_attempt_every_num_sweeps=1):
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
            num_start_chains=num_start_chains,
            num_replicas_per_start=num_replicas_per_start,
            pt_p_hot=pt_p_hot,
            pt_num_temperatures=pt_num_temperatures,
            pt_swap_attempt_every_num_sweeps=(
                pt_swap_attempt_every_num_sweeps
            ),
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
        scan_result["q0_start_sector_labels"] = result[
            "q0_start_sector_labels"
        ]
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
    num_start_chains = task_data.get("num_start_chains")
    num_replicas_per_start = task_data.get("num_replicas_per_start", 1)
    pt_p_hot = task_data.get("pt_p_hot")
    pt_num_temperatures = task_data.get("pt_num_temperatures")
    pt_swap_attempt_every_num_sweeps = task_data.get(
        "pt_swap_attempt_every_num_sweeps",
        1,
    )
    seed = task_data["seed"]
    code_family = task_data["code_family"]

    parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
        code_family=code_family,
        lattice_size=lattice_size,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family=code_family,
        lattice_size=lattice_size,
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
        num_start_chains=num_start_chains,
        num_replicas_per_start=num_replicas_per_start,
        pt_p_hot=pt_p_hot,
        pt_num_temperatures=pt_num_temperatures,
        pt_swap_attempt_every_num_sweeps=(
            pt_swap_attempt_every_num_sweeps
        ),
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
        q0_num_start_chains=4,
        num_start_chains=None,
        num_replicas_per_start=1,
        pt_p_hot=None,
        pt_num_temperatures=None,
        pt_swap_attempt_every_num_sweeps=1,
        code_family="2d_toric"):
    """
    扫描多个 toric code 尺寸，并在内部对 burn-in 做线性放大。
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
        parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
            code_family=code_family,
            lattice_size=int(lattice_size),
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
                "num_start_chains": num_start_chains,
                "num_replicas_per_start": num_replicas_per_start,
                "pt_p_hot": pt_p_hot,
                "pt_num_temperatures": pt_num_temperatures,
                "pt_swap_attempt_every_num_sweeps": (
                    pt_swap_attempt_every_num_sweeps
                ),
                "code_family": code_family,
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
        q0_num_start_sector_generators = int(num_logical_qubits_list[0])
        q0_start_sector_labels = _build_q0_start_sector_labels(
            q0_num_start_chains=q0_num_start_chains,
            num_start_sector_generators=q0_num_start_sector_generators,
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

    output_dir = _ensure_local_run_dir("q0_geometric_multistart_local")
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
