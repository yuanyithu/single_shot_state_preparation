"""
Parallel tempering in data-error probability p.

在 K 条并行链上运行相同的 (disorder, observed syndrome)，每条链使用自己的
`data_error_probability`（因此也有不同的 `log_odds_data`）。`syndrome_error_probability`
在所有温度共享，因此 swap 的 acceptance ratio 只依赖 data-term weight：

    log ratio = (log_odds_j - log_odds_i) * (W_i - W_j)

其中 W_i = sum(data_term_bits_i)。相邻温度对 (i, i+1) 以 alternating even/odd 轮询尝试 swap。

只交换 (chain_bits, data_term_bits, syndrome_term_bits)；disorder 与 observed syndrome
不变（在所有温度共享同一个 disorder 样本）。
"""

import numpy as np

from mcmc import (
    accumulate_logical_observables,
    initialize_mcmc_state,
)
from main import (
    _build_kernel_basis_from_linear_section,
    _build_measurement_diagnostic_config,
    _compute_log_odds,
    _compute_logical_observable_values,
    _count_zero_syndrome_proposals,
    _count_zero_syndrome_proposals_split,
    _has_zero_syndrome_proposals,
    _run_measurement_update_cycle,
)
from linear_section import build_linear_section


def _attempt_replica_swaps(
        chain_bits_list,
        data_term_bits_list,
        syndrome_term_bits_list,
        data_weight_per_temperature,
        log_odds_data_per_temperature,
        rng,
        parity_index,
        swap_accept_counts,
        swap_attempt_counts):
    """
    对相邻温度对做一次 alternating even/odd swap 扫描。

    parity_index:
        0 → 尝试 (0,1), (2,3), ...
        1 → 尝试 (1,2), (3,4), ...

    chain_bits_list / data_term_bits_list / syndrome_term_bits_list 是 list of ndarray。
    Swap 就交换 list 中的对象引用，不复制缓冲区。
    """
    num_temperatures = len(chain_bits_list)
    offset = parity_index % 2
    for i in range(offset, num_temperatures - 1, 2):
        j = i + 1
        data_weight_i = int(data_weight_per_temperature[i])
        data_weight_j = int(data_weight_per_temperature[j])
        log_ratio = (
            (log_odds_data_per_temperature[j] - log_odds_data_per_temperature[i])
            * (data_weight_i - data_weight_j)
        )
        if log_ratio >= 0.0:
            accepted = True
        else:
            accepted = bool(rng.random() < np.exp(log_ratio))
        swap_attempt_counts[i] += 1
        if accepted:
            swap_accept_counts[i] += 1
            chain_bits_list[i], chain_bits_list[j] = (
                chain_bits_list[j],
                chain_bits_list[i],
            )
            data_term_bits_list[i], data_term_bits_list[j] = (
                data_term_bits_list[j],
                data_term_bits_list[i],
            )
            syndrome_term_bits_list[i], syndrome_term_bits_list[j] = (
                syndrome_term_bits_list[j],
                syndrome_term_bits_list[i],
            )
            data_weight_per_temperature[i], data_weight_per_temperature[j] = (
                data_weight_per_temperature[j],
                data_weight_per_temperature[i],
            )


def run_parallel_tempering_measurement(
        parity_check_matrix,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability_ladder,
        logical_observable_masks,
        checks_touching_each_qubit,
        num_burn_in_sweeps,
        num_measurements,
        num_sweeps_between_measurements,
        rng,
        zero_syndrome_move_data=None,
        kernel_basis=None,
        initial_chain_bits_per_temperature=None,
        num_zero_syndrome_sweeps_per_cycle=1,
        winding_repeat_factor=1,
        swap_attempt_every_num_sweeps=1,
        return_diagnostics=False,
        record_all_temperature_trajectories=False):
    """
    在 p 温度 ladder 上做 parallel tempering 采样。

    data_error_probability_ladder: 1D array，长度 K。建议 index 0 是 "cold"
        （physics 目标 p），index K-1 是 "hot"（>p_c）。

    返回 dict：
        "data_error_probability_ladder"
        "m_u_values_per_temperature"               shape (K, num_masks)
        "q_top_value_per_temperature"              shape (K,)
        "single_bit_acceptance_rate_per_temperature"          shape (K,)
        "contractible_acceptance_rate_per_temperature"        shape (K,)
        "winding_acceptance_rate_per_temperature"             shape (K,)
        "swap_accept_counts"                       shape (K-1,)
        "swap_attempt_counts"                      shape (K-1,)
        "swap_acceptance_rates"                    shape (K-1,)

    return_diagnostics=True 时，另外返回每温度每 measurement 的 logical observable trace。
    """
    data_error_probability_ladder = np.asarray(
        data_error_probability_ladder,
        dtype=np.float64,
    )
    if data_error_probability_ladder.ndim != 1:
        raise ValueError("data_error_probability_ladder must be 1D")
    num_temperatures = int(data_error_probability_ladder.shape[0])
    if num_temperatures < 1:
        raise ValueError("data_error_probability_ladder must be non-empty")

    num_checks, num_qubits = parity_check_matrix.shape
    diagnostic_config = _build_measurement_diagnostic_config(
        num_zero_syndrome_sweeps_per_cycle=(
            num_zero_syndrome_sweeps_per_cycle
        ),
        winding_repeat_factor=winding_repeat_factor,
        record_measurement_trajectories=return_diagnostics,
    )
    if zero_syndrome_move_data is None and kernel_basis is None:
        linear_section_data = build_linear_section(parity_check_matrix)
        kernel_basis = _build_kernel_basis_from_linear_section(
            parity_check_matrix=parity_check_matrix,
            linear_section_data=linear_section_data,
        )
    num_zero_syndrome_proposals = _count_zero_syndrome_proposals(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=winding_repeat_factor,
    )
    use_hybrid_zero_syndrome_sweeps = _has_zero_syndrome_proposals(
        zero_syndrome_move_data=zero_syndrome_move_data,
        kernel_basis=kernel_basis,
        winding_repeat_factor=winding_repeat_factor,
    )

    log_odds_data_per_temperature = np.array(
        [
            _compute_log_odds(float(probability))
            for probability in data_error_probability_ladder
        ],
        dtype=np.float64,
    )
    log_odds_syndrome = _compute_log_odds(syndrome_error_probability)

    chain_bits_list = []
    data_term_bits_list = []
    syndrome_term_bits_list = []
    data_weight_per_temperature = np.empty(num_temperatures, dtype=np.int64)
    for temperature_index in range(num_temperatures):
        if initial_chain_bits_per_temperature is None:
            initial_chain_bits = None
        else:
            initial_chain_bits = initial_chain_bits_per_temperature[
                temperature_index
            ]
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
        chain_bits_list.append(current_chain_bits)
        data_term_bits_list.append(current_data_term_bits)
        syndrome_term_bits_list.append(current_syndrome_term_bits)
        data_weight_per_temperature[temperature_index] = np.count_nonzero(
            current_data_term_bits
        )

    single_bit_accepted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    single_bit_attempted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    contractible_accepted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    contractible_attempted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    winding_accepted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    winding_attempted_per_temperature = np.zeros(
        num_temperatures, dtype=np.int64,
    )
    swap_accept_counts = np.zeros(
        max(num_temperatures - 1, 0), dtype=np.int64,
    )
    swap_attempt_counts = np.zeros(
        max(num_temperatures - 1, 0), dtype=np.int64,
    )
    qubit_order_buffer = np.arange(num_qubits, dtype=np.int32)

    def _run_one_sweep_for_all_temperatures():
        for temperature_index in range(num_temperatures):
            cycle_result = _run_measurement_update_cycle(
                current_chain_bits=chain_bits_list[temperature_index],
                current_data_term_bits=(
                    data_term_bits_list[temperature_index]
                ),
                current_syndrome_term_bits=(
                    syndrome_term_bits_list[temperature_index]
                ),
                syndrome_error_probability=syndrome_error_probability,
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=(
                    log_odds_data_per_temperature[temperature_index]
                ),
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
                qubit_order_buffer=qubit_order_buffer,
            )
            data_weight_per_temperature[temperature_index] += (
                cycle_result["data_weight_delta"]
            )
            single_bit_accepted_per_temperature[temperature_index] += (
                cycle_result["single_bit_accepted_count"]
            )
            single_bit_attempted_per_temperature[temperature_index] += (
                cycle_result["single_bit_attempted_count"]
            )
            contractible_accepted_per_temperature[temperature_index] += (
                cycle_result["contractible_accepted_count"]
            )
            contractible_attempted_per_temperature[temperature_index] += (
                cycle_result["contractible_attempted_count"]
            )
            winding_accepted_per_temperature[temperature_index] += (
                cycle_result["winding_accepted_count"]
            )
            winding_attempted_per_temperature[temperature_index] += (
                cycle_result["winding_attempted_count"]
            )

    swap_parity_counter = 0

    def _maybe_attempt_swap(sweep_counter):
        nonlocal swap_parity_counter
        if num_temperatures < 2:
            return
        if swap_attempt_every_num_sweeps <= 0:
            return
        if sweep_counter % swap_attempt_every_num_sweeps != 0:
            return
        _attempt_replica_swaps(
            chain_bits_list=chain_bits_list,
            data_term_bits_list=data_term_bits_list,
            syndrome_term_bits_list=syndrome_term_bits_list,
            data_weight_per_temperature=data_weight_per_temperature,
            log_odds_data_per_temperature=log_odds_data_per_temperature,
            rng=rng,
            parity_index=swap_parity_counter,
            swap_accept_counts=swap_accept_counts,
            swap_attempt_counts=swap_attempt_counts,
        )
        swap_parity_counter += 1

    sweep_counter = 0
    for _ in range(num_burn_in_sweeps):
        _run_one_sweep_for_all_temperatures()
        sweep_counter += 1
        _maybe_attempt_swap(sweep_counter)

    num_masks = logical_observable_masks.shape[0]
    logical_observable_sum_per_temperature = np.zeros(
        (num_temperatures, num_masks), dtype=np.int64,
    )
    if diagnostic_config["record_measurement_trajectories"]:
        if record_all_temperature_trajectories:
            diagnostic_temperature_indices = np.arange(
                num_temperatures,
                dtype=np.int64,
            )
        else:
            diagnostic_temperature_indices = np.array([0], dtype=np.int64)
        logical_observable_values_per_measurement = np.empty(
            (
                diagnostic_temperature_indices.shape[0],
                num_measurements,
                num_masks,
            ),
            dtype=np.int8,
        )
    else:
        diagnostic_temperature_indices = None
        logical_observable_values_per_measurement = None

    for measurement_index in range(num_measurements):
        for _ in range(num_sweeps_between_measurements):
            _run_one_sweep_for_all_temperatures()
            sweep_counter += 1
            _maybe_attempt_swap(sweep_counter)
        for temperature_index in range(num_temperatures):
            accumulate_logical_observables(
                current_chain_bits=chain_bits_list[temperature_index],
                logical_observable_masks=logical_observable_masks,
                logical_observable_sum_values=(
                    logical_observable_sum_per_temperature[temperature_index]
                ),
            )
        if diagnostic_config["record_measurement_trajectories"]:
            for diagnostic_slot, temperature_index in enumerate(
                    diagnostic_temperature_indices):
                logical_observable_values_per_measurement[
                    diagnostic_slot,
                    measurement_index,
                ] = _compute_logical_observable_values(
                    current_chain_bits=chain_bits_list[temperature_index],
                    logical_observable_masks=logical_observable_masks,
                )

    m_u_values_per_temperature = (
        logical_observable_sum_per_temperature.astype(np.float64)
        / float(num_measurements)
    )
    q_top_value_per_temperature = np.mean(
        m_u_values_per_temperature ** 2, axis=1,
    )

    def _safe_rate(accepted, attempted):
        rates = np.zeros_like(accepted, dtype=np.float64)
        nonzero_mask = attempted > 0
        rates[nonzero_mask] = (
            accepted[nonzero_mask].astype(np.float64)
            / attempted[nonzero_mask].astype(np.float64)
        )
        return rates

    single_bit_acceptance_rate_per_temperature = _safe_rate(
        single_bit_accepted_per_temperature,
        single_bit_attempted_per_temperature,
    )
    contractible_acceptance_rate_per_temperature = _safe_rate(
        contractible_accepted_per_temperature,
        contractible_attempted_per_temperature,
    )
    winding_acceptance_rate_per_temperature = _safe_rate(
        winding_accepted_per_temperature,
        winding_attempted_per_temperature,
    )
    swap_acceptance_rates = _safe_rate(
        swap_accept_counts,
        swap_attempt_counts,
    )

    result = {
        "data_error_probability_ladder": data_error_probability_ladder,
        "m_u_values_per_temperature": m_u_values_per_temperature,
        "q_top_value_per_temperature": q_top_value_per_temperature,
        "single_bit_accepted_count_per_temperature": (
            single_bit_accepted_per_temperature
        ),
        "contractible_accepted_count_per_temperature": (
            contractible_accepted_per_temperature
        ),
        "winding_accepted_count_per_temperature": (
            winding_accepted_per_temperature
        ),
        "single_bit_acceptance_rate_per_temperature": (
            single_bit_acceptance_rate_per_temperature
        ),
        "contractible_acceptance_rate_per_temperature": (
            contractible_acceptance_rate_per_temperature
        ),
        "winding_acceptance_rate_per_temperature": (
            winding_acceptance_rate_per_temperature
        ),
        "single_bit_attempted_count_per_temperature": (
            single_bit_attempted_per_temperature
        ),
        "contractible_attempted_count_per_temperature": (
            contractible_attempted_per_temperature
        ),
        "winding_attempted_count_per_temperature": (
            winding_attempted_per_temperature
        ),
        "swap_accept_counts": swap_accept_counts,
        "swap_attempt_counts": swap_attempt_counts,
        "swap_acceptance_rates": swap_acceptance_rates,
    }
    if diagnostic_config["record_measurement_trajectories"]:
        result["logical_observable_values_per_measurement_per_temperature"] = (
            logical_observable_values_per_measurement
        )
        result["diagnostic_temperature_indices"] = diagnostic_temperature_indices
    return result
