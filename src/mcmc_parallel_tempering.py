"""
Parallel tempering in error probabilities.

在 K 条并行链上运行相同的 (disorder, observed syndrome)。每条链使用自己的
`data_error_probability`；可选地也可以使用自己的 `syndrome_error_probability`。
swap 的 acceptance ratio 同时包含 data 和 syndrome 权重：

    log ratio = (log_odds_p[j] - log_odds_p[i]) * (W_data_i - W_data_j)
              + (log_odds_q[j] - log_odds_q[i]) * (W_syn_i - W_syn_j)

旧的 data-only PT 传入标量 q 即可，此时 syndrome 项为 0。相邻温度对
(i, i+1) 以 alternating even/odd 轮询尝试 swap。

只交换 (chain_bits, data_term_bits, syndrome_term_bits)；disorder 与 observed syndrome
不变（在所有温度共享同一个 disorder 样本）。
"""

import time

import numpy as np

from cluster_update import (
    build_cluster_controller,
    freeze_cluster_controller,
    make_disabled_cluster_summary,
    maybe_run_cluster_update,
    summarize_cluster_controller,
)
from mcmc import initialize_mcmc_state
from mcmc_diagnostics import (
    build_adaptive_pt_flow_tracker,
    record_adaptive_pt_flow,
    summarize_adaptive_pt_flow,
)
from main import (
    _accumulate_logical_observables_fast,
    _build_kernel_basis_from_linear_section,
    _build_measurement_diagnostic_config,
    _build_numba_update_kernel_data,
    _compute_logical_observable_values,
    _compute_log_odds,
    _count_zero_syndrome_proposals,
    _count_zero_syndrome_proposals_split,
    _has_zero_syndrome_proposals,
    _run_measurement_update_cycle,
)
from linear_section import build_linear_section
from linear_section import apply_section, build_syndrome_representative_section


def _attempt_replica_swaps(
        chain_bits_list,
        data_term_bits_list,
        syndrome_term_bits_list,
        data_weight_per_temperature,
        log_odds_data_per_temperature,
        rng,
        parity_index,
        swap_accept_counts,
        swap_attempt_counts,
        syndrome_weight_per_temperature=None,
        log_odds_syndrome_per_temperature=None,
        replica_id_per_temperature=None):
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
        if (
                syndrome_weight_per_temperature is not None
                and log_odds_syndrome_per_temperature is not None):
            syndrome_weight_i = int(syndrome_weight_per_temperature[i])
            syndrome_weight_j = int(syndrome_weight_per_temperature[j])
            log_ratio += (
                (
                    log_odds_syndrome_per_temperature[j]
                    - log_odds_syndrome_per_temperature[i]
                )
                * (syndrome_weight_i - syndrome_weight_j)
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
            if syndrome_weight_per_temperature is not None:
                (
                    syndrome_weight_per_temperature[i],
                    syndrome_weight_per_temperature[j],
                ) = (
                    syndrome_weight_per_temperature[j],
                    syndrome_weight_per_temperature[i],
                )
            if replica_id_per_temperature is not None:
                replica_id_per_temperature[i], replica_id_per_temperature[j] = (
                    replica_id_per_temperature[j],
                    replica_id_per_temperature[i],
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
        record_all_temperature_trajectories=False,
        cluster_update_enabled=True,
        cluster_budget_fraction_rho=0.05,
        cluster_update_debug=False,
        section_data=None,
        single_bit_proposal_fraction=1.0,
        observable_temperature_mode="all",
        syndrome_error_probability_ladder=None,
        adaptive_pt_flow_enabled=False,
        track_logical_sector_diagnostics=False,
        logical_sector_diagnostic_stride=1,
        num_logical_qubits=None):
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
    if observable_temperature_mode not in ("all", "cold"):
        raise ValueError("observable_temperature_mode must be all or cold")
    logical_sector_diagnostic_stride = int(logical_sector_diagnostic_stride)
    if logical_sector_diagnostic_stride < 1:
        raise ValueError("logical_sector_diagnostic_stride must be >= 1")
    if syndrome_error_probability_ladder is None:
        syndrome_error_probability_ladder = np.full(
            num_temperatures,
            float(syndrome_error_probability),
            dtype=np.float64,
        )
    else:
        syndrome_error_probability_ladder = np.asarray(
            syndrome_error_probability_ladder,
            dtype=np.float64,
        )
        if syndrome_error_probability_ladder.ndim != 1:
            raise ValueError("syndrome_error_probability_ladder must be 1D")
        if syndrome_error_probability_ladder.shape != data_error_probability_ladder.shape:
            raise ValueError(
                "syndrome_error_probability_ladder must match data ladder shape"
            )
        if not np.isclose(
                float(syndrome_error_probability_ladder[0]),
                float(syndrome_error_probability),
                rtol=1e-12,
                atol=0.0):
            raise ValueError(
                "syndrome_error_probability_ladder[0] must match "
                "syndrome_error_probability"
            )
    has_nonconstant_syndrome_ladder = bool(
        np.any(
            syndrome_error_probability_ladder
            != syndrome_error_probability_ladder[0]
        )
    )
    if has_nonconstant_syndrome_ladder and cluster_update_enabled:
        raise ValueError(
            "cluster update is not supported with nonconstant PT q ladder"
        )

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
    if section_data is None:
        section_data = build_syndrome_representative_section(
            parity_check_matrix
        )
    disorder_syndrome_bits = (
        parity_check_matrix.astype(np.uint8)
        @ disorder_data_error_bits.astype(np.uint8)
    ) % 2
    disorder_syndrome_representative_bits = apply_section(
        disorder_syndrome_bits.astype(bool),
        section_data,
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
    numba_update_kernel_data = _build_numba_update_kernel_data(
        checks_touching_each_qubit=checks_touching_each_qubit,
        zero_syndrome_move_data=zero_syndrome_move_data,
        num_qubits=num_qubits,
    )

    log_odds_data_per_temperature = np.array(
        [
            _compute_log_odds(float(probability))
            for probability in data_error_probability_ladder
        ],
        dtype=np.float64,
    )
    log_odds_syndrome_per_temperature = np.array(
        [
            _compute_log_odds(float(probability))
            for probability in syndrome_error_probability_ladder
        ],
        dtype=np.float64,
    )

    chain_bits_list = []
    data_term_bits_list = []
    syndrome_term_bits_list = []
    data_weight_per_temperature = np.empty(num_temperatures, dtype=np.int64)
    syndrome_weight_per_temperature = np.empty(num_temperatures, dtype=np.int64)
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
        syndrome_weight_per_temperature[temperature_index] = np.count_nonzero(
            current_syndrome_term_bits
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
    replica_id_per_temperature = np.arange(num_temperatures, dtype=np.int64)
    replica_last_endpoint_per_replica = np.full(
        num_temperatures,
        -1,
        dtype=np.int64,
    )
    replica_cold_visit_count = np.zeros(num_temperatures, dtype=np.int64)
    replica_hot_visit_count = np.zeros(num_temperatures, dtype=np.int64)
    replica_cold_to_hot_passage_count = np.zeros(
        num_temperatures,
        dtype=np.int64,
    )
    replica_hot_to_cold_passage_count = np.zeros(
        num_temperatures,
        dtype=np.int64,
    )
    replica_min_temperature_visited = np.full(
        num_temperatures,
        num_temperatures,
        dtype=np.int64,
    )
    replica_max_temperature_visited = np.full(
        num_temperatures,
        -1,
        dtype=np.int64,
    )
    transport_tracking_started = False
    transport_position_sample_count = 0
    flow_tracker = (
        build_adaptive_pt_flow_tracker(num_temperatures)
        if adaptive_pt_flow_enabled
        else None
    )
    cluster_controller = build_cluster_controller(
        parity_check_matrix=parity_check_matrix,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability_ladder=data_error_probability_ladder,
        enabled=cluster_update_enabled,
        budget_fraction_rho=cluster_budget_fraction_rho,
        debug_assertions=cluster_update_debug,
    )

    def _run_one_sweep_for_all_temperatures():
        nonlocal ordinary_update_wall_time
        ordinary_elapsed_per_temperature = np.empty(
            num_temperatures,
            dtype=np.float64,
        )
        for temperature_index in range(num_temperatures):
            ordinary_started_at = time.perf_counter()
            cycle_result = _run_measurement_update_cycle(
                current_chain_bits=chain_bits_list[temperature_index],
                current_data_term_bits=(
                    data_term_bits_list[temperature_index]
                ),
                current_syndrome_term_bits=(
                    syndrome_term_bits_list[temperature_index]
                ),
                syndrome_error_probability=float(
                    syndrome_error_probability_ladder[temperature_index]
                ),
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=(
                    log_odds_data_per_temperature[temperature_index]
                ),
                log_odds_syndrome=(
                    log_odds_syndrome_per_temperature[temperature_index]
                ),
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
                numba_update_kernel_data=numba_update_kernel_data,
                single_bit_proposal_fraction=single_bit_proposal_fraction,
            )
            ordinary_elapsed_per_temperature[temperature_index] = (
                time.perf_counter() - ordinary_started_at
            )
            data_weight_per_temperature[temperature_index] += (
                cycle_result["data_weight_delta"]
            )
            syndrome_weight_per_temperature[temperature_index] = (
                np.count_nonzero(syndrome_term_bits_list[temperature_index])
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
        cluster_result = maybe_run_cluster_update(
            controller=cluster_controller,
            chain_bits_list=chain_bits_list,
            data_term_bits_list=data_term_bits_list,
            syndrome_term_bits_list=syndrome_term_bits_list,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            ordinary_elapsed_per_temperature=ordinary_elapsed_per_temperature,
            rng=rng,
        )
        if cluster_result["attempted"]:
            data_weight_per_temperature[
                cluster_result["temperature_index"]
            ] += cluster_result["data_weight_delta"]
            syndrome_weight_per_temperature[
                cluster_result["temperature_index"]
            ] = np.count_nonzero(
                syndrome_term_bits_list[cluster_result["temperature_index"]]
            )
        ordinary_update_wall_time += float(np.sum(ordinary_elapsed_per_temperature))

    swap_parity_counter = 0

    def _maybe_attempt_swap(sweep_counter):
        nonlocal swap_parity_counter
        nonlocal pt_swap_wall_time
        if num_temperatures < 2:
            return
        if swap_attempt_every_num_sweeps <= 0:
            return
        if sweep_counter % swap_attempt_every_num_sweeps != 0:
            return
        swap_started_at = time.perf_counter()
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
            syndrome_weight_per_temperature=syndrome_weight_per_temperature,
            log_odds_syndrome_per_temperature=(
                log_odds_syndrome_per_temperature
            ),
            replica_id_per_temperature=replica_id_per_temperature,
        )
        record_adaptive_pt_flow(
            flow_tracker=flow_tracker,
            replica_id_per_temperature=replica_id_per_temperature,
        )
        _record_replica_transport_position()
        pt_swap_wall_time += time.perf_counter() - swap_started_at
        swap_parity_counter += 1

    def _record_replica_transport_position():
        nonlocal transport_position_sample_count
        if not transport_tracking_started:
            return
        transport_position_sample_count += 1
        hot_index = num_temperatures - 1
        for temperature_index in range(num_temperatures):
            replica_id = int(replica_id_per_temperature[temperature_index])
            if temperature_index < replica_min_temperature_visited[replica_id]:
                replica_min_temperature_visited[replica_id] = temperature_index
            if temperature_index > replica_max_temperature_visited[replica_id]:
                replica_max_temperature_visited[replica_id] = temperature_index
            if temperature_index == 0:
                previous_endpoint = int(
                    replica_last_endpoint_per_replica[replica_id]
                )
                if previous_endpoint != 0:
                    replica_cold_visit_count[replica_id] += 1
                    if previous_endpoint == hot_index:
                        replica_hot_to_cold_passage_count[replica_id] += 1
                    replica_last_endpoint_per_replica[replica_id] = 0
            elif temperature_index == hot_index:
                previous_endpoint = int(
                    replica_last_endpoint_per_replica[replica_id]
                )
                if previous_endpoint != hot_index:
                    replica_hot_visit_count[replica_id] += 1
                    if previous_endpoint == 0:
                        replica_cold_to_hot_passage_count[replica_id] += 1
                    replica_last_endpoint_per_replica[replica_id] = hot_index

    sweep_counter = 0
    ordinary_update_wall_time = 0.0
    pt_swap_wall_time = 0.0
    observable_wall_time = 0.0
    measurement_started_at = time.perf_counter()
    for _ in range(num_burn_in_sweeps):
        _run_one_sweep_for_all_temperatures()
        sweep_counter += 1
        _maybe_attempt_swap(sweep_counter)

    if cluster_controller is not None and num_burn_in_sweeps > 0:
        freeze_cluster_controller(cluster_controller)
    transport_tracking_started = True
    _record_replica_transport_position()

    num_masks = logical_observable_masks.shape[0]
    if num_logical_qubits is None:
        inferred_num_logical_qubits = int(round(np.log2(num_masks + 1)))
        if (1 << inferred_num_logical_qubits) - 1 != num_masks:
            inferred_num_logical_qubits = min(num_masks, inferred_num_logical_qubits)
        num_logical_qubits = inferred_num_logical_qubits
    num_logical_qubits = int(num_logical_qubits)
    if num_logical_qubits < 1 or num_logical_qubits > num_masks:
        raise ValueError("num_logical_qubits must be in [1, num_masks]")
    logical_sector_count = 1 << num_logical_qubits
    logical_sector_bit_weights = (
        1 << np.arange(num_logical_qubits, dtype=np.int64)
    )
    logical_observable_sum_per_temperature = np.zeros(
        (num_temperatures, num_masks), dtype=np.int64,
    )
    if observable_temperature_mode == "all":
        observable_temperature_indices = np.arange(
            num_temperatures,
            dtype=np.int64,
        )
    else:
        observable_temperature_indices = np.array([0], dtype=np.int64)
    logical_observable_count_per_temperature = np.zeros(
        num_temperatures,
        dtype=np.int64,
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
    if track_logical_sector_diagnostics:
        sector_previous_signature_per_temperature = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_flip_count_per_temperature = np.zeros(
            num_temperatures,
            dtype=np.int64,
        )
        sector_first_change_index_per_temperature = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_histogram_per_temperature = np.zeros(
            (num_temperatures, logical_sector_count),
            dtype=np.int64,
        )
        sector_last_hot_signature_by_replica = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_last_delivered_hot_signature_by_replica = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_last_cold_signature_by_replica = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_last_hot_measurement_by_replica = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_last_cold_measurement_by_replica = np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        )
        sector_diagnostic_sample_count = 0
        hot_to_cold_sector_delivery_count = 0
        hot_to_cold_sector_change_delivery_count = 0
    else:
        sector_previous_signature_per_temperature = None
        sector_flip_count_per_temperature = None
        sector_first_change_index_per_temperature = None
        sector_histogram_per_temperature = None
        sector_last_hot_signature_by_replica = None
        sector_last_delivered_hot_signature_by_replica = None
        sector_last_cold_signature_by_replica = None
        sector_last_hot_measurement_by_replica = None
        sector_last_cold_measurement_by_replica = None
        sector_diagnostic_sample_count = 0
        hot_to_cold_sector_delivery_count = 0
        hot_to_cold_sector_change_delivery_count = 0

    def _compute_logical_values_without_accumulating(temperature_index):
        return _compute_logical_observable_values(
            current_chain_bits=chain_bits_list[temperature_index],
            logical_observable_masks=logical_observable_masks,
            section_data=section_data,
            disorder_data_error_bits=disorder_data_error_bits,
            disorder_syndrome_representative_bits=(
                disorder_syndrome_representative_bits
            ),
            current_syndrome_term_bits=(
                syndrome_term_bits_list[temperature_index]
            ),
            observed_syndrome_bits=observed_syndrome_bits,
        )

    def _signature_from_logical_values(logical_observable_values):
        primitive_values = np.asarray(
            logical_observable_values[:num_logical_qubits],
            dtype=np.int8,
        )
        parity_bits = (primitive_values < 0).astype(np.int64)
        return int(parity_bits @ logical_sector_bit_weights)

    for measurement_index in range(num_measurements):
        for _ in range(num_sweeps_between_measurements):
            if (
                    cluster_controller is not None
                    and cluster_controller["enabled"]
                    and num_burn_in_sweeps == 0):
                cluster_controller["production_used_adaptive"] = True
            _run_one_sweep_for_all_temperatures()
            sweep_counter += 1
            _maybe_attempt_swap(sweep_counter)
        observable_started_at = time.perf_counter()
        measured_logical_values = {}
        for temperature_index in observable_temperature_indices:
            temperature_index = int(temperature_index)
            logical_observable_values = _accumulate_logical_observables_fast(
                current_chain_bits=chain_bits_list[temperature_index],
                logical_observable_masks=logical_observable_masks,
                logical_observable_sum_values=(
                    logical_observable_sum_per_temperature[temperature_index]
                ),
                section_data=section_data,
                disorder_data_error_bits=disorder_data_error_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                current_syndrome_term_bits=(
                    syndrome_term_bits_list[temperature_index]
                ),
                observed_syndrome_bits=observed_syndrome_bits,
            )
            logical_observable_count_per_temperature[temperature_index] += 1
            measured_logical_values[temperature_index] = logical_observable_values
        if diagnostic_config["record_measurement_trajectories"]:
            for diagnostic_slot, temperature_index in enumerate(
                    diagnostic_temperature_indices):
                temperature_index = int(temperature_index)
                if temperature_index in measured_logical_values:
                    logical_observable_values = measured_logical_values[
                        temperature_index
                    ]
                else:
                    logical_observable_values = _accumulate_logical_observables_fast(
                        current_chain_bits=chain_bits_list[temperature_index],
                        logical_observable_masks=logical_observable_masks,
                        logical_observable_sum_values=(
                            logical_observable_sum_per_temperature[
                                temperature_index
                            ]
                        ),
                        section_data=section_data,
                        disorder_data_error_bits=disorder_data_error_bits,
                        disorder_syndrome_representative_bits=(
                            disorder_syndrome_representative_bits
                        ),
                        current_syndrome_term_bits=(
                            syndrome_term_bits_list[temperature_index]
                        ),
                        observed_syndrome_bits=observed_syndrome_bits,
                    )
                    logical_observable_count_per_temperature[
                        temperature_index
                    ] += 1
                logical_observable_values_per_measurement[
                    diagnostic_slot,
                    measurement_index,
                ] = logical_observable_values
        if (
                track_logical_sector_diagnostics
                and measurement_index % logical_sector_diagnostic_stride == 0):
            sector_diagnostic_sample_count += 1
            sector_signatures = np.empty(num_temperatures, dtype=np.int64)
            for temperature_index in range(num_temperatures):
                if temperature_index in measured_logical_values:
                    logical_observable_values = measured_logical_values[
                        temperature_index
                    ]
                else:
                    logical_observable_values = (
                        _compute_logical_values_without_accumulating(
                            temperature_index
                        )
                    )
                signature = _signature_from_logical_values(
                    logical_observable_values
                )
                sector_signatures[temperature_index] = signature
                sector_histogram_per_temperature[
                    temperature_index,
                    signature,
                ] += 1
                previous_signature = int(
                    sector_previous_signature_per_temperature[
                        temperature_index
                    ]
                )
                if previous_signature >= 0 and signature != previous_signature:
                    sector_flip_count_per_temperature[temperature_index] += 1
                    if (
                            sector_first_change_index_per_temperature[
                                temperature_index
                            ]
                            < 0):
                        sector_first_change_index_per_temperature[
                            temperature_index
                        ] = measurement_index
                sector_previous_signature_per_temperature[
                    temperature_index
                ] = signature
            if num_temperatures > 1:
                hot_replica_id = int(replica_id_per_temperature[-1])
                cold_replica_id = int(replica_id_per_temperature[0])
                hot_signature = int(sector_signatures[-1])
                cold_signature = int(sector_signatures[0])
                previous_cold_signature = int(
                    sector_last_cold_signature_by_replica[cold_replica_id]
                )
                previous_cold_measurement = int(
                    sector_last_cold_measurement_by_replica[cold_replica_id]
                )
                sector_last_hot_signature_by_replica[
                    hot_replica_id
                ] = hot_signature
                sector_last_hot_measurement_by_replica[
                    hot_replica_id
                ] = measurement_index
                delivered_signature = int(
                    sector_last_hot_signature_by_replica[cold_replica_id]
                )
                delivered_after_hot_visit = (
                    int(sector_last_hot_measurement_by_replica[cold_replica_id])
                    > previous_cold_measurement
                )
                if (
                        delivered_signature >= 0
                        and delivered_signature == cold_signature
                        and delivered_signature
                        != int(
                            sector_last_delivered_hot_signature_by_replica[
                                cold_replica_id
                            ]
                        )):
                    hot_to_cold_sector_delivery_count += 1
                    if (
                            previous_cold_signature >= 0
                            and cold_signature != previous_cold_signature
                            and delivered_after_hot_visit):
                        hot_to_cold_sector_change_delivery_count += 1
                    sector_last_delivered_hot_signature_by_replica[
                        cold_replica_id
                    ] = delivered_signature
                sector_last_cold_signature_by_replica[
                    cold_replica_id
                ] = cold_signature
                sector_last_cold_measurement_by_replica[
                    cold_replica_id
                ] = measurement_index
        observable_wall_time += time.perf_counter() - observable_started_at

    m_u_values_per_temperature = np.full(
        (num_temperatures, num_masks),
        np.nan,
        dtype=np.float64,
    )
    for temperature_index in range(num_temperatures):
        count = int(logical_observable_count_per_temperature[temperature_index])
        if count > 0:
            m_u_values_per_temperature[temperature_index] = (
                logical_observable_sum_per_temperature[temperature_index].astype(
                    np.float64
                )
                / float(count)
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
    measurement_wall_time = time.perf_counter() - measurement_started_at

    result = {
        "data_error_probability_ladder": data_error_probability_ladder,
        "syndrome_error_probability_ladder": syndrome_error_probability_ladder,
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
        "pt_transport_position_sample_count": np.int64(
            transport_position_sample_count
        ),
        "pt_replica_cold_visit_count": replica_cold_visit_count,
        "pt_replica_hot_visit_count": replica_hot_visit_count,
        "pt_replica_cold_to_hot_passage_count": (
            replica_cold_to_hot_passage_count
        ),
        "pt_replica_hot_to_cold_passage_count": (
            replica_hot_to_cold_passage_count
        ),
        "pt_replica_endpoint_round_trip_count": np.minimum(
            replica_cold_to_hot_passage_count,
            replica_hot_to_cold_passage_count,
        ),
        "pt_replica_min_temperature_visited": replica_min_temperature_visited,
        "pt_replica_max_temperature_visited": replica_max_temperature_visited,
        "ordinary_update_wall_time": np.float64(ordinary_update_wall_time),
        "pt_swap_wall_time": np.float64(pt_swap_wall_time),
        "observable_wall_time": np.float64(observable_wall_time),
        "measurement_wall_time": np.float64(measurement_wall_time),
    }
    adaptive_flow_summary = summarize_adaptive_pt_flow(flow_tracker)
    if adaptive_flow_summary is not None:
        result["adaptive_pt_flow"] = adaptive_flow_summary
    if track_logical_sector_diagnostics:
        result["pt_sector_diagnostics_enabled"] = np.bool_(True)
        result["pt_sector_flip_count_per_temperature"] = (
            sector_flip_count_per_temperature
        )
        result["pt_first_sector_change_index_per_temperature"] = (
            sector_first_change_index_per_temperature
        )
        result["pt_sector_histogram_per_temperature"] = (
            sector_histogram_per_temperature
        )
        result["pt_sector_diagnostic_stride"] = np.int64(
            logical_sector_diagnostic_stride
        )
        result["pt_sector_diagnostic_sample_count"] = np.int64(
            sector_diagnostic_sample_count
        )
        result["pt_hot_to_cold_sector_delivery_count"] = np.int64(
            hot_to_cold_sector_delivery_count
        )
        result["pt_hot_to_cold_sector_change_delivery_count"] = np.int64(
            hot_to_cold_sector_change_delivery_count
        )
    else:
        result["pt_sector_diagnostics_enabled"] = np.bool_(False)
    if diagnostic_config["record_measurement_trajectories"]:
        result["logical_observable_values_per_measurement_per_temperature"] = (
            logical_observable_values_per_measurement
        )
        result["diagnostic_temperature_indices"] = diagnostic_temperature_indices
    if cluster_controller is None:
        result.update(make_disabled_cluster_summary(
            num_temperatures=num_temperatures,
            requested_enabled=cluster_update_enabled,
            budget_fraction_rho=cluster_budget_fraction_rho,
        ))
    else:
        result.update(summarize_cluster_controller(cluster_controller))
    return result
