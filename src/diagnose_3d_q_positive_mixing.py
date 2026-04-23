import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from build_toric_code_examples import (
    build_3d_toric_code,
    build_3d_toric_zero_syndrome_move_data,
)
from exact_enumeration import (
    compute_exact_logical_observable_means,
    compute_exact_logical_sector_weights,
)
from linear_section import build_linear_section
from main import (
    _build_q0_initial_chain_bits_per_start,
    _run_single_disorder_measurement,
)
from mcmc import draw_disorder_sample_from_uniform_values
from mcmc_parallel_tempering import run_parallel_tempering_measurement
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "q_positive_mixing_diagnostics"
)
DEFAULT_DISORDER_SEEDS = (
    2026042201,
    2026042202,
    2026042203,
    2026042204,
)
DEFAULT_BASE_NUM_BURN_IN_SWEEPS = 1200
DEFAULT_BURN_IN_MULTIPLIER = 4
DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS = 18
DEFAULT_NUM_MEASUREMENTS = 4096
DEFAULT_NUM_SWEEPS_BETWEEN_MEASUREMENTS = 6
DEFAULT_NUM_REPLICAS_PER_START = 2
DEFAULT_WINDOW_SIZE = 128
DEFAULT_Q0_NUM_START_CHAINS = 8
DEFAULT_SUITE = "first_batch"
DEFAULT_Q_VALUES_B1 = (0.0, 1e-6, 1e-4, 1e-3, 5e-3)
DEFAULT_Q_VALUES_D1 = DEFAULT_Q_VALUES_B1
DEFAULT_C1_CONFIGS = (
    {"label": "H0", "num_zero_syndrome_sweeps_per_cycle": 1, "winding_repeat_factor": 1},
    {"label": "H1", "num_zero_syndrome_sweeps_per_cycle": 4, "winding_repeat_factor": 1},
    {"label": "H2", "num_zero_syndrome_sweeps_per_cycle": 8, "winding_repeat_factor": 4},
)
DEFAULT_C2_P_HOT = 0.44
DEFAULT_C2_NUM_TEMPERATURES = 9


def _timestamp_tag():
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(output_path, data):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )


def _parse_float_csv(csv_value):
    return [
        float(token.strip())
        for token in csv_value.split(",")
        if token.strip()
    ]


def _parse_int_csv(csv_value):
    return [
        int(token.strip())
        for token in csv_value.split(",")
        if token.strip()
    ]


def _effective_num_burn_in_sweeps(
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        num_qubits,
        burn_in_scaling_reference_num_qubits):
    return int(np.ceil(
        base_num_burn_in_sweeps
        * burn_in_multiplier
        * (num_qubits / burn_in_scaling_reference_num_qubits)
    ))


def _build_context(lattice_size):
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
    return {
        "parity_check_matrix": parity_check_matrix,
        "dual_logical_z_basis": dual_logical_z_basis,
        "linear_section_data": linear_section_data,
        "logical_observable_masks": logical_observable_masks,
        "checks_touching_each_qubit": checks_touching_each_qubit,
        "zero_syndrome_move_data": zero_syndrome_move_data,
        "num_qubits": int(parity_check_matrix.shape[1]),
        "num_checks": int(parity_check_matrix.shape[0]),
        "num_masks": int(logical_observable_masks.shape[0]),
        "num_logical_qubits": int(dual_logical_z_basis.shape[0]),
    }


def _draw_uniform_disorder(context, disorder_seed):
    disorder_rng = np.random.default_rng(disorder_seed)
    syndrome_uniform_values = disorder_rng.random(context["num_checks"])
    data_uniform_values = disorder_rng.random(context["num_qubits"])
    return syndrome_uniform_values, data_uniform_values


def _draw_disorder_sample_for_q_p(
        syndrome_uniform_values,
        data_uniform_values,
        syndrome_error_probability,
        data_error_probability):
    return draw_disorder_sample_from_uniform_values(
        syndrome_uniform_values=syndrome_uniform_values,
        data_uniform_values=data_uniform_values,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
    )


def _build_start_states(
        context,
        observed_syndrome_bits,
        q0_num_start_chains=DEFAULT_Q0_NUM_START_CHAINS):
    initial_chain_bits_per_start, start_sector_labels = (
        _build_q0_initial_chain_bits_per_start(
            observed_syndrome_bits=observed_syndrome_bits,
            linear_section_data=context["linear_section_data"],
            zero_syndrome_move_data=context["zero_syndrome_move_data"],
            q0_num_start_chains=q0_num_start_chains,
        )
    )
    return initial_chain_bits_per_start, start_sector_labels


def _running_window_mean(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    num_measurements = values.shape[0]
    if window_size < 1 or window_size > num_measurements:
        raise ValueError("window_size must be in [1, num_measurements]")
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    cumulative = np.concatenate(
        (np.zeros((1, values.shape[1]), dtype=np.float64), cumulative),
        axis=0,
    )
    window_sums = cumulative[window_size:] - cumulative[:-window_size]
    return window_sums / window_size


def _compute_windowed_observables(
        logical_observable_values_per_measurement,
        window_size):
    effective_window_size = min(
        int(window_size),
        int(logical_observable_values_per_measurement.shape[0]),
    )
    windowed_m_u = _running_window_mean(
        logical_observable_values_per_measurement,
        window_size=effective_window_size,
    )
    windowed_q_top = np.mean(windowed_m_u ** 2, axis=1)
    return windowed_m_u, windowed_q_top


def _autocovariance_fft(values):
    values = np.asarray(values, dtype=np.float64)
    num_values = values.shape[0]
    centered_values = values - np.mean(values)
    variance = float(np.var(centered_values))
    if num_values < 2 or variance == 0.0:
        return np.zeros(num_values, dtype=np.float64)
    fft_length = 1 << (2 * num_values - 1).bit_length()
    fft_values = np.fft.rfft(centered_values, n=fft_length)
    autocovariance = np.fft.irfft(
        fft_values * np.conj(fft_values),
        n=fft_length,
    )[:num_values]
    normalization = np.arange(num_values, 0, -1, dtype=np.float64)
    return autocovariance / normalization


def _integrated_autocorrelation_time(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return 1.0
    autocovariance = _autocovariance_fft(values)
    if autocovariance[0] <= 0.0:
        return 1.0
    autocorrelation = autocovariance / autocovariance[0]
    tau_int = 1.0
    for lag_index in range(1, values.shape[0] - 1, 2):
        paired_sum = (
            autocorrelation[lag_index]
            + autocorrelation[lag_index + 1]
        )
        if paired_sum <= 0.0:
            break
        tau_int += 2.0 * paired_sum
    return float(max(tau_int, 1.0))


def _split_r_hat(chains):
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError("chains must have shape (num_chains, num_samples)")
    num_chains, num_samples = chains.shape
    if num_chains < 2 or num_samples < 4:
        return np.nan
    half_length = num_samples // 2
    split_chains = np.concatenate(
        (
            chains[:, :half_length],
            chains[:, num_samples - half_length:],
        ),
        axis=0,
    )
    num_split_chains = split_chains.shape[0]
    chain_means = np.mean(split_chains, axis=1)
    chain_vars = np.var(split_chains, axis=1, ddof=1)
    within_chain = float(np.mean(chain_vars))
    between_chain = float(
        half_length * np.var(chain_means, ddof=1)
    )
    if within_chain == 0.0:
        return 1.0
    variance_hat = (
        (half_length - 1) / half_length * within_chain
        + between_chain / half_length
    )
    return float(np.sqrt(variance_hat / within_chain))


def _signature_indices_from_logical_observable_values(
        logical_observable_values_per_measurement,
        num_logical_qubits):
    parity_bits = (
        logical_observable_values_per_measurement[:, :num_logical_qubits] < 0
    ).astype(np.int64)
    bit_weights = 1 << np.arange(num_logical_qubits, dtype=np.int64)
    return parity_bits @ bit_weights


def _signature_histogram(signature_indices, num_signatures):
    histogram = np.bincount(
        np.asarray(signature_indices, dtype=np.int64),
        minlength=num_signatures,
    )
    if histogram.shape[0] != num_signatures:
        histogram = histogram[:num_signatures]
    return histogram.astype(np.int64, copy=False)


def _analyze_chain_diagnostics(
        logical_observable_values_per_measurement,
        window_size,
        num_logical_qubits):
    num_measurements = logical_observable_values_per_measurement.shape[0]
    windowed_m_u, windowed_q_top = _compute_windowed_observables(
        logical_observable_values_per_measurement,
        window_size=window_size,
    )
    tau_int_per_mask = np.array(
        [
            _integrated_autocorrelation_time(
                logical_observable_values_per_measurement[:, mask_index]
            )
            for mask_index in range(
                logical_observable_values_per_measurement.shape[1]
            )
        ],
        dtype=np.float64,
    )
    max_tau_int = float(np.max(tau_int_per_mask))
    effective_sample_size = float(num_measurements / max_tau_int)
    signature_indices = _signature_indices_from_logical_observable_values(
        logical_observable_values_per_measurement,
        num_logical_qubits=num_logical_qubits,
    )
    signature_histogram = _signature_histogram(
        signature_indices=signature_indices,
        num_signatures=1 << num_logical_qubits,
    )
    signature_probabilities = (
        signature_histogram / np.sum(signature_histogram)
    ).astype(np.float64, copy=False)
    first_signature_change_index = -1
    initial_signature_index = int(signature_indices[0])
    changed_signature_positions = np.flatnonzero(
        signature_indices != initial_signature_index
    )
    if changed_signature_positions.size > 0:
        first_signature_change_index = int(changed_signature_positions[0])
    primitive_flip_counts = np.zeros(num_logical_qubits, dtype=np.int64)
    for logical_qubit_index in range(num_logical_qubits):
        primitive_flip_counts[logical_qubit_index] = int(np.count_nonzero(
            np.diff(
                logical_observable_values_per_measurement[
                    :min(512, num_measurements),
                    logical_qubit_index,
                ]
            )
        ))
    return {
        "windowed_m_u": windowed_m_u,
        "windowed_q_top": windowed_q_top,
        "tau_int_per_mask": tau_int_per_mask,
        "max_tau_int": max_tau_int,
        "effective_sample_size": effective_sample_size,
        "signature_indices": signature_indices,
        "signature_histogram": signature_histogram,
        "signature_probabilities": signature_probabilities,
        "first_signature_change_index": np.int64(first_signature_change_index),
        "primitive_flip_counts_first_512": primitive_flip_counts,
    }


def _run_single_chain(
        context,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        chain_seed,
        initial_chain_bits,
        num_zero_syndrome_sweeps_per_cycle,
        winding_repeat_factor,
        window_size):
    measurement_result = _run_single_disorder_measurement(
        parity_check_matrix=context["parity_check_matrix"],
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        logical_observable_masks=context["logical_observable_masks"],
        checks_touching_each_qubit=context["checks_touching_each_qubit"],
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements_per_disorder=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        rng=np.random.default_rng(chain_seed),
        zero_syndrome_move_data=context["zero_syndrome_move_data"],
        initial_chain_bits=initial_chain_bits,
        num_zero_syndrome_sweeps_per_cycle=(
            num_zero_syndrome_sweeps_per_cycle
        ),
        winding_repeat_factor=winding_repeat_factor,
        return_diagnostics=True,
    )
    chain_analysis = _analyze_chain_diagnostics(
        logical_observable_values_per_measurement=measurement_result[
            "logical_observable_values_per_measurement"
        ],
        window_size=window_size,
        num_logical_qubits=context["num_logical_qubits"],
    )
    return {
        "measurement_result": measurement_result,
        "chain_analysis": chain_analysis,
    }


def _run_pt_single_chain(
        context,
        observed_syndrome_bits,
        disorder_data_error_bits,
        syndrome_error_probability,
        data_error_probability_ladder,
        num_burn_in_sweeps,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        chain_seed,
        initial_chain_bits_cold,
        num_zero_syndrome_sweeps_per_cycle,
        winding_repeat_factor,
        window_size):
    """
    运行一次 PT，返回冷链（ladder[0]）的 measurement_result / chain_analysis，
    shape 与 _run_single_chain 完全兼容，便于复用 _summarize_multi_chain_batch。

    冷链从 `initial_chain_bits_cold` 起；其它温度沿用相同初态（syndrome 一致即可）。
    """
    num_temperatures = int(len(data_error_probability_ladder))
    if initial_chain_bits_cold is None:
        initial_chain_bits_per_temperature = None
    else:
        initial_chain_bits_per_temperature = np.broadcast_to(
            np.asarray(initial_chain_bits_cold, dtype=bool),
            (num_temperatures, initial_chain_bits_cold.shape[0]),
        ).copy()

    pt_result = run_parallel_tempering_measurement(
        parity_check_matrix=context["parity_check_matrix"],
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability_ladder=data_error_probability_ladder,
        logical_observable_masks=context["logical_observable_masks"],
        checks_touching_each_qubit=context["checks_touching_each_qubit"],
        num_burn_in_sweeps=num_burn_in_sweeps,
        num_measurements=num_measurements_per_disorder,
        num_sweeps_between_measurements=num_sweeps_between_measurements,
        rng=np.random.default_rng(chain_seed),
        zero_syndrome_move_data=context["zero_syndrome_move_data"],
        initial_chain_bits_per_temperature=initial_chain_bits_per_temperature,
        num_zero_syndrome_sweeps_per_cycle=(
            num_zero_syndrome_sweeps_per_cycle
        ),
        winding_repeat_factor=winding_repeat_factor,
        return_diagnostics=True,
    )

    cold_logical_trace = pt_result[
        "logical_observable_values_per_measurement_per_temperature"
    ][0]
    measurement_result = {
        "m_u_values": pt_result["m_u_values_per_temperature"][0],
        "q_top_value": float(pt_result["q_top_value_per_temperature"][0]),
        "single_bit_acceptance_rate": float(
            pt_result["single_bit_acceptance_rate_per_temperature"][0]
        ),
        "contractible_acceptance_rate": float(
            pt_result["contractible_acceptance_rate_per_temperature"][0]
        ),
        "winding_acceptance_rate": float(
            pt_result["winding_acceptance_rate_per_temperature"][0]
        ),
        "winding_attempted_count": int(
            pt_result["winding_attempted_count_per_temperature"][0]
        ),
        "logical_observable_values_per_measurement": cold_logical_trace,
        "pt_ladder": pt_result["data_error_probability_ladder"],
        "pt_q_top_value_per_temperature": (
            pt_result["q_top_value_per_temperature"]
        ),
        "pt_winding_acceptance_rate_per_temperature": (
            pt_result["winding_acceptance_rate_per_temperature"]
        ),
        "pt_swap_acceptance_rates": pt_result["swap_acceptance_rates"],
    }
    chain_analysis = _analyze_chain_diagnostics(
        logical_observable_values_per_measurement=cold_logical_trace,
        window_size=window_size,
        num_logical_qubits=context["num_logical_qubits"],
    )
    return {
        "measurement_result": measurement_result,
        "chain_analysis": chain_analysis,
    }


def _equal_log_odds_ladder(p_cold, p_hot, num_temperatures):
    L_cold = np.log(p_cold / (1.0 - p_cold))
    L_hot = np.log(p_hot / (1.0 - p_hot))
    log_odds = np.linspace(L_cold, L_hot, num_temperatures)
    return np.exp(log_odds) / (1.0 + np.exp(log_odds))


def _aggregate_r_hat(logical_observable_values_tensor):
    num_masks = logical_observable_values_tensor.shape[-1]
    r_hat_per_mask = np.empty(num_masks, dtype=np.float64)
    for mask_index in range(num_masks):
        r_hat_per_mask[mask_index] = _split_r_hat(
            logical_observable_values_tensor[:, :, mask_index]
        )
    return r_hat_per_mask


def _summarize_multi_chain_batch(
        measurement_results,
        chain_analyses,
        start_sector_labels,
        context,
        config_label,
        p_value,
        q_value):
    num_disorders = measurement_results.shape[0]
    num_starts = measurement_results.shape[1]
    num_replicas = measurement_results.shape[2]
    num_masks = context["num_masks"]
    num_signatures = 1 << context["num_logical_qubits"]
    num_measurements = (
        measurement_results[0, 0, 0]["logical_observable_values_per_measurement"]
        .shape[0]
    )
    final_m_u_values = np.empty(
        (num_disorders, num_starts, num_replicas, num_masks),
        dtype=np.float64,
    )
    final_q_top_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    max_tau_int_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    effective_sample_size_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    signature_probabilities = np.empty(
        (
            num_disorders,
            num_starts,
            num_replicas,
            num_signatures,
        ),
        dtype=np.float64,
    )
    signature_histograms = np.empty(
        (
            num_disorders,
            num_starts,
            num_replicas,
            num_signatures,
        ),
        dtype=np.int64,
    )
    first_signature_change_index_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.int64,
    )
    contractible_acceptance_rate_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    winding_acceptance_rate_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    single_bit_acceptance_rate_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.float64,
    )
    winding_attempted_count_values = np.empty(
        (num_disorders, num_starts, num_replicas),
        dtype=np.int64,
    )
    logical_observable_values_tensor = np.empty(
        (
            num_disorders,
            num_starts,
            num_replicas,
            num_measurements,
            num_masks,
        ),
        dtype=np.int8,
    )

    for disorder_index in range(num_disorders):
        for start_index in range(num_starts):
            for replica_index in range(num_replicas):
                measurement_result = measurement_results[
                    disorder_index,
                    start_index,
                    replica_index,
                ]
                chain_analysis = chain_analyses[
                    disorder_index,
                    start_index,
                    replica_index,
                ]
                final_m_u_values[disorder_index, start_index, replica_index] = (
                    measurement_result["m_u_values"]
                )
                final_q_top_values[disorder_index, start_index, replica_index] = (
                    measurement_result["q_top_value"]
                )
                max_tau_int_values[disorder_index, start_index, replica_index] = (
                    chain_analysis["max_tau_int"]
                )
                effective_sample_size_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = chain_analysis["effective_sample_size"]
                signature_probabilities[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = chain_analysis["signature_probabilities"]
                signature_histograms[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = chain_analysis["signature_histogram"]
                first_signature_change_index_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = int(chain_analysis["first_signature_change_index"])
                contractible_acceptance_rate_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = float(
                    measurement_result.get(
                        "contractible_acceptance_rate",
                        float("nan"),
                    )
                )
                winding_acceptance_rate_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = float(
                    measurement_result.get(
                        "winding_acceptance_rate",
                        float("nan"),
                    )
                )
                single_bit_acceptance_rate_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = float(
                    measurement_result.get(
                        "single_bit_acceptance_rate",
                        float("nan"),
                    )
                )
                winding_attempted_count_values[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = int(
                    measurement_result.get(
                        "winding_attempted_count",
                        0,
                    )
                )
                logical_observable_values_tensor[
                    disorder_index,
                    start_index,
                    replica_index,
                ] = measurement_result[
                    "logical_observable_values_per_measurement"
                ]

    q_top_spread_per_disorder = (
        np.max(final_q_top_values, axis=(1, 2))
        - np.min(final_q_top_values, axis=(1, 2))
    )
    m_u_spread_linf_per_disorder = np.empty(num_disorders, dtype=np.float64)
    max_r_hat_per_disorder = np.empty(num_disorders, dtype=np.float64)
    start_signature_l1_spread_per_disorder = np.empty(
        num_disorders,
        dtype=np.float64,
    )

    for disorder_index in range(num_disorders):
        pairwise_m_u_diff = np.abs(
            final_m_u_values[disorder_index][:, None, :, None, :]
            - final_m_u_values[disorder_index][None, :, None, :, :]
        )
        m_u_spread_linf_per_disorder[disorder_index] = float(
            np.max(pairwise_m_u_diff)
        )
        flattened_chains = logical_observable_values_tensor[
            disorder_index
        ].reshape(
            num_starts * num_replicas,
            num_measurements,
            num_masks,
        )
        r_hat_per_mask = _aggregate_r_hat(flattened_chains)
        max_r_hat_per_disorder[disorder_index] = float(
            np.nanmax(r_hat_per_mask)
        )
        mean_signature_probabilities_by_start = np.mean(
            signature_probabilities[disorder_index],
            axis=1,
        )
        start_signature_l1_spread_per_disorder[disorder_index] = float(
            np.max(np.sum(
                np.abs(
                    mean_signature_probabilities_by_start[:, None, :]
                    - mean_signature_probabilities_by_start[None, :, :]
                ),
                axis=2,
            ))
        )

    num_chains_that_never_flipped_sector = int(
        np.count_nonzero(first_signature_change_index_values == -1)
    )
    total_num_chains = int(first_signature_change_index_values.size)
    winding_acceptance_max = (
        float(np.nanmax(winding_acceptance_rate_values))
        if winding_acceptance_rate_values.size > 0
        else 0.0
    )
    winding_acceptance_mean = (
        float(np.nanmean(winding_acceptance_rate_values))
        if winding_acceptance_rate_values.size > 0
        else 0.0
    )
    contractible_acceptance_mean = (
        float(np.nanmean(contractible_acceptance_rate_values))
        if contractible_acceptance_rate_values.size > 0
        else 0.0
    )

    return {
        "config_label": config_label,
        "p_value": float(p_value),
        "q_value": float(q_value),
        "start_sector_labels": start_sector_labels,
        "mean_q_top": float(np.mean(final_q_top_values)),
        "q_top_sem": float(
            np.std(final_q_top_values.reshape(-1), ddof=1)
            / np.sqrt(final_q_top_values.size)
        ),
        "mean_q_top_spread_per_disorder": float(
            np.mean(q_top_spread_per_disorder)
        ),
        "mean_m_u_spread_linf_per_disorder": float(
            np.mean(m_u_spread_linf_per_disorder)
        ),
        "max_r_hat_across_disorders": float(np.max(max_r_hat_per_disorder)),
        "mean_max_tau_int": float(np.mean(max_tau_int_values)),
        "min_effective_sample_size": float(np.min(effective_sample_size_values)),
        "mean_signature_l1_spread_per_disorder": float(
            np.mean(start_signature_l1_spread_per_disorder)
        ),
        "num_chains_that_never_flipped_sector": (
            num_chains_that_never_flipped_sector
        ),
        "total_num_chains": total_num_chains,
        "winding_acceptance_rate_mean": winding_acceptance_mean,
        "winding_acceptance_rate_max": winding_acceptance_max,
        "contractible_acceptance_rate_mean": contractible_acceptance_mean,
        "final_q_top_per_chain": final_q_top_values,
        "final_m_u_per_chain": final_m_u_values,
        "signature_histograms_per_chain": signature_histograms,
        "first_signature_change_index_per_chain": (
            first_signature_change_index_values
        ),
        "contractible_acceptance_rate_per_chain": (
            contractible_acceptance_rate_values
        ),
        "winding_acceptance_rate_per_chain": (
            winding_acceptance_rate_values
        ),
        "single_bit_acceptance_rate_per_chain": (
            single_bit_acceptance_rate_values
        ),
        "winding_attempted_count_per_chain": (
            winding_attempted_count_values
        ),
        "q_top_spread_per_disorder": q_top_spread_per_disorder,
        "m_u_spread_linf_per_disorder": m_u_spread_linf_per_disorder,
        "max_r_hat_per_disorder": max_r_hat_per_disorder,
    }


def _build_chain_seed(
        disorder_seed,
        start_index,
        replica_index,
        config_index=0,
        q_index=0):
    return int(
        disorder_seed
        + 1009 * start_index
        + 10007 * replica_index
        + 100003 * config_index
        + 1000003 * q_index
    )


def _run_a1(
        run_root,
        lattice_sizes,
        p_values,
        q_value,
        disorder_seeds,
        num_replicas_per_start,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    summary_entries = []
    for lattice_size in lattice_sizes:
        context = _build_context(lattice_size)
        num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            base_num_burn_in_sweeps=base_num_burn_in_sweeps,
            burn_in_multiplier=burn_in_multiplier,
            num_qubits=context["num_qubits"],
            burn_in_scaling_reference_num_qubits=(
                burn_in_scaling_reference_num_qubits
            ),
        )
        for p_value in p_values:
            num_disorders = len(disorder_seeds)
            measurement_results = np.empty(
                (
                    num_disorders,
                    DEFAULT_Q0_NUM_START_CHAINS,
                    num_replicas_per_start,
                ),
                dtype=object,
            )
            chain_analyses = np.empty_like(measurement_results)
            start_sector_labels = None
            disorder_uniforms = []
            for disorder_index, disorder_seed in enumerate(disorder_seeds):
                disorder_uniforms.append(
                    _draw_uniform_disorder(context, disorder_seed)
                )
                (
                    syndrome_uniform_values,
                    data_uniform_values,
                ) = disorder_uniforms[-1]
                (
                    observed_syndrome_bits,
                    disorder_data_error_bits,
                ) = _draw_disorder_sample_for_q_p(
                    syndrome_uniform_values=syndrome_uniform_values,
                    data_uniform_values=data_uniform_values,
                    syndrome_error_probability=q_value,
                    data_error_probability=p_value,
                )
                initial_chain_bits_per_start, start_sector_labels = (
                    _build_start_states(context, observed_syndrome_bits)
                )
                for start_index in range(DEFAULT_Q0_NUM_START_CHAINS):
                    for replica_index in range(num_replicas_per_start):
                        chain_output = _run_single_chain(
                            context=context,
                            observed_syndrome_bits=observed_syndrome_bits,
                            disorder_data_error_bits=disorder_data_error_bits,
                            syndrome_error_probability=q_value,
                            data_error_probability=p_value,
                            num_burn_in_sweeps=num_burn_in_sweeps,
                            num_measurements_per_disorder=(
                                num_measurements_per_disorder
                            ),
                            num_sweeps_between_measurements=(
                                num_sweeps_between_measurements
                            ),
                            chain_seed=_build_chain_seed(
                                disorder_seed=disorder_seed,
                                start_index=start_index,
                                replica_index=replica_index,
                            ),
                            initial_chain_bits=(
                                initial_chain_bits_per_start[start_index]
                            ),
                            num_zero_syndrome_sweeps_per_cycle=1,
                            winding_repeat_factor=1,
                            window_size=window_size,
                        )
                        measurement_results[
                            disorder_index,
                            start_index,
                            replica_index,
                        ] = chain_output["measurement_result"]
                        chain_analyses[
                            disorder_index,
                            start_index,
                            replica_index,
                        ] = chain_output["chain_analysis"]
            summary_entries.append({
                "lattice_size": int(lattice_size),
                "num_burn_in_sweeps": int(num_burn_in_sweeps),
                **_summarize_multi_chain_batch(
                    measurement_results=measurement_results,
                    chain_analyses=chain_analyses,
                    start_sector_labels=start_sector_labels,
                    context=context,
                    config_label="A1-H0",
                    p_value=p_value,
                    q_value=q_value,
                ),
            })
            output_stem = f"a1_L{lattice_size:02d}_p{p_value:0.4f}".replace(".", "p")
            np.savez_compressed(
                run_root / f"{output_stem}.npz",
                measurement_results=measurement_results,
                chain_analyses=chain_analyses,
                disorder_seeds=np.asarray(disorder_seeds, dtype=np.int64),
                start_sector_labels=np.asarray(start_sector_labels),
                lattice_size=np.int64(lattice_size),
                p_value=np.float64(p_value),
                q_value=np.float64(q_value),
                num_burn_in_sweeps=np.int64(num_burn_in_sweeps),
                num_measurements_per_disorder=np.int64(
                    num_measurements_per_disorder
                ),
            )
    _write_json(run_root / "a1_summary.json", {"entries": summary_entries})
    return summary_entries


def _run_c2(
        run_root,
        lattice_sizes,
        p_cold_values,
        p_hot,
        num_temperatures,
        q_value,
        disorder_seeds,
        num_replicas_per_start,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    """
    C2 suite: 用 parallel tempering 在 p 上跑 multi-start per-disorder，
    复用 A1/A2 的诊断 summary。冷链的 per-(disorder, start, replica)
    breakdown 可直接和 A1 基线对照。
    """
    summary_entries = []
    for lattice_size in lattice_sizes:
        context = _build_context(lattice_size)
        num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            base_num_burn_in_sweeps=base_num_burn_in_sweeps,
            burn_in_multiplier=burn_in_multiplier,
            num_qubits=context["num_qubits"],
            burn_in_scaling_reference_num_qubits=(
                burn_in_scaling_reference_num_qubits
            ),
        )
        for p_cold in p_cold_values:
            ladder = _equal_log_odds_ladder(
                p_cold=float(p_cold),
                p_hot=float(p_hot),
                num_temperatures=int(num_temperatures),
            )
            num_disorders = len(disorder_seeds)
            measurement_results = np.empty(
                (
                    num_disorders,
                    DEFAULT_Q0_NUM_START_CHAINS,
                    num_replicas_per_start,
                ),
                dtype=object,
            )
            chain_analyses = np.empty_like(measurement_results)
            start_sector_labels = None
            disorder_uniforms = []
            for disorder_index, disorder_seed in enumerate(disorder_seeds):
                disorder_uniforms.append(
                    _draw_uniform_disorder(context, disorder_seed)
                )
                (
                    syndrome_uniform_values,
                    data_uniform_values,
                ) = disorder_uniforms[-1]
                (
                    observed_syndrome_bits,
                    disorder_data_error_bits,
                ) = _draw_disorder_sample_for_q_p(
                    syndrome_uniform_values=syndrome_uniform_values,
                    data_uniform_values=data_uniform_values,
                    syndrome_error_probability=q_value,
                    data_error_probability=float(p_cold),
                )
                initial_chain_bits_per_start, start_sector_labels = (
                    _build_start_states(context, observed_syndrome_bits)
                )
                for start_index in range(DEFAULT_Q0_NUM_START_CHAINS):
                    for replica_index in range(num_replicas_per_start):
                        chain_output = _run_pt_single_chain(
                            context=context,
                            observed_syndrome_bits=observed_syndrome_bits,
                            disorder_data_error_bits=(
                                disorder_data_error_bits
                            ),
                            syndrome_error_probability=q_value,
                            data_error_probability_ladder=ladder,
                            num_burn_in_sweeps=num_burn_in_sweeps,
                            num_measurements_per_disorder=(
                                num_measurements_per_disorder
                            ),
                            num_sweeps_between_measurements=(
                                num_sweeps_between_measurements
                            ),
                            chain_seed=_build_chain_seed(
                                disorder_seed=disorder_seed,
                                start_index=start_index,
                                replica_index=replica_index,
                            ),
                            initial_chain_bits_cold=(
                                initial_chain_bits_per_start[start_index]
                            ),
                            num_zero_syndrome_sweeps_per_cycle=1,
                            winding_repeat_factor=1,
                            window_size=window_size,
                        )
                        measurement_results[
                            disorder_index,
                            start_index,
                            replica_index,
                        ] = chain_output["measurement_result"]
                        chain_analyses[
                            disorder_index,
                            start_index,
                            replica_index,
                        ] = chain_output["chain_analysis"]
            config_label = (
                f"C2-pt-K{int(num_temperatures):02d}"
                f"-phot{float(p_hot):0.3f}".replace(".", "p")
            )
            pt_ladder_info = {
                "pt_ladder": ladder,
                "pt_p_hot": float(p_hot),
                "pt_num_temperatures": int(num_temperatures),
            }
            summary_entries.append({
                "lattice_size": int(lattice_size),
                "num_burn_in_sweeps": int(num_burn_in_sweeps),
                **pt_ladder_info,
                **_summarize_multi_chain_batch(
                    measurement_results=measurement_results,
                    chain_analyses=chain_analyses,
                    start_sector_labels=start_sector_labels,
                    context=context,
                    config_label=config_label,
                    p_value=float(p_cold),
                    q_value=q_value,
                ),
            })
            output_stem = (
                f"c2_L{lattice_size:02d}"
                f"_pcold{float(p_cold):0.4f}"
            ).replace(".", "p")
            np.savez_compressed(
                run_root / f"{output_stem}.npz",
                measurement_results=measurement_results,
                chain_analyses=chain_analyses,
                disorder_seeds=np.asarray(disorder_seeds, dtype=np.int64),
                start_sector_labels=np.asarray(start_sector_labels),
                pt_ladder=ladder,
                lattice_size=np.int64(lattice_size),
                p_cold=np.float64(p_cold),
                p_hot=np.float64(p_hot),
                q_value=np.float64(q_value),
                num_burn_in_sweeps=np.int64(num_burn_in_sweeps),
                num_measurements_per_disorder=np.int64(
                    num_measurements_per_disorder
                ),
            )
    _write_json(run_root / "c2_summary.json", {"entries": summary_entries})
    return summary_entries


def _run_a2(
        run_root,
        q_value,
        p_values,
        disorder_seeds,
        num_replicas_per_start,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    lattice_size = 5
    context = _build_context(lattice_size)
    num_burn_in_sweeps = _effective_num_burn_in_sweeps(
        base_num_burn_in_sweeps=base_num_burn_in_sweeps,
        burn_in_multiplier=burn_in_multiplier,
        num_qubits=context["num_qubits"],
        burn_in_scaling_reference_num_qubits=(
            burn_in_scaling_reference_num_qubits
        ),
    )
    summary_entries = []
    for p_value in p_values:
        per_disorder_records = []
        for disorder_seed in disorder_seeds:
            syndrome_uniform_values, data_uniform_values = _draw_uniform_disorder(
                context,
                disorder_seed,
            )
            observed_syndrome_bits, disorder_data_error_bits = (
                _draw_disorder_sample_for_q_p(
                    syndrome_uniform_values=syndrome_uniform_values,
                    data_uniform_values=data_uniform_values,
                    syndrome_error_probability=q_value,
                    data_error_probability=p_value,
                )
            )
            initial_chain_bits_per_start, start_sector_labels = (
                _build_start_states(context, observed_syndrome_bits)
            )
            start_weights = np.count_nonzero(
                initial_chain_bits_per_start,
                axis=1,
            )
            low_weight_start_index = int(np.argmin(start_weights))
            high_weight_start_index = int(np.argmax(start_weights))
            selected_start_indices = (
                low_weight_start_index,
                high_weight_start_index,
            )
            per_start_results = []
            for selected_order, start_index in enumerate(selected_start_indices):
                for replica_index in range(num_replicas_per_start):
                    chain_output = _run_single_chain(
                        context=context,
                        observed_syndrome_bits=observed_syndrome_bits,
                        disorder_data_error_bits=disorder_data_error_bits,
                        syndrome_error_probability=q_value,
                        data_error_probability=p_value,
                        num_burn_in_sweeps=num_burn_in_sweeps,
                        num_measurements_per_disorder=(
                            num_measurements_per_disorder
                        ),
                        num_sweeps_between_measurements=(
                            num_sweeps_between_measurements
                        ),
                        chain_seed=_build_chain_seed(
                            disorder_seed=disorder_seed,
                            start_index=start_index,
                            replica_index=replica_index,
                            config_index=selected_order,
                        ),
                        initial_chain_bits=initial_chain_bits_per_start[start_index],
                        num_zero_syndrome_sweeps_per_cycle=1,
                        winding_repeat_factor=1,
                        window_size=window_size,
                    )
                    per_start_results.append({
                        "start_index": int(start_index),
                        "start_label": str(start_sector_labels[start_index]),
                        "replica_index": int(replica_index),
                        "measurement_result": chain_output["measurement_result"],
                        "chain_analysis": chain_output["chain_analysis"],
                    })
            per_disorder_records.append({
                "disorder_seed": int(disorder_seed),
                "low_weight_start_index": low_weight_start_index,
                "high_weight_start_index": high_weight_start_index,
                "low_weight_start_label": str(
                    start_sector_labels[low_weight_start_index]
                ),
                "high_weight_start_label": str(
                    start_sector_labels[high_weight_start_index]
                ),
                "per_start_results": per_start_results,
            })
        summary_entries.append({
            "lattice_size": lattice_size,
            "p_value": float(p_value),
            "q_value": float(q_value),
            "num_burn_in_sweeps": int(num_burn_in_sweeps),
            "records": per_disorder_records,
        })
    _write_json(run_root / "a2_summary.json", {"entries": summary_entries})
    return summary_entries


def _run_c1(
        run_root,
        lattice_sizes,
        p_values,
        q_value,
        disorder_seeds,
        num_replicas_per_start,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    summary_entries = []
    for lattice_size in lattice_sizes:
        context = _build_context(lattice_size)
        num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            base_num_burn_in_sweeps=base_num_burn_in_sweeps,
            burn_in_multiplier=burn_in_multiplier,
            num_qubits=context["num_qubits"],
            burn_in_scaling_reference_num_qubits=(
                burn_in_scaling_reference_num_qubits
            ),
        )
        for p_value in p_values:
            config_summaries = []
            for config_index, config in enumerate(DEFAULT_C1_CONFIGS):
                measurement_results = np.empty(
                    (
                        len(disorder_seeds),
                        DEFAULT_Q0_NUM_START_CHAINS,
                        num_replicas_per_start,
                    ),
                    dtype=object,
                )
                chain_analyses = np.empty_like(measurement_results)
                start_sector_labels = None
                for disorder_index, disorder_seed in enumerate(disorder_seeds):
                    syndrome_uniform_values, data_uniform_values = (
                        _draw_uniform_disorder(context, disorder_seed)
                    )
                    observed_syndrome_bits, disorder_data_error_bits = (
                        _draw_disorder_sample_for_q_p(
                            syndrome_uniform_values=syndrome_uniform_values,
                            data_uniform_values=data_uniform_values,
                            syndrome_error_probability=q_value,
                            data_error_probability=p_value,
                        )
                    )
                    initial_chain_bits_per_start, start_sector_labels = (
                        _build_start_states(context, observed_syndrome_bits)
                    )
                    for start_index in range(DEFAULT_Q0_NUM_START_CHAINS):
                        for replica_index in range(num_replicas_per_start):
                            chain_output = _run_single_chain(
                                context=context,
                                observed_syndrome_bits=observed_syndrome_bits,
                                disorder_data_error_bits=(
                                    disorder_data_error_bits
                                ),
                                syndrome_error_probability=q_value,
                                data_error_probability=p_value,
                                num_burn_in_sweeps=num_burn_in_sweeps,
                                num_measurements_per_disorder=(
                                    num_measurements_per_disorder
                                ),
                                num_sweeps_between_measurements=(
                                    num_sweeps_between_measurements
                                ),
                                chain_seed=_build_chain_seed(
                                    disorder_seed=disorder_seed,
                                    start_index=start_index,
                                    replica_index=replica_index,
                                    config_index=config_index,
                                ),
                                initial_chain_bits=(
                                    initial_chain_bits_per_start[start_index]
                                ),
                                num_zero_syndrome_sweeps_per_cycle=(
                                    config["num_zero_syndrome_sweeps_per_cycle"]
                                ),
                                winding_repeat_factor=(
                                    config["winding_repeat_factor"]
                                ),
                                window_size=window_size,
                            )
                            measurement_results[
                                disorder_index,
                                start_index,
                                replica_index,
                            ] = chain_output["measurement_result"]
                            chain_analyses[
                                disorder_index,
                                start_index,
                                replica_index,
                            ] = chain_output["chain_analysis"]
                config_summaries.append(_summarize_multi_chain_batch(
                    measurement_results=measurement_results,
                    chain_analyses=chain_analyses,
                    start_sector_labels=start_sector_labels,
                    context=context,
                    config_label=config["label"],
                    p_value=p_value,
                    q_value=q_value,
                ))
            summary_entries.append({
                "lattice_size": int(lattice_size),
                "p_value": float(p_value),
                "q_value": float(q_value),
                "num_burn_in_sweeps": int(num_burn_in_sweeps),
                "configs": config_summaries,
            })
    _write_json(run_root / "c1_summary.json", {"entries": summary_entries})
    return summary_entries


def _run_b1(
        run_root,
        lattice_sizes,
        p_values,
        q_values,
        disorder_seeds,
        num_replicas_per_start,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    summary_entries = []
    for lattice_size in lattice_sizes:
        context = _build_context(lattice_size)
        num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            base_num_burn_in_sweeps=base_num_burn_in_sweeps,
            burn_in_multiplier=burn_in_multiplier,
            num_qubits=context["num_qubits"],
            burn_in_scaling_reference_num_qubits=(
                burn_in_scaling_reference_num_qubits
            ),
        )
        for p_value in p_values:
            disorder_uniforms = [
                _draw_uniform_disorder(context, disorder_seed)
                for disorder_seed in disorder_seeds
            ]
            q_summaries = []
            for q_index, q_value in enumerate(q_values):
                measurement_results = np.empty(
                    (
                        len(disorder_seeds),
                        DEFAULT_Q0_NUM_START_CHAINS,
                        num_replicas_per_start,
                    ),
                    dtype=object,
                )
                chain_analyses = np.empty_like(measurement_results)
                start_sector_labels = None
                for disorder_index, disorder_seed in enumerate(disorder_seeds):
                    syndrome_uniform_values, data_uniform_values = (
                        disorder_uniforms[disorder_index]
                    )
                    observed_syndrome_bits, disorder_data_error_bits = (
                        _draw_disorder_sample_for_q_p(
                            syndrome_uniform_values=syndrome_uniform_values,
                            data_uniform_values=data_uniform_values,
                            syndrome_error_probability=q_value,
                            data_error_probability=p_value,
                        )
                    )
                    initial_chain_bits_per_start, start_sector_labels = (
                        _build_start_states(context, observed_syndrome_bits)
                    )
                    for start_index in range(DEFAULT_Q0_NUM_START_CHAINS):
                        for replica_index in range(num_replicas_per_start):
                            chain_output = _run_single_chain(
                                context=context,
                                observed_syndrome_bits=observed_syndrome_bits,
                                disorder_data_error_bits=(
                                    disorder_data_error_bits
                                ),
                                syndrome_error_probability=q_value,
                                data_error_probability=p_value,
                                num_burn_in_sweeps=num_burn_in_sweeps,
                                num_measurements_per_disorder=(
                                    num_measurements_per_disorder
                                ),
                                num_sweeps_between_measurements=(
                                    num_sweeps_between_measurements
                                ),
                                chain_seed=_build_chain_seed(
                                    disorder_seed=disorder_seed,
                                    start_index=start_index,
                                    replica_index=replica_index,
                                    q_index=q_index,
                                ),
                                initial_chain_bits=(
                                    initial_chain_bits_per_start[start_index]
                                ),
                                num_zero_syndrome_sweeps_per_cycle=1,
                                winding_repeat_factor=1,
                                window_size=window_size,
                            )
                            measurement_results[
                                disorder_index,
                                start_index,
                                replica_index,
                            ] = chain_output["measurement_result"]
                            chain_analyses[
                                disorder_index,
                                start_index,
                                replica_index,
                            ] = chain_output["chain_analysis"]
                q_summaries.append(_summarize_multi_chain_batch(
                    measurement_results=measurement_results,
                    chain_analyses=chain_analyses,
                    start_sector_labels=start_sector_labels,
                    context=context,
                    config_label="B1-H0",
                    p_value=p_value,
                    q_value=q_value,
                ))
            summary_entries.append({
                "lattice_size": int(lattice_size),
                "p_value": float(p_value),
                "num_burn_in_sweeps": int(num_burn_in_sweeps),
                "q_summaries": q_summaries,
            })
    _write_json(run_root / "b1_summary.json", {"entries": summary_entries})
    return summary_entries


def _run_d1(
        run_root,
        p_values,
        q_values,
        disorder_seeds,
        base_num_burn_in_sweeps,
        burn_in_multiplier,
        burn_in_scaling_reference_num_qubits,
        num_measurements_per_disorder,
        num_sweeps_between_measurements,
        window_size):
    lattice_size = 2
    context = _build_context(lattice_size)
    num_burn_in_sweeps = _effective_num_burn_in_sweeps(
        base_num_burn_in_sweeps=base_num_burn_in_sweeps,
        burn_in_multiplier=burn_in_multiplier,
        num_qubits=context["num_qubits"],
        burn_in_scaling_reference_num_qubits=(
            burn_in_scaling_reference_num_qubits
        ),
    )
    summary_entries = []
    for p_value in p_values:
        for q_index, q_value in enumerate(q_values):
            per_disorder_results = []
            for disorder_seed in disorder_seeds:
                syndrome_uniform_values, data_uniform_values = _draw_uniform_disorder(
                    context,
                    disorder_seed,
                )
                observed_syndrome_bits, disorder_data_error_bits = (
                    _draw_disorder_sample_for_q_p(
                        syndrome_uniform_values=syndrome_uniform_values,
                        data_uniform_values=data_uniform_values,
                        syndrome_error_probability=q_value,
                        data_error_probability=p_value,
                    )
                )
                exact_result = compute_exact_logical_observable_means(
                    parity_check_matrix=context["parity_check_matrix"],
                    observed_syndrome_bits=observed_syndrome_bits,
                    disorder_data_error_bits=disorder_data_error_bits,
                    syndrome_error_probability=q_value,
                    data_error_probability=p_value,
                    logical_observable_masks=context["logical_observable_masks"],
                )
                exact_sector_result = compute_exact_logical_sector_weights(
                    parity_check_matrix=context["parity_check_matrix"],
                    observed_syndrome_bits=observed_syndrome_bits,
                    disorder_data_error_bits=disorder_data_error_bits,
                    syndrome_error_probability=q_value,
                    data_error_probability=p_value,
                    logical_observable_masks=context["logical_observable_masks"],
                    num_primitive_logical_qubits=context["num_logical_qubits"],
                )
                initial_chain_bits_per_start, start_sector_labels = (
                    _build_start_states(context, observed_syndrome_bits)
                )
                mcmc_m_u_values = []
                mcmc_q_top_values = []
                aggregated_signature_histogram = np.zeros(
                    1 << context["num_logical_qubits"],
                    dtype=np.int64,
                )
                for start_index in range(DEFAULT_Q0_NUM_START_CHAINS):
                    chain_output = _run_single_chain(
                        context=context,
                        observed_syndrome_bits=observed_syndrome_bits,
                        disorder_data_error_bits=disorder_data_error_bits,
                        syndrome_error_probability=q_value,
                        data_error_probability=p_value,
                        num_burn_in_sweeps=num_burn_in_sweeps,
                        num_measurements_per_disorder=(
                            num_measurements_per_disorder
                        ),
                        num_sweeps_between_measurements=(
                            num_sweeps_between_measurements
                        ),
                        chain_seed=_build_chain_seed(
                            disorder_seed=disorder_seed,
                            start_index=start_index,
                            replica_index=0,
                            q_index=q_index,
                        ),
                        initial_chain_bits=initial_chain_bits_per_start[start_index],
                        num_zero_syndrome_sweeps_per_cycle=1,
                        winding_repeat_factor=1,
                        window_size=window_size,
                    )
                    measurement_result = chain_output["measurement_result"]
                    chain_analysis = chain_output["chain_analysis"]
                    mcmc_m_u_values.append(measurement_result["m_u_values"])
                    mcmc_q_top_values.append(measurement_result["q_top_value"])
                    aggregated_signature_histogram += (
                        chain_analysis["signature_histogram"]
                    )
                mcmc_m_u_mean = np.mean(np.asarray(mcmc_m_u_values), axis=0)
                mcmc_signature_probabilities = (
                    aggregated_signature_histogram
                    / np.sum(aggregated_signature_histogram)
                ).astype(np.float64, copy=False)
                per_disorder_results.append({
                    "disorder_seed": int(disorder_seed),
                    "start_sector_labels": start_sector_labels,
                    "exact_m_u_values": exact_result["m_u_values"],
                    "exact_q_top_value": exact_result["q_top_value"],
                    "exact_signature_probabilities": (
                        exact_sector_result["signature_probabilities"]
                    ),
                    "exact_signature_labels": (
                        exact_sector_result["signature_labels"]
                    ),
                    "mcmc_mean_m_u_values": mcmc_m_u_mean,
                    "mcmc_mean_q_top_value": float(np.mean(mcmc_q_top_values)),
                    "mcmc_signature_probabilities": (
                        mcmc_signature_probabilities
                    ),
                    "mcmc_signature_l1_error": float(np.sum(np.abs(
                        mcmc_signature_probabilities
                        - exact_sector_result["signature_probabilities"]
                    ))),
                    "mcmc_m_u_linf_error": float(np.max(np.abs(
                        mcmc_m_u_mean - exact_result["m_u_values"]
                    ))),
                    "mcmc_q_top_abs_error": float(abs(
                        float(np.mean(mcmc_q_top_values))
                        - exact_result["q_top_value"]
                    )),
                })
            summary_entries.append({
                "lattice_size": lattice_size,
                "p_value": float(p_value),
                "q_value": float(q_value),
                "num_burn_in_sweeps": int(num_burn_in_sweeps),
                "per_disorder_results": per_disorder_results,
            })
    _write_json(run_root / "d1_summary.json", {"entries": summary_entries})
    return summary_entries


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnostics for 3D toric q>0 mixing."
    )
    parser.add_argument(
        "--suite",
        choices=("first_batch", "a1", "a2", "c1", "c2", "b1", "d1", "all"),
        default=DEFAULT_SUITE,
    )
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--disorder-seeds",
        default=",".join(str(seed) for seed in DEFAULT_DISORDER_SEEDS),
    )
    parser.add_argument(
        "--base-num-burn-in-sweeps",
        type=int,
        default=DEFAULT_BASE_NUM_BURN_IN_SWEEPS,
    )
    parser.add_argument(
        "--burn-in-multiplier",
        type=int,
        default=DEFAULT_BURN_IN_MULTIPLIER,
    )
    parser.add_argument(
        "--burn-in-scaling-reference-num-qubits",
        type=int,
        default=DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS,
    )
    parser.add_argument(
        "--num-measurements-per-disorder",
        type=int,
        default=DEFAULT_NUM_MEASUREMENTS,
    )
    parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        default=DEFAULT_NUM_SWEEPS_BETWEEN_MEASUREMENTS,
    )
    parser.add_argument(
        "--num-replicas-per-start",
        type=int,
        default=DEFAULT_NUM_REPLICAS_PER_START,
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--a1-lattice-sizes", default="3,4,5")
    parser.add_argument("--a1-p-values", default="0.22,0.26,0.30")
    parser.add_argument("--a1-q-value", type=float, default=0.005)
    parser.add_argument("--a2-p-values", default="0.22,0.26")
    parser.add_argument("--a2-q-value", type=float, default=0.005)
    parser.add_argument("--c1-lattice-sizes", default="5")
    parser.add_argument("--c1-p-values", default="0.22,0.26,0.30")
    parser.add_argument("--c1-q-value", type=float, default=0.005)
    parser.add_argument("--c2-lattice-sizes", default="5")
    parser.add_argument("--c2-p-cold-values", default="0.22,0.26")
    parser.add_argument("--c2-p-hot", type=float, default=DEFAULT_C2_P_HOT)
    parser.add_argument(
        "--c2-num-temperatures",
        type=int,
        default=DEFAULT_C2_NUM_TEMPERATURES,
    )
    parser.add_argument("--c2-q-value", type=float, default=0.005)
    parser.add_argument("--b1-lattice-sizes", default="3,4,5")
    parser.add_argument("--b1-p-values", default="0.22,0.26")
    parser.add_argument(
        "--b1-q-values",
        default=",".join(f"{value:.6g}" for value in DEFAULT_Q_VALUES_B1),
    )
    parser.add_argument("--d1-p-values", default="0.22,0.26")
    parser.add_argument(
        "--d1-q-values",
        default=",".join(f"{value:.6g}" for value in DEFAULT_Q_VALUES_D1),
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.run_root is None:
        run_root = DEFAULT_RUN_ROOT / f"diagnostic_{_timestamp_tag()}"
    else:
        run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    disorder_seeds = _parse_int_csv(args.disorder_seeds)
    suite_summary = {
        "suite": args.suite,
        "run_root": str(run_root),
        "disorder_seeds": disorder_seeds,
    }

    if args.suite in ("first_batch", "a1", "all"):
        suite_summary["a1"] = _run_a1(
            run_root=run_root,
            lattice_sizes=_parse_int_csv(args.a1_lattice_sizes),
            p_values=_parse_float_csv(args.a1_p_values),
            q_value=float(args.a1_q_value),
            disorder_seeds=disorder_seeds,
            num_replicas_per_start=int(args.num_replicas_per_start),
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    if args.suite in ("a2", "all"):
        suite_summary["a2"] = _run_a2(
            run_root=run_root,
            q_value=float(args.a2_q_value),
            p_values=_parse_float_csv(args.a2_p_values),
            disorder_seeds=disorder_seeds,
            num_replicas_per_start=int(args.num_replicas_per_start),
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    if args.suite in ("first_batch", "c1", "all"):
        suite_summary["c1"] = _run_c1(
            run_root=run_root,
            lattice_sizes=_parse_int_csv(args.c1_lattice_sizes),
            p_values=_parse_float_csv(args.c1_p_values),
            q_value=float(args.c1_q_value),
            disorder_seeds=disorder_seeds,
            num_replicas_per_start=int(args.num_replicas_per_start),
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    if args.suite in ("c2", "all"):
        suite_summary["c2"] = _run_c2(
            run_root=run_root,
            lattice_sizes=_parse_int_csv(args.c2_lattice_sizes),
            p_cold_values=_parse_float_csv(args.c2_p_cold_values),
            p_hot=float(args.c2_p_hot),
            num_temperatures=int(args.c2_num_temperatures),
            q_value=float(args.c2_q_value),
            disorder_seeds=disorder_seeds,
            num_replicas_per_start=int(args.num_replicas_per_start),
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    if args.suite in ("b1", "all"):
        suite_summary["b1"] = _run_b1(
            run_root=run_root,
            lattice_sizes=_parse_int_csv(args.b1_lattice_sizes),
            p_values=_parse_float_csv(args.b1_p_values),
            q_values=_parse_float_csv(args.b1_q_values),
            disorder_seeds=disorder_seeds,
            num_replicas_per_start=int(args.num_replicas_per_start),
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    if args.suite in ("d1", "all"):
        suite_summary["d1"] = _run_d1(
            run_root=run_root,
            p_values=_parse_float_csv(args.d1_p_values),
            q_values=_parse_float_csv(args.d1_q_values),
            disorder_seeds=disorder_seeds,
            base_num_burn_in_sweeps=int(args.base_num_burn_in_sweeps),
            burn_in_multiplier=int(args.burn_in_multiplier),
            burn_in_scaling_reference_num_qubits=int(
                args.burn_in_scaling_reference_num_qubits
            ),
            num_measurements_per_disorder=int(
                args.num_measurements_per_disorder
            ),
            num_sweeps_between_measurements=int(
                args.num_sweeps_between_measurements
            ),
            window_size=int(args.window_size),
        )

    _write_json(run_root / "suite_summary.json", suite_summary)
    print(json.dumps(suite_summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
