import numpy as np


SYNC_PT_LADDER_SEMANTICS = "common_beta_heat_scale"
DATA_ONLY_PT_LADDER_SEMANTICS = "data_only_log_odds"
DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR = 1.5


def probability_to_odds(probability):
    probability = float(probability)
    if not (0.0 < probability < 0.5):
        raise ValueError("probability must be in (0, 0.5)")
    return probability / (1.0 - probability)


def odds_to_probability(odds):
    odds = float(odds)
    if odds <= 0.0:
        raise ValueError("odds must be positive")
    return odds / (1.0 + odds)


def probability_to_coupling(probability):
    probability = float(probability)
    if not (0.0 < probability < 0.5):
        raise ValueError("probability must be in (0, 0.5)")
    return float(np.log((1.0 - probability) / probability))


def coupling_to_probability(coupling):
    coupling = float(coupling)
    if coupling < 0.0:
        raise ValueError("coupling must be non-negative")
    return float(1.0 / (1.0 + np.exp(coupling)))


def equal_log_odds_ladder(p_cold, p_hot, num_temperatures):
    if num_temperatures < 1:
        raise ValueError("num_temperatures must be >= 1")
    log_odds_cold = np.log(p_cold / (1.0 - p_cold))
    log_odds_hot = np.log(p_hot / (1.0 - p_hot))
    log_odds_values = np.linspace(
        log_odds_cold,
        log_odds_hot,
        int(num_temperatures),
    )
    return np.exp(log_odds_values) / (1.0 + np.exp(log_odds_values))


def sync_pt_enlarge_ladder(q_cold, q_hot, num_temperatures):
    num_temperatures = int(num_temperatures)
    if num_temperatures < 1:
        raise ValueError("num_temperatures must be >= 1")
    if num_temperatures == 1:
        return np.asarray([1.0], dtype=np.float64)
    q_cold_coupling = probability_to_coupling(q_cold)
    q_hot_coupling = probability_to_coupling(q_hot)
    beta_hot = q_hot_coupling / q_cold_coupling
    if not (0.0 < beta_hot < 1.0):
        raise ValueError("q_hot must be greater than q_cold and below 0.5")
    hot_enlarge = 1.0 / beta_hot
    return np.exp(
        np.linspace(0.0, np.log(hot_enlarge), num_temperatures)
    ).astype(np.float64)


def sync_pt_ladders_from_enlarge(p_cold, q_cold, pt_enlarge):
    pt_enlarge = np.asarray(pt_enlarge, dtype=np.float64)
    if pt_enlarge.ndim != 1:
        raise ValueError("pt_enlarge must be 1D")
    if np.any(~np.isfinite(pt_enlarge)) or np.any(pt_enlarge <= 0.0):
        raise ValueError("pt_enlarge values must be finite and positive")
    if np.any(pt_enlarge < 1.0):
        raise ValueError("sync PT heat scales must be >= 1")
    p_cold_coupling = probability_to_coupling(p_cold)
    q_cold_coupling = probability_to_coupling(q_cold)
    beta_ladder = 1.0 / pt_enlarge
    p_ladder = np.asarray(
        [
            coupling_to_probability(beta * p_cold_coupling)
            for beta in beta_ladder
        ],
        dtype=np.float64,
    )
    q_ladder = np.asarray(
        [
            coupling_to_probability(beta * q_cold_coupling)
            for beta in beta_ladder
        ],
        dtype=np.float64,
    )
    if np.any(p_ladder >= 0.5) or np.any(q_ladder >= 0.5):
        raise ValueError("sync PT ladder must keep p_k and q_k below 0.5")
    return p_ladder, q_ladder


def build_adaptive_pt_flow_tracker(num_temperatures):
    return {
        "replica_tags": np.zeros(num_temperatures, dtype=np.int8),
        "l_count_per_temperature": np.zeros(num_temperatures, dtype=np.int64),
        "r_count_per_temperature": np.zeros(num_temperatures, dtype=np.int64),
        "unlabeled_count_per_temperature": np.zeros(
            num_temperatures,
            dtype=np.int64,
        ),
        "num_observations": 0,
    }


def record_adaptive_pt_flow(flow_tracker, replica_id_per_temperature):
    if flow_tracker is None:
        return
    replica_id_per_temperature = np.asarray(
        replica_id_per_temperature,
        dtype=np.int64,
    )
    num_temperatures = int(replica_id_per_temperature.shape[0])
    if num_temperatures < 2:
        return
    flow_tracker["replica_tags"][int(replica_id_per_temperature[0])] = 1
    flow_tracker["replica_tags"][int(replica_id_per_temperature[-1])] = 2
    for temperature_index, replica_id in enumerate(replica_id_per_temperature):
        tag = int(flow_tracker["replica_tags"][int(replica_id)])
        if tag == 1:
            flow_tracker["l_count_per_temperature"][temperature_index] += 1
        elif tag == 2:
            flow_tracker["r_count_per_temperature"][temperature_index] += 1
        else:
            flow_tracker["unlabeled_count_per_temperature"][
                temperature_index
            ] += 1
    flow_tracker["num_observations"] += 1


def _fill_flow_nan_values(f_raw):
    f_raw = np.asarray(f_raw, dtype=np.float64)
    if f_raw.size == 0:
        return f_raw.copy()
    finite_mask = np.isfinite(f_raw)
    if not np.any(finite_mask):
        return np.linspace(1.0, 0.0, f_raw.size)
    indices = np.arange(f_raw.size, dtype=np.float64)
    filled = np.interp(
        indices,
        indices[finite_mask],
        f_raw[finite_mask],
    )
    filled[0] = 1.0
    if f_raw.size > 1:
        filled[-1] = 0.0
    return filled


def _monotone_flow_profile(f_values):
    f_values = np.asarray(f_values, dtype=np.float64)
    if f_values.size == 0:
        return f_values.copy()
    f_mono = np.empty_like(f_values)
    f_mono[0] = 1.0
    for index in range(1, f_values.size - 1):
        f_mono[index] = min(float(f_values[index]), float(f_mono[index - 1]))
    if f_values.size > 1:
        f_mono[-1] = 0.0
    return f_mono


def summarize_adaptive_pt_flow(flow_tracker):
    if flow_tracker is None:
        return None
    l_counts = np.asarray(flow_tracker["l_count_per_temperature"], dtype=np.int64)
    r_counts = np.asarray(flow_tracker["r_count_per_temperature"], dtype=np.int64)
    unlabeled_counts = np.asarray(
        flow_tracker["unlabeled_count_per_temperature"],
        dtype=np.int64,
    )
    denominator = l_counts + r_counts
    f_raw = np.full(l_counts.shape[0], np.nan, dtype=np.float64)
    mask = denominator > 0
    f_raw[mask] = l_counts[mask].astype(np.float64) / denominator[mask]
    f_filled = _fill_flow_nan_values(f_raw)
    f_mono = _monotone_flow_profile(f_filled)
    return {
        "l_count_per_temperature": l_counts,
        "r_count_per_temperature": r_counts,
        "unlabeled_count_per_temperature": unlabeled_counts,
        "num_observations": int(flow_tracker["num_observations"]),
        "f_raw": f_raw,
        "f_mono": f_mono,
        "f_target": np.linspace(1.0, 0.0, l_counts.shape[0]),
    }


def _cap_adaptive_log_ladder_gaps(
        proposed_log_enlarge,
        max_log_gap_factor=DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR):
    proposed_log_enlarge = np.asarray(proposed_log_enlarge, dtype=np.float64)
    if proposed_log_enlarge.size < 3:
        return proposed_log_enlarge.copy(), False
    total_log_span = float(
        proposed_log_enlarge[-1] - proposed_log_enlarge[0]
    )
    if total_log_span <= 0.0:
        return proposed_log_enlarge.copy(), False
    uniform_gap = total_log_span / float(proposed_log_enlarge.size - 1)
    max_allowed_gap = float(max_log_gap_factor) * uniform_gap
    if max_allowed_gap < uniform_gap:
        raise ValueError("max_log_gap_factor must be >= 1")
    proposed_gaps = np.diff(proposed_log_enlarge)
    max_proposed_gap = float(np.max(proposed_gaps))
    if max_proposed_gap <= max_allowed_gap + 1e-12:
        return proposed_log_enlarge.copy(), False

    uniform_log_enlarge = np.linspace(
        proposed_log_enlarge[0],
        proposed_log_enlarge[-1],
        proposed_log_enlarge.size,
    )
    if max_proposed_gap <= uniform_gap + 1e-12:
        blend_fraction = 0.0
    else:
        blend_fraction = (
            (max_allowed_gap - uniform_gap)
            / (max_proposed_gap - uniform_gap)
        )
        blend_fraction = float(np.clip(blend_fraction, 0.0, 1.0))
    capped_log_enlarge = (
        uniform_log_enlarge
        + blend_fraction * (proposed_log_enlarge - uniform_log_enlarge)
    )
    capped_log_enlarge[0] = proposed_log_enlarge[0]
    capped_log_enlarge[-1] = proposed_log_enlarge[-1]
    capped_log_enlarge = np.maximum.accumulate(capped_log_enlarge)
    return capped_log_enlarge, True


def adaptive_ladder_from_flow(
        pt_enlarge,
        f_mono,
        max_log_gap_factor=DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR):
    pt_enlarge = np.asarray(pt_enlarge, dtype=np.float64)
    f_mono = np.asarray(f_mono, dtype=np.float64)
    if pt_enlarge.size != f_mono.size:
        raise ValueError("pt_enlarge and f_mono must have the same length")
    if pt_enlarge.size < 2:
        return pt_enlarge.copy(), "single_temperature"

    target_flow = np.linspace(1.0, 0.0, pt_enlarge.size)
    log_enlarge = np.log(pt_enlarge)
    reverse_flow = f_mono[::-1]
    reverse_log_enlarge = log_enlarge[::-1]
    unique_flow = []
    unique_log_enlarge = []
    for flow_value, log_value in zip(reverse_flow, reverse_log_enlarge):
        flow_value = float(flow_value)
        if not unique_flow or flow_value > unique_flow[-1] + 1e-12:
            unique_flow.append(flow_value)
            unique_log_enlarge.append(float(log_value))
        else:
            unique_log_enlarge[-1] = min(unique_log_enlarge[-1], float(log_value))
    if len(unique_flow) < 2:
        return pt_enlarge.copy(), "degenerate_flow"
    new_log_enlarge = np.interp(
        target_flow,
        np.asarray(unique_flow, dtype=np.float64),
        np.asarray(unique_log_enlarge, dtype=np.float64),
    )
    new_log_enlarge[0] = log_enlarge[0]
    new_log_enlarge[-1] = log_enlarge[-1]
    new_log_enlarge = np.maximum.accumulate(new_log_enlarge)
    new_log_enlarge, capped = _cap_adaptive_log_ladder_gaps(
        new_log_enlarge,
        max_log_gap_factor=max_log_gap_factor,
    )
    return np.exp(new_log_enlarge), ("ok_capped_gap" if capped else "ok")


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


def integrated_autocorrelation_time(values):
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


def split_r_hat(chains):
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


def aggregate_r_hat(logical_observable_values_tensor):
    logical_observable_values_tensor = np.asarray(
        logical_observable_values_tensor,
        dtype=np.float64,
    )
    if logical_observable_values_tensor.ndim != 3:
        raise ValueError(
            "logical_observable_values_tensor must have shape "
            "(num_chains, num_samples, num_masks)"
        )
    num_masks = logical_observable_values_tensor.shape[-1]
    r_hat_per_mask = np.empty(num_masks, dtype=np.float64)
    for mask_index in range(num_masks):
        r_hat_per_mask[mask_index] = split_r_hat(
            logical_observable_values_tensor[:, :, mask_index]
        )
    return r_hat_per_mask


def signature_indices_from_logical_observable_values(
        logical_observable_values_per_measurement,
        num_logical_qubits):
    parity_bits = (
        np.asarray(logical_observable_values_per_measurement)
        [:, :num_logical_qubits] < 0
    ).astype(np.int64)
    bit_weights = 1 << np.arange(num_logical_qubits, dtype=np.int64)
    return parity_bits @ bit_weights


def analyze_chain_diagnostics(
        logical_observable_values_per_measurement,
        num_logical_qubits):
    logical_observable_values_per_measurement = np.asarray(
        logical_observable_values_per_measurement,
        dtype=np.int8,
    )
    num_measurements = logical_observable_values_per_measurement.shape[0]
    tau_int_per_mask = np.array(
        [
            integrated_autocorrelation_time(
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
    signature_indices = signature_indices_from_logical_observable_values(
        logical_observable_values_per_measurement,
        num_logical_qubits=num_logical_qubits,
    )
    first_signature_change_index = -1
    initial_signature_index = int(signature_indices[0])
    changed_signature_positions = np.flatnonzero(
        signature_indices != initial_signature_index
    )
    if changed_signature_positions.size > 0:
        first_signature_change_index = int(changed_signature_positions[0])
    return {
        "tau_int_per_mask": tau_int_per_mask,
        "max_tau_int": max_tau_int,
        "effective_sample_size": effective_sample_size,
        "signature_indices": signature_indices,
        "first_signature_change_index": np.int64(
            first_signature_change_index
        ),
    }


def summarize_multi_chain_convergence(
        chain_m_u_values,
        chain_q_top_values,
        chain_effective_sample_size_values,
        chain_first_signature_change_index_values,
        logical_observable_values_tensor):
    chain_m_u_values = np.asarray(chain_m_u_values, dtype=np.float64)
    chain_q_top_values = np.asarray(chain_q_top_values, dtype=np.float64)
    chain_effective_sample_size_values = np.asarray(
        chain_effective_sample_size_values,
        dtype=np.float64,
    )
    chain_first_signature_change_index_values = np.asarray(
        chain_first_signature_change_index_values,
        dtype=np.int64,
    )
    logical_observable_values_tensor = np.asarray(
        logical_observable_values_tensor,
        dtype=np.int8,
    )

    pairwise_m_u_diff = np.abs(
        chain_m_u_values[:, None, :]
        - chain_m_u_values[None, :, :]
    )
    q_top_spread = float(
        np.max(chain_q_top_values) - np.min(chain_q_top_values)
    )
    m_u_spread_linf = float(np.max(pairwise_m_u_diff))
    r_hat_per_mask = aggregate_r_hat(logical_observable_values_tensor)
    finite_r_hat_values = r_hat_per_mask[np.isfinite(r_hat_per_mask)]
    if finite_r_hat_values.size == 0:
        max_r_hat = np.nan
    else:
        max_r_hat = float(np.max(finite_r_hat_values))
    return {
        "q_top_spread": q_top_spread,
        "m_u_spread_linf": m_u_spread_linf,
        "r_hat_per_mask": r_hat_per_mask,
        "max_r_hat": max_r_hat,
        "min_effective_sample_size": float(
            np.min(chain_effective_sample_size_values)
        ),
        "num_chains_that_never_flipped_sector": int(
            np.count_nonzero(
                chain_first_signature_change_index_values == -1
            )
        ),
    }
