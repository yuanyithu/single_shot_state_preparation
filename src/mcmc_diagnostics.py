import numpy as np


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
