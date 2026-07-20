import numpy as np


def basis_character_traces(label_traces, k):
    traces = np.asarray(label_traces, dtype=np.uint64)
    return np.stack([
        1.0 - 2.0 * ((traces >> np.uint64(bit)) & np.uint64(1)).astype(np.float64)
        for bit in range(k)
    ], axis=1)


def split_rhat(chains):
    chains = np.asarray(chains, dtype=np.float64)
    half = chains.shape[1] // 2
    if half < 2:
        return np.inf
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1))
    if within == 0:
        return 1.0 if np.all(split == split[0, 0]) else np.inf
    between = half * np.var(np.mean(split, axis=1), ddof=1)
    return float(np.sqrt(((half - 1) / half * within + between / half) / within))


def bulk_ess(chains):
    chains = np.asarray(chains, dtype=np.float64)
    total = chains.size
    centered = chains - chains.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered)
    if variance == 0:
        return float(total)
    rho_sum = 0.0
    previous_pair = np.inf
    for lag in range(1, min(chains.shape[1] - 1, 1000), 2):
        pair = 0.0
        for offset in (0, 1):
            value = np.mean(centered[:, :-(lag + offset)] * centered[:, lag + offset:]) / variance
            pair += value
        pair = min(pair, previous_pair)
        if pair <= 0:
            break
        rho_sum += pair
        previous_pair = pair
    return float(min(total, total / max(1.0 + 2.0 * rho_sum, 1.0)))


def evaluate_gate(results, gate, k, require_trace_gate=True):
    failures = []
    traces = np.stack([result["labels"] for result in results])
    if len({int(result["seed"]) for result in results}) != 4:
        failures.append("duplicate_instance_seed")
    for index, result in enumerate(results):
        if result["max_hard_coset_residual"]:
            failures.append(f"instance_{index}:hard_coset")
        rates = result["swap_accepts"] / np.maximum(result["swap_attempts"], 1)
        if np.any(rates < gate["min_swap_rate"]) or np.any(result["swap_accepts"] < gate["min_swap_accepts"]):
            failures.append(f"instance_{index}:swap")
        if result["round_trips"] < gate["min_round_trips"]:
            failures.append(f"instance_{index}:round_trips")
        if result["sector_changing_round_trips"] < gate["min_sector_changing_round_trips"]:
            failures.append(f"instance_{index}:sector_transport")
        hot_attempts = result["logical_attempts"][-1]
        hot_accepts = result["logical_accepts"][-1]
        hot_rates = hot_accepts / np.maximum(hot_attempts, 1)
        if np.any(hot_accepts < gate["min_hot_logical_accepts_per_basis"]) or np.any(hot_rates < gate["min_hot_logical_rate"]):
            failures.append(f"instance_{index}:hot_logical")
    characters = basis_character_traces(traces, k)
    rhats = np.full(k, np.nan)
    esses = np.full(k, np.nan)
    statuses = np.full(k, "nonconstant", dtype="U40")
    for bit in range(k):
        values = characters[:, bit, :]
        if np.all(values == values[:, :1]):
            same_sign = np.unique(values).size == 1
            transported = all(result["sector_changing_round_trips"] >= gate["min_sector_changing_round_trips"] for result in results)
            residual_ok = all(result["max_hard_coset_residual"] == 0 for result in results)
            swap_ok = all(np.all(result["swap_accepts"] >= gate["min_swap_accepts"]) for result in results)
            hot_ok = all(np.all(result["logical_accepts"][-1] >= gate["min_hot_logical_accepts_per_basis"]) for result in results)
            if same_sign and transported and residual_ok and swap_ok and hot_ok:
                statuses[bit] = "constant_with_certified_transport"
                rhats[bit], esses[bit] = 1.0, float(values.size)
            else:
                statuses[bit] = "constant_rejected"
                if require_trace_gate:
                    failures.append(f"basis_{bit}:constant_untrusted")
        else:
            rhats[bit], esses[bit] = split_rhat(values), bulk_ess(values)
            spread = np.ptp(values.mean(axis=1))
            if require_trace_gate and (
                    rhats[bit] > gate["max_rhat"] or esses[bit] < gate["min_ess"]
                    or spread > gate["max_instance_mean_spread"]):
                failures.append(f"basis_{bit}:trace")
    return not failures, sorted(set(failures)), rhats, esses, statuses
