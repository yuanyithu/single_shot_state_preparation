import math
import time

import numpy as np


CLUSTER_EWMA_ALPHA = 0.10
CLUSTER_SCORE_EPSILON = 0.05
CLUSTER_OLD_OPS_CONSTANT = 3.0


def _compute_log_odds(probability):
    if probability == 0.0:
        return -np.inf
    if probability == 0.5:
        return 0.0
    if probability == 1.0:
        return np.inf
    return float(np.log(probability / (1.0 - probability)))


def _update_ewma(current_value, new_value, alpha=CLUSTER_EWMA_ALPHA):
    new_value = float(new_value)
    if not np.isfinite(current_value):
        return new_value
    return float((1.0 - alpha) * current_value + alpha * new_value)


def _build_check_supports(parity_check_matrix):
    return [
        np.flatnonzero(parity_check_matrix[check_index]).astype(np.int32)
        for check_index in range(parity_check_matrix.shape[0])
    ]


def build_cluster_controller(
        parity_check_matrix,
        syndrome_error_probability,
        data_error_probability_ladder,
        syndrome_error_probability_ladder=None,
        enabled=True,
        budget_fraction_rho=0.05,
        debug_assertions=False):
    """
    Build a mutable dict controller for the q>0 cluster update.
    """
    data_error_probability_ladder = np.asarray(
        data_error_probability_ladder,
        dtype=np.float64,
    )
    if data_error_probability_ladder.ndim != 1:
        raise ValueError("data_error_probability_ladder must be 1D")
    if syndrome_error_probability_ladder is None:
        syndrome_error_probability_ladder = np.full(
            data_error_probability_ladder.shape,
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
        if (
                syndrome_error_probability_ladder.shape
                != data_error_probability_ladder.shape):
            raise ValueError(
                "syndrome_error_probability_ladder must match data ladder shape"
            )

    num_temperatures = int(data_error_probability_ladder.shape[0])
    num_checks, num_qubits = parity_check_matrix.shape
    valid_probabilities = (
        np.all(syndrome_error_probability_ladder > 0.0)
        and np.all(syndrome_error_probability_ladder < 0.5)
        and np.all(data_error_probability_ladder > 0.0)
        and np.all(data_error_probability_ladder < 0.5)
    )
    effective_enabled = bool(enabled and valid_probabilities)
    log_odds = np.array(
        [_compute_log_odds(float(p)) for p in data_error_probability_ladder],
        dtype=np.float64,
    )
    if num_temperatures == 1:
        hotness = np.ones(1, dtype=np.float64)
    else:
        log_min = float(np.min(log_odds))
        log_max = float(np.max(log_odds))
        if log_max == log_min:
            hotness = np.ones(num_temperatures, dtype=np.float64)
        else:
            hotness = (log_odds - log_min) / (log_max - log_min)

    return {
        "enabled": effective_enabled,
        "requested_enabled": bool(enabled),
        "adaptive": effective_enabled,
        "frozen": False,
        "production_used_adaptive": False,
        "budget_fraction_rho": float(budget_fraction_rho),
        "budget": 0.0,
        "debug_assertions": bool(debug_assertions),
        "syndrome_error_probability": float(syndrome_error_probability),
        "syndrome_error_probability_ladder": syndrome_error_probability_ladder,
        "data_error_probability_ladder": data_error_probability_ladder,
        "log_odds_data_per_temperature": log_odds,
        "hotness": hotness,
        "num_temperatures": num_temperatures,
        "num_checks": int(num_checks),
        "num_qubits": int(num_qubits),
        "check_supports": _build_check_supports(parity_check_matrix),
        "parity_check_matrix": parity_check_matrix.astype(bool, copy=False),
        "ordinary_time_ewma": np.full(num_temperatures, np.nan, dtype=np.float64),
        "cluster_time_ewma": np.full(num_temperatures, np.nan, dtype=np.float64),
        "nullity_ewma": np.full(num_temperatures, np.nan, dtype=np.float64),
        "nonzero_ewma": np.full(num_temperatures, np.nan, dtype=np.float64),
        "move_fraction_ewma": np.full(num_temperatures, np.nan, dtype=np.float64),
        "attempt_count": np.zeros(num_temperatures, dtype=np.int64),
        "success_nonzero_count": np.zeros(num_temperatures, dtype=np.int64),
        "cluster_wall_time": np.zeros(num_temperatures, dtype=np.float64),
        "ordinary_wall_time": np.zeros(num_temperatures, dtype=np.float64),
        "nullity_sum": np.zeros(num_temperatures, dtype=np.float64),
        "move_fraction_sum": np.zeros(num_temperatures, dtype=np.float64),
        "nullity_histogram": np.zeros(num_qubits + 1, dtype=np.int64),
    }


def freeze_cluster_controller(controller):
    if controller is not None:
        controller["frozen"] = True


def _estimate_cluster_cost(controller, temperature_index):
    cluster_time_ewma = controller["cluster_time_ewma"][temperature_index]
    if np.isfinite(cluster_time_ewma) and cluster_time_ewma > 0.0:
        return float(cluster_time_ewma)

    ordinary_time_ewma = controller["ordinary_time_ewma"][temperature_index]
    if not np.isfinite(ordinary_time_ewma) or ordinary_time_ewma <= 0.0:
        ordinary_time_ewma = 1e-6

    n = controller["num_qubits"]
    m = controller["num_checks"]
    p_value = float(controller["data_error_probability_ladder"][temperature_index])
    q_value = float(
        controller["syndrome_error_probability_ladder"][temperature_index]
    )
    f_est = max(1, int(math.floor(2.0 * p_value * n)))
    r_est = max(1, int(math.floor((1.0 - 2.0 * q_value) * m)))
    cluster_ops_est = r_est * f_est * f_est / 64.0
    old_ops_est = CLUSTER_OLD_OPS_CONSTANT * n
    return float(ordinary_time_ewma * cluster_ops_est / max(old_ops_est, 1.0))


def _compute_scores(controller):
    hotness = controller["hotness"]
    n = controller["num_qubits"]
    base_scores = CLUSTER_SCORE_EPSILON + hotness ** 2

    attempt_count = controller["attempt_count"]
    if int(np.sum(attempt_count)) == 0:
        scores = base_scores
    else:
        nonzero = np.nan_to_num(controller["nonzero_ewma"], nan=0.0)
        move_fraction = np.nan_to_num(controller["move_fraction_ewma"], nan=0.0)
        nullity = np.nan_to_num(controller["nullity_ewma"], nan=0.0)
        scores = (
            base_scores
            * (CLUSTER_SCORE_EPSILON + nonzero)
            * (
                CLUSTER_SCORE_EPSILON
                + np.minimum(1.0, move_fraction / 0.05)
            )
            * (
                CLUSTER_SCORE_EPSILON
                + np.minimum(
                    1.0,
                    np.log1p(nullity) / np.log1p(max(4.0, math.sqrt(n))),
                )
            )
        )
    total_score = float(np.sum(scores))
    if total_score <= 0.0 or not np.isfinite(total_score):
        return np.full(controller["num_temperatures"], 1.0 / controller["num_temperatures"])
    return scores / total_score


def _record_ordinary_timings(controller, ordinary_elapsed_per_temperature):
    ordinary_elapsed_per_temperature = np.asarray(
        ordinary_elapsed_per_temperature,
        dtype=np.float64,
    )
    if not controller["enabled"]:
        return
    for temperature_index, elapsed in enumerate(ordinary_elapsed_per_temperature):
        elapsed = max(float(elapsed), 0.0)
        controller["ordinary_wall_time"][temperature_index] += elapsed
        if not controller["frozen"]:
            controller["ordinary_time_ewma"][temperature_index] = _update_ewma(
                controller["ordinary_time_ewma"][temperature_index],
                elapsed,
            )
        controller["budget"] += controller["budget_fraction_rho"] * elapsed

    max_estimated_cost = max(
        _estimate_cluster_cost(controller, temperature_index)
        for temperature_index in range(controller["num_temperatures"])
    )
    if max_estimated_cost > 0.0:
        controller["budget"] = min(controller["budget"], 5.0 * max_estimated_cost)


def _set_packed_bit(row, column_index):
    word_index = column_index // 64
    bit_index = column_index % 64
    row[word_index] |= np.uint64(1) << np.uint64(bit_index)


def _get_packed_bit(row, column_index):
    word_index = column_index // 64
    bit_index = column_index % 64
    mask = np.uint64(1) << np.uint64(bit_index)
    return bool(row[word_index] & mask)


def _packed_parity(row):
    parity = 0
    for value in row:
        parity ^= int(value).bit_count() & 1
    return bool(parity)


def _build_active_restricted_rows(controller, active_check_indices, free_indices):
    num_free = int(free_indices.shape[0])
    num_words = (num_free + 63) // 64
    rows = np.zeros((active_check_indices.shape[0], num_words), dtype=np.uint64)
    free_column_by_qubit = np.full(controller["num_qubits"], -1, dtype=np.int32)
    free_column_by_qubit[free_indices] = np.arange(num_free, dtype=np.int32)

    for row_index, check_index in enumerate(active_check_indices):
        support = controller["check_supports"][int(check_index)]
        restricted_columns = free_column_by_qubit[support]
        restricted_columns = restricted_columns[restricted_columns >= 0]
        for column_index in restricted_columns:
            _set_packed_bit(rows[row_index], int(column_index))
    return rows


def _rref_packed_rows(rows, num_columns):
    rows = np.asarray(rows, dtype=np.uint64).copy()
    num_rows = rows.shape[0]
    rank = 0
    pivot_columns = []

    for column_index in range(num_columns):
        if rank >= num_rows:
            break
        word_index = column_index // 64
        bit_index = column_index % 64
        mask = np.uint64(1) << np.uint64(bit_index)
        candidate_offsets = np.flatnonzero((rows[rank:, word_index] & mask) != 0)
        if candidate_offsets.size == 0:
            continue
        pivot_row = rank + int(candidate_offsets[0])
        if pivot_row != rank:
            rows[[rank, pivot_row]] = rows[[pivot_row, rank]]
        for row_index in range(num_rows):
            if row_index == rank:
                continue
            if rows[row_index, word_index] & mask:
                rows[row_index] ^= rows[rank]
        pivot_columns.append(column_index)
        rank += 1

    return rows[:rank], np.asarray(pivot_columns, dtype=np.int32)


def _sample_kernel_vector_from_rref(rref_rows, pivot_columns, num_columns, rng):
    pivot_mask = np.zeros(num_columns, dtype=bool)
    pivot_mask[pivot_columns] = True
    free_columns = np.flatnonzero(~pivot_mask).astype(np.int32)
    nullity = int(free_columns.shape[0])
    num_words = (num_columns + 63) // 64
    sampled_vector = np.zeros(num_words, dtype=np.uint64)

    if nullity == 0:
        return sampled_vector, 0

    free_coefficients = rng.integers(0, 2, size=nullity, dtype=np.uint8)
    selected_free_columns = free_columns[free_coefficients.astype(bool)]
    for column_index in selected_free_columns:
        _set_packed_bit(sampled_vector, int(column_index))

    for row_index, pivot_column in enumerate(pivot_columns):
        row_projection = rref_rows[row_index] & sampled_vector
        if _packed_parity(row_projection):
            _set_packed_bit(sampled_vector, int(pivot_column))

    return sampled_vector, nullity


def _packed_vector_to_free_indices(sampled_vector, free_indices, num_columns):
    changed_free_columns = []
    for column_index in range(num_columns):
        if _get_packed_bit(sampled_vector, column_index):
            changed_free_columns.append(column_index)
    if not changed_free_columns:
        return np.empty(0, dtype=np.int32)
    return free_indices[np.asarray(changed_free_columns, dtype=np.int32)]


def _attempt_cluster_update(
        controller,
        temperature_index,
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        observed_syndrome_bits,
        disorder_data_error_bits,
        checks_touching_each_qubit,
        rng):
    start_time = time.perf_counter()
    n = controller["num_qubits"]
    p_value = float(controller["data_error_probability_ladder"][temperature_index])
    q_value = float(
        controller["syndrome_error_probability_ladder"][temperature_index]
    )
    a_value = q_value / (1.0 - q_value)
    b_value = p_value / (1.0 - p_value)

    active_check_mask = (
        (~current_syndrome_term_bits)
        & (rng.random(controller["num_checks"]) < (1.0 - a_value))
    )
    active_pin_mask = (
        (~current_data_term_bits)
        & (rng.random(n) < (1.0 - b_value))
    )
    active_check_indices = np.flatnonzero(active_check_mask).astype(np.int32)
    free_indices = np.flatnonzero(~active_pin_mask).astype(np.int32)
    active_ops_estimate = (
        active_check_indices.shape[0]
        * free_indices.shape[0]
        * free_indices.shape[0]
        / 64.0
    )
    ordinary_time_ewma = controller["ordinary_time_ewma"][temperature_index]
    if not np.isfinite(ordinary_time_ewma) or ordinary_time_ewma <= 0.0:
        ordinary_time_ewma = 1e-6
    old_ops_estimate = CLUSTER_OLD_OPS_CONSTANT * n
    active_time_estimate = (
        ordinary_time_ewma
        * active_ops_estimate
        / max(old_ops_estimate, 1.0)
    )

    if active_time_estimate > max(controller["budget"], 0.0):
        elapsed = time.perf_counter() - start_time
        return {
            "attempted": True,
            "nonzero": False,
            "skipped_for_budget": True,
            "elapsed_time": float(elapsed),
            "nullity": 0,
            "move_fraction": 0.0,
            "data_weight_delta": 0,
            "active_check_indices": active_check_indices,
            "active_pin_mask": active_pin_mask,
            "changed_qubit_indices": np.empty(0, dtype=np.int32),
        }

    if free_indices.size == 0:
        elapsed = time.perf_counter() - start_time
        return {
            "attempted": True,
            "nonzero": False,
            "skipped_for_budget": False,
            "elapsed_time": float(elapsed),
            "nullity": 0,
            "move_fraction": 0.0,
            "data_weight_delta": 0,
            "active_check_indices": active_check_indices,
            "active_pin_mask": active_pin_mask,
            "changed_qubit_indices": np.empty(0, dtype=np.int32),
        }

    restricted_rows = _build_active_restricted_rows(
        controller=controller,
        active_check_indices=active_check_indices,
        free_indices=free_indices,
    )
    rref_rows, pivot_columns = _rref_packed_rows(
        rows=restricted_rows,
        num_columns=int(free_indices.shape[0]),
    )
    sampled_vector, nullity = _sample_kernel_vector_from_rref(
        rref_rows=rref_rows,
        pivot_columns=pivot_columns,
        num_columns=int(free_indices.shape[0]),
        rng=rng,
    )
    changed_qubit_indices = _packed_vector_to_free_indices(
        sampled_vector=sampled_vector,
        free_indices=free_indices,
        num_columns=int(free_indices.shape[0]),
    )

    if nullity == 0 or changed_qubit_indices.size == 0:
        elapsed = time.perf_counter() - start_time
        return {
            "attempted": True,
            "nonzero": False,
            "skipped_for_budget": False,
            "elapsed_time": float(elapsed),
            "nullity": int(nullity),
            "move_fraction": 0.0,
            "data_weight_delta": 0,
            "active_check_indices": active_check_indices,
            "active_pin_mask": active_pin_mask,
            "changed_qubit_indices": np.empty(0, dtype=np.int32),
        }

    old_ones_on_changed = int(np.count_nonzero(
        current_data_term_bits[changed_qubit_indices]
    ))
    data_weight_delta = int(changed_qubit_indices.size - 2 * old_ones_on_changed)

    current_chain_bits[changed_qubit_indices] ^= True
    current_data_term_bits[changed_qubit_indices] ^= True
    for qubit_index in changed_qubit_indices:
        current_syndrome_term_bits[
            checks_touching_each_qubit[int(qubit_index)]
        ] ^= True

    if controller["debug_assertions"]:
        if np.any(current_chain_bits[active_pin_mask] != disorder_data_error_bits[active_pin_mask]):
            raise AssertionError("cluster active pins were not preserved")
        if np.any(current_syndrome_term_bits[active_check_mask]):
            raise AssertionError("cluster active checks were not preserved")
        expected_data = current_chain_bits ^ disorder_data_error_bits
        if not np.array_equal(current_data_term_bits, expected_data):
            raise AssertionError("cluster data cache mismatch")
        expected_syndrome = (
            (
                controller["parity_check_matrix"].astype(np.uint8)
                @ current_chain_bits.astype(np.uint8)
            ) % 2
        ).astype(bool)
        expected_syndrome ^= observed_syndrome_bits
        if not np.array_equal(current_syndrome_term_bits, expected_syndrome):
            raise AssertionError("cluster syndrome cache mismatch")

    elapsed = time.perf_counter() - start_time
    return {
        "attempted": True,
        "nonzero": True,
        "skipped_for_budget": False,
        "elapsed_time": float(elapsed),
        "nullity": int(nullity),
        "move_fraction": float(changed_qubit_indices.size / n),
        "data_weight_delta": data_weight_delta,
        "active_check_indices": active_check_indices,
        "active_pin_mask": active_pin_mask,
        "changed_qubit_indices": changed_qubit_indices,
    }


def _record_cluster_result(controller, temperature_index, result):
    elapsed = float(result["elapsed_time"])
    nullity = int(result["nullity"])
    move_fraction = float(result["move_fraction"])
    nonzero_value = 1.0 if result["nonzero"] else 0.0

    controller["attempt_count"][temperature_index] += 1
    controller["success_nonzero_count"][temperature_index] += int(result["nonzero"])
    controller["cluster_wall_time"][temperature_index] += elapsed
    controller["nullity_sum"][temperature_index] += nullity
    controller["move_fraction_sum"][temperature_index] += move_fraction
    if 0 <= nullity < controller["nullity_histogram"].shape[0]:
        controller["nullity_histogram"][nullity] += 1

    if not controller["frozen"]:
        controller["cluster_time_ewma"][temperature_index] = _update_ewma(
            controller["cluster_time_ewma"][temperature_index],
            max(elapsed, 1e-12),
        )
        controller["nullity_ewma"][temperature_index] = _update_ewma(
            controller["nullity_ewma"][temperature_index],
            nullity,
        )
        controller["nonzero_ewma"][temperature_index] = _update_ewma(
            controller["nonzero_ewma"][temperature_index],
            nonzero_value,
        )
        controller["move_fraction_ewma"][temperature_index] = _update_ewma(
            controller["move_fraction_ewma"][temperature_index],
            move_fraction,
        )


def maybe_run_cluster_update(
        controller,
        chain_bits_list,
        data_term_bits_list,
        syndrome_term_bits_list,
        observed_syndrome_bits,
        disorder_data_error_bits,
        checks_touching_each_qubit,
        ordinary_elapsed_per_temperature,
        rng,
        before_update_callback=None):
    """
    Budgeted adaptive scheduler. Returns a small result dict.
    """
    if controller is None or not controller["enabled"]:
        return {
            "attempted": False,
            "temperature_index": -1,
            "data_weight_delta": 0,
            "elapsed_time": 0.0,
        }

    _record_ordinary_timings(controller, ordinary_elapsed_per_temperature)
    scores = _compute_scores(controller)
    temperature_index = int(rng.choice(controller["num_temperatures"], p=scores))
    estimated_cost = _estimate_cluster_cost(controller, temperature_index)
    if controller["budget"] < estimated_cost:
        return {
            "attempted": False,
            "temperature_index": temperature_index,
            "data_weight_delta": 0,
            "elapsed_time": 0.0,
        }

    if before_update_callback is not None:
        before_update_callback(temperature_index)
    result = _attempt_cluster_update(
        controller=controller,
        temperature_index=temperature_index,
        current_chain_bits=chain_bits_list[temperature_index],
        current_data_term_bits=data_term_bits_list[temperature_index],
        current_syndrome_term_bits=syndrome_term_bits_list[temperature_index],
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        checks_touching_each_qubit=checks_touching_each_qubit,
        rng=rng,
    )
    controller["budget"] = max(0.0, controller["budget"] - result["elapsed_time"])
    _record_cluster_result(controller, temperature_index, result)
    result["temperature_index"] = temperature_index
    return result


def _histogram_median(histogram):
    total = int(np.sum(histogram))
    if total <= 0:
        return 0.0
    threshold = (total - 1) // 2
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += int(count)
        if cumulative > threshold:
            return float(index)
    return float(histogram.shape[0] - 1)


def summarize_cluster_controller(controller):
    if controller is None:
        return make_disabled_cluster_summary()

    attempts = controller["attempt_count"].astype(np.int64)
    nonzero = controller["success_nonzero_count"].astype(np.int64)
    total_attempts = int(np.sum(attempts))
    total_nonzero = int(np.sum(nonzero))
    total_cluster_wall_time = float(np.sum(controller["cluster_wall_time"]))
    total_ordinary_wall_time = float(np.sum(controller["ordinary_wall_time"]))
    total_wall = total_cluster_wall_time + total_ordinary_wall_time

    by_temperature_nonzero_rate = np.zeros_like(attempts, dtype=np.float64)
    by_temperature_mean_nullity = np.zeros_like(attempts, dtype=np.float64)
    by_temperature_mean_move_fraction = np.zeros_like(attempts, dtype=np.float64)
    nonzero_mask = attempts > 0
    by_temperature_nonzero_rate[nonzero_mask] = (
        nonzero[nonzero_mask].astype(np.float64)
        / attempts[nonzero_mask].astype(np.float64)
    )
    by_temperature_mean_nullity[nonzero_mask] = (
        controller["nullity_sum"][nonzero_mask]
        / attempts[nonzero_mask].astype(np.float64)
    )
    by_temperature_mean_move_fraction[nonzero_mask] = (
        controller["move_fraction_sum"][nonzero_mask]
        / attempts[nonzero_mask].astype(np.float64)
    )

    if total_attempts == 0:
        nullity_mean = 0.0
        move_fraction_mean = 0.0
    else:
        nullity_mean = float(np.sum(controller["nullity_sum"]) / total_attempts)
        move_fraction_mean = float(
            np.sum(controller["move_fraction_sum"]) / total_attempts
        )

    return {
        "cluster_update_enabled": np.bool_(controller["enabled"]),
        "cluster_update_requested_enabled": np.bool_(controller["requested_enabled"]),
        "cluster_update_adaptive": np.bool_(controller["adaptive"]),
        "cluster_budget_fraction_rho": np.float64(
            controller["budget_fraction_rho"]
        ),
        "cluster_num_attempts": np.int64(total_attempts),
        "cluster_num_nonzero_moves": np.int64(total_nonzero),
        "cluster_total_wall_time": np.float64(total_cluster_wall_time),
        "cluster_wall_time_fraction": np.float64(
            0.0 if total_wall <= 0.0 else total_cluster_wall_time / total_wall
        ),
        "cluster_nullity_mean": np.float64(nullity_mean),
        "cluster_nullity_median": np.float64(
            _histogram_median(controller["nullity_histogram"])
        ),
        "cluster_nullity_histogram": controller["nullity_histogram"].copy(),
        "cluster_move_fraction_mean": np.float64(move_fraction_mean),
        "cluster_by_temperature_attempts": attempts.copy(),
        "cluster_by_temperature_nonzero_rate": by_temperature_nonzero_rate,
        "cluster_by_temperature_mean_nullity": by_temperature_mean_nullity,
        "cluster_by_temperature_mean_move_fraction": (
            by_temperature_mean_move_fraction
        ),
        "cluster_by_temperature_wall_time": (
            controller["cluster_wall_time"].copy()
        ),
        "cluster_controller_frozen": np.bool_(controller["frozen"]),
    }


def make_disabled_cluster_summary(num_temperatures=1, requested_enabled=False,
                                  budget_fraction_rho=0.05):
    return {
        "cluster_update_enabled": np.bool_(False),
        "cluster_update_requested_enabled": np.bool_(requested_enabled),
        "cluster_update_adaptive": np.bool_(False),
        "cluster_budget_fraction_rho": np.float64(budget_fraction_rho),
        "cluster_num_attempts": np.int64(0),
        "cluster_num_nonzero_moves": np.int64(0),
        "cluster_total_wall_time": np.float64(0.0),
        "cluster_wall_time_fraction": np.float64(0.0),
        "cluster_nullity_mean": np.float64(0.0),
        "cluster_nullity_median": np.float64(0.0),
        "cluster_nullity_histogram": np.zeros(1, dtype=np.int64),
        "cluster_move_fraction_mean": np.float64(0.0),
        "cluster_by_temperature_attempts": np.zeros(
            num_temperatures,
            dtype=np.int64,
        ),
        "cluster_by_temperature_nonzero_rate": np.zeros(
            num_temperatures,
            dtype=np.float64,
        ),
        "cluster_by_temperature_mean_nullity": np.zeros(
            num_temperatures,
            dtype=np.float64,
        ),
        "cluster_by_temperature_mean_move_fraction": np.zeros(
            num_temperatures,
            dtype=np.float64,
        ),
        "cluster_by_temperature_wall_time": np.zeros(
            num_temperatures,
            dtype=np.float64,
        ),
        "cluster_controller_frozen": np.bool_(False),
    }


def combine_cluster_summaries(summaries):
    summaries = list(summaries)
    if not summaries:
        return make_disabled_cluster_summary()

    max_num_temperatures = max(
        int(np.asarray(summary["cluster_by_temperature_attempts"]).shape[0])
        for summary in summaries
    )
    max_histogram_size = max(
        int(np.asarray(summary["cluster_nullity_histogram"]).shape[0])
        for summary in summaries
    )

    attempts_by_temperature = np.zeros(max_num_temperatures, dtype=np.int64)
    nonzero_by_temperature = np.zeros(max_num_temperatures, dtype=np.float64)
    nullity_sum_by_temperature = np.zeros(max_num_temperatures, dtype=np.float64)
    move_sum_by_temperature = np.zeros(max_num_temperatures, dtype=np.float64)
    wall_time_by_temperature = np.zeros(max_num_temperatures, dtype=np.float64)
    histogram = np.zeros(max_histogram_size, dtype=np.int64)
    total_cluster_wall_time = 0.0
    weighted_wall_fraction_sum = 0.0
    enabled = False
    requested_enabled = False
    adaptive = False
    frozen = True
    rho = float(summaries[0]["cluster_budget_fraction_rho"])

    for summary in summaries:
        attempts = np.asarray(
            summary["cluster_by_temperature_attempts"],
            dtype=np.int64,
        )
        count = attempts.shape[0]
        attempts_by_temperature[:count] += attempts
        nonzero_by_temperature[:count] += (
            np.asarray(summary["cluster_by_temperature_nonzero_rate"])
            * attempts
        )
        nullity_sum_by_temperature[:count] += (
            np.asarray(summary["cluster_by_temperature_mean_nullity"])
            * attempts
        )
        move_sum_by_temperature[:count] += (
            np.asarray(summary["cluster_by_temperature_mean_move_fraction"])
            * attempts
        )
        wall_time = np.asarray(summary["cluster_by_temperature_wall_time"])
        wall_time_by_temperature[:count] += wall_time
        total_cluster_wall_time += float(summary["cluster_total_wall_time"])
        weighted_wall_fraction_sum += float(summary["cluster_wall_time_fraction"])
        current_histogram = np.asarray(summary["cluster_nullity_histogram"])
        histogram[:current_histogram.shape[0]] += current_histogram
        enabled = enabled or bool(summary["cluster_update_enabled"])
        requested_enabled = requested_enabled or bool(
            summary.get("cluster_update_requested_enabled", False)
        )
        adaptive = adaptive or bool(summary["cluster_update_adaptive"])
        frozen = frozen and bool(summary["cluster_controller_frozen"])

    total_attempts = int(np.sum(attempts_by_temperature))
    total_nonzero = int(round(float(np.sum(nonzero_by_temperature))))
    nonzero_rate = np.zeros(max_num_temperatures, dtype=np.float64)
    mean_nullity = np.zeros(max_num_temperatures, dtype=np.float64)
    mean_move = np.zeros(max_num_temperatures, dtype=np.float64)
    mask = attempts_by_temperature > 0
    nonzero_rate[mask] = nonzero_by_temperature[mask] / attempts_by_temperature[mask]
    mean_nullity[mask] = nullity_sum_by_temperature[mask] / attempts_by_temperature[mask]
    mean_move[mask] = move_sum_by_temperature[mask] / attempts_by_temperature[mask]

    if total_attempts == 0:
        nullity_mean = 0.0
        move_fraction_mean = 0.0
    else:
        nullity_mean = float(np.sum(nullity_sum_by_temperature) / total_attempts)
        move_fraction_mean = float(np.sum(move_sum_by_temperature) / total_attempts)

    return {
        "cluster_update_enabled": np.bool_(enabled),
        "cluster_update_requested_enabled": np.bool_(requested_enabled),
        "cluster_update_adaptive": np.bool_(adaptive),
        "cluster_budget_fraction_rho": np.float64(rho),
        "cluster_num_attempts": np.int64(total_attempts),
        "cluster_num_nonzero_moves": np.int64(total_nonzero),
        "cluster_total_wall_time": np.float64(total_cluster_wall_time),
        "cluster_wall_time_fraction": np.float64(
            weighted_wall_fraction_sum / len(summaries)
        ),
        "cluster_nullity_mean": np.float64(nullity_mean),
        "cluster_nullity_median": np.float64(_histogram_median(histogram)),
        "cluster_nullity_histogram": histogram,
        "cluster_move_fraction_mean": np.float64(move_fraction_mean),
        "cluster_by_temperature_attempts": attempts_by_temperature,
        "cluster_by_temperature_nonzero_rate": nonzero_rate,
        "cluster_by_temperature_mean_nullity": mean_nullity,
        "cluster_by_temperature_mean_move_fraction": mean_move,
        "cluster_by_temperature_wall_time": wall_time_by_temperature,
        "cluster_controller_frozen": np.bool_(frozen),
    }
