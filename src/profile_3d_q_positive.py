import argparse
import json
import math
import multiprocessing
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration dependency
    njit = None

from build_toric_code_examples import (
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from cluster_update import (
    build_cluster_controller,
    freeze_cluster_controller,
    maybe_run_cluster_update,
    summarize_cluster_controller,
)
from linear_section import (
    apply_section,
    build_linear_section,
    build_syndrome_representative_section,
)
from main import (
    _build_q0_initial_chain_bits_per_start,
    _compute_log_odds,
    _run_one_sweep_safe,
)
from mcmc import (
    compute_logical_observable_values,
    draw_disorder_sample_from_uniform_values,
    initialize_mcmc_state,
)
from mcmc_diagnostics import (
    aggregate_r_hat,
    equal_log_odds_ladder,
    integrated_autocorrelation_time,
)
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
    / "profile_3d_q_positive"
)
STAGE_NAMES = (
    "single_bit",
    "contractible",
    "winding",
    "cluster",
    "pt_swap",
    "observable",
)
DEFAULT_Q_VALUE = 0.005
DEFAULT_SEED_BASE = 2026052301
NUMBA_AVAILABLE = njit is not None


if njit is not None:
    @njit(cache=True)
    def _numba_shuffle_int32_inplace(values):
        for index in range(values.shape[0] - 1, 0, -1):
            swap_index = np.random.randint(0, index + 1)
            temporary_value = values[index]
            values[index] = values[swap_index]
            values[swap_index] = temporary_value


    @njit(cache=True)
    def _numba_single_bit_stage_3d(
            current_chain_bits,
            current_data_term_bits,
            current_syndrome_term_bits,
            checks_touching_each_qubit_array,
            qubit_order_buffer,
            log_odds_data,
            log_odds_syndrome,
            random_seed):
        np.random.seed(random_seed)
        _numba_shuffle_int32_inplace(qubit_order_buffer)
        accepted_count = 0
        data_weight_delta = 0
        for order_position in range(qubit_order_buffer.shape[0]):
            qubit_index = qubit_order_buffer[order_position]
            if current_data_term_bits[qubit_index]:
                delta_data_weight = -1
            else:
                delta_data_weight = 1

            delta_syndrome_weight = 0
            check_index_0 = checks_touching_each_qubit_array[qubit_index, 0]
            check_index_1 = checks_touching_each_qubit_array[qubit_index, 1]
            check_index_2 = checks_touching_each_qubit_array[qubit_index, 2]
            check_index_3 = checks_touching_each_qubit_array[qubit_index, 3]
            if current_syndrome_term_bits[check_index_0]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_1]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_2]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_3]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1

            log_acceptance = (
                delta_data_weight * log_odds_data
                + delta_syndrome_weight * log_odds_syndrome
            )
            accepted = False
            if log_acceptance >= 0.0:
                accepted = True
            elif np.random.random() < math.exp(log_acceptance):
                accepted = True
            if accepted:
                current_chain_bits[qubit_index] = (
                    not current_chain_bits[qubit_index]
                )
                current_data_term_bits[qubit_index] = (
                    not current_data_term_bits[qubit_index]
                )
                current_syndrome_term_bits[check_index_0] = (
                    not current_syndrome_term_bits[check_index_0]
                )
                current_syndrome_term_bits[check_index_1] = (
                    not current_syndrome_term_bits[check_index_1]
                )
                current_syndrome_term_bits[check_index_2] = (
                    not current_syndrome_term_bits[check_index_2]
                )
                current_syndrome_term_bits[check_index_3] = (
                    not current_syndrome_term_bits[check_index_3]
                )
                accepted_count += 1
                data_weight_delta += delta_data_weight
        return accepted_count, data_weight_delta


    @njit(cache=True)
    def _numba_single_bit_stage_3d_sparse(
            current_chain_bits,
            current_data_term_bits,
            current_syndrome_term_bits,
            checks_touching_each_qubit_array,
            qubit_order_buffer,
            num_attempts,
            log_odds_data,
            log_odds_syndrome,
            random_seed):
        np.random.seed(random_seed)
        accepted_count = 0
        data_weight_delta = 0
        num_qubits = qubit_order_buffer.shape[0]
        if num_attempts > num_qubits:
            num_attempts = num_qubits
        if num_attempts < 0:
            num_attempts = 0
        for order_position in range(num_attempts):
            swap_index = np.random.randint(order_position, num_qubits)
            temporary_value = qubit_order_buffer[order_position]
            qubit_order_buffer[order_position] = qubit_order_buffer[swap_index]
            qubit_order_buffer[swap_index] = temporary_value

            qubit_index = qubit_order_buffer[order_position]
            if current_data_term_bits[qubit_index]:
                delta_data_weight = -1
            else:
                delta_data_weight = 1

            delta_syndrome_weight = 0
            check_index_0 = checks_touching_each_qubit_array[qubit_index, 0]
            check_index_1 = checks_touching_each_qubit_array[qubit_index, 1]
            check_index_2 = checks_touching_each_qubit_array[qubit_index, 2]
            check_index_3 = checks_touching_each_qubit_array[qubit_index, 3]
            if current_syndrome_term_bits[check_index_0]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_1]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_2]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1
            if current_syndrome_term_bits[check_index_3]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1

            log_acceptance = (
                delta_data_weight * log_odds_data
                + delta_syndrome_weight * log_odds_syndrome
            )
            accepted = False
            if log_acceptance >= 0.0:
                accepted = True
            elif np.random.random() < math.exp(log_acceptance):
                accepted = True
            if accepted:
                current_chain_bits[qubit_index] = (
                    not current_chain_bits[qubit_index]
                )
                current_data_term_bits[qubit_index] = (
                    not current_data_term_bits[qubit_index]
                )
                current_syndrome_term_bits[check_index_0] = (
                    not current_syndrome_term_bits[check_index_0]
                )
                current_syndrome_term_bits[check_index_1] = (
                    not current_syndrome_term_bits[check_index_1]
                )
                current_syndrome_term_bits[check_index_2] = (
                    not current_syndrome_term_bits[check_index_2]
                )
                current_syndrome_term_bits[check_index_3] = (
                    not current_syndrome_term_bits[check_index_3]
                )
                accepted_count += 1
                data_weight_delta += delta_data_weight
        return accepted_count, data_weight_delta


    @njit(cache=True)
    def _numba_zero_syndrome_stage_fixed_support(
            current_chain_bits,
            current_data_term_bits,
            move_supports,
            order_buffer,
            log_odds_data,
            random_seed):
        np.random.seed(random_seed)
        _numba_shuffle_int32_inplace(order_buffer)
        accepted_count = 0
        data_weight_delta = 0
        support_size = move_supports.shape[1]
        for order_position in range(order_buffer.shape[0]):
            move_index = order_buffer[order_position]
            current_ones_on_support = 0
            for support_position in range(support_size):
                support_qubit = move_supports[move_index, support_position]
                if current_data_term_bits[support_qubit]:
                    current_ones_on_support += 1
            delta_data_weight = support_size - 2 * current_ones_on_support
            log_acceptance = delta_data_weight * log_odds_data
            accepted = False
            if log_acceptance >= 0.0:
                accepted = True
            elif np.random.random() < math.exp(log_acceptance):
                accepted = True
            if accepted:
                for support_position in range(support_size):
                    support_qubit = move_supports[move_index, support_position]
                    current_chain_bits[support_qubit] = (
                        not current_chain_bits[support_qubit]
                    )
                    current_data_term_bits[support_qubit] = (
                        not current_data_term_bits[support_qubit]
                    )
                accepted_count += 1
                data_weight_delta += delta_data_weight
        return accepted_count, data_weight_delta
else:
    _numba_single_bit_stage_3d = None
    _numba_single_bit_stage_3d_sparse = None
    _numba_zero_syndrome_stage_fixed_support = None


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamp_tag():
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _log(message):
    print(f"[{_timestamp()}] {message}", flush=True)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
    temporary_path.replace(path)


def _parse_int_csv(value):
    return [int(token.strip()) for token in str(value).split(",") if token.strip()]


def _parse_float_csv(value):
    return [float(token.strip()) for token in str(value).split(",") if token.strip()]


def _probability_tag(value):
    return f"{float(value):0.4f}".replace(".", "p")


def _sanitize_label(value):
    return (
        str(value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace("/", "_")
    )


def _safe_rate(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator


def _new_stage_profile():
    return {
        stage_name: {
            "wall_time": 0.0,
            "attempted": 0,
            "accepted": 0,
            "sector_changes": 0,
            "data_weight_delta": 0,
        }
        for stage_name in STAGE_NAMES
    }


def _record_stage(
        stage_profile,
        stage_name,
        wall_time=0.0,
        attempted=0,
        accepted=0,
        sector_changed=False,
        data_weight_delta=0):
    entry = stage_profile[stage_name]
    entry["wall_time"] += float(wall_time)
    entry["attempted"] += int(attempted)
    entry["accepted"] += int(accepted)
    entry["sector_changes"] += int(bool(sector_changed))
    entry["data_weight_delta"] += int(data_weight_delta)


def _build_context(lattice_size):
    parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
        code_family="3d_toric",
        lattice_size=int(lattice_size),
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family="3d_toric",
        lattice_size=int(lattice_size),
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    section_data = build_syndrome_representative_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    primitive_mask_indices = np.asarray(
        [(1 << index) - 1 for index in range(dual_logical_z_basis.shape[0])],
        dtype=np.int64,
    )
    return {
        "lattice_size": int(lattice_size),
        "parity_check_matrix": parity_check_matrix,
        "parity_check_matrix_uint8": parity_check_matrix.astype(np.uint8),
        "dual_logical_z_basis": dual_logical_z_basis,
        "linear_section_data": linear_section_data,
        "logical_observable_masks": logical_observable_masks,
        "checks_touching_each_qubit": checks_touching_each_qubit,
        "zero_syndrome_move_data": zero_syndrome_move_data,
        "section_data": section_data,
        "contractible_move_supports": np.asarray(
            zero_syndrome_move_data["contractible_move_supports"],
            dtype=np.int32,
        ),
        "winding_move_supports": np.asarray(
            zero_syndrome_move_data["winding_move_supports"],
            dtype=np.int32,
        ),
        "checks_touching_each_qubit_array": np.asarray(
            checks_touching_each_qubit,
            dtype=np.int32,
        ),
        "num_checks": int(parity_check_matrix.shape[0]),
        "num_qubits": int(parity_check_matrix.shape[1]),
        "num_logical_qubits": int(dual_logical_z_basis.shape[0]),
        "num_masks": int(logical_observable_masks.shape[0]),
        "primitive_mask_indices": primitive_mask_indices,
    }


def _draw_common_disorder(context, q_value, p_value, disorder_seed):
    rng = np.random.default_rng(int(disorder_seed))
    syndrome_uniform_values = rng.random(context["num_checks"])
    data_uniform_values = rng.random(context["num_qubits"])
    observed_syndrome_bits, disorder_data_error_bits = (
        draw_disorder_sample_from_uniform_values(
            syndrome_uniform_values=syndrome_uniform_values,
            data_uniform_values=data_uniform_values,
            syndrome_error_probability=float(q_value),
            data_error_probability=float(p_value),
        )
    )
    return observed_syndrome_bits, disorder_data_error_bits


def _compute_disorder_syndrome_representative(
        context,
        disorder_data_error_bits):
    disorder_syndrome_bits = (
        context["parity_check_matrix_uint8"]
        @ disorder_data_error_bits.astype(np.uint8)
    ) % 2
    return apply_section(
        disorder_syndrome_bits.astype(bool),
        context["section_data"],
    )


def _logical_values_for_state(
        context,
        current_chain_bits,
        disorder_data_error_bits,
        disorder_syndrome_representative_bits,
        current_syndrome_term_bits=None,
        observed_syndrome_bits=None):
    if current_syndrome_term_bits is not None:
        if observed_syndrome_bits is None:
            raise ValueError(
                "observed_syndrome_bits is required with current_syndrome_term_bits"
            )
        chain_syndrome_bits = (
            np.asarray(current_syndrome_term_bits, dtype=bool)
            ^ np.asarray(observed_syndrome_bits, dtype=bool)
        )
        chain_syndrome_representative_bits = apply_section(
            chain_syndrome_bits,
            context["section_data"],
        )
        logical_chain_bits = (
            current_chain_bits
            ^ disorder_data_error_bits
            ^ chain_syndrome_representative_bits
            ^ disorder_syndrome_representative_bits
        )
        masked_bits = context["logical_observable_masks"] & logical_chain_bits
        parity_bits = np.bitwise_xor.reduce(masked_bits, axis=1)
        return (1 - 2 * parity_bits.astype(np.int8)).astype(
            np.int8,
            copy=False,
        )

    return compute_logical_observable_values(
        current_chain_bits=current_chain_bits,
        logical_observable_masks=context["logical_observable_masks"],
        parity_check_matrix=context["parity_check_matrix"],
        section_data=context["section_data"],
        disorder_data_error_bits=disorder_data_error_bits,
        disorder_syndrome_representative_bits=(
            disorder_syndrome_representative_bits
        ),
    )


def _signature_from_logical_values(context, logical_values):
    primitive_values = logical_values[context["primitive_mask_indices"]]
    parity_bits = (primitive_values < 0).astype(np.int64)
    bit_weights = 1 << np.arange(context["num_logical_qubits"], dtype=np.int64)
    return int(parity_bits @ bit_weights)


def _signature_for_state(
        context,
        current_chain_bits,
        disorder_data_error_bits,
        disorder_syndrome_representative_bits,
        current_syndrome_term_bits=None,
        observed_syndrome_bits=None):
    logical_values = _logical_values_for_state(
        context=context,
        current_chain_bits=current_chain_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        disorder_syndrome_representative_bits=(
            disorder_syndrome_representative_bits
        ),
        current_syndrome_term_bits=current_syndrome_term_bits,
        observed_syndrome_bits=observed_syndrome_bits,
    )
    return _signature_from_logical_values(context, logical_values)


def _signatures_for_temperatures(
        context,
        chain_bits_list,
        syndrome_term_bits_list,
        observed_syndrome_bits,
        disorder_data_error_bits,
        disorder_syndrome_representative_bits):
    signatures = np.empty(len(chain_bits_list), dtype=np.int64)
    for temperature_index, chain_bits in enumerate(chain_bits_list):
        signatures[temperature_index] = _signature_for_state(
            context=context,
            current_chain_bits=chain_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            disorder_syndrome_representative_bits=(
                disorder_syndrome_representative_bits
            ),
            current_syndrome_term_bits=syndrome_term_bits_list[
                temperature_index
            ],
            observed_syndrome_bits=observed_syndrome_bits,
        )
    return signatures


def _attempt_zero_syndrome_support_update(
        current_chain_bits,
        current_data_term_bits,
        move_support_indices,
        log_odds_data,
        rng):
    support_size = int(move_support_indices.shape[0])
    if support_size == 0:
        return False, 0
    current_ones_on_support = int(
        np.count_nonzero(current_data_term_bits[move_support_indices])
    )
    delta_data_weight = support_size - 2 * current_ones_on_support
    log_acceptance = delta_data_weight * log_odds_data
    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))
    if not accepted:
        return False, 0
    current_chain_bits[move_support_indices] ^= True
    current_data_term_bits[move_support_indices] ^= True
    return True, int(delta_data_weight)


def _run_zero_syndrome_support_sweep(
        current_chain_bits,
        current_data_term_bits,
        move_supports,
        log_odds_data,
        rng):
    accepted_count = 0
    data_weight_delta = 0
    for move_index in rng.permutation(move_supports.shape[0]):
        accepted, accepted_data_weight_delta = (
            _attempt_zero_syndrome_support_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                move_support_indices=move_supports[int(move_index)],
                log_odds_data=log_odds_data,
                rng=rng,
            )
        )
        accepted_count += int(accepted)
        data_weight_delta += int(accepted_data_weight_delta)
    return accepted_count, int(data_weight_delta)


def _run_sparse_single_bit_sweep(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng,
        qubit_order_buffer,
        num_attempts):
    num_qubits = int(current_chain_bits.shape[0])
    num_attempts = max(0, min(int(num_attempts), num_qubits))
    accepted_count = 0
    data_weight_delta = 0
    for order_position in range(num_attempts):
        swap_index = int(rng.integers(order_position, num_qubits))
        temporary_value = qubit_order_buffer[order_position]
        qubit_order_buffer[order_position] = qubit_order_buffer[swap_index]
        qubit_order_buffer[swap_index] = temporary_value

        qubit_index = int(qubit_order_buffer[order_position])
        touched_checks = checks_touching_each_qubit[qubit_index]
        if current_data_term_bits[qubit_index]:
            delta_data_weight = -1
        else:
            delta_data_weight = 1

        delta_syndrome_weight = 0
        for check_index in touched_checks:
            if current_syndrome_term_bits[check_index]:
                delta_syndrome_weight -= 1
            else:
                delta_syndrome_weight += 1

        log_acceptance = (
            delta_data_weight * log_odds_data
            + delta_syndrome_weight * log_odds_syndrome
        )
        if log_acceptance >= 0.0:
            accepted = True
        else:
            accepted = bool(rng.random() < np.exp(log_acceptance))
        if not accepted:
            continue

        current_chain_bits[qubit_index] ^= True
        current_data_term_bits[qubit_index] ^= True
        current_syndrome_term_bits[touched_checks] ^= True
        accepted_count += 1
        data_weight_delta += delta_data_weight
    return accepted_count, int(data_weight_delta)


def _attempt_profiled_replica_swaps(
        chain_bits_list,
        data_term_bits_list,
        syndrome_term_bits_list,
        data_weight_per_temperature,
        log_odds_data_per_temperature,
        replica_id_per_temperature,
        rng,
        parity_index,
        swap_accept_counts,
        swap_attempt_counts):
    num_temperatures = len(chain_bits_list)
    accepted_count = 0
    attempted_count = 0
    offset = parity_index % 2
    for i in range(offset, num_temperatures - 1, 2):
        j = i + 1
        log_ratio = (
            (log_odds_data_per_temperature[j] - log_odds_data_per_temperature[i])
            * (int(data_weight_per_temperature[i]) - int(data_weight_per_temperature[j]))
        )
        if log_ratio >= 0.0:
            accepted = True
        else:
            accepted = bool(rng.random() < np.exp(log_ratio))
        swap_attempt_counts[i] += 1
        attempted_count += 1
        if not accepted:
            continue
        swap_accept_counts[i] += 1
        accepted_count += 1
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
        replica_id_per_temperature[i], replica_id_per_temperature[j] = (
            replica_id_per_temperature[j],
            replica_id_per_temperature[i],
        )
    return attempted_count, accepted_count


def _update_pt_transport(transport, replica_id_per_temperature, signatures):
    if signatures.shape[0] < 2:
        return
    hot_temperature = signatures.shape[0] - 1
    hot_replica = int(replica_id_per_temperature[hot_temperature])
    cold_replica = int(replica_id_per_temperature[0])
    hot_signature = int(signatures[hot_temperature])
    cold_signature = int(signatures[0])

    transport["last_hot_signature_by_replica"][hot_replica] = hot_signature
    transport["visited_hot_since_cold"][hot_replica] = True
    transport["last_cold_signature_by_replica"][cold_replica] = cold_signature
    transport["visited_cold_since_hot"][cold_replica] = True

    if transport["visited_hot_since_cold"][cold_replica]:
        transport["hot_to_cold_round_trip_count"] += 1
        transport["visited_hot_since_cold"][cold_replica] = False
        delivered_signature = int(
            transport["last_hot_signature_by_replica"][cold_replica]
        )
        if (
                delivered_signature >= 0
                and delivered_signature == cold_signature
                and delivered_signature
                != int(transport["last_delivered_hot_signature"][cold_replica])):
            transport["hot_to_cold_sector_delivery_count"] += 1
            transport["last_delivered_hot_signature"][cold_replica] = (
                delivered_signature
            )

    if transport["visited_cold_since_hot"][hot_replica]:
        transport["cold_to_hot_trip_count"] += 1
        transport["visited_cold_since_hot"][hot_replica] = False


def _build_pt_transport(num_temperatures):
    return {
        "visited_hot_since_cold": np.zeros(num_temperatures, dtype=bool),
        "visited_cold_since_hot": np.zeros(num_temperatures, dtype=bool),
        "last_hot_signature_by_replica": np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        ),
        "last_cold_signature_by_replica": np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        ),
        "last_delivered_hot_signature": np.full(
            num_temperatures,
            -1,
            dtype=np.int64,
        ),
        "hot_to_cold_round_trip_count": 0,
        "cold_to_hot_trip_count": 0,
        "hot_to_cold_sector_delivery_count": 0,
    }


def _summarize_cluster_result(result):
    if not result.get("attempted", False):
        return {
            "attempted": 0,
            "nonzero": 0,
            "move_fraction": 0.0,
            "skipped_for_budget": 0,
        }
    return {
        "attempted": 1,
        "nonzero": int(bool(result.get("nonzero", False))),
        "move_fraction": float(result.get("move_fraction", 0.0)),
        "skipped_for_budget": int(bool(result.get("skipped_for_budget", False))),
    }


def _run_profile_chain(
        context,
        observed_syndrome_bits,
        disorder_data_error_bits,
        disorder_syndrome_representative_bits,
        initial_chain_bits,
        task,
        chain_seed):
    rng = np.random.default_rng(int(chain_seed))
    p_value = float(task["p_value"])
    q_value = float(task["q_value"])
    data_error_probability_ladder = np.asarray(
        task["data_error_probability_ladder"],
        dtype=np.float64,
    )
    log_odds_data_per_temperature = np.asarray(
        [_compute_log_odds(float(value)) for value in data_error_probability_ladder],
        dtype=np.float64,
    )
    log_odds_syndrome = _compute_log_odds(q_value)
    num_temperatures = int(data_error_probability_ladder.shape[0])
    num_measurements = int(task["num_measurements"])
    num_sweeps_between_measurements = int(task["num_sweeps_between_measurements"])
    num_burn_in_sweeps = int(task["num_burn_in_sweeps"])
    num_zero_syndrome_sweeps_per_cycle = int(
        task["num_zero_syndrome_sweeps_per_cycle"]
    )
    winding_repeat_factor = int(task["winding_repeat_factor"])
    swap_attempt_every_num_sweeps = int(task["pt_swap_attempt_every_num_sweeps"])
    single_bit_proposal_fraction = float(
        task.get("single_bit_proposal_fraction", 1.0)
    )
    if not (0.0 < single_bit_proposal_fraction <= 1.0):
        raise ValueError("single_bit_proposal_fraction must be in (0, 1]")
    single_bit_attempts_per_temperature = max(
        1,
        int(math.ceil(single_bit_proposal_fraction * context["num_qubits"])),
    )
    observable_temperature_mode = str(
        task.get("observable_temperature_mode", "all")
    )
    if observable_temperature_mode not in ("all", "cold"):
        raise ValueError("observable_temperature_mode must be all or cold")
    stage_signature_mode = str(task.get("stage_signature_mode", "stage"))
    if stage_signature_mode not in ("stage", "none"):
        raise ValueError("stage_signature_mode must be stage or none")
    stage_signature_enabled = stage_signature_mode == "stage"

    chain_bits_list = []
    data_term_bits_list = []
    syndrome_term_bits_list = []
    data_weight_per_temperature = np.empty(num_temperatures, dtype=np.int64)
    for temperature_index in range(num_temperatures):
        (
            current_chain_bits,
            current_data_term_bits,
            current_syndrome_term_bits,
        ) = initialize_mcmc_state(
            num_qubits=context["num_qubits"],
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            parity_check_matrix=context["parity_check_matrix"],
            rng=rng,
            initial_chain_bits=initial_chain_bits,
        )
        chain_bits_list.append(current_chain_bits)
        data_term_bits_list.append(current_data_term_bits)
        syndrome_term_bits_list.append(current_syndrome_term_bits)
        data_weight_per_temperature[temperature_index] = np.count_nonzero(
            current_data_term_bits
        )

    cluster_controller = build_cluster_controller(
        parity_check_matrix=context["parity_check_matrix"],
        syndrome_error_probability=q_value,
        data_error_probability_ladder=data_error_probability_ladder,
        enabled=bool(task["cluster_update_enabled"]),
        budget_fraction_rho=float(task["cluster_budget_fraction_rho"]),
        debug_assertions=bool(task["cluster_update_debug"]),
    )

    logical_trace_per_temperature = np.empty(
        (
            num_temperatures,
            num_measurements,
            context["num_masks"],
        ),
        dtype=np.int8,
    )
    temperature_signature_trace = np.empty(
        (num_temperatures, num_measurements),
        dtype=np.int64,
    )
    replica_id_trace = np.empty(
        (num_temperatures, num_measurements),
        dtype=np.int64,
    )
    logical_observable_sum_per_temperature = np.zeros(
        (num_temperatures, context["num_masks"]),
        dtype=np.int64,
    )

    stage_profile = _new_stage_profile()
    cluster_attempt_count = 0
    cluster_nonzero_count = 0
    cluster_skipped_for_budget_count = 0
    cluster_sector_change_count = 0
    cluster_move_fraction_sum = 0.0
    swap_accept_counts = np.zeros(max(num_temperatures - 1, 0), dtype=np.int64)
    swap_attempt_counts = np.zeros(max(num_temperatures - 1, 0), dtype=np.int64)
    replica_id_per_temperature = np.arange(num_temperatures, dtype=np.int64)
    transport = _build_pt_transport(num_temperatures)
    qubit_order_buffer_per_temperature = [
        np.arange(context["num_qubits"], dtype=np.int32)
        for _ in range(num_temperatures)
    ]
    contractible_order_buffer_per_temperature = [
        np.arange(context["contractible_move_supports"].shape[0], dtype=np.int32)
        for _ in range(num_temperatures)
    ]
    winding_order_buffer_per_temperature = [
        np.arange(context["winding_move_supports"].shape[0], dtype=np.int32)
        for _ in range(num_temperatures)
    ]
    use_numba_split_stages = (
        NUMBA_AVAILABLE
        and context["checks_touching_each_qubit_array"].shape
        == (context["num_qubits"], 4)
        and isinstance(context["contractible_move_supports"], np.ndarray)
        and isinstance(context["winding_move_supports"], np.ndarray)
        and context["contractible_move_supports"].ndim == 2
        and context["winding_move_supports"].ndim == 2
    )

    total_signature_probe_time = 0.0
    sweep_counter = 0
    swap_parity_counter = 0
    chain_started_at = time.perf_counter()

    def _signature(temperature_index):
        nonlocal total_signature_probe_time
        started_at = time.perf_counter()
        signature = _signature_for_state(
            context=context,
            current_chain_bits=chain_bits_list[temperature_index],
            disorder_data_error_bits=disorder_data_error_bits,
            disorder_syndrome_representative_bits=(
                disorder_syndrome_representative_bits
            ),
            current_syndrome_term_bits=syndrome_term_bits_list[
                temperature_index
            ],
            observed_syndrome_bits=observed_syndrome_bits,
        )
        total_signature_probe_time += time.perf_counter() - started_at
        return signature

    def _all_signatures():
        nonlocal total_signature_probe_time
        started_at = time.perf_counter()
        signatures = _signatures_for_temperatures(
            context=context,
            chain_bits_list=chain_bits_list,
            syndrome_term_bits_list=syndrome_term_bits_list,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            disorder_syndrome_representative_bits=(
                disorder_syndrome_representative_bits
            ),
        )
        total_signature_probe_time += time.perf_counter() - started_at
        return signatures

    def _run_one_cycle_all_temperatures():
        nonlocal cluster_attempt_count
        nonlocal cluster_nonzero_count
        nonlocal cluster_skipped_for_budget_count
        nonlocal cluster_sector_change_count
        nonlocal cluster_move_fraction_sum
        ordinary_elapsed_per_temperature = np.zeros(
            num_temperatures,
            dtype=np.float64,
        )

        for temperature_index in range(num_temperatures):
            before_signature = (
                _signature(temperature_index)
                if stage_signature_enabled
                else None
            )
            started_at = time.perf_counter()
            if use_numba_split_stages:
                random_seed = int(
                    rng.integers(0, np.iinfo(np.int32).max, dtype=np.int32)
                )
                if single_bit_attempts_per_temperature >= context["num_qubits"]:
                    single_bit_accepted_count, single_bit_data_weight_delta = (
                        _numba_single_bit_stage_3d(
                            current_chain_bits=(
                                chain_bits_list[temperature_index]
                            ),
                            current_data_term_bits=(
                                data_term_bits_list[temperature_index]
                            ),
                            current_syndrome_term_bits=(
                                syndrome_term_bits_list[temperature_index]
                            ),
                            checks_touching_each_qubit_array=(
                                context["checks_touching_each_qubit_array"]
                            ),
                            qubit_order_buffer=(
                                qubit_order_buffer_per_temperature[
                                    temperature_index
                                ]
                            ),
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            log_odds_syndrome=log_odds_syndrome,
                            random_seed=random_seed,
                        )
                    )
                else:
                    single_bit_accepted_count, single_bit_data_weight_delta = (
                        _numba_single_bit_stage_3d_sparse(
                            current_chain_bits=(
                                chain_bits_list[temperature_index]
                            ),
                            current_data_term_bits=(
                                data_term_bits_list[temperature_index]
                            ),
                            current_syndrome_term_bits=(
                                syndrome_term_bits_list[temperature_index]
                            ),
                            checks_touching_each_qubit_array=(
                                context["checks_touching_each_qubit_array"]
                            ),
                            qubit_order_buffer=(
                                qubit_order_buffer_per_temperature[
                                    temperature_index
                                ]
                            ),
                            num_attempts=single_bit_attempts_per_temperature,
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            log_odds_syndrome=log_odds_syndrome,
                            random_seed=random_seed,
                        )
                    )
            else:
                if single_bit_attempts_per_temperature >= context["num_qubits"]:
                    single_bit_accepted_count, single_bit_data_weight_delta = (
                        _run_one_sweep_safe(
                            current_chain_bits=(
                                chain_bits_list[temperature_index]
                            ),
                            current_data_term_bits=(
                                data_term_bits_list[temperature_index]
                            ),
                            current_syndrome_term_bits=(
                                syndrome_term_bits_list[temperature_index]
                            ),
                            checks_touching_each_qubit=(
                                context["checks_touching_each_qubit"]
                            ),
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            log_odds_syndrome=log_odds_syndrome,
                            rng=rng,
                            qubit_order_buffer=(
                                qubit_order_buffer_per_temperature[
                                    temperature_index
                                ]
                            ),
                        )
                    )
                else:
                    single_bit_accepted_count, single_bit_data_weight_delta = (
                        _run_sparse_single_bit_sweep(
                            current_chain_bits=(
                                chain_bits_list[temperature_index]
                            ),
                            current_data_term_bits=(
                                data_term_bits_list[temperature_index]
                            ),
                            current_syndrome_term_bits=(
                                syndrome_term_bits_list[temperature_index]
                            ),
                            checks_touching_each_qubit=(
                                context["checks_touching_each_qubit"]
                            ),
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            log_odds_syndrome=log_odds_syndrome,
                            rng=rng,
                            qubit_order_buffer=(
                                qubit_order_buffer_per_temperature[
                                    temperature_index
                                ]
                            ),
                            num_attempts=single_bit_attempts_per_temperature,
                        )
                    )
            elapsed = time.perf_counter() - started_at
            data_weight_per_temperature[temperature_index] += (
                single_bit_data_weight_delta
            )
            after_signature = (
                _signature(temperature_index)
                if stage_signature_enabled
                else None
            )
            ordinary_elapsed_per_temperature[temperature_index] += elapsed
            _record_stage(
                stage_profile=stage_profile,
                stage_name="single_bit",
                wall_time=elapsed,
                attempted=single_bit_attempts_per_temperature,
                accepted=single_bit_accepted_count,
                sector_changed=(
                    stage_signature_enabled
                    and after_signature != before_signature
                ),
                data_weight_delta=single_bit_data_weight_delta,
            )

        for _zero_sweep_index in range(num_zero_syndrome_sweeps_per_cycle):
            for temperature_index in range(num_temperatures):
                before_signature = (
                    _signature(temperature_index)
                    if stage_signature_enabled
                    else None
                )
                started_at = time.perf_counter()
                if use_numba_split_stages:
                    random_seed = int(
                        rng.integers(0, np.iinfo(np.int32).max, dtype=np.int32)
                    )
                    accepted_count, data_weight_delta = (
                        _numba_zero_syndrome_stage_fixed_support(
                            current_chain_bits=chain_bits_list[
                                temperature_index
                            ],
                            current_data_term_bits=data_term_bits_list[
                                temperature_index
                            ],
                            move_supports=context["contractible_move_supports"],
                            order_buffer=(
                                contractible_order_buffer_per_temperature[
                                    temperature_index
                                ]
                            ),
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            random_seed=random_seed,
                        )
                    )
                else:
                    accepted_count, data_weight_delta = (
                        _run_zero_syndrome_support_sweep(
                        current_chain_bits=chain_bits_list[temperature_index],
                        current_data_term_bits=(
                            data_term_bits_list[temperature_index]
                        ),
                        move_supports=context["contractible_move_supports"],
                        log_odds_data=(
                            log_odds_data_per_temperature[temperature_index]
                        ),
                        rng=rng,
                    )
                    )
                elapsed = time.perf_counter() - started_at
                data_weight_per_temperature[temperature_index] += (
                    data_weight_delta
                )
                after_signature = (
                    _signature(temperature_index)
                    if stage_signature_enabled
                    else None
                )
                ordinary_elapsed_per_temperature[temperature_index] += elapsed
                _record_stage(
                    stage_profile=stage_profile,
                    stage_name="contractible",
                    wall_time=elapsed,
                    attempted=context["contractible_move_supports"].shape[0],
                    accepted=accepted_count,
                    sector_changed=(
                        stage_signature_enabled
                        and after_signature != before_signature
                    ),
                    data_weight_delta=data_weight_delta,
                )

            for _winding_repeat_index in range(winding_repeat_factor):
                for temperature_index in range(num_temperatures):
                    before_signature = (
                        _signature(temperature_index)
                        if stage_signature_enabled
                        else None
                    )
                    started_at = time.perf_counter()
                    if use_numba_split_stages:
                        random_seed = int(
                            rng.integers(
                                0,
                                np.iinfo(np.int32).max,
                                dtype=np.int32,
                            )
                        )
                        accepted_count, data_weight_delta = (
                            _numba_zero_syndrome_stage_fixed_support(
                                current_chain_bits=chain_bits_list[
                                    temperature_index
                                ],
                                current_data_term_bits=data_term_bits_list[
                                    temperature_index
                                ],
                                move_supports=context["winding_move_supports"],
                                order_buffer=(
                                    winding_order_buffer_per_temperature[
                                        temperature_index
                                    ]
                                ),
                                log_odds_data=(
                                    log_odds_data_per_temperature[
                                        temperature_index
                                    ]
                                ),
                                random_seed=random_seed,
                            )
                        )
                    else:
                        accepted_count, data_weight_delta = (
                            _run_zero_syndrome_support_sweep(
                            current_chain_bits=chain_bits_list[
                                temperature_index
                            ],
                            current_data_term_bits=data_term_bits_list[
                                temperature_index
                            ],
                            move_supports=context["winding_move_supports"],
                            log_odds_data=(
                                log_odds_data_per_temperature[temperature_index]
                            ),
                            rng=rng,
                        )
                        )
                    elapsed = time.perf_counter() - started_at
                    data_weight_per_temperature[temperature_index] += (
                        data_weight_delta
                    )
                    after_signature = (
                        _signature(temperature_index)
                        if stage_signature_enabled
                        else None
                    )
                    ordinary_elapsed_per_temperature[temperature_index] += elapsed
                    _record_stage(
                        stage_profile=stage_profile,
                        stage_name="winding",
                        wall_time=elapsed,
                        attempted=context["winding_move_supports"].shape[0],
                        accepted=accepted_count,
                        sector_changed=(
                            stage_signature_enabled
                            and after_signature != before_signature
                        ),
                        data_weight_delta=data_weight_delta,
                    )

        before_signatures = _all_signatures() if stage_signature_enabled else None
        started_at = time.perf_counter()
        cluster_result = maybe_run_cluster_update(
            controller=cluster_controller,
            chain_bits_list=chain_bits_list,
            data_term_bits_list=data_term_bits_list,
            syndrome_term_bits_list=syndrome_term_bits_list,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            checks_touching_each_qubit=context["checks_touching_each_qubit"],
            ordinary_elapsed_per_temperature=ordinary_elapsed_per_temperature,
            rng=rng,
        )
        elapsed = time.perf_counter() - started_at
        if cluster_result.get("attempted", False):
            temperature_index = int(cluster_result["temperature_index"])
            data_weight_per_temperature[temperature_index] += int(
                cluster_result.get("data_weight_delta", 0)
            )
        after_signatures = _all_signatures() if stage_signature_enabled else None
        cluster_summary = _summarize_cluster_result(cluster_result)
        cluster_attempt_count += cluster_summary["attempted"]
        cluster_nonzero_count += cluster_summary["nonzero"]
        cluster_skipped_for_budget_count += cluster_summary["skipped_for_budget"]
        cluster_move_fraction_sum += cluster_summary["move_fraction"]
        cluster_sector_changed = (
            stage_signature_enabled
            and bool(np.any(after_signatures != before_signatures))
        )
        cluster_sector_change_count += int(cluster_sector_changed)
        _record_stage(
            stage_profile=stage_profile,
            stage_name="cluster",
            wall_time=elapsed,
            attempted=cluster_summary["attempted"],
            accepted=cluster_summary["nonzero"],
            sector_changed=cluster_sector_changed,
            data_weight_delta=int(cluster_result.get("data_weight_delta", 0)),
        )

    def _maybe_attempt_swap():
        nonlocal swap_parity_counter
        if num_temperatures < 2:
            return
        if swap_attempt_every_num_sweeps <= 0:
            return
        if sweep_counter % swap_attempt_every_num_sweeps != 0:
            return
        before_signatures = _all_signatures() if stage_signature_enabled else None
        started_at = time.perf_counter()
        attempted_count, accepted_count = _attempt_profiled_replica_swaps(
            chain_bits_list=chain_bits_list,
            data_term_bits_list=data_term_bits_list,
            syndrome_term_bits_list=syndrome_term_bits_list,
            data_weight_per_temperature=data_weight_per_temperature,
            log_odds_data_per_temperature=log_odds_data_per_temperature,
            replica_id_per_temperature=replica_id_per_temperature,
            rng=rng,
            parity_index=swap_parity_counter,
            swap_accept_counts=swap_accept_counts,
            swap_attempt_counts=swap_attempt_counts,
        )
        elapsed = time.perf_counter() - started_at
        swap_parity_counter += 1
        after_signatures = _all_signatures() if stage_signature_enabled else None
        _record_stage(
            stage_profile=stage_profile,
            stage_name="pt_swap",
            wall_time=elapsed,
            attempted=attempted_count,
            accepted=accepted_count,
            sector_changed=(
                stage_signature_enabled
                and bool(np.any(after_signatures != before_signatures))
            ),
        )
        if stage_signature_enabled:
            _update_pt_transport(
                transport=transport,
                replica_id_per_temperature=replica_id_per_temperature,
                signatures=after_signatures,
            )

    for _burn_index in range(num_burn_in_sweeps):
        _run_one_cycle_all_temperatures()
        sweep_counter += 1
        _maybe_attempt_swap()

    if num_burn_in_sweeps > 0:
        freeze_cluster_controller(cluster_controller)

    for measurement_index in range(num_measurements):
        for _sweep_between_index in range(num_sweeps_between_measurements):
            if (
                    cluster_controller is not None
                    and cluster_controller["enabled"]
                    and num_burn_in_sweeps == 0):
                cluster_controller["production_used_adaptive"] = True
            _run_one_cycle_all_temperatures()
            sweep_counter += 1
            _maybe_attempt_swap()

        started_at = time.perf_counter()
        observable_temperature_indices = (
            range(num_temperatures)
            if observable_temperature_mode == "all"
            else range(1)
        )
        for temperature_index in observable_temperature_indices:
            logical_values = _logical_values_for_state(
                context=context,
                current_chain_bits=chain_bits_list[temperature_index],
                disorder_data_error_bits=disorder_data_error_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                current_syndrome_term_bits=syndrome_term_bits_list[
                    temperature_index
                ],
                observed_syndrome_bits=observed_syndrome_bits,
            )
            logical_trace_per_temperature[
                temperature_index,
                measurement_index,
            ] = logical_values
            logical_observable_sum_per_temperature[temperature_index] += (
                logical_values.astype(np.int64)
            )
            temperature_signature_trace[
                temperature_index,
                measurement_index,
            ] = _signature_from_logical_values(context, logical_values)
        if observable_temperature_mode == "cold" and num_temperatures > 1:
            cold_signature = int(temperature_signature_trace[0, measurement_index])
            temperature_signature_trace[1:, measurement_index] = cold_signature
        replica_id_trace[:, measurement_index] = replica_id_per_temperature
        elapsed = time.perf_counter() - started_at
        _record_stage(
            stage_profile=stage_profile,
            stage_name="observable",
            wall_time=elapsed,
        )

    total_wall_time = time.perf_counter() - chain_started_at
    m_u_values_per_temperature = (
        logical_observable_sum_per_temperature.astype(np.float64)
        / float(num_measurements)
    )
    q_top_value_per_temperature = np.mean(
        m_u_values_per_temperature ** 2,
        axis=1,
    )
    cold_logical_trace = logical_trace_per_temperature[0]
    tau_int_per_mask = np.asarray(
        [
            integrated_autocorrelation_time(cold_logical_trace[:, mask_index])
            for mask_index in range(context["num_masks"])
        ],
        dtype=np.float64,
    )
    max_tau_int = float(np.max(tau_int_per_mask))
    effective_sample_size = float(num_measurements / max(max_tau_int, 1.0))
    cold_signature_trace = temperature_signature_trace[0]
    first_sector_change_measurement = -1
    changed_positions = np.flatnonzero(
        cold_signature_trace != int(cold_signature_trace[0])
    )
    if changed_positions.size > 0:
        first_sector_change_measurement = int(changed_positions[0])
    cold_sector_flip_count = int(np.count_nonzero(np.diff(cold_signature_trace)))
    hot_sector_flip_count = 0
    if num_temperatures > 1:
        hot_sector_flip_count = int(np.count_nonzero(
            np.diff(temperature_signature_trace[-1])
        ))

    stage_total_wall_time = float(
        sum(entry["wall_time"] for entry in stage_profile.values())
    )
    instrumentation_wall_time = max(0.0, total_wall_time - stage_total_wall_time)
    for stage_name, entry in stage_profile.items():
        entry["acceptance_rate"] = _safe_rate(
            entry["accepted"],
            entry["attempted"],
        )
        entry["wall_fraction_of_stages"] = _safe_rate(
            entry["wall_time"],
            stage_total_wall_time,
        )

    if swap_attempt_counts.size == 0:
        swap_acceptance_rates = np.empty(0, dtype=np.float64)
    else:
        swap_acceptance_rates = np.zeros_like(
            swap_attempt_counts,
            dtype=np.float64,
        )
        mask = swap_attempt_counts > 0
        swap_acceptance_rates[mask] = (
            swap_accept_counts[mask].astype(np.float64)
            / swap_attempt_counts[mask].astype(np.float64)
        )

    cluster_controller_summary = summarize_cluster_controller(cluster_controller)
    cluster_nonzero_rate = _safe_rate(cluster_nonzero_count, cluster_attempt_count)
    cluster_mean_move_fraction = _safe_rate(
        cluster_move_fraction_sum,
        cluster_attempt_count,
    )

    summary = {
        "p_value": p_value,
        "q_value": q_value,
        "data_error_probability_ladder": data_error_probability_ladder,
        "m_u_values_per_temperature": m_u_values_per_temperature,
        "q_top_value_per_temperature": q_top_value_per_temperature,
        "cold_m_u_values": m_u_values_per_temperature[0],
        "cold_q_top_value": float(q_top_value_per_temperature[0]),
        "tau_int_per_mask": tau_int_per_mask,
        "max_tau_int": max_tau_int,
        "effective_sample_size": effective_sample_size,
        "ess_per_total_second": _safe_rate(effective_sample_size, total_wall_time),
        "ess_per_stage_second": _safe_rate(
            effective_sample_size,
            stage_total_wall_time,
        ),
        "cold_signature_histogram": np.bincount(
            cold_signature_trace,
            minlength=1 << context["num_logical_qubits"],
        ),
        "first_sector_change_measurement": int(first_sector_change_measurement),
        "cold_sector_flip_count": cold_sector_flip_count,
        "hot_sector_flip_count": hot_sector_flip_count,
        "stage_profile": stage_profile,
        "stage_total_wall_time": stage_total_wall_time,
        "total_wall_time": float(total_wall_time),
        "signature_probe_wall_time": float(total_signature_probe_time),
        "instrumentation_wall_time": float(instrumentation_wall_time),
        "swap_accept_counts": swap_accept_counts,
        "swap_attempt_counts": swap_attempt_counts,
        "swap_acceptance_rates": swap_acceptance_rates,
        "pt_transport": {
            "hot_to_cold_round_trip_count": int(
                transport["hot_to_cold_round_trip_count"]
            ),
            "cold_to_hot_trip_count": int(transport["cold_to_hot_trip_count"]),
            "hot_to_cold_sector_delivery_count": int(
                transport["hot_to_cold_sector_delivery_count"]
            ),
        },
        "cluster_profile": {
            "attempt_count": int(cluster_attempt_count),
            "nonzero_count": int(cluster_nonzero_count),
            "nonzero_rate": float(cluster_nonzero_rate),
            "skipped_for_budget_count": int(cluster_skipped_for_budget_count),
            "mean_move_fraction": float(cluster_mean_move_fraction),
            "sector_change_count": int(cluster_sector_change_count),
        },
        "cluster_controller_summary": cluster_controller_summary,
        "numba_split_stages_enabled": bool(use_numba_split_stages),
        "single_bit_proposal_fraction": float(single_bit_proposal_fraction),
        "single_bit_attempts_per_temperature": int(
            single_bit_attempts_per_temperature
        ),
        "observable_temperature_mode": observable_temperature_mode,
        "stage_signature_mode": stage_signature_mode,
        "stage_signature_instrumentation_enabled": bool(stage_signature_enabled),
    }
    raw = {
        "logical_trace_per_temperature": logical_trace_per_temperature,
        "temperature_signature_trace": temperature_signature_trace,
        "replica_id_trace": replica_id_trace,
    }
    return summary, raw


def _task_raw_stem(task):
    return (
        f"L{int(task['lattice_size']):02d}_"
        f"p{_probability_tag(task['p_value'])}_"
        f"{_sanitize_label(task['config_label'])}_"
        f"d{int(task['disorder_index']):03d}"
    )


def _summarize_task_chains(context, task, chain_summaries):
    num_chains = len(chain_summaries)
    num_measurements = int(task["num_measurements"])
    chain_m_u_values = np.asarray(
        [summary["cold_m_u_values"] for summary in chain_summaries],
        dtype=np.float64,
    )
    chain_q_top_values = np.asarray(
        [summary["cold_q_top_value"] for summary in chain_summaries],
        dtype=np.float64,
    )
    chain_ess_values = np.asarray(
        [summary["effective_sample_size"] for summary in chain_summaries],
        dtype=np.float64,
    )
    chain_ess_per_second = np.asarray(
        [summary["ess_per_total_second"] for summary in chain_summaries],
        dtype=np.float64,
    )
    logical_trace_tensor = np.asarray(
        [
            summary["_cold_logical_trace"]
            for summary in chain_summaries
        ],
        dtype=np.int8,
    )
    r_hat_per_mask = aggregate_r_hat(logical_trace_tensor)
    finite_r_hat = r_hat_per_mask[np.isfinite(r_hat_per_mask)]
    max_r_hat = float(np.max(finite_r_hat)) if finite_r_hat.size else np.nan

    if num_chains < 2:
        m_u_spread_linf = 0.0
        q_top_spread = 0.0
    else:
        pairwise_m_u_diff = np.abs(
            chain_m_u_values[:, None, :]
            - chain_m_u_values[None, :, :]
        )
        m_u_spread_linf = float(np.max(pairwise_m_u_diff))
        q_top_spread = float(np.max(chain_q_top_values) - np.min(chain_q_top_values))

    stage_totals = _new_stage_profile()
    for summary in chain_summaries:
        for stage_name in STAGE_NAMES:
            source = summary["stage_profile"][stage_name]
            target = stage_totals[stage_name]
            for key in ("wall_time", "attempted", "accepted", "sector_changes", "data_weight_delta"):
                target[key] += source[key]
    stage_total_wall_time = float(
        sum(entry["wall_time"] for entry in stage_totals.values())
    )
    for stage_name, entry in stage_totals.items():
        entry["acceptance_rate"] = _safe_rate(
            entry["accepted"],
            entry["attempted"],
        )
        entry["wall_fraction_of_stages"] = _safe_rate(
            entry["wall_time"],
            stage_total_wall_time,
        )

    swap_accept_counts = np.sum(
        [
            np.asarray(summary["swap_accept_counts"], dtype=np.int64)
            for summary in chain_summaries
        ],
        axis=0,
    )
    swap_attempt_counts = np.sum(
        [
            np.asarray(summary["swap_attempt_counts"], dtype=np.int64)
            for summary in chain_summaries
        ],
        axis=0,
    )
    swap_acceptance_rates = np.zeros_like(swap_attempt_counts, dtype=np.float64)
    swap_mask = swap_attempt_counts > 0
    swap_acceptance_rates[swap_mask] = (
        swap_accept_counts[swap_mask].astype(np.float64)
        / swap_attempt_counts[swap_mask].astype(np.float64)
    )
    first_sector_change_measurements = np.asarray(
        [
            summary["first_sector_change_measurement"]
            for summary in chain_summaries
        ],
        dtype=np.int64,
    )
    cluster_attempts = int(sum(
        summary["cluster_profile"]["attempt_count"]
        for summary in chain_summaries
    ))
    cluster_nonzero = int(sum(
        summary["cluster_profile"]["nonzero_count"]
        for summary in chain_summaries
    ))
    cluster_move_fraction_sum = float(sum(
        summary["cluster_profile"]["mean_move_fraction"]
        * summary["cluster_profile"]["attempt_count"]
        for summary in chain_summaries
    ))
    total_sweeps_per_chain = (
        int(task["num_burn_in_sweeps"])
        + int(task["num_measurements"])
        * int(task["num_sweeps_between_measurements"])
    )
    winding_sector_changes_per_1000_sweeps = _safe_rate(
        1000.0 * stage_totals["winding"]["sector_changes"],
        total_sweeps_per_chain * max(num_chains, 1),
    )
    task_summary = {
        "task_id": task["task_id"],
        "config_label": task["config_label"],
        "lattice_size": int(task["lattice_size"]),
        "p_value": float(task["p_value"]),
        "q_value": float(task["q_value"]),
        "disorder_index": int(task["disorder_index"]),
        "num_chains": int(num_chains),
        "num_measurements": int(num_measurements),
        "mean_q_top": float(np.mean(chain_q_top_values)),
        "q_top_spread": q_top_spread,
        "m_u_spread_linf": m_u_spread_linf,
        "max_r_hat": max_r_hat,
        "r_hat_per_mask": r_hat_per_mask,
        "min_effective_sample_size": float(np.min(chain_ess_values)),
        "mean_effective_sample_size": float(np.mean(chain_ess_values)),
        "mean_ess_per_total_second": float(np.mean(chain_ess_per_second)),
        "num_chains_that_never_flipped_sector": int(
            np.count_nonzero(first_sector_change_measurements == -1)
        ),
        "stage_totals": stage_totals,
        "stage_total_wall_time": stage_total_wall_time,
        "total_wall_time": float(sum(
            summary["total_wall_time"] for summary in chain_summaries
        )),
        "signature_probe_wall_time": float(sum(
            summary["signature_probe_wall_time"] for summary in chain_summaries
        )),
        "instrumentation_wall_time": float(sum(
            summary["instrumentation_wall_time"] for summary in chain_summaries
        )),
        "swap_accept_counts": swap_accept_counts,
        "swap_attempt_counts": swap_attempt_counts,
        "swap_acceptance_rates": swap_acceptance_rates,
        "hot_to_cold_sector_delivery_count": int(sum(
            summary["pt_transport"]["hot_to_cold_sector_delivery_count"]
            for summary in chain_summaries
        )),
        "hot_to_cold_round_trip_count": int(sum(
            summary["pt_transport"]["hot_to_cold_round_trip_count"]
            for summary in chain_summaries
        )),
        "cold_to_hot_trip_count": int(sum(
            summary["pt_transport"]["cold_to_hot_trip_count"]
            for summary in chain_summaries
        )),
        "cold_sector_flip_count": int(sum(
            summary["cold_sector_flip_count"] for summary in chain_summaries
        )),
        "hot_sector_flip_count": int(sum(
            summary["hot_sector_flip_count"] for summary in chain_summaries
        )),
        "cluster_attempt_count": cluster_attempts,
        "cluster_nonzero_count": cluster_nonzero,
        "cluster_nonzero_rate": _safe_rate(cluster_nonzero, cluster_attempts),
        "cluster_mean_move_fraction": _safe_rate(
            cluster_move_fraction_sum,
            cluster_attempts,
        ),
        "cluster_sector_change_count": int(sum(
            summary["cluster_profile"]["sector_change_count"]
            for summary in chain_summaries
        )),
        "winding_sector_changes_per_1000_sweeps": (
            winding_sector_changes_per_1000_sweeps
        ),
        "chain_q_top_values": chain_q_top_values,
        "chain_m_u_values": chain_m_u_values,
        "first_sector_change_measurements": (
            first_sector_change_measurements
        ),
    }
    return task_summary


def _run_profile_task(task):
    task_started_at = time.perf_counter()
    raw_dir = Path(task["raw_dir"])
    json_dir = Path(task["json_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    context = _build_context(task["lattice_size"])
    observed_syndrome_bits, disorder_data_error_bits = _draw_common_disorder(
        context=context,
        q_value=task["q_value"],
        p_value=task["p_value"],
        disorder_seed=task["disorder_seed"],
    )
    disorder_syndrome_representative_bits = (
        _compute_disorder_syndrome_representative(
            context=context,
            disorder_data_error_bits=disorder_data_error_bits,
        )
    )
    initial_chain_bits_per_start, start_sector_labels = (
        _build_q0_initial_chain_bits_per_start(
            observed_syndrome_bits=np.zeros(context["num_checks"], dtype=bool),
            section_data=context["section_data"],
            zero_syndrome_move_data=context["zero_syndrome_move_data"],
            q0_num_start_chains=int(task["num_start_chains"]),
        )
    )

    chain_summaries = []
    chain_logical_traces = []
    chain_temperature_signature_traces = []
    chain_replica_id_traces = []
    for start_index in range(int(task["num_start_chains"])):
        for replica_index in range(int(task["num_replicas_per_start"])):
            chain_seed = int(
                task["chain_seed_base"]
                + 1009 * start_index
                + 10007 * replica_index
            )
            chain_summary, chain_raw = _run_profile_chain(
                context=context,
                observed_syndrome_bits=observed_syndrome_bits,
                disorder_data_error_bits=disorder_data_error_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                initial_chain_bits=initial_chain_bits_per_start[start_index],
                task=task,
                chain_seed=chain_seed,
            )
            chain_summary["start_index"] = int(start_index)
            chain_summary["start_sector_label"] = str(
                start_sector_labels[start_index]
            )
            chain_summary["replica_index"] = int(replica_index)
            chain_summary["_cold_logical_trace"] = (
                chain_raw["logical_trace_per_temperature"][0]
            )
            chain_summaries.append(chain_summary)
            chain_logical_traces.append(chain_raw["logical_trace_per_temperature"])
            chain_temperature_signature_traces.append(
                chain_raw["temperature_signature_trace"]
            )
            chain_replica_id_traces.append(chain_raw["replica_id_trace"])

    task_summary = _summarize_task_chains(
        context=context,
        task=task,
        chain_summaries=chain_summaries,
    )
    task_summary["duration_seconds"] = float(time.perf_counter() - task_started_at)
    task_summary["start_sector_labels"] = np.asarray(start_sector_labels)
    task_summary["data_error_probability_ladder"] = np.asarray(
        task["data_error_probability_ladder"],
        dtype=np.float64,
    )

    raw_stem = _task_raw_stem(task)
    raw_npz_path = raw_dir / f"{raw_stem}.npz"
    raw_json_path = json_dir / f"{raw_stem}.json"
    public_chain_summaries = []
    for summary in chain_summaries:
        public_summary = {
            key: value
            for key, value in summary.items()
            if key != "_cold_logical_trace"
        }
        public_chain_summaries.append(public_summary)

    np.savez_compressed(
        raw_npz_path,
        logical_trace_per_chain=np.asarray(chain_logical_traces, dtype=np.int8),
        temperature_signature_trace_per_chain=np.asarray(
            chain_temperature_signature_traces,
            dtype=np.int64,
        ),
        replica_id_trace_per_chain=np.asarray(
            chain_replica_id_traces,
            dtype=np.int64,
        ),
        chain_q_top_values=task_summary["chain_q_top_values"],
        chain_m_u_values=task_summary["chain_m_u_values"],
        start_sector_labels=np.asarray(start_sector_labels),
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        data_error_probability_ladder=np.asarray(
            task["data_error_probability_ladder"],
            dtype=np.float64,
        ),
        lattice_size=np.int64(task["lattice_size"]),
        p_value=np.float64(task["p_value"]),
        q_value=np.float64(task["q_value"]),
        disorder_index=np.int64(task["disorder_index"]),
        num_burn_in_sweeps=np.int64(task["num_burn_in_sweeps"]),
        num_measurements=np.int64(task["num_measurements"]),
        num_sweeps_between_measurements=np.int64(
            task["num_sweeps_between_measurements"]
        ),
    )
    _write_json(
        raw_json_path,
        {
            "task": task,
            "task_summary": task_summary,
            "chain_summaries": public_chain_summaries,
            "raw_npz_path": str(raw_npz_path),
        },
    )
    task_summary["raw_npz_path"] = str(raw_npz_path)
    task_summary["raw_json_path"] = str(raw_json_path)
    return task_summary


def _base_config():
    return {
        "config_label": "base_PT7_hot0p44_cluster_rho0p05",
        "pt_num_temperatures": 7,
        "pt_p_hot": 0.44,
        "pt_swap_attempt_every_num_sweeps": 1,
        "cluster_update_enabled": True,
        "cluster_budget_fraction_rho": 0.05,
        "num_zero_syndrome_sweeps_per_cycle": 1,
        "winding_repeat_factor": 1,
        "num_start_chains": 4,
        "num_replicas_per_start": 1,
        "single_bit_proposal_fraction": 1.0,
        "observable_temperature_mode": "all",
        "stage_signature_mode": "stage",
    }


def _config_with(label, **updates):
    config = _base_config()
    config.update(updates)
    config["config_label"] = label
    return config


def _build_experiment_points(args):
    q_value = float(args.q)
    if args.suite == "smoke":
        lattice_sizes = _parse_int_csv(args.lattice_sizes or "4")
        p_values = _parse_float_csv(args.p_values or "0.26")
        config = _config_with(
            "smoke_PT3_hot0p44_cluster_rho0p05",
            pt_num_temperatures=3,
            pt_p_hot=0.44,
            num_start_chains=max(1, int(args.num_start_chains or 2)),
            num_replicas_per_start=max(1, int(args.num_replicas_per_start or 1)),
        )
        return [
            {
                "lattice_size": lattice_size,
                "p_value": p_value,
                "q_value": q_value,
                "config": config,
                "num_disorders": int(args.num_disorders or 1),
            }
            for lattice_size in lattice_sizes
            for p_value in p_values
        ]

    if args.suite == "calibration":
        return [
            {
                "lattice_size": 4,
                "p_value": 0.26,
                "q_value": q_value,
                "config": _config_with(
                    "calibration_base_PT5_hot0p44_cluster_rho0p05",
                    pt_num_temperatures=5,
                    num_start_chains=2,
                    num_replicas_per_start=1,
                ),
                "num_disorders": int(args.num_disorders or 1),
            },
            {
                "lattice_size": 4,
                "p_value": 0.26,
                "q_value": q_value,
                "config": _config_with(
                    "calibration_no_PT",
                    pt_num_temperatures=1,
                    pt_p_hot=None,
                    num_start_chains=2,
                    num_replicas_per_start=1,
                ),
                "num_disorders": int(args.num_disorders or 1),
            },
        ]

    if args.suite == "optimization":
        p_values = _parse_float_csv(args.p_values or "0.20")
        lattice_sizes = _parse_int_csv(args.lattice_sizes or "4,5")
        num_disorders = int(args.num_disorders or 2)
        if args.l5_num_disorders is not None:
            l5_num_disorders = int(args.l5_num_disorders)
        else:
            l5_num_disorders = num_disorders
        optimization_configs = (
            _config_with(
                "opt_no_cluster_PT7_full_single_coldobs",
                cluster_update_enabled=False,
                single_bit_proposal_fraction=1.0,
                observable_temperature_mode="cold",
                stage_signature_mode=args.stage_signature_mode,
            ),
            _config_with(
                "opt_no_cluster_PT7_single_0p25_coldobs",
                cluster_update_enabled=False,
                single_bit_proposal_fraction=0.25,
                observable_temperature_mode="cold",
                stage_signature_mode=args.stage_signature_mode,
            ),
            _config_with(
                "opt_no_cluster_PT7_single_0p10_coldobs",
                cluster_update_enabled=False,
                single_bit_proposal_fraction=0.10,
                observable_temperature_mode="cold",
                stage_signature_mode=args.stage_signature_mode,
            ),
            _config_with(
                "opt_no_cluster_PT7_single_0p05_coldobs",
                cluster_update_enabled=False,
                single_bit_proposal_fraction=0.05,
                observable_temperature_mode="cold",
                stage_signature_mode=args.stage_signature_mode,
            ),
            _config_with(
                "opt_noPT_no_cluster_full_single",
                pt_num_temperatures=1,
                pt_p_hot=None,
                cluster_update_enabled=False,
                single_bit_proposal_fraction=1.0,
                observable_temperature_mode="all",
                stage_signature_mode=args.stage_signature_mode,
            ),
            _config_with(
                "opt_noPT_no_cluster_single_0p10",
                pt_num_temperatures=1,
                pt_p_hot=None,
                cluster_update_enabled=False,
                single_bit_proposal_fraction=0.10,
                observable_temperature_mode="all",
                stage_signature_mode=args.stage_signature_mode,
            ),
        )
        if args.config_labels:
            requested_labels = {
                token.strip()
                for token in args.config_labels.split(",")
                if token.strip()
            }
            optimization_configs = tuple(
                config for config in optimization_configs
                if config["config_label"] in requested_labels
            )
        return [
            {
                "lattice_size": lattice_size,
                "p_value": p_value,
                "q_value": q_value,
                "config": dict(config),
                "num_disorders": (
                    l5_num_disorders if int(lattice_size) == 5 else num_disorders
                ),
            }
            for lattice_size in lattice_sizes
            for p_value in p_values
            for config in optimization_configs
        ]

    points = []
    seen = set()

    def add(lattice_size, p_value, config, num_disorders):
        key = (
            int(lattice_size),
            float(p_value),
            str(config["config_label"]),
            int(num_disorders),
        )
        if key in seen:
            return
        seen.add(key)
        points.append({
            "lattice_size": int(lattice_size),
            "p_value": float(p_value),
            "q_value": q_value,
            "config": dict(config),
            "num_disorders": int(num_disorders),
        })

    main_configs = (
        _base_config(),
        _config_with("no_PT", pt_num_temperatures=1, pt_p_hot=None),
        _config_with("cluster_off", cluster_update_enabled=False),
        _config_with("winding_repeat_4", winding_repeat_factor=4),
    )
    for p_value in (0.22, 0.26, 0.30):
        for config in main_configs:
            add(4, p_value, config, int(args.num_disorders or 3))

    sensitivity_configs = [
        _config_with("PT_K5_hot0p44_swap1", pt_num_temperatures=5),
        _config_with("PT_K7_hot0p44_swap1", pt_num_temperatures=7),
        _config_with("PT_K9_hot0p44_swap1", pt_num_temperatures=9),
        _config_with("PT_K7_hot0p36_swap1", pt_p_hot=0.36),
        _config_with("PT_K7_hot0p48_swap1", pt_p_hot=0.48),
        _config_with("PT_K7_hot0p44_swap4", pt_swap_attempt_every_num_sweeps=4),
        _config_with("zero_sweeps_4_winding_repeat_1", num_zero_syndrome_sweeps_per_cycle=4),
        _config_with("zero_sweeps_1_winding_repeat_4", winding_repeat_factor=4),
        _config_with("zero_sweeps_4_winding_repeat_4", num_zero_syndrome_sweeps_per_cycle=4, winding_repeat_factor=4),
        _config_with("cluster_disabled_sensitivity", cluster_update_enabled=False),
        _config_with("cluster_rho0p05", cluster_budget_fraction_rho=0.05),
        _config_with("cluster_rho0p15", cluster_budget_fraction_rho=0.15),
        _config_with("starts4_reps1", num_start_chains=4, num_replicas_per_start=1),
        _config_with("starts8_reps1", num_start_chains=8, num_replicas_per_start=1),
        _config_with("starts4_reps2", num_start_chains=4, num_replicas_per_start=2),
    ]
    for config in sensitivity_configs:
        add(4, 0.26, config, int(args.num_disorders or 3))

    l5_configs = (
        _base_config(),
        _config_with("L5_no_PT", pt_num_temperatures=1, pt_p_hot=None),
        _config_with("L5_cluster_off", cluster_update_enabled=False),
        _config_with("L5_winding_repeat_4", winding_repeat_factor=4),
        _config_with("L5_PT_K9", pt_num_temperatures=9),
        _config_with("L5_hot0p48", pt_p_hot=0.48),
        _config_with("L5_cluster_rho0p15", cluster_budget_fraction_rho=0.15),
    )
    for config in l5_configs:
        add(5, 0.26, config, int(args.l5_num_disorders or 2))

    if args.lattice_sizes:
        allowed_lattice_sizes = set(_parse_int_csv(args.lattice_sizes))
        points = [
            point for point in points
            if int(point["lattice_size"]) in allowed_lattice_sizes
        ]
    if args.p_values:
        allowed_p_values = set(_parse_float_csv(args.p_values))
        points = [
            point for point in points
            if float(point["p_value"]) in allowed_p_values
        ]
    if args.config_labels:
        requested_labels = {
            token.strip()
            for token in args.config_labels.split(",")
            if token.strip()
        }
        points = [
            point for point in points
            if point["config"]["config_label"] in requested_labels
        ]
    return points


def _build_tasks(args, run_root):
    points = _build_experiment_points(args)
    if not points:
        raise ValueError("empty profiling matrix after filters")
    raw_dir = Path(run_root) / "raw_npz"
    json_dir = Path(run_root) / "raw_json"
    tasks = []
    task_index = 0
    for point_index, point in enumerate(points):
        config = dict(point["config"])
        p_value = float(point["p_value"])
        if int(config["pt_num_temperatures"]) <= 1:
            ladder = np.asarray([p_value], dtype=np.float64)
        else:
            ladder = equal_log_odds_ladder(
                p_cold=p_value,
                p_hot=float(config["pt_p_hot"]),
                num_temperatures=int(config["pt_num_temperatures"]),
            )
        for disorder_index in range(int(point["num_disorders"])):
            p_seed_component = int(round(p_value * 1_000_000))
            q_seed_component = int(round(float(point["q_value"]) * 1_000_000))
            disorder_seed_offset = (
                1000003 * int(point["lattice_size"])
                + 37 * p_seed_component
                + 104729 * q_seed_component
                + 9176 * disorder_index
            )
            chain_seed_offset = 1000003 * point_index + 9176 * disorder_index
            task_id = (
                f"task{task_index:04d}_"
                f"L{int(point['lattice_size']):02d}_"
                f"p{_probability_tag(point['p_value'])}_"
                f"{_sanitize_label(config['config_label'])}_"
                f"d{disorder_index:03d}"
            )
            tasks.append({
                "task_id": task_id,
                "task_index": int(task_index),
                "matrix_point_index": int(point_index),
                "code_family": "3d_toric",
                "lattice_size": int(point["lattice_size"]),
                "p_value": p_value,
                "q_value": float(point["q_value"]),
                "disorder_index": int(disorder_index),
                "config_label": str(config["config_label"]),
                "data_error_probability_ladder": ladder,
                "pt_num_temperatures": int(config["pt_num_temperatures"]),
                "pt_p_hot": (
                    None if config["pt_p_hot"] is None else float(config["pt_p_hot"])
                ),
                "pt_swap_attempt_every_num_sweeps": int(
                    config["pt_swap_attempt_every_num_sweeps"]
                ),
                "cluster_update_enabled": bool(config["cluster_update_enabled"]),
                "cluster_budget_fraction_rho": float(
                    config["cluster_budget_fraction_rho"]
                ),
                "cluster_update_debug": bool(args.cluster_debug_assertions),
                "num_zero_syndrome_sweeps_per_cycle": int(
                    config["num_zero_syndrome_sweeps_per_cycle"]
                ),
                "winding_repeat_factor": int(config["winding_repeat_factor"]),
                "num_start_chains": int(config["num_start_chains"]),
                "num_replicas_per_start": int(config["num_replicas_per_start"]),
                "single_bit_proposal_fraction": float(
                    config.get("single_bit_proposal_fraction", 1.0)
                ),
                "observable_temperature_mode": str(
                    config.get("observable_temperature_mode", "all")
                ),
                "stage_signature_mode": str(
                    config.get("stage_signature_mode", "stage")
                ),
                "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
                "num_measurements": int(args.num_measurements),
                "num_sweeps_between_measurements": int(
                    args.num_sweeps_between_measurements
                ),
                "seed_base": int(args.seed_base),
                "disorder_seed_scope": (
                    "lattice_size,p_value,q_value,disorder_index"
                ),
                "disorder_seed": int(args.seed_base + disorder_seed_offset + 19),
                "chain_seed_base": int(args.seed_base + chain_seed_offset + 101),
                "raw_dir": str(raw_dir),
                "json_dir": str(json_dir),
            })
            task_index += 1
    tasks.sort(
        key=lambda task: (
            -int(task["lattice_size"]),
            int(task["matrix_point_index"]),
            int(task["disorder_index"]),
        )
    )
    return tasks


def _compute_worker_count(requested_workers, num_tasks):
    if num_tasks <= 0:
        return 0
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(int(requested_workers), int(cpu_count), int(num_tasks)))


def _multiprocessing_context():
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _resolve_git_commit_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _run_tasks(tasks, workers, max_wall_seconds):
    if workers <= 1:
        results = []
        started_at = time.monotonic()
        skipped_tasks = []
        for task in tasks:
            if (
                    max_wall_seconds is not None
                    and time.monotonic() - started_at >= max_wall_seconds):
                skipped_tasks.append(task)
                continue
            _log(f"Starting {task['task_id']}")
            result = _run_profile_task(task)
            results.append(result)
            _log(
                f"Completed {task['task_id']} "
                f"in {result['duration_seconds']:.1f}s"
            )
        return results, skipped_tasks

    started_at = time.monotonic()
    results = []
    skipped_tasks = []
    next_task_index = 0
    futures = {}
    context = _multiprocessing_context()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        while next_task_index < len(tasks) and len(futures) < workers:
            task = tasks[next_task_index]
            _log(f"Submitting {task['task_id']}")
            futures[executor.submit(_run_profile_task, task)] = task
            next_task_index += 1

        while futures:
            done, _not_done = wait(
                futures,
                return_when=FIRST_COMPLETED,
                timeout=5.0,
            )
            if not done:
                if (
                        max_wall_seconds is not None
                        and time.monotonic() - started_at >= max_wall_seconds):
                    skipped_tasks.extend(tasks[next_task_index:])
                    next_task_index = len(tasks)
                continue
            for future in done:
                task = futures.pop(future)
                result = future.result()
                results.append(result)
                _log(
                    f"Completed {task['task_id']} "
                    f"({len(results)}/{len(tasks)}) "
                    f"in {result['duration_seconds']:.1f}s"
                )
            while next_task_index < len(tasks) and len(futures) < workers:
                if (
                        max_wall_seconds is not None
                        and time.monotonic() - started_at >= max_wall_seconds):
                    skipped_tasks.extend(tasks[next_task_index:])
                    next_task_index = len(tasks)
                    break
                task = tasks[next_task_index]
                _log(f"Submitting {task['task_id']}")
                futures[executor.submit(_run_profile_task, task)] = task
                next_task_index += 1
    return results, skipped_tasks


def _aggregate_group(group_results):
    stage_totals = _new_stage_profile()
    for result in group_results:
        for stage_name in STAGE_NAMES:
            source = result["stage_totals"][stage_name]
            target = stage_totals[stage_name]
            for key in ("wall_time", "attempted", "accepted", "sector_changes", "data_weight_delta"):
                target[key] += source[key]
    stage_total_wall_time = float(
        sum(entry["wall_time"] for entry in stage_totals.values())
    )
    for stage_name, entry in stage_totals.items():
        entry["acceptance_rate"] = _safe_rate(
            entry["accepted"],
            entry["attempted"],
        )
        entry["wall_fraction_of_stages"] = _safe_rate(
            entry["wall_time"],
            stage_total_wall_time,
        )
    q_top_values = np.asarray(
        [result["mean_q_top"] for result in group_results],
        dtype=np.float64,
    )
    ess_values = np.asarray(
        [result["mean_ess_per_total_second"] for result in group_results],
        dtype=np.float64,
    )
    swap_accept_counts = np.sum(
        [np.asarray(result["swap_accept_counts"], dtype=np.int64) for result in group_results],
        axis=0,
    )
    swap_attempt_counts = np.sum(
        [np.asarray(result["swap_attempt_counts"], dtype=np.int64) for result in group_results],
        axis=0,
    )
    swap_acceptance_rates = np.zeros_like(swap_attempt_counts, dtype=np.float64)
    swap_mask = swap_attempt_counts > 0
    swap_acceptance_rates[swap_mask] = (
        swap_accept_counts[swap_mask].astype(np.float64)
        / swap_attempt_counts[swap_mask].astype(np.float64)
    )
    cluster_attempts = int(sum(result["cluster_attempt_count"] for result in group_results))
    cluster_nonzero = int(sum(result["cluster_nonzero_count"] for result in group_results))
    cluster_move_fraction_sum = float(sum(
        result["cluster_mean_move_fraction"] * result["cluster_attempt_count"]
        for result in group_results
    ))
    total_wall_time = float(sum(result["total_wall_time"] for result in group_results))
    summary = {
        "config_label": group_results[0]["config_label"],
        "lattice_size": int(group_results[0]["lattice_size"]),
        "p_value": float(group_results[0]["p_value"]),
        "q_value": float(group_results[0]["q_value"]),
        "num_disorders_completed": int(len(group_results)),
        "mean_q_top": float(np.mean(q_top_values)),
        "std_q_top_across_disorders": (
            0.0 if q_top_values.size < 2 else float(np.std(q_top_values, ddof=1))
        ),
        "mean_q_top_spread": float(np.mean([
            result["q_top_spread"] for result in group_results
        ])),
        "max_q_top_spread": float(np.max([
            result["q_top_spread"] for result in group_results
        ])),
        "mean_m_u_spread_linf": float(np.mean([
            result["m_u_spread_linf"] for result in group_results
        ])),
        "max_r_hat": float(np.nanmax([
            result["max_r_hat"] for result in group_results
        ])),
        "min_effective_sample_size": float(np.min([
            result["min_effective_sample_size"] for result in group_results
        ])),
        "mean_ess_per_total_second": float(np.mean(ess_values)),
        "stage_totals": stage_totals,
        "stage_total_wall_time": stage_total_wall_time,
        "total_wall_time": total_wall_time,
        "signature_probe_wall_time": float(sum(
            result["signature_probe_wall_time"] for result in group_results
        )),
        "instrumentation_wall_time": float(sum(
            result["instrumentation_wall_time"] for result in group_results
        )),
        "swap_accept_counts": swap_accept_counts,
        "swap_attempt_counts": swap_attempt_counts,
        "swap_acceptance_rates": swap_acceptance_rates,
        "hot_to_cold_sector_delivery_count": int(sum(
            result["hot_to_cold_sector_delivery_count"]
            for result in group_results
        )),
        "hot_to_cold_round_trip_count": int(sum(
            result["hot_to_cold_round_trip_count"]
            for result in group_results
        )),
        "cold_to_hot_trip_count": int(sum(
            result["cold_to_hot_trip_count"]
            for result in group_results
        )),
        "cold_sector_flip_count": int(sum(
            result["cold_sector_flip_count"] for result in group_results
        )),
        "hot_sector_flip_count": int(sum(
            result["hot_sector_flip_count"] for result in group_results
        )),
        "num_chains_that_never_flipped_sector": int(sum(
            result["num_chains_that_never_flipped_sector"]
            for result in group_results
        )),
        "cluster_attempt_count": cluster_attempts,
        "cluster_nonzero_count": cluster_nonzero,
        "cluster_nonzero_rate": _safe_rate(cluster_nonzero, cluster_attempts),
        "cluster_mean_move_fraction": _safe_rate(
            cluster_move_fraction_sum,
            cluster_attempts,
        ),
        "cluster_sector_change_count": int(sum(
            result["cluster_sector_change_count"] for result in group_results
        )),
        "winding_sector_changes_per_1000_sweeps": float(np.mean([
            result["winding_sector_changes_per_1000_sweeps"]
            for result in group_results
        ])),
        "raw_npz_paths": [result["raw_npz_path"] for result in group_results],
        "raw_json_paths": [result["raw_json_path"] for result in group_results],
    }
    return summary


def _aggregate_results(args, run_root, tasks, results, skipped_tasks, started_at):
    groups = {}
    for result in results:
        key = (
            int(result["lattice_size"]),
            float(result["p_value"]),
            str(result["config_label"]),
        )
        groups.setdefault(key, []).append(result)
    group_summaries = [
        _aggregate_group(group_results)
        for _key, group_results in sorted(groups.items())
    ]

    stage_rankings = {}
    for group in group_summaries:
        sorted_stages = sorted(
            STAGE_NAMES,
            key=lambda stage_name: group["stage_totals"][stage_name]["wall_time"],
            reverse=True,
        )
        stage_rankings[
            f"L{group['lattice_size']}_p{_probability_tag(group['p_value'])}_{group['config_label']}"
        ] = [
            {
                "stage": stage_name,
                "wall_time": group["stage_totals"][stage_name]["wall_time"],
                "wall_fraction_of_stages": group["stage_totals"][stage_name][
                    "wall_fraction_of_stages"
                ],
            }
            for stage_name in sorted_stages
        ]

    summary = {
        "created_at": _timestamp(),
        "started_at": started_at,
        "completed_at": _timestamp(),
        "hostname": socket.gethostname(),
        "git_commit_sha": _resolve_git_commit_sha(),
        "suite": args.suite,
        "code_family": args.code_family,
        "q_value": float(args.q),
        "num_tasks": int(len(tasks)),
        "num_completed_tasks": int(len(results)),
        "num_skipped_tasks": int(len(skipped_tasks)),
        "tasks": tasks,
        "skipped_task_ids": [task["task_id"] for task in skipped_tasks],
        "group_summaries": group_summaries,
        "stage_rankings": stage_rankings,
        "analysis_criteria": {
            "pt_effective_if_cold_ess_or_flip_improves": ">=1.5x versus no_PT",
            "cluster_effective_if_ess_or_spread_improves": ">=1.3x versus cluster_off",
            "winding_report_unit": "accepted/sector-changing winding stage changes per 1000 sweeps",
        },
    }
    _write_json(Path(run_root) / "profile_summary.json", summary)
    _write_markdown_summary(Path(run_root) / "profile_summary.md", summary)
    return summary


def _format_float(value, digits=4):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def _write_markdown_summary(path, summary):
    lines = []
    lines.append("# 3D q>0 profiling summary")
    lines.append("")
    lines.append(f"- created_at: `{summary['created_at']}`")
    lines.append(f"- suite: `{summary['suite']}`")
    lines.append(f"- q: `{summary['q_value']}`")
    lines.append(
        f"- completed/skipped tasks: `{summary['num_completed_tasks']}` / `{summary['num_skipped_tasks']}`"
    )
    lines.append(f"- git_commit_sha: `{summary['git_commit_sha']}`")
    lines.append("")
    lines.append("## Config summaries")
    lines.append("")
    lines.append(
        "| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |"
    )
    lines.append(
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for group in summary["group_summaries"]:
        top_stage = max(
            STAGE_NAMES,
            key=lambda stage_name: group["stage_totals"][stage_name]["wall_time"],
        )
        top_stage_fraction = group["stage_totals"][top_stage][
            "wall_fraction_of_stages"
        ]
        lines.append(
            "| "
            f"{group['lattice_size']} | "
            f"{group['p_value']:.4f} | "
            f"`{group['config_label']}` | "
            f"{group['num_disorders_completed']} | "
            f"{_format_float(group['mean_q_top'])} | "
            f"{_format_float(group['mean_ess_per_total_second'])} | "
            f"{_format_float(group['max_r_hat'])} | "
            f"{_format_float(group['mean_q_top_spread'])} | "
            f"{_format_float(group['mean_m_u_spread_linf'])} | "
            f"{group['cold_sector_flip_count']} | "
            f"{group['hot_sector_flip_count']} | "
            f"{group['hot_to_cold_sector_delivery_count']} | "
            f"{_format_float(group['cluster_nonzero_rate'])} | "
            f"{top_stage} ({_format_float(top_stage_fraction)}) |"
        )
    lines.append("")
    lines.append("## Stage Wall-Time Rankings")
    lines.append("")
    for key, ranking in summary["stage_rankings"].items():
        lines.append(f"### {key}")
        for item in ranking:
            lines.append(
                f"- {item['stage']}: "
                f"{_format_float(item['wall_time'])}s, "
                f"fraction={_format_float(item['wall_fraction_of_stages'])}"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Opt-in 3D toric q>0 profiling runner.",
    )
    parser.add_argument("--code-family", default="3d_toric")
    parser.add_argument("--q", type=float, default=DEFAULT_Q_VALUE)
    parser.add_argument(
        "--suite",
        choices=("default", "calibration", "smoke", "optimization"),
        default="default",
    )
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--lattice-sizes", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--config-labels", default=None)
    parser.add_argument("--num-disorders", type=int, default=None)
    parser.add_argument("--l5-num-disorders", type=int, default=None)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=256)
    parser.add_argument("--num-measurements", type=int, default=768)
    parser.add_argument("--num-sweeps-between-measurements", type=int, default=4)
    parser.add_argument("--num-start-chains", type=int, default=None)
    parser.add_argument("--num-replicas-per-start", type=int, default=None)
    parser.add_argument(
        "--stage-signature-mode",
        choices=("stage", "none"),
        default="stage",
        help=(
            "stage records logical-sector changes before/after every profiled "
            "stage; none disables those expensive probes."
        ),
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--cluster-debug-assertions", action="store_true")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.code_family != "3d_toric":
        raise ValueError("--code-family must be 3d_toric for this profiler")
    if float(args.q) <= 0.0:
        raise ValueError("--q must be > 0 for this profiler")
    if args.num_burn_in_sweeps < 0:
        raise ValueError("--num-burn-in-sweeps must be >= 0")
    if args.num_measurements < 1:
        raise ValueError("--num-measurements must be >= 1")
    if args.num_sweeps_between_measurements < 1:
        raise ValueError("--num-sweeps-between-measurements must be >= 1")

    if args.run_root is None:
        run_root = DEFAULT_RUN_ROOT / f"{args.suite}_{_timestamp_tag()}"
    else:
        run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()

    tasks = _build_tasks(args, run_root)
    workers = _compute_worker_count(args.workers, len(tasks))
    manifest = {
        "created_at": started_at,
        "hostname": socket.gethostname(),
        "git_commit_sha": _resolve_git_commit_sha(),
        "run_root": str(run_root),
        "suite": args.suite,
        "code_family": args.code_family,
        "q_value": float(args.q),
        "workers": int(workers),
        "max_wall_seconds": args.max_wall_seconds,
        "num_tasks": len(tasks),
        "tasks": tasks,
    }
    _write_json(run_root / "profile_manifest.json", manifest)
    _log(f"Profile run_root={run_root}")
    _log(f"Task count={len(tasks)} workers={workers}")
    results, skipped_tasks = _run_tasks(
        tasks=tasks,
        workers=workers,
        max_wall_seconds=args.max_wall_seconds,
    )
    summary = _aggregate_results(
        args=args,
        run_root=run_root,
        tasks=tasks,
        results=results,
        skipped_tasks=skipped_tasks,
        started_at=started_at,
    )
    _log(f"Wrote {run_root / 'profile_summary.json'}")
    _log(f"Wrote {run_root / 'profile_summary.md'}")
    if summary["num_completed_tasks"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
