"""No-extra-randomness transport trace for the exhausted exp102 PT-v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from .discovery import (
    DISCOVERY_RAW_VERSION,
    load_discovery_config,
    validate_discovery_raw,
)
from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .labels import bits_to_uint64
from .ladders import q0_config_from_candidate, validate_pt_candidate
from .q0_pa import PaConflictError
from .q0_pt import (
    _acceptance_tables,
    _supports_to_csr,
    coupling_ladder,
)
from .registry import load_frozen_code, load_registry
from .worker import build_model

try:
    from numba import njit
except ImportError:  # pragma: no cover - autopsy preflight requires Numba
    njit = None


AUTOPSY_VERSION = "exp102.transport_autopsy.v1"
AUTOPSY_RAW_VERSION = "exp102.transport_autopsy.raw.v1"
AUTOPSY_REPORT_VERSION = "exp102.transport_autopsy.report.v1"
AUTOPSY_TASKS_VERSION = "exp102.transport_autopsy.tasks.v1"
AUTOPSY_NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
PARENT_SOURCE_COMMIT = "da69528b43f4a9d1635083c21d713ba63ccec4ab"
PARENT_RUN_ID = "exp102_discovery_v2_20260720_da69528"
PARENT_TRANSPORT_CONTROL = "5480511a57d1"
AUTOPSY_LADDERS = ("D0", "D4")
AUTOPSY_CELLS = (
    {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
)
PARENT_RELATIVE_PATHS = {
    ("D0", "m06_c00"): f"transport/{PARENT_TRANSPORT_CONTROL}/nd-3/D0_S64_b2000_m8000/m06_c00/p0.04_d00.npz",
    ("D0", "m08_c06"): f"transport/{PARENT_TRANSPORT_CONTROL}/nd-2/D0_S64_b2000_m8000/m08_c06/p0.04_d00.npz",
    ("D4", "m06_c00"): f"transport/{PARENT_TRANSPORT_CONTROL}/nd-3/D4_S64_b2000_m8000/m06_c00/p0.04_d00.npz",
    ("D4", "m08_c06"): f"transport/{PARENT_TRANSPORT_CONTROL}/nd-1/D4_S64_b2000_m8000/m08_c06/p0.04_d00.npz",
}

AUTOPSY_RAW_FIELDS = {
    "raw_version", "autopsy_version", "task_fingerprint", "source_commit",
    "parent_source_commit", "parent_raw_sha256", "parent_task_fingerprint",
    "parent_relative_path", "registry_sha256", "discovery_config_sha256",
    "cell_json", "candidate_json", "ladder_id", "uniform_seed",
    "instance_seeds", "model_fingerprint", "section_fingerprint",
    "logical_frame_fingerprint", "labels", "swap_attempts", "swap_accepts",
    "logical_attempts", "logical_accepts", "hot_touches",
    "hot_updated_visits", "uncertified_round_trips", "round_trips",
    "sector_changing_round_trips", "hot_touches_per_replica",
    "hot_updated_visits_per_replica", "uncertified_round_trips_per_replica",
    "round_trips_per_replica", "sector_changing_round_trips_per_replica",
    "max_hard_coset_residual", "replica_at_rung_by_round",
    "replica_weight_by_round", "replica_label_by_round",
    "replica_phase_by_round", "round_min_rung_by_replica",
    "round_max_rung_by_replica", "direction_by_round",
    "endpoint_events_by_round", "edge_attempts_by_phase_direction",
    "edge_accepts_by_phase_direction", "first_hot_touch_round",
    "first_hot_update_round", "first_uncertified_return_round",
    "first_certified_return_round", "frontier_max_rung", "rung_churn",
    "direction_reversals", "post_hot_return_records",
    "post_hot_return_record_count", "classification", "classification_json",
    "core_seconds", "wall_seconds", "engine",
}


if njit is not None:
    @njit(cache=True, inline="always")
    def _nb_next_uint64(state):
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << np.uint64(23))
        x = x ^ (x >> np.uint64(17))
        x = x ^ y ^ (y >> np.uint64(26))
        state[1] = x
        return x + y


    @njit(cache=True, inline="always")
    def _nb_random(state):
        return float(_nb_next_uint64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


    @njit(cache=True, inline="always")
    def _nb_fill_permutation(state, buffer):
        for index in range(buffer.size):
            buffer[index] = index
        for index in range(buffer.size - 1, 0, -1):
            selected = int(_nb_next_uint64(state) % np.uint64(index + 1))
            temporary = buffer[index]
            buffer[index] = buffer[selected]
            buffer[selected] = temporary


    @njit(cache=True)
    def _run_autopsy_core(
        states,
        weights,
        state_labels,
        stabilizer_indices,
        stabilizer_offsets,
        logical_indices,
        logical_offsets,
        check_indices,
        check_offsets,
        syndrome,
        move_acceptance,
        swap_acceptance,
        rng_state,
        burn_rounds,
        measurement_rounds,
        sweeps_per_round,
        logical_move_repeat,
        swap_sweeps_per_round,
    ):
        temperatures = states.shape[0]
        num_stabilizers = stabilizer_offsets.size - 1
        num_logicals = logical_offsets.size - 1
        logical_attempts = np.zeros((temperatures, num_logicals), dtype=np.int64)
        logical_accepts = np.zeros((temperatures, num_logicals), dtype=np.int64)
        swap_attempts = np.zeros(temperatures - 1, dtype=np.int64)
        swap_accepts = np.zeros(temperatures - 1, dtype=np.int64)
        labels = np.zeros(measurement_rounds, dtype=np.uint64)
        replica_at = np.arange(temperatures, dtype=np.int64)
        phase = np.zeros(temperatures, dtype=np.int8)
        arrival_hot_label = np.zeros(temperatures, dtype=np.uint64)
        hot_updated_label = np.zeros(temperatures, dtype=np.uint64)
        hot_touches = np.zeros(temperatures, dtype=np.int64)
        hot_updates = np.zeros(temperatures, dtype=np.int64)
        uncertified_returns = np.zeros(temperatures, dtype=np.int64)
        certified_returns = np.zeros(temperatures, dtype=np.int64)
        changing_returns = np.zeros(temperatures, dtype=np.int64)
        stabilizer_order = np.empty(num_stabilizers, dtype=np.int64)
        logical_order = np.empty(num_logicals, dtype=np.int64)
        parity = 0

        replica_at_trace = np.empty((measurement_rounds, temperatures), dtype=np.uint16)
        weight_trace = np.empty((measurement_rounds, temperatures), dtype=np.int32)
        label_trace = np.empty((measurement_rounds, temperatures), dtype=np.uint64)
        phase_trace = np.empty((measurement_rounds, temperatures), dtype=np.int8)
        min_rung_trace = np.empty((measurement_rounds, temperatures), dtype=np.uint16)
        max_rung_trace = np.empty((measurement_rounds, temperatures), dtype=np.uint16)
        direction_trace = np.empty((measurement_rounds, temperatures), dtype=np.int8)
        event_trace = np.zeros((measurement_rounds, temperatures), dtype=np.uint8)
        conditional_attempts = np.zeros((4, 2, temperatures - 1), dtype=np.int64)
        conditional_accepts = np.zeros((4, 2, temperatures - 1), dtype=np.int64)
        first_hot_touch = np.full(temperatures, -1, dtype=np.int64)
        first_hot_update = np.full(temperatures, -1, dtype=np.int64)
        first_uncertified_return = np.full(temperatures, -1, dtype=np.int64)
        first_certified_return = np.full(temperatures, -1, dtype=np.int64)
        frontier_max = np.zeros(temperatures, dtype=np.int64)
        rung_churn = np.zeros(temperatures, dtype=np.int64)
        direction_reversals = np.zeros(temperatures, dtype=np.int64)
        previous_step_direction = np.zeros(temperatures, dtype=np.int8)
        last_hot_update_round = np.full(temperatures, -1, dtype=np.int64)
        return_records = np.full((measurement_rounds, 5), -1, dtype=np.int64)
        return_record_count = 0
        rung_of = np.empty(temperatures, dtype=np.int64)
        round_min = np.empty(temperatures, dtype=np.int64)
        round_max = np.empty(temperatures, dtype=np.int64)

        total_rounds = burn_rounds + measurement_rounds
        for round_index in range(total_rounds):
            measure = round_index >= burn_rounds
            measurement_index = round_index - burn_rounds
            if measure:
                for rung in range(temperatures):
                    replica = replica_at[rung]
                    rung_of[replica] = rung
                    round_min[replica] = rung
                    round_max[replica] = rung
                    if rung > frontier_max[replica]:
                        frontier_max[replica] = rung

            for rung in range(temperatures):
                replica = replica_at[rung]
                for _ in range(sweeps_per_round):
                    _nb_fill_permutation(rng_state, stabilizer_order)
                    for order_index in range(num_stabilizers):
                        move = stabilizer_order[order_index]
                        start = stabilizer_offsets[move]
                        stop = stabilizer_offsets[move + 1]
                        ones = 0
                        for position in range(start, stop):
                            ones += int(states[replica, stabilizer_indices[position]])
                        delta = (stop - start) - 2 * ones
                        uniform = _nb_random(rng_state)
                        if delta <= 0 or uniform < move_acceptance[rung, delta]:
                            for position in range(start, stop):
                                states[replica, stabilizer_indices[position]] ^= np.uint8(1)
                            weights[replica] += delta
                    for _ in range(logical_move_repeat):
                        _nb_fill_permutation(rng_state, logical_order)
                        for order_index in range(num_logicals):
                            bit = logical_order[order_index]
                            start = logical_offsets[bit]
                            stop = logical_offsets[bit + 1]
                            ones = 0
                            for position in range(start, stop):
                                ones += int(states[replica, logical_indices[position]])
                            delta = (stop - start) - 2 * ones
                            logical_attempts[rung, bit] += 1
                            uniform = _nb_random(rng_state)
                            if delta <= 0 or uniform < move_acceptance[rung, delta]:
                                logical_accepts[rung, bit] += 1
                                for position in range(start, stop):
                                    states[replica, logical_indices[position]] ^= np.uint8(1)
                                weights[replica] += delta
                                state_labels[replica] ^= np.uint64(1) << np.uint64(bit)

            if measure:
                hot_replica = replica_at[-1]
                if phase[hot_replica] == 2:
                    phase[hot_replica] = 3
                    hot_updates[hot_replica] += 1
                    hot_updated_label[hot_replica] = state_labels[hot_replica]
                    event_trace[measurement_index, hot_replica] |= np.uint8(2)
                    last_hot_update_round[hot_replica] = measurement_index
                    if first_hot_update[hot_replica] < 0:
                        first_hot_update[hot_replica] = measurement_index

            for _ in range(swap_sweeps_per_round):
                for edge in range(parity, temperatures - 1, 2):
                    swap_attempts[edge] += 1
                    left_replica = replica_at[edge]
                    right_replica = replica_at[edge + 1]
                    if measure:
                        conditional_attempts[phase[left_replica], 0, edge] += 1
                        conditional_attempts[phase[right_replica], 1, edge] += 1
                    difference = weights[left_replica] - weights[right_replica]
                    uniform = _nb_random(rng_state)
                    accepted = (
                        difference >= 0
                        or uniform < swap_acceptance[edge, -difference]
                    )
                    if accepted:
                        swap_accepts[edge] += 1
                        if measure:
                            conditional_accepts[phase[left_replica], 0, edge] += 1
                            conditional_accepts[phase[right_replica], 1, edge] += 1
                            for replica, old_rung, new_rung, step_direction in (
                                    (left_replica, edge, edge + 1, 1),
                                    (right_replica, edge + 1, edge, -1)):
                                rung_churn[replica] += 1
                                if (previous_step_direction[replica] != 0
                                        and previous_step_direction[replica] != step_direction):
                                    direction_reversals[replica] += 1
                                previous_step_direction[replica] = step_direction
                                rung_of[replica] = new_rung
                                if new_rung < round_min[replica]:
                                    round_min[replica] = new_rung
                                if new_rung > round_max[replica]:
                                    round_max[replica] = new_rung
                                if new_rung > frontier_max[replica]:
                                    frontier_max[replica] = new_rung
                        replica_at[edge] = right_replica
                        replica_at[edge + 1] = left_replica
                parity ^= 1

                if measure:
                    hot_replica = replica_at[-1]
                    if phase[hot_replica] == 1:
                        phase[hot_replica] = 2
                        hot_touches[hot_replica] += 1
                        arrival_hot_label[hot_replica] = state_labels[hot_replica]
                        event_trace[measurement_index, hot_replica] |= np.uint8(1)
                        if first_hot_touch[hot_replica] < 0:
                            first_hot_touch[hot_replica] = measurement_index

                    cold_replica = replica_at[0]
                    if phase[cold_replica] == 2:
                        uncertified_returns[cold_replica] += 1
                        event_trace[measurement_index, cold_replica] |= np.uint8(4)
                        if first_uncertified_return[cold_replica] < 0:
                            first_uncertified_return[cold_replica] = measurement_index
                        phase[cold_replica] = 1
                    elif phase[cold_replica] == 3:
                        certified_returns[cold_replica] += 1
                        changed = int(
                            state_labels[cold_replica] != arrival_hot_label[cold_replica]
                        )
                        changing_returns[cold_replica] += changed
                        event_trace[measurement_index, cold_replica] |= np.uint8(8)
                        if first_certified_return[cold_replica] < 0:
                            first_certified_return[cold_replica] = measurement_index
                        if return_record_count < measurement_rounds:
                            hot_round = last_hot_update_round[cold_replica]
                            return_records[return_record_count, 0] = cold_replica
                            return_records[return_record_count, 1] = hot_round
                            return_records[return_record_count, 2] = measurement_index
                            return_records[return_record_count, 3] = measurement_index - hot_round
                            return_records[return_record_count, 4] = changed
                            return_record_count += 1
                        phase[cold_replica] = 1
                    elif phase[cold_replica] == 0:
                        phase[cold_replica] = 1

            if measure:
                cold_replica = replica_at[0]
                labels[measurement_index] = state_labels[cold_replica]
                for rung in range(temperatures):
                    replica_at_trace[measurement_index, rung] = replica_at[rung]
                for replica in range(temperatures):
                    weight_trace[measurement_index, replica] = weights[replica]
                    label_trace[measurement_index, replica] = state_labels[replica]
                    phase_trace[measurement_index, replica] = phase[replica]
                    min_rung_trace[measurement_index, replica] = round_min[replica]
                    max_rung_trace[measurement_index, replica] = round_max[replica]
                    direction_trace[measurement_index, replica] = (
                        1 if phase[replica] == 1 else
                        -1 if phase[replica] == 2 or phase[replica] == 3 else 0
                    )
            elif round_index + 1 == burn_rounds:
                phase[:] = 0
                phase[replica_at[0]] = 1

        max_residual = 0
        for replica in range(temperatures):
            residual_weight = 0
            for check in range(check_offsets.size - 1):
                parity_bit = syndrome[check]
                for position in range(check_offsets[check], check_offsets[check + 1]):
                    parity_bit ^= states[replica, check_indices[position]]
                residual_weight += int(parity_bit)
            if residual_weight > max_residual:
                max_residual = residual_weight
        return (
            labels, swap_attempts, swap_accepts, logical_attempts, logical_accepts,
            hot_touches, hot_updates, uncertified_returns, certified_returns,
            changing_returns, max_residual, replica_at_trace, weight_trace,
            label_trace, phase_trace, min_rung_trace, max_rung_trace,
            direction_trace, event_trace, conditional_attempts, conditional_accepts,
            first_hot_touch, first_hot_update, first_uncertified_return,
            first_certified_return, frontier_max, rung_churn, direction_reversals,
            return_records, return_record_count,
        )
else:  # pragma: no cover
    _run_autopsy_core = None


def _default_parent_root():
    return Path(__file__).resolve().parents[1] / "raw" / "discovery" / PARENT_RUN_ID


def _resolve_parent_path(parent_root, relative):
    """Resolve the canonical server path or the deliberately flattened local cache."""
    root = Path(parent_root)
    relative = Path(relative)
    canonical = root / relative
    if canonical.is_file():
        return canonical
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("transport", PARENT_TRANSPORT_CONTROL):
        flattened = root / "transport" / Path(*parts[2:])
        if flattened.is_file():
            return flattened
    return canonical


def build_autopsy_config(registry_path, discovery_config_path, parent_root=None):
    registry = load_registry(registry_path)
    discovery = load_discovery_config(discovery_config_path, registry)
    root = _default_parent_root() if parent_root is None else Path(parent_root)
    parents = []
    for ladder_id in AUTOPSY_LADDERS:
        for cell in AUTOPSY_CELLS:
            relative = PARENT_RELATIVE_PATHS[(ladder_id, cell["code_id"])]
            path = _resolve_parent_path(root, relative)
            with np.load(path, allow_pickle=False) as data:
                if str(data["raw_version"].item()) != DISCOVERY_RAW_VERSION:
                    raise ValueError("autopsy parent raw version mismatch")
                candidate = validate_pt_candidate(json.loads(str(data["candidate_json"].item())))
                if (candidate["ladder_id"] != ladder_id
                        or candidate["swap_sweeps_per_round"] != 64
                        or candidate["burn_rounds"] != 2000
                        or candidate["measurement_rounds"] != 8000
                        or json.loads(str(data["cell_json"].item())) != cell):
                    raise ValueError("autopsy parent task differs from the frozen plan")
                parents.append({
                    "ladder_id": ladder_id,
                    "cell": dict(cell),
                    "parent_relative_path": relative,
                    "parent_raw_sha256": sha256_file(path),
                    "parent_task_fingerprint": str(data["task_fingerprint"].item()),
                    "candidate": candidate,
                    "uniform_seed": int(data["uniform_seed"].item()),
                    "instance_seeds": data["instance_seeds"].astype(np.int64).tolist(),
                })
    return {
        "autopsy_version": AUTOPSY_VERSION,
        "raw_version": AUTOPSY_RAW_VERSION,
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "parent_run_id": PARENT_RUN_ID,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": discovery["discovery_config_sha256"],
        "conditional_min_attempts": 200,
        "conditional_min_edge_rate": 0.05,
        "parents": parents,
    }


def write_autopsy_config(registry_path, discovery_config_path, output_path,
                          parent_root=None):
    config = build_autopsy_config(registry_path, discovery_config_path, parent_root)
    atomic_json(output_path, config)
    return config


def load_autopsy_config(path, registry=None, discovery=None):
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    expected_fields = {
        "autopsy_version", "raw_version", "parent_source_commit", "parent_run_id",
        "registry_sha256", "discovery_config_sha256", "conditional_min_attempts",
        "conditional_min_edge_rate", "parents",
    }
    if set(raw) != expected_fields or raw["autopsy_version"] != AUTOPSY_VERSION:
        raise ValueError("autopsy config schema/version mismatch")
    if (raw["raw_version"] != AUTOPSY_RAW_VERSION
            or raw["parent_source_commit"] != PARENT_SOURCE_COMMIT
            or raw["parent_run_id"] != PARENT_RUN_ID
            or raw["conditional_min_attempts"] != 200
            or raw["conditional_min_edge_rate"] != 0.05):
        raise ValueError("autopsy config differs from the frozen protocol")
    if registry is not None and raw["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("autopsy config registry mismatch")
    if (discovery is not None
            and raw["discovery_config_sha256"] != discovery["discovery_config_sha256"]):
        raise ValueError("autopsy config discovery identity mismatch")
    keys = []
    for parent in raw["parents"]:
        if set(parent) != {
                "ladder_id", "cell", "parent_relative_path", "parent_raw_sha256",
                "parent_task_fingerprint", "candidate", "uniform_seed", "instance_seeds"}:
            raise ValueError("autopsy parent config schema mismatch")
        candidate = validate_pt_candidate(parent["candidate"])
        if candidate != parent["candidate"] or candidate["ladder_id"] != parent["ladder_id"]:
            raise ValueError("autopsy candidate is noncanonical")
        if (re.fullmatch(r"[0-9a-f]{64}", parent["parent_raw_sha256"]) is None
                or re.fullmatch(r"[0-9a-f]{64}", parent["parent_task_fingerprint"]) is None
                or len(parent["instance_seeds"]) != 4):
            raise ValueError("autopsy parent hashes/seeds are invalid")
        keys.append((parent["ladder_id"], canonical_json(parent["cell"])))
    expected_keys = [
        (ladder, canonical_json(cell))
        for ladder in AUTOPSY_LADDERS for cell in AUTOPSY_CELLS
    ]
    if keys != expected_keys:
        raise ValueError("autopsy parent order differs from the frozen four tasks")
    return {**raw, "autopsy_config_sha256": sha256_json(raw),
            "config_path": str(Path(path).resolve())}


def autopsy_task_identity(config, source_commit, parent):
    if re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None:
        raise ValueError("autopsy source commit must be a full lowercase Git SHA")
    if parent not in config["parents"]:
        raise ValueError("autopsy parent is outside the frozen config")
    return {
        "raw_version": AUTOPSY_RAW_VERSION,
        "autopsy_version": AUTOPSY_VERSION,
        "source_commit": source_commit,
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "parent_raw_sha256": parent["parent_raw_sha256"],
        "parent_task_fingerprint": parent["parent_task_fingerprint"],
        "parent_relative_path": parent["parent_relative_path"],
        "ladder_id": parent["ladder_id"],
        "cell": parent["cell"],
        "candidate": parent["candidate"],
        "uniform_seed": parent["uniform_seed"],
        "instance_seeds": parent["instance_seeds"],
        "registry_sha256": config["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "autopsy_config_sha256": config["autopsy_config_sha256"],
        "engine": "numba_trace_no_extra_rng",
    }


def autopsy_tasks(config, source_commit):
    return [autopsy_task_identity(config, source_commit, parent)
            for parent in config["parents"]]


def write_autopsy_task_manifest(config_path, source_commit, output_path):
    config = load_autopsy_config(config_path)
    manifest = {
        "manifest_version": AUTOPSY_TASKS_VERSION,
        "source_commit": source_commit,
        "autopsy_config_sha256": config["autopsy_config_sha256"],
        "tasks": autopsy_tasks(config, source_commit),
    }
    atomic_json(output_path, manifest)
    return manifest


def autopsy_task_cost(task):
    candidate = task["candidate"]
    m = int(task["cell"]["code_id"][1:3])
    return float(
        m * m * candidate["num_temperatures"]
        * (candidate["burn_rounds"] + candidate["measurement_rounds"])
        * candidate["swap_sweeps_per_round"]
    )


def fixed_autopsy_ownership(tasks, nodes, source_commit, control_sha256):
    nodes = list(nodes)
    if (len(nodes) < 2 or len(nodes) != len(set(nodes))
            or not set(nodes) <= set(AUTOPSY_NODE_CAPACITY)):
        raise ValueError("autopsy ownership requires at least two known nodes")
    loads = {node: 0.0 for node in nodes}
    owners = {}
    for task in sorted(tasks, key=lambda value: (-autopsy_task_cost(value), sha256_json(value))):
        node = min(nodes, key=lambda value: (
            loads[value] / AUTOPSY_NODE_CAPACITY[value], value,
        ))
        fingerprint = sha256_json(task)
        owners[fingerprint] = node
        loads[node] += autopsy_task_cost(task)
    identity = {
        "source_commit": source_commit,
        "control_sha256": control_sha256,
        "nodes": nodes,
        "task_owner": owners,
    }
    return {
        "ownership_version": "exp102.transport_autopsy.ownership.v1",
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "weighted_load": loads,
        "capacity": {node: AUTOPSY_NODE_CAPACITY[node] for node in nodes},
    }


def _run_trace_instance(model, frame, syndrome, p_cold, candidate, seed, initial_label):
    from exp101_certified_src.prng import PortablePrng
    from exp101_certified_src.reference_mcmc import _logical_supports, _stab_supports

    config = q0_config_from_candidate(candidate)
    K, _ = coupling_ladder(
        p_cold, config.p_hot, config.num_temperatures, config.gamma,
        config.ladder_x_q32,
    )
    base = model.logical_sector_section.apply(syndrome, strict=True)
    vector = base.copy()
    for bit in range(model.k):
        if (int(initial_label) >> bit) & 1:
            vector ^= model.logical_move_basis[bit]
    states = np.repeat(vector[None, :], config.num_temperatures, axis=0)
    states = np.ascontiguousarray(states, dtype=np.uint8)
    weights = np.full(config.num_temperatures, int(vector.sum()), dtype=np.int64)
    initial_state_label = bits_to_uint64(frame.label_of(vector))
    state_labels = np.full(config.num_temperatures, initial_state_label, dtype=np.uint64)
    stabilizers = _stab_supports(model)
    logicals = _logical_supports(model)
    check_supports = [np.flatnonzero(row).astype(np.int64) for row in model.H_check]
    stab_indices, stab_offsets = _supports_to_csr(stabilizers)
    logical_indices, logical_offsets = _supports_to_csr(logicals)
    check_indices, check_offsets = _supports_to_csr(check_supports)
    max_move_weight = max(
        max(int(value.size) for value in stabilizers),
        max(int(value.size) for value in logicals),
    )
    move_acceptance, swap_acceptance = _acceptance_tables(
        tuple(float(value) for value in K), model.num_qubits, max_move_weight,
    )
    output = _run_autopsy_core(
        states, weights, state_labels, stab_indices, stab_offsets,
        logical_indices, logical_offsets, check_indices, check_offsets,
        np.asarray(syndrome, dtype=np.uint8), move_acceptance, swap_acceptance,
        PortablePrng(seed).state_array(), config.burn_rounds,
        config.measurement_rounds, config.sweeps_per_round,
        config.logical_move_repeat, config.swap_sweeps_per_round,
    )
    names = (
        "labels", "swap_attempts", "swap_accepts", "logical_attempts",
        "logical_accepts", "hot_touches_per_replica",
        "hot_updated_visits_per_replica", "uncertified_round_trips_per_replica",
        "round_trips_per_replica", "sector_changing_round_trips_per_replica",
        "max_hard_coset_residual", "replica_at_rung_by_round",
        "replica_weight_by_round", "replica_label_by_round",
        "replica_phase_by_round", "round_min_rung_by_replica",
        "round_max_rung_by_replica", "direction_by_round",
        "endpoint_events_by_round", "edge_attempts_by_phase_direction",
        "edge_accepts_by_phase_direction", "first_hot_touch_round",
        "first_hot_update_round", "first_uncertified_return_round",
        "first_certified_return_round", "frontier_max_rung", "rung_churn",
        "direction_reversals", "post_hot_return_records",
        "post_hot_return_record_count",
    )
    result = dict(zip(names, output))
    for total, vector_name in (
            ("hot_touches", "hot_touches_per_replica"),
            ("hot_updated_visits", "hot_updated_visits_per_replica"),
            ("uncertified_round_trips", "uncertified_round_trips_per_replica"),
            ("round_trips", "round_trips_per_replica"),
            ("sector_changing_round_trips", "sector_changing_round_trips_per_replica")):
        result[total] = int(result[vector_name].sum())
    return result


def classify_transport(results, config):
    attempts = np.sum([
        result["edge_attempts_by_phase_direction"] for result in results
    ], axis=0)
    accepts = np.sum([
        result["edge_accepts_by_phase_direction"] for result in results
    ], axis=0)
    swap_attempts = np.sum([result["swap_attempts"] for result in results], axis=0)
    swap_accepts = np.sum([result["swap_accepts"] for result in results], axis=0)
    aggregate_rate = swap_accepts / np.maximum(swap_attempts, 1)
    minimum_attempts = int(config["conditional_min_attempts"])
    minimum_rate = float(config["conditional_min_edge_rate"])
    outbound_attempts = attempts[1, 0]
    outbound_rate = accepts[1, 0] / np.maximum(outbound_attempts, 1)
    hot_updates = sum(result["hot_updated_visits"] for result in results)
    frontier = max(int(result["frontier_max_rung"].max()) for result in results)
    temperatures = results[0]["swap_attempts"].size + 1
    detail = {
        "aggregate_edge_rate": aggregate_rate.tolist(),
        "outbound_attempts": outbound_attempts.tolist(),
        "outbound_edge_rate": outbound_rate.tolist(),
        "hot_updated_visits": int(hot_updates),
        "frontier_max_rung": frontier,
        "hot_rung": temperatures - 1,
    }
    if np.any(outbound_attempts < minimum_attempts):
        classification = "INCONCLUSIVE"
        detail["reason"] = "outbound conditional attempts below 200"
    elif np.any((outbound_rate < minimum_rate) & (aggregate_rate >= minimum_rate)):
        classification = "CONDITIONAL_EDGE_BOTTLENECK"
        detail["reason"] = "phase-1 upward edge rate is narrow while aggregate rate is not"
    elif hot_updates == 0 or frontier < temperatures - 1:
        classification = "GLOBAL_DIFFUSION_OR_RELAXATION_LIMITED"
        detail["reason"] = "no conditional narrow edge, but certified hot relaxation was not reached"
    else:
        inbound_attempts = attempts[3, 1]
        inbound_rate = accepts[3, 1] / np.maximum(inbound_attempts, 1)
        detail["post_hot_return_attempts"] = inbound_attempts.tolist()
        detail["post_hot_return_edge_rate"] = inbound_rate.tolist()
        if np.any(inbound_attempts < minimum_attempts):
            classification = "INCONCLUSIVE"
            detail["reason"] = "post-hot return attempts below 200"
        elif np.any(inbound_rate < minimum_rate):
            classification = "POST_HOT_HYSTERESIS"
            detail["reason"] = "certified-hot return edge rate is below 0.05"
        else:
            classification = "GLOBAL_DIFFUSION_OR_RELAXATION_LIMITED"
            detail["reason"] = "no local conditional bottleneck explains the zero round trips"
    return classification, detail


def _parent_arrays(parent):
    names = (
        "labels", "swap_attempts", "swap_accepts", "logical_attempts",
        "logical_accepts", "hot_touches", "hot_updated_visits",
        "uncertified_round_trips", "round_trips", "sector_changing_round_trips",
        "hot_touches_per_replica", "hot_updated_visits_per_replica",
        "uncertified_round_trips_per_replica", "round_trips_per_replica",
        "sector_changing_round_trips_per_replica", "residual",
    )
    return {name: parent[name].copy() for name in names}


def run_autopsy_task(registry_path, discovery_config_path, autopsy_config_path,
                     source_commit, task, parent_root, output_path):
    if _run_autopsy_core is None:
        raise RuntimeError("Numba is required for PT transport autopsy")
    registry = load_registry(registry_path)
    discovery = load_discovery_config(discovery_config_path, registry)
    config = load_autopsy_config(autopsy_config_path, registry, discovery)
    parent_record = next((value for value in config["parents"]
                          if value["parent_raw_sha256"] == task.get("parent_raw_sha256")), None)
    expected = autopsy_task_identity(config, source_commit, parent_record)
    if task != expected:
        raise ValueError("autopsy task identity is noncanonical or tampered")
    fingerprint = sha256_json(expected)
    parent_path = _resolve_parent_path(parent_root, expected["parent_relative_path"])
    if not parent_path.is_file() or sha256_file(parent_path) != expected["parent_raw_sha256"]:
        raise ValueError("autopsy parent raw is missing or has the wrong SHA256")
    validated_parent = validate_discovery_raw(
        parent_path, registry, discovery, PARENT_SOURCE_COMMIT,
    )
    if validated_parent["task_fingerprint"] != expected["parent_task_fingerprint"]:
        raise ValueError("autopsy parent task fingerprint mismatch")
    with np.load(parent_path, allow_pickle=False) as parent:
        parent_arrays = _parent_arrays(parent)
        if (int(parent["uniform_seed"].item()) != expected["uniform_seed"]
                or not np.array_equal(parent["instance_seeds"], expected["instance_seeds"])):
            raise ValueError("autopsy parent seeds differ from the frozen config")

    cell = expected["cell"]
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    uniforms = np.random.Generator(np.random.PCG64(expected["uniform_seed"])).random(
        model.num_qubits
    )
    epsilon = (uniforms < cell["p"]).astype(np.uint8)
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    from .labels import initial_labels
    results = []
    wall_start = time.monotonic()
    core_start = time.process_time()
    for instance, initial_label in enumerate(initial_labels(model.k)):
        results.append(_run_trace_instance(
            model, frame, syndrome, cell["p"], expected["candidate"],
            expected["instance_seeds"][instance], initial_label,
        ))
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    core_names = (
        "labels", "swap_attempts", "swap_accepts", "logical_attempts",
        "logical_accepts", "hot_touches", "hot_updated_visits",
        "uncertified_round_trips", "round_trips", "sector_changing_round_trips",
        "hot_touches_per_replica", "hot_updated_visits_per_replica",
        "uncertified_round_trips_per_replica", "round_trips_per_replica",
        "sector_changing_round_trips_per_replica",
    )
    for name in core_names:
        replay = np.asarray([result[name] for result in results])
        if not np.array_equal(replay, parent_arrays[name]):
            raise PaConflictError(f"autopsy replay differs from parent raw: {name}")
    residual = np.asarray([result["max_hard_coset_residual"] for result in results])
    if not np.array_equal(residual, parent_arrays["residual"]):
        raise PaConflictError("autopsy replay differs from parent raw: residual")
    classification, classification_detail = classify_transport(results, config)
    measurements = expected["candidate"]["measurement_rounds"]
    record_arrays = np.full((4, measurements, 5), -1, dtype=np.int64)
    record_counts = np.empty(4, dtype=np.int64)
    for instance, result in enumerate(results):
        count = int(result["post_hot_return_record_count"])
        record_counts[instance] = count
        record_arrays[instance, :count] = result["post_hot_return_records"][:count]
    arrays = {}
    trace_names = (
        *core_names, "replica_at_rung_by_round", "replica_weight_by_round",
        "replica_label_by_round", "replica_phase_by_round",
        "round_min_rung_by_replica", "round_max_rung_by_replica",
        "direction_by_round", "endpoint_events_by_round",
        "edge_attempts_by_phase_direction", "edge_accepts_by_phase_direction",
        "first_hot_touch_round", "first_hot_update_round",
        "first_uncertified_return_round", "first_certified_return_round",
        "frontier_max_rung", "rung_churn", "direction_reversals",
    )
    for name in trace_names:
        arrays[name] = np.asarray([result[name] for result in results])
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"autopsy raw already exists: {output_path}")
    atomic_npz(
        output_path,
        raw_version=np.array(AUTOPSY_RAW_VERSION),
        autopsy_version=np.array(AUTOPSY_VERSION),
        task_fingerprint=np.array(fingerprint),
        source_commit=np.array(source_commit),
        parent_source_commit=np.array(PARENT_SOURCE_COMMIT),
        parent_raw_sha256=np.array(expected["parent_raw_sha256"]),
        parent_task_fingerprint=np.array(expected["parent_task_fingerprint"]),
        parent_relative_path=np.array(expected["parent_relative_path"]),
        registry_sha256=np.array(registry["registry_sha256"]),
        discovery_config_sha256=np.array(discovery["discovery_config_sha256"]),
        cell_json=np.array(canonical_json(cell)),
        candidate_json=np.array(canonical_json(expected["candidate"])),
        ladder_id=np.array(expected["ladder_id"]),
        uniform_seed=np.array(expected["uniform_seed"], dtype=np.int64),
        instance_seeds=np.asarray(expected["instance_seeds"], dtype=np.int64),
        model_fingerprint=np.array(model.fingerprint()),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
        **arrays,
        max_hard_coset_residual=residual,
        post_hot_return_records=record_arrays,
        post_hot_return_record_count=record_counts,
        classification=np.array(classification),
        classification_json=np.array(canonical_json(classification_detail)),
        core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds),
        engine=np.array("numba_trace_no_extra_rng"),
    )
    return "computed"


def validate_autopsy_raw(path, registry, discovery, config, source_commit,
                         parent_root):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != AUTOPSY_RAW_FIELDS:
            raise ValueError("autopsy raw schema mismatch")
        parent_sha = str(data["parent_raw_sha256"].item())
        parent = next((value for value in config["parents"]
                       if value["parent_raw_sha256"] == parent_sha), None)
        if parent is None:
            raise ValueError("autopsy raw references an unknown parent")
        expected = autopsy_task_identity(config, source_commit, parent)
        code = next(
            row for row in registry["codes"]
            if row["code_id"] == parent["cell"]["code_id"]
        )
        scalars = {
            "raw_version": AUTOPSY_RAW_VERSION,
            "autopsy_version": AUTOPSY_VERSION,
            "task_fingerprint": sha256_json(expected),
            "source_commit": source_commit,
            "parent_source_commit": PARENT_SOURCE_COMMIT,
            "parent_task_fingerprint": parent["parent_task_fingerprint"],
            "parent_relative_path": parent["parent_relative_path"],
            "registry_sha256": registry["registry_sha256"],
            "discovery_config_sha256": discovery["discovery_config_sha256"],
            "cell_json": canonical_json(parent["cell"]),
            "candidate_json": canonical_json(parent["candidate"]),
            "ladder_id": parent["ladder_id"],
            "uniform_seed": parent["uniform_seed"],
            "engine": "numba_trace_no_extra_rng",
        }
        for field, expected_value in scalars.items():
            if str(data[field].item()) != str(expected_value):
                raise ValueError(f"autopsy raw identity mismatch: {field}")
        if not np.array_equal(data["instance_seeds"], parent["instance_seeds"]):
            raise ValueError("autopsy instance seeds mismatch")
        parent_path = _resolve_parent_path(parent_root, parent["parent_relative_path"])
        if sha256_file(parent_path) != parent_sha:
            raise ValueError("autopsy parent SHA no longer matches")
        with np.load(parent_path, allow_pickle=False) as old:
            comparisons = {
                "labels": "labels", "swap_attempts": "swap_attempts",
                "swap_accepts": "swap_accepts", "logical_attempts": "logical_attempts",
                "logical_accepts": "logical_accepts", "hot_touches": "hot_touches",
                "hot_updated_visits": "hot_updated_visits",
                "uncertified_round_trips": "uncertified_round_trips",
                "round_trips": "round_trips",
                "sector_changing_round_trips": "sector_changing_round_trips",
                "hot_touches_per_replica": "hot_touches_per_replica",
                "hot_updated_visits_per_replica": "hot_updated_visits_per_replica",
                "uncertified_round_trips_per_replica": "uncertified_round_trips_per_replica",
                "round_trips_per_replica": "round_trips_per_replica",
                "sector_changing_round_trips_per_replica": "sector_changing_round_trips_per_replica",
                "max_hard_coset_residual": "residual",
            }
            for new_name, old_name in comparisons.items():
                if not np.array_equal(data[new_name], old[old_name]):
                    raise ValueError(f"autopsy core replay mismatch: {new_name}")
        candidate = parent["candidate"]
        instances = 4
        measurements = candidate["measurement_rounds"]
        temperatures = candidate["num_temperatures"]
        permutation = data["replica_at_rung_by_round"]
        weights = data["replica_weight_by_round"]
        labels = data["replica_label_by_round"]
        phases = data["replica_phase_by_round"]
        minimum = data["round_min_rung_by_replica"]
        maximum = data["round_max_rung_by_replica"]
        directions = data["direction_by_round"]
        events = data["endpoint_events_by_round"]
        trace_shape = (instances, measurements, temperatures)
        for name, value in (
                ("replica_at_rung_by_round", permutation),
                ("replica_weight_by_round", weights),
                ("replica_label_by_round", labels),
                ("replica_phase_by_round", phases),
                ("round_min_rung_by_replica", minimum),
                ("round_max_rung_by_replica", maximum),
                ("direction_by_round", directions),
                ("endpoint_events_by_round", events)):
            if value.shape != trace_shape:
                raise ValueError(f"autopsy trace shape mismatch: {name}")
        expected_permutation = np.arange(temperatures, dtype=np.uint16)
        if not np.all(np.sort(permutation, axis=2) == expected_permutation):
            raise ValueError("autopsy round permutation is invalid")
        cold_replicas = permutation[:, :, 0].astype(np.int64)
        cold_labels = np.take_along_axis(
            labels, cold_replicas[:, :, None], axis=2,
        )[:, :, 0]
        if not np.array_equal(cold_labels, data["labels"]):
            raise ValueError("autopsy cold labels disagree with the round trace")
        if (np.any(weights < 0) or np.any(weights > code["n"])
                or (code["k"] < 64 and np.any(labels >> np.uint64(code["k"])))):
            raise ValueError("autopsy weight/label trace is out of range")
        if np.any((phases < 0) | (phases > 3)):
            raise ValueError("autopsy phase trace is invalid")
        expected_directions = np.where(
            phases == 1, 1, np.where((phases == 2) | (phases == 3), -1, 0),
        ).astype(np.int8)
        if not np.array_equal(directions, expected_directions):
            raise ValueError("autopsy direction trace disagrees with phase")
        if (np.any(minimum > maximum) or np.any(maximum >= temperatures)
                or np.any(events > 15)):
            raise ValueError("autopsy round extrema/event trace is invalid")
        frontier = maximum.max(axis=1).astype(np.int64)
        if not np.array_equal(frontier, data["frontier_max_rung"]):
            raise ValueError("autopsy frontier cannot be recomputed from round extrema")
        attempts = data["edge_attempts_by_phase_direction"]
        accepts = data["edge_accepts_by_phase_direction"]
        conditional_shape = (instances, 4, 2, temperatures - 1)
        if (attempts.shape != conditional_shape or accepts.shape != conditional_shape
                or np.any(attempts < 0) or np.any(accepts < 0)
                or np.any(accepts > attempts)):
            raise ValueError("autopsy conditional edge counters are invalid")
        sub_sweeps = measurements * candidate["swap_sweeps_per_round"]
        start_parity = (
            candidate["burn_rounds"] * candidate["swap_sweeps_per_round"]
        ) % 2
        expected_edge_attempts = np.asarray([
            (sub_sweeps + 1) // 2 if edge % 2 == start_parity else sub_sweeps // 2
            for edge in range(temperatures - 1)
        ])
        if (not np.array_equal(
                attempts[:, :, 0].sum(axis=1),
                np.broadcast_to(expected_edge_attempts, (instances, temperatures - 1)))
                or not np.array_equal(
                    attempts[:, :, 1].sum(axis=1),
                    np.broadcast_to(expected_edge_attempts, (instances, temperatures - 1)))):
            raise ValueError("autopsy conditional attempts do not cover every swap proposal")

        first_fields = (
            ("first_hot_touch_round", 1), ("first_hot_update_round", 2),
            ("first_uncertified_return_round", 4),
            ("first_certified_return_round", 8),
        )
        for field, bit in first_fields:
            expected_first = np.full((instances, temperatures), -1, dtype=np.int64)
            for instance in range(instances):
                for replica in range(temperatures):
                    matches = np.flatnonzero(events[instance, :, replica] & bit)
                    if matches.size:
                        expected_first[instance, replica] = int(matches[0])
            if not np.array_equal(data[field], expected_first):
                raise ValueError(f"autopsy first-passage mismatch: {field}")
        records = data["post_hot_return_records"]
        counts = data["post_hot_return_record_count"]
        if records.shape != (instances, measurements, 5) or counts.shape != (instances,):
            raise ValueError("autopsy post-hot return record shape mismatch")
        for instance, count_value in enumerate(counts):
            count = int(count_value)
            if not 0 <= count <= measurements or np.any(records[instance, count:] != -1):
                raise ValueError("autopsy post-hot return record count mismatch")
            active = records[instance, :count]
            if count and (np.any(active[:, 0] < 0) or np.any(active[:, 0] >= temperatures)
                          or np.any(active[:, 3] != active[:, 2] - active[:, 1])
                          or np.any((active[:, 4] < 0) | (active[:, 4] > 1))):
                raise ValueError("autopsy post-hot return record is invalid")
        results = []
        trace_names = (
            "swap_attempts", "swap_accepts", "hot_updated_visits",
            "frontier_max_rung", "edge_attempts_by_phase_direction",
            "edge_accepts_by_phase_direction",
        )
        for instance in range(instances):
            result = {name: data[name][instance].copy() for name in trace_names}
            result["hot_updated_visits"] = int(result["hot_updated_visits"])
            results.append(result)
        classification, detail = classify_transport(results, config)
        if (str(data["classification"].item()) != classification
                or str(data["classification_json"].item()) != canonical_json(detail)):
            raise ValueError("autopsy classification transcript mismatch")
        for field in ("core_seconds", "wall_seconds"):
            value = float(data[field].item())
            if not np.isfinite(value) or value < 0:
                raise ValueError("autopsy timing is invalid")
        return {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "task_fingerprint": sha256_json(expected), "ladder_id": parent["ladder_id"],
            "cell": parent["cell"], "classification": classification,
            "classification_detail": detail,
        }


def _verified_autopsy_paths(raw_root, config, source_commit):
    raw_root = Path(raw_root).resolve()
    manifests = []
    for path in raw_root.rglob("raw_manifest.json"):
        value = json.loads(path.read_text(encoding="ascii"))
        if value.get("raw_manifest_version") == AUTOPSY_RAW_VERSION:
            manifests.append((path.resolve(), value))
    if not manifests:
        return None
    evidence = {sha256_file(path): path.resolve() for path in raw_root.rglob("*.json")}
    control_sha = manifests[0][1].get("control_sha256")
    ownership_sha = manifests[0][1].get("ownership_sha256")
    if control_sha not in evidence or ownership_sha not in evidence:
        raise ValueError("autopsy control/ownership evidence is missing")
    control = json.loads(evidence[control_sha].read_text(encoding="ascii"))
    if (set(control) != {
            "manifest_version", "source_commit", "autopsy_config_sha256", "tasks"}
            or control["manifest_version"] != AUTOPSY_TASKS_VERSION
            or control["source_commit"] != source_commit
            or control["autopsy_config_sha256"] != config["autopsy_config_sha256"]
            or control["tasks"] != autopsy_tasks(config, source_commit)):
        raise ValueError("autopsy control manifest is noncanonical")
    ownership = json.loads(evidence[ownership_sha].read_text(encoding="ascii"))
    expected_ownership = fixed_autopsy_ownership(
        control["tasks"], ownership.get("nodes", []), source_commit, control_sha,
    )
    if ownership != expected_ownership:
        raise ValueError("autopsy ownership evidence is noncanonical")
    expected_fingerprints = {sha256_json(task) for task in control["tasks"]}
    listed = {}
    seen_nodes = set()
    source_identity = None
    for manifest_path, manifest in manifests:
        if set(manifest) != {
                "raw_manifest_version", "node", "stage", "stage_fingerprint",
                "source_commit", "control_sha256", "ownership_sha256",
                "source_identity", "files"}:
            raise ValueError("autopsy node manifest schema mismatch")
        node = manifest["node"]
        identity = manifest["source_identity"]
        if (manifest["stage"] != "transport_autopsy" or node in seen_nodes
                or node not in ownership["nodes"]
                or manifest["source_commit"] != source_commit
                or manifest["control_sha256"] != control_sha
                or manifest["ownership_sha256"] != ownership_sha
                or manifest["stage_fingerprint"] != ownership["stage_fingerprint"]
                or not isinstance(identity, dict)
                or identity.get("source_commit") != source_commit
                or identity.get("mode") != "archive"):
            raise ValueError("autopsy node manifest identity mismatch")
        if (set(identity) != {
                "source_commit", "mode", "archive_sha256", "manifest_sha256",
                "file_count"}
                or re.fullmatch(r"[0-9a-f]{64}", str(identity["archive_sha256"])) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(identity["manifest_sha256"])) is None
                or isinstance(identity["file_count"], bool)
                or not isinstance(identity["file_count"], int)
                or identity["file_count"] <= 0):
            raise ValueError("autopsy source archive identity is invalid")
        seen_nodes.add(node)
        if source_identity is None:
            source_identity = identity
        elif source_identity != identity:
            raise ValueError("autopsy nodes used inconsistent source archives")
        root = manifest_path.parent
        status_path, success_path = root / "stage_status.json", root / "SUCCESS"
        if (not status_path.is_file() or not success_path.is_file()
                or (root / "RUNNING").exists() or (root / "FAILED").exists()):
            raise ValueError("autopsy stage lacks an exclusive SUCCESS marker")
        status = json.loads(status_path.read_text(encoding="ascii"))
        success = json.loads(success_path.read_text(encoding="ascii"))
        if (set(status) != {
                "status", "node", "stage_fingerprint", "expected", "computed",
                "reused", "raw_manifest_sha256"}
                or status["status"] != "SUCCESS" or status["node"] != node
                or status["stage_fingerprint"] != ownership["stage_fingerprint"]
                or status["raw_manifest_sha256"] != sha256_file(manifest_path)
                or status["expected"] != len(manifest["files"])
                or status["computed"] + status["reused"] != status["expected"]
                or set(success) != {"stage_fingerprint", "completed_utc"}
                or success["stage_fingerprint"] != ownership["stage_fingerprint"]
                or not isinstance(success["completed_utc"], str)
                or not success["completed_utc"]):
            raise ValueError("autopsy SUCCESS/status marker identity mismatch")
        assigned = {
            fingerprint for fingerprint, owner in ownership["task_owner"].items()
            if owner == node
        }
        covered = set()
        for item in manifest["files"]:
            relative = Path(item.get("path", ""))
            path = (root / relative).resolve()
            fingerprint = item.get("task_fingerprint")
            if (set(item) != {"task_fingerprint", "path", "sha256"}
                    or fingerprint in covered or fingerprint not in assigned
                    or relative.is_absolute() or ".." in relative.parts
                    or root not in path.parents or path.suffix != ".npz"
                    or re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is None
                    or not path.is_file()
                    or sha256_file(path) != item["sha256"] or path in listed):
                raise ValueError("autopsy node raw coverage/hash mismatch")
            covered.add(fingerprint)
            listed[path] = fingerprint
        if covered != assigned:
            raise ValueError("autopsy node manifest misses assigned tasks")
    if (seen_nodes != set(ownership["nodes"])
            or set(listed.values()) != expected_fingerprints):
        raise ValueError("autopsy completed nodes do not cover the frozen task set")
    actual = {
        path.resolve() for manifest_path, _ in manifests
        for path in manifest_path.parent.rglob("*.npz")
    }
    if actual != set(listed):
        raise ValueError("autopsy node directories contain unmanifested raw files")
    return {"paths": sorted(listed), "source_identity": source_identity,
            "stage_fingerprint": ownership["stage_fingerprint"]}


def analyze_autopsy(raw_dir, registry_path, discovery_config_path,
                     autopsy_config_path, source_commit, parent_root, output_path=None):
    registry = load_registry(registry_path)
    discovery = load_discovery_config(discovery_config_path, registry)
    config = load_autopsy_config(autopsy_config_path, registry, discovery)
    verified = _verified_autopsy_paths(raw_dir, config, source_commit)
    paths = (
        verified["paths"] if verified is not None
        else sorted(Path(raw_dir).rglob("*.npz"))
    )
    records = [
        validate_autopsy_raw(
            path, registry, discovery, config, source_commit, parent_root,
        )
        for path in paths
    ]
    expected = {sha256_json(task) for task in autopsy_tasks(config, source_commit)}
    actual = [record["task_fingerprint"] for record in records]
    if len(actual) != len(set(actual)) or not set(actual) <= expected:
        raise ValueError("autopsy raw contains duplicate or unexpected tasks")
    status = "PASS" if set(actual) == expected else "INCOMPLETE"
    report = {
        "report_version": AUTOPSY_REPORT_VERSION,
        "status": status,
        "source_commit": source_commit,
        "parent_source_commit": PARENT_SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": discovery["discovery_config_sha256"],
        "autopsy_config_sha256": config["autopsy_config_sha256"],
        "expected_tasks": len(expected), "present_tasks": len(actual),
        "remote_evidence": verified,
        "tasks": records,
    }
    report["analysis_sha256"] = sha256_json(report)
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-config")
    make.add_argument("registry"); make.add_argument("discovery_config")
    make.add_argument("output"); make.add_argument("--parent-root")
    plan = sub.add_parser("plan")
    plan.add_argument("config"); plan.add_argument("source_commit"); plan.add_argument("output")
    run = sub.add_parser("run-task")
    run.add_argument("registry"); run.add_argument("discovery_config"); run.add_argument("config")
    run.add_argument("source_commit"); run.add_argument("task_json"); run.add_argument("parent_root")
    run.add_argument("output")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("raw_dir"); analyze.add_argument("registry")
    analyze.add_argument("discovery_config"); analyze.add_argument("config")
    analyze.add_argument("source_commit"); analyze.add_argument("parent_root")
    analyze.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "make-config":
        result = write_autopsy_config(
            args.registry, args.discovery_config, args.output, args.parent_root,
        )
    elif args.command == "plan":
        result = write_autopsy_task_manifest(
            args.config, args.source_commit, args.output,
        )
    elif args.command == "run-task":
        task = json.loads(Path(args.task_json).read_text(encoding="ascii"))
        result = run_autopsy_task(
            args.registry, args.discovery_config, args.config, args.source_commit,
            task, args.parent_root, args.output,
        )
    else:
        result = analyze_autopsy(
            args.raw_dir, args.registry, args.discovery_config, args.config,
            args.source_commit, args.parent_root, args.output,
        )
    print(sha256_json(result) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
