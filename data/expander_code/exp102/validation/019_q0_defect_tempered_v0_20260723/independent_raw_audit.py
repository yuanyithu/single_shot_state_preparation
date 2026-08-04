#!/usr/bin/env python3
"""Independent raw-only audit for the frozen DTC21-S1 V0 preflight.

This intentionally never invokes the defect-tempered transition kernel.  It
reconstructs the frozen HGP cell, all legal P/U/L starts, labels, defects,
fixed-clock D=0 masks, counters, and the transport gate from saved NPZ files.
The runner's deterministic replay is useful, but it must not be the only
reader of its own evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import frozen_character_set
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "local_m8_d0_transport_v0"
REGISTRY_PATH = Path("data/expander_code/exp102/registry/registry.json")
MANIFEST_SHA256 = "751f76bec3831fd8fad39ee96972bd2a5e54a3da4a2e87a90ba202554decb337"
CONTRACT_VERSION = "exp102.q0_defect_tempered.v0"
RAW_VERSION = "exp102.q0_defect_tempered.v0.raw.v1"
ENGINE_RAW_VERSION = "exp102.q0_defect_tempered.raw.v0"
ENGINE_KERNEL = "syndrome_penalty_replica_exchange_iid_hot.v1"
ENGINE_VERSION = "exp102.q0_defect_tempered.v0"
L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
AUDIT_VERSION = "exp102.q0_defect_tempered.v0.independent_raw_audit.v1"


class AuditConflict(RuntimeError):
    """A frozen raw artifact or its conclusion cannot be trusted."""


def require(condition, message):
    if not condition:
        raise AuditConflict(message)


def scalar(value, name):
    array = np.asarray(value)
    require(array.shape == (), f"{name} must be scalar")
    return array.item()


def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise AuditConflict(f"cannot load {path}: {exc}") from exc


def load_raw(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise AuditConflict(f"cannot load raw {path}: {exc}") from exc
    require(not any(value.dtype.hasobject for value in raw.values()), "raw contains object dtype")
    return raw


def pack_state(state):
    return np.packbits(np.asarray(state, dtype=np.uint8), bitorder="little")


def unpack_states(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, count=int(num_qubits),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def label_of(frame, state):
    """Build uint64 signatures explicitly, including bit 63."""
    bits = np.asarray(frame.label_of(np.asarray(state, dtype=np.uint8)), dtype=np.uint8)
    require(bits.ndim == 1 and bits.size <= 64, "logical label bits are invalid")
    value = np.uint64(0)
    for bit in np.flatnonzero(bits):
        value |= np.uint64(1) << np.uint64(bit)
    return value


def labels_of(frame, states):
    return np.asarray([label_of(frame, state) for state in states], dtype=np.uint64)


def residuals_of(model, states, syndrome):
    values = np.asarray(states, dtype=np.uint8)
    if values.ndim == 1:
        values = values[None, :]
    return (
        model.H_check.astype(np.int64) @ values.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]


def rank_masks(masks, k):
    pivots = {}
    for raw in masks:
        value = int(np.uint64(raw))
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    require(len(pivots) <= int(k), "label-delta rank exceeds logical dimension")
    return len(pivots)


def character_signs(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    result = np.empty((labels.size, masks.size), dtype=np.int8)
    for row, label in enumerate(labels):
        for column, mask in enumerate(masks):
            result[row, column] = 1 if (int(label & mask).bit_count() & 1) == 0 else -1
    return result


def leave_return(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    if labels.size < 3:
        return np.zeros(np.asarray(masks).size, dtype=bool)
    signs = character_signs(labels, masks)
    origin = signs[0]
    left = np.zeros(signs.shape[1], dtype=bool)
    returned = np.zeros(signs.shape[1], dtype=bool)
    for values in signs[1:]:
        changed = values != origin
        returned |= left & ~changed
        left |= changed
    return returned


def reduce_logical_basis(rows):
    """Independent deterministic implementation of the frozen reduction rule."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8).copy()
    require(rows.ndim == 2, "logical basis is not two-dimensional")
    while True:
        best = None
        for i in range(rows.shape[0]):
            old_weight = int(rows[i].sum())
            for j in range(rows.shape[0]):
                if i == j:
                    continue
                improvement = old_weight - int(np.count_nonzero(rows[i] ^ rows[j]))
                if improvement <= 0:
                    continue
                candidate = (-improvement, i, j)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return rows
        _, i, j = best
        rows[i] ^= rows[j]


def select_l_move(epsilon, model, frame):
    reduced = reduce_logical_basis(model.logical_move_basis)
    seen = set()
    selected = None
    candidate_count = 0
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = pack_state(move).tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(label_of(frame, move))
            require(signature != 0, "L-start candidate has zero signature")
            require(not residuals_of(model, move, np.zeros(model.num_checks, dtype=np.uint8)).any(),
                    "L-start candidate leaves the kernel")
            candidate_count += 1
            key = (int(np.count_nonzero(epsilon ^ move)), int(move.sum()), signature, packed)
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    require(selected is not None, "no L-start candidate exists")
    key, move = selected
    return move, {
        "rule": L_START_RULE,
        "candidate_orders": [1, 2, 3],
        "candidate_count": candidate_count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": array_sha256(move),
    }


def seed_from_identity(identity, stage, role="stream", index=0):
    expected = {
        "source_commit", "config_sha256", "registry_sha256", "cell_fingerprint",
        "method_id", "resource_tier", "init_family", "trajectory_index",
        "trajectory_namespace",
    }
    require(set(identity) == expected, "seed identity fields changed")
    return derive_seed(
        ENGINE_VERSION,
        identity["source_commit"], identity["config_sha256"], identity["registry_sha256"],
        identity["cell_fingerprint"], identity["method_id"], identity["resource_tier"],
        identity["init_family"], int(identity["trajectory_index"]),
        identity["trajectory_namespace"], str(stage), str(role), int(index),
    )


def uniform_hard_coset_state(model, syndrome, seed):
    """Rebuild the K=0 affine initializer without using the DTC kernel."""
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    state = model.logical_sector_section.apply(syndrome, strict=True).astype(np.uint8)
    rng = PortablePrng(int(seed))
    for row in model.stabilizer_rows:
        if rng.randbelow(2):
            state ^= row
    for row in model.logical_move_basis:
        if rng.randbelow(2):
            state ^= row
    return np.ascontiguousarray(state, dtype=np.uint8)


def initial_state(task, context):
    family = task["init_family"]
    if family == "P":
        state = context["epsilon"].copy()
    elif family == "U":
        state = uniform_hard_coset_state(
            context["model"], context["syndrome"],
            seed_from_identity(task["seed_identity"], "initialize", "hard_coset"),
        )
    elif family == "L":
        state = context["epsilon"] ^ context["l_move"]
    else:
        raise AuditConflict(f"unknown initial family {family}")
    require(not residuals_of(context["model"], state, context["syndrome"]).any(),
            f"{family} start leaves hard coset")
    return np.ascontiguousarray(state, dtype=np.uint8)


def engine_digest(raw):
    """Recompute the runner digest without importing its functions."""
    fields = {name[4:]: raw[name] for name in raw if name.startswith("dtc_")}
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_defect_tempered.v0.trajectory_digest.v1\0")
    for name in sorted(fields):
        value = np.asarray(fields[name])
        require(not value.dtype.hasobject, f"engine field {name} has object dtype")
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        dtype = value.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def expected_raw_fields():
    outer = {
        "raw_version", "contract_version", "manifest_sha256", "task_fingerprint",
        "task_json", "source_binding_json", "config_sha256", "registry_sha256",
        "uniform_seed", "syndrome", "character_masks", "character_sha256",
        "l_start_json", "l_move_sha256", "trajectory_digest", "core_seconds", "wall_seconds",
    }
    engine = {
        "raw_version", "kernel", "method_id", "sampler_config_json", "sampler_config_sha256",
        "seed_identity_json", "initial_state_packed", "burn_state_packed", "final_state_packed",
        "burn_states_packed", "burn_labels", "burn_weights", "burn_defects", "burn_d0_mask",
        "measurement_states_packed", "measurement_labels", "measurement_weights",
        "measurement_defects", "measurement_d0_mask", "measurement_block", "initial_label",
        "burn_label", "final_label", "ladder_kq", "ladder_sha256", "burn_bit_counters_by_rung",
        "measurement_bit_counters_by_rung", "burn_hot_refresh_bit_changes",
        "measurement_hot_refresh_bit_changes", "burn_swap_attempts", "burn_swap_accepts",
        "measurement_swap_attempts", "measurement_swap_accepts", "burn_hot_visits_by_origin",
        "burn_cold_visits_by_origin", "measurement_hot_visits_by_origin",
        "measurement_cold_visits_by_origin", "burn_cold_d0_leaves", "burn_cold_d0_returns",
        "burn_cold_d0_label_changes", "measurement_cold_d0_leaves",
        "measurement_cold_d0_returns", "measurement_cold_d0_label_changes",
        "measurement_residual_weights", "engine",
    }
    return outer | {f"dtc_{name}" for name in engine}


def phase_transition_counts(start_state, states, labels, model, frame, syndrome):
    start_residual = residuals_of(model, start_state, syndrome)[0]
    previous_d0 = not bool(start_residual.any())
    previous_label = label_of(frame, start_state)
    leaves = returns = label_changes = 0
    for state, label in zip(states, labels):
        current_d0 = not bool(residuals_of(model, state, syndrome)[0].any())
        if previous_d0 and not current_d0:
            leaves += 1
        elif not previous_d0 and current_d0:
            returns += 1
        elif previous_d0 and current_d0 and np.uint64(label) != previous_label:
            label_changes += 1
        if current_d0:
            previous_label = np.uint64(label)
        previous_d0 = current_d0
    return leaves, returns, label_changes


def expected_swap_attempts(rounds, edges):
    attempts = np.zeros(edges, dtype=np.int64)
    for round_index in range(int(rounds)):
        for lower in range(round_index & 1, edges, 2):
            attempts[lower] += 1
    return attempts


def validate_phase(raw, prefix, start_state, context, rounds):
    model = context["model"]
    states = unpack_states(raw[f"dtc_{prefix}_states_packed"], model.num_qubits)
    require(states.shape == (rounds, model.num_qubits), f"{prefix} state shape changed")
    labels = np.asarray(raw[f"dtc_{prefix}_labels"], dtype=np.uint64)
    weights = np.asarray(raw[f"dtc_{prefix}_weights"], dtype=np.int32)
    defects = np.asarray(raw[f"dtc_{prefix}_defects"], dtype=np.int32)
    d0_mask = np.asarray(raw[f"dtc_{prefix}_d0_mask"], dtype=np.uint8)
    require(labels.shape == (rounds,), f"{prefix} label shape changed")
    require(weights.shape == (rounds,), f"{prefix} weight shape changed")
    require(defects.shape == (rounds,), f"{prefix} defect shape changed")
    require(d0_mask.shape == (rounds,), f"{prefix} D0 mask shape changed")
    residuals = residuals_of(model, states, context["syndrome"])
    expected_defects = residuals.sum(axis=1).astype(np.int32)
    require(np.array_equal(defects, expected_defects), f"{prefix} defect counts changed")
    require(np.array_equal(d0_mask, (expected_defects == 0).astype(np.uint8)),
            f"{prefix} D0 mask changed")
    require(np.array_equal(weights, states.sum(axis=1).astype(np.int32)), f"{prefix} weights changed")
    require(np.array_equal(labels, labels_of(context["frame"], states)), f"{prefix} labels changed")
    counters = np.asarray(raw[f"dtc_{prefix}_bit_counters_by_rung"], dtype=np.int64)
    rungs = len(context["kq_values"])
    require(counters.shape == (rungs, 2), f"{prefix} bit counter shape changed")
    expected_attempts = np.full(rungs, rounds * model.num_qubits, dtype=np.int64)
    expected_attempts[-1] = 0
    require(np.array_equal(counters[:, 0], expected_attempts), f"{prefix} bit attempts changed")
    require(np.all((0 <= counters[:, 1]) & (counters[:, 1] <= counters[:, 0])),
            f"{prefix} bit changes invalid")
    swap_attempts = np.asarray(raw[f"dtc_{prefix}_swap_attempts"], dtype=np.int64)
    swap_accepts = np.asarray(raw[f"dtc_{prefix}_swap_accepts"], dtype=np.int64)
    require(swap_attempts.shape == (rungs - 1,), f"{prefix} swap attempt shape changed")
    require(np.array_equal(swap_attempts, expected_swap_attempts(rounds, rungs - 1)),
            f"{prefix} swap attempts changed")
    require(np.all((0 <= swap_accepts) & (swap_accepts <= swap_attempts)),
            f"{prefix} swap accepts invalid")
    for name in ("hot_visits_by_origin", "cold_visits_by_origin"):
        visits = np.asarray(raw[f"dtc_{prefix}_{name}"], dtype=np.int64)
        require(visits.shape == (rungs,) and np.all(visits >= 0) and int(visits.sum()) == rounds,
                f"{prefix} {name} changed")
    refresh_changes = int(scalar(raw[f"dtc_{prefix}_hot_refresh_bit_changes"], f"{prefix} refresh"))
    require(0 <= refresh_changes <= rounds * model.num_qubits, f"{prefix} hot refresh count invalid")
    leaves, returns, label_changes = phase_transition_counts(
        start_state, states, labels, model, context["frame"], context["syndrome"],
    )
    for suffix, expected in (("cold_d0_leaves", leaves), ("cold_d0_returns", returns),
                             ("cold_d0_label_changes", label_changes)):
        require(int(scalar(raw[f"dtc_{prefix}_{suffix}"], f"{prefix} {suffix}")) == expected,
                f"{prefix} {suffix} changed")
    return states, labels, defects, d0_mask


def raw_path(task):
    return OUTPUT / "raw" / task["init_family"] / f"t{task['trajectory_index']:02d}.npz"


def validate_raw(raw, task, manifest, context):
    require(set(raw) == expected_raw_fields(), "raw schema changed")
    for name, expected in {
        "raw_version": RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"],
        "registry_sha256": manifest["registry_sha256"],
    }.items():
        require(scalar(raw[name], name) == expected, f"raw {name} changed")
    require(scalar(raw["task_fingerprint"], "task fingerprint") == sha256_json(task),
            "raw task fingerprint changed")
    require(json.loads(scalar(raw["task_json"], "task JSON")) == task, "raw task JSON changed")
    require(json.loads(scalar(raw["source_binding_json"], "source binding")) == manifest["source_binding"],
            "raw source binding changed")
    require(int(scalar(raw["uniform_seed"], "uniform seed")) == context["uniform_seed"],
            "raw uniform seed changed")
    require(np.array_equal(raw["syndrome"], context["syndrome"]), "raw syndrome changed")
    require(np.array_equal(raw["character_masks"], context["characters"].masks),
            "raw character masks changed")
    require(scalar(raw["character_sha256"], "character SHA") == context["characters"].character_sha256,
            "raw character SHA changed")
    require(json.loads(scalar(raw["l_start_json"], "L start")) == context["l_start"],
            "raw L-start metadata changed")
    require(scalar(raw["l_move_sha256"], "L move SHA") == array_sha256(context["l_move"]),
            "raw L move changed")
    require(scalar(raw["trajectory_digest"], "trajectory digest") == engine_digest(raw),
            "raw trajectory digest changed")
    for name in ("core_seconds", "wall_seconds"):
        value = float(scalar(raw[name], name))
        require(math.isfinite(value) and value >= 0.0, f"{name} is invalid")

    require(scalar(raw["dtc_raw_version"], "engine raw version") == ENGINE_RAW_VERSION,
            "engine raw version changed")
    require(scalar(raw["dtc_kernel"], "engine kernel") == ENGINE_KERNEL, "engine kernel changed")
    require(scalar(raw["dtc_method_id"], "method ID") == task["method_id"], "method ID changed")
    require(json.loads(scalar(raw["dtc_sampler_config_json"], "sampler config")) == task["sampler_config"],
            "sampler config changed")
    require(scalar(raw["dtc_sampler_config_sha256"], "sampler config SHA")
            == sha256_json(task["sampler_config"]), "sampler config SHA changed")
    require(json.loads(scalar(raw["dtc_seed_identity_json"], "seed identity")) == task["seed_identity"],
            "seed identity changed")
    require(np.array_equal(raw["dtc_ladder_kq"], np.asarray(context["kq_values"], dtype=np.float64)),
            "ladder values changed")
    require(scalar(raw["dtc_ladder_sha256"], "ladder SHA") == manifest["ladder_sha256"],
            "ladder SHA changed")
    require(scalar(raw["dtc_engine"], "engine") == "numba", "engine changed")

    expected_initial = initial_state(task, context)
    require(np.array_equal(raw["dtc_initial_state_packed"], pack_state(expected_initial)),
            "initial state changed")
    require(int(scalar(raw["dtc_initial_label"], "initial label")) == int(label_of(context["frame"], expected_initial)),
            "initial label changed")

    burn_states, burn_labels, burn_defects, burn_d0 = validate_phase(
        raw, "burn", expected_initial, context, context["burn_rounds"],
    )
    burn_endpoint = burn_states[-1]
    require(np.array_equal(raw["dtc_burn_state_packed"], pack_state(burn_endpoint)),
            "burn endpoint changed")
    require(int(scalar(raw["dtc_burn_label"], "burn label")) == int(label_of(context["frame"], burn_endpoint)),
            "burn label changed")

    measurement_states, measurement_labels, measurement_defects, measurement_d0 = validate_phase(
        raw, "measurement", burn_endpoint, context, context["measurement_rounds"],
    )
    final_state = measurement_states[-1]
    require(np.array_equal(raw["dtc_final_state_packed"], pack_state(final_state)), "final state changed")
    require(int(scalar(raw["dtc_final_label"], "final label")) == int(label_of(context["frame"], final_state)),
            "final label changed")
    require(np.array_equal(raw["dtc_measurement_residual_weights"], measurement_defects),
            "measurement residual weights changed")
    require(np.array_equal(
        raw["dtc_measurement_block"],
        np.repeat(np.arange(8, dtype=np.int8), context["measurement_rounds"] // 8),
    ), "measurement blocks changed")

    d0_labels = measurement_labels[measurement_d0.astype(bool)]
    deltas = d0_labels[1:] ^ d0_labels[:-1] if d0_labels.size > 1 else np.empty(0, dtype=np.uint64)
    changed = deltas[deltas != 0]
    return {
        "task": task,
        "d0_observations": int(d0_labels.size),
        "d0_label_changes": int(changed.size),
        "d0_deltas": [int(value) for value in changed],
        "d0_leave_return": leave_return(d0_labels, context["characters"].masks).astype(np.uint8),
        "d0_returns": int(scalar(raw["dtc_measurement_cold_d0_returns"], "D0 returns")),
        "d0_leaves": int(scalar(raw["dtc_measurement_cold_d0_leaves"], "D0 leaves")),
        "min_swap_rate": float(np.min(
            np.asarray(raw["dtc_measurement_swap_accepts"], dtype=np.float64)
            / np.maximum(np.asarray(raw["dtc_measurement_swap_attempts"], dtype=np.float64), 1.0)
        )),
        "wall_seconds": float(scalar(raw["wall_seconds"], "wall seconds")),
    }


def family_summary(rows, context):
    gates = context["config"]["gates"]
    masks = context["characters"].masks
    all_deltas = [np.uint64(value) for row in rows for value in row["d0_deltas"]]
    returns = np.asarray([row["d0_leave_return"] for row in rows], dtype=np.uint8)
    failures = []
    for index, row in enumerate(rows):
        if row["d0_observations"] < gates["minimum_d0_observations_per_trajectory"]:
            failures.append(f"t{index:02d}:d0_observations")
        if row["d0_returns"] < gates["minimum_d0_leave_returns_per_trajectory"]:
            failures.append(f"t{index:02d}:d0_returns")
    summary = {
        "chain_count": len(rows),
        "d0_observations_per_trajectory": [row["d0_observations"] for row in rows],
        "d0_leave_returns_per_trajectory": [row["d0_returns"] for row in rows],
        "d0_leaves_per_trajectory": [row["d0_leaves"] for row in rows],
        "d0_label_changes_per_trajectory": [row["d0_label_changes"] for row in rows],
        "family_d0_label_changes": int(sum(row["d0_label_changes"] for row in rows)),
        "chains_with_eight_d0_label_changes": int(sum(
            row["d0_label_changes"] >= 8 for row in rows
        )),
        "d0_label_delta_rank": rank_masks(all_deltas, context["model"].k),
        "basis_characters_with_d0_leave_return": int(np.count_nonzero(returns[:, :context["model"].k].sum(axis=0))),
        "nonbasis_characters_with_d0_leave_return": int(np.count_nonzero(returns[:, context["model"].k:].sum(axis=0))),
        "minimum_adjacent_swap_rate_diagnostic": float(min(row["min_swap_rate"] for row in rows)),
        "median_wall_seconds": float(np.median([row["wall_seconds"] for row in rows])),
        "transport_gate": "D0 observations, D0 leave-return loops, and D0 label deltas only",
        "not_a_posterior_or_qtop_gate": True,
    }
    if summary["chain_count"] != context["config"]["trajectories_per_family"]:
        failures.append("chain_count")
    if summary["family_d0_label_changes"] < gates["minimum_family_d0_label_changes"]:
        failures.append("family_d0_label_changes")
    if summary["chains_with_eight_d0_label_changes"] < gates[
            "minimum_trajectories_with_eight_d0_label_changes_per_family"]:
        failures.append("chains_with_eight_d0_label_changes")
    if summary["d0_label_delta_rank"] < gates["minimum_d0_label_delta_rank_per_family"]:
        failures.append("d0_label_delta_rank")
    if summary["basis_characters_with_d0_leave_return"] < gates[
            "minimum_d0_sign_leave_returns_per_character_set_per_family"]:
        failures.append("basis_d0_leave_return")
    if summary["nonbasis_characters_with_d0_leave_return"] < gates[
            "minimum_d0_sign_leave_returns_per_character_set_per_family"]:
        failures.append("nonbasis_d0_leave_return")
    summary["failures"] = failures
    summary["passes_transport_gate"] = not failures
    return summary


def main():
    require(not (OUTPUT / "INDEPENDENT_AUDIT.json").exists(), "audit output already exists")
    require((OUTPUT / "SUCCESS.json").is_file(), "raw stage did not succeed")
    require(not (OUTPUT / "FAILED.json").exists(), "raw stage failed")
    manifest = load_json(OUTPUT / "MANIFEST.json")
    report = load_json(OUTPUT / "REPORT.json")
    manifest_core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    report_core = {name: value for name, value in report.items() if name != "report_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(manifest_core), "manifest SHA changed")
    require(manifest["manifest_sha256"] == MANIFEST_SHA256, "foreign manifest")
    require(report["manifest_sha256"] == MANIFEST_SHA256, "report uses foreign manifest")
    require(report["report_sha256"] == sha256_json(report_core), "report SHA changed")
    require(manifest["contract_version"] == CONTRACT_VERSION, "manifest contract changed")
    require(manifest["raw_version"] == RAW_VERSION, "manifest raw version changed")
    require(len(manifest["tasks"]) == 24, "task count changed")

    source = manifest["source_binding"]
    require(source["source_files"] == {
        name: sha256_file(name) for name in source["source_files"]
    }, "source file binding changed")
    current_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(current_commit == source["source_commit"], "source commit changed")
    require(source["source_binding_sha256"] == sha256_json({
        "source_commit": source["source_commit"], "source_files": source["source_files"],
    }), "source binding SHA changed")

    registry = load_registry(REGISTRY_PATH)
    require(registry["registry_sha256"] == manifest["registry_sha256"], "registry changed")
    _, code, H = load_frozen_code(REGISTRY_PATH, manifest["cell"]["code_id"])
    H = np.ascontiguousarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(manifest["cell"]["disorder_index"]), "uniforms",
    )
    require(int(uniform_seed) == int(manifest["uniform_seed"]), "uniform seed derivation changed")
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(manifest["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    require(bool(syndrome.any()), "planted syndrome unexpectedly vanishes")
    require(array_sha256(H) == manifest["H_sha256"], "H matrix changed")
    require(array_sha256(syndrome) == manifest["syndrome_sha256"], "syndrome changed")
    require(model.fingerprint() == manifest["model_fingerprint"], "model fingerprint changed")
    require(frame.fingerprint() == manifest["logical_frame_fingerprint"], "frame fingerprint changed")

    character_seed = derive_seed(
        manifest["config"]["character_seed_namespace"], registry["registry_sha256"],
        code["code_id"], "d0_transport_characters",
    )
    characters = frozen_character_set(
        model.k, character_seed, manifest["config"]["num_nonbasis_characters"],
    )
    require(characters.character_sha256 == manifest["character_sha256"], "character SHA changed")
    require([int(value) for value in characters.masks] == manifest["character_masks"],
            "character masks changed")
    l_move, l_start = select_l_move(epsilon, model, frame)
    require(l_start == manifest["l_start"], "L-start selection changed")
    require(manifest["config"]["method"]["id"] == "DTC21-S1", "method changed")
    require(manifest["config"]["resource"] == {
        "burn_rounds": 256, "measurement_rounds": 2048, "name": "V0",
    }, "resource changed")
    kq_values = tuple(float(value) for value in manifest["config"]["method"]["kq_values"])
    require(len(kq_values) == 21 and kq_values[-1] == 0.0, "ladder changed")

    expected_files = {raw_path(task) for task in manifest["tasks"]}
    observed_files = set((OUTPUT / "raw").glob("*/*.npz"))
    require(observed_files == expected_files, "raw files are missing or unexpected")
    context = {
        "config": manifest["config"], "model": model, "frame": frame,
        "uniform_seed": int(uniform_seed), "epsilon": epsilon, "syndrome": syndrome,
        "characters": characters, "l_move": l_move, "l_start": l_start,
        "kq_values": kq_values, "burn_rounds": 256, "measurement_rounds": 2048,
    }
    rows = [validate_raw(load_raw(raw_path(task)), task, manifest, context)
            for task in manifest["tasks"]]
    summaries = {
        family: family_summary(
            [row for row in rows if row["task"]["init_family"] == family], context,
        )
        for family in manifest["config"]["init_families"]
    }
    require(summaries == report["families"], "independent transport summary differs from report")
    require(report["status"] == "LOCAL_D0_TRANSPORT_NOT_VIABLE", "unexpected terminal status")
    core = {
        "audit_version": AUDIT_VERSION,
        "manifest_sha256": MANIFEST_SHA256,
        "report_sha256": report["report_sha256"],
        "status": "INDEPENDENT_RAW_AUDIT_PASS",
        "raw_count": len(rows),
        "families": summaries,
    }
    audit = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(OUTPUT / "INDEPENDENT_AUDIT.json", audit)
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
