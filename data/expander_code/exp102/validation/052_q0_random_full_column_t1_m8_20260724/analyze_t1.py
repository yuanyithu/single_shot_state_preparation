"""Independent raw analyzer for the random-full-column m8 T1 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.diagnostics import bulk_ess, split_rhat
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    CharacterSet,
    character_d2_estimate,
    character_means,
    character_qtop_estimate,
    character_values,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _qubit_signatures,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import (
    BCharacterSet,
    _b_character_bits,
    _b_log_likelihood,
)
_workflow = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
CONTRACT_VERSION = _workflow.CONTRACT_VERSION
FAMILIES = _workflow.FAMILIES
NODE_REPORT_VERSION = _workflow.NODE_REPORT_VERSION
RAW_VERSION = _workflow.RAW_VERSION
_load_canonical_json = _workflow._load_canonical_json
_load_config = _workflow._load_config
_load_control = _workflow._load_control
_verify_self_hash = _workflow._verify_self_hash


REPORT_VERSION = "exp102.q0_random_full_column.t1_m8.report.v0"


class AnalysisConflictError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AnalysisConflictError(message)


def _unpack(packed, width):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, count=int(width),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def _columns_to_b_packed(columns, r):
    columns = np.asarray(columns, dtype=np.uint32)
    bits = (
        columns[:, None, :] >> np.arange(r, dtype=np.uint32)[None, :, None]
    ) & np.uint32(1)
    return np.packbits(
        bits.astype(np.uint8).reshape(columns.shape[0], r * r),
        axis=1, bitorder="little",
    )


def _state_b_columns(state, H):
    r, n = H.shape
    block = np.asarray(state, dtype=np.uint8)[n * n:].reshape(r, r)
    powers = np.left_shift(np.uint32(1), np.arange(r, dtype=np.uint32))
    return np.einsum(
        "ij,i->j", block.astype(np.uint32), powers, optimize=False,
    ).astype(np.uint32)


def _labels_from_states(states, signatures, chunk=256):
    result = np.empty(states.shape[0], dtype=np.uint64)
    signatures = np.asarray(signatures, dtype=np.uint64)
    for start in range(0, states.shape[0], chunk):
        stop = min(start + chunk, states.shape[0])
        result[start:stop] = np.bitwise_xor.reduce(
            np.where(states[start:stop].astype(bool), signatures[None, :], np.uint64(0)),
            axis=1,
        )
    return result


def _verify_hgp_syndrome(states, H, syndrome, chunk=256):
    r, n = H.shape
    target = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    for start in range(0, states.shape[0], chunk):
        stop = min(start + chunk, states.shape[0])
        batch = states[start:stop]
        A = batch[:, :n * n].reshape(-1, n, n).astype(np.int64)
        B = batch[:, n * n:].reshape(-1, r, r).astype(np.int64)
        observed = (
            np.einsum("ij,tjk->tik", H.astype(np.int64), A, optimize=False)
            + np.einsum("tij,jk->tik", B, H.astype(np.int64), optimize=False)
        ) & 1
        _require(np.array_equal(
            observed.astype(np.uint8), np.repeat(target[None, :, :], stop - start, axis=0),
        ), "measurement state left the hard coset")


def _replay_b_transcript(initial, selected, old, new):
    state = np.asarray(initial, dtype=np.uint32).copy()
    trace = np.empty((len(selected), state.size), dtype=np.uint32)
    changes = 0
    changed_bits = 0
    for clock, column_value in enumerate(selected):
        column = int(column_value)
        _require(0 <= column < state.size and int(state[column]) == int(old[clock]),
                 "B transcript old column mismatch")
        delta = int(old[clock]) ^ int(new[clock])
        if delta:
            changes += 1
            changed_bits += delta.bit_count()
        state[column] = new[clock]
        trace[clock] = state
    return trace, changes, changed_bits


def _load_and_verify_raw(path, task, context, schedule, log_mass):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required_metadata = {
        "archive_sha256", "config_sha256", "contract_version",
        "control_content_sha256", "model_fingerprint", "raw_version",
        "replay_seconds", "sampling_seconds", "schedule_sha256", "source_commit",
        "source_manifest_sha256", "syndrome_packed", "task_fingerprint",
        "task_json", "version",
    }
    required_kernel = {
        "burn__counters", "burn__final_b_columns", "burn__selected_columns",
        "burn__old_columns", "burn__new_columns", "final_b_columns",
        "final_state_packed", "initial_b_columns", "initial_state_packed",
        "measurement__counters", "measurement__selected_columns",
        "measurement__old_columns", "measurement__new_columns",
        "measurement__b_columns", "measurement__b_likelihood",
        "measurement__b_weights", "measurement__blocks", "measurement__labels",
        "measurement__states_packed", "measurement__weights", "seed_identity_sha256",
    }
    _require(set(data) == required_metadata | required_kernel, "raw schema changed")
    source = schedule["source_identity"]
    scalar_identity = {
        "archive_sha256": source["archive_sha256"],
        "config_sha256": context["config_sha"],
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "model_fingerprint": context["model"].fingerprint(),
        "raw_version": RAW_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_commit": source["source_commit"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
    }
    for name, expected in scalar_identity.items():
        _require(str(data[name].item()) == expected, f"raw identity mismatch: {name}")
    _require(np.array_equal(data["syndrome_packed"], context["arrays"]["syndrome_packed"])
             and math.isfinite(float(data["sampling_seconds"]))
             and float(data["sampling_seconds"]) > 0.0
             and math.isfinite(float(data["replay_seconds"]))
             and float(data["replay_seconds"]) > 0.0,
             "raw syndrome or timing invalid")
    seed_identity = hashlib.sha256(
        np.asarray([
            task["burn_update_seed"], task["measurement_update_seed"],
            task["observation_seed"],
        ], dtype=">u8").tobytes()
        + np.asarray(0.04, dtype=">f8").tobytes()
        + np.asarray([
            context["config"]["resource"]["burn_updates"],
            context["config"]["resource"]["measurement_updates"],
        ], dtype=">u8").tobytes()
    ).hexdigest()
    _require(str(data["seed_identity_sha256"].item()) == seed_identity,
             "raw seed identity mismatch")

    initial = _unpack(data["initial_state_packed"], context["model"].num_qubits)
    family = task["family"]
    if family == "P":
        expected = context["fixed_states"][0]
    elif family == "M0":
        expected = context["fixed_states"][1]
    elif family == "M1":
        expected = context["fixed_states"][2]
    elif family == "S":
        expected = context["fixed_states"][3 + int(task["index"])]
    else:
        from data.expander_code.exp102.exp102_pipeline.q0_global import uniform_hard_coset_state
        expected = uniform_hard_coset_state(
            context["model"], context["syndrome"], task["initialization_seed"],
        )
    _require(np.array_equal(initial, expected), "raw initial state mismatch")
    _require(np.array_equal(_state_b_columns(initial, context["H"]),
                            data["initial_b_columns"]),
             "raw initial B columns mismatch")
    burn_trace, burn_changes, burn_bits = _replay_b_transcript(
        data["initial_b_columns"], data["burn__selected_columns"],
        data["burn__old_columns"], data["burn__new_columns"],
    )
    _require(np.array_equal(burn_trace[-1], data["burn__final_b_columns"]),
             "raw burn endpoint mismatch")
    measurement_trace, measurement_changes, measurement_bits = _replay_b_transcript(
        data["burn__final_b_columns"], data["measurement__selected_columns"],
        data["measurement__old_columns"], data["measurement__new_columns"],
    )
    _require(np.array_equal(measurement_trace, data["measurement__b_columns"])
             and np.array_equal(measurement_trace[-1], data["final_b_columns"]),
             "raw measurement B transcript mismatch")
    burn_counters = data["burn__counters"]
    measurement_counters = data["measurement__counters"]
    _require(np.array_equal(
        burn_counters[:4], [burn_trace.shape[0], burn_changes, burn_bits, 0],
    ) and int(burn_counters[4]) == 0,
             "raw burn counters mismatch")

    states = _unpack(data["measurement__states_packed"], context["model"].num_qubits)
    _verify_hgp_syndrome(states, context["H"], context["syndrome"])
    n = context["H"].shape[1]
    r = context["H"].shape[0]
    observed_b = np.packbits(
        states[:, n * n:].reshape(-1, r * r), axis=1, bitorder="little",
    )
    stored_b = _columns_to_b_packed(measurement_trace, r)
    _require(np.array_equal(observed_b, stored_b), "raw state/B mismatch")
    labels = _labels_from_states(states, _qubit_signatures(context["frame"]))
    _require(np.array_equal(labels, data["measurement__labels"])
             and np.array_equal(states.sum(axis=1), data["measurement__weights"]),
             "raw label or weight mismatch")
    label_changes = int(labels[0] != state_label(context["frame"], initial))
    label_changes += int(np.count_nonzero(labels[1:] != labels[:-1]))
    _require(np.array_equal(
        measurement_counters[:4], [
            measurement_trace.shape[0], measurement_changes, measurement_bits,
            measurement_trace.shape[0],
        ],
    ) and int(measurement_counters[4]) == label_changes,
             "raw measurement counters mismatch")
    b_weights = np.bitwise_count(stored_b).sum(axis=1).astype(np.int16)
    b_likelihood = _b_log_likelihood(
        stored_b, context["H"], context["syndrome"], log_mass,
    )
    _require(np.array_equal(b_weights, data["measurement__b_weights"])
             and np.array_equal(b_likelihood, data["measurement__b_likelihood"]),
             "raw B weight or likelihood mismatch")
    _require(np.array_equal(
        data["measurement__blocks"],
        np.minimum(7, 8 * np.arange(labels.size) // labels.size).astype(np.int8),
    ), "raw block clock mismatch")
    _require(np.array_equal(
        _unpack(data["final_state_packed"], context["model"].num_qubits), states[-1],
    ), "raw final state mismatch")
    return {
        "b_likelihood": b_likelihood,
        "b_packed": stored_b,
        "b_weights": b_weights.astype(np.float64),
        "burn_b_packed": _columns_to_b_packed(burn_trace, r),
        "burn_changes": burn_changes,
        "family": family,
        "index": int(task["index"]),
        "initial_b_packed": _columns_to_b_packed(
            data["initial_b_columns"][None, :], r,
        )[0],
        "initial_label": int(state_label(context["frame"], initial)),
        "label_changes": label_changes,
        "labels": labels,
        "measurement_changes": measurement_changes,
        "path": path,
        "sampling_seconds": float(data["sampling_seconds"]),
        "weights": data["measurement__weights"].astype(np.float64),
    }


def _observable_diagnostic(chains):
    chains = np.asarray(chains, dtype=np.float64)
    _require(chains.ndim == 2 and chains.shape[0] == 8, "diagnostic chain shape changed")
    if np.unique(chains).size == 1:
        return {"bulk_ess": float(chains.size), "degenerate": True, "split_rhat": 1.0}
    return {
        "bulk_ess": float(bulk_ess(chains)),
        "degenerate": False,
        "split_rhat": float(split_rhat(chains)),
    }


def _trajectory_mean_and_se(chains, normalizer):
    trajectory = np.asarray(chains, dtype=np.float64).mean(axis=1) / float(normalizer)
    return float(trajectory.mean()), float(trajectory.std(ddof=1) / math.sqrt(trajectory.size)), trajectory


def _collision_diagnostic(label_chains):
    frequencies = []
    for labels in label_chains:
        values, counts = np.unique(labels, return_counts=True)
        frequencies.append({int(value): int(count) / labels.size for value, count in zip(values, counts)})
    overlaps = []
    for left in range(len(frequencies)):
        for right in range(left + 1, len(frequencies)):
            a, b = frequencies[left], frequencies[right]
            if len(a) > len(b):
                a, b = b, a
            overlaps.append(sum(value * b.get(label, 0.0) for label, value in a.items()))
    return float(np.mean(overlaps))


def _family_summary(records, context, logical_set, b_set):
    records = sorted(records, key=lambda row: row["index"])
    _require([row["index"] for row in records] == list(range(8)),
             "family trajectories are incomplete")
    gates = context["config"]["gates"]
    label_chains = [row["labels"] for row in records]
    logical_means, _counts = character_means(label_chains, logical_set.masks)
    qtop = character_qtop_estimate(logical_set, logical_means)
    weights = np.stack([row["weights"] for row in records])
    b_weights = np.stack([row["b_weights"] for row in records])
    b_likelihood = np.stack([row["b_likelihood"] for row in records])
    weight_mean, weight_se, weight_trajectory = _trajectory_mean_and_se(
        weights, context["model"].num_qubits,
    )
    b_weight_mean, b_weight_se, b_weight_trajectory = _trajectory_mean_and_se(
        b_weights, b_set.dimension,
    )
    likelihood_mean, likelihood_se, likelihood_trajectory = _trajectory_mean_and_se(
        b_likelihood, context["H"].shape[1],
    )

    diagnostic_masks = np.concatenate((
        logical_set.masks[logical_set.basis_positions],
        np.delete(logical_set.masks, logical_set.basis_positions)[:64],
    ))
    logical_diagnostics = []
    for mask in diagnostic_masks:
        chains = np.stack([
            character_values(labels, [mask])[:, 0] for labels in label_chains
        ])
        logical_diagnostics.append(_observable_diagnostic(chains))
    b_bits = [
        1.0 - 2.0 * _b_character_bits(
            row["b_packed"], b_set.masks_packed,
        ).astype(np.float64)
        for row in records
    ]
    b_chains = np.stack(b_bits)
    b_means = b_chains.mean(axis=1)
    b_diagnostics = [
        _observable_diagnostic(b_chains[:, :, index]) for index in range(b_set.size)
    ]
    scalar_diagnostics = {
        "b_likelihood": _observable_diagnostic(b_likelihood),
        "b_weight": _observable_diagnostic(b_weights),
        "weight": _observable_diagnostic(weights),
    }
    all_diagnostics = [*logical_diagnostics, *b_diagnostics, *scalar_diagnostics.values()]
    nondegenerate_ess = [row["bulk_ess"] for row in all_diagnostics if not row["degenerate"]]
    max_rhat = max(row["split_rhat"] for row in all_diagnostics)
    min_ess = min(nondegenerate_ess) if nondegenerate_ess else float("inf")
    dense = b_diagnostics[b_set.dense_start:]
    dense_nondegenerate = sum(not row["degenerate"] for row in dense)
    failures = []
    if not math.isfinite(qtop["q_top_total_se"]) or qtop["q_top_total_se"] > gates["max_q_top_se"]:
        failures.append("q_top_se")
    if max_rhat > gates["max_rhat"]:
        failures.append("rhat")
    if min_ess < gates["min_bulk_ess"]:
        failures.append("bulk_ess")
    if dense_nondegenerate < gates["min_dense_b_characters_nondegenerate"]:
        failures.append("b_dense_characters_uninformative")
    if any(row["burn_changes"] < gates["min_burn_column_changes_per_trajectory"] for row in records):
        failures.append("burn_column_changes")
    if any(row["measurement_changes"] < gates["min_measurement_column_changes_per_trajectory"] for row in records):
        failures.append("measurement_column_changes")
    if any(row["label_changes"] < gates["min_measurement_label_changes_per_trajectory"] for row in records):
        failures.append("measurement_label_changes")
    return {
        "b_dense_nondegenerate": dense_nondegenerate,
        "b_likelihood_mean_per_factor": likelihood_mean,
        "b_likelihood_mean_per_factor_se": likelihood_se,
        "b_weight_mean_normalized": b_weight_mean,
        "b_weight_mean_normalized_se": b_weight_se,
        "collision_q_top_diagnostic": _collision_diagnostic(label_chains),
        "failures": failures,
        "max_rhat": float(max_rhat),
        "min_nondegenerate_bulk_ess": float(min_ess),
        "normalized_weight_mean": weight_mean,
        "normalized_weight_mean_se": weight_se,
        "q_top": float(qtop["q_top"]),
        "q_top_character_se": float(qtop["q_top_character_se"]),
        "q_top_total_se": float(qtop["q_top_total_se"]),
        "q_top_trajectory_se": float(qtop["q_top_trajectory_se"]),
        "transition_counts": [{
            "burn_column_changes": row["burn_changes"],
            "index": row["index"],
            "measurement_column_changes": row["measurement_changes"],
            "measurement_label_changes": row["label_changes"],
        } for row in records],
        "valid": not failures,
        "_b_means": b_means,
        "_b_likelihood_trajectory": likelihood_trajectory,
        "_b_weight_trajectory": b_weight_trajectory,
        "_logical_means": logical_means,
        "_records": records,
        "_weight_trajectory": weight_trajectory,
    }


def _u_square(means):
    means = np.asarray(means, dtype=np.float64)
    return (
        np.square(means.sum(axis=0)) - np.square(means).sum(axis=0)
    ) / (means.shape[0] * (means.shape[0] - 1))


def _b_d2(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    def estimate(a, b):
        return float(np.mean(
            _u_square(a) + _u_square(b) - 2.0 * a.mean(axis=0) * b.mean(axis=0)
        ))

    value = estimate(left, right)
    delete_left = np.asarray([
        estimate(np.delete(left, index, axis=0), right) for index in range(left.shape[0])
    ])
    delete_right = np.asarray([
        estimate(left, np.delete(right, index, axis=0)) for index in range(right.shape[0])
    ])
    variance = 0.0
    for delete in (delete_left, delete_right):
        variance += (
            (delete.size - 1) / delete.size
            * float(np.square(delete - delete.mean()).sum())
        )
    return value, math.sqrt(variance)


def _pooled_b_rhat(left, right, b_set):
    records = [*left["_records"], *right["_records"]]
    weight = np.stack([row["b_weights"] for row in records])
    likelihood = np.stack([row["b_likelihood"] for row in records])
    maximum = max(float(split_rhat(weight)), float(split_rhat(likelihood)))
    for start in range(0, b_set.size, 32):
        stop = min(start + 32, b_set.size)
        masks = b_set.masks_packed[start:stop]
        blocks = [
            1.0 - 2.0 * _b_character_bits(row["b_packed"], masks).astype(np.float64)
            for row in records
        ]
        values = np.stack(blocks)
        for index in range(values.shape[2]):
            if np.unique(values[:, :, index]).size > 1:
                maximum = max(maximum, float(split_rhat(values[:, :, index])))
    return maximum


def _pair_comparison(left_name, right_name, left, right, logical_set, b_set, context):
    gates = context["config"]["gates"]
    logical_d2 = character_d2_estimate(
        logical_set, left["_logical_means"], right["_logical_means"],
    )
    delta_q = abs(left["q_top"] - right["q_top"])
    delta_q_se = math.hypot(left["q_top_total_se"], right["q_top_total_se"])
    weight_delta = abs(left["normalized_weight_mean"] - right["normalized_weight_mean"])
    weight_se = math.hypot(
        left["normalized_weight_mean_se"], right["normalized_weight_mean_se"],
    )
    b_weight_delta = abs(
        left["b_weight_mean_normalized"] - right["b_weight_mean_normalized"]
    )
    b_weight_se = math.hypot(
        left["b_weight_mean_normalized_se"], right["b_weight_mean_normalized_se"],
    )
    likelihood_delta = abs(
        left["b_likelihood_mean_per_factor"] - right["b_likelihood_mean_per_factor"]
    )
    likelihood_se = math.hypot(
        left["b_likelihood_mean_per_factor_se"],
        right["b_likelihood_mean_per_factor_se"],
    )
    b_d2, b_d2_se = _b_d2(left["_b_means"], right["_b_means"])
    pooled_b_rhat = _pooled_b_rhat(left, right, b_set)
    b_mean_delta = np.abs(
        left["_b_means"].mean(axis=0) - right["_b_means"].mean(axis=0)
    )
    b_mean_se = np.sqrt(
        left["_b_means"].var(axis=0, ddof=1) / left["_b_means"].shape[0]
        + right["_b_means"].var(axis=0, ddof=1) / right["_b_means"].shape[0]
    )
    b_character_pass = bool(np.all(
        (b_mean_delta <= gates["max_abs_b_character_mean_delta"])
        & (b_mean_delta <= gates["delta_sigma_multiplier"] * b_mean_se
                           + gates["b_character_delta_sigma_slack"])
    ))
    checks = {
        "b_character_d2": max(0.0, b_d2) + 3.0 * b_d2_se
        <= gates["max_b_character_d2_upper"],
        "b_character_means": b_character_pass,
        "b_likelihood": likelihood_delta <= gates["max_b_log_likelihood_delta_per_factor"]
        and likelihood_delta <= 3.0 * likelihood_se + 1.0 / context["H"].shape[1],
        "b_weight": b_weight_delta <= gates["max_b_normalized_weight_delta"]
        and b_weight_delta <= 3.0 * b_weight_se + 1.0 / (context["H"].shape[0] ** 2),
        "logical_d2": max(0.0, logical_d2["d2_norm"])
        + 3.0 * logical_d2["d2_total_se"] <= gates["max_d2_upper"],
        "pooled_b_rhat": pooled_b_rhat <= gates["max_rhat"],
        "q_top": delta_q <= gates["max_abs_delta_q_top"]
        and delta_q <= gates["delta_sigma_multiplier"] * delta_q_se
        + gates["delta_sigma_slack"],
        "weight": weight_delta <= gates["max_normalized_weight_delta"]
        and weight_delta <= 3.0 * weight_se
        + gates["normalized_weight_sigma_slack_qubits"] / context["model"].num_qubits,
    }
    return {
        "b_character_d2": b_d2,
        "b_character_d2_se": b_d2_se,
        "b_character_failed_count": int(np.count_nonzero(
            ~((b_mean_delta <= gates["max_abs_b_character_mean_delta"])
              & (b_mean_delta <= gates["delta_sigma_multiplier"] * b_mean_se
                 + gates["b_character_delta_sigma_slack"]))
        )),
        "b_likelihood_delta_per_factor": likelihood_delta,
        "b_likelihood_delta_per_factor_se": likelihood_se,
        "b_weight_delta_normalized": b_weight_delta,
        "b_weight_delta_normalized_se": b_weight_se,
        "checks": checks,
        "left": left_name,
        "logical_d2": float(logical_d2["d2_norm"]),
        "logical_d2_total_se": float(logical_d2["d2_total_se"]),
        "pooled_b_rhat": pooled_b_rhat,
        "q_top_delta": delta_q,
        "q_top_delta_se": delta_q_se,
        "right": right_name,
        "valid": all(checks.values()),
        "weight_delta_normalized": weight_delta,
        "weight_delta_normalized_se": weight_se,
    }


def _constant_b_freeze_failures(all_records, b_set):
    measurement = np.concatenate([row["b_packed"] for row in all_records], axis=0)
    values = 1 - 2 * _b_character_bits(measurement, b_set.masks_packed).astype(np.int8)
    failures = []
    for character in range(b_set.size):
        unique = np.unique(values[:, character])
        if unique.size != 1:
            continue
        common = int(unique[0])
        mask = b_set.masks_packed[character:character + 1]
        for row in all_records:
            initial = int(1 - 2 * _b_character_bits(
                row["initial_b_packed"][None, :], mask,
            ).astype(np.int8)[0, 0])
            if initial == common:
                continue
            burn = 1 - 2 * _b_character_bits(row["burn_b_packed"], mask).astype(np.int8)
            if not np.any(burn[:, 0] == common):
                failures.append({
                    "character": character, "family": row["family"], "index": row["index"],
                })
    return failures


def _map_bridge_gate(family_summaries, context):
    gates = context["config"]["gates"]
    fixed_b = context["arrays"]["fixed_b_blocks"]
    map0 = fixed_b[1]
    map1 = fixed_b[2]
    result = {}
    for family, source, target in (("M0", map0, map1), ("M1", map1, map0)):
        counts = []
        for row in family_summaries[family]["_records"]:
            combined = np.concatenate((row["burn_b_packed"], row["b_packed"]), axis=0)
            bits = _unpack(combined, map0.size).reshape(-1, *map0.shape)
            source_distance = np.count_nonzero(bits ^ source[None, :, :], axis=(1, 2))
            target_distance = np.count_nonzero(bits ^ target[None, :, :], axis=(1, 2))
            counts.append(int(np.count_nonzero(target_distance < source_distance)))
        result[family] = {
            "aggregate_opposite_basin_visits": int(sum(counts)),
            "opposite_basin_visits_per_trajectory": counts,
            "trajectories_visiting_opposite_basin": int(sum(value > 0 for value in counts)),
            "valid": sum(counts) >= gates["min_aggregate_opposite_map_basin_visits"]
            and sum(value > 0 for value in counts)
            >= gates["min_trajectories_visiting_opposite_map_basin"],
        }
    return result


def _public_family(summary):
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    _require(not output.exists(), "analysis output already exists")
    config, config_sha = _load_config()
    context = _load_control(run_root / "control", config, config_sha)
    schedule = _load_canonical_json(run_root / "control/schedule.json")
    _verify_self_hash(schedule, "schedule_sha256")
    preflight = _load_canonical_json(run_root / "preflight/aggregate.json")
    _verify_self_hash(preflight, "preflight_sha256")
    _require(preflight["status"] == "PASS" and preflight["exact_consensus"] is True
             and preflight["schedule_sha256"] == schedule["schedule_sha256"],
             "measurement lacks PASS aggregate preflight")
    task_by_fingerprint = {task["task_fingerprint"]: task for task in schedule["tasks"]}
    raw_by_fingerprint = {}
    node_reports = {}
    for node in config["resource"]["allowed_nodes"]:
        report_path = run_root / f"measurement/{node}/node_report.json"
        report = _load_canonical_json(report_path)
        _verify_self_hash(report, "node_report_sha256")
        _require(report["node_report_version"] == NODE_REPORT_VERSION
                 and report["node"] == node and report["status"] == "COMPLETE"
                 and report["schedule_sha256"] == schedule["schedule_sha256"],
                 "node report identity changed")
        node_reports[node] = report["node_report_sha256"]
        for record in report["raw_records"]:
            path = run_root / f"measurement/{node}/raw/{record['file']}"
            _require(sha256_file(path) == record["raw_sha256"], "node raw hash mismatch")
            fingerprint = record["task_fingerprint"]
            _require(fingerprint not in raw_by_fingerprint, "duplicate raw fingerprint")
            raw_by_fingerprint[fingerprint] = path
    _require(set(raw_by_fingerprint) == set(task_by_fingerprint),
             "raw task set is incomplete")
    mass = build_classical_coset_mass(context["H"], 0.04, engine="numba")
    log_mass = np.log(mass)
    records = []
    for task in schedule["tasks"]:
        records.append(_load_and_verify_raw(
            raw_by_fingerprint[task["task_fingerprint"]], task, context, schedule, log_mass,
        ))
    logical_set = CharacterSet(
        masks=context["arrays"]["logical_character_masks"],
        basis_positions=context["arrays"]["logical_basis_positions"],
        tier="sampled", k=context["model"].k,
        random_seed=context["metadata"]["logical_character_seed"],
        character_sha256=context["metadata"]["logical_character_sha256"],
    )
    b_set = BCharacterSet(
        masks_packed=context["arrays"]["b_character_masks_packed"],
        r=context["H"].shape[0],
        dense_count=config["statistics"]["b_dense_character_count"],
        random_seed=context["metadata"]["b_character_seed"],
        character_sha256=context["metadata"]["b_character_sha256"],
    )
    family_summaries = {
        family: _family_summary(
            [row for row in records if row["family"] == family],
            context, logical_set, b_set,
        ) for family in FAMILIES
    }
    comparisons = []
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1:]:
            comparisons.append(_pair_comparison(
                left, right, family_summaries[left], family_summaries[right],
                logical_set, b_set, context,
            ))
    constant_failures = _constant_b_freeze_failures(records, b_set)
    bridge = _map_bridge_gate(family_summaries, context)
    checks = {
        "all_families": all(summary["valid"] for summary in family_summaries.values()),
        "all_pairwise_comparisons": all(row["valid"] for row in comparisons),
        "constant_b_freeze": not constant_failures,
        "map_bridge": all(row["valid"] for row in bridge.values()),
        "raw_identity_and_algebra": True,
    }
    status = (
        "DIAGNOSTIC_RFCG_T1_M8_VIABLE"
        if all(checks.values()) else "UNRESOLVED_RFCG_T1_M8"
    )
    raw_paths = sorted(raw_by_fingerprint.values(), key=lambda path: path.as_posix())
    raw_set_sha = hashlib.sha256("".join(
        f"{path.relative_to(run_root).as_posix()}:{sha256_file(path)}\n"
        for path in raw_paths
    ).encode("ascii")).hexdigest()
    core = {
        "checks": checks,
        "comparisons": comparisons,
        "config_sha256": config_sha,
        "constant_b_freeze_failures": constant_failures,
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "families": {family: _public_family(summary)
                     for family, summary in family_summaries.items()},
        "map_bridge": bridge,
        "node_report_sha256": node_reports,
        "preflight_sha256": preflight["preflight_sha256"],
        "raw_count": len(raw_paths),
        "raw_set_sha256": raw_set_sha,
        "report_version": REPORT_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "scope": config["scope"],
        "source_identity": schedule["source_identity"],
        "status": status,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(output, report)
    print(canonical_json({
        "checks": checks,
        "report_sha256": report["report_sha256"],
        "status": status,
    }))


if __name__ == "__main__":
    main()
