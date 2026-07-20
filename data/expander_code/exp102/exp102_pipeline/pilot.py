"""Deterministic pilot scheduling, raw selection, and fail-closed freezing."""

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from .config import PILOT_CANDIDATE_FIELDS, load_config, validate_pilot_candidate
from .diagnostics import evaluate_gate
from .io import atomic_json, canonical_json, sha256_file, sha256_json
from .pilot_cell import _gate, pilot_task_identity
from .registry import load_registry
from .seeds import derive_seed


TUNING_P_VALUES = (0.04, 0.07, 0.10)
PILOT_STAGES = ("ladder", "gamma", "rounds", "held_out")
CANDIDATE_FIELDS = PILOT_CANDIDATE_FIELDS
PILOT_RAW_FIELDS = {
    "task_fingerprint", "namespace", "stage", "code_id", "m", "p",
    "disorder_index", "candidate_json", "attempt", "valid", "failure_reason",
    "labels", "swap_attempts", "swap_accepts", "swap_rates",
    "logical_attempts", "logical_accepts", "logical_rates", "round_trips",
    "sector_changing_round_trips", "residual", "rhat", "ess",
    "constant_status", "core_seconds", "wall_seconds", "engine",
    "source_commit", "model_fingerprint", "registry_sha256", "config_sha256",
    "section_fingerprint", "logical_frame_fingerprint",
}


def _validate_status_raw(data, task):
    """Reject readable pilot fragments before reporting operational completion."""
    if set(data.files) != PILOT_RAW_FIELDS:
        raise ValueError("pilot status raw schema mismatch")
    fingerprint = sha256_json(task)
    expected_scalars = {
        "task_fingerprint": fingerprint,
        **{field: task[field] for field in (
            "namespace", "stage", "code_id", "p", "disorder_index", "engine",
            "source_commit", "registry_sha256", "config_sha256",
        )},
        "candidate_json": _candidate_key(task["candidate"]),
    }
    for field, expected in expected_scalars.items():
        if str(_scalar(data, field)) != str(expected):
            raise ValueError(f"pilot status raw identity mismatch: {field}")
    if int(_scalar(data, "attempt")) < 0 or int(_scalar(data, "m")) not in range(3, 9):
        raise ValueError("pilot status raw m/attempt is invalid")
    candidate = task["candidate"]
    instances, temperatures = 4, int(candidate["num_temperatures"])
    measurements = int(candidate["measurement_rounds"])
    labels = _array(data, "labels", (instances, measurements), np.dtype(np.uint64))
    swap_attempts = _array(data, "swap_attempts", (instances, temperatures - 1))
    swap_accepts = _array(data, "swap_accepts", swap_attempts.shape)
    logical_attempts = data["logical_attempts"].copy()
    if (logical_attempts.ndim != 3 or logical_attempts.shape[:2] != (instances, temperatures)
            or logical_attempts.shape[2] <= 0 or logical_attempts.shape[2] > 64):
        raise ValueError("pilot status logical attempts have the wrong shape")
    k = logical_attempts.shape[2]
    logical_accepts = _array(data, "logical_accepts", logical_attempts.shape)
    for attempts, accepts in ((swap_attempts, swap_accepts),
                              (logical_attempts, logical_accepts)):
        if (attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu"
                or np.any(attempts < 0) or np.any(accepts < 0)
                or np.any(accepts > attempts)):
            raise ValueError("pilot status counters are invalid")
    if not np.array_equal(
            _array(data, "swap_rates", swap_attempts.shape),
            swap_accepts / np.maximum(swap_attempts, 1)):
        raise ValueError("pilot status swap rates disagree with counters")
    if not np.array_equal(
            _array(data, "logical_rates", logical_attempts.shape),
            logical_accepts / np.maximum(logical_attempts, 1)):
        raise ValueError("pilot status logical rates disagree with counters")
    for field in ("round_trips", "sector_changing_round_trips", "residual"):
        values = _array(data, field, (instances,))
        if values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"pilot status {field} is invalid")
    _array(data, "rhat", (k,)); _array(data, "ess", (k,))
    if _array(data, "constant_status", (k,)).dtype.kind != "U":
        raise ValueError("pilot status constant status dtype is invalid")
    for field in ("failure_reason", "model_fingerprint", "section_fingerprint",
                  "logical_frame_fingerprint"):
        _scalar(data, field)
    for field in ("core_seconds", "wall_seconds"):
        value = float(_scalar(data, field))
        if not np.isfinite(value) or value < 0:
            raise ValueError("pilot status timing is invalid")
    return fingerprint, bool(_scalar(data, "valid"))


def pilot_status(expected_manifest, raw_dir):
    manifest = json.loads(Path(expected_manifest).read_text(encoding="ascii"))
    raw_dir = Path(raw_dir)
    counts = {key: 0 for key in ("expected", "computed", "reused", "invalid", "missing", "conflict")}
    fingerprints = defaultdict(list)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("pilot status requires a deployment manifest with concrete tasks")
    counts["expected"] = len(tasks)
    expected = {sha256_json(task): task for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("pilot deployment manifest contains duplicate tasks")
    for path in sorted(raw_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                claimed = str(_scalar(data, "task_fingerprint"))
                if claimed not in expected:
                    raise ValueError("unexpected pilot task fingerprint")
                fingerprint, valid = _validate_status_raw(data, expected[claimed])
        except Exception:
            counts["conflict"] += 1
            continue
        fingerprints[fingerprint].append({
            "path": str(path), "sha256": sha256_file(path), "valid": valid,
        })
    files = {}
    for fingerprint in expected:
        matches = fingerprints.get(fingerprint, [])
        if not matches:
            counts["missing"] += 1
            continue
        hashes = {item["sha256"] for item in matches}
        if len(hashes) > 1:
            counts["conflict"] += 1
            continue
        if not all(item["valid"] for item in matches):
            counts["invalid"] += 1
        else:
            counts["reused" if len(matches) > 1 else "computed"] += 1
        files[fingerprint] = sorted(item["path"] for item in matches)
    return {"pilot_status_version": "exp102.pilot.status.v1", **counts, "files": files}


def pilot_schedule(registry_path, config_path, output_path):
    registry = load_registry(registry_path)
    config = load_config(config_path)
    tuning = []
    for code in registry["codes"]:
        for p in TUNING_P_VALUES:
            for disorder in range(4):
                tuning.append({"stage": "tuning", "namespace": "pilot_tuning_v1",
                               "code_id": code["code_id"], "p": p, "disorder_index": disorder})
    held_out = []
    for code in registry["codes"]:
        for p in config["p_values"]:
            for disorder in range(8):
                held_out.append({"stage": "held_out", "namespace": "pilot_held_out_v1",
                                 "code_id": code["code_id"], "p": p, "disorder_index": disorder})
    candidates = [{"p_hot": ladder["p_hot"],
                   "num_temperatures": ladder["num_temperatures"], "gamma": gamma,
                   "burn_rounds": rounds[0], "measurement_rounds": rounds[1],
                   "sweeps_per_round": 1, "logical_move_repeat": 1}
                  for ladder, gamma, rounds in product(
                      config["pilot"]["ladder_candidates"],
                      config["pilot"]["gamma_candidates"],
                      config["pilot"]["round_candidates"])]
    schedule = {"pilot_schedule_version": "exp102.pilot.v1",
                "registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
                "selection_policy": "ordered_ladder_pairs_then_min_core_time_gamma_then_raise_round_budget",
                "candidates": candidates, "tuning_tasks": tuning, "held_out_tasks": held_out}
    atomic_json(output_path, schedule)
    return schedule


def _scalar(data, field):
    if field not in data or data[field].shape != ():
        raise ValueError(f"pilot raw field {field!r} must be a scalar")
    return data[field].item()


def _candidate_key(candidate):
    return canonical_json(candidate)


def _candidate(config, p_hot, num_temperatures, gamma, rounds):
    return {
        "p_hot": float(p_hot), "num_temperatures": int(num_temperatures),
        "gamma": float(gamma), "burn_rounds": int(rounds[0]),
        "measurement_rounds": int(rounds[1]), "sweeps_per_round": 1,
        "logical_move_repeat": 1,
    }


def _array(data, field, shape, dtype=None):
    if field not in data or data[field].shape != shape:
        raise ValueError(f"pilot raw field {field!r} has the wrong shape")
    value = data[field].copy()
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"pilot raw field {field!r} has the wrong dtype")
    return value


def _validate_raw(path, registry, config, expected_source_commit):
    """Load one pilot NPZ and independently recompute every stored gate result."""
    path = Path(path)
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    try:
        data_context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot read pilot raw {path}: {exc}") from exc
    with data_context as data:
        stage = str(_scalar(data, "stage"))
        if stage not in PILOT_STAGES:
            raise ValueError(f"unknown pilot stage in {path}")
        code_id = str(_scalar(data, "code_id"))
        if code_id not in code_by_id:
            raise ValueError(f"unknown pilot code_id in {path}")
        code = code_by_id[code_id]
        m = int(_scalar(data, "m"))
        p = float(_scalar(data, "p"))
        disorder = int(_scalar(data, "disorder_index"))
        attempt = int(_scalar(data, "attempt"))
        if m != int(code["m"]) or attempt < 0:
            raise ValueError(f"pilot m/attempt identity mismatch in {path}")
        expected_p = config["p_values"] if stage == "held_out" else TUNING_P_VALUES
        max_disorder = 8 if stage == "held_out" else 4
        if p not in expected_p or not 0 <= disorder < max_disorder:
            raise ValueError(f"pilot cell lies outside the {stage} schedule in {path}")

        try:
            candidate_unchecked = json.loads(str(_scalar(data, "candidate_json")))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid candidate JSON in {path}") from exc
        candidate = validate_pilot_candidate(candidate_unchecked, config)
        if str(_scalar(data, "candidate_json")) != _candidate_key(candidate):
            raise ValueError(f"non-canonical candidate JSON in {path}")
        namespace = (f"pilot_held_out_m{m}_attempt{attempt}" if stage == "held_out"
                     else f"pilot_{stage}_m{m}_attempt{attempt}")
        scalar_expected = {
            "namespace": namespace, "engine": "numba", "source_commit": expected_source_commit,
            "registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
            "section_fingerprint": code["section_fingerprint"],
            "logical_frame_fingerprint": code["logical_frame_fingerprint"],
            "model_fingerprint": sha256_json({"n": code["n"], "k": code["k"]}),
        }
        for field, expected in scalar_expected.items():
            if str(_scalar(data, field)) != str(expected):
                raise ValueError(f"pilot raw identity mismatch in {path}: {field}")
        identity = pilot_task_identity(
            registry["registry_sha256"], config["config_sha256"], code_id, m,
            p, disorder, candidate, attempt, stage, expected_source_commit,
        )
        fingerprint = sha256_json(identity)
        if str(_scalar(data, "task_fingerprint")) != fingerprint:
            raise ValueError(f"pilot task fingerprint mismatch in {path}")

        instances, k = 4, int(code["k"])
        temperatures = candidate["num_temperatures"]
        measurements = candidate["measurement_rounds"]
        labels = _array(data, "labels", (instances, measurements), np.dtype(np.uint64))
        swap_attempts = _array(data, "swap_attempts", (instances, temperatures - 1))
        swap_accepts = _array(data, "swap_accepts", (instances, temperatures - 1))
        logical_attempts = _array(data, "logical_attempts", (instances, temperatures, k))
        logical_accepts = _array(data, "logical_accepts", (instances, temperatures, k))
        round_trips = _array(data, "round_trips", (instances,))
        changing_round_trips = _array(data, "sector_changing_round_trips", (instances,))
        residual = _array(data, "residual", (instances,))
        for field, attempts, accepts in (
                ("swap", swap_attempts, swap_accepts),
                ("logical", logical_attempts, logical_accepts)):
            if attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu":
                raise ValueError(f"pilot {field} counters are not integer arrays in {path}")
            if np.any(attempts < 0) or np.any(accepts < 0) or np.any(accepts > attempts):
                raise ValueError(f"pilot {field} counters are invalid in {path}")
        if any(value.dtype.kind not in "iu" for value in (round_trips, changing_round_trips, residual)):
            raise ValueError(f"pilot diagnostic counters are not integers in {path}")
        if np.any(round_trips < 0) or np.any(changing_round_trips < 0) or np.any(residual < 0):
            raise ValueError(f"pilot diagnostic counters are negative in {path}")
        stored_swap_rates = _array(data, "swap_rates", swap_attempts.shape)
        stored_logical_rates = _array(data, "logical_rates", logical_attempts.shape)
        if not np.array_equal(stored_swap_rates, swap_accepts / np.maximum(swap_attempts, 1)):
            raise ValueError(f"pilot swap rates were not derived from counters in {path}")
        if not np.array_equal(stored_logical_rates, logical_accepts / np.maximum(logical_attempts, 1)):
            raise ValueError(f"pilot logical rates were not derived from counters in {path}")

        results = []
        for instance in range(instances):
            seed = derive_seed(namespace, registry["registry_sha256"], code_id, disorder,
                               f"p={p:.8f}", instance)
            results.append({
                "labels": labels[instance], "swap_attempts": swap_attempts[instance],
                "swap_accepts": swap_accepts[instance], "logical_attempts": logical_attempts[instance],
                "logical_accepts": logical_accepts[instance], "round_trips": int(round_trips[instance]),
                "sector_changing_round_trips": int(changing_round_trips[instance]),
                "max_hard_coset_residual": int(residual[instance]), "seed": seed,
            })
        valid, failures, rhats, esses, statuses = evaluate_gate(
            results, _gate(config, stage), k,
            require_trace_gate=stage not in {"ladder", "gamma"},
        )
        if bool(_scalar(data, "valid")) != valid:
            raise ValueError(f"pilot stored validity disagrees with recomputed gates in {path}")
        if str(_scalar(data, "failure_reason")) != ";".join(failures):
            raise ValueError(f"pilot stored failures disagree with recomputed gates in {path}")
        stored_rhat = _array(data, "rhat", (k,))
        stored_ess = _array(data, "ess", (k,))
        stored_status = _array(data, "constant_status", (k,))
        if not np.array_equal(stored_rhat, rhats, equal_nan=True):
            raise ValueError(f"pilot stored R-hat disagrees with recomputation in {path}")
        if not np.array_equal(stored_ess, esses, equal_nan=True):
            raise ValueError(f"pilot stored ESS disagrees with recomputation in {path}")
        if not np.array_equal(stored_status, statuses):
            raise ValueError(f"pilot constant status disagrees with recomputation in {path}")
        core_seconds = float(_scalar(data, "core_seconds"))
        wall_seconds = float(_scalar(data, "wall_seconds"))
        if not np.isfinite(core_seconds) or not np.isfinite(wall_seconds) or core_seconds < 0 or wall_seconds < 0:
            raise ValueError(f"pilot timing is invalid in {path}")
    return {
        "path": str(path.resolve()), "task_fingerprint": fingerprint, "stage": stage,
        "code_id": code_id, "m": m, "p": p, "disorder_index": disorder,
        "attempt": attempt, "candidate": candidate, "candidate_key": _candidate_key(candidate),
        "valid": bool(valid), "failure_reason": ";".join(failures),
        "core_seconds": core_seconds, "wall_seconds": wall_seconds,
    }


def _read_source_commit(path):
    with np.load(path, allow_pickle=False) as data:
        return str(_scalar(data, "source_commit"))


def _load_records(paths, registry, config, expected_source_commit=None):
    paths = sorted(Path(path).resolve() for path in paths)
    if not paths:
        raise ValueError("pilot merge-select has no raw NPZ evidence")
    source_commits = {_read_source_commit(path) for path in paths}
    if len(source_commits) != 1:
        raise ValueError("pilot raw mixes source commits")
    source_commit = source_commits.pop()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("pilot raw source commit must be a full lowercase Git SHA")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ValueError("pilot raw source commit differs from the report")
    records = [_validate_raw(path, registry, config, source_commit) for path in paths]
    fingerprints = [record["task_fingerprint"] for record in records]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("pilot evidence contains duplicate task fingerprints")
    cell_keys = [(record["stage"], record["m"], record["attempt"], record["candidate_key"],
                  record["code_id"], record["p"], record["disorder_index"])
                 for record in records]
    if len(set(cell_keys)) != len(cell_keys):
        raise ValueError("pilot evidence contains duplicate logical cells")
    return records, source_commit


def _expected_cells(code_ids, p_values, disorders):
    return {(code_id, float(p), disorder)
            for code_id in code_ids for p in p_values for disorder in range(disorders)}


def _group_records(records, registry, config):
    code_ids_by_m = defaultdict(list)
    for code in registry["codes"]:
        code_ids_by_m[int(code["m"])].append(code["code_id"])
    groups = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["stage"], record["m"], record["attempt"], record["candidate_key"])].append(record)
    for key, rows in grouped.items():
        stage, m, attempt, candidate_key = key
        expected = _expected_cells(
            sorted(code_ids_by_m[m]), config["p_values"] if stage == "held_out" else TUNING_P_VALUES,
            8 if stage == "held_out" else 4,
        )
        actual = {(row["code_id"], row["p"], row["disorder_index"]) for row in rows}
        unexpected = actual - expected
        missing = expected - actual
        failure_counts = Counter(reason for row in rows for reason in row["failure_reason"].split(";") if reason)
        groups[key] = {
            "stage": stage, "m": m, "attempt": attempt,
            "candidate": json.loads(candidate_key), "candidate_key": candidate_key,
            "expected_cells": len(expected), "present_cells": len(actual),
            "valid_cells": sum(row["valid"] for row in rows),
            "missing_cells": len(missing), "unexpected_cells": len(unexpected),
            "complete": not missing and not unexpected,
            "all_pass": not missing and not unexpected and all(row["valid"] for row in rows),
            "core_seconds": float(sum(row["core_seconds"] for row in rows)),
            "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
            "failure_counts": dict(sorted(failure_counts.items())),
        }
    return groups


def _public_group(group):
    if group is None:
        return None
    return {key: value for key, value in group.items() if key not in {"candidate_key", "m"}}


def _candidate_group(groups, stage, m, candidate):
    """Return the sole raw group for a candidate, retaining partial attempts."""
    candidate_key = _candidate_key(candidate)
    matches = [group for key, group in groups.items()
               if key[0] == stage and key[1] == m and key[3] == candidate_key]
    if len(matches) > 1:
        return None, "duplicate_candidate_attempts"
    return (matches[0], None) if matches else (None, "missing")


def _pending_trial(candidate, reason="missing"):
    return {"candidate": candidate, "complete": False, "all_pass": False, "reason": reason}


def _select_one_m(m, groups, config):
    """Consume the adaptive pilot protocol in order for one code size.

    Evidence after a missing prefix is never selected.  A failed held-out
    attempt advances the round budget, and exhaustion of that budget advances
    to the next ordered ladder pair where gamma is selected again.
    """
    pilot = config["pilot"]
    base_rounds = pilot["round_candidates"][0]
    ladder_trials, gamma_trials, rounds_trials, held_out_trials, cycles = [], [], [], [], []
    active_ladder = active_gamma = active_rounds = None
    final_tuning_group = held_out_group = final_candidate = None

    held_groups = sorted((group for key, group in groups.items()
                          if key[0] == "held_out" and key[1] == m),
                         key=lambda group: (group["attempt"], group["candidate_key"]))
    by_attempt = defaultdict(list)
    for group in held_groups:
        by_attempt[group["attempt"]].append(group)
    if by_attempt:
        expected_attempts = set(range(max(by_attempt) + 1))
        if set(by_attempt) != expected_attempts:
            conflict = "held_out_attempt_gap"
        elif any(len(candidates) != 1 for candidates in by_attempt.values()):
            conflict = "ambiguous_held_out_attempt"
        else:
            conflict = None
    else:
        conflict = None
    held_attempt = 0

    def result(state, next_action=None, conflict_reason=None):
        return {
            "ladder": {"selected": active_ladder, "trials": ladder_trials},
            "gamma": {"selected": active_gamma, "trials": gamma_trials},
            "rounds": {"selected": active_rounds, "trials": rounds_trials},
            "held_out": {
                "selected_attempt": None if held_out_group is None else held_out_group["attempt"],
                "trials": held_out_trials,
            },
            "all_tuning_pass": bool(final_tuning_group is not None and final_tuning_group["all_pass"]),
            "all_held_out_pass": bool(held_out_group is not None and held_out_group["all_pass"]),
            "num_tuning_cells": 0 if final_tuning_group is None else final_tuning_group["present_cells"],
            "num_held_out_cells": 0 if held_out_group is None else held_out_group["present_cells"],
            "selected_config": final_candidate,
            "state": state, "next_action": next_action, "cycles": cycles,
            "conflict_reason": conflict_reason,
        }

    if conflict is not None:
        return result("CONFLICT", conflict_reason=conflict)

    tie_order = {1.0: 0, 0.75: 1, 1.5: 2}
    for ladder_index, ladder in enumerate(pilot["ladder_candidates"]):
        active_ladder = active_gamma = active_rounds = None
        final_tuning_group = final_candidate = None
        ladder_candidate = _candidate(
            config, ladder["p_hot"], ladder["num_temperatures"], 1.0, base_rounds,
        )
        cycle = {"ladder_index": ladder_index, "ladder_candidate": ladder_candidate,
                 "gamma_selected": None, "outcome": None}
        cycles.append(cycle)
        ladder_group, reason = _candidate_group(groups, "ladder", m, ladder_candidate)
        ladder_trials.append(_public_group(ladder_group) if ladder_group is not None
                             else _pending_trial(ladder_candidate, reason))
        if reason == "duplicate_candidate_attempts":
            cycle["outcome"] = "conflict"
            return result("CONFLICT", conflict_reason="duplicate_ladder_candidate")
        if ladder_group is None or not ladder_group["complete"]:
            cycle["outcome"] = "waiting_ladder"
            return result("WAITING_LADDER", {
                "stage": "ladder", "candidate": ladder_candidate,
                "reason": reason if ladder_group is None else "partial",
            })
        if not ladder_group["all_pass"]:
            cycle["outcome"] = "ladder_failed"
            continue
        active_ladder = ladder_candidate

        gamma_groups = []
        missing_gamma = []
        for gamma in pilot["gamma_candidates"]:
            candidate = _candidate(
                config, ladder["p_hot"], ladder["num_temperatures"], gamma, base_rounds,
            )
            group, reason = _candidate_group(groups, "gamma", m, candidate)
            gamma_trials.append(_public_group(group) if group is not None
                                else _pending_trial(candidate, reason))
            if reason == "duplicate_candidate_attempts":
                cycle["outcome"] = "conflict"
                return result("CONFLICT", conflict_reason="duplicate_gamma_candidate")
            if group is None or not group["complete"]:
                missing_gamma.append({"candidate": candidate,
                                      "reason": reason if group is None else "partial"})
            else:
                gamma_groups.append(group)
        if missing_gamma:
            cycle["outcome"] = "waiting_gamma"
            return result("WAITING_GAMMA", {"stage": "gamma", "candidates": missing_gamma})
        passing_gamma = [group for group in gamma_groups if group["all_pass"]]
        if not passing_gamma:
            cycle["outcome"] = "gamma_failed"
            continue
        gamma_group = min(passing_gamma, key=lambda group: (
            group["core_seconds"], tie_order[float(group["candidate"]["gamma"])]))
        active_gamma = gamma_group["candidate"]
        cycle["gamma_selected"] = active_gamma

        for round_index, rounds in enumerate(pilot["round_candidates"]):
            candidate = _candidate(
                config, ladder["p_hot"], ladder["num_temperatures"],
                active_gamma["gamma"], rounds,
            )
            round_group, reason = _candidate_group(groups, "rounds", m, candidate)
            rounds_trials.append(_public_group(round_group) if round_group is not None
                                 else _pending_trial(candidate, reason))
            if reason == "duplicate_candidate_attempts":
                cycle["outcome"] = "conflict"
                return result("CONFLICT", conflict_reason="duplicate_rounds_candidate")
            if round_group is None or not round_group["complete"]:
                active_rounds = None
                cycle["outcome"] = "waiting_rounds"
                return result("WAITING_ROUNDS", {
                    "stage": "rounds", "candidate": candidate,
                    "reason": reason if round_group is None else "partial",
                })
            if not round_group["all_pass"]:
                continue

            active_rounds = candidate
            final_tuning_group, final_candidate = round_group, candidate
            held_candidates = by_attempt.get(held_attempt, [])
            if not held_candidates:
                cycle["outcome"] = "waiting_held_out"
                return result("WAITING_HELD_OUT", {
                    "stage": "held_out", "attempt": held_attempt, "candidate": candidate,
                })
            held_group = held_candidates[0]
            if held_group["candidate_key"] != _candidate_key(candidate):
                cycle["outcome"] = "conflict"
                return result("CONFLICT", conflict_reason="held_out_candidate_mismatch")
            held_out_trials.append(_public_group(held_group))
            if not held_group["complete"]:
                cycle["outcome"] = "waiting_held_out"
                return result("WAITING_HELD_OUT", {
                    "stage": "held_out", "attempt": held_attempt,
                    "candidate": candidate, "reason": "partial",
                })
            held_attempt += 1
            if held_group["all_pass"]:
                if held_attempt != len(by_attempt):
                    cycle["outcome"] = "conflict"
                    return result("CONFLICT", conflict_reason="held_out_after_pass")
                held_out_group = held_group
                cycle["outcome"] = "passed"
                return result("PASSED")

            # This candidate is rejected by held-out evidence.  The next round
            # budget, or the next ladder pair after the maximum, must earn new
            # tuning evidence before another held-out attempt can be consumed.
            final_tuning_group = final_candidate = active_rounds = None
            cycle["outcome"] = ("held_out_failed" if round_index + 1 < len(pilot["round_candidates"])
                                else "held_out_exhausted")

        if cycle["outcome"] == "held_out_failed":
            cycle["outcome"] = "rounds_exhausted_after_held_out"
        elif cycle["outcome"] is None:
            cycle["outcome"] = "rounds_exhausted"

    active_ladder = active_gamma = active_rounds = None
    final_tuning_group = final_candidate = None
    if held_attempt != len(by_attempt):
        return result("CONFLICT", conflict_reason="unconsumed_held_out_attempt")
    return result("EXHAUSTED")


def _analyze_records(records, registry, config):
    groups = _group_records(records, registry, config)
    return {str(m): _select_one_m(m, groups, config) for m in range(3, 9)}


def _relative_evidence(paths, report_path):
    root = Path(report_path).resolve().parent
    evidence = []
    for path in sorted(Path(path).resolve() for path in paths):
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("pilot report must be written above all raw evidence") from exc
        evidence.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    return evidence


def _verified_stage_raw_paths(raw_dir):
    """Bind merge-select to the hash manifests emitted by completed stages."""
    raw_dir = Path(raw_dir).resolve()
    manifests = sorted(raw_dir.rglob("raw_manifest.json"))
    if not manifests:
        raise ValueError("pilot merge-select found no stage raw manifests")
    listed = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if set(manifest) != {
                "raw_manifest_version", "node", "stage", "attempt", "source_commit",
                "registry_sha256", "config_sha256", "files"}:
            raise ValueError(f"pilot raw manifest schema mismatch: {manifest_path}")
        if manifest["raw_manifest_version"] != "exp102.pilot.raw.v1":
            raise ValueError(f"pilot raw manifest version mismatch: {manifest_path}")
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"pilot raw manifest has no files: {manifest_path}")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError(f"pilot raw manifest file entry is invalid: {manifest_path}")
            path = (manifest_path.parent / item["path"]).resolve()
            if (manifest_path.parent.resolve() not in path.parents or path.suffix != ".npz"
                    or not path.is_file()):
                raise ValueError(f"pilot raw manifest path is invalid: {item['path']}")
            if path in listed:
                raise ValueError(f"pilot raw file is listed more than once: {path}")
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"pilot stage raw hash mismatch: {path}")
            listed[path] = item["sha256"]
    actual = {path.resolve() for path in raw_dir.rglob("*.npz")}
    if actual != set(listed):
        raise ValueError("pilot raw files differ from completed stage manifests")
    return sorted(listed)


def merge_select(raw_dir, registry_path, config_path, output_path):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    registry = load_registry(registry_path)
    config = load_config(config_path)
    paths = _verified_stage_raw_paths(raw_dir)
    records, source_commit = _load_records(paths, registry, config)
    report = {
        "report_version": "exp102.pilot.report.v1", "generated_by": "pilot.merge-select.v1",
        "registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
        "source_commit": source_commit, "engine": "numba",
        "registry_path": str(Path(registry_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "raw_evidence": _relative_evidence(paths, output_path),
        "by_m": _analyze_records(records, registry, config),
    }
    report["analysis_sha256"] = sha256_json(report["by_m"])
    atomic_json(output_path, report)
    return report


def _verify_report_evidence(report, report_path):
    if report.get("generated_by") != "pilot.merge-select.v1":
        raise ValueError("pilot report was not generated by merge-select")
    evidence = report.get("raw_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("pilot report has no raw evidence")
    root = Path(report_path).resolve().parent
    seen = set()
    paths = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or item["path"] in seen:
            raise ValueError("invalid or duplicate raw evidence entry")
        seen.add(item["path"])
        path = (root / item["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"raw evidence is missing or outside report tree: {item['path']}")
        if path.suffix != ".npz" or sha256_file(path) != item["sha256"]:
            raise ValueError(f"raw evidence hash mismatch: {item['path']}")
        paths.append(path)
    return paths


def _assert_report_matches_recomputed(report, by_m):
    if report.get("analysis_sha256") != sha256_json(by_m):
        raise ValueError("pilot report analysis digest does not match recomputed raw gates")
    if report.get("by_m") != by_m:
        raise ValueError("pilot report selection or pass fields were edited after merge-select")


def recompute_frozen(report_path, registry_path=None, config_path=None):
    report = json.loads(Path(report_path).read_text(encoding="ascii"))
    if report.get("report_version") != "exp102.pilot.report.v1":
        raise ValueError("wrong pilot report version")
    paths = _verify_report_evidence(report, report_path)
    registry_path = registry_path or report.get("registry_path")
    config_path = config_path or report.get("config_path")
    if not registry_path or not config_path:
        raise ValueError("pilot report has no registry/config verification context")
    registry = load_registry(registry_path)
    config = load_config(config_path)
    required_identity = {"registry_sha256", "config_sha256", "source_commit", "engine"}
    if not required_identity <= report.keys() or report.get("engine") != "numba":
        raise ValueError("pilot report production identity is incomplete")
    if report["registry_sha256"] != registry["registry_sha256"] or report["config_sha256"] != config["config_sha256"]:
        raise ValueError("pilot report registry/config identity mismatch")
    records, source_commit = _load_records(paths, registry, config, report["source_commit"])
    if source_commit != report["source_commit"]:
        raise ValueError("pilot report source identity mismatch")
    by_m = _analyze_records(records, registry, config)
    _assert_report_matches_recomputed(report, by_m)
    if set(by_m) != {str(m) for m in range(3, 9)}:
        raise ValueError("pilot report must cover m=3..8")

    frozen = {
        "status": "FROZEN_HELD_OUT_PASS", "pilot_report_sha256": sha256_json(report),
        "raw_evidence_sha256": sha256_json(report["raw_evidence"]),
        **{key: report[key] for key in required_identity}, "by_m": {},
        "held_out_attempt_by_m": {},
    }
    for m, record in by_m.items():
        if (record.get("state") != "PASSED" or not record["all_tuning_pass"]
                or not record["all_held_out_pass"]):
            raise ValueError(f"m={m} did not pass both recomputed pilot stages")
        if record["num_tuning_cells"] != 96 or record["num_held_out_cells"] != 448:
            raise ValueError(f"m={m} recomputed pilot cell count is incomplete")
        selected = record.get("selected_config")
        if not isinstance(selected, dict) or set(selected) != CANDIDATE_FIELDS:
            raise ValueError(f"m={m} selected config fields are incomplete")
        frozen["by_m"][m] = selected
        frozen["held_out_attempt_by_m"][m] = int(record["held_out"]["selected_attempt"])
    return frozen


def freeze_from_report(report_path, output_path, registry_path=None, config_path=None):
    frozen = recompute_frozen(report_path, registry_path, config_path)
    atomic_json(output_path, frozen)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("registry"); schedule.add_argument("config"); schedule.add_argument("output")
    merge = sub.add_parser("merge-select")
    merge.add_argument("raw_dir"); merge.add_argument("registry"); merge.add_argument("config"); merge.add_argument("output")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("report"); freeze.add_argument("output")
    freeze.add_argument("--registry"); freeze.add_argument("--config")
    status = sub.add_parser("status")
    status.add_argument("manifest"); status.add_argument("raw_dir")
    args = parser.parse_args(argv)
    if args.command == "schedule":
        result = pilot_schedule(args.registry, args.config, args.output)
    elif args.command == "merge-select":
        result = merge_select(args.raw_dir, args.registry, args.config, args.output)
    elif args.command == "freeze":
        result = freeze_from_report(args.report, args.output, args.registry, args.config)
    else:
        result = pilot_status(args.manifest, args.raw_dir)
    print(sha256_json(result))


if __name__ == "__main__":
    main()
