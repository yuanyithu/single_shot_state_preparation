"""Deterministic pilot scheduling, raw selection, and fail-closed freezing."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from .config import load_config
from .diagnostics import evaluate_gate
from .io import atomic_json, canonical_json, sha256_file, sha256_json
from .pilot_cell import _gate
from .registry import load_registry
from .seeds import derive_seed


TUNING_P_VALUES = (0.04, 0.07, 0.10)
PILOT_STAGES = ("ladder", "gamma", "rounds", "held_out")
CANDIDATE_FIELDS = {
    "p_hot", "num_temperatures", "gamma", "burn_rounds", "measurement_rounds",
    "sweeps_per_round", "logical_move_repeat",
}


def pilot_status(expected_manifest, raw_dir):
    manifest = json.loads(Path(expected_manifest).read_text(encoding="ascii"))
    raw_dir = Path(raw_dir)
    counts = {key: 0 for key in ("expected", "computed", "reused", "invalid", "missing", "conflict")}
    fingerprints = {}
    tasks = manifest.get("tasks", manifest.get("tuning_tasks", []))
    counts["expected"] = len(tasks)
    for task in tasks:
        fingerprint = sha256_json(task)
        matches = list(raw_dir.rglob(f"{fingerprint}*.npz"))
        if not matches:
            counts["missing"] += 1
            continue
        hashes = {sha256_file(path) for path in matches}
        if len(hashes) > 1:
            counts["conflict"] += 1
            continue
        with np.load(matches[0], allow_pickle=False) as data:
            if str(data["task_fingerprint"].item()) != fingerprint:
                counts["conflict"] += 1
            elif not bool(data["valid"].item()):
                counts["invalid"] += 1
            else:
                counts["reused" if len(matches) > 1 else "computed"] += 1
        fingerprints[fingerprint] = sorted(str(path) for path in matches)
    return {"pilot_status_version": "exp102.pilot.status.v1", **counts, "files": fingerprints}


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
    candidates = [{"p_hot": ph, "num_temperatures": r, "gamma": gamma,
                   "burn_rounds": rounds[0], "measurement_rounds": rounds[1],
                   "sweeps_per_round": 1, "logical_move_repeat": 1}
                  for ph, r, gamma, rounds in product(
                      config["pilot"]["p_hot_candidates"],
                      config["pilot"]["num_temperatures_candidates"],
                      config["pilot"]["gamma_candidates"],
                      config["pilot"]["round_candidates"])]
    schedule = {"pilot_schedule_version": "exp102.pilot.v1",
                "registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
                "selection_policy": "raise_p_hot_and_R_then_min_core_time_gamma_then_raise_round_budget",
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


def _validate_candidate(candidate, config):
    if set(candidate) != CANDIDATE_FIELDS:
        raise ValueError("pilot candidate fields are incomplete")
    normalized = {
        "p_hot": float(candidate["p_hot"]),
        "num_temperatures": int(candidate["num_temperatures"]),
        "gamma": float(candidate["gamma"]),
        "burn_rounds": int(candidate["burn_rounds"]),
        "measurement_rounds": int(candidate["measurement_rounds"]),
        "sweeps_per_round": int(candidate["sweeps_per_round"]),
        "logical_move_repeat": int(candidate["logical_move_repeat"]),
    }
    pilot = config["pilot"]
    if normalized["p_hot"] not in pilot["p_hot_candidates"]:
        raise ValueError("pilot candidate p_hot is outside the frozen schedule")
    if normalized["num_temperatures"] not in pilot["num_temperatures_candidates"]:
        raise ValueError("pilot candidate temperature count is outside the frozen schedule")
    if normalized["gamma"] not in pilot["gamma_candidates"]:
        raise ValueError("pilot candidate gamma is outside the frozen schedule")
    if [normalized["burn_rounds"], normalized["measurement_rounds"]] not in pilot["round_candidates"]:
        raise ValueError("pilot candidate round budget is outside the frozen schedule")
    if normalized["sweeps_per_round"] != 1 or normalized["logical_move_repeat"] != 1:
        raise ValueError("pilot candidate move counts differ from the frozen schedule")
    return normalized


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
        candidate = _validate_candidate(candidate_unchecked, config)
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
        identity = {
            "namespace": namespace, "stage": stage, "code_id": code_id, "p": p,
            "disorder_index": disorder, "candidate": candidate,
            "registry_sha256": registry["registry_sha256"],
            "config_sha256": config["config_sha256"], "source_commit": expected_source_commit,
            "engine": "numba",
        }
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


def _complete_group(groups, stage, m, candidate):
    candidate_key = _candidate_key(candidate)
    matches = [group for key, group in groups.items()
               if key[0] == stage and key[1] == m and key[3] == candidate_key and group["complete"]]
    if not matches:
        return None
    return min(matches, key=lambda group: group["attempt"])


def _public_group(group):
    if group is None:
        return None
    return {key: value for key, value in group.items() if key not in {"candidate_key", "m"}}


def _select_one_m(m, groups, config):
    pilot = config["pilot"]
    base_rounds = pilot["round_candidates"][0]
    ladder_trials = []
    ladder_group = None
    ladder_candidate = None
    for p_hot in pilot["p_hot_candidates"]:
        for temperatures in pilot["num_temperatures_candidates"]:
            candidate = _candidate(config, p_hot, temperatures, 1.0, base_rounds)
            group = _complete_group(groups, "ladder", m, candidate)
            ladder_trials.append(_public_group(group) if group is not None else {
                "candidate": candidate, "complete": False, "all_pass": False,
            })
            if group is None:
                break
            if group["all_pass"]:
                ladder_group, ladder_candidate = group, candidate
                break
        if ladder_group is not None or (ladder_trials and not ladder_trials[-1]["complete"]):
            break

    gamma_trials = []
    gamma_group = None
    gamma_candidate = None
    if ladder_candidate is not None:
        gamma_groups = []
        for gamma in pilot["gamma_candidates"]:
            candidate = _candidate(config, ladder_candidate["p_hot"],
                                   ladder_candidate["num_temperatures"], gamma, base_rounds)
            group = _complete_group(groups, "gamma", m, candidate)
            gamma_trials.append(_public_group(group) if group is not None else {
                "candidate": candidate, "complete": False, "all_pass": False,
            })
            if group is not None:
                gamma_groups.append(group)
        if len(gamma_groups) == len(pilot["gamma_candidates"]):
            passing = [group for group in gamma_groups if group["all_pass"]]
            tie_order = {1.0: 0, 0.75: 1, 1.5: 2}
            if passing:
                gamma_group = min(passing, key=lambda group: (
                    group["core_seconds"], tie_order[float(group["candidate"]["gamma"])]))
                gamma_candidate = gamma_group["candidate"]

    rounds_trials = []
    rounds_group = None
    rounds_candidate = None
    if gamma_candidate is not None:
        for rounds in pilot["round_candidates"]:
            candidate = _candidate(config, gamma_candidate["p_hot"],
                                   gamma_candidate["num_temperatures"], gamma_candidate["gamma"], rounds)
            group = _complete_group(groups, "rounds", m, candidate)
            rounds_trials.append(_public_group(group) if group is not None else {
                "candidate": candidate, "complete": False, "all_pass": False,
            })
            if group is None:
                break
            if group["all_pass"]:
                rounds_group, rounds_candidate = group, candidate
                break

    held_out_trials = []
    held_out_group = None
    final_tuning_group = rounds_group
    final_candidate = rounds_candidate
    held_groups = sorted((group for key, group in groups.items()
                          if key[0] == "held_out" and key[1] == m),
                         key=lambda group: (group["attempt"], group["candidate_key"]))
    by_attempt = defaultdict(list)
    for group in held_groups:
        by_attempt[group["attempt"]].append(group)
    expected_held_candidate = rounds_candidate
    if held_groups and expected_held_candidate is not None:
        max_attempt = max(by_attempt)
        for attempt in range(max_attempt + 1):
            candidates = by_attempt.get(attempt, [])
            if len(candidates) != 1:
                held_out_trials.append({"attempt": attempt, "complete": False, "all_pass": False,
                                        "reason": "missing_or_ambiguous_attempt"})
                break
            group = candidates[0]
            held_out_trials.append(_public_group(group))
            if not group["complete"]:
                break
            if group["candidate_key"] != _candidate_key(expected_held_candidate):
                break
            matching_tuning = _complete_group(groups, "rounds", m, group["candidate"])
            if matching_tuning is None or not matching_tuning["all_pass"]:
                break
            if group["all_pass"]:
                held_out_group = group
                final_tuning_group = matching_tuning
                final_candidate = group["candidate"]
                break
            round_index = next(
                index for index, rounds in enumerate(pilot["round_candidates"])
                if list(rounds) == [expected_held_candidate["burn_rounds"],
                                    expected_held_candidate["measurement_rounds"]]
            )
            if round_index + 1 >= len(pilot["round_candidates"]):
                break
            expected_held_candidate = _candidate(
                config, gamma_candidate["p_hot"], gamma_candidate["num_temperatures"],
                gamma_candidate["gamma"], pilot["round_candidates"][round_index + 1],
            )

    return {
        "ladder": {"selected": ladder_candidate, "trials": ladder_trials},
        "gamma": {"selected": gamma_candidate, "trials": gamma_trials},
        "rounds": {"selected": rounds_candidate, "trials": rounds_trials},
        "held_out": {"selected_attempt": None if held_out_group is None else held_out_group["attempt"],
                     "trials": held_out_trials},
        "all_tuning_pass": bool(final_tuning_group is not None and final_tuning_group["all_pass"]),
        "all_held_out_pass": bool(held_out_group is not None and held_out_group["all_pass"]),
        "num_tuning_cells": 0 if final_tuning_group is None else final_tuning_group["present_cells"],
        "num_held_out_cells": 0 if held_out_group is None else held_out_group["present_cells"],
        "selected_config": final_candidate,
    }


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


def merge_select(raw_dir, registry_path, config_path, output_path):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    registry = load_registry(registry_path)
    config = load_config(config_path)
    paths = sorted(raw_dir.rglob("*.npz"))
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


def freeze_from_report(report_path, output_path, registry_path=None, config_path=None):
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
    }
    for m, record in by_m.items():
        if not record["all_tuning_pass"] or not record["all_held_out_pass"]:
            raise ValueError(f"m={m} did not pass both recomputed pilot stages")
        if record["num_tuning_cells"] != 96 or record["num_held_out_cells"] != 448:
            raise ValueError(f"m={m} recomputed pilot cell count is incomplete")
        selected = record.get("selected_config")
        if not isinstance(selected, dict) or set(selected) != CANDIDATE_FIELDS:
            raise ValueError(f"m={m} selected config fields are incomplete")
        frozen["by_m"][m] = selected
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
