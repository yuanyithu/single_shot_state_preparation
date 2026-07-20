"""Build deterministic three-node production ownership from held-out timings."""

import argparse
import json
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.pilot import (
    _analyze_records,
    _assert_report_matches_recomputed,
    _candidate_key,
    _load_records,
    _verify_report_evidence,
    recompute_frozen,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}


def selected_held_out_records(records, by_m, registry):
    selected = []
    attempts = {}
    for m in range(3, 9):
        result = by_m[str(m)]
        attempt = result["held_out"]["selected_attempt"]
        candidate = result["selected_config"]
        if not result["all_tuning_pass"] or not result["all_held_out_pass"]:
            raise ValueError(f"m={m} is not held-out certified")
        if attempt is None or candidate is None:
            raise ValueError(f"m={m} has no selected held-out attempt")
        attempts[str(m)] = int(attempt)
        candidate_key = _candidate_key(candidate)
        rows = [record for record in records
                if record["stage"] == "held_out" and record["m"] == m
                and record["attempt"] == attempt
                and record["candidate_key"] == candidate_key]
        cells = {(row["code_id"], row["p"], row["disorder_index"]) for row in rows}
        if len(rows) != 448 or len(cells) != 448 or not all(row["valid"] for row in rows):
            raise ValueError(f"m={m} selected held-out evidence is not 448 unique valid cells")
        selected.extend(rows)
    expected_codes = {code["code_id"] for code in registry["codes"]}
    if {row["code_id"] for row in selected} != expected_codes or len(selected) != 2688:
        raise ValueError("selected held-out evidence does not cover the frozen registry")
    return selected, attempts


def lpt_assign(timings):
    core_load = {node: 0.0 for node in CAPACITY}
    owners = {}
    for code_id, seconds in sorted(timings.items(), key=lambda item: (-item[1], item[0])):
        code_core_seconds = 128 * 7 * float(seconds)
        node = min(CAPACITY, key=lambda name: (
            (core_load[name] + code_core_seconds) / CAPACITY[name], name,
        ))
        owners[code_id] = node
        core_load[node] += code_core_seconds
    wall_load = {node: core_load[node] / CAPACITY[node] for node in CAPACITY}
    return owners, core_load, wall_load


def build_manifest(registry_path, config_path, frozen_path, report_path):
    registry = load_registry(registry_path)
    config = load_config(config_path)
    frozen = json.loads(Path(frozen_path).read_text(encoding="ascii"))
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="ascii"))
    expected_frozen = recompute_frozen(report_path, registry_path, config_path)
    if frozen != expected_frozen:
        raise ValueError("freezer differs from the report/raw recomputation")
    if frozen.get("status") != "FROZEN_HELD_OUT_PASS":
        raise ValueError("held-out freezer is required")
    if frozen.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("freezer registry identity mismatch")
    if frozen.get("config_sha256") != config["config_sha256"]:
        raise ValueError("freezer config identity mismatch")
    if frozen.get("pilot_report_sha256") != sha256_json(report):
        raise ValueError("freezer pilot report hash mismatch")
    if frozen.get("raw_evidence_sha256") != sha256_json(report.get("raw_evidence")):
        raise ValueError("freezer raw evidence hash mismatch")
    paths = _verify_report_evidence(report, report_path)
    records, source_commit = _load_records(paths, registry, config, report.get("source_commit"))
    by_m = _analyze_records(records, registry, config)
    _assert_report_matches_recomputed(report, by_m)
    if (source_commit != frozen.get("source_commit") or report.get("engine") != "numba"
            or frozen.get("engine") != "numba"):
        raise ValueError("held-out source or engine identity mismatch")
    selected, selected_attempts = selected_held_out_records(records, by_m, registry)
    selected_configs = {str(m): by_m[str(m)]["selected_config"] for m in range(3, 9)}
    if frozen.get("by_m") != selected_configs:
        raise ValueError("freezer selected configs differ from recomputed held-out evidence")
    if frozen.get("held_out_attempt_by_m") != selected_attempts:
        raise ValueError("freezer held-out attempts differ from recomputed evidence")
    raw_timings = {code["code_id"]: [] for code in registry["codes"]}
    for record in selected:
        raw_timings[record["code_id"]].append(record["core_seconds"])
    timings = {}
    for code in registry["codes"]:
        values = raw_timings[code["code_id"]]
        if len(values) != 56:
            raise ValueError(f"held-out timing coverage for {code['code_id']} is {len(values)}/56")
        timings[code["code_id"]] = float(np.median(values))
    owners, core_load, wall_load = lpt_assign(timings)
    manifest = {
        "deployment_version": "exp102.production.deployment.v1",
        "capacity": CAPACITY, "code_owner": owners,
        "estimated_node_core_seconds": core_load,
        "estimated_node_seconds": wall_load,
        "held_out_median_core_seconds": timings,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": sha256_json(frozen),
        "source_commit": frozen["source_commit"],
        "pilot_report_sha256": frozen["pilot_report_sha256"],
        "raw_evidence_sha256": frozen["raw_evidence_sha256"],
        "selected_attempt_by_m": selected_attempts,
        "selected_cell_count": len(selected),
        "selected_held_out_evidence_sha256": sha256_json(sorted(
            ({"task_fingerprint": record["task_fingerprint"],
              "sha256": sha256_file(record["path"])} for record in selected),
            key=lambda item: item["task_fingerprint"],
        )),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry"); parser.add_argument("config"); parser.add_argument("frozen")
    parser.add_argument("report"); parser.add_argument("output")
    args = parser.parse_args()
    manifest = build_manifest(args.registry, args.config, args.frozen, args.report)
    atomic_json(args.output, manifest)
    print(json.dumps(manifest["estimated_node_seconds"], sort_keys=True))


if __name__ == "__main__":
    main()
