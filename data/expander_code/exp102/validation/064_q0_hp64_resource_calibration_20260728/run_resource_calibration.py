#!/usr/bin/env python3
"""Read-only validation 013 audit and outcome-blind HP64 resource projections.

The resource model and the scientific discrepancy audit are deliberately
separate entry points.  In particular, ``build_resource_model`` accepts only
timing, schedule, capacity, and provenance inputs.  It cannot inspect logical
labels or any scientific diagnostic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONFIG_VERSION = "exp102.q0_hp64.resource_calibration.config.v1"
RESOURCE_REPORT_VERSION = "exp102.q0_hp64.resource_calibration.report.v1"
SCIENCE_AUDIT_VERSION = "exp102.q0_hp64.discrepancy_audit.v1"
RECEIPT_VERSION = "exp102.q0_hp64.resource_calibration.receipt.v1"
CALIBRATION_CONTRACT_VERSION = "exp102.q0_hp64.resource_calibration.local.v1"
VALIDATION_RELATIVE_DIR = (
    "data/expander_code/exp102/validation/"
    "064_q0_hp64_resource_calibration_20260728"
)
BOUND_IMPLEMENTATION_NAMES = (
    "run_resource_calibration.py",
    "audit_resource_calibration.py",
    "test_resource_calibration.py",
    "README.md",
    "PRE_RUN_RED_TEAM.md",
)
RUNNER_OUTPUT_NAMES = (
    "resource_calibration_report.json",
    "discrepancy_audit.json",
    "timing_coverage.csv",
    "resource_scenarios.csv",
    "RESOURCE_CALIBRATION_REPORT.md",
    "RUN_RECEIPT.json",
)
INDEPENDENT_AUDIT_OUTPUT_NAME = "independent_package_audit.json"
ALL_EXPECTED_OUTPUT_NAMES = RUNNER_OUTPUT_NAMES + (INDEPENDENT_AUDIT_OUTPUT_NAME,)

EXPECTED_013 = {
    "run_name": "exp102_q0_hgp_screen_v2_20260722_4d134ee",
    "report_self_sha256": "bb2b8ef99dfbb1ba008bfddf1a64bc0ad9fccabc350bf2e0bd28b48d19dca062",
    "manifest_self_sha256": "1d1fe8bfc5c6e336acde74237ff3c488af320dd0d37be3c5f25d435c7adf78f2",
    "validation013_source_commit": "4d134ee7ca25125d341eb11cbfa34d6856514101",
    "validation013_archive_sha256": "ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027",
    "validation013_source_manifest_sha256": "5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8",
    "raw_count": 384,
    "resource_tier": "T3",
    "input_file_sha256": {
        "report": "7e3e9bc56d93b8cb2a361eb8bafd867b3f75ed14cf09688a230ca8553e729c2e",
        "manifest": "361c8adced9e456143c6ad8b2f282d998c71f5e6cdd9fd3dcf6eb2768729ba8e",
        "runtime_nd-1": "99033538fba7eb171378bc37bec0ec8a0d1ab0b352b56ba0fdc019dca7e4b801",
        "runtime_nd-2": "ad33343401eedb5dfb1cbf2d9e4f62a46f2cdad90249131924bf5613550666db",
        "runtime_nd-3": "f7c973efb70b5af48df78af58c033a00d6986c0d1338fd3ee5028341ae5768ff",
    },
}

HP64_TIMING_CELLS = (
    (3, "m03_c00", 0.10, "global_fresh_v1"),
    (4, "m04_c00", 0.07, "global_fresh_v1"),
    (5, "m05_c00", 0.10, "global_fresh_v1"),
    (6, "m06_c00", 0.04, "attempt022"),
    (8, "m08_c06", 0.04, "attempt022"),
)
HARD_AUDIT_CELLS = (
    ("m06_c00", 0.04, "attempt022"),
    ("m08_c06", 0.04, "attempt022"),
)
SCIENCE_METHODS = ("HP64", "MAM-IMH8")
INIT_FAMILIES = ("P", "U")
REPORT_TOLERANCE = 5.0e-14


class CalibrationConflict(RuntimeError):
    """The frozen evidence disagrees with its declared identity or values."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationConflict(message)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise CalibrationConflict(f"non-finite JSON constant in {path}: {token}")

    value = json.loads(
        path.read_text(encoding="ascii"), parse_constant=reject_constant,
    )
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    _require(field in value, f"missing self-hash field {field}")
    claimed = str(value[field])
    identity = {key: item for key, item in value.items() if key != field}
    _require(claimed == sha256_json(identity), f"self-hash mismatch for {field}")
    return claimed


def _assert_finite_json(value: Any, context: str = "output") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite float in {context}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{context}[{index}]")


def _exclusive_text(path: Path, text: str) -> None:
    """Install a new file atomically without ever replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CalibrationConflict(f"one-shot output already exists: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _exclusive_json(path: Path, value: Any) -> None:
    _assert_finite_json(value, path.name)
    _exclusive_text(path, canonical_json(value) + "\n")


def require_expected_outputs_absent(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    existing = [name for name in ALL_EXPECTED_OUTPUT_NAMES if (output_dir / name).exists()]
    _require(not existing, f"one-shot output set is not empty: {existing}")


def _git(worktree: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(worktree), *arguments), check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        raise CalibrationConflict(
            f"git source verification failed: {' '.join(arguments)}"
        ) from error
    return result.stdout.strip()


def _validate_authority(config: Mapping[str, Any]) -> None:
    authority = config.get("authority")
    _require(isinstance(authority, dict), "resource config authority is missing")
    exact = {
        "contract_version": CALIBRATION_CONTRACT_VERSION,
        "maximum_resource_status": "RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE",
        "science_audit_status": "PASS",
        "science_audit_authority": "HISTORICAL_DIAGNOSTIC_RECOMPUTATION_ONLY",
        "receipt_status": "LOCAL_AUDIT_COMPLETE_NO_REMOTE_AUTHORITY",
        "independent_audit_status": "INDEPENDENT_PACKAGE_AUDIT_PASS",
        "formal_authorization": False,
        "production_authorization": False,
        "remote_launch_authorization": False,
    }
    for field, expected in exact.items():
        _require(authority.get(field) == expected, f"authority field changed: {field}")
    files = authority.get("implementation_files")
    expected_paths = {
        f"{VALIDATION_RELATIVE_DIR}/{name}" for name in BOUND_IMPLEMENTATION_NAMES
    }
    _require(isinstance(files, dict) and set(files) == expected_paths, "bound implementation path set changed")
    _require(
        all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in files.values()),
        "bound implementation SHA is malformed",
    )


def verify_calibration_source(
    config_path: Path | str, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a clean, tracked, bytecode-free source tree bound by config."""
    config_path = Path(config_path).resolve()
    worktree = Path(_git(config_path.parent, "rev-parse", "--show-toplevel")).resolve()
    try:
        config_relative = config_path.relative_to(worktree).as_posix()
    except ValueError as error:
        raise CalibrationConflict("resource config is outside its Git worktree") from error
    expected_config_relative = f"{VALIDATION_RELATIVE_DIR}/resource_model_config.json"
    _require(config_relative == expected_config_relative, "resource config path changed")
    dirty = _git(worktree, "status", "--porcelain", "--untracked-files=all")
    _require(dirty == "", "calibration requires an entirely clean Git worktree")
    bytecode = []
    for directory, names, filenames in os.walk(worktree):
        names[:] = [name for name in names if name != ".git"]
        directory_path = Path(directory)
        if directory_path.name == "__pycache__":
            bytecode.append(str(directory_path.relative_to(worktree)))
            names[:] = []
            continue
        bytecode.extend(
            str((directory_path / name).relative_to(worktree))
            for name in filenames if name.endswith((".pyc", ".pyo"))
        )
    _require(not bytecode, f"bytecode artifacts are forbidden: {bytecode[:8]}")
    bound = dict(config["authority"]["implementation_files"])
    for relative_path, expected_sha in sorted(bound.items()):
        _git(worktree, "ls-files", "--error-unmatch", "--", relative_path)
        actual_path = worktree / relative_path
        _require(actual_path.is_file(), f"bound implementation file is missing: {relative_path}")
        _require(sha256_file(actual_path) == expected_sha, f"bound implementation SHA mismatch: {relative_path}")
    _git(worktree, "ls-files", "--error-unmatch", "--", config_relative)
    source_commit = _git(worktree, "rev-parse", "HEAD")
    source_tree = _git(worktree, "rev-parse", "HEAD^{tree}")
    object_format = _git(worktree, "rev-parse", "--show-object-format")
    _require(re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is not None, "source commit OID malformed")
    _require(re.fullmatch(r"[0-9a-f]{40,64}", source_tree) is not None, "source tree OID malformed")
    return {
        "calibration_source_commit": source_commit,
        "calibration_source_tree_sha": source_tree,
        "git_object_format": object_format,
        "worktree_clean": True,
        "bytecode_absent": True,
        "bound_implementation_sha256": bound,
    }


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, float, int, str]:
    return (
        str(cell["code_id"]),
        float(cell["p"]),
        int(cell["disorder_index"]),
        str(cell["disorder_source"]),
    )


def load_config(path: Path | str) -> dict[str, Any]:
    config = _load_json(Path(path))
    _require(config.get("version") == CONFIG_VERSION, "resource config version changed")
    _require(config.get("m_values") == [3, 4, 5, 6, 7, 8], "m grid changed")
    _require(
        config.get("calibration_p_values") == [0.04, 0.07, 0.1],
        "calibration p grid changed",
    )
    _require(
        config.get("formal_p_values") == [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "formal p grid changed",
    )
    _require(config.get("trajectory_options") == [8, 16, 32], "trajectory options changed")
    _require(set(config.get("clock_options", {})) == {"T1", "T2", "T3", "2T"}, "clock options changed")
    _require(float(config.get("safety_factor", 0.0)) == 2.0, "safety factor changed")
    _require(int(config.get("replay_passes", -1)) == 1, "replay pass count changed")
    _require(float(config.get("report_tolerance", 0.0)) == REPORT_TOLERANCE, "report tolerance changed")
    _validate_authority(config)
    _require(config.get("validation013_evidence") == EXPECTED_013, "validation 013 evidence binding changed")
    return config


def load_validation013_metadata(
    run_root: Path | str, expected_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify immutable validation 013 metadata without opening trajectory labels."""
    run_root = Path(run_root).resolve()
    report_path = run_root / "control" / "hgp_report.json"
    manifest_path = run_root / "control" / "hgp_measurement_control.json"
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    _require(run_root.name == expected_evidence["run_name"], "validation 013 run name changed")
    report_self_hash = _verify_self_hash(report, "report_sha256")
    manifest_self_hash = _verify_self_hash(manifest, "manifest_sha256")
    _require(report_self_hash == expected_evidence["report_self_sha256"], "unexpected validation 013 report")
    _require(manifest_self_hash == expected_evidence["manifest_self_sha256"], "unexpected validation 013 manifest")
    field_map = {
        "source_commit": "validation013_source_commit",
        "archive_sha256": "validation013_archive_sha256",
        "source_manifest_sha256": "validation013_source_manifest_sha256",
        "resource_tier": "resource_tier",
    }
    for report_field, evidence_field in field_map.items():
        _require(report.get(report_field) == expected_evidence[evidence_field], f"report {report_field} changed")
        _require(manifest.get(report_field) == expected_evidence[evidence_field], f"manifest {report_field} changed")
    _require(int(report.get("raw_count", -1)) == expected_evidence["raw_count"], "report raw count changed")
    _require(int(manifest.get("task_count", -1)) == expected_evidence["raw_count"], "manifest task count changed")
    _require(len(manifest.get("tasks", [])) == expected_evidence["raw_count"], "manifest task list changed")
    _require(report.get("manifest_sha256") == manifest_self_hash, "report/manifest hash mismatch")

    runtime_paths = {
        node: run_root / "hgp_global" / "preflight" / "nodes" / node / "runtime.json"
        for node in ("nd-1", "nd-2", "nd-3")
    }
    runtimes = {node: _load_json(path) for node, path in runtime_paths.items()}
    for node, runtime in runtimes.items():
        for runtime_field, evidence_field in field_map.items():
            if runtime_field == "resource_tier":
                continue
            _require(runtime.get(runtime_field) == expected_evidence[evidence_field], f"{node} runtime {runtime_field} changed")

    raw_root = run_root / "hgp_global" / "raw"
    _require(raw_root.is_dir(), "validation 013 raw directory is missing")
    file_sha256 = {
        "report": sha256_file(report_path),
        "manifest": sha256_file(manifest_path),
        **{f"runtime_{node}": sha256_file(path) for node, path in runtime_paths.items()},
    }
    _require(file_sha256 == expected_evidence["input_file_sha256"], "validation 013 input file hash changed")
    return {
        "run_root": run_root,
        "raw_root": raw_root,
        "report": report,
        "manifest": manifest,
        "runtimes": runtimes,
        "paths": {
            "report": report_path,
            "manifest": manifest_path,
            **{f"runtime_{node}": path for node, path in runtime_paths.items()},
        },
        "file_sha256": file_sha256,
    }


def build_provenance(
    *,
    config: Mapping[str, Any],
    config_path: Path | str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the unambiguous calibration/validation-013 provenance split."""
    provenance = {
        "calibration_source": dict(source_identity),
        "implementation_authority_sha256": sha256_json(config["authority"]),
        "resource_config": {
            "path": f"{VALIDATION_RELATIVE_DIR}/resource_model_config.json",
            "file_sha256": sha256_file(config_path),
        },
        "validation013_evidence": dict(config["validation013_evidence"]),
    }
    _assert_finite_json(provenance, "provenance")
    return provenance


def _hp64_manifest_counts(manifest: Mapping[str, Any]) -> dict[tuple[str, float, int, str], int]:
    counts: dict[tuple[str, float, int, str], int] = defaultdict(int)
    for entry in manifest["tasks"]:
        task = entry["task"]
        if task["method_id"] == "HP64":
            counts[_cell_key(task["cell"])] += 1
    return dict(counts)


def extract_hp64_timing_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only timing and task-count fields from the frozen report."""
    report = metadata["report"]
    manifest = metadata["manifest"]
    counts = _hp64_manifest_counts(manifest)
    summaries = {
        _cell_key(row["cell"]): row
        for row in report["cell_summaries"]
        if row["method_id"] == "HP64"
    }
    observations = []
    for m, code_id, p_value, source in HP64_TIMING_CELLS:
        key = (code_id, p_value, 0, source)
        _require(key in summaries, f"missing HP64 timing summary: {key}")
        _require(counts.get(key) == 32, f"HP64 historical trajectory count changed: {key}")
        core_seconds = float(summaries[key]["core_seconds"])
        _require(math.isfinite(core_seconds) and core_seconds > 0.0, f"invalid HP64 timing: {key}")
        observations.append({
            "m": m,
            "code_id": code_id,
            "p": p_value,
            "disorder_index": 0,
            "disorder_source": source,
            "resource_tier": "T3",
            "trajectory_count": counts[key],
            "cell_core_seconds": core_seconds,
            "seconds_per_trajectory_t3": core_seconds / counts[key],
            "evidence_class": "EMPIRICAL_SINGLE_CELL_SINGLE_DISORDER",
        })
    _require(len(summaries) == len(observations), "unexpected HP64 report timing cells")
    return {
        "observations": observations,
        "missing_m_values": [7],
        "historical_trajectory_count": 32,
    }


def extract_analysis_timing_evidence(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Conservatively select the largest stored analyzer benchmark by field."""
    runtimes = metadata["runtimes"]
    fields = (
        "family_benchmark_seconds",
        "comparison_benchmark_seconds",
        "trace_benchmark_seconds",
    )
    maxima: dict[str, dict[str, Any]] = {}
    rounds = set()
    tier_scales: dict[str, set[float]] = {tier: set() for tier in ("T1", "T2", "T3")}
    for node, runtime in runtimes.items():
        payload = runtime["payload"]
        timings = payload["b_analysis_timings"]
        rounds.add(int(timings["benchmark_measurement_rounds"]))
        for field in fields:
            value = float(timings[field])
            _require(math.isfinite(value) and value >= 0.0, f"invalid {node} analyzer timing")
            if field not in maxima or value > maxima[field]["seconds"]:
                maxima[field] = {"seconds": value, "node": node}
        for tier in tier_scales:
            tier_scales[tier].add(float(payload["tiers"][tier]["analysis_workload"]["b_diagnostic_scale"]))
    _require(rounds == {32768}, "analyzer benchmark measurement length changed")
    for tier, values in tier_scales.items():
        _require(len(values) == 1, f"cross-node analyzer scale mismatch for {tier}")
    scales = {tier: values.pop() for tier, values in tier_scales.items()}
    scales["2T"] = float(config["analysis_proxy"]["two_t_scale_from_t3"])
    family = maxima["family_benchmark_seconds"]["seconds"]
    comparison = maxima["comparison_benchmark_seconds"]["seconds"]
    proxy = (
        int(config["analysis_proxy"]["families_per_evaluation"]) * family
        + int(config["analysis_proxy"]["comparisons_per_evaluation"]) * comparison
    )
    return {
        "benchmark_measurement_rounds": 32768,
        "maxima_by_field": maxima,
        "analysis_scale_by_clock": scales,
        "t3_proxy_seconds_per_evaluation": proxy,
        "proxy_components": {
            "families_per_evaluation": int(config["analysis_proxy"]["families_per_evaluation"]),
            "comparisons_per_evaluation": int(config["analysis_proxy"]["comparisons_per_evaluation"]),
        },
        "trace_postprocess_accounting": "INCLUDED_IN_COMPLETE_REPLAY_NOT_ADDED_TO_ANALYSIS_PROXY",
        "coverage_limitation": "VALIDATION_013_B_FAMILY_AND_B_COMPARISON_PROXY_ONLY",
        "two_t_scale_status": "CONFIG_EXTRAPOLATION_UNVALIDATED",
    }


def build_timing_coverage(
    config: Mapping[str, Any], timing_evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    observed = {(int(row["m"]), float(row["p"])): row for row in timing_evidence["observations"]}
    by_m = {int(row["m"]): float(row["seconds_per_trajectory_t3"]) for row in timing_evidence["observations"]}
    global_max = max(by_m.values())
    rows = []
    for m in config["m_values"]:
        for p_value in config["calibration_p_values"]:
            exact = observed.get((int(m), float(p_value)))
            same_m_proxy = by_m.get(int(m), global_max)
            rows.append({
                "m": int(m),
                "p": float(p_value),
                "coverage_status": (
                    "EMPIRICAL_SINGLE_CELL_SINGLE_DISORDER"
                    if exact is not None else "MISSING_EMPIRICAL_COVERAGE"
                ),
                "observed_code_id": None if exact is None else exact["code_id"],
                "observed_disorder_source": None if exact is None else exact["disorder_source"],
                "observed_cell_core_seconds_t3_32_trajectories": (
                    None if exact is None else exact["cell_core_seconds"]
                ),
                "observed_seconds_per_trajectory_t3": (
                    None if exact is None else exact["seconds_per_trajectory_t3"]
                ),
                "same_m_proxy_seconds_per_trajectory_t3": same_m_proxy,
                "same_m_proxy_provenance": (
                    f"m{int(m)}_single_observed_cell"
                    if int(m) in by_m else "global_largest_observed_proxy_for_missing_m7"
                ),
                "global_max_proxy_seconds_per_trajectory_t3": global_max,
            })
    _require(len(rows) == 18, "timing coverage matrix size changed")
    return rows


def project_resource_option(
    *,
    config: Mapping[str, Any],
    evaluation_count_by_m: Mapping[int, int],
    seconds_per_t3_trajectory_by_m: Mapping[int, float],
    t3_analysis_seconds_per_evaluation: float,
    analysis_scale: float,
    trajectory_count: int,
    clock_name: str,
) -> dict[str, float]:
    """Pure timing arithmetic; it has no scientific-outcome inputs."""
    clock = config["clock_options"][clock_name]
    generation_t3 = sum(
        int(evaluation_count_by_m.get(int(m), 0))
        * float(seconds_per_t3_trajectory_by_m[int(m)])
        for m in config["m_values"]
    ) * int(trajectory_count)
    generation = generation_t3 * float(clock["generation_scale_from_t3"])
    replay = generation * int(config["replay_passes"])
    evaluation_count = sum(int(value) for value in evaluation_count_by_m.values())
    analysis = evaluation_count * float(t3_analysis_seconds_per_evaluation) * float(analysis_scale)
    unsafetied = generation + replay + analysis
    safety_adjusted = unsafetied * float(config["safety_factor"])
    cores_166 = int(config["capacity_cores"]["nd2_and_nd3_historical"])
    cores_75 = int(config["capacity_cores"]["nd2_only_contingency"])
    return {
        "generation_core_seconds": generation,
        "replay_core_seconds": replay,
        "analysis_proxy_core_seconds": analysis,
        "unsafetied_total_core_seconds": unsafetied,
        "safety_adjusted_total_core_seconds": safety_adjusted,
        "safety_adjusted_core_hours": safety_adjusted / 3600.0,
        "ideal_wall_hours_166_cores": safety_adjusted / (cores_166 * 3600.0),
        "ideal_wall_hours_75_cores": safety_adjusted / (cores_75 * 3600.0),
    }


def build_resource_model(
    *,
    config: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    analysis_timing_evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit all frozen resource scenarios and select none of them."""
    forbidden = {"q_top", "labels", "ess", "valid", "passed"}
    _require(not (forbidden & set(inspect.signature(build_resource_model).parameters)), "resource signature leaked outcomes")
    by_m = {
        int(row["m"]): float(row["seconds_per_trajectory_t3"])
        for row in timing_evidence["observations"]
    }
    global_max = max(by_m.values())
    same_m = {int(m): by_m.get(int(m), global_max) for m in config["m_values"]}
    global_proxy = {int(m): global_max for m in config["m_values"]}
    scenarios = {
        "same_m_proxy_with_m7_global_max": same_m,
        "global_observed_max_proxy": global_proxy,
    }
    per_m_calibration = int(config["codes_per_m"]) * int(config["disorders_per_code"]) * len(config["calibration_p_values"])
    per_m_formal = int(config["codes_per_m"]) * int(config["disorders_per_code"]) * len(config["formal_p_values"])
    stages = {
        "m3_easy_block_128": {3: 128},
        "calibration_grid_3p": {int(m): per_m_calibration for m in config["m_values"]},
        "formal_grid_7p": {int(m): per_m_formal for m in config["m_values"]},
    }
    _require(sum(stages["calibration_grid_3p"].values()) == 18432, "three-p grid count changed")
    _require(sum(stages["formal_grid_7p"].values()) == 43008, "formal grid count changed")

    rows = []
    for stage_name, count_by_m in stages.items():
        for scenario_name, seconds_by_m in scenarios.items():
            for trajectory_count in config["trajectory_options"]:
                for clock_name in ("T1", "T2", "T3", "2T"):
                    projected = project_resource_option(
                        config=config,
                        evaluation_count_by_m=count_by_m,
                        seconds_per_t3_trajectory_by_m=seconds_by_m,
                        t3_analysis_seconds_per_evaluation=analysis_timing_evidence[
                            "t3_proxy_seconds_per_evaluation"
                        ],
                        analysis_scale=analysis_timing_evidence["analysis_scale_by_clock"][clock_name],
                        trajectory_count=int(trajectory_count),
                        clock_name=clock_name,
                    )
                    rows.append({
                        "stage": stage_name,
                        "scenario": scenario_name,
                        "estimate_class": "SCENARIO_PROXY_NOT_CONFIDENCE_BOUND",
                        "evaluation_count": sum(count_by_m.values()),
                        "trajectory_count": int(trajectory_count),
                        "clock": clock_name,
                        "burn": int(config["clock_options"][clock_name]["burn"]),
                        "measurement": int(config["clock_options"][clock_name]["measurement"]),
                        "selected": False,
                        **projected,
                    })
    _require(len(rows) == 72, "resource scenario row count changed")
    coverage = build_timing_coverage(config, timing_evidence)
    authority = config["authority"]
    report_identity = {
        "report_version": RESOURCE_REPORT_VERSION,
        "contract_version": authority["contract_version"],
        "authority": "RESOURCE_SCENARIOS_ONLY",
        "status": authority["maximum_resource_status"],
        "formal_authorization": authority["formal_authorization"],
        "production_authorization": authority["production_authorization"],
        "remote_launch_authorization": authority["remote_launch_authorization"],
        "selection": None,
        "provenance": dict(provenance),
        "grid_evaluation_counts": {
            name: sum(values.values()) for name, values in stages.items()
        },
        "strict_empirical_estimates": {
            name: {
                "safety_adjusted_total_core_seconds": None,
                "reason": "INSUFFICIENT_MULTI_P_MULTI_CODE_MULTI_DISORDER_TIMING_COVERAGE",
            }
            for name in stages
        },
        "timing_observations": list(timing_evidence["observations"]),
        "timing_coverage": coverage,
        "analysis_timing_evidence": dict(analysis_timing_evidence),
        "scenario_definitions": {
            "same_m_proxy_with_m7_global_max": {
                "seconds_per_t3_trajectory_by_m": same_m,
                "meaning": "SAME_M_SINGLE_CELL_PROXY; M7_USES_GLOBAL_LARGEST_OBSERVED",
            },
            "global_observed_max_proxy": {
                "seconds_per_t3_trajectory_by_m": global_proxy,
                "meaning": "GLOBAL_LARGEST_OBSERVED_ASSIGNED_TO_EVERY_EVALUATION",
            },
        },
        "resource_scenarios": rows,
        "limitations": [
            "NO_EMPIRICAL_M7_TIMING",
            "MOST_M_P_CELLS_UNOBSERVED",
            "NO_MULTI_CODE_OR_MULTI_DISORDER_TIMING_DISTRIBUTION",
            "IDEAL_WALL_TIME_OMITS_LOAD_IMBALANCE_AND_SERIAL_OVERHEAD",
            "ANALYSIS_PROXY_DOES_NOT_COVER_A_FUTURE_SCHEMA_COMPLETELY",
            "NO_PROXY_ROW_IS_A_CONFIDENCE_BOUND",
        ],
    }
    _assert_finite_json(report_identity, "resource_report")
    serialized = canonical_json(report_identity).lower()
    _require("q_top" not in serialized and "logical_label" not in serialized, "resource report leaked science")
    return {**report_identity, "report_sha256": sha256_json(report_identity)}


def _np_scalar(raw: Mapping[str, np.ndarray], name: str) -> Any:
    _require(name in raw, f"raw field missing: {name}")
    value = raw[name]
    _require(value.shape == (), f"raw scalar has wrong shape: {name}")
    return value.item()


def _validate_task_entry(entry: Mapping[str, Any]) -> str:
    task = entry["task"]
    fingerprint = sha256_json(task)
    _require(entry.get("task_fingerprint") == fingerprint, "manifest task fingerprint changed")
    _require(entry.get("output_relpath") == f"trajectories/{fingerprint}.npz", "task output path changed")
    return fingerprint


def validate_hp64_raw_timing(
    metadata: Mapping[str, Any], timing_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-sum raw scalar timings without opening state or label arrays."""
    raw_root = Path(metadata["raw_root"])
    expected = {
        (row["code_id"], float(row["p"]), int(row["disorder_index"]), row["disorder_source"]): row
        for row in timing_evidence["observations"]
    }
    sums: dict[tuple[str, float, int, str], float] = defaultdict(float)
    counts: dict[tuple[str, float, int, str], int] = defaultdict(int)
    wall_sums: dict[tuple[str, float, int, str], float] = defaultdict(float)
    for entry in metadata["manifest"]["tasks"]:
        task = entry["task"]
        if task["method_id"] != "HP64":
            continue
        fingerprint = _validate_task_entry(entry)
        path = raw_root / entry["output_relpath"]
        with np.load(path, allow_pickle=False) as raw:
            _require(str(_np_scalar(raw, "task_fingerprint")) == fingerprint, "raw task fingerprint changed")
            raw_task = json.loads(str(_np_scalar(raw, "task_json")))
            _require(raw_task == task, "raw task identity changed")
            core = float(_np_scalar(raw, "core_seconds"))
            wall = float(_np_scalar(raw, "wall_seconds"))
        _require(math.isfinite(core) and core > 0.0, "raw core_seconds is invalid")
        _require(math.isfinite(wall) and wall > 0.0, "raw wall_seconds is invalid")
        key = _cell_key(task["cell"])
        sums[key] += core
        wall_sums[key] += wall
        counts[key] += 1
    _require(set(sums) == set(expected), "HP64 raw timing cell set changed")
    rows = []
    for key in sorted(sums):
        row = expected[key]
        _require(counts[key] == int(row["trajectory_count"]), f"raw timing count changed: {key}")
        _require(
            math.isclose(sums[key], float(row["cell_core_seconds"]), rel_tol=0.0, abs_tol=1.0e-9),
            f"report/raw core_seconds mismatch: {key}",
        )
        rows.append({
            "cell": {
                "code_id": key[0], "p": key[1], "disorder_index": key[2],
                "disorder_source": key[3],
            },
            "trajectory_count": counts[key],
            "raw_core_seconds_sum": sums[key],
            "raw_wall_seconds_sum": wall_sums[key],
            "report_core_seconds": row["cell_core_seconds"],
        })
    return {"status": "PASS", "cells": rows, "allow_pickle": False}


def character_means_from_labels(
    labels: np.ndarray, masks: np.ndarray, *, chunk_size: int = 128
) -> np.ndarray:
    labels = np.asarray(labels)
    masks = np.asarray(masks)
    _require(labels.ndim == 1 and labels.dtype == np.uint64 and labels.size > 0, "labels are malformed")
    _require(masks.ndim == 1 and masks.dtype == np.uint64 and masks.size > 0, "character masks are malformed")
    result = np.empty(masks.size, dtype=np.float64)
    for start in range(0, masks.size, int(chunk_size)):
        stop = min(start + int(chunk_size), masks.size)
        parity = np.bitwise_count(labels[:, None] & masks[None, start:stop]) & np.uint8(1)
        result[start:stop] = 1.0 - 2.0 * parity.mean(axis=0)
    return result


def u_statistic_squares(means: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    _require(means.ndim == 2 and means.shape[0] >= 2, "U-statistic needs independent trajectories")
    sums = means.sum(axis=0)
    return (np.square(sums) - np.square(means).sum(axis=0)) / (
        means.shape[0] * (means.shape[0] - 1)
    )


def infer_character_design(masks: np.ndarray, k: int) -> dict[str, Any]:
    masks = np.asarray(masks)
    _require(masks.ndim == 1 and masks.dtype == np.uint64, "character masks are malformed")
    _require(1 <= int(k) <= 64, "logical dimension is invalid")
    _require(np.unique(masks).size == masks.size and np.all(masks != 0), "character masks repeat or include zero")
    positions = []
    for bit in range(int(k)):
        target = np.uint64(1 << bit)
        matches = np.flatnonzero(masks == target)
        _require(matches.size == 1, f"basis character missing or repeated: bit {bit}")
        positions.append(int(matches[0]))
    total = (1 << int(k)) - 1
    tier = "full" if masks.size == total else "sampled"
    if tier == "sampled":
        _require(masks.size > int(k) and masks.size - int(k) <= total - int(k), "sampled character design is invalid")
    return {
        "k": int(k),
        "tier": tier,
        "basis_positions": positions,
        "total_nonzero_characters": total,
    }


def character_population_estimate(
    values: np.ndarray, masks: np.ndarray, k: int
) -> tuple[float, float, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    design = infer_character_design(masks, int(k))
    _require(values.shape == np.asarray(masks).shape and np.all(np.isfinite(values)), "character values are malformed")
    if design["tier"] == "full":
        return float(values.mean()), 0.0, design
    basis = np.zeros(values.size, dtype=bool)
    basis[np.asarray(design["basis_positions"], dtype=np.int64)] = True
    sampled = values[~basis]
    remaining = design["total_nonzero_characters"] - int(k)
    estimate = (float(values[basis].sum()) + remaining * float(sampled.mean())) / design[
        "total_nonzero_characters"
    ]
    if sampled.size <= 1 or remaining <= 1 or sampled.size == remaining:
        finite_se = 0.0
    else:
        fraction = sampled.size / remaining
        finite_se = (
            remaining / design["total_nonzero_characters"]
            * math.sqrt((1.0 - fraction) * float(sampled.var(ddof=1)) / sampled.size)
        )
    return float(estimate), float(finite_se), design


def qtop_estimate(means: np.ndarray, masks: np.ndarray, k: int) -> dict[str, Any]:
    means = np.asarray(means, dtype=np.float64)
    per_character = u_statistic_squares(means)
    estimate, character_se, design = character_population_estimate(per_character, masks, k)
    delete = []
    if means.shape[0] >= 3:
        for omitted in range(means.shape[0]):
            value, _, _ = character_population_estimate(
                u_statistic_squares(np.delete(means, omitted, axis=0)), masks, k
            )
            delete.append(value)
    delete_array = np.asarray(delete, dtype=np.float64)
    trajectory_se = (
        math.sqrt(
            (delete_array.size - 1) / delete_array.size
            * float(np.square(delete_array - delete_array.mean()).sum())
        )
        if delete_array.size else float("nan")
    )
    return {
        "estimate": estimate,
        "trajectory_se": float(trajectory_se),
        "character_se": character_se,
        "total_se": float(math.hypot(trajectory_se, character_se)),
        "per_character": per_character,
        "delete_one": delete_array,
        "character_design": design,
    }


def paired_qtop_delta(
    left_means: np.ndarray, right_means: np.ndarray, masks: np.ndarray, k: int
) -> dict[str, Any]:
    left = qtop_estimate(left_means, masks, k)
    right = qtop_estimate(right_means, masks, k)
    per_character = left["per_character"] - right["per_character"]
    signed, character_se, design = character_population_estimate(per_character, masks, k)
    left_contrasts = left["delete_one"] - right["estimate"]
    right_contrasts = left["estimate"] - right["delete_one"]
    variance = 0.0
    for contrasts in (left_contrasts, right_contrasts):
        _require(contrasts.size >= 2, "paired jackknife needs at least three trajectories per side")
        variance += (
            (contrasts.size - 1) / contrasts.size
            * float(np.square(contrasts - contrasts.mean()).sum())
        )
    trajectory_se = math.sqrt(variance)
    return {
        "signed_delta": float(signed),
        "absolute_delta": abs(float(signed)),
        "trajectory_se": float(trajectory_se),
        "character_se": float(character_se),
        "total_se": float(math.hypot(trajectory_se, character_se)),
        "z_abs_delta_over_total_se": (
            abs(float(signed)) / math.hypot(trajectory_se, character_se)
            if math.hypot(trajectory_se, character_se) > 0.0 else float("inf")
        ),
        "delete_one_left_contrasts": left_contrasts,
        "delete_one_right_contrasts": right_contrasts,
        "character_design": design,
    }


def _science_raw_worker(argument: tuple[str, dict[str, Any], int]) -> dict[str, Any]:
    path_text, entry, expected_measurements = argument
    path = Path(path_text)
    fingerprint = _validate_task_entry(entry)
    file_hash = sha256_file(path)
    with np.load(path, allow_pickle=False) as raw:
        _require(str(_np_scalar(raw, "task_fingerprint")) == fingerprint, "science raw fingerprint changed")
        raw_task = json.loads(str(_np_scalar(raw, "task_json")))
        _require(raw_task == entry["task"], "science raw task identity changed")
        labels = np.asarray(raw["sampler_measurement_labels"])
        masks = np.asarray(raw["character_masks"])
        k = int(_np_scalar(raw, "k"))
        core_seconds = float(_np_scalar(raw, "core_seconds"))
        wall_seconds = float(_np_scalar(raw, "wall_seconds"))
        character_sha256 = str(_np_scalar(raw, "character_sha256"))
        _require(labels.shape == (expected_measurements,), "measurement label length changed")
        means = character_means_from_labels(labels, masks)
    task = entry["task"]
    return {
        "output_relpath": entry["output_relpath"],
        "file_sha256": file_hash,
        "task_fingerprint": fingerprint,
        "task": task,
        "k": k,
        "masks": masks,
        "means": means,
        "character_sha256": character_sha256,
        "core_seconds": core_seconds,
        "wall_seconds": wall_seconds,
    }


def load_science_records(
    metadata: Mapping[str, Any], *, num_workers: int = 1
) -> list[dict[str, Any]]:
    """Load only the two frozen hard cells and recompute character means."""
    hard_keys = {(code, p, 0, source) for code, p, source in HARD_AUDIT_CELLS}
    selected = []
    for entry in metadata["manifest"]["tasks"]:
        task = entry["task"]
        if task["method_id"] in SCIENCE_METHODS and _cell_key(task["cell"]) in hard_keys:
            selected.append(entry)
    selected.sort(key=lambda entry: (
        _cell_key(entry["task"]["cell"]), entry["task"]["method_id"],
        entry["task"]["init_family"], int(entry["task"]["trajectory_index"]),
    ))
    _require(len(selected) == 128, "science raw selection changed")
    arguments = [
        (str(Path(metadata["raw_root"]) / entry["output_relpath"]), entry, 32768)
        for entry in selected
    ]
    if int(num_workers) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_workers)) as pool:
            records = list(pool.map(_science_raw_worker, arguments, chunksize=1))
    else:
        records = [_science_raw_worker(argument) for argument in arguments]
    return records


def _report_summary_index(report: Mapping[str, Any]) -> dict[tuple[tuple[str, float, int, str], str], Mapping[str, Any]]:
    return {
        (_cell_key(row["cell"]), str(row["method_id"])): row
        for row in report["cell_summaries"]
    }


def _report_comparison_index(report: Mapping[str, Any]) -> dict[tuple[tuple[str, float, int, str], str], Mapping[str, Any]]:
    return {
        (_cell_key(row["cell"]), str(row["init_family"])): row
        for row in report["comparisons"]
        if row["hp_method_id"] == "HP64" and row["map_method_id"] == "MAM-IMH8"
    }


def _assert_close(actual: float, expected: float, name: str, tolerance: float) -> None:
    _require(
        math.isfinite(actual) and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"raw recomputation disagrees with report for {name}: {actual} != {expected}",
    )


def _public_qtop(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "q_top": float(value["estimate"]),
        "trajectory_se": float(value["trajectory_se"]),
        "character_se": float(value["character_se"]),
        "total_se": float(value["total_se"]),
        "character_design": value["character_design"],
    }


def _public_delta(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "signed_delta": float(value["signed_delta"]),
        "absolute_delta": float(value["absolute_delta"]),
        "trajectory_se": float(value["trajectory_se"]),
        "character_se": float(value["character_se"]),
        "total_se": float(value["total_se"]),
        "z_abs_delta_over_total_se": float(value["z_abs_delta_over_total_se"]),
        "character_design": value["character_design"],
    }


def build_discrepancy_audit(
    *,
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    authority: Mapping[str, Any],
    tolerance: float = REPORT_TOLERANCE,
) -> dict[str, Any]:
    """Independently reconstruct the validation 013 headline character statistics."""
    grouped: dict[tuple[tuple[str, float, int, str], str, str], list[Mapping[str, Any]]] = defaultdict(list)
    catalog = []
    for record in records:
        task = record["task"]
        grouped[(_cell_key(task["cell"]), task["method_id"], task["init_family"])].append(record)
        catalog.append({
            "output_relpath": record["output_relpath"],
            "task_fingerprint": record["task_fingerprint"],
            "file_sha256": record["file_sha256"],
        })
    catalog.sort(key=lambda row: row["output_relpath"])
    _require(len(catalog) == 128 and len({row["output_relpath"] for row in catalog}) == 128, "science raw catalog changed")
    report_summaries = _report_summary_index(report)
    report_comparisons = _report_comparison_index(report)
    cell_results = []
    for code_id, p_value, source in HARD_AUDIT_CELLS:
        cell_key = (code_id, p_value, 0, source)
        method_internal: dict[str, dict[str, Any]] = {}
        method_public: dict[str, Any] = {}
        for method in SCIENCE_METHODS:
            family_internal = {}
            family_public = {}
            reference_masks = None
            reference_k = None
            for family in INIT_FAMILIES:
                rows = sorted(
                    grouped[(cell_key, method, family)],
                    key=lambda row: int(row["task"]["trajectory_index"]),
                )
                _require(len(rows) == 16, f"science family trajectory count changed: {cell_key}/{method}/{family}")
                _require([int(row["task"]["trajectory_index"]) for row in rows] == list(range(16)), "trajectory indices changed")
                masks = np.asarray(rows[0]["masks"], dtype=np.uint64)
                k = int(rows[0]["k"])
                for row in rows[1:]:
                    _require(int(row["k"]) == k and np.array_equal(row["masks"], masks), "within-family characters changed")
                means = np.stack([row["means"] for row in rows])
                estimate = qtop_estimate(means, masks, k)
                stored = report_summaries[(cell_key, method)]["families"][family]
                _assert_close(estimate["estimate"], float(stored["q_top"]), f"{cell_key}/{method}/{family}/q_top", tolerance)
                _assert_close(estimate["trajectory_se"], float(stored["q_top_trajectory_se"]), f"{cell_key}/{method}/{family}/trajectory_se", tolerance)
                _assert_close(estimate["character_se"], float(stored["q_top_character_se"]), f"{cell_key}/{method}/{family}/character_se", tolerance)
                family_internal[family] = {"means": means, "estimate": estimate}
                family_public[family] = _public_qtop(estimate)
                if reference_masks is None:
                    reference_masks, reference_k = masks, k
                else:
                    _require(reference_k == k and np.array_equal(reference_masks, masks), "P/U character sets changed")
            combined = qtop_estimate(
                np.vstack([family_internal[family]["means"] for family in INIT_FAMILIES]),
                reference_masks,
                int(reference_k),
            )
            stored_summary = report_summaries[(cell_key, method)]
            _assert_close(combined["estimate"], float(stored_summary["q_top"]), f"{cell_key}/{method}/combined_q_top", tolerance)
            initialization = paired_qtop_delta(
                family_internal["P"]["means"], family_internal["U"]["means"],
                reference_masks, int(reference_k),
            )
            stored_initialization = stored_summary["initialization_delta"]
            _assert_close(initialization["absolute_delta"], float(stored_initialization["delta_q_top"]), f"{cell_key}/{method}/P_U_delta", tolerance)
            _assert_close(initialization["total_se"], float(stored_initialization["se_delta_q_top"]), f"{cell_key}/{method}/P_U_se", tolerance)
            method_internal[method] = {
                "families": family_internal,
                "combined": combined,
                "masks": reference_masks,
                "k": int(reference_k),
            }
            method_public[method] = {
                "families": family_public,
                "combined": _public_qtop(combined),
                "P_minus_U": _public_delta(initialization),
            }
        _require(
            method_internal["HP64"]["k"] == method_internal["MAM-IMH8"]["k"]
            and np.array_equal(method_internal["HP64"]["masks"], method_internal["MAM-IMH8"]["masks"]),
            "cross-method character sets changed",
        )
        cross_family = {}
        for family in INIT_FAMILIES:
            comparison = paired_qtop_delta(
                method_internal["HP64"]["families"][family]["means"],
                method_internal["MAM-IMH8"]["families"][family]["means"],
                method_internal["HP64"]["masks"], method_internal["HP64"]["k"],
            )
            stored = report_comparisons[(cell_key, family)]["q_top"]
            _assert_close(comparison["absolute_delta"], float(stored["delta_q_top"]), f"{cell_key}/{family}/HP64_MAM_delta", tolerance)
            _assert_close(comparison["total_se"], float(stored["se_delta_q_top"]), f"{cell_key}/{family}/HP64_MAM_se", tolerance)
            cross_family[family] = _public_delta(comparison)
        combined_difference = (
            method_internal["HP64"]["combined"]["estimate"]
            - method_internal["MAM-IMH8"]["combined"]["estimate"]
        )
        cell_results.append({
            "cell": {
                "code_id": code_id, "p": p_value, "disorder_index": 0,
                "disorder_source": source,
            },
            "methods": method_public,
            "cross_method_by_initialization_family": cross_family,
            "combined_hp64_minus_mam": float(combined_difference),
            "combined_absolute_difference": abs(float(combined_difference)),
        })
    m8 = next(row for row in cell_results if row["cell"]["code_id"] == "m08_c06")
    m6 = next(row for row in cell_results if row["cell"]["code_id"] == "m06_c00")
    identity = {
        "audit_version": SCIENCE_AUDIT_VERSION,
        "contract_version": authority["contract_version"],
        "status": authority["science_audit_status"],
        "authority": authority["science_audit_authority"],
        "formal_authorization": authority["formal_authorization"],
        "production_authorization": authority["production_authorization"],
        "remote_launch_authorization": authority["remote_launch_authorization"],
        "provenance": dict(provenance),
        "allow_pickle": False,
        "report_tolerance": float(tolerance),
        "selected_raw_count": len(catalog),
        "selected_raw_catalog_sha256": sha256_json(catalog),
        "selected_raw_catalog": catalog,
        "cells": cell_results,
        "headline_checks": {
            "m8_hp64_P_q_top": m8["methods"]["HP64"]["families"]["P"]["q_top"],
            "m8_hp64_U_q_top": m8["methods"]["HP64"]["families"]["U"]["q_top"],
            "m8_hp64_combined_q_top": m8["methods"]["HP64"]["combined"]["q_top"],
            "m8_mam_combined_q_top": m8["methods"]["MAM-IMH8"]["combined"]["q_top"],
            "m8_interpretation": "0.91317_VS_0.99273_IS_HP64_VS_MAM_NOT_HP64_P_VS_U",
            "m6_P_hp64_mam_absolute_delta": m6["cross_method_by_initialization_family"]["P"]["absolute_delta"],
            "m6_P_hp64_mam_paired_se": m6["cross_method_by_initialization_family"]["P"]["total_se"],
            "m6_P_hp64_mam_z": m6["cross_method_by_initialization_family"]["P"]["z_abs_delta_over_total_se"],
        },
        "limitations": [
            "NO_NEW_SAMPLING",
            "NO_NEW_CONVERGENCE_AUTHORITY",
            "HP64_AND_HP32_REMAIN_ONE_MECHANISM",
            "HISTORICAL_P_AND_EXACT_K0_U_IDENTITIES_PRESERVED",
        ],
    }
    _assert_finite_json(identity, "discrepancy_audit")
    return {**identity, "audit_sha256": sha256_json(identity)}


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    return stream.getvalue()


def render_resource_markdown(resource: Mapping[str, Any]) -> str:
    example_rows = [
        row for row in resource["resource_scenarios"]
        if row["trajectory_count"] == 32 and row["clock"] in {"T3", "2T"}
    ]
    lines = [
        "# HP64 resource calibration report",
        "",
        f"Status: `{resource['status']}`.",
        "",
        "No option is selected. Every numerical total below is a planning proxy, not a confidence bound.",
        "Strict empirical totals remain `null` because validation 013 lacks m7, most p values, and multi-code/multi-disorder timing distributions.",
        "",
        "## Grid sizes",
        "",
        "| Stage | Evaluations |",
        "|---|---:|",
    ]
    for name, count in resource["grid_evaluation_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## T3/2T, 32-trajectory proxy examples",
        "",
        "| Stage | Scenario | Clock | Safety core-hours | Ideal 166-core hours | Ideal 75-core hours |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in example_rows:
        lines.append(
            f"| `{row['stage']}` | `{row['scenario']}` | `{row['clock']}` | "
            f"{row['safety_adjusted_core_hours']:.3f} | {row['ideal_wall_hours_166_cores']:.3f} | "
            f"{row['ideal_wall_hours_75_cores']:.3f} |"
        )
    lines += [
        "",
        "The full 72-row option matrix is in `resource_scenarios.csv`; the 18-row empirical coverage matrix is in `timing_coverage.csv`.",
        "Ideal wall times omit scheduling imbalance, serial stages, filesystem contention, current load, and failures.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    resource: Mapping[str, Any],
    discrepancy: Mapping[str, Any],
    timing_raw_audit: Mapping[str, Any],
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    require_expected_outputs_absent(output_dir)
    _assert_finite_json(resource, "resource_report")
    _assert_finite_json(discrepancy, "discrepancy_audit")
    _assert_finite_json(timing_raw_audit, "timing_raw_audit")
    _require(resource.get("provenance") == discrepancy.get("provenance"), "resource/science provenance mismatch")
    resource_path = output_dir / "resource_calibration_report.json"
    discrepancy_path = output_dir / "discrepancy_audit.json"
    coverage_path = output_dir / "timing_coverage.csv"
    scenarios_path = output_dir / "resource_scenarios.csv"
    markdown_path = output_dir / "RESOURCE_CALIBRATION_REPORT.md"
    _exclusive_json(resource_path, resource)
    _exclusive_json(discrepancy_path, discrepancy)
    coverage_fields = (
        "m", "p", "coverage_status", "observed_code_id", "observed_disorder_source",
        "observed_cell_core_seconds_t3_32_trajectories", "observed_seconds_per_trajectory_t3",
        "same_m_proxy_seconds_per_trajectory_t3", "same_m_proxy_provenance",
        "global_max_proxy_seconds_per_trajectory_t3",
    )
    scenario_fields = (
        "stage", "scenario", "estimate_class", "evaluation_count", "trajectory_count",
        "clock", "burn", "measurement", "selected", "generation_core_seconds",
        "replay_core_seconds", "analysis_proxy_core_seconds", "unsafetied_total_core_seconds",
        "safety_adjusted_total_core_seconds", "safety_adjusted_core_hours",
        "ideal_wall_hours_166_cores", "ideal_wall_hours_75_cores",
    )
    _exclusive_text(coverage_path, _csv_text(resource["timing_coverage"], coverage_fields))
    _exclusive_text(scenarios_path, _csv_text(resource["resource_scenarios"], scenario_fields))
    _exclusive_text(markdown_path, render_resource_markdown(resource))
    artifacts = {
        path.name: sha256_file(path)
        for path in (resource_path, discrepancy_path, coverage_path, scenarios_path, markdown_path)
    }
    receipt_identity = {
        "receipt_version": RECEIPT_VERSION,
        "contract_version": config["authority"]["contract_version"],
        "authority": "LOCAL_RESOURCE_AND_HISTORICAL_DIAGNOSTIC_AUDIT_ONLY",
        "status": config["authority"]["receipt_status"],
        "provenance": resource["provenance"],
        "resource_report_sha256": resource["report_sha256"],
        "discrepancy_audit_sha256": discrepancy["audit_sha256"],
        "config_file_sha256": sha256_file(config_path),
        "timing_raw_audit": timing_raw_audit,
        "artifact_file_sha256": artifacts,
        "formal_authorization": config["authority"]["formal_authorization"],
        "production_authorization": config["authority"]["production_authorization"],
        "remote_launch_authorization": config["authority"]["remote_launch_authorization"],
    }
    _assert_finite_json(receipt_identity, "run_receipt")
    receipt = {**receipt_identity, "receipt_sha256": sha256_json(receipt_identity)}
    receipt_path = output_dir / "RUN_RECEIPT.json"
    _exclusive_json(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation013-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=here / "resource_model_config.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require(args.num_workers >= 1, "num-workers must be positive")
    config = load_config(args.config)
    require_expected_outputs_absent(args.output_dir)
    source_identity = verify_calibration_source(args.config, config)
    metadata = load_validation013_metadata(
        args.validation013_run, config["validation013_evidence"],
    )
    timing = extract_hp64_timing_evidence(metadata)
    analysis_timing = extract_analysis_timing_evidence(metadata, config)
    provenance = build_provenance(
        config=config, config_path=args.config, source_identity=source_identity,
    )

    # Resource arithmetic is completed before any logical-label array is read.
    resource = build_resource_model(
        config=config,
        timing_evidence=timing,
        analysis_timing_evidence=analysis_timing,
        provenance=provenance,
    )
    timing_raw_audit = validate_hp64_raw_timing(metadata, timing)
    science_records = load_science_records(metadata, num_workers=args.num_workers)
    discrepancy = build_discrepancy_audit(
        report=metadata["report"], records=science_records, provenance=provenance,
        authority=config["authority"], tolerance=float(config["report_tolerance"]),
    )
    receipt = write_outputs(
        output_dir=args.output_dir,
        resource=resource,
        discrepancy=discrepancy,
        timing_raw_audit=timing_raw_audit,
        config_path=args.config,
        config=config,
    )
    print(canonical_json({
        "status": receipt["status"],
        "resource_report_sha256": receipt["resource_report_sha256"],
        "discrepancy_audit_sha256": receipt["discrepancy_audit_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
