import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.pa_discovery import (
    PA_RAW_VERSION,
    _verified_pa_paths,
    fixed_pa_ownership,
    load_pa_discovery_config,
    pa_task_manifest,
)
from data.expander_code.exp102.exp102_pipeline.discovery import load_discovery_config
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.transport_autopsy import (
    AUTOPSY_RAW_VERSION,
    _verified_autopsy_paths,
    autopsy_tasks,
    fixed_autopsy_ownership,
    load_autopsy_config,
    write_autopsy_task_manifest,
)


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_pa.discovery.v1.json"
DISCOVERY_CONFIG_PATH = EXP102_ROOT / "config/discovery.v2.json"
AUTOPSY_CONFIG_PATH = EXP102_ROOT / "config/transport_autopsy.v1.json"
SOURCE_COMMIT = "6" * 40
validate_runtime_report = import_module(
    "data.expander_code.exp102.validation.006_pa_discovery_20260721.orchestrate_pa"
).validate_runtime_report
pa_workflow = import_module(
    "data.expander_code.exp102.exp102_pipeline.pa_discovery"
)


def _stage_evidence(root):
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    control_path = root / "control/hard_screen.json"
    control = pa_task_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "hard_screen",
        [record["method_id"] for record in config["base_methods"]], control_path,
    )
    control_sha = sha256_file(control_path)
    ownership = fixed_pa_ownership(
        control["tasks"], ["nd-2", "nd-3"], SOURCE_COMMIT,
        control_sha, "hard_screen",
    )
    ownership_path = root / "control/ownership.json"
    atomic_json(ownership_path, ownership)
    ownership_sha = sha256_file(ownership_path)
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "file_count": 20,
    }
    task_by_fingerprint = {
        sha256_json(task): task for task in control["tasks"]
    }
    node_roots = {}
    for node in ownership["nodes"]:
        node_root = root / "hard_screen" / control_sha[:12] / node
        node_roots[node] = node_root
        files = []
        assigned = sorted(
            fingerprint for fingerprint, owner in ownership["task_owner"].items()
            if owner == node
        )
        for index, fingerprint in enumerate(assigned):
            path = node_root / "raw" / f"{index:03d}.npz"
            atomic_npz(path, marker=np.array(index, dtype=np.int16))
            files.append({
                "task_fingerprint": fingerprint,
                "path": path.relative_to(node_root).as_posix(),
                "sha256": sha256_file(path),
            })
        raw_manifest = {
            "raw_manifest_version": PA_RAW_VERSION,
            "node": node,
            "stage": "hard_screen",
            "stage_fingerprint": ownership["stage_fingerprint"],
            "source_commit": SOURCE_COMMIT,
            "control_sha256": control_sha,
            "ownership_sha256": ownership_sha,
            "source_identity": source_identity,
            "files": files,
        }
        raw_manifest_path = node_root / "raw_manifest.json"
        atomic_json(raw_manifest_path, raw_manifest)
        atomic_json(node_root / "stage_status.json", {
            "status": "SUCCESS", "node": node,
            "stage_fingerprint": ownership["stage_fingerprint"],
            "expected": len(files), "computed": len(files), "reused": 0,
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
        })
        atomic_json(node_root / "SUCCESS", {
            "stage_fingerprint": ownership["stage_fingerprint"],
            "completed_utc": "2026-07-21T00:00:00Z",
        })
    return registry, config, control_path, ownership, node_roots


def test_pa_remote_evidence_binds_lpt_ownership_source_hashes_and_markers(tmp_path):
    registry, config, control, ownership, _ = _stage_evidence(tmp_path)
    verified = _verified_pa_paths(
        tmp_path, control, registry, config, "hard_screen", SOURCE_COMMIT,
    )
    assert len(verified["paths"]) == 64
    assert verified["source_identity"]["archive_sha256"] == "a" * 64
    assert verified["stage_fingerprint"] == ownership["stage_fingerprint"]


def test_pa_remote_evidence_rejects_tampered_success_marker(tmp_path):
    registry, config, control, _, roots = _stage_evidence(tmp_path)
    marker = roots["nd-2"] / "SUCCESS"
    value = json.loads(marker.read_text(encoding="ascii"))
    value["stage_fingerprint"] = "c" * 64
    atomic_json(marker, value)
    with pytest.raises(ValueError, match="SUCCESS marker"):
        _verified_pa_paths(
            tmp_path, control, registry, config, "hard_screen", SOURCE_COMMIT,
        )


def test_pa_fixed_ownership_rejects_capacity_or_owner_tampering(tmp_path):
    _, _, control_path, ownership, _ = _stage_evidence(tmp_path)
    control = json.loads(control_path.read_text(encoding="ascii"))
    recomputed = fixed_pa_ownership(
        control["tasks"], ownership["nodes"], SOURCE_COMMIT,
        sha256_file(control_path), "hard_screen",
    )
    assert recomputed == ownership
    altered = dict(ownership)
    altered["capacity"] = dict(ownership["capacity"])
    altered["capacity"]["nd-2"] -= 1
    assert altered != recomputed


def _runtime_report(control):
    rows = [
        {"m": m, "kernel": kernel,
         "differential_us_per_particle_sweep": 10.0}
        for m in (6, 8) for kernel in ("coordinate", "block4")
    ]
    return {
        "benchmark_version": "exp102.q0_pa.runtime.v1",
        "source_commit": SOURCE_COMMIT,
        "source_identity": {
            "source_commit": SOURCE_COMMIT, "mode": "archive",
            "archive_sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "file_count": 20,
        },
        "environment": {
            "system": "Linux", "machine": "x86_64", "hostname": "nd-2",
            "python": "3.12.0", "numpy": "2.0.0",
        },
        "registry_sha256": control["registry_sha256"],
        "discovery_config_sha256": control["discovery_config_sha256"],
        "rows": rows,
        "conservative_seconds_per_particle_sweep": {
            "coordinate": 1e-5, "block4": 1e-5,
        },
        "startup_seconds": 1.0,
        "max_population_minutes": 1.0,
        "projected_core_seconds": 60.0,
        "projection_nodes": ["nd-2", "nd-3"],
        "projection_capacity": 166,
        "projected_minutes_with_safety_factor_2": 1.0,
        "projected_confirmation_methods": ["B384-2", "C192-2"],
        "checks": {
            "m8_slowest_kernel_us": True,
            "startup_seconds": True,
            "max_population_minutes": True,
            "full_schedule_minutes_with_safety_factor_2": True,
        },
        "status": "PASS",
    }


def test_pa_runtime_launch_gate_requires_clean_linux_archive_and_numeric_pass(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    control_path = tmp_path / "hard.json"
    control = pa_task_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "hard_screen",
        [record["method_id"] for record in config["base_methods"]], control_path,
    )
    report_path = tmp_path / "runtime.json"
    report = _runtime_report(control)
    atomic_json(report_path, report)
    assert validate_runtime_report(
        report_path, control, SOURCE_COMMIT, "a" * 64, "b" * 64,
    )["status"] == "PASS"

    report["environment"]["system"] = "Darwin"
    atomic_json(report_path, report)
    with pytest.raises(ValueError, match="clean-source Linux"):
        validate_runtime_report(
            report_path, control, SOURCE_COMMIT, "a" * 64, "b" * 64,
        )


def _hard_summaries(config, passing, separated=False):
    summaries = {}
    for method_index, method in enumerate(config["base_methods"]):
        method_id = method["method_id"]
        cells = {}
        for cell in config["hard_screen"]["cells"]:
            q_top = 0.2 + (0.2 * method_index if separated else 0.005 * method_index)
            cells[json.dumps(cell, sort_keys=True, separators=(",", ":"))] = {
                "cell": cell, "pass": method_id in passing,
                "q_top": q_top, "q_top_mcse": 0.01,
                "core_seconds": 1.0,
            }
        summaries[method_id] = cells
    return summaries


@pytest.mark.parametrize(
    "passing,separated,expected",
    [
        (set(), False, "EXHAUSTED"),
        ({"C192-2"}, False, "RESCUE_REQUIRED"),
        ({"C192-2", "B96-1"}, True, "EXHAUSTED"),
        ({"C192-2", "B96-1"}, False, "READY_FOR_CONFIRMATION"),
    ],
)
def test_pa_hard_screen_obeys_zero_one_two_and_consistency_branches(
    monkeypatch, passing, separated, expected,
):
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    summaries = _hard_summaries(config, passing, separated)
    stage = {
        "manifest": {
            "source_commit": SOURCE_COMMIT,
            "method_ids": [method["method_id"] for method in config["base_methods"]],
            "stage": "hard_screen",
        },
        "complete": True,
        "records": [],
    }
    monkeypatch.setattr(pa_workflow, "load_pa_stage", lambda *args, **kwargs: stage)
    monkeypatch.setattr(
        pa_workflow, "_summarize_records", lambda *args, **kwargs: summaries,
    )
    report = pa_workflow.analyze_hard_screen(
        "raw", "manifest", REGISTRY_PATH, CONFIG_PATH,
    )
    assert report["status"] == expected
    if expected == "READY_FOR_CONFIRMATION":
        assert [report["primary"], report["backup"]] == ["B96-1", "C192-2"]


@pytest.mark.parametrize("rescue_pass,expected", [(False, "EXHAUSTED"), (True, "READY_FOR_CONFIRMATION")])
def test_pa_rescue_is_the_only_extension_after_exactly_one_base_pass(
    monkeypatch, rescue_pass, expected,
):
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    base_summaries = _hard_summaries(config, {"C192-2"})
    rescue_id = config["rescue_method"]["method_id"]
    rescue_summaries = {
        rescue_id: {
            json.dumps(cell, sort_keys=True, separators=(",", ":")): {
                "cell": cell, "pass": rescue_pass,
                "q_top": 0.202, "q_top_mcse": 0.01,
                "core_seconds": 2.0,
            }
            for cell in config["hard_screen"]["cells"]
        }
    }
    base = {
        "manifest": {
            "source_commit": SOURCE_COMMIT,
            "method_ids": [method["method_id"] for method in config["base_methods"]],
            "stage": "hard_screen",
        },
        "complete": True, "records": [],
    }
    rescue = {
        "manifest": {
            "source_commit": SOURCE_COMMIT,
            "method_ids": [rescue_id], "stage": "rescue",
        },
        "complete": True, "records": [],
    }
    monkeypatch.setattr(
        pa_workflow, "load_pa_stage",
        lambda *args, **kwargs: base if args[4] == "hard_screen" else rescue,
    )
    monkeypatch.setattr(
        pa_workflow, "_summarize_records",
        lambda stage, *args, **kwargs: (
            base_summaries if stage["manifest"]["stage"] == "hard_screen"
            else rescue_summaries
        ),
    )
    report = pa_workflow.analyze_hard_screen(
        "base_raw", "base_manifest", REGISTRY_PATH, CONFIG_PATH,
        "rescue_raw", "rescue_manifest",
    )
    assert report["status"] == expected
    if rescue_pass:
        assert [report["primary"], report["backup"]] == ["C192-2", rescue_id]


def _valid_hard_report(config, registry):
    primary, backup = "B96-1", "C192-2"
    cell_rows = {
        json.dumps(cell, sort_keys=True, separators=(",", ":")): {
            "cell": cell, "pass": True,
        }
        for cell in config["hard_screen"]["cells"]
    }
    report = {
        "report_version": "exp102.q0_pa.report.v1",
        "report_kind": "hard_screen",
        "status": "READY_FOR_CONFIRMATION",
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "base_complete": True,
        "rescue_complete": None,
        "base_passing_methods": [primary, backup],
        "candidate_methods": [primary, backup],
        "method_cells": {primary: cell_rows, backup: cell_rows},
        "pair_evidence": [{
            "methods": [primary, backup],
            "cells": [
                {"cell": cell, "pass": True, "abs_delta": 0.0,
                 "sigma_limit": 0.01}
                for cell in config["hard_screen"]["cells"]
            ],
            "pass": True,
        }],
        "primary": primary,
        "backup": backup,
        "raw_evidence": [],
    }
    report["analysis_sha256"] = sha256_json({
        key: value for key, value in report.items() if key != "raw_evidence"
    })
    return report


def test_confirmation_freeze_revalidates_hard_report_and_exact_task_counts(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    hard_path = tmp_path / "hard.json"
    atomic_json(hard_path, _valid_hard_report(config, registry))
    freeze = pa_workflow.freeze_confirmation_manifests(
        hard_path, REGISTRY_PATH, CONFIG_PATH,
        tmp_path / "confirmation.json", tmp_path / "resolution.json",
        tmp_path / "freeze.json",
    )
    assert freeze["confirmation_task_count"] == 272
    assert freeze["resolution_task_count"] == 96

    tampered = _valid_hard_report(config, registry)
    tampered["primary"] = "B192-1"
    tampered_path = tmp_path / "tampered_hard.json"
    atomic_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="not eligible"):
        pa_workflow.freeze_confirmation_manifests(
            tampered_path, REGISTRY_PATH, CONFIG_PATH,
            tmp_path / "confirmation_2.json", tmp_path / "resolution_2.json",
            tmp_path / "freeze_2.json",
        )


def _autopsy_stage_evidence(root):
    registry = load_registry(REGISTRY_PATH)
    discovery = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    config = load_autopsy_config(AUTOPSY_CONFIG_PATH, registry, discovery)
    control_path = root / "control/autopsy.json"
    control = write_autopsy_task_manifest(
        AUTOPSY_CONFIG_PATH, SOURCE_COMMIT, control_path,
    )
    control_sha = sha256_file(control_path)
    ownership = fixed_autopsy_ownership(
        control["tasks"], ["nd-2", "nd-3"], SOURCE_COMMIT, control_sha,
    )
    ownership_path = root / "control/autopsy_ownership.json"
    atomic_json(ownership_path, ownership)
    ownership_sha = sha256_file(ownership_path)
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "file_count": 20,
    }
    task_by_fingerprint = {
        sha256_json(task): task for task in autopsy_tasks(config, SOURCE_COMMIT)
    }
    node_roots = {}
    for node in ownership["nodes"]:
        node_root = root / "transport_autopsy" / control_sha[:12] / node
        node_roots[node] = node_root
        files = []
        assigned = sorted(
            fingerprint for fingerprint, owner in ownership["task_owner"].items()
            if owner == node
        )
        for index, fingerprint in enumerate(assigned):
            assert fingerprint in task_by_fingerprint
            path = node_root / "raw" / f"{index:03d}.npz"
            atomic_npz(path, marker=np.array(index, dtype=np.int16))
            files.append({
                "task_fingerprint": fingerprint,
                "path": path.relative_to(node_root).as_posix(),
                "sha256": sha256_file(path),
            })
        raw_manifest = {
            "raw_manifest_version": AUTOPSY_RAW_VERSION,
            "node": node,
            "stage": "transport_autopsy",
            "stage_fingerprint": ownership["stage_fingerprint"],
            "source_commit": SOURCE_COMMIT,
            "control_sha256": control_sha,
            "ownership_sha256": ownership_sha,
            "source_identity": source_identity,
            "files": files,
        }
        raw_manifest_path = node_root / "raw_manifest.json"
        atomic_json(raw_manifest_path, raw_manifest)
        atomic_json(node_root / "stage_status.json", {
            "status": "SUCCESS", "node": node,
            "stage_fingerprint": ownership["stage_fingerprint"],
            "expected": len(files), "computed": len(files), "reused": 0,
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
        })
        atomic_json(node_root / "SUCCESS", {
            "stage_fingerprint": ownership["stage_fingerprint"],
            "completed_utc": "2026-07-21T00:00:00Z",
        })
    return config, ownership, node_roots


def test_autopsy_remote_evidence_binds_ownership_hashes_and_markers(tmp_path):
    config, ownership, _ = _autopsy_stage_evidence(tmp_path)
    verified = _verified_autopsy_paths(tmp_path, config, SOURCE_COMMIT)
    assert len(verified["paths"]) == 4
    assert verified["source_identity"]["archive_sha256"] == "a" * 64
    assert verified["stage_fingerprint"] == ownership["stage_fingerprint"]


def test_autopsy_remote_evidence_rejects_tampered_success_marker(tmp_path):
    config, _, roots = _autopsy_stage_evidence(tmp_path)
    marker = roots["nd-2"] / "SUCCESS"
    value = json.loads(marker.read_text(encoding="ascii"))
    value["stage_fingerprint"] = "c" * 64
    atomic_json(marker, value)
    with pytest.raises(ValueError, match="SUCCESS/status marker"):
        _verified_autopsy_paths(tmp_path, config, SOURCE_COMMIT)
