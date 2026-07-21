import importlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import global_discovery as gd
from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    GLOBAL_TASKS_VERSION,
    NODE_CAPACITY,
    build_bias_manifest,
    build_measurement_manifest,
    build_ti_anchor_manifest,
    combine_global_readiness,
    fixed_global_ownership,
    freeze_global_schedule,
    freeze_postselection_plan,
    load_global_discovery_config,
    run_ti_anchor_task,
    ti_anchor_task_identity,
    validate_ti_anchor_raw,
    validate_global_control_manifest,
    verify_global_remote_evidence,
)
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_global.discovery.v1.json"
SOURCE_COMMIT = "1" * 40


def test_verified_analysis_cli_requires_the_complete_evidence_tuple():
    empty = SimpleNamespace(
        run_root=None, ownership=None, deployment_root=None, schedule=None,
        manifest="control.json",
    )
    assert gd._cli_verified_evidence(empty) is None
    partial = SimpleNamespace(
        run_root="run", ownership=None, deployment_root=None, schedule=None,
        manifest="control.json",
    )
    with pytest.raises(ValueError, match="requires run root"):
        gd._cli_verified_evidence(partial)
    complete = SimpleNamespace(
        run_root="run", ownership="ownership.json", deployment_root="repo",
        schedule="schedule.json", manifest="control.json",
    )
    assert gd._cli_verified_evidence(complete) == {
        "run_root": "run", "control_path": "control.json",
        "ownership_path": "ownership.json", "deployment_root": "repo",
        "schedule_path": "schedule.json",
    }


def test_global_bias_and_hard_measurement_manifests_are_frozen_and_complete(tmp_path):
    bias_path = tmp_path / "bias_manifest.json"
    bias = build_bias_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("DT16", "T1"), ("DT32", "T1"), ("DT64", "T1")], bias_path,
    )
    assert bias["manifest_version"] == GLOBAL_TASKS_VERSION
    assert len(bias["tasks"]) == 3 * 5
    assert len({entry["task_fingerprint"] for entry in bias["tasks"]}) == 15
    assert all(entry["output_relpath"].startswith("bias/") for entry in bias["tasks"])

    measurement_path = tmp_path / "measurement_manifest.json"
    measurement = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("RC8-QC1", "T1"), ("RC8-J08", "T1")], measurement_path,
    )
    assert len(measurement["tasks"]) == 2 * 5 * 2 * 16
    assert len({entry["task_fingerprint"] for entry in measurement["tasks"]}) == 320
    assert all(entry["bias_relpath"] is None for entry in measurement["tasks"])
    assert json.loads(measurement_path.read_text(encoding="ascii")) == measurement

    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    assert validate_global_control_manifest(bias, registry, config)
    assert validate_global_control_manifest(measurement, registry, config)
    ti = build_ti_anchor_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, tmp_path / "ti.json",
    )
    assert validate_global_control_manifest(ti, registry, config)

    tampered = json.loads(json.dumps(measurement))
    tampered["tasks"][0]["output_relpath"] = "../../escaped.npz"
    with pytest.raises(gd.GlobalConflictError, match="fingerprint/path"):
        validate_global_control_manifest(tampered, registry, config)


def test_global_ownership_is_deterministic_complete_and_capacity_aware(tmp_path):
    manifest = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("RC8-QC1", "T1"), ("RC8-J16", "T1")],
        tmp_path / "measurement.json",
    )
    tasks = [entry["task"] for entry in manifest["tasks"]]
    ownership = fixed_global_ownership(
        tasks, ["nd-2", "nd-3"], SOURCE_COMMIT, "a" * 64,
        "screen:measurement",
    )
    repeated = fixed_global_ownership(
        tasks, ["nd-2", "nd-3"], SOURCE_COMMIT, "a" * 64,
        "screen:measurement",
    )
    assert ownership == repeated
    assert set(ownership["task_owner"]) == {sha256_json(task) for task in tasks}
    assert set(ownership["task_owner"].values()) == {"nd-2", "nd-3"}
    normalized = {
        node: ownership["weighted_load"][node] / NODE_CAPACITY[node]
        for node in ownership["nodes"]
    }
    assert max(normalized.values()) / min(normalized.values()) < 1.10


def test_defect_measurement_manifest_refuses_unfrozen_or_missing_bias(tmp_path):
    with pytest.raises(ValueError, match="bias evidence"):
        build_measurement_manifest(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
            [("DT16", "T1")], tmp_path / "measurement.json",
        )


@pytest.fixture(scope="module")
def local_runtime_report():
    benchmark = importlib.import_module(
        "data.expander_code.exp102.validation."
        "007_q0_global_discovery_20260721.benchmark_global"
    )
    return benchmark.run_benchmark(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT,
    )


@pytest.fixture(scope="module")
def passing_runtime_report(local_runtime_report):
    """Canonical PASS-shaped fixture for deterministic consensus tests.

    The live report remains a real performance probe.  Whether the current
    host fits the frozen wall window belongs to the dedicated three-node
    preflight gate, not to a platform-independent unit-test assertion.
    """
    report = json.loads(json.dumps(local_runtime_report))
    for projection in report["projections"]:
        projection["pass"] = True
        projection["projected_hours_with_safety_factor_2"] = min(
            1.0, float(projection["projected_hours_with_safety_factor_2"]),
        )
        projection["eligible_methods"] = list(
            projection["trajectory_seconds_m8"]
        )
    report["selected_resource_tier"] = report["projections"][-1][
        "resource_tier"
    ]
    report["selected_eligible_methods"] = report["projections"][-1][
        "eligible_methods"
    ]
    report["ti_anchor_projection"]["pass"] = True
    report["checks"] = {key: True for key in report["checks"]}
    report["status"] = "PASS"
    return report


def test_global_runtime_report_is_self_consistent_on_host(local_runtime_report):
    report = local_runtime_report
    assert report["status"] == (
        "PASS" if all(report["checks"].values()) else "RUNTIME_EXHAUSTED"
    )
    passing = [value for value in report["projections"] if value["pass"]]
    expected_tier = passing[-1]["resource_tier"] if passing else None
    assert report["selected_resource_tier"] == expected_tier
    assert all(
        value["pass"]
        == (value["projected_hours_with_safety_factor_2"] <= 58.0)
        for value in report["projections"]
    )
    if expected_tier is not None:
        selected = passing[-1]
        assert report["selected_eligible_methods"] == selected["eligible_methods"]
        assert set(selected["eligible_methods"]) >= {"RC8-QC1", "DT16"}
    assert isinstance(report["ti_anchor_projection"]["pass"], bool)


def test_three_node_runtime_and_digest_consensus_are_fail_closed(
        tmp_path, passing_runtime_report):
    benchmark = importlib.import_module(
        "data.expander_code.exp102.validation."
        "007_q0_global_discovery_20260721.benchmark_global"
    )
    cross = importlib.import_module(
        "data.expander_code.exp102.validation."
        "007_q0_global_discovery_20260721.cross_node_global"
    )
    source_identity = {
        "source_commit": SOURCE_COMMIT, "mode": "archive",
        "archive_sha256": "a" * 64, "manifest_sha256": "b" * 64,
        "file_count": 10,
    }
    runtime_paths = {}
    digest_paths = {}
    digest_base = cross.canonical_digest(REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT)
    for node in ("nd-1", "nd-2", "nd-3"):
        runtime = json.loads(json.dumps(passing_runtime_report))
        runtime.update({
            "node": node, "source_identity": source_identity,
            "completed_unix": time.time(),
        })
        runtime["environment"]["system"] = "Linux"
        runtime_path = tmp_path / f"runtime_{node}.json"
        atomic_json(runtime_path, runtime)
        runtime_paths[node] = runtime_path

        digest = {
            **digest_base,
            "source_commit": SOURCE_COMMIT,
            "source_identity": source_identity,
            "node": node,
            "completed_unix": time.time(),
            "environment": {"system": "Linux"},
        }
        digest_path = tmp_path / f"digest_{node}.json"
        atomic_json(digest_path, digest)
        digest_paths[node] = digest_path
    runtime_consensus = benchmark.combine_runtime_reports(runtime_paths)
    digest_consensus = cross.combine_digest_reports(digest_paths)
    assert runtime_consensus["status"] == "PASS"
    assert digest_consensus["status"] == "PASS"
    tampered = json.loads(runtime_paths["nd-3"].read_text(encoding="ascii"))
    tampered["source_identity"]["archive_sha256"] = "c" * 64
    atomic_json(runtime_paths["nd-3"], tampered)
    with pytest.raises(ValueError, match="consensus evidence"):
        benchmark.combine_runtime_reports(runtime_paths)


def test_preflight_wmc_evidence_binds_source_config_and_ordered_panel():
    preflight = importlib.import_module(
        "data.expander_code.exp102.validation."
        "007_q0_global_discovery_20260721.orchestrate_global_preflight"
    )
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    source_identity = {
        "source_commit": SOURCE_COMMIT, "mode": "archive",
        "archive_sha256": "a" * 64, "manifest_sha256": "b" * 64,
        "file_count": 10,
    }
    report = {
        "report_version": "exp102.q0_global.wmc_feasibility.v1",
        "status": "INCONCLUSIVE",
        "source_commit": SOURCE_COMMIT,
        "source_identity": source_identity,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "node": "nd-1", "environment": {"system": "Linux"},
        "timeout_seconds_per_cell": 7200.0,
        "completed_unix": time.time(),
        "records": [
            {"cell": cell, "status": "INCONCLUSIVE_WIDTH"}
            for cell in config["panels"]["SMALL6"]["cells"]
        ],
    }
    assert preflight.validate_wmc_report(
        report, registry, config, SOURCE_COMMIT, source_identity,
        time.time() + 60.0,
    ) == report
    report["records"] = list(reversed(report["records"]))
    with pytest.raises(ValueError, match="malformed or unverified"):
        preflight.validate_wmc_report(
            report, registry, config, SOURCE_COMMIT, source_identity,
            time.time() + 60.0,
        )


def test_global_control_source_registry_and_config_axes_change_fingerprint(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    first_path = tmp_path / "first.json"
    first = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("RC8-QC1", "T1")], first_path,
    )
    second = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, "2" * 40, "screen",
        [("RC8-QC1", "T1")], tmp_path / "second.json",
    )
    assert sha256_json(first) != sha256_json(second)
    assert first["registry_sha256"] == registry["registry_sha256"]
    assert first["discovery_config_sha256"] == config["discovery_config_sha256"]
    assert json.loads(first_path.read_text(encoding="ascii")) == first


def _synthetic_records(manifest, biased_methods=()):
    records = []
    masks = np.asarray([1, 2, 3], dtype=np.uint64)
    for entry in manifest["tasks"]:
        task = entry["task"]
        seed = int(entry["task_fingerprint"][:16], 16)
        rng = np.random.default_rng(seed)
        if task["method_id"] in biased_methods:
            labels = rng.choice(4, size=1024, p=[0.82, 0.06, 0.06, 0.06]).astype(np.uint64)
        else:
            labels = rng.integers(0, 4, size=1024, dtype=np.uint64)
        records.append({
            "cell": task["cell"],
            "method_id": task["method_id"],
            "resource_tier": task["resource_tier"],
            "init_family": task["init_family"],
            "trajectory_index": task["trajectory_index"],
            "labels": labels,
            "weights": rng.binomial(100, 0.1, size=1024).astype(np.int32),
            "valid_mask": np.ones(1024, dtype=bool),
            "burn_labels": rng.integers(0, 4, size=256, dtype=np.uint64),
            "initial_label": task["trajectory_index"] % 4,
            "num_qubits": 100,
            "k": 2,
            "character_masks": masks,
            "core_seconds": 1.0,
            "task_fingerprint": entry["task_fingerprint"],
        })
    return records


def test_stage_analyzer_includes_method_comparisons_in_status(tmp_path, monkeypatch):
    one_path = tmp_path / "one.json"
    one = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("RC8-QC1", "T1")], one_path,
    )
    monkeypatch.setattr(
        gd, "_load_measurement_records",
        lambda *args, **kwargs: _synthetic_records(one),
    )
    assert gd.analyze_measurement_stage(
        tmp_path, one_path, REGISTRY_PATH, CONFIG_PATH,
    )["status"] == "PASS"

    two_path = tmp_path / "two.json"
    two = build_measurement_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "screen",
        [("RC8-QC1", "T1"), ("RC8-J08", "T1")], two_path,
    )
    monkeypatch.setattr(
        gd, "_load_measurement_records",
        lambda *args, **kwargs: _synthetic_records(two, {"RC8-J08"}),
    )
    report = gd.analyze_measurement_stage(
        tmp_path, two_path, REGISTRY_PATH, CONFIG_PATH,
    )
    assert all(value["valid"] for value in report["method_status"])
    assert not all(value["valid"] for value in report["comparisons"])
    assert report["status"] == "SAMPLING_INSUFFICIENT"


def test_defect_bias_replay_is_cached_across_trajectories(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    trajectories = raw_root / "trajectories"
    bias_dir = raw_root / "bias"
    trajectories.mkdir(parents=True)
    bias_dir.mkdir()
    bias_path = bias_dir / "shared.npz"
    bias_path.write_bytes(b"bias")
    entries = []
    for index in range(2):
        path = trajectories / f"t{index}.npz"
        path.write_bytes(b"trajectory")
        task = {"method_id": "DT16"}
        entries.append({
            "task": task,
            "task_fingerprint": str(index),
            "output_relpath": f"trajectories/t{index}.npz",
            "bias_relpath": "bias/shared.npz",
        })
    manifest = {"source_commit": SOURCE_COMMIT, "tasks": entries}
    calls = {"bias": 0, "defect": 0}
    cached = {
        "path": str(bias_path.resolve()), "sha256": sha256_file(bias_path),
        "task": {
            "registry_sha256": "r", "discovery_config_sha256": "c",
            "source_commit": SOURCE_COMMIT,
        },
    }

    def fake_bias(*args):
        calls["bias"] += 1
        return cached

    def fake_defect(path, registry, config, source, bound, *,
                    _validated_bias_record=None):
        calls["defect"] += 1
        assert _validated_bias_record is cached
        index = int(Path(path).stem[1:])
        return {"task_fingerprint": str(index)}

    monkeypatch.setattr(gd, "validate_bias_raw", fake_bias)
    monkeypatch.setattr(gd, "validate_defect_raw", fake_defect)
    records = gd._load_measurement_records(
        raw_root, manifest,
        {"registry_sha256": "r"},
        {"discovery_config_sha256": "c"},
    )
    assert len(records) == 2
    assert calls == {"bias": 1, "defect": 2}


def _fake_ti_result(k):
    labels = list(range(1 << k))
    weights = np.full(1 << k, 1.0 / (1 << k))
    return {
        "labels": labels,
        "kp_grid": np.asarray([0.0, 1.0]),
        "delta_f": np.zeros(1 << k),
        "delta_f_infinite_mask": np.zeros(1 << k, dtype=bool),
        "delta_f_stderr": np.zeros(1 << k),
        "acceptance_per_label": {value: np.ones(2) for value in labels},
        "weights_absolute": weights,
        "characters_absolute": np.zeros((1 << k) - 1),
        "q_top": 0.0,
        "q_top_stderr": 0.0,
        "grid_tv": 0.0,
        "grid_q_top_abs_diff": 0.0,
        "flags": "PASS",
        "valid_for_aggregation": True,
        "proposal_summary": {"num_stab": 1, "signature_group_sizes": []},
    }


def test_ti_anchor_manifest_raw_and_replay_are_fail_closed(tmp_path, monkeypatch):
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    cell = config["panels"]["SMALL6"]["cells"][0]
    task = ti_anchor_task_identity(registry, config, SOURCE_COMMIT, cell)
    _, code, H = load_frozen_code(REGISTRY_PATH, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed = gd.uniform_seed_for_cell(registry, code, cell)
    fake = _fake_ti_result(model.k)
    monkeypatch.setattr(
        gd, "_execute_ti_anchor",
        lambda *args: (code, model, frame, uniform_seed, fake),
    )
    raw = tmp_path / "anchor.npz"
    assert run_ti_anchor_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, task, raw,
    ) == "computed"
    record = validate_ti_anchor_raw(
        raw, registry, config, REGISTRY_PATH, SOURCE_COMMIT,
    )
    assert record["valid_for_aggregation"]
    with np.load(raw, allow_pickle=False) as data:
        assert set(data.files) == gd.TI_ANCHOR_RAW_FIELDS

    manifest_path = tmp_path / "ti_manifest.json"
    manifest = build_ti_anchor_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, manifest_path,
    )
    assert len(manifest["tasks"]) == 3


def _make_remote_evidence(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    archive = deployment / "SOURCE.tar"
    archive.write_bytes(b"verified archive")
    archive_sha = sha256_file(archive)
    (deployment / "SOURCE_COMMIT").write_text(SOURCE_COMMIT + "\n", encoding="ascii")
    (deployment / "ARCHIVE_SHA256").write_text(archive_sha + "\n", encoding="ascii")
    source_manifest = {
        "source_identity_version": "exp102.source.v1",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": archive_sha,
        "files": [{"path": "dummy", "sha256": "0" * 64}],
    }
    atomic_json(deployment / "SOURCE_MANIFEST.json", source_manifest)
    schedule_path = tmp_path / "schedule.json"
    freeze_global_schedule(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, archive_sha,
        sha256_file(deployment / "SOURCE_MANIFEST.json"), schedule_path,
        started_unix=time.time() - 60.0,
    )
    schedule = json.loads(schedule_path.read_text(encoding="ascii"))
    tasks = []
    entries = []
    for index in range(2):
        task = {
            "cell": {"code_id": "m03_c00"},
            "method_id": "RC8-QC1",
            "sampler_config": {"burn_sweeps": 1, "measurement_sweeps": 8,
                               "cluster_repeats": 1, "joint_block_size": 0},
            "trajectory_index": index,
        }
        fingerprint = sha256_json(task)
        tasks.append(task)
        entries.append({
            "task": task, "task_fingerprint": fingerprint,
            "output_relpath": f"trajectories/{fingerprint}.npz",
            "bias_relpath": None,
        })
    control = {
        "manifest_version": GLOBAL_TASKS_VERSION,
        "kind": "measurement", "stage": "screen",
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "tasks": entries,
    }
    control_path = tmp_path / "control.json"
    atomic_json(control_path, control)
    control_sha = sha256_file(control_path)
    ownership = fixed_global_ownership(
        tasks, ["nd-2", "nd-3"], SOURCE_COMMIT, control_sha,
        "screen:measurement",
    )
    ownership_path = tmp_path / "ownership.json"
    atomic_json(ownership_path, ownership)
    ownership_sha = sha256_file(ownership_path)
    run_root = tmp_path / "run"
    evidence_root = run_root / "global/screen"
    for entry in entries:
        raw = evidence_root / entry["output_relpath"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(entry["task_fingerprint"].encode("ascii"))
    for node in ownership["nodes"]:
        owned = [
            entry for entry in entries
            if ownership["task_owner"][entry["task_fingerprint"]] == node
        ]
        node_root = evidence_root / "node_manifests" / control_sha[:12] / node
        marker_root = evidence_root / "markers" / control_sha[:12] / node
        raw_manifest = {
            "raw_manifest_version": "measurement", "node": node,
            "stage": "screen", "kind": "measurement",
            "stage_fingerprint": ownership["stage_fingerprint"],
            "source_commit": SOURCE_COMMIT, "control_sha256": control_sha,
            "ownership_sha256": ownership_sha,
            "schedule_file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "source_identity": {
                "source_commit": SOURCE_COMMIT, "mode": "archive",
                "archive_sha256": archive_sha,
                "manifest_sha256": sha256_file(deployment / "SOURCE_MANIFEST.json"),
            },
            "files": [{
                "task_fingerprint": entry["task_fingerprint"],
                "path": entry["output_relpath"],
                "sha256": sha256_file(evidence_root / entry["output_relpath"]),
            } for entry in owned],
        }
        atomic_json(node_root / "raw_manifest.json", raw_manifest)
        atomic_json(node_root / "stage_status.json", {
            "status": "SUCCESS",
            "stage_fingerprint": ownership["stage_fingerprint"],
            "raw_manifest_sha256": sha256_file(node_root / "raw_manifest.json"),
        })
        atomic_json(marker_root / "SUCCESS", {
            "stage_fingerprint": ownership["stage_fingerprint"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return {
        "run_root": run_root,
        "control_path": control_path,
        "ownership_path": ownership_path,
        "deployment_root": deployment,
        "schedule_path": schedule_path,
    }, evidence_root / entries[0]["output_relpath"]


def test_remote_evidence_verifies_schedule_source_markers_and_raw_sha(tmp_path):
    arguments, first_raw = _make_remote_evidence(tmp_path)
    result = verify_global_remote_evidence(**arguments)
    assert result["schedule_sha256"]
    first_raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw SHA256"):
        verify_global_remote_evidence(**arguments)


def _synthetic_stage_report(stage, method_tiers, cells, registry, config,
                            schedule, schedule_path):
    summaries = [{
        "cell": cell,
        "method_id": method,
        "resource_tier": tier,
        "valid": True,
        "failures": [],
    } for cell in cells for method, tier in method_tiers]
    comparisons = []
    for cell in cells:
        for left_index, left in enumerate(method_tiers):
            for right in method_tiers[left_index + 1:]:
                comparisons.append({
                    "cell": cell,
                    "left": {"method_id": left[0], "resource_tier": left[1]},
                    "right": {"method_id": right[0], "resource_tier": right[1]},
                    "valid": True,
                })
    return {
        "report_version": gd.GLOBAL_REPORT_VERSION,
        "stage": stage,
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "raw_count": len(summaries) * 32,
        "cell_summaries": summaries,
        "method_status": [{
            "method_id": method, "resource_tier": tier,
            "cells_total": len(cells), "cells_passed": len(cells),
            "valid": True,
        } for method, tier in method_tiers],
        "comparisons": comparisons,
        "verified_remote_evidence": {
            "schedule_file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "completed_unix_max": time.time(),
        },
        "status": "PASS",
    }


def _frozen_postselection(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    schedule_path = tmp_path / "schedule.json"
    schedule = freeze_global_schedule(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "a" * 64, "b" * 64,
        schedule_path, started_unix=time.time() - 60.0,
    )
    selection_identity = {
        "selection_version": gd.GLOBAL_SELECTION_VERSION,
        "status": "FROZEN_DISCOVERY_METHODS",
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "selected": [
            {"method_id": "RC8-QC1", "resource_tier": "T1"},
            {"method_id": "DT16", "resource_tier": "T1"},
        ],
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_sha256": schedule["schedule_sha256"],
    }
    selection = {
        **selection_identity,
        "selection_sha256": sha256_json(selection_identity),
    }
    selection_path = tmp_path / "selection.json"
    atomic_json(selection_path, selection)
    plan_path = tmp_path / "postselection.json"
    freeze_postselection_plan(
        selection_path, REGISTRY_PATH, CONFIG_PATH, schedule_path, plan_path,
    )
    controls_dir = tmp_path / "controls"
    controls_index = controls_dir / "control_freeze.json"
    controls = gd.prepare_postselection_controls(
        selection_path, plan_path, REGISTRY_PATH, CONFIG_PATH, schedule_path,
        controls_dir, controls_index,
    )
    return {
        "registry": registry,
        "config": config,
        "schedule": schedule,
        "schedule_path": schedule_path,
        "selection_path": selection_path,
        "plan_path": plan_path,
        "controls": controls,
        "controls_index": controls_index,
    }


def test_postselection_materialization_rejects_tampered_controls(
        tmp_path, monkeypatch):
    frozen = _frozen_postselection(tmp_path)

    def fake_bias(path, registry, config, source_commit):
        return {
            "task_fingerprint": Path(path).stem,
            "sha256": "c" * 64,
            "bias_sha256": "d" * 64,
        }

    monkeypatch.setattr(gd, "validate_bias_raw", fake_bias)
    output_path = tmp_path / "hard_fresh_measurement.json"
    manifest = gd.materialize_postselection_measurement(
        "hard_fresh", frozen["controls_index"], frozen["plan_path"],
        REGISTRY_PATH, CONFIG_PATH, tmp_path / "bias_raw", output_path,
    )
    controls = frozen["controls"]
    assert manifest["postselection_plan_sha256"] == controls[
        "postselection_plan_sha256"
    ]
    assert manifest["control_freeze_sha256"] == controls[
        "control_freeze_sha256"
    ]
    assert manifest["bias_manifest_sha256"] == controls[
        "bias_controls"
    ]["hard_fresh"]["manifest_sha256"]

    bias_path = frozen["controls_index"].parent / controls[
        "bias_controls"
    ]["hard_fresh"]["filename"]
    original_bias = bias_path.read_bytes()
    bias_path.write_bytes(original_bias + b"\n")
    with pytest.raises(gd.GlobalConflictError, match="bias control changed"):
        gd.materialize_postselection_measurement(
            "hard_fresh", frozen["controls_index"], frozen["plan_path"],
            REGISTRY_PATH, CONFIG_PATH, tmp_path / "bias_raw", output_path,
        )
    bias_path.write_bytes(original_bias)

    tampered_index = json.loads(
        frozen["controls_index"].read_text(encoding="ascii")
    )
    tampered_index["source_commit"] = "2" * 40
    atomic_json(frozen["controls_index"], tampered_index)
    with pytest.raises(gd.GlobalConflictError, match="control index is invalid"):
        gd.materialize_postselection_measurement(
            "hard_fresh", frozen["controls_index"], frozen["plan_path"],
            REGISTRY_PATH, CONFIG_PATH, tmp_path / "bias_raw", output_path,
        )


def test_readiness_combiner_requires_every_stage_ti_and_frozen_plan(tmp_path):
    frozen = _frozen_postselection(tmp_path)
    registry = frozen["registry"]
    config = frozen["config"]
    schedule = frozen["schedule"]
    schedule_path = frozen["schedule_path"]
    selection_path = frozen["selection_path"]
    plan_path = frozen["plan_path"]
    controls = frozen["controls"]
    controls_index = frozen["controls_index"]
    selected = [("RC8-QC1", "T1"), ("DT16", "T1")]
    reports = {
        "hard": _synthetic_stage_report(
            "hard_fresh",
            [(method, tier) for method, _ in selected for tier in ("T1", "2T1")],
            gd._stage_cells(config, "hard_fresh"), registry, config,
            schedule, schedule_path,
        ),
        "confirmation": _synthetic_stage_report(
            "confirmation", [(method, "2T1") for method, _ in selected],
            gd._stage_cells(config, "confirmation"), registry, config,
            schedule, schedule_path,
        ),
        "resolution": _synthetic_stage_report(
            "resolution", selected, gd._stage_cells(config, "resolution"),
            registry, config, schedule, schedule_path,
        ),
    }
    for report in reports.values():
        stage = report["stage"]
        report.update({
            "postselection_plan_sha256": controls["postselection_plan_sha256"],
            "control_freeze_sha256": controls["control_freeze_sha256"],
            "bias_manifest_sha256": controls[
                "bias_controls"
            ][stage]["manifest_sha256"],
        })
    report_paths = {}
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        atomic_json(path, report)
        report_paths[name] = path
    ti_cells = gd._stage_cells(config, "ti_anchors")
    ti_report = {
        "report_version": gd.TI_ANCHOR_REPORT_VERSION,
        "stage": "ti_anchors", "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "manifest_sha256": controls["ti_control"]["manifest_sha256"],
        "raw_count": len(ti_cells),
        "anchors": [
            {"cell": cell, "valid_for_aggregation": True} for cell in ti_cells
        ],
        "verified_remote_evidence": {
            "schedule_file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "completed_unix_max": time.time(),
        },
        "status": "PASS",
    }
    ti_path = tmp_path / "ti.json"
    atomic_json(ti_path, ti_report)
    ti_comparison = {
        "report_version": gd.TI_COMPARISON_VERSION,
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "confirmation_report_sha256": sha256_json(reports["confirmation"]),
        "ti_report_sha256": sha256_json(ti_report),
        "method_tiers": [["DT16", "2T1"], ["RC8-QC1", "2T1"]],
        "comparisons": [
            {"cell": cell, "method_id": method, "resource_tier": "2T1",
             "valid": True}
            for cell in ti_cells for method in ("RC8-QC1", "DT16")
        ],
        "status": "PASS",
    }
    ti_comparison_path = tmp_path / "ti_comparison.json"
    atomic_json(ti_comparison_path, ti_comparison)
    result = combine_global_readiness(
        selection_path, report_paths["hard"], report_paths["confirmation"],
        report_paths["resolution"], ti_path, ti_comparison_path,
        REGISTRY_PATH, CONFIG_PATH, schedule_path, plan_path, controls_index,
    )
    assert result["status"] == "READY_FOR_FORMAL"
    tampered_hard = json.loads(
        report_paths["hard"].read_text(encoding="ascii")
    )
    tampered_hard["control_freeze_sha256"] = "0" * 64
    atomic_json(report_paths["hard"], tampered_hard)
    with pytest.raises(gd.GlobalConflictError, match="frozen controls"):
        combine_global_readiness(
            selection_path, report_paths["hard"], report_paths["confirmation"],
            report_paths["resolution"], ti_path, ti_comparison_path,
            REGISTRY_PATH, CONFIG_PATH, schedule_path, plan_path, controls_index,
        )
    atomic_json(report_paths["hard"], reports["hard"])
    ti_comparison["comparisons"][0]["valid"] = False
    ti_comparison["status"] = "SAMPLING_INSUFFICIENT"
    atomic_json(ti_comparison_path, ti_comparison)
    result = combine_global_readiness(
        selection_path, report_paths["hard"], report_paths["confirmation"],
        report_paths["resolution"], ti_path, ti_comparison_path,
        REGISTRY_PATH, CONFIG_PATH, schedule_path, plan_path, controls_index,
    )
    assert result["status"] == "UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET"
