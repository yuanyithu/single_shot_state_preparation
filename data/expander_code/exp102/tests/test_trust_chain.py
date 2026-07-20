import shutil
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, atomic_npz, sha256_json
from data.expander_code.exp102.exp102_pipeline.pilot import pilot_status
from data.expander_code.exp102.exp102_pipeline.pilot_cell import pilot_task_identity, run_cell
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.worker import run_task


EXP102_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_STAGE = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.run_production_stage"
)
LADDER_STAGE = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.run_ladder_stage"
)


def test_pilot_status_requires_concrete_tasks_and_counts_exact_outputs(tmp_path):
    tasks = [
        {"namespace": "pilot_ladder_m3_attempt0", "cell": 0},
        {"namespace": "pilot_ladder_m3_attempt0", "cell": 1},
    ]
    manifest = tmp_path / "deployment.json"
    raw = tmp_path / "raw"
    atomic_json(manifest, {"tasks": tasks})
    atomic_npz(
        raw / "cell.npz",
        task_fingerprint=np.array(sha256_json(tasks[0])),
        valid=np.array(True),
    )

    status = pilot_status(manifest, raw)
    assert {key: status[key] for key in (
        "expected", "computed", "reused", "invalid", "missing", "conflict",
    )} == {
        "expected": 2, "computed": 0, "reused": 0,
        "invalid": 0, "missing": 2, "conflict": 1,
    }

    shutil.copy2(raw / "cell.npz", raw / "duplicate.npz")
    status = pilot_status(manifest, raw)
    assert status["computed"] == 0
    assert status["reused"] == 0
    assert status["missing"] == 2
    assert status["conflict"] == 2

    atomic_json(manifest, {"tuning_tasks": tasks})
    with pytest.raises(ValueError, match="concrete tasks"):
        pilot_status(manifest, raw)


def test_pilot_status_accepts_a_complete_runner_output(tmp_path):
    registry_path = EXP102_ROOT / "registry/registry.json"
    config_path = EXP102_ROOT / "config/production.v1.json"
    registry = load_registry(registry_path)
    config = load_config(config_path)
    code = next(row for row in registry["codes"] if row["code_id"] == "m03_c00")
    candidate = _approved_candidate(config)
    source_commit = "1" * 40
    task = pilot_task_identity(
        registry["registry_sha256"], config["config_sha256"], code["code_id"],
        code["m"], 0.04, 0, candidate, 0, "ladder", source_commit,
    )
    manifest = tmp_path / "deployment.json"
    raw = tmp_path / "raw"
    output = raw / "cell.npz"
    atomic_json(manifest, {"tasks": [task]})
    assert run_cell(
        registry_path, config_path, code["code_id"], 0.04, 0, candidate,
        0, "ladder", source_commit, output,
    ) == "computed"

    status = pilot_status(manifest, raw)
    assert status["expected"] == 1
    assert status["computed"] + status["invalid"] == 1
    assert status["missing"] == status["conflict"] == 0


def test_stage_manifest_contains_worker_canonical_task_identities():
    registry = load_registry(EXP102_ROOT / "registry/registry.json")
    config = load_config(EXP102_ROOT / "config/production.v1.json")
    candidate = _approved_candidate(config)
    tasks = LADDER_STAGE.build_manifest_tasks(
        registry, config, {3: candidate}, {3}, "ladder", 7, "1" * 40,
    )

    assert len(tasks) == 96
    assert len({sha256_json(task) for task in tasks}) == 96
    assert {task["namespace"] for task in tasks} == {"pilot_ladder_m3_attempt7"}
    assert {task["candidate"]["num_temperatures"] for task in tasks} == {8}


def _runner_inputs(tmp_path, frozen, deployment):
    paths = {
        "registry": tmp_path / "registry.json",
        "config": tmp_path / "config.json",
        "frozen": tmp_path / "frozen.json",
        "report": tmp_path / "report.json",
        "plan": tmp_path / "plan.json",
        "deployment": tmp_path / "deployment.json",
        "run_root": tmp_path / "run",
    }
    for key in ("registry", "config", "report", "plan"):
        atomic_json(paths[key], {})
    atomic_json(paths["frozen"], frozen)
    atomic_json(paths["deployment"], deployment)
    return paths


def _patch_runner_cli(monkeypatch, paths):
    monkeypatch.setattr(PRODUCTION_STAGE, "load_registry", lambda _: {
        "registry_sha256": "registry", "codes": [],
    })
    monkeypatch.setattr(PRODUCTION_STAGE, "load_config", lambda _: {
        "config_sha256": "config",
    })
    monkeypatch.setattr(sys, "argv", [
        "run_production_stage.py", "nd-1", "--num-workers", "75",
        "--run-root", str(paths["run_root"]),
        "--registry", str(paths["registry"]),
        "--config", str(paths["config"]),
        "--frozen", str(paths["frozen"]),
        "--pilot-report", str(paths["report"]),
        "--task-plan", str(paths["plan"]),
        "--deployment-manifest", str(paths["deployment"]),
    ])


def test_production_stage_rejects_tampered_freezer(monkeypatch, tmp_path):
    frozen = {"status": "FORGED", "engine": "numba", "source_commit": "1" * 40}
    paths = _runner_inputs(tmp_path, frozen, {})
    _patch_runner_cli(monkeypatch, paths)
    monkeypatch.setattr(PRODUCTION_STAGE, "verify_source_identity", lambda *_: {"mode": "test"})
    monkeypatch.setattr(PRODUCTION_STAGE, "build_manifest", lambda *_: {})

    with pytest.raises(ValueError, match="held-out certified"):
        PRODUCTION_STAGE.main()


def test_production_stage_rejects_tampered_deployment(monkeypatch, tmp_path):
    frozen = {
        "status": "FROZEN_HELD_OUT_PASS", "engine": "numba",
        "source_commit": "1" * 40,
    }
    paths = _runner_inputs(tmp_path, frozen, {"tampered": True})
    _patch_runner_cli(monkeypatch, paths)
    monkeypatch.setattr(PRODUCTION_STAGE, "verify_source_identity", lambda *_: {"mode": "test"})
    monkeypatch.setattr(PRODUCTION_STAGE, "build_manifest", lambda *_: {"tampered": False})

    with pytest.raises(ValueError, match="differs from recomputed"):
        PRODUCTION_STAGE.main()


def test_production_stage_rejects_source_mismatch(monkeypatch, tmp_path):
    frozen = {
        "status": "FROZEN_HELD_OUT_PASS", "engine": "numba",
        "source_commit": "1" * 40,
    }
    paths = _runner_inputs(tmp_path, frozen, {})
    _patch_runner_cli(monkeypatch, paths)

    def reject_source(*_):
        raise ValueError("source identity mismatch")

    monkeypatch.setattr(PRODUCTION_STAGE, "verify_source_identity", reject_source)
    with pytest.raises(ValueError, match="source identity mismatch"):
        PRODUCTION_STAGE.main()


def _approved_candidate(config):
    ladder = config["pilot"]["ladder_candidates"][0]
    rounds = config["pilot"]["round_candidates"][0]
    return {
        "p_hot": ladder["p_hot"], "num_temperatures": ladder["num_temperatures"],
        "gamma": 1.0, "burn_rounds": rounds[0], "measurement_rounds": rounds[1],
        "sweeps_per_round": 1, "logical_move_repeat": 1,
    }


def test_pilot_cell_does_not_reuse_fingerprint_only_npz(tmp_path):
    registry_path = EXP102_ROOT / "registry/registry.json"
    config_path = EXP102_ROOT / "config/production.v1.json"
    registry = load_registry(registry_path)
    config = load_config(config_path)
    candidate = _approved_candidate(config)
    source_commit = "1" * 40
    identity = {
        "namespace": "pilot_ladder_m3_attempt0", "stage": "ladder",
        "code_id": "m03_c00", "p": 0.04, "disorder_index": 0,
        "candidate": candidate, "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"], "source_commit": source_commit,
        "engine": "numba",
    }
    output = tmp_path / "pilot.npz"
    atomic_npz(output, task_fingerprint=np.array(sha256_json(identity)))

    with pytest.raises(ValueError):
        run_cell(
            registry_path, config_path, "m03_c00", 0.04, 0, candidate,
            0, "ladder", source_commit, output,
        )


def test_production_worker_does_not_reuse_fingerprint_only_npz(monkeypatch, tmp_path):
    registry_path = EXP102_ROOT / "registry/registry.json"
    config_path = EXP102_ROOT / "config/production.v1.json"
    registry = load_registry(registry_path)
    config = load_config(config_path)
    source_commit = "1" * 40
    frozen = {
        "status": "FROZEN_HELD_OUT_PASS", "engine": "numba",
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "by_m": {str(m): _approved_candidate(config) for m in range(3, 9)},
    }
    frozen_path = tmp_path / "frozen.json"
    atomic_json(frozen_path, frozen)
    frozen_hash = sha256_json(frozen)
    monkeypatch.setenv("EXP102_FROZEN_VERIFIED_SHA256", frozen_hash)
    monkeypatch.setenv("EXP102_SOURCE_VERIFIED_COMMIT", source_commit)
    identity = {
        "code_id": "m03_c00", "disorder_index": 0,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": frozen_hash, "namespace": "production",
    }
    output = tmp_path / "production.npz"
    atomic_npz(output, task_fingerprint=np.array(sha256_json(identity)))

    with pytest.raises(ValueError):
        run_task(registry_path, config_path, frozen_path, "m03_c00", 0, output)
