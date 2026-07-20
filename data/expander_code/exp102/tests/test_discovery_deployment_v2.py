import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest


orchestrator = importlib.import_module(
    "data.expander_code.exp102.validation.005_pt_v2_discovery_20260720.orchestrate_discovery"
)


def _task(ladder, swap_sweeps, code_id):
    return {
        "cell": {"code_id": code_id},
        "candidate": {
            "ladder_id": ladder,
            "ladder_fingerprint": ladder * 64,
            "num_temperatures": 8,
            "burn_rounds": 5,
            "measurement_rounds": 20,
            "swap_sweeps_per_round": swap_sweeps,
        },
        "ladder_fingerprint": ladder * 64,
    }


def test_fixed_ownership_binds_source_control_ladder_swap_and_m_set():
    tasks = [_task("a", 4, "m05_c00"), _task("b", 16, "m08_c00")]
    first = orchestrator.fixed_ownership(
        tasks, ("nd-1", "nd-2", "nd-3"), "1" * 40, "2" * 64, "screen",
    )
    repeat = orchestrator.fixed_ownership(
        tasks, ("nd-1", "nd-2", "nd-3"), "1" * 40, "2" * 64, "screen",
    )
    assert first == repeat
    assert set(first["task_owner"].values()) <= {"nd-1", "nd-2", "nd-3"}
    assert first["m_values"] == [5, 8]

    variants = [
        orchestrator.fixed_ownership(tasks, ("nd-2", "nd-3"), "1" * 40, "2" * 64, "screen"),
        orchestrator.fixed_ownership(tasks, ("nd-1", "nd-2", "nd-3"), "3" * 40, "2" * 64, "screen"),
        orchestrator.fixed_ownership(tasks, ("nd-1", "nd-2", "nd-3"), "1" * 40, "4" * 64, "screen"),
        orchestrator.fixed_ownership(
            [_task("a", 64, "m05_c00"), tasks[1]],
            ("nd-1", "nd-2", "nd-3"), "1" * 40, "2" * 64, "screen",
        ),
    ]
    assert all(item["stage_fingerprint"] != first["stage_fingerprint"] for item in variants)


def test_wrapper_rejects_marker_from_another_stage_identity(tmp_path):
    wrapper = Path(__file__).resolve().parents[1] / (
        "validation/005_pt_v2_discovery_20260720/run_discovery_wrapper.sh"
    )
    stage = tmp_path / "stage"
    log = tmp_path / "stage.log"
    stage.mkdir()
    (stage / "SUCCESS").write_text(
        json.dumps({"stage_fingerprint": "a" * 64}), encoding="ascii",
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake_flock = binary_dir / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    fake_flock.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
    result = subprocess.run(
        ("bash", str(wrapper), str(stage), str(log), "b" * 64, "true"),
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 74
    assert "identity conflict" in result.stderr


def test_node_choice_falls_back_before_ownership_when_nd1_is_busy(monkeypatch):
    loads = {"nd-1": (20.0, 10.0, 5.0), "nd-2": (0.1, 0.1, 0.1), "nd-3": (0.2, 0.2, 0.2)}
    monkeypatch.setattr(orchestrator, "probe_node_load", lambda node: loads[node])
    nodes, measured = orchestrator.choose_nodes("", 10.0)
    assert nodes == ("nd-2", "nd-3")
    assert measured == loads
