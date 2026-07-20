from importlib import import_module
import json

import pytest


ORCHESTRATOR = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.orchestrate_production"
)


def test_builds_three_verified_production_launches(tmp_path):
    launches = [ORCHESTRATOR.build_node_launch(
        "exp102_test", "1" * 40, "2" * 64, "3" * 64, node, home=tmp_path,
    ) for node in ORCHESTRATOR.WORKERS]
    assert [launch.workers for launch in launches] == [75, 75, 91]
    assert len({launch.stage_dir for launch in launches}) == 3
    for launch in launches:
        command = launch.ssh_command[2]
        assert "run_verified_source.sh" in command
        assert "run_production_stage.py" in command
        assert "--pilot-report" in command
        assert "conda run -n 11 --no-capture-output" in command


def test_reconciles_exactly_6144_tasks_with_common_identity(tmp_path):
    run_root = tmp_path / "run"
    counts = {"nd-1": 1792, "nd-2": 1792, "nd-3": 2560}
    common = {
        "registry_sha256": "r", "config_sha256": "c", "frozen_config_sha256": "f",
        "source_commit": "1" * 40, "task_plan_sha256": "t",
        "deployment_manifest_sha256": "d", "pilot_report_sha256_file": "p",
    }
    for node, expected in counts.items():
        path = run_root / "status" / f"production_{node}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "SUCCESS", "node": node, "expected": expected,
            "computed": expected, "reused": 0, **common,
        }), encoding="ascii")
    assert set(ORCHESTRATOR.reconcile_statuses(run_root)) == set(counts)

    path = run_root / "status/production_nd-3.json"
    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["source_commit"] = "2" * 40
    path.write_text(json.dumps(tampered), encoding="ascii")
    with pytest.raises(ValueError, match="differ"):
        ORCHESTRATOR.reconcile_statuses(run_root)
