from importlib import import_module
import json

import pytest


ORCHESTRATOR = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.orchestrate_stage"
)


def _build(tmp_path, node):
    config = tmp_path / "by_m.json"
    config.write_text(json.dumps({"3": {"candidate": "test"}}), encoding="ascii")
    return ORCHESTRATOR.build_node_launch(
        "exp102_pilot_test", "1" * 40, "2" * 64, "3" * 64,
        "gamma", 7, config.resolve(), (3,), node, home=tmp_path,
    )


def test_builds_verified_independent_dual_node_commands(tmp_path):
    nd2 = _build(tmp_path, "nd-2")
    nd3 = _build(tmp_path, "nd-3")

    assert (nd2.workers, nd3.workers) == (75, 91)
    assert nd2.stage_dir != nd3.stage_dir
    assert nd2.log_file != nd3.log_file
    assert nd2.screen_name != nd3.screen_name
    for launch, workers in ((nd2, 75), (nd3, 91)):
        command = ORCHESTRATOR.remote_command(launch.stage_command)
        assert "run_ladder_stage.py" in command
        assert "conda run -n 11 --no-capture-output" in command
        assert f"--num-workers {workers}" in command
        assert "--stage gamma --attempt 7" in command
        assert "--by-m-config" in command
        assert "--by-m-config-sha256" in command
        assert "--m-values 3" in command
        assert "run_stage_wrapper.sh" in launch.shell
        assert "run_verified_source.sh" in launch.shell
        assert "2" * 64 in launch.shell
        assert "3" * 64 in launch.shell
        assert launch.ssh_command[:2] == ("ssh", launch.node)
        assert "screen -dmS" in launch.ssh_command[2]


@pytest.mark.parametrize("marker", ORCHESTRATOR.MARKERS)
def test_rejects_any_existing_stage_marker(tmp_path, marker):
    launch = _build(tmp_path, "nd-2")
    launch.stage_dir.mkdir(parents=True)
    (launch.stage_dir / marker).write_text("{}\n", encoding="ascii")

    with pytest.raises(FileExistsError, match=marker):
        ORCHESTRATOR.check_marker_conflicts((launch,))


def test_validates_by_m_config_keys_and_canonicalizes_m_values(tmp_path):
    config = tmp_path / "by_m.json"
    config.write_text(json.dumps({"3": {}, "5": {}}), encoding="ascii")

    values = ORCHESTRATOR.parse_m_values("5,3")
    assert values == (3, 5)
    assert ORCHESTRATOR.validate_by_m_config(config, values) == config.resolve()

    with pytest.raises(ValueError, match="exactly match"):
        ORCHESTRATOR.validate_by_m_config(config, (3, 4, 5))


def test_snapshots_canonical_by_m_config_into_shared_run_control(tmp_path):
    config = tmp_path / "input.json"
    config.write_text('{"5": {"b": 2}, "3": {"a": 1}}\n', encoding="ascii")
    target, digest = ORCHESTRATOR.snapshot_by_m_config(
        config, (3, 5), tmp_path / "run", "gamma", 4,
    )
    assert target == tmp_path / "run/control/gamma_attempt_004.json"
    assert target.read_text(encoding="ascii") == '{"3":{"a":1},"5":{"b":2}}\n'
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="already exists"):
        ORCHESTRATOR.snapshot_by_m_config(
            config, (3, 5), tmp_path / "run", "gamma", 4,
        )


def test_wait_requires_both_success_markers_and_fails_closed(tmp_path):
    nd2 = _build(tmp_path, "nd-2")
    nd3 = _build(tmp_path, "nd-3")
    for launch in (nd2, nd3):
        launch.stage_dir.mkdir(parents=True)
        (launch.stage_dir / "SUCCESS").write_text("{}\n", encoding="ascii")

    assert ORCHESTRATOR.wait_for_terminal_markers((nd2, nd3), 1.0) == {
        "nd-2": "SUCCESS", "nd-3": "SUCCESS",
    }

    (nd3.stage_dir / "SUCCESS").unlink()
    (nd3.stage_dir / "FAILED").write_text("{}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="nd-3"):
        ORCHESTRATOR.wait_for_terminal_markers((nd2, nd3), 1.0)
