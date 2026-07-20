from importlib import import_module


ORCHESTRATOR = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.orchestrate_preflight"
)


def test_builds_verified_preflight_for_all_three_nodes(tmp_path):
    launches = [ORCHESTRATOR.build_node_launch(
        "exp102_test", "1" * 40, "2" * 64, "3" * 64, node, home=tmp_path,
    ) for node in ORCHESTRATOR.WORKERS]
    assert len(launches) == 3
    for launch in launches:
        command = launch.ssh_command[2]
        assert "preflight.py" in command
        assert "cross_node_smoke.py" in command
        assert "pytest" in command
        assert "run_verified_source.sh" in command
