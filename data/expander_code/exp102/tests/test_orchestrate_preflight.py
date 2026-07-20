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


def test_attestation_requires_one_cross_node_digest(tmp_path):
    digest = "a" * 64
    probes = {}
    launches = []
    for node in ORCHESTRATOR.WORKERS:
        log = tmp_path / f"{node}.log"
        log.write_text(f"preflight output\n{digest}\n", encoding="ascii")
        launches.append(ORCHESTRATOR.PreflightLaunch(
            node=node, stage_dir=tmp_path / node, log_file=log,
            screen_name=f"screen_{node}", ssh_command=("ssh", node, "true"),
        ))
        probes[node] = {"idle": True}
    attestation = ORCHESTRATOR.write_attestation(
        tmp_path / "pass", "1" * 40, "2" * 64, "3" * 64, launches, probes,
    )
    assert attestation["smoke_digest"] == digest
    assert {row["smoke_digest"] for row in attestation["nodes"].values()} == {digest}

    launches[-1].log_file.write_text("preflight output\n" + "b" * 64 + "\n", encoding="ascii")
    import pytest
    with pytest.raises(ValueError, match="digests differ"):
        ORCHESTRATOR.write_attestation(
            tmp_path / "fail", "1" * 40, "2" * 64, "3" * 64, launches, probes,
        )
