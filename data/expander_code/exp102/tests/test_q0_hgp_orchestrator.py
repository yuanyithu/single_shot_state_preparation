import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time

import pytest


orchestrator_module = importlib.import_module(
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.orchestrate_hgp"
)
local_audit_module = importlib.import_module(
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.local_preflight_audit"
)

SOURCE_COMMIT = "1" * 40
ARCHIVE_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64
CONFIG_SHA256 = "c" * 64
LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation/013_q0_hgp_global_screen_20260722"
    / "launch_hgp_orchestrator.sh"
)


def _config():
    return {
        "execution": {
            "capacities": {"nd-2": 75, "nd-3": 91},
            "execution_nodes": ["nd-2", "nd-3"],
            "analysis": {"node": "nd-3", "capacity": 91, "num_workers": 91},
        },
        "resource_tiers": {"T1": {}, "T2": {}, "T3": {}},
        "task_counts": {
            "hp_measurement": 320,
            "map_measurement": 64,
            "total_measurement": 384,
        },
    }


def _orchestrator(tmp_path):
    return orchestrator_module.HgpOrchestrator(
        run_id="exp102_q0_hgp_test", source_commit=SOURCE_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        deployment_root=tmp_path / "repos/exp102_q0_hgp_test",
        run_root=tmp_path / "runs/exp102_q0_hgp_test",
        config=_config(), config_file_sha256=CONFIG_SHA256,
        poll_seconds=0.1,
    )


def _schedule():
    identity = {
        "schedule_version": "schedule.v1",
        "contract_version": "contract.v1",
        "run_id": "exp102_q0_hgp_test",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_identity": {
            "mode": "archive", "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "registry_file_sha256": "d" * 64,
        "config_file_sha256": CONFIG_SHA256,
        "started_unix": 1000,
        "preflight_deadline_unix": 2000,
        "control_freeze_deadline_unix": 3000,
        "screen_deadline_unix": 4000,
        "analysis_deadline_unix": 5000,
    }
    return {
        **identity,
        "schedule_sha256": hashlib.sha256(
            orchestrator_module._canonical_json(identity).encode("ascii")
        ).hexdigest(),
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        orchestrator_module._canonical_json(value) + "\n", encoding="ascii",
    )


def _write_success(path, stage):
    _write_json(path, {
        "stage": stage,
        "source_commit": SOURCE_COMMIT,
        "stage_fingerprint": "f" * 64,
        "prerequisite_success_sha256": [],
        "completed_utc": "2026-07-22T00:00:00Z",
    })


def _launch_metadata(args, command_sha="9" * 64):
    return {
        "archive_sha256": args.archive_sha256,
        "command_sha256": command_sha,
        "launcher_version": orchestrator_module.ND0_LAUNCHER_VERSION,
        "local_attestation_sha256": (
            args.local_attestation_sha256
            if args.phase == "measurement" else None
        ),
        "manifest_sha256": args.source_manifest_sha256,
        "phase": args.phase,
        "run_id": args.run_id,
        "source_commit": args.source_commit,
    }


def test_stage_graph_is_verified_screen_only_and_preserves_frozen_placement(
    tmp_path,
):
    runner = _orchestrator(tmp_path)
    schedule = runner.schedule_stage()
    artifact = runner.artifact_stage(schedule.success)
    preflight_nodes = runner.preflight_node_stages(artifact.success)
    combine = runner.preflight_combine_stage(
        tuple(stage.success for stage in preflight_nodes),
    )
    control = runner.control_stage(combine.success)
    screens = runner.screen_stages(control.success)
    analysis = runner.analysis_stage(tuple(stage.success for stage in screens))

    assert schedule.node == artifact.node == combine.node == control.node == "nd-1"
    assert [stage.node for stage in preflight_nodes] == ["nd-1", "nd-2", "nd-3"]
    assert [stage.node for stage in screens] == ["nd-2", "nd-3"]
    assert analysis.node == "nd-3"
    assert artifact.prerequisites == (schedule.success,)
    assert combine.prerequisites == tuple(stage.success for stage in preflight_nodes)
    assert control.prerequisites == (combine.success,)
    assert all(stage.prerequisites == (control.success,) for stage in screens)
    assert analysis.prerequisites == tuple(stage.success for stage in screens)

    nd2_workers = screens[0].workflow_argv.index("--num-workers") + 1
    nd3_workers = screens[1].workflow_argv.index("--num-workers") + 1
    analysis_workers = analysis.workflow_argv.index("--num-workers") + 1
    assert screens[0].workflow_argv[nd2_workers] == 75
    assert screens[1].workflow_argv[nd3_workers] == 91
    assert analysis.workflow_argv[analysis_workers] == 91

    for stage in (
            schedule, artifact, *preflight_nodes, combine, control, *screens,
            analysis):
        shell = orchestrator_module._verified_stage_shell(
            runner.deployment_root, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, stage.stage, stage.stage_dir,
            stage.log_file, stage.prerequisites, stage.workflow_argv,
        )
        assert "tar -xOf" in shell
        assert orchestrator_module.VERIFY_RELATIVE in shell
        assert orchestrator_module.WRAPPER_RELATIVE in shell
        assert f"python -m {orchestrator_module.WORKFLOW_MODULE}" in shell
        assert "/usr/bin/true" not in shell
        assert "ecedc1fb3e8e6fbe9680f5047eb3dbee7" not in shell


def test_stage_launcher_uses_remote_screen_and_verified_bootstrap(
    tmp_path, monkeypatch,
):
    runner = _orchestrator(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrator_module.subprocess, "run", fake_run)
    runner._launch(runner.schedule_stage())
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ("ssh", "-o", "BatchMode=yes")
    assert command[-2] == "nd-1"
    remote = command[-1]
    assert "screen -dmS" in remote
    assert "tar -xOf" in remote
    assert orchestrator_module.VERIFY_RELATIVE in remote
    assert orchestrator_module.WRAPPER_RELATIVE in remote
    assert " build-schedule " in remote
    assert kwargs == {"check": True}


def test_nd0_outer_launcher_uses_exact_nohup_setsid_shape():
    source = LAUNCHER_PATH.read_text(encoding="ascii")
    normalized = " ".join(source.split())
    guard = 'mkdir -m 700 -- "$launch_guard"'
    detached = (
        "EXP102_HGP_ORCHESTRATOR_PERSISTENCE=$persistence_token \\\n"
        "EXP102_HGP_ORCHESTRATOR_GUARD=$launch_guard \\\n"
        '  /usr/bin/nohup /usr/bin/setsid /bin/bash -lc "$inner" \\\n'
        '  </dev/null >>"$log" 2>&1 &'
    )

    assert guard in source
    assert detached in source
    assert source.index(guard) < source.index("/usr/bin/nohup")
    assert "screen -dmS" not in source
    assert (
        "conda run -n 11 --no-capture-output /usr/bin/setsid python -m"
        in normalized
    )
    assert (
        'expected_attestation=$run_root/control/'
        'HGP_LOCAL_PREFLIGHT_ATTESTATION.json' in source
    )
    assert '[[ $local_attestation == "$expected_attestation" ]]' in source
    assert '[[ $phase == preflight && $# -ne 5 ]]' in source
    assert '[[ $# -eq 7 && $7 =~ ^[0-9a-f]{64}$ ]]' in source


def test_nd0_persistence_guard_binds_phase_and_publishes_pid(
    tmp_path, monkeypatch,
):
    base = tmp_path / ".single_shot"
    (base / "logs").mkdir(parents=True)
    args = SimpleNamespace(
        run_id="exp102_q0_hgp_test",
        phase="measurement",
        source_commit=SOURCE_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        local_attestation_sha256="e" * 64,
    )
    token = hashlib.sha256(args.run_id.encode("ascii")).hexdigest()[:8]
    guard = (
        base / "logs"
        / f".{args.run_id}_hgp_orchestrator_{token}_{args.phase}.launch"
    )
    guard.mkdir()
    _write_json(guard / "LAUNCH.json", _launch_metadata(args))
    monkeypatch.setenv(
        "EXP102_HGP_ORCHESTRATOR_PERSISTENCE",
        orchestrator_module.ND0_PERSISTENCE_TOKEN,
    )
    monkeypatch.setenv("EXP102_HGP_ORCHESTRATOR_GUARD", str(guard))
    monkeypatch.setattr(orchestrator_module.os, "getpid", lambda: 4242)
    monkeypatch.setattr(orchestrator_module.os, "getsid", lambda _pid: 4242)

    assert orchestrator_module._validate_nd0_persistence(args, base) == guard
    assert (guard / "ORCHESTRATOR_PID").read_text(encoding="ascii") == "4242\n"
    with pytest.raises(ValueError, match="PID metadata already exists"):
        orchestrator_module._validate_nd0_persistence(args, base)

    (guard / "ORCHESTRATOR_PID").unlink()
    changed = _launch_metadata(args)
    changed["local_attestation_sha256"] = None
    _write_json(guard / "LAUNCH.json", changed)
    with pytest.raises(ValueError, match="metadata identity"):
        orchestrator_module._validate_nd0_persistence(args, base)


def test_nd0_publishes_orchestrator_pid_before_archive_hashing(
    tmp_path, monkeypatch,
):
    home = tmp_path / "home"
    base = home / ".single_shot"
    run_id = "exp102_q0_hgp_test"
    deployment = base / "repos" / run_id
    (deployment / "source").mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    (deployment / "SOURCE.tar").write_bytes(b"archive\n")
    (deployment / "SOURCE_MANIFEST.json").write_bytes(b"manifest\n")
    (deployment / "SOURCE_COMMIT").write_text(
        SOURCE_COMMIT + "\n", encoding="ascii",
    )
    args = SimpleNamespace(
        run_id=run_id,
        phase="preflight",
        source_commit=SOURCE_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
    )
    events = []

    def record_persistence(_args, _base):
        events.append("pid")

    def record_hash(path):
        events.append(Path(path).name)
        return {
            "SOURCE.tar": ARCHIVE_SHA256,
            "SOURCE_MANIFEST.json": SOURCE_MANIFEST_SHA256,
        }[Path(path).name]

    monkeypatch.setenv("EXP102_SOURCE_COMMIT", SOURCE_COMMIT)
    monkeypatch.setattr(orchestrator_module.platform, "node", lambda: "nd-0")
    monkeypatch.setattr(
        orchestrator_module, "_validate_nd0_persistence", record_persistence,
    )
    monkeypatch.setattr(orchestrator_module, "_sha256_file", record_hash)

    roots = orchestrator_module._require_verified_launch(args, home)

    assert roots == (deployment, base / "runs" / run_id)
    assert events == ["pid", "SOURCE.tar", "SOURCE_MANIFEST.json"]


def test_nd0_launcher_atomic_guard_and_argument_rejection(tmp_path):
    home = tmp_path / "home"
    server_root = home / ".single_shot"
    run_id = "exp102_q0_hgp_test"
    deployment = server_root / "repos" / run_id
    (deployment / "source").mkdir(parents=True)
    (server_root / "logs").mkdir(parents=True)
    archive = deployment / "SOURCE.tar"
    manifest = deployment / "SOURCE_MANIFEST.json"
    archive.write_bytes(b"archive-test\n")
    manifest.write_bytes(b"manifest-test\n")
    (deployment / "SOURCE_COMMIT").write_text(
        SOURCE_COMMIT + "\n", encoding="ascii",
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    env = {**dict(os.environ), "HOME": str(home), "HOSTNAME": "nd-0"}

    token = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:8]
    guard = (
        server_root / "logs"
        / f".{run_id}_hgp_orchestrator_{token}_preflight.launch"
    )
    guard.mkdir()
    command = [
        "bash", str(LAUNCHER_PATH), run_id, SOURCE_COMMIT, archive_sha,
        manifest_sha, "preflight",
    ]
    completed = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 69
    assert "launch guard already exists" in completed.stderr
    assert not (
        server_root / "logs" / f"{run_id}_hgp_orchestrator_preflight.log"
    ).exists()

    missing_measurement_args = subprocess.run(
        [*command[:-1], "measurement"], env=env, text=True,
        capture_output=True, check=False,
    )
    assert missing_measurement_args.returncode == 65
    extra_preflight_args = subprocess.run(
        [*command, "unexpected", "0" * 64], env=env, text=True,
        capture_output=True, check=False,
    )
    assert extra_preflight_args.returncode == 65

    injected = tmp_path / "injected"
    malicious_run_id = f"bad;touch {injected}"
    injection_attempt = subprocess.run(
        [
            "bash", str(LAUNCHER_PATH), malicious_run_id, SOURCE_COMMIT,
            archive_sha, manifest_sha, "preflight",
        ],
        env=env, text=True, capture_output=True, check=False,
    )
    assert injection_attempt.returncode == 65
    assert not injected.exists()

    run_root = server_root / "runs" / run_id
    run_root.mkdir(parents=True)
    wrong_attestation = tmp_path / "attestation.json"
    wrong_attestation.write_text("{}\n", encoding="ascii")
    wrong_sha = hashlib.sha256(wrong_attestation.read_bytes()).hexdigest()
    path_drift = subprocess.run(
        [
            "bash", str(LAUNCHER_PATH), run_id, SOURCE_COMMIT, archive_sha,
            manifest_sha, "measurement", str(wrong_attestation), wrong_sha,
        ],
        env=env, text=True, capture_output=True, check=False,
    )
    assert path_drift.returncode == 65
    assert "attestation path is not canonical" in path_drift.stderr


def test_preflight_phase_stops_for_local_audit_without_attestation(
    tmp_path, monkeypatch,
):
    runner = _orchestrator(tmp_path)
    schedule = _schedule()
    _write_json(runner.schedule, schedule)
    _write_json(runner.artifact_manifest, {
        "artifact_manifest_sha256": "f" * 64,
    })
    _write_json(runner.preflight, {
        "status": "PASS",
        "selected_resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "config_file_sha256": CONFIG_SHA256,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": {
            "mode": "archive", "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "completed_unix": 1500,
    })
    batches = []

    def fake_run_batch(stages, deadline_unix):
        batches.append((tuple(stage.key for stage in stages), deadline_unix))
        return tuple(stage.success for stage in stages)

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)
    result = runner.run_preflight()

    assert result["event"] == "preflight_ready_for_local_audit"
    assert result["selected_resource_tier"] == "T3"
    assert [keys for keys, _ in batches] == [
        ("00_schedule",),
        ("01_artifacts",),
        ("02_preflight_nd-1", "02_preflight_nd-2", "02_preflight_nd-3"),
        ("03_preflight_combine",),
    ]


def test_measurement_rejects_missing_or_tampered_attestation_before_control(
    tmp_path, monkeypatch,
):
    runner = _orchestrator(tmp_path)
    schedule = _schedule()
    _write_json(runner.schedule, schedule)
    _write_json(runner.artifact_manifest, {
        "artifact_manifest_sha256": "f" * 64,
    })
    _write_json(runner.preflight, {
        "status": "PASS",
        "selected_resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "config_file_sha256": CONFIG_SHA256,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": {
            "mode": "archive", "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "completed_unix": 1500,
    })
    schedule_stage = runner.schedule_stage()
    artifact_stage = runner.artifact_stage(schedule_stage.success)
    preflight_stages = runner.preflight_node_stages(artifact_stage.success)
    combine_stage = runner.preflight_combine_stage(
        tuple(stage.success for stage in preflight_stages),
    )
    _write_success(schedule_stage.success, "build-schedule")
    _write_success(artifact_stage.success, "build-artifacts")
    for stage in preflight_stages:
        _write_success(stage.success, "preflight")
    _write_success(combine_stage.success, "preflight")

    control_calls = []

    def forbidden_control(_preflight_success):
        control_calls.append(True)
        raise AssertionError("control stage must not be constructed")

    monkeypatch.setattr(runner, "control_stage", forbidden_control)
    with pytest.raises(ValueError, match="requires a local attestation"):
        runner.run_measurement()
    assert control_calls == []

    attestation = runner.control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
    _write_json(attestation, {"tampered": True})
    registry = tmp_path / "registry.json"
    _write_json(registry, {"registry": "test"})
    runner.registry = registry
    runner.local_attestation = attestation
    runner.local_attestation_sha256 = "0" * 64
    with pytest.raises(ValueError, match="file SHA mismatch"):
        runner.run_measurement()
    assert control_calls == []


def test_aggregate_preflight_must_be_pass_selected_and_archive_bound(tmp_path):
    schedule = _schedule()
    report = {
        "status": "PASS",
        "selected_resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "config_file_sha256": CONFIG_SHA256,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": {
            "mode": "archive", "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "completed_unix": 1500,
    }
    path = tmp_path / "preflight.json"
    _write_json(path, report)
    assert orchestrator_module._validate_aggregate_preflight(
        path, schedule, _config(), SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
    ) == report

    mutations = [
        ("status", "RUNTIME_EXHAUSTED"),
        ("selected_resource_tier", None),
        ("selected_resource_tier", "T4"),
        ("completed_unix", 2500),
        ("source_commit", "2" * 40),
        ("config_file_sha256", "e" * 64),
    ]
    for name, value in mutations:
        changed = copy.deepcopy(report)
        changed[name] = value
        _write_json(path, changed)
        with pytest.raises(RuntimeError, match="not a PASS authority"):
            orchestrator_module._validate_aggregate_preflight(
                path, schedule, _config(), SOURCE_COMMIT, ARCHIVE_SHA256,
                SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
            )

    changed = copy.deepcopy(report)
    changed["source_identity"]["mode"] = "git"
    _write_json(path, changed)
    with pytest.raises(RuntimeError, match="not a PASS authority"):
        orchestrator_module._validate_aggregate_preflight(
            path, schedule, _config(), SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
        )


def test_schedule_and_control_are_fail_closed(tmp_path):
    schedule = _schedule()
    schedule_path = tmp_path / "schedule.json"
    _write_json(schedule_path, schedule)
    assert orchestrator_module._validate_schedule_output(
        schedule_path, "exp102_q0_hgp_test", SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
    ) == schedule

    future_identity = dict(schedule)
    future_identity.pop("schedule_sha256")
    offset = time.time() + 3600
    future_identity.update({
        "started_unix": offset,
        "preflight_deadline_unix": offset + 1000,
        "control_freeze_deadline_unix": offset + 2000,
        "screen_deadline_unix": offset + 3000,
        "analysis_deadline_unix": offset + 4000,
    })
    future = {
        **future_identity,
        "schedule_sha256": hashlib.sha256(
            orchestrator_module._canonical_json(future_identity).encode("ascii")
        ).hexdigest(),
    }
    _write_json(schedule_path, future)
    with pytest.raises(ValueError, match="schedule output"):
        orchestrator_module._validate_schedule_output(
            schedule_path, "exp102_q0_hgp_test", SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
        )

    preflight = {"selected_resource_tier": "T3"}
    control = {
        "resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "task_count": 384,
        "tasks": [{} for _ in range(384)],
        "execution_nodes": ["nd-2", "nd-3"],
    }
    control_path = tmp_path / "control.json"
    _write_json(control_path, control)
    assert orchestrator_module._validate_control_output(
        control_path, preflight, schedule, _config(), SOURCE_COMMIT,
        ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
    ) == control
    control["resource_tier"] = "T2"
    _write_json(control_path, control)
    with pytest.raises(RuntimeError, match="measurement control"):
        orchestrator_module._validate_control_output(
            control_path, preflight, schedule, _config(), SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
        )


def test_success_marker_requires_exact_wrapper_schema(tmp_path):
    marker = {
        "stage": "preflight", "source_commit": SOURCE_COMMIT,
        "stage_fingerprint": "f" * 64,
        "prerequisite_success_sha256": ["e" * 64],
        "completed_utc": "2026-07-22T00:00:00Z",
    }
    path = tmp_path / "SUCCESS"
    _write_json(path, marker)
    assert orchestrator_module._validate_success_marker(
        path, expected_stage="preflight", source_commit=SOURCE_COMMIT,
    ) == marker
    marker["extra"] = True
    _write_json(path, marker)
    with pytest.raises(ValueError, match="SUCCESS marker"):
        orchestrator_module._validate_success_marker(
            path, expected_stage="preflight", source_commit=SOURCE_COMMIT,
        )


def test_measurement_attestation_is_hash_bound_and_exact(tmp_path):
    control_root = tmp_path / "control"
    schedule = _schedule()
    schedule_path = control_root / "HGP_GLOBAL_24H_SCHEDULE.json"
    artifact_path = control_root / "hgp_artifacts.json"
    preflight_path = control_root / "hgp_preflight.json"
    _write_json(schedule_path, schedule)
    _write_json(artifact_path, {
        "artifact": "bound", "artifact_manifest_sha256": "f" * 64,
    })
    remote_digest = "d" * 64
    expected_is = [{
        "cell_fingerprint": "1" * 64,
        "transcript_sha256": "2" * 64,
    }]
    preflight = {
        "canonical_digest_sha256": remote_digest,
        "canonical_digest": {
            "importance_sampling_transcript_sha256": expected_is,
        },
    }
    _write_json(preflight_path, preflight)
    identity = {
        "attestation_version": orchestrator_module.LOCAL_ATTESTATION_VERSION,
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "registry_file_sha256": "e" * 64,
        "config_file_sha256": CONFIG_SHA256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": orchestrator_module._sha256_file(schedule_path),
        "artifact_manifest_sha256": "f" * 64,
        "artifact_manifest_file_sha256": orchestrator_module._sha256_file(
            artifact_path,
        ),
        "preflight_file_sha256": orchestrator_module._sha256_file(preflight_path),
        "remote_canonical_digest_sha256": remote_digest,
        "local_canonical_digest_sha256": remote_digest,
        "exact_canonical_match": True,
        "mismatch_paths": [],
        "importance_sampling_transcript_sha256": expected_is,
        "solver_identity_policy": orchestrator_module.LOCAL_SOLVER_POLICY,
        "local_environment": {
            "system": "Darwin", "machine": "arm64", "python": "3.12",
            "numpy": "2.4.1", "scipy": "1.17.0",
            "map_solver_identity_current": "local-test",
        },
        "portability_review": None,
        "completed_unix": 1500,
    }
    attestation = {
        **identity,
        "attestation_sha256": hashlib.sha256(
            orchestrator_module._canonical_json(identity).encode("ascii")
        ).hexdigest(),
    }
    path = control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
    _write_json(path, attestation)
    file_sha = orchestrator_module._sha256_file(path)
    assert orchestrator_module._validate_local_attestation(
        path, file_sha, schedule, preflight, artifact_path,
        "e" * 64, CONFIG_SHA256, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256,
    ) == attestation

    with pytest.raises(ValueError, match="file SHA mismatch"):
        orchestrator_module._validate_local_attestation(
            path, "9" * 64, schedule, preflight, artifact_path,
            "e" * 64, CONFIG_SHA256, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256,
        )
    changed = copy.deepcopy(attestation)
    changed["exact_canonical_match"] = False
    changed_identity = dict(changed)
    changed_identity.pop("attestation_sha256")
    changed["attestation_sha256"] = hashlib.sha256(
        orchestrator_module._canonical_json(changed_identity).encode("ascii")
    ).hexdigest()
    _write_json(path, changed)
    with pytest.raises(ValueError, match="exact local attestation"):
        orchestrator_module._validate_local_attestation(
            path, orchestrator_module._sha256_file(path), schedule, preflight,
            artifact_path, "e" * 64, CONFIG_SHA256, SOURCE_COMMIT,
            ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256,
        )

    # A hand-written PORTABLE_PASS remains forbidden even when its self-hash
    # and externally supplied file SHA are recomputed.
    forged = copy.deepcopy(attestation)
    forged["status"] = "PORTABLE_PASS"
    forged["exact_canonical_match"] = False
    forged["mismatch_paths"] = ["$.cells[0].mass"]
    forged["portability_review"] = {
        "review_version": "exp102.q0_hgp_global.screen.portability_review.v1",
        "discrete_transcripts_exact": True,
        "review_contract_sha256": "3" * 64,
        "comparator_source_sha256": "4" * 64,
        "approved_float_differences": [{
            "field": "$.cells[0].mass", "observed_ulp": 1, "max_ulp": 999,
        }],
    }
    forged_identity = dict(forged)
    forged_identity.pop("attestation_sha256")
    forged["attestation_sha256"] = hashlib.sha256(
        orchestrator_module._canonical_json(forged_identity).encode("ascii")
    ).hexdigest()
    _write_json(path, forged)
    with pytest.raises(ValueError, match="identity is invalid"):
        orchestrator_module._validate_local_attestation(
            path, orchestrator_module._sha256_file(path), schedule, preflight,
            artifact_path, "e" * 64, CONFIG_SHA256, SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
        )

    forged = copy.deepcopy(attestation)
    forged["registry_file_sha256"] = "9" * 64
    forged["importance_sampling_transcript_sha256"] = []
    forged_identity = dict(forged)
    forged_identity.pop("attestation_sha256")
    forged["attestation_sha256"] = hashlib.sha256(
        orchestrator_module._canonical_json(forged_identity).encode("ascii")
    ).hexdigest()
    _write_json(path, forged)
    with pytest.raises(ValueError, match="identity is invalid"):
        orchestrator_module._validate_local_attestation(
            path, orchestrator_module._sha256_file(path), schedule, preflight,
            artifact_path, "e" * 64, CONFIG_SHA256, SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
        )


def test_local_audit_reports_exact_mismatch_paths_without_float_tolerance():
    remote = {
        "state": [0, 1],
        "derived": {"energy": 1.0},
        "extra": "remote",
    }
    local = {
        "state": [0, 0],
        "derived": {"energy": 1.0000000000000002},
    }
    assert local_audit_module._mismatch_paths(remote, local) == [
        "$.derived.energy", "$.extra:missing", "$.state[1]",
    ]
    assert not hasattr(local_audit_module, "ULP_TOLERANCE")
