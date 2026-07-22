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
BOOT_ID = "01234567-89ab-cdef-0123-456789abcdef"
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
        "importance_sampling": {"num_samples_per_cell": 50000},
    }


def _clock_authority():
    return {
        "clock_authority_version": orchestrator_module.CLOCK_AUTHORITY_VERSION,
        "clock_authority_node": "nd-0",
        "clock_authority_boot_id": BOOT_ID,
        "boottime_before_ns": 5_000_000_000_000,
        "authority_unix_ns": 1_000_000_000_000,
        "boottime_after_ns": 5_000_000_001_000,
    }


def _orchestrator(tmp_path):
    return orchestrator_module.HgpOrchestrator(
        run_id="exp102_q0_hgp_test", source_commit=SOURCE_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        deployment_root=tmp_path / "repos/exp102_q0_hgp_test",
        run_root=tmp_path / "runs/exp102_q0_hgp_test",
        config=_config(), config_file_sha256=CONFIG_SHA256,
        clock_authority=_clock_authority(),
        poll_seconds=0.1,
    )


def _schedule(*, registry_sha="d" * 64, config_sha=CONFIG_SHA256):
    clock_authority = _clock_authority()
    started_unix = 1000
    started_boottime_ns = clock_authority["boottime_before_ns"]
    identity = {
        "schedule_version": orchestrator_module.SCHEDULE_VERSION,
        "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
        "run_id": "exp102_q0_hgp_test",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_identity": {
            "mode": "archive", "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "registry_file_sha256": registry_sha,
        "config_file_sha256": config_sha,
        "clock_authority": clock_authority,
        "started_unix": started_unix,
        "started_boottime_ns": started_boottime_ns,
    }
    for name, hours in (
            ("preflight", 6), ("control_freeze", 8),
            ("screen", 22), ("analysis", 24)):
        identity[f"{name}_deadline_unix"] = started_unix + hours * 3600
        identity[f"{name}_deadline_boottime_ns"] = (
            started_boottime_ns
            + hours * 3600 * orchestrator_module.NANOSECONDS_PER_SECOND
        )
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


def _write_stage_success(stage):
    fingerprint, prerequisite_sha256 = (
        orchestrator_module._expected_stage_fingerprint(stage, SOURCE_COMMIT)
    )
    _write_json(stage.success, {
        "stage": stage.stage,
        "source_commit": SOURCE_COMMIT,
        "stage_fingerprint": fingerprint,
        "prerequisite_success_sha256": prerequisite_sha256,
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


def _add_self_hash(identity, field):
    return {
        **identity,
        field: hashlib.sha256(
            orchestrator_module._canonical_json(identity).encode("ascii")
        ).hexdigest(),
    }


def _canonical_evidence(*, full_marker="6", is_full_marker="7"):
    decision_catalog = [{
        "cell_fingerprint": "5" * 64,
        "method_id": "MAM-IMH8",
        "init_family": "P",
        "acceptance_decision_sha256": "4" * 64,
    }]
    decision_sha = hashlib.sha256(
        orchestrator_module._canonical_json(decision_catalog).encode("ascii")
    ).hexdigest()
    full_is = [{
        "cell_fingerprint": "5" * 64,
        "full_transcript_sha256": is_full_marker * 64,
        "portable_transcript_sha256": "8" * 64,
        "nonportable_float_sha256": full_marker * 64,
        "field_manifest_sha256": "9" * 64,
    }]
    portable_is = [{
        "cell_fingerprint": "5" * 64,
        "portable_transcript_sha256": "8" * 64,
        "field_manifest_sha256": "9" * 64,
    }]
    full_payload = {
        "schema_version": "full.v1",
        "platform_float_marker": full_marker,
        "acceptance_decision_catalog": decision_catalog,
        "acceptance_decision_catalog_sha256": decision_sha,
        "importance_sampling_transcript_summary": full_is,
    }
    portable_payload = {
        "schema_version": "portable.v1",
        "discrete_marker": "exact",
        "acceptance_decision_catalog": decision_catalog,
        "acceptance_decision_catalog_sha256": decision_sha,
        "importance_sampling_transcript_summary": portable_is,
    }
    return {
        "canonical_full_payload": full_payload,
        "canonical_full_payload_sha256": hashlib.sha256(
            orchestrator_module._canonical_json(full_payload).encode("ascii")
        ).hexdigest(),
        "canonical_portable_payload": portable_payload,
        "canonical_portable_payload_sha256": hashlib.sha256(
            orchestrator_module._canonical_json(portable_payload).encode(
                "ascii"
            )
        ).hexdigest(),
    }


def _base_canonical_evidence(*, full_marker="6"):
    evidence = _canonical_evidence(full_marker=full_marker)
    full_payload = dict(evidence["canonical_full_payload"])
    portable_payload = dict(evidence["canonical_portable_payload"])
    full_payload.pop("importance_sampling_transcript_summary")
    portable_payload.pop("importance_sampling_transcript_summary")
    return {
        "canonical_full_payload": full_payload,
        "canonical_full_payload_sha256": orchestrator_module._sha256_json(
            full_payload,
        ),
        "canonical_portable_payload": portable_payload,
        "canonical_portable_payload_sha256": (
            orchestrator_module._sha256_json(portable_payload)
        ),
    }


def _write_phase_launch(runner, phase, attestation_file_sha=None):
    metadata = {
        "archive_sha256": ARCHIVE_SHA256,
        "command_sha256": "9" * 64,
        "launcher_version": orchestrator_module.ND0_LAUNCHER_VERSION,
        "local_attestation_sha256": (
            attestation_file_sha if phase == "measurement" else None
        ),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "phase": phase,
        "run_id": runner.run_id,
        "source_commit": SOURCE_COMMIT,
    }
    path = (
        runner.log_root
        / f".{runner.run_id}_hgp_orchestrator_{runner.token}_{phase}.launch"
        / "LAUNCH.json"
    )
    _write_json(path, metadata)
    return path


def _write_phase_manifest(
        runner, path, phase, stages, bound_files, bound_identity,
        published_boottime_ns, prior_manifest=None):
    identity = runner._phase_acceptance_identity(
        phase, stages, bound_files, bound_identity,
        published_boottime_ns, prior_manifest,
    )
    value = _add_self_hash(identity, "manifest_sha256")
    _write_json(path, value)
    return value


def _terminal_evidence(tmp_path):
    source_root = tmp_path / "source"
    registry_path = source_root / "registry.json"
    config_path = source_root / "config.json"
    config = _config()
    _write_json(registry_path, {"registry": "test"})
    _write_json(config_path, config)
    registry_sha = orchestrator_module._sha256_file(registry_path)
    config_sha = orchestrator_module._sha256_file(config_path)
    log_root = tmp_path / "logs"
    runner = orchestrator_module.HgpOrchestrator(
        run_id="exp102_q0_hgp_test", source_commit=SOURCE_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        deployment_root=tmp_path / "repos/exp102_q0_hgp_test",
        run_root=tmp_path / "runs/exp102_q0_hgp_test",
        config=config, config_file_sha256=config_sha,
        clock_authority=_clock_authority(),
        launch_guard=log_root / "placeholder", poll_seconds=0.1,
    )
    runner.registry = registry_path
    runner.config_path = config_path
    schedule = _schedule(registry_sha=registry_sha, config_sha=config_sha)
    _write_json(runner.schedule, schedule)

    artifact = {"artifact_manifest_sha256": "4" * 64}
    _write_json(runner.artifact_manifest, artifact)
    evidence = _canonical_evidence()
    preflight = {
        "status": "PASS",
        "remote_full_consensus": True,
        "nodes": ["nd-1", "nd-2", "nd-3"],
        "selected_resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "config_file_sha256": config_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "selection_basis": (
            "runtime_only_worst_node_and_frozen_elapsed_deadlines"
        ),
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1500,
        **evidence,
    }
    _write_json(runner.preflight, preflight)

    schedule_stage = runner.schedule_stage()
    _write_stage_success(schedule_stage)
    artifact_stage = runner.artifact_stage(schedule_stage.success)
    _write_stage_success(artifact_stage)
    preflight_nodes = runner.preflight_node_stages(artifact_stage.success)
    for stage in preflight_nodes:
        _write_stage_success(stage)
    combine_stage = runner.preflight_combine_stage(
        tuple(stage.success for stage in preflight_nodes)
    )
    _write_stage_success(combine_stage)
    preflight_stages = (
        schedule_stage, artifact_stage, *preflight_nodes, combine_stage,
    )
    _write_phase_launch(runner, "preflight")
    for stage in preflight_stages:
        deadline = runner._stage_acceptance_spec(stage)[-1]
        marker, marker_sha = (
            orchestrator_module._validate_stage_success_snapshot(
                stage, SOURCE_COMMIT,
            )
        )
        runner._record_stage_acceptance(
            stage, marker, marker_sha, deadline - 1, deadline,
        )
    preflight_bound_files = {
        "schedule": runner.schedule,
        "artifact_manifest": runner.artifact_manifest,
        "aggregate_preflight": runner.preflight,
    }
    preflight_bound_identity = {
        "schedule_sha256": schedule["schedule_sha256"],
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "preflight_status": "PASS",
        "selected_resource_tier": "T3",
        "registry_file_sha256": registry_sha,
        "config_file_sha256": config_sha,
    }
    preflight_manifest = _write_phase_manifest(
        runner, runner.preflight_acceptance_manifest, "preflight",
        preflight_stages, preflight_bound_files, preflight_bound_identity,
        schedule["preflight_deadline_boottime_ns"] - 1,
    )

    attestation_identity = {
        "attestation_version": orchestrator_module.LOCAL_ATTESTATION_VERSION,
        "status": "PASS_EXACT",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "registry_file_sha256": registry_sha,
        "config_file_sha256": config_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": orchestrator_module._sha256_file(
            runner.schedule
        ),
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": orchestrator_module._sha256_file(
            runner.artifact_manifest
        ),
        "preflight_file_sha256": orchestrator_module._sha256_file(
            runner.preflight
        ),
        "preflight_acceptance_manifest_file_sha256": (
            orchestrator_module._sha256_file(
                runner.preflight_acceptance_manifest
            )
        ),
        "preflight_acceptance_manifest_sha256": preflight_manifest[
            "manifest_sha256"
        ],
        "remote_full_payload_sha256": evidence[
            "canonical_full_payload_sha256"
        ],
        "local_full_payload_sha256": evidence[
            "canonical_full_payload_sha256"
        ],
        "remote_portable_payload_sha256": evidence[
            "canonical_portable_payload_sha256"
        ],
        "local_portable_payload_sha256": evidence[
            "canonical_portable_payload_sha256"
        ],
        "exact_canonical_match": True,
        "portable_canonical_match": True,
        "acceptance_decisions_exact": True,
        "acceptance_decision_catalog_sha256": evidence[
            "canonical_portable_payload"
        ]["acceptance_decision_catalog_sha256"],
        "remote_full_consensus": True,
        "remote_full_consensus_nodes": ["nd-1", "nd-2", "nd-3"],
        "mismatch_paths": [],
        "importance_sampling_portable_summary": evidence[
            "canonical_portable_payload"
        ]["importance_sampling_transcript_summary"],
        "remote_importance_sampling_full_summary": evidence[
            "canonical_full_payload"
        ]["importance_sampling_transcript_summary"],
        "local_importance_sampling_full_summary": evidence[
            "canonical_full_payload"
        ]["importance_sampling_transcript_summary"],
        "solver_identity_policy": orchestrator_module.LOCAL_SOLVER_POLICY,
        "full_mismatch_policy": (
            orchestrator_module.LOCAL_FULL_MISMATCH_POLICY
        ),
        "local_environment": {
            "system": "Darwin", "machine": "arm64", "python": "3.12",
            "numpy": "2.4", "scipy": "1.17",
            "map_solver_identity_current": "solver-test",
        },
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1600,
    }
    attestation = _add_self_hash(
        attestation_identity, "attestation_sha256",
    )
    attestation_path = (
        runner.control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
    )
    _write_json(attestation_path, attestation)
    attestation_file_sha = orchestrator_module._sha256_file(attestation_path)
    runner.local_attestation = attestation_path
    runner.local_attestation_sha256 = attestation_file_sha

    map_cells = (
        {
            "code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
            "disorder_source": "attempt022",
        },
        {
            "code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
            "disorder_source": "attempt022",
        },
    )
    tasks = []
    for index in range(384):
        task = {
            "fixture_task_index": index,
            "method_id": "HP32" if index < 160 else "HP64",
        }
        if index >= 320:
            cell = map_cells[(index - 320) // 32]
            task.update({
                "method_id": "MAM-IMH8",
                "cell": cell,
                "map_artifact": {
                    "fixture_artifact": cell["code_id"],
                },
            })
        fingerprint = hashlib.sha256(
            orchestrator_module._canonical_json(task).encode("ascii")
        ).hexdigest()
        tasks.append({
            "task": task,
            "task_fingerprint": fingerprint,
            "output_relpath": f"trajectories/{fingerprint}.npz",
            "owner": orchestrator_module.EXECUTION_NODES[index % 2],
        })
    control_identity = {
        "manifest_version": "fixture.v1",
        "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
        "resource_tier": "T3",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "importance_sampling": {
            "raw_version": "fixture.is.v1",
            "num_samples_per_cell": 50000,
            "seed_namespace": "fixture_is",
            "used_for_gate_or_selection": False,
            "outputs": [
                "importance_sampling/"
                + hashlib.sha256(
                    orchestrator_module._canonical_json(cell).encode("ascii")
                ).hexdigest()
                + ".npz"
                for cell in map_cells
            ],
        },
        "task_count": len(tasks),
        "tasks": tasks,
        "execution_nodes": list(orchestrator_module.EXECUTION_NODES),
    }
    control = _add_self_hash(control_identity, "manifest_sha256")
    _write_json(runner.control, control)
    for node in orchestrator_module.EXECUTION_NODES:
        _write_json(runner.node_report_root / f"{node}.json", {"node": node})

    report_identity = {
        "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_sha256": control["manifest_sha256"],
        "raw_count": 384,
        "status": "UNRESOLVED_NO_HP_PASS",
        "selected_pair": None,
        "formal_authorization": False,
        "production_authorization": False,
    }
    report = _add_self_hash(report_identity, "report_sha256")
    _write_json(runner.report, report)
    frozen_records = orchestrator_module._terminal_control_records(
        control, config,
    )
    frozen_records.sort(
        key=lambda value: (value["kind"], value["fingerprint"]),
    )
    evidence_by_identity = {}
    evidence_rows = []
    for record in frozen_records:
        evidence_fields = (
            orchestrator_module.TERMINAL_MEASUREMENT_SUMMARY_FIELDS
            if record["kind"] == "measurement"
            else orchestrator_module.TERMINAL_TRANSCRIPT_SUMMARY_FIELDS
        )
        raw_evidence = {
            name: hashlib.sha256(
                f"{record['kind']}:{record['fingerprint']}:{name}".encode(
                    "ascii"
                )
            ).hexdigest()
            for name in evidence_fields
        }
        evidence_by_identity[(record["kind"], record["fingerprint"])] = (
            raw_evidence
        )
        evidence_rows.append({
            "kind": record["kind"],
            "fingerprint": record["fingerprint"],
            "output_relpath": record["output_relpath"],
            "sha256": "0" * 64,
            "claim_sha256": "1" * 64,
            **raw_evidence,
        })
    raw_evidence_summary = (
        orchestrator_module._terminal_raw_evidence_summary(evidence_rows)
    )
    decision_identity = {
        "decision_version": orchestrator_module.TERMINAL_DECISION_VERSION,
        "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": orchestrator_module._sha256_file(
            runner.schedule
        ),
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": orchestrator_module._sha256_file(
            runner.artifact_manifest
        ),
        "preflight_file_sha256": orchestrator_module._sha256_file(
            runner.preflight
        ),
        "control_file_sha256": orchestrator_module._sha256_file(runner.control),
        "manifest_sha256": control["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "report_file_sha256": orchestrator_module._sha256_file(runner.report),
        "status": report["status"],
        "selected_pair": None,
        "raw_evidence_summary": raw_evidence_summary,
        "formal_authorization": False,
        "production_authorization": False,
    }
    decision = _add_self_hash(decision_identity, "decision_sha256")
    _write_json(runner.decision, decision)
    raw_root = runner.raw_root
    raw_files = []
    for record in frozen_records:
        raw_path = raw_root / record["output_relpath"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(b"raw\n")
        claim_path = raw_root / record["claim_relpath"]
        _write_json(claim_path, {
            "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
            "kind": record["kind"],
            "fingerprint": record["fingerprint"],
            "manifest_sha256": control["manifest_sha256"],
            "node": record["owner"],
            "pid": 4242,
            "claimed_unix": 1700.0,
        })
        raw_files.append({
            "kind": record["kind"],
            "fingerprint": record["fingerprint"],
            "output_relpath": record["output_relpath"],
            "sha256": orchestrator_module._sha256_file(raw_path),
            "claim_sha256": orchestrator_module._sha256_file(claim_path),
            **evidence_by_identity[
                (record["kind"], record["fingerprint"])
            ],
        })
    package_identity = {
        "package_version": orchestrator_module.TERMINAL_PACKAGE_VERSION,
        "contract_version": orchestrator_module.HGP_CONTRACT_VERSION,
        "source_identity": schedule["source_identity"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": orchestrator_module._sha256_file(
            runner.schedule
        ),
        "artifact_manifest_file_sha256": orchestrator_module._sha256_file(
            runner.artifact_manifest
        ),
        "preflight_file_sha256": orchestrator_module._sha256_file(
            runner.preflight
        ),
        "control_file_sha256": orchestrator_module._sha256_file(runner.control),
        "execution_report_file_sha256": {
            node: orchestrator_module._sha256_file(
                runner.node_report_root / f"{node}.json"
            ) for node in orchestrator_module.EXECUTION_NODES
        },
        "report_file_sha256": orchestrator_module._sha256_file(runner.report),
        "decision_file_sha256": orchestrator_module._sha256_file(
            runner.decision
        ),
        "decision_sha256": decision["decision_sha256"],
        "raw_evidence_summary": raw_evidence_summary,
        "status": report["status"],
        "raw_file_count": len(raw_files),
        "raw_files": raw_files,
        "formal_authorization": False,
        "production_authorization": False,
    }
    package = _add_self_hash(package_identity, "package_sha256")
    _write_json(runner.package, package)

    control_stage = runner.control_stage(combine_stage.success)
    _write_stage_success(control_stage)
    screen_stages = runner.screen_stages(control_stage.success)
    for stage in screen_stages:
        _write_stage_success(stage)
    analysis_stage = runner.analysis_stage(
        tuple(stage.success for stage in screen_stages)
    )
    _write_stage_success(analysis_stage)
    measurement_stages = (control_stage, *screen_stages, analysis_stage)
    _write_phase_launch(runner, "measurement", attestation_file_sha)
    for stage in measurement_stages:
        deadline = runner._stage_acceptance_spec(stage)[-1]
        marker, marker_sha = (
            orchestrator_module._validate_stage_success_snapshot(
                stage, SOURCE_COMMIT,
            )
        )
        runner._record_stage_acceptance(
            stage, marker, marker_sha, deadline - 1, deadline,
        )
    measurement_bound_files = {
        "schedule": runner.schedule,
        "artifact_manifest": runner.artifact_manifest,
        "aggregate_preflight": runner.preflight,
        "local_attestation": attestation_path,
        "measurement_control": runner.control,
        "nd2_node_report": runner.node_report_root / "nd-2.json",
        "nd3_node_report": runner.node_report_root / "nd-3.json",
        "terminal_report": runner.report,
        "terminal_decision": runner.decision,
        "terminal_package": runner.package,
    }
    measurement_bound_identity = {
        "schedule_sha256": schedule["schedule_sha256"],
        "selected_resource_tier": "T3",
        "local_attestation_sha256": attestation["attestation_sha256"],
        "terminal_status": package["status"],
        "terminal_package_sha256": package["package_sha256"],
    }
    measurement_manifest = _write_phase_manifest(
        runner, runner.measurement_acceptance_manifest, "measurement",
        measurement_stages, measurement_bound_files,
        measurement_bound_identity,
        schedule["analysis_deadline_boottime_ns"] - 1,
        prior_manifest=runner.preflight_acceptance_manifest,
    )
    evidence = {
        "runner": runner,
        "registry_path": registry_path,
        "config_path": config_path,
        "schedule": schedule,
        "preflight_manifest": preflight_manifest,
        "measurement_manifest": measurement_manifest,
        "preflight_stages": preflight_stages,
        "measurement_stages": measurement_stages,
        "attestation_path": attestation_path,
    }
    _validate_terminal_evidence(evidence)
    return evidence


def _validate_terminal_evidence(evidence):
    runner = evidence["runner"]
    return orchestrator_module.validate_measurement_acceptance_offline(
        runner.measurement_acceptance_manifest, runner.run_root,
        evidence["registry_path"], evidence["config_path"], SOURCE_COMMIT,
        ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
    )


def _rehash_json(path, field):
    value = json.loads(Path(path).read_text(encoding="ascii"))
    identity = dict(value)
    identity.pop(field, None)
    value = _add_self_hash(identity, field)
    _write_json(path, value)
    return value


def _rehash_acceptance_in_manifest(evidence, phase, key):
    runner = evidence["runner"]
    acceptance = runner.acceptance_root / phase / f"{key}.json"
    acceptance_value = _rehash_json(acceptance, "acceptance_sha256")
    manifest_path = (
        runner.preflight_acceptance_manifest if phase == "preflight"
        else runner.measurement_acceptance_manifest
    )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for row in manifest["stage_acceptances"]:
        if row["stage_key"] == key:
            row["acceptance_sha256"] = acceptance_value["acceptance_sha256"]
            row["acceptance_file_sha256"] = (
                orchestrator_module._sha256_file(acceptance)
            )
            break
    identity = dict(manifest)
    identity.pop("manifest_sha256", None)
    manifest = _add_self_hash(identity, "manifest_sha256")
    _write_json(manifest_path, manifest)
    return manifest


def _replace_with_symlink(path):
    path = Path(path)
    target = path.with_name(path.name + ".target")
    path.rename(target)
    path.symlink_to(target)


def test_offline_joint_terminal_requires_package_and_measurement_acceptance(
    tmp_path,
):
    evidence = _terminal_evidence(tmp_path)

    result = _validate_terminal_evidence(evidence)

    assert result["joint_terminal_version"] == (
        orchestrator_module.JOINT_TERMINAL_VERSION
    )
    assert result["status"] == "UNRESOLVED_NO_HP_PASS"
    assert result["formal_authorization"] is False
    assert result["production_authorization"] is False
    identity = dict(result)
    stored_sha = identity.pop("joint_terminal_sha256")
    assert stored_sha == hashlib.sha256(
        orchestrator_module._canonical_json(identity).encode("ascii")
    ).hexdigest()


def test_offline_joint_terminal_rejects_missing_raw(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    package = json.loads(runner.package.read_text(encoding="ascii"))
    (runner.raw_root / package["raw_files"][0]["output_relpath"]).unlink()

    with pytest.raises(ValueError, match="raw file"):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra", "duplicate", "path_traversal", "raw_tamper",
        "claim_tamper", "claim_identity", "symlink",
    ),
)
def test_terminal_raw_evidence_is_exact_and_fail_closed(tmp_path, mutation):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    control = json.loads(runner.control.read_text(encoding="ascii"))
    config = json.loads(evidence["config_path"].read_text(encoding="ascii"))
    package = json.loads(runner.package.read_text(encoding="ascii"))
    row = package["raw_files"][0]
    raw_path = runner.raw_root / row["output_relpath"]
    fingerprint = row["fingerprint"]
    claim_dir = ".claims" if row["kind"] == "measurement" else ".claims_is"
    claim_path = runner.raw_root / claim_dir / f"{fingerprint}.json"

    if mutation == "extra":
        (runner.raw_root / "unexpected.bin").write_bytes(b"extra")
    elif mutation == "duplicate":
        package["raw_files"][1] = copy.deepcopy(package["raw_files"][0])
    elif mutation == "path_traversal":
        row["output_relpath"] = "../escaped.npz"
    elif mutation == "raw_tamper":
        raw_path.write_bytes(raw_path.read_bytes() + b"tampered")
    elif mutation == "claim_tamper":
        claim_path.write_bytes(claim_path.read_bytes() + b" ")
    elif mutation == "claim_identity":
        claim = json.loads(claim_path.read_text(encoding="ascii"))
        claim["node"] = "nd-1"
        _write_json(claim_path, claim)
        row["claim_sha256"] = orchestrator_module._sha256_file(claim_path)
    else:
        _replace_with_symlink(raw_path)

    with pytest.raises(ValueError):
        orchestrator_module._validate_terminal_raw_evidence(
            runner.run_root, control, config, package,
        )


@pytest.mark.parametrize(
    ("kind", "mutation"),
    (
        ("measurement", "missing_evidence"),
        ("importance_sampling", "extra_evidence"),
        ("measurement", "changed_evidence"),
        ("importance_sampling", "changed_evidence"),
    ),
)
def test_terminal_package_requires_exact_v2_raw_evidence_schema(
    tmp_path, kind, mutation,
):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    package = json.loads(runner.package.read_text(encoding="ascii"))
    row = next(value for value in package["raw_files"] if value["kind"] == kind)
    if mutation == "missing_evidence":
        row.pop("acceptance_decision_sha256")
    elif mutation == "extra_evidence":
        row["acceptance_decision_sha256"] = "e" * 64
    else:
        row["portable_transcript_sha256"] = "e" * 64
    identity = dict(package)
    identity.pop("package_sha256")
    _write_json(
        runner.package, _add_self_hash(identity, "package_sha256"),
    )

    with pytest.raises(ValueError, match="terminal package is invalid"):
        orchestrator_module._validate_terminal_output(
            runner.package, SOURCE_COMMIT, evidence["schedule"],
        )


def test_offline_terminal_cross_binds_decision_and_raw_evidence_summary(
    tmp_path,
):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    package = json.loads(runner.package.read_text(encoding="ascii"))
    measurement = next(
        row for row in package["raw_files"]
        if row["kind"] == "measurement"
    )
    measurement["portable_transcript_sha256"] = "e" * 64
    package["raw_evidence_summary"] = (
        orchestrator_module._terminal_raw_evidence_summary(
            package["raw_files"],
        )
    )
    package_identity = dict(package)
    package_identity.pop("package_sha256")
    package = _add_self_hash(package_identity, "package_sha256")
    _write_json(runner.package, package)

    manifest = json.loads(
        runner.measurement_acceptance_manifest.read_text(encoding="ascii")
    )
    manifest["bound_file_sha256"]["terminal_package"] = (
        orchestrator_module._sha256_file(runner.package)
    )
    manifest["bound_identity"]["terminal_package_sha256"] = package[
        "package_sha256"
    ]
    manifest_identity = dict(manifest)
    manifest_identity.pop("manifest_sha256")
    _write_json(
        runner.measurement_acceptance_manifest,
        _add_self_hash(manifest_identity, "manifest_sha256"),
    )

    with pytest.raises(ValueError, match="report/decision identity"):
        _validate_terminal_evidence(evidence)


def test_online_terminal_checks_raw_before_acceptance_publish(
    tmp_path, monkeypatch,
):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    runner.measurement_acceptance_manifest.unlink()
    monkeypatch.setattr(orchestrator_module, "_current_boot_id", lambda: BOOT_ID)
    monkeypatch.setattr(
        runner, "run_batch",
        lambda stages, _deadline: tuple(stage.success for stage in stages),
    )
    checks = []
    publishes = []

    def reject_raw(*_args, **_kwargs):
        checks.append(True)
        raise ValueError("raw evidence rejected")

    monkeypatch.setattr(
        orchestrator_module, "_validate_terminal_raw_evidence", reject_raw,
    )
    monkeypatch.setattr(
        runner, "_write_phase_acceptance_manifest",
        lambda *_args, **_kwargs: publishes.append(True),
    )

    with pytest.raises(ValueError, match="raw evidence rejected"):
        runner.run_measurement()
    assert checks == [True]
    assert publishes == []


def test_offline_stage_deadline_is_strict_at_one_nanosecond(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    key = "04_control"
    acceptance_path = runner.acceptance_root / "measurement" / f"{key}.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="ascii"))
    deadline = acceptance["deadline_boottime_ns"]
    assert acceptance["observed_boottime_ns"] == deadline - 1
    _validate_terminal_evidence(evidence)

    acceptance["observed_boottime_ns"] = deadline
    _write_json(acceptance_path, acceptance)
    _rehash_acceptance_in_manifest(evidence, "measurement", key)

    with pytest.raises(ValueError, match="stage identity"):
        _validate_terminal_evidence(evidence)


def test_offline_phase_deadline_is_strict_at_one_nanosecond(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    path = runner.measurement_acceptance_manifest
    manifest = json.loads(path.read_text(encoding="ascii"))
    deadline = manifest["phase_deadline_boottime_ns"]
    assert manifest["published_boottime_ns"] == deadline - 1
    _validate_terminal_evidence(evidence)

    manifest["published_boottime_ns"] = deadline
    _write_json(path, manifest)
    _rehash_json(path, "manifest_sha256")

    with pytest.raises(ValueError, match="measurement acceptance identity"):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize("target", ("success", "acceptance", "manifest"))
def test_offline_rejects_evidence_byte_tampering(tmp_path, target):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    paths = {
        "success": runner.marker_root / "04_control/SUCCESS",
        "acceptance": (
            runner.acceptance_root / "measurement/04_control.json"
        ),
        "manifest": runner.measurement_acceptance_manifest,
    }
    path = paths[target]
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError):
        _validate_terminal_evidence(evidence)


def test_offline_rejects_acceptance_self_hash_tamper(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    key = "04_control"
    acceptance_path = runner.acceptance_root / "measurement" / f"{key}.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="ascii"))
    acceptance["acceptance_sha256"] = "0" * 64
    _write_json(acceptance_path, acceptance)
    manifest_path = runner.measurement_acceptance_manifest
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for row in manifest["stage_acceptances"]:
        if row["stage_key"] == key:
            row["acceptance_sha256"] = "0" * 64
            row["acceptance_file_sha256"] = (
                orchestrator_module._sha256_file(acceptance_path)
            )
    identity = dict(manifest)
    identity.pop("manifest_sha256")
    _write_json(manifest_path, _add_self_hash(identity, "manifest_sha256"))

    with pytest.raises(ValueError, match="stage identity"):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize("mutation", ("missing", "extra", "order", "duplicate"))
def test_offline_rejects_acceptance_set_and_order_changes(tmp_path, mutation):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    acceptance_dir = runner.acceptance_root / "measurement"
    manifest_path = runner.measurement_acceptance_manifest
    if mutation == "missing":
        (acceptance_dir / "04_control.json").unlink()
    elif mutation == "extra":
        _write_json(acceptance_dir / "unexpected.json", {"unexpected": True})
    else:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if mutation == "order":
            manifest["stage_acceptances"][0:2] = reversed(
                manifest["stage_acceptances"][0:2]
            )
        else:
            manifest["stage_acceptances"][1] = copy.deepcopy(
                manifest["stage_acceptances"][0]
            )
        identity = dict(manifest)
        identity.pop("manifest_sha256")
        _write_json(
            manifest_path, _add_self_hash(identity, "manifest_sha256"),
        )

    with pytest.raises(ValueError):
        _validate_terminal_evidence(evidence)


def test_offline_rejects_stage_boot_id_change(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    key = "04_control"
    path = runner.acceptance_root / "measurement" / f"{key}.json"
    acceptance = json.loads(path.read_text(encoding="ascii"))
    acceptance["clock_authority_boot_id"] = (
        "fedcba98-7654-3210-fedc-ba9876543210"
    )
    _write_json(path, acceptance)
    _rehash_acceptance_in_manifest(evidence, "measurement", key)

    with pytest.raises(ValueError, match="stage identity"):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("phase", "preflight"),
        ("source_commit", "2" * 40),
        ("archive_sha256", "2" * 64),
        ("local_attestation_sha256", "2" * 64),
    ),
)
def test_offline_rejects_measurement_launch_metadata_misbinding(
        tmp_path, field, replacement):
    evidence = _terminal_evidence(tmp_path)
    path = evidence["runner"].measurement_acceptance_manifest
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest["launch_metadata"][field] = replacement
    manifest["launch_metadata_file_sha256"] = hashlib.sha256(
        (orchestrator_module._canonical_json(
            manifest["launch_metadata"]
        ) + "\n").encode("ascii")
    ).hexdigest()
    identity = dict(manifest)
    identity.pop("manifest_sha256")
    _write_json(path, _add_self_hash(identity, "manifest_sha256"))

    with pytest.raises(ValueError, match="measurement acceptance identity"):
        _validate_terminal_evidence(evidence)


def test_offline_rejects_attestation_bound_to_wrong_preflight_manifest(
    tmp_path,
):
    evidence = _terminal_evidence(tmp_path)
    path = evidence["attestation_path"]
    attestation = json.loads(path.read_text(encoding="ascii"))
    attestation["preflight_acceptance_manifest_sha256"] = "0" * 64
    identity = dict(attestation)
    identity.pop("attestation_sha256")
    _write_json(path, _add_self_hash(identity, "attestation_sha256"))

    with pytest.raises(ValueError, match="attestation identity"):
        _validate_terminal_evidence(evidence)


def test_offline_rejects_changed_prior_preflight_manifest(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    path = evidence["runner"].measurement_acceptance_manifest
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest["prior_manifest"]["manifest_sha256"] = "0" * 64
    identity = dict(manifest)
    identity.pop("manifest_sha256")
    _write_json(path, _add_self_hash(identity, "manifest_sha256"))

    with pytest.raises(ValueError, match="measurement acceptance identity"):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize(
    "target",
    (
        "success", "acceptance", "manifest", "bound_terminal",
        "bound_parent",
    ),
)
def test_offline_rejects_symlinked_evidence(tmp_path, target):
    evidence = _terminal_evidence(tmp_path)
    runner = evidence["runner"]
    if target == "bound_parent":
        real_control = runner.run_root / "real_control"
        runner.control_root.rename(real_control)
        runner.control_root.symlink_to(real_control, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            _validate_terminal_evidence(evidence)
        return
    paths = {
        "success": runner.marker_root / "04_control/SUCCESS",
        "acceptance": (
            runner.acceptance_root / "measurement/04_control.json"
        ),
        "manifest": runner.measurement_acceptance_manifest,
        "bound_terminal": runner.report,
    }
    _replace_with_symlink(paths[target])

    with pytest.raises(ValueError):
        _validate_terminal_evidence(evidence)


def test_offline_rejects_package_without_measurement_manifest(tmp_path):
    evidence = _terminal_evidence(tmp_path)
    evidence["runner"].measurement_acceptance_manifest.unlink()

    with pytest.raises(ValueError):
        _validate_terminal_evidence(evidence)


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_offline_rejects_measurement_manifest_without_exact_package(
    tmp_path, mutation,
):
    evidence = _terminal_evidence(tmp_path)
    package = evidence["runner"].package
    if mutation == "missing":
        package.unlink()
    else:
        package.write_bytes(package.read_bytes() + b" ")

    with pytest.raises(ValueError):
        _validate_terminal_evidence(evidence)


def test_local_audit_environment_uses_map_solver_identity():
    identity = local_audit_module._environment_identity()

    assert identity["map_solver_identity_current"]
    assert "numpy=" in identity["map_solver_identity_current"]
    assert "scipy=" in identity["map_solver_identity_current"]
    assert "highs=" in identity["map_solver_identity_current"]


def test_nd0_clock_authority_capture_is_boottime_bound(monkeypatch):
    samples = iter((10_000, 10_250))
    monkeypatch.setattr(orchestrator_module, "_boottime_ns", lambda: next(samples))
    monkeypatch.setattr(
        orchestrator_module.time, "time_ns", lambda: 20_000,
    )
    monkeypatch.setattr(
        orchestrator_module, "_current_boot_id", lambda: BOOT_ID,
    )

    assert orchestrator_module._capture_clock_authority() == {
        "clock_authority_version": orchestrator_module.CLOCK_AUTHORITY_VERSION,
        "clock_authority_node": "nd-0",
        "clock_authority_boot_id": BOOT_ID,
        "boottime_before_ns": 10_000,
        "authority_unix_ns": 20_000,
        "boottime_after_ns": 10_250,
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
    assert kwargs == {"check": True, "timeout": 180.0}


def test_schedule_stage_fingerprint_binds_clock_authority(tmp_path):
    runner = _orchestrator(tmp_path)
    stage = runner.schedule_stage()
    _write_stage_success(stage)
    orchestrator_module._validate_stage_success(stage, SOURCE_COMMIT)

    changed = copy.deepcopy(runner.clock_authority)
    changed["authority_unix_ns"] += 1
    runner.clock_authority = changed
    with pytest.raises(ValueError, match="command changed"):
        orchestrator_module._validate_stage_success(
            runner.schedule_stage(), SOURCE_COMMIT,
        )


def test_run_batch_rejects_success_first_observed_at_deadline(
    tmp_path, monkeypatch,
):
    runner = _orchestrator(tmp_path)
    stage = runner.schedule_stage()
    deadline = runner._stage_acceptance_spec(stage)[-1]
    samples = iter((
        deadline - 3, deadline - 2, deadline - 1, deadline,
    ))
    monkeypatch.setattr(runner, "_authority_now_ns", lambda: next(samples))
    monkeypatch.setattr(
        runner, "_launch", lambda launched: _write_stage_success(launched),
    )
    monkeypatch.setattr(runner, "_stop", lambda _stage: None)

    with pytest.raises(TimeoutError, match="frozen deadline"):
        runner.run_batch((stage,), deadline)


def test_run_batch_accepts_success_one_nanosecond_before_deadline(
    tmp_path, monkeypatch,
):
    runner = _orchestrator(tmp_path)
    stage = runner.schedule_stage()
    deadline = runner._stage_acceptance_spec(stage)[-1]
    samples = iter((
        deadline - 4, deadline - 3, deadline - 2, deadline - 1,
    ))
    monkeypatch.setattr(runner, "_authority_now_ns", lambda: next(samples))
    monkeypatch.setattr(
        runner, "_launch", lambda launched: _write_stage_success(launched),
    )

    assert runner.run_batch((stage,), deadline) == (stage.success,)


def test_run_batch_rechecks_deadline_before_each_launch(tmp_path, monkeypatch):
    runner = _orchestrator(tmp_path)
    schedule = runner.schedule_stage()
    _write_stage_success(schedule)
    artifact = runner.artifact_stage(schedule.success)
    _write_stage_success(artifact)
    first, second = runner.preflight_node_stages(artifact.success)[:2]
    deadline = runner._stage_acceptance_spec(first)[-1]
    samples = iter((deadline - 1, deadline))
    launches = []

    def fake_launch(stage):
        launches.append(stage.key)
        _write_stage_success(stage)

    monkeypatch.setattr(runner, "_authority_now_ns", lambda: next(samples))
    monkeypatch.setattr(runner, "_launch", fake_launch)

    with pytest.raises(TimeoutError, match="expired before launch"):
        runner.run_batch((first, second), deadline)
    assert launches == [first.key]


def test_authority_clock_rejects_boot_change(tmp_path, monkeypatch):
    runner = _orchestrator(tmp_path)
    monkeypatch.setattr(
        orchestrator_module, "_current_boot_id",
        lambda: "fedcba98-7654-3210-fedc-ba9876543210",
    )

    with pytest.raises(RuntimeError, match="reboot invalidated"):
        runner._authority_now_ns()


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
    (deployment / "ARCHIVE_SHA256").write_text(
        ARCHIVE_SHA256 + "\n", encoding="ascii",
    )
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

    guard = base / "logs/test-guard"

    def record_persistence(_args, _base):
        events.append("pid")
        return guard

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

    assert roots == (deployment, base / "runs" / run_id, guard)
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
    (deployment / "ARCHIVE_SHA256").write_text(
        archive_sha + "\n", encoding="ascii",
    )
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
    monkeypatch.setattr(orchestrator_module, "_current_boot_id", lambda: BOOT_ID)
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
        "selection_basis": (
            "runtime_only_worst_node_and_frozen_elapsed_deadlines"
        ),
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1500,
    })
    batches = []

    def fake_run_batch(stages, deadline_unix):
        batches.append((tuple(stage.key for stage in stages), deadline_unix))
        return tuple(stage.success for stage in stages)

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)
    def fake_acceptance_manifest(path, *_args, **_kwargs):
        value = {"manifest_sha256": "7" * 64}
        _write_json(path, value)
        return value

    monkeypatch.setattr(
        runner, "_write_phase_acceptance_manifest",
        fake_acceptance_manifest,
    )
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
    monkeypatch.setattr(orchestrator_module, "_current_boot_id", lambda: BOOT_ID)
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
        "selection_basis": (
            "runtime_only_worst_node_and_frozen_elapsed_deadlines"
        ),
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1500,
    })
    schedule_stage = runner.schedule_stage()
    artifact_stage = runner.artifact_stage(schedule_stage.success)
    preflight_stages = runner.preflight_node_stages(artifact_stage.success)
    combine_stage = runner.preflight_combine_stage(
        tuple(stage.success for stage in preflight_stages),
    )
    _write_stage_success(schedule_stage)
    _write_stage_success(artifact_stage)
    for stage in preflight_stages:
        _write_stage_success(stage)
    _write_stage_success(combine_stage)

    control_calls = []

    def forbidden_control(_preflight_success):
        control_calls.append(True)
        raise AssertionError("control stage must not be constructed")

    monkeypatch.setattr(runner, "control_stage", forbidden_control)
    monkeypatch.setattr(
        runner, "_validate_phase_acceptance_manifest",
        lambda *_args, **_kwargs: {"manifest_sha256": "7" * 64},
    )
    with pytest.raises(ValueError, match="requires a local attestation"):
        runner.run_measurement()
    assert control_calls == []

    attestation = runner.control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
    _write_json(attestation, {"tampered": True})
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
        "selection_basis": (
            "runtime_only_worst_node_and_frozen_elapsed_deadlines"
        ),
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1500,
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
        ("completed_local_unix", "invalid"),
        ("clock_domain", "nd-1_epoch"),
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


def test_schedule_and_control_are_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "_current_boot_id", lambda: BOOT_ID)
    schedule = _schedule()
    schedule_path = tmp_path / "schedule.json"
    _write_json(schedule_path, schedule)
    assert orchestrator_module._validate_schedule_output(
        schedule_path, "exp102_q0_hgp_test", SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
    ) == schedule
    monkeypatch.setattr(orchestrator_module.time, "time", lambda: -1e12)
    assert orchestrator_module._validate_schedule_output(
        schedule_path, "exp102_q0_hgp_test", SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, CONFIG_SHA256,
    ) == schedule

    future_identity = dict(schedule)
    future_identity.pop("schedule_sha256")
    for name in (
            "started_unix", "preflight_deadline_unix",
            "control_freeze_deadline_unix", "screen_deadline_unix",
            "analysis_deadline_unix"):
        future_identity[name] += 3600
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
    acceptance_path = control_root / "HGP_ND0_PREFLIGHT_ACCEPTANCE.json"
    _write_json(schedule_path, schedule)
    _write_json(artifact_path, {
        "artifact": "bound", "artifact_manifest_sha256": "f" * 64,
    })
    evidence = _canonical_evidence()
    preflight = {
        **evidence,
        "remote_full_consensus": True,
        "nodes": ["nd-1", "nd-2", "nd-3"],
    }
    _write_json(preflight_path, preflight)
    _write_json(acceptance_path, {"manifest_sha256": "7" * 64})
    identity = {
        "attestation_version": orchestrator_module.LOCAL_ATTESTATION_VERSION,
        "status": "PASS_EXACT",
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
        "preflight_acceptance_manifest_file_sha256": (
            orchestrator_module._sha256_file(acceptance_path)
        ),
        "preflight_acceptance_manifest_sha256": "7" * 64,
        "remote_full_payload_sha256": evidence[
            "canonical_full_payload_sha256"
        ],
        "local_full_payload_sha256": evidence[
            "canonical_full_payload_sha256"
        ],
        "remote_portable_payload_sha256": evidence[
            "canonical_portable_payload_sha256"
        ],
        "local_portable_payload_sha256": evidence[
            "canonical_portable_payload_sha256"
        ],
        "exact_canonical_match": True,
        "portable_canonical_match": True,
        "acceptance_decisions_exact": True,
        "acceptance_decision_catalog_sha256": evidence[
            "canonical_portable_payload"
        ]["acceptance_decision_catalog_sha256"],
        "remote_full_consensus": True,
        "remote_full_consensus_nodes": ["nd-1", "nd-2", "nd-3"],
        "mismatch_paths": [],
        "importance_sampling_portable_summary": evidence[
            "canonical_portable_payload"
        ]["importance_sampling_transcript_summary"],
        "remote_importance_sampling_full_summary": evidence[
            "canonical_full_payload"
        ]["importance_sampling_transcript_summary"],
        "local_importance_sampling_full_summary": evidence[
            "canonical_full_payload"
        ]["importance_sampling_transcript_summary"],
        "solver_identity_policy": orchestrator_module.LOCAL_SOLVER_POLICY,
        "full_mismatch_policy": (
            orchestrator_module.LOCAL_FULL_MISMATCH_POLICY
        ),
        "local_environment": {
            "system": "Darwin", "machine": "arm64", "python": "3.12",
            "numpy": "2.4.1", "scipy": "1.17.0",
            "map_solver_identity_current": "local-test",
        },
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": 1500,
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
        acceptance_path, "e" * 64, CONFIG_SHA256, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256,
    ) == attestation

    with pytest.raises(ValueError, match="file SHA mismatch"):
        orchestrator_module._validate_local_attestation(
            path, "9" * 64, schedule, preflight, artifact_path,
            acceptance_path, "e" * 64, CONFIG_SHA256, SOURCE_COMMIT,
            ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256,
        )
    portable = copy.deepcopy(attestation)
    portable["status"] = "PORTABLE_PASS"
    portable["exact_canonical_match"] = False
    portable["local_full_payload_sha256"] = "e" * 64
    portable["mismatch_paths"] = ["$.cells[0].nonportable_float_sha256"]
    portable["local_importance_sampling_full_summary"] = copy.deepcopy(
        portable["local_importance_sampling_full_summary"],
    )
    portable["local_importance_sampling_full_summary"][0][
        "full_transcript_sha256"
    ] = "a" * 64
    portable["local_importance_sampling_full_summary"][0][
        "nonportable_float_sha256"
    ] = "b" * 64
    portable_identity = dict(portable)
    portable_identity.pop("attestation_sha256")
    portable["attestation_sha256"] = orchestrator_module._sha256_json(
        portable_identity,
    )
    _write_json(path, portable)
    assert orchestrator_module._validate_local_attestation(
        path, orchestrator_module._sha256_file(path), schedule, preflight,
        artifact_path, acceptance_path, "e" * 64, CONFIG_SHA256,
        SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
    ) == portable

    for field, value in (
            ("acceptance_decisions_exact", False),
            ("portable_canonical_match", False),
            ("local_portable_payload_sha256", "9" * 64)):
        forged = copy.deepcopy(portable)
        forged[field] = value
        forged_identity = dict(forged)
        forged_identity.pop("attestation_sha256")
        forged["attestation_sha256"] = orchestrator_module._sha256_json(
            forged_identity,
        )
        _write_json(path, forged)
        with pytest.raises(ValueError, match="identity is invalid"):
            orchestrator_module._validate_local_attestation(
                path, orchestrator_module._sha256_file(path), schedule,
                preflight, artifact_path, acceptance_path, "e" * 64,
                CONFIG_SHA256, SOURCE_COMMIT, ARCHIVE_SHA256,
                SOURCE_MANIFEST_SHA256,
            )

    old = copy.deepcopy(portable)
    old["attestation_version"] = (
        "exp102.q0_hgp_global.screen.local_attestation.v3"
    )
    old_identity = dict(old)
    old_identity.pop("attestation_sha256")
    old["attestation_sha256"] = orchestrator_module._sha256_json(old_identity)
    _write_json(path, old)
    with pytest.raises(ValueError, match="identity is invalid"):
        orchestrator_module._validate_local_attestation(
            path, orchestrator_module._sha256_file(path), schedule, preflight,
            artifact_path, acceptance_path, "e" * 64, CONFIG_SHA256,
            SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
        )

    no_consensus = dict(preflight)
    no_consensus["remote_full_consensus"] = False
    _write_json(path, portable)
    with pytest.raises(ValueError, match="identity is invalid"):
        orchestrator_module._validate_local_attestation(
            path, orchestrator_module._sha256_file(path), schedule,
            no_consensus, artifact_path, acceptance_path, "e" * 64,
            CONFIG_SHA256, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(("remote_full_marker", "expected_status"), (
    ("6", "PASS_EXACT"),
    ("a", "PORTABLE_PASS"),
))
def test_local_audit_complete_exact_or_portable_success_path(
        tmp_path, monkeypatch, remote_full_marker, expected_status):
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config.json"
    schedule_path = tmp_path / "schedule.json"
    artifact_root = tmp_path / "artifacts"
    artifact_manifest_path = tmp_path / "artifact_manifest.json"
    preflight_path = tmp_path / "preflight.json"
    preflight_acceptance_path = tmp_path / "preflight_acceptance.json"
    work_root = tmp_path / "work"
    output_path = tmp_path / "attestation.json"
    for path in (
            registry_path, config_path, schedule_path,
            artifact_manifest_path, preflight_path,
            preflight_acceptance_path):
        _write_json(path, {"placeholder": path.name})
    artifact_root.mkdir()

    schedule = {"schedule_sha256": "3" * 64}
    artifact_manifest = {"artifact_manifest_sha256": "4" * 64}
    evidence = _canonical_evidence(full_marker=remote_full_marker)
    base_evidence = _base_canonical_evidence()
    preflight = {
        "status": "PASS",
        "remote_full_consensus": True,
        "nodes": ["nd-1", "nd-2", "nd-3"],
        "selected_resource_tier": "T3",
        **evidence,
    }
    monkeypatch.setenv("EXP102_SOURCE_COMMIT", SOURCE_COMMIT)
    monkeypatch.setattr(
        local_audit_module.workflow, "_load_registry", lambda _path: {},
    )
    monkeypatch.setattr(
        local_audit_module.workflow, "_load_config", lambda *_args: {},
    )
    monkeypatch.setattr(
        local_audit_module.workflow, "_validate_schedule",
        lambda *_args: (schedule, schedule_path),
    )
    monkeypatch.setattr(
        local_audit_module.workflow, "_validate_artifact_manifest",
        lambda *_args: (artifact_manifest, artifact_manifest_path),
    )
    monkeypatch.setattr(
        local_audit_module.workflow, "_validate_preflight",
        lambda *_args: preflight,
    )
    monkeypatch.setattr(
        local_audit_module.orchestration,
        "validate_preflight_acceptance_offline",
        lambda *_args, **_kwargs: {"manifest_sha256": "7" * 64},
    )
    digest_calls = []

    def replay_digest(*_args, **kwargs):
        digest_calls.append(kwargs)
        return copy.deepcopy(base_evidence)

    monkeypatch.setattr(
        local_audit_module.pipeline, "hgp_screen_preflight_digest",
        replay_digest,
    )
    monkeypatch.setattr(
        local_audit_module.pipeline, "_map_cells", lambda _config: [{"cell": 0}],
    )
    monkeypatch.setattr(
        local_audit_module.pipeline, "_cell_fingerprint", lambda _cell: "5" * 64,
    )

    def write_is(_registry, _config, _commit, _archive, _manifest, _cell,
                 _artifact_root, path, **_kwargs):
        Path(path).write_bytes(b"frozen-is")
        return {
            "full_transcript_sha256": "7" * 64,
            "portable_transcript_sha256": "8" * 64,
            "nonportable_float_sha256": "6" * 64,
            "field_manifest_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        local_audit_module.pipeline, "run_hgp_map_is_diagnostic", write_is,
    )
    monkeypatch.setattr(
        local_audit_module.pipeline, "validate_hgp_map_is_diagnostic",
        lambda *_args, **_kwargs: {
            "full_transcript_sha256": "7" * 64,
            "portable_transcript_sha256": "8" * 64,
            "nonportable_float_sha256": "6" * 64,
            "field_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        local_audit_module, "_environment_identity",
        lambda: {
            "system": "Darwin", "machine": "arm64", "python": "3.12",
            "numpy": "2.4.1", "scipy": "1.17.0",
            "map_solver_identity_current": "solver-test",
        },
    )

    attestation = local_audit_module.audit_local_preflight(
        registry_path, config_path, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, schedule_path, artifact_root,
        artifact_manifest_path, preflight_path, preflight_acceptance_path,
        work_root, output_path,
    )

    assert attestation["status"] == expected_status
    assert digest_calls == [{}]
    assert attestation["exact_canonical_match"] is (
        expected_status == "PASS_EXACT"
    )
    assert bool(attestation["mismatch_paths"]) is (
        expected_status == "PORTABLE_PASS"
    )
    assert attestation["importance_sampling_portable_summary"] == evidence[
        "canonical_portable_payload"
    ]["importance_sampling_transcript_summary"]
    identity = dict(attestation)
    stored_sha = identity.pop("attestation_sha256")
    assert stored_sha == local_audit_module.workflow._sha256_json(identity)
    assert json.loads(output_path.read_text(encoding="ascii")) == attestation


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
