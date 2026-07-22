import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import time

import numpy as np
import pytest


workflow = importlib.import_module(
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.workflow"
)
orchestrator = importlib.import_module(
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.orchestrate_hgp"
)


SOURCE_COMMIT = "1" * 40
ARCHIVE_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64
EXP102_ROOT = Path(__file__).resolve().parents[1]


def _fake_model_and_frame():
    model = SimpleNamespace(
        num_qubits=3,
        num_checks=1,
        k=1,
        H_check=np.array([[1, 1, 0]], dtype=np.uint8),
    )
    frame = SimpleNamespace(
        num_qubits=3,
        k=1,
        W_basis=np.array([[1, 0, 0]], dtype=np.uint8),
    )
    return model, frame


def test_artifact_manifest_is_canonical_and_rejects_extra_files(
    tmp_path, monkeypatch,
):
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config.json"
    registry_path.write_text("registry\n", encoding="ascii")
    config_path.write_text("config\n", encoding="ascii")
    artifact_root = tmp_path / "artifacts"
    artifact_path = artifact_root / "map_artifacts/a.npz"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"artifact")
    descriptors = [{
        "artifact_relpath": "map_artifacts/a.npz",
        "cell_fingerprint": "c" * 64,
    }]
    monkeypatch.setattr(
        workflow, "_load_map_artifact_descriptors", lambda *args: descriptors,
    )
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text("schedule\n", encoding="ascii")
    schedule = {"schedule_sha256": "d" * 64}
    identity = workflow._artifact_manifest_identity(
        registry_path, config_path, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, schedule, schedule_path, descriptors,
    )
    manifest = {
        **identity,
        "artifact_manifest_sha256": workflow._sha256_json(identity),
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        workflow._canonical_json(manifest) + "\n", encoding="ascii",
    )
    validated, _ = workflow._validate_artifact_manifest(
        manifest_path, registry_path, config_path, SOURCE_COMMIT,
        ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
        schedule, schedule_path,
    )
    assert validated == manifest

    extra = artifact_root / "map_artifacts/extra.txt"
    extra.write_text("not allowed", encoding="ascii")
    with pytest.raises(ValueError, match="incomplete or has extras"):
        workflow._validate_artifact_manifest(
            manifest_path, registry_path, config_path, SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
            schedule, schedule_path,
        )
    extra.unlink()
    symlink = artifact_root / "map_artifacts/link.npz"
    symlink.symlink_to(artifact_path)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        workflow._validate_artifact_manifest(
            manifest_path, registry_path, config_path, SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
            schedule, schedule_path,
        )
    symlink.unlink()
    manifest["artifact_count"] = 2
    manifest_path.write_text(
        workflow._canonical_json(manifest) + "\n", encoding="ascii",
    )
    with pytest.raises(ValueError, match="noncanonical or stale"):
        workflow._validate_artifact_manifest(
            manifest_path, registry_path, config_path, SOURCE_COMMIT,
            ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
            schedule, schedule_path,
        )


def test_importance_records_follow_manifest_scope_without_fixed_counts():
    cells = [
        {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0},
        {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0},
    ]
    descriptors = [
        {"cell_fingerprint": str(index) * 64}
        for index in (1, 2)
    ]
    tasks = []
    for index in range(384):
        cell_index = index % 2
        task = {
            "cell": cells[cell_index],
            "method_id": "MAM-IMH8" if index < 64 else "HP32",
        }
        if index < 64:
            task["map_artifact"] = descriptors[cell_index]
        tasks.append({"task": task})
    manifest = {
        "manifest_sha256": "3" * 64,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "execution_nodes": ["nd-2", "nd-3"],
        "tasks": tasks,
        "importance_sampling": {
            "outputs": [
                "importance_sampling/a.npz",
                "importance_sampling/b.npz",
            ],
            "raw_version": "is.v1",
            "num_samples_per_cell": 50_000,
            "used_for_gate_or_selection": False,
        },
    }
    config = {"importance_sampling": {"num_samples_per_cell": 50_000}}
    assert len(workflow._manifest_records({
        "tasks": [
            {
                "task": {"index": index},
                "task_fingerprint": workflow._sha256_json({"index": index}),
                "output_relpath": f"trajectories/{index}.npz",
                "owner": "nd-2" if index % 2 == 0 else "nd-3",
            }
            for index in range(384)
        ],
    })) == 384
    records = workflow._importance_records(manifest, config)
    assert len(records) == 2
    assert [record["cell"] for record in records] == cells
    assert [record["owner"] for record in records] == ["nd-2", "nd-3"]


def _preflight_digest_bundle(*, full_probe=1.0):
    pipeline = workflow._pipeline()
    catalog = [{
        "cell_fingerprint": "1" * 64,
        "method_id": "MAM-IMH8",
        "init_family": "P",
        "acceptance_decision_sha256": "2" * 64,
    }]
    common = {
        "evidence_schema_version": pipeline.HGP_PREFLIGHT_EVIDENCE_VERSION,
        "acceptance_decision_catalog": catalog,
        "acceptance_decision_catalog_sha256": workflow._sha256_json(catalog),
    }
    return workflow._build_preflight_evidence_bundle(
        {**common, "evidence_projection": "full", "probe": full_probe},
        {**common, "evidence_projection": "portable", "probe": 1},
    )


def test_preflight_full_and_portable_evidence_are_sealed_and_exact():
    summaries = [{
        "cell_fingerprint": "3" * 64,
        "full_transcript_sha256": "4" * 64,
        "portable_transcript_sha256": "5" * 64,
        "nonportable_float_sha256": "6" * 64,
        "field_manifest_sha256": "7" * 64,
    }]
    bundled = workflow._add_preflight_is_summaries(
        _preflight_digest_bundle(), summaries,
    )
    assert set(bundled) == workflow._PREFLIGHT_DIGEST_FIELDS
    assert bundled["canonical_full_payload"][
        "importance_sampling_transcript_summary"
    ] == summaries
    assert bundled["canonical_portable_payload"][
        "importance_sampling_transcript_summary"
    ] == [{
        "cell_fingerprint": "3" * 64,
        "portable_transcript_sha256": "5" * 64,
        "field_manifest_sha256": "7" * 64,
    }]
    consensus = {
        node: copy.deepcopy(bundled)
        for node in workflow.EXPECTED_PREFLIGHT_NODES
    }
    assert workflow._linux_preflight_digest_consensus(consensus) == bundled

    # One representable-float step is still a byte-level Linux conflict.
    changed = _preflight_digest_bundle(full_probe=np.nextafter(1.0, 2.0))
    changed = workflow._add_preflight_is_summaries(changed, summaries)
    consensus["nd-3"] = changed
    with pytest.raises(ValueError, match="full payload differs"):
        workflow._linux_preflight_digest_consensus(consensus)

    tampered = copy.deepcopy(bundled)
    tampered["canonical_full_payload"][
        "acceptance_decision_catalog"
    ][0]["decision_transcript_sha256"] = "8" * 64
    tampered["canonical_full_payload"][
        "acceptance_decision_catalog_sha256"
    ] = workflow._sha256_json(tampered["canonical_full_payload"][
        "acceptance_decision_catalog"
    ])
    tampered["canonical_full_payload_sha256"] = workflow._sha256_json(
        tampered["canonical_full_payload"],
    )
    with pytest.raises(ValueError, match="acceptance row"):
        workflow._validate_preflight_digest_bundle(tampered)


def test_staging_measurement_checks_structure_without_full_sampler_replay(
    tmp_path, monkeypatch,
):
    model, frame = _fake_model_and_frame()
    monkeypatch.setattr(
        workflow, "_load_model_for_cell", lambda *args: (model, frame),
    )
    monkeypatch.setattr(
        workflow, "_load_registry", lambda *args: {"registry_sha256": "r"},
    )
    monkeypatch.setattr(
        workflow,
        "_pipeline",
        lambda: SimpleNamespace(
            COLLAPSED_RAW_VERSION="collapsed.v1", MAP_RAW_VERSION="map.v1",
        ),
    )
    monkeypatch.setattr(
        workflow, "_validate_raw",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("staging invoked the full sampler replay")
        ),
    )
    evidence = {
        name: str(index + 1) * 64
        for index, name in enumerate(workflow._MEASUREMENT_SUMMARY_FIELDS)
    }
    monkeypatch.setattr(
        workflow, "_validate_stored_measurement_evidence",
        lambda path: dict(evidence),
    )
    task = {
        "method_id": "HP32",
        "cell": {"code_id": "m06_c00", "p": 0.04},
    }
    config = {
        "hgp_screen_config_sha256": "c",
        "raw_versions": {"hp": "outer.v1", "map": "map.outer.v1"},
    }
    packed_state = np.array([0], dtype=np.uint8)
    packed_measurements = np.zeros((8, 1), dtype=np.uint8)
    sampler = {
        "raw_version": np.array("collapsed.v1"),
        "initial_state_packed": packed_state,
        "burn_state_packed": packed_state,
        "final_state_packed": packed_state,
        "measurement_states_packed": packed_measurements,
        "measurement_weights": np.zeros(8, dtype=np.int32),
        "measurement_labels": np.zeros(8, dtype=np.uint64),
        "measurement_residual_weights": np.zeros(8, dtype=np.int32),
        "measurement_block": np.arange(8, dtype=np.int8),
    }
    payload = {
        "raw_version": np.array("outer.v1"),
        "contract_version": np.array(workflow.CONTRACT_VERSION),
        "task_json": np.array(workflow._canonical_json(task)),
        "task_fingerprint": np.array(workflow._sha256_json(task)),
        "source_commit": np.array(SOURCE_COMMIT),
        "archive_sha256": np.array(ARCHIVE_SHA256),
        "source_manifest_sha256": np.array(SOURCE_MANIFEST_SHA256),
        "registry_sha256": np.array("r"),
        "hgp_screen_config_sha256": np.array("c"),
        "cell_json": np.array(workflow._canonical_json(task["cell"])),
        "uniform_seed": np.array(1, dtype=np.uint64),
        "syndrome_packed": np.array([0], dtype=np.uint8),
        "syndrome_sha256": np.array("s"),
        "model_fingerprint": np.array("m"),
        "section_fingerprint": np.array("x"),
        "logical_frame_fingerprint": np.array("f"),
        "character_masks": np.array([1], dtype=np.uint64),
        "character_sha256": np.array("h"),
        "b_character_masks_packed": np.zeros((65, 1), dtype=np.uint8),
        "b_character_sha256": np.array("b" * 64),
        "b_character_count": np.array(65, dtype=np.int32),
        "b_dimension": np.array(1, dtype=np.int32),
        "b_dense_character_count": np.array(64, dtype=np.int16),
        "num_qubits": np.array(3, dtype=np.int32),
        "k": np.array(1, dtype=np.int16),
        "field_manifest_json": np.array("{}"),
        "field_manifest_sha256": np.array(
            evidence["field_manifest_sha256"],
        ),
        "full_transcript_sha256": np.array(
            evidence["full_transcript_sha256"],
        ),
        "portable_transcript_sha256": np.array(
            evidence["portable_transcript_sha256"],
        ),
        "nonportable_float_sha256": np.array(
            evidence["nonportable_float_sha256"],
        ),
        "acceptance_decision_sha256": np.array(
            evidence["acceptance_decision_sha256"],
        ),
        "core_seconds": np.array(1.0),
        "wall_seconds": np.array(1.0),
    }
    payload.update({f"sampler_{name}": value for name, value in sampler.items()})
    path = tmp_path / "measurement.npz"
    np.savez(path, **payload)
    assert workflow._validate_staging_measurement(
        path, tmp_path / "registry.json", config, task, SOURCE_COMMIT,
        ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, tmp_path,
    ) == evidence


def test_staging_map_rebuilds_coordinates_and_transition_counters(tmp_path):
    artifact_path = tmp_path / "map.npz"
    np.savez(
        artifact_path,
        coordinate_packed_reference=np.array([0], dtype=np.uint8),
        coordinate_packed_basis=np.array([[1], [2]], dtype=np.uint8),
        anchors=np.zeros((2, 3), dtype=np.uint8),
        proposal_component_weights=np.array([0.5, 0.5]),
    )
    payload = {
        "sampler_initial_coordinate_packed": np.array([0], dtype=np.uint8),
        "sampler_burn_coordinate_packed": np.array([1], dtype=np.uint8),
        "sampler_final_coordinate_packed": np.array([3], dtype=np.uint8),
        "sampler_initial_state_packed": np.array([0], dtype=np.uint8),
        "sampler_burn_state_packed": np.array([1], dtype=np.uint8),
        "sampler_final_state_packed": np.array([3], dtype=np.uint8),
    }
    for stage, proposal, state in (
            ("burn", 1, 1), ("measurement", 3, 3)):
        payload.update({
            f"sampler_{stage}_proposal_coordinates_packed": np.array(
                [[proposal]], dtype=np.uint8,
            ),
            f"sampler_{stage}_proposal_states_packed": np.array(
                [[state]], dtype=np.uint8,
            ),
            f"sampler_{stage}_states_packed": np.array(
                [[state]], dtype=np.uint8,
            ),
            f"sampler_{stage}_accepted": np.array([1], dtype=np.uint8),
            f"sampler_{stage}_state_changed": np.array([1], dtype=np.uint8),
            f"sampler_{stage}_accept_uniform": np.array([0.25]),
            f"sampler_{stage}_proposal_anchor_index": np.array(
                [1], dtype=np.int16,
            ),
            f"sampler_{stage}_proposal_component_index": np.array(
                [1], dtype=np.int8,
            ),
            f"sampler_{stage}_attempts": np.array(1, dtype=np.int64),
            f"sampler_{stage}_accepts": np.array(1, dtype=np.int64),
            f"sampler_{stage}_state_changes": np.array(1, dtype=np.int64),
        })
    raw_path = tmp_path / "map_raw.npz"
    np.savez(raw_path, **payload)
    model = SimpleNamespace(num_qubits=3)
    with np.load(raw_path, allow_pickle=False) as data:
        workflow._validate_map_staging_algebra(data, artifact_path, model)

    changed = dict(payload)
    changed["sampler_measurement_proposal_coordinates_packed"] = np.array(
        [[2]], dtype=np.uint8,
    )
    np.savez(raw_path, **changed)
    with np.load(raw_path, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="coordinate/state mismatch"):
            workflow._validate_map_staging_algebra(data, artifact_path, model)

    changed = dict(payload)
    changed["sampler_measurement_accept_uniform"] = np.array([1.0])
    np.savez(raw_path, **changed)
    with np.load(raw_path, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="shape/range"):
            workflow._validate_map_staging_algebra(data, artifact_path, model)


def test_staging_is_checks_structure_without_redrawing_samples(
    tmp_path, monkeypatch,
):
    model, _ = _fake_model_and_frame()
    monkeypatch.setattr(
        workflow, "_load_model_for_cell", lambda *args: (model, None),
    )
    monkeypatch.setattr(
        workflow, "_validate_is",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("staging redrew the importance samples")
        ),
    )
    evidence = {
        name: format(index + 10, "x") * 64
        for index, name in enumerate(workflow._TRANSCRIPT_SUMMARY_FIELDS)
    }
    monkeypatch.setattr(
        workflow, "_validate_stored_is_evidence",
        lambda path: dict(evidence),
    )
    artifact_root = tmp_path / "artifacts"
    artifact_path = artifact_root / "map_artifacts/a.npz"
    artifact_path.parent.mkdir(parents=True)
    np.savez(
        artifact_path,
        metadata_json=np.array(workflow._canonical_json({"num_checks": 1})),
        syndrome_packed=np.array([0], dtype=np.uint8),
    )
    descriptor = {
        "artifact_relpath": "map_artifacts/a.npz",
        "artifact_file_sha256": workflow._sha256_file(artifact_path),
    }
    record = {
        "cell": {"code_id": "m06_c00", "p": 0.04},
        "artifact_descriptor": descriptor,
    }
    config = {
        "hgp_screen_config_sha256": "c",
        "importance_sampling": {"num_samples_per_cell": 2},
        "raw_versions": {"importance_sampling": "is.v2"},
    }
    identity = {
        "contract_version": workflow.CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "hgp_screen_config_sha256": "c",
        "cell": record["cell"],
        "artifact_descriptor": descriptor,
        "num_samples": 2,
        "used_for_gate_or_selection": False,
        "raw_version": "is.v2",
    }
    arrays = {
        "sample_states_packed": np.zeros((2, 1), dtype=np.uint8),
        "sample_coordinates_packed": np.zeros((2, 1), dtype=np.uint8),
        "sample_physical_weights": np.zeros(2, dtype=np.int32),
        "sample_log_q": np.zeros(2, dtype=np.float64),
        "sample_log_importance_weight": np.zeros(2, dtype=np.float64),
        "sample_anchor_index": np.zeros(2, dtype=np.int16),
        "sample_component_index": np.zeros(2, dtype=np.int16),
    }
    path = tmp_path / "is.npz"
    diagnostics = {"num_samples": 2}
    field_manifest = {"manifest_version": "test.v1"}
    np.savez(
        path,
        identity_json=np.array(workflow._canonical_json(identity)),
        diagnostics_json=np.array(workflow._canonical_json(diagnostics)),
        field_manifest_json=np.array(
            workflow._canonical_json(field_manifest),
        ),
        field_manifest_sha256=np.array(evidence["field_manifest_sha256"]),
        full_transcript_sha256=np.array(
            evidence["full_transcript_sha256"],
        ),
        portable_transcript_sha256=np.array(
            evidence["portable_transcript_sha256"],
        ),
        nonportable_float_sha256=np.array(
            evidence["nonportable_float_sha256"],
        ),
        **arrays,
    ) == evidence
    assert workflow._validate_staging_is(
        path, tmp_path / "registry.json", config, record, SOURCE_COMMIT,
        ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
    )


def test_claim_is_exclusive(tmp_path):
    path = tmp_path / "claim.json"
    workflow._claim_raw(path, "f", "nd-2", "m", "measurement")
    with pytest.raises(FileExistsError):
        workflow._claim_raw(path, "f", "nd-2", "m", "measurement")


def test_runtime_accounting_accepts_exact_placement_and_rejects_double_count():
    pipeline = workflow._pipeline()
    registry_path = EXP102_ROOT / "registry/registry.json"
    config_path = EXP102_ROOT / "config/q0_hgp_global.screen.v2.json"
    registry = workflow._load_registry(registry_path)
    config = workflow._load_config(config_path, registry)
    timings = {
        "HP32": {
            "benchmark_seconds": 0.0, "benchmark_steps": 160,
            "seconds_per_step": 0.0, "setup_seconds_per_task": 1.0,
        },
        "HP64": {
            "benchmark_seconds": 0.0, "benchmark_steps": 160,
            "seconds_per_step": 0.0, "setup_seconds_per_task": 2.0,
        },
        "MAM-IMH8": {
            "benchmark_seconds": 0.0, "benchmark_steps": 160,
            "seconds_per_step": 0.0, "setup_seconds_per_task": 3.0,
        },
    }
    map_cells = pipeline._map_cells(config)
    is_seconds = {
        pipeline._cell_fingerprint(cell): float(index + 4)
        for index, cell in enumerate(map_cells)
    }
    b_analysis_timings = {
        "benchmark_measurement_rounds": 32768,
        "trace_benchmark_seconds": 3276.8,
        "trace_seconds_per_round": 0.1,
        "family_benchmark_seconds": 2.0,
        "comparison_benchmark_seconds": 3.0,
    }
    tiers, owner_counts = pipeline._runtime_tier_projections(
        config, timings, is_seconds, 6.0, b_analysis_timings,
    )
    runtime_context = {
        "timings": timings,
        "is_seconds_by_cell": is_seconds,
        "artifact_generation_wall_seconds": 6.0,
        "owner_task_counts": owner_counts,
        "b_analysis_timings": b_analysis_timings,
    }
    artifact_manifest = {
        "map_artifacts": [{"generation_wall_seconds": 6.0}],
    }
    row = {**tiers["T1"], "resource_tier": "T1"}
    accounting = workflow._runtime_accounting(config)
    workflow._validate_projection_accounting(
        row, accounting, config, runtime_context, artifact_manifest,
    )

    changed = copy.deepcopy(row)
    changed["projected_complete_schedule_seconds"] *= 2.0
    with pytest.raises(ValueError, match="double-counted"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )
    changed = copy.deepcopy(row)
    changed["analysis_workload"]["sampler_replay_passes_per_task"] = 0
    with pytest.raises(ValueError, match="placement or replay mode"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )
    changed = copy.deepcopy(row)
    del changed["analysis_workload"]["b_statistical_diagnostics_seconds"]
    with pytest.raises(ValueError, match="analysis replay LPT"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )
    changed_context = copy.deepcopy(runtime_context)
    del changed_context["b_analysis_timings"]
    with pytest.raises(ValueError, match="omitted frozen replay accounting"):
        workflow._validate_projection_accounting(
            row, accounting, config, changed_context, artifact_manifest,
        )
    changed = copy.deepcopy(row)
    changed["analysis_workload"]["b_family_diagnostics_seconds"] *= 0.5
    with pytest.raises(ValueError, match="analysis replay LPT"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )
    changed = copy.deepcopy(row)
    changed["eligible"] = not changed["eligible"]
    with pytest.raises(ValueError, match="eligibility"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )
    changed = copy.deepcopy(row)
    changed["per_node_generation_workload"]["nd-2"][
        "sampler_generation_lpt_seconds"
    ] += 1.0
    changed["per_node_generation_workload"]["nd-2"][
        "projected_generation_wall_seconds"
    ] += 1.0
    with pytest.raises(ValueError, match="LPT projection"):
        workflow._validate_projection_accounting(
            changed, accounting, config, runtime_context, artifact_manifest,
        )


def test_cli_freezes_provenance_and_analysis_placement():
    parser = workflow._parser()
    common = [
        "--source-commit", SOURCE_COMMIT,
        "--archive-sha256", ARCHIVE_SHA256,
        "--source-manifest-sha256", SOURCE_MANIFEST_SHA256,
        "--artifact-root", "/artifacts",
        "--artifact-manifest", "/artifact-manifest.json",
        "--schedule", "/schedule.json",
    ]
    args = parser.parse_args([
        "analyze", "nd-3", *common,
        "--control", "/control.json", "--preflight", "/preflight.json",
        "--node-report", "nd-2=/nd2.json",
        "--node-report", "nd-3=/nd3.json", "--raw-root", "/raw",
        "--output", "/report.json", "--decision-output", "/decision.json",
        "--package-output", "/package.json", "--num-workers", "91",
    ])
    assert args.node == "nd-3"
    assert args.num_workers == 91
    with pytest.raises(SystemExit):
        parser.parse_args([
            "analyze", "nd-3", *common,
            "--control", "/control.json", "--preflight", "/preflight.json",
            "--node-report", "nd-2=/nd2.json",
            "--node-report", "nd-3=/nd3.json", "--raw-root", "/raw",
            "--output", "/report.json", "--decision-output", "/decision.json",
            "--package-output", "/package.json",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "run-node", "nd-2", *common,
            "--control", "/control.json", "--raw-root", "/raw",
            "--output", "/node.json", "--num-workers", "75",
        ])


def test_schedule_is_canonical_deadlined_and_ignores_local_epoch(
    tmp_path, monkeypatch,
):
    registry_path = EXP102_ROOT / "registry/registry.json"
    config_path = EXP102_ROOT / "config/q0_hgp_global.screen.v2.json"
    registry = workflow._load_registry(registry_path)
    config = workflow._load_config(config_path, registry)
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "file_count": 1,
    }
    monkeypatch.setattr(
        workflow, "_verify_provenance", lambda *args: source_identity,
    )
    clock_authority = {
        "clock_authority_version": workflow.CLOCK_AUTHORITY_VERSION,
        "clock_authority_node": "nd-0",
        "clock_authority_boot_id": (
            "01234567-89ab-cdef-0123-456789abcdef"
        ),
        "boottime_before_ns": 5_000_000_000_000,
        "authority_unix_ns": 1_000_000_000_000,
        "boottime_after_ns": 5_000_000_001_000,
    }
    started = 1000
    started_boottime_ns = clock_authority["boottime_before_ns"]
    identity = {
        "schedule_version": workflow.SCHEDULE_VERSION,
        "contract_version": workflow.CONTRACT_VERSION,
        "run_id": "exp102_test",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_identity": source_identity,
        "registry_file_sha256": workflow._registry_sha(registry_path),
        "config_file_sha256": workflow._config_sha(config_path),
        "clock_authority": clock_authority,
        "started_unix": started,
        "started_boottime_ns": started_boottime_ns,
        **workflow._schedule_deadlines(
            config, started, started_boottime_ns,
        ),
    }
    schedule = {**identity, "schedule_sha256": workflow._sha256_json(identity)}
    path = tmp_path / "schedule.json"
    path.write_text(workflow._canonical_json(schedule) + "\n", encoding="ascii")
    assert workflow._validate_schedule(
        path, registry_path, config_path, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, config,
    )[0] == schedule
    monkeypatch.setattr(workflow.time, "time", lambda: 1e30)
    assert workflow._validate_schedule(
        path, registry_path, config_path, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, config,
    )[0] == schedule

    future = dict(schedule)
    future["started_unix"] += 3600
    path.write_text(workflow._canonical_json(future) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="start identity"):
        workflow._validate_schedule(
            path, registry_path, config_path, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, config,
        )


def test_remote_provenance_rejects_git_mode(monkeypatch):
    monkeypatch.setattr(
        workflow, "_verify_source",
        lambda commit: {"source_commit": commit, "mode": "git"},
    )
    with pytest.raises(ValueError, match="verified source archive"):
        workflow._verify_provenance(
            SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256,
        )


def test_wrapper_rejects_action_token_spoof_and_help_marker(tmp_path):
    wrapper = (
        EXP102_ROOT
        / "validation/013_q0_hgp_global_screen_20260722/run_hgp_wrapper.sh"
    )
    environment = os.environ.copy()
    environment["EXP102_SOURCE_COMMIT"] = SOURCE_COMMIT
    spoof_stage = tmp_path / "spoof-stage"
    spoof = subprocess.run(
        [
            "bash", str(wrapper), "freeze-control", str(spoof_stage),
            str(tmp_path / "spoof.log"), "--", "/usr/bin/true",
            "build-control", "unused", "unused2",
        ],
        check=False, capture_output=True, text=True, env=environment,
    )
    assert spoof.returncode == 68
    assert not (spoof_stage / "SUCCESS").exists()

    help_stage = tmp_path / "help-stage"
    help_result = subprocess.run(
        [
            "bash", str(wrapper), "build-schedule", str(help_stage),
            str(tmp_path / "help.log"), "--", "python", "-m",
            workflow.__name__, "build-schedule", "--help",
        ],
        check=False, capture_output=True, text=True, env=environment,
    )
    assert help_result.returncode == 68
    assert not (help_stage / "SUCCESS").exists()


def test_wrapper_build_schedule_supports_empty_prerequisites_with_nounset(
    tmp_path,
):
    wrapper = (
        EXP102_ROOT
        / "validation/013_q0_hgp_global_screen_20260722/run_hgp_wrapper.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == -c ]]; then\n"
        "  exec \"$REAL_PYTHON\" \"$@\"\n"
        "fi\n"
        "exit 0\n",
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    fake_flock.chmod(0o755)
    environment = os.environ.copy()
    environment["EXP102_SOURCE_COMMIT"] = SOURCE_COMMIT
    environment["REAL_PYTHON"] = sys.executable
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    stage_dir = tmp_path / "schedule-stage"
    log_file = tmp_path / "schedule.log"

    completed = subprocess.run(
        [
            "bash", str(wrapper), "build-schedule", str(stage_dir),
            str(log_file), "--", "python", "-m", workflow.__name__,
            "build-schedule",
        ],
        check=False, capture_output=True, text=True, env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    success = json.loads(
        (stage_dir / "SUCCESS").read_text(encoding="ascii")
    )
    assert success["stage"] == "build-schedule"
    assert success["prerequisite_success_sha256"] == []
    stage = orchestrator._Stage(
        key="schedule", node="nd-1", stage="build-schedule",
        workflow_argv=("build-schedule",), stage_dir=stage_dir,
        log_file=log_file, bootstrap_log=tmp_path / "bootstrap.log",
        prerequisites=(), session="test",
    )
    assert orchestrator._validate_stage_success(stage, SOURCE_COMMIT) == success
    assert not (stage_dir / "RUNNING").exists()
