import importlib
import json
from pathlib import Path

import pytest


VALIDATION_ROOT = (
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722"
)
audit = importlib.import_module(VALIDATION_ROOT + ".local_terminal_audit")
workflow = importlib.import_module(VALIDATION_ROOT + ".workflow")

SOURCE_COMMIT = "1" * 40
ARCHIVE_SHA256 = "2" * 64
SOURCE_MANIFEST_SHA256 = "3" * 64


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(workflow._canonical_json(value) + "\n", encoding="ascii")


def _sealed(identity, field):
    return {**identity, field: workflow._sha256_json(identity)}


def _fixture_run(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    control_root = run_root / "control"
    raw_root = run_root / "hgp_global/raw"
    artifact_root = run_root / "hgp_global/artifacts"
    artifact_root.mkdir(parents=True)
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config.json"
    registry_path.write_text("registry\n", encoding="ascii")
    config_path.write_text("config\n", encoding="ascii")

    schedule_identity = {
        "run_id": "exp102_terminal_test",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "registry_file_sha256": workflow._sha256_file(registry_path),
        "config_file_sha256": workflow._sha256_file(config_path),
    }
    schedule = _sealed(schedule_identity, "schedule_sha256")
    artifact_manifest = _sealed(
        {"artifact_count": 2}, "artifact_manifest_sha256",
    )
    cells = [
        {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0},
        {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0},
    ]
    descriptors = [
        {"cell_fingerprint": str(index) * 64}
        for index in (4, 5)
    ]
    tasks = []
    for index in range(audit.MEASUREMENT_COUNT):
        cell_index = index % 2
        task = {
            "index": index,
            "cell": cells[cell_index],
            "method_id": "MAM-IMH8" if index < 64 else "HP32",
        }
        if index < 64:
            task["map_artifact"] = descriptors[cell_index]
        fingerprint = workflow._sha256_json(task)
        tasks.append({
            "task": task,
            "task_fingerprint": fingerprint,
            "output_relpath": f"trajectories/{fingerprint}.npz",
            "owner": "nd-2" if index % 2 == 0 else "nd-3",
        })
    control_identity = {
        "contract_version": workflow.CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "execution_nodes": ["nd-2", "nd-3"],
        "task_count": audit.MEASUREMENT_COUNT,
        "tasks": tasks,
        "importance_sampling": {
            "outputs": [
                "importance_sampling/a.npz",
                "importance_sampling/b.npz",
            ],
            "raw_version": "is.v2",
            "num_samples_per_cell": 2,
            "used_for_gate_or_selection": False,
        },
    }
    control = _sealed(control_identity, "manifest_sha256")
    config = {"importance_sampling": {"num_samples_per_cell": 2}}
    monkeypatch.setattr(workflow, "_load_registry", lambda path: {})
    monkeypatch.setattr(workflow, "_load_config", lambda path, registry: config)
    measurement_records = workflow._manifest_records(control)
    is_records = workflow._importance_records(control, config)

    measurement_evidence = {
        name: format(index + 6, "x") * 64
        for index, name in enumerate(audit.MEASUREMENT_EVIDENCE_FIELDS)
    }
    is_evidence = {
        name: format(index + 11, "x") * 64
        for index, name in enumerate(audit.IS_EVIDENCE_FIELDS)
    }
    monkeypatch.setattr(
        workflow, "_validate_staging_measurement",
        lambda *args, **kwargs: dict(measurement_evidence),
    )
    monkeypatch.setattr(
        workflow, "_validate_staging_is",
        lambda *args, **kwargs: dict(is_evidence),
    )
    expected = [{
        "kind": "measurement",
        "fingerprint": record["task_fingerprint"],
        "output_relpath": record["output_relpath"],
        "owner": record["owner"],
        "claim_relpath": f".claims/{record['task_fingerprint']}.json",
        "evidence": measurement_evidence,
    } for record in measurement_records] + [{
        "kind": "importance_sampling",
        "fingerprint": record["is_fingerprint"],
        "output_relpath": record["output_relpath"],
        "owner": record["owner"],
        "claim_relpath": f".claims_is/{record['is_fingerprint']}.json",
        "evidence": is_evidence,
    } for record in is_records]
    expected.sort(key=lambda value: (value["kind"], value["fingerprint"]))
    rows = []
    for record in expected:
        raw_path = raw_root / record["output_relpath"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(record["fingerprint"].encode("ascii"))
        claim_path = raw_root / record["claim_relpath"]
        claim = {
            "contract_version": workflow.CONTRACT_VERSION,
            "kind": record["kind"],
            "fingerprint": record["fingerprint"],
            "manifest_sha256": control["manifest_sha256"],
            "node": record["owner"],
            "pid": 123,
            "claimed_unix": 1.0,
        }
        _write_json(claim_path, claim)
        rows.append({
            "kind": record["kind"],
            "fingerprint": record["fingerprint"],
            "output_relpath": record["output_relpath"],
            "sha256": workflow._sha256_file(raw_path),
            "claim_sha256": workflow._sha256_file(claim_path),
            **record["evidence"],
        })
    measurement_rows = [row for row in rows if row["kind"] == "measurement"]
    is_rows = [row for row in rows if row["kind"] == "importance_sampling"]
    package_identity = {
        "status": "UNRESOLVED_NO_HP_PASS",
        "raw_file_count": len(rows),
        "raw_files": rows,
        "raw_evidence_summary": audit._raw_evidence_summary(
            measurement_rows, is_rows,
        ),
    }
    package = _sealed(package_identity, "package_sha256")
    preflight_acceptance = _sealed(
        {"phase": "preflight"}, "manifest_sha256",
    )
    measurement_acceptance = _sealed(
        {"phase": "measurement"}, "manifest_sha256",
    )
    _write_json(
        control_root / "HGP_GLOBAL_24H_SCHEDULE.json", schedule,
    )
    _write_json(control_root / "hgp_artifacts.json", artifact_manifest)
    _write_json(control_root / "hgp_measurement_control.json", control)
    _write_json(control_root / "hgp_terminal_package.json", package)
    _write_json(
        control_root / "HGP_ND0_PREFLIGHT_ACCEPTANCE.json",
        preflight_acceptance,
    )
    _write_json(
        control_root / "HGP_ND0_MEASUREMENT_ACCEPTANCE.json",
        measurement_acceptance,
    )
    calls = []

    def offline(*args):
        calls.append("offline")
        return {
            "status": package["status"],
            "joint_terminal_sha256": "f" * 64,
            "formal_authorization": False,
            "production_authorization": False,
        }

    monkeypatch.setattr(
        audit.orchestration, "validate_measurement_acceptance_offline", offline,
    )
    return {
        "run_root": run_root,
        "registry": registry_path,
        "config": config_path,
        "package_path": control_root / "hgp_terminal_package.json",
        "package": package,
        "rows": rows,
        "calls": calls,
    }


def _run_audit(fixture, work_root):
    return audit.audit_terminal_run(
        fixture["run_root"], fixture["registry"], fixture["config"],
        SOURCE_COMMIT, ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, work_root,
    )


def test_local_terminal_audit_validates_exact_386_file_catalog(
    tmp_path, monkeypatch,
):
    fixture = _fixture_run(tmp_path, monkeypatch)
    work_root = tmp_path / "audit"
    result = _run_audit(fixture, work_root)
    assert fixture["calls"] == ["offline"]
    assert result["status"] == "PASS"
    assert result["validated_measurement_count"] == 384
    assert result["validated_importance_sampling_count"] == 2
    assert result["validated_raw_count"] == 386
    assert len(result["ordered_evidence"]) == 386
    identity = dict(result)
    stored = identity.pop("attestation_sha256")
    assert stored == workflow._sha256_json(identity)
    output = work_root / "local_terminal_attestation.json"
    assert json.loads(output.read_text(encoding="ascii")) == result
    with pytest.raises(FileExistsError, match="work root already exists"):
        _run_audit(fixture, work_root)


def test_local_terminal_audit_rejects_summary_tamper_and_symlink(
    tmp_path, monkeypatch,
):
    fixture = _fixture_run(tmp_path, monkeypatch)
    package = dict(fixture["package"])
    rows = [dict(row) for row in package["raw_files"]]
    rows[0]["portable_transcript_sha256"] = "0" * 64
    package_identity = dict(package)
    package_identity.pop("package_sha256")
    package_identity["raw_files"] = rows
    package = _sealed(package_identity, "package_sha256")
    _write_json(fixture["package_path"], package)
    with pytest.raises(ValueError, match="stored transcript evidence"):
        _run_audit(fixture, tmp_path / "audit_tamper")

    fixture = _fixture_run(tmp_path / "second", monkeypatch)
    first = fixture["rows"][0]
    raw_path = fixture["run_root"] / "hgp_global/raw" / first["output_relpath"]
    target = tmp_path / "outside.npz"
    target.write_bytes(raw_path.read_bytes())
    raw_path.unlink()
    raw_path.symlink_to(target)
    with pytest.raises(ValueError, match="traverses a symlink"):
        _run_audit(fixture, tmp_path / "audit_symlink")
