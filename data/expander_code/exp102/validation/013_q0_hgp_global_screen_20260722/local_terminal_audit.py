"""Lightweight macmini audit of a pulled HGP terminal run.

This audit never consumes a sampler RNG stream.  It first validates the joint
nd-0 terminal acceptance, then independently rehashes and algebraically checks
all 384 measurement files and both auxiliary IS files through the workflow's
stored-evidence validators.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Mapping


VALIDATION_ROOT = (
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722"
)
workflow = importlib.import_module(VALIDATION_ROOT + ".workflow")
orchestration = importlib.import_module(VALIDATION_ROOT + ".orchestrate_hgp")

ATTESTATION_VERSION = (
    "exp102.q0_hgp_global.screen.local_terminal_attestation.v1"
)
MEASUREMENT_COUNT = 384
IMPORTANCE_SAMPLING_COUNT = 2
MEASUREMENT_EVIDENCE_FIELDS = (
    "full_transcript_sha256", "portable_transcript_sha256",
    "nonportable_float_sha256", "field_manifest_sha256",
    "acceptance_decision_sha256",
)
IS_EVIDENCE_FIELDS = MEASUREMENT_EVIDENCE_FIELDS[:-1]
SHA1_CHARS = frozenset("0123456789abcdef")


def _canonical_json(value):
    return workflow._canonical_json(value)


def _sha256_json(value):
    return workflow._sha256_json(value)


def _sha256_file(path):
    return workflow._sha256_file(path)


def _is_sha(value, length):
    return (
        isinstance(value, str) and len(value) == length
        and all(character in SHA1_CHARS for character in value)
    )


def _read_canonical_json(path, label):
    try:
        text = Path(path).read_text(encoding="ascii")
        value = json.loads(text)
    except Exception as exc:
        raise ValueError(f"{label} is not readable canonical JSON") from exc
    if text != _canonical_json(value) + "\n":
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular non-symlink file")
    return path.resolve(strict=True)


def _require_in_root(path, root, label):
    root = Path(root).resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped its root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink")
    resolved = _require_regular_file(lexical, label)
    if resolved != root / relative:
        raise ValueError(f"{label} changed after path resolution")
    return resolved


def _prepare_fresh_work(work_root, output_path):
    work_root = Path(os.path.abspath(work_root))
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError("local terminal audit work root already exists")
    parent = work_root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("local terminal audit work parent is invalid")
    os.mkdir(work_root, 0o755)
    output = (
        work_root / "local_terminal_attestation.json"
        if output_path is None else Path(os.path.abspath(output_path))
    )
    if output.parent != work_root or output.exists() or output.is_symlink():
        raise ValueError("local terminal audit output must be fresh inside work root")
    return work_root, output


def _write_exclusive_json(path, value):
    payload = (_canonical_json(value) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_claim(path, record, manifest_sha256, package_row):
    claim = _read_canonical_json(path, "local terminal raw claim")
    claimed_unix = claim.get("claimed_unix")
    pid = claim.get("pid")
    if (set(claim) != {
            "contract_version", "kind", "fingerprint",
            "manifest_sha256", "node", "pid", "claimed_unix"}
            or claim.get("contract_version") != workflow.CONTRACT_VERSION
            or claim.get("kind") != record["kind"]
            or claim.get("fingerprint") != record["fingerprint"]
            or claim.get("manifest_sha256") != manifest_sha256
            or claim.get("node") != record["owner"]
            or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or isinstance(claimed_unix, bool)
            or not isinstance(claimed_unix, (int, float))
            or not math.isfinite(float(claimed_unix))
            or package_row.get("claim_sha256") != _sha256_file(path)):
        raise ValueError("local terminal raw claim identity changed")


def _raw_evidence_summary(measurement_rows, is_rows):
    measurement = sorted((
        {
            "task_fingerprint": row["fingerprint"],
            **{name: row[name] for name in MEASUREMENT_EVIDENCE_FIELDS},
        }
        for row in measurement_rows
    ), key=lambda value: value["task_fingerprint"])
    measurement_portable = [{
        "task_fingerprint": value["task_fingerprint"],
        "portable_transcript_sha256": value["portable_transcript_sha256"],
        "field_manifest_sha256": value["field_manifest_sha256"],
        "acceptance_decision_sha256": value[
            "acceptance_decision_sha256"
        ],
    } for value in measurement]
    decisions = [{
        "task_fingerprint": value["task_fingerprint"],
        "acceptance_decision_sha256": value[
            "acceptance_decision_sha256"
        ],
    } for value in measurement]
    importance = sorted((
        {
            "is_fingerprint": row["fingerprint"],
            **{name: row[name] for name in IS_EVIDENCE_FIELDS},
        }
        for row in is_rows
    ), key=lambda value: value["is_fingerprint"])
    importance_portable = [{
        "is_fingerprint": value["is_fingerprint"],
        "portable_transcript_sha256": value["portable_transcript_sha256"],
        "field_manifest_sha256": value["field_manifest_sha256"],
    } for value in importance]
    return {
        "measurement_full_evidence_sha256": _sha256_json(measurement),
        "measurement_portable_evidence_sha256": _sha256_json(
            measurement_portable,
        ),
        "acceptance_decision_catalog_sha256": _sha256_json(decisions),
        "importance_sampling_full_evidence_sha256": _sha256_json(importance),
        "importance_sampling_portable_evidence_sha256": _sha256_json(
            importance_portable,
        ),
    }


def _fixed_run_files(run_root):
    control_root = run_root / "control"
    return {
        "schedule": control_root / "HGP_GLOBAL_24H_SCHEDULE.json",
        "artifact_manifest": control_root / "hgp_artifacts.json",
        "control": control_root / "hgp_measurement_control.json",
        "package": control_root / "hgp_terminal_package.json",
        "measurement_acceptance": (
            control_root / "HGP_ND0_MEASUREMENT_ACCEPTANCE.json"
        ),
        "preflight_acceptance": (
            control_root / "HGP_ND0_PREFLIGHT_ACCEPTANCE.json"
        ),
    }


def audit_terminal_run(
        run_root, registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, work_root, output_path=None):
    if not _is_sha(source_commit, 40):
        raise ValueError("local terminal source commit is invalid")
    if not _is_sha(archive_sha256, 64) or not _is_sha(
            source_manifest_sha256, 64):
        raise ValueError("local terminal source archive identity is invalid")
    lexical_run_root = Path(os.path.abspath(run_root))
    if (not lexical_run_root.is_dir() or lexical_run_root.is_symlink()
            or lexical_run_root.resolve(strict=True) != lexical_run_root):
        raise ValueError("local terminal run root is invalid")
    run_root = lexical_run_root
    registry_path = _require_regular_file(registry_path, "local registry")
    config_path = _require_regular_file(config_path, "local HGP config")
    work_root, output = _prepare_fresh_work(work_root, output_path)

    files = _fixed_run_files(run_root)
    files = {
        name: _require_in_root(path, run_root, f"local terminal {name}")
        for name, path in files.items()
    }
    joint = orchestration.validate_measurement_acceptance_offline(
        files["measurement_acceptance"], run_root, registry_path, config_path,
        source_commit, archive_sha256, source_manifest_sha256,
    )
    if (joint.get("formal_authorization") is not False
            or joint.get("production_authorization") is not False):
        raise ValueError("joint terminal audit granted unauthorized production")

    schedule = _read_canonical_json(files["schedule"], "local schedule")
    artifact_manifest = _read_canonical_json(
        files["artifact_manifest"], "local artifact manifest",
    )
    control = _read_canonical_json(files["control"], "local task manifest")
    package = _read_canonical_json(files["package"], "local terminal package")
    measurement_acceptance = _read_canonical_json(
        files["measurement_acceptance"], "local measurement acceptance",
    )
    preflight_acceptance = _read_canonical_json(
        files["preflight_acceptance"], "local preflight acceptance",
    )
    registry_file_sha256 = _sha256_file(registry_path)
    config_file_sha256 = _sha256_file(config_path)
    if (schedule.get("source_commit") != source_commit
            or schedule.get("archive_sha256") != archive_sha256
            or schedule.get("source_manifest_sha256")
            != source_manifest_sha256
            or schedule.get("registry_file_sha256") != registry_file_sha256
            or schedule.get("config_file_sha256") != config_file_sha256
            or control.get("source_commit") != source_commit
            or control.get("archive_sha256") != archive_sha256
            or control.get("source_manifest_sha256")
            != source_manifest_sha256
            or package.get("status") != joint.get("status")):
        raise ValueError("local terminal source/control identity changed")
    for value, field, label in (
            (schedule, "schedule_sha256", "schedule"),
            (artifact_manifest, "artifact_manifest_sha256", "artifact"),
            (control, "manifest_sha256", "control"),
            (package, "package_sha256", "package"),
            (measurement_acceptance, "manifest_sha256", "measurement acceptance"),
            (preflight_acceptance, "manifest_sha256", "preflight acceptance")):
        identity = dict(value)
        stored = identity.pop(field, None)
        if stored != _sha256_json(identity):
            raise ValueError(f"local terminal {label} self-hash changed")

    registry = workflow._load_registry(registry_path)
    config = workflow._load_config(config_path, registry)
    measurement_records = workflow._manifest_records(control)
    is_records = workflow._importance_records(control, config)
    if (len(measurement_records) != MEASUREMENT_COUNT
            or len(is_records) != IMPORTANCE_SAMPLING_COUNT
            or control.get("task_count") != MEASUREMENT_COUNT):
        raise ValueError("local terminal frozen task counts changed")
    expected = [{
        "kind": "measurement",
        "fingerprint": record["task_fingerprint"],
        "output_relpath": record["output_relpath"],
        "owner": record["owner"],
        "claim_relpath": f".claims/{record['task_fingerprint']}.json",
        "record": record,
    } for record in measurement_records] + [{
        "kind": "importance_sampling",
        "fingerprint": record["is_fingerprint"],
        "output_relpath": record["output_relpath"],
        "owner": record["owner"],
        "claim_relpath": f".claims_is/{record['is_fingerprint']}.json",
        "record": record,
    } for record in is_records]
    expected.sort(key=lambda value: (value["kind"], value["fingerprint"]))
    package_rows = package.get("raw_files")
    if (not isinstance(package_rows, list)
            or package.get("raw_file_count") != len(expected)
            or len(package_rows) != MEASUREMENT_COUNT + IMPORTANCE_SAMPLING_COUNT):
        raise ValueError("local terminal package raw count changed")

    raw_root = run_root / "hgp_global/raw"
    if (not raw_root.is_dir() or raw_root.is_symlink()
            or raw_root.resolve(strict=True) != raw_root):
        raise ValueError("local terminal raw root is invalid")
    artifact_root = run_root / "hgp_global/artifacts"
    if (not artifact_root.is_dir() or artifact_root.is_symlink()
            or artifact_root.resolve(strict=True) != artifact_root):
        raise ValueError("local terminal artifact root is invalid")
    ordered_evidence = []
    measurement_rows = []
    is_rows = []
    expected_files = set()
    for expected_row, package_row in zip(expected, package_rows):
        if not isinstance(package_row, Mapping):
            raise ValueError("local terminal package raw row is invalid")
        base_fields = {
            "kind", "fingerprint", "output_relpath", "sha256", "claim_sha256",
        }
        evidence_fields = (
            MEASUREMENT_EVIDENCE_FIELDS
            if expected_row["kind"] == "measurement" else IS_EVIDENCE_FIELDS
        )
        if (set(package_row) != base_fields | set(evidence_fields)
                or any(package_row.get(name) != expected_row[name]
                       for name in ("kind", "fingerprint", "output_relpath"))
                or any(not _is_sha(package_row.get(name), 64)
                       for name in ("sha256", "claim_sha256", *evidence_fields))):
            raise ValueError("local terminal package raw row changed")
        relative = workflow._safe_relative_path(
            expected_row["output_relpath"], field="terminal raw path",
        )
        raw_path = _require_in_root(
            raw_root / relative, raw_root, "local terminal raw",
        )
        claim_path = _require_in_root(
            raw_root / expected_row["claim_relpath"], raw_root,
            "local terminal claim",
        )
        expected_files.update({
            raw_path.relative_to(raw_root).as_posix(),
            claim_path.relative_to(raw_root).as_posix(),
        })
        if package_row["sha256"] != _sha256_file(raw_path):
            raise ValueError("local terminal raw file SHA changed")
        _validate_claim(
            claim_path, expected_row, control["manifest_sha256"], package_row,
        )
        if expected_row["kind"] == "measurement":
            task = workflow._task_payload(control, expected_row["record"])
            validated = workflow._validate_staging_measurement(
                raw_path, registry_path, config, task, source_commit,
                archive_sha256, source_manifest_sha256, artifact_root,
            )
            measurement_rows.append(dict(package_row))
        else:
            validated = workflow._validate_staging_is(
                raw_path, registry_path, config, expected_row["record"],
                source_commit, archive_sha256, source_manifest_sha256,
                artifact_root,
            )
            is_rows.append(dict(package_row))
        if any(validated.get(name) != package_row[name]
               for name in evidence_fields):
            raise ValueError("local terminal stored transcript evidence changed")
        evidence_identity = {
            "kind": package_row["kind"],
            "fingerprint": package_row["fingerprint"],
            "output_relpath": package_row["output_relpath"],
            "raw_file_sha256": package_row["sha256"],
            "claim_file_sha256": package_row["claim_sha256"],
            **{name: package_row[name] for name in evidence_fields},
        }
        ordered_evidence.append({
            "kind": package_row["kind"],
            "fingerprint": package_row["fingerprint"],
            "evidence_sha256": _sha256_json(evidence_identity),
        })

    actual_files = set()
    for path in raw_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("local terminal raw tree contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(raw_root).as_posix())
        elif not path.is_dir():
            raise ValueError("local terminal raw tree contains a special file")
    if actual_files != expected_files:
        raise ValueError("local terminal raw tree has missing or extra files")
    raw_evidence_summary = _raw_evidence_summary(measurement_rows, is_rows)
    if package.get("raw_evidence_summary") != raw_evidence_summary:
        raise ValueError("local terminal package evidence summary changed")

    identity = {
        "attestation_version": ATTESTATION_VERSION,
        "status": "PASS",
        "run_id": schedule.get("run_id"),
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_file_sha256": registry_file_sha256,
        "config_file_sha256": config_file_sha256,
        "schedule_file_sha256": _sha256_file(files["schedule"]),
        "schedule_sha256": schedule["schedule_sha256"],
        "artifact_manifest_file_sha256": _sha256_file(
            files["artifact_manifest"],
        ),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "control_file_sha256": _sha256_file(files["control"]),
        "control_manifest_sha256": control["manifest_sha256"],
        "terminal_package_file_sha256": _sha256_file(files["package"]),
        "terminal_package_sha256": package["package_sha256"],
        "preflight_acceptance_manifest_file_sha256": _sha256_file(
            files["preflight_acceptance"],
        ),
        "preflight_acceptance_manifest_sha256": preflight_acceptance[
            "manifest_sha256"
        ],
        "measurement_acceptance_manifest_file_sha256": _sha256_file(
            files["measurement_acceptance"],
        ),
        "measurement_acceptance_manifest_sha256": measurement_acceptance[
            "manifest_sha256"
        ],
        "joint_terminal_sha256": joint.get("joint_terminal_sha256"),
        "terminal_status": package.get("status"),
        "raw_evidence_summary": raw_evidence_summary,
        "ordered_evidence_count": len(ordered_evidence),
        "ordered_evidence": ordered_evidence,
        "ordered_evidence_catalog_sha256": _sha256_json(ordered_evidence),
        "validated_measurement_count": len(measurement_rows),
        "validated_importance_sampling_count": len(is_rows),
        "validated_raw_count": len(ordered_evidence),
        "formal_authorization": False,
        "production_authorization": False,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    if (not _is_sha(identity["joint_terminal_sha256"], 64)
            or identity["ordered_evidence_count"]
            != MEASUREMENT_COUNT + IMPORTANCE_SAMPLING_COUNT
            or identity["validated_measurement_count"] != MEASUREMENT_COUNT
            or identity["validated_importance_sampling_count"]
            != IMPORTANCE_SAMPLING_COUNT):
        raise ValueError("local terminal validated evidence count changed")
    attestation = {
        **identity, "attestation_sha256": _sha256_json(identity),
    }
    _write_exclusive_json(output, attestation)
    return attestation


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--registry", default=workflow.DEFAULT_REGISTRY)
    parser.add_argument("--config", default=workflow.DEFAULT_CONFIG)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    attestation = audit_terminal_run(
        args.run_root, args.registry, args.config, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, args.work_root,
        args.output,
    )
    print(_canonical_json({
        "status": attestation["status"],
        "attestation_sha256": attestation["attestation_sha256"],
        "validated_raw_count": attestation["validated_raw_count"],
    }))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
