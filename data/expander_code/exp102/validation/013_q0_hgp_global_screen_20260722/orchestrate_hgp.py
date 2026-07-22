"""Orchestrate the immutable 24-hour HGP diagnostic on nd-0.

This module never calls the scientific workflow directly.  Every workflow
action is launched in a fresh remote screen through the verified source
archive and ``run_hgp_wrapper.sh``.  The wrapper's immutable SUCCESS markers
form the only authority chain between stages.

The default ``preflight`` phase stops after aggregate runtime/digest consensus
so the frozen artifact can be replayed locally.  A separate explicit
``measurement`` phase revalidates that authority chain before freezing the
control and launching any sampler task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import tarfile
import time


WORKFLOW_MODULE = (
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.workflow"
)
VERIFY_RELATIVE = (
    "data/expander_code/exp102/validation/"
    "002_numba_smoke_20260719/run_verified_source.sh"
)
WRAPPER_RELATIVE = (
    "data/expander_code/exp102/validation/"
    "013_q0_hgp_global_screen_20260722/run_hgp_wrapper.sh"
)
CONFIG_RELATIVE = (
    "data/expander_code/exp102/config/q0_hgp_global.screen.v2.json"
)
REGISTRY_RELATIVE = "data/expander_code/exp102/registry/registry.json"

PREFLIGHT_NODES = ("nd-1", "nd-2", "nd-3")
EXECUTION_NODES = ("nd-2", "nd-3")
LOCAL_ATTESTATION_VERSION = "exp102.q0_hgp_global.screen.local_attestation.v4"
LOCAL_SOLVER_POLICY = (
    "stored_generation_identity_exact_artifact_replay_no_local_milp"
)
LOCAL_FULL_MISMATCH_POLICY = (
    "diagnostic_only_after_remote_full_consensus_and_exact_portable_decisions"
)
LOCAL_ATTESTATION_STATUSES = {"PASS_EXACT", "PORTABLE_PASS"}
LOCAL_FULL_CONSENSUS_NODES = ("nd-1", "nd-2", "nd-3")
LOCAL_IS_SUMMARY_KEY = "importance_sampling_transcript_summary"
LOCAL_IS_FULL_FIELDS = {
    "cell_fingerprint", "full_transcript_sha256",
    "portable_transcript_sha256", "nonportable_float_sha256",
    "field_manifest_sha256",
}
LOCAL_IS_PORTABLE_FIELDS = {
    "cell_fingerprint", "portable_transcript_sha256",
    "field_manifest_sha256",
}
LOCAL_DECISION_FIELDS = {
    "cell_fingerprint", "method_id", "init_family",
    "acceptance_decision_sha256",
}
CLOCK_AUTHORITY_VERSION = "exp102.q0_hgp.nd0_boottime.v1"
SCHEDULE_VERSION = "exp102.q0_hgp_global.screen.schedule.v2"
STAGE_ACCEPTANCE_VERSION = "exp102.q0_hgp.nd0_stage_acceptance.v1"
ACCEPTANCE_MANIFEST_VERSION = "exp102.q0_hgp.nd0_acceptance_manifest.v1"
JOINT_TERMINAL_VERSION = "exp102.q0_hgp.offline_joint_terminal.v1"
HGP_CONTRACT_VERSION = "exp102.q0_hgp_global.screen.v2"
TERMINAL_DECISION_VERSION = "exp102.q0_hgp_global.screen.decision.v2"
TERMINAL_PACKAGE_VERSION = "exp102.q0_hgp_global.screen.terminal_package.v2"
ND0_PERSISTENCE_TOKEN = "exp102_q0_hgp_nd0_nohup_setsid_v1"
ND0_LAUNCHER_VERSION = "exp102.q0_hgp.nd0_nohup_setsid.v1"
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_CLOCK_CAPTURE_SPAN_NS = NANOSECONDS_PER_SECOND
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=4",
)
STAGE_ACCEPTANCE_SPECS = {
    "00_schedule": ("preflight", "build-schedule", "nd-1",
                    "schedule_bootstrap_15m", 900),
    "01_artifacts": ("preflight", "build-artifacts", "nd-1",
                     "preflight_6h", 6 * 3600),
    "02_preflight_nd-1": ("preflight", "preflight", "nd-1",
                          "preflight_6h", 6 * 3600),
    "02_preflight_nd-2": ("preflight", "preflight", "nd-2",
                          "preflight_6h", 6 * 3600),
    "02_preflight_nd-3": ("preflight", "preflight", "nd-3",
                          "preflight_6h", 6 * 3600),
    "03_preflight_combine": ("preflight", "preflight", "nd-1",
                             "preflight_6h", 6 * 3600),
    "04_control": ("measurement", "freeze-control", "nd-1",
                   "control_freeze_8h", 8 * 3600),
    "05_screen_nd-2": ("measurement", "screen", "nd-2",
                       "screen_22h", 22 * 3600),
    "05_screen_nd-3": ("measurement", "screen", "nd-3",
                       "screen_22h", 22 * 3600),
    "06_analyze": ("measurement", "analyze", "nd-3",
                   "analysis_24h", 24 * 3600),
}
PHASE_STAGE_KEYS = {
    "preflight": (
        "00_schedule", "01_artifacts", "02_preflight_nd-1",
        "02_preflight_nd-2", "02_preflight_nd-3",
        "03_preflight_combine",
    ),
    "measurement": (
        "04_control", "05_screen_nd-2", "05_screen_nd-3", "06_analyze",
    ),
}
STAGE_PREREQUISITE_KEYS = {
    "00_schedule": (),
    "01_artifacts": ("00_schedule",),
    "02_preflight_nd-1": ("01_artifacts",),
    "02_preflight_nd-2": ("01_artifacts",),
    "02_preflight_nd-3": ("01_artifacts",),
    "03_preflight_combine": (
        "02_preflight_nd-1", "02_preflight_nd-2", "02_preflight_nd-3",
    ),
    "04_control": ("03_preflight_combine",),
    "05_screen_nd-2": ("04_control",),
    "05_screen_nd-3": ("04_control",),
    "06_analyze": ("05_screen_nd-2", "05_screen_nd-3"),
}
STAGE_ACCEPTANCE_FIELDS = {
    "acceptance_version", "run_id", "phase", "source_commit",
    "archive_sha256", "source_manifest_sha256",
    "clock_authority_sha256", "clock_authority_boot_id",
    "stage_key", "stage", "node", "observed_boottime_ns",
    "deadline_kind", "deadline_boottime_ns", "success_relpath",
    "success_file_sha256", "success_marker", "acceptance_sha256",
}
ACCEPTANCE_MANIFEST_FIELDS = {
    "manifest_version", "phase", "run_id", "source_commit",
    "archive_sha256", "source_manifest_sha256",
    "clock_authority_sha256", "clock_authority_boot_id",
    "phase_deadline_boottime_ns", "published_boottime_ns",
    "launch_metadata", "launch_metadata_file_sha256",
    "stage_acceptances", "bound_file_sha256", "bound_identity",
    "prior_manifest", "manifest_sha256",
}
LAUNCH_METADATA_FIELDS = {
    "archive_sha256", "command_sha256", "launcher_version",
    "local_attestation_sha256", "manifest_sha256", "phase",
    "run_id", "source_commit",
}
TERMINAL_PACKAGE_FIELDS = {
    "package_version", "contract_version", "source_identity",
    "schedule_sha256", "schedule_file_sha256",
    "artifact_manifest_file_sha256", "preflight_file_sha256",
    "control_file_sha256", "execution_report_file_sha256",
    "report_file_sha256", "decision_file_sha256", "decision_sha256",
    "raw_evidence_summary", "status", "raw_file_count", "raw_files",
    "formal_authorization", "production_authorization", "package_sha256",
}
TERMINAL_DECISION_FIELDS = {
    "decision_version", "contract_version", "source_commit",
    "archive_sha256", "source_manifest_sha256", "schedule_sha256",
    "schedule_file_sha256", "artifact_manifest_sha256",
    "artifact_manifest_file_sha256", "preflight_file_sha256",
    "control_file_sha256", "manifest_sha256", "report_sha256",
    "report_file_sha256", "status", "selected_pair",
    "raw_evidence_summary",
    "formal_authorization", "production_authorization", "decision_sha256",
}
TERMINAL_TRANSCRIPT_SUMMARY_FIELDS = (
    "full_transcript_sha256", "portable_transcript_sha256",
    "nonportable_float_sha256", "field_manifest_sha256",
)
TERMINAL_MEASUREMENT_SUMMARY_FIELDS = (
    *TERMINAL_TRANSCRIPT_SUMMARY_FIELDS, "acceptance_decision_sha256",
)
TERMINAL_RAW_BASE_FIELDS = {
    "kind", "fingerprint", "output_relpath", "sha256", "claim_sha256",
}
TERMINAL_RAW_EVIDENCE_SUMMARY_FIELDS = {
    "measurement_full_evidence_sha256",
    "measurement_portable_evidence_sha256",
    "acceptance_decision_catalog_sha256",
    "importance_sampling_full_evidence_sha256",
    "importance_sampling_portable_evidence_sha256",
}
TERMINAL_STATUSES = {
    "DIAGNOSTIC_HARD_PAIR_FOUND", "UNRESOLVED_NO_HP_PASS",
    "UNRESOLVED_MAP_MIXTURE_FAIL",
    "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT",
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value):
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _read_canonical_json(path, label):
    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid canonical JSON") from exc
    expected = (_canonical_json(value) + "\n").encode("ascii")
    if payload != expected:
        raise ValueError(f"{label} bytes are noncanonical")
    return value


def _write_exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(value) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _require_regular_file_in_root(path, root, label):
    """Reject both leaf and parent symlinks in a pulled evidence tree."""
    lexical_root = Path(root).absolute()
    if lexical_root.is_symlink():
        raise ValueError(f"{label} cannot contain a symlink")
    root = lexical_root.resolve(strict=True)
    raw_path = Path(path).absolute()
    if raw_path.is_symlink():
        raise ValueError(f"{label} cannot contain a symlink")
    try:
        relative = raw_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the HGP run root") from exc
    if ".." in relative.parts:
        raise ValueError(f"{label} escaped the HGP run root")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} cannot contain a symlink")
    if not current.is_file():
        raise ValueError(f"{label} is not a regular file")
    return current


def _terminal_control_records(control, config):
    """Reconstruct the exact raw/claim identities from the frozen control."""
    identity = dict(control)
    stored_manifest_sha = identity.pop("manifest_sha256", None)
    if stored_manifest_sha != hashlib.sha256(
            _canonical_json(identity).encode("ascii")
    ).hexdigest():
        raise ValueError("HGP terminal control self-hash changed")

    execution_nodes = tuple(control.get("execution_nodes", ()))
    tasks = control.get("tasks")
    configured_count = config.get("task_counts", {}).get(
        "total_measurement"
    )
    if (execution_nodes != EXECUTION_NODES
            or isinstance(configured_count, bool)
            or configured_count != 384
            or not isinstance(tasks, list)
            or len(tasks) != configured_count
            or control.get("task_count") != configured_count):
        raise ValueError("HGP terminal control task set changed")

    measurement = []
    seen_fingerprints = set()
    seen_outputs = set()
    map_cells = []
    map_descriptors = {}
    for entry in tasks:
        if (not isinstance(entry, dict) or set(entry) != {
                "task", "task_fingerprint", "output_relpath", "owner",
        } or not isinstance(entry.get("task"), dict)):
            raise ValueError("HGP terminal control task schema changed")
        task = entry["task"]
        fingerprint = entry.get("task_fingerprint")
        expected_fingerprint = hashlib.sha256(
            _canonical_json(task).encode("ascii")
        ).hexdigest()
        expected_output = f"trajectories/{expected_fingerprint}.npz"
        if (fingerprint != expected_fingerprint
                or entry.get("output_relpath") != expected_output
                or entry.get("owner") not in EXECUTION_NODES
                or fingerprint in seen_fingerprints
                or expected_output in seen_outputs):
            raise ValueError("HGP terminal control task identity changed")
        seen_fingerprints.add(fingerprint)
        seen_outputs.add(expected_output)
        measurement.append({
            "kind": "measurement",
            "fingerprint": fingerprint,
            "output_relpath": expected_output,
            "owner": entry["owner"],
            "claim_relpath": f".claims/{fingerprint}.json",
        })

        descriptor = task.get("map_artifact")
        if descriptor is not None:
            cell = task.get("cell")
            if not isinstance(descriptor, dict) or not isinstance(cell, dict):
                raise ValueError("HGP terminal MAP task identity changed")
            cell_key = _canonical_json(cell)
            if cell_key not in map_descriptors:
                map_cells.append(cell)
                map_descriptors[cell_key] = descriptor
            elif map_descriptors[cell_key] != descriptor:
                raise ValueError("HGP terminal MAP artifact identity changed")

    spec = control.get("importance_sampling")
    configured_is = config.get("importance_sampling", {})
    if not isinstance(spec, dict) or set(spec) != {
            "raw_version", "num_samples_per_cell", "seed_namespace",
            "used_for_gate_or_selection", "outputs",
    }:
        raise ValueError("HGP terminal IS control schema changed")
    outputs = spec.get("outputs")
    if (len(map_cells) != 2 or not isinstance(outputs, list)
            or len(outputs) != 2
            or spec.get("num_samples_per_cell")
            != configured_is.get("num_samples_per_cell")
            or spec.get("used_for_gate_or_selection") is not False):
        raise ValueError("HGP terminal IS control set changed")

    importance_sampling = []
    for index, (cell, output) in enumerate(zip(map_cells, outputs)):
        cell_fingerprint = hashlib.sha256(
            _canonical_json(cell).encode("ascii")
        ).hexdigest()
        expected_output = f"importance_sampling/{cell_fingerprint}.npz"
        if output != expected_output or expected_output in seen_outputs:
            raise ValueError("HGP terminal IS output identity changed")
        is_identity = {
            "contract_version": HGP_CONTRACT_VERSION,
            "manifest_sha256": stored_manifest_sha,
            "archive_sha256": control.get("archive_sha256"),
            "source_manifest_sha256": control.get(
                "source_manifest_sha256"
            ),
            "cell": cell,
            "output_relpath": expected_output,
            "raw_version": spec.get("raw_version"),
            "num_samples": spec.get("num_samples_per_cell"),
        }
        fingerprint = hashlib.sha256(
            _canonical_json(is_identity).encode("ascii")
        ).hexdigest()
        seen_outputs.add(expected_output)
        importance_sampling.append({
            "kind": "importance_sampling",
            "fingerprint": fingerprint,
            "output_relpath": expected_output,
            "owner": execution_nodes[index % len(execution_nodes)],
            "claim_relpath": f".claims_is/{fingerprint}.json",
        })
    return measurement + importance_sampling


def _terminal_raw_evidence_summary(rows):
    """Rebuild the exact workflow-v2 transcript catalogs from package rows."""
    if not isinstance(rows, list):
        raise ValueError("HGP terminal raw evidence rows are invalid")
    measurement = []
    importance = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("HGP terminal raw evidence row is invalid")
        kind = row.get("kind")
        if kind == "measurement":
            evidence_fields = TERMINAL_MEASUREMENT_SUMMARY_FIELDS
        elif kind == "importance_sampling":
            evidence_fields = TERMINAL_TRANSCRIPT_SUMMARY_FIELDS
        else:
            raise ValueError("HGP terminal raw evidence kind changed")
        if (set(row) != TERMINAL_RAW_BASE_FIELDS | set(evidence_fields)
                or SHA256_RE.fullmatch(str(
                    row.get("fingerprint", ""),
                )) is None
                or not isinstance(row.get("output_relpath"), str)
                or not row["output_relpath"]
                or any(SHA256_RE.fullmatch(str(row.get(name, ""))) is None
                       for name in (
                           "sha256", "claim_sha256", *evidence_fields,
                       ))):
            raise ValueError("HGP terminal raw evidence row schema changed")
        if kind == "measurement":
            measurement.append({
                "task_fingerprint": row["fingerprint"],
                **{name: row[name] for name in evidence_fields},
            })
        else:
            importance.append({
                "is_fingerprint": row["fingerprint"],
                **{name: row[name] for name in evidence_fields},
            })

    measurement.sort(key=lambda value: value["task_fingerprint"])
    importance.sort(key=lambda value: value["is_fingerprint"])
    measurement_portable = [{
        "task_fingerprint": value["task_fingerprint"],
        "portable_transcript_sha256": value[
            "portable_transcript_sha256"
        ],
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
    importance_portable = [{
        "is_fingerprint": value["is_fingerprint"],
        "portable_transcript_sha256": value[
            "portable_transcript_sha256"
        ],
        "field_manifest_sha256": value["field_manifest_sha256"],
    } for value in importance]
    return {
        "measurement_full_evidence_sha256": _sha256_json(measurement),
        "measurement_portable_evidence_sha256": _sha256_json(
            measurement_portable,
        ),
        "acceptance_decision_catalog_sha256": _sha256_json(decisions),
        "importance_sampling_full_evidence_sha256": _sha256_json(
            importance,
        ),
        "importance_sampling_portable_evidence_sha256": _sha256_json(
            importance_portable,
        ),
    }


def _validate_terminal_raw_evidence(run_root, control, config, package):
    """Bind the terminal package to every frozen raw and claim byte."""
    run_root = Path(run_root).resolve(strict=True)
    raw_root = run_root / "hgp_global/raw"
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("HGP terminal raw root is absent or is a symlink")

    expected = _terminal_control_records(control, config)
    expected.sort(key=lambda value: (value["kind"], value["fingerprint"]))
    rows = package.get("raw_files")
    if (not isinstance(rows, list)
            or package.get("raw_file_count") != len(expected)
            or len(rows) != len(expected)):
        raise ValueError("HGP terminal package raw set changed")
    raw_evidence_summary = _terminal_raw_evidence_summary(rows)
    if package.get("raw_evidence_summary") != raw_evidence_summary:
        raise ValueError("HGP terminal package raw evidence summary changed")
    row_identities = [
        (row.get("kind"), row.get("fingerprint"), row.get("output_relpath"))
        if isinstance(row, dict) else None
        for row in rows
    ]
    expected_identities = [
        (record["kind"], record["fingerprint"], record["output_relpath"])
        for record in expected
    ]
    if row_identities != expected_identities:
        raise ValueError("HGP terminal package raw identity/order changed")

    expected_files = set()
    for record, row in zip(expected, rows):
        raw_path = raw_root / record["output_relpath"]
        claim_path = raw_root / record["claim_relpath"]
        raw_path = _require_regular_file_in_root(
            raw_path, run_root, "HGP terminal raw file",
        )
        claim_path = _require_regular_file_in_root(
            claim_path, run_root, "HGP terminal raw claim",
        )
        expected_files.update({
            raw_path.relative_to(raw_root).as_posix(),
            claim_path.relative_to(raw_root).as_posix(),
        })
        if (row.get("sha256") != _sha256_file(raw_path)
                or row.get("claim_sha256") != _sha256_file(claim_path)):
            raise ValueError("HGP terminal raw or claim SHA changed")
        claim = _read_canonical_json(claim_path, "HGP terminal raw claim")
        claimed_unix = claim.get("claimed_unix")
        pid = claim.get("pid")
        if (set(claim) != {
                "contract_version", "kind", "fingerprint",
                "manifest_sha256", "node", "pid", "claimed_unix",
        } or claim.get("contract_version") != HGP_CONTRACT_VERSION
                or claim.get("kind") != record["kind"]
                or claim.get("fingerprint") != record["fingerprint"]
                or claim.get("manifest_sha256")
                != control.get("manifest_sha256")
                or claim.get("node") != record["owner"]
                or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or isinstance(claimed_unix, bool)
                or not isinstance(claimed_unix, (int, float))
                or not math.isfinite(float(claimed_unix))):
            raise ValueError("HGP terminal raw claim identity changed")

    actual_files = set()
    for path in raw_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("HGP terminal raw tree contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(raw_root).as_posix())
        elif not path.is_dir():
            raise ValueError("HGP terminal raw tree contains a special file")
    if actual_files != expected_files:
        raise ValueError("HGP terminal raw/claim file set changed")
    return {
        "measurement_count": 384,
        "importance_sampling_count": 2,
        "raw_file_count": len(expected),
        "raw_evidence_summary": raw_evidence_summary,
    }


def _joint_terminal_result(
        schedule, package, preflight_manifest_path, preflight_manifest,
        measurement_manifest_path, measurement_manifest):
    identity = {
        "joint_terminal_version": JOINT_TERMINAL_VERSION,
        "run_id": schedule["run_id"],
        "source_commit": schedule["source_commit"],
        "archive_sha256": schedule["archive_sha256"],
        "source_manifest_sha256": schedule["source_manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "status": package["status"],
        "terminal_package_sha256": package["package_sha256"],
        "terminal_package_file_sha256": _sha256_file(
            Path(measurement_manifest_path).parent
            / "hgp_terminal_package.json"
        ),
        "preflight_acceptance_manifest_sha256": preflight_manifest[
            "manifest_sha256"
        ],
        "preflight_acceptance_manifest_file_sha256": _sha256_file(
            preflight_manifest_path
        ),
        "measurement_acceptance_manifest_sha256": measurement_manifest[
            "manifest_sha256"
        ],
        "measurement_acceptance_manifest_file_sha256": _sha256_file(
            measurement_manifest_path
        ),
        "formal_authorization": False,
        "production_authorization": False,
    }
    return {
        **identity,
        "joint_terminal_sha256": hashlib.sha256(
            _canonical_json(identity).encode("ascii")
        ).hexdigest(),
    }


def _current_boot_id():
    value = BOOT_ID_PATH.read_text(encoding="ascii").strip()
    if BOOT_ID_RE.fullmatch(value) is None:
        raise ValueError("nd-0 boot ID is invalid")
    return value


def _boottime_ns():
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is None or not hasattr(time, "clock_gettime_ns"):
        raise RuntimeError("nd-0 CLOCK_BOOTTIME is unavailable")
    return time.clock_gettime_ns(clock)


def _validate_clock_authority(value):
    if not isinstance(value, dict) or set(value) != {
            "clock_authority_version", "clock_authority_node",
            "clock_authority_boot_id", "boottime_before_ns",
            "authority_unix_ns", "boottime_after_ns"}:
        raise ValueError("HGP clock authority schema is invalid")
    integer_fields = (
        "boottime_before_ns", "authority_unix_ns", "boottime_after_ns",
    )
    if (value.get("clock_authority_version") != CLOCK_AUTHORITY_VERSION
            or value.get("clock_authority_node") != "nd-0"
            or BOOT_ID_RE.fullmatch(str(
                value.get("clock_authority_boot_id", ""),
            )) is None
            or any(isinstance(value.get(name), bool)
                   or not isinstance(value.get(name), int)
                   or value[name] <= 0 for name in integer_fields)
            or value["boottime_after_ns"] < value["boottime_before_ns"]
            or value["boottime_after_ns"] - value["boottime_before_ns"]
            > MAX_CLOCK_CAPTURE_SPAN_NS):
        raise ValueError("HGP clock authority identity is invalid")
    return dict(value)


def _capture_clock_authority():
    before = _boottime_ns()
    unix_ns = time.time_ns()
    after = _boottime_ns()
    value = {
        "clock_authority_version": CLOCK_AUTHORITY_VERSION,
        "clock_authority_node": "nd-0",
        "clock_authority_boot_id": _current_boot_id(),
        "boottime_before_ns": before,
        "authority_unix_ns": unix_ns,
        "boottime_after_ns": after,
    }
    return _validate_clock_authority(value)


def _read_frozen_config(deployment_root, expected_archive_sha256):
    archive = Path(deployment_root) / "SOURCE.tar"
    if _sha256_file(archive) != expected_archive_sha256:
        raise ValueError("HGP orchestrator source archive SHA mismatch")
    with tarfile.open(archive, "r:*") as handle:
        try:
            member = handle.getmember(CONFIG_RELATIVE)
        except KeyError as exc:
            raise ValueError("HGP config is absent from the source archive") from exc
        if not member.isfile():
            raise ValueError("HGP config archive member is not a regular file")
        stream = handle.extractfile(member)
        if stream is None:
            raise ValueError("HGP config archive member is not a regular file")
        payload = stream.read()
    config = json.loads(payload.decode("ascii"))
    execution = config.get("execution", {})
    capacities = execution.get("capacities", {})
    analysis = execution.get("analysis", {})
    if (tuple(execution.get("execution_nodes", ())) != EXECUTION_NODES
            or set(capacities) != set(EXECUTION_NODES)
            or any(isinstance(capacities[node], bool)
                   or int(capacities[node]) <= 0 for node in EXECUTION_NODES)
            or analysis.get("node") != "nd-3"
            or isinstance(analysis.get("capacity"), bool)
            or int(analysis.get("capacity", 0)) <= 0
            or int(analysis.get("num_workers", -1))
            != int(analysis.get("capacity", 0))):
        raise ValueError("HGP archived execution topology is invalid")
    if set(config.get("resource_tiers", {})) != {"T1", "T2", "T3"}:
        raise ValueError("HGP archived resource tiers changed")
    return config, hashlib.sha256(payload).hexdigest()


def _validate_nd0_persistence(args, base):
    if os.environ.get("EXP102_HGP_ORCHESTRATOR_PERSISTENCE") != (
            ND0_PERSISTENCE_TOKEN):
        raise ValueError("HGP orchestrator requires the nd-0 setsid launcher")
    if os.getsid(0) != os.getpid():
        raise ValueError("HGP orchestrator must be a detached session leader")

    token = hashlib.sha256(args.run_id.encode("ascii")).hexdigest()[:8]
    expected_guard = (
        Path(base) / "logs"
        / f".{args.run_id}_hgp_orchestrator_{token}_{args.phase}.launch"
    )
    supplied_guard = os.environ.get("EXP102_HGP_ORCHESTRATOR_GUARD")
    if supplied_guard is None:
        raise ValueError("HGP orchestrator launch guard is absent")
    try:
        guard = Path(supplied_guard).resolve(strict=True)
        canonical_guard = expected_guard.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("HGP orchestrator launch guard is invalid") from exc
    if (guard != canonical_guard or not guard.is_dir()
            or expected_guard.is_symlink()):
        raise ValueError("HGP orchestrator launch guard is not canonical")

    metadata_path = guard / "LAUNCH.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("HGP orchestrator launch metadata is invalid")
    metadata = _read_json(metadata_path)
    expected_attestation_sha = (
        args.local_attestation_sha256
        if args.phase == "measurement" else None
    )
    if (set(metadata) != {
            "archive_sha256", "command_sha256", "launcher_version",
            "local_attestation_sha256", "manifest_sha256", "phase",
            "run_id", "source_commit"}
            or metadata.get("launcher_version") != ND0_LAUNCHER_VERSION
            or metadata.get("run_id") != args.run_id
            or metadata.get("phase") != args.phase
            or metadata.get("source_commit") != args.source_commit
            or metadata.get("archive_sha256") != args.archive_sha256
            or metadata.get("manifest_sha256")
            != args.source_manifest_sha256
            or metadata.get("local_attestation_sha256")
            != expected_attestation_sha
            or SHA256_RE.fullmatch(str(metadata.get("command_sha256", "")))
            is None):
        raise ValueError("HGP orchestrator launch metadata identity is invalid")

    pid_path = guard / "ORCHESTRATOR_PID"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(pid_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(f"{os.getpid()}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("HGP orchestrator PID metadata already exists") from exc
    return guard


def _require_verified_launch(args, home):
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError("HGP run ID is invalid")
    if SHA1_RE.fullmatch(args.source_commit) is None:
        raise ValueError("HGP source commit must be a full lowercase SHA")
    if any(SHA256_RE.fullmatch(value) is None for value in (
            args.archive_sha256, args.source_manifest_sha256)):
        raise ValueError("HGP deployment SHA256 is invalid")
    if os.environ.get("EXP102_SOURCE_COMMIT") != args.source_commit:
        raise ValueError("HGP orchestrator itself must run from verified source")
    if platform.node().split(".", 1)[0] != "nd-0":
        raise ValueError("HGP orchestrator is owned by storage node nd-0")

    base = (Path(home).resolve() / ".single_shot")
    deployment_root = base / "repos" / args.run_id
    run_root = base / "runs" / args.run_id
    if not (base / "logs").is_dir():
        raise ValueError("HGP server log root is absent")
    if not deployment_root.is_dir():
        raise FileNotFoundError("HGP launch deployment is absent")
    if args.phase == "preflight" and run_root.exists():
        raise FileExistsError("HGP preflight requires a fresh run root")
    if args.phase == "measurement" and not run_root.is_dir():
        raise FileNotFoundError(
            "HGP measurement requires the completed preflight run root"
        )

    # Publish the detached session-leader PID before any potentially slow
    # archive I/O, so the outer launcher never times out on an untracked
    # orchestrator that can continue in its own session.
    launch_guard = _validate_nd0_persistence(args, base)
    archive = deployment_root / "SOURCE.tar"
    archive_marker = deployment_root / "ARCHIVE_SHA256"
    manifest = deployment_root / "SOURCE_MANIFEST.json"
    marker = deployment_root / "SOURCE_COMMIT"
    source = deployment_root / "source"
    if not (archive.is_file() and archive_marker.is_file()
            and manifest.is_file() and marker.is_file()
            and source.is_dir()):
        raise ValueError("HGP deployment evidence is incomplete")
    if (_sha256_file(archive) != args.archive_sha256
            or archive_marker.read_text(encoding="ascii").strip()
            != args.archive_sha256
            or _sha256_file(manifest) != args.source_manifest_sha256
            or marker.read_text(encoding="ascii").strip()
            != args.source_commit):
        raise ValueError("HGP deployment identity does not match launch arguments")
    return deployment_root, run_root, launch_guard


def _verified_stage_shell(deployment_root, source_commit, archive_sha256,
                          source_manifest_sha256, stage, stage_dir, log_file,
                          prerequisites, workflow_argv):
    archive = Path(deployment_root) / "SOURCE.tar"
    wrapper_arguments = [
        "bash", WRAPPER_RELATIVE, stage, str(stage_dir), str(log_file),
    ]
    for marker in prerequisites:
        wrapper_arguments.extend(("--require-success", str(marker)))
    wrapper_arguments.extend(("--", "python", "-m", WORKFLOW_MODULE))
    wrapper_arguments.extend(str(value) for value in workflow_argv)
    verified_arguments = [
        str(deployment_root), source_commit, archive_sha256,
        source_manifest_sha256, "conda", "run", "-n", "11",
        "--no-capture-output", *wrapper_arguments,
    ]
    checksum_line = f"{archive_sha256}  {archive}"
    return "\n".join((
        "set -euo pipefail",
        f"printf '%s\\n' {shlex.quote(checksum_line)} | sha256sum -c - >/dev/null",
        (
            f"tar -xOf {shlex.quote(str(archive))} "
            f"{shlex.quote(VERIFY_RELATIVE)} | bash -s -- "
            + shlex.join(verified_arguments)
        ),
    ))


def _remote_command(arguments):
    return shlex.join(tuple(str(value) for value in arguments))


class _Stage:
    def __init__(self, *, key, node, stage, workflow_argv, stage_dir,
                 log_file, bootstrap_log, prerequisites, session):
        self.key = key
        self.node = node
        self.stage = stage
        self.workflow_argv = tuple(workflow_argv)
        self.stage_dir = Path(stage_dir)
        self.log_file = Path(log_file)
        self.bootstrap_log = Path(bootstrap_log)
        self.prerequisites = tuple(Path(value) for value in prerequisites)
        self.session = session

    @property
    def success(self):
        return self.stage_dir / "SUCCESS"

    @property
    def failed(self):
        return self.stage_dir / "FAILED"


class HgpOrchestrator:
    def __init__(self, *, run_id, source_commit, archive_sha256,
                 source_manifest_sha256, deployment_root, run_root, config,
                 config_file_sha256, clock_authority, local_attestation=None,
                 local_attestation_sha256=None, launch_guard=None,
                 poll_seconds=5.0):
        self.run_id = run_id
        self.source_commit = source_commit
        self.archive_sha256 = archive_sha256
        self.source_manifest_sha256 = source_manifest_sha256
        self.deployment_root = Path(deployment_root)
        self.run_root = Path(run_root)
        self.config = config
        self.config_file_sha256 = config_file_sha256
        self.clock_authority = _validate_clock_authority(clock_authority)
        self.local_attestation = (
            None if local_attestation is None else Path(local_attestation)
        )
        self.local_attestation_sha256 = local_attestation_sha256
        self.launch_guard = (
            None if launch_guard is None else Path(launch_guard)
        )
        self.log_root = (
            self.launch_guard.parent if self.launch_guard is not None
            else Path.home() / ".single_shot/logs"
        )
        self.poll_seconds = float(poll_seconds)
        self.registry = REGISTRY_RELATIVE
        self.config_path = CONFIG_RELATIVE
        self.token = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:8]
        self.control_root = self.run_root / "control"
        self.artifact_root = self.run_root / "hgp_global/artifacts"
        self.artifact_manifest = self.control_root / "hgp_artifacts.json"
        self.schedule = self.control_root / "HGP_GLOBAL_24H_SCHEDULE.json"
        self.preflight_root = self.run_root / "hgp_global/preflight"
        self.preflight = self.control_root / "hgp_preflight.json"
        self.control = self.control_root / "hgp_measurement_control.json"
        self.raw_root = self.run_root / "hgp_global/raw"
        self.node_report_root = self.run_root / "hgp_global/node_reports"
        self.report = self.control_root / "hgp_report.json"
        self.decision = self.control_root / "hgp_decision.json"
        self.package = self.control_root / "hgp_terminal_package.json"
        self.acceptance_root = self.run_root / "hgp_global/nd0_acceptance"
        self.preflight_acceptance_manifest = (
            self.control_root / "HGP_ND0_PREFLIGHT_ACCEPTANCE.json"
        )
        self.measurement_acceptance_manifest = (
            self.control_root / "HGP_ND0_MEASUREMENT_ACCEPTANCE.json"
        )
        self.marker_root = self.run_root / "hgp_global/markers"

    def _common(self, *, artifact_manifest=False):
        values = [
            "--source-commit", self.source_commit,
            "--archive-sha256", self.archive_sha256,
            "--source-manifest-sha256", self.source_manifest_sha256,
            "--artifact-root", self.artifact_root,
        ]
        if artifact_manifest:
            values.extend(("--artifact-manifest", self.artifact_manifest))
        values.extend((
            "--schedule", self.schedule,
            "--registry", self.registry,
            "--config", self.config_path,
        ))
        return values

    def _stage(self, key, node, stage, action, arguments, prerequisites=()):
        safe_key = key.replace("_", "-")
        return _Stage(
            key=key, node=node, stage=stage,
            workflow_argv=(action, *arguments),
            stage_dir=self.marker_root / key,
            log_file=self.log_root / f"{self.run_id}_hgp_{safe_key}.log",
            bootstrap_log=(
                self.log_root / f"{self.run_id}_hgp_{safe_key}_bootstrap.log"
            ),
            prerequisites=prerequisites,
            session=f"e102h_{self.token}_{safe_key[:18]}",
        )

    def schedule_stage(self):
        arguments = [
            "--source-commit", self.source_commit,
            "--archive-sha256", self.archive_sha256,
            "--source-manifest-sha256", self.source_manifest_sha256,
            "--registry", self.registry, "--config", self.config_path,
            "--run-id", self.run_id,
            "--clock-authority-json", _canonical_json(self.clock_authority),
            "--output", self.schedule,
        ]
        return self._stage(
            "00_schedule", "nd-1", "build-schedule", "build-schedule",
            arguments,
        )

    def artifact_stage(self, schedule_success):
        return self._stage(
            "01_artifacts", "nd-1", "build-artifacts", "build-artifacts",
            [*self._common(), "--output", self.artifact_manifest],
            (schedule_success,),
        )

    def preflight_node_stages(self, artifact_success):
        return [
            self._stage(
                f"02_preflight_{node}", node, "preflight", "preflight-node",
                [
                    node, *self._common(artifact_manifest=True),
                    "--output-root", self.preflight_root,
                ],
                (artifact_success,),
            )
            for node in PREFLIGHT_NODES
        ]

    def preflight_combine_stage(self, node_successes):
        arguments = [*self._common(artifact_manifest=True)]
        for node in PREFLIGHT_NODES:
            arguments.extend((
                "--node-report",
                f"{node}={self.preflight_root / 'nodes' / node / 'preflight.json'}",
            ))
        arguments.extend(("--output", self.preflight))
        return self._stage(
            "03_preflight_combine", "nd-1", "preflight",
            "combine-preflight", arguments, node_successes,
        )

    def control_stage(self, preflight_success):
        return self._stage(
            "04_control", "nd-1", "freeze-control", "build-control",
            [
                *self._common(artifact_manifest=True),
                "--preflight", self.preflight,
                "--output", self.control,
            ],
            (preflight_success,),
        )

    def screen_stages(self, control_success):
        capacities = self.config["execution"]["capacities"]
        return [
            self._stage(
                f"05_screen_{node}", node, "screen", "run-node",
                [
                    node, *self._common(artifact_manifest=True),
                    "--control", self.control,
                    "--preflight", self.preflight,
                    "--raw-root", self.raw_root,
                    "--output", self.node_report_root / f"{node}.json",
                    "--num-workers", int(capacities[node]),
                ],
                (control_success,),
            )
            for node in EXECUTION_NODES
        ]

    def analysis_stage(self, screen_successes):
        analysis = self.config["execution"]["analysis"]
        arguments = [
            analysis["node"], *self._common(artifact_manifest=True),
            "--control", self.control,
            "--preflight", self.preflight,
        ]
        for node in EXECUTION_NODES:
            arguments.extend((
                "--node-report", f"{node}={self.node_report_root / (node + '.json')}",
            ))
        arguments.extend((
            "--raw-root", self.raw_root,
            "--output", self.report,
            "--decision-output", self.decision,
            "--package-output", self.package,
            "--num-workers", int(analysis["num_workers"]),
        ))
        return self._stage(
            "06_analyze", analysis["node"], "analyze", "analyze",
            arguments, screen_successes,
        )

    def _launch(self, stage):
        for marker in stage.prerequisites:
            _validate_success_marker(
                marker, source_commit=self.source_commit,
            )
        if any(path.exists() for path in (
                stage.stage_dir / "RUNNING", stage.success, stage.failed,
                stage.log_file, stage.bootstrap_log,
                self._acceptance_path(stage))):
            raise FileExistsError(f"HGP stage evidence already exists: {stage.key}")
        verified_shell = _verified_stage_shell(
            self.deployment_root, self.source_commit, self.archive_sha256,
            self.source_manifest_sha256, stage.stage, stage.stage_dir,
            stage.log_file, stage.prerequisites, stage.workflow_argv,
        )
        # Redirect the entire bootstrap in the screen's login shell.  The
        # scientific workflow itself still writes its immutable stage log.
        wrapped_shell = "\n".join((
            "set -euo pipefail",
            f"{{\n{verified_shell}\n}} "
            f"> {shlex.quote(str(stage.bootstrap_log))} 2>&1",
        ))
        remote = _remote_command((
            "screen", "-dmS", stage.session, "bash", "-lc", wrapped_shell,
        ))
        subprocess.run((
            "ssh", *SSH_OPTIONS, stage.node, remote,
        ), check=True, timeout=180.0)
        print(_canonical_json({
            "event": "launched", "key": stage.key, "node": stage.node,
            "screen": stage.session, "stage": stage.stage,
        }), flush=True)

    def _session_alive(self, stage):
        remote = _remote_command((
            "screen", "-S", stage.session, "-Q", "select", ".",
        ))
        try:
            completed = subprocess.run((
                "ssh", *SSH_OPTIONS, stage.node, remote,
            ), check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=60.0)
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _stop(self, stage):
        remote = _remote_command(("screen", "-S", stage.session, "-X", "quit"))
        try:
            subprocess.run((
                "ssh", *SSH_OPTIONS, stage.node, remote,
            ), check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=60.0)
        except subprocess.TimeoutExpired:
            pass

    def _authority_now_ns(self):
        if _current_boot_id() != self.clock_authority[
                "clock_authority_boot_id"]:
            raise RuntimeError("nd-0 reboot invalidated HGP clock authority")
        return _boottime_ns()

    def _acceptance_path(self, stage):
        phase = self._stage_acceptance_spec(stage)[0]
        return self.acceptance_root / phase / f"{stage.key}.json"

    def _stage_acceptance_spec(self, stage):
        spec = STAGE_ACCEPTANCE_SPECS.get(stage.key)
        if (spec is None or stage.stage != spec[1] or stage.node != spec[2]):
            raise ValueError(f"HGP stage deadline identity changed: {stage.key}")
        deadline = (
            self.clock_authority["boottime_before_ns"]
            + spec[4] * NANOSECONDS_PER_SECOND
        )
        return (*spec, deadline)

    def _validate_stage_acceptance(self, stage, deadline_boottime_ns):
        path = self._acceptance_path(stage)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"HGP nd-0 acceptance is invalid: {path}")
        value = _read_json(path)
        identity = dict(value)
        stored_sha = identity.pop("acceptance_sha256", None)
        marker, success_file_sha256 = _validate_stage_success_snapshot(
            stage, self.source_commit,
        )
        observed = value.get("observed_boottime_ns")
        deadline = value.get("deadline_boottime_ns")
        phase, _stage_name, _node, deadline_kind, _seconds, expected_deadline = (
            self._stage_acceptance_spec(stage)
        )
        try:
            success_relpath = str(stage.success.relative_to(self.run_root))
        except ValueError as exc:
            raise ValueError("HGP SUCCESS marker is outside the run root") from exc
        clock_sha = hashlib.sha256(
            _canonical_json(self.clock_authority).encode("ascii")
        ).hexdigest()
        if (set(value) != STAGE_ACCEPTANCE_FIELDS
                or value.get("acceptance_version")
                != STAGE_ACCEPTANCE_VERSION
                or value.get("run_id") != self.run_id
                or value.get("phase") != phase
                or value.get("source_commit") != self.source_commit
                or value.get("archive_sha256") != self.archive_sha256
                or value.get("source_manifest_sha256")
                != self.source_manifest_sha256
                or value.get("clock_authority_sha256") != clock_sha
                or value.get("clock_authority_boot_id")
                != self.clock_authority["clock_authority_boot_id"]
                or value.get("stage_key") != stage.key
                or value.get("stage") != stage.stage
                or value.get("node") != stage.node
                or isinstance(observed, bool) or not isinstance(observed, int)
                or observed < self.clock_authority["boottime_after_ns"]
                or isinstance(deadline, bool) or not isinstance(deadline, int)
                or deadline != expected_deadline
                or deadline != int(deadline_boottime_ns)
                or value.get("deadline_kind") != deadline_kind
                or observed >= deadline
                or value.get("success_relpath") != success_relpath
                or value.get("success_file_sha256") != success_file_sha256
                or value.get("success_marker") != marker
                or stored_sha != hashlib.sha256(
                    _canonical_json(identity).encode("ascii")
                ).hexdigest()):
            raise ValueError(f"HGP nd-0 acceptance is invalid: {path}")
        return value

    def _record_stage_acceptance(
            self, stage, marker, success_file_sha256, observed_boottime_ns,
            deadline_boottime_ns):
        phase, _stage_name, _node, deadline_kind, _seconds, expected_deadline = (
            self._stage_acceptance_spec(stage)
        )
        observed_boottime_ns = int(observed_boottime_ns)
        if (expected_deadline != int(deadline_boottime_ns)
                or observed_boottime_ns
                < self.clock_authority["boottime_after_ns"]
                or observed_boottime_ns >= expected_deadline):
            raise TimeoutError("HGP stage acceptance missed its frozen deadline")
        success_relpath = str(stage.success.relative_to(self.run_root))
        identity = {
            "acceptance_version": STAGE_ACCEPTANCE_VERSION,
            "run_id": self.run_id,
            "phase": phase,
            "source_commit": self.source_commit,
            "archive_sha256": self.archive_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "clock_authority_sha256": hashlib.sha256(
                _canonical_json(self.clock_authority).encode("ascii")
            ).hexdigest(),
            "clock_authority_boot_id": self.clock_authority[
                "clock_authority_boot_id"
            ],
            "stage_key": stage.key,
            "stage": stage.stage,
            "node": stage.node,
            "observed_boottime_ns": observed_boottime_ns,
            "deadline_kind": deadline_kind,
            "deadline_boottime_ns": expected_deadline,
            "success_relpath": success_relpath,
            "success_file_sha256": success_file_sha256,
            "success_marker": marker,
        }
        value = {
            **identity,
            "acceptance_sha256": hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest(),
        }
        path = self._acceptance_path(stage)
        _write_exclusive_json(path, value)
        self._validate_stage_acceptance(stage, deadline_boottime_ns)
        return path

    def _phase_launch_metadata(self, phase):
        if phase not in PHASE_STAGE_KEYS:
            raise ValueError("HGP acceptance phase is invalid")
        guard = (
            self.log_root
            / f".{self.run_id}_hgp_orchestrator_{self.token}_{phase}.launch"
        )
        metadata_path = guard / "LAUNCH.json"
        if (not guard.is_dir() or guard.is_symlink()
                or not metadata_path.is_file() or metadata_path.is_symlink()):
            raise ValueError("HGP phase launch metadata is absent")
        metadata = _read_json(metadata_path)
        expected_attestation_sha = (
            None if phase == "preflight" else self.local_attestation_sha256
        )
        if (set(metadata) != LAUNCH_METADATA_FIELDS
                or metadata.get("launcher_version") != ND0_LAUNCHER_VERSION
                or metadata.get("run_id") != self.run_id
                or metadata.get("phase") != phase
                or metadata.get("source_commit") != self.source_commit
                or metadata.get("archive_sha256") != self.archive_sha256
                or metadata.get("manifest_sha256")
                != self.source_manifest_sha256
                or metadata.get("local_attestation_sha256")
                != expected_attestation_sha
                or SHA256_RE.fullmatch(str(
                    metadata.get("command_sha256", ""),
                )) is None):
            raise ValueError("HGP phase launch metadata identity is invalid")
        return metadata, _sha256_file(metadata_path)

    def _phase_acceptance_identity(
            self, phase, stages, bound_files, bound_identity,
            published_boottime_ns, prior_manifest=None):
        stages = tuple(stages)
        if tuple(stage.key for stage in stages) != PHASE_STAGE_KEYS.get(phase):
            raise ValueError("HGP acceptance stage order changed")
        stage_rows = []
        for stage in stages:
            deadline = self._stage_acceptance_spec(stage)[-1]
            acceptance = self._validate_stage_acceptance(stage, deadline)
            path = self._acceptance_path(stage)
            stage_rows.append({
                "stage_key": stage.key,
                "acceptance_relpath": str(path.relative_to(self.run_root)),
                "acceptance_sha256": acceptance["acceptance_sha256"],
                "acceptance_file_sha256": _sha256_file(path),
            })
        acceptance_dir = self.acceptance_root / phase
        actual_names = (
            {path.name for path in acceptance_dir.iterdir()}
            if acceptance_dir.is_dir() and not acceptance_dir.is_symlink()
            else set()
        )
        expected_names = {f"{stage.key}.json" for stage in stages}
        if actual_names != expected_names:
            raise ValueError("HGP phase acceptance file set changed")
        launch_metadata, launch_metadata_file_sha256 = (
            self._phase_launch_metadata(phase)
        )
        phase_seconds = 6 * 3600 if phase == "preflight" else 24 * 3600
        phase_deadline = (
            self.clock_authority["boottime_before_ns"]
            + phase_seconds * NANOSECONDS_PER_SECOND
        )
        if (isinstance(published_boottime_ns, bool)
                or not isinstance(published_boottime_ns, int)
                or published_boottime_ns
                < self.clock_authority["boottime_after_ns"]
                or published_boottime_ns >= phase_deadline):
            raise TimeoutError("HGP phase acceptance missed its frozen deadline")
        bound_hashes = {}
        for name, path in sorted(bound_files.items()):
            path = Path(path)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"HGP bound phase file is invalid: {name}")
            bound_hashes[name] = _sha256_file(path)
        prior = None
        if prior_manifest is not None:
            prior_path = Path(prior_manifest)
            if not prior_path.is_file() or prior_path.is_symlink():
                raise ValueError("HGP prior acceptance manifest is invalid")
            prior_value = _read_json(prior_path)
            prior = {
                "manifest_relpath": str(prior_path.relative_to(self.run_root)),
                "manifest_file_sha256": _sha256_file(prior_path),
                "manifest_sha256": prior_value.get("manifest_sha256"),
            }
        return {
            "manifest_version": ACCEPTANCE_MANIFEST_VERSION,
            "phase": phase,
            "run_id": self.run_id,
            "source_commit": self.source_commit,
            "archive_sha256": self.archive_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "clock_authority_sha256": hashlib.sha256(
                _canonical_json(self.clock_authority).encode("ascii")
            ).hexdigest(),
            "clock_authority_boot_id": self.clock_authority[
                "clock_authority_boot_id"
            ],
            "phase_deadline_boottime_ns": phase_deadline,
            "published_boottime_ns": published_boottime_ns,
            "launch_metadata": launch_metadata,
            "launch_metadata_file_sha256": launch_metadata_file_sha256,
            "stage_acceptances": stage_rows,
            "bound_file_sha256": bound_hashes,
            "bound_identity": dict(bound_identity),
            "prior_manifest": prior,
        }

    def _write_phase_acceptance_manifest(
            self, path, phase, stages, bound_files, bound_identity,
            prior_manifest=None):
        published_boottime_ns = self._authority_now_ns()
        identity = self._phase_acceptance_identity(
            phase, stages, bound_files, bound_identity,
            published_boottime_ns, prior_manifest,
        )
        value = {
            **identity,
            "manifest_sha256": hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest(),
        }
        _write_exclusive_json(path, value)
        return self._validate_phase_acceptance_manifest(
            path, phase, stages, bound_files, bound_identity, prior_manifest,
        )

    def _validate_phase_acceptance_manifest(
            self, path, phase, stages, bound_files, bound_identity,
            prior_manifest=None):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"HGP nd-0 {phase} acceptance manifest is invalid")
        value = _read_json(path)
        identity = self._phase_acceptance_identity(
            phase, stages, bound_files, bound_identity,
            value.get("published_boottime_ns"), prior_manifest,
        )
        expected = {
            **identity,
            "manifest_sha256": hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest(),
        }
        if value != expected:
            raise ValueError(f"HGP nd-0 {phase} acceptance manifest is invalid")
        return value

    def run_batch(self, stages, deadline_boottime_ns):
        stages = tuple(stages)
        if not stages:
            raise ValueError("HGP orchestration batch cannot be empty")
        deadline_ns = int(deadline_boottime_ns)
        if (isinstance(deadline_boottime_ns, bool)
                or deadline_ns != deadline_boottime_ns):
            raise ValueError("HGP boottime deadline is invalid")
        if any(self._stage_acceptance_spec(stage)[-1] != deadline_ns
               for stage in stages):
            raise ValueError("HGP stage batch deadline changed")
        launched = []
        try:
            for stage in stages:
                if self._authority_now_ns() >= deadline_ns:
                    raise TimeoutError("HGP stage deadline expired before launch")
                self._launch(stage)
                launched.append(stage)
            last_probe_ns = self._authority_now_ns()
            pending = {stage.key: stage for stage in stages}
            while pending:
                now_ns = self._authority_now_ns()
                if now_ns >= deadline_ns:
                    raise TimeoutError(
                        "HGP stage batch exceeded its frozen deadline"
                    )
                for key, stage in tuple(pending.items()):
                    if stage.failed.exists():
                        raise RuntimeError(
                            f"HGP stage failed: {key}: "
                            f"{stage.failed.read_text(encoding='ascii').strip()}"
                        )
                    if stage.success.exists():
                        marker, success_file_sha256 = (
                            _validate_stage_success_snapshot(
                                stage, self.source_commit,
                            )
                        )
                        # Acceptance is timed after reading and validating the
                        # exact SUCCESS bytes that the evidence record hashes.
                        observed_ns = self._authority_now_ns()
                        if observed_ns >= deadline_ns:
                            raise TimeoutError(
                                "HGP SUCCESS was first observed at or after "
                                "its frozen deadline"
                            )
                        acceptance = self._record_stage_acceptance(
                            stage, marker, success_file_sha256, observed_ns,
                            deadline_ns,
                        )
                        del pending[key]
                        print(_canonical_json({
                            "event": "completed", "key": key,
                            "success": str(stage.success),
                            "acceptance": str(acceptance),
                            "acceptance_file_sha256": _sha256_file(acceptance),
                        }), flush=True)
                if not pending:
                    break
                if now_ns - last_probe_ns >= 60 * NANOSECONDS_PER_SECOND:
                    for stage in pending.values():
                        if not self._session_alive(stage):
                            raise RuntimeError(
                                f"HGP screen exited without a terminal marker: {stage.key}"
                            )
                    last_probe_ns = now_ns
                remaining_seconds = (
                    deadline_ns - now_ns
                ) / NANOSECONDS_PER_SECOND
                time.sleep(min(
                    self.poll_seconds, max(0.001, remaining_seconds),
                ))
        except BaseException:
            for stage in launched:
                if not stage.success.exists() and not stage.failed.exists():
                    self._stop(stage)
            raise
        return tuple(stage.success for stage in stages)

    def run_preflight(self):
        schedule_stage = self.schedule_stage()
        self.run_batch((schedule_stage,), (
            self.clock_authority["boottime_before_ns"]
            + 900 * NANOSECONDS_PER_SECOND
        ))
        schedule = _validate_schedule_output(
            self.schedule, self.run_id, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )

        artifact = self.artifact_stage(schedule_stage.success)
        self.run_batch(
            (artifact,), schedule["preflight_deadline_boottime_ns"],
        )
        preflight_nodes = self.preflight_node_stages(artifact.success)
        node_successes = self.run_batch(
            preflight_nodes, schedule["preflight_deadline_boottime_ns"],
        )
        combine = self.preflight_combine_stage(node_successes)
        self.run_batch(
            (combine,), schedule["preflight_deadline_boottime_ns"],
        )
        preflight = _validate_aggregate_preflight(
            self.preflight, schedule, self.config, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        preflight_stages = (
            schedule_stage, artifact, *preflight_nodes, combine,
        )
        preflight_acceptance = self._write_phase_acceptance_manifest(
            self.preflight_acceptance_manifest, "preflight",
            preflight_stages,
            {
                "schedule": self.schedule,
                "artifact_manifest": self.artifact_manifest,
                "aggregate_preflight": self.preflight,
            },
            {
                "schedule_sha256": schedule["schedule_sha256"],
                "artifact_manifest_sha256": _read_json(
                    self.artifact_manifest,
                )["artifact_manifest_sha256"],
                "preflight_status": preflight["status"],
                "selected_resource_tier": preflight[
                    "selected_resource_tier"
                ],
                "registry_file_sha256": _sha256_file(self.registry),
                "config_file_sha256": self.config_file_sha256,
            },
        )
        result = {
            "event": "preflight_ready_for_local_audit",
            "run_id": self.run_id,
            "selected_resource_tier": preflight["selected_resource_tier"],
            "schedule": str(self.schedule),
            "schedule_file_sha256": _sha256_file(self.schedule),
            "artifact_manifest": str(self.artifact_manifest),
            "artifact_manifest_file_sha256": _sha256_file(
                self.artifact_manifest,
            ),
            "artifact_root": str(self.artifact_root),
            "preflight": str(self.preflight),
            "preflight_file_sha256": _sha256_file(self.preflight),
            "nd0_acceptance_manifest": str(
                self.preflight_acceptance_manifest,
            ),
            "nd0_acceptance_manifest_file_sha256": _sha256_file(
                self.preflight_acceptance_manifest,
            ),
            "nd0_acceptance_manifest_sha256": preflight_acceptance[
                "manifest_sha256"
            ],
            "control_freeze_deadline_unix": schedule[
                "control_freeze_deadline_unix"
            ],
        }
        print(_canonical_json(result), flush=True)
        return result

    def run_measurement(self):
        schedule_stage = self.schedule_stage()
        artifact = self.artifact_stage(schedule_stage.success)
        preflight_nodes = self.preflight_node_stages(artifact.success)
        combine = self.preflight_combine_stage(
            tuple(stage.success for stage in preflight_nodes),
        )
        _validate_stage_success(schedule_stage, self.source_commit)
        _validate_stage_success(artifact, self.source_commit)
        for stage in preflight_nodes:
            _validate_stage_success(stage, self.source_commit)
        _validate_stage_success(combine, self.source_commit)
        schedule = _validate_schedule_output(
            self.schedule, self.run_id, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        preflight = _validate_aggregate_preflight(
            self.preflight, schedule, self.config, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        preflight_stages = (
            schedule_stage, artifact, *preflight_nodes, combine,
        )
        self._validate_phase_acceptance_manifest(
            self.preflight_acceptance_manifest, "preflight",
            preflight_stages,
            {
                "schedule": self.schedule,
                "artifact_manifest": self.artifact_manifest,
                "aggregate_preflight": self.preflight,
            },
            {
                "schedule_sha256": schedule["schedule_sha256"],
                "artifact_manifest_sha256": _read_json(
                    self.artifact_manifest,
                )["artifact_manifest_sha256"],
                "preflight_status": preflight["status"],
                "selected_resource_tier": preflight[
                    "selected_resource_tier"
                ],
                "registry_file_sha256": _sha256_file(self.registry),
                "config_file_sha256": self.config_file_sha256,
            },
        )
        if self.local_attestation is None:
            raise ValueError("HGP measurement requires a local attestation")
        expected_attestation = (
            self.control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
        )
        if self.local_attestation.resolve(strict=True) != expected_attestation.resolve(
                strict=True):
            raise ValueError("HGP local attestation path is not canonical")
        _validate_local_attestation(
            self.local_attestation, self.local_attestation_sha256,
            schedule, preflight, self.artifact_manifest,
            self.preflight_acceptance_manifest,
            _sha256_file(self.registry), self.config_file_sha256,
            self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
        )

        control = self.control_stage(combine.success)
        self.run_batch(
            (control,), schedule["control_freeze_deadline_boottime_ns"],
        )
        _validate_control_output(
            self.control, preflight, schedule, self.config,
            self.source_commit, self.archive_sha256,
            self.source_manifest_sha256,
        )

        screens = self.screen_stages(control.success)
        screen_successes = self.run_batch(
            screens, schedule["screen_deadline_boottime_ns"],
        )
        analysis = self.analysis_stage(screen_successes)
        self.run_batch((analysis,), schedule["analysis_deadline_boottime_ns"])
        package = _validate_terminal_output(
            self.package, self.source_commit, schedule,
        )
        _validate_terminal_raw_evidence(
            self.run_root, _read_json(self.control), self.config, package,
        )
        measurement_stages = (control, *screens, analysis)
        measurement_acceptance = self._write_phase_acceptance_manifest(
            self.measurement_acceptance_manifest, "measurement",
            measurement_stages,
            {
                "schedule": self.schedule,
                "artifact_manifest": self.artifact_manifest,
                "aggregate_preflight": self.preflight,
                "local_attestation": self.local_attestation,
                "measurement_control": self.control,
                "nd2_node_report": self.node_report_root / "nd-2.json",
                "nd3_node_report": self.node_report_root / "nd-3.json",
                "terminal_report": self.report,
                "terminal_decision": self.decision,
                "terminal_package": self.package,
            },
            {
                "schedule_sha256": schedule["schedule_sha256"],
                "selected_resource_tier": preflight[
                    "selected_resource_tier"
                ],
                "local_attestation_sha256": _read_json(
                    self.local_attestation,
                )["attestation_sha256"],
                "terminal_status": package["status"],
                "terminal_package_sha256": package["package_sha256"],
            },
            prior_manifest=self.preflight_acceptance_manifest,
        )
        preflight_acceptance = _read_json(
            self.preflight_acceptance_manifest,
        )
        joint_terminal = _joint_terminal_result(
            schedule, package, self.preflight_acceptance_manifest,
            preflight_acceptance, self.measurement_acceptance_manifest,
            measurement_acceptance,
        )
        result = {
            "event": "terminal", "run_id": self.run_id,
            "status": package["status"],
            "terminal_package": str(self.package),
            "package_sha256": package["package_sha256"],
            "nd0_acceptance_manifest": str(
                self.measurement_acceptance_manifest,
            ),
            "nd0_acceptance_manifest_file_sha256": _sha256_file(
                self.measurement_acceptance_manifest,
            ),
            "nd0_acceptance_manifest_sha256": measurement_acceptance[
                "manifest_sha256"
            ],
            "joint_terminal_sha256": joint_terminal[
                "joint_terminal_sha256"
            ],
        }
        print(_canonical_json(result), flush=True)
        return result

    def run(self, phase):
        if phase == "preflight":
            return self.run_preflight()
        if phase == "measurement":
            return self.run_measurement()
        raise ValueError("unknown HGP orchestration phase")


def _validate_success_value(marker, path, expected_stage=None,
                            source_commit=None):
    if (set(marker) != {
            "stage", "source_commit", "stage_fingerprint",
            "prerequisite_success_sha256", "completed_utc"}
            or (expected_stage is not None
                and marker.get("stage") != expected_stage)
            or (source_commit is not None
                and marker.get("source_commit") != source_commit)
            or SHA256_RE.fullmatch(str(marker.get("stage_fingerprint", "")))
            is None
            or not isinstance(marker.get("prerequisite_success_sha256"), list)
            or any(SHA256_RE.fullmatch(str(value)) is None
                   for value in marker["prerequisite_success_sha256"])
            or not isinstance(marker.get("completed_utc"), str)
            or not marker["completed_utc"]):
        raise ValueError(f"HGP SUCCESS marker is invalid: {path}")
    return marker


def _success_snapshot(path, expected_stage=None, source_commit=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"HGP SUCCESS marker is invalid: {path}")
    payload = path.read_bytes()
    try:
        marker = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"HGP SUCCESS marker is invalid: {path}") from exc
    _validate_success_value(marker, path, expected_stage, source_commit)
    return marker, hashlib.sha256(payload).hexdigest()


def _validate_success_marker(path, expected_stage=None, source_commit=None):
    return _success_snapshot(path, expected_stage, source_commit)[0]


def _expected_stage_fingerprint(stage, source_commit):
    prerequisite_sha256 = [
        _sha256_file(path) for path in stage.prerequisites
    ]
    command = [
        "python", "-m", WORKFLOW_MODULE,
        *(str(value) for value in stage.workflow_argv),
    ]
    identity = {
        "stage": stage.stage,
        "source_commit": source_commit,
        "prerequisite_success_sha256": prerequisite_sha256,
        "command": command,
    }
    return hashlib.sha256(
        _canonical_json(identity).encode("ascii")
    ).hexdigest(), prerequisite_sha256


def _validate_stage_success(stage, source_commit):
    return _validate_stage_success_snapshot(stage, source_commit)[0]


def _validate_stage_success_snapshot(stage, source_commit):
    marker, file_sha256 = _success_snapshot(
        stage.success, expected_stage=stage.stage,
        source_commit=source_commit,
    )
    expected_fingerprint, prerequisite_sha256 = _expected_stage_fingerprint(
        stage, source_commit,
    )
    if (marker["stage_fingerprint"] != expected_fingerprint
            or marker["prerequisite_success_sha256"]
            != prerequisite_sha256):
        raise ValueError(f"HGP SUCCESS marker command changed: {stage.success}")
    return marker, file_sha256


def validate_preflight_acceptance_offline(
        path, run_root, schedule, source_commit, archive_sha256,
        source_manifest_sha256, bound_files, bound_identity):
    """Validate the complete nd-0 preflight deadline chain after transfer."""
    run_root = Path(run_root).resolve(strict=True)
    expected_path = run_root / "control/HGP_ND0_PREFLIGHT_ACCEPTANCE.json"
    try:
        path = _require_regular_file_in_root(
            path, run_root, "HGP offline preflight acceptance manifest",
        )
    except ValueError as exc:
        raise ValueError(
            "HGP offline preflight acceptance manifest is invalid"
        ) from exc
    if path != expected_path:
        raise ValueError("HGP offline preflight acceptance path is noncanonical")
    manifest = _read_canonical_json(
        path, "HGP offline preflight acceptance manifest",
    )
    identity = dict(manifest)
    stored_sha = identity.pop("manifest_sha256", None)
    clock = _validate_clock_authority(schedule.get("clock_authority"))
    clock_sha = hashlib.sha256(
        _canonical_json(clock).encode("ascii")
    ).hexdigest()
    deadline = (
        clock["boottime_before_ns"]
        + 6 * 3600 * NANOSECONDS_PER_SECOND
    )
    published = manifest.get("published_boottime_ns")
    launch = manifest.get("launch_metadata")
    if (set(manifest) != ACCEPTANCE_MANIFEST_FIELDS
            or manifest.get("manifest_version")
            != ACCEPTANCE_MANIFEST_VERSION
            or manifest.get("phase") != "preflight"
            or manifest.get("run_id") != schedule.get("run_id")
            or manifest.get("source_commit") != source_commit
            or manifest.get("archive_sha256") != archive_sha256
            or manifest.get("source_manifest_sha256")
            != source_manifest_sha256
            or manifest.get("clock_authority_sha256") != clock_sha
            or manifest.get("clock_authority_boot_id")
            != clock["clock_authority_boot_id"]
            or manifest.get("phase_deadline_boottime_ns") != deadline
            or isinstance(published, bool) or not isinstance(published, int)
            or published < clock["boottime_after_ns"]
            or published >= deadline
            or not isinstance(launch, dict)
            or set(launch) != LAUNCH_METADATA_FIELDS
            or launch.get("phase") != "preflight"
            or launch.get("run_id") != schedule.get("run_id")
            or launch.get("source_commit") != source_commit
            or launch.get("archive_sha256") != archive_sha256
            or launch.get("manifest_sha256") != source_manifest_sha256
            or launch.get("local_attestation_sha256") is not None
            or launch.get("launcher_version") != ND0_LAUNCHER_VERSION
            or SHA256_RE.fullmatch(str(launch.get("command_sha256", "")))
            is None
            or SHA256_RE.fullmatch(str(
                manifest.get("launch_metadata_file_sha256", ""),
            )) is None
            or manifest.get("launch_metadata_file_sha256")
            != hashlib.sha256(
                (_canonical_json(launch) + "\n").encode("ascii")
            ).hexdigest()
            or manifest.get("bound_identity") != bound_identity
            or manifest.get("prior_manifest") is not None
            or stored_sha != hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest()):
        raise ValueError("HGP offline preflight acceptance identity is invalid")
    expected_bound_hashes = {}
    for name, bound_path in sorted(bound_files.items()):
        bound_path = Path(bound_path)
        _require_regular_file_in_root(
            bound_path, run_root, f"HGP offline bound file {name}",
        )
        expected_bound_hashes[name] = _sha256_file(bound_path)
    if manifest.get("bound_file_sha256") != expected_bound_hashes:
        raise ValueError("HGP offline preflight bound files changed")

    rows = manifest.get("stage_acceptances")
    if (not isinstance(rows, list)
            or [row.get("stage_key") for row in rows]
            != list(PHASE_STAGE_KEYS["preflight"])):
        raise ValueError("HGP offline preflight acceptance order changed")
    acceptance_dir = run_root / "hgp_global/nd0_acceptance/preflight"
    expected_names = {f"{key}.json" for key in PHASE_STAGE_KEYS["preflight"]}
    if (not acceptance_dir.is_dir() or acceptance_dir.is_symlink()
            or {item.name for item in acceptance_dir.iterdir()}
            != expected_names):
        raise ValueError("HGP offline preflight acceptance file set changed")
    for row in rows:
        if set(row) != {
                "stage_key", "acceptance_relpath", "acceptance_sha256",
                "acceptance_file_sha256"}:
            raise ValueError("HGP offline preflight acceptance row changed")
        key = row["stage_key"]
        phase, stage_name, node, deadline_kind, seconds = (
            STAGE_ACCEPTANCE_SPECS[key]
        )
        relative = f"hgp_global/nd0_acceptance/preflight/{key}.json"
        acceptance_path = run_root / relative
        if (row.get("acceptance_relpath") != relative
                or not acceptance_path.is_file()
                or acceptance_path.is_symlink()
                or row.get("acceptance_file_sha256")
                != _sha256_file(acceptance_path)):
            raise ValueError("HGP offline stage acceptance file changed")
        acceptance = _read_json(acceptance_path)
        acceptance_identity = dict(acceptance)
        acceptance_sha = acceptance_identity.pop("acceptance_sha256", None)
        expected_deadline = (
            clock["boottime_before_ns"]
            + seconds * NANOSECONDS_PER_SECOND
        )
        observed = acceptance.get("observed_boottime_ns")
        success_relative = f"hgp_global/markers/{key}/SUCCESS"
        success_path = run_root / success_relative
        try:
            _require_regular_file_in_root(
                acceptance_path, run_root,
                "HGP offline preflight stage acceptance",
            )
            _require_regular_file_in_root(
                success_path, run_root, "HGP offline preflight SUCCESS marker",
            )
            expected_prerequisites = [
                _sha256_file(_require_regular_file_in_root(
                    run_root / f"hgp_global/markers/{prerequisite}/SUCCESS",
                    run_root, "HGP offline prerequisite SUCCESS marker",
                ))
                for prerequisite in STAGE_PREREQUISITE_KEYS[key]
            ]
        except ValueError as exc:
            raise ValueError("HGP offline stage evidence is invalid") from exc
        if (set(acceptance) != STAGE_ACCEPTANCE_FIELDS
                or acceptance.get("acceptance_version")
                != STAGE_ACCEPTANCE_VERSION
                or acceptance.get("run_id") != schedule.get("run_id")
                or acceptance.get("phase") != phase
                or acceptance.get("source_commit") != source_commit
                or acceptance.get("archive_sha256") != archive_sha256
                or acceptance.get("source_manifest_sha256")
                != source_manifest_sha256
                or acceptance.get("clock_authority_sha256") != clock_sha
                or acceptance.get("clock_authority_boot_id")
                != clock["clock_authority_boot_id"]
                or acceptance.get("stage_key") != key
                or acceptance.get("stage") != stage_name
                or acceptance.get("node") != node
                or acceptance.get("deadline_kind") != deadline_kind
                or acceptance.get("deadline_boottime_ns") != expected_deadline
                or isinstance(observed, bool) or not isinstance(observed, int)
                or observed < clock["boottime_after_ns"]
                or observed >= expected_deadline
                or acceptance.get("success_relpath") != success_relative
                or acceptance.get("success_file_sha256")
                != _sha256_file(success_path)
                or acceptance.get("success_marker") != _read_json(success_path)
                or acceptance_sha != row.get("acceptance_sha256")
                or acceptance_sha != hashlib.sha256(
                    _canonical_json(acceptance_identity).encode("ascii")
                ).hexdigest()):
            raise ValueError("HGP offline stage acceptance identity changed")
        marker = _validate_success_value(
            acceptance["success_marker"], success_path,
            expected_stage=stage_name, source_commit=source_commit,
        )
        if marker["prerequisite_success_sha256"] != expected_prerequisites:
            raise ValueError("HGP offline preflight prerequisite graph changed")
    return manifest


def validate_measurement_acceptance_offline(
        path, run_root, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256):
    """Validate the pulled terminal package and nd-0 acceptance as one result."""
    run_root = Path(run_root).resolve(strict=True)
    path = Path(path)
    registry_path = Path(registry_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    control_root = run_root / "control"
    fixed_paths = {
        "schedule": control_root / "HGP_GLOBAL_24H_SCHEDULE.json",
        "artifact_manifest": control_root / "hgp_artifacts.json",
        "aggregate_preflight": control_root / "hgp_preflight.json",
        "preflight_acceptance": (
            control_root / "HGP_ND0_PREFLIGHT_ACCEPTANCE.json"
        ),
        "local_attestation": (
            control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
        ),
        "measurement_control": control_root / "hgp_measurement_control.json",
        "nd2_node_report": run_root / "hgp_global/node_reports/nd-2.json",
        "nd3_node_report": run_root / "hgp_global/node_reports/nd-3.json",
        "terminal_report": control_root / "hgp_report.json",
        "terminal_decision": control_root / "hgp_decision.json",
        "terminal_package": control_root / "hgp_terminal_package.json",
    }
    for name, fixed_path in fixed_paths.items():
        _require_regular_file_in_root(
            fixed_path, run_root, f"HGP offline terminal file {name}",
        )
    expected_manifest_path = control_root / "HGP_ND0_MEASUREMENT_ACCEPTANCE.json"
    path = _require_regular_file_in_root(
        path, run_root, "HGP offline measurement acceptance manifest",
    )
    if path != expected_manifest_path:
        raise ValueError("HGP offline measurement manifest path is noncanonical")

    registry_sha = _sha256_file(registry_path)
    config_sha = _sha256_file(config_path)
    schedule_value = _read_json(fixed_paths["schedule"])
    run_id = schedule_value.get("run_id")
    if RUN_ID_RE.fullmatch(str(run_id or "")) is None:
        raise ValueError("HGP offline terminal schedule identity changed")
    schedule = _validate_schedule_output(
        fixed_paths["schedule"], run_id, source_commit, archive_sha256,
        source_manifest_sha256, config_sha, check_current_boot=False,
    )
    if schedule.get("registry_file_sha256") != registry_sha:
        raise ValueError("HGP offline terminal schedule identity changed")
    clock = _validate_clock_authority(schedule.get("clock_authority"))

    artifact = _read_json(fixed_paths["artifact_manifest"])
    config = _read_json(config_path)
    preflight = _validate_aggregate_preflight(
        fixed_paths["aggregate_preflight"], schedule, config, source_commit,
        archive_sha256, source_manifest_sha256, config_sha,
    )
    preflight_bound_files = {
        name: fixed_paths[name]
        for name in ("schedule", "artifact_manifest", "aggregate_preflight")
    }
    preflight_bound_identity = {
        "schedule_sha256": schedule["schedule_sha256"],
        "artifact_manifest_sha256": artifact.get("artifact_manifest_sha256"),
        "preflight_status": preflight.get("status"),
        "selected_resource_tier": preflight.get("selected_resource_tier"),
        "registry_file_sha256": registry_sha,
        "config_file_sha256": config_sha,
    }
    preflight_acceptance = validate_preflight_acceptance_offline(
        fixed_paths["preflight_acceptance"], run_root, schedule,
        source_commit, archive_sha256, source_manifest_sha256,
        preflight_bound_files, preflight_bound_identity,
    )

    local_attestation_sha = _sha256_file(fixed_paths["local_attestation"])
    local_attestation = _validate_local_attestation(
        fixed_paths["local_attestation"], local_attestation_sha,
        schedule, preflight, fixed_paths["artifact_manifest"],
        fixed_paths["preflight_acceptance"], registry_sha, config_sha,
        source_commit, archive_sha256, source_manifest_sha256,
    )
    control = _validate_control_output(
        fixed_paths["measurement_control"], preflight, schedule, config,
        source_commit, archive_sha256, source_manifest_sha256,
    )
    package = _validate_terminal_output(
        fixed_paths["terminal_package"], source_commit, schedule,
    )
    if (package.get("source_identity", {}).get("archive_sha256")
            != archive_sha256
            or package.get("source_identity", {}).get("manifest_sha256")
            != source_manifest_sha256):
        raise ValueError("HGP offline terminal package source identity changed")

    report = _read_json(fixed_paths["terminal_report"])
    report_identity = dict(report)
    report_sha = report_identity.pop("report_sha256", None)
    decision = _read_json(fixed_paths["terminal_decision"])
    decision_identity = dict(decision)
    decision_sha = decision_identity.pop("decision_sha256", None)
    if (report_sha != hashlib.sha256(
                _canonical_json(report_identity).encode("ascii")
            ).hexdigest()
            or report.get("source_commit") != source_commit
            or report.get("archive_sha256") != archive_sha256
            or report.get("source_manifest_sha256")
            != source_manifest_sha256
            or report.get("contract_version") != HGP_CONTRACT_VERSION
            or report.get("status") != package.get("status")
            or report.get("formal_authorization") is not False
            or report.get("production_authorization") is not False
            or set(decision) != TERMINAL_DECISION_FIELDS
            or decision.get("decision_version")
            != TERMINAL_DECISION_VERSION
            or decision.get("contract_version") != HGP_CONTRACT_VERSION
            or decision_sha != hashlib.sha256(
                _canonical_json(decision_identity).encode("ascii")
            ).hexdigest()
            or decision.get("source_commit") != source_commit
            or decision.get("archive_sha256") != archive_sha256
            or decision.get("source_manifest_sha256")
            != source_manifest_sha256
            or decision.get("schedule_sha256")
            != schedule["schedule_sha256"]
            or decision.get("schedule_file_sha256")
            != _sha256_file(fixed_paths["schedule"])
            or decision.get("artifact_manifest_file_sha256")
            != _sha256_file(fixed_paths["artifact_manifest"])
            or decision.get("artifact_manifest_sha256")
            != artifact.get("artifact_manifest_sha256")
            or decision.get("preflight_file_sha256")
            != _sha256_file(fixed_paths["aggregate_preflight"])
            or decision.get("control_file_sha256")
            != _sha256_file(fixed_paths["measurement_control"])
            or decision.get("report_sha256") != report_sha
            or decision.get("report_file_sha256")
            != _sha256_file(fixed_paths["terminal_report"])
            or decision.get("status") != package.get("status")
            or decision.get("raw_evidence_summary")
            != package.get("raw_evidence_summary")
            or decision.get("formal_authorization") is not False
            or decision.get("production_authorization") is not False
            or package.get("decision_sha256") != decision_sha):
        raise ValueError("HGP offline terminal report/decision identity changed")

    expected_package_hashes = {
        "schedule_file_sha256": _sha256_file(fixed_paths["schedule"]),
        "artifact_manifest_file_sha256": _sha256_file(
            fixed_paths["artifact_manifest"],
        ),
        "preflight_file_sha256": _sha256_file(
            fixed_paths["aggregate_preflight"],
        ),
        "control_file_sha256": _sha256_file(
            fixed_paths["measurement_control"],
        ),
        "report_file_sha256": _sha256_file(fixed_paths["terminal_report"]),
        "decision_file_sha256": _sha256_file(
            fixed_paths["terminal_decision"],
        ),
    }
    if (any(package.get(name) != value
            for name, value in expected_package_hashes.items())
            or package.get("execution_report_file_sha256") != {
                "nd-2": _sha256_file(fixed_paths["nd2_node_report"]),
                "nd-3": _sha256_file(fixed_paths["nd3_node_report"]),
            }):
        raise ValueError("HGP offline terminal package file binding changed")
    _validate_terminal_raw_evidence(run_root, control, config, package)

    measurement_bound_files = {
        name: fixed_paths[name] for name in (
            "schedule", "artifact_manifest", "aggregate_preflight",
            "local_attestation", "measurement_control", "nd2_node_report",
            "nd3_node_report", "terminal_report", "terminal_decision",
            "terminal_package",
        )
    }
    measurement_bound_identity = {
        "schedule_sha256": schedule["schedule_sha256"],
        "selected_resource_tier": preflight.get("selected_resource_tier"),
        "local_attestation_sha256": local_attestation["attestation_sha256"],
        "terminal_status": package.get("status"),
        "terminal_package_sha256": package.get("package_sha256"),
    }
    manifest = _read_canonical_json(
        path, "HGP offline measurement acceptance manifest",
    )
    manifest_identity = dict(manifest)
    stored_manifest_sha = manifest_identity.pop("manifest_sha256", None)
    clock_sha = hashlib.sha256(
        _canonical_json(clock).encode("ascii")
    ).hexdigest()
    deadline = (
        clock["boottime_before_ns"] + 24 * 3600 * NANOSECONDS_PER_SECOND
    )
    published = manifest.get("published_boottime_ns")
    launch = manifest.get("launch_metadata")
    expected_bound_hashes = {
        name: _sha256_file(bound_path)
        for name, bound_path in sorted(measurement_bound_files.items())
    }
    expected_prior = {
        "manifest_relpath": "control/HGP_ND0_PREFLIGHT_ACCEPTANCE.json",
        "manifest_file_sha256": _sha256_file(
            fixed_paths["preflight_acceptance"],
        ),
        "manifest_sha256": preflight_acceptance["manifest_sha256"],
    }
    canonical_launch_file_sha = hashlib.sha256(
        (_canonical_json(launch) + "\n").encode("ascii")
    ).hexdigest() if isinstance(launch, dict) else None
    if (set(manifest) != ACCEPTANCE_MANIFEST_FIELDS
            or manifest.get("manifest_version")
            != ACCEPTANCE_MANIFEST_VERSION
            or manifest.get("phase") != "measurement"
            or manifest.get("run_id") != schedule.get("run_id")
            or manifest.get("source_commit") != source_commit
            or manifest.get("archive_sha256") != archive_sha256
            or manifest.get("source_manifest_sha256")
            != source_manifest_sha256
            or manifest.get("clock_authority_sha256") != clock_sha
            or manifest.get("clock_authority_boot_id")
            != clock["clock_authority_boot_id"]
            or manifest.get("phase_deadline_boottime_ns") != deadline
            or isinstance(published, bool) or not isinstance(published, int)
            or published < clock["boottime_after_ns"]
            or published >= deadline
            or not isinstance(launch, dict)
            or set(launch) != LAUNCH_METADATA_FIELDS
            or launch.get("phase") != "measurement"
            or launch.get("run_id") != schedule.get("run_id")
            or launch.get("source_commit") != source_commit
            or launch.get("archive_sha256") != archive_sha256
            or launch.get("manifest_sha256") != source_manifest_sha256
            or launch.get("local_attestation_sha256")
            != local_attestation_sha
            or launch.get("launcher_version") != ND0_LAUNCHER_VERSION
            or SHA256_RE.fullmatch(str(launch.get("command_sha256", "")))
            is None
            or manifest.get("launch_metadata_file_sha256")
            != canonical_launch_file_sha
            or manifest.get("bound_file_sha256") != expected_bound_hashes
            or manifest.get("bound_identity") != measurement_bound_identity
            or manifest.get("prior_manifest") != expected_prior
            or stored_manifest_sha != hashlib.sha256(
                _canonical_json(manifest_identity).encode("ascii")
            ).hexdigest()):
        raise ValueError("HGP offline measurement acceptance identity changed")

    rows = manifest.get("stage_acceptances")
    if (not isinstance(rows, list)
            or [row.get("stage_key") for row in rows]
            != list(PHASE_STAGE_KEYS["measurement"])
            or any(set(row) != {
                "stage_key", "acceptance_relpath", "acceptance_sha256",
                "acceptance_file_sha256",
            } for row in rows)):
        raise ValueError("HGP offline measurement acceptance order changed")
    acceptance_dir = run_root / "hgp_global/nd0_acceptance/measurement"
    expected_names = {
        f"{key}.json" for key in PHASE_STAGE_KEYS["measurement"]
    }
    if (not acceptance_dir.is_dir() or acceptance_dir.is_symlink()
            or {item.name for item in acceptance_dir.iterdir()}
            != expected_names):
        raise ValueError("HGP offline measurement acceptance file set changed")
    for row in rows:
        key = row["stage_key"]
        phase, stage_name, node, deadline_kind, seconds = (
            STAGE_ACCEPTANCE_SPECS[key]
        )
        relative = f"hgp_global/nd0_acceptance/measurement/{key}.json"
        acceptance_path = run_root / relative
        if (row.get("acceptance_relpath") != relative
                or not acceptance_path.is_file()
                or acceptance_path.is_symlink()
                or row.get("acceptance_file_sha256")
                != _sha256_file(acceptance_path)):
            raise ValueError("HGP offline measurement acceptance file changed")
        acceptance = _read_json(acceptance_path)
        acceptance_identity = dict(acceptance)
        acceptance_sha = acceptance_identity.pop("acceptance_sha256", None)
        expected_deadline = (
            clock["boottime_before_ns"] + seconds * NANOSECONDS_PER_SECOND
        )
        observed = acceptance.get("observed_boottime_ns")
        success_relative = f"hgp_global/markers/{key}/SUCCESS"
        success_path = run_root / success_relative
        try:
            _require_regular_file_in_root(
                acceptance_path, run_root,
                "HGP offline measurement stage acceptance",
            )
            _require_regular_file_in_root(
                success_path, run_root,
                "HGP offline measurement SUCCESS marker",
            )
            expected_prerequisites = [
                _sha256_file(_require_regular_file_in_root(
                    run_root / f"hgp_global/markers/{prerequisite}/SUCCESS",
                    run_root, "HGP offline prerequisite SUCCESS marker",
                ))
                for prerequisite in STAGE_PREREQUISITE_KEYS[key]
            ]
        except ValueError as exc:
            raise ValueError(
                "HGP offline measurement stage evidence is invalid"
            ) from exc
        if (set(acceptance) != STAGE_ACCEPTANCE_FIELDS
                or acceptance.get("acceptance_version")
                != STAGE_ACCEPTANCE_VERSION
                or acceptance.get("run_id") != schedule.get("run_id")
                or acceptance.get("phase") != phase
                or acceptance.get("source_commit") != source_commit
                or acceptance.get("archive_sha256") != archive_sha256
                or acceptance.get("source_manifest_sha256")
                != source_manifest_sha256
                or acceptance.get("clock_authority_sha256") != clock_sha
                or acceptance.get("clock_authority_boot_id")
                != clock["clock_authority_boot_id"]
                or acceptance.get("stage_key") != key
                or acceptance.get("stage") != stage_name
                or acceptance.get("node") != node
                or acceptance.get("deadline_kind") != deadline_kind
                or acceptance.get("deadline_boottime_ns") != expected_deadline
                or isinstance(observed, bool) or not isinstance(observed, int)
                or observed < clock["boottime_after_ns"]
                or observed >= expected_deadline
                or acceptance.get("success_relpath") != success_relative
                or acceptance.get("success_file_sha256")
                != _sha256_file(success_path)
                or acceptance.get("success_marker") != _read_json(success_path)
                or acceptance_sha != row.get("acceptance_sha256")
                or acceptance_sha != hashlib.sha256(
                    _canonical_json(acceptance_identity).encode("ascii")
                ).hexdigest()):
            raise ValueError("HGP offline measurement stage identity changed")
        marker = _validate_success_value(
            acceptance["success_marker"], success_path,
            expected_stage=stage_name, source_commit=source_commit,
        )
        if marker["prerequisite_success_sha256"] != expected_prerequisites:
            raise ValueError("HGP offline measurement prerequisite graph changed")

    return _joint_terminal_result(
        schedule, package, fixed_paths["preflight_acceptance"],
        preflight_acceptance, path, manifest,
    )


def _validate_schedule_output(
        path, run_id, source_commit, archive_sha256,
        source_manifest_sha256, config_file_sha256, *,
        check_current_boot=True):
    schedule = _read_json(path)
    unix_fields = (
        "started_unix", "preflight_deadline_unix",
        "control_freeze_deadline_unix", "screen_deadline_unix",
        "analysis_deadline_unix",
    )
    boottime_fields = (
        "started_boottime_ns", "preflight_deadline_boottime_ns",
        "control_freeze_deadline_boottime_ns",
        "screen_deadline_boottime_ns", "analysis_deadline_boottime_ns",
    )
    clock_authority = _validate_clock_authority(
        schedule.get("clock_authority")
    )
    identity = dict(schedule)
    stored_sha = identity.pop("schedule_sha256", None)
    if (schedule.get("schedule_version") != SCHEDULE_VERSION
            or schedule.get("run_id") != run_id
            or schedule.get("source_commit") != source_commit
            or schedule.get("archive_sha256") != archive_sha256
            or schedule.get("source_manifest_sha256")
            != source_manifest_sha256
            or schedule.get("config_file_sha256") != config_file_sha256
            or schedule.get("source_identity", {}).get("mode") != "archive"
            or schedule.get("source_identity", {}).get("source_commit")
            != source_commit
            or schedule.get("source_identity", {}).get("archive_sha256")
            != archive_sha256
            or schedule.get("source_identity", {}).get("manifest_sha256")
            != source_manifest_sha256
            or stored_sha != hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest()
            or any(isinstance(schedule.get(name), bool)
                   or not isinstance(schedule.get(name), int)
                   for name in (*unix_fields, *boottime_fields))
            or not all(float(schedule[left]) < float(schedule[right])
                       for left, right in zip(
                           unix_fields, unix_fields[1:],
                       ))
            or not all(schedule[left] < schedule[right]
                       for left, right in zip(
                           boottime_fields, boottime_fields[1:],
                       ))
            or schedule["started_unix"]
            != clock_authority["authority_unix_ns"]
            // NANOSECONDS_PER_SECOND
            or schedule["started_boottime_ns"]
            != clock_authority["boottime_before_ns"]
            or any(
                schedule[f"{name}_deadline_unix"]
                - schedule["started_unix"] != hours * 3600
                or schedule[f"{name}_deadline_boottime_ns"]
                - schedule["started_boottime_ns"]
                != hours * 3600 * NANOSECONDS_PER_SECOND
                for name, hours in (
                    ("preflight", 6), ("control_freeze", 8),
                    ("screen", 22), ("analysis", 24),
                )
            )
            or (check_current_boot and _current_boot_id()
                != clock_authority["clock_authority_boot_id"])):
        raise ValueError("HGP frozen schedule output is invalid")
    return schedule


def _validate_aggregate_preflight(path, schedule, config, source_commit,
                                  archive_sha256, source_manifest_sha256,
                                  config_file_sha256):
    report = _read_json(path)
    tier = report.get("selected_resource_tier")
    if (report.get("status") != "PASS"
            or tier not in config.get("resource_tiers", {})
            or report.get("source_commit") != source_commit
            or report.get("archive_sha256") != archive_sha256
            or report.get("source_manifest_sha256")
            != source_manifest_sha256
            or report.get("config_file_sha256") != config_file_sha256
            or report.get("schedule_sha256") != schedule["schedule_sha256"]
            or not isinstance(report.get("source_identity"), dict)
            or report["source_identity"].get("mode") != "archive"
            or report["source_identity"].get("source_commit") != source_commit
            or report["source_identity"].get("archive_sha256")
            != archive_sha256
            or report["source_identity"].get("manifest_sha256")
            != source_manifest_sha256
            or report.get("selection_basis")
            != "runtime_only_worst_node_and_frozen_elapsed_deadlines"
            or report.get("clock_domain")
            != "unsynchronized_local_diagnostic"
            or isinstance(report.get("completed_local_unix"), bool)
            or not isinstance(
                report.get("completed_local_unix"), (int, float),
            )
            or not math.isfinite(float(report["completed_local_unix"]))):
        raise RuntimeError(
            "HGP aggregate preflight is not a PASS authority for measurement"
        )
    return report


def _validate_control_output(path, preflight, schedule, config, source_commit,
                             archive_sha256, source_manifest_sha256):
    control = _read_json(path)
    expected_count = int(config["task_counts"]["total_measurement"])
    if (control.get("resource_tier")
            != preflight.get("selected_resource_tier")
            or control.get("source_commit") != source_commit
            or control.get("archive_sha256") != archive_sha256
            or control.get("source_manifest_sha256")
            != source_manifest_sha256
            or int(control.get("task_count", -1)) != expected_count
            or len(control.get("tasks", ())) != expected_count
            or tuple(control.get("execution_nodes", ())) != EXECUTION_NODES):
        raise RuntimeError("HGP frozen measurement control is invalid")
    return control


def _validate_attestation_decisions(payload, context):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} portable payload is invalid")
    catalog = payload.get("acceptance_decision_catalog")
    digest = payload.get("acceptance_decision_catalog_sha256")
    if (not isinstance(catalog, list) or not catalog
            or SHA256_RE.fullmatch(str(digest)) is None):
        raise ValueError(f"{context} acceptance decisions are invalid")
    seen = set()
    for entry in catalog:
        if (not isinstance(entry, dict)
                or set(entry) != LOCAL_DECISION_FIELDS
                or SHA256_RE.fullmatch(str(entry.get(
                    "cell_fingerprint",
                ))) is None
                or entry.get("method_id") != "MAM-IMH8"
                or entry.get("init_family") not in {"P", "U"}
                or SHA256_RE.fullmatch(str(entry.get(
                    "acceptance_decision_sha256",
                ))) is None):
            raise ValueError(f"{context} acceptance catalog is invalid")
        identity = (
            entry["cell_fingerprint"], entry["method_id"],
            entry["init_family"],
        )
        if identity in seen:
            raise ValueError(f"{context} acceptance catalog is duplicated")
        seen.add(identity)
    if _sha256_json(catalog) != digest:
        raise ValueError(f"{context} acceptance decision SHA is invalid")
    return catalog, digest


def _validate_attestation_is_summary(summary, fields, context):
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"{context} IS summary is invalid")
    seen = set()
    for entry in summary:
        if (not isinstance(entry, dict) or set(entry) != fields
                or any(SHA256_RE.fullmatch(str(entry[name])) is None
                       for name in fields)):
            raise ValueError(f"{context} IS summary schema is invalid")
        fingerprint = entry["cell_fingerprint"]
        if fingerprint in seen:
            raise ValueError(f"{context} IS summary is duplicated")
        seen.add(fingerprint)
    return summary


def _attestation_portable_is_projection(summary):
    return [{
        name: entry[name] for name in sorted(LOCAL_IS_PORTABLE_FIELDS)
    } for entry in summary]


def _validate_local_attestation(
        path, expected_file_sha256, schedule, preflight,
        artifact_manifest_path, preflight_acceptance_manifest_path,
        registry_file_sha256, config_file_sha256,
        source_commit,
        archive_sha256, source_manifest_sha256):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("HGP local attestation must be a regular file")
    path = path.resolve(strict=True)
    if (SHA256_RE.fullmatch(str(expected_file_sha256)) is None
            or _sha256_file(path) != expected_file_sha256):
        raise ValueError("HGP local attestation file SHA mismatch")
    value = _read_json(path)
    expected_fields = {
        "attestation_version", "status", "source_commit", "archive_sha256",
        "source_manifest_sha256", "registry_file_sha256",
        "config_file_sha256", "schedule_sha256", "schedule_file_sha256",
        "artifact_manifest_sha256", "artifact_manifest_file_sha256",
        "preflight_file_sha256",
        "preflight_acceptance_manifest_file_sha256",
        "preflight_acceptance_manifest_sha256",
        "remote_full_payload_sha256", "local_full_payload_sha256",
        "remote_portable_payload_sha256", "local_portable_payload_sha256",
        "exact_canonical_match", "portable_canonical_match",
        "acceptance_decisions_exact", "acceptance_decision_catalog_sha256",
        "remote_full_consensus", "remote_full_consensus_nodes",
        "mismatch_paths", "importance_sampling_portable_summary",
        "remote_importance_sampling_full_summary",
        "local_importance_sampling_full_summary",
        "solver_identity_policy", "full_mismatch_policy",
        "local_environment", "clock_domain", "completed_local_unix",
        "attestation_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("HGP local attestation schema is invalid")
    identity = dict(value)
    stored_sha = identity.pop("attestation_sha256", None)
    status = value.get("status")
    exact = value.get("exact_canonical_match")
    remote_full = preflight.get("canonical_full_payload")
    remote_portable = preflight.get("canonical_portable_payload")
    remote_full_sha = preflight.get("canonical_full_payload_sha256")
    remote_portable_sha = preflight.get("canonical_portable_payload_sha256")
    if (not isinstance(remote_full, dict)
            or not isinstance(remote_portable, dict)
            or SHA256_RE.fullmatch(str(remote_full_sha)) is None
            or SHA256_RE.fullmatch(str(remote_portable_sha)) is None
            or _sha256_json(remote_full) != remote_full_sha
            or _sha256_json(remote_portable) != remote_portable_sha):
        raise ValueError("HGP preflight canonical evidence is invalid")
    full_decisions, full_decision_sha = _validate_attestation_decisions(
        remote_full, "remote full",
    )
    portable_decisions, portable_decision_sha = (
        _validate_attestation_decisions(remote_portable, "remote portable")
    )
    if (full_decisions != portable_decisions
            or full_decision_sha != portable_decision_sha):
        raise ValueError("HGP preflight full/portable decisions disagree")
    expected_remote_is = _validate_attestation_is_summary(
        remote_full.get(LOCAL_IS_SUMMARY_KEY), LOCAL_IS_FULL_FIELDS,
        "remote full",
    )
    expected_portable_is = _validate_attestation_is_summary(
        remote_portable.get(LOCAL_IS_SUMMARY_KEY), LOCAL_IS_PORTABLE_FIELDS,
        "remote portable",
    )
    if _attestation_portable_is_projection(
            expected_remote_is) != expected_portable_is:
        raise ValueError("HGP preflight full/portable IS summaries disagree")
    local_full_is = _validate_attestation_is_summary(
        value.get("local_importance_sampling_full_summary"),
        LOCAL_IS_FULL_FIELDS, "local full",
    )
    environment = value.get("local_environment")
    mismatch_paths = value.get("mismatch_paths")
    common_valid = (
        value.get("attestation_version") == LOCAL_ATTESTATION_VERSION
        and status in LOCAL_ATTESTATION_STATUSES
        and value.get("source_commit") == source_commit
        and value.get("archive_sha256") == archive_sha256
        and value.get("source_manifest_sha256") == source_manifest_sha256
        and value.get("registry_file_sha256") == registry_file_sha256
        and value.get("config_file_sha256") == config_file_sha256
        and value.get("schedule_sha256") == schedule["schedule_sha256"]
        and value.get("schedule_file_sha256")
        == _sha256_file(Path(path).parent / "HGP_GLOBAL_24H_SCHEDULE.json")
        and value.get("artifact_manifest_file_sha256")
        == _sha256_file(artifact_manifest_path)
        and value.get("artifact_manifest_sha256")
        == _read_json(artifact_manifest_path).get("artifact_manifest_sha256")
        and value.get("preflight_file_sha256")
        == _sha256_file(Path(path).parent / "hgp_preflight.json")
        and value.get("preflight_acceptance_manifest_file_sha256")
        == _sha256_file(preflight_acceptance_manifest_path)
        and value.get("preflight_acceptance_manifest_sha256")
        == _read_json(preflight_acceptance_manifest_path).get(
            "manifest_sha256"
        )
        and value.get("remote_full_payload_sha256") == remote_full_sha
        and SHA256_RE.fullmatch(str(value.get(
            "local_full_payload_sha256",
        ))) is not None
        and value.get("remote_portable_payload_sha256") == remote_portable_sha
        and value.get("local_portable_payload_sha256") == remote_portable_sha
        and value.get("portable_canonical_match") is True
        and value.get("acceptance_decisions_exact") is True
        and value.get("acceptance_decision_catalog_sha256")
        == portable_decision_sha
        and preflight.get("remote_full_consensus") is True
        and tuple(preflight.get("nodes", ()))
        == LOCAL_FULL_CONSENSUS_NODES
        and value.get("remote_full_consensus") is True
        and tuple(value.get("remote_full_consensus_nodes", ()))
        == LOCAL_FULL_CONSENSUS_NODES
        and isinstance(mismatch_paths, list)
        and all(isinstance(item, str) and item for item in mismatch_paths)
        and value.get("importance_sampling_portable_summary")
        == expected_portable_is
        and value.get("remote_importance_sampling_full_summary")
        == expected_remote_is
        and _attestation_portable_is_projection(local_full_is)
        == expected_portable_is
        and value.get("solver_identity_policy") == LOCAL_SOLVER_POLICY
        and value.get("full_mismatch_policy")
        == LOCAL_FULL_MISMATCH_POLICY
        and isinstance(environment, dict)
        and set(environment) == {
            "system", "machine", "python", "numpy", "scipy",
            "map_solver_identity_current",
        }
        and all(isinstance(item, str) and item for item in environment.values())
        and value.get("clock_domain")
        == "unsynchronized_local_diagnostic"
        and not isinstance(value.get("completed_local_unix"), bool)
        and isinstance(value.get("completed_local_unix"), (int, float))
        and math.isfinite(float(value["completed_local_unix"]))
        and stored_sha == hashlib.sha256(
            _canonical_json(identity).encode("ascii")
        ).hexdigest()
    )
    if not common_valid:
        raise ValueError("HGP local attestation identity is invalid")
    local_full_sha = value["local_full_payload_sha256"]
    if status == "PASS_EXACT":
        consistent = (
            exact is True and local_full_sha == remote_full_sha
            and mismatch_paths == []
        )
    else:
        consistent = (
            exact is False and local_full_sha != remote_full_sha
            and bool(mismatch_paths)
        )
    if not consistent:
        raise ValueError("HGP local attestation status is inconsistent")
    return value


def _validate_terminal_output(path, source_commit, schedule):
    package = _read_json(path)
    identity = dict(package)
    stored_sha = identity.pop("package_sha256", None)
    source_identity = package.get("source_identity")
    raw_files = package.get("raw_files")
    raw_count = package.get("raw_file_count")
    try:
        raw_evidence_summary = _terminal_raw_evidence_summary(raw_files)
    except ValueError as exc:
        raise ValueError("HGP terminal package is invalid") from exc
    file_hash_fields = (
        "schedule_file_sha256", "artifact_manifest_file_sha256",
        "preflight_file_sha256", "control_file_sha256",
        "report_file_sha256", "decision_file_sha256", "decision_sha256",
    )
    if (set(package) != TERMINAL_PACKAGE_FIELDS
            or package.get("package_version") != TERMINAL_PACKAGE_VERSION
            or package.get("contract_version") != HGP_CONTRACT_VERSION
            or not isinstance(source_identity, dict)
            or source_identity.get("mode") != "archive"
            or source_identity.get("source_commit")
            != source_commit
            or source_identity.get("archive_sha256")
            != schedule.get("archive_sha256")
            or source_identity.get("manifest_sha256")
            != schedule.get("source_manifest_sha256")
            or package.get("schedule_sha256") != schedule["schedule_sha256"]
            or package.get("status") not in TERMINAL_STATUSES
            or any(SHA256_RE.fullmatch(str(package.get(name, ""))) is None
                   for name in file_hash_fields)
            or not isinstance(
                package.get("execution_report_file_sha256"), dict,
            )
            or set(package["execution_report_file_sha256"])
            != set(EXECUTION_NODES)
            or any(SHA256_RE.fullmatch(str(value)) is None for value in
                   package["execution_report_file_sha256"].values())
            or isinstance(raw_count, bool) or not isinstance(raw_count, int)
            or raw_count < 0 or not isinstance(raw_files, list)
            or raw_count != len(raw_files)
            or not isinstance(package.get("raw_evidence_summary"), dict)
            or set(package["raw_evidence_summary"])
            != TERMINAL_RAW_EVIDENCE_SUMMARY_FIELDS
            or any(SHA256_RE.fullmatch(str(value)) is None for value in
                   package["raw_evidence_summary"].values())
            or package["raw_evidence_summary"] != raw_evidence_summary
            or package.get("formal_authorization") is not False
            or package.get("production_authorization") is not False
            or stored_sha != hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest()):
        raise ValueError("HGP terminal package is invalid")
    return package


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("preflight", "measurement"),
        default="preflight",
    )
    parser.add_argument("--local-attestation")
    parser.add_argument("--local-attestation-sha256")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not 0.1 <= float(args.poll_seconds) <= 60.0:
        raise ValueError("HGP orchestrator poll interval is invalid")
    if args.phase == "preflight" and (
            args.local_attestation is not None
            or args.local_attestation_sha256 is not None):
        raise ValueError("HGP preflight cannot accept a local attestation")
    if args.phase == "measurement" and (
            args.local_attestation is None
            or args.local_attestation_sha256 is None):
        raise ValueError("HGP measurement requires local attestation path and SHA")
    deployment_root, run_root, launch_guard = _require_verified_launch(
        args, Path.home(),
    )
    config, config_file_sha256 = _read_frozen_config(
        deployment_root, args.archive_sha256,
    )
    if args.phase == "preflight":
        clock_authority = _capture_clock_authority()
        print(_canonical_json({
            "event": "clock_authority_frozen",
            "clock_authority": clock_authority,
            "clock_authority_sha256": hashlib.sha256(
                _canonical_json(clock_authority).encode("ascii")
            ).hexdigest(),
        }), flush=True)
    else:
        frozen_schedule = _read_json(
            Path(run_root) / "control/HGP_GLOBAL_24H_SCHEDULE.json"
        )
        clock_authority = _validate_clock_authority(
            frozen_schedule.get("clock_authority")
        )
    orchestrator = HgpOrchestrator(
        run_id=args.run_id, source_commit=args.source_commit,
        archive_sha256=args.archive_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        deployment_root=deployment_root, run_root=run_root, config=config,
        config_file_sha256=config_file_sha256,
        clock_authority=clock_authority,
        local_attestation=args.local_attestation,
        local_attestation_sha256=args.local_attestation_sha256,
        launch_guard=launch_guard,
        poll_seconds=args.poll_seconds,
    )
    orchestrator.run(args.phase)


if __name__ == "__main__":
    main()
