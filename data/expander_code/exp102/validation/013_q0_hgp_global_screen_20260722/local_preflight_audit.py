"""Replay the frozen remote HGP preflight under local conda 12.

The command emits a launch-authority attestation only when the complete
canonical digest is exactly equal.  A mismatch is recorded for a later,
field-specific portability review; this command does not guess an ULP
tolerance or silently downgrade the comparison.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


VALIDATION_ROOT = (
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722"
)
workflow = importlib.import_module(VALIDATION_ROOT + ".workflow")
pipeline = importlib.import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_hgp_screen"
)

ATTESTATION_VERSION = "exp102.q0_hgp_global.screen.local_attestation.v1"
SOLVER_POLICY = "stored_generation_identity_exact_artifact_replay_no_local_milp"


def _mismatch_paths(left, right, path="$", limit=64):
    result = []

    def visit(a, b, current):
        if len(result) >= limit:
            return
        if type(a) is not type(b):
            result.append(current + ":type")
        elif isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for key in keys:
                if key not in a or key not in b:
                    result.append(f"{current}.{key}:missing")
                else:
                    visit(a[key], b[key], f"{current}.{key}")
                if len(result) >= limit:
                    break
        elif isinstance(a, list):
            if len(a) != len(b):
                result.append(current + ":length")
            for index, (left_value, right_value) in enumerate(zip(a, b)):
                visit(left_value, right_value, f"{current}[{index}]")
                if len(result) >= limit:
                    break
        elif a != b:
            result.append(current)

    visit(left, right, path)
    return result


def _environment_identity():
    import scipy

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "map_solver_identity_current": pipeline._solver_identity(),
    }


def audit_local_preflight(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, schedule_path, artifact_root,
        artifact_manifest_path, preflight_path, work_root, output_path):
    registry_path = Path(registry_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    schedule_path = Path(schedule_path).resolve(strict=True)
    artifact_root = Path(artifact_root).resolve(strict=True)
    artifact_manifest_path = Path(artifact_manifest_path).resolve(strict=True)
    preflight_path = Path(preflight_path).resolve(strict=True)
    work_root = Path(work_root).resolve()
    output_path = Path(output_path).resolve()
    if work_root.exists() or output_path.exists():
        raise FileExistsError("local HGP audit requires fresh work/output paths")
    if os.environ.get("EXP102_SOURCE_COMMIT") != source_commit:
        raise ValueError("local HGP audit must run from the verified source archive")

    registry = workflow._load_registry(registry_path)
    config = workflow._load_config(config_path, registry)
    schedule, _ = workflow._validate_schedule(
        schedule_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, config,
    )
    artifact_manifest, _ = workflow._validate_artifact_manifest(
        artifact_manifest_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root, schedule,
        schedule_path,
    )
    remote_preflight = workflow._validate_preflight(
        preflight_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_manifest_path,
        artifact_root, config, schedule, schedule_path,
    )
    if remote_preflight.get("status") != "PASS":
        raise RuntimeError("remote aggregate preflight is not PASS")

    work_root.mkdir(parents=True)
    local_payload = pipeline.hgp_screen_preflight_digest(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )
    is_root = work_root / "importance_sampling"
    is_root.mkdir()
    is_digests = []
    for cell in pipeline._map_cells(config):
        fingerprint = pipeline._cell_fingerprint(cell)
        path = is_root / f"{fingerprint}.npz"
        generated = pipeline.run_hgp_map_is_diagnostic(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root, path,
        )
        validated = pipeline.validate_hgp_map_is_diagnostic(
            path, registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root,
        )
        if generated["transcript_sha256"] != validated["transcript_sha256"]:
            raise RuntimeError("local HGP IS generation/replay digest changed")
        is_digests.append({
            "cell_fingerprint": fingerprint,
            "transcript_sha256": validated["transcript_sha256"],
        })
    local_payload["importance_sampling_transcript_sha256"] = is_digests
    remote_payload = remote_preflight["canonical_digest"]
    exact = workflow._canonical_json(local_payload) == workflow._canonical_json(
        remote_payload,
    )
    local_digest = workflow._sha256_json(local_payload)
    remote_digest = workflow._sha256_json(remote_payload)
    if remote_digest != remote_preflight["canonical_digest_sha256"]:
        raise ValueError("remote preflight canonical digest self-hash changed")

    comparison = {
        "comparison_version": "exp102.q0_hgp_global.screen.local_compare.v1",
        "remote_payload": remote_payload,
        "local_payload": local_payload,
        "mismatch_paths": _mismatch_paths(remote_payload, local_payload),
    }
    comparison_path = work_root / "canonical_comparison.json"
    workflow._write_exclusive_json(comparison_path, comparison)
    completed_unix = time.time()
    identity = {
        "attestation_version": ATTESTATION_VERSION,
        "status": "PASS" if exact else "MISMATCH_REQUIRES_PORTABILITY_REVIEW",
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_file_sha256": workflow._sha256_file(registry_path),
        "config_file_sha256": workflow._sha256_file(config_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": workflow._sha256_file(schedule_path),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "artifact_manifest_file_sha256": workflow._sha256_file(
            artifact_manifest_path,
        ),
        "preflight_file_sha256": workflow._sha256_file(preflight_path),
        "remote_canonical_digest_sha256": remote_digest,
        "local_canonical_digest_sha256": local_digest,
        "exact_canonical_match": bool(exact),
        "mismatch_paths": comparison["mismatch_paths"],
        "importance_sampling_transcript_sha256": is_digests,
        "solver_identity_policy": SOLVER_POLICY,
        "local_environment": _environment_identity(),
        "portability_review": None,
        "completed_unix": completed_unix,
    }
    attestation = {
        **identity, "attestation_sha256": workflow._sha256_json(identity),
    }
    workflow._write_exclusive_json(output_path, attestation)
    return attestation


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    attestation = audit_local_preflight(
        args.registry, args.config, args.source_commit, args.archive_sha256,
        args.source_manifest_sha256, args.schedule, args.artifact_root,
        args.artifact_manifest, args.preflight, args.work_root, args.output,
    )
    print(workflow._canonical_json({
        "status": attestation["status"],
        "attestation": str(Path(args.output).resolve()),
        "attestation_sha256": attestation["attestation_sha256"],
        "mismatch_paths": attestation["mismatch_paths"],
    }))
    if attestation["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
