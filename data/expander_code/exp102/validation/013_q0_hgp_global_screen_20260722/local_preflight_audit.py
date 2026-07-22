"""Replay the frozen remote HGP preflight under local conda 12.

The v4 attestation keeps remote full replay and cross-platform portability as
separate evidence.  A portable pass requires exact portable and acceptance-
decision transcripts; full-payload differences remain diagnostic only.
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
orchestration = importlib.import_module(VALIDATION_ROOT + ".orchestrate_hgp")
pipeline = importlib.import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_hgp_screen"
)
map_pipeline = importlib.import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_map_mixture"
)

ATTESTATION_VERSION = "exp102.q0_hgp_global.screen.local_attestation.v4"
COMPARISON_VERSION = "exp102.q0_hgp_global.screen.local_compare.v2"
SOLVER_POLICY = "stored_generation_identity_exact_artifact_replay_no_local_milp"
FULL_MISMATCH_POLICY = (
    "diagnostic_only_after_remote_full_consensus_and_exact_portable_decisions"
)
REMOTE_FULL_CONSENSUS_NODES = ("nd-1", "nd-2", "nd-3")
EVIDENCE_BUNDLE_FIELDS = {
    "canonical_full_payload", "canonical_full_payload_sha256",
    "canonical_portable_payload", "canonical_portable_payload_sha256",
}
IS_SUMMARY_KEY = "importance_sampling_transcript_summary"
IS_FULL_SUMMARY_FIELDS = {
    "cell_fingerprint", "full_transcript_sha256",
    "portable_transcript_sha256", "nonportable_float_sha256",
    "field_manifest_sha256",
}
IS_PORTABLE_SUMMARY_FIELDS = {
    "cell_fingerprint", "portable_transcript_sha256",
    "field_manifest_sha256",
}
DECISION_CATALOG_FIELDS = {
    "cell_fingerprint", "method_id", "init_family",
    "acceptance_decision_sha256",
}


def _is_sha256(value):
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_decision_evidence(payload, *, context):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} portable payload is not an object")
    catalog = payload.get("acceptance_decision_catalog")
    digest = payload.get("acceptance_decision_catalog_sha256")
    if not isinstance(catalog, list) or not catalog or not _is_sha256(digest):
        raise ValueError(f"{context} acceptance-decision evidence is invalid")
    seen = set()
    for entry in catalog:
        if (not isinstance(entry, dict)
                or set(entry) != DECISION_CATALOG_FIELDS
                or not _is_sha256(entry.get("cell_fingerprint"))
                or entry.get("method_id") != "MAM-IMH8"
                or entry.get("init_family") not in {"P", "U"}
                or not _is_sha256(entry.get("acceptance_decision_sha256"))):
            raise ValueError(
                f"{context} acceptance-decision catalog is invalid",
            )
        identity = (
            entry["cell_fingerprint"], entry["method_id"],
            entry["init_family"],
        )
        if identity in seen:
            raise ValueError(
                f"{context} acceptance-decision catalog is duplicated",
            )
        seen.add(identity)
    if workflow._sha256_json(catalog) != digest:
        raise ValueError(f"{context} acceptance-decision SHA is invalid")
    return catalog, digest


def _validate_is_summary(summary, *, portable, context):
    fields = (
        IS_PORTABLE_SUMMARY_FIELDS if portable else IS_FULL_SUMMARY_FIELDS
    )
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"{context} IS summary is invalid")
    seen = set()
    for entry in summary:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"{context} IS summary schema is invalid")
        if any(not _is_sha256(entry[name]) for name in fields):
            raise ValueError(f"{context} IS summary SHA is invalid")
        fingerprint = entry["cell_fingerprint"]
        if fingerprint in seen:
            raise ValueError(f"{context} IS summary is duplicated")
        seen.add(fingerprint)
    return summary


def _portable_is_projection(summary):
    return [{
        name: entry[name] for name in sorted(IS_PORTABLE_SUMMARY_FIELDS)
    } for entry in summary]


def _validate_evidence_bundle(bundle, *, context, require_is):
    if not isinstance(bundle, dict) or set(bundle) != EVIDENCE_BUNDLE_FIELDS:
        raise ValueError(f"{context} canonical evidence bundle is invalid")
    full_payload = bundle["canonical_full_payload"]
    portable_payload = bundle["canonical_portable_payload"]
    full_sha = bundle["canonical_full_payload_sha256"]
    portable_sha = bundle["canonical_portable_payload_sha256"]
    if (not isinstance(full_payload, dict)
            or not isinstance(portable_payload, dict)
            or not _is_sha256(full_sha)
            or not _is_sha256(portable_sha)
            or workflow._sha256_json(full_payload) != full_sha
            or workflow._sha256_json(portable_payload) != portable_sha):
        raise ValueError(f"{context} canonical evidence self-hash is invalid")
    full_catalog, full_decision_sha = _validate_decision_evidence(
        full_payload, context=f"{context} full",
    )
    portable_catalog, portable_decision_sha = _validate_decision_evidence(
        portable_payload, context=f"{context} portable",
    )
    if (full_catalog != portable_catalog
            or full_decision_sha != portable_decision_sha):
        raise ValueError(f"{context} full/portable decisions disagree")
    if require_is:
        full_is = _validate_is_summary(
            full_payload.get(IS_SUMMARY_KEY), portable=False,
            context=f"{context} full",
        )
        portable_is = _validate_is_summary(
            portable_payload.get(IS_SUMMARY_KEY), portable=True,
            context=f"{context} portable",
        )
        if _portable_is_projection(full_is) != portable_is:
            raise ValueError(f"{context} full/portable IS summaries disagree")
    elif IS_SUMMARY_KEY in full_payload or IS_SUMMARY_KEY in portable_payload:
        raise ValueError(f"{context} base evidence unexpectedly contains IS")
    return bundle


def _attach_is_summaries(bundle, full_summary):
    _validate_evidence_bundle(bundle, context="local base", require_is=False)
    full_payload = dict(bundle["canonical_full_payload"])
    portable_payload = dict(bundle["canonical_portable_payload"])
    full_payload[IS_SUMMARY_KEY] = full_summary
    portable_payload[IS_SUMMARY_KEY] = _portable_is_projection(full_summary)
    result = {
        "canonical_full_payload": full_payload,
        "canonical_full_payload_sha256": workflow._sha256_json(full_payload),
        "canonical_portable_payload": portable_payload,
        "canonical_portable_payload_sha256": workflow._sha256_json(
            portable_payload,
        ),
    }
    return _validate_evidence_bundle(
        result, context="local complete", require_is=True,
    )


def _is_full_summary_entry(fingerprint, generated, validated):
    if not _is_sha256(fingerprint):
        raise ValueError("local IS cell fingerprint is invalid")
    digest_fields = IS_FULL_SUMMARY_FIELDS - {"cell_fingerprint"}
    generated_summary = {
        name: generated.get(name) for name in digest_fields
    }
    validated_summary = {
        name: validated.get(name) for name in digest_fields
    }
    if (generated_summary != validated_summary
            or any(not _is_sha256(value)
                   for value in generated_summary.values())):
        raise RuntimeError("local HGP IS generation/full replay changed")
    return {"cell_fingerprint": fingerprint, **generated_summary}


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
        "map_solver_identity_current": map_pipeline.map_solver_identity(),
    }


def audit_local_preflight(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, schedule_path, artifact_root,
        artifact_manifest_path, preflight_path, preflight_acceptance_path,
        work_root, output_path):
    registry_path = Path(registry_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    schedule_path = Path(schedule_path).resolve(strict=True)
    artifact_root = Path(artifact_root).resolve(strict=True)
    artifact_manifest_path = Path(artifact_manifest_path).resolve(strict=True)
    preflight_path = Path(preflight_path).resolve(strict=True)
    preflight_acceptance_path = Path(
        preflight_acceptance_path,
    ).resolve(strict=True)
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
    if (remote_preflight.get("remote_full_consensus") is not True
            or tuple(remote_preflight.get("nodes", ()))
            != REMOTE_FULL_CONSENSUS_NODES):
        raise RuntimeError("remote aggregate preflight lacks full consensus")
    try:
        remote_bundle = {
            name: remote_preflight[name] for name in EVIDENCE_BUNDLE_FIELDS
        }
    except KeyError as exc:
        raise ValueError("remote preflight lacks canonical evidence") from exc
    _validate_evidence_bundle(
        remote_bundle, context="remote", require_is=True,
    )
    preflight_acceptance = orchestration.validate_preflight_acceptance_offline(
        preflight_acceptance_path, schedule_path.parent.parent, schedule,
        source_commit, archive_sha256, source_manifest_sha256,
        {
            "schedule": schedule_path,
            "artifact_manifest": artifact_manifest_path,
            "aggregate_preflight": preflight_path,
        },
        {
            "schedule_sha256": schedule["schedule_sha256"],
            "artifact_manifest_sha256": artifact_manifest[
                "artifact_manifest_sha256"
            ],
            "preflight_status": remote_preflight["status"],
            "selected_resource_tier": remote_preflight[
                "selected_resource_tier"
            ],
            "registry_file_sha256": workflow._sha256_file(registry_path),
            "config_file_sha256": workflow._sha256_file(config_path),
        },
    )

    work_root.mkdir(parents=True)
    local_base_bundle = pipeline.hgp_screen_preflight_digest(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )
    is_root = work_root / "importance_sampling"
    is_root.mkdir()
    local_is_summary = []
    for cell in pipeline._map_cells(config):
        fingerprint = pipeline._cell_fingerprint(cell)
        path = is_root / f"{fingerprint}.npz"
        generated = pipeline.run_hgp_map_is_diagnostic(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root, path,
            seed_namespace=pipeline.HGP_SCREEN_PREFLIGHT_IS_ROOT,
        )
        validated = pipeline.validate_hgp_map_is_diagnostic(
            path, registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root,
            seed_namespace=pipeline.HGP_SCREEN_PREFLIGHT_IS_ROOT,
            replay_evidence="full",
        )
        local_is_summary.append(_is_full_summary_entry(
            fingerprint, generated, validated,
        ))
    local_bundle = _attach_is_summaries(
        local_base_bundle, local_is_summary,
    )
    remote_full = remote_bundle["canonical_full_payload"]
    local_full = local_bundle["canonical_full_payload"]
    remote_portable = remote_bundle["canonical_portable_payload"]
    local_portable = local_bundle["canonical_portable_payload"]
    exact = workflow._canonical_json(local_full) == workflow._canonical_json(
        remote_full,
    )
    portable_exact = (
        workflow._canonical_json(local_portable)
        == workflow._canonical_json(remote_portable)
        and local_bundle["canonical_portable_payload_sha256"]
        == remote_bundle["canonical_portable_payload_sha256"]
    )
    remote_decisions, remote_decision_sha = _validate_decision_evidence(
        remote_portable, context="remote portable",
    )
    local_decisions, local_decision_sha = _validate_decision_evidence(
        local_portable, context="local portable",
    )
    decisions_exact = (
        remote_decisions == local_decisions
        and remote_decision_sha == local_decision_sha
    )

    comparison = {
        "comparison_version": COMPARISON_VERSION,
        "remote_full_payload": remote_full,
        "local_full_payload": local_full,
        "remote_portable_payload": remote_portable,
        "local_portable_payload": local_portable,
        "full_mismatch_paths": _mismatch_paths(remote_full, local_full),
        "portable_mismatch_paths": _mismatch_paths(
            remote_portable, local_portable,
        ),
        "acceptance_decisions_exact": bool(decisions_exact),
    }
    comparison_path = work_root / "canonical_comparison.json"
    workflow._write_exclusive_json(comparison_path, comparison)
    if not portable_exact:
        raise RuntimeError("local/remote portable canonical evidence differs")
    if not decisions_exact:
        raise RuntimeError("local/remote acceptance decisions differ")
    completed_local_unix = time.time()
    remote_full_is = remote_full[IS_SUMMARY_KEY]
    local_full_is = local_full[IS_SUMMARY_KEY]
    portable_is = remote_portable[IS_SUMMARY_KEY]
    identity = {
        "attestation_version": ATTESTATION_VERSION,
        "status": "PASS_EXACT" if exact else "PORTABLE_PASS",
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
        "preflight_acceptance_manifest_file_sha256": workflow._sha256_file(
            preflight_acceptance_path,
        ),
        "preflight_acceptance_manifest_sha256": preflight_acceptance[
            "manifest_sha256"
        ],
        "remote_full_payload_sha256": remote_bundle[
            "canonical_full_payload_sha256"
        ],
        "local_full_payload_sha256": local_bundle[
            "canonical_full_payload_sha256"
        ],
        "remote_portable_payload_sha256": remote_bundle[
            "canonical_portable_payload_sha256"
        ],
        "local_portable_payload_sha256": local_bundle[
            "canonical_portable_payload_sha256"
        ],
        "exact_canonical_match": bool(exact),
        "portable_canonical_match": True,
        "acceptance_decisions_exact": True,
        "acceptance_decision_catalog_sha256": remote_decision_sha,
        "remote_full_consensus": True,
        "remote_full_consensus_nodes": list(REMOTE_FULL_CONSENSUS_NODES),
        "mismatch_paths": comparison["full_mismatch_paths"],
        "importance_sampling_portable_summary": portable_is,
        "remote_importance_sampling_full_summary": remote_full_is,
        "local_importance_sampling_full_summary": local_full_is,
        "solver_identity_policy": SOLVER_POLICY,
        "full_mismatch_policy": FULL_MISMATCH_POLICY,
        "local_environment": _environment_identity(),
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": completed_local_unix,
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
    parser.add_argument("--preflight-acceptance", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    attestation = audit_local_preflight(
        args.registry, args.config, args.source_commit, args.archive_sha256,
        args.source_manifest_sha256, args.schedule, args.artifact_root,
        args.artifact_manifest, args.preflight, args.preflight_acceptance,
        args.work_root, args.output,
    )
    print(workflow._canonical_json({
        "status": attestation["status"],
        "attestation": str(Path(args.output).resolve()),
        "attestation_sha256": attestation["attestation_sha256"],
        "mismatch_paths": attestation["mismatch_paths"],
    }))
    if attestation["status"] not in {"PASS_EXACT", "PORTABLE_PASS"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
