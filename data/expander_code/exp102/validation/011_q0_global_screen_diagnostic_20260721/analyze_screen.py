"""Verify both remote stages, replay raw, and seal a diagnostic-only decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity

from . import benchmark_screen, cross_node_screen
from .common import (
    CONTRACT_VERSION,
    DECISION_VERSION,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    DIGEST_CONSENSUS_VERSION,
    EXPECTED_PREFLIGHT_NODES,
    PREFLIGHT_NODE_VERSION,
    PREFLIGHT_VERSION,
    analyze_measurement,
    atomic_json,
    config_sha256,
    load_config,
    load_registry,
    pipeline_attr,
    sha256_file,
    sha256_json,
    validate_runtime_consensus,
    validate_schedule,
    verify_remote_stage,
)


TERMINAL_PACKAGE_VERSION = (
    "exp102.q0_global.screen_diagnostic.verified_terminal_package.v1"
)


def _require_new(*paths):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "diagnostic analysis output already exists: " + ",".join(existing)
        )


def _validate_preflight(path, digest_path, runtime_path, run_root, schedule,
                        registry, config, source_commit,
                        schedule_file_sha256):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    digest = json.loads(Path(digest_path).read_text(encoding="ascii"))
    runtime = json.loads(Path(runtime_path).read_text(encoding="ascii"))
    report_fields = {
        "report_version", "contract_version", "status", "stage",
        "source_commit", "archive_sha256", "source_manifest_sha256",
        "schedule_file_sha256", "schedule_sha256", "registry_sha256",
        "diagnostic_config_sha256", "registry_relative", "config_relative",
        "nodes", "stage_fingerprint", "node_report_sha256",
        "runtime_consensus_sha256", "digest_consensus_sha256",
        "selected_resource_tier", "selected_eligible_methods",
        "canonical_digest", "excluded_work", "maximum_terminal_status",
        "completed_unix",
    }
    if (set(report) != report_fields
            or report.get("report_version") != PREFLIGHT_VERSION
            or report.get("contract_version") != CONTRACT_VERSION
            or report.get("status") != "PASS"
            or report.get("source_commit") != source_commit
            or report.get("registry_sha256") != registry["registry_sha256"]
            or report.get("diagnostic_config_sha256")
            != config_sha256(config)
            or report.get("archive_sha256") != schedule["archive_sha256"]
            or report.get("source_manifest_sha256")
            != schedule["source_manifest_sha256"]
            or report.get("schedule_file_sha256") != schedule_file_sha256
            or report.get("schedule_sha256") != schedule["schedule_sha256"]
            or report.get("runtime_consensus_sha256")
            != sha256_file(runtime_path)
            or report.get("digest_consensus_sha256")
            != sha256_file(digest_path)
            or report.get("excluded_work") != ["full_sector_ti", "wmc"]
            or report.get("maximum_terminal_status")
            != "DIAGNOSTIC_SCREEN_PAIR_FOUND"
            or not math.isfinite(float(
                report.get("completed_unix", math.nan)
            ))
            or float(report["completed_unix"])
            > float(schedule["deadlines_unix"]["preflight"])
            or digest.get("report_version") != DIGEST_CONSENSUS_VERSION
            or digest.get("contract_version") != CONTRACT_VERSION
            or digest.get("status") != "PASS"
            or digest.get("source_commit") != source_commit
            or digest.get("registry_sha256") != registry["registry_sha256"]
            or digest.get("diagnostic_config_sha256")
            != config_sha256(config)
            or digest.get("canonical_digest")
            != report.get("canonical_digest")
            or not isinstance(digest.get("source_identity"), dict)
            or digest["source_identity"].get("mode") != "archive"
            or digest["source_identity"].get("archive_sha256")
            != schedule["archive_sha256"]
            or digest["source_identity"].get("manifest_sha256")
            != schedule["source_manifest_sha256"]):
        raise ValueError("diagnostic preflight/digest evidence is invalid")
    identity = {
        key: report[key] for key in (
            "contract_version", "stage", "source_commit", "archive_sha256",
            "source_manifest_sha256", "schedule_file_sha256",
            "schedule_sha256", "registry_sha256",
            "diagnostic_config_sha256", "registry_relative",
            "config_relative", "nodes",
        )
    }
    if (report["stage"] != "preflight"
            or report["nodes"] != list(EXPECTED_PREFLIGHT_NODES)
            or report["stage_fingerprint"] != sha256_json(identity)
            or set(report.get("node_report_sha256", {}))
            != set(EXPECTED_PREFLIGHT_NODES)
            or report["selected_resource_tier"]
            != runtime.get("selected_resource_tier")
            or report["selected_eligible_methods"]
            != runtime.get("selected_eligible_methods")):
        raise ValueError("diagnostic aggregate preflight identity is invalid")

    output_root = (
        Path(run_root).resolve(strict=True) / "screen_diagnostic/preflight"
    )
    runtime_paths = {}
    digest_paths = {}
    source_identity = None
    node_report_fields = {
        "report_version", "contract_version", "status", "node",
        "source_commit", "source_identity", "environment",
        "pytest_returncode", "pytest_log_sha256", "digest_path",
        "digest_sha256", "runtime_path", "runtime_sha256",
        "excluded_work", "started_unix", "completed_unix",
    }
    for node in EXPECTED_PREFLIGHT_NODES:
        node_root = output_root / "nodes" / node
        node_report_path = node_root / "preflight.json"
        node_report = json.loads(
            node_report_path.read_text(encoding="ascii")
        )
        runtime_node_path = node_root / "runtime.json"
        digest_node_path = node_root / "digest.json"
        pytest_log_path = node_root / "pytest.log"
        started = float(node_report.get("started_unix", math.nan))
        completed = float(node_report.get("completed_unix", math.nan))
        if (set(node_report) != node_report_fields
                or node_report.get("report_version") != PREFLIGHT_NODE_VERSION
                or node_report.get("contract_version") != CONTRACT_VERSION
                or node_report.get("status") != "PASS"
                or node_report.get("node") != node
                or node_report.get("source_commit") != source_commit
                or node_report.get("environment", {}).get("system") != "Linux"
                or node_report.get("pytest_returncode") != 0
                or node_report.get("excluded_work")
                != ["full_sector_ti", "wmc"]
                or not math.isfinite(started)
                or not math.isfinite(completed)
                or not 0.0 < started <= completed
                <= float(schedule["deadlines_unix"]["preflight"])
                or node_report.get("digest_path")
                != f"nodes/{node}/digest.json"
                or node_report.get("runtime_path")
                != f"nodes/{node}/runtime.json"
                or sha256_file(node_report_path)
                != report["node_report_sha256"].get(node)
                or sha256_file(digest_node_path)
                != node_report.get("digest_sha256")
                or sha256_file(runtime_node_path)
                != node_report.get("runtime_sha256")
                or sha256_file(pytest_log_path)
                != node_report.get("pytest_log_sha256")):
            raise ValueError(
                f"diagnostic preflight node evidence is invalid: {node}"
            )
        if source_identity is None:
            source_identity = node_report["source_identity"]
        elif node_report["source_identity"] != source_identity:
            raise ValueError("diagnostic preflight source identities differ")
        runtime_paths[node] = runtime_node_path
        digest_paths[node] = digest_node_path
    reconstructed_runtime = benchmark_screen.combine_runtime_reports(
        runtime_paths,
    )
    reconstructed_digest = cross_node_screen.combine_digest_reports(
        digest_paths,
    )
    if (runtime != reconstructed_runtime
            or digest != reconstructed_digest
            or report["runtime_consensus_sha256"] != sha256_file(runtime_path)
            or report["digest_consensus_sha256"] != sha256_file(digest_path)
            or report["canonical_digest"]
            != reconstructed_digest["canonical_digest"]):
        raise ValueError("diagnostic aggregate preflight was not reconstructed")
    return report, digest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--deployment-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--digest-report", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--bias-control", required=True)
    parser.add_argument("--bias-ownership", required=True)
    parser.add_argument("--measurement-control", required=True)
    parser.add_argument("--measurement-ownership", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--decision-output", required=True)
    parser.add_argument("--package-output", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args(argv)
    if args.num_workers <= 0:
        raise ValueError("diagnostic analyzer worker count must be positive")
    output_paths = [
        Path(value).resolve() for value in (
            args.report_output, args.decision_output, args.package_output,
        )
    ]
    if len(set(output_paths)) != 3:
        raise ValueError("diagnostic analyzer outputs must be distinct paths")
    args.report_output, args.decision_output, args.package_output = map(
        str, output_paths,
    )
    _require_new(
        args.report_output, args.decision_output, args.package_output,
    )

    registry_path = Path(args.registry)
    config_path = Path(args.config)
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    measurement_control = json.loads(
        Path(args.measurement_control).read_text(encoding="ascii")
    )
    source_commit = measurement_control["source_commit"]
    schedule = validate_schedule(
        args.schedule, registry, config, source_commit,
    )
    analysis_source_identity = verify_source_identity(
        Path.cwd(), source_commit,
    )
    if (analysis_source_identity.get("mode") != "archive"
            or analysis_source_identity.get("archive_sha256")
            != schedule["archive_sha256"]
            or analysis_source_identity.get("manifest_sha256")
            != schedule["source_manifest_sha256"]):
        raise ValueError(
            "diagnostic analyzer must run from the scheduled archive source"
        )
    if time.time() > float(schedule["deadlines_unix"]["analysis"]):
        raise TimeoutError("diagnostic analysis deadline has expired")
    validate_runtime_consensus(
        args.runtime_report, source_commit, registry["registry_sha256"],
        config_sha256(config), schedule["archive_sha256"],
        schedule["source_manifest_sha256"],
    )
    preflight, digest = _validate_preflight(
        args.preflight_report, args.digest_report, args.runtime_report,
        args.run_root, schedule, registry, config, source_commit,
        sha256_file(args.schedule),
    )

    bias_control = json.loads(
        Path(args.bias_control).read_text(encoding="ascii")
    )
    if (bias_control.get("source_commit") != source_commit
            or measurement_control.get("bias_manifest_sha256")
            != sha256_json(bias_control)):
        raise ValueError("measurement control is not bound to bias control")
    bias_evidence = verify_remote_stage(
        args.run_root, args.raw_root, args.bias_control,
        args.bias_ownership, args.deployment_root, args.schedule,
        args.runtime_report, registry_path, config_path,
    )
    measurement_evidence = verify_remote_stage(
        args.run_root, args.raw_root, args.measurement_control,
        args.measurement_ownership, args.deployment_root, args.schedule,
        args.runtime_report, registry_path, config_path,
    )
    if (bias_evidence["raw_count"] != 15
            or measurement_evidence["raw_count"] != 1280
            or bias_evidence["source_commit"] != source_commit
            or measurement_evidence["source_commit"] != source_commit
            or bias_evidence["schedule_sha256"]
            != measurement_evidence["schedule_sha256"]
            or bias_evidence["runtime_report_sha256"]
            != measurement_evidence["runtime_report_sha256"]):
        raise ValueError("diagnostic bias/measurement evidence axes conflict")

    report = analyze_measurement(
        args.raw_root, args.measurement_control, registry_path, config_path,
        args.report_output, args.num_workers,
    )
    validate_report = pipeline_attr("validate_screen_report")
    validate_report(report, registry, config)
    if (verify_source_identity(Path.cwd(), source_commit)
            != analysis_source_identity):
        raise ValueError("diagnostic analysis changed its verified source")
    if time.time() > float(schedule["deadlines_unix"]["analysis"]):
        raise TimeoutError("diagnostic raw replay exceeded analysis deadline")
    if (report.get("formal_authorization") is not False
            or report.get("production_authorization") is not False):
        raise ValueError("diagnostic report attempted formal authorization")
    terminal = pipeline_attr(
        "terminal_decision", "build_terminal_decision",
    )(
        args.report_output, registry_path, config_path,
        args.decision_output,
    )
    if (terminal.get("formal_authorization") is not False
            or terminal.get("production_authorization") is not False
            or terminal.get("maximum_possible_status")
            != "DIAGNOSTIC_SCREEN_PAIR_FOUND"
            or terminal.get("status") in {
                "READY_FOR_FORMAL", "FROZEN_HELD_OUT_PASS", "REPORTABLE",
            }):
        raise ValueError("diagnostic terminal decision exceeded its authority")

    if (verify_source_identity(Path.cwd(), source_commit)
            != analysis_source_identity):
        raise ValueError("diagnostic decision changed its verified source")
    completed = time.time()
    if completed > float(schedule["deadlines_unix"]["analysis"]):
        raise TimeoutError("diagnostic decision exceeded analysis deadline")

    identity = {
        "package_version": TERMINAL_PACKAGE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": terminal["status"],
        "maximum_possible_status": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
        "archive_sha256": schedule["archive_sha256"],
        "source_manifest_sha256": schedule["source_manifest_sha256"],
        "analysis_source_identity": analysis_source_identity,
        "schedule_file_sha256": sha256_file(args.schedule),
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_report_sha256": sha256_file(args.runtime_report),
        "digest_report_sha256": sha256_file(args.digest_report),
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "canonical_digest": digest["canonical_digest"],
        "bias_control_sha256": sha256_file(args.bias_control),
        "bias_ownership_sha256": sha256_file(args.bias_ownership),
        "bias_evidence": bias_evidence,
        "measurement_control_sha256": sha256_file(args.measurement_control),
        "measurement_ownership_sha256": sha256_file(
            args.measurement_ownership
        ),
        "measurement_evidence": measurement_evidence,
        "analysis_report_sha256": sha256_file(args.report_output),
        "analysis_report_identity_sha256": report["report_sha256"],
        "terminal_decision_sha256": sha256_file(args.decision_output),
        "terminal_decision_identity_sha256": terminal["decision_sha256"],
        "selected_pair": terminal["selected_pair"],
        "formal_authorization": False,
        "production_authorization": False,
        "completed_unix": completed,
    }
    package = {**identity, "package_sha256": sha256_json(identity)}
    atomic_json(args.package_output, package)
    print(json.dumps({
        "status": package["status"],
        "selected_pair": package["selected_pair"],
        "report": args.report_output,
        "decision": args.decision_output,
        "package": args.package_output,
        "package_sha256": package["package_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
