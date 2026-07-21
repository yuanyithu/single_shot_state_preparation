"""Launch immutable three-node tests, digest, and sampler-only preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

from . import benchmark_screen, cross_node_screen
from .common import (
    CONTRACT_VERSION,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    EXPECTED_PREFLIGHT_NODES,
    PREFLIGHT_NODE_VERSION,
    PREFLIGHT_VERSION,
    atomic_json,
    config_sha256,
    load_config,
    load_registry,
    remote_command,
    sha256_file,
    sha256_json,
    validate_schedule,
    validate_source_and_hashes,
    verified_bootstrap,
    wait_for_markers,
    write_exclusive,
)


MODULE_ROOT = (
    "data.expander_code.exp102.validation."
    "011_q0_global_screen_diagnostic_20260721"
)


def _validate_node_report(path, node, output_root, source_commit,
                          source_identity, deadline):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    if (report.get("report_version") != PREFLIGHT_NODE_VERSION
            or report.get("contract_version") != CONTRACT_VERSION
            or report.get("status") != "PASS"
            or report.get("node") != node
            or report.get("source_commit") != source_commit
            or report.get("source_identity") != source_identity
            or report.get("environment", {}).get("system") != "Linux"
            or report.get("pytest_returncode") != 0
            or report.get("excluded_work") != ["full_sector_ti", "wmc"]
            or not 0.0 < float(report.get("started_unix", 0.0))
            <= float(report.get("completed_unix", 0.0)) <= deadline):
        raise ValueError(f"diagnostic preflight node report is invalid: {node}")
    for path_key, hash_key in (
        ("digest_path", "digest_sha256"),
        ("runtime_path", "runtime_sha256"),
    ):
        relative = Path(str(report.get(path_key, "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("diagnostic preflight report path is unsafe")
        target = Path(output_root) / relative
        if (not target.is_file()
                or sha256_file(target) != report.get(hash_key)):
            raise ValueError("diagnostic preflight report hash mismatch")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    parser.add_argument("--registry-relative", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config-relative", default=str(DEFAULT_CONFIG_RELATIVE))
    args = parser.parse_args(argv)
    validate_source_and_hashes(
        args.source_commit, args.archive_sha256, args.manifest_sha256,
        args.schedule_file_sha256,
    )

    registry_path = Path(args.registry_relative)
    config_path = Path(args.config_relative)
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    schedule_input = Path(args.schedule).resolve(strict=True)
    if sha256_file(schedule_input) != args.schedule_file_sha256:
        raise ValueError("diagnostic preflight schedule file SHA mismatch")
    schedule = validate_schedule(
        schedule_input, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != args.archive_sha256
            or schedule["source_manifest_sha256"] != args.manifest_sha256):
        raise ValueError("diagnostic schedule/source identity mismatch")
    if float(schedule["started_unix"]) > time.time():
        raise ValueError("diagnostic schedule cannot start in the future")
    deadline = float(schedule["deadlines_unix"]["preflight"])
    timeout = deadline - time.time()
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise TimeoutError("diagnostic preflight deadline has expired")

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    if (sha256_file(deployment_root / "SOURCE.tar") != args.archive_sha256
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json")
            != args.manifest_sha256):
        raise ValueError("diagnostic deployment hashes mismatch")
    run_root = home / ".single_shot/runs" / args.run_id
    output_root = run_root / "screen_diagnostic/preflight"
    schedule_path = run_root / "control/SCREEN_DIAGNOSTIC_24H_SCHEDULE.json"
    if schedule_path.exists():
        if sha256_file(schedule_path) != args.schedule_file_sha256:
            raise FileExistsError("frozen diagnostic schedule conflicts")
    elif write_exclusive(schedule_path, schedule) != args.schedule_file_sha256:
        raise ValueError("diagnostic schedule is not canonical JSON")

    identity = {
        "contract_version": CONTRACT_VERSION,
        "stage": "preflight",
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.manifest_sha256,
        "schedule_file_sha256": args.schedule_file_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
        "registry_relative": str(registry_path),
        "config_relative": str(config_path),
        "nodes": list(EXPECTED_PREFLIGHT_NODES),
    }
    fingerprint = sha256_json(identity)
    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    run_token = sha256_json({"run_id": args.run_id})[:8]
    try:
        for node in EXPECTED_PREFLIGHT_NODES:
            stage_dir = output_root / "markers" / fingerprint[:12] / node
            stage_dirs[node] = stage_dir
            if any((stage_dir / name).exists()
                   for name in ("RUNNING", "SUCCESS", "FAILED")):
                raise FileExistsError(
                    f"diagnostic preflight marker already exists for {node}"
                )
            log = home / ".single_shot/logs" / (
                f"{args.run_id}_screen_diagnostic_preflight_{node}.log"
            )
            command = (
                "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
                "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "PYTHONDONTWRITEBYTECODE=1", "conda", "run", "-n", "11",
                "--no-capture-output", "python", "-m",
                MODULE_ROOT + ".run_screen_preflight_node", node,
                "--source-commit", args.source_commit,
                "--output-root", output_root,
                "--registry-relative", registry_path,
                "--config-relative", config_path,
            )
            shell = verified_bootstrap(
                deployment_root, args.source_commit, args.archive_sha256,
                args.manifest_sha256, stage_dir, log, fingerprint, command,
            )
            screen = (
                f"exp102_sd_{run_token}_pre_{fingerprint[:8]}_{node}"
            )
            screens[node] = screen
            subprocess.run((
                "ssh", node,
                remote_command((
                    "screen", "-dmS", screen, "bash", "-lc", shell,
                )),
            ), check=True)
            print(f"launched node={node} screen={screen}", flush=True)
        wait_for_markers(stage_dirs, fingerprint, timeout)
    except BaseException:
        for node, screen in screens.items():
            subprocess.run((
                "ssh", node,
                remote_command(("screen", "-S", screen, "-X", "quit")),
            ), check=False)
        raise

    runtime_paths = {
        node: output_root / "nodes" / node / "runtime.json"
        for node in EXPECTED_PREFLIGHT_NODES
    }
    digest_paths = {
        node: output_root / "nodes" / node / "digest.json"
        for node in EXPECTED_PREFLIGHT_NODES
    }
    source_identity = json.loads(
        runtime_paths[EXPECTED_PREFLIGHT_NODES[0]].read_text(encoding="ascii")
    )["source_identity"]
    node_reports = {}
    for node in EXPECTED_PREFLIGHT_NODES:
        node_reports[node] = _validate_node_report(
            output_root / "nodes" / node / "preflight.json", node,
            output_root, args.source_commit, source_identity, deadline,
        )
    runtime_path = run_root / "control/screen_runtime_consensus.json"
    digest_path = run_root / "control/screen_digest_consensus.json"
    if runtime_path.exists() or digest_path.exists():
        raise FileExistsError("diagnostic consensus output already exists")
    runtime = benchmark_screen.combine_runtime_reports(
        runtime_paths, runtime_path,
    )
    digest = cross_node_screen.combine_digest_reports(
        digest_paths, digest_path,
    )
    completed = time.time()
    if completed > deadline:
        raise TimeoutError("diagnostic preflight completed after its deadline")
    report = {
        "report_version": PREFLIGHT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": runtime["status"],
        **identity,
        "stage_fingerprint": fingerprint,
        "node_report_sha256": {
            node: sha256_file(
                output_root / "nodes" / node / "preflight.json"
            ) for node in EXPECTED_PREFLIGHT_NODES
        },
        "runtime_consensus_sha256": sha256_file(runtime_path),
        "digest_consensus_sha256": sha256_file(digest_path),
        "selected_resource_tier": runtime["selected_resource_tier"],
        "selected_eligible_methods": runtime["selected_eligible_methods"],
        "canonical_digest": digest["canonical_digest"],
        "excluded_work": ["full_sector_ti", "wmc"],
        "maximum_terminal_status": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        "completed_unix": completed,
    }
    report_path = run_root / "control/screen_preflight_report.json"
    write_exclusive(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "selected_resource_tier": report["selected_resource_tier"],
        "canonical_digest": report["canonical_digest"],
        "report": str(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
