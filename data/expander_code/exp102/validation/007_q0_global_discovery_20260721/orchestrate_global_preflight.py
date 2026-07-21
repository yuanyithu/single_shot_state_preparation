"""Launch and combine the frozen three-node q=0 global-discovery preflight."""

import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import time

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    load_global_discovery_config,
    validate_global_schedule,
)
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry

VALIDATION_MODULE = (
    "data.expander_code.exp102.validation."
    "007_q0_global_discovery_20260721"
)
benchmark_global = importlib.import_module(VALIDATION_MODULE + ".benchmark_global")
cross_node_global = importlib.import_module(VALIDATION_MODULE + ".cross_node_global")
orchestrate_global = importlib.import_module(VALIDATION_MODULE + ".orchestrate_global")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
remote_command = orchestrate_global.remote_command
verified_bootstrap = orchestrate_global.verified_bootstrap
wait_for_markers = orchestrate_global.wait_for_markers


def validate_wmc_report(wmc, registry, config, source_commit,
                        source_identity, deadline):
    records = wmc.get("records", [])
    statuses = {
        "EXACT", "INCONCLUSIVE_WIDTH", "INCONCLUSIVE_TIMEOUT",
    }
    expected_status = (
        "EXACT" if records and all(value.get("status") == "EXACT"
                                   for value in records)
        else "INCONCLUSIVE"
    )
    if (wmc.get("report_version")
            != "exp102.q0_global.wmc_feasibility.v1"
            or wmc.get("source_commit") != source_commit
            or wmc.get("source_identity") != source_identity
            or wmc.get("registry_sha256") != registry["registry_sha256"]
            or wmc.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or wmc.get("node") != "nd-1"
            or wmc.get("environment", {}).get("system") != "Linux"
            or wmc.get("status") != expected_status
            or len(records) != len(config["panels"]["SMALL6"]["cells"])
            or [value.get("cell") for value in records]
            != config["panels"]["SMALL6"]["cells"]
            or any(value.get("status") not in statuses for value in records)
            or not 0.0 < float(wmc.get("timeout_seconds_per_cell", 0.0)) <= 7200.0
            or not 0.0 < float(wmc.get("completed_unix", 0.0)) <= deadline):
        raise ValueError("preflight WMC report is malformed or unverified")
    return wmc


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    args = parser.parse_args(argv)
    if FULL_SHA.fullmatch(args.source_commit) is None:
        raise ValueError("preflight source commit must be a full lowercase Git SHA")
    if any(SHA256.fullmatch(value) is None for value in (
            args.archive_sha256, args.manifest_sha256,
            args.schedule_file_sha256)):
        raise ValueError("preflight hashes must be lowercase SHA256")

    registry_path = Path("data/expander_code/exp102/registry/registry.json")
    config_path = Path("data/expander_code/exp102/config/q0_global.discovery.v1.json")
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    schedule_input = Path(args.schedule).resolve(strict=True)
    if sha256_file(schedule_input) != args.schedule_file_sha256:
        raise ValueError("preflight schedule file SHA mismatch")
    schedule = validate_global_schedule(
        schedule_input, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != args.archive_sha256
            or schedule["source_manifest_sha256"] != args.manifest_sha256):
        raise ValueError("preflight schedule/source identity mismatch")
    deadline = float(schedule["deadlines_unix"]["digest_runtime"])
    timeout = deadline - time.time()
    if timeout <= 0.0:
        raise TimeoutError("preflight initial eight-hour window has expired")

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    if (sha256_file(deployment_root / "SOURCE.tar") != args.archive_sha256
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json")
            != args.manifest_sha256):
        raise ValueError("preflight deployment hashes mismatch")
    run_root = home / ".single_shot/runs" / args.run_id
    output_root = run_root / "global/preflight"
    schedule_path = run_root / "control/GLOBAL_72H_SCHEDULE.json"
    if schedule_path.exists():
        if sha256_file(schedule_path) != args.schedule_file_sha256:
            raise FileExistsError("preflight frozen schedule conflicts")
    else:
        atomic_json(schedule_path, schedule)
        if sha256_file(schedule_path) != args.schedule_file_sha256:
            raise ValueError("preflight schedule is not canonical JSON")

    identity = {
        "stage": "preflight", "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.manifest_sha256,
        "schedule_file_sha256": args.schedule_file_sha256,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "nodes": ["nd-1", "nd-2", "nd-3"],
    }
    fingerprint = sha256_json(identity)
    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    for node in identity["nodes"]:
        stage_dir = output_root / "markers" / fingerprint[:12] / node
        stage_dirs[node] = stage_dir
        if any((stage_dir / name).exists() for name in ("RUNNING", "SUCCESS", "FAILED")):
            raise FileExistsError(f"preflight marker already exists for {node}")
        log = home / ".single_shot/logs" / f"{args.run_id}_global_preflight_{node}.log"
        command = (
            "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
            "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1",
            "conda", "run", "-n", "11", "--no-capture-output", "python",
            "-m", VALIDATION_MODULE + ".run_global_preflight_node", node,
            "--source-commit", args.source_commit,
            "--output-root", output_root,
        )
        shell = verified_bootstrap(
            deployment_root, args.source_commit, args.archive_sha256,
            args.manifest_sha256, stage_dir, log, fingerprint, command,
        )
        screen = f"exp102_global_preflight_{fingerprint[:8]}_{node}"
        screens[node] = screen
        subprocess.run((
            "ssh", node,
            remote_command(("screen", "-dmS", screen, "bash", "-lc", shell)),
        ), check=True)
        print(f"launched node={node} screen={screen}", flush=True)
    try:
        wait_for_markers(stage_dirs, fingerprint, timeout)
    except BaseException:
        for node, screen in screens.items():
            subprocess.run(
                ("ssh", node,
                 remote_command(("screen", "-S", screen, "-X", "quit"))),
                check=False,
            )
        raise

    runtime_paths = {
        node: output_root / "nodes" / node / "runtime.json"
        for node in identity["nodes"]
    }
    digest_paths = {
        node: output_root / "nodes" / node / "digest.json"
        for node in identity["nodes"]
    }
    runtime_path = run_root / "control/runtime_consensus.json"
    digest_path = run_root / "control/digest_consensus.json"
    runtime = benchmark_global.combine_runtime_reports(runtime_paths, runtime_path)
    digest = cross_node_global.combine_digest_reports(digest_paths, digest_path)
    wmc_path = output_root / "nodes/nd-1/wmc.json"
    wmc = json.loads(wmc_path.read_text(encoding="ascii"))
    validate_wmc_report(
        wmc, registry, config, args.source_commit,
        runtime["source_identity"], deadline,
    )
    completed = time.time()
    if completed > deadline:
        raise TimeoutError("preflight completed after the frozen eight-hour deadline")
    report = {
        "report_version": "exp102.q0_global.preflight.v1",
        "status": runtime["status"],
        **identity,
        "schedule_sha256": schedule["schedule_sha256"],
        "node_report_sha256": {
            node: sha256_file(output_root / "nodes" / node / "preflight.json")
            for node in identity["nodes"]
        },
        "runtime_consensus_sha256": sha256_file(runtime_path),
        "digest_consensus_sha256": sha256_file(digest_path),
        "wmc_sha256": sha256_file(wmc_path),
        "selected_resource_tier": runtime["selected_resource_tier"],
        "canonical_digest": digest["canonical_digest"],
        "completed_unix": completed,
    }
    report_path = run_root / "control/preflight_report.json"
    atomic_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "selected_resource_tier": report["selected_resource_tier"],
        "canonical_digest": report["canonical_digest"],
        "report": str(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
