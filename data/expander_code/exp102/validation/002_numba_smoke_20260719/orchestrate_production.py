"""Launch and reconcile the three-node exp102 production run from nd-0."""

import sys

# Loading the shared launcher helper must not mutate the verified source tree.
sys.dont_write_bytecode = True

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess


_HELPER_PATH = Path(__file__).with_name("orchestrate_stage.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("exp102_orchestrate_stage", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"cannot load production launcher helpers from {_HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(HELPERS)
WORKERS = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
RELATIVE = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719")


@dataclass(frozen=True)
class ProductionLaunch:
    node: str
    workers: int
    stage_dir: Path
    log_file: Path
    screen_name: str
    ssh_command: tuple


def build_node_launch(run_id, source_commit, archive_sha256, manifest_sha256,
                      node, home=None):
    if HELPERS.RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run id must contain only letters, digits, dot, underscore, and dash")
    if node not in WORKERS:
        raise ValueError(f"unsupported production node: {node}")
    home = Path.home() if home is None else Path(home)
    deployment_root = home / ".single_shot/repos" / run_id
    source = deployment_root / "source"
    run_root = home / ".single_shot/runs" / run_id
    workers = WORKERS[node]
    stage_dir = run_root / "production_stage" / node
    log_file = home / ".single_shot/logs" / f"{run_id}_production_{node}.log"
    screen_name = f"exp102_{run_id}_production_{node}"
    command = (
        "env",
        f"NUMBA_CACHE_DIR={deployment_root / ('numba-cache-' + node)}",
        "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "conda", "run", "-n", "11", "--no-capture-output", "python",
        RELATIVE / "run_production_stage.py", node,
        "--num-workers", workers,
        "--run-root", run_root,
        "--registry", source / "data/expander_code/exp102/registry/registry.json",
        "--config", source / "data/expander_code/exp102/config/production.v1.json",
        "--frozen", run_root / "frozen.json",
        "--pilot-report", run_root / "pilot_report.json",
        "--task-plan", run_root / "task_plan.json",
        "--deployment-manifest", run_root / "production_deployment.json",
    )
    shell = HELPERS.verified_bootstrap(
        deployment_root, source_commit, archive_sha256, manifest_sha256,
        stage_dir, log_file, command,
    )
    remote = HELPERS.remote_command(("screen", "-dmS", screen_name, "bash", "-lc", shell))
    return ProductionLaunch(
        node=node, workers=workers, stage_dir=stage_dir, log_file=log_file,
        screen_name=screen_name, ssh_command=("ssh", node, remote),
    )


def reconcile_statuses(run_root):
    run_root = Path(run_root)
    statuses = {}
    for node in WORKERS:
        path = run_root / "status" / f"production_{node}.json"
        if not path.is_file():
            raise ValueError(f"production status is missing: {node}")
        statuses[node] = json.loads(path.read_text(encoding="ascii"))
    common_fields = (
        "registry_sha256", "config_sha256", "frozen_config_sha256", "source_commit",
        "task_plan_sha256", "deployment_manifest_sha256", "pilot_report_sha256_file",
    )
    reference = statuses["nd-1"]
    for node, status in statuses.items():
        if status.get("status") != "SUCCESS" or status.get("node") != node:
            raise ValueError(f"production status is not successful: {node}")
        if status.get("computed", 0) + status.get("reused", 0) != status.get("expected"):
            raise ValueError(f"production status counts are inconsistent: {node}")
        if any(status.get(field) != reference.get(field) for field in common_fields):
            raise ValueError(f"production status identities differ across nodes: {node}")
    if sum(status["expected"] for status in statuses.values()) != 6144:
        raise ValueError("production statuses do not cover exactly 6144 tasks")
    return statuses


def require_idle_capacity(node, workers):
    command = (
        "set -e; getconf _NPROCESSORS_ONLN; cut -d' ' -f1 /proc/loadavg; "
        "pgrep -u \"$(id -u)\" -af '[p]ython.*exp102|[r]un_production_stage|[r]un_ladder_stage' || true"
    )
    result = subprocess.run(
        ("ssh", node, command), check=True, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        raise ValueError(f"production resource probe is incomplete: {node}")
    cpus, load1 = int(lines[0]), float(lines[1])
    if cpus < workers or load1 >= 5.0 or lines[2:]:
        raise RuntimeError(
            f"production node is not idle: {node} cpus={cpus} load1={load1} processes={lines[2:]}"
        )


def validate_preflight_attestation(attestation, source_commit, archive_sha256, manifest_sha256):
    smoke_digest = str(attestation.get("smoke_digest", ""))
    nodes = attestation.get("nodes", {})
    if (attestation.get("preflight_version") != "exp102.preflight.v1"
            or attestation.get("status") != "PASS"
            or attestation.get("source_commit") != source_commit
            or attestation.get("archive_sha256") != archive_sha256
            or attestation.get("manifest_sha256") != manifest_sha256
            or re.fullmatch(r"[0-9a-f]{64}", smoke_digest) is None
            or not isinstance(nodes, dict)
            or set(nodes) != set(WORKERS)
            or any(not row.get("idle") or row.get("smoke_digest") != smoke_digest
                   for row in nodes.values())):
        raise ValueError("three-node preflight attestation is invalid")
    return smoke_digest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 24 * 3600.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout must be a positive finite number")

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    run_root = home / ".single_shot/runs" / args.run_id
    HELPERS.verify_shared_deployment(
        deployment_root, args.source_commit, args.archive_sha256, args.manifest_sha256,
    )
    attestation_path = run_root / "preflight_attestation.json"
    if not attestation_path.is_file():
        raise ValueError("three-node preflight attestation is missing")
    attestation = json.loads(attestation_path.read_text(encoding="ascii"))
    validate_preflight_attestation(
        attestation, args.source_commit, args.archive_sha256, args.manifest_sha256,
    )
    for name in ("frozen.json", "pilot_report.json", "task_plan.json", "production_deployment.json"):
        if not (run_root / name).is_file():
            raise ValueError(f"production control file is missing: {name}")
    for node, workers in WORKERS.items():
        require_idle_capacity(node, workers)
    launches = tuple(build_node_launch(
        args.run_id, args.source_commit, args.archive_sha256,
        args.manifest_sha256, node, home,
    ) for node in WORKERS)
    HELPERS.check_marker_conflicts(launches)
    for launch in launches:
        subprocess.run(launch.ssh_command, check=True)
        print(
            f"launched node={launch.node} workers={launch.workers} "
            f"screen={launch.screen_name} log={launch.log_file}",
            flush=True,
        )
    HELPERS.wait_for_terminal_markers(launches, args.timeout_seconds, poll_seconds=10.0)
    statuses = reconcile_statuses(run_root)
    print(json.dumps({
        "status": "SUCCESS", "num_tasks": sum(row["expected"] for row in statuses.values()),
        "nodes": sorted(statuses),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
