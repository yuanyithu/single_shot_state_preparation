"""Freeze PA stage ownership and launch it from the nd-0 storage node."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from data.expander_code.exp102.exp102_pipeline.pa_discovery import (
    PA_NODE_CAPACITY,
    PA_TASKS_VERSION,
    fixed_pa_ownership,
)


RELATIVE = Path("data/expander_code/exp102/validation/006_pa_discovery_20260721")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_command(arguments):
    return " ".join(shlex.quote(str(value)) for value in arguments)


def probe_node_load(node):
    output = subprocess.run(
        ("ssh", node, "cat /proc/loadavg"), check=True, capture_output=True, text=True,
    ).stdout.split()
    if len(output) < 3:
        raise ValueError(f"cannot parse load average from {node}")
    return tuple(float(value) for value in output[:3])


def choose_nodes(requested, nd1_busy_threshold):
    loads = {node: probe_node_load(node) for node in PA_NODE_CAPACITY}
    if requested:
        nodes = tuple(requested.split(","))
        if len(nodes) != len(set(nodes)) or not set(nodes) <= set(PA_NODE_CAPACITY):
            raise ValueError("--nodes must be a unique subset of nd-1,nd-2,nd-3")
    elif loads["nd-1"][0] > nd1_busy_threshold:
        nodes = ("nd-2", "nd-3")
    else:
        nodes = ("nd-1", "nd-2", "nd-3")
    if len(nodes) < 2:
        raise ValueError("PA discovery requires at least two frozen nodes")
    return nodes, loads


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def validate_runtime_report(path, control, source_commit, archive_sha256,
                            manifest_sha256):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    if set(report) != {
            "benchmark_version", "source_commit", "source_identity", "environment",
            "registry_sha256", "discovery_config_sha256", "rows",
            "conservative_seconds_per_particle_sweep", "startup_seconds",
            "max_population_minutes", "projected_core_seconds", "projection_nodes",
            "projection_capacity", "projected_minutes_with_safety_factor_2",
            "projected_confirmation_methods", "checks", "status"}:
        raise ValueError("PA runtime report schema mismatch")
    identity = report["source_identity"]
    environment = report["environment"]
    if (report["benchmark_version"] != "exp102.q0_pa.runtime.v1"
            or report["source_commit"] != source_commit
            or report["registry_sha256"] != control.get("registry_sha256")
            or report["discovery_config_sha256"]
            != control.get("discovery_config_sha256")
            or not isinstance(identity, dict)
            or set(identity) != {
                "source_commit", "mode", "archive_sha256", "manifest_sha256",
                "file_count"}
            or identity["source_commit"] != source_commit
            or identity["mode"] != "archive"
            or identity["archive_sha256"] != archive_sha256
            or identity["manifest_sha256"] != manifest_sha256
            or isinstance(identity["file_count"], bool)
            or not isinstance(identity["file_count"], int)
            or identity["file_count"] <= 0
            or not isinstance(environment, dict)
            or environment.get("system") != "Linux"):
        raise ValueError("PA runtime report is not clean-source Linux evidence")
    rows = report["rows"]
    row_keys = {
        (int(row["m"]), str(row["kernel"])) for row in rows
        if isinstance(row, dict) and "m" in row and "kernel" in row
    }
    slopes = [
        float(row["differential_us_per_particle_sweep"]) for row in rows
        if isinstance(row, dict) and "differential_us_per_particle_sweep" in row
    ]
    expected_rows = {
        (6, "coordinate"), (6, "block4"),
        (8, "coordinate"), (8, "block4"),
    }
    if (len(rows) != 4 or row_keys != expected_rows or len(slopes) != 4
            or not all(math.isfinite(value) and value >= 0.0 for value in slopes)):
        raise ValueError("PA runtime report benchmark rows are invalid")
    startup = float(report["startup_seconds"])
    population_minutes = float(report["max_population_minutes"])
    schedule_minutes = float(report["projected_minutes_with_safety_factor_2"])
    checks = {
        "m8_slowest_kernel_us": max(
            float(row["differential_us_per_particle_sweep"])
            for row in rows if int(row["m"]) == 8
        ) <= 200.0,
        "startup_seconds": startup <= 120.0,
        "max_population_minutes": population_minutes <= 20.0,
        "full_schedule_minutes_with_safety_factor_2": schedule_minutes <= 180.0,
    }
    if (not all(math.isfinite(value) and value >= 0.0 for value in (
            startup, population_minutes, schedule_minutes,
            float(report["projected_core_seconds"])))
            or report["projection_nodes"] != ["nd-2", "nd-3"]
            or int(report["projection_capacity"])
            != PA_NODE_CAPACITY["nd-2"] + PA_NODE_CAPACITY["nd-3"]
            or report["checks"] != checks or not all(checks.values())
            or report["status"] != "PASS"):
        raise ValueError("PA runtime budget gate did not pass")
    return report


def freeze_runtime_report(source_path, control_root):
    source_path = Path(source_path).resolve(strict=True)
    digest = sha256_file(source_path)
    destination = control_root / f"runtime_{digest[:12]}.json"
    value = json.loads(source_path.read_text(encoding="ascii"))
    if destination.exists():
        if sha256_file(destination) != digest:
            raise FileExistsError("frozen PA runtime report conflicts with this launch")
    elif write_exclusive(destination, value) != digest:
        raise ValueError("PA runtime report is not canonical JSON")
    return destination, digest


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file, stage_fingerprint,
                       command):
    archive = deployment_root / "SOURCE.tar"
    verifier = Path(
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/"
        "run_verified_source.sh"
    )
    wrapper = RELATIVE / "run_pa_wrapper.sh"
    guarded = (
        "set -euo pipefail; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(verifier.as_posix())} "
        f"| bash -s -- {shlex.quote(str(deployment_root))} {source_commit} "
        f"{archive_sha256} {manifest_sha256} {remote_command(command)}"
    )
    return (
        "set -euo pipefail; "
        f"printf '%s  %s\\n' {archive_sha256} {shlex.quote(str(archive))} "
        "| sha256sum -c - >/dev/null; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(wrapper.as_posix())} "
        f"| bash -s -- {shlex.quote(str(stage_dir))} {shlex.quote(str(log_file))} "
        f"{stage_fingerprint} bash -c {shlex.quote(guarded)}"
    )


def wait_for_markers(stage_dirs, stage_fingerprint, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        states = {}
        for node, stage_dir in stage_dirs.items():
            present = [name for name in ("RUNNING", "SUCCESS", "FAILED")
                       if (stage_dir / name).exists()]
            states[node] = "+".join(present) if present else "MISSING"
            for name in present:
                marker = json.loads((stage_dir / name).read_text(encoding="ascii"))
                if marker.get("stage_fingerprint") != stage_fingerprint:
                    raise ValueError(f"PA marker fingerprint mismatch on {node}")
        if states != previous:
            print(" ".join(f"{node}={state}" for node, state in sorted(states.items())), flush=True)
            previous = states
        if any("FAILED" in state for state in states.values()):
            raise RuntimeError("PA stage failed")
        if all(state == "SUCCESS" for state in states.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("PA stage exceeded its frozen timeout")
        time.sleep(2.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--nd1-busy-threshold", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    if FULL_SHA.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if (SHA256.fullmatch(args.archive_sha256) is None
            or SHA256.fullmatch(args.manifest_sha256) is None):
        raise ValueError("deployment hashes must be lowercase SHA256")
    control_input = Path(args.control).resolve(strict=True)
    control = json.loads(control_input.read_text(encoding="ascii"))
    if (control.get("manifest_version") != PA_TASKS_VERSION
            or control.get("source_commit") != args.source_commit):
        raise ValueError("PA control source/version mismatch")
    stage_timeout = 7200.0 if control.get("stage") == "hard_screen" else 14400.0
    timeout_seconds = stage_timeout if args.timeout_seconds is None else args.timeout_seconds
    if (not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= stage_timeout):
        raise ValueError("PA timeout exceeds the frozen stage deadline")
    validate_runtime_report(
        args.runtime_report, control, args.source_commit,
        args.archive_sha256, args.manifest_sha256,
    )
    nodes, loads = choose_nodes(args.nodes, args.nd1_busy_threshold)
    print(" ".join(f"{node}_load1={loads[node][0]:.2f}" for node in sorted(loads)), flush=True)

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    archive = deployment_root / "SOURCE.tar"
    manifest = deployment_root / "SOURCE_MANIFEST.json"
    if (not archive.is_file() or sha256_file(archive) != args.archive_sha256
            or not manifest.is_file() or sha256_file(manifest) != args.manifest_sha256):
        raise ValueError("shared PA deployment hashes do not match")
    run_root = home / ".single_shot/runs" / args.run_id
    runtime_path, runtime_sha256 = freeze_runtime_report(
        args.runtime_report, run_root / "control",
    )
    control_path = run_root / "control" / (
        f"{control['stage']}_{sha256_file(control_input)[:12]}.json"
    )
    control_sha256 = write_exclusive(control_path, control)
    ownership = fixed_pa_ownership(
        control["tasks"], nodes, args.source_commit, control_sha256, control["stage"],
    )
    ownership_path = run_root / "control" / f"ownership_{control_sha256[:12]}.json"
    ownership_sha256 = write_exclusive(ownership_path, ownership)
    stage_fingerprint = ownership["stage_fingerprint"]
    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    for node in nodes:
        stage_dir = run_root / control["stage"] / control_sha256[:12] / node
        stage_dirs[node] = stage_dir
        if any((stage_dir / marker).exists() for marker in ("RUNNING", "SUCCESS", "FAILED")):
            raise FileExistsError(f"PA stage marker already exists for {node}")
        log = home / ".single_shot/logs" / (
            f"{args.run_id}_{control['stage']}_{control_sha256[:12]}_{node}.log"
        )
        command = (
            "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
            "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "conda", "run", "-n", "11",
            "--no-capture-output", "python", RELATIVE / "run_pa_stage.py", node,
            "--num-workers", PA_NODE_CAPACITY[node], "--run-id", args.run_id,
            "--source-commit", args.source_commit, "--control", control_path,
            "--control-sha256", control_sha256, "--ownership", ownership_path,
            "--ownership-sha256", ownership_sha256,
            "--stage-fingerprint", stage_fingerprint,
        )
        shell = verified_bootstrap(
            deployment_root, args.source_commit, args.archive_sha256,
            args.manifest_sha256, stage_dir, log, stage_fingerprint, command,
        )
        screen = f"exp102_pa_{control['stage']}_{control_sha256[:8]}_{node}"
        screens[node] = screen
        subprocess.run((
            "ssh", node,
            remote_command(("screen", "-dmS", screen, "bash", "-lc", shell)),
        ), check=True)
        print(f"launched node={node} workers={PA_NODE_CAPACITY[node]} screen={screen}", flush=True)
    try:
        wait_for_markers(stage_dirs, stage_fingerprint, timeout_seconds)
    except BaseException:
        for node, screen in screens.items():
            subprocess.run(
                ("ssh", node, remote_command(("screen", "-S", screen, "-X", "quit"))),
                check=False,
            )
        raise
    print(json.dumps({
        "status": "SUCCESS", "nodes": nodes, "loads": loads,
        "control_sha256": control_sha256,
        "ownership_sha256": ownership_sha256,
        "runtime_report": str(runtime_path),
        "runtime_report_sha256": runtime_sha256,
        "stage_fingerprint": stage_fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
