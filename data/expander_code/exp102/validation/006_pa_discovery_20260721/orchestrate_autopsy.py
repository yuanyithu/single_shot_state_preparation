"""Freeze ownership and launch the four transport-autopsy tasks from nd-0."""

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

from data.expander_code.exp102.exp102_pipeline.transport_autopsy import (
    AUTOPSY_NODE_CAPACITY,
    AUTOPSY_TASKS_VERSION,
    fixed_autopsy_ownership,
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


def choose_nodes(requested, nd1_busy_threshold):
    loads = {}
    for node in AUTOPSY_NODE_CAPACITY:
        values = subprocess.run(
            ("ssh", node, "cat /proc/loadavg"), check=True,
            capture_output=True, text=True,
        ).stdout.split()
        loads[node] = tuple(float(value) for value in values[:3])
    if requested:
        nodes = tuple(requested.split(","))
    elif loads["nd-1"][0] > nd1_busy_threshold:
        nodes = ("nd-2", "nd-3")
    else:
        nodes = ("nd-1", "nd-2", "nd-3")
    if (len(nodes) < 2 or len(nodes) != len(set(nodes))
            or not set(nodes) <= set(AUTOPSY_NODE_CAPACITY)):
        raise ValueError("autopsy nodes must be a unique known set of size >=2")
    return nodes, loads


def bootstrap(deployment_root, commit, archive_sha, manifest_sha, stage_dir,
              log_path, stage_fingerprint, command):
    archive = deployment_root / "SOURCE.tar"
    verifier = Path(
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/"
        "run_verified_source.sh"
    )
    wrapper = RELATIVE / "run_pa_wrapper.sh"
    guarded = (
        "set -euo pipefail; "
        f"tar -xOf {shlex.quote(str(archive))} {verifier.as_posix()} "
        f"| bash -s -- {shlex.quote(str(deployment_root))} {commit} "
        f"{archive_sha} {manifest_sha} {remote_command(command)}"
    )
    return (
        "set -euo pipefail; "
        f"printf '%s  %s\\n' {archive_sha} {shlex.quote(str(archive))} "
        "| sha256sum -c - >/dev/null; "
        f"tar -xOf {shlex.quote(str(archive))} {wrapper.as_posix()} "
        f"| bash -s -- {shlex.quote(str(stage_dir))} {shlex.quote(str(log_path))} "
        f"{stage_fingerprint} bash -c {shlex.quote(guarded)}"
    )


def wait(stage_dirs, fingerprint, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        states = {}
        for node, root in stage_dirs.items():
            found = [name for name in ("RUNNING", "SUCCESS", "FAILED")
                     if (root / name).exists()]
            states[node] = "+".join(found) if found else "MISSING"
            for name in found:
                marker = json.loads((root / name).read_text(encoding="ascii"))
                if marker.get("stage_fingerprint") != fingerprint:
                    raise ValueError("autopsy marker fingerprint mismatch")
        if states != previous:
            print(" ".join(f"{node}={state}" for node, state in sorted(states.items())), flush=True)
            previous = states
        if any("FAILED" in value for value in states.values()):
            raise RuntimeError("autopsy stage failed")
        if all(value == "SUCCESS" for value in states.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("autopsy stage timed out")
        time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--nd1-busy-threshold", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    args = parser.parse_args(argv)
    if FULL_SHA.fullmatch(args.source_commit) is None:
        raise ValueError("autopsy source commit must be a full lowercase Git SHA")
    if (SHA256.fullmatch(args.archive_sha256) is None
            or SHA256.fullmatch(args.manifest_sha256) is None):
        raise ValueError("autopsy deployment hashes must be lowercase SHA256")
    if (not math.isfinite(args.timeout_seconds)
            or not 0 < args.timeout_seconds <= 14400.0):
        raise ValueError("autopsy timeout must lie in (0,4h]")
    control_input = Path(args.control).resolve(strict=True)
    control = json.loads(control_input.read_text(encoding="ascii"))
    if (control.get("manifest_version") != AUTOPSY_TASKS_VERSION
            or control.get("source_commit") != args.source_commit):
        raise ValueError("autopsy control source/version mismatch")
    nodes, loads = choose_nodes(args.nodes, args.nd1_busy_threshold)
    print(" ".join(f"{node}_load1={loads[node][0]:.2f}" for node in sorted(loads)), flush=True)
    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    if (sha256_file(deployment_root / "SOURCE.tar") != args.archive_sha256
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json") != args.manifest_sha256):
        raise ValueError("autopsy deployment hashes mismatch")
    run_root = home / ".single_shot/runs" / args.run_id
    control_path = run_root / "control" / f"autopsy_{sha256_file(control_input)[:12]}.json"
    control_sha = write_exclusive(control_path, control)
    ownership = fixed_autopsy_ownership(
        control["tasks"], nodes, args.source_commit, control_sha,
    )
    ownership_path = run_root / "control" / f"ownership_{control_sha[:12]}.json"
    ownership_sha = write_exclusive(ownership_path, ownership)
    fingerprint = ownership["stage_fingerprint"]
    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    for node in nodes:
        stage_dir = run_root / "transport_autopsy" / control_sha[:12] / node
        stage_dirs[node] = stage_dir
        log = home / ".single_shot/logs" / f"{args.run_id}_autopsy_{node}.log"
        command = (
            "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
            "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "conda", "run", "-n", "11",
            "--no-capture-output", "python", RELATIVE / "run_autopsy_stage.py", node,
            "--num-workers", AUTOPSY_NODE_CAPACITY[node], "--run-id", args.run_id,
            "--source-commit", args.source_commit, "--control", control_path,
            "--control-sha256", control_sha, "--ownership", ownership_path,
            "--ownership-sha256", ownership_sha, "--stage-fingerprint", fingerprint,
        )
        shell = bootstrap(
            deployment_root, args.source_commit, args.archive_sha256,
            args.manifest_sha256, stage_dir, log, fingerprint, command,
        )
        screen = f"exp102_autopsy_{control_sha[:8]}_{node}"
        screens[node] = screen
        subprocess.run((
            "ssh", node,
            remote_command(("screen", "-dmS", screen, "bash", "-lc", shell)),
        ), check=True)
        print(f"launched node={node} screen={screen}", flush=True)
    try:
        wait(stage_dirs, fingerprint, args.timeout_seconds)
    except BaseException:
        for node, screen in screens.items():
            subprocess.run(
                ("ssh", node, remote_command(("screen", "-S", screen, "-X", "quit"))),
                check=False,
            )
        raise
    print(json.dumps({
        "status": "SUCCESS", "nodes": nodes, "loads": loads,
        "control_sha256": control_sha, "ownership_sha256": ownership_sha,
        "stage_fingerprint": fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
