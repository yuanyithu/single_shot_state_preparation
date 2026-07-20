"""Freeze ownership and launch one PT-v2 discovery manifest from nd-0."""

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


NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
RELATIVE = Path("data/expander_code/exp102/validation/005_pt_v2_discovery_20260720")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


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
    loads = {node: probe_node_load(node) for node in NODE_CAPACITY}
    if requested:
        nodes = tuple(requested.split(","))
        if len(nodes) != len(set(nodes)) or not set(nodes) <= set(NODE_CAPACITY):
            raise ValueError("--nodes must be a duplicate-free subset of nd-1,nd-2,nd-3")
    elif loads["nd-1"][0] > nd1_busy_threshold:
        nodes = ("nd-2", "nd-3")
    else:
        nodes = ("nd-1", "nd-2", "nd-3")
    if len(nodes) < 2:
        raise ValueError("discovery requires at least two fixed nodes")
    return nodes, loads


def task_cost(task):
    candidate = task["candidate"]
    code_id = task["cell"]["code_id"]
    m = int(code_id[1:3])
    return (
        float(m * m)
        * candidate["num_temperatures"]
        * (candidate["burn_rounds"] + candidate["measurement_rounds"])
    )


def fixed_ownership(tasks, nodes, source_commit, control_sha256, stage):
    loads = {node: 0.0 for node in nodes}
    owners = {}
    for task in sorted(tasks, key=lambda item: (-task_cost(item), sha256_json(item))):
        node = min(nodes, key=lambda name: (loads[name] / NODE_CAPACITY[name], name))
        fingerprint = sha256_json(task)
        owners[fingerprint] = node
        loads[node] += task_cost(task)
    identity = {
        "source_commit": source_commit,
        "control_sha256": control_sha256,
        "stage": stage,
        "nodes": list(nodes),
        "task_owner": owners,
        "candidate_transport": sorted({
            (task["candidate"]["ladder_fingerprint"]
             if "ladder_fingerprint" in task["candidate"] else task["ladder_fingerprint"],
             task["candidate"]["swap_sweeps_per_round"])
            for task in tasks
        }),
        "m_values": sorted({int(task["cell"]["code_id"][1:3]) for task in tasks}),
    }
    stage_fingerprint = sha256_json(identity)
    return {
        "ownership_version": "exp102.discovery.ownership.v2",
        **identity,
        "stage_fingerprint": stage_fingerprint,
        "weighted_load": loads,
        "capacity": {node: NODE_CAPACITY[node] for node in nodes},
    }


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


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file, stage_fingerprint,
                       command):
    archive = deployment_root / "SOURCE.tar"
    verifier = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh")
    wrapper = RELATIVE / "run_discovery_wrapper.sh"
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
                    raise ValueError(f"marker fingerprint mismatch on {node}")
        if states != previous:
            print(" ".join(f"{node}={state}" for node, state in sorted(states.items())), flush=True)
            previous = states
        if any("FAILED" in state for state in states.values()):
            raise RuntimeError("discovery stage failed")
        if all(state == "SUCCESS" for state in states.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("discovery stage timed out")
        time.sleep(2.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--nd1-busy-threshold", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args(argv)
    if FULL_SHA.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if (SHA256.fullmatch(args.archive_sha256) is None
            or SHA256.fullmatch(args.manifest_sha256) is None):
        raise ValueError("deployment hashes must be lowercase SHA256")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive and finite")
    control_input = Path(args.control).resolve(strict=True)
    control = json.loads(control_input.read_text(encoding="ascii"))
    if (control.get("manifest_version") != "exp102.discovery.tasks.v2"
            or control.get("source_commit") != args.source_commit):
        raise ValueError("control manifest source/version mismatch")
    nodes, loads = choose_nodes(args.nodes, args.nd1_busy_threshold)
    print(" ".join(f"{node}_load1={loads[node][0]:.2f}" for node in sorted(loads)), flush=True)

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    archive = deployment_root / "SOURCE.tar"
    manifest = deployment_root / "SOURCE_MANIFEST.json"
    if (not archive.is_file() or sha256_file(archive) != args.archive_sha256
            or not manifest.is_file() or sha256_file(manifest) != args.manifest_sha256):
        raise ValueError("shared deployment hashes do not match")
    run_root = home / ".single_shot/runs" / args.run_id
    control_path = run_root / "control" / f"{control['stage']}_{sha256_file(control_input)[:12]}.json"
    control_sha256 = write_exclusive(control_path, control)
    ownership = fixed_ownership(
        control["tasks"], nodes, args.source_commit, control_sha256, control["stage"],
    )
    ownership_path = run_root / "control" / f"ownership_{control_sha256[:12]}.json"
    ownership_sha256 = write_exclusive(ownership_path, ownership)
    stage_fingerprint = ownership["stage_fingerprint"]
    source = deployment_root / "source"
    stage_dirs = {}
    for node in nodes:
        stage_dir = run_root / control["stage"] / control_sha256[:12] / node
        stage_dirs[node] = stage_dir
        if any((stage_dir / marker).exists() for marker in ("RUNNING", "SUCCESS", "FAILED")):
            raise FileExistsError(f"discovery marker already exists for {node}")
        log = home / ".single_shot/logs" / (
            f"{args.run_id}_{control['stage']}_{control_sha256[:12]}_{node}.log"
        )
        command = (
            "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
            "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "conda", "run", "-n", "11",
            "--no-capture-output", "python", RELATIVE / "run_discovery_stage.py", node,
            "--num-workers", NODE_CAPACITY[node], "--run-id", args.run_id,
            "--source-commit", args.source_commit, "--control", control_path,
            "--control-sha256", control_sha256, "--ownership", ownership_path,
            "--ownership-sha256", ownership_sha256,
            "--stage-fingerprint", stage_fingerprint,
        )
        shell = verified_bootstrap(
            deployment_root, args.source_commit, args.archive_sha256,
            args.manifest_sha256, stage_dir, log, stage_fingerprint, command,
        )
        screen = f"exp102_dv2_{control['stage']}_{control_sha256[:8]}_{node}"
        subprocess.run(("ssh", node, remote_command(("screen", "-dmS", screen, "bash", "-lc", shell))), check=True)
        print(f"launched node={node} workers={NODE_CAPACITY[node]} screen={screen}", flush=True)
    wait_for_markers(stage_dirs, stage_fingerprint, args.timeout_seconds)
    print(json.dumps({
        "status": "SUCCESS", "nodes": nodes, "loads": loads,
        "control_sha256": control_sha256, "ownership_sha256": ownership_sha256,
        "stage_fingerprint": stage_fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
