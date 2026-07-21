"""Freeze ownership and launch a global-discovery stage from nd-0."""

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

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    GLOBAL_TASKS_VERSION,
    NODE_CAPACITY,
    fixed_global_ownership,
    load_global_discovery_config,
    validate_global_control_manifest,
    validate_global_schedule,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


RELATIVE = Path("data/expander_code/exp102/validation/007_q0_global_discovery_20260721")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_STAGE_SECONDS = {
    "screen": 12 * 3600.0,
    "hard_fresh": 24 * 3600.0,
    "confirmation": 22 * 3600.0,
    "resolution": 22 * 3600.0,
    "diagnostic_boundary": 12 * 3600.0,
    "ti_anchors": 22 * 3600.0,
}


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
    for node in NODE_CAPACITY:
        values = subprocess.run(
            ("ssh", node, "cat /proc/loadavg"), check=True,
            capture_output=True, text=True,
        ).stdout.split()
        if len(values) < 3:
            raise ValueError(f"cannot parse load average from {node}")
        loads[node] = tuple(float(value) for value in values[:3])
    if requested:
        nodes = tuple(requested.split(","))
    elif loads["nd-1"][0] > nd1_busy_threshold:
        nodes = ("nd-2", "nd-3")
    else:
        nodes = ("nd-1", "nd-2", "nd-3")
    if (len(nodes) < 2 or len(nodes) != len(set(nodes))
            or not set(nodes) <= set(NODE_CAPACITY)):
        raise ValueError("global nodes must be a unique known set of size >=2")
    return nodes, loads


def validate_runtime_report(path, control, source_commit, archive_sha256,
                            manifest_sha256):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    if (report.get("benchmark_version")
            != "exp102.q0_global.runtime_consensus.v1"
            or report.get("source_commit") != source_commit
            or report.get("registry_sha256") != control.get("registry_sha256")
            or report.get("discovery_config_sha256")
            != control.get("discovery_config_sha256")
            or report.get("status") != "PASS"):
        raise ValueError("global runtime report identity/status mismatch")
    identity = report.get("source_identity")
    if (not isinstance(identity, dict)
            or identity.get("source_commit") != source_commit
            or identity.get("mode") != "archive"
            or identity.get("archive_sha256") != archive_sha256
            or identity.get("manifest_sha256") != manifest_sha256
            or report.get("environment", {}).get("system") != "Linux"):
        raise ValueError("global runtime report is not clean-source Linux evidence")
    selected = report.get("selected_resource_tier")
    eligible = set(report.get("selected_eligible_methods", []))
    if not isinstance(selected, str) or not eligible:
        raise ValueError("global runtime report lacks a selected resource tier")
    if control.get("kind") != "ti_anchor":
        for method, tier in control.get("method_tiers", []):
            base = tier[1:] if str(tier).startswith("2") else tier
            if method not in eligible or base != selected:
                raise ValueError("global control method/tier is not runtime eligible")
    projections = {value["resource_tier"]: value for value in report.get("projections", [])}
    if selected not in projections or not projections[selected].get("pass"):
        raise ValueError("global selected runtime projection did not pass")
    projected = float(projections[selected]["projected_hours_with_safety_factor_2"])
    if not math.isfinite(projected) or projected > 58.0:
        raise ValueError("global factor-two runtime projection exceeds 58 hours")
    return report


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file, stage_fingerprint,
                       command):
    archive = deployment_root / "SOURCE.tar"
    verifier = Path(
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/"
        "run_verified_source.sh"
    )
    wrapper = RELATIVE / "run_global_wrapper.sh"
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
        for node, root in stage_dirs.items():
            found = [name for name in ("RUNNING", "SUCCESS", "FAILED") if (root / name).exists()]
            states[node] = "+".join(found) if found else "MISSING"
            for name in found:
                marker = json.loads((root / name).read_text(encoding="ascii"))
                if marker.get("stage_fingerprint") != stage_fingerprint:
                    raise ValueError("global marker fingerprint mismatch")
        if states != previous:
            print(" ".join(f"{node}={state}" for node, state in sorted(states.items())), flush=True)
            previous = states
        if any("FAILED" in value for value in states.values()):
            raise RuntimeError("global stage failed")
        if all(value == "SUCCESS" for value in states.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("global stage exceeded its frozen timeout")
        time.sleep(2.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--nd1-busy-threshold", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    if FULL_SHA.fullmatch(args.source_commit) is None:
        raise ValueError("global source commit must be a full lowercase Git SHA")
    if (SHA256.fullmatch(args.archive_sha256) is None
            or SHA256.fullmatch(args.manifest_sha256) is None
            or SHA256.fullmatch(args.schedule_file_sha256) is None):
        raise ValueError("global deployment hashes must be lowercase SHA256")
    control_input = Path(args.control).resolve(strict=True)
    control = json.loads(control_input.read_text(encoding="ascii"))
    if (control.get("manifest_version") != GLOBAL_TASKS_VERSION
            or control.get("source_commit") != args.source_commit
            or control.get("kind") not in {
                "defect_bias", "measurement", "ti_anchor",
            }):
        raise ValueError("global control source/version/kind mismatch")
    maximum = MAX_STAGE_SECONDS[control["stage"]]
    if (args.timeout_seconds is not None
            and (not math.isfinite(args.timeout_seconds)
                 or not 0 < args.timeout_seconds <= maximum)):
        raise ValueError("global timeout exceeds the frozen stage window")
    schedule_input = Path(args.schedule).resolve(strict=True)
    if sha256_file(schedule_input) != args.schedule_file_sha256:
        raise ValueError("global schedule file SHA256 mismatch")
    registry_path = Path("data/expander_code/exp102/registry/registry.json")
    config_path = Path("data/expander_code/exp102/config/q0_global.discovery.v1.json")
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    validate_global_control_manifest(control, registry, config)
    schedule = validate_global_schedule(
        schedule_input, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != args.archive_sha256
            or schedule["source_manifest_sha256"] != args.manifest_sha256):
        raise ValueError("global schedule source archive identity mismatch")
    deadline = float(schedule["deadlines_unix"][control["stage"]])
    now = time.time()
    available = deadline - now
    timeout = min(maximum, available) if args.timeout_seconds is None else args.timeout_seconds
    if (now < float(schedule["started_unix"]) or available <= 0.0
            or timeout > available):
        raise TimeoutError("global stage cannot fit inside its frozen cumulative deadline")
    runtime_report = validate_runtime_report(
        args.runtime_report, control, args.source_commit,
        args.archive_sha256, args.manifest_sha256,
    )
    if control["kind"] == "ti_anchor":
        predicted_wall = float(runtime_report["ti_anchor_projection"][
            "factor_two_stage_seconds_two_node_contingency"
        ])
        if (not runtime_report["ti_anchor_projection"].get("pass")
                or predicted_wall > available):
            raise TimeoutError(
                "TI anchor factor-two projection exceeds the remaining frozen wall"
            )
    else:
        selected_tier = runtime_report["selected_resource_tier"]
        projection = next(
            value for value in runtime_report["projections"]
            if value["resource_tier"] == selected_tier
        )
        if control["kind"] == "defect_bias":
            predicted_seconds = sum(
                float(runtime_report["bias_tuning_seconds_m8"][
                    entry["task"]["method_id"]
                ])
                for entry in control["tasks"]
            )
        else:
            predicted_seconds = 0.0
            for entry in control["tasks"]:
                task = entry["task"]
                multiplier = (
                    2.0 if str(task["resource_tier"]).startswith("2") else 1.0
                )
                predicted_seconds += multiplier * float(
                    projection["trajectory_seconds_m8"][task["method_id"]]
                )
        predicted_wall = 2.0 * predicted_seconds / 166.0
        if predicted_wall > available:
            raise TimeoutError(
                "factor-two runtime projection exceeds the remaining frozen wall"
            )
    nodes, loads = choose_nodes(args.nodes, args.nd1_busy_threshold)
    print(" ".join(f"{node}_load1={loads[node][0]:.2f}" for node in sorted(loads)), flush=True)

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    if (sha256_file(deployment_root / "SOURCE.tar") != args.archive_sha256
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json") != args.manifest_sha256):
        raise ValueError("global shared deployment hashes mismatch")
    run_root = home / ".single_shot/runs" / args.run_id
    schedule_path = run_root / "control" / "GLOBAL_72H_SCHEDULE.json"
    if not schedule_path.exists():
        frozen_schedule_sha = write_exclusive(schedule_path, schedule)
        if frozen_schedule_sha != args.schedule_file_sha256:
            raise ValueError("global schedule is not canonical JSON")
    elif sha256_file(schedule_path) != args.schedule_file_sha256:
        raise FileExistsError("global frozen schedule conflicts")
    runtime_digest = sha256_file(args.runtime_report)
    runtime_path = run_root / "control" / f"runtime_{runtime_digest[:12]}.json"
    if not runtime_path.exists():
        frozen = json.loads(Path(args.runtime_report).read_text(encoding="ascii"))
        if write_exclusive(runtime_path, frozen) != runtime_digest:
            raise ValueError("global runtime report is not canonical JSON")
    elif sha256_file(runtime_path) != runtime_digest:
        raise FileExistsError("global frozen runtime report conflicts")
    control_path = run_root / "control" / (
        f"{control['stage']}_{control['kind']}_{sha256_file(control_input)[:12]}.json"
    )
    control_sha = write_exclusive(control_path, control)
    tasks = [entry["task"] for entry in control["tasks"]]
    ownership = fixed_global_ownership(
        tasks, nodes, args.source_commit, control_sha,
        f"{control['stage']}:{control['kind']}",
    )
    ownership_path = run_root / "control" / f"ownership_{control_sha[:12]}.json"
    ownership_sha = write_exclusive(ownership_path, ownership)
    fingerprint = ownership["stage_fingerprint"]
    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    for node in nodes:
        stage_dir = (
            run_root / "global" / control["stage"] / "markers"
            / control_sha[:12] / node
        )
        stage_dirs[node] = stage_dir
        if any((stage_dir / name).exists() for name in ("RUNNING", "SUCCESS", "FAILED")):
            raise FileExistsError(f"global stage marker already exists for {node}")
        log = home / ".single_shot/logs" / (
            f"{args.run_id}_{control['stage']}_{control['kind']}_{node}.log"
        )
        module = (
            "data.expander_code.exp102.validation."
            "007_q0_global_discovery_20260721.run_global_stage"
        )
        command = (
            "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
            "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "conda", "run", "-n", "11",
            "--no-capture-output", "python", "-m", module, node,
            "--num-workers", NODE_CAPACITY[node], "--run-id", args.run_id,
            "--source-commit", args.source_commit, "--control", control_path,
            "--control-sha256", control_sha, "--ownership", ownership_path,
            "--ownership-sha256", ownership_sha,
            "--stage-fingerprint", fingerprint,
            "--schedule", schedule_path,
            "--schedule-file-sha256", args.schedule_file_sha256,
        )
        shell = verified_bootstrap(
            deployment_root, args.source_commit, args.archive_sha256,
            args.manifest_sha256, stage_dir, log, fingerprint, command,
        )
        screen = f"exp102_global_{control['stage']}_{control['kind']}_{control_sha[:8]}_{node}"
        screens[node] = screen
        subprocess.run((
            "ssh", node,
            remote_command(("screen", "-dmS", screen, "bash", "-lc", shell)),
        ), check=True)
        print(f"launched node={node} workers={NODE_CAPACITY[node]} screen={screen}", flush=True)
    try:
        wait_for_markers(stage_dirs, fingerprint, timeout)
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
        "runtime_report_sha256": runtime_digest,
        "stage_fingerprint": fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
