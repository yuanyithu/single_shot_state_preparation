"""Launch one exp102 pilot stage on nd-2 and nd-3 from nd-0."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


WORKERS = {"nd-2": 75, "nd-3": 91}
STAGES = ("ladder", "gamma", "rounds", "held_out")
RELATIVE = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719")
FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MARKERS = ("RUNNING", "SUCCESS", "FAILED")


@dataclass(frozen=True)
class NodeLaunch:
    node: str
    workers: int
    stage_dir: Path
    log_file: Path
    screen_name: str
    stage_command: tuple
    shell: str
    ssh_command: tuple


def remote_command(arguments):
    return " ".join(shlex.quote(str(value)) for value in arguments)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_shared_deployment(deployment_root, source_commit, archive_sha256, manifest_sha256):
    deployment_root = Path(deployment_root)
    source = deployment_root / "source"
    archive = deployment_root / "SOURCE.tar"
    commit_marker = deployment_root / "SOURCE_COMMIT"
    archive_marker = deployment_root / "ARCHIVE_SHA256"
    manifest_path = deployment_root / "SOURCE_MANIFEST.json"
    if (not source.is_dir()
            or any(not path.is_file() for path in (
                archive, commit_marker, archive_marker, manifest_path,
            ))):
        raise ValueError("shared deployment bundle is incomplete")
    if commit_marker.read_text(encoding="ascii").strip() != source_commit:
        raise ValueError("shared deployment commit marker mismatch")
    if archive_marker.read_text(encoding="ascii").strip() != archive_sha256:
        raise ValueError("shared deployment archive marker mismatch")
    if _sha256_file(archive) != archive_sha256:
        raise ValueError("shared deployment archive SHA256 mismatch")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("shared deployment manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if (manifest.get("source_commit") != source_commit
            or manifest.get("archive_sha256") != archive_sha256):
        raise ValueError("shared deployment manifest identity mismatch")


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file, command):
    """Build a shell command whose first executable bytes come from the archive."""
    if FULL_GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(archive_sha256) is None:
        raise ValueError("archive SHA256 must be 64 lowercase hex characters")
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest SHA256 must be 64 lowercase hex characters")
    archive = deployment_root / "SOURCE.tar"
    verifier = RELATIVE / "run_verified_source.sh"
    wrapper = RELATIVE / "run_stage_wrapper.sh"
    guarded_command = (
        "set -euo pipefail; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(verifier.as_posix())} "
        f"| bash -s -- {shlex.quote(str(deployment_root))} "
        f"{shlex.quote(source_commit)} {shlex.quote(archive_sha256)} "
        f"{shlex.quote(manifest_sha256)} {remote_command(command)}"
    )
    return (
        "set -euo pipefail; "
        f"printf '%s  %s\\n' {shlex.quote(archive_sha256)} {shlex.quote(str(archive))} "
        "| sha256sum -c - >/dev/null; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(wrapper.as_posix())} "
        f"| bash -s -- {shlex.quote(str(stage_dir))} {shlex.quote(str(log_file))} "
        f"bash -c {shlex.quote(guarded_command)}"
    )


def parse_m_values(value):
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("m-values must be comma-separated integers") from error
    if not parsed or len(parsed) != len(set(parsed)) or not set(parsed) <= set(range(3, 9)):
        raise ValueError("m-values must be a nonempty, duplicate-free subset of 3..8")
    return tuple(sorted(parsed))


def validate_by_m_config(path, m_values):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("by-m config must be a regular file")
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("by-m config must be valid ASCII JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("by-m config must be a JSON object")
    try:
        keys = {int(key) for key in raw}
    except (TypeError, ValueError) as error:
        raise ValueError("by-m config keys must be integer m values") from error
    if len(keys) != len(raw) or keys != set(m_values):
        raise ValueError("by-m config keys must exactly match m-values")
    if not all(isinstance(candidate, dict) for candidate in raw.values()):
        raise ValueError("each by-m config value must be a JSON object")
    return path


def snapshot_by_m_config(path, m_values, run_root, stage, attempt):
    """Publish one immutable shared control file before either node launches."""
    source = validate_by_m_config(path, m_values)
    raw = json.loads(source.read_text(encoding="ascii"))
    payload = (json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("ascii")
    target = Path(run_root) / "control" / f"{stage}_attempt_{attempt:03d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise FileExistsError(f"stage control snapshot already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target, hashlib.sha256(payload).hexdigest()


def build_node_launch(run_id, source_commit, archive_sha256, manifest_sha256,
                      stage, attempt, by_m_config, m_values, node, home=None):
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run id must contain only letters, digits, dot, underscore, and dash")
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError("attempt must be a nonnegative integer")
    if node not in WORKERS:
        raise ValueError(f"unsupported node: {node}")
    if FULL_GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(archive_sha256) is None:
        raise ValueError("archive SHA256 must be 64 lowercase hex characters")
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest SHA256 must be 64 lowercase hex characters")

    home = Path.home() if home is None else Path(home)
    deployment_root = home / ".single_shot/repos" / run_id
    source = deployment_root / "source"
    run_root = home / ".single_shot/runs" / run_id
    workers = WORKERS[node]
    by_m_config = Path(by_m_config)
    by_m_config_sha256 = hashlib.sha256(by_m_config.read_bytes()).hexdigest()
    stage_dir = run_root / stage / f"attempt_{attempt:03d}" / node
    log_file = home / ".single_shot/logs" / f"{run_id}_{stage}_a{attempt:03d}_{node}.log"
    screen_name = f"exp102_{run_id}_{stage}_a{attempt:03d}_{node}"
    stage_command = (
        "env",
        f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
        "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "conda", "run", "-n", "11", "--no-capture-output", "python",
        RELATIVE / "run_ladder_stage.py", node,
        "--num-workers", workers,
        "--run-id", run_id,
        "--source-commit", source_commit,
        "--stage", stage,
        "--attempt", attempt,
        "--by-m-config", Path(by_m_config),
        "--by-m-config-sha256", by_m_config_sha256,
        "--m-values", ",".join(str(m) for m in m_values),
    )
    shell = verified_bootstrap(
        deployment_root, source_commit, archive_sha256, manifest_sha256,
        stage_dir, log_file, stage_command,
    )
    remote = remote_command(("screen", "-dmS", screen_name, "bash", "-lc", shell))
    return NodeLaunch(
        node=node,
        workers=workers,
        stage_dir=stage_dir,
        log_file=log_file,
        screen_name=screen_name,
        stage_command=stage_command,
        shell=shell,
        ssh_command=("ssh", node, remote),
    )


def marker_state(launch):
    present = [marker for marker in MARKERS if (launch.stage_dir / marker).exists()]
    return "+".join(present) if present else "MISSING"


def check_marker_conflicts(launches):
    conflicts = []
    for launch in launches:
        for marker in MARKERS:
            marker_path = launch.stage_dir / marker
            if marker_path.exists():
                conflicts.append(str(marker_path))
    if conflicts:
        raise FileExistsError("stage markers already exist: " + ", ".join(conflicts))


def wait_for_terminal_markers(launches, timeout_seconds, poll_seconds=2.0):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be a positive finite number")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll interval must be a positive finite number")
    launches = tuple(launches)
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        states = {launch.node: marker_state(launch) for launch in launches}
        if states != previous:
            print(" ".join(f"{node}={state}" for node, state in sorted(states.items())), flush=True)
            previous = states
        failed = [node for node, state in states.items() if "FAILED" in state.split("+")]
        if failed:
            raise RuntimeError("stage failed on " + ", ".join(sorted(failed)))
        if states and all(state == "SUCCESS" for state in states.values()):
            return states
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = ", ".join(f"{node}={state}" for node, state in sorted(states.items()))
            raise TimeoutError(f"stage produced no dual-node SUCCESS within {timeout_seconds:g} seconds ({detail})")
        time.sleep(min(poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--by-m-config", required=True)
    parser.add_argument("--m-values", required=True)
    parser.add_argument(
        "--timeout-seconds", "--attempt-timeout-seconds",
        dest="timeout_seconds", type=float, default=86400.0,
    )
    args = parser.parse_args(argv)

    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run id must contain only letters, digits, dot, underscore, and dash")
    if args.attempt < 0:
        raise ValueError("attempt must be nonnegative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout must be a positive finite number")
    m_values = parse_m_values(args.m_values)
    input_by_m_config = validate_by_m_config(args.by_m_config, m_values)
    deployment_root = Path.home() / ".single_shot/repos" / args.run_id
    verify_shared_deployment(
        deployment_root, args.source_commit, args.archive_sha256, args.manifest_sha256,
    )
    for node in WORKERS:
        subprocess.run(("ssh", node, "true"), check=True)
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    preliminary = tuple(
        build_node_launch(
            args.run_id, args.source_commit, args.archive_sha256,
            args.manifest_sha256, args.stage, args.attempt,
            input_by_m_config, m_values, node,
        )
        for node in WORKERS
    )
    check_marker_conflicts(preliminary)
    by_m_config, _ = snapshot_by_m_config(
        input_by_m_config, m_values, run_root, args.stage, args.attempt,
    )
    launches = tuple(
        build_node_launch(
            args.run_id, args.source_commit, args.archive_sha256,
            args.manifest_sha256, args.stage, args.attempt,
            by_m_config, m_values, node,
        )
        for node in WORKERS
    )

    for launch in launches:
        subprocess.run(launch.ssh_command, check=True)
        print(
            f"launched node={launch.node} workers={launch.workers} "
            f"screen={launch.screen_name} log={launch.log_file}",
            flush=True,
        )
    wait_for_terminal_markers(launches, args.timeout_seconds)
    print(
        json.dumps({
            "attempt": args.attempt,
            "m_values": list(m_values),
            "stage": args.stage,
            "status": "SUCCESS",
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
