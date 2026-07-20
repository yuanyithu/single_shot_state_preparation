"""Run source, test, and R128 smoke gates on all three compute nodes."""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess


_HELPER_PATH = Path(__file__).with_name("orchestrate_stage.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("exp102_preflight_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"cannot load preflight helpers from {_HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(HELPERS)

WORKERS = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
RELATIVE = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719")
SMOKE_DIGEST = "5516e079846adaf95fa32504f6dc040101b90ceb164b543009f43d01c11dcb9d"


@dataclass(frozen=True)
class PreflightLaunch:
    node: str
    stage_dir: Path
    log_file: Path
    screen_name: str
    ssh_command: tuple


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_idle_node(node, workers):
    command = (
        "set -e; getconf _NPROCESSORS_ONLN; cut -d' ' -f1 /proc/loadavg; "
        "pgrep -u \"$(id -u)\" -af '[p]ython.*exp102|[r]un_production_stage|[r]un_ladder_stage' || true"
    )
    result = subprocess.run(
        ("ssh", node, command), check=True, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        raise ValueError(f"node resource probe is incomplete: {node}")
    cpus, load1 = int(lines[0]), float(lines[1])
    processes = lines[2:]
    if cpus < workers or load1 >= 5.0 or processes:
        raise RuntimeError(
            f"node is not idle for exp102: {node} cpus={cpus} load1={load1} processes={processes}"
        )
    return {"workers": workers, "online_cpus": cpus, "load1_before": load1,
            "user_compute_processes": processes, "idle": True}


def build_node_launch(run_id, source_commit, archive_sha256, manifest_sha256,
                      node, home=None):
    home = Path.home() if home is None else Path(home)
    deployment_root = home / ".single_shot/repos" / run_id
    run_root = home / ".single_shot/runs" / run_id
    stage_dir = run_root / "preflight" / node
    log_file = home / ".single_shot/logs" / f"{run_id}_preflight_{node}.log"
    screen_name = f"exp102_{run_id}_preflight_{node}"
    script = (
        "set -euo pipefail; "
        f"conda run -n 11 --no-capture-output python {RELATIVE / 'preflight.py'}; "
        "conda run -n 11 --no-capture-output python -m pytest "
        "data/expander_code/exp102/tests -q -p no:cacheprovider; "
        f"conda run -n 11 --no-capture-output python {RELATIVE / 'cross_node_smoke.py'}"
    )
    shell = HELPERS.verified_bootstrap(
        deployment_root, source_commit, archive_sha256, manifest_sha256,
        stage_dir, log_file, ("bash", "-lc", script),
    )
    remote = HELPERS.remote_command(("screen", "-dmS", screen_name, "bash", "-lc", shell))
    return PreflightLaunch(
        node=node, stage_dir=stage_dir, log_file=log_file,
        screen_name=screen_name, ssh_command=("ssh", node, remote),
    )


def write_attestation(run_root, source_commit, archive_sha256, manifest_sha256,
                      launches, probes):
    nodes = {}
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for launch in launches:
        text = launch.log_file.read_text(encoding="utf-8")
        digests = [line.strip() for line in text.splitlines()
                   if digest_pattern.fullmatch(line.strip())]
        if not digests or digests[-1] != SMOKE_DIGEST:
            raise ValueError(f"cross-node smoke digest mismatch: {launch.node}")
        nodes[launch.node] = {
            **probes[launch.node], "smoke_digest": digests[-1],
            "log_sha256": _sha256_file(launch.log_file),
        }
    if set(nodes) != set(WORKERS):
        raise ValueError("preflight attestation does not cover all production nodes")
    attestation = {
        "preflight_version": "exp102.preflight.v1", "status": "PASS",
        "source_commit": source_commit, "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256, "nodes": nodes,
    }
    output = Path(run_root) / "preflight_attestation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with open(temporary, "x", encoding="ascii") as handle:
        json.dump(attestation, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    if output.exists():
        temporary.unlink()
        raise FileExistsError(f"preflight attestation already exists: {output}")
    os.replace(temporary, output)
    return attestation


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if HELPERS.RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run id must contain only letters, digits, dot, underscore, and dash")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout must be a positive finite number")
    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    run_root = home / ".single_shot/runs" / args.run_id
    HELPERS.verify_shared_deployment(
        deployment_root, args.source_commit, args.archive_sha256, args.manifest_sha256,
    )
    probes = {node: probe_idle_node(node, workers) for node, workers in WORKERS.items()}
    launches = tuple(build_node_launch(
        args.run_id, args.source_commit, args.archive_sha256,
        args.manifest_sha256, node, home,
    ) for node in WORKERS)
    HELPERS.check_marker_conflicts(launches)
    for launch in launches:
        subprocess.run(launch.ssh_command, check=True)
    HELPERS.wait_for_terminal_markers(launches, args.timeout_seconds)
    attestation = write_attestation(
        run_root, args.source_commit, args.archive_sha256, args.manifest_sha256,
        launches, probes,
    )
    print(json.dumps(attestation, sort_keys=True))


if __name__ == "__main__":
    main()
