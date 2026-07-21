"""Run the immutable Linux test, digest, runtime, and WMC preflight on one node."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    verify_source_identity,
)

VALIDATION_MODULE = (
    "data.expander_code.exp102.validation."
    "007_q0_global_discovery_20260721"
)
benchmark_global = importlib.import_module(VALIDATION_MODULE + ".benchmark_global")
cross_node_global = importlib.import_module(VALIDATION_MODULE + ".cross_node_global")
wmc_feasibility = importlib.import_module(VALIDATION_MODULE + ".wmc_feasibility")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=("nd-1", "nd-2", "nd-3"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    output_root = Path(args.output_root).resolve()
    node_root = output_root / "nodes" / args.node
    registry = source / "data/expander_code/exp102/registry/registry.json"
    config = source / "data/expander_code/exp102/config/q0_global.discovery.v1.json"
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "data/expander_code/exp102/tests",
        "data/expander_code/exp101/tests",
    ]
    completed = subprocess.run(
        command, cwd=source, env=environment, check=False,
        capture_output=True, text=True,
    )
    test_log = completed.stdout + completed.stderr
    node_root.mkdir(parents=True, exist_ok=True)
    test_log_path = node_root / "pytest.log"
    test_log_path.write_text(test_log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Linux regression suite failed on {args.node}")

    digest = cross_node_global.canonical_digest(
        registry, config, args.source_commit,
    )
    digest.update({
        "source_commit": args.source_commit,
        "source_identity": source_identity,
        "node": args.node,
        "completed_unix": time.time(),
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
        },
    })
    digest_path = node_root / "digest.json"
    atomic_json(digest_path, digest)

    runtime = benchmark_global.run_benchmark(
        registry, config, args.source_commit,
        verified_source=True, node=args.node,
    )
    runtime_path = node_root / "runtime.json"
    atomic_json(runtime_path, runtime)

    wmc_path = None
    if args.node == "nd-1":
        wmc = wmc_feasibility.run_panel(
            registry, config, max_width=24, timeout_seconds=7200.0,
        )
        wmc.update({
            "source_commit": args.source_commit,
            "source_identity": source_identity,
            "node": args.node,
            "completed_unix": time.time(),
            "environment": {"system": platform.system()},
        })
        wmc_path = node_root / "wmc.json"
        atomic_json(wmc_path, wmc)

    report = {
        "report_version": "exp102.q0_global.preflight_node.v1",
        "status": "PASS",
        "node": args.node,
        "source_commit": args.source_commit,
        "source_identity": source_identity,
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "pytest_returncode": completed.returncode,
        "pytest_log_sha256": hashlib.sha256(
            test_log.encode("utf-8")
        ).hexdigest(),
        "digest_path": digest_path.relative_to(output_root).as_posix(),
        "runtime_path": runtime_path.relative_to(output_root).as_posix(),
        "wmc_path": (
            None if wmc_path is None
            else wmc_path.relative_to(output_root).as_posix()
        ),
        "started_unix": started,
        "completed_unix": time.time(),
    }
    atomic_json(node_root / "preflight.json", report)
    print(json.dumps({
        "node": args.node, "status": report["status"],
        "wall_seconds": report["completed_unix"] - started,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
