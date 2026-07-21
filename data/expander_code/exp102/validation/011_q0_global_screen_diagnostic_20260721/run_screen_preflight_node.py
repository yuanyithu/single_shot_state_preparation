"""Run Linux tests, digest, and sampler-only timing on one verified node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity

from . import benchmark_screen, cross_node_screen
from .common import (
    CONTRACT_VERSION,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    EXPECTED_PREFLIGHT_NODES,
    PREFLIGHT_NODE_VERSION,
    atomic_json,
    resolve_source_path,
    sha256_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=EXPECTED_PREFLIGHT_NODES)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry-relative", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config-relative", default=str(DEFAULT_CONFIG_RELATIVE))
    args = parser.parse_args(argv)

    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    registry = resolve_source_path(source, args.registry_relative)
    config = resolve_source_path(source, args.config_relative)
    output_root = Path(args.output_root).resolve()
    node_root = output_root / "nodes" / args.node
    node_root.mkdir(parents=True, exist_ok=True)
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
    test_log_path = node_root / "pytest.log"
    test_log_path.write_text(test_log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"diagnostic Linux tests failed on {args.node}")
    if verify_source_identity(source, args.source_commit) != source_identity:
        raise RuntimeError("diagnostic Linux tests changed verified source")

    digest = cross_node_screen.canonical_digest(
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

    runtime = benchmark_screen.run_benchmark(
        registry, config, args.source_commit,
        verified_source=True, node=args.node,
    )
    runtime_path = node_root / "runtime.json"
    atomic_json(runtime_path, runtime)

    if verify_source_identity(source, args.source_commit) != source_identity:
        raise RuntimeError("diagnostic preflight changed verified source")
    report = {
        "report_version": PREFLIGHT_NODE_VERSION,
        "contract_version": CONTRACT_VERSION,
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
        "digest_sha256": sha256_file(digest_path),
        "runtime_path": runtime_path.relative_to(output_root).as_posix(),
        "runtime_sha256": sha256_file(runtime_path),
        "excluded_work": ["full_sector_ti", "wmc"],
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
