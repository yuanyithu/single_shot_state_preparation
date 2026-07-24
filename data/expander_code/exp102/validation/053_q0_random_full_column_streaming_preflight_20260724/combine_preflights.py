"""Fail-closed aggregate for three immutable streaming preflight reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)


VERSION = "exp102.q0_random_full_column_streaming.preflight.aggregate.v1"
NODES = ("nd-1", "nd-2", "nd-3")
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = EXP102_ROOT / "config/q0_random_full_column_streaming.preflight.v1.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CombineError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CombineError(message)


def load_canonical(path):
    serialized = Path(path).read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_hash(value, field):
    claimed = value[field]
    require(SHA256_RE.fullmatch(str(claimed)) is not None, f"invalid {field}")
    core = {key: item for key, item in value.items() if key != field}
    require(sha256_json(core) == claimed, f"self-hash mismatch: {field}")
    return claimed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("EXP102_SOURCE_COMMIT") == args.source_commit,
            "combiner is outside the verified source wrapper")
    require(not args.output.exists(), "immutable aggregate output exists")
    config_sha = sha256_file(CONFIG_PATH)
    reports = []
    for node in NODES:
        report = load_canonical(args.run_root / f"preflight/{node}.json")
        verify_hash(report, "report_sha256")
        require(report["node"] == node and report["config_sha256"] == config_sha,
                f"node/config identity mismatch: {node}")
        require(report["source_identity"]["source_commit"] == args.source_commit,
                f"source identity mismatch: {node}")
        reports.append(report)

    source_identities = {canonical_json(report["source_identity"]) for report in reports}
    cdf_catalogs = {
        tuple(
            (row["state"], row["column"], row["cdf_sha256"], row["equal"])
            for row in report["equivalence"]["probes"]
        )
        for report in reports
    }
    transcript_catalogs = {
        tuple((row["index"], row["state"], row["transcript_sha256"])
              for row in report["runtime"])
        for report in reports
    }
    exact_consensus = (
        len(source_identities) == 1
        and len(cdf_catalogs) == 1
        and len(transcript_catalogs) == 1
        and all(report["checks"]["m8_cdf_byte_equivalence"] for report in reports)
    )
    if not exact_consensus or any(report["status"] == "CONFLICT" for report in reports):
        status = "CONFLICT"
    elif not all(
        report["status"] == "STREAMING_PREFLIGHT_NODE_PASS"
        and report["checks"]["minimum_speedup"]
        and report["checks"]["runtime_projection"]
        for report in reports
    ):
        status = "RUNTIME_EXHAUSTED"
    else:
        status = "PASS"
    core = {
        "combiner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha,
        "exact_consensus": exact_consensus,
        "node_report_sha256": {
            report["node"]: report["report_sha256"] for report in reports
        },
        "node_status": {report["node"]: report["status"] for report in reports},
        "source_commit": args.source_commit,
        "source_identity_sha256": reports[0]["source_identity"]["source_identity_sha256"],
        "stage_runner_sha256": sha256_file(ROOT / "run_stage.sh"),
        "status": status,
        "version": VERSION,
        "worst_projected_replay_inclusive_t1_seconds": max(
            report["worst_projected_replay_inclusive_t1_seconds"]
            for report in reports
        ),
    }
    aggregate = {**core, "aggregate_sha256": sha256_json(core)}
    atomic_json(args.output, aggregate)
    print(canonical_json(aggregate))


if __name__ == "__main__":
    main()
