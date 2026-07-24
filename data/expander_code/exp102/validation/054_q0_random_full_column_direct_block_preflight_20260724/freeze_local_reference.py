"""Freeze portable direct-block digests before any remote run exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_json,
)


ORIGIN_COMMIT = "a0d4dbf6451240f0c2e07057d45206427ef09db0"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_canonical(path):
    serialized = Path(path).read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", "local report is noncanonical")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "immutable portable reference exists")
    report = load_canonical(args.local_report)
    claimed = report["report_sha256"]
    require(SHA256_RE.fullmatch(str(claimed)) is not None, "invalid report hash")
    report_core = {key: value for key, value in report.items() if key != "report_sha256"}
    require(sha256_json(report_core) == claimed, "local report self-hash mismatch")
    require(report["status"] == "DIRECT_BLOCK_PREFLIGHT_LOCAL_PASS",
            "local preflight did not pass")
    require(report["source_identity"]["source_commit"] == ORIGIN_COMMIT,
            "local preflight source changed")
    require(all(report["checks"].values()), "local preflight checks did not all pass")
    block_catalog = [
        {
            "block_subtotals_sha256": row["block_subtotals_sha256"],
            "column": int(row["column"]),
            "state": row["state"],
        }
        for row in report["correctness"]["probes"]
    ]
    transcript_catalog = [
        {
            "index": int(row["index"]),
            "state": row["state"],
            "transcript_sha256": row["transcript_sha256"],
        }
        for row in report["runtime"]
    ]
    core = {
        "block_subtotal_catalog": block_catalog,
        "origin_config_sha256": report["config_sha256"],
        "origin_report_sha256": report["report_sha256"],
        "origin_source_commit": ORIGIN_COMMIT,
        "origin_source_identity_sha256": (
            report["source_identity"]["source_identity_sha256"]
        ),
        "runtime_transcript_catalog": transcript_catalog,
        "version": "exp102.q0_random_full_column_direct_block.portable_reference.v1",
    }
    reference = {**core, "reference_sha256": sha256_json(core)}
    atomic_json(args.output, reference)
    print(canonical_json(reference))


if __name__ == "__main__":
    main()
