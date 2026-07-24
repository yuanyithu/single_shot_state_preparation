"""Fail-closed aggregate for direct-block three-node preflights."""

from __future__ import annotations

import argparse
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


VERSION = "exp102.q0_random_full_column_direct_block.preflight.aggregate.v1"
NODES = ("nd-1", "nd-2", "nd-3")
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = EXP102_ROOT / "config/q0_random_full_column_direct_block.preflight.v1.json"
REFERENCE_PATH = ROOT / "portable_reference.v1.json"
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


def block_catalog(report):
    return tuple(
        (row["state"], int(row["column"]), row["block_subtotals_sha256"])
        for row in report["correctness"]["probes"]
    )


def transcript_catalog(report):
    return tuple(
        (int(row["index"]), row["state"], row["transcript_sha256"])
        for row in report["runtime"]
    )


def verify_source_identity(identity, expected_commit):
    core = {
        "files": identity["files"],
        "source_commit": identity["source_commit"],
    }
    require(identity["source_commit"] == expected_commit,
            "source identity commit changed")
    require(sha256_json(core) == identity["source_identity_sha256"],
            "source identity self-hash changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("EXP102_SOURCE_COMMIT") == args.source_commit,
            "combiner is outside the verified source wrapper")
    require(not args.output.exists(), "immutable direct-block aggregate exists")
    config = load_canonical(CONFIG_PATH)
    config_sha = sha256_file(CONFIG_PATH)
    reference = load_canonical(REFERENCE_PATH)
    verify_hash(reference, "reference_sha256")
    require(sha256_file(REFERENCE_PATH) == config["portable_reference"]["file_sha256"],
            "portable reference file changed")
    require(reference["reference_sha256"]
            == config["portable_reference"]["reference_sha256"],
            "portable reference identity changed")
    reports = []
    for node in NODES:
        report = load_canonical(args.run_root / f"preflight/{node}.json")
        verify_hash(report, "report_sha256")
        require(report["node"] == node and report["config_sha256"] == config_sha,
                f"node/config identity mismatch: {node}")
        require(report["source_identity"]["source_commit"] == args.source_commit,
                f"source identity mismatch: {node}")
        verify_source_identity(report["source_identity"], args.source_commit)
        require(report["portable_reference_sha256"] == reference["reference_sha256"],
                f"portable reference mismatch: {node}")
        reports.append(report)

    source_identities = {canonical_json(report["source_identity"]) for report in reports}
    block_catalogs = {block_catalog(report) for report in reports}
    transcript_catalogs = {transcript_catalog(report) for report in reports}
    exact_consensus = (
        len(source_identities) == 1
        and len(block_catalogs) == 1
        and len(transcript_catalogs) == 1
        and all(report["checks"]["full_m8_weight_identity"] for report in reports)
        and all(report["checks"]["portable_reference"] for report in reports)
    )
    if not exact_consensus or any(report["status"] == "CONFLICT" for report in reports):
        status = "CONFLICT"
    elif not all(
        report["status"] == "DIRECT_BLOCK_PREFLIGHT_NODE_PASS"
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
        "portable_reference_sha256": reference["reference_sha256"],
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
