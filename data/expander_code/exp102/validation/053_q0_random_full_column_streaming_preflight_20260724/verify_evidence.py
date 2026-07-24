"""Independent fail-closed audit for validation 053 evidence."""

from __future__ import annotations

import argparse
import json
import math
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


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parents[1] / "config/q0_random_full_column_streaming.preflight.v1.json"
REMOTE = ROOT / "remote_evidence"
SOURCE_COMMIT = "de68bbc06aa729063b24c1f40ba23cc404a44c9c"
ARCHIVE_SHA256 = "e8f14f856cad43d8bf787d7954990054a989a3b49e41efdfa3209ee279986586"
MANIFEST_SHA256 = "08ddeff372f05d8f296893fce37815e97aabc4dd2678495ce4b3079e28460271"
NODES = ("nd-1", "nd-2", "nd-3")
STATES = ("P", "M0", "S0", "U0")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def load_canonical(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_self_hash(value, field):
    claimed = str(value[field])
    require(SHA256_RE.fullmatch(claimed) is not None, f"invalid {field}")
    core = {key: item for key, item in value.items() if key != field}
    require(sha256_json(core) == claimed, f"self-hash mismatch: {field}")
    return claimed


def cdf_catalog(report):
    return tuple(
        (row["state"], int(row["column"]), row["cdf_sha256"])
        for row in report["equivalence"]["probes"]
    )


def transcript_catalog(report):
    return tuple(
        (int(row["index"]), row["state"], row["transcript_sha256"])
        for row in report["runtime"]
    )


def audit(output):
    output = Path(output)
    require(not output.exists(), "immutable audit output exists")
    config = load_canonical(CONFIG)
    config_sha = sha256_file(CONFIG)
    require(config_sha == "6dbd28a893ae6c1532c044ef98645dfa67fec852e19fff50379da4b3eb81a899",
            "config identity changed")

    local = load_canonical(ROOT / "local_preflight.json")
    verify_self_hash(local, "report_sha256")
    require(local["status"] == "STREAMING_PREFLIGHT_LOCAL_PASS",
            "authoritative local preflight did not pass")
    require(local["source_identity"]["source_commit"] == SOURCE_COMMIT,
            "local source identity changed")
    require(local["config_sha256"] == config_sha, "local config identity changed")

    superseded = load_canonical(ROOT / "superseded_local_preflight_7d57bcb.json")
    verify_self_hash(superseded, "report_sha256")
    require(
        superseded["source_identity"]["source_commit"]
        == "7d57bcbbf439eec8a4570c9769ba7f29ddd3aef0",
        "superseded report is not bound to its historical source",
    )

    reports = []
    for node in NODES:
        report_path = REMOTE / f"preflight/{node}.json"
        report = load_canonical(report_path)
        verify_self_hash(report, "report_sha256")
        require(report["node"] == node, f"node identity changed: {node}")
        require(report["status"] == "CONFLICT", f"node did not fail closed: {node}")
        require(report["config_sha256"] == config_sha, f"config changed: {node}")
        require(report["source_identity"] == local["source_identity"],
                f"source identity changed: {node}")
        require((REMOTE / f"preflight/stages/{node}/RUNNING").is_file(),
                f"missing RUNNING marker: {node}")
        require((REMOTE / f"preflight/stages/{node}/SUCCESS").is_file(),
                f"missing SUCCESS marker: {node}")
        require(not (REMOTE / f"preflight/stages/{node}/FAILED").exists(),
                f"unexpected FAILED marker: {node}")
        require((REMOTE / f"logs/{node}.log").read_bytes() == report_path.read_bytes(),
                f"stage log/report mismatch: {node}")

        probes = report["equivalence"]["probes"]
        mismatches = [
            (row["state"], int(row["column"]))
            for row in probes if not row["equal"]
        ]
        require(mismatches == [("U0", 11)],
                f"unexpected dense/streaming mismatch panel: {node}")
        require(not report["checks"]["m8_cdf_byte_equivalence"],
                f"CDF conflict was not persisted: {node}")
        require(not report["checks"]["minimum_speedup"],
                f"speed failure was not persisted: {node}")
        require(not report["checks"]["runtime_projection"],
                f"runtime failure was not persisted: {node}")

        rows = report["runtime"]
        require(tuple(row["state"] for row in rows) == STATES,
                f"runtime state order changed: {node}")
        for row in rows:
            updates = int(row["updates"])
            expected_rate = (row["sampling_seconds"] + row["replay_seconds"]) / updates
            expected_projection = (
                expected_rate
                * (config["resource"]["t1_burn_updates"]
                   + config["resource"]["t1_measurement_updates"])
                * config["resource"]["safety_factor"]
            )
            require(math.isclose(row["replay_inclusive_seconds_per_update"], expected_rate,
                                 rel_tol=0.0, abs_tol=1e-15),
                    f"runtime rate arithmetic changed: {node}")
            require(math.isclose(row["projected_replay_inclusive_t1_seconds"],
                                 expected_projection, rel_tol=0.0, abs_tol=1e-9),
                    f"runtime projection arithmetic changed: {node}")
        worst = max(row["projected_replay_inclusive_t1_seconds"] for row in rows)
        require(worst == report["worst_projected_replay_inclusive_t1_seconds"],
                f"worst runtime projection changed: {node}")
        require(worst > config["resource"]["trajectory_wall_cap_seconds"],
                f"runtime cap unexpectedly passed: {node}")
        reports.append(report)

    require(len({cdf_catalog(report) for report in reports + [local]}) == 1,
            "streaming CDF digest catalogs differ across machines")
    require(len({transcript_catalog(report) for report in reports + [local]}) == 1,
            "portable transcript catalogs differ across machines")

    aggregate_path = REMOTE / "preflight/aggregate.json"
    aggregate = load_canonical(aggregate_path)
    verify_self_hash(aggregate, "aggregate_sha256")
    require(aggregate["status"] == "CONFLICT" and not aggregate["exact_consensus"],
            "aggregate did not preserve the fail-closed outcome")
    require(aggregate["node_status"] == {node: "CONFLICT" for node in NODES},
            "aggregate node status changed")
    require(aggregate["node_report_sha256"] == {
        report["node"]: report["report_sha256"] for report in reports
    }, "aggregate report bindings changed")
    expected_worst = max(
        report["worst_projected_replay_inclusive_t1_seconds"] for report in reports
    )
    require(aggregate["worst_projected_replay_inclusive_t1_seconds"] == expected_worst,
            "aggregate worst runtime changed")
    require((REMOTE / "preflight/stages/combine/RUNNING").is_file(),
            "missing aggregate RUNNING marker")
    require((REMOTE / "preflight/stages/combine/SUCCESS").is_file(),
            "missing aggregate SUCCESS marker")
    require(not (REMOTE / "preflight/stages/combine/FAILED").exists(),
            "unexpected aggregate FAILED marker")
    require((REMOTE / "logs/combine.log").read_bytes() == aggregate_path.read_bytes(),
            "aggregate stage log/report mismatch")

    evidence_paths = sorted(
        path for path in REMOTE.rglob("*") if path.is_file()
    ) + [ROOT / "local_preflight.json", ROOT / "superseded_local_preflight_7d57bcb.json"]
    file_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in evidence_paths
    }
    core = {
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "archive_sha256": ARCHIVE_SHA256,
        "cdf_catalog_sha256": sha256_json(cdf_catalog(local)),
        "config_sha256": config_sha,
        "evidence_file_sha256": file_hashes,
        "manifest_sha256": MANIFEST_SHA256,
        "node_report_sha256": aggregate["node_report_sha256"],
        "portable_transcript_catalog_sha256": sha256_json(transcript_catalog(local)),
        "source_commit": SOURCE_COMMIT,
        "status": "INDEPENDENT_AUDIT_PASS_CONFLICT_AND_RUNTIME_EXHAUSTION_CONFIRMED",
        "worst_projected_replay_inclusive_t1_seconds": expected_worst,
    }
    result = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(output, result)
    print(canonical_json(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output)


if __name__ == "__main__":
    main()
