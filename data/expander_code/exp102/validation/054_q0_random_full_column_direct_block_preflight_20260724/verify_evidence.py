"""Independent fail-closed audit for validation 054 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parents[1] / "config/q0_random_full_column_direct_block.preflight.v1.json"
REFERENCE = ROOT / "portable_reference.v1.json"
REMOTE = ROOT / "remote_evidence"

SOURCE_COMMIT = "61d605a5e27db0970457736c72d1c45d72a12b10"
ARCHIVE_SHA256 = "61bb87e70320f7371504ea99c320e49baf1140b4ac9d3050fc9a3b742d5a7bec"
MANIFEST_SHA256 = "a6be723a7aa59b7d1305e518b859fdbef50f6b0f881ca08d04088ebe2dcdb49f"
CONFIG_SHA256 = "500c0cb36874168ce1d49501dc37548abc315da6451dbab289ac04f164a1ab78"
REFERENCE_FILE_SHA256 = (
    "03d5b235b05fa23982f18e916f00c483b14be24adf1492fa63e6e8aa2da92ff1"
)
REFERENCE_SHA256 = "a5f20a2d324798db289756d6e9cb1c09fad3fad34672ea2a23b7dee4e38f2d4f"
NODES = ("nd-1", "nd-2", "nd-3")
STATES = ("P", "M0", "S0", "U0")
COLUMNS = (0, 11, 17)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def reference_block_catalog(reference):
    return tuple(
        (row["state"], int(row["column"]), row["block_subtotals_sha256"])
        for row in reference["block_subtotal_catalog"]
    )


def reference_transcript_catalog(reference):
    return tuple(
        (int(row["index"]), row["state"], row["transcript_sha256"])
        for row in reference["runtime_transcript_catalog"]
    )


def verify_stage(stage_dir, expected_stage):
    require(stage_dir.is_dir(), f"missing stage directory: {stage_dir}")
    require({path.name for path in stage_dir.iterdir()} == {"RUNNING", "SUCCESS"},
            f"unexpected marker set: {stage_dir}")
    running = json.loads((stage_dir / "RUNNING").read_text(encoding="ascii"))
    success = json.loads((stage_dir / "SUCCESS").read_text(encoding="ascii"))
    require(running["source_commit"] == success["source_commit"] == SOURCE_COMMIT,
            f"stage source identity changed: {stage_dir}")
    require(running["stage"] == success["stage"] == expected_stage,
            f"stage action changed: {stage_dir}")
    require("started_utc" in running and "completed_utc" in success,
            f"stage timestamps missing: {stage_dir}")


def verify_source_identity(report, expected):
    identity = report["source_identity"]
    core = {"files": identity["files"], "source_commit": identity["source_commit"]}
    require(identity["source_commit"] == SOURCE_COMMIT,
            "report source commit changed")
    require(sha256_json(core) == identity["source_identity_sha256"],
            "source identity self-hash mismatch")
    if expected is not None:
        require(identity == expected, "source identity differs across machines")
    return identity


def verify_report(report, config, reference, expected_node, expected_source):
    verify_self_hash(report, "report_sha256")
    require(report["node"] == expected_node, f"node identity changed: {expected_node}")
    expected_status = (
        "DIRECT_BLOCK_PREFLIGHT_LOCAL_PASS"
        if expected_node == "macmini"
        else "DIRECT_BLOCK_PREFLIGHT_NODE_PASS"
    )
    require(report["status"] == expected_status,
            f"preflight did not pass: {expected_node}")
    require(report["config_sha256"] == CONFIG_SHA256,
            f"config identity changed: {expected_node}")
    require(report["portable_reference_sha256"] == REFERENCE_SHA256,
            f"portable reference identity changed: {expected_node}")
    source = verify_source_identity(report, expected_source)

    checks = report["checks"]
    require(checks == {
        "full_m8_weight_identity": True,
        "local_minimum_streaming_speedup": True,
        "portable_reference": True,
        "runtime_projection": True,
    }, f"preflight checks changed: {expected_node}")
    correctness = report["correctness"]
    require(correctness["all_pass"] is True,
            f"weight identity did not pass: {expected_node}")
    probes = correctness["probes"]
    require(
        tuple((row["state"], int(row["column"])) for row in probes)
        == tuple((state, column) for state in STATES for column in COLUMNS),
        f"correctness panel changed: {expected_node}",
    )
    margin_log = (
        math.log(float.fromhex("0x1.0000000000000p-1022"))
        + math.log(config["correctness"]["minimum_normal_margin_factor"])
    )
    for row in probes:
        require(row["normal_positive_weights"] is True and row["passed"] is True,
                f"nonpositive or failed direct weights: {expected_node}")
        require(math.isfinite(row["block_total"]) and row["block_total"] > 0.0,
                f"invalid block total: {expected_node}")
        require(row["direct_min_weight"] > 0.0
                and math.isfinite(row["direct_max_weight"]),
                f"invalid direct-weight range: {expected_node}")
        require(row["log_candidate_weight_lower_bound"] > margin_log,
                f"underflow margin failed: {expected_node}")
        require(row["max_scaled_weight_absolute_error"]
                <= config["correctness"]["max_scaled_weight_absolute_error"],
                f"scaled weight error failed: {expected_node}")
        require(row["max_relative_weight_error"]
                <= config["correctness"]["max_relative_weight_error"],
                f"relative weight error failed: {expected_node}")
        require(row["total_variation"]
                <= config["correctness"]["max_total_variation"],
                f"total variation failed: {expected_node}")
    require(block_catalog(report) == reference_block_catalog(reference),
            f"portable block catalog changed: {expected_node}")

    timing = report["timing"]
    require(len(timing["direct_seconds"]) == len(timing["streaming_seconds"]) == 3,
            f"timing panel changed: {expected_node}")
    require(all(math.isfinite(value) and value > 0.0
                for value in timing["direct_seconds"] + timing["streaming_seconds"]),
            f"invalid timing: {expected_node}")
    direct_median = statistics.median(timing["direct_seconds"])
    streaming_median = statistics.median(timing["streaming_seconds"])
    require(timing["direct_seconds_median"] == direct_median
            and timing["streaming_seconds_median"] == streaming_median,
            f"timing median changed: {expected_node}")
    require(timing["speedup_over_streaming"] == streaming_median / direct_median,
            f"speedup arithmetic changed: {expected_node}")
    require(timing["speedup_over_streaming"]
            >= config["resource"]["local_min_streaming_speedup"],
            f"speedup gate failed: {expected_node}")

    runtime = report["runtime"]
    require(tuple((int(row["index"]), row["state"]) for row in runtime)
            == tuple(enumerate(STATES)), f"runtime panel changed: {expected_node}")
    for row in runtime:
        updates = int(row["updates"])
        require(updates == (
            config["resource"]["runtime_probe_burn_updates"]
            + config["resource"]["runtime_probe_measurement_updates"]
        ), f"runtime probe clock changed: {expected_node}")
        rate = (row["sampling_seconds"] + row["replay_seconds"]) / updates
        projection = (
            rate
            * (config["resource"]["t1_burn_updates"]
               + config["resource"]["t1_measurement_updates"])
            * config["resource"]["safety_factor"]
        )
        require(math.isclose(row["replay_inclusive_seconds_per_update"], rate,
                             rel_tol=0.0, abs_tol=1e-15),
                f"runtime rate arithmetic changed: {expected_node}")
        require(math.isclose(row["projected_replay_inclusive_t1_seconds"], projection,
                             rel_tol=0.0, abs_tol=1e-9),
                f"runtime projection arithmetic changed: {expected_node}")
        require(projection <= config["resource"]["trajectory_wall_cap_seconds"],
                f"runtime cap failed: {expected_node}")
    worst = max(row["projected_replay_inclusive_t1_seconds"] for row in runtime)
    require(report["worst_projected_replay_inclusive_t1_seconds"] == worst,
            f"worst runtime projection changed: {expected_node}")
    require(transcript_catalog(report) == reference_transcript_catalog(reference),
            f"portable transcript catalog changed: {expected_node}")
    return source


def audit(output):
    output = Path(output)
    require(not output.exists(), "immutable audit output exists")
    require(sha256_file(CONFIG) == CONFIG_SHA256, "config identity changed")
    config = load_canonical(CONFIG)
    require(sha256_file(REFERENCE) == REFERENCE_FILE_SHA256,
            "portable reference file identity changed")
    reference = load_canonical(REFERENCE)
    require(verify_self_hash(reference, "reference_sha256") == REFERENCE_SHA256,
            "portable reference self-hash changed")
    require(config["portable_reference"]["file_sha256"] == REFERENCE_FILE_SHA256
            and config["portable_reference"]["reference_sha256"] == REFERENCE_SHA256,
            "config/reference binding changed")

    local = load_canonical(ROOT / "local_preflight.json")
    source_identity = verify_report(local, config, reference, "macmini", None)

    origin = load_canonical(ROOT / "reference_origin_local_preflight_a0d4dbf.json")
    verify_self_hash(origin, "report_sha256")
    require(origin["report_sha256"] == reference["origin_report_sha256"]
            and origin["source_identity"]["source_commit"]
            == reference["origin_source_commit"],
            "portable reference origin changed")
    superseded = load_canonical(ROOT / "superseded_local_preflight_f5f2976.json")
    verify_self_hash(superseded, "report_sha256")
    require(superseded["source_identity"]["source_commit"]
            == "f5f2976922ced2276f3bcb890bf24410cbc1db00",
            "superseded report source changed")

    reports = []
    for node in NODES:
        report_path = REMOTE / f"preflight/{node}.json"
        report = load_canonical(report_path)
        verify_report(report, config, reference, node, source_identity)
        verify_stage(REMOTE / f"preflight/stages/{node}", "preflight-node")
        require((REMOTE / f"logs/{node}.log").read_bytes() == report_path.read_bytes(),
                f"node log/report mismatch: {node}")
        reports.append(report)

    require(len({block_catalog(report) for report in reports + [local]}) == 1,
            "block-subtotal catalogs differ across machines")
    require(len({transcript_catalog(report) for report in reports + [local]}) == 1,
            "portable transcript catalogs differ across machines")

    aggregate_path = REMOTE / "preflight/aggregate.json"
    aggregate = load_canonical(aggregate_path)
    verify_self_hash(aggregate, "aggregate_sha256")
    require(aggregate["source_commit"] == SOURCE_COMMIT
            and aggregate["source_identity_sha256"]
            == source_identity["source_identity_sha256"]
            and aggregate["config_sha256"] == CONFIG_SHA256
            and aggregate["portable_reference_sha256"] == REFERENCE_SHA256,
            "aggregate source/config/reference binding changed")
    require(aggregate["node_report_sha256"] == {
        report["node"]: report["report_sha256"] for report in reports
    }, "aggregate node bindings changed")
    require(aggregate["node_status"] == {
        node: "DIRECT_BLOCK_PREFLIGHT_NODE_PASS" for node in NODES
    }, "aggregate node statuses changed")
    expected_worst = max(
        report["worst_projected_replay_inclusive_t1_seconds"] for report in reports
    )
    require(aggregate["exact_consensus"] is True
            and aggregate["status"] == "PASS"
            and aggregate["worst_projected_replay_inclusive_t1_seconds"] == expected_worst
            and expected_worst <= config["resource"]["trajectory_wall_cap_seconds"],
            "aggregate terminal decision changed")
    verify_stage(REMOTE / "preflight/stages/combine", "preflight-combine")
    require((REMOTE / "logs/combine.log").read_bytes() == aggregate_path.read_bytes(),
            "aggregate log/report mismatch")

    evidence_paths = sorted(path for path in REMOTE.rglob("*") if path.is_file())
    evidence_paths += [
        ROOT / "local_preflight.json",
        ROOT / "portable_reference.v1.json",
        ROOT / "reference_origin_local_preflight_a0d4dbf.json",
        ROOT / "superseded_local_preflight_f5f2976.json",
    ]
    file_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in evidence_paths
    }
    core = {
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "archive_sha256": ARCHIVE_SHA256,
        "block_subtotal_catalog_sha256": sha256_json(block_catalog(local)),
        "config_sha256": CONFIG_SHA256,
        "evidence_file_sha256": file_hashes,
        "manifest_sha256": MANIFEST_SHA256,
        "node_report_sha256": aggregate["node_report_sha256"],
        "portable_reference_sha256": REFERENCE_SHA256,
        "portable_transcript_catalog_sha256": sha256_json(transcript_catalog(local)),
        "source_commit": SOURCE_COMMIT,
        "status": "INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED",
        "worst_projected_replay_inclusive_t1_seconds": expected_worst,
    }
    result = {**core, "audit_sha256": sha256_json(core)}
    output.write_text(canonical_json(result) + "\n", encoding="ascii")
    print(canonical_json(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output)


if __name__ == "__main__":
    main()
