"""Independent audit of validation 055's immutable pre-measurement evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "remote_run_r3"
FAILED = ROOT / "failed_schedule_runs"
CONFIG = ROOT.parents[1] / "config/q0_random_full_column_direct_block.t1_m8.v1.json"
OLD_SCHEDULE = (
    ROOT.parent / "052_q0_random_full_column_t1_m8_20260724"
    / "remote_run/control/schedule.json"
)
OUTPUT = ROOT / "independent_preflight_audit.json"

SOURCE_COMMIT = "146ef550591a72435641c47baa8794c338f7a27e"
ARCHIVE_SHA256 = "b960250283d6b986bc7bb20c1ff4aca3238a9c4ecbae7da88512bc6f591e3c48"
MANIFEST_SHA256 = "8daf8d94ece1adeb52b954d13ea34a7062a2bc8995f3e0043144af3eeac144da"
CONFIG_SHA256 = "19d5f64b59170e60c0dc4727da2d3086e299c48934cb81577a33826ff1f32c71"
CONTROL_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.control.v1"
CONTROL_CONTENT_SHA256 = "982fb9318fe423a1d642c118c4efccac247e446da07b6bdea4d8a64dab1b8421"
CONTROL_FILE_SHA256 = "c84579cb2fcd593b176308610a5c69e0fe47f54136b61b9f70a7fff6d94c4168"
CONTROL_MANIFEST_SHA256 = "03847ffe8fa95f4d015298e91da7e663e6f9a20312dee57ecac2a2f4ca41ff2e"
SCHEDULE_SHA256 = "bbc2e268d6e9ed39a2fcae296db3d4dbcb2c49a1f6bf60e6b5678b72b8ee731a"
PORTABLE_CONFIG_SHA256 = "500c0cb36874168ce1d49501dc37548abc315da6451dbab289ac04f164a1ab78"
PORTABLE_REFERENCE_SHA256 = (
    "a5f20a2d324798db289756d6e9cb1c09fad3fad34672ea2a23b7dee4e38f2d4f"
)
PORTABLE_AGGREGATE_SHA256 = (
    "ae356c9e061ae4aea81b6c7a30baec8a744319bcfd8419483f15f8338cfb35ac"
)
T1_AGGREGATE_SHA256 = "7fffcdda598422fab3b33ded26c4acdf77f035932c2cc308c6f8a22a7420f461"
NODES = ("nd-1", "nd-2", "nd-3")
FAMILIES = ("P", "U", "M0", "M1", "S")
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


def control_content_sha(metadata, arrays):
    core = dict(metadata)
    core.pop("control_content_sha256", None)
    digest = hashlib.sha256(CONTROL_VERSION.encode("ascii") + b"\0")
    digest.update(canonical_json(core).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def verify_stage(stage_dir, expected_stage, expected_terminal="SUCCESS"):
    stage_dir = Path(stage_dir)
    require(stage_dir.is_dir(), f"missing stage directory: {stage_dir}")
    names = {path.name for path in stage_dir.iterdir()}
    require(names == {"RUNNING", expected_terminal},
            f"unexpected stage markers: {stage_dir}")
    running = json.loads((stage_dir / "RUNNING").read_text(encoding="ascii"))
    terminal = json.loads(
        (stage_dir / expected_terminal).read_text(encoding="ascii")
    )
    require(running["source_commit"] == terminal["source_commit"] == SOURCE_COMMIT,
            f"stage source changed: {stage_dir}")
    require(running["stage"] == terminal["stage"] == expected_stage,
            f"stage action changed: {stage_dir}")


def portable_block_catalog(report):
    return tuple(
        (row["state"], int(row["column"]), row["block_subtotals_sha256"])
        for row in report["correctness"]["probes"]
    )


def portable_transcript_catalog(report):
    return tuple(
        (int(row["index"]), row["state"], row["transcript_sha256"])
        for row in report["runtime"]
    )


def main():
    require(not OUTPUT.exists(), "immutable preflight audit exists")
    require(sha256_file(CONFIG) == CONFIG_SHA256, "config identity changed")
    config = load_canonical(CONFIG)
    require(config["resource"]["burn_updates"] == 2048
            and config["resource"]["measurement_updates"] == 8192
            and config["resource"]["safety_factor"] == 2.0
            and config["resource"]["trajectory_wall_cap_seconds"] == 7200.0,
            "T1 resource contract changed")

    for run_name in ("r1", "r2"):
        run = FAILED / run_name
        verify_stage(run / "stages/schedule", "schedule", "FAILED")
        message = (run / "logs/schedule.log").read_text(encoding="ascii")
        require("run root must be fresh" in message,
                f"unexpected failed-schedule cause: {run_name}")
        require(not (run / "control/schedule.json").exists()
                and not (run / "preflight").exists()
                and not (run / "measurement").exists(),
                f"failed schedule produced forbidden work: {run_name}")

    verify_stage(RUN / "schedule_stage", "schedule")
    schedule_log = load_canonical(RUN / "schedule.log")
    require(schedule_log == {
        "schedule_sha256": SCHEDULE_SHA256,
        "status": "SCHEDULE_FROZEN",
        "task_count": 40,
    }, "schedule log changed")

    manifest = load_canonical(RUN / "control/control_manifest.json")
    require(verify_self_hash(manifest, "manifest_sha256")
            == CONTROL_MANIFEST_SHA256, "control manifest changed")
    require(manifest["config_sha256"] == CONFIG_SHA256
            and manifest["control_file_sha256"] == CONTROL_FILE_SHA256
            and manifest["control_content_sha256"] == CONTROL_CONTENT_SHA256,
            "control manifest binding changed")
    control_path = RUN / "control/control.npz"
    local_control = ROOT / "control/control.npz"
    require(sha256_file(control_path) == sha256_file(local_control)
            == CONTROL_FILE_SHA256, "remote/local control differs")
    with np.load(control_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: archive[name].copy()
            for name in archive.files if name != "metadata_json"
        }
    require(control_content_sha(metadata, arrays) == CONTROL_CONTENT_SHA256,
            "independent control-content digest changed")

    schedule = load_canonical(RUN / "control/schedule.json")
    require(verify_self_hash(schedule, "schedule_sha256") == SCHEDULE_SHA256,
            "schedule identity changed")
    source = {
        "archive_sha256": ARCHIVE_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": MANIFEST_SHA256,
    }
    require(schedule["source_identity"] == source
            and schedule["config_sha256"] == CONFIG_SHA256
            and schedule["control_content_sha256"] == CONTROL_CONTENT_SHA256,
            "schedule source/control binding changed")
    tasks = schedule["tasks"]
    require(len(tasks) == 40
            and Counter(row["family"] for row in tasks)
            == Counter({family: 8 for family in FAMILIES}),
            "task family panel changed")
    require({row["method_id"] for row in tasks} == {"RFCG-C24-DPB12-S1"},
            "task method changed")
    require({node: len(schedule["ownership"][node]) for node in NODES}
            == {"nd-1": 14, "nd-2": 13, "nd-3": 13},
            "task ownership changed")
    old_tasks = load_canonical(OLD_SCHEDULE)["tasks"]
    for field in (
        "initialization_seed", "burn_update_seed", "measurement_update_seed",
        "observation_seed",
    ):
        fresh = {row[field] for row in tasks}
        old = {row[field] for row in old_tasks}
        require(len(fresh) == 40 and not fresh.intersection(old),
                f"fresh seed schedule collided: {field}")

    portable_reports = []
    for node in NODES:
        path = RUN / f"portable_preflight/preflight/{node}.json"
        report = load_canonical(path)
        verify_self_hash(report, "report_sha256")
        identity = report["source_identity"]
        identity_core = {
            "files": identity["files"], "source_commit": identity["source_commit"],
        }
        require(identity["source_commit"] == SOURCE_COMMIT
                and sha256_json(identity_core) == identity["source_identity_sha256"],
                f"portable source identity changed: {node}")
        require(report["node"] == node
                and report["status"] == "DIRECT_BLOCK_PREFLIGHT_NODE_PASS"
                and report["config_sha256"] == PORTABLE_CONFIG_SHA256
                and report["portable_reference_sha256"]
                == PORTABLE_REFERENCE_SHA256
                and all(report["checks"].values()),
                f"portable node did not pass: {node}")
        resource = config["resource"]
        for row in report["runtime"]:
            rate = (row["sampling_seconds"] + row["replay_seconds"]) / row["updates"]
            projection = rate * (
                resource["burn_updates"] + resource["measurement_updates"]
            ) * 1.2
            require(math.isclose(row["replay_inclusive_seconds_per_update"], rate,
                                 rel_tol=0.0, abs_tol=1e-15)
                    and math.isclose(
                        row["projected_replay_inclusive_t1_seconds"], projection,
                        rel_tol=0.0, abs_tol=1e-9,
                    ), f"portable runtime arithmetic changed: {node}")
        require(report["worst_projected_replay_inclusive_t1_seconds"] <= 7200.0,
                f"portable runtime cap failed: {node}")
        verify_stage(RUN / f"portable_preflight/stages/{node}", "preflight-node")
        require((RUN / f"portable_preflight/logs/{node}.log").read_bytes()
                == path.read_bytes(), f"portable log mismatch: {node}")
        portable_reports.append(report)
    require(len({portable_block_catalog(row) for row in portable_reports}) == 1
            and len({portable_transcript_catalog(row) for row in portable_reports}) == 1,
            "portable cross-node catalog consensus failed")

    portable_aggregate_path = RUN / "portable_preflight/preflight/aggregate.json"
    portable_aggregate = load_canonical(portable_aggregate_path)
    require(verify_self_hash(portable_aggregate, "aggregate_sha256")
            == PORTABLE_AGGREGATE_SHA256, "portable aggregate changed")
    require(portable_aggregate["status"] == "PASS"
            and portable_aggregate["exact_consensus"] is True
            and portable_aggregate["worst_projected_replay_inclusive_t1_seconds"]
            == max(row["worst_projected_replay_inclusive_t1_seconds"]
                   for row in portable_reports),
            "portable aggregate decision changed")
    verify_stage(RUN / "portable_preflight/stages/combine", "preflight-combine")
    require((RUN / "portable_preflight/logs/combine.log").read_bytes()
            == portable_aggregate_path.read_bytes(), "portable aggregate log mismatch")

    t1_reports = []
    for node in NODES:
        path = RUN / f"preflight/{node}.json"
        report = load_canonical(path)
        verify_self_hash(report, "preflight_sha256")
        require(report["node"] == node
                and report["source_identity"] == source
                and report["schedule_sha256"] == SCHEDULE_SHA256
                and report["config_sha256"] == CONFIG_SHA256
                and report["control_content_sha256"] == CONTROL_CONTENT_SHA256,
                f"T1 preflight identity changed: {node}")
        for row in report["probes"]:
            require(row["total_probe_updates"] == 10, "short probe clock changed")
            rate = row["elapsed_seconds"] / row["total_probe_updates"]
            require(row["seconds_per_weighted_update"] == rate,
                    f"short probe rate changed: {node}")
        projection = max(
            row["seconds_per_weighted_update"] for row in report["probes"]
        ) * 10240 * 2.0
        require(report["projected_replay_inclusive_trajectory_seconds"] == projection
                and report["pass_runtime"] is False
                and report["status"] == "RUNTIME_EXHAUSTED"
                and projection > 7200.0,
                f"T1 runtime exhaustion changed: {node}")
        verify_stage(RUN / f"stages/preflight-{node}", "preflight-node")
        require((RUN / f"logs/preflight-{node}.log").read_bytes() == path.read_bytes(),
                f"T1 node log mismatch: {node}")
        t1_reports.append(report)
    require(len({row["mass_sha256"] for row in t1_reports}) == 1
            and len({tuple(probe["portable_transcript_sha256"]
                           for probe in row["probes"]) for row in t1_reports}) == 1,
            "T1 exact consensus changed")

    t1_aggregate_path = RUN / "preflight/aggregate.json"
    t1_aggregate = load_canonical(t1_aggregate_path)
    require(verify_self_hash(t1_aggregate, "preflight_sha256")
            == T1_AGGREGATE_SHA256, "T1 aggregate changed")
    require(t1_aggregate["status"] == "RUNTIME_EXHAUSTED"
            and t1_aggregate["exact_consensus"] is True
            and t1_aggregate["node_status"]
            == {node: "RUNTIME_EXHAUSTED" for node in NODES},
            "T1 aggregate terminal decision changed")
    verify_stage(RUN / "stages/preflight-combine", "preflight-combine")
    require((RUN / "logs/preflight-combine.log").read_bytes()
            == t1_aggregate_path.read_bytes(), "T1 aggregate log mismatch")
    require(not (RUN / "measurement").exists(), "measurement raw exists after failed gate")
    require([path.relative_to(RUN).as_posix() for path in RUN.rglob("*.npz")]
            == ["control/control.npz"], "unexpected NPZ after failed gate")

    evidence_paths = sorted(
        [path for path in RUN.rglob("*") if path.is_file()]
        + [path for path in FAILED.rglob("*") if path.is_file()]
    )
    file_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in evidence_paths
    }
    core = {
        "archive_sha256": ARCHIVE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "control_content_sha256": CONTROL_CONTENT_SHA256,
        "evidence_file_sha256": file_hashes,
        "failed_schedule_runs": ["r1", "r2"],
        "manifest_sha256": MANIFEST_SHA256,
        "portable_aggregate_sha256": PORTABLE_AGGREGATE_SHA256,
        "portable_worst_projection_seconds": (
            portable_aggregate["worst_projected_replay_inclusive_t1_seconds"]
        ),
        "schedule_sha256": SCHEDULE_SHA256,
        "source_commit": SOURCE_COMMIT,
        "status": (
            "INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_"
            "T1_RUNTIME_EXHAUSTED_CONFIRMED"
        ),
        "t1_aggregate_sha256": T1_AGGREGATE_SHA256,
        "t1_worst_projection_seconds": (
            t1_aggregate["worst_projected_replay_inclusive_trajectory_seconds"]
        ),
    }
    result = {**core, "audit_sha256": sha256_json(core)}
    OUTPUT.write_text(canonical_json(result) + "\n", encoding="ascii")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
