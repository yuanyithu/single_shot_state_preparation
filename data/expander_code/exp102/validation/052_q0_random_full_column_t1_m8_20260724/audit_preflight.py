"""Independent audit of the immutable three-node T1 preflight evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "remote_run"
OUTPUT = ROOT / "preflight_audit.json"

SOURCE_COMMIT = "6fa489f838dffea15b07e1ef3b3fbee3951dd3c0"
ARCHIVE_SHA256 = "670f0768b163ecd7daf9112e1da405787c1df117b805dac8c57d6c2ec02b398e"
SOURCE_MANIFEST_SHA256 = (
    "d5dc8c118fdcfc1b895a46718ca5a9be642d46e419b5ee697d2d6e55a67d8bb3"
)
CONFIG_SHA256 = "952c65491883423b21e4c51015d167b56489f33b99b369ff0dfdebd2db5c0a85"
CONTROL_FILE_SHA256 = (
    "a43865186be0865ba8f1eac35ec22354ebe92ea6528091ce32e6f6dcaa118a41"
)
CONTROL_CONTENT_SHA256 = (
    "b99fb047e787fd999cde113bd3c64a1e9ef0e41e805d79a3d6d5f7995b6b8df6"
)
CONTROL_MANIFEST_SHA256 = (
    "336d3e24a0f65970d4fcaa24de7f292798aedd89a6dcb548a06a37c73afb33cc"
)
SCHEDULE_SHA256 = "22de98602f48ef0aceed23b18e564875a9a6d8c41a4971a9cf8356256cc538eb"
NODES = ("nd-1", "nd-2", "nd-3")
FAMILIES = ("P", "U", "M0", "M1", "S")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical(path: Path) -> dict:
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_self_hash(value: dict, field: str) -> str:
    claimed = value[field]
    core = {key: item for key, item in value.items() if key != field}
    actual = hashlib.sha256(canonical_json(core).encode("ascii")).hexdigest()
    require(claimed == actual, f"self-hash mismatch: {field}")
    return claimed


def control_content_sha(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    core = dict(metadata)
    core.pop("control_content_sha256", None)
    digest = hashlib.sha256(b"exp102.q0_random_full_column.t1_m8.control.v0\0")
    digest.update(canonical_json(core).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def verify_stage(stage_dir: Path, expected_stage: str) -> None:
    require(stage_dir.is_dir(), f"missing stage directory: {stage_dir}")
    require({path.name for path in stage_dir.iterdir()} == {"RUNNING", "SUCCESS"},
            f"unexpected stage marker set: {stage_dir}")
    running = json.loads((stage_dir / "RUNNING").read_text(encoding="ascii"))
    success = json.loads((stage_dir / "SUCCESS").read_text(encoding="ascii"))
    require(running["source_commit"] == success["source_commit"] == SOURCE_COMMIT,
            "stage source identity mismatch")
    require(running["stage"] == success["stage"] == expected_stage,
            "stage action mismatch")
    require("started_utc" in running and "completed_utc" in success,
            "stage timestamps missing")


def evidence_package_sha() -> tuple[str, int]:
    paths = sorted(
        [path for path in RUN.rglob("*") if path.is_file()]
        + [ROOT / "remote_schedule.log"]
        + sorted(path for path in (ROOT / "remote_schedule_stage").rglob("*") if path.is_file())
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(paths)


def main() -> None:
    require(not OUTPUT.exists(), "immutable preflight audit already exists")
    require(RUN.is_dir(), "remote run evidence is missing")

    manifest = load_canonical(RUN / "control/control_manifest.json")
    require(verify_self_hash(manifest, "manifest_sha256") == CONTROL_MANIFEST_SHA256,
            "control manifest identity mismatch")
    require(manifest["config_sha256"] == CONFIG_SHA256
            and manifest["control_file_sha256"] == CONTROL_FILE_SHA256
            and manifest["control_content_sha256"] == CONTROL_CONTENT_SHA256,
            "control binding mismatch")
    remote_control = RUN / "control/control.npz"
    local_control = ROOT / "control/control.npz"
    require(sha256_file(remote_control) == sha256_file(local_control) == CONTROL_FILE_SHA256,
            "remote control differs from frozen local control")
    with np.load(remote_control, allow_pickle=False) as archive:
        require("metadata_json" in archive.files, "control metadata missing")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: archive[name].copy() for name in archive.files if name != "metadata_json"
        }
    require(control_content_sha(metadata, arrays) == CONTROL_CONTENT_SHA256,
            "independent control-content digest mismatch")

    schedule = load_canonical(RUN / "control/schedule.json")
    require(verify_self_hash(schedule, "schedule_sha256") == SCHEDULE_SHA256,
            "schedule identity mismatch")
    source_identity = {
        "archive_sha256": ARCHIVE_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    require(schedule["source_identity"] == source_identity
            and schedule["config_sha256"] == CONFIG_SHA256
            and schedule["control_file_sha256"] == CONTROL_FILE_SHA256
            and schedule["control_content_sha256"] == CONTROL_CONTENT_SHA256
            and schedule["control_manifest_sha256"] == CONTROL_MANIFEST_SHA256,
            "schedule source/control binding mismatch")
    tasks = schedule["tasks"]
    require(len(tasks) == 40, "task count changed")
    require(Counter(task["family"] for task in tasks) == Counter({name: 8 for name in FAMILIES}),
            "initialization-family panel changed")
    require(len({task["task_fingerprint"] for task in tasks}) == 40,
            "task fingerprints collided")
    for seed_field in (
        "initialization_seed", "burn_update_seed", "measurement_update_seed",
        "observation_seed",
    ):
        require(len({task[seed_field] for task in tasks}) == 40,
                f"seed collision: {seed_field}")
    ownership_counts = {node: len(schedule["ownership"][node]) for node in NODES}
    require(ownership_counts == {"nd-1": 14, "nd-2": 13, "nd-3": 13},
            "ownership changed")
    require(all(
        schedule["ownership"][node]
        == [task["task_fingerprint"] for task in tasks if task["owner"] == node]
        for node in NODES
    ), "task ownership ordering changed")

    reports = []
    for node in NODES:
        report_path = RUN / f"preflight/{node}.json"
        report = load_canonical(report_path)
        verify_self_hash(report, "preflight_sha256")
        require(report["node"] == node and report["source_identity"] == source_identity
                and report["schedule_sha256"] == SCHEDULE_SHA256
                and report["config_sha256"] == CONFIG_SHA256
                and report["control_content_sha256"] == CONTROL_CONTENT_SHA256,
                f"node identity mismatch: {node}")
        probes = report["probes"]
        require([row["index"] for row in probes] == [0, 1, 2, 3],
                f"probe catalog changed: {node}")
        for row in probes:
            require(math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0,
                    f"invalid probe timing: {node}")
            require(row["total_probe_updates"] == 10, f"probe clock changed: {node}")
            require(row["seconds_per_weighted_update"]
                    == row["elapsed_seconds"] / row["total_probe_updates"],
                    f"probe timing derivation mismatch: {node}")
        projection = max(row["seconds_per_weighted_update"] for row in probes) * 10240 * 2.0
        require(report["projected_replay_inclusive_trajectory_seconds"] == projection,
                f"runtime projection mismatch: {node}")
        require(report["pass_runtime"] is False
                and report["status"] == "RUNTIME_EXHAUSTED"
                and projection > 7200.0,
                f"runtime terminal status mismatch: {node}")
        require((RUN / f"logs/preflight-node-{node}.log").read_bytes()
                == report_path.read_bytes(), f"node log/report mismatch: {node}")
        verify_stage(RUN / f"stages/preflight-node-{node}", "preflight-node")
        reports.append(report)

    require(len({report["mass_sha256"] for report in reports}) == 1,
            "mass-table consensus failed")
    transcript_catalogs = {
        tuple(row["portable_transcript_sha256"] for row in report["probes"])
        for report in reports
    }
    require(len(transcript_catalogs) == 1, "portable-transcript consensus failed")

    aggregate_path = RUN / "preflight/aggregate.json"
    aggregate = load_canonical(aggregate_path)
    aggregate_self_hash = verify_self_hash(aggregate, "preflight_sha256")
    require(aggregate["source_identity"] == source_identity
            and aggregate["schedule_sha256"] == SCHEDULE_SHA256
            and aggregate["config_sha256"] == CONFIG_SHA256,
            "aggregate identity mismatch")
    require(aggregate["node_preflight_sha256"]
            == {report["node"]: report["preflight_sha256"] for report in reports},
            "aggregate node hashes changed")
    require(aggregate["node_status"]
            == {node: "RUNTIME_EXHAUSTED" for node in NODES},
            "aggregate node status changed")
    worst_projection = max(
        report["projected_replay_inclusive_trajectory_seconds"] for report in reports
    )
    require(aggregate["exact_consensus"] is True
            and aggregate["status"] == "RUNTIME_EXHAUSTED"
            and aggregate["worst_projected_replay_inclusive_trajectory_seconds"]
            == worst_projection,
            "aggregate terminal decision mismatch")
    require((RUN / "logs/preflight-combine.log").read_bytes() == aggregate_path.read_bytes(),
            "aggregate log/report mismatch")
    verify_stage(RUN / "stages/preflight-combine", "preflight-combine")
    verify_stage(ROOT / "remote_schedule_stage", "schedule")
    require((ROOT / "remote_schedule.log").read_text(encoding="ascii")
            == canonical_json({
                "schedule_sha256": SCHEDULE_SHA256,
                "status": "SCHEDULE_FROZEN",
                "task_count": 40,
            }) + "\n", "schedule log mismatch")

    measurement_files = []
    if (RUN / "measurement").exists():
        measurement_files = [path for path in (RUN / "measurement").rglob("*") if path.is_file()]
    require(not measurement_files, "measurement evidence exists despite failed runtime gate")
    package_sha256, package_file_count = evidence_package_sha()
    core = {
        "aggregate_preflight_sha256": aggregate_self_hash,
        "config_sha256": CONFIG_SHA256,
        "evidence_package_file_count": package_file_count,
        "evidence_package_sha256": package_sha256,
        "exact_consensus": True,
        "measurement_raw_count": 0,
        "node_projection_seconds": {
            report["node"]: report["projected_replay_inclusive_trajectory_seconds"]
            for report in reports
        },
        "node_status": {node: "RUNTIME_EXHAUSTED" for node in NODES},
        "ownership_counts": ownership_counts,
        "schedule_sha256": SCHEDULE_SHA256,
        "source_identity": source_identity,
        "status": "INDEPENDENT_PREFLIGHT_AUDIT_PASS_RUNTIME_EXHAUSTED_CONFIRMED",
        "task_count": len(tasks),
        "version": "exp102.q0_random_full_column.t1_m8.preflight_audit.v0",
        "worst_projected_replay_inclusive_trajectory_seconds": worst_projection,
    }
    report = {
        **core,
        "audit_sha256": hashlib.sha256(canonical_json(core).encode("ascii")).hexdigest(),
    }
    OUTPUT.write_text(canonical_json(report) + "\n", encoding="ascii")
    print(canonical_json(report))


if __name__ == "__main__":
    main()
