"""Independently verify the archived third q=0 global preflight attempt."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "remote_run"
NODES = ("nd-1", "nd-2", "nd-3")
SOURCE_COMMIT = "204b37d8e00e7d11ffa2b6766b90d947892e179d"
ARCHIVE_SHA256 = "1583dce6b8bb81ad7780f323d21300b158ad435d710f3c0226b7b3028b8eb7f7"
MANIFEST_SHA256 = "b69290798a11a3bf548483c6e223f96a64e0d9c7be0e48b89fa6e54a28a57ea3"
CONFIG_SHA256 = "1d0a453f2bf8445ad6587c612c2eabb3049e76e2d73b59c230b8b1358b06e565"
REGISTRY_SHA256 = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
SCHEDULE_FILE_SHA256 = "7874a0d967ba866d8834cf380b408947af614bdf3bec7b50c0f30fb4a332465c"
SCHEDULE_SHA256 = "35e08b457f6a96eea252bc8d6653950aecb231ac85b6ac66c129f799ca0d02c1"
STAGE_FINGERPRINT = "0ae347a314c23ecba2d7239af6d6203f6e4e8f91dd5111810bdb3a1fb89ab538"
CANONICAL_DIGEST = "a3730d7380575976f88e35f5490b24a9b6949e3817b2fb3880775736cf2ad364"
METHODS = (
    "RC8-QC1", "RC8-QC4", "RC8-J08", "RC8-J12", "RC8-J16",
    "DT16", "DT32", "DT64",
)
EXPECTED_RUNTIME_STATUS = {
    "nd-1": "PASS",
    "nd-2": "RUNTIME_EXHAUSTED",
    "nd-3": "RUNTIME_EXHAUSTED",
}
EXPECTED_TI_SECONDS = {
    "nd-1": 78705.40290523804,
    "nd-2": 116274.88993334108,
    "nd-3": 251240.6735108179,
}
EXPECTED_LOG_SHA256 = {
    "nd-1": "f38c0cc53083e92d09e8ae8ac22b0fafe5f90b734c68416c22112bc2ed62ee6c",
    "nd-2": "f1891937d9114b3d4e72eb71dc5514e53f043f41242420fff9e17b22f0b7e405",
    "nd-3": "8dcd964de246c871d7b1449befdf61ec26d59868a6899451878517c12704d0e0",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def verify_source_axes(report):
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["registry_sha256"] == REGISTRY_SHA256
    assert report["discovery_config_sha256"] == CONFIG_SHA256
    assert report["source_identity"] == {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "file_count": 776,
    }


def main():
    schedule_path = RUN_ROOT / "control/GLOBAL_72H_SCHEDULE.json"
    assert sha256_file(schedule_path) == SCHEDULE_FILE_SHA256
    schedule = load_json(schedule_path)
    identity = {key: value for key, value in schedule.items()
                if key != "schedule_sha256"}
    assert schedule["schedule_sha256"] == SCHEDULE_SHA256 == sha256_json(identity)
    assert schedule["source_commit"] == SOURCE_COMMIT
    assert schedule["archive_sha256"] == ARCHIVE_SHA256
    assert schedule["source_manifest_sha256"] == MANIFEST_SHA256
    assert schedule["registry_sha256"] == REGISTRY_SHA256
    assert schedule["discovery_config_sha256"] == CONFIG_SHA256

    preflight_root = RUN_ROOT / "global/preflight"
    marker_root = preflight_root / "markers" / STAGE_FINGERPRINT[:12]
    t3_hours = {}
    ti_seconds = {}
    for node in NODES:
        node_root = preflight_root / "nodes" / node
        preflight = load_json(node_root / "preflight.json")
        assert preflight["report_version"] == "exp102.q0_global.preflight_node.v1"
        assert preflight["status"] == "PASS"
        assert preflight["node"] == node
        assert preflight["pytest_returncode"] == 0
        assert schedule["started_unix"] <= preflight["started_unix"]
        assert preflight["completed_unix"] <= schedule["deadlines_unix"]["digest_runtime"]
        assert preflight["source_commit"] == SOURCE_COMMIT
        assert preflight["source_identity"]["archive_sha256"] == ARCHIVE_SHA256
        assert preflight["source_identity"]["manifest_sha256"] == MANIFEST_SHA256
        assert sha256_file(node_root / "pytest.log") == preflight["pytest_log_sha256"]

        markers = sorted(path.name for path in (marker_root / node).iterdir()
                         if path.name != "stage.lock")
        assert markers == ["SUCCESS"]
        marker = load_json(marker_root / node / "SUCCESS")
        assert marker["stage_fingerprint"] == STAGE_FINGERPRINT

        digest = load_json(node_root / "digest.json")
        verify_source_axes(digest)
        assert digest["node"] == node
        assert digest["canonical_digest"] == CANONICAL_DIGEST

        runtime = load_json(node_root / "runtime.json")
        verify_source_axes(runtime)
        assert runtime["benchmark_version"] == "exp102.q0_global.runtime.v1"
        assert runtime["node"] == node
        assert runtime["status"] == EXPECTED_RUNTIME_STATUS[node]
        assert runtime["selected_resource_tier"] == "T3"
        assert tuple(runtime["selected_eligible_methods"]) == METHODS
        assert runtime["checks"]["all_numeric_finite"]
        assert runtime["checks"]["at_least_T1_fits"]
        assert runtime["checks"]["hard_method_available"]
        assert runtime["checks"]["defect_method_available"]
        assert runtime["checks"]["ti_anchor_fits_confirmation_window"] is (
            node == "nd-1"
        )
        t3 = next(value for value in runtime["projections"]
                  if value["resource_tier"] == "T3")
        assert t3["pass"] and math.isfinite(t3["projected_hours_with_safety_factor_2"])
        t3_hours[node] = t3["projected_hours_with_safety_factor_2"]
        ti = runtime["ti_anchor_projection"]
        assert ti["pass"] is (node == "nd-1")
        assert ti["factor_two_stage_seconds_two_node_contingency"] == (
            EXPECTED_TI_SECONDS[node]
        )
        ti_seconds[node] = ti["factor_two_stage_seconds_two_node_contingency"]

        log = ROOT / "remote_logs" / (
            f"exp102_q0_global_20260721_204b37d_global_preflight_{node}.log"
        )
        log_sha256 = sha256_file(log)
        assert log_sha256 == EXPECTED_LOG_SHA256[node], (
            node, log_sha256, EXPECTED_LOG_SHA256[node],
        )

    wmc = load_json(preflight_root / "nodes/nd-1/wmc.json")
    verify_source_axes(wmc)
    assert wmc["report_version"] == "exp102.q0_global.wmc_feasibility.v1"
    assert wmc["status"] == "INCONCLUSIVE"
    assert wmc["node"] == "nd-1"
    assert len(wmc["records"]) == 6
    assert all(value["status"] == "INCONCLUSIVE_WIDTH"
               for value in wmc["records"])

    # The old orchestrator raised while combining the valid exhausted reports.
    assert not (RUN_ROOT / "control/runtime_consensus.json").exists()
    assert not (RUN_ROOT / "control/preflight_report.json").exists()
    assert sorted(path.name for path in (RUN_ROOT / "global").iterdir()) == [
        "preflight"
    ]

    print(json.dumps({
        "status": "VERIFIED_RUNTIME_EXHAUSTED",
        "source_commit": SOURCE_COMMIT,
        "schedule_file_sha256": SCHEDULE_FILE_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "canonical_digest": CANONICAL_DIGEST,
        "node_runtime_status": EXPECTED_RUNTIME_STATUS,
        "t3_projected_hours_with_safety_factor_2": t3_hours,
        "ti_factor_two_stage_seconds": ti_seconds,
        "wmc_status": wmc["status"],
        "sampler_raw_files": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
