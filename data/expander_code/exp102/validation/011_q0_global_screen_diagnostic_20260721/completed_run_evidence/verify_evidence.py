"""Verify the metadata-only evidence for the completed 342dd5b screen run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_evidence"
CONTROL = REMOTE / "control"
DEPLOYMENT = ROOT / "deployment_identity"
INDEPENDENT = ROOT / "independent_replay"

RUN_ID = "exp102_q0_screen_diagnostic_20260721_342dd5b"
SOURCE_COMMIT = "342dd5bc0fb2c7694dbc58a8d0f2d92689c24991"
ARCHIVE_SHA256 = "4a54ba28f3ee2add94e93dd052e4bda567d5e008691f84a098c21768b4fe11f3"
MANIFEST_SHA256 = "2b8ab6d238d6319ea73c4c5da0ecf815a3d2e2ea28932dddc30bd40afe158b01"
REGISTRY_SHA256 = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
CONFIG_SHA256 = "e5fa2ebdc2f22f25342d3d8d5c5ab05027685a4def6aecf8e48e666fa72f468b"
CONTRACT_VERSION = "exp102.q0_global.screen_diagnostic.v1"

SCHEDULE_FILE_SHA256 = "f9aeccd95640a56fabe813796d0e1ce388cffa1bcccf2405a6bafcd913520832"
SCHEDULE_SHA256 = "cd09b4701d54b061f59db5ce50df191edac0b23da62e411ccc5a597400426cb9"
RUNTIME_SHA256 = "5776a8c309b8599e2abeff930e46fd60a6f89dfe5e403c1e753e0a4e030d3cd7"
DIGEST_SHA256 = "65eb086653a9b6d21ea10545050de0106e45e9ebc0f89353b5c124e3982b48a1"
PREFLIGHT_SHA256 = "3225e4c80c11a559fffae88b76fbd2dd2ef63a1d98eebdf1692889189aaa59ee"
CANONICAL_DIGEST = "080b3170ca168dc3f237d22a4d18403eb2c0b0b2455e6d1e3ca876aae39c86a9"
GAMMA_SHA256 = "a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a"
PREFLIGHT_FINGERPRINT = "32884c4f7fbbb93edbf2ac5379518f7434d3c4466443a8ea729b4004e537dcc2"

BIAS_CONTROL_SHA256 = "5661ba2d718c70f730355433fe1fe7f8a25e0d849e5d40575bd071c168621837"
BIAS_MANIFEST_SHA256 = "2096e73918b327f5dce74693ff2d34d2ad199556066b7ed0e88200efdf7bd9a5"
BIAS_OWNERSHIP_SHA256 = "7264d2753bf251919f20bea37de6fdb7be9098448b2b0430d07dcae97cd8744f"
BIAS_EVIDENCE_SHA256 = "5ecfce4eeb888ad3a50ef317d3bd006b1b6e54c717138b6ed77a26266f3056c7"
BIAS_FINGERPRINT = "a0933e485c0337c9633d7afe6b4afffbae78ce6ae67b43e9e437f8c87641f9d4"

MEASUREMENT_CONTROL_SHA256 = "16d19159c05766ed229be8b451ed71f0cae61b26d7bc024d3fcde84c171b4c9b"
MEASUREMENT_MANIFEST_SHA256 = "d6bc446c857e11df5e430968d6ce98775527c070e6a7ac48b5c0c4018bab4674"
MEASUREMENT_OWNERSHIP_SHA256 = "31f9b49c3b301f21de92f779490e39cfc0d9b0d37bb6432d36bbef69c5b9d2f3"
MEASUREMENT_EVIDENCE_SHA256 = "f682c42306b367891cc63965c6b6fe5c069990e5f0302156cf6276b5189143b8"
MEASUREMENT_FINGERPRINT = "e89ec6bfbb5cc3d90bcfdd4f836ac3914c079ff22ea0ab3970ebdfad740224d5"

REPORT_FILE_SHA256 = "04598fc8263488b46ad6efe135b2dba06af2d2464c94d79731e97f8a54d65628"
REPORT_IDENTITY_SHA256 = "70b9b6e6d1618058a292a49b5095bfe7e56a5e8f4cad4497ae788cead370ce36"
DECISION_FILE_SHA256 = "45491758ffa8e7e36509a1a02faa12fa1c1cb11277e133706460a271fb17aa1b"
DECISION_IDENTITY_SHA256 = "d267bccf9314d7520a6e89dc238d8ddc0f35777f2872fe0376e3cd626f17afcd"
ORIGINAL_PACKAGE_FILE_SHA256 = (
    "83155d17e54fa2597ba8bce48ac99a8667a3dfc4296589a93e89dcc0cfd5cae7"
)
ORIGINAL_PACKAGE_IDENTITY_SHA256 = (
    "0e0fb2f950eb609c984b29f5647321694c82f8f7a6810609fd1742d1472a990a"
)
INDEPENDENT_REPORT_FILE_SHA256 = (
    "39f88606e7c9767701779dd955de617cf57c86c21e722a9d68e01e0cb12a05ee"
)
INDEPENDENT_REPORT_IDENTITY_SHA256 = (
    "35230903c9e13c5ed1be8b7e1904060decc5076e09a78cc9c7c7b8eb2af85389"
)
INDEPENDENT_DECISION_FILE_SHA256 = (
    "aad7708e3e89d5aef9049f9cde36744cf88762223632906b524eb1e32c058ee8"
)
INDEPENDENT_DECISION_IDENTITY_SHA256 = (
    "8b5aa4c2a497617db866a617d487f258b1026c127d45ee453d73205040cf8f8d"
)
INDEPENDENT_PACKAGE_FILE_SHA256 = (
    "84f23cb942db8e85a5c323470437dccec04e7bcfea2c4e64a6ee371902b4a25a"
)
INDEPENDENT_PACKAGE_IDENTITY_SHA256 = (
    "e4a9838b9bfba73993cd3699f7629a848b04be819b6526b18e9b467999f2de60"
)
DRIVER_SHA256 = "1bd1180ffd5777c65c74f00b3498f711736f29a170d7ec3f5fb2ba1ac636a5b5"
DRIVER_LOG_SHA256 = "cb621886d2cc9be7b999f96f37e7c3a1ed8aca6145a74be805290f0d8c769586"
RAW_SHA256SUMS_SHA256 = "4affe3dc996bfbd60382778fda5e12be9cd77f3cdd0b6b9bafed67873ccde79a"

NODES = ("nd-1", "nd-2", "nd-3")
EXECUTION_NODES = ("nd-1", "nd-3")
HARD_METHODS = ("RC8-QC1", "RC8-QC4", "RC8-J08", "RC8-J12", "RC8-J16")
DEFECT_METHODS = ("DT16", "DT32", "DT64")
METHODS = HARD_METHODS + DEFECT_METHODS
METHOD_TIERS = [[method, "T3"] for method in METHODS]
FORMAL_BLOCKERS = [
    "NO_T_VS_2T",
    "NO_FRESH_HARD2_CONFIRMATION",
    "NO_CONF17_RES6_GAP8_SMALL6",
    "NO_TI_OR_REVIEWED_INDEPENDENT_ORACLE",
    "NO_HELD_OUT",
]
FINAL_STATUS = "UNRESOLVED_NO_HARD_COSET_PASS"

SOURCE_IDENTITY = {
    "source_commit": SOURCE_COMMIT,
    "mode": "archive",
    "archive_sha256": ARCHIVE_SHA256,
    "manifest_sha256": MANIFEST_SHA256,
    "file_count": 854,
}

DIGEST_RECORDS = [
    {
        "kind": "catalog_and_characters",
        "catalog_sha256": "7ec00dd84ce3302d5b0c7948804512c55fe6f3bec73c88691fc5bfd1164bb273",
        "catalog_size": 512,
        "character_sha256": "56c47deedab1187cd80c8d52b10055b25a67cfd31c8ce169709b014d159427c8",
        "character_count": 4160,
    },
    {
        "kind": "RC8-QC1",
        "digest": "129e51641e2d6eddd7aa0f43f5059912afe53906710d2ad1afc568bf1c02e7c2",
    },
    {
        "kind": "RC8-J08",
        "digest": "e52eb63a2272125c39f03027f5323b63cea026d10ad83dec8ca2e7148d57a0b3",
    },
    {
        "kind": "DT16_fixed_bias",
        "digest": "0b71fe603c48010bcc6c29267c4dde8dc8bcdff29f3cf3f6ccece7eebe7892f0",
    },
    {
        "kind": "DT16_bias_tuning",
        "digest": "79ce0c5e387d668175157adc100eae3b0edce343ea18a7737e9fdf407e812ecf",
        "gamma_count": 4096,
        "gamma_sha256": GAMMA_SHA256,
    },
]

CELLS = (
    {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m03_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m04_c00", "p": 0.07, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m05_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SUM_LINE_RE = re.compile(r"([0-9a-f]{64})  ([^\x00\r\n]+)")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _ordered_float_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    sign = 1 << 63
    return (~bits & ((1 << 64) - 1)) if bits & sign else bits | sign


def _verify_replay_report_equivalence(original, independent):
    """Allow only the audited cross-platform reduction/FFT roundoff."""
    allowed_float_keys = {"core_seconds", "min_nondegenerate_bulk_ess"}
    stats = {"float_differences": 0, "max_ulp": 0}

    def compare(left, right, path):
        _require(type(left) is type(right), f"replay type mismatch: {path}")
        if isinstance(left, dict):
            _require(set(left) == set(right), f"replay key mismatch: {path}")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list):
            _require(len(left) == len(right), f"replay length mismatch: {path}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left == right:
            return
        leaf = path.rsplit(".", 1)[-1]
        _require(
            isinstance(left, float) and leaf in allowed_float_keys,
            f"replay value mismatch outside float whitelist: {path}",
        )
        _require(
            math.isfinite(left) and math.isfinite(right),
            f"replay non-finite float: {path}",
        )
        ulp = abs(_ordered_float_bits(left) - _ordered_float_bits(right))
        _require(ulp <= 4, f"replay float differs by more than 4 ULP: {path}")
        stats["float_differences"] += 1
        stats["max_ulp"] = max(stats["max_ulp"], ulp)

    original_body = {
        key: value for key, value in original.items() if key != "report_sha256"
    }
    independent_body = {
        key: value for key, value in independent.items() if key != "report_sha256"
    }
    compare(original_body, independent_body, "report")
    return stats


def _parse_sums(path):
    rows = {}
    order = []
    for line_number, line in enumerate(
            Path(path).read_text(encoding="ascii").splitlines(), start=1):
        match = SUM_LINE_RE.fullmatch(line)
        _require(match is not None, f"malformed checksum line {line_number}: {path}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute()
            and "\\" not in relative
            and relative == pure.as_posix()
            and pure.parts
            and all(part not in {"", ".", ".."} for part in pure.parts),
            f"unsafe checksum path: {relative}",
        )
        _require(relative not in rows, f"duplicate checksum path: {relative}")
        rows[relative] = digest
        order.append(relative)
    _require(order == sorted(order), f"checksum rows are not sorted: {path}")
    return rows


def _verify_evidence_closure():
    expected = _parse_sums(ROOT / "EVIDENCE_SHA256SUMS")
    _require("EVIDENCE_SHA256SUMS" not in expected, "checksum file is self-listed")
    entries = list(ROOT.rglob("*"))
    _require(not any(path.is_symlink() for path in entries), "evidence contains symlinks")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in entries
        if path.is_file() and path != ROOT / "EVIDENCE_SHA256SUMS"
    }
    _require(set(expected) == actual, "EVIDENCE_SHA256SUMS closure mismatch")
    _require(not any(path.endswith(".npz") for path in actual), "raw NPZ entered metadata bundle")
    for relative, digest in expected.items():
        _require(_sha256(ROOT / relative) == digest, f"evidence hash mismatch: {relative}")


def _verify_source_identity(value, context):
    _require(value == SOURCE_IDENTITY, f"source identity mismatch: {context}")


def _verify_axes(value, context, *, config_key="diagnostic_config_sha256"):
    _require(value.get("source_commit") == SOURCE_COMMIT, f"source mismatch: {context}")
    _require(value.get("registry_sha256") == REGISTRY_SHA256, f"registry mismatch: {context}")
    _require(value.get(config_key) == CONFIG_SHA256, f"config mismatch: {context}")


def _verify_no_authorization(value, context):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "formal_authorization", "formal_readiness_authorized",
                "formal_production_authorized", "production_authorization",
                "production_authorized",
            }:
                _require(child is False, f"authorization present: {context}.{key}")
            _verify_no_authorization(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_no_authorization(child, f"{context}[{index}]")


def _verify_deployment_identity():
    _require(
        (DEPLOYMENT / "SOURCE_COMMIT").read_text(encoding="ascii")
        == SOURCE_COMMIT + "\n",
        "deployment commit file mismatch",
    )
    _require(
        (DEPLOYMENT / "ARCHIVE_SHA256").read_text(encoding="ascii")
        == ARCHIVE_SHA256 + "\n",
        "deployment archive file mismatch",
    )
    manifest_path = DEPLOYMENT / "SOURCE_MANIFEST.json"
    _require(_sha256(manifest_path) == MANIFEST_SHA256, "source manifest hash mismatch")
    manifest = _load(manifest_path)
    _require(set(manifest) == {
        "source_identity_version", "source_commit", "archive_sha256", "files",
    }, "source manifest schema mismatch")
    _require(manifest["source_identity_version"] == "exp102.source.v1", "source version mismatch")
    _require(manifest["source_commit"] == SOURCE_COMMIT, "source manifest commit mismatch")
    _require(manifest["archive_sha256"] == ARCHIVE_SHA256, "source manifest archive mismatch")
    files = manifest["files"]
    _require(isinstance(files, list) and len(files) == 854, "source file count mismatch")
    _require(
        all(set(row) == {"path", "sha256"} and SHA256_RE.fullmatch(row["sha256"])
            for row in files),
        "source manifest row malformed",
    )
    paths = [row["path"] for row in files]
    _require(paths == sorted(paths) and len(paths) == len(set(paths)), "source paths not canonical")
    for relative in paths:
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute()
            and "\\" not in relative
            and relative == pure.as_posix()
            and pure.parts
            and all(part not in {"", ".", ".."} for part in pure.parts),
            f"unsafe source manifest path: {relative}",
        )
    source_hashes = {row["path"]: row["sha256"] for row in files}
    expected_source_hashes = {
        "data/expander_code/exp102/config/q0_global.screen_diagnostic.v1.json":
            "db9eb38d562bc15f4aaab44c8629c41d68f201bb2ee6a355df3d60c18a5c171f",
        "data/expander_code/exp102/registry/registry.json":
            "8016f21935a707a972044c9ab7144e7d3812567af76880f56e3e0fe80d503024",
        "data/expander_code/exp102/exp102_pipeline/q0_global.py":
            "ab42914a9c45b81fd99ec2ce8e230ac98dc4649d2aa8b77914ec3fea87649396",
        "data/expander_code/exp102/tests/test_q0_global.py":
            "e9aa2ec08ffc2fda88073b31f2c9556bb0c0d3c9b74dc131c52e0d9c9f3af71e",
    }
    for relative, digest in expected_source_hashes.items():
        _require(source_hashes.get(relative) == digest, f"key source hash mismatch: {relative}")
    _require(_sha256(DEPLOYMENT / "SCREEN_DRIVER.sh") == DRIVER_SHA256, "driver hash mismatch")


def _verify_schedule():
    path = CONTROL / "SCREEN_DIAGNOSTIC_24H_SCHEDULE.json"
    _require(_sha256(path) == SCHEDULE_FILE_SHA256, "schedule file hash mismatch")
    schedule = _load(path)
    identity = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    _require(schedule["schedule_sha256"] == SCHEDULE_SHA256, "schedule identity changed")
    _require(_sha256_json(identity) == SCHEDULE_SHA256, "schedule self-hash mismatch")
    _verify_axes(schedule, "schedule", config_key="screen_config_sha256")
    _require(schedule["schedule_version"] == "exp102.q0_global.screen_diagnostic.schedule.v1", "schedule version mismatch")
    _require(schedule["status"] == "FROZEN_24H_DIAGNOSTIC", "schedule status mismatch")
    _require(schedule["archive_sha256"] == ARCHIVE_SHA256, "schedule archive mismatch")
    _require(schedule["source_manifest_sha256"] == MANIFEST_SHA256, "schedule manifest mismatch")
    _require(schedule["wall_limit_hours"] == 24, "schedule wall changed")
    _require(schedule["production_authorized"] is False, "schedule authorized production")
    started = float(schedule["started_unix"])
    offsets = {"preflight": 8, "bias": 12, "measurement": 22, "analysis": 24}
    _require(math.isfinite(started) and started > 0.0, "schedule start invalid")
    for stage, hours in offsets.items():
        _require(
            float(schedule["deadlines_unix"][stage]) == started + hours * 3600.0,
            f"schedule deadline changed: {stage}",
        )
    return schedule


def _verify_success_marker(directory, fingerprint):
    _require(directory.is_dir(), f"missing marker directory: {directory}")
    _require(
        sorted(path.name for path in directory.iterdir()) == ["SUCCESS", "stage.lock"],
        f"marker is not exclusive SUCCESS: {directory}",
    )
    _require((directory / "stage.lock").read_bytes() == b"", f"stage lock is not empty: {directory}")
    marker = _load(directory / "SUCCESS")
    _require(set(marker) == {"stage_fingerprint", "completed_utc"}, "SUCCESS schema mismatch")
    _require(marker["stage_fingerprint"] == fingerprint, "SUCCESS fingerprint mismatch")


def _verify_preflight(schedule):
    runtime_path = CONTROL / "screen_runtime_consensus.json"
    digest_path = CONTROL / "screen_digest_consensus.json"
    preflight_path = CONTROL / "screen_preflight_report.json"
    _require(_sha256(runtime_path) == RUNTIME_SHA256, "runtime consensus hash mismatch")
    _require(_sha256(digest_path) == DIGEST_SHA256, "digest consensus hash mismatch")
    _require(_sha256(preflight_path) == PREFLIGHT_SHA256, "preflight report hash mismatch")
    runtime = _load(runtime_path)
    digest = _load(digest_path)
    preflight = _load(preflight_path)

    _verify_axes(runtime, "runtime consensus")
    _verify_source_identity(runtime.get("source_identity"), "runtime consensus")
    _require(runtime["benchmark_version"] == "exp102.q0_global.screen_diagnostic.runtime_consensus.v1", "runtime version mismatch")
    _require(runtime["contract_version"] == CONTRACT_VERSION and runtime["status"] == "PASS", "runtime did not pass")
    _require(runtime["selected_resource_tier"] == "T3", "runtime tier mismatch")
    _require(runtime["selected_eligible_methods"] == list(METHODS), "runtime methods mismatch")
    _require(set(runtime["node_report_sha256"]) == set(NODES), "runtime node hashes changed")
    _require(runtime["excluded_work"] == ["full_sector_ti", "wmc"], "runtime scope changed")
    _require(runtime["environment"] == {"nodes": list(NODES), "system": "Linux"}, "runtime environment mismatch")
    _require(
        schedule["started_unix"] <= runtime["completed_unix_max"]
        <= schedule["deadlines_unix"]["preflight"],
        "runtime consensus timing mismatch",
    )
    projections = runtime["projections"]
    _require([row["resource_tier"] for row in projections] == ["T1", "T2", "T3"], "runtime projections changed")
    for row in projections:
        _require(row["pass"] is True, "runtime projection failed")
        _require(row["eligible_methods"] == list(METHODS), "runtime eligibility changed")
        _require(row["execution_nodes"] == list(EXECUTION_NODES), "runtime execution nodes changed")
        _require(row["execution_capacity"] == 166 and row["safety_factor"] == 2.0, "runtime capacity changed")

    _verify_axes(digest, "digest consensus")
    _verify_source_identity(digest.get("source_identity"), "digest consensus")
    _require(digest["report_version"] == "exp102.q0_global.screen_diagnostic.digest_consensus.v2", "digest version mismatch")
    _require(digest["contract_version"] == CONTRACT_VERSION and digest["status"] == "PASS", "digest did not pass")
    _require(digest["nodes"] == list(NODES), "digest nodes changed")
    _require(set(digest["node_report_sha256"]) == set(NODES), "digest node hashes changed")
    _require(digest["canonical_digest"] == CANONICAL_DIGEST, "canonical digest changed")
    _require(digest["records"] == DIGEST_RECORDS, "digest records changed")
    _require(
        schedule["started_unix"] <= digest["completed_unix_max"]
        <= schedule["deadlines_unix"]["preflight"],
        "digest consensus timing mismatch",
    )
    _require(
        _sha256_json(DIGEST_RECORDS) == CANONICAL_DIGEST,
        "canonical digest is not linked to its records",
    )

    _verify_axes(preflight, "preflight")
    _require(preflight["report_version"] == "exp102.q0_global.screen_diagnostic.preflight.v1", "preflight version mismatch")
    _require(preflight["contract_version"] == CONTRACT_VERSION and preflight["status"] == "PASS", "preflight did not pass")
    _require(preflight["stage"] == "preflight" and preflight["nodes"] == list(NODES), "preflight nodes changed")
    _require(set(preflight["node_report_sha256"]) == set(NODES), "preflight node hashes changed")
    _require(preflight["archive_sha256"] == ARCHIVE_SHA256, "preflight archive mismatch")
    _require(preflight["source_manifest_sha256"] == MANIFEST_SHA256, "preflight manifest mismatch")
    _require(preflight["schedule_file_sha256"] == SCHEDULE_FILE_SHA256, "preflight schedule file mismatch")
    _require(preflight["schedule_sha256"] == SCHEDULE_SHA256, "preflight schedule mismatch")
    _require(preflight["runtime_consensus_sha256"] == RUNTIME_SHA256, "preflight runtime link mismatch")
    _require(preflight["digest_consensus_sha256"] == DIGEST_SHA256, "preflight digest link mismatch")
    _require(preflight["canonical_digest"] == CANONICAL_DIGEST, "preflight canonical digest mismatch")
    _require(preflight["selected_resource_tier"] == "T3", "preflight tier mismatch")
    _require(preflight["selected_eligible_methods"] == list(METHODS), "preflight methods mismatch")
    _require(preflight["excluded_work"] == ["full_sector_ti", "wmc"], "preflight scope changed")
    _require(preflight["maximum_terminal_status"] == "DIAGNOSTIC_SCREEN_PAIR_FOUND", "preflight authority changed")
    _require(
        schedule["started_unix"] <= float(preflight["completed_unix"])
        <= schedule["deadlines_unix"]["preflight"],
        "preflight timing mismatch",
    )
    fingerprint_identity = {
        key: preflight[key] for key in (
            "contract_version", "stage", "source_commit", "archive_sha256",
            "source_manifest_sha256", "schedule_file_sha256", "schedule_sha256",
            "registry_sha256", "diagnostic_config_sha256", "registry_relative",
            "config_relative", "nodes",
        )
    }
    _require(_sha256_json(fingerprint_identity) == PREFLIGHT_FINGERPRINT, "preflight fingerprint invalid")
    _require(preflight["stage_fingerprint"] == PREFLIGHT_FINGERPRINT, "preflight fingerprint changed")

    node_root = REMOTE / "screen_diagnostic/preflight/nodes"
    marker_root = REMOTE / f"screen_diagnostic/preflight/markers/{PREFLIGHT_FINGERPRINT[:12]}"
    for node in NODES:
        root = node_root / node
        report_path = root / "preflight.json"
        node_runtime_path = root / "runtime.json"
        node_digest_path = root / "digest.json"
        pytest_path = root / "pytest.log"
        report = _load(report_path)
        node_runtime = _load(node_runtime_path)
        node_digest = _load(node_digest_path)
        _require(_sha256(report_path) == preflight["node_report_sha256"][node], f"node preflight hash mismatch: {node}")
        _require(_sha256(node_runtime_path) == runtime["node_report_sha256"][node], f"node runtime hash mismatch: {node}")
        _require(_sha256(node_digest_path) == digest["node_report_sha256"][node], f"node digest hash mismatch: {node}")
        _require(report["report_version"] == "exp102.q0_global.screen_diagnostic.preflight_node.v1", f"node preflight version: {node}")
        _require(report["contract_version"] == CONTRACT_VERSION and report["status"] == "PASS", f"node preflight failed: {node}")
        _require(report["node"] == node and report["source_commit"] == SOURCE_COMMIT, f"node preflight identity: {node}")
        _verify_source_identity(report.get("source_identity"), f"node preflight {node}")
        _require(report["environment"].get("system") == "Linux", f"node preflight platform: {node}")
        _require(report["pytest_returncode"] == 0, f"node tests failed: {node}")
        _require(report["excluded_work"] == ["full_sector_ti", "wmc"], f"node scope changed: {node}")
        _require(report["digest_path"] == f"nodes/{node}/digest.json", f"node digest path: {node}")
        _require(report["runtime_path"] == f"nodes/{node}/runtime.json", f"node runtime path: {node}")
        _require(report["digest_sha256"] == _sha256(node_digest_path), f"node digest link: {node}")
        _require(report["runtime_sha256"] == _sha256(node_runtime_path), f"node runtime link: {node}")
        _require(report["pytest_log_sha256"] == _sha256(pytest_path), f"node pytest link: {node}")
        pytest_summary = {
            "nd-1": "619 passed, 3 skipped, 2 warnings",
            "nd-2": "619 passed, 3 skipped, 2 warnings",
            "nd-3": "622 passed, 2 warnings",
        }[node]
        _require(pytest_summary in pytest_path.read_text(encoding="utf-8"), f"node test count: {node}")
        _require(schedule["started_unix"] <= report["started_unix"] <= report["completed_unix"] <= schedule["deadlines_unix"]["preflight"], f"node timing: {node}")

        _verify_axes(node_runtime, f"node runtime {node}")
        _verify_source_identity(node_runtime.get("source_identity"), f"node runtime {node}")
        _require(node_runtime["benchmark_version"] == "exp102.q0_global.screen_diagnostic.runtime_node.v1", f"node runtime version: {node}")
        _require(node_runtime["contract_version"] == CONTRACT_VERSION and node_runtime["status"] == "PASS", f"node runtime failed: {node}")
        _require(node_runtime["node"] == node and node_runtime["environment"].get("system") == "Linux", f"node runtime platform: {node}")
        _require(all(node_runtime["checks"].values()), f"node runtime checks: {node}")
        _require(node_runtime["selected_resource_tier"] == "T3", f"node runtime tier: {node}")
        _require(node_runtime["selected_eligible_methods"] == list(METHODS), f"node runtime methods: {node}")
        _require(schedule["started_unix"] <= node_runtime["completed_unix"] <= schedule["deadlines_unix"]["preflight"], f"node runtime timing: {node}")
        _require([row["resource_tier"] for row in node_runtime["projections"]] == ["T1", "T2", "T3"], f"node runtime projections: {node}")
        _require(all(row["pass"] and row["eligible_methods"] == list(METHODS) for row in node_runtime["projections"]), f"node runtime projection failed: {node}")

        _verify_axes(node_digest, f"node digest {node}")
        _verify_source_identity(node_digest.get("source_identity"), f"node digest {node}")
        _require(node_digest["digest_version"] == "exp102.q0_global.screen_diagnostic.digest_node.v2", f"node digest version: {node}")
        _require(node_digest["contract_version"] == CONTRACT_VERSION, f"node digest contract: {node}")
        _require(node_digest["node"] == node and node_digest["environment"].get("system") == "Linux", f"node digest platform: {node}")
        _require(node_digest["canonical_digest"] == CANONICAL_DIGEST, f"node canonical digest: {node}")
        _require(node_digest["records"] == DIGEST_RECORDS, f"node digest records: {node}")
        _require(schedule["started_unix"] <= node_digest["completed_unix"] <= schedule["deadlines_unix"]["preflight"], f"node digest timing: {node}")
        _verify_success_marker(marker_root / node, PREFLIGHT_FINGERPRINT)
    return runtime, digest, preflight


def _cell_key(cell):
    return _sha256_json(cell)


def _expected_sampler(task):
    method = task["method_id"]
    common = {
        "method_id": method,
        "p": task["cell"]["p"],
        "burn_sweeps": 8192,
        "measurement_sweeps": 32768,
    }
    if method in DEFECT_METHODS:
        return {
            **common,
            "dmax": {"DT16": 16, "DT32": 32, "DT64": 64}[method],
            "K_q": 0.0,
            "tuning_chains": 8,
            "tuning_sweeps": 4096,
        }
    if method in {"RC8-QC1", "RC8-QC4"}:
        return {
            **common,
            "cluster_repeats": {"RC8-QC1": 1, "RC8-QC4": 4}[method],
            "joint_block_size": 0,
        }
    return {
        **common,
        "cluster_repeats": 0,
        "joint_block_size": {
            "RC8-J08": 8, "RC8-J12": 12, "RC8-J16": 16,
        }[method],
    }


def _verify_seed(seed, task, *, family, index, namespace):
    _require(set(seed) == {
        "seed_root", "trajectory_namespace", "source_commit",
        "registry_sha256", "config_sha256", "cell_fingerprint",
        "method_id", "resource_tier", "init_family", "trajectory_index",
    }, "seed identity schema mismatch")
    _require(seed["seed_root"] == "q0_global_screen_diagnostic_v1", "seed root changed")
    _require(seed["trajectory_namespace"] == namespace, "seed namespace changed")
    _require(seed["source_commit"] == SOURCE_COMMIT, "seed source changed")
    _require(seed["registry_sha256"] == REGISTRY_SHA256, "seed registry changed")
    _require(seed["config_sha256"] == CONFIG_SHA256, "seed config changed")
    _require(seed["cell_fingerprint"] == _cell_key(task["cell"]), "seed cell changed")
    _require(seed["method_id"] == task["method_id"], "seed method changed")
    _require(seed["resource_tier"] == "T3", "seed tier changed")
    _require(seed["init_family"] == family, "seed family changed")
    _require(seed["trajectory_index"] == index, "seed index changed")


def _verify_control(kind):
    if kind == "bias":
        input_name = "screen_bias_input.json"
        frozen_name = f"screen_bias_{BIAS_CONTROL_SHA256[:12]}.json"
        control_sha = BIAS_CONTROL_SHA256
        manifest_sha = BIAS_MANIFEST_SHA256
        expected_count = 15
        expected_methods = DEFECT_METHODS
        raw_folder = "bias"
        control_kind = "diagnostic_defect_bias"
    else:
        input_name = "screen_measurement_input.json"
        frozen_name = f"screen_measurement_{MEASUREMENT_CONTROL_SHA256[:12]}.json"
        control_sha = MEASUREMENT_CONTROL_SHA256
        manifest_sha = MEASUREMENT_MANIFEST_SHA256
        expected_count = 1280
        expected_methods = METHODS
        raw_folder = "trajectories"
        control_kind = "diagnostic_measurement"
    input_path = CONTROL / input_name
    frozen_path = CONTROL / frozen_name
    _require(_sha256(input_path) == control_sha, f"{kind} input hash mismatch")
    _require(_sha256(frozen_path) == control_sha, f"{kind} frozen hash mismatch")
    _require(input_path.read_bytes() == frozen_path.read_bytes(), f"{kind} control copies differ")
    control = _load(input_path)
    _verify_axes(control, f"{kind} control", config_key="screen_config_sha256")
    _require(control["contract_version"] == CONTRACT_VERSION, f"{kind} contract mismatch")
    _require(control["stage"] == "screen" and control["kind"] == control_kind, f"{kind} kind mismatch")
    _require(control["manifest_version"] == "exp102.q0_global.screen_diagnostic.tasks.v1", f"{kind} manifest version")
    _require(control["method_tiers"] == [[method, "T3"] for method in expected_methods], f"{kind} methods changed")
    _require(len(control["tasks"]) == expected_count, f"{kind} task count mismatch")
    _require(_sha256_json(control) == manifest_sha, f"{kind} semantic manifest hash mismatch")

    entries = {}
    coordinate_counts = {}
    for entry in control["tasks"]:
        expected_entry_keys = {"task", "task_fingerprint", "output_relpath"}
        if kind == "measurement":
            expected_entry_keys.add("bias_relpath")
        _require(set(entry) == expected_entry_keys, f"{kind} control entry schema")
        task = entry["task"]
        fingerprint = entry["task_fingerprint"]
        _require(SHA256_RE.fullmatch(fingerprint) is not None, f"{kind} fingerprint malformed")
        _require(fingerprint == _sha256_json(task), f"{kind} task fingerprint mismatch")
        _require(fingerprint not in entries, f"{kind} duplicate task")
        _require(entry["output_relpath"] == f"{raw_folder}/{fingerprint}.npz", f"{kind} output path mismatch")
        _verify_axes(task, f"{kind} task", config_key="screen_config_sha256")
        _require(task["contract_version"] == CONTRACT_VERSION, f"{kind} task contract")
        _require(task["stage"] == "screen" and task["engine"] == "numba", f"{kind} task engine")
        _require(task["resource_tier"] == "T3", f"{kind} task tier")
        _require(task["method_id"] in expected_methods, f"{kind} task method")
        _require(task["cell"] in CELLS, f"{kind} task cell")
        _require(task["task_version"] == "exp102.q0_global.screen_diagnostic.tasks.v1", f"{kind} task version")
        _require(task["sampler_config"] == _expected_sampler(task), f"{kind} sampler config")
        if kind == "bias":
            _require(set(task) == {
                "cell", "contract_version", "engine", "method_id",
                "raw_version", "registry_sha256", "resource_tier",
                "sampler_config", "screen_config_sha256", "source_commit",
                "stage", "task_version", "tuning_seed_identities",
            }, "bias task schema mismatch")
            _require(task["raw_version"] == "exp102.q0_global.screen_diagnostic.defect_bias.raw.v1", "bias raw version")
            seeds = task["tuning_seed_identities"]
            _require(isinstance(seeds, list) and len(seeds) == 8, "bias seed count")
            for seed_index, seed in enumerate(seeds):
                _verify_seed(
                    seed, task, family="TUNE", index=seed_index,
                    namespace="q0_global_screen_bias_v1",
                )
            key = (_cell_key(task["cell"]), task["method_id"])
        else:
            _require(set(task) == {
                "bias_binding", "cell", "contract_version", "engine",
                "init_family", "method_id", "raw_version",
                "registry_sha256", "resource_tier", "sampler_config",
                "screen_config_sha256", "seed_identity", "source_commit",
                "stage", "task_version", "trajectory_index",
            }, "measurement task schema mismatch")
            expected_raw_version = (
                "exp102.q0_global.screen_diagnostic.hardcoset.raw.v1"
                if task["method_id"] in HARD_METHODS
                else "exp102.q0_global.screen_diagnostic.defect_trace.raw.v1"
            )
            _require(task["raw_version"] == expected_raw_version, "measurement raw version")
            _require(task["init_family"] in {"P", "U"}, "measurement init family")
            _require(task["trajectory_index"] in range(16), "measurement trajectory index")
            seed = task["seed_identity"]
            _verify_seed(
                seed, task, family=task["init_family"],
                index=task["trajectory_index"],
                namespace="q0_global_screen_measurement_v1",
            )
            key = (
                _cell_key(task["cell"]), task["method_id"],
                task["init_family"], task["trajectory_index"],
            )
        coordinate_counts[key] = coordinate_counts.get(key, 0) + 1
        entries[fingerprint] = entry
    _require(all(value == 1 for value in coordinate_counts.values()), f"{kind} duplicate coordinates")
    if kind == "bias":
        _require(len(coordinate_counts) == 5 * 3, "bias coordinate count")
    else:
        _require(len(coordinate_counts) == 5 * 8 * 2 * 16, "measurement coordinate count")
        _require(control["bias_manifest_sha256"] == BIAS_MANIFEST_SHA256, "measurement bias link")
    return control, entries


def _verify_stage(kind, schedule, package_evidence, control, entries,
                  bias_entries, bias_raw_hashes):
    if kind == "bias":
        control_sha = BIAS_CONTROL_SHA256
        ownership_sha = BIAS_OWNERSHIP_SHA256
        evidence_sha = BIAS_EVIDENCE_SHA256
        fingerprint = BIAS_FINGERPRINT
        expected_count = 15
        expected_node_counts = {"nd-1": 7, "nd-3": 8}
        control_kind = "diagnostic_defect_bias"
    else:
        control_sha = MEASUREMENT_CONTROL_SHA256
        ownership_sha = MEASUREMENT_OWNERSHIP_SHA256
        evidence_sha = MEASUREMENT_EVIDENCE_SHA256
        fingerprint = MEASUREMENT_FINGERPRINT
        expected_count = 1280
        expected_node_counts = {"nd-1": 576, "nd-3": 704}
        control_kind = "diagnostic_measurement"
    ownership_path = CONTROL / f"screen_ownership_{control_sha[:12]}.json"
    evidence_path = CONTROL / f"screen_{kind}_evidence_{control_sha[:12]}.json"
    _require(_sha256(ownership_path) == ownership_sha, f"{kind} ownership hash")
    _require(_sha256(evidence_path) == evidence_sha, f"{kind} evidence hash")
    ownership = _load(ownership_path)
    evidence = _load(evidence_path)
    _verify_axes(ownership, f"{kind} ownership")
    _verify_axes(evidence, f"{kind} evidence")
    _require(ownership["contract_version"] == CONTRACT_VERSION, f"{kind} ownership contract")
    _require(ownership["ownership_version"] == "exp102.q0_global.screen_diagnostic.ownership.v1", f"{kind} ownership version")
    _require(ownership["stage"] == "screen" and ownership["kind"] == control_kind, f"{kind} ownership kind")
    _require(ownership["nodes"] == list(EXECUTION_NODES), f"{kind} ownership nodes")
    _require(ownership["capacity"] == {"nd-1": 75, "nd-3": 91}, f"{kind} ownership capacity")
    _require(ownership["control_sha256"] == control_sha, f"{kind} ownership control")
    _require(ownership["schedule_file_sha256"] == SCHEDULE_FILE_SHA256, f"{kind} ownership schedule file")
    _require(ownership["schedule_sha256"] == SCHEDULE_SHA256, f"{kind} ownership schedule")
    _require(ownership["runtime_report_sha256"] == RUNTIME_SHA256, f"{kind} ownership runtime")
    ownership_identity = {
        key: ownership[key] for key in (
            "ownership_version", "contract_version", "source_commit",
            "registry_sha256", "diagnostic_config_sha256",
            "schedule_file_sha256", "schedule_sha256", "control_sha256",
            "runtime_report_sha256", "stage", "kind", "nodes", "task_owner",
        )
    }
    _require(_sha256_json(ownership_identity) == fingerprint, f"{kind} ownership fingerprint")
    _require(ownership["stage_fingerprint"] == fingerprint, f"{kind} stage fingerprint")
    _require(set(ownership["task_owner"]) == set(entries), f"{kind} ownership task set")
    _require(set(ownership["task_owner"].values()) == set(EXECUTION_NODES), f"{kind} ownership values")

    _require(evidence["contract_version"] == CONTRACT_VERSION, f"{kind} evidence contract")
    _require(evidence["stage"] == kind and evidence["raw_count"] == expected_count, f"{kind} evidence count")
    _require(evidence["archive_sha256"] == ARCHIVE_SHA256, f"{kind} evidence archive")
    _require(evidence["source_manifest_sha256"] == MANIFEST_SHA256, f"{kind} evidence source manifest")
    _require(evidence["control_sha256"] == control_sha, f"{kind} evidence control")
    _require(evidence["ownership_sha256"] == ownership_sha, f"{kind} evidence ownership")
    _require(evidence["schedule_file_sha256"] == SCHEDULE_FILE_SHA256, f"{kind} evidence schedule file")
    _require(evidence["schedule_sha256"] == SCHEDULE_SHA256, f"{kind} evidence schedule")
    _require(evidence["runtime_report_sha256"] == RUNTIME_SHA256, f"{kind} evidence runtime")
    _require(evidence["stage_fingerprint"] == fingerprint, f"{kind} evidence fingerprint")
    _require(evidence == package_evidence, f"{kind} package evidence differs")

    manifest_union = {}
    _require(
        isinstance(evidence["nodes"], list)
        and [row.get("node") for row in evidence["nodes"]]
        == list(EXECUTION_NODES),
        f"{kind} evidence nodes",
    )
    _require(all(
        set(row) == {
            "node", "completed_unix", "success_sha256", "status_sha256",
            "raw_manifest_sha256",
        }
        and all(
            SHA256_RE.fullmatch(row[key]) is not None
            for key in ("success_sha256", "status_sha256", "raw_manifest_sha256")
        )
        for row in evidence["nodes"]
    ), f"{kind} node evidence schema")
    node_evidence = {row["node"]: row for row in evidence["nodes"]}
    completed_values = []
    for node in EXECUTION_NODES:
        marker_dir = (
            REMOTE / f"screen_diagnostic/stages/{kind}/markers/{control_sha[:12]}/{node}"
        )
        manifest_dir = (
            REMOTE / f"screen_diagnostic/stages/{kind}/node_manifests/{control_sha[:12]}/{node}"
        )
        _verify_success_marker(marker_dir, fingerprint)
        manifest_path = manifest_dir / "raw_manifest.json"
        status_path = manifest_dir / "stage_status.json"
        manifest = _load(manifest_path)
        status = _load(status_path)
        record = node_evidence[node]
        _require(_sha256(manifest_path) == record["raw_manifest_sha256"], f"{kind} manifest link: {node}")
        _require(_sha256(status_path) == record["status_sha256"], f"{kind} status link: {node}")
        _require(_sha256(marker_dir / "SUCCESS") == record["success_sha256"], f"{kind} SUCCESS link: {node}")
        _verify_axes(manifest, f"{kind} manifest {node}")
        _verify_source_identity(manifest.get("source_identity"), f"{kind} manifest {node}")
        _require(manifest["raw_manifest_version"] == "exp102.q0_global.screen_diagnostic.remote_raw_manifest.v1", f"{kind} manifest version: {node}")
        _require(manifest["contract_version"] == CONTRACT_VERSION, f"{kind} manifest contract: {node}")
        _require(manifest["node"] == node and manifest["stage"] == "screen", f"{kind} manifest node: {node}")
        _require(manifest["kind"] == control_kind, f"{kind} manifest kind: {node}")
        _require(manifest["stage_fingerprint"] == fingerprint, f"{kind} manifest fingerprint: {node}")
        _require(manifest["control_sha256"] == control_sha, f"{kind} manifest control: {node}")
        _require(manifest["ownership_sha256"] == ownership_sha, f"{kind} manifest ownership: {node}")
        _require(manifest["schedule_file_sha256"] == SCHEDULE_FILE_SHA256, f"{kind} manifest schedule file: {node}")
        _require(manifest["schedule_sha256"] == SCHEDULE_SHA256, f"{kind} manifest schedule: {node}")
        _require(manifest["runtime_report_sha256"] == RUNTIME_SHA256, f"{kind} manifest runtime: {node}")
        expected_fingerprints = {
            task for task, owner in ownership["task_owner"].items() if owner == node
        }
        _require(len(manifest["files"]) == expected_node_counts[node], f"{kind} manifest count: {node}")
        _require({row["task_fingerprint"] for row in manifest["files"]} == expected_fingerprints, f"{kind} manifest task set: {node}")
        for row in manifest["files"]:
            _require(set(row) == {"task_fingerprint", "path", "sha256"}, f"{kind} manifest row schema")
            task = row["task_fingerprint"]
            _require(row["path"] == entries[task]["output_relpath"], f"{kind} control output mismatch")
            _require(SHA256_RE.fullmatch(row["sha256"]) is not None, f"{kind} raw SHA malformed")
            _require(row["path"] not in manifest_union, f"{kind} duplicate raw path")
            manifest_union[row["path"]] = row["sha256"]
        _require(set(status) == {
            "status", "node", "stage_fingerprint", "expected", "computed",
            "reused", "raw_manifest_sha256", "completed_unix",
        }, f"{kind} status schema: {node}")
        _require(status["status"] == "SUCCESS" and status["node"] == node, f"{kind} status failed: {node}")
        _require(status["stage_fingerprint"] == fingerprint, f"{kind} status fingerprint: {node}")
        _require(status["expected"] == status["computed"] == expected_node_counts[node], f"{kind} status count: {node}")
        _require(status["reused"] == 0, f"{kind} reused raw: {node}")
        _require(status["raw_manifest_sha256"] == _sha256(manifest_path), f"{kind} status manifest link: {node}")
        _require(status["completed_unix"] == record["completed_unix"], f"{kind} evidence timestamp: {node}")
        _require(schedule["started_unix"] <= status["completed_unix"] <= schedule["deadlines_unix"][kind], f"{kind} timing: {node}")
        completed_values.append(status["completed_unix"])
    _require(len(manifest_union) == expected_count, f"{kind} raw manifest union count")
    _require(evidence["completed_unix_max"] == max(completed_values), f"{kind} evidence max time")

    if kind == "measurement":
        bias_by_coordinate = {
            (_cell_key(entry["task"]["cell"]), entry["task"]["method_id"]): entry
            for entry in bias_entries.values()
        }
        binding_by_bias = {}
        for entry in entries.values():
            task = entry["task"]
            if task["method_id"] in DEFECT_METHODS:
                expected_bias = bias_by_coordinate[
                    (_cell_key(task["cell"]), task["method_id"])
                ]
                _require(
                    entry["bias_relpath"] == expected_bias["output_relpath"],
                    "measurement bias path mismatch",
                )
                _require(
                    task["bias_binding"]["bias_task_fingerprint"]
                    == expected_bias["task_fingerprint"],
                    "measurement bias binding task",
                )
                _require(
                    task["bias_binding"]["bias_raw_sha256"]
                    == bias_raw_hashes[expected_bias["output_relpath"]],
                    "measurement bias binding raw SHA",
                )
                _require(
                    set(task["bias_binding"]) == {
                        "bias_task_fingerprint", "bias_raw_sha256",
                        "bias_sha256",
                    }
                    and SHA256_RE.fullmatch(
                        task["bias_binding"]["bias_sha256"]
                    ) is not None,
                    "measurement bias semantic binding malformed",
                )
                bias_fingerprint = expected_bias["task_fingerprint"]
                previous = binding_by_bias.setdefault(
                    bias_fingerprint, task["bias_binding"],
                )
                _require(
                    previous == task["bias_binding"],
                    "measurement bias semantic binding is inconsistent",
                )
            else:
                _require(entry["bias_relpath"] is None and task["bias_binding"] is None, "hard task has bias")
        _require(len(binding_by_bias) == 15, "measurement bias binding count")
    return manifest_union


def _verify_report(path, *, file_sha256, identity_sha256):
    _require(_sha256(path) == file_sha256, f"report file hash mismatch: {path}")
    report = _load(path)
    identity = {key: value for key, value in report.items() if key != "report_sha256"}
    _require(_sha256_json(identity) == report["report_sha256"], "report self-hash invalid")
    _require(report["report_sha256"] == identity_sha256, "report identity changed")
    _verify_axes(report, "report", config_key="screen_config_sha256")
    _require(report["report_version"] == "exp102.q0_global.screen_diagnostic.report.v1", "report version mismatch")
    _require(report["contract_version"] == CONTRACT_VERSION and report["stage"] == "screen", "report contract mismatch")
    _require(report["status"] == "NO_HARD_COSET_PASS", "report status mismatch")
    _require(report["raw_count"] == 1280, "report raw count mismatch")
    _require(report["manifest_sha256"] == MEASUREMENT_MANIFEST_SHA256, "report measurement link")
    _require(report["bias_manifest_sha256"] == BIAS_MANIFEST_SHA256, "report bias link")
    _require(
        report["screen_panel_sha256"]
        == _sha256_json(list(CELLS))
        == "ad9219e3a1b55375adb3d096a047306771cd100bf98c2402a22a3dd5be0f4f85",
        "report panel changed",
    )
    _require(report["method_tiers"] == METHOD_TIERS, "report methods changed")
    _require(report["formal_authorization"] is False and report["production_authorization"] is False, "report authorized work")
    _require(report["primary_pair"] is None and report["selected_pair"] is None, "report selected a pair")
    _require(len(report["cell_summaries"]) == 40, "report cell count mismatch")
    _require(len(report["comparisons"]) == 75, "report comparison count mismatch")
    statuses = report["method_status"]
    _require([row["method_id"] for row in statuses] == list(METHODS), "method status order changed")
    _require(all(
        row["resource_tier"] == "T3" and row["cells_total"] == 5
        and row["cells_passed"] == 0 and row["valid"] is False
        for row in statuses
    ), "a method unexpectedly passed")
    pairs = report["pair_status"]
    _require(len(pairs) == 15, "pair status count mismatch")
    _require(all(
        row["cells_total"] == 5 and row["cells_passed"] == 0
        and row["valid"] is False
        for row in pairs
    ), "a pair unexpectedly passed")
    _verify_no_authorization(report, "report")
    return report


def _verify_decision(path, report_path, *, file_sha256, identity_sha256,
                     report_identity_sha256):
    _require(_sha256(path) == file_sha256, f"decision file hash mismatch: {path}")
    decision = _load(path)
    identity = {key: value for key, value in decision.items() if key != "decision_sha256"}
    _require(_sha256_json(identity) == decision["decision_sha256"], "decision self-hash invalid")
    _require(decision["decision_sha256"] == identity_sha256, "decision identity changed")
    _verify_axes(decision, "decision", config_key="screen_config_sha256")
    _require(decision["decision_version"] == "exp102.q0_global.screen_diagnostic.decision.v1", "decision version mismatch")
    _require(decision["contract_version"] == CONTRACT_VERSION, "decision contract mismatch")
    _require(decision["status"] == FINAL_STATUS, "decision status mismatch")
    _require(decision["maximum_possible_status"] == "DIAGNOSTIC_SCREEN_PAIR_FOUND", "decision authority changed")
    _require(decision["report_sha256"] == report_identity_sha256, "decision report identity link")
    _require(decision["report_file_sha256"] == _sha256(report_path), "decision report file link")
    _require(decision["selected_pair"] is None, "decision selected a pair")
    _require(decision["formal_authorization"] is False and decision["production_authorization"] is False, "decision authorized work")
    _require(decision["formal_blockers"] == FORMAL_BLOCKERS, "decision blockers changed")
    _verify_no_authorization(decision, "decision")
    return decision


def _verify_package(path, report_path, decision_path, schedule, bias_evidence,
                    measurement_evidence, *, file_sha256, identity_sha256,
                    report_identity_sha256, decision_identity_sha256):
    _require(_sha256(path) == file_sha256, "package file hash mismatch")
    package = _load(path)
    identity = {key: value for key, value in package.items() if key != "package_sha256"}
    _require(package["package_sha256"] == _sha256_json(identity), "package self-hash invalid")
    _require(package["package_sha256"] == identity_sha256, "package identity changed")
    _verify_axes(package, "package")
    _verify_source_identity(package.get("analysis_source_identity"), "package analysis")
    _require(package["package_version"] == "exp102.q0_global.screen_diagnostic.verified_terminal_package.v1", "package version mismatch")
    _require(package["contract_version"] == CONTRACT_VERSION, "package contract mismatch")
    _require(package["status"] == FINAL_STATUS, "package status mismatch")
    _require(package["maximum_possible_status"] == "DIAGNOSTIC_SCREEN_PAIR_FOUND", "package authority changed")
    _require(package["archive_sha256"] == ARCHIVE_SHA256, "package archive mismatch")
    _require(package["source_manifest_sha256"] == MANIFEST_SHA256, "package manifest mismatch")
    _require(package["schedule_file_sha256"] == SCHEDULE_FILE_SHA256, "package schedule file link")
    _require(package["schedule_sha256"] == SCHEDULE_SHA256, "package schedule link")
    _require(package["runtime_report_sha256"] == RUNTIME_SHA256, "package runtime link")
    _require(package["digest_report_sha256"] == DIGEST_SHA256, "package digest link")
    _require(package["preflight_report_sha256"] == PREFLIGHT_SHA256, "package preflight link")
    _require(package["canonical_digest"] == CANONICAL_DIGEST, "package digest changed")
    _require(package["bias_control_sha256"] == BIAS_CONTROL_SHA256, "package bias control link")
    _require(package["bias_ownership_sha256"] == BIAS_OWNERSHIP_SHA256, "package bias ownership link")
    _require(package["measurement_control_sha256"] == MEASUREMENT_CONTROL_SHA256, "package measurement control link")
    _require(package["measurement_ownership_sha256"] == MEASUREMENT_OWNERSHIP_SHA256, "package measurement ownership link")
    _require(package["bias_evidence"] == bias_evidence, "package bias evidence mismatch")
    _require(package["measurement_evidence"] == measurement_evidence, "package measurement evidence mismatch")
    _require(package["analysis_report_sha256"] == _sha256(report_path), "package report file link")
    _require(package["analysis_report_identity_sha256"] == report_identity_sha256, "package report identity link")
    _require(package["terminal_decision_sha256"] == _sha256(decision_path), "package decision file link")
    _require(package["terminal_decision_identity_sha256"] == decision_identity_sha256, "package decision identity link")
    _require(package["selected_pair"] is None, "package selected a pair")
    _require(package["formal_authorization"] is False and package["production_authorization"] is False, "package authorized work")
    _require(math.isfinite(float(package["completed_unix"])), "package timestamp invalid")
    _require(
        float(package["completed_unix"])
        >= float(measurement_evidence["completed_unix_max"]),
        "package predates measurement evidence",
    )
    _require(float(package["completed_unix"]) <= schedule["deadlines_unix"]["analysis"], "package missed analysis deadline")
    _verify_no_authorization(package, "package")
    return package


def _verify_marker_closure():
    expected = {
        f"screen_diagnostic/preflight/markers/{PREFLIGHT_FINGERPRINT[:12]}/{node}/SUCCESS"
        for node in NODES
    }
    expected.update(
        f"screen_diagnostic/stages/{stage}/markers/{control[:12]}/{node}/SUCCESS"
        for stage, control in (
            ("bias", BIAS_CONTROL_SHA256),
            ("measurement", MEASUREMENT_CONTROL_SHA256),
        )
        for node in EXECUTION_NODES
    )
    actual = {
        path.relative_to(REMOTE).as_posix()
        for path in REMOTE.rglob("*")
        if path.is_file() and path.name in {"SUCCESS", "FAILED", "RUNNING"}
    }
    _require(actual == expected and len(actual) == 7, "remote markers are not seven exclusive SUCCESS states")

    expected_manifests = {
        f"screen_diagnostic/stages/{stage}/node_manifests/{control[:12]}/{node}/raw_manifest.json"
        for stage, control in (
            ("bias", BIAS_CONTROL_SHA256),
            ("measurement", MEASUREMENT_CONTROL_SHA256),
        )
        for node in EXECUTION_NODES
    }
    expected_statuses = {
        relative.replace("raw_manifest.json", "stage_status.json")
        for relative in expected_manifests
    }
    actual_manifests = {
        path.relative_to(REMOTE).as_posix()
        for path in REMOTE.rglob("raw_manifest.json")
    }
    actual_statuses = {
        path.relative_to(REMOTE).as_posix()
        for path in REMOTE.rglob("stage_status.json")
    }
    _require(actual_manifests == expected_manifests, "raw manifest path closure mismatch")
    _require(actual_statuses == expected_statuses, "stage status path closure mismatch")


def _verify_driver_log():
    path = ROOT / "remote_screen_driver.log"
    _require(_sha256(path) == DRIVER_LOG_SHA256, "remote driver log hash mismatch")
    text = path.read_text(encoding="utf-8")
    _require("SCREEN_DRIVER_COMPLETE" in text, "driver did not complete")
    _require("SCREEN_DRIVER_EXIT status=0" in text, "driver exit was not successful")
    _require(f'"status": "{FINAL_STATUS}"' in text, "driver terminal status mismatch")
    _require(f'"package_sha256": "{ORIGINAL_PACKAGE_IDENTITY_SHA256}"' in text, "driver package identity mismatch")


def main():
    if not __debug__:
        raise RuntimeError("evidence verification forbids optimized Python")
    _verify_evidence_closure()
    _verify_deployment_identity()
    schedule = _verify_schedule()
    _verify_preflight(schedule)

    original_report_path = CONTROL / "screen_report.json"
    original_decision_path = CONTROL / "screen_decision.json"
    original_package_path = CONTROL / "screen_terminal_package.json"
    independent_report_path = INDEPENDENT / "screen_report.json"
    independent_decision_path = INDEPENDENT / "screen_decision.json"
    independent_package_path = INDEPENDENT / "screen_terminal_package.json"

    original_report = _verify_report(
        original_report_path,
        file_sha256=REPORT_FILE_SHA256,
        identity_sha256=REPORT_IDENTITY_SHA256,
    )
    independent_report = _verify_report(
        independent_report_path,
        file_sha256=INDEPENDENT_REPORT_FILE_SHA256,
        identity_sha256=INDEPENDENT_REPORT_IDENTITY_SHA256,
    )
    replay_stats = _verify_replay_report_equivalence(
        original_report, independent_report,
    )
    _require(
        replay_stats == {"float_differences": 80, "max_ulp": 4},
        "independent replay roundoff audit changed",
    )
    original_decision = _verify_decision(
        original_decision_path, original_report_path,
        file_sha256=DECISION_FILE_SHA256,
        identity_sha256=DECISION_IDENTITY_SHA256,
        report_identity_sha256=REPORT_IDENTITY_SHA256,
    )
    independent_decision = _verify_decision(
        independent_decision_path, independent_report_path,
        file_sha256=INDEPENDENT_DECISION_FILE_SHA256,
        identity_sha256=INDEPENDENT_DECISION_IDENTITY_SHA256,
        report_identity_sha256=INDEPENDENT_REPORT_IDENTITY_SHA256,
    )
    decision_link_fields = {
        "decision_sha256", "report_file_sha256", "report_sha256",
    }
    _require(
        {
            key for key in set(original_decision) | set(independent_decision)
            if original_decision.get(key) != independent_decision.get(key)
        } == decision_link_fields,
        "independent decision differs outside replay hash links",
    )
    _require(
        {key: value for key, value in original_decision.items()
         if key not in decision_link_fields}
        == {key: value for key, value in independent_decision.items()
            if key not in decision_link_fields},
        "independent decision semantics differ",
    )

    bias_control, bias_entries = _verify_control("bias")
    measurement_control, measurement_entries = _verify_control("measurement")
    bias_evidence = _load(CONTROL / f"screen_bias_evidence_{BIAS_CONTROL_SHA256[:12]}.json")
    measurement_evidence = _load(
        CONTROL / f"screen_measurement_evidence_{MEASUREMENT_CONTROL_SHA256[:12]}.json"
    )
    original_package = _verify_package(
        original_package_path, original_report_path, original_decision_path,
        schedule, bias_evidence, measurement_evidence,
        file_sha256=ORIGINAL_PACKAGE_FILE_SHA256,
        identity_sha256=ORIGINAL_PACKAGE_IDENTITY_SHA256,
        report_identity_sha256=REPORT_IDENTITY_SHA256,
        decision_identity_sha256=DECISION_IDENTITY_SHA256,
    )
    independent_package = _verify_package(
        independent_package_path, independent_report_path,
        independent_decision_path, schedule, bias_evidence,
        measurement_evidence,
        file_sha256=INDEPENDENT_PACKAGE_FILE_SHA256,
        identity_sha256=INDEPENDENT_PACKAGE_IDENTITY_SHA256,
        report_identity_sha256=INDEPENDENT_REPORT_IDENTITY_SHA256,
        decision_identity_sha256=INDEPENDENT_DECISION_IDENTITY_SHA256,
    )
    package_replay_fields = {
        "analysis_report_identity_sha256", "analysis_report_sha256",
        "completed_unix", "package_sha256",
        "terminal_decision_identity_sha256", "terminal_decision_sha256",
    }
    differing_package_fields = {
        key for key in set(original_package) | set(independent_package)
        if original_package.get(key) != independent_package.get(key)
    }
    _require(
        differing_package_fields == package_replay_fields,
        "independent package differs outside replay links/timestamp/self-hash",
    )
    _require(
        independent_package["completed_unix"] > original_package["completed_unix"],
        "independent replay does not postdate original analysis",
    )
    original_stable = {
        key: value for key, value in original_package.items()
        if key not in package_replay_fields
    }
    independent_stable = {
        key: value for key, value in independent_package.items()
        if key not in package_replay_fields
    }
    _require(original_stable == independent_stable, "package stable identity differs")

    bias_union = _verify_stage(
        "bias", schedule, bias_evidence, bias_control, bias_entries,
        bias_entries, {},
    )
    measurement_union = _verify_stage(
        "measurement", schedule, measurement_evidence, measurement_control,
        measurement_entries, bias_entries, bias_union,
    )
    _require(
        bias_evidence["completed_unix_max"]
        <= measurement_evidence["completed_unix_max"]
        <= original_package["completed_unix"],
        "bias/measurement/analysis chronology mismatch",
    )
    manifest_union = {**bias_union, **measurement_union}
    _require(len(manifest_union) == 1295, "combined raw manifest count mismatch")
    raw_sums_path = ROOT / "RAW_SHA256SUMS"
    _require(_sha256(raw_sums_path) == RAW_SHA256SUMS_SHA256, "RAW_SHA256SUMS file hash changed")
    raw_sums = _parse_sums(raw_sums_path)
    _require(len(raw_sums) == 1295, "RAW_SHA256SUMS count mismatch")
    _require(raw_sums == manifest_union, "RAW_SHA256SUMS differs from control/manifest union")

    _verify_marker_closure()
    _verify_driver_log()
    for json_path in ROOT.rglob("*.json"):
        _verify_no_authorization(
            _load(json_path), json_path.relative_to(ROOT).as_posix(),
        )
    print(json.dumps({
        "status": "VERIFIED_UNRESOLVED_NO_HARD_COSET_PASS",
        "run_id": RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "canonical_digest": CANONICAL_DIGEST,
        "gamma_sha256": GAMMA_SHA256,
        "bias_raw_count": len(bias_union),
        "measurement_raw_count": len(measurement_union),
        "replay_float_differences": replay_stats["float_differences"],
        "replay_max_ulp": replay_stats["max_ulp"],
        "selected_pair": None,
        "formal_blockers": FORMAL_BLOCKERS,
        "formal_authorization": False,
        "production_authorization": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
