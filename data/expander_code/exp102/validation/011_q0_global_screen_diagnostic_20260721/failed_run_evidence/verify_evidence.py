"""Verify the metadata-only evidence for the closed 5e1f5aa screen run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_evidence"
SOURCE_COMMIT = "5e1f5aa0d1046a3555126196533d2988d4bf14b6"
CONFIG_SHA256 = "e5fa2ebdc2f22f25342d3d8d5c5ab05027685a4def6aecf8e48e666fa72f468b"
REGISTRY_SHA256 = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
SCHEDULE_SHA256 = "5d2d1476112e83de45b64ec216b23ddedda1063dca67f73884fcf87a7eb03ecb"
BIAS_CONTROL_SHA256 = "471d9185eb06eadb2850e4915882683c144369c492d3c4fb0842a4bab92a7456"
PREFLIGHT_FINGERPRINT = "bea2ea18422edbffe39bba162fedf6a31b944116823de8ef00488793e1b3c5f0"
BIAS_FINGERPRINT = "aebfbe5f4bedf1cf68f5c26f85d5d3214c1348f8642d8fdc3106608dc411d507"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _verify_checksums():
    lines = (ROOT / "EVIDENCE_SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected = {}
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert relative not in expected
        expected[relative] = digest
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "EVIDENCE_SHA256SUMS"
    }
    assert set(expected) == actual
    for relative, digest in expected.items():
        assert _sha256(ROOT / relative) == digest


def _verify_success_marker(path, fingerprint):
    marker = _load(path)
    assert marker["stage_fingerprint"] == fingerprint
    assert set(marker) == {"stage_fingerprint", "completed_utc"}


def main():
    if not __debug__:
        raise RuntimeError("evidence verification forbids optimized Python")
    _verify_checksums()
    conflict = _load(ROOT / "CONFLICT_CROSS_NODE_GAMMA_LIBM.json")
    assert conflict["status"] == "CONFLICT_CROSS_NODE_GAMMA_LIBM"
    assert conflict["source_commit"] == SOURCE_COMMIT
    assert conflict["diagnostic_config_sha256"] == CONFIG_SHA256
    assert conflict["registry_sha256"] == REGISTRY_SHA256
    assert conflict["schedule_sha256"] == SCHEDULE_SHA256
    assert conflict["bias_control_sha256"] == BIAS_CONTROL_SHA256
    assert conflict["bias_raw_count"] == 15
    assert sum(row["raw_count"] for row in conflict["gamma_patterns"]) == 15
    assert conflict["cross_node_difference"]["differing_indices"] == [3171, 4082]
    assert conflict["measurement_artifacts"] == {
        "decision_exists": False,
        "measurement_control_exists": False,
        "measurement_raw_count": 0,
        "report_exists": False,
        "terminal_package_exists": False,
    }

    control = REMOTE / "control"
    schedule = _load(control / "SCREEN_DIAGNOSTIC_24H_SCHEDULE.json")
    preflight = _load(control / "screen_preflight_report.json")
    bias_control = _load(control / "screen_bias_input.json")
    ownership = _load(control / "screen_ownership_471d9185eb06.json")
    evidence = _load(control / "screen_bias_evidence_471d9185eb06.json")
    for record in (schedule, preflight, bias_control, ownership, evidence):
        assert record["source_commit"] == SOURCE_COMMIT
        assert record["registry_sha256"] == REGISTRY_SHA256
        assert record.get("screen_config_sha256", record.get(
            "diagnostic_config_sha256"
        )) == CONFIG_SHA256
    for record in (schedule, preflight, ownership, evidence):
        assert record["schedule_sha256"] == SCHEDULE_SHA256
    assert preflight["status"] == "PASS"
    assert preflight["stage_fingerprint"] == PREFLIGHT_FINGERPRINT
    assert _sha256(control / "screen_bias_input.json") == BIAS_CONTROL_SHA256
    assert len(bias_control["tasks"]) == 15
    assert ownership["control_sha256"] == BIAS_CONTROL_SHA256
    assert evidence["raw_count"] == 15

    preflight_marker_root = (
        REMOTE / "screen_diagnostic/preflight/markers/bea2ea18422e"
    )
    for node in ("nd-1", "nd-2", "nd-3"):
        _verify_success_marker(
            preflight_marker_root / node / "SUCCESS", PREFLIGHT_FINGERPRINT,
        )

    bias_root = REMOTE / "screen_diagnostic/stages/bias"
    task_fingerprints = set()
    counts = {"nd-1": 7, "nd-3": 8}
    for node, expected_count in counts.items():
        _verify_success_marker(
            bias_root / "markers/471d9185eb06" / node / "SUCCESS",
            BIAS_FINGERPRINT,
        )
        manifest = _load(
            bias_root / "node_manifests/471d9185eb06" / node
            / "raw_manifest.json"
        )
        assert manifest["node"] == node
        assert manifest["source_commit"] == SOURCE_COMMIT
        assert manifest["control_sha256"] == BIAS_CONTROL_SHA256
        assert manifest["stage_fingerprint"] == BIAS_FINGERPRINT
        assert len(manifest["files"]) == expected_count
        for row in manifest["files"]:
            assert row["path"] == f"bias/{row['task_fingerprint']}.npz"
            task_fingerprints.add(row["task_fingerprint"])
    assert len(task_fingerprints) == 15
    assert task_fingerprints == {
        row["task_fingerprint"] for row in bias_control["tasks"]
    }

    forbidden = (
        "screen_measurement_input.json",
        "screen_report.json",
        "screen_decision.json",
        "screen_terminal_package.json",
    )
    assert not any((control / name).exists() for name in forbidden)
    assert not (REMOTE / "screen_diagnostic/raw/measurement").exists()
    print("VERIFIED_CONFLICT_CROSS_NODE_GAMMA_LIBM")


if __name__ == "__main__":
    main()
