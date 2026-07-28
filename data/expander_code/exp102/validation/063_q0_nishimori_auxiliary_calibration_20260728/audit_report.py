"""Independent scientific audit of the exact Nishimori calibration report."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from independent_oracle import (
    build_oracle_calibration,
    canonical,
)


CONFIG_PATH = ROOT / "nishimori_config.json"
SCHEMA_PATH = ROOT / "nishimori_raw_schema.v1.json"
REPORT_PATH = ROOT / "exact_calibration_report.json"
OUTPUT_PATH = ROOT / "independent_audit.json"
CALIBRATED_STATUS = "NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS"
INSUFFICIENT_STATUS = "NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT"
EXPECTED_AUTHORITY = {
    "formal_authorization": False,
    "maximum_status": CALIBRATED_STATUS,
    "posterior_estimation": False,
    "production_authorization": False,
    "remote_authorization": False,
    "sole_confirmer_authorization": False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    actual = hashlib.sha256(canonical(unsigned).encode("ascii")).hexdigest()
    if expected != actual:
        raise RuntimeError(f"self hash changed: {field}")


def _git(*args):
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, env=env,
    ).stdout


def verify_audit_source_identity(config, report):
    bound = config["implementation"]["bound_files"]
    for role, descriptor in sorted(bound.items()):
        path = PROJECT_ROOT / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"bound source changed during audit: {role}")
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", descriptor["path"]],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).returncode != 0:
            raise RuntimeError(f"untracked bound audit source: {role}")
    config_relative = CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    if subprocess.run(
        ["git", "ls-files", "--error-unmatch", config_relative],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).returncode != 0:
        raise RuntimeError("untracked Nishimori config during audit")
    allowed = {f"?? {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}"}
    status_lines = set(filter(None, _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()))
    if status_lines - allowed:
        raise RuntimeError("audit source worktree has changes beyond the immutable report")
    bytecode = [
        path for path in PROJECT_ROOT.rglob("*")
        if path.name == "__pycache__" or (path.is_file() and path.suffix == ".pyc")
    ]
    if bytecode:
        raise RuntimeError("audit source worktree contains Python bytecode")
    source_commit = _git("rev-parse", "HEAD").strip()
    tree_core = {
        "bound_files": bound,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_commit": source_commit,
    }
    expected_tree = hashlib.sha256(canonical(tree_core).encode("ascii")).hexdigest()
    if report["source_commit"] != source_commit or report["source_tree_sha256"] != expected_tree:
        raise RuntimeError("report source-tree binding changed")
    if report["bound_files"] != bound or report["config_sha256"] != tree_core["config_sha256"]:
        raise RuntimeError("report bound-file identity changed")


def assert_nested_close(actual, expected, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"mapping keys changed at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"list shape changed at {path}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            raise RuntimeError(f"nonfinite/non-numeric value at {path}")
        if not math.isclose(float(actual), float(expected), rel_tol=2e-13, abs_tol=2e-13):
            raise RuntimeError(f"numeric value changed at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise RuntimeError(f"value changed at {path}: {actual!r} != {expected!r}")


def terminal_status(calibration_gate):
    if not isinstance(calibration_gate, dict):
        raise RuntimeError("calibration gate is missing")
    passed = calibration_gate.get("passed")
    failures = calibration_gate.get("failures")
    if not isinstance(passed, bool) or not isinstance(failures, list):
        raise RuntimeError("calibration gate fields changed")
    if any(not isinstance(failure, str) or not failure for failure in failures):
        raise RuntimeError("calibration gate failure is invalid")
    if passed != (len(failures) == 0):
        raise RuntimeError("calibration gate result contradicts its failures")
    return CALIBRATED_STATUS if passed else INSUFFICIENT_STATUS


def validate_report_envelope(report, config, schema):
    verify_self_hash(report, "report_sha256")
    expected_status = terminal_status(report.get("calibration_gate"))
    if report.get("status") != expected_status:
        raise RuntimeError("report status contradicts the calibration gate")
    if report.get("authority") != EXPECTED_AUTHORITY or config.get("authority") != EXPECTED_AUTHORITY:
        raise RuntimeError("report/config authority changed")
    if report.get("version") != "exp102.q0_nishimori_auxiliary_calibration.v2":
        raise RuntimeError("report version changed")
    if report.get("universal_q_top_bias_bound_from_identity") is not None:
        raise RuntimeError("Nishimori identity was upgraded to a q_top bound")
    if report.get("schema_sha256") != sha256_file(SCHEMA_PATH):
        raise RuntimeError("schema hash changed")
    if schema.get("raw_version") != "exp102.q0_nishimori_auxiliary.raw.v1":
        raise RuntimeError("raw schema version changed")
    if schema.get("aggregation_rule") != "ALL_PLANNED_FRESH_IID_DISORDERS_MUST_PASS_OR_AUDIT_NOT_COMPUTED":
        raise RuntimeError("fail-closed ensemble aggregation changed")
    runner_sha = config["implementation"]["bound_files"]["runner"]["sha256"]
    if report.get("runner_sha256") != runner_sha:
        raise RuntimeError("runner hash changed")
    if report["calibration_gate"].get("universal_q_top_bias_bound") is not None:
        raise RuntimeError("calibration gate created a q_top bound")


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("independent audit already exists")
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="ascii"))
    report = json.loads(REPORT_PATH.read_text(encoding="ascii"))
    validate_report_envelope(report, config, schema)
    verify_audit_source_identity(config, report)
    expected = build_oracle_calibration(config, include_power=True)
    for field in (
        "calibration_gate", "chain_level_control_metrics", "exact_control_rows",
        "golden_rows", "power_rows",
    ):
        assert_nested_close(report[field], expected[field], field)
    core = {
        "audit_runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": report["config_sha256"],
        "independent_oracle_sha256": config["implementation"]["bound_files"]["independent_oracle"]["sha256"],
        "report_file_sha256": sha256_file(REPORT_PATH),
        "report_sha256": report["report_sha256"],
        "schema_sha256": report["schema_sha256"],
        "source_commit": report["source_commit"],
        "source_tree_sha256": report["source_tree_sha256"],
        "status": "INDEPENDENT_SCIENTIFIC_AUDIT_PASS_" + report["status"],
        "version": "exp102.q0_nishimori_auxiliary_calibration.audit.v2",
    }
    core["audit_sha256"] = hashlib.sha256(canonical(core).encode("ascii")).hexdigest()
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
