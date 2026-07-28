#!/usr/bin/env python3
"""Independent, source-bound verifier for generated validation 064 artifacts.

This module intentionally does not import ``run_resource_calibration``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_VERSION = "exp102.q0_hp64.resource_calibration.config.v1"
CONTRACT_VERSION = "exp102.q0_hp64.resource_calibration.local.v1"
RESOURCE_REPORT_VERSION = "exp102.q0_hp64.resource_calibration.report.v1"
SCIENCE_AUDIT_VERSION = "exp102.q0_hp64.discrepancy_audit.v1"
RECEIPT_VERSION = "exp102.q0_hp64.resource_calibration.receipt.v1"
INDEPENDENT_AUDIT_VERSION = "exp102.q0_hp64.independent_package_audit.v1"
VALIDATION_RELATIVE_DIR = (
    "data/expander_code/exp102/validation/"
    "064_q0_hp64_resource_calibration_20260728"
)
BOUND_IMPLEMENTATION_NAMES = (
    "run_resource_calibration.py",
    "audit_resource_calibration.py",
    "test_resource_calibration.py",
    "README.md",
    "PRE_RUN_RED_TEAM.md",
)
RUNNER_OUTPUT_NAMES = (
    "resource_calibration_report.json",
    "discrepancy_audit.json",
    "timing_coverage.csv",
    "resource_scenarios.csv",
    "RESOURCE_CALIBRATION_REPORT.md",
    "RUN_RECEIPT.json",
)
INDEPENDENT_OUTPUT_NAME = "independent_package_audit.json"


class AuditFailure(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def assert_finite_json(value: Any, context: str = "value") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite float in {context}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{context}[{index}]")


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise AuditFailure(f"non-finite JSON constant in {path}: {token}")

    value = json.loads(
        path.read_text(encoding="ascii"), parse_constant=reject_constant,
    )
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    assert_finite_json(value, path.name)
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    require(field in value, f"missing {field}")
    claimed = str(value[field])
    identity = {key: item for key, item in value.items() if key != field}
    require(claimed == sha256_json(identity), f"{field} mismatch")
    return claimed


def _git(worktree: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(worktree), *arguments), check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        raise AuditFailure(f"git source verification failed: {' '.join(arguments)}") from error
    return result.stdout.strip()


def validate_authority(config: Mapping[str, Any]) -> None:
    require(config.get("version") == CONFIG_VERSION, "config version changed")
    authority = config.get("authority")
    require(isinstance(authority, dict), "config authority missing")
    exact = {
        "contract_version": CONTRACT_VERSION,
        "maximum_resource_status": "RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE",
        "science_audit_status": "PASS",
        "science_audit_authority": "HISTORICAL_DIAGNOSTIC_RECOMPUTATION_ONLY",
        "receipt_status": "LOCAL_AUDIT_COMPLETE_NO_REMOTE_AUTHORITY",
        "independent_audit_status": "INDEPENDENT_PACKAGE_AUDIT_PASS",
        "formal_authorization": False,
        "production_authorization": False,
        "remote_launch_authorization": False,
    }
    for field, expected in exact.items():
        require(authority.get(field) == expected, f"authority field changed: {field}")
    expected_paths = {
        f"{VALIDATION_RELATIVE_DIR}/{name}" for name in BOUND_IMPLEMENTATION_NAMES
    }
    files = authority.get("implementation_files")
    require(isinstance(files, dict) and set(files) == expected_paths, "implementation path set changed")
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in files.values()),
        "implementation SHA malformed",
    )
    evidence = config.get("validation013_evidence")
    require(isinstance(evidence, dict), "validation 013 evidence binding missing")
    require(evidence.get("validation013_source_commit") == "4d134ee7ca25125d341eb11cbfa34d6856514101", "validation 013 source changed")
    require(evidence.get("raw_count") == 384 and evidence.get("resource_tier") == "T3", "validation 013 scope changed")
    require(float(config.get("report_tolerance", 0.0)) == 5.0e-14, "report tolerance changed")


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    validate_authority(config)
    return config


def verify_auditor_source(
    config_path: Path, config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Bind the auditor while allowing only the six immutable runner outputs."""
    config_path = config_path.resolve()
    worktree = Path(_git(config_path.parent, "rev-parse", "--show-toplevel")).resolve()
    config_relative = config_path.relative_to(worktree).as_posix()
    require(
        config_relative == f"{VALIDATION_RELATIVE_DIR}/resource_model_config.json",
        "config path changed",
    )
    status_lines = _git(worktree, "status", "--porcelain", "--untracked-files=all").splitlines()
    allowed_untracked = set()
    try:
        output_relative = output_dir.resolve().relative_to(worktree)
    except ValueError:
        output_relative = None
    if output_relative is not None:
        allowed_untracked = {
            f"?? {(output_relative / name).as_posix()}" for name in RUNNER_OUTPUT_NAMES
        }
    require(set(status_lines) <= allowed_untracked, "auditor source worktree has unrelated changes")
    require(
        all((output_dir / name).is_file() for name in RUNNER_OUTPUT_NAMES),
        "runner output package is incomplete",
    )
    bytecode = []
    for directory, names, filenames in os.walk(worktree):
        names[:] = [name for name in names if name != ".git"]
        directory_path = Path(directory)
        if directory_path.name == "__pycache__":
            bytecode.append(str(directory_path.relative_to(worktree)))
            names[:] = []
            continue
        bytecode.extend(
            str((directory_path / name).relative_to(worktree))
            for name in filenames if name.endswith((".pyc", ".pyo"))
        )
    require(not bytecode, f"bytecode artifacts are forbidden: {bytecode[:8]}")
    bound = config["authority"]["implementation_files"]
    for relative_path, expected_sha in sorted(bound.items()):
        _git(worktree, "ls-files", "--error-unmatch", "--", relative_path)
        require(sha256_file(worktree / relative_path) == expected_sha, f"implementation SHA mismatch: {relative_path}")
    _git(worktree, "ls-files", "--error-unmatch", "--", config_relative)
    return {
        "calibration_source_commit": _git(worktree, "rev-parse", "HEAD"),
        "calibration_source_tree_sha": _git(worktree, "rev-parse", "HEAD^{tree}"),
        "git_object_format": _git(worktree, "rev-parse", "--show-object-format"),
        "worktree_clean": True,
        "bytecode_absent": True,
        "bound_implementation_sha256": dict(bound),
    }


def expected_provenance(
    config: Mapping[str, Any], config_path: Path, source_identity: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "calibration_source": dict(source_identity),
        "implementation_authority_sha256": sha256_json(config["authority"]),
        "resource_config": {
            "path": f"{VALIDATION_RELATIVE_DIR}/resource_model_config.json",
            "file_sha256": sha256_file(config_path),
        },
        "validation013_evidence": dict(config["validation013_evidence"]),
    }


def _float_equal(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    require(set(value) == expected, f"{context} field set changed")


def audit(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    resource_path = output_dir / "resource_calibration_report.json"
    discrepancy_path = output_dir / "discrepancy_audit.json"
    receipt_path = output_dir / "RUN_RECEIPT.json"
    coverage_path = output_dir / "timing_coverage.csv"
    scenarios_path = output_dir / "resource_scenarios.csv"
    markdown_path = output_dir / "RESOURCE_CALIBRATION_REPORT.md"
    required_paths = (
        resource_path, discrepancy_path, receipt_path, coverage_path,
        scenarios_path, markdown_path,
    )
    require(all(path.is_file() for path in required_paths), "calibration package is incomplete")
    resource = load_json(resource_path)
    discrepancy = load_json(discrepancy_path)
    receipt = load_json(receipt_path)
    resource_hash = verify_self_hash(resource, "report_sha256")
    discrepancy_hash = verify_self_hash(discrepancy, "audit_sha256")
    receipt_hash = verify_self_hash(receipt, "receipt_sha256")
    authority = config["authority"]
    provenance = expected_provenance(config, config_path, source_identity)

    _require_exact_keys(resource, {
        "report_version", "contract_version", "authority", "status",
        "formal_authorization", "production_authorization",
        "remote_launch_authorization", "selection", "provenance",
        "grid_evaluation_counts", "strict_empirical_estimates",
        "timing_observations", "timing_coverage", "analysis_timing_evidence",
        "scenario_definitions", "resource_scenarios", "limitations",
        "report_sha256",
    }, "resource report")
    require(resource["report_version"] == RESOURCE_REPORT_VERSION, "resource version changed")
    require(resource["contract_version"] == CONTRACT_VERSION, "resource contract changed")
    require(resource["authority"] == "RESOURCE_SCENARIOS_ONLY", "resource authority changed")
    require(resource["status"] == authority["maximum_resource_status"], "resource status changed")
    require(resource["provenance"] == provenance, "resource provenance changed")
    require(resource["selection"] is None, "resource option was selected")
    for field in ("formal_authorization", "production_authorization", "remote_launch_authorization"):
        require(resource[field] is False, f"resource {field} leaked")
    serialized_resource = canonical_json(resource).lower()
    require("q_top" not in serialized_resource and "logical_label" not in serialized_resource, "resource payload contains science")
    require(len(resource["resource_scenarios"]) == 72, "resource row count changed")
    require(len(resource["timing_coverage"]) == 18, "coverage row count changed")
    require(all(row["selected"] is False for row in resource["resource_scenarios"]), "selected proxy row found")
    require(all(
        value["safety_adjusted_total_core_seconds"] is None
        for value in resource["strict_empirical_estimates"].values()
    ), "strict estimate is non-null")
    require(resource["grid_evaluation_counts"] == {
        "m3_easy_block_128": 128,
        "calibration_grid_3p": 18432,
        "formal_grid_7p": 43008,
    }, "grid evaluation counts changed")
    require(resource["analysis_timing_evidence"]["coverage_limitation"] == "VALIDATION_013_B_FAMILY_AND_B_COMPARISON_PROXY_ONLY", "analysis proxy authority changed")
    require(_float_equal(resource["analysis_timing_evidence"]["t3_proxy_seconds_per_evaluation"], 118.95883645396679), "analysis proxy changed")

    _require_exact_keys(discrepancy, {
        "audit_version", "contract_version", "status", "authority",
        "formal_authorization", "production_authorization",
        "remote_launch_authorization", "provenance", "allow_pickle",
        "report_tolerance", "selected_raw_count",
        "selected_raw_catalog_sha256", "selected_raw_catalog", "cells",
        "headline_checks", "limitations", "audit_sha256",
    }, "discrepancy audit")
    require(discrepancy["audit_version"] == SCIENCE_AUDIT_VERSION, "discrepancy version changed")
    require(discrepancy["contract_version"] == CONTRACT_VERSION, "discrepancy contract changed")
    require(discrepancy["status"] == authority["science_audit_status"], "discrepancy status changed")
    require(discrepancy["authority"] == authority["science_audit_authority"], "discrepancy authority changed")
    require(discrepancy["provenance"] == provenance, "discrepancy provenance changed")
    require(discrepancy["allow_pickle"] is False, "pickle was not prohibited")
    require(float(discrepancy["report_tolerance"]) == float(config["report_tolerance"]), "report tolerance changed")
    require(len(discrepancy["cells"]) == 2, "hard-cell audit scope changed")
    for field in ("formal_authorization", "production_authorization", "remote_launch_authorization"):
        require(discrepancy[field] is False, f"discrepancy {field} leaked")
    catalog = discrepancy["selected_raw_catalog"]
    require(discrepancy["selected_raw_count"] == 128 and len(catalog) == 128, "science raw count changed")
    require(discrepancy["selected_raw_catalog_sha256"] == sha256_json(catalog), "science catalog hash mismatch")
    headline = discrepancy["headline_checks"]
    expected_headline = {
        "m8_hp64_P_q_top": 0.9128439674802393,
        "m8_hp64_U_q_top": 0.913491773670944,
        "m8_hp64_combined_q_top": 0.9131680270339482,
        "m8_mam_combined_q_top": 0.9927278950353573,
        "m6_P_hp64_mam_absolute_delta": 0.016596369769588087,
        "m6_P_hp64_mam_paired_se": 0.0005425377386565906,
        "m6_P_hp64_mam_z": 30.590258680775513,
    }
    for name, value in expected_headline.items():
        require(_float_equal(headline[name], value), f"headline value changed: {name}")
    require(headline["m8_interpretation"] == "0.91317_VS_0.99273_IS_HP64_VS_MAM_NOT_HP64_P_VS_U", "m8 interpretation changed")

    _require_exact_keys(receipt, {
        "receipt_version", "contract_version", "authority", "status",
        "provenance", "resource_report_sha256", "discrepancy_audit_sha256",
        "config_file_sha256", "timing_raw_audit", "artifact_file_sha256",
        "formal_authorization", "production_authorization",
        "remote_launch_authorization", "receipt_sha256",
    }, "receipt")
    require(receipt["receipt_version"] == RECEIPT_VERSION, "receipt version changed")
    require(receipt["contract_version"] == CONTRACT_VERSION, "receipt contract changed")
    require(receipt["authority"] == "LOCAL_RESOURCE_AND_HISTORICAL_DIAGNOSTIC_AUDIT_ONLY", "receipt authority changed")
    require(receipt["status"] == authority["receipt_status"], "receipt status changed")
    require(receipt["provenance"] == provenance, "receipt provenance changed")
    require(receipt["config_file_sha256"] == sha256_file(config_path), "receipt config hash changed")
    require(receipt["resource_report_sha256"] == resource_hash, "receipt resource hash mismatch")
    require(receipt["discrepancy_audit_sha256"] == discrepancy_hash, "receipt discrepancy hash mismatch")
    require(receipt["timing_raw_audit"].get("status") == "PASS", "timing raw audit status changed")
    require(receipt["timing_raw_audit"].get("allow_pickle") is False, "timing raw audit pickle flag changed")
    require(len(receipt["timing_raw_audit"].get("cells", [])) == 5, "timing raw audit scope changed")
    for field in ("formal_authorization", "production_authorization", "remote_launch_authorization"):
        require(receipt[field] is False, f"receipt {field} leaked")

    with coverage_path.open("r", encoding="ascii", newline="") as handle:
        coverage_csv = list(csv.DictReader(handle))
    with scenarios_path.open("r", encoding="ascii", newline="") as handle:
        scenarios_csv = list(csv.DictReader(handle))
    require(len(coverage_csv) == len(resource["timing_coverage"]), "coverage CSV/JSON count mismatch")
    require(len(scenarios_csv) == len(resource["resource_scenarios"]), "scenario CSV/JSON count mismatch")
    for csv_row, json_row in zip(coverage_csv, resource["timing_coverage"]):
        require(int(csv_row["m"]) == int(json_row["m"]), "coverage CSV m mismatch")
        require(_float_equal(csv_row["p"], json_row["p"]), "coverage CSV p mismatch")
        require(csv_row["coverage_status"] == json_row["coverage_status"], "coverage CSV status mismatch")
    for csv_row, json_row in zip(scenarios_csv, resource["resource_scenarios"]):
        require(csv_row["stage"] == json_row["stage"], "scenario CSV stage mismatch")
        require(csv_row["scenario"] == json_row["scenario"], "scenario CSV name mismatch")
        require(csv_row["clock"] == json_row["clock"], "scenario CSV clock mismatch")
        require(int(csv_row["trajectory_count"]) == int(json_row["trajectory_count"]), "scenario CSV trajectory mismatch")
        require(_float_equal(csv_row["safety_adjusted_total_core_seconds"], json_row["safety_adjusted_total_core_seconds"]), "scenario CSV total mismatch")
    artifact_paths = {
        resource_path.name: resource_path,
        discrepancy_path.name: discrepancy_path,
        coverage_path.name: coverage_path,
        scenarios_path.name: scenarios_path,
        markdown_path.name: markdown_path,
    }
    require(set(receipt["artifact_file_sha256"]) == set(artifact_paths), "receipt artifact set mismatch")
    for name, path in artifact_paths.items():
        require(receipt["artifact_file_sha256"][name] == sha256_file(path), f"artifact hash mismatch: {name}")
    return {
        "verified_status": "PACKAGE_CONTENT_PASS",
        "resource_report_sha256": resource_hash,
        "discrepancy_audit_sha256": discrepancy_hash,
        "receipt_sha256": receipt_hash,
        "provenance": provenance,
    }


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"one-shot independent audit already exists: {path}")
    assert_finite_json(value, path.name)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditFailure(f"one-shot independent audit already exists: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--config", type=Path, default=here / "resource_model_config.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output_dir / INDEPENDENT_OUTPUT_NAME
    require(not output_path.exists(), f"one-shot independent audit already exists: {output_path}")
    config = load_config(args.config)
    source_identity = verify_auditor_source(args.config, config, args.output_dir)
    package = audit(
        args.output_dir, config=config, config_path=args.config,
        source_identity=source_identity,
    )
    auditor_relative = f"{VALIDATION_RELATIVE_DIR}/audit_resource_calibration.py"
    identity = {
        "audit_version": INDEPENDENT_AUDIT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": config["authority"]["independent_audit_status"],
        "authority": "SERIALIZED_PACKAGE_INTEGRITY_ONLY",
        "calibration_source": source_identity,
        "implementation_authority_sha256": sha256_json(config["authority"]),
        "auditor": {
            "path": auditor_relative,
            "file_sha256": sha256_file(Path(__file__).resolve()),
        },
        "resource_config_file_sha256": sha256_file(args.config),
        "verified_package": package,
        "formal_authorization": False,
        "production_authorization": False,
        "remote_launch_authorization": False,
    }
    require(
        identity["auditor"]["file_sha256"]
        == config["authority"]["implementation_files"][auditor_relative],
        "auditor self SHA does not match authority",
    )
    report = {**identity, "audit_sha256": sha256_json(identity)}
    exclusive_json(output_path, report)
    print(canonical_json({
        "status": report["status"],
        "audit_sha256": report["audit_sha256"],
        "output": output_path.name,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
