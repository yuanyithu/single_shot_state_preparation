#!/usr/bin/env python3
"""Build the deterministic scan-v3 aggregation-safety certification.

The evidence path deliberately avoids a long stochastic simulation.  It
feeds fingerprint-valid synthetic disorder chunks through the real v3 merge,
exercises the publication loader, and calls the real bounds producers with
deterministic fixtures.  A certification is complete only when this evidence
and the entire exp101 pytest suite pass in conda environment ``12``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
EXP101_ROOT = HERE.parents[1]
REPOSITORY_ROOT = EXP101_ROOT.parents[2]
if str(EXP101_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import exact_reference  # noqa: E402
from src.graphs import repetition_parity_check_matrix  # noqa: E402
from src.hgp import hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import (  # noqa: E402
    PHYSICS_CONTRACT_VERSION,
    assemble_sector_model,
    disorder_from_uniforms,
    wire_ensemble,
)
from src.observables import (  # noqa: E402
    ObservableSet,
    build_observable_frame,
    posterior_statistics,
)
from src.run_scan import (  # noqa: E402
    AGGREGATION_POLICY,
    PROTOCOL_VERSION,
    _build_specs,
    _sampled_estimator_result,
    implementation_fingerprint,
    merge,
)
from src.scan_results import (  # noqa: E402
    ESTIMATED_MAP_PURITY_BOUNDS_LABEL,
    NonPublicationEnsembleError,
    PublicationPointNotReportableError,
    UnsupportedScanContractError,
    load_publication_q_top,
    map_success_bound_plot_label,
)
from src.sector_ti import (  # noqa: E402
    SectorTiConfig,
    _attach_full_sector_statistics,
    run_sector_ti,
)


EVIDENCE_JSON = HERE / "deterministic_aggregation_bounds_evidence.json"
EVIDENCE_MD = HERE / "deterministic_aggregation_bounds_evidence.md"
ENVIRONMENT_JSON = HERE / "environment.json"
PYTEST_OUTPUT = HERE / "pytest_full_output.txt"
PYTEST_EXIT = HERE / "pytest_exit_code.txt"
SUMMARY_MD = HERE / "summary.md"
COUNT_NAMES = ("planned", "present", "valid", "invalid", "missing")


class Audit:
    """Count named assertions and fail with an auditable message."""

    def __init__(self):
        self.assertions: list[str] = []

    def require(self, condition, name):
        if not bool(condition):
            raise AssertionError(name)
        self.assertions.append(str(name))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def _git_environment():
    rc_sha, sha = _command_output(
        ["git", "rev-parse", "HEAD"], REPOSITORY_ROOT
    )
    rc_status, status = _command_output(
        ["git", "status", "--short"], REPOSITORY_ROOT
    )
    return {
        "commit_sha": sha if rc_sha == 0 else "unknown",
        "commit_query_exit_code": rc_sha,
        "worktree_dirty": bool(status) if rc_status == 0 else None,
        "status_query_exit_code": rc_status,
    }


def _collect_environment(pytest_command):
    conda_name = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    prefix_name = Path(conda_prefix).name if conda_prefix else None
    environment_is_12 = conda_name == "12" and prefix_name == "12"
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "required_conda_environment": "12",
        "conda_default_env": conda_name,
        "conda_prefix": conda_prefix,
        "conda_environment_verified": environment_is_12,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pytest_version": importlib.metadata.version("pytest"),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "implementation_fingerprint": implementation_fingerprint(),
        "git": _git_environment(),
        "pytest_command": pytest_command,
        "pytest_command_shell": shlex.join(pytest_command),
        "pytest_cwd": str(EXP101_ROOT),
    }


def _coverage_inventory(evidence):
    aggregation = evidence.get("aggregation", {})
    cases = aggregation.get("cases", {})
    loader_checks = aggregation.get("loader_checks", {})
    bounds = evidence.get("bounds", {}).get("producers", {})
    return {
        "aggregation_cases": {
            name: case.get("status") for name, case in cases.items()
        },
        "loader_checks": {
            name: result.get("outcome", result.get("exception"))
            for name, result in loader_checks.items()
        },
        "bounds_producer_kinds": {
            name: result.get("kind") for name, result in bounds.items()
        },
        "invalid_weight_rejection_count": len(
            evidence.get("bounds", {}).get("invalid_weight_rejections", [])
        ),
    }


def _finalize_environment(
    environment,
    evidence,
    *,
    pytest_status,
    pytest_exit_code,
    overall_status,
):
    environment.update({
        "pytest_status": pytest_status,
        "pytest_exit_code": pytest_exit_code,
        "pytest_log_sha256": _sha256_file(PYTEST_OUTPUT),
        "evidence_assertion_count": evidence.get("assertion_count", 0),
        "evidence_coverage_inventory": _coverage_inventory(evidence),
        "evidence_json_sha256": _sha256_file(EVIDENCE_JSON),
        "overall_status": overall_status,
    })
    _write_json(ENVIRONMENT_JSON, environment)


def _write_chunk(spec, result):
    path = Path(spec["chunk_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "result": result,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def _synthetic_result(
    spec,
    *,
    q_top,
    valid,
    formal_only=False,
    numerical_valid=None,
):
    numerical_valid = valid if numerical_valid is None else numerical_valid
    values = [0.1]
    purity = None if q_top is None else (1.0 + float(q_top)) / 2.0
    result = {
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "git_commit_sha": "deterministic-synthetic-fixture",
        "git_worktree_dirty": False,
        "family": {"family": "surface", "size": 2},
        "k": 1,
        "code_fingerprint": "deterministic-code",
        "section_fingerprint": "deterministic-section",
        "observable_frame_fingerprint": "deterministic-frame",
        "observable_set_fingerprint": "deterministic-observable-set",
        "resolved_engine": spec["resolved_engine"],
        "resolved_engine_config": spec["engine_config"],
        "character_count": 1,
        "u_bitmasks": [1],
        "character_means_absolute": values,
        "character_means_relative": values,
        "m2_u_pooled_square_raw": values,
        "m2_u_debiased": values,
        "m2_u_debiased_jackknife_se": [0.01],
        "q_top_estimate": q_top,
        "q_top_absolute": q_top,
        "q_top_relative": q_top,
        "q_top_estimator_name": "deterministic_synthetic_fixture",
        "posterior_purity": purity,
        "posterior_mass_on_planted_class": None,
        "map_success_probability": None,
        "map_success_algebraic_lower_bound": None,
        "map_success_algebraic_upper_bound": None,
        "map_success_estimated_lower_bound": (
            purity if valid and not formal_only else None
        ),
        "map_success_estimated_upper_bound": (
            math.sqrt(purity) if valid and not formal_only else None
        ),
        "map_success_bound_kind": (
            "sampled_u_statistic_plugin_no_coverage"
            if valid and not formal_only
            else "unavailable"
        ),
        "map_success_bound_has_confidence_coverage": False,
        "weights_are_exact_sector_posterior": False,
        "weights_cover_all_sectors": False,
        "valid_for_aggregation": bool(valid),
        "numerically_valid": bool(numerical_valid),
        "formal_only": bool(formal_only),
        "failure_reasons": [] if numerical_valid else ["synthetic_invalid"],
        "flags": (
            "FORMAL_ONLY:legacy_delta_only"
            if formal_only
            else "PASS" if valid else "INVALID:synthetic_invalid"
        ),
    }
    if formal_only:
        result.update({
            "q_top_estimate": None,
            "q_top_absolute": None,
            "q_top_relative": None,
            "q_top_estimator_name": None,
            "posterior_purity": None,
            "formal_q_top": q_top,
            "formal_q_top_absolute": q_top,
            "formal_q_top_relative": q_top,
            "formal_q_top_estimator_name": (
                "deterministic_synthetic_fixture"
            ),
        })
    return result


def _archive_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "status": str(data["aggregation_status_per_point"][0, 0]),
            "reason": str(
                data["aggregation_failure_reasons_per_point"][0, 0]
            ),
            "reportable": bool(
                data["reportable_for_crossing_fss"][0, 0]
            ),
            "counts": {
                name: int(data[f"{name}_disorder_count"][0, 0])
                for name in COUNT_NAMES
            },
            "numerically_valid_count": int(
                data["numerically_valid_disorder_count"][0, 0]
            ),
            "formal_only_count": int(
                data["formal_only_disorder_count"][0, 0]
            ),
            "paper_aggregation_fraction": float(
                data["paper_aggregation_fraction"][0, 0]
            ),
            "numerical_pass_fraction": float(
                data["numerical_pass_fraction"][0, 0]
            ),
            "mean": float(data["mean_q_top_estimate"][0, 0]),
            "sem": float(data["disorder_sem_q_top_estimate"][0, 0]),
            "conditional_mean": float(
                data[
                    "conditional_mean_q_top_estimate_valid_only"
                ][0, 0]
            ),
            "conditional_sem": float(
                data[
                    "conditional_disorder_sem_q_top_estimate_valid_only"
                ][0, 0]
            ),
            "raw_q_top": data[
                "q_top_estimate_per_disorder"
            ][0, 0].astype(float),
            "crossing_input": data[
                "q_top_crossing_input_per_disorder"
            ][0, 0].astype(float),
            "valid_mask": data["valid_for_aggregation"][0, 0].astype(bool),
        }


def _make_merge_case(
    root,
    name,
    q_top_values,
    valid_values,
    *,
    ensemble="true_posterior",
    present_values=None,
    formal_only=False,
):
    output = root / name
    num_disorders = len(q_top_values)
    specs = _build_specs(
        output,
        "surface",
        [2],
        0.1,
        [0.05],
        num_disorders,
        "x_error",
        ensemble,
        "direct",
        {},
        "full_rank",
        None,
        64,
    )
    if present_values is None:
        present_values = [True] * num_disorders
    for spec, q_top, valid, present in zip(
        specs, q_top_values, valid_values, present_values
    ):
        if not present:
            continue
        result = _synthetic_result(
            spec,
            q_top=q_top,
            valid=valid,
            formal_only=formal_only,
            numerical_valid=True if formal_only else valid,
        )
        _write_chunk(spec, result)
    path = merge(
        output,
        "surface",
        [2],
        0.1,
        [0.05],
        num_disorders,
        "x_error",
        ensemble,
        "direct",
        {},
        "full_rank",
        expected_specs=specs,
    )
    return path, _archive_snapshot(path)


def _expect_loader_rejection(audit, path, exception_type, fragments, name):
    try:
        load_publication_q_top(path)
    except exception_type as error:
        message = str(error)
    else:
        raise AssertionError(f"{name}: loader unexpectedly accepted file")
    for fragment in fragments:
        audit.require(fragment in message, f"{name}: message contains {fragment}")
    return {"exception": exception_type.__name__, "message": message}


def _make_v2_copy(source, destination):
    with np.load(source, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    manifest = json.loads(str(payload["manifest_json"].item()))
    manifest["protocol"] = "exp101.scan.v2"
    manifest["scan_contract_version"] = "exp101.scan.v2"
    payload["scan_contract_version"] = np.asarray("exp101.scan.v2")
    payload["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, ensure_ascii=True)
    )
    np.savez_compressed(destination, **payload)


def _build_aggregation_evidence(audit, root):
    paths = {}
    cases = {}
    paths["reportable"], cases["reportable"] = _make_merge_case(
        root, "reportable", [0.2, 0.6], [True, True]
    )
    paths["invalid"], cases["invalid"] = _make_merge_case(
        root, "invalid", [0.2, 0.6, 0.99], [True, True, False]
    )
    paths["missing"], cases["missing"] = _make_merge_case(
        root,
        "missing",
        [0.2, 0.6, 0.8],
        [True, True, True],
        present_values=[True, True, False],
    )
    paths["legacy"], cases["legacy"] = _make_merge_case(
        root,
        "legacy",
        [0.2, 0.6],
        [False, False],
        ensemble="legacy_delta_only",
        formal_only=True,
    )
    paths["single"], cases["single"] = _make_merge_case(
        root, "single", [0.3], [True]
    )

    reportable = cases["reportable"]
    audit.require(reportable["status"] == "REPORTABLE", "all-valid status")
    audit.require(reportable["reportable"], "all-valid publication gate")
    audit.require(reportable["counts"] == {
        "planned": 2,
        "present": 2,
        "valid": 2,
        "invalid": 0,
        "missing": 0,
    }, "all-valid counts")
    audit.require(np.isclose(reportable["mean"], 0.4), "all-valid mean")
    audit.require(np.isclose(reportable["sem"], 0.2), "all-valid SEM")
    audit.require(
        np.isclose(reportable["conditional_mean"], reportable["mean"]),
        "all-valid conditional mean equals official mean",
    )
    audit.require(
        np.isclose(reportable["conditional_sem"], reportable["sem"]),
        "all-valid conditional SEM equals official SEM",
    )
    audit.require(
        np.allclose(reportable["crossing_input"], [0.2, 0.6]),
        "all-valid crossing input",
    )

    invalid = cases["invalid"]
    audit.require(
        invalid["status"] == "SAMPLING_INSUFFICIENT",
        "invalid status is sampling insufficient",
    )
    audit.require(not invalid["reportable"], "invalid gate is closed")
    audit.require(invalid["counts"] == {
        "planned": 3,
        "present": 3,
        "valid": 2,
        "invalid": 1,
        "missing": 0,
    }, "invalid counts")
    audit.require(
        invalid["reason"] == "invalid_disorders_present",
        "invalid reason",
    )
    audit.require(np.isnan(invalid["mean"]), "invalid official mean is NaN")
    audit.require(np.isnan(invalid["sem"]), "invalid official SEM is NaN")
    audit.require(
        np.isnan(invalid["crossing_input"]).all(),
        "invalid whole-point crossing input is NaN",
    )
    audit.require(
        np.isclose(invalid["conditional_mean"], 0.4),
        "invalid conditional mean retained",
    )
    audit.require(
        np.isclose(invalid["conditional_sem"], 0.2),
        "invalid conditional SEM retained",
    )
    audit.require(
        np.isclose(invalid["raw_q_top"][2], 0.99),
        "invalid raw estimator retained",
    )
    audit.require(
        np.isclose(invalid["paper_aggregation_fraction"], 2.0 / 3.0),
        "invalid paper fraction uses planned denominator",
    )
    audit.require(
        np.isclose(invalid["numerical_pass_fraction"], 2.0 / 3.0),
        "invalid numerical fraction uses planned denominator",
    )

    missing = cases["missing"]
    audit.require(missing["status"] == "INCOMPLETE", "missing status")
    audit.require(not missing["reportable"], "missing gate is closed")
    audit.require(missing["counts"] == {
        "planned": 3,
        "present": 2,
        "valid": 2,
        "invalid": 0,
        "missing": 1,
    }, "missing counts")
    audit.require(
        missing["reason"] == "missing_disorders_present",
        "missing reason",
    )
    audit.require(np.isnan(missing["mean"]), "missing official mean is NaN")
    audit.require(
        np.isnan(missing["crossing_input"]).all(),
        "missing whole-point crossing input is NaN",
    )
    audit.require(
        np.isclose(missing["conditional_mean"], 0.4),
        "missing conditional mean retained",
    )
    audit.require(
        np.isclose(missing["paper_aggregation_fraction"], 2.0 / 3.0),
        "missing paper fraction uses planned denominator",
    )
    audit.require(
        np.isclose(missing["numerical_pass_fraction"], 2.0 / 3.0),
        "missing numerical fraction uses planned denominator",
    )

    legacy = cases["legacy"]
    audit.require(legacy["status"] == "FORMAL_ONLY", "legacy status")
    audit.require(not legacy["reportable"], "legacy gate is closed")
    audit.require(legacy["counts"] == {
        "planned": 2,
        "present": 2,
        "valid": 0,
        "invalid": 0,
        "missing": 0,
    }, "legacy counts")
    audit.require(
        legacy["formal_only_count"] == 2,
        "legacy formal-only count",
    )
    audit.require(
        legacy["paper_aggregation_fraction"] == 0.0,
        "legacy paper fraction",
    )
    audit.require(
        legacy["numerical_pass_fraction"] == 1.0,
        "legacy numerical fraction uses planned denominator",
    )
    audit.require(np.isnan(legacy["mean"]), "legacy official mean is NaN")
    audit.require(
        np.isnan(legacy["crossing_input"]).all(),
        "legacy crossing input is NaN",
    )

    single = cases["single"]
    audit.require(single["status"] == "REPORTABLE", "single status")
    audit.require(single["reportable"], "single reportable gate")
    audit.require(single["mean"] == 0.3, "single mean")
    audit.require(np.isnan(single["sem"]), "single SEM is undefined NaN")
    audit.require(
        np.isnan(single["conditional_sem"]),
        "single conditional SEM is undefined NaN",
    )

    loaded = load_publication_q_top(paths["reportable"])
    audit.require(loaded.selected_point_count == 1, "loader accepts reportable")
    audit.require(
        np.isclose(loaded.mean_q_top_estimate[0, 0], 0.4),
        "loader reportable mean",
    )
    audit.require(
        loaded.map_success_bound_plot_label_per_disorder[0, 0, 0]
        == ESTIMATED_MAP_PURITY_BOUNDS_LABEL,
        "loader canonical plug-in label",
    )
    loader_checks = {
        "reportable": {
            "outcome": "accepted",
            "selected_point_count": loaded.selected_point_count,
        }
    }
    loader_checks["invalid"] = _expect_loader_rejection(
        audit,
        paths["invalid"],
        PublicationPointNotReportableError,
        ["size=2", "q=0.05", "SAMPLING_INSUFFICIENT"],
        "invalid loader rejection",
    )
    loader_checks["missing"] = _expect_loader_rejection(
        audit,
        paths["missing"],
        PublicationPointNotReportableError,
        ["size=2", "q=0.05", "INCOMPLETE"],
        "missing loader rejection",
    )
    loader_checks["legacy"] = _expect_loader_rejection(
        audit,
        paths["legacy"],
        NonPublicationEnsembleError,
        ["legacy_delta_only", "formal-only"],
        "legacy loader rejection",
    )
    v2_path = root / "scan_v2_audit_only.npz"
    _make_v2_copy(paths["reportable"], v2_path)
    loader_checks["scan_v2"] = _expect_loader_rejection(
        audit,
        v2_path,
        UnsupportedScanContractError,
        ["exp101.scan.v2", "audit-only", "conditional means"],
        "scan-v2 loader rejection",
    )

    return {
        "source": (
            "fingerprint-validated synthetic chunks merged by "
            "src.run_scan.merge"
        ),
        "aggregation_policy": dict(AGGREGATION_POLICY),
        "cases": cases,
        "loader_checks": loader_checks,
    }


def _build_model_and_wiring(p, q, ensemble="true_posterior"):
    H_Z, H_X = hgp_from_H(repetition_parity_check_matrix(2))
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(
        H_X, H_Z, logicals, sector="x_error"
    )
    frame = build_observable_frame(model)
    disorder = disorder_from_uniforms(
        model,
        p,
        q,
        np.ones(model.num_qubits, dtype=np.float64),
        np.ones(model.num_checks, dtype=np.float64),
    )
    wiring = wire_ensemble(model, disorder, ensemble, frame)
    return model, frame, wiring


def _bound_snapshot(result, source):
    return {
        "source": source,
        "kind": result["map_success_bound_kind"],
        "has_confidence_coverage": result[
            "map_success_bound_has_confidence_coverage"
        ],
        "weights_are_exact_sector_posterior": result.get(
            "weights_are_exact_sector_posterior"
        ),
        "posterior_purity": result.get("posterior_purity"),
        "map_success_probability": result.get("map_success_probability"),
        "algebraic_lower": result.get(
            "map_success_algebraic_lower_bound"
        ),
        "algebraic_upper": result.get(
            "map_success_algebraic_upper_bound"
        ),
        "estimated_lower": result.get(
            "map_success_estimated_lower_bound"
        ),
        "estimated_upper": result.get(
            "map_success_estimated_upper_bound"
        ),
    }


def _check_bound_partition(audit, result, expected_kind, category, name):
    audit.require(
        result["map_success_bound_kind"] == expected_kind,
        f"{name}: bound kind",
    )
    audit.require(
        result["map_success_bound_has_confidence_coverage"] is False,
        f"{name}: no confidence coverage",
    )
    algebraic = (
        result.get("map_success_algebraic_lower_bound"),
        result.get("map_success_algebraic_upper_bound"),
    )
    estimated = (
        result.get("map_success_estimated_lower_bound"),
        result.get("map_success_estimated_upper_bound"),
    )
    if category == "algebraic":
        audit.require(
            all(value is not None for value in algebraic),
            f"{name}: algebraic fields populated",
        )
        audit.require(
            all(value is None for value in estimated),
            f"{name}: estimated fields empty",
        )
    elif category == "estimated":
        audit.require(
            all(value is None for value in algebraic),
            f"{name}: algebraic fields empty",
        )
        audit.require(
            all(value is not None for value in estimated),
            f"{name}: estimated fields populated",
        )
    elif category == "unavailable":
        audit.require(
            all(value is None for value in algebraic + estimated),
            f"{name}: all bound fields empty",
        )
    else:
        raise AssertionError(f"unknown bounds category {category}")


def _sampled_observable_set():
    return ObservableSet(
        tier="full",
        k=1,
        u_bitmasks=np.asarray([1], dtype=np.int64),
        W_rows=np.zeros((1, 1), dtype=np.uint8),
        basis_positions=np.asarray([0], dtype=np.int64),
        num_random_u=0,
    )


def _build_bounds_evidence(audit):
    model, frame, wiring = _build_model_and_wiring(0.19, 0.11)
    exact = exact_reference(model, frame, wiring, force_python=True)

    endpoint_model, endpoint_frame, endpoint_wiring = (
        _build_model_and_wiring(0.5, 0.11)
    )
    endpoint = run_sector_ti(
        endpoint_model,
        endpoint_frame,
        endpoint_wiring,
        SectorTiConfig(),
        seed=0,
    )

    ti = _attach_full_sector_statistics(
        {"labels": list(range(1 << model.k))},
        model,
        wiring,
        np.asarray([0.73, 0.27]),
        q_top_stderr=0.01,
        estimator_name="deterministic_ti_producer_fixture",
        bound_kind="full_sector_ti_plugin_no_coverage",
    )

    observable_set = _sampled_observable_set()
    valid_means = np.asarray([[0.2], [0.3], [0.4], [0.5]])
    sampled = _sampled_estimator_result(
        observable_set,
        valid_means,
        valid_means,
        "true_posterior",
        planted_class=0,
        minimum_chains=4,
    )
    invalid_means = np.asarray([[1.0], [-1.0], [1.0], [-1.0]])
    sampled_invalid = _sampled_estimator_result(
        observable_set,
        invalid_means,
        invalid_means,
        "true_posterior",
        planted_class=0,
        minimum_chains=4,
    )
    sampled_legacy = _sampled_estimator_result(
        observable_set,
        valid_means,
        valid_means,
        "legacy_delta_only",
        planted_class=0,
        minimum_chains=4,
    )
    direct_statistics = posterior_statistics(
        np.asarray([0.1, 0.9]), planted_class=0
    )

    _check_bound_partition(
        audit,
        exact,
        "exact_posterior_algebraic",
        "algebraic",
        "exact producer",
    )
    audit.require(
        exact["weights_are_exact_sector_posterior"] is True,
        "exact producer weights exact",
    )
    _check_bound_partition(
        audit,
        endpoint,
        "analytic_endpoint_algebraic",
        "algebraic",
        "analytic endpoint producer",
    )
    audit.require(
        endpoint["weights_are_exact_sector_posterior"] is True,
        "analytic endpoint weights exact",
    )
    _check_bound_partition(
        audit,
        ti,
        "full_sector_ti_plugin_no_coverage",
        "estimated",
        "ordinary TI producer",
    )
    audit.require(
        ti["weights_are_exact_sector_posterior"] is False,
        "ordinary TI weights are estimated",
    )
    _check_bound_partition(
        audit,
        sampled,
        "sampled_u_statistic_plugin_no_coverage",
        "estimated",
        "sampled producer",
    )
    audit.require(
        sampled["weights_are_exact_sector_posterior"] is False,
        "sampled weights are estimated",
    )
    audit.require(
        sampled["map_success_probability"] is None,
        "sampled producer does not claim MAP success",
    )
    audit.require(
        not sampled["estimator_failure_reasons"],
        "sampled valid fixture passes estimator gates",
    )
    _check_bound_partition(
        audit,
        sampled_invalid,
        "unavailable",
        "unavailable",
        "sampled invalid producer",
    )
    audit.require(
        sampled_invalid["q_top_estimate"] < 0.0,
        "sampled invalid raw U-statistic remains negative",
    )
    audit.require(
        "debiased_posterior_purity_out_of_range"
        in sampled_invalid["estimator_failure_reasons"],
        "sampled invalid physical-range gate",
    )
    audit.require(
        sampled_invalid["map_success_probability"] is None,
        "sampled invalid does not claim MAP success",
    )
    _check_bound_partition(
        audit,
        sampled_legacy,
        "unavailable",
        "unavailable",
        "legacy sampled producer",
    )
    audit.require(
        sampled_legacy["posterior_purity"] is None,
        "legacy posterior purity hidden",
    )
    _check_bound_partition(
        audit,
        direct_statistics,
        "exact_posterior_algebraic",
        "algebraic",
        "posterior_statistics",
    )
    audit.require(
        direct_statistics["posterior_mass_on_planted_class"] == 0.1,
        "posterior planted mass",
    )
    audit.require(
        direct_statistics["map_success_probability"] == 0.9,
        "posterior MAP success differs from planted mass",
    )

    rejected_weights = []
    malformed = (
        ([-0.1, 1.1], "nonnegative"),
        ([0.1, 0.8], "sum to 1"),
        ([0.5, 0.5 + 2e-12], "sum to 1"),
        ([float("nan"), 1.0], "finite"),
        ([float("inf"), 0.0], "finite"),
    )
    for weights, expected_message in malformed:
        try:
            posterior_statistics(np.asarray(weights, dtype=np.float64))
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError(f"malformed weights accepted: {weights}")
        audit.require(
            expected_message in message,
            f"malformed weights reject with {expected_message}",
        )
        rejected_weights.append({
            "weights": weights,
            "exception": "ValueError",
            "message": message,
        })

    for kind in (
        "full_sector_ti_plugin_no_coverage",
        "sampled_u_statistic_plugin_no_coverage",
    ):
        audit.require(
            map_success_bound_plot_label(kind)
            == ESTIMATED_MAP_PURITY_BOUNDS_LABEL,
            f"canonical plot label for {kind}",
        )

    return {
        "producers": {
            "exact": _bound_snapshot(
                exact, "src.enumerate_exact.exact_reference"
            ),
            "analytic_endpoint": _bound_snapshot(
                endpoint, "src.sector_ti.run_sector_ti analytic endpoint"
            ),
            "ordinary_ti": _bound_snapshot(
                ti,
                "src.sector_ti._attach_full_sector_statistics producer",
            ),
            "sampled_valid": _bound_snapshot(
                sampled, "src.run_scan._sampled_estimator_result"
            ),
            "sampled_invalid": {
                **_bound_snapshot(
                    sampled_invalid,
                    "src.run_scan._sampled_estimator_result",
                ),
                "raw_q_top_estimate": sampled_invalid["q_top_estimate"],
                "estimator_failure_reasons": sampled_invalid[
                    "estimator_failure_reasons"
                ],
            },
            "legacy": _bound_snapshot(
                sampled_legacy,
                "src.run_scan._sampled_estimator_result legacy path",
            ),
            "posterior_statistics": _bound_snapshot(
                direct_statistics, "src.observables.posterior_statistics"
            ),
        },
        "invalid_weight_rejections": rejected_weights,
        "plug_in_plot_label": ESTIMATED_MAP_PURITY_BOUNDS_LABEL,
    }


def _evidence_markdown(evidence):
    lines = [
        "# Deterministic aggregation and bounds evidence",
        "",
        f"Status: **{evidence['status']}**",
        "",
        "This evidence uses the real scan-v3 merge and producer functions;",
        "no long stochastic sampling is part of the certification fixture.",
        "",
        "## Parameter-point aggregation",
        "",
        "| case | status | planned/present/valid/invalid/missing | "
        "official mean | conditional mean | crossing finite |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, case in evidence["aggregation"]["cases"].items():
        counts = case["counts"]
        count_text = "/".join(
            str(counts[key])
            for key in COUNT_NAMES
        )
        crossing_finite = bool(
            np.isfinite(np.asarray(case["crossing_input"], dtype=float)).all()
        )
        lines.append(
            f"| {name} | {case['status']} | {count_text} | "
            f"{_json_safe(case['mean'])} | "
            f"{_json_safe(case['conditional_mean'])} | "
            f"{crossing_finite} |"
        )
    lines.extend([
        "",
        "Invalid and missing points retain raw and valid-only diagnostics, "
        "but their official mean, SEM, and entire crossing row are NaN. "
        "The valid-only statistics are diagnostics-only because conditioning "
        "on gate success can introduce selection bias.",
        "",
        "## Publication loader",
        "",
        "| case | outcome |",
        "|---|---|",
    ])
    for name, result in evidence["aggregation"]["loader_checks"].items():
        outcome = result.get("outcome", result.get("exception"))
        lines.append(f"| {name} | {outcome} |")
    lines.extend([
        "",
        "## MAP-purity bounds",
        "",
        "| producer | kind | exact weights | algebraic | estimated |",
        "|---|---|---:|---:|---:|",
    ])
    for name, result in evidence["bounds"]["producers"].items():
        algebraic = result["algebraic_lower"] is not None
        estimated = result["estimated_lower"] is not None
        lines.append(
            f"| {name} | {result['kind']} | "
            f"{result['weights_are_exact_sector_posterior']} | "
            f"{algebraic} | {estimated} |"
        )
    lines.extend([
        "",
        "Plug-in plot label: `"
        + evidence["bounds"]["plug_in_plot_label"]
        + "`.",
        "",
        f"Assertions passed: **{evidence['assertion_count']}**.",
        "",
    ])
    return "\n".join(lines)


def _build_evidence():
    audit = Audit()
    audit.require(
        PROTOCOL_VERSION == "exp101.scan.v3",
        "scan contract is exp101.scan.v3",
    )
    audit.require(
        PHYSICS_CONTRACT_VERSION == "exp101.physics.v2",
        "physics contract remains exp101.physics.v2",
    )
    audit.require(
        AGGREGATION_POLICY == {
            "point_eligibility": "all_planned_disorders_valid",
            "fraction_denominator": "planned_disorders",
            "maximum_invalid_disorders": 0,
            "maximum_missing_disorders": 0,
            "conditional_statistics_purpose": "diagnostics_only",
            "conditional_statistics_are_publication_eligible": False,
            "crossing_input_policy": "whole_point_nan_unless_reportable",
        },
        "fixed fail-closed aggregation policy",
    )
    with tempfile.TemporaryDirectory(
        prefix=".tmp_aggregation_safety_", dir=HERE
    ) as temporary:
        root = Path(temporary)
        aggregation = _build_aggregation_evidence(audit, root)
        bounds = _build_bounds_evidence(audit)
    return {
        "status": "PASS",
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "implementation_fingerprint": implementation_fingerprint(),
        "aggregation": aggregation,
        "bounds": bounds,
        "assertion_count": len(audit.assertions),
        "assertions": audit.assertions,
    }


def _summary(status, environment, evidence, pytest_status, pytest_code):
    evidence_status = evidence.get("status", "FAIL")
    assertion_count = evidence.get("assertion_count", 0)
    lines = [
        "# validation/015 - aggregation safety",
        "",
        f"Status: **{status}**",
        "",
        "- Physics contract: `exp101.physics.v2` (unchanged)",
        "- Scan contract: `exp101.scan.v3`",
        "- Conda environment `12`: "
        + ("PASS" if environment["conda_environment_verified"] else "FAIL"),
        f"- Deterministic aggregation/bounds evidence: {evidence_status} "
        f"({assertion_count} assertions)",
        f"- Full pytest suite: {pytest_status}",
        "- Pytest command: `" + environment["pytest_command_shell"] + "`",
        "- Pytest exit record: `pytest_exit_code.txt` "
        f"({pytest_code})",
        "",
    ]
    if status == "PASS":
        lines.extend([
            "This directory certifies publication-safe parameter-point "
            "aggregation and MAP-bound naming for scan v3. Validation 014 "
            "continues to certify physics v2 only; its scan-v2 aggregation "
            "is historical audit evidence.",
            "",
        ])
    elif status == "INCOMPLETE":
        lines.extend([
            "Deterministic evidence passed, but `--skip-pytest` was used. "
            "This run does not certify scan v3. Re-run without "
            "`--skip-pytest` in conda environment `12`.",
            "",
        ])
    else:
        lines.extend([
            "Certification failed. See "
            "`deterministic_aggregation_bounds_evidence.json` and "
            "`pytest_full_output.txt` before changing status to DONE.",
            "",
        ])
    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Certify exp101 scan-v3 aggregation and bounds semantics."
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Build deterministic evidence only; summary remains INCOMPLETE.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(EXP101_ROOT / "tests"),
    ]
    environment = _collect_environment(pytest_command)
    _write_json(ENVIRONMENT_JSON, environment)

    if not environment["conda_environment_verified"]:
        evidence = {
            "status": "FAIL",
            "error": (
                "runner requires CONDA_DEFAULT_ENV=12 and a CONDA_PREFIX "
                "whose basename is 12"
            ),
            "assertion_count": 0,
        }
        _write_json(EVIDENCE_JSON, evidence)
        EVIDENCE_MD.write_text(
            "# Deterministic aggregation and bounds evidence\n\n"
            "Status: **FAIL** - conda environment `12` was not active.\n",
            encoding="utf-8",
        )
        PYTEST_OUTPUT.write_text(
            "NOT RUN: conda environment 12 was not verified.\n",
            encoding="utf-8",
        )
        PYTEST_EXIT.write_text("NOT_RUN\n", encoding="utf-8")
        _finalize_environment(
            environment,
            evidence,
            pytest_status="NOT_RUN",
            pytest_exit_code=None,
            overall_status="FAIL",
        )
        SUMMARY_MD.write_text(
            _summary("FAIL", environment, evidence, "NOT RUN", "NOT_RUN"),
            encoding="utf-8",
        )
        return 2

    try:
        evidence = _build_evidence()
    except Exception as error:
        evidence = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "assertion_count": 0,
        }
        _write_json(EVIDENCE_JSON, evidence)
        EVIDENCE_MD.write_text(
            "# Deterministic aggregation and bounds evidence\n\n"
            f"Status: **FAIL** - {type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        PYTEST_OUTPUT.write_text(
            "NOT RUN: deterministic evidence failed.\n",
            encoding="utf-8",
        )
        PYTEST_EXIT.write_text("NOT_RUN\n", encoding="utf-8")
        _finalize_environment(
            environment,
            evidence,
            pytest_status="NOT_RUN",
            pytest_exit_code=None,
            overall_status="FAIL",
        )
        SUMMARY_MD.write_text(
            _summary("FAIL", environment, evidence, "NOT RUN", "NOT_RUN"),
            encoding="utf-8",
        )
        return 1

    _write_json(EVIDENCE_JSON, evidence)
    EVIDENCE_MD.write_text(_evidence_markdown(evidence), encoding="utf-8")

    if args.skip_pytest:
        PYTEST_OUTPUT.write_text(
            "NOT RUN: --skip-pytest requested; certification incomplete.\n",
            encoding="utf-8",
        )
        PYTEST_EXIT.write_text("NOT_RUN\n", encoding="utf-8")
        _finalize_environment(
            environment,
            evidence,
            pytest_status="NOT_RUN",
            pytest_exit_code=None,
            overall_status="INCOMPLETE",
        )
        SUMMARY_MD.write_text(
            _summary(
                "INCOMPLETE", environment, evidence, "NOT RUN", "NOT_RUN"
            ),
            encoding="utf-8",
        )
        return 0

    completed = subprocess.run(
        pytest_command,
        cwd=EXP101_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    PYTEST_OUTPUT.write_text(completed.stdout, encoding="utf-8")
    PYTEST_EXIT.write_text(f"{completed.returncode}\n", encoding="utf-8")
    status = "PASS" if completed.returncode == 0 else "FAIL"
    _finalize_environment(
        environment,
        evidence,
        pytest_status=status,
        pytest_exit_code=completed.returncode,
        overall_status=status,
    )
    SUMMARY_MD.write_text(
        _summary(
            status,
            environment,
            evidence,
            status,
            str(completed.returncode),
        ),
        encoding="utf-8",
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
