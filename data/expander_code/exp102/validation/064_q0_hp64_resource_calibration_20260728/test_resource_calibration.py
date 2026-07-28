import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import audit_resource_calibration as independent_audit
import run_resource_calibration as calibration


HERE = Path(__file__).resolve().parent


def test_u_statistic_squares_matches_explicit_cross_trajectory_products():
    means = np.asarray([
        [0.2, -0.3, 0.7],
        [0.6, 0.1, -0.2],
        [-0.4, 0.8, 0.5],
        [0.9, -0.5, 0.0],
    ])
    expected = []
    for character in range(means.shape[1]):
        products = [
            means[left, character] * means[right, character]
            for left in range(means.shape[0])
            for right in range(means.shape[0])
            if left != right
        ]
        expected.append(np.mean(products))
    np.testing.assert_allclose(
        calibration.u_statistic_squares(means), expected, rtol=0.0, atol=2e-16
    )


def test_full_character_population_is_a_census_with_zero_character_se():
    masks = np.asarray([1, 2, 3], dtype=np.uint64)
    values = np.asarray([0.1, 0.7, -0.2])
    estimate, finite_se, design = calibration.character_population_estimate(
        values, masks, 2
    )
    assert estimate == pytest.approx(values.mean(), abs=1e-16)
    assert finite_se == 0.0
    assert design["tier"] == "full"
    assert design["basis_positions"] == [0, 1]


def test_sampled_character_population_uses_basis_census_and_fpc():
    masks = np.asarray([1, 2, 4, 8, 3, 5, 6], dtype=np.uint64)
    values = np.asarray([0.1, 0.2, 0.3, 0.4, -0.2, 0.6, 0.9])
    estimate, finite_se, design = calibration.character_population_estimate(
        values, masks, 4
    )
    total = 15
    remaining = 11
    sampled = values[4:]
    expected = (values[:4].sum() + remaining * sampled.mean()) / total
    expected_se = (
        remaining / total
        * math.sqrt((1.0 - sampled.size / remaining) * sampled.var(ddof=1) / sampled.size)
    )
    assert estimate == pytest.approx(expected, abs=1e-16)
    assert finite_se == pytest.approx(expected_se, abs=1e-16)
    assert design["tier"] == "sampled"


def test_paired_delta_uses_two_delete_one_jackknife_variances():
    masks = np.asarray([1, 2, 3], dtype=np.uint64)
    left = np.asarray([
        [0.1, 0.2, -0.1],
        [0.4, -0.3, 0.2],
        [-0.2, 0.5, 0.7],
        [0.8, 0.1, -0.4],
    ])
    right = np.asarray([
        [-0.1, 0.3, 0.4],
        [0.2, -0.2, 0.1],
        [0.5, 0.4, -0.6],
        [-0.3, 0.6, 0.2],
        [0.7, -0.1, 0.3],
    ])
    result = calibration.paired_qtop_delta(left, right, masks, 2)
    left_q = calibration.qtop_estimate(left, masks, 2)
    right_q = calibration.qtop_estimate(right, masks, 2)
    assert result["signed_delta"] == pytest.approx(
        left_q["estimate"] - right_q["estimate"], abs=2e-16
    )
    expected_variance = 0.0
    for contrasts in (
        left_q["delete_one"] - right_q["estimate"],
        left_q["estimate"] - right_q["delete_one"],
    ):
        expected_variance += (
            (contrasts.size - 1) / contrasts.size
            * np.square(contrasts - contrasts.mean()).sum()
        )
    assert result["trajectory_se"] == pytest.approx(
        math.sqrt(expected_variance), abs=2e-16
    )
    assert result["character_se"] == 0.0


def test_uint64_bit_63_is_a_basis_character_without_int64_conversion():
    basis = [np.uint64(1 << bit) for bit in range(64)]
    masks = np.asarray(basis + [np.uint64(3), np.uint64(5), np.uint64(6)])
    design = calibration.infer_character_design(masks, 64)
    assert design["basis_positions"] == list(range(64))
    assert int(masks[63]) == 1 << 63
    assert design["tier"] == "sampled"


def test_character_means_from_uint64_labels_matches_python_parity():
    labels = np.asarray([0, 1, 2, 3, 1 << 63], dtype=np.uint64)
    masks = np.asarray([1, 2, 3, 1 << 63], dtype=np.uint64)
    actual = calibration.character_means_from_labels(labels, masks, chunk_size=2)
    expected = []
    for mask in masks:
        signs = [1 - 2 * ((int(label) & int(mask)).bit_count() & 1) for label in labels]
        expected.append(np.mean(signs))
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1e-16)


def test_resource_arithmetic_includes_generation_replay_analysis_and_safety():
    config = calibration.load_config(HERE / "resource_model_config.json")
    result = calibration.project_resource_option(
        config=config,
        evaluation_count_by_m={3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        seconds_per_t3_trajectory_by_m={m: 10.0 for m in config["m_values"]},
        t3_analysis_seconds_per_evaluation=100.0,
        analysis_scale=13.0 / 60.0,
        trajectory_count=8,
        clock_name="T1",
    )
    assert result["generation_core_seconds"] == 40.0
    assert result["replay_core_seconds"] == 40.0
    assert result["analysis_proxy_core_seconds"] == pytest.approx(130.0 / 3.0)
    assert result["unsafetied_total_core_seconds"] == pytest.approx(370.0 / 3.0)
    assert result["safety_adjusted_total_core_seconds"] == pytest.approx(740.0 / 3.0)


def _fake_timing_evidence():
    return {
        "observations": [
            {
                "m": m,
                "code_id": code,
                "p": p,
                "disorder_index": 0,
                "disorder_source": source,
                "resource_tier": "T3",
                "trajectory_count": 32,
                "cell_core_seconds": float(1000 * m),
                "seconds_per_trajectory_t3": float(1000 * m / 32),
                "evidence_class": "EMPIRICAL_SINGLE_CELL_SINGLE_DISORDER",
            }
            for m, code, p, source in calibration.HP64_TIMING_CELLS
        ],
        "missing_m_values": [7],
        "historical_trajectory_count": 32,
    }


def _fake_analysis_evidence():
    return {
        "benchmark_measurement_rounds": 32768,
        "maxima_by_field": {},
        "analysis_scale_by_clock": {
            "T1": 13.0 / 60.0,
            "T2": 7.0 / 15.0,
            "T3": 1.0,
            "2T": 2.0,
        },
        "t3_proxy_seconds_per_evaluation": 118.95883645396679,
        "proxy_components": {
            "families_per_evaluation": 2,
            "comparisons_per_evaluation": 1,
        },
        "trace_postprocess_accounting": "INCLUDED_IN_COMPLETE_REPLAY_NOT_ADDED_TO_ANALYSIS_PROXY",
        "coverage_limitation": "VALIDATION_013_B_FAMILY_AND_B_COMPARISON_PROXY_ONLY",
        "two_t_scale_status": "CONFIG_EXTRAPOLATION_UNVALIDATED",
    }


def test_resource_model_emits_all_options_selects_none_and_keeps_strict_null():
    config = calibration.load_config(HERE / "resource_model_config.json")
    report = calibration.build_resource_model(
        config=config,
        timing_evidence=_fake_timing_evidence(),
        analysis_timing_evidence=_fake_analysis_evidence(),
        provenance={"test": "identity"},
    )
    assert len(report["resource_scenarios"]) == 72
    assert len(report["timing_coverage"]) == 18
    assert report["grid_evaluation_counts"] == {
        "m3_easy_block_128": 128,
        "calibration_grid_3p": 18432,
        "formal_grid_7p": 43008,
    }
    assert all(row["selected"] is False for row in report["resource_scenarios"])
    assert report["selection"] is None
    assert all(
        value["safety_adjusted_total_core_seconds"] is None
        for value in report["strict_empirical_estimates"].values()
    )
    serialized = calibration.canonical_json(report).lower()
    assert "q_top" not in serialized
    assert "logical_label" not in serialized


def test_resource_and_science_entry_points_are_functionally_separate():
    resource_parameters = set(inspect.signature(calibration.build_resource_model).parameters)
    science_parameters = set(inspect.signature(calibration.build_discrepancy_audit).parameters)
    assert resource_parameters == {
        "config", "timing_evidence", "analysis_timing_evidence", "provenance"
    }
    assert {"report", "records", "provenance", "authority", "tolerance"} == science_parameters
    assert not ({"q_top", "labels", "ess", "valid", "passed"} & resource_parameters)


def test_resource_report_self_hash_and_json_round_trip():
    config = calibration.load_config(HERE / "resource_model_config.json")
    report = calibration.build_resource_model(
        config=config,
        timing_evidence=_fake_timing_evidence(),
        analysis_timing_evidence=_fake_analysis_evidence(),
        provenance={"test": "identity"},
    )
    round_trip = json.loads(calibration.canonical_json(report))
    claimed = round_trip.pop("report_sha256")
    assert claimed == calibration.sha256_json(round_trip)


def test_independent_package_auditor_accepts_a_self_consistent_fixture(tmp_path):
    config = calibration.load_config(HERE / "resource_model_config.json")
    source_identity = {
        "calibration_source_commit": "1" * 40,
        "calibration_source_tree_sha": "2" * 40,
        "git_object_format": "sha1",
        "worktree_clean": True,
        "bytecode_absent": True,
        "bound_implementation_sha256": dict(config["authority"]["implementation_files"]),
    }
    provenance = calibration.build_provenance(
        config=config,
        config_path=HERE / "resource_model_config.json",
        source_identity=source_identity,
    )
    resource = calibration.build_resource_model(
        config=config,
        timing_evidence=_fake_timing_evidence(),
        analysis_timing_evidence=_fake_analysis_evidence(),
        provenance=provenance,
    )
    catalog = [
        {
            "output_relpath": f"trajectories/{index:064x}.npz",
            "task_fingerprint": f"{index:064x}",
            "file_sha256": f"{index + 128:064x}",
        }
        for index in range(128)
    ]
    discrepancy_identity = {
        "audit_version": calibration.SCIENCE_AUDIT_VERSION,
        "contract_version": calibration.CALIBRATION_CONTRACT_VERSION,
        "status": config["authority"]["science_audit_status"],
        "authority": config["authority"]["science_audit_authority"],
        "formal_authorization": False,
        "production_authorization": False,
        "remote_launch_authorization": False,
        "provenance": provenance,
        "allow_pickle": False,
        "report_tolerance": config["report_tolerance"],
        "selected_raw_count": 128,
        "selected_raw_catalog": catalog,
        "selected_raw_catalog_sha256": calibration.sha256_json(catalog),
        "cells": [{"cell": "m6"}, {"cell": "m8"}],
        "headline_checks": {
            "m8_hp64_P_q_top": 0.9128439674802393,
            "m8_hp64_U_q_top": 0.913491773670944,
            "m8_hp64_combined_q_top": 0.9131680270339482,
            "m8_mam_combined_q_top": 0.9927278950353573,
            "m8_interpretation": "0.91317_VS_0.99273_IS_HP64_VS_MAM_NOT_HP64_P_VS_U",
            "m6_P_hp64_mam_absolute_delta": 0.016596369769588087,
            "m6_P_hp64_mam_paired_se": 0.0005425377386565906,
            "m6_P_hp64_mam_z": 30.590258680775513,
        },
        "limitations": [
            "NO_NEW_SAMPLING",
            "NO_NEW_CONVERGENCE_AUTHORITY",
            "HP64_AND_HP32_REMAIN_ONE_MECHANISM",
            "HISTORICAL_P_AND_EXACT_K0_U_IDENTITIES_PRESERVED",
        ],
    }
    discrepancy = {
        **discrepancy_identity,
        "audit_sha256": calibration.sha256_json(discrepancy_identity),
    }
    calibration.write_outputs(
        output_dir=tmp_path,
        resource=resource,
        discrepancy=discrepancy,
        timing_raw_audit={
            "status": "PASS", "allow_pickle": False,
            "cells": [{"cell": index} for index in range(5)],
        },
        config_path=HERE / "resource_model_config.json",
        config=config,
    )
    result = independent_audit.audit(
        tmp_path,
        config=config,
        config_path=HERE / "resource_model_config.json",
        source_identity=source_identity,
    )
    assert result["verified_status"] == "PACKAGE_CONTENT_PASS"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(repo, *args):
    return subprocess.run(
        ("git", "-C", str(repo), *args), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _make_bound_git_repo(tmp_path):
    repo = tmp_path / "repo"
    validation = repo / calibration.VALIDATION_RELATIVE_DIR
    validation.mkdir(parents=True)
    for index, name in enumerate(calibration.BOUND_IMPLEMENTATION_NAMES):
        (validation / name).write_text(f"bound-{index}\n", encoding="ascii")
    config = json.loads((HERE / "resource_model_config.json").read_text(encoding="ascii"))
    config["authority"]["implementation_files"] = {
        f"{calibration.VALIDATION_RELATIVE_DIR}/{name}": _sha256(validation / name)
        for name in calibration.BOUND_IMPLEMENTATION_NAMES
    }
    config_path = validation / "resource_model_config.json"
    config_path.write_text(calibration.canonical_json(config) + "\n", encoding="ascii")
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.invalid")
    _run_git(repo, "config", "user.name", "Validation Test")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-q", "-m", "fixture")
    return repo, config_path, calibration.load_config(config_path)


def test_source_gate_rejects_dirty_tracked_file(tmp_path):
    _, config_path, config = _make_bound_git_repo(tmp_path)
    calibration.verify_calibration_source(config_path, config)
    runner = config_path.parent / "run_resource_calibration.py"
    runner.write_text("tampered\n", encoding="ascii")
    with pytest.raises(calibration.CalibrationConflict, match="clean Git worktree"):
        calibration.verify_calibration_source(config_path, config)


def test_source_gate_rejects_untracked_file(tmp_path):
    repo, config_path, config = _make_bound_git_repo(tmp_path)
    (repo / "untracked.txt").write_text("untracked\n", encoding="ascii")
    with pytest.raises(calibration.CalibrationConflict, match="clean Git worktree"):
        calibration.verify_calibration_source(config_path, config)


def test_tampered_authority_is_rejected():
    config = json.loads((HERE / "resource_model_config.json").read_text(encoding="ascii"))
    config["authority"]["formal_authorization"] = True
    with pytest.raises(calibration.CalibrationConflict, match="authority field changed"):
        calibration._validate_authority(config)


def test_nonfinite_json_and_output_payload_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        calibration.canonical_json({"bad": float("nan")})
    with pytest.raises(calibration.CalibrationConflict, match="non-finite"):
        calibration._assert_finite_json({"bad": float("inf")})
    config = calibration.load_config(HERE / "resource_model_config.json")
    resource = calibration.build_resource_model(
        config=config,
        timing_evidence=_fake_timing_evidence(),
        analysis_timing_evidence=_fake_analysis_evidence(),
        provenance={"test": "identity"},
    )
    resource["resource_scenarios"][0]["generation_core_seconds"] = float("nan")
    with pytest.raises(calibration.CalibrationConflict, match="non-finite"):
        calibration.write_outputs(
            output_dir=tmp_path,
            resource=resource,
            discrepancy={"provenance": {"test": "identity"}},
            timing_raw_audit={"status": "PASS"},
            config_path=HERE / "resource_model_config.json",
            config=config,
        )
    assert not any(tmp_path.iterdir())


def test_one_shot_output_preflight_rejects_existing_artifact(tmp_path):
    (tmp_path / "discrepancy_audit.json").write_text("existing\n", encoding="ascii")
    with pytest.raises(calibration.CalibrationConflict, match="not empty"):
        calibration.require_expected_outputs_absent(tmp_path)


def test_independent_audit_never_replaces_existing_output(tmp_path):
    path = tmp_path / independent_audit.INDEPENDENT_OUTPUT_NAME
    path.write_text("immutable\n", encoding="ascii")
    with pytest.raises(independent_audit.AuditFailure, match="already exists"):
        independent_audit.exclusive_json(path, {"status": "would_overwrite"})
    assert path.read_text(encoding="ascii") == "immutable\n"
