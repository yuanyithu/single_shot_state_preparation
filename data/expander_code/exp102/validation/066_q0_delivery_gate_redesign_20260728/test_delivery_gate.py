import ast
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("q0_delivery_gate_runner", "run_delivery_gate.py")
auditor = load_module("q0_delivery_gate_auditor", "audit_delivery_gate.py")


def load_config():
    return json.loads((ROOT / "delivery_gate_config.json").read_text(encoding="ascii"))


def bound_report(config=None):
    return runner.verify_bound_validation_062(config or load_config())


def scenarios(config=None):
    config = config or load_config()
    return runner.build_scenarios(config, bound_report(config))


def test_bound_062_report_and_independent_audit_are_hash_and_status_bound():
    config = load_config()
    report = bound_report(config)
    spec = config["bound_validation_062"]
    assert report["report_sha256"] == spec["report"]["report_sha256"]
    assert report["status"] == "CHARACTER_GATE_REDESIGN_REQUIRED"
    audit = runner.load_json_strict(runner._exp102_path(spec["audit"]["path"]))
    assert audit["status"] == (
        "INDEPENDENT_AUDIT_PASS_CHARACTER_GATE_REDESIGN_REQUIRED"
    )
    assert audit["report_sha256"] == report["report_sha256"]


def test_inverse_walsh_reconstructs_all_twelve_complete_label_distributions():
    config = load_config()
    report = bound_report(config)
    row_filter = config["bound_validation_062"]["source_row_filter"]
    rows = [
        row for row in report["selection_rows"]
        if all(row.get(name) == value for name, value in row_filter.items())
    ]
    assert len(rows) == 12
    for row in rows:
        labels, probabilities, k = runner.inverse_walsh(row["base_logical_means"])
        assert labels.dtype == np.uint64
        assert np.all(probabilities >= 0.0)
        assert probabilities.sum() == pytest.approx(1.0, abs=2e-13)
        assert runner.normalized_q_top(probabilities, k) == pytest.approx(
            row["true_q_top_left"], abs=2e-13,
        )
        recovered = runner.character_means_from_distribution(
            labels, probabilities, np.arange(1, 1 << k, dtype=np.uint64),
        )
        assert np.allclose(recovered, row["base_logical_means"], atol=2e-13)


def test_scenario_registry_covers_profiles_qtop_d2_and_known_blind_controls():
    rows = scenarios()
    assert len(rows) == 134
    assert {row["k"] for row in rows} == {1, 4, 9, 16, 36, 64}
    classifications = {row["classification"] for row in rows}
    assert classifications == {
        "null", "good_q_top", "boundary_q_top", "bad_q_top", "d2_null",
        "good_d2", "boundary_d2", "bad_d2", "known_blind",
    }
    assert sum(row["classification"] == "known_blind" for row in rows) == 2
    assert runner.fail_hypotheses_per_stage_point(rows) == 139
    assert runner.fail_hypothesis_count(load_config(), rows) == 1390
    metadata = [runner.scenario_public_metadata(row, load_config()) for row in rows]
    assert auditor.fail_hypotheses_per_stage_point(metadata) == 139
    assert auditor.fail_hypothesis_count(load_config(), metadata) == 1390
    assert runner.fail_adjusted_confidence(load_config(), rows) == pytest.approx(
        1.0 - 0.05 / 1390.0,
    )


@pytest.mark.parametrize("k", (9, 16, 36, 64))
@pytest.mark.parametrize("base", (0.05, 0.15, 0.55, 0.90))
def test_sparse_qtop_profiles_and_exact_deltas_are_legal(k, base):
    config = load_config()
    labels, probabilities = runner.sparse_profile_distribution(k, base, 256)
    assert labels.dtype == np.uint64
    assert len(set(map(int, labels))) == 256
    assert max(map(int, labels)) < (1 << k)
    assert runner.normalized_q_top(probabilities, k) == pytest.approx(base, abs=2e-13)
    for delta in config["sparse_label_stress"]["q_top_deltas"]:
        _, shifted = runner.shift_distribution_q_top(labels, probabilities, k, delta)
        assert abs(
            runner.normalized_q_top(shifted, k)
            - runner.normalized_q_top(probabilities, k)
        ) == pytest.approx(delta, abs=2e-13)


@pytest.mark.parametrize("k", (9, 16, 36, 64))
@pytest.mark.parametrize("distance", (0.0, 0.02, 0.04, 0.06))
def test_same_purity_controlled_d2_levels_are_exact(k, distance):
    left, p_left, right, p_right = runner.controlled_same_purity_d2_pair(
        k, distance,
    )
    truth = runner.true_distribution_metrics(left, p_left, right, p_right, k)
    assert truth["q_top_delta_abs"] == pytest.approx(0.0, abs=2e-13)
    assert truth["d2_norm"] == pytest.approx(distance, abs=2e-13)


def test_k64_uses_uint64_bit63_max_label_and_stable_normalization():
    config = load_config()
    masks = runner.diagnostic_masks(64, config)
    assert masks.dtype == np.uint64
    assert np.uint64(1 << 63) in masks
    labels = np.asarray(
        [np.uint64(0), np.uint64(1 << 63), np.uint64((1 << 64) - 1)],
        dtype=np.uint64,
    )
    assert runner.labels_hex(labels) == [
        "0x0000000000000000", "0x8000000000000000", "0xffffffffffffffff",
    ]
    size = 1 << 64
    for collision in (0.0, 0.5, 1.0):
        expected = collision + (collision - 1.0) / (size - 1)
        assert runner.normalize_collision(collision, 64) == expected
    for raw_d2 in (-0.25, 0.0, 0.06):
        expected = raw_d2 + raw_d2 / (size - 1)
        assert runner.normalize_d2(raw_d2, 64) == expected


def manual_collision(histograms):
    values = []
    for left in range(histograms.shape[0]):
        for right in range(histograms.shape[0]):
            if left != right:
                values.append(float(np.dot(histograms[left], histograms[right])))
    return sum(values) / len(values)


def test_full_label_collision_estimators_match_direct_pair_loops():
    labels = np.asarray([0, 1, 2], dtype=np.uint64)
    left = np.asarray([[[4, 0, 0], [2, 2, 0], [0, 4, 0]]], dtype=np.int64)
    right = np.asarray([[[0, 0, 4], [0, 2, 2], [0, 0, 4]]], dtype=np.int64)
    masks = np.asarray([1, 2, 3], dtype=np.uint64)
    result = runner.estimate_trial_metrics(labels, left, right, 2, masks)
    p_left = left[0] / 4.0
    p_right = right[0] / 4.0
    c_left = manual_collision(p_left)
    c_right = manual_collision(p_right)
    cross = np.mean([np.dot(a, b) for a in p_left for b in p_right])
    assert result["q_top_left"][0] == pytest.approx(
        runner.normalize_collision(c_left, 2),
    )
    assert result["q_top_right"][0] == pytest.approx(
        runner.normalize_collision(c_right, 2),
    )
    assert result["d2_norm"][0] == pytest.approx(
        runner.normalize_d2(c_left + c_right - 2.0 * cross, 2),
    )


def independent_groupwise_se(left, right):
    total = np.zeros(left.shape[0])
    for values in (left, right):
        centered = values - values.mean(axis=1, keepdims=True)
        total += (values.shape[1] - 1) / values.shape[1] * np.sum(centered ** 2, axis=1)
    return np.sqrt(total)


def test_four_delete_one_arrays_reproduce_groupwise_jackknife_se():
    labels = np.asarray([0, 1], dtype=np.uint64)
    left = np.asarray([
        [[8, 0], [6, 2], [4, 4], [2, 6]],
        [[7, 1], [5, 3], [3, 5], [1, 7]],
    ], dtype=np.int64)
    right = left[:, ::-1].copy()
    result = runner.estimate_trial_metrics(
        labels, left, right, 1, np.asarray([1], dtype=np.uint64),
    )
    assert np.allclose(
        result["q_top_delta_se"], independent_groupwise_se(
            result["delete_one_q_top_delta_left"],
            result["delete_one_q_top_delta_right"],
        ),
    )
    assert np.allclose(
        result["d2_se"], independent_groupwise_se(
            result["delete_one_d2_left"], result["delete_one_d2_right"],
        ),
    )


def test_runner_and_auditor_replay_receipts_are_identical_and_tamper_evident():
    config = load_config()
    scenario = next(row for row in scenarios(config) if row["classification"] == "null")
    metrics, left_seed, right_seed, receipt = runner.simulate_metrics(
        scenario, config, "receipt-test", "selection", 4, 32, 3,
    )
    metadata = runner.scenario_public_metadata(scenario, config)
    replayed, audited_receipt = auditor.replay_trial_metrics(
        metadata, config, 3, 4, 32, left_seed, right_seed,
    )
    assert receipt == audited_receipt
    assert set(receipt) == {
        "all_trial_metrics_sha256", "histogram_counts_sha256", "schema",
    }
    assert all(np.array_equal(metrics[name], replayed[name]) for name in metrics)
    changed = metrics.copy()
    changed["d2_norm"] = changed["d2_norm"].copy()
    changed["d2_norm"][0] += 1e-15
    labels = np.asarray([0], dtype=np.uint64)
    left = np.ones((1, 4, 1), dtype=np.int64)
    right = left.copy()
    baseline = runner.replay_receipt(labels, left, right, metrics)
    assert runner.replay_receipt(
        labels, left, right, changed,
    )["all_trial_metrics_sha256"] != baseline["all_trial_metrics_sha256"]
    changed_left = left.copy()
    changed_left[0, 0, 0] += 1
    assert runner.replay_receipt(
        labels, changed_left, right, metrics,
    )["histogram_counts_sha256"] != baseline["histogram_counts_sha256"]


def test_compact_worst_case_report_probe_stays_under_ten_mib_without_arrays():
    config = load_config()
    small = copy.deepcopy(config)
    small["replications"]["calibration_trials"] = 2
    small["replications"]["selection_trials"] = 2
    all_scenarios = scenarios(config)
    scenario = next(
        row for row in all_scenarios if row["classification"] == "null"
    )
    calibration = runner.calibration_point(
        small, "size-probe", [scenario], 8, 2048,
    )
    calibration["scenario_quantiles"] *= 132
    row = runner.compact_evaluation_row(runner.evaluate_row(
        scenario, small, "size-probe", "selection", 8, 2048, 3.0,
        all_scenarios,
    ))
    payload = {
        "calibration_points": [calibration] * 5,
        "confirmation_rows": [row] * 134,
        "scenario_registry": [
            runner.scenario_public_metadata(value, config)
            for value in all_scenarios
        ],
        "selection_rows": [row] * (5 * 134),
    }
    encoded = runner.canonical(payload).encode("ascii")
    assert len(encoded) < config["report_contract"]["maximum_bytes"]
    assert b"raw_trial_metrics" not in encoded
    assert b"delete_one" not in encoded


@pytest.mark.parametrize("control_suffix", ("common_freeze", "distinct_freeze_same_set"))
def test_known_blind_controls_pass_distribution_gate_without_mixing_authority(
        control_suffix):
    config = load_config()
    scenario = next(row for row in scenarios(config) if row["id"].endswith(control_suffix))
    metrics, _, _, _ = runner.simulate_metrics(
        scenario, config, "test-config", "selection", 8, 32, 4,
    )
    decisions = runner.decision_arrays(metrics, 3.0, config)
    assert np.all(decisions["pass"])
    assert scenario["truth"]["d2_norm"] > 0.06
    if control_suffix == "distinct_freeze_same_set":
        assert np.all(metrics["d2_norm"] < 0.0)
        assert np.all(metrics["d2_se"] == 0.0)
    assert scenario["classification"] == "known_blind"


def test_qtop_and_d2_have_separate_three_state_rules_and_character_is_diagnostic():
    config = load_config()
    metrics = {
        "q_top_delta_abs": np.asarray([0.01, 0.05, 0.08]),
        "q_top_delta_se": np.asarray([0.001, 0.02, 0.001]),
        "d2_norm": np.asarray([0.01, 0.01, 0.01]),
        "d2_se": np.asarray([0.001, 0.001, 0.001]),
        "character_max_abs_delta_diagnostic": np.asarray([0.0, 10.0, 1e6]),
    }
    decision = runner.decision_arrays(metrics, 3.0, config)
    assert decision["pass"].tolist() == [True, False, False]
    assert decision["inconclusive"].tolist() == [False, True, False]
    assert decision["fail"].tolist() == [False, False, True]
    changed = dict(metrics, character_max_abs_delta_diagnostic=np.zeros(3))
    for name, values in runner.decision_arrays(changed, 3.0, config).items():
        assert np.array_equal(values, decision[name])


def test_wilson_lower_upper_and_rate_taxonomy_are_independent():
    lower = runner.wilson_lower(95, 100, 0.95)
    upper = runner.wilson_upper(95, 100, 0.95)
    assert lower < 0.95 < upper
    assert lower == pytest.approx(auditor.wilson_lower(95, 100, 0.95))
    assert upper == pytest.approx(auditor.wilson_upper(95, 100, 0.95))
    assert runner.rate_verdict(0.96, 0.99, 0.95) == "PASS"
    assert runner.rate_verdict(0.90, 0.94, 0.95) == "FAIL"
    assert runner.rate_verdict(0.90, 0.97, 0.95) == "INCONCLUSIVE"


def test_bonferroni_fail_evidence_changes_marginal_fail_to_inconclusive():
    config = load_config()
    rows = [_fake_summary_row(row["classification"]) for row in scenarios(config)]
    marginal = rows[0]
    adjusted = runner.fail_adjusted_confidence(config, rows)
    marginal["candidate_pass_wilson_lower"] = runner.wilson_lower(271, 300, 0.95)
    marginal["candidate_pass_wilson_upper"] = runner.wilson_upper(271, 300, 0.95)
    marginal["candidate_pass_fail_adjusted_wilson_lower"] = runner.wilson_lower(
        271, 300, adjusted,
    )
    marginal["candidate_pass_fail_adjusted_wilson_upper"] = runner.wilson_upper(
        271, 300, adjusted,
    )
    assert marginal["candidate_pass_wilson_upper"] < 0.95
    assert marginal["candidate_pass_fail_adjusted_wilson_upper"] >= 0.95
    assert runner.summarize_point(
        rows, config, 8, 2048, 3.0, "selection",
    )["decision"] == "INCONCLUSIVE"

    severe = copy.deepcopy(rows)
    severe[0]["candidate_pass_fail_adjusted_wilson_upper"] = runner.wilson_upper(
        0, 300, adjusted,
    )
    assert runner.summarize_point(
        severe, config, 8, 2048, 3.0, "selection",
    )["decision"] == "FAIL"


def test_invalid_multiplier_and_confirmation_fail_terminal_taxonomy():
    invalid = runner.invalid_calibration_selection_point(8, 2048)
    assert invalid["decision"] == "INCONCLUSIVE"
    assert runner.terminal_status_for([invalid], None, None) == (
        "DELIVERY_GATE_CALIBRATION_INCONCLUSIVE"
    )
    selected = {
        "decision": "PASS", "eligible": True, "trajectory_count": 8,
        "draws_per_trajectory": 2048, "multiplier": 3.0,
    }
    confirmation = dict(selected, decision="FAIL", eligible=False)
    assert runner.terminal_status_for([selected], selected, confirmation) == (
        "SELECTED_POINT_CONFIRMATION_FAILED_REDESIGN_REQUIRED"
    )


def _fake_summary_row(classification, trials=300):
    row = {
        "candidate_fail_wilson_lower": 0.99 if classification in {"bad_q_top", "bad_d2"} else 0.0,
        "candidate_fail_wilson_upper": 1.0 if classification in {"bad_q_top", "bad_d2"} else 0.01,
        "candidate_pass_wilson_lower": 0.99 if classification not in {"bad_q_top", "bad_d2"} else 0.0,
        "candidate_pass_wilson_upper": 1.0 if classification not in {"bad_q_top", "bad_d2"} else 0.01,
        "classification": classification,
        "coverage_applicable": classification != "known_blind",
        "expected_interpretation": (
            "EXPECTED_KNOWN_BLIND" if classification == "known_blind"
            else "IID_DELIVERY_GATE_CALIBRATION"
        ),
        "_raw_trial_metrics": {
            "d2_norm": np.zeros(trials),
            "d2_se": np.zeros(trials),
            "q_top_delta_se": np.zeros(trials),
            "q_top_delta_signed": np.zeros(trials),
        },
        "true_d2_norm": 0.0,
        "true_q_top_delta_signed": 0.0,
    }
    for scalar, bad_class in (("q_top", "bad_q_top"), ("d2", "bad_d2")):
        is_bad = classification == bad_class
        row[f"candidate_{scalar}_fail_wilson_lower"] = 0.99 if is_bad else 0.0
        row[f"candidate_{scalar}_fail_wilson_upper"] = 1.0 if is_bad else 0.01
        row[f"candidate_{scalar}_pass_wilson_lower"] = 0.0 if is_bad else 0.99
        row[f"candidate_{scalar}_pass_wilson_upper"] = 0.01 if is_bad else 1.0
    for prefix in ("candidate_pass", "candidate_q_top_fail", "candidate_q_top_pass",
                   "candidate_d2_fail", "candidate_d2_pass"):
        row[f"{prefix}_fail_adjusted_wilson_lower"] = row[
            f"{prefix}_wilson_lower"
        ]
        row[f"{prefix}_fail_adjusted_wilson_upper"] = row[
            f"{prefix}_wilson_upper"
        ]
    return row


def test_point_summary_distinguishes_pass_fail_and_inconclusive():
    config = load_config()
    classifications = [row["classification"] for row in scenarios(config)]
    rows = [_fake_summary_row(name) for name in classifications]
    passed = runner.summarize_point(rows, config, 8, 2048, 3.0, "selection")
    assert passed["decision"] == "PASS"
    assert runner.canonical(passed) == auditor.canonical(
        auditor.summarize(rows, config, 8, 2048, 3.0, "selection")
    )
    failed_rows = copy.deepcopy(rows)
    failed_rows[0]["candidate_pass_wilson_lower"] = 0.90
    failed_rows[0]["candidate_pass_fail_adjusted_wilson_upper"] = 0.94
    failed = runner.summarize_point(
        failed_rows, config, 8, 2048, 3.0, "selection",
    )
    assert failed["decision"] == "FAIL"
    uncertain_rows = copy.deepcopy(rows)
    uncertain_rows[0]["candidate_pass_wilson_lower"] = 0.90
    uncertain_rows[0]["candidate_pass_fail_adjusted_wilson_upper"] = 0.97
    uncertain = runner.summarize_point(
        uncertain_rows, config, 8, 2048, 3.0, "selection",
    )
    assert uncertain["decision"] == "INCONCLUSIVE"

    wrong_scalar = copy.deepcopy(rows)
    bad_q = next(row for row in wrong_scalar if row["classification"] == "bad_q_top")
    bad_q["candidate_q_top_fail_wilson_lower"] = 0.0
    bad_q["candidate_q_top_fail_fail_adjusted_wilson_upper"] = 0.01
    bad_q["candidate_q_top_pass_wilson_lower"] = 0.99
    bad_q["candidate_q_top_pass_wilson_upper"] = 1.0
    assert runner.summarize_point(
        wrong_scalar, config, 8, 2048, 3.0, "selection",
    )["decision"] == "FAIL"


def test_outer_calibration_maximum_is_scenario_simultaneous_not_pooled():
    config = load_config()
    small = copy.deepcopy(config)
    small["replications"]["calibration_trials"] = 5
    subset = [
        next(row for row in scenarios(config) if row["classification"] == "null"),
        next(row for row in scenarios(config) if row["classification"] == "bad_d2"),
    ]
    point = runner.calibration_point(small, "config", subset, 4, 64)
    registry = [runner.scenario_public_metadata(row, small) for row in subset]
    auditor.verify_calibration_point(point, small, "config", registry)
    tampered = copy.deepcopy(point)
    tampered["scenario_quantiles"][0]["replay_receipt"][
        "all_trial_metrics_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="replay receipt changed"):
        auditor.verify_calibration_point(tampered, small, "config", registry)


def test_calibration_selection_confirmation_seed_namespaces_are_disjoint():
    config = load_config()
    values = {
        runner._scenario_seeds(config, "config", stage, 8, 2048, "scenario")
        for stage in ("calibration", "selection", "confirmation")
    }
    assert len(values) == 3
    assert all(left != right for left, right in values)


def test_independent_auditor_recomputes_raw_delete_one_and_decisions():
    config = load_config()
    small = copy.deepcopy(config)
    small["replications"]["selection_trials"] = 4
    scenario = next(row for row in scenarios(config) if row["classification"] == "good_q_top")
    row = runner.evaluate_row(
        scenario, small, "config", "selection", 4, 64, 2.0, scenarios(config),
    )
    metadata = runner.scenario_public_metadata(scenario, small)
    auditor.verify_row(
        row, small, "config", metadata, "selection", 4, 64, 2.0,
        [runner.scenario_public_metadata(value, small) for value in scenarios(config)],
    )
    tampered = copy.deepcopy(row)
    tampered["replay_receipt"]["histogram_counts_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="seed replay receipt changed"):
        auditor.verify_row(
            tampered, small, "config", metadata, "selection", 4, 64, 2.0,
            [runner.scenario_public_metadata(value, small) for value in scenarios(config)],
        )

    alternate = runner.evaluate_row(
        scenario, small, "coordinated-other-config", "selection", 4, 64, 2.0,
        scenarios(config),
    )
    coordinated = copy.deepcopy(row)
    coordinated["replay_receipt"] = alternate["replay_receipt"]
    for name, value in alternate.items():
        if name.startswith("candidate_") or name.startswith("joint_coverage_"):
            coordinated[name] = value
    with pytest.raises(RuntimeError, match="seed replay receipt changed"):
        auditor.verify_row(
            coordinated, small, "config", metadata, "selection", 4, 64, 2.0,
            [runner.scenario_public_metadata(value, small) for value in scenarios(config)],
        )


def test_runner_and_auditor_reconstruct_identical_registry_independently():
    config = load_config()
    report = bound_report(config)
    observed = [
        runner.scenario_public_metadata(row, config)
        for row in runner.build_scenarios(config, report)
    ]
    expected = auditor.expected_scenario_registry(config, report)
    assert runner.canonical(observed) == auditor.canonical(expected)


def test_config_self_hash_and_all_source_artifact_hashes_are_bound():
    config = load_config()
    runner.validate_config(config)
    assert set(config["source_artifacts"]) == {
        "auditor", "readme", "red_team", "runner", "tests",
    }
    for spec in config["source_artifacts"].values():
        path = runner._exp102_path(spec["path"])
        assert path.is_file()
        assert runner.sha256_file(path) == spec["sha256"]


def test_dirty_worktree_bytecode_and_existing_output_are_fail_closed(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "_git", lambda args, text=True: "?? rogue.txt\n"
        if args[0] == "status" else "",
    )
    with pytest.raises(RuntimeError, match="completely clean worktree"):
        runner.require_completely_clean_worktree()
    clean = tmp_path / "clean"
    clean.mkdir()
    runner.reject_validation_bytecode(clean)
    bytecode = clean / "__pycache__"
    bytecode.mkdir()
    (bytecode / "module.pyc").write_bytes(b"not bytecode")
    with pytest.raises(RuntimeError, match="contains bytecode"):
        runner.reject_validation_bytecode(clean)


@pytest.mark.parametrize("module", (runner, auditor))
def test_frozen_leaf_and_parent_escape_symlinks_are_rejected(
        module, monkeypatch, tmp_path):
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    target = frozen_root / "target.json"
    target.write_text("{}\n", encoding="ascii")
    leaf = frozen_root / "leaf.json"
    leaf.symlink_to(target)
    monkeypatch.setattr(module, "EXP102_ROOT", frozen_root)
    with pytest.raises(RuntimeError, match="may not be a symlink"):
        module._exp102_path("leaf.json")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_text("pass\n", encoding="ascii")
    (frozen_root / "parent_link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escapes exp102 root"):
        module._exp102_path("parent_link/source.py")


def test_strict_json_self_hash_and_nonfinite_rejection(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value":NaN}\n', encoding="ascii")
    with pytest.raises(ValueError, match="non-finite"):
        runner.load_json_strict(invalid)
    payload = {"field": 1}
    payload["self_sha"] = runner.sha256_bytes(
        runner.canonical(payload).encode("ascii")
    )
    runner.verify_self_hash(payload, "self_sha")
    payload["field"] = 2
    with pytest.raises(RuntimeError, match="self hash changed"):
        runner.verify_self_hash(payload, "self_sha")


def test_auditor_is_independent_and_all_sources_are_ascii_python():
    audit_source = (ROOT / "audit_delivery_gate.py").read_text(encoding="ascii")
    assert "run_delivery_gate" not in audit_source
    for filename in (
            "run_delivery_gate.py", "audit_delivery_gate.py", "test_delivery_gate.py"):
        source = (ROOT / filename).read_text(encoding="ascii")
        ast.parse(source, filename=filename)
        assert all(ord(character) < 128 for character in source)
