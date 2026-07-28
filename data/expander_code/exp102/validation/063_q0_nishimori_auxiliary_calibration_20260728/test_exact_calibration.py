import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[4]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("q0_nishimori_exact_calibration", "run_exact_calibration.py")
oracle = load_module("q0_nishimori_independent_oracle", "independent_oracle.py")
validator = load_module("q0_nishimori_future_raw_validator", "validate_future_raw.py")
auditor = load_module("q0_nishimori_report_auditor", "audit_report.py")


def load_config():
    return json.loads((ROOT / "nishimori_config.json").read_text(encoding="ascii"))


def load_schema():
    return json.loads((ROOT / "nishimori_raw_schema.v1.json").read_text(encoding="ascii"))


def assert_nested_close(left, right):
    if isinstance(right, dict):
        assert set(left) == set(right)
        for key in right:
            assert_nested_close(left[key], right[key])
    elif isinstance(right, list):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            assert_nested_close(left_value, right_value)
    elif isinstance(right, float):
        assert math.isclose(float(left), right, rel_tol=2e-13, abs_tol=2e-13)
    else:
        assert left == right


def runner_ensemble(model_spec, p):
    model, frame = runner.build_model(np.asarray(model_spec["H"], dtype=np.uint8))
    states = runner.all_binary_states(model.num_qubits)
    syndromes = runner.packed_syndromes(model, states)
    labels = runner.integer_labels(frame, states)
    golden, posterior, y_probabilities = runner.runner_golden(
        model, frame, states, syndromes, labels, p, model_spec["id"]
    )
    return model, golden, posterior, y_probabilities


@pytest.mark.parametrize("model_index", [0, 1])
@pytest.mark.parametrize("p", [0.04, 0.1, 0.25])
def test_independent_physics_v2_oracle_matches_every_runner_state(model_index, p):
    model_spec = load_config()["exact_models"][model_index]
    model, runner_golden, runner_posterior, runner_y = runner_ensemble(model_spec, p)
    oracle_golden, oracle_posterior, oracle_y = oracle.enumerate_physics_v2(
        model_spec["H"], p, model_spec["id"]
    )
    assert model.num_qubits == model_spec["expected_n"]
    assert model.k == model_spec["expected_k"]
    assert runner_golden["state_table_sha256"] == oracle_golden["state_table_sha256"]
    assert runner_golden["syndrome_keys_hex"] == oracle_golden["syndrome_keys_hex"]
    assert runner_golden["support_counts"] == oracle_golden["support_counts"]
    assert runner_golden["model_fingerprint"] == oracle_golden["model_fingerprint"]
    assert runner_golden["frame_fingerprint"] == oracle_golden["frame_fingerprint"]
    assert runner_golden["section_fingerprint"] == oracle_golden["section_fingerprint"]
    assert runner_golden["hard_coset_support_verified"] is True
    assert oracle_golden["hard_coset_support_verified"] is True
    assert runner_golden["max_log_b_weight_ratio_error"] <= 2e-13
    assert oracle_golden["max_log_b_weight_ratio_error"] <= 2e-13
    assert np.allclose(runner_y, oracle_y, atol=2e-13, rtol=0.0)
    assert np.allclose(runner_posterior, oracle_posterior, atol=2e-13, rtol=0.0)


@pytest.mark.parametrize("model_index", [0, 1])
@pytest.mark.parametrize("p", [0.04, 0.1, 0.25])
def test_independent_oracle_recomputes_all_exact_control_metrics(model_index, p):
    config = load_config()
    model_spec = config["exact_models"][model_index]
    _, _, target, y_probabilities = runner_ensemble(model_spec, p)
    wrong_p = float(config["wrong_temperature_map"][str(p)])
    _, _, wrong, _ = runner_ensemble(model_spec, wrong_p)
    runner_controls = runner.control_candidates(target, wrong)
    oracle_controls = oracle.oracle_controls(target, wrong)
    assert set(runner_controls) == set(config["controls"]["exact"]) == set(oracle_controls)
    for name in sorted(runner_controls):
        assert np.array_equal(runner_controls[name], oracle_controls[name])
        runner_metrics = runner.candidate_metrics(target, runner_controls[name], y_probabilities)
        oracle_metrics = oracle.oracle_candidate_metrics(target, oracle_controls[name], y_probabilities)
        assert_nested_close(runner_metrics, oracle_metrics)


def test_power_reports_omnibus_basis_and_nonbasis_without_diluting_sparse_control():
    config = load_config()
    reduced = copy.deepcopy(config)
    reduced["power"]["ensemble_sizes"] = [128]
    reduced["power"]["replications"] = 32
    model_spec = config["exact_models"][0]
    _, _, target, y_probabilities = runner_ensemble(model_spec, 0.1)
    candidate = runner.control_candidates(target, target)["label_permutation"]
    runner_rows = runner.power_curve(target, candidate, y_probabilities, reduced, (model_spec["id"], 0.1, "label_permutation"))
    oracle_rows = oracle.oracle_power_curve(target, candidate, y_probabilities, reduced, (model_spec["id"], 0.1, "label_permutation"))
    assert_nested_close(runner_rows, oracle_rows)
    assert set(runner_rows[0]["statistics"]) == {"omnibus", "basis_max", "nonbasis_max"}
    assert all(runner_rows[0]["statistics"][name]["applicable"] for name in runner_rows[0]["statistics"])

    small_spec = config["exact_models"][1]
    _, _, small_target, small_y = runner_ensemble(small_spec, 0.1)
    small_rows = runner.power_curve(
        small_target, small_target, small_y, reduced,
        (small_spec["id"], 0.1, "correct_posterior"),
    )
    assert small_rows[0]["statistics"]["nonbasis_max"] == {
        "applicable": False,
        "diagnostic_equivalence_pass_rate": None,
        "equality_rejection_rate": None,
        "rate_mcse_upper_bound": None,
    }


def test_chain_level_blind_controls_use_the_same_character_cross_product_estimator():
    actual = runner.chain_level_controls()
    expected = oracle.oracle_chain_controls()
    assert_nested_close(actual, expected)
    common = actual["common_planted_freeze"]
    assert common["collision_mass_u_statistic"] == common["planted_hit"] == 1.0
    assert np.max(np.abs(common["per_character_identity_difference"])) == 0.0
    four = actual["four_distinct_label_freeze"]
    assert four["collision_mass_u_statistic"] == four["planted_hit"] == 0.0
    assert abs(four["scalar_identity_difference"]) <= 1e-13
    assert abs(four["group_exact_metrics"]["omnibus"]["exact_effect"]) <= 1e-13
    assert four["group_exact_metrics"]["basis_max"]["max_abs_exact_effect"] > 0.005
    assert four["group_exact_metrics"]["nonbasis_max"]["max_abs_exact_effect"] > 0.005
    two = actual["two_label_equal_moment_counterexample"]
    assert two["collision_mass"] == two["planted_hit"] == 0.5
    assert two["target_q_top"] == 0.64 and two["candidate_q_top"] == 0.0


def test_full_optimistic_power_catalog_failure_is_scientific_insufficiency():
    payload = runner.build_calibration_payload(load_config(), include_power=True)
    gate = payload["calibration_gate"]
    assert gate["passed"] is False
    assert len(gate["failures"]) == 14
    assert all(failure.startswith("equivalence gate failed:") for failure in gate["failures"])
    assert all("correct_posterior" in failure for failure in gate["failures"])
    assert not any("detection gate failed:" in failure for failure in gate["failures"])
    assert gate["power_is_optimistic_no_sampler_noise"] is True
    assert gate["universal_q_top_bias_bound"] is None
    report = runner.build_report_core(load_config(), {"source_commit": "0" * 40}, payload)
    assert report["status"] == runner.INSUFFICIENT_STATUS
    assert report["authority"] == runner.EXPECTED_AUTHORITY
    assert report["authority"]["maximum_status"] == runner.CALIBRATED_STATUS


def test_only_a_passing_gate_can_receive_the_calibrated_terminal_status():
    passing_payload = {
        "calibration_gate": {
            "failures": [],
            "passed": True,
            "power_is_optimistic_no_sampler_noise": True,
            "universal_q_top_bias_bound": None,
        }
    }
    report = runner.build_report_core(load_config(), {"source_commit": "0" * 40}, passing_payload)
    assert report["status"] == runner.CALIBRATED_STATUS
    with pytest.raises(RuntimeError):
        runner.terminal_status({"failures": ["scientific failure"], "passed": True})


def fake_manifest():
    core = {
        "code_id": "m03_c00",
        "code_sha256": "1" * 64,
        "generation_contract": "fresh_iid_bernoulli_truth.v1",
        "generation_config_sha256": "2" * 64,
        "p": 0.1,
        "planned_count": 2,
        "planned_disorders": [
            {"disorder_index": 0, "disorder_seed_identity": "fresh-0"},
            {"disorder_index": 1, "disorder_seed_identity": "fresh-1"},
        ],
        "registry_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "source_tree_sha256": "5" * 64,
        "version": validator.MANIFEST_VERSION,
    }
    return validator.build_planned_manifest(core)


def fake_raw(manifest, disorder_index, *, gate_pass=True):
    k = 64
    masks = np.asarray([*(1 << index for index in range(k)), 3], dtype=np.uint64)
    character_count = masks.size
    chain_count = 4
    failures = [] if gate_pass else ["B_CHARACTER_RHAT"]
    arrays = {
        "basis_character_mask": np.asarray([(int(value) & (int(value) - 1)) == 0 for value in masks], dtype=np.bool_),
        "character_masks_uint64": masks,
        "collision_mass_u_statistic": np.asarray(0.5, dtype=np.float64),
        "finite_population_batch_means": np.asarray([0.0, 0.1], dtype=np.float64),
        "m2_debiased_per_character": np.zeros(character_count, dtype=np.float64),
        "per_disorder_sampler_gate_failures_json": np.asarray(json.dumps(failures), dtype="<U4096"),
        "per_disorder_sampler_gate_pass": np.asarray(gate_pass, dtype=np.bool_),
        "planted_cross_moment_per_character": np.zeros(character_count, dtype=np.float64),
        "posterior_mass_on_planted_class_estimate": np.asarray(0.5, dtype=np.float64),
        "scoring_chain_character_means": np.zeros((chain_count, character_count), dtype=np.float64),
        "scoring_chain_seed_ids": np.arange(chain_count, dtype=np.uint64),
        "trajectory_jackknife_values": np.zeros((chain_count, character_count), dtype=np.float64),
        "truth_blind_scoring_chain_mask": np.ones(chain_count, dtype=np.bool_),
        "truth_character_signs": np.ones(character_count, dtype=np.int8),
    }
    identity = {
        "H_check_role": "H_Z",
        "H_check_sha256": "6" * 64,
        "analyzer_sha256": "7" * 64,
        "character_count": int(character_count),
        "character_sha256": "8" * 64,
        "character_weighting_rule": validator.WEIGHTING_RULE,
        "code_id": manifest["code_id"],
        "code_sha256": manifest["code_sha256"],
        "config_sha256": "9" * 64,
        "disorder_index": disorder_index,
        "disorder_seed_identity": manifest["planned_disorders"][disorder_index]["disorder_seed_identity"],
        "finite_population_batch_count": 2,
        "generation_config_sha256": manifest["generation_config_sha256"],
        "k": k,
        "logical_frame_fingerprint": "a" * 64,
        "model_fingerprint": "b" * 64,
        "nonbasis_population_size": (1 << k) - 1 - k,
        "nonbasis_sampled_count": 1,
        "p": manifest["p"],
        "physics_contract_version": "exp101.physics.v2",
        "planned_manifest_sha256": manifest["manifest_sha256"],
        "posterior_ensemble": "true_posterior",
        "q": 0.0,
        "raw_self_sha256": "0" * 64,
        "raw_version": validator.RAW_VERSION,
        "registry_sha256": manifest["registry_sha256"],
        "sampler_config_sha256": "c" * 64,
        "schedule_sha256": "d" * 64,
        "schema_sha256": "e" * 64,
        "scoring_chain_initial_state_sha256": [f"{index + 1:064x}" for index in range(chain_count)],
        "scoring_chain_initialization_families": ["U", "MAP", "S0", "S1"],
        "scoring_chain_seed_identities": [f"scoring-{index}" for index in range(chain_count)],
        "section_fingerprint": "f" * 64,
        "sector": "x_error",
        "source_archive_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "source_tree_sha256": "3" * 64,
        "truth_blind_scoring_chain_count": chain_count,
    }
    identity["raw_self_sha256"] = validator.compute_raw_self_hash(identity, arrays)
    return identity, arrays


def test_future_raw_schema_accepts_uint64_bit63_and_complete_planned_ensemble_only():
    schema = load_schema()
    manifest = fake_manifest()
    records = [fake_raw(manifest, 0), fake_raw(manifest, 1)]
    assert int(records[0][1]["character_masks_uint64"][63]) == 1 << 63
    result = validator.validate_complete_planned_ensemble(records, manifest, schema)
    assert result["planned_count"] == 2
    assert result["status"] == "COMPLETE_PLANNED_IID_ENSEMBLE_ELIGIBLE_FOR_AUXILIARY_AUDIT"


def test_future_raw_schema_rejects_missing_failed_selected_or_tampered_records():
    schema = load_schema()
    manifest = fake_manifest()
    with pytest.raises(validator.EnsembleAuditNotComputable):
        validator.validate_complete_planned_ensemble([fake_raw(manifest, 0)], manifest, schema)
    with pytest.raises(validator.EnsembleAuditNotComputable):
        validator.validate_complete_planned_ensemble(
            [fake_raw(manifest, 0), fake_raw(manifest, 1, gate_pass=False)], manifest, schema
        )
    identity, arrays = fake_raw(manifest, 0)
    arrays["truth_blind_scoring_chain_mask"][0] = False
    identity["raw_self_sha256"] = validator.compute_raw_self_hash(identity, arrays)
    with pytest.raises(validator.FutureRawConflictError):
        validator.validate_raw_record(identity, arrays, manifest, schema)
    identity, arrays = fake_raw(manifest, 0)
    arrays["m2_debiased_per_character"][0] = 0.25
    with pytest.raises(validator.FutureRawConflictError):
        validator.validate_raw_record(identity, arrays, manifest, schema)


def signed_report(config, schema, *, gate_pass=True, **updates):
    gate = {
        "failures": [] if gate_pass else ["frozen optimistic power gate failed"],
        "passed": gate_pass,
        "universal_q_top_bias_bound": None,
    }
    report = {
        "authority": runner.EXPECTED_AUTHORITY,
        "calibration_gate": gate,
        "runner_sha256": config["implementation"]["bound_files"]["runner"]["sha256"],
        "schema_sha256": runner.sha256_file(ROOT / "nishimori_raw_schema.v1.json"),
        "status": runner.CALIBRATED_STATUS if gate_pass else runner.INSUFFICIENT_STATUS,
        "universal_q_top_bias_bound_from_identity": None,
        "version": "exp102.q0_nishimori_auxiliary_calibration.v2",
    }
    report.update(updates)
    report["report_sha256"] = hashlib.sha256(runner.canonical(report).encode("ascii")).hexdigest()
    return report


def test_machine_authority_envelope_rejects_status_or_authority_laundering():
    config = load_config()
    schema = load_schema()
    auditor.validate_report_envelope(signed_report(config, schema), config, schema)
    auditor.validate_report_envelope(
        signed_report(config, schema, gate_pass=False), config, schema
    )
    bad_authority = dict(runner.EXPECTED_AUTHORITY)
    bad_authority["formal_authorization"] = True
    with pytest.raises(RuntimeError):
        auditor.validate_report_envelope(signed_report(config, schema, authority=bad_authority), config, schema)
    with pytest.raises(RuntimeError):
        auditor.validate_report_envelope(signed_report(config, schema, status="READY_FOR_FORMAL"), config, schema)
    with pytest.raises(RuntimeError):
        auditor.validate_report_envelope(
            signed_report(
                config, schema, gate_pass=False, status=runner.CALIBRATED_STATUS
            ),
            config,
            schema,
        )


def test_config_binds_runner_oracle_auditor_validator_tests_docs_and_schema():
    config = load_config()
    expected_roles = {
        "audit_runner", "independent_oracle", "pre_run_red_team", "raw_schema",
        "raw_validator", "readme", "runner", "tests",
    }
    assert set(config["implementation"]["bound_files"]) == expected_roles
    for descriptor in config["implementation"]["bound_files"].values():
        path = PROJECT_ROOT / descriptor["path"]
        assert path.is_file()
        assert runner.sha256_file(path) == descriptor["sha256"]
    assert not (ROOT / "exact_calibration_report.json").exists()
    assert not (ROOT / "independent_audit.json").exists()
