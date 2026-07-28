"""Focused tests for the validation-065 numerical audit rebind."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return load_module("exp102_v065_audit_test", "audit_rebind.py")


@pytest.fixture(scope="module")
def verifier():
    return load_module("exp102_v065_verifier_test", "verify_audit_rebind.py")


@pytest.fixture(scope="module")
def recomputed(audit):
    config = audit.load_json(audit.CONFIG_PATH)
    report, old_config, conflict = audit.verify_original_evidence(config)
    expected, failures, report_legacy, oracle_legacy, mismatches, ties = audit.recompute_and_compare(
        config, report, old_config,
    )
    return (
        config, report, old_config, conflict, expected, failures,
        report_legacy, oracle_legacy, mismatches, ties,
    )


def test_config_has_zero_authority_and_fixed_input(audit):
    config = audit.load_json(audit.CONFIG_PATH)
    assert config["version"] == "exp102.q0_nishimori_audit_rebind.config.v1"
    assert config["authority"]
    assert not any(config["authority"].values())
    assert config["input_063"]["report_self_sha256"] == (
        "134228b993a7b856c143d5410de7940f51657f0d1b7c38bcad1fd6cd917af441"
    )


def test_original_report_and_historical_blobs_are_immutable(recomputed):
    _, report, _, conflict, _, _, _, _, _, _ = recomputed
    assert report["status"] == "NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT"
    assert report["universal_q_top_bias_bound_from_identity"] is None
    assert conflict["independent_audit_created"] is False
    assert conflict["status"] == "CONFLICT_INDEPENDENT_AUDIT_MESSAGE_TAXONOMY_MISMATCH"


def test_full_payload_recomputes_and_yields_fourteen_rate_failures(recomputed):
    config, report, _, _, expected, failures, _, _, mismatches, ties = recomputed
    assert len(expected["golden_rows"]) == 6
    assert len(expected["exact_control_rows"]) == 30
    assert len(expected["power_rows"]) == 30
    assert len(failures) == config["input_063"]["expected_calibration_failure_count"] == 14
    assert len(report["calibration_gate"]["failures"]) == 14
    assert {row["control"] for row in failures} == {"correct_posterior"}
    assert {tuple(row["reason_codes"]) for row in failures} == {
        ("EQUIVALENCE_RATE_BELOW_MINIMUM",)
    }
    assert len(mismatches) == config["input_063"]["expected_full_payload_mismatch_count"] == 11
    assert len(ties) == config["input_063"]["expected_map_tie_case_count"] == 3
    assert max(row["absolute_difference"] for row in mismatches) == pytest.approx(
        config["input_063"]["expected_maximum_absolute_mismatch"], abs=1e-15,
    )
    assert all(row["mathematical_weight_enumerator_tie"] for row in ties)


def test_legacy_prefixes_rebind_to_the_same_numeric_identities(recomputed):
    _, _, _, _, _, failures, report_legacy, oracle_legacy, _, _ = recomputed
    numeric = {row["failure_id"] for row in failures}
    assert set(report_legacy) == set(oracle_legacy) == numeric
    assert set(report_legacy.values()) == {"equivalence gate failed"}
    assert set(oracle_legacy.values()) == {"equivalence power failed"}


def test_unknown_or_malformed_legacy_grammar_is_rejected(audit):
    aliases = {
        "equivalence gate failed": "EXACT_CONTROL_GATE",
        "equivalence power failed": "EXACT_CONTROL_GATE",
    }
    with pytest.raises(audit.AuditError, match="unrecognized legacy"):
        audit.parse_legacy_failures(
            ["approximately equivalent: hgp_n10_k4/0.04/correct_posterior/basis_max"],
            aliases,
        )
    with pytest.raises(audit.AuditError, match="duplicate legacy"):
        message = "equivalence gate failed: hgp_n10_k4/0.04/correct_posterior/basis_max"
        audit.parse_legacy_failures([message, message], aliases)


def test_numeric_gate_not_legacy_text_controls_the_decision(audit, recomputed):
    _, _, old_config, _, expected, failures, _, _, _, _ = recomputed
    changed = copy.deepcopy(expected["power_rows"])
    first = failures[0]
    for row in changed:
        if (
            row["model_id"] == first["model_id"]
            and float(row["p"]) == float(first["p"])
            and row["control"] == first["control"]
        ):
            gate_row = next(item for item in row["rows"] if item["ensemble_size"] == 2048)
            gate_row["statistics"][first["character_group"]][
                "diagnostic_equivalence_pass_rate"
            ] = 1.0
            break
    rebuilt = audit.rebuild_structured_failures(
        old_config, expected["exact_control_rows"], changed,
        expected["chain_level_control_metrics"],
    )
    assert len(rebuilt) == 13
    assert first["failure_id"] not in {row["failure_id"] for row in rebuilt}


def test_independent_verifier_rebuild_matches_without_importing_audit(verifier, recomputed):
    _, _, old_config, _, expected, failures, _, _, _, _ = recomputed
    rebuilt = verifier.independent_failures(
        old_config, expected["exact_control_rows"], expected["power_rows"],
        expected["chain_level_control_metrics"],
    )
    assert json.loads(json.dumps(rebuilt, sort_keys=True)) == json.loads(
        json.dumps(failures, sort_keys=True)
    )


def test_independent_verifier_reproduces_all_three_map_ties(verifier, recomputed):
    _, report, old_config, _, expected, _, _, _, mismatches, ties = recomputed
    oracle = verifier.frozen_oracle(verifier.strict_json(verifier.CONFIG_PATH))
    rebuilt_ties = verifier.independent_tie_witnesses(
        oracle, old_config, report, expected,
    )
    assert rebuilt_ties == ties
    assert {(row["p"], row["syndrome_hex"], row["report_choice"]["label"], row["oracle_choice"]["label"]) for row in ties} == {
        (0.04, "05", 0, 15),
        (0.04, "06", 0, 5),
        (0.1, "03", 10, 0),
    }
    assert len(mismatches) == 11


def test_one_shot_outputs_are_not_precreated(audit, verifier):
    assert not audit.OUTPUT_PATH.exists()
    assert not verifier.OUTPUT_PATH.exists()
