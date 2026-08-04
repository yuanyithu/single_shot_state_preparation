import json

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp103.exp103_pipeline.aggregate import aggregate_decoder_scan
from data.expander_code.exp103.exp103_pipeline.config import CODE_IDS, ensure_config
from data.expander_code.exp103.exp103_pipeline.raw import raw_filename, save_raw


def _write_cell(root, raw_factory, code_id, p_token="0.02", mutator=None):
    paths = []
    for shard_index in range(4):
        raw = raw_factory(code_id, p_token, shard_index, failures=shard_index)
        if mutator is not None:
            mutator(raw, shard_index)
        path = root / code_id / raw_filename(code_id, p_token, shard_index)
        save_raw(path, raw)
        paths.append(path)
    return paths


def test_registry_retains_exact_full_panel_including_all_distance_two_codes(
    frozen_config,
):
    registry = load_registry(frozen_config["registry_path"])
    ids = [row["code_id"] for row in registry["codes"]]
    assert ids == CODE_IDS
    assert len(ids) == len(set(ids)) == 48
    distance_two = {
        row["code_id"] for row in registry["codes"] if row["classical_distance"] == 2
    }
    assert distance_two == {
        "m04_c01", "m04_c06", "m04_c07", "m05_c01",
        "m05_c06", "m06_c00", "m06_c06", "m08_c04",
    }


def test_config_rejects_panel_grid_decoder_and_claimed_hash_mutations(
    frozen_config, clone_payload,
):
    mutations = (
        ("m_values", [3, 4, 5, 6, 7]),
        ("p_tokens", frozen_config["p_tokens"][:-1]),
        ("codes_per_m", 7),
        ("trials_per_shard", 2499),
    )
    for field, value in mutations:
        bad = clone_payload(frozen_config)
        bad[field] = value
        with pytest.raises(ValueError):
            ensure_config(bad)

    bad_decoder = clone_payload(frozen_config)
    bad_decoder["decoder"]["lsd_order"] = 1
    with pytest.raises(ValueError):
        ensure_config(bad_decoder)

    bad_claim = clone_payload(frozen_config)
    bad_claim["config_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="config SHA256 mismatch"):
        ensure_config(bad_claim)

    bad_environment = clone_payload(frozen_config)
    bad_environment.pop("config_sha256")
    bad_environment["environment"]["hostname"] = "other.local"
    with pytest.raises(ValueError, match="environment"):
        ensure_config(bad_environment)

    non_boolean_attestation = clone_payload(frozen_config)
    non_boolean_attestation.pop("config_sha256")
    non_boolean_attestation["environment"]["conda_prefix_matches_python"] = 1
    with pytest.raises(ValueError, match="boolean"):
        ensure_config(non_boolean_attestation)


def test_tampered_registry_is_rejected_before_aggregation(
    tmp_path, frozen_config, clone_payload,
):
    source = json.loads(open(frozen_config["registry_path"], encoding="ascii").read())
    source["codes"] = source["codes"][:-1]
    tampered_registry = tmp_path / "registry.json"
    tampered_registry.write_text(json.dumps(source), encoding="ascii")
    config = clone_payload(frozen_config)
    config.pop("config_sha256")
    config.pop("config_path", None)
    config["registry_path"] = str(tampered_registry)
    with pytest.raises(ValueError, match="registry"):
        aggregate_decoder_scan(tmp_path / "raw", config)


def test_partial_eight_code_m_point_stays_nan_until_every_code_is_reportable(
    tmp_path, frozen_config, raw_factory,
):
    for code_id in CODE_IDS[:7]:
        _write_cell(tmp_path, raw_factory, code_id)
    partial = aggregate_decoder_scan(tmp_path, frozen_config)
    assert np.all(partial["code_status"][:7, 0] == "REPORTABLE")
    assert partial["code_status"][7, 0] == "INCOMPLETE"
    assert partial["m_status"][0, 0] == "INCOMPLETE"
    assert np.isnan(partial["primary_mean"][0, 0])
    assert np.isnan(partial["primary_median"][0, 0])
    assert partial["terminal_status"] == "EXP103_INCOMPLETE"

    _write_cell(tmp_path, raw_factory, CODE_IDS[7])
    reportable = aggregate_decoder_scan(tmp_path, frozen_config)
    expected_rates = np.asarray([sum(range(4)) / 10_000] * 8)
    assert reportable["m_status"][0, 0] == "REPORTABLE"
    assert reportable["primary_mean"][0, 0] == pytest.approx(expected_rates.mean())
    assert reportable["primary_median"][0, 0] == pytest.approx(np.median(expected_rates))
    assert reportable["overall_status"] == "INCOMPLETE"
    assert not np.isfinite(reportable["delta38"]).any()


def test_syndrome_mismatch_is_a_valid_failure_not_infrastructure_invalid(
    tmp_path, frozen_config, raw_factory,
):
    def mismatch_first_trial(raw, shard_index):
        raw["syndrome_match"][100] = False
        raw["failure_flags"][100] = True

    _write_cell(tmp_path, raw_factory, "m03_c00", mutator=mismatch_first_trial)
    result = aggregate_decoder_scan(tmp_path, frozen_config)
    assert result["code_status"][0, 0] == "REPORTABLE"
    assert result["failure_counts"][0, 0] == 10
    assert result["syndrome_mismatch_rate"][0, 0] == pytest.approx(4 / 10_000)
    assert result["bp_convergence_rate"][0, 0] == 0.0


@pytest.mark.parametrize(
    "tamper",
    ["source", "host", "prefix_type", "dtype", "shape", "exception"],
)
def test_invalid_shard_closes_code_m_and_formal_crossing(
    tmp_path, frozen_config, raw_factory, tamper,
):
    raw = raw_factory("m03_c00", "0.02", 0)
    if tamper == "source":
        raw["source_tree_sha256"] = "f" * 64
    elif tamper == "host":
        raw["hostname"] = "other.local"
    elif tamper == "prefix_type":
        raw["conda_prefix_matches_python"] = 1
    elif tamper == "dtype":
        raw["logical_labels"] = raw["logical_labels"].astype(np.int64)
    elif tamper == "shape":
        raw["failure_flags"] = raw["failure_flags"][:-1]
    else:
        raw["status"] = "INVALID"
        raw["invalid_reason"] = "trial_infrastructure_error"
        raw["exception_type"] = "RuntimeError"
        raw["exception_message"] = "deliberate"
    save_raw(tmp_path / f"{tamper}.npz", raw)

    aggregate = aggregate_decoder_scan(tmp_path, frozen_config)
    assert aggregate["code_status"][0, 0] == "INVALID"
    assert aggregate["m_status"][0, 0] == "INVALID"
    assert aggregate["overall_status"] == "INVALID"
    assert aggregate["terminal_status"] == "EXP103_INVALID"
    assert np.isnan(aggregate["primary_mean"][0, 0])
    assert not np.isfinite(aggregate["delta38"]).any()


def test_conflicting_duplicate_shard_is_invalid(
    tmp_path, frozen_config, raw_factory,
):
    first = raw_factory("m03_c00", "0.02", 0)
    second = raw_factory("m03_c00", "0.02", 0)
    second["bp_iterations"][0] = 2
    save_raw(tmp_path / "copy_a.npz", first)
    save_raw(tmp_path / "copy_b.npz", second)
    aggregate = aggregate_decoder_scan(tmp_path, frozen_config)
    assert aggregate["code_status"][0, 0] == "INVALID"
    assert aggregate["overall_status"] == "INVALID"


def test_well_formed_unplanned_raw_key_is_not_silently_ignored(
    tmp_path, frozen_config, raw_factory,
):
    raw = raw_factory("m03_c00", "0.02", 0)
    raw["code_id"] = "unplanned_code"
    save_raw(tmp_path / "rogue.npz", raw)
    aggregate = aggregate_decoder_scan(tmp_path, frozen_config)
    assert aggregate["overall_status"] == "INVALID"
    assert "unplanned" in aggregate["unexpected_raw_errors_json"]


def test_malformed_unexpected_npz_invalidates_aggregate(tmp_path, frozen_config):
    np.savez(tmp_path / "malformed.npz", foreign=np.asarray([1], dtype=np.int64))
    aggregate = aggregate_decoder_scan(tmp_path, frozen_config)
    assert aggregate["overall_status"] == "INVALID"
    assert aggregate["terminal_status"] == "EXP103_INVALID"
    assert "malformed.npz" in aggregate["unexpected_raw_errors_json"]
