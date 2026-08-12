"""A cell is reportable only when every task behind it is complete and valid."""

import numpy as np
import pytest

from data.expander_code.exp106.exp106_pipeline.aggregate import (
    aggregate_scan,
    panel_layout,
)
from data.expander_code.exp106.exp106_pipeline.raw import task_counts
from data.expander_code.exp106.exp106_pipeline.preflight import _synthetic_raw
from data.expander_code.exp106.exp106_pipeline.raw import raw_filename, save_raw


def _identity(config):
    return {
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "python_version": config["environment"]["python"],
        "numpy_version": config["environment"]["numpy"],
        "scipy_version": config["environment"]["scipy"],
        "ldpc_version": config["environment"]["ldpc"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
    }


def _write_task(directory, config, m, block_index, **overrides):
    raw = _synthetic_raw(m, config, _identity(config))
    raw["block_index"] = int(block_index)
    raw.update(overrides)
    save_raw(directory / raw_filename(config, m, block_index), raw)
    return raw


def test_an_empty_raw_tree_is_incomplete(tmp_path, frozen_config):
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert aggregate["overall_status"] == "INCOMPLETE"
    assert aggregate["terminal_status"] == "INCOMPLETE"
    assert np.all(aggregate["m_status"] == "INCOMPLETE")
    assert np.all(np.isnan(aggregate["delta38"]))
    assert np.isnan(aggregate["bootstrap_half_width"])
    assert np.isnan(aggregate["p_cross"])
    assert aggregate["p_cross_reason"] == "aggregate_incomplete"


def test_one_valid_task_reports_only_its_own_codes(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0)
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert aggregate["overall_status"] == "INCOMPLETE"
    m_values, _, trials_by_m, offsets, _ = panel_layout(frozen_config)
    block = int(frozen_config["codes_per_task"]["8"])
    base = offsets[8]
    reported = aggregate["code_status"][base:base + block]
    assert np.all(reported == "REPORTABLE")
    assert np.all(aggregate["code_status"][base + block] == "MISSING")
    assert np.all(
        aggregate["trial_counts"][base:base + block] == trials_by_m[8]
    )
    # An incomplete panel still publishes nothing.
    assert np.all(np.isnan(aggregate["primary_mean"]))


def test_an_invalid_task_poisons_its_cells_rather_than_being_dropped(
    tmp_path, frozen_config,
):
    _write_task(
        tmp_path, frozen_config, 8, 0,
        status="INVALID", invalid_reason="trial_infrastructure_error",
        completed_codes=0,
    )
    aggregate = aggregate_scan(tmp_path, frozen_config)
    m_values, _, _, offsets, _ = panel_layout(frozen_config)
    m8 = m_values.index(8)
    block = int(frozen_config["codes_per_task"]["8"])
    base = offsets[8]
    assert np.all(aggregate["code_status"][base:base + block] == "INVALID")
    # The contract's token must survive storage intact, not be truncated.
    assert np.all(aggregate["m_status"][m8] == "SAMPLING_INSUFFICIENT")
    assert aggregate["overall_status"] == "INCOMPLETE"


def test_a_schema_mismatch_is_recorded_as_an_unexpected_error(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0, schema_version="exp106.raw.v0")
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert aggregate["unexpected_raw_errors_json"] != "[]"
    assert "schema_mismatch" in aggregate["unexpected_raw_errors_json"]
    assert aggregate["overall_status"] == "INCOMPLETE"


def test_a_grid_mismatch_is_recorded_as_an_unexpected_error(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0, p_tokens="0.02,0.03")
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert "grid_mismatch" in aggregate["unexpected_raw_errors_json"]


def test_unplanned_raw_evidence_is_refused(tmp_path, frozen_config):
    raw = _synthetic_raw(8, frozen_config, _identity(frozen_config))
    raw["block_index"] = task_counts(frozen_config)[8] + 5
    save_raw(tmp_path / "m08__b9999.npz", raw)
    with pytest.raises(ValueError, match="unplanned"):
        aggregate_scan(tmp_path, frozen_config)


def test_duplicate_task_identities_are_refused(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0)
    raw = _synthetic_raw(8, frozen_config, _identity(frozen_config))
    save_raw(tmp_path / "duplicate.npz", raw)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_scan(tmp_path, frozen_config)


def test_counts_and_rates_agree_for_an_ingested_task(tmp_path, frozen_config):
    raw = _write_task(tmp_path, frozen_config, 8, 0)
    aggregate = aggregate_scan(tmp_path, frozen_config)
    _, _, trials_by_m, offsets, _ = panel_layout(frozen_config)
    trials = trials_by_m[8]
    base = offsets[8]
    for slot in range(int(frozen_config["codes_per_task"]["8"])):
        for p_index in range(len(frozen_config["p_tokens"])):
            expected = int(raw["failure_flags"][slot, p_index].sum())
            assert aggregate["failure_counts"][base + slot, p_index] == expected
            assert aggregate["code_rates"][base + slot, p_index] == expected / trials
            assert (
                aggregate["wilson_low"][base + slot, p_index]
                <= expected / trials
                <= aggregate["wilson_high"][base + slot, p_index]
            )


@pytest.mark.parametrize("foreign_q", ["0.05", "0.0", "0.011"])
def test_a_q_mismatch_is_recorded_as_an_unexpected_error(
    tmp_path, frozen_config, foreign_q,
):
    """exp106 is a fixed-q experiment; raw from another q must never merge.

    `0.05` is exp105's q and `0.0` is exp104's, and no filename anywhere encodes
    q -- raw files, configs and aggregates are name-identical across the three.
    This check inside the aggregator is what actually keeps their evidence
    apart, so it is exercised with the two values that could really turn up and
    with one that is merely close.
    """
    assert foreign_q != frozen_config["q_token"]
    _write_task(tmp_path, frozen_config, 8, 0, q_token=foreign_q)
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert "q_mismatch" in aggregate["unexpected_raw_errors_json"]
    assert aggregate["overall_status"] == "INCOMPLETE"
