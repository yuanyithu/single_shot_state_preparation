"""A cell is reportable only when every task behind it is complete and valid."""

import numpy as np
import pytest

from data.expander_code.exp104.exp104_pipeline.aggregate import aggregate_scan
from data.expander_code.exp104.exp104_pipeline.config import (
    CODES_PER_M,
    CODES_PER_TASK,
    M_VALUES,
)
from data.expander_code.exp104.exp104_pipeline.preflight import _synthetic_raw
from data.expander_code.exp104.exp104_pipeline.raw import raw_filename, save_raw


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
    save_raw(directory / raw_filename(m, block_index), raw)
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
    m8 = M_VALUES.index(8)
    base = m8 * CODES_PER_M
    reported = aggregate["code_status"][base:base + CODES_PER_TASK[8]]
    assert np.all(reported == "REPORTABLE")
    assert np.all(aggregate["code_status"][base + CODES_PER_TASK[8]] == "MISSING")
    assert np.all(aggregate["trial_counts"][base:base + CODES_PER_TASK[8]] == 4)
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
    m8 = M_VALUES.index(8)
    base = m8 * CODES_PER_M
    assert np.all(aggregate["code_status"][base:base + CODES_PER_TASK[8]] == "INVALID")
    # The contract's token must survive storage intact, not be truncated.
    assert np.all(aggregate["m_status"][m8] == "SAMPLING_INSUFFICIENT")
    assert aggregate["overall_status"] == "INCOMPLETE"


def test_a_schema_mismatch_is_recorded_as_an_unexpected_error(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0, schema_version="exp104.raw.v0")
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert aggregate["unexpected_raw_errors_json"] != "[]"
    assert "schema_mismatch" in aggregate["unexpected_raw_errors_json"]
    assert aggregate["overall_status"] == "INCOMPLETE"


def test_a_grid_mismatch_is_recorded_as_an_unexpected_error(tmp_path, frozen_config):
    _write_task(tmp_path, frozen_config, 8, 0, p_tokens="0.02,0.03")
    aggregate = aggregate_scan(tmp_path, frozen_config)
    assert "grid_mismatch" in aggregate["unexpected_raw_errors_json"]


def test_unplanned_raw_evidence_is_refused(tmp_path, frozen_config):
    from data.expander_code.exp104.exp104_pipeline.config import TASKS_PER_M

    raw = _synthetic_raw(8, frozen_config, _identity(frozen_config))
    raw["block_index"] = TASKS_PER_M[8] + 5
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
    base = M_VALUES.index(8) * CODES_PER_M
    for slot in range(CODES_PER_TASK[8]):
        for p_index in range(len(frozen_config["p_tokens"])):
            expected = int(raw["failure_flags"][slot, p_index].sum())
            assert aggregate["failure_counts"][base + slot, p_index] == expected
            assert aggregate["code_rates"][base + slot, p_index] == expected / 4
            assert (
                aggregate["wilson_low"][base + slot, p_index]
                <= expected / 4
                <= aggregate["wilson_high"][base + slot, p_index]
            )
