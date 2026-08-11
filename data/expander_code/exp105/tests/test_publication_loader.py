"""A published aggregate must survive independent re-derivation."""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.aggregate import ARRAY_FIELDS, SCALAR_FIELDS
from data.expander_code.exp105.exp105_pipeline.crossing import CERTIFIED
from data.expander_code.exp105.exp105_pipeline.loader import load_exp105_crossing


def test_loader_accepts_the_complete_aggregate(complete_aggregate_factory, frozen_config):
    payload = complete_aggregate_factory()
    accepted = load_exp105_crossing(payload, frozen_config)
    assert accepted["overall_status"] == "COMPLETE"
    assert accepted["terminal_status"] == CERTIFIED
    assert accepted["replay_status"] == "PASS"
    assert np.isfinite(accepted["p_cross"])
    assert (
        accepted["crossing_bracket_low"]
        <= accepted["p_cross"]
        <= accepted["crossing_bracket_high"]
    )
    assert accepted["p_cross_low"] <= accepted["p_cross"] <= accepted["p_cross_high"]


def test_loader_round_trips_through_an_npz_file(
    tmp_path, complete_aggregate_factory, frozen_config,
):
    payload = complete_aggregate_factory()
    path = tmp_path / "ensemble_crossing.npz"
    np.savez_compressed(path, **{
        key: np.asarray(payload[key]) for key in ARRAY_FIELDS + SCALAR_FIELDS
    })
    accepted = load_exp105_crossing(path, frozen_config)
    assert accepted["terminal_status"] == payload["terminal_status"]
    assert accepted["p_cross"] == pytest.approx(payload["p_cross"])
    # Status tokens must not be truncated by the stored dtype.
    assert set(np.unique(accepted["m_status"])) == {"REPORTABLE"}


def test_loader_rejects_a_tampered_payload_without_rehashing(
    complete_aggregate_factory, frozen_config,
):
    payload = complete_aggregate_factory()
    payload["failure_counts"][0, 0] = (payload["failure_counts"][0, 0] + 1) % 5
    with pytest.raises(ValueError, match="payload SHA256"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_recomputes_rates_from_counts(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["code_rates"][3, 2] += 0.25
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="per-code rates"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_recomputes_the_pooled_mean(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["primary_mean"][1, 4] += 0.05
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="primary mean"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_recomputes_the_cluster_standard_error(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["cluster_se"][1, 1] *= 0.5
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="cluster standard error"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_recomputes_the_strata_table(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["strata_failures"][0, 1, 0] += 7
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="strata failure counts"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_rejects_a_relabelled_terminal_status(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["terminal_status"] = "EXP105_NO_CERTIFIED_CROSSING"
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="terminal status"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_rejects_a_moved_crossing_location(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["p_cross"] = float(payload["p_cross"]) + 0.01
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="p_cross"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_rejects_a_widened_or_narrowed_band(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["bootstrap_half_width"] = float(payload["bootstrap_half_width"]) * 0.5
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="half-width"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_rejects_a_foreign_config(
    complete_aggregate_factory, frozen_config, foreign_config,
):
    assert foreign_config["config_sha256"] != frozen_config["config_sha256"]
    payload = complete_aggregate_factory()
    with pytest.raises(ValueError, match="identity mismatch"):
        load_exp105_crossing(payload, foreign_config)


def test_loader_refuses_an_incomplete_publication(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["overall_status"] = "INCOMPLETE"
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="not COMPLETE"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_refuses_a_missing_replay_gate(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["replay_status"] = "MISSING"
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="replay gate"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_refuses_unexpected_raw_errors(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["unexpected_raw_errors_json"] = '[{"m": 8}]'
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="unexpected raw errors"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_refuses_cells_that_are_not_reportable(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["code_status"][17, 3] = "INVALID"
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="REPORTABLE"):
        load_exp105_crossing(payload, frozen_config)


def test_loader_refuses_counts_outside_the_trial_allocation(
    complete_aggregate_factory, frozen_config, rehash_aggregate,
):
    payload = complete_aggregate_factory()
    payload["failure_counts"][5, 5] = 9
    payload["code_rates"][5, 5] = 9 / 4
    rehash_aggregate(payload)
    with pytest.raises(ValueError, match="legal range"):
        load_exp105_crossing(payload, frozen_config)
