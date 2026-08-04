import json

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline.aggregate import (
    aggregate_decoder_scan,
    save_aggregate,
)
from data.expander_code.exp103.exp103_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp103.exp103_pipeline.loader import load_exp103_crossing


STAGE1_CROSSING_ARRAYS = (
    "stage1_delta35", "stage1_band_low", "stage1_band_high",
    "stage1_adjacent_delta", "stage1_adjacent_band_low",
    "stage1_adjacent_band_high", "stage1_primary_band_low",
    "stage1_primary_band_high",
)
FINAL_CROSSING_ARRAYS = (
    "primary_band_low", "primary_band_high", "delta38", "delta38_band_low",
    "delta38_band_high", "adjacent_delta", "adjacent_band_low",
    "adjacent_band_high",
)


def _withhold_stage1_crossing(aggregate):
    for field in STAGE1_CROSSING_ARRAYS:
        aggregate[field].fill(np.nan)
    aggregate["stage1_status"] = "INCOMPLETE"
    aggregate["stage1_bracket_low"] = np.nan
    aggregate["stage1_bracket_high"] = np.nan
    aggregate["stage1_bootstrap_half_width"] = np.nan
    aggregate["stage1_compatible_triple_json"] = "null"


def _mark_final_crossing_invalid(aggregate):
    for field in FINAL_CROSSING_ARRAYS:
        aggregate[field].fill(np.nan)
    aggregate["overall_status"] = "INVALID"
    aggregate["terminal_status"] = "EXP103_INVALID"
    aggregate["crossing_bracket_low"] = np.nan
    aggregate["crossing_bracket_high"] = np.nan
    aggregate["bootstrap_half_width"] = np.nan
    aggregate["compatible_triple_json"] = "null"


def test_complete_synthetic_aggregate_loads_readonly_and_preserves_full_panel(
    tmp_path, complete_aggregate_factory,
):
    aggregate = complete_aggregate_factory()
    path = tmp_path / "complete.npz"
    save_aggregate(path, aggregate)
    loaded = load_exp103_crossing(path)

    assert loaded["terminal_status"] == "EXP103_DECODER_CROSSING_RESOLVED"
    assert loaded["code_ids"].shape == (48,)
    assert loaded["p_values"].shape == (13,)
    assert loaded["primary_mean"].shape == (6, 13)
    assert (loaded["crossing_bracket_low"], loaded["crossing_bracket_high"]) == (
        0.07, 0.08,
    )
    with pytest.raises(TypeError):
        loaded["new_field"] = 1
    with pytest.raises(ValueError):
        loaded["primary_mean"][0, 0] = 0.0


def test_all_reportable_panel_without_replay_can_only_load_as_invalid(
    tmp_path, complete_aggregate_factory, rehash_aggregate,
):
    aggregate = complete_aggregate_factory()
    aggregate.update({
        "replay_status": "NOT_REQUIRED_INCOMPLETE",
        "replay_scope": "none",
        "replay_report_sha256": "",
        "raw_manifest_sha256": "",
        "replay_report_json": "{}",
    })
    _withhold_stage1_crossing(aggregate)
    rehash_aggregate(aggregate)

    claimed_complete_path = tmp_path / "complete_without_replay.npz"
    save_aggregate(claimed_complete_path, aggregate)
    with pytest.raises(ValueError, match="overall status"):
        load_exp103_crossing(claimed_complete_path)

    _mark_final_crossing_invalid(aggregate)
    rehash_aggregate(aggregate)
    invalid_path = tmp_path / "invalid_without_replay.npz"
    save_aggregate(invalid_path, aggregate)
    loaded = load_exp103_crossing(invalid_path)
    assert loaded["overall_status"] == "INVALID"
    assert loaded["terminal_status"] == "EXP103_INVALID"
    assert np.all(np.isnan(loaded["delta38"]))


def test_all_reportable_panel_with_only_stage1_replay_cannot_publish_final_crossing(
    tmp_path, complete_aggregate_factory, rehash_aggregate,
):
    aggregate = complete_aggregate_factory()
    replay_bundle = json.loads(aggregate["replay_report_json"])
    stage1_report = replay_bundle["stage1"]
    aggregate.update({
        "replay_status": "PASS",
        "replay_scope": "stage1",
        "replay_report_sha256": sha256_json(stage1_report),
        "raw_manifest_sha256": stage1_report["raw_manifest_sha256"],
        "replay_report_json": canonical_json(stage1_report),
    })

    claimed_complete_path = tmp_path / "complete_with_stage1_replay.npz"
    save_aggregate(claimed_complete_path, aggregate)
    with pytest.raises(ValueError, match="overall status"):
        load_exp103_crossing(claimed_complete_path)

    _mark_final_crossing_invalid(aggregate)
    rehash_aggregate(aggregate)
    invalid_path = tmp_path / "invalid_with_stage1_replay.npz"
    save_aggregate(invalid_path, aggregate)
    loaded = load_exp103_crossing(invalid_path)
    assert loaded["overall_status"] == "INVALID"
    assert loaded["terminal_status"] == "EXP103_INVALID"
    assert loaded["stage1_status"].startswith("STAGE1_RESTRICTED_")
    assert np.all(np.isnan(loaded["delta38"]))


def test_loader_accepts_only_the_preregistered_full_point_mask(
    tmp_path, complete_aggregate_factory,
):
    path = tmp_path / "complete.npz"
    save_aggregate(path, complete_aggregate_factory())
    full = np.ones(13, dtype=np.bool_)
    assert load_exp103_crossing(path, point_mask="full")["p_values"].shape == (13,)
    assert load_exp103_crossing(path, point_mask=full)["p_values"].shape == (13,)
    for mask in (
        np.asarray([True] * 12 + [False], dtype=np.bool_),
        np.ones(12, dtype=np.bool_),
        np.ones(13, dtype=np.int8),
    ):
        with pytest.raises(ValueError, match="point mask"):
            load_exp103_crossing(path, point_mask=mask)


@pytest.mark.parametrize(
    "field",
    [
        "config_sha256",
        "registry_sha256",
        "source_commit",
        "source_tree_sha256",
        "bplsd_binary_sha256",
    ],
)
def test_loader_rejects_frozen_scalar_identity_drift(
    tmp_path, complete_aggregate_factory, field,
):
    aggregate = complete_aggregate_factory()
    aggregate[field] = "f" * len(aggregate[field])
    path = tmp_path / f"tampered_{field}.npz"
    save_aggregate(path, aggregate)
    with pytest.raises(ValueError, match="identity|hash|SHA|source|binary"):
        load_exp103_crossing(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "exp102.aggregate.v1"),
        ("experiment_id", "exp102.q_top.v1"),
    ],
)
def test_loader_refuses_exp101_exp102_or_foreign_schema(
    tmp_path, complete_aggregate_factory, field, replacement,
):
    aggregate = complete_aggregate_factory()
    aggregate[field] = replacement
    path = tmp_path / f"foreign_{field}.npz"
    save_aggregate(path, aggregate)
    with pytest.raises(ValueError):
        load_exp103_crossing(path)


def test_payload_hash_and_axis_tampering_are_rejected(
    tmp_path, complete_aggregate_factory, rehash_aggregate,
):
    hash_tamper = complete_aggregate_factory()
    hash_tamper["primary_mean"][0, 0] += 0.01
    hash_path = tmp_path / "bad_payload_hash.npz"
    save_aggregate(hash_path, hash_tamper)
    with pytest.raises(ValueError, match="payload hash"):
        load_exp103_crossing(hash_path)

    axis_tamper = complete_aggregate_factory()
    axis_tamper["code_ids"][[0, 1]] = axis_tamper["code_ids"][[1, 0]]
    rehash_aggregate(axis_tamper)
    axis_path = tmp_path / "bad_axis.npz"
    save_aggregate(axis_path, axis_tamper)
    with pytest.raises(ValueError, match="code axis"):
        load_exp103_crossing(axis_path)


def test_loader_recomputes_counts_means_and_crossing_decision(
    tmp_path, complete_aggregate_factory, rehash_aggregate,
):
    count_tamper = complete_aggregate_factory()
    count_tamper["failure_counts"][0, 0] += 1
    rehash_aggregate(count_tamper)
    count_path = tmp_path / "bad_count.npz"
    save_aggregate(count_path, count_tamper)
    with pytest.raises(ValueError, match="rate|count"):
        load_exp103_crossing(count_path)

    mean_tamper = complete_aggregate_factory()
    mean_tamper["primary_mean"][0, 0] += 0.001
    rehash_aggregate(mean_tamper)
    mean_path = tmp_path / "bad_mean.npz"
    save_aggregate(mean_path, mean_tamper)
    with pytest.raises(ValueError, match="mean"):
        load_exp103_crossing(mean_path)

    decision_tamper = complete_aggregate_factory()
    decision_tamper["terminal_status"] = "EXP103_PAIRWISE_BRACKET_ONLY"
    decision_path = tmp_path / "bad_decision.npz"
    save_aggregate(decision_path, decision_tamper)
    with pytest.raises(ValueError, match="decision"):
        load_exp103_crossing(decision_path)


def test_loader_rejects_tampered_compatible_triple(
    tmp_path, complete_aggregate_factory,
):
    aggregate = complete_aggregate_factory()
    aggregate["compatible_triple_json"] = "null"
    path = tmp_path / "bad_triple.npz"
    save_aggregate(path, aggregate)
    with pytest.raises(ValueError, match="triple"):
        load_exp103_crossing(path)


def test_nonreportable_counts_and_valid_only_primary_cannot_leak(
    tmp_path, frozen_config, rehash_aggregate,
):
    raw_root = tmp_path / "empty_raw"
    raw_root.mkdir()
    base = aggregate_decoder_scan(raw_root, frozen_config)
    clean_path = tmp_path / "incomplete_clean.npz"
    save_aggregate(clean_path, base)
    assert load_exp103_crossing(clean_path)["terminal_status"] == "EXP103_INCOMPLETE"

    leaked_count = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in base.items()
    }
    leaked_count["failure_counts"][0, 0] = 1
    leaked_count["trial_counts"][0, 0] = 1
    rehash_aggregate(leaked_count)
    count_path = tmp_path / "incomplete_count_leak.npz"
    save_aggregate(count_path, leaked_count)
    with pytest.raises(ValueError, match="nonreportable|count"):
        load_exp103_crossing(count_path)

    valid_only = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in base.items()
    }
    valid_only["primary_mean"][0, 0] = 0.1
    rehash_aggregate(valid_only)
    valid_only_path = tmp_path / "valid_only.npz"
    save_aggregate(valid_only_path, valid_only)
    with pytest.raises(ValueError, match="valid-only"):
        load_exp103_crossing(valid_only_path)
