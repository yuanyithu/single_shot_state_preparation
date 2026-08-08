"""The frozen contract must reject every mutation of itself."""

import copy

import pytest

from data.expander_code.exp104.exp104_pipeline import config as config_module
from data.expander_code.exp104.exp104_pipeline.config import (
    CODES_PER_M,
    CODES_PER_TASK,
    M_VALUES,
    P_TOKENS,
    TASKS_PER_M,
    TRIALS_PER_CODE_P,
    block_code_indices,
    code_id,
    ensure_config,
    normalize_p_token,
    parse_code_id,
)


def test_grid_and_panel_are_the_frozen_ones(frozen_config):
    assert frozen_config["p_tokens"] == [
        "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.10",
    ]
    assert frozen_config["m_values"] == [3, 4, 5, 6, 7, 8]
    assert frozen_config["codes_per_m"] == 2000
    assert frozen_config["trials_per_code_p"] == 4
    assert frozen_config["objective"] == "bposd_ensemble_block_logical_failure_crossing_q0"


def test_decoder_identity_matches_the_frozen_exp103_decoder(frozen_config):
    # exp104 is only comparable with exp103 if this stays byte for byte equal.
    assert frozen_config["decoder"] == {
        "bp_method": "product_sum",
        "max_iter": "n",
        "schedule": "serial",
        "serial_schedule_order": "natural",
        "osd_method": "osd_0",
        "osd_order": 0,
        "omp_thread_count": 1,
    }
    assert frozen_config["decoder_binary"]["module"] == "ldpc.bposd_decoder._bposd_decoder"


def test_ensemble_rule_declares_no_post_selection(frozen_config):
    assert frozen_config["ensemble"]["post_selection"] == "none"
    assert frozen_config["ensemble"]["acceptance_rule"] == "full_row_rank_and_unique_H"
    assert frozen_config["ensemble"]["d_A"] == 3 and frozen_config["ensemble"]["d_B"] == 4


def test_crossing_rule_is_primary_only_and_not_adjacent_bound(frozen_config):
    crossing = frozen_config["crossing"]
    assert crossing["primary_contrast"] == "m08_minus_m03"
    assert crossing["bracket_requires_adjacent_grid_points"] is False
    assert crossing["simultaneous_band_scope"] == "primary_contrast_grid_only"
    assert crossing["adjacent_contrasts"] == "diagnostic_pointwise_only"


def test_replay_policy_is_a_committed_subsample(frozen_config):
    assert frozen_config["replay"] == {
        "policy": "committed_random_subsample",
        "fraction": 0.10,
        "always_include_block_index": 0,
    }


def test_remote_profile_pins_nd3_and_the_authorized_caps(remote_config):
    profile = remote_config["execution_profile"]
    assert profile["compute_host"] == "nd-3"
    assert profile["entry_host"] == "yuany"
    assert profile["num_workers"] == 64
    assert profile["omp_thread_count"] == 1
    assert profile["stage_core_hour_cap"] == 900.0
    assert profile["stage_wall_hour_cap"] == 16.0
    assert profile["peak_rss_gib_cap"] == 128.0
    assert profile["reserve_multiplier"] == 2.0


def test_namespaces_are_disjoint_and_experiment_scoped(frozen_config):
    namespaces = frozen_config["namespaces"]
    assert len(set(namespaces.values())) == len(namespaces)
    assert all(value.startswith("exp104.") for value in namespaces.values())


def test_task_blocking_partitions_every_code_exactly_once():
    for m in M_VALUES:
        assert CODES_PER_M % CODES_PER_TASK[m] == 0
        assert TASKS_PER_M[m] * CODES_PER_TASK[m] == CODES_PER_M
        seen = []
        for block in range(TASKS_PER_M[m]):
            indices = block_code_indices(m, block)
            assert len(indices) == CODES_PER_TASK[m]
            seen.extend(indices)
        assert seen == list(range(CODES_PER_M))


def test_block_index_outside_the_plan_is_rejected():
    for m in M_VALUES:
        with pytest.raises(ValueError):
            block_code_indices(m, TASKS_PER_M[m])
        with pytest.raises(ValueError):
            block_code_indices(m, -1)
    with pytest.raises(ValueError):
        block_code_indices(9, 0)


def test_code_id_round_trip_and_range():
    for m in M_VALUES:
        assert parse_code_id(code_id(m, 0)) == (m, 0)
        assert parse_code_id(code_id(m, CODES_PER_M - 1)) == (m, CODES_PER_M - 1)
        with pytest.raises(ValueError):
            code_id(m, CODES_PER_M)
    for bad in ("m03_c0000", "m09_c00000", "x03_c00000", "m03_c99999"):
        with pytest.raises(ValueError):
            parse_code_id(bad)


def test_p_token_normalization_rejects_everything_off_grid():
    assert normalize_p_token("0.06") == "0.06"
    assert normalize_p_token(0.06) == "0.06"
    for bad in ("0.11", "0.065", 0.11, 0.0, 1.0, "abc"):
        with pytest.raises(ValueError):
            normalize_p_token(bad)


@pytest.mark.parametrize("field", [
    "p_tokens", "m_values", "codes_per_m", "trials_per_code_p", "decoder",
    "ensemble", "namespaces", "bootstrap", "replay", "crossing", "preflight",
    "master_seed_hex", "objective", "registry_path", "codes_per_task",
])
def test_mutating_any_frozen_field_is_rejected(frozen_config, field):
    broken = copy.deepcopy(frozen_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    value = broken[field]
    if isinstance(value, dict):
        broken[field] = dict(value)
        broken[field].pop(next(iter(value)))
    elif isinstance(value, list):
        broken[field] = value[:-1]
    elif isinstance(value, int):
        broken[field] = value + 1
    else:
        broken[field] = str(value) + "x"
    with pytest.raises(ValueError):
        ensure_config(broken)


def test_unknown_or_missing_top_level_fields_are_rejected(frozen_config):
    broken = copy.deepcopy(frozen_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    broken["extra"] = 1
    with pytest.raises(ValueError):
        ensure_config(broken)
    broken = copy.deepcopy(frozen_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    broken.pop("crossing")
    with pytest.raises(ValueError):
        ensure_config(broken)


def test_config_sha_mismatch_is_rejected(frozen_config):
    broken = copy.deepcopy(frozen_config)
    broken.pop("config_path", None)
    broken["config_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ensure_config(broken)


def test_config_must_be_the_canonical_artifact(tmp_path, frozen_config):
    import json

    copied = tmp_path / "ensemble_mc.v1.json"
    payload = {
        key: value for key, value in frozen_config.items()
        if key not in {"config_sha256", "config_path"}
    }
    copied.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError):
        config_module.load_config(copied)


def test_remote_config_rejects_a_placeholder_source_identity(remote_config):
    broken = copy.deepcopy(remote_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    broken["source_commit"] = "0" * 40
    with pytest.raises(ValueError):
        ensure_config(broken)
