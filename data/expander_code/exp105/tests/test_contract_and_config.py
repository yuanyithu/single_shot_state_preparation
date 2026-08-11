"""The frozen contract must reject every mutation of itself.

exp105 has two config phases. The locating pilot is frozen now; the production
plan is deliberately absent from the source until Validation 003 evaluates the
contract's freezing rules on pilot measurements. Both facts are asserted here,
because "not yet frozen" has to be a state the code enforces rather than a note
in a document.
"""

import copy

import pytest

from data.expander_code.exp105.exp105_pipeline import config as config_module
from data.expander_code.exp105.exp105_pipeline.config import (
    ANCHOR_M_VALUES,
    M_VALUES,
    PILOT_CODES_PER_M,
    PILOT_CODES_PER_TASK,
    PILOT_M_VALUES,
    PILOT_P_TOKENS,
    PILOT_TRIALS_PER_CODE_P,
    ProductionPlanNotFrozen,
    Q_TOKEN,
    block_code_indices,
    code_id,
    ensure_config,
    normalize_p_token,
    normalize_q_token,
    parse_code_id,
    plan_for_phase,
    require_production_plan_frozen,
    tasks_per_m,
)


def test_the_production_plan_freeze_is_all_or_nothing(monkeypatch):
    """The freeze is a state the code enforces, not a note in a document.

    Before Validation 003 every production entry point must refuse; after it,
    every production constant must be present. Half-frozen is the dangerous
    state and is what this asserts cannot exist.
    """
    assert config_module.PRODUCTION_PLAN_FROZEN is True
    for name in ("P_TOKENS", "CODES_PER_M", "TRIALS_PER_CODE_P", "CODES_PER_TASK"):
        assert getattr(config_module, name) is not None
    require_production_plan_frozen()
    assert plan_for_phase("production")[1] == config_module.P_TOKENS

    monkeypatch.setattr(config_module, "PRODUCTION_PLAN_FROZEN", False)
    with pytest.raises(ProductionPlanNotFrozen):
        require_production_plan_frozen()
    with pytest.raises(ProductionPlanNotFrozen):
        plan_for_phase("production")

    monkeypatch.setattr(config_module, "PRODUCTION_PLAN_FROZEN", True)
    monkeypatch.setattr(config_module, "CODES_PER_M", None)
    with pytest.raises(ProductionPlanNotFrozen):
        require_production_plan_frozen()


def test_the_frozen_production_plan_is_the_rule_output():
    """The panel is the Validation 003 arithmetic, not a round number."""
    from data.expander_code.exp105.exp105_pipeline.config import (
        CODES_PER_M as codes, CODES_PER_TASK as blocks,
        TRIALS_PER_CODE_P as trials, P_TOKENS as grid, FALLBACK_P_TOKENS,
    )

    assert grid == FALLBACK_P_TOKENS, "the pilot found no sign change"
    assert set(codes) == set(M_VALUES)
    assert all(trials[m] == 6 for m in M_VALUES), (
        "the pilot measured sigma_c below its resolution, so trials sit at the cap"
    )
    for m in M_VALUES:
        assert codes[m] % blocks[m] == 0
    assert codes[3] > codes[8] > codes[7], (
        "panels must be unequal: an m=8 trial costs about 70 times an m=3 trial"
    )
    assert sum(codes.values()) == 167005


def test_q_is_fixed_at_the_contract_value(frozen_config):
    assert Q_TOKEN == "0.05"
    assert frozen_config["q_token"] == "0.05"
    assert normalize_q_token("0.05") == "0.05"
    assert normalize_q_token(0.05) == "0.05"
    for bad in ("0.5", 0.0, 0.06, "abc", None):
        with pytest.raises(ValueError):
            normalize_q_token(bad)


def test_pilot_grid_and_panel_are_the_frozen_ones(frozen_config):
    assert frozen_config["phase"] == "pilot"
    assert frozen_config["p_tokens"] == PILOT_P_TOKENS
    assert frozen_config["m_values"] == PILOT_M_VALUES == [3, 8]
    assert frozen_config["codes_per_m"] == {
        str(m): PILOT_CODES_PER_M for m in PILOT_M_VALUES
    }
    assert frozen_config["trials_per_code_p"] == {
        str(m): PILOT_TRIALS_PER_CODE_P for m in PILOT_M_VALUES
    }
    assert frozen_config["objective"] == (
        "bposd_ensemble_block_logical_failure_crossing_q005"
    )


def test_pilot_draws_from_its_own_ensemble_namespace(frozen_config):
    """No code that helps choose the frozen grid may also be measured on it."""
    assert frozen_config["registry_path"].endswith("ensemble_registry.pilot.v1.npz")
    assert frozen_config["registry_path"] != config_module.REGISTRY_PATH


def test_decoder_identity_is_exp104_plus_the_augmented_block(frozen_config):
    assert frozen_config["decoder"] == {
        "bp_method": "product_sum",
        "max_iter": "n_plus_nc",
        "schedule": "serial",
        "serial_schedule_order": "natural",
        "osd_method": "osd_0",
        "osd_order": 0,
        "omp_thread_count": 1,
        "augmented_matrix": "H_Z_hstack_identity",
        "error_channel": "p_on_data_q_on_checks",
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
    assert crossing["no_crossing_is_a_legitimate_terminal"] is True


def test_replay_policy_is_a_committed_subsample(frozen_config):
    assert frozen_config["replay"] == {
        "policy": "committed_random_subsample",
        "fraction": 0.10,
        "always_include_block_index": 0,
    }


def test_namespaces_are_disjoint_and_experiment_scoped(frozen_config):
    namespaces = frozen_config["namespaces"]
    assert len(set(namespaces.values())) == len(namespaces)
    assert all(value.startswith("exp105.") for value in namespaces.values())


def test_task_blocking_partitions_every_code_exactly_once(frozen_config):
    counts = {int(k): int(v) for k, v in frozen_config["codes_per_m"].items()}
    sizes = {int(k): int(v) for k, v in frozen_config["codes_per_task"].items()}
    per_m = tasks_per_m(counts, sizes)
    for m in frozen_config["m_values"]:
        seen = []
        for block in range(per_m[m]):
            indices = block_code_indices(frozen_config, m, block)
            assert len(indices) == sizes[m]
            seen.extend(indices)
        assert seen == list(range(counts[m]))


def test_unequal_panels_are_supported_by_the_blocking_arithmetic():
    """The whole point of the allocation rule is that panels differ per m."""
    assert tasks_per_m({3: 60000, 8: 7000}, {3: 500, 8: 5}) == {3: 120, 8: 1400}
    with pytest.raises(ValueError):
        tasks_per_m({3: 1001}, {3: 500})


def test_block_index_outside_the_plan_is_rejected(frozen_config):
    counts = {int(k): int(v) for k, v in frozen_config["codes_per_m"].items()}
    sizes = {int(k): int(v) for k, v in frozen_config["codes_per_task"].items()}
    per_m = tasks_per_m(counts, sizes)
    for m in frozen_config["m_values"]:
        with pytest.raises(ValueError):
            block_code_indices(frozen_config, m, per_m[m])
        with pytest.raises(ValueError):
            block_code_indices(frozen_config, m, -1)
    with pytest.raises(ValueError):
        block_code_indices(frozen_config, 5, 0)


def test_code_id_round_trip_and_range():
    for m in sorted(set(M_VALUES) | set(ANCHOR_M_VALUES)):
        assert parse_code_id(code_id(m, 0)) == (m, 0)
        assert parse_code_id(code_id(m, 999999)) == (m, 999999)
        with pytest.raises(ValueError):
            code_id(m, 1000000)
    assert code_id(3, 0) == "m03_c000000"
    for bad in ("m03_c0000", "m09_c000000", "x03_c000000", "m03_c00000"):
        with pytest.raises(ValueError):
            parse_code_id(bad)


def test_p_token_normalization_rejects_everything_off_grid(frozen_config):
    tokens = frozen_config["p_tokens"]
    assert normalize_p_token("0.02", tokens) == "0.02"
    assert normalize_p_token(0.02, tokens) == "0.02"
    for bad in ("0.11", "0.065", 0.11, 0.0, 1.0, "abc"):
        with pytest.raises(ValueError):
            normalize_p_token(bad, tokens)


@pytest.mark.parametrize("field", [
    "p_tokens", "m_values", "codes_per_m", "trials_per_code_p", "decoder",
    "ensemble", "namespaces", "bootstrap", "replay", "crossing", "preflight",
    "master_seed_hex", "objective", "registry_path", "codes_per_task",
    "q_token", "phase",
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

    copied = tmp_path / "noisy_mc.pilot.v1.json"
    payload = {
        key: value for key, value in frozen_config.items()
        if key not in {"config_sha256", "config_path"}
    }
    copied.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError):
        config_module.load_config(copied)


def test_phase_and_schema_must_agree(frozen_config):
    broken = copy.deepcopy(frozen_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    broken["schema_version"] = config_module.REMOTE_CONFIG_SCHEMA
    with pytest.raises(ValueError, match="phase and schema disagree"):
        ensure_config(broken)
