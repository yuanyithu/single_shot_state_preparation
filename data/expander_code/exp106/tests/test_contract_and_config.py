"""The frozen contract must reject every mutation of itself.

exp106 has two config phases. The locating pilot is frozen now; the production
plan is deliberately absent from the source until Validation 003 evaluates the
contract's freezing rules on pilot measurements. Both facts are asserted here,
because "not yet frozen" has to be a state the code enforces rather than a note
in a document.
"""

import copy

import pytest

from data.expander_code.exp106.exp106_pipeline import config as config_module
from data.expander_code.exp106.exp106_pipeline.config import (
    CENSUS_ONLY_M_VALUES,
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


PRODUCTION_CONSTANTS = ("P_TOKENS", "CODES_PER_M", "TRIALS_PER_CODE_P", "CODES_PER_TASK")


def test_the_production_plan_freeze_is_all_or_nothing(monkeypatch):
    """The freeze is a state the code enforces, not a note in a document.

    Before Validation 003 every production entry point must refuse; after it,
    every production constant must be present. Half-frozen is the dangerous
    state and is what this asserts cannot exist -- in either direction, so this
    test is the same before and after the freeze.
    """
    frozen = config_module.PRODUCTION_PLAN_FROZEN
    present = [getattr(config_module, name) is not None for name in PRODUCTION_CONSTANTS]
    assert all(present) == frozen, "the flag and the constants must agree"

    if frozen:
        require_production_plan_frozen()
        assert plan_for_phase("production")[1] == config_module.P_TOKENS
    else:
        with pytest.raises(ProductionPlanNotFrozen):
            require_production_plan_frozen()
        with pytest.raises(ProductionPlanNotFrozen):
            plan_for_phase("production")

    monkeypatch.setattr(config_module, "PRODUCTION_PLAN_FROZEN", False)
    with pytest.raises(ProductionPlanNotFrozen):
        require_production_plan_frozen()

    monkeypatch.setattr(config_module, "PRODUCTION_PLAN_FROZEN", True)
    monkeypatch.setattr(config_module, "CODES_PER_M", None)
    with pytest.raises(ProductionPlanNotFrozen):
        require_production_plan_frozen()


def test_the_pilot_phase_is_usable_before_the_freeze():
    """The pilot is what freezes the plan, so it cannot depend on it."""
    m_values, p_tokens, codes, trials, blocks = plan_for_phase("pilot")
    assert m_values == PILOT_M_VALUES
    assert p_tokens == PILOT_P_TOKENS
    assert codes == {m: PILOT_CODES_PER_M for m in PILOT_M_VALUES}
    assert trials == {m: PILOT_TRIALS_PER_CODE_P for m in PILOT_M_VALUES}
    assert blocks == PILOT_CODES_PER_TASK
    # ... and so is its remote form, which is how compute-host costs get measured
    # before the allocation rule that spends them is evaluated.
    assert plan_for_phase("pilot_remote") == plan_for_phase("pilot")


@pytest.mark.skipif(
    not config_module.PRODUCTION_PLAN_FROZEN,
    reason="the production plan is frozen by Validation 003",
)
def test_the_frozen_production_plan_is_the_rule_output():
    """The panel must be the section 6 arithmetic, not a round number.

    Deliberately an assertion about the *rule*, not about a remembered answer.
    exp105's version of this test hard-coded its own pilot outcome -- a
    fallback grid, six trials everywhere, 17,617 codes -- which would be a
    false gate here: at q = 0.01 the pilot may well find a bracket, and if it
    does the correct plan is the one that brackets it.
    """
    from data.expander_code.exp106.exp106_pipeline.config import (
        CODES_PER_M as codes, CODES_PER_TASK as blocks,
        TRIALS_PER_CODE_P as trials, P_TOKENS as grid,
        FALLBACK_P_TOKENS, PRODUCTION_GRID_POINTS, TRIALS_PER_CODE_P_RANGE,
    )

    assert 2 <= len(grid) <= PRODUCTION_GRID_POINTS
    assert grid == sorted(grid, key=float), "the grid must be ordered"
    assert len(set(grid)) == len(grid), "the grid must be deduplicated"
    if grid != FALLBACK_P_TOKENS:
        assert len(grid) == PRODUCTION_GRID_POINTS, (
            "the bracket branch produces exactly the frozen number of points"
        )

    assert set(codes) == set(M_VALUES) == set(trials) == set(blocks)
    low, high = TRIALS_PER_CODE_P_RANGE
    assert all(low <= trials[m] <= high for m in M_VALUES)
    for m in M_VALUES:
        assert codes[m] % blocks[m] == 0, "blocking must partition the panel"
        assert codes[m] >= blocks[m], "every m must have at least one whole task"
    assert codes[3] > codes[8], (
        "panels must be unequal: an m=8 trial costs about fifty times an m=3 trial, "
        "so equal panels would spend the budget on the smaller variance term"
    )


def test_q_is_fixed_at_the_contract_value(frozen_config):
    assert Q_TOKEN == "0.01"
    assert frozen_config["q_token"] == "0.01"
    assert normalize_q_token("0.01") == "0.01"
    assert normalize_q_token(0.01) == "0.01"
    # 0.05 is exp105's value and 0.0 is exp104's. Both are rejected here, which
    # is the point: no filename encodes q, so the config is the only thing
    # standing between the two experiments' streams.
    for bad in ("0.1", 0.0, 0.05, 0.011, "abc", None):
        with pytest.raises(ValueError):
            normalize_q_token(bad)


def test_pilot_grid_and_panel_are_the_frozen_ones(pilot_config):
    assert pilot_config["phase"] == "pilot"
    assert pilot_config["p_tokens"] == PILOT_P_TOKENS
    assert pilot_config["m_values"] == PILOT_M_VALUES == [3, 8]
    assert pilot_config["codes_per_m"] == {
        str(m): PILOT_CODES_PER_M for m in PILOT_M_VALUES
    }
    assert pilot_config["trials_per_code_p"] == {
        str(m): PILOT_TRIALS_PER_CODE_P for m in PILOT_M_VALUES
    }
    assert pilot_config["objective"] == (
        "bposd_ensemble_block_logical_failure_crossing_q001"
    )


def test_pilot_draws_from_its_own_ensemble_namespace(pilot_config):
    """No code that helps choose the frozen grid may also be measured on it."""
    assert pilot_config["registry_path"].endswith("ensemble_registry.pilot.v1.npz")
    assert pilot_config["registry_path"] != config_module.REGISTRY_PATH


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
    assert all(value.startswith("exp106.") for value in namespaces.values())


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
        block_code_indices(frozen_config, 2, 0)


def test_code_id_round_trip_and_range():
    for m in sorted(set(M_VALUES) | set(CENSUS_ONLY_M_VALUES)):
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
    # Named by position: the two phases have different grids.
    assert normalize_p_token(tokens[0], tokens) == tokens[0]
    assert normalize_p_token(float(tokens[0]), tokens) == tokens[0]
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


def test_phase_and_schema_must_agree(pilot_config):
    broken = copy.deepcopy(pilot_config)
    broken.pop("config_sha256", None)
    broken.pop("config_path", None)
    broken["schema_version"] = config_module.REMOTE_CONFIG_SCHEMA
    with pytest.raises(ValueError, match="phase and schema disagree"):
        ensure_config(broken)


def test_every_qualification_group_path_exists():
    """A mistyped path would silently shrink the gate rather than fail it.

    pytest exits 0 with "no tests ran" on a path that matches nothing in some
    configurations, and the expected-pass check would then be comparing against
    a count nobody measured. Checking the paths resolve is cheap; discovering
    this on nd-3 after a deployment round trip is not.
    """
    from pathlib import Path

    from data.expander_code.exp106.exp106_pipeline.remote_cli import (
        QUALIFICATION_EXPECTED_PASSES,
        QUALIFICATION_GROUPS,
    )

    repo_root = Path(__file__).resolve().parents[4]
    names = [name for name, _ in QUALIFICATION_GROUPS]
    assert names == sorted(set(names), key=names.index), "group names must be unique"
    assert set(names) == set(QUALIFICATION_EXPECTED_PASSES)
    for name, paths in QUALIFICATION_GROUPS:
        assert paths, f"{name} has no paths"
        for path in paths:
            assert (repo_root / path).exists(), f"{name}: missing {path}"


def test_qualification_refuses_unmeasured_pass_counts(monkeypatch):
    """An unset count must block the gate, not sail through it.

    exp106's own count can only be measured once the production plan is frozen,
    because 29 tests skip until then and qualification allows no skips. This
    test is written to hold on both sides of that freeze, so it is not something
    that has to be edited at exactly the moment it matters.
    """
    from data.expander_code.exp106.exp106_pipeline import remote_cli

    counts = dict(remote_cli.QUALIFICATION_EXPECTED_PASSES)
    if config_module.PRODUCTION_PLAN_FROZEN:
        assert all(isinstance(value, int) and value > 0 for value in counts.values())
        remote_cli.require_expected_pass_counts()
    else:
        assert counts["exp106"] is None
        with pytest.raises(ValueError, match="pass counts are unset"):
            remote_cli.require_expected_pass_counts()

    monkeypatch.setattr(
        remote_cli, "QUALIFICATION_EXPECTED_PASSES", dict(counts, exp106=None),
    )
    with pytest.raises(ValueError, match="pass counts are unset"):
        remote_cli.require_expected_pass_counts()


def test_the_frozen_execution_profile_has_one_definition():
    """cli.py and _validate must read the same numbers, not two copies.

    exp105 kept two copies of the execution profile and two copies of the
    benchmark grid selection; the second pair drifted and raised a KeyError in
    the middle of a resource gate.
    """
    from data.expander_code.exp106.exp106_pipeline.config import (
        COMPUTE_HOST, GENERATION_BUDGET_CORE_HOURS,
        REMOTE_EXECUTION_DEFAULTS, REMOTE_EXECUTION_FIELDS,
    )

    assert set(REMOTE_EXECUTION_DEFAULTS) | {"conda_environment"} == REMOTE_EXECUTION_FIELDS
    # The host is named once and read everywhere; it was spelled inline at six
    # sites before the run moved from nd-3 to nd-2.
    assert REMOTE_EXECUTION_DEFAULTS["compute_host"] == COMPUTE_HOST
    workers = REMOTE_EXECUTION_DEFAULTS["num_workers"]
    assert isinstance(workers, int) and workers > 0
    # The caps are discipline 11 applied to the frozen generation budget:
    # 2 x (G + 0.1 G replay + 1 analysis + 1 overhead), and (G + 0.1 G)/workers + 2.
    budget = GENERATION_BUDGET_CORE_HOURS
    assert REMOTE_EXECUTION_DEFAULTS["reserve_multiplier"] == 2.0
    assert REMOTE_EXECUTION_DEFAULTS["stage_core_hour_cap"] >= 2 * (budget * 1.1 + 2)
    assert REMOTE_EXECUTION_DEFAULTS["stage_wall_hour_cap"] >= (budget * 1.1) / workers + 2


def test_the_compute_host_is_named_once(monkeypatch):
    """Changing the host must not require finding six string literals.

    The nd-3 resource gate blocked and the run moved to nd-2 on the same day.
    Every consumer of the host -- the execution profile, the expected remote
    environment, and the allocation rule's refusal to accept costs from the wrong
    machine -- now reads one constant.
    """
    from data.expander_code.exp106.exp106_pipeline import pilot as pilot_module
    from data.expander_code.exp106.exp106_pipeline.config import COMPUTE_HOST
    from data.expander_code.exp106.exp106_pipeline.identity import DEVICE_BY_HOSTNAME

    assert COMPUTE_HOST in DEVICE_BY_HOSTNAME
    assert DEVICE_BY_HOSTNAME[COMPUTE_HOST] == COMPUTE_HOST
    assert pilot_module.COMPUTE_HOST == COMPUTE_HOST


def test_the_pilot_registry_is_the_shape_the_constant_declares(pilot_registry):
    """`PILOT_REGISTRY_CODES_PER_M` must describe the file, not just intentions.

    The pilot *panel* is the primary pair; the pilot *registry* carries all six
    sizes so that the nd-3 cost benchmark can time every one of them before the
    production registry exists. Those extra rows are inert -- never scanned,
    never aggregated -- which is exactly the kind of thing that rots silently if
    nothing checks it.
    """
    from data.expander_code.exp106.exp106_pipeline.config import (
        PILOT_M_VALUES, PILOT_REGISTRY_CODES_PER_M,
    )

    counts = {
        int(m): int(c) for m, c in pilot_registry["metadata"]["codes_per_m"].items()
    }
    assert counts == PILOT_REGISTRY_CODES_PER_M
    for m in PILOT_M_VALUES:
        assert counts[m] == PILOT_CODES_PER_M
    for m in set(M_VALUES) - set(PILOT_M_VALUES):
        assert 0 < counts[m] < PILOT_CODES_PER_M, (
            "diagnostic sizes need enough codes to bound a cost and no more"
        )
    assert set(counts) == set(M_VALUES), (
        "the cost benchmark times every production size"
    )


def test_every_config_schema_has_exactly_one_canonical_filename():
    """One lookup, not a hard-coded name at each call site.

    exp106 has two *remote* schemas -- the production config and the pilot
    remote config the nd-3 cost benchmark runs under -- so every place that
    spelled `noisy_mc.remote.v1.json` inline silently meant "production only".
    That assumption was wrong in three places, and the deployment verifier's
    copy of it failed on nd-3 after a full bundle round trip.
    """
    from pathlib import Path

    from data.expander_code.exp106.exp106_pipeline.config import (
        SCHEMA_BY_PHASE, canonical_config_filename,
    )

    config_dir = Path(__file__).resolve().parents[1] / "config"
    names = set()
    for phase, schema in SCHEMA_BY_PHASE.items():
        filename = canonical_config_filename(schema)
        assert filename not in names, f"{filename} claimed by two schemas"
        names.add(filename)
        # the pilot phases exist now; the production ones appear at the freeze
        if phase in ("pilot", "pilot_remote"):
            assert (config_dir / filename).is_file(), f"{phase}: {filename} missing"

    with pytest.raises(ValueError, match="unknown exp106 config schema"):
        canonical_config_filename("exp106.config.not.a.schema")


def test_the_deployment_ships_both_remote_configs_and_both_predecessors():
    """The archive has to carry what the gates on nd-3 will reach for.

    exp105's first nd-3 qualification failed because the production registry was
    gitignored and therefore absent from the archive; a file the deployed tree
    needs but does not have is the same class of failure.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "_exp106_deploy",
        root / "data/expander_code/exp106/deployment/build_remote_deployment.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = set(module.SOURCE_PATHS)
    assert "data/expander_code/exp106/config" in paths, "both remote configs"
    # the two equality gates decode against these packages, and both suites are
    # qualification groups
    for needed in (
        "data/expander_code/exp104/exp104_pipeline",
        "data/expander_code/exp104/config",
        "data/expander_code/exp105/exp105_pipeline",
        "data/expander_code/exp105/config",
        "data/expander_code/exp105/tests",
    ):
        assert needed in paths, needed
    for entry in paths:
        assert (root / entry).exists(), f"SOURCE_PATHS names a missing path: {entry}"
