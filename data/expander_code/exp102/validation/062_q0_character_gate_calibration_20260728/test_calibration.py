import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibration = load_module("q0_character_gate_calibration", "run_calibration.py")
auditor = load_module("q0_character_gate_auditor", "audit_report.py")


def load_config():
    return json.loads((ROOT / "calibration_config.json").read_text(encoding="ascii"))


def bit_rows(num_bits):
    values = np.arange(1 << num_bits, dtype=np.uint64)
    return ((values[:, None] >> np.arange(num_bits, dtype=np.uint64)) & 1).astype(
        np.uint8
    )


def state_key(state):
    return np.packbits(state, bitorder="little").tobytes()


def independent_hard_coset(model, syndrome):
    all_states = bit_rows(model.num_qubits)
    residual = (
        all_states.astype(np.int64) @ model.H_check.T.astype(np.int64) % 2
    ).astype(np.uint8)
    return all_states[np.all(residual == syndrome[None, :], axis=1)]


@pytest.mark.parametrize(
    "H_values",
    (
        [[1, 1, 1]],
        [[1, 1, 0], [0, 1, 1]],
    ),
)
@pytest.mark.parametrize("syndrome_kind", ("zero", "nonzero"))
def test_n10_n13_full_hard_coset_weights_and_labels_match_independent_bruteforce(
        H_values, syndrome_kind):
    H = np.asarray(H_values, dtype=np.uint8)
    model, frame = calibration.build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if syndrome_kind == "nonzero":
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)

    expected = independent_hard_coset(model, syndrome)
    actual = calibration.coset_states(model, syndrome)
    expected_keys = {state_key(state) for state in expected}
    actual_keys = {state_key(state) for state in actual}
    assert actual_keys == expected_keys
    assert len(actual_keys) == len(actual)

    direct_logical_bits = (
        actual.astype(np.int64) @ frame.W_basis.T.astype(np.int64) % 2
    ).astype(np.uint8)
    direct_logical = np.asarray([
        sum(int(bit) << index for index, bit in enumerate(bits))
        for bits in direct_logical_bits
    ], dtype=np.int64)
    actual_logical = calibration.integer_labels(model, frame, actual)
    assert np.array_equal(actual_logical, direct_logical)

    r, n = H.shape
    direct_B_bits = actual[:, n * n:n * n + r * r]
    direct_B = np.asarray([
        sum(int(bit) << index for index, bit in enumerate(bits))
        for bits in direct_B_bits
    ], dtype=np.int64)
    assert np.array_equal(calibration.b_labels(H, actual), direct_B)

    weights = actual.sum(axis=1, dtype=np.int64)
    for p in (0.04, 0.10, 0.25):
        ratio = p / (1.0 - p)
        direct = np.asarray([ratio ** int(weight) for weight in weights])
        direct /= direct.sum()
        observed = calibration.posterior_probabilities(actual, p)
        direct_by_state = dict(zip(map(state_key, actual), direct))
        observed_by_state = dict(zip(map(state_key, actual), observed))
        assert direct_by_state.keys() == observed_by_state.keys()
        for key in direct_by_state:
            assert np.isclose(direct_by_state[key], observed_by_state[key])


def test_exact_scenario_grid_and_complete_catalog_sizes():
    scenarios = calibration.build_exact_scenarios(load_config())
    assert len(scenarios) == 24
    assert {row["catalog"] for row in scenarios} == {"logical", "collapsed_B"}
    assert max(row["catalog_size"] for row in scenarios) == 15
    for row in scenarios:
        assert np.isclose(row["probabilities"].sum(), 1.0)
        assert np.all(row["probabilities"] > 0.0)
        assert row["signs"].shape == (
            row["probabilities"].size, row["catalog_size"],
        )
        assert set(np.unique(row["signs"])) <= {-1.0, 1.0}


def test_exact_logical_purity_and_collapsed_B_induced_q_top_are_accounted():
    config = load_config()
    scenarios = calibration.build_exact_scenarios(config)
    logical = next(row for row in scenarios if row["catalog"] == "logical")
    collapsed_B = next(row for row in scenarios if row["catalog"] == "collapsed_B")
    logical_row = calibration.evaluate_exact_row(
        logical, 0.06, 8, 4, 32, 1.0, config, "config", "selection", 0,
    )
    b_row = calibration.evaluate_exact_row(
        collapsed_B, 0.06, 8, 4, 32, 1.0, config, "config", "selection", 1,
    )
    assert logical_row["q_top_accounting"] == "exact_full_logical_catalog"
    assert np.isclose(
        logical_row["true_character_purity_left"], logical_row["true_q_top_left"],
    )
    assert np.isclose(
        logical_row["true_character_purity_right"], logical_row["true_q_top_right"],
    )
    assert b_row["q_top_accounting"] == (
        "induced_from_retained_conditional_logical_signs"
    )
    shifted, _, _, _ = calibration.tilted_probabilities(
        collapsed_B["probabilities"], collapsed_B["signs"], 0.06,
    )
    expected_q_top = np.mean((shifted @ collapsed_B["logical_signs"]) ** 2)
    assert np.isclose(b_row["true_q_top_right"], expected_q_top)
    assert len(b_row["true_character_shifts"]) == b_row["catalog_size"]
    expected_shift = np.asarray(b_row["base_character_means"]) - np.asarray(
        b_row["shifted_character_means"]
    )
    assert np.allclose(b_row["true_character_shifts"], expected_shift)


def test_distributed_nonphysical_stress_exposes_purity_blind_spot_compactly():
    config = load_config()
    catalog = next(
        row for row in config["synthetic_multiplicity_stress"]["catalogs"]
        if row["size"] == 4160
    )
    row = calibration.evaluate_synthetic_row(
        catalog, 0.8, 0.04, "distributed_all_characters", 4, 4, 16, 1.0,
        config, "config", "selection",
    )
    assert row["physical_distribution"] is False
    assert row["shifted_character_mean_minimum"] == pytest.approx(0.76)
    assert row["shifted_character_mean_maximum"] == pytest.approx(0.76)
    assert row["true_character_purity_delta"] == pytest.approx(0.0624)
    assert row["true_max_abs_character_shift"] == pytest.approx(0.04)
    assert row["true_character_shift_summary"]["nonzero_count"] == 4160
    assert "true_character_shifts" not in row
    assert "base_character_means" not in row
    shifts = [0.8 - (0.8 - 0.04)] * 4160
    assert row["true_character_shift_vector_sha256"] == calibration.vector_sha256(
        shifts
    )
    tolerance = config["candidate_rule"]["tolerance"]
    assert 2.0 * tolerance == 0.08
    assert row["true_character_purity_delta"] <= 2.0 * tolerance


def test_wilson_is_one_sided_and_recomputed_independently():
    value = calibration.wilson_lower(98, 100, 0.95)
    assert 0.90 < value < 0.98
    assert value == pytest.approx(auditor.wilson_lower(98, 100, 0.95))
    assert calibration.wilson_lower(100, 100, 0.95) < 1.0


def test_role_multipliers_are_shared_across_base_means_and_catalog_scales():
    config = load_config()
    small = copy.deepcopy(config)
    small["replications"]["calibration_trials"] = 8
    result = calibration.common_multipliers(
        small, "config", [], trajectories=8, draws=512,
    )
    assert set(result["role_multipliers"]) == {"logical", "collapsed_B"}
    for role, multiplier in result["role_multipliers"].items():
        role_rows = [
            row for row in result["scenario_quantiles"]
            if row["simultaneous_role"] == role
        ]
        assert {row["base_character_mean"] for row in role_rows} == {0.0, 0.8}
        assert multiplier == max(row["quantile"] for row in role_rows)
    assert {
        row["catalog_size"] for row in result["scenario_quantiles"]
    } == {688, 4160}


def test_fixed_cost_selection_and_fresh_confirmation_cannot_mix_points():
    config = load_config()
    ordered = calibration.ordered_operating_points(config)
    assert ordered == sorted(ordered, key=lambda row: (row[0] * row[1], row[0], row[1]))
    points = [
        {"eligible": False, "trajectory_count": 8, "draws_per_trajectory": 8192,
         "role_multipliers": {"logical": 5.0, "collapsed_B": 4.0}},
        {"eligible": True, "trajectory_count": 16, "draws_per_trajectory": 8192,
         "role_multipliers": {"logical": 5.1, "collapsed_B": 4.1}},
        {"eligible": True, "trajectory_count": 32, "draws_per_trajectory": 8192,
         "role_multipliers": {"logical": 5.2, "collapsed_B": 4.2}},
    ]
    selected = calibration.first_eligible_operating_point(points)
    assert selected is points[1]
    matching = dict(selected)
    assert calibration.confirmation_matches_selection(selected, matching)
    incompatible_point = dict(matching, draws_per_trajectory=16384)
    assert not calibration.confirmation_matches_selection(selected, incompatible_point)
    incompatible_multiplier = copy.deepcopy(matching)
    incompatible_multiplier["role_multipliers"]["logical"] += 0.1
    assert not calibration.confirmation_matches_selection(
        selected, incompatible_multiplier,
    )
    failed_confirmation = dict(matching, eligible=False)
    assert not calibration.confirmation_matches_selection(
        selected, failed_confirmation,
    )


def test_confirmation_seed_namespace_is_fresh():
    config = load_config()
    common = (
        config["seed_namespace"], "config", "synthetic", 16, 8192,
        "synthetic_logical_4160", 0.8, "distributed_all_characters", 0.04,
    )
    selection = calibration.derive_seed(common[0], common[1], "selection", *common[2:])
    confirmation = calibration.derive_seed(
        common[0], common[1], "confirmation", *common[2:],
    )
    calibration_seed = calibration.derive_seed(
        common[0], common[1], "common-z-synthetic", *common[3:5], 4160, 0.8,
    )
    assert len({selection, confirmation, calibration_seed}) == 3


def test_registered_stress_sizes_roles_and_required_categories_are_frozen():
    config = load_config()
    catalogs = config["synthetic_multiplicity_stress"]["catalogs"]
    assert {row["size"] for row in catalogs} == {15, 163, 511, 688, 4160}
    by_size = {row["size"]: row["role"] for row in catalogs}
    assert by_size == {
        15: "collapsed_B", 163: "collapsed_B", 511: "logical",
        688: "collapsed_B", 4160: "logical",
    }
    calibration.validate_config(config)


def test_independent_auditor_reconstructs_complete_point_and_seed_grid():
    config = load_config()
    small = copy.deepcopy(config)
    small["replications"]["selection_trials"] = 6
    scenarios = calibration.build_exact_scenarios(small)
    multipliers = {"collapsed_B": 1.25, "logical": 1.5}
    config_sha = calibration.sha256_file(ROOT / "calibration_config.json")
    rows = calibration.evaluate_point(
        small, config_sha, scenarios, trajectories=4, draws=64,
        role_multipliers=multipliers, stage="selection",
    )
    auditor.verify_rows(
        rows, small, "selection", trajectories=4, draws=64,
        role_multipliers=multipliers,
    )
    runner_summary = calibration.summarize_operating_point(
        rows, small, trajectories=4, draws=64,
        role_multipliers=multipliers, stage="selection",
    )
    audit_summary = auditor.summarize(
        rows, small, trajectories=4, draws=64,
        role_multipliers=multipliers, stage="selection",
    )
    assert calibration.canonical(runner_summary) == auditor.canonical(audit_summary)


def test_config_self_hash_and_all_source_artifact_hashes_are_bound():
    config = load_config()
    calibration.verify_config_self_hash(config)
    assert set(config["source_artifacts"]) == {
        "auditor", "readme", "red_team", "runner", "tests",
    }
    for spec in config["source_artifacts"].values():
        path = calibration._exp102_path(spec["path"])
        assert path.is_file()
        assert calibration.sha256_file(path) == spec["sha256"]


def test_dirty_worktree_and_bytecode_are_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration, "_git", lambda args, text=True: "?? rogue.txt\n"
        if args[0] == "status" else "",
    )
    with pytest.raises(RuntimeError, match="completely clean worktree"):
        calibration.require_completely_clean_worktree()

    clean = tmp_path / "clean"
    clean.mkdir()
    calibration.reject_validation_bytecode(clean)
    bytecode = clean / "__pycache__"
    bytecode.mkdir()
    (bytecode / "module.pyc").write_bytes(b"not bytecode")
    with pytest.raises(RuntimeError, match="contains bytecode"):
        calibration.reject_validation_bytecode(clean)


def test_auditor_is_independent_and_sources_are_ascii_python():
    audit_source = (ROOT / "audit_report.py").read_text(encoding="ascii")
    assert "run_calibration" not in audit_source
    for filename in ("run_calibration.py", "audit_report.py", "test_calibration.py"):
        source = (ROOT / filename).read_text(encoding="ascii")
        ast.parse(source, filename=filename)
