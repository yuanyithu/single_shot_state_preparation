import json
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import discovery
from data.expander_code.exp102.exp102_pipeline.discovery import (
    DISCOVERY_RAW_FIELDS,
    _confirmation_analysis,
    _screen_analysis,
    _trajectory_seed,
    _transport_analysis,
    _uniform_seed,
    confirmation_candidate,
    default_discovery_config,
    discovery_task_identity,
    load_discovery_config,
    run_discovery_cell,
    screen_candidates,
    transport_candidates,
    validate_discovery_raw,
)
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, atomic_npz, canonical_json
from data.expander_code.exp102.exp102_pipeline.pilot import _validate_raw as validate_formal_pilot_raw
from data.expander_code.exp102.exp102_pipeline.q0_pt import expected_swap_attempts
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
DISCOVERY_CONFIG_PATH = EXP102_ROOT / "config/discovery.v2.json"
SOURCE_COMMIT = "1" * 40


def test_discovery_config_freezes_ladders_and_prospective_fresh_panel(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    assert [item["ladder_id"] for item in config["ladders"]] == [
        "D0", "D1", "D2", "D3", "D4",
    ]
    assert len(config["screen"]["cells"]) == 9
    assert len(config["confirmation"]["cells"]) == 17
    assert len(config["confirmation"]["fresh_cells"]) == 8
    assert config["pt_contract_version"] == "exp102.q0_pt.v2"

    tampered = json.loads(DISCOVERY_CONFIG_PATH.read_text(encoding="ascii"))
    tampered["confirmation"]["fresh_cells"][0]["disorder_index"] += 1
    path = tmp_path / "tampered.json"
    atomic_json(path, tampered)
    with pytest.raises(ValueError, match="frozen protocol"):
        load_discovery_config(path, registry)


def test_disorder_seed_is_candidate_independent_but_transport_trajectory_is_bound():
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    code = next(item for item in registry["codes"] if item["code_id"] == "m06_c00")
    cell = config["transport"]["cells"][0]
    candidates = transport_candidates(config, ["D0"])
    assert _uniform_seed(registry, code, cell) == _uniform_seed(registry, code, cell)
    seeds = {
        _trajectory_seed(
            registry, config, SOURCE_COMMIT, "transport", code, cell, candidate, 0,
        )
        for candidate in candidates
    }
    assert len(seeds) == len(candidates)

    screen = screen_candidates(config)
    screen_seeds = {
        _trajectory_seed(
            registry, config, SOURCE_COMMIT, "screen", code,
            config["screen"]["cells"][2], candidate, 0,
        )
        for candidate in screen
    }
    assert len(screen_seeds) == 1


def _fake_result(candidate, k, seed):
    temperatures = candidate["num_temperatures"]
    measurements = candidate["measurement_rounds"]
    total_rounds = candidate["burn_rounds"] + measurements
    swap_attempts = expected_swap_attempts(
        temperatures, total_rounds, candidate["swap_sweeps_per_round"],
    )
    logical_attempts = np.full(
        (temperatures, k),
        total_rounds * candidate["sweeps_per_round"] * candidate["logical_move_repeat"],
        dtype=np.int64,
    )
    zero = np.zeros(temperatures, dtype=np.int64)
    return {
        "labels": np.full(measurements, np.uint64(seed & 1), dtype=np.uint64),
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_attempts // 2,
        "logical_attempts": logical_attempts,
        "logical_accepts": logical_attempts // 2,
        "hot_touches": 0,
        "hot_updated_visits": 0,
        "uncertified_round_trips": 0,
        "round_trips": 0,
        "sector_changing_round_trips": 0,
        "hot_touches_per_replica": zero.copy(),
        "hot_updated_visits_per_replica": zero.copy(),
        "uncertified_round_trips_per_replica": zero.copy(),
        "round_trips_per_replica": zero.copy(),
        "sector_changing_round_trips_per_replica": zero.copy(),
        "max_hard_coset_residual": 0,
    }


def test_discovery_raw_is_self_validating_and_formal_pilot_rejects_it(monkeypatch, tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    candidate = screen_candidates(config)[0]
    cell = config["screen"]["cells"][0]
    task = discovery_task_identity(
        registry, config, SOURCE_COMMIT, "screen", cell, candidate,
    )

    def fake_engine(model, frame, syndrome, p, pt_config, seed, initial_label, engine):
        assert engine == "numba"
        return _fake_result(candidate, model.k, seed)

    monkeypatch.setattr(discovery, "run_q0_pt_instance", fake_engine)
    output = tmp_path / "screen.npz"
    assert run_discovery_cell(
        REGISTRY_PATH, DISCOVERY_CONFIG_PATH, SOURCE_COMMIT, task, output,
    ) == "computed"
    record = validate_discovery_raw(output, registry, config, SOURCE_COMMIT)
    assert record["valid"]
    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == DISCOVERY_RAW_FIELDS

    formal_config = json.loads((EXP102_ROOT / "config/production.v1.json").read_text())
    formal_config["config_sha256"] = "not-used"
    with pytest.raises(ValueError, match="unknown pilot stage"):
        validate_formal_pilot_raw(output, registry, formal_config, SOURCE_COMMIT)

    with np.load(output, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["swap_sweeps_per_round"] = np.array(4, dtype=np.int16)
    atomic_npz(output, **arrays)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_discovery_raw(output, registry, config, SOURCE_COMMIT)


def _group(candidate, expected, core, all_pass=True, min_hot=2):
    return {
        "candidate": candidate,
        "present": expected,
        "missing": 0,
        "unexpected": 0,
        "valid": expected if all_pass else expected - 1,
        "all_pass": all_pass,
        "core_seconds": float(core),
        "wall_seconds_sum": float(core),
        "min_hot_updated_visits": min_hot,
        "failure_counts": {} if all_pass else {"instance_0:round_trips": 1},
    }


def test_discovery_selection_requires_two_distinct_confirmed_ladders():
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    groups = {}
    screen = screen_candidates(config)
    for candidate in screen:
        groups[("screen", canonical_json(candidate))] = _group(candidate, 9, 10)
    screen_result = _screen_analysis(groups, config)
    assert screen_result["passing_ladder_ids"] == ["D0", "D1", "D2", "D3", "D4"]

    transport = transport_candidates(config, ["D0", "D1"])
    for candidate in transport:
        passed = candidate["swap_sweeps_per_round"] == 16
        groups[("transport", canonical_json(candidate))] = _group(
            candidate, 2, 2 if candidate["ladder_id"] == "D1" else 3,
            all_pass=passed,
        )
    transport_result = _transport_analysis(groups, config, ["D0", "D1"])
    assert {item["ladder_id"] for item in transport_result["ranked_candidates"]} == {
        "D0", "D1",
    }

    for transport_candidate in transport_result["ranked_candidates"]:
        candidate = confirmation_candidate(
            config, transport_candidate["ladder_id"], 16, (2000, 8000),
        )
        core = 100 if candidate["ladder_id"] == "D0" else 90
        groups[("confirmation", canonical_json(candidate))] = _group(candidate, 17, core)
    confirmation = _confirmation_analysis(
        groups, config, transport_result["ranked_candidates"],
    )
    assert confirmation["primary"]["ladder_id"] == "D1"
    assert confirmation["backup"]["ladder_id"] == "D0"
    assert confirmation["complete"]
