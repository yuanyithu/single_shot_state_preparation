from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("legacy_pilot_runner", "run_legacy_pilot.py")
analysis = load_module("legacy_pilot_analysis", "analyze_legacy_pilot.py")


def test_first_wave_has_exact_unique_task_grid():
    pairs = runner.parse_pairs("0.230:0.012,0.230:0.022,0.230:0.030")
    tasks = runner.build_tasks(pairs, [3, 7], 48, 950000)
    assert len(tasks) == 288
    assert len({task["task_id"] for task in tasks}) == 288
    assert all(task["lattice_size"] == 7 for task in tasks[:144])
    assert {task["disorder_seed"] for task in tasks} == set(range(950000, 950048))
    assert len({task["sample_seed"] for task in tasks}) == 288


def checkpoint_payload(task, config_hash):
    weights = np.arange(1, 9, dtype=np.float64)
    weights /= weights.sum()
    return {
        "task_id": np.array(task["task_id"]),
        "task_hash": np.array(task["task_hash"]),
        "config_hash": np.array(config_hash),
        "lattice_size": np.int64(task["lattice_size"]),
        "p_value": np.float64(task["p_value"]),
        "q_value": np.float64(task["q_value"]),
        "disorder_index": np.int64(task["disorder_index"]),
        "seed": np.int64(task["seed"]),
        "disorder_seed": np.int64(task["disorder_seed"]),
        "sample_seed": np.int64(task["sample_seed"]),
        "projection_mode": np.array("linear"),
        "disorder_seed_scope": np.array("disorder_index"),
        "disorder_realization_mode": np.array("rng_stream"),
        "delta_f": np.arange(8, dtype=np.float64),
        "weights": weights,
        "delta_f_stderr": np.full(8, 0.1),
        "weights_stderr": np.full(8, 0.01),
        "q_top": np.float64(0.5),
        "q_top_stderr": np.float64(0.01),
        "q_top_ci95": np.array([0.48, 0.52]),
        "grid_tv": np.float64(0.0),
        "grid_q_top_abs_diff": np.float64(0.0),
        "flags": np.array("PASS"),
        "wall_time_seconds": np.float64(1.0),
        "num_burn_in_sweeps": np.int64(512),
        "max_effective_num_burn_in_sweeps": np.int64(512),
    }


def test_atomic_checkpoints_merge_without_overwrite(tmp_path):
    tasks = runner.build_tasks([(0.23, 0.012)], [3, 7], 2, 950000)
    config_hash = "a" * 64
    for task in tasks:
        path = runner.checkpoint_path(tmp_path, task)
        runner.atomic_savez(path, **checkpoint_payload(task, config_hash))
        runner.validate_checkpoint(path, task, config_hash)
    for lattice_size in (3, 7):
        cell_tasks = [task for task in tasks if task["lattice_size"] == lattice_size]
        cell_path = runner.merge_cell(tmp_path, cell_tasks, config_hash)
        original_hash = runner.sha256_file(cell_path)
        assert runner.merge_cell(tmp_path, cell_tasks, config_hash) == cell_path
        assert runner.sha256_file(cell_path) == original_hash
        with np.load(cell_path, allow_pickle=False) as loaded:
            assert int(loaded["num_disorder_samples"]) == 2
            assert loaded["delta_f_per_disorder"].shape == (2, 8)


def stat(state: str, eligible: bool = True):
    return {"state": state, "eligible": eligible}


def test_adaptive_state_machine_all_authorized_branches():
    assert analysis.decide_q(0.012, {0.22: stat("NEG")})["next_p"] == 0.23
    assert analysis.decide_q(0.012, {0.22: stat("NEG"), 0.23: stat("POS")})[
        "next_p"
    ] == 0.225
    bracket = analysis.decide_q(
        0.012,
        {0.22: stat("NEG"), 0.225: stat("NEG"), 0.23: stat("POS")},
    )
    assert (bracket["bracket_low"], bracket["bracket_high"]) == (0.225, 0.23)
    assert analysis.decide_q(0.012, {0.22: stat("NEG"), 0.23: stat("NEG")})[
        "next_p"
    ] == 0.24
    assert analysis.decide_q(
        0.012,
        {0.22: stat("NEG"), 0.23: stat("NEG"), 0.24: stat("POS")},
    )["next_p"] == 0.235
    bracket = analysis.decide_q(
        0.012,
        {
            0.22: stat("NEG"),
            0.23: stat("NEG"),
            0.235: stat("POS"),
            0.24: stat("POS"),
        },
    )
    assert (bracket["bracket_low"], bracket["bracket_high"]) == (0.23, 0.235)
    stopped = analysis.decide_q(
        0.012,
        {0.22: stat("NEG"), 0.23: stat("UNRESOLVED", eligible=False)},
    )
    assert stopped["status"] == "STOP"


def test_fail_closed_shapes_and_flags():
    with pytest.raises(ValueError):
        analysis.softmax_w0(np.zeros((4, 7)))
    with pytest.raises(ValueError):
        analysis.validate_flags(np.array(["MISSING"]), Path("fixture.npz"))
