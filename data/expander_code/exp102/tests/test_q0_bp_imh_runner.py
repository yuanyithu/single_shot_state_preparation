"""Pre-raw regression checks for the frozen BP-IMH local runner."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "046_q0_bp_imh_local_v1_20260724" / "run_local_viability.py"
CONFIG = ROOT / "config" / "q0_bp_imh.local.v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("bp_imh_runner_regression", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bp_imh_runner_binds_complete_source_trees():
    runner = _load_runner()
    identity = runner._source_identity(CONFIG)
    assert identity["source_identity_sha256"]
    assert {
        "q0_bp_imh.py", "q0_bp_systematic.py", "q0_global.py",
        "q0_iid_importance.py", "registry.py", "seeds.py", "worker.py",
    } <= set(identity["exp102_pipeline_files"])
    assert {"prng.py", "model.py"} <= set(identity["exp101_src_files"])


def test_bp_imh_runner_canonicalizes_relative_output_path():
    runner = _load_runner()
    relative = RUNNER.parent.relative_to(Path.cwd())
    assert runner._canonical_output_dir(relative) == RUNNER.parent.resolve()
    assert runner._canonical_output_dir(RUNNER.parent.resolve()) == RUNNER.parent.resolve()


def test_full_label_d2_catches_equal_purity_equal_basis_marginal_distributions():
    runner = _load_runner()
    # A is uniform on {00,11}; B is uniform on {01,10}.  Both have purity 1/2
    # and zero one-bit character means, but their supports are disjoint.
    left = np.tile(np.asarray([0, 3], dtype=np.uint64), (8, 64))
    right = np.tile(np.asarray([1, 2], dtype=np.uint64), (8, 64))
    left_q = runner._q_top(left)
    right_q = runner._q_top(right)
    assert left_q["q_top"] == pytest.approx(right_q["q_top"], abs=1e-15)
    left_char = runner._character_traces(left).mean(axis=(0, 2))
    right_char = runner._character_traces(right).mean(axis=(0, 2))
    assert np.array_equal(left_char, right_char)
    d2 = runner._distribution_d2(left, right)
    assert d2["d2_norm"] == pytest.approx(1.0, abs=1e-15)
    assert d2["jackknife_se"] == pytest.approx(0.0, abs=1e-15)
