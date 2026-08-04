"""Focused regression checks for the frozen UASRE local runner boundary."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "024_q0_aux_stabilizer_v0_20260724" / "run_local_viability.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("uasre_runner_regression", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uasre_runner_exposes_complete_score_for_raw_revalidation():
    runner = _load_runner()
    assert callable(runner.collapsed_complete_score)


def test_uasre_runner_binds_audit_and_all_transition_dependencies():
    runner = _load_runner()
    binding = runner._source_binding(runner.DEFAULT_CONFIG)
    assert binding["source_binding_sha256"]
    assert {
        "independent_raw_audit.py",
        "q0_hgp_aux_stabilizer.py",
        "q0_hgp_aux_stabilizer_pt.py",
        "q0_hgp_uniform_anchor_pt.py",
        "q0_hgp_collapsed.py",
        "q0_hgp_full_row_gibbs.py",
        "q0_global.py",
        "exp101_bridge.py",
        "registry.json",
    } <= set(binding["files"])
    assert len(binding["exp101_src_files"]) >= 1
