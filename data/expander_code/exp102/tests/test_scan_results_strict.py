from pathlib import Path
import shutil

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import PHYSICS_VERSION, PT_VERSION, SCAN_VERSION
from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, atomic_npz, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline import scan_results


EXP102_ROOT = Path(__file__).resolve().parents[1]


def _bundle(tmp_path, m_status_shape=(6, 7)):
    registry_source = EXP102_ROOT / "registry/registry.json"
    config_source = EXP102_ROOT / "config/production.v1.json"
    registry = load_registry(registry_source)
    config = load_config(config_source)
    shutil.copy2(registry_source, tmp_path / "registry.json")
    shutil.copy2(config_source, tmp_path / "production.v1.json")
    candidate = {
        "p_hot": 0.49, "num_temperatures": 96, "gamma": 1.0,
        "burn_rounds": 500, "measurement_rounds": 2000,
        "sweeps_per_round": 1, "logical_move_repeat": 1,
    }
    frozen = {
        "status": "FROZEN_HELD_OUT_PASS", "engine": "numba",
        "source_commit": "1" * 40,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "pilot_report_sha256": "2" * 64, "raw_evidence_sha256": "3" * 64,
        "by_m": {str(m): candidate for m in range(3, 9)},
        "held_out_attempt_by_m": {str(m): 0 for m in range(3, 9)},
    }
    atomic_json(tmp_path / "frozen.json", frozen)

    mu_code = np.arange(48, dtype=np.float64)[:, None] * 0.001
    mu_code = mu_code + np.arange(7, dtype=np.float64)[None, :] * 0.01
    mu_m = np.empty((6, 7), dtype=np.float64)
    error = np.empty((6, 7), dtype=np.float64)
    for m_index in range(6):
        for p_index in range(7):
            values = mu_code[m_index * 8:(m_index + 1) * 8, p_index]
            mu_m[m_index, p_index] = np.mean(values)
            error[m_index, p_index] = np.std(values, ddof=1) / np.sqrt(8)
    result = tmp_path / "exp102_results.npz"
    atomic_npz(
        result,
        physics_contract_version=np.array(PHYSICS_VERSION),
        pt_contract_version=np.array(PT_VERSION), scan_contract_version=np.array(SCAN_VERSION),
        registry_sha256=np.array(registry["registry_sha256"]),
        config_sha256=np.array(config["config_sha256"]),
        frozen_config_sha256=np.array(sha256_json(frozen)), engine=np.array("numba"),
        source_commit=np.array(frozen["source_commit"]),
        p_values=np.asarray(config["p_values"], dtype=np.float64),
        m_values=np.asarray(config["m_values"], dtype=np.int8),
        code_ids=np.asarray([row["code_id"] for row in registry["codes"]], dtype="U8"),
        mu_code=mu_code, mu_m=mu_m, errorbar_between_code_sem=error,
        code_status=np.full((48, 7), "REPORTABLE", dtype="U24"),
        m_status=np.full(m_status_shape, "REPORTABLE", dtype="U24"),
        present_disorders=np.full((48, 7), 128, dtype=np.int16),
        valid_disorders=np.full((48, 7), 128, dtype=np.int16),
        main_errorbar_definition=np.array("std(code_means,ddof=1)/sqrt(8)"),
    )
    manifest = {
        "aggregation_version": "exp102.aggregation.v1", "result_file": result.name,
        "result_sha256": sha256_file(result),
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": sha256_json(frozen), "engine": "numba",
        "source_commit": frozen["source_commit"], "planned_tasks": 6144,
        "present_tasks": 6144,
        "main_errorbar_definition": "std(code_means,ddof=1)/sqrt(8)",
        "registry_file_sha256": sha256_file(tmp_path / "registry.json"),
        "config_file_sha256": sha256_file(tmp_path / "production.v1.json"),
        "frozen_file_sha256": sha256_file(tmp_path / "frozen.json"),
        "pilot_report_sha256": frozen["pilot_report_sha256"],
        "pilot_raw_evidence_sha256": frozen["raw_evidence_sha256"],
        "held_out_attempt_by_m": frozen["held_out_attempt_by_m"],
        "production_raw_manifest_sha256": {
            "nd-1": "4" * 64, "nd-2": "5" * 64, "nd-3": "6" * 64,
        },
    }
    atomic_json(tmp_path / "aggregation_manifest.json", manifest)
    return result


def test_publication_loader_recomputes_point_eligibility(monkeypatch, tmp_path):
    monkeypatch.setattr(scan_results, "_verify_recorded_git_support", lambda *_: None)
    result = _bundle(tmp_path)
    loaded = scan_results.load_exp102_publication_q_top(result)
    assert loaded["q_top"].shape == (6, 7)
    assert np.all(loaded["point_mask"])


def test_publication_loader_rejects_broadcastable_status_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(scan_results, "_verify_recorded_git_support", lambda *_: None)
    result = _bundle(tmp_path, m_status_shape=(1, 1))
    with pytest.raises(ValueError, match="shape"):
        scan_results.load_exp102_publication_q_top(result)


def test_publication_loader_requires_nonempty_boolean_mask(monkeypatch, tmp_path):
    monkeypatch.setattr(scan_results, "_verify_recorded_git_support", lambda *_: None)
    result = _bundle(tmp_path)
    with pytest.raises(ValueError, match="boolean"):
        scan_results.load_exp102_publication_q_top(result, np.ones((6, 7), dtype=np.int8))
    with pytest.raises(ValueError, match="at least one"):
        scan_results.load_exp102_publication_q_top(result, np.zeros((6, 7), dtype=bool))


def test_publication_loader_masks_unselected_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(scan_results, "_verify_recorded_git_support", lambda *_: None)
    result = _bundle(tmp_path)
    mask = np.zeros((6, 7), dtype=bool)
    mask[2, 3] = True
    loaded = scan_results.load_exp102_publication_q_top(result, mask)
    assert np.isfinite(loaded["q_top"][2, 3])
    assert np.isfinite(loaded["errorbar"][2, 3])
    assert np.all(np.isnan(loaded["q_top"][~mask]))
    assert np.all(np.isnan(loaded["errorbar"][~mask]))
