import json
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.diagnostics import evaluate_gate
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.pilot import (
    TUNING_P_VALUES,
    _assert_report_matches_recomputed,
    _candidate,
    _group_records,
    _select_one_m,
    _validate_raw,
    _verify_report_evidence,
)
from data.expander_code.exp102.exp102_pipeline.pilot_cell import _gate
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


EXP102_ROOT = Path(__file__).resolve().parents[1]


def _write_valid_ladder_raw(path, registry, config, stored_valid=True):
    code = next(row for row in registry["codes"] if row["code_id"] == "m03_c00")
    candidate = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    source_commit = "test-source"
    namespace = "pilot_ladder_m3_attempt0"
    p = 0.04
    disorder = 0
    rng = np.random.default_rng(1234)
    labels = rng.integers(0, 1 << code["k"], size=(4, 2000), dtype=np.uint64)
    swap_attempts = np.full((4, 7), 1000, dtype=np.int64)
    swap_accepts = np.full((4, 7), 500, dtype=np.int64)
    logical_attempts = np.full((4, 8, code["k"]), 2500, dtype=np.int64)
    logical_accepts = np.full((4, 8, code["k"]), 1000, dtype=np.int64)
    round_trips = np.full(4, 10, dtype=np.int64)
    changing = np.full(4, 5, dtype=np.int64)
    residual = np.zeros(4, dtype=np.int64)
    results = []
    for instance in range(4):
        results.append({
            "labels": labels[instance], "swap_attempts": swap_attempts[instance],
            "swap_accepts": swap_accepts[instance], "logical_attempts": logical_attempts[instance],
            "logical_accepts": logical_accepts[instance], "round_trips": int(round_trips[instance]),
            "sector_changing_round_trips": int(changing[instance]),
            "max_hard_coset_residual": 0,
            "seed": derive_seed(namespace, registry["registry_sha256"], code["code_id"],
                                disorder, f"p={p:.8f}", instance),
        })
    valid, failures, rhats, esses, statuses = evaluate_gate(
        results, _gate(config, "ladder"), code["k"], require_trace_gate=False,
    )
    assert valid and not failures
    identity = {
        "namespace": namespace, "stage": "ladder", "code_id": code["code_id"], "p": p,
        "disorder_index": disorder, "candidate": candidate,
        "registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
        "source_commit": source_commit, "engine": "numba",
    }
    atomic_npz(
        path, task_fingerprint=np.array(sha256_json(identity)), namespace=np.array(namespace),
        stage=np.array("ladder"), code_id=np.array(code["code_id"]), m=np.array(3, dtype=np.int8),
        p=np.array(p), disorder_index=np.array(disorder, dtype=np.int16),
        candidate_json=np.array(canonical_json(candidate)), attempt=np.array(0, dtype=np.int16),
        valid=np.array(stored_valid), failure_reason=np.array("", dtype="U4096"), labels=labels,
        swap_attempts=swap_attempts, swap_accepts=swap_accepts,
        swap_rates=swap_accepts / swap_attempts,
        logical_attempts=logical_attempts, logical_accepts=logical_accepts,
        logical_rates=logical_accepts / logical_attempts,
        round_trips=round_trips, sector_changing_round_trips=changing, residual=residual,
        rhat=rhats, ess=esses, constant_status=statuses,
        core_seconds=np.array(1.0), wall_seconds=np.array(2.0), engine=np.array("numba"),
        source_commit=np.array(source_commit),
        model_fingerprint=np.array(sha256_json({"n": code["n"], "k": code["k"]})),
        registry_sha256=np.array(registry["registry_sha256"]),
        config_sha256=np.array(config["config_sha256"]),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
    )
    return source_commit


def test_raw_validity_is_recomputed_instead_of_trusted(tmp_path):
    registry = load_registry(EXP102_ROOT / "registry/registry.json")
    config = load_config(EXP102_ROOT / "config/production.v1.json")
    path = tmp_path / "cell.npz"
    source_commit = _write_valid_ladder_raw(path, registry, config, stored_valid=True)
    assert _validate_raw(path, registry, config, source_commit)["valid"]

    _write_valid_ladder_raw(path, registry, config, stored_valid=False)
    with pytest.raises(ValueError, match="stored validity disagrees"):
        _validate_raw(path, registry, config, source_commit)


def _records_for_group(stage, m, candidate, valid, config, core_seconds=1.0, attempt=0):
    p_values = config["p_values"] if stage == "held_out" else TUNING_P_VALUES
    disorders = 8 if stage == "held_out" else 4
    rows = []
    for code_index in range(8):
        for p in p_values:
            for disorder in range(disorders):
                rows.append({
                    "stage": stage, "m": m, "attempt": attempt,
                    "candidate": candidate, "candidate_key": canonical_json(candidate),
                    "code_id": f"m{m:02d}_c{code_index:02d}", "p": float(p),
                    "disorder_index": disorder, "valid": valid,
                    "failure_reason": "instance_0:swap" if not valid else "",
                    "core_seconds": core_seconds, "wall_seconds": core_seconds,
                })
    return rows


def test_selection_uses_complete_prefix_and_recomputed_cell_results():
    config = {
        "p_values": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "pilot": {
            "p_hot_candidates": [0.45], "num_temperatures_candidates": [8, 12],
            "gamma_candidates": [0.75, 1.0, 1.5], "round_candidates": [[500, 2000]],
        },
    }
    registry = {"codes": [{"code_id": f"m03_c{i:02d}", "m": 3} for i in range(8)]}
    r8 = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    r12 = _candidate(config, 0.45, 12, 1.0, [500, 2000])
    records = _records_for_group("ladder", 3, r8, False, config)
    records += _records_for_group("ladder", 3, r12, True, config)
    for gamma, core in ((0.75, 3.0), (1.0, 1.0), (1.5, 2.0)):
        records += _records_for_group(
            "gamma", 3, _candidate(config, 0.45, 12, gamma, [500, 2000]), True, config,
            core_seconds=core,
        )
    records += _records_for_group("rounds", 3, r12, True, config)
    records += _records_for_group("held_out", 3, r12, True, config)
    selected = _select_one_m(3, _group_records(records, registry, config), config)
    assert selected["ladder"]["selected"]["num_temperatures"] == 12
    assert selected["gamma"]["selected"]["gamma"] == 1.0
    assert selected["all_tuning_pass"] and selected["num_tuning_cells"] == 96
    assert selected["all_held_out_pass"] and selected["num_held_out_cells"] == 448


def test_report_evidence_hash_is_rechecked(tmp_path):
    raw_path = tmp_path / "cell.npz"
    np.savez(raw_path, value=np.array(1))
    report_path = tmp_path / "report.json"
    report = {
        "generated_by": "pilot.merge-select.v1",
        "raw_evidence": [{"path": raw_path.name, "sha256": sha256_file(raw_path)}],
    }
    atomic_json(report_path, report)
    assert _verify_report_evidence(report, report_path) == [raw_path.resolve()]
    np.savez(raw_path, value=np.array(2))
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_report_evidence(report, report_path)


def test_report_pass_booleans_cannot_be_hand_edited():
    recomputed = {"3": {"all_tuning_pass": False, "all_held_out_pass": False}}
    forged = {"3": {"all_tuning_pass": True, "all_held_out_pass": True}}
    report = {"by_m": forged, "analysis_sha256": sha256_json(forged)}
    with pytest.raises(ValueError, match="recomputed raw gates"):
        _assert_report_matches_recomputed(report, recomputed)
