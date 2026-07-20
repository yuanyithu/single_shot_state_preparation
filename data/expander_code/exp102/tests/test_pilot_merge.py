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
from data.expander_code.exp102.exp102_pipeline.pilot_cell import run_cell
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


EXP102_ROOT = Path(__file__).resolve().parents[1]


def test_config_allows_r96_r128_only_at_the_approved_ladder_tail(tmp_path):
    path = EXP102_ROOT / "config/production.v1.json"
    config = load_config(path)
    assert config["pilot"]["ladder_candidates"][-2:] == [
        {"p_hot": 0.49, "num_temperatures": 96},
        {"p_hot": 0.49, "num_temperatures": 128},
    ]

    edited = json.loads(path.read_text(encoding="ascii"))
    edited["pilot"]["ladder_candidates"].insert(
        7, {"p_hot": 0.45, "num_temperatures": 96},
    )
    tampered = tmp_path / "production.json"
    tampered.write_text(json.dumps(edited), encoding="ascii")
    with pytest.raises(ValueError, match="ordered pilot ladder"):
        load_config(tampered)

    edited = json.loads(path.read_text(encoding="ascii"))
    edited["production_gate"]["min_swap_rate"] = 0.0
    tampered.write_text(json.dumps(edited), encoding="ascii")
    with pytest.raises(ValueError, match="production gate"):
        load_config(tampered)


def test_pilot_cell_rejects_unapproved_ladder_pair_before_running(tmp_path):
    candidate = {
        "p_hot": 0.45, "num_temperatures": 96, "gamma": 1.0,
        "burn_rounds": 500, "measurement_rounds": 2000,
        "sweeps_per_round": 1, "logical_move_repeat": 1,
    }
    with pytest.raises(ValueError, match="ladder pair"):
        run_cell(
            EXP102_ROOT / "registry/registry.json",
            EXP102_ROOT / "config/production.v1.json",
            "m03_c00", 0.04, 0, candidate, 0, "ladder", "1" * 40,
            tmp_path / "must-not-exist.npz",
        )
    assert not (tmp_path / "must-not-exist.npz").exists()


def _write_valid_ladder_raw(path, registry, config, stored_valid=True):
    code = next(row for row in registry["codes"] if row["code_id"] == "m03_c00")
    candidate = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    source_commit = "1" * 40
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
            "ladder_candidates": [
                {"p_hot": 0.45, "num_temperatures": 8},
                {"p_hot": 0.45, "num_temperatures": 12},
            ],
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


def _fallback_selection_context():
    config = {
        "p_values": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "pilot": {
            "ladder_candidates": [
                {"p_hot": 0.45, "num_temperatures": 8},
                {"p_hot": 0.49, "num_temperatures": 12},
            ],
            "gamma_candidates": [0.75, 1.0, 1.5],
            "round_candidates": [[500, 2000], [1000, 4000]],
        },
    }
    registry = {"codes": [{"code_id": f"m03_c{i:02d}", "m": 3} for i in range(8)]}
    return config, registry


def test_held_out_max_round_failure_reselects_gamma_on_next_ladder():
    config, registry = _fallback_selection_context()
    first = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    first_long = _candidate(config, 0.45, 8, 1.0, [1000, 4000])
    second = _candidate(config, 0.49, 12, 0.75, [500, 2000])
    records = []
    records += _records_for_group("ladder", 3, first, True, config)
    records += _records_for_group(
        "ladder", 3, _candidate(config, 0.49, 12, 1.0, [500, 2000]), True, config,
    )
    for gamma, core in ((0.75, 3.0), (1.0, 1.0), (1.5, 2.0)):
        records += _records_for_group(
            "gamma", 3, _candidate(config, 0.45, 8, gamma, [500, 2000]),
            True, config, core_seconds=core,
        )
    for gamma, core in ((0.75, 1.0), (1.0, 2.0), (1.5, 3.0)):
        records += _records_for_group(
            "gamma", 3, _candidate(config, 0.49, 12, gamma, [500, 2000]),
            True, config, core_seconds=core,
        )
    records += _records_for_group("rounds", 3, first, True, config)
    records += _records_for_group("held_out", 3, first, False, config, attempt=0)
    records += _records_for_group("rounds", 3, first_long, True, config)
    records += _records_for_group("held_out", 3, first_long, False, config, attempt=1)
    records += _records_for_group("rounds", 3, second, True, config)
    records += _records_for_group("held_out", 3, second, True, config, attempt=2)

    selected = _select_one_m(3, _group_records(records, registry, config), config)
    assert selected["state"] == "PASSED"
    assert [cycle["outcome"] for cycle in selected["cycles"]] == [
        "held_out_exhausted", "passed",
    ]
    assert selected["ladder"]["selected"]["p_hot"] == 0.49
    assert selected["gamma"]["selected"]["gamma"] == 0.75
    assert selected["rounds"]["selected"] == second
    assert selected["held_out"]["selected_attempt"] == 2
    assert selected["all_tuning_pass"] and selected["num_tuning_cells"] == 96
    assert selected["all_held_out_pass"] and selected["num_held_out_cells"] == 448


def test_selection_does_not_jump_over_a_missing_ladder_prefix():
    config, registry = _fallback_selection_context()
    future = _candidate(config, 0.49, 12, 1.0, [500, 2000])
    records = _records_for_group("ladder", 3, future, True, config)
    for gamma in config["pilot"]["gamma_candidates"]:
        records += _records_for_group(
            "gamma", 3, _candidate(config, 0.49, 12, gamma, [500, 2000]),
            True, config,
        )
    records += _records_for_group("rounds", 3, future, True, config)
    records += _records_for_group("held_out", 3, future, True, config)

    selected = _select_one_m(3, _group_records(records, registry, config), config)
    assert selected["state"] == "WAITING_LADDER"
    assert selected["next_action"]["candidate"]["p_hot"] == 0.45
    assert selected["selected_config"] is None
    assert not selected["all_tuning_pass"] and not selected["all_held_out_pass"]


def test_held_out_attempt_candidate_mismatch_is_fail_closed():
    config, registry = _fallback_selection_context()
    base = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    wrong = _candidate(config, 0.45, 8, 1.0, [1000, 4000])
    records = _records_for_group("ladder", 3, base, True, config)
    for gamma in config["pilot"]["gamma_candidates"]:
        records += _records_for_group(
            "gamma", 3, _candidate(config, 0.45, 8, gamma, [500, 2000]),
            True, config, core_seconds=1.0 if gamma == 1.0 else 2.0,
        )
    records += _records_for_group("rounds", 3, base, True, config)
    records += _records_for_group("held_out", 3, wrong, True, config, attempt=0)

    selected = _select_one_m(3, _group_records(records, registry, config), config)
    assert selected["state"] == "CONFLICT"
    assert selected["conflict_reason"] == "held_out_candidate_mismatch"
    assert not selected["all_held_out_pass"]


def test_held_out_attempt_gap_is_fail_closed():
    config, registry = _fallback_selection_context()
    candidate = _candidate(config, 0.45, 8, 1.0, [500, 2000])
    records = _records_for_group("held_out", 3, candidate, True, config, attempt=1)

    selected = _select_one_m(3, _group_records(records, registry, config), config)
    assert selected["state"] == "CONFLICT"
    assert selected["conflict_reason"] == "held_out_attempt_gap"
    assert not selected["all_tuning_pass"] and not selected["all_held_out_pass"]


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
