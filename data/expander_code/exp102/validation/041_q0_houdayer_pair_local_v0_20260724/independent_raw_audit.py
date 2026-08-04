"""Pickle-free raw-only audit for the frozen HCA-RHB1 local run.

This analyzer intentionally does not import the pair sampler or runner.  It
rebuilds the disorder, legal starts, hard-coset checks, labels, weights, and
all pre-registered pair-level gates from raw NPZ files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import reduce_logical_basis, uniform_hard_coset_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


AUDIT_VERSION = "exp102.q0_houdayer_pair.local.raw_audit.v0"
RUN_VERSION = "exp102.q0_houdayer_pair.local.v0"
RAW_VERSION = "exp102.q0_houdayer_pair.local.raw.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class HoudayerRawAuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise HoudayerRawAuditError(message)


def _label(W_basis, states):
    bits = (W_basis.astype(np.int64) @ states.T.astype(np.int64) % 2).astype(np.uint8)
    values = np.zeros(states.shape[0], dtype=np.uint64)
    for bit in range(W_basis.shape[0]):
        values |= bits[bit].astype(np.uint64) << np.uint64(bit)
    return values


def _low_starts(model, frame, planted, count=2):
    reduced = np.ascontiguousarray(reduce_logical_basis(model.logical_move_basis), dtype=np.uint8)
    candidates = {}
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(model.k), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            label = _label(frame.W_basis, move[None, :])[0]
            _require(label, "raw audit low logical candidate has zero label")
            state = np.ascontiguousarray(planted ^ move, dtype=np.uint8)
            candidates[packed] = (int(state.sum()), int(move.sum()), int(label), packed, state)
    by_label = {}
    for record in sorted(candidates.values(), key=lambda row: row[:4]):
        by_label.setdefault(record[2], record)
    selected = sorted(by_label.values(), key=lambda row: row[:4])[:count]
    _require(len(selected) == count, "raw audit low logical catalog is incomplete")
    return [record[4] for record in selected]


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _pair_characters(left_labels, right_labels, k):
    positions = np.arange(int(k), dtype=np.uint64)
    left = ((left_labels[:, None] >> positions[None, :]) & np.uint64(1)).astype(np.float64)
    right = ((right_labels[:, None] >> positions[None, :]) & np.uint64(1)).astype(np.float64)
    return 1.0 - (left + right)


def _check_hash_document(path, key):
    document = json.loads(Path(path).read_text(encoding="ascii"))
    claimed = document.pop(key, None)
    _require(claimed == sha256_json(document), f"{Path(path).name} self-hash is invalid")
    document[key] = claimed
    return document


def _load_config(path):
    raw = Path(path).read_text(encoding="ascii")
    config = json.loads(raw)
    _require(raw == canonical_json(config) + "\n", "raw audit config is not canonical")
    _require(config["version"] == RUN_VERSION and config["resource"] == {
        "burn_clocks": 128, "measurement_clocks": 1024, "trajectories_per_family": 8,
    }, "raw audit config changed")
    return config, sha256_file(path)


def _summary(rows):
    return {
        "pair_weight": np.asarray([row["pair_weight"] for row in rows], dtype=np.float64),
        "pair_characters": np.asarray([row["pair_characters"] for row in rows], dtype=np.float64),
        "first_weight": np.asarray([row["first_weight"] for row in rows], dtype=np.float64),
        "last_weight": np.asarray([row["last_weight"] for row in rows], dtype=np.float64),
        "first_characters": np.asarray([row["first_characters"] for row in rows], dtype=np.float64),
        "last_characters": np.asarray([row["last_characters"] for row in rows], dtype=np.float64),
    }


def _comparison(left_name, left, right_name, right, gates):
    left_weight, left_weight_se = _mean_se(left["pair_weight"])
    right_weight, right_weight_se = _mean_se(right["pair_weight"])
    difference = abs(float(left_weight - right_weight))
    se = float(np.hypot(left_weight_se, right_weight_se))
    weight_pass = (
        difference <= gates["normalized_weight_absolute_difference"]
        and difference <= 3.0 * se + gates["normalized_weight_se_slack"]
    )
    left_characters, left_character_se = _mean_se(left["pair_characters"])
    right_characters, right_character_se = _mean_se(right["pair_characters"])
    character_difference = np.abs(left_characters - right_characters)
    character_se = np.hypot(left_character_se, right_character_se)
    character_pass = bool(
        np.all(character_difference <= gates["basis_character_absolute_difference"])
        and np.all(character_difference <= 3.0 * character_se + gates["basis_character_se_slack"])
    )
    return {
        "families": [left_name, right_name],
        "normalized_weight_difference": difference,
        "normalized_weight_se": se,
        "normalized_weight_pass": bool(weight_pass),
        "basis_character_maximum_difference": float(character_difference.max()),
        "basis_character_maximum_se": float(character_se.max()),
        "basis_character_pass": character_pass,
        "pass": bool(weight_pass and character_pass),
    }


def _stability(family, values, gates):
    weight_delta = values["last_weight"] - values["first_weight"]
    weight_mean, weight_se = _mean_se(weight_delta)
    weight_pass = (
        abs(float(weight_mean)) <= gates["normalized_weight_absolute_difference"]
        and abs(float(weight_mean)) <= 3.0 * float(weight_se) + gates["normalized_weight_se_slack"]
    )
    character_delta = values["last_characters"] - values["first_characters"]
    character_mean, character_se = _mean_se(character_delta)
    character_pass = bool(
        np.all(np.abs(character_mean) <= gates["basis_character_absolute_difference"])
        and np.all(np.abs(character_mean) <= 3.0 * character_se + gates["basis_character_se_slack"])
    )
    return {
        "family": family,
        "normalized_weight_drift": float(weight_mean),
        "normalized_weight_drift_se": float(weight_se),
        "normalized_weight_pass": bool(weight_pass),
        "basis_character_maximum_drift": float(np.abs(character_mean).max()),
        "basis_character_maximum_drift_se": float(character_se.max()),
        "basis_character_pass": character_pass,
        "pass": bool(weight_pass and character_pass),
    }


def run_audit(config_path, run_dir):
    config, config_sha256 = _load_config(config_path)
    run_dir = Path(run_dir)
    manifest = _check_hash_document(run_dir / "MANIFEST.json", "manifest_sha256")
    complete = _check_hash_document(run_dir / "RUN_COMPLETE.json", "run_complete_sha256")
    report = _check_hash_document(run_dir / "REPORT.json", "report_sha256")
    _require(manifest["manifest_version"] == RUN_VERSION and manifest["config_sha256"] == config_sha256,
             "raw audit manifest identity changed")
    _require(complete["manifest_sha256"] == manifest["manifest_sha256"]
             and report["manifest_sha256"] == manifest["manifest_sha256"],
             "raw audit manifest/report binding changed")
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "raw audit registry changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    _seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    starts = _low_starts(model, frame, planted)
    expected_raw_paths = []
    grouped = {family: [] for family in ("PP", "UU", "LL", "PL")}
    raw_schema = {
        "burn_left_packed", "burn_right_packed", "counter_names", "counter_values",
        "final_left_packed", "final_right_packed", "initial_left_packed", "initial_right_packed",
        "kernel", "measurement_component_counts", "measurement_left_labels",
        "measurement_left_states_packed", "measurement_left_weights", "measurement_new_unordered_pair",
        "measurement_residual_weights", "measurement_right_labels", "measurement_right_states_packed",
        "measurement_right_weights", "measurement_whole_pair_exchange", "metadata_json",
    }
    for task in manifest["tasks"]:
        family = task["family"]
        index = int(task["trajectory_index"])
        path = run_dir / "raw" / f"{family}_{index:02d}.npz"
        expected_raw_paths.append({"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)})
        if family == "PP":
            expected_left, expected_right = planted, planted
        elif family == "UU":
            expected_left = uniform_hard_coset_state(model, syndrome, task["left_uniform_seed"])
            expected_right = uniform_hard_coset_state(model, syndrome, task["right_uniform_seed"])
        elif family == "LL":
            expected_left, expected_right = starts
        elif family == "PL":
            expected_left, expected_right = planted, starts[0]
        else:
            raise HoudayerRawAuditError("raw audit has an unknown initial pair family")
        with np.load(path, allow_pickle=False) as data:
            _require(set(data.files) == raw_schema, "raw audit raw schema changed")
            metadata = json.loads(str(data["metadata_json"].item()))
            _require(metadata["raw_version"] == RAW_VERSION
                     and metadata["manifest_sha256"] == manifest["manifest_sha256"]
                     and metadata["task"] == task,
                     "raw audit raw metadata changed")
            _require(metadata["initial_left_sha256"] == hashlib.sha256(expected_left.tobytes()).hexdigest()
                     and metadata["initial_right_sha256"] == hashlib.sha256(expected_right.tobytes()).hexdigest(),
                     "raw audit initial-state identity changed")
            _require(np.array_equal(
                np.unpackbits(data["initial_left_packed"], count=model.num_qubits, bitorder="little"),
                expected_left,
            ) and np.array_equal(
                np.unpackbits(data["initial_right_packed"], count=model.num_qubits, bitorder="little"),
                expected_right,
            ), "raw audit initial state bytes changed")
            left_states = np.unpackbits(
                data["measurement_left_states_packed"], axis=1, count=model.num_qubits,
                bitorder="little",
            ).astype(np.uint8, copy=False)
            right_states = np.unpackbits(
                data["measurement_right_states_packed"], axis=1, count=model.num_qubits,
                bitorder="little",
            ).astype(np.uint8, copy=False)
            _require(left_states.shape == right_states.shape == (1024, model.num_qubits),
                     "raw audit measurement dimensions changed")
            all_states = np.concatenate((left_states, right_states), axis=0)
            residual = (
                model.H_check.astype(np.int64) @ all_states.T.astype(np.int64) % 2
            ).T.astype(np.uint8) ^ syndrome[None, :]
            _require(not residual.any() and not np.asarray(data["measurement_residual_weights"]).any(),
                     "raw audit hard-coset residual changed")
            left_labels = _label(frame.W_basis, left_states)
            right_labels = _label(frame.W_basis, right_states)
            _require(np.array_equal(left_labels, data["measurement_left_labels"])
                     and np.array_equal(right_labels, data["measurement_right_labels"]),
                     "raw audit label mismatch")
            left_weights = left_states.sum(axis=1).astype(np.int32)
            right_weights = right_states.sum(axis=1).astype(np.int32)
            _require(np.array_equal(left_weights, data["measurement_left_weights"])
                     and np.array_equal(right_weights, data["measurement_right_weights"]),
                     "raw audit weight mismatch")
            _require(str(data["kernel"].item())
                     == "reduced_coordinate_random_scan_heatbath_plus_houdayer.v0",
                     "raw audit kernel identity changed")
            _require(np.all((data["measurement_new_unordered_pair"] == 0)
                            | (data["measurement_new_unordered_pair"] == 1))
                     and np.all((data["measurement_whole_pair_exchange"] == 0)
                                | (data["measurement_whole_pair_exchange"] == 1)),
                     "raw audit HCA flags are not binary")
            pair_weight = (left_weights + right_weights).astype(np.float64) / (2.0 * model.num_qubits)
            characters = _pair_characters(left_labels, right_labels, model.k)
            grouped[family].append({
                "pair_weight": float(pair_weight.mean()),
                "pair_characters": characters.mean(axis=0),
                "first_weight": float(pair_weight[:512].mean()),
                "last_weight": float(pair_weight[512:].mean()),
                "first_characters": characters[:512].mean(axis=0),
                "last_characters": characters[512:].mean(axis=0),
                "new_pair_events": int(data["measurement_new_unordered_pair"].sum()),
                "whole_pair_exchanges": int(data["measurement_whole_pair_exchange"].sum()),
            })
    _require(complete["raw_files"] == expected_raw_paths,
             "raw audit run-complete raw hashes changed")
    values = {family: _summary(rows) for family, rows in grouped.items()}
    comparisons = [_comparison(left, values[left], right, values[right], config["gates"])
                   for left, right in itertools.combinations(("PP", "UU", "LL", "PL"), 2)]
    stability = [_stability(family, values[family], config["gates"])
                 for family in ("PP", "UU", "LL", "PL")]
    ll_events = sum(row["new_pair_events"] for row in grouped["LL"])
    ll_pairs = sum(row["new_pair_events"] > 0 for row in grouped["LL"])
    hca_gate = {
        "ll_measurement_new_pair_events": int(ll_events),
        "ll_pairs_with_one_measurement_new_pair": int(ll_pairs),
        "minimum_events": config["gates"]["ll_minimum_measurement_new_pair_events"],
        "minimum_pairs": config["gates"]["ll_minimum_pairs_with_one_measurement_new_pair"],
        "pass": bool(
            ll_events >= config["gates"]["ll_minimum_measurement_new_pair_events"]
            and ll_pairs >= config["gates"]["ll_minimum_pairs_with_one_measurement_new_pair"]
        ),
    }
    gate_pass = bool(hca_gate["pass"] and all(item["pass"] for item in comparisons)
                     and all(item["pass"] for item in stability))
    expected_status = "LOCAL_HOUDAYER_PAIR_TRANSPORT_VIABLE" if gate_pass else "LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED"
    _require(report["analysis"]["status"] == expected_status
             and report["analysis"]["cross_family_comparisons"] == comparisons
             and report["analysis"]["fixed_clock_stability"] == stability
             and report["analysis"]["hca_substance_gate"] == hca_gate,
             "raw audit recomputed gate report disagrees")
    core = {
        "audit_version": AUDIT_VERSION,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "run_complete_sha256": complete["run_complete_sha256"],
        "report_sha256": report["report_sha256"],
        "raw_file_count": len(expected_raw_paths),
        "recomputed_status": expected_status,
        "hca_substance_gate": hca_gate,
    }
    return {**core, "audit_sha256": sha256_json(core)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "local_houdayer_pair_viability_v0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "INDEPENDENT_AUDIT.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace raw audit: {output}")
    audit = run_audit(args.config, args.run_dir)
    atomic_json(output, audit)
    print(audit["audit_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
