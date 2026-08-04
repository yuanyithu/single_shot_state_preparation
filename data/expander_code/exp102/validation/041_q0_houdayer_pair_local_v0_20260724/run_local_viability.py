"""Frozen local adversarial-pair viability run for HCA-RHB1."""

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
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import uniform_hard_coset_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    HOUDAYER_PAIR_KERNEL,
    HOUDAYER_PAIR_VERSION,
    build_reduced_houdayer_pair_kernel,
    deterministic_low_energy_logical_starts,
    run_houdayer_pair_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


RUN_VERSION = "exp102.q0_houdayer_pair.local.v0"
RAW_VERSION = "exp102.q0_houdayer_pair.local.raw.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
PREFLIGHT_PATH = (
    EXP102_ROOT / "validation" / "042_q0_houdayer_pair_runtime_rebind_20260724"
    / "runtime_preflight_rebind.json"
)


class HoudayerLocalViabilityError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise HoudayerLocalViabilityError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise HoudayerLocalViabilityError("Houdayer local config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "Houdayer local config is not canonical")
    expected = {
        "cell", "config_version", "contract_version", "gates", "initial_pairs", "kernel",
        "low_energy_catalog", "preflight_report_sha256", "registry_sha256", "resource", "scope",
        "version",
    }
    _require(set(config) == expected and config["version"] == RUN_VERSION
             and config["contract_version"] == RUN_VERSION
             and config["config_version"] == "exp102.q0_houdayer_pair.local.config.v0",
             "Houdayer local config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "Houdayer local cell changed")
    _require(config["kernel"] == {
        "coordinate_basis": "h_x_plus_canonical_reduced_logical_complement.v0",
        "local_kernel": "independent_random_scan_coordinate_heatbath.v0",
        "local_updates_per_clock": 832,
        "pair_kernel": "complete_component_houdayer_swap.v0",
    }, "Houdayer local kernel changed")
    _require(config["low_energy_catalog"] == {
        "candidate_orders": [1, 2, 3],
        "count": 2,
        "selection": "per_nonzero_signature_min_state_weight_then_move_weight_signature_packed_move",
    }, "Houdayer local L-start catalog changed")
    _require(config["initial_pairs"] == {
        "LL": "first_two_label_distinct_p_derived_low_energy_starts",
        "PL": "planted_and_first_p_derived_low_energy_start",
        "PP": "planted_and_planted",
        "UU": "two_independent_exact_K0_uniform_hard_coset_states",
    }, "Houdayer local initial pairs changed")
    _require(config["resource"] == {
        "burn_clocks": 128,
        "measurement_clocks": 1024,
        "trajectories_per_family": 8,
    }, "Houdayer local resource tier changed")
    _require(config["gates"] == {
        "basis_character_absolute_difference": 0.1,
        "basis_character_se_slack": 0.02,
        "ll_minimum_measurement_new_pair_events": 16,
        "ll_minimum_pairs_with_one_measurement_new_pair": 6,
        "normalized_weight_absolute_difference": 0.01,
        "normalized_weight_se_slack": 0.000625,
        "require_all_family_pairs_and_stability": True,
    }, "Houdayer local gates changed")
    _require(config["preflight_report_sha256"]
             == "f0337841fcb4806cecba905d922c8639a395df6f5c148551a8ba0b815a0ae5a1",
             "Houdayer local runtime preflight binding changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "Houdayer local registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "local_adversarial_pair_transport_viability_only",
        "remote_authorization": False,
    }, "Houdayer local scope changed")
    return config, sha256_file(path)


def _verify_preflight(config):
    report = json.loads(PREFLIGHT_PATH.read_text(encoding="ascii"))
    claimed = report.pop("report_sha256", None)
    _require(claimed == sha256_json(report)
             and claimed == config["preflight_report_sha256"],
             "Houdayer local runtime preflight hash is invalid")
    _require(report["preflight"]["status"] == "RUNTIME_PASS",
             "Houdayer local runtime preflight did not pass")
    return claimed


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer.py"),
        "pair_kernel": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer_pair.py"),
        "preflight": sha256_file(PREFLIGHT_PATH),
        "registry": sha256_file(REGISTRY_PATH),
        "runner": sha256_file(Path(__file__)),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "Houdayer local registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    kernel = build_reduced_houdayer_pair_kernel(H, model, frame, syndrome, config["cell"]["p"])
    starts = deterministic_low_energy_logical_starts(
        model, frame, planted, count=config["low_energy_catalog"]["count"],
        orders=config["low_energy_catalog"]["candidate_orders"],
    )
    _require(kernel.coordinate_count == config["kernel"]["local_updates_per_clock"],
             "Houdayer local coordinate count changed")
    return registry, code, H, model, frame, int(uniform_seed), planted, syndrome, kernel, starts


def _task_seed(source_binding, config_sha256, registry_sha256, family, trajectory_index, role):
    return derive_seed(
        RUN_VERSION, source_binding["source_commit"], config_sha256, registry_sha256,
        "m08_c06", "attempt022", "p0.04", family, int(trajectory_index), role,
    )


def _build_manifest(config, config_sha256, source_binding, registry, starts):
    tasks = []
    for family in ("PP", "UU", "LL", "PL"):
        for trajectory_index in range(config["resource"]["trajectories_per_family"]):
            task = {
                "family": family,
                "trajectory_index": trajectory_index,
                "transition_seed": _task_seed(
                    source_binding, config_sha256, registry["registry_sha256"],
                    family, trajectory_index, "transition",
                ),
            }
            if family == "UU":
                task["left_uniform_seed"] = _task_seed(
                    source_binding, config_sha256, registry["registry_sha256"],
                    family, trajectory_index, "left_uniform",
                )
                task["right_uniform_seed"] = _task_seed(
                    source_binding, config_sha256, registry["registry_sha256"],
                    family, trajectory_index, "right_uniform",
                )
            tasks.append(task)
    start_records = [{
        "index": int(record["index"]),
        "logical_signature": int(record["signature"]),
        "move_weight": int(record["move_weight"]),
        "state_weight": int(record["state_weight"]),
        "state_sha256": hashlib.sha256(record["state"].tobytes()).hexdigest(),
    } for record in starts]
    core = {
        "manifest_version": RUN_VERSION,
        "config_sha256": config_sha256,
        "registry_sha256": registry["registry_sha256"],
        "source_binding": source_binding,
        "initial_pairs": config["initial_pairs"],
        "low_energy_starts": start_records,
        "tasks": tasks,
    }
    return {**core, "manifest_sha256": sha256_json(core)}


def _initial_pair(task, family, model, syndrome, planted, starts):
    if family == "PP":
        return planted.copy(), planted.copy()
    if family == "UU":
        return (
            uniform_hard_coset_state(model, syndrome, task["left_uniform_seed"]),
            uniform_hard_coset_state(model, syndrome, task["right_uniform_seed"]),
        )
    if family == "LL":
        return starts[0]["state"].copy(), starts[1]["state"].copy()
    if family == "PL":
        return planted.copy(), starts[0]["state"].copy()
    raise HoudayerLocalViabilityError("unknown Houdayer local initial pair")


def _raw_metadata(manifest, task, initial_left, initial_right):
    return {
        "raw_version": RAW_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "task": task,
        "initial_left_sha256": hashlib.sha256(initial_left.tobytes()).hexdigest(),
        "initial_right_sha256": hashlib.sha256(initial_right.tobytes()).hexdigest(),
    }


def _write_raw(path, trace, metadata):
    counters = trace.pop("counters")
    kernel = trace.pop("kernel")
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays["counter_names"] = np.asarray(sorted(counters), dtype="<U64")
    arrays["counter_values"] = np.asarray([counters[name] for name in sorted(counters)], dtype=np.int64)
    arrays["kernel"] = np.asarray(kernel, dtype="<U128")
    arrays["metadata_json"] = np.asarray(canonical_json(metadata), dtype="<U2048")
    atomic_npz(path, **arrays)


def _label_from_states(W_basis, states):
    bits = (W_basis.astype(np.int64) @ states.T.astype(np.int64) % 2).astype(np.uint8)
    values = np.zeros(states.shape[0], dtype=np.uint64)
    for bit in range(W_basis.shape[0]):
        values |= bits[bit].astype(np.uint64) << np.uint64(bit)
    return values


def _pair_character_means(left_labels, right_labels, k):
    masks = np.arange(int(k), dtype=np.uint64)
    left_bits = ((left_labels[:, None] >> masks[None, :]) & np.uint64(1)).astype(np.float64)
    right_bits = ((right_labels[:, None] >> masks[None, :]) & np.uint64(1)).astype(np.float64)
    values = 1.0 - (left_bits + right_bits)
    return values.mean(axis=0)


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    _require(values.shape[0] >= 2, "Houdayer local needs at least two pair trajectories")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _load_and_summarize_raw(path, metadata, kernel, config):
    expected_keys = {
        "burn_left_packed", "burn_right_packed", "counter_names", "counter_values",
        "final_left_packed", "final_right_packed", "initial_left_packed", "initial_right_packed",
        "kernel", "measurement_component_counts", "measurement_left_labels",
        "measurement_left_states_packed", "measurement_left_weights", "measurement_new_unordered_pair",
        "measurement_residual_weights", "measurement_right_labels", "measurement_right_states_packed",
        "measurement_right_weights", "measurement_whole_pair_exchange", "metadata_json",
    }
    with np.load(path, allow_pickle=False) as data:
        _require(set(data.files) == expected_keys, "Houdayer local raw schema changed")
        saved_metadata = json.loads(str(data["metadata_json"].item()))
        _require(saved_metadata == metadata, "Houdayer local raw metadata changed")
        _require(str(data["kernel"].item()) == HOUDAYER_PAIR_KERNEL,
                 "Houdayer local raw kernel changed")
        left_states = np.unpackbits(
            data["measurement_left_states_packed"], axis=1, count=kernel.num_qubits,
            bitorder="little",
        ).astype(np.uint8, copy=False)
        right_states = np.unpackbits(
            data["measurement_right_states_packed"], axis=1, count=kernel.num_qubits,
            bitorder="little",
        ).astype(np.uint8, copy=False)
        _require(left_states.shape == right_states.shape
                 == (config["resource"]["measurement_clocks"], kernel.num_qubits),
                 "Houdayer local measurement shape changed")
        stacked = np.concatenate((left_states, right_states), axis=0)
        residual = (
            kernel.H_check.astype(np.int64) @ stacked.T.astype(np.int64) % 2
        ).T.astype(np.uint8) ^ kernel.syndrome[None, :]
        _require(not residual.any() and not np.asarray(data["measurement_residual_weights"]).any(),
                 "Houdayer local raw leaves the hard coset")
        left_labels = _label_from_states(kernel.W_basis, left_states)
        right_labels = _label_from_states(kernel.W_basis, right_states)
        _require(np.array_equal(left_labels, data["measurement_left_labels"])
                 and np.array_equal(right_labels, data["measurement_right_labels"]),
                 "Houdayer local raw labels drifted")
        left_weights = left_states.sum(axis=1).astype(np.int32)
        right_weights = right_states.sum(axis=1).astype(np.int32)
        _require(np.array_equal(left_weights, data["measurement_left_weights"])
                 and np.array_equal(right_weights, data["measurement_right_weights"]),
                 "Houdayer local raw weights drifted")
        blocks = 8
        block_size = config["resource"]["measurement_clocks"] // blocks
        pair_weight = (left_weights + right_weights).astype(np.float64) / (2.0 * kernel.num_qubits)
        pair_characters = _pair_character_means(left_labels, right_labels, kernel.logical_count)
        first_weight = pair_weight[:4 * block_size].mean()
        last_weight = pair_weight[4 * block_size:].mean()
        first_characters = _pair_character_means(
            left_labels[:4 * block_size], right_labels[:4 * block_size], kernel.logical_count,
        )
        last_characters = _pair_character_means(
            left_labels[4 * block_size:], right_labels[4 * block_size:], kernel.logical_count,
        )
        counter_names = [str(value) for value in data["counter_names"]]
        counter_values = [int(value) for value in data["counter_values"]]
        counters = dict(zip(counter_names, counter_values, strict=True))
        return {
            "pair_weight": float(pair_weight.mean()),
            "pair_characters": pair_characters,
            "first_weight": float(first_weight),
            "last_weight": float(last_weight),
            "first_characters": first_characters,
            "last_characters": last_characters,
            "measurement_new_pair_events": int(data["measurement_new_unordered_pair"].sum()),
            "measurement_whole_pair_exchanges": int(data["measurement_whole_pair_exchange"].sum()),
            "counters": counters,
            "raw_sha256": sha256_file(path),
        }


def _comparison(left_name, left, right_name, right, gates):
    left_weight, left_weight_se = _mean_se(left["pair_weight"])
    right_weight, right_weight_se = _mean_se(right["pair_weight"])
    weight_difference = abs(float(left_weight - right_weight))
    weight_se = float(np.hypot(left_weight_se, right_weight_se))
    weight_pass = (
        weight_difference <= gates["normalized_weight_absolute_difference"]
        and weight_difference <= 3.0 * weight_se + gates["normalized_weight_se_slack"]
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
        "normalized_weight_difference": weight_difference,
        "normalized_weight_se": weight_se,
        "normalized_weight_pass": bool(weight_pass),
        "basis_character_maximum_difference": float(character_difference.max()),
        "basis_character_maximum_se": float(character_se.max()),
        "basis_character_pass": character_pass,
        "pass": bool(weight_pass and character_pass),
    }


def _stability(family, values, gates):
    weight_delta = np.asarray(values["last_weight"]) - np.asarray(values["first_weight"])
    weight_mean, weight_se = _mean_se(weight_delta)
    weight_pass = (
        abs(float(weight_mean)) <= gates["normalized_weight_absolute_difference"]
        and abs(float(weight_mean)) <= 3.0 * float(weight_se) + gates["normalized_weight_se_slack"]
    )
    character_delta = np.asarray(values["last_characters"]) - np.asarray(values["first_characters"])
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


def _analyze(run_dir, manifest, kernel, config, model, planted, starts):
    summaries = {}
    raw_files = []
    for task in manifest["tasks"]:
        family = task["family"]
        left, right = _initial_pair(task, family, model, kernel.syndrome, planted, starts)
        metadata = _raw_metadata(manifest, task, left, right)
        path = run_dir / "raw" / f"{family}_{task['trajectory_index']:02d}.npz"
        summary = _load_and_summarize_raw(path, metadata, kernel, config)
        summaries.setdefault(family, []).append(summary)
        raw_files.append({"path": str(path.relative_to(run_dir)), "sha256": summary["raw_sha256"]})
    family_values = {}
    for family, rows in summaries.items():
        _require(len(rows) == config["resource"]["trajectories_per_family"],
                 "Houdayer local family trajectory count changed")
        family_values[family] = {
            "pair_weight": np.asarray([row["pair_weight"] for row in rows], dtype=np.float64),
            "pair_characters": np.asarray([row["pair_characters"] for row in rows], dtype=np.float64),
            "first_weight": np.asarray([row["first_weight"] for row in rows], dtype=np.float64),
            "last_weight": np.asarray([row["last_weight"] for row in rows], dtype=np.float64),
            "first_characters": np.asarray([row["first_characters"] for row in rows], dtype=np.float64),
            "last_characters": np.asarray([row["last_characters"] for row in rows], dtype=np.float64),
        }
    comparisons = [_comparison(left, family_values[left], right, family_values[right], config["gates"])
                   for left, right in itertools.combinations(("PP", "UU", "LL", "PL"), 2)]
    stability = [_stability(family, family_values[family], config["gates"])
                 for family in ("PP", "UU", "LL", "PL")]
    ll_rows = summaries["LL"]
    ll_events = sum(row["measurement_new_pair_events"] for row in ll_rows)
    ll_pairs = sum(row["measurement_new_pair_events"] > 0 for row in ll_rows)
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
    family_summary = {}
    for family, rows in summaries.items():
        mean_weight, se_weight = _mean_se(family_values[family]["pair_weight"])
        mean_characters, se_characters = _mean_se(family_values[family]["pair_characters"])
        family_summary[family] = {
            "pair_count": len(rows),
            "normalized_pair_weight_mean": float(mean_weight),
            "normalized_pair_weight_se": float(se_weight),
            "basis_character_min_max": [float(mean_characters.min()), float(mean_characters.max())],
            "basis_character_maximum_se": float(se_characters.max()),
            "measurement_new_unordered_pair_events": [
                int(row["measurement_new_pair_events"]) for row in rows
            ],
            "measurement_whole_pair_exchanges": [
                int(row["measurement_whole_pair_exchanges"]) for row in rows
            ],
        }
    gate_pass = bool(hca_gate["pass"] and all(item["pass"] for item in comparisons)
                     and all(item["pass"] for item in stability))
    return {
        "status": "LOCAL_HOUDAYER_PAIR_TRANSPORT_VIABLE" if gate_pass
                  else "LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED",
        "raw_files": raw_files,
        "family_summary": family_summary,
        "cross_family_comparisons": comparisons,
        "fixed_clock_stability": stability,
        "hca_substance_gate": hca_gate,
        "does_not_establish": [
            "A posterior, q_top, purity, or parameter-point physics result.",
            "A rigorous unobserved-tail or normalizer bound.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def run(config_path, output_dir):
    config, config_sha256 = _load_config(config_path)
    preflight_sha256 = _verify_preflight(config)
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), "refusing to replace a Houdayer local run directory")
    source_binding = _source_binding(config_path)
    registry, code, H, model, frame, uniform_seed, planted, syndrome, kernel, starts = _context(config)
    manifest = _build_manifest(config, config_sha256, source_binding, registry, starts)
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "MANIFEST.json", manifest)
    atomic_json(output_dir / "RUNNING.json", {
        "run_version": RUN_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "preflight_report_sha256": preflight_sha256,
    })
    for task in manifest["tasks"]:
        family = task["family"]
        left, right = _initial_pair(task, family, model, syndrome, planted, starts)
        metadata = _raw_metadata(manifest, task, left, right)
        trace = run_houdayer_pair_trajectory(
            kernel, left, right, task["transition_seed"],
            config["resource"]["burn_clocks"], config["resource"]["measurement_clocks"],
            config["kernel"]["local_updates_per_clock"],
        )
        _write_raw(output_dir / "raw" / f"{family}_{task['trajectory_index']:02d}.npz", trace, metadata)
    analysis = _analyze(output_dir, manifest, kernel, config, model, planted, starts)
    complete_core = {
        "run_version": RUN_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_files": analysis["raw_files"],
    }
    run_complete = {**complete_core, "run_complete_sha256": sha256_json(complete_core)}
    atomic_json(output_dir / "RUN_COMPLETE.json", run_complete)
    report_core = {
        "run_version": RUN_VERSION,
        "config_sha256": config_sha256,
        "preflight_report_sha256": preflight_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_binding": source_binding,
        "cell": config["cell"],
        "disorder_uniform_seed": uniform_seed,
        "kernel": HOUDAYER_PAIR_KERNEL,
        "analysis": analysis,
    }
    report = {**report_core, "report_sha256": sha256_json(report_core)}
    atomic_json(output_dir / "REPORT.json", report)
    atomic_json(output_dir / "SUCCESS.json", {
        "run_version": RUN_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "status": analysis["status"],
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "local_houdayer_pair_viability_v0")
    args = parser.parse_args(argv)
    report = run(args.config, args.output)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
