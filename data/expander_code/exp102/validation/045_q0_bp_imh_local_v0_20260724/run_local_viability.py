"""Run the frozen local BP-systematic independence-MH viability experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.diagnostics import bulk_ess, split_rhat
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    load_npz_no_pickle,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_bp_imh import (
    BP_IMH_RAW_VERSION,
    BP_IMH_VERSION,
    combine_bp_proposals,
    replay_bp_imh_trajectory,
    run_bp_imh_trajectory,
    validate_bp_imh_transcript,
)
from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    deterministic_low_energy_logical_starts,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_bp_imh.local.v0"
REPORT_VERSION = "exp102.q0_bp_imh.local.report.v0"
TASK_VERSION = "exp102.q0_bp_imh.local.tasks.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
FAMILIES = ("P", "U", "L")


class LocalBpImhError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalBpImhError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise LocalBpImhError("BP-IMH config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n", "BP-IMH config is not canonical")
    expected = {
        "bp", "cell", "combined_weights", "config_version", "contract_version", "gates",
        "initialization", "method_id", "proposal_component_weights", "registry_sha256",
        "resource", "scope", "seed_namespace", "version",
    }
    _require(set(config) == expected
             and config["version"] == CONTRACT_VERSION
             and config["contract_version"] == CONTRACT_VERSION
             and config["config_version"] == "exp102.q0_bp_imh.local.config.v0",
             "BP-IMH config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "BP-IMH cell changed")
    _require(config["bp"] == {
        "damping": 0.5, "iterations": 64, "llr_cap": 30.0, "min_probability": 1e-5,
    }, "BP-IMH BP settings changed")
    _require(config["combined_weights"] == [0.5, 0.5]
             and config["proposal_component_weights"] == [0.9, 0.09, 0.01],
             "BP-IMH proposal weights changed")
    _require(config["method_id"] == "BPIMH-FR64", "BP-IMH method changed")
    _require(config["initialization"] == {
        "families": ["P", "U", "L"], "l_candidate_orders": [1, 2, 3],
        "l_catalog_count": 8,
    }, "BP-IMH initialization schedule changed")
    _require(config["resource"] == {
        "burn_steps": 256, "measurement_steps": 2048,
        "trajectory_count_per_family": 8,
    }, "BP-IMH fixed clock changed")
    _require(config["gates"] == {
        "character_d2_max": 0.04,
        "character_max_abs_delta": 0.04,
        "max_abs_b_weight_delta": 0.02,
        "max_abs_q_top_delta": 0.04,
        "max_abs_weight_delta": 0.01,
        "max_burn_end_normalized_weight_u": 0.15,
        "max_q_top_se": 0.03,
        "max_rhat": 1.05,
        "min_bulk_ess": 400.0,
        "min_burn_state_changes_u": 1,
        "min_each_internal_component_draws_per_trajectory": 1,
        "min_measurement_state_change_rate": 0.01,
        "min_measurement_state_changes": 16,
        "min_outer_source_draws_per_trajectory": 256,
        "q_top_se_slack": 0.005,
        "se_multiple": 3.0,
    }, "BP-IMH gates changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "BP-IMH registry identity changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "maximum_terminal_status": "LOCAL_BP_IMH_TRANSPORT_VIABLE_FOR_HARD2",
        "posterior_estimation": "diagnostic_only",
        "production_authorization": False,
        "remote_authorization": False,
    }, "BP-IMH scope changed")
    _require(config["seed_namespace"] == "exp102.q0_bp_imh.local.v0.20260724",
             "BP-IMH seed namespace changed")
    return config, sha256_file(path)


def _source_identity(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(len(source_commit) == 40
             and all(value in "0123456789abcdef" for value in source_commit),
             "BP-IMH source commit is invalid")
    paths = {
        "config": Path(config_path),
        "registry": REGISTRY_PATH,
        "runner": Path(__file__),
    }
    pipeline_root = EXP102_ROOT / "exp102_pipeline"
    pipeline_files = {
        path.relative_to(pipeline_root).as_posix(): sha256_file(path)
        for path in sorted(pipeline_root.rglob("*.py"))
    }
    exp101_root = EXP102_ROOT.parent / "exp101" / "src"
    exp101_files = {
        path.relative_to(exp101_root).as_posix(): sha256_file(path)
        for path in sorted(exp101_root.rglob("*.py"))
    }
    _require(pipeline_files and exp101_files, "BP-IMH source tree is incomplete")
    try:
        import numba
        numba_version = str(numba.__version__)
    except ImportError:
        numba_version = "missing"
    core = {
        "source_commit": source_commit,
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "exp102_pipeline_files": pipeline_files,
        "exp101_src_files": exp101_files,
        "runtime": {
            "numba": numba_version,
            "numpy": str(np.__version__),
            "python": sys.version.split()[0],
        },
    }
    return {**core, "source_identity_sha256": sha256_json(core)}


def _order(name, size):
    if name == "forward":
        return np.arange(int(size), dtype=np.int32)
    if name == "reverse":
        return np.arange(int(size) - 1, -1, -1, dtype=np.int32)
    raise LocalBpImhError("unknown BP-IMH systematic order")


def _hard_residual(H_check, state, syndrome):
    return (
        H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "BP-IMH registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600
             and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160,
             "BP-IMH m8 model identity changed")
    residual = (model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2).astype(np.uint8)
    _require(np.array_equal(residual, syndrome), "BP-IMH planted start is illegal")
    proposals = []
    for name in ("forward", "reverse"):
        proposals.append(build_bp_systematic_proposal(
            model, syndrome, config["cell"]["p"],
            column_order=_order(name, model.num_qubits),
            bp_iterations=config["bp"]["iterations"],
            bp_damping=config["bp"]["damping"],
            bp_llr_cap=config["bp"]["llr_cap"],
            min_probability=config["bp"]["min_probability"],
            component_weights=config["proposal_component_weights"],
        ))
    combined = combine_bp_proposals(proposals, config["combined_weights"])
    logical_starts = deterministic_low_energy_logical_starts(
        model, frame, planted, count=config["initialization"]["l_catalog_count"],
        orders=tuple(config["initialization"]["l_candidate_orders"]),
    )
    planted_label = int(state_label(frame, planted))
    logical_labels = [int(state_label(frame, record["state"])) for record in logical_starts]
    _require(len(set(logical_labels)) == len(logical_labels)
             and planted_label not in logical_labels,
             "BP-IMH logical starts are not label-distinct from P")
    for record in logical_starts:
        _require(not _hard_residual(model.H_check, record["state"], syndrome).any(),
                 "BP-IMH logical start is outside the hard coset")
    packed_l = np.stack([
        np.packbits(record["state"], bitorder="little") for record in logical_starts
    ])
    l_sha = hashlib.sha256(
        packed_l.tobytes() + np.asarray(
            [record["signature"] for record in logical_starts], dtype=">u8",
        ).tobytes()
    ).hexdigest()
    return {
        "registry": registry, "code": code, "H": H, "model": model, "frame": frame,
        "uniform_seed": int(uniform_seed), "planted": planted, "syndrome": syndrome,
        "proposals": tuple(proposals), "combined": combined,
        "logical_starts": logical_starts, "logical_start_sha256": l_sha,
    }


def _task_manifest(context, config, config_sha256, source_identity):
    registry = context["registry"]
    tasks = []
    for family in FAMILIES:
        for trajectory in range(config["resource"]["trajectory_count_per_family"]):
            identity = {
                "cell": config["cell"],
                "combined_proposal_sha256": context["combined"].proposal_sha256,
                "config_sha256": config_sha256,
                "init_family": family,
                "initialization_seed": derive_seed(
                    config["seed_namespace"], config_sha256, registry["registry_sha256"],
                    sha256_json(config["cell"]), family, trajectory, "initialization",
                ),
                "logical_start_sha256": context["logical_start_sha256"],
                "method_id": config["method_id"],
                "raw_version": BP_IMH_RAW_VERSION,
                "registry_sha256": registry["registry_sha256"],
                "resource": config["resource"],
                "sampler_seed": derive_seed(
                    config["seed_namespace"], config_sha256, registry["registry_sha256"],
                    sha256_json(config["cell"]), family, trajectory, "trajectory",
                ),
                "source_identity_sha256": source_identity["source_identity_sha256"],
                "source_proposal_sha256": [
                    proposal.proposal_sha256 for proposal in context["proposals"]
                ],
                "task_version": TASK_VERSION,
                "trajectory_index": trajectory,
            }
            tasks.append({**identity, "task_fingerprint": sha256_json(identity)})
    _require(len(tasks) == 24 and len({task["task_fingerprint"] for task in tasks}) == 24,
             "BP-IMH task manifest is incomplete or duplicated")
    core = {
        "contract_version": CONTRACT_VERSION,
        "config_sha256": config_sha256,
        "source_identity": source_identity,
        "tasks": tasks,
    }
    return {**core, "manifest_sha256": sha256_json(core)}


def _initial_state(context, task):
    family = task["init_family"]
    if family == "P":
        return context["planted"].copy()
    if family == "U":
        return uniform_hard_coset_state(
            context["model"], context["syndrome"], task["initialization_seed"],
        )
    if family == "L":
        return context["logical_starts"][task["trajectory_index"]]["state"].copy()
    raise LocalBpImhError("unknown BP-IMH initialization family")


def _raw_payload(raw, task, context, config_sha256, source_identity):
    metadata = {
        "raw_version": np.array(BP_IMH_RAW_VERSION),
        "contract_version": np.array(CONTRACT_VERSION),
        "bp_imh_version": np.array(BP_IMH_VERSION),
        "task_json": np.array(canonical_json(task)),
        "task_fingerprint": np.array(task["task_fingerprint"]),
        "config_sha256": np.array(config_sha256),
        "registry_sha256": np.array(context["registry"]["registry_sha256"]),
        "source_identity_sha256": np.array(source_identity["source_identity_sha256"]),
        "combined_proposal_sha256": np.array(context["combined"].proposal_sha256),
        "source_proposal_sha256": np.asarray(
            [proposal.proposal_sha256 for proposal in context["proposals"]], dtype="U64",
        ),
        "logical_start_sha256": np.array(context["logical_start_sha256"]),
        "syndrome_packed": np.packbits(context["syndrome"], bitorder="little"),
        "model_fingerprint": np.array(context["model"].fingerprint()),
        "section_fingerprint": np.array(context["model"].logical_sector_section.fingerprint()),
        "frame_fingerprint": np.array(context["frame"].fingerprint()),
    }
    return {**metadata, **{f"sampler__{key}": value for key, value in raw.items()}}


def _sampler_from_payload(payload):
    return {
        key.removeprefix("sampler__"): np.asarray(value)
        for key, value in payload.items() if key.startswith("sampler__")
    }


def _trajectory_frequencies(labels):
    labels = np.asarray(labels, dtype=np.uint64)
    _require(labels.ndim == 2 and labels.shape[0] >= 4,
             "BP-IMH collision estimator needs independent trajectory rows")
    frequencies = []
    for row in labels:
        unique, counts = np.unique(row, return_counts=True)
        frequencies.append(dict(zip(
            (int(value) for value in unique),
            (int(count) / row.size for count in counts),
        )))
    return frequencies


def _frequency_overlap(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(label, 0.0) for label, value in left.items())


def _within_collision(frequencies):
    collisions = [
        _frequency_overlap(frequencies[left], frequencies[right])
        for left in range(len(frequencies))
        for right in range(left + 1, len(frequencies))
    ]
    _require(bool(collisions), "BP-IMH collision estimator has too few trajectories")
    return float(np.mean(collisions))


def _cross_collision(left, right):
    collisions = [
        _frequency_overlap(left_value, right_value)
        for left_value in left for right_value in right
    ]
    _require(bool(collisions), "BP-IMH cross collision estimator is empty")
    return float(np.mean(collisions))


def _q_top(labels):
    labels = np.asarray(labels, dtype=np.uint64)

    def estimate(rows):
        purity = _within_collision(_trajectory_frequencies(rows))
        uniform = math.ldexp(1.0, -64)
        return (purity - uniform) / (1.0 - uniform)

    value = estimate(labels)
    leave_one = np.asarray([
        estimate(np.delete(labels, index, axis=0)) for index in range(labels.shape[0])
    ], dtype=np.float64)
    mean = float(leave_one.mean())
    se = math.sqrt((labels.shape[0] - 1) / labels.shape[0]
                   * float(np.sum((leave_one - mean) ** 2)))
    return {"q_top": value, "jackknife_se": se, "leave_one": leave_one.tolist()}


def _distribution_d2(left_labels, right_labels):
    """Unbiased full-label D2 with separate-side trajectory jackknife."""
    left_labels = np.asarray(left_labels, dtype=np.uint64)
    right_labels = np.asarray(right_labels, dtype=np.uint64)
    _require(left_labels.ndim == right_labels.ndim == 2
             and left_labels.shape[0] >= 4 and right_labels.shape[0] >= 4,
             "BP-IMH D2 needs two independent trajectory ensembles")
    uniform = math.ldexp(1.0, -64)
    scale = 1.0 - uniform

    def estimate(left_rows, right_rows):
        left = _trajectory_frequencies(left_rows)
        right = _trajectory_frequencies(right_rows)
        return (
            _within_collision(left) + _within_collision(right)
            - 2.0 * _cross_collision(left, right)
        ) / scale

    value = estimate(left_labels, right_labels)
    delete_by_side = []
    for side, source in enumerate((left_labels, right_labels)):
        delete_by_side.append(np.asarray([
            estimate(
                np.delete(left_labels, omitted, axis=0) if side == 0 else left_labels,
                np.delete(right_labels, omitted, axis=0) if side == 1 else right_labels,
            )
            for omitted in range(source.shape[0])
        ], dtype=np.float64))
    variance = sum(
        (values.size - 1.0) / values.size
        * float(np.square(values - values.mean()).sum(dtype=np.float64))
        for values in delete_by_side
    )
    return {
        "d2_norm": float(value),
        "jackknife_se": float(math.sqrt(variance)),
        "delete_one_left": delete_by_side[0].tolist(),
        "delete_one_right": delete_by_side[1].tolist(),
    }


def _chain_mean(values):
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(means.size))


def _character_traces(labels):
    labels = np.asarray(labels, dtype=np.uint64)
    return np.stack([
        1.0 - 2.0 * ((labels >> np.uint64(bit)) & np.uint64(1)).astype(np.float64)
        for bit in range(64)
    ], axis=1)


def _trace_gate(weight, b_weight, characters, gates):
    observables = [("physical_weight", weight), ("b_weight", b_weight)]
    observables.extend((f"basis_{bit}", characters[:, bit, :]) for bit in range(64))
    records = {}
    passed = True
    for name, values in observables:
        values = np.asarray(values, dtype=np.float64)
        constant = bool(np.all(values == values.flat[0]))
        if constant:
            rhat, ess = 1.0, float(values.size)
        else:
            rhat, ess = split_rhat(values), bulk_ess(values)
        gate = bool(rhat <= gates["max_rhat"] and ess >= gates["min_bulk_ess"])
        records[name] = {
            "constant": constant, "split_rhat": float(rhat),
            "bulk_ess": float(ess), "pass": gate,
        }
        passed &= gate
    return records, bool(passed)


def _segment_stability(labels, weight, b_weight, num_qubits, b_dimension, gates):
    half = labels.shape[1] // 2
    q_left, q_right = _q_top(labels[:, :half]), _q_top(labels[:, half:])
    q_delta = abs(q_left["q_top"] - q_right["q_top"])
    q_se = math.hypot(q_left["jackknife_se"], q_right["jackknife_se"])

    def compare(values, scale, absolute, slack):
        left_mean, left_se = _chain_mean(values[:, :half] / scale)
        right_mean, right_se = _chain_mean(values[:, half:] / scale)
        delta = abs(left_mean - right_mean)
        se = math.hypot(left_se, right_se)
        return {
            "left": left_mean, "right": right_mean, "absolute_difference": delta,
            "difference_se": se,
            "pass": bool(delta <= absolute and delta <= gates["se_multiple"] * se + slack),
        }

    weight_result = compare(weight, num_qubits, gates["max_abs_weight_delta"], 1.0 / num_qubits)
    b_result = compare(b_weight, b_dimension, gates["max_abs_b_weight_delta"], 1.0 / b_dimension)
    q_result = {
        "left": q_left, "right": q_right, "absolute_difference": q_delta,
        "difference_se": q_se,
        "pass": bool(q_delta <= gates["max_abs_q_top_delta"]
                     and q_delta <= gates["se_multiple"] * q_se + gates["q_top_se_slack"]),
    }
    return {
        "q_top": q_result, "physical_weight": weight_result, "b_weight": b_result,
        "pass": bool(q_result["pass"] and weight_result["pass"] and b_result["pass"]),
    }


def _family_data(items, context, gates):
    labels = np.stack([np.asarray(raw["measurement_labels"], dtype=np.uint64)
                       for _task, raw in items])
    weights = np.stack([np.asarray(raw["measurement_weights"], dtype=np.float64)
                        for _task, raw in items])
    b_start = context["H"].shape[1] ** 2
    b_dimension = context["H"].shape[0] ** 2
    b_weights = []
    for _task, raw in items:
        states = np.unpackbits(
            raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
            bitorder="little",
        )
        b_weights.append(states[:, b_start:b_start + b_dimension].sum(axis=1))
    b_weights = np.asarray(b_weights, dtype=np.float64)
    characters = _character_traces(labels)
    q_top = _q_top(labels)
    weight_mean, weight_se = _chain_mean(weights / context["model"].num_qubits)
    b_mean, b_se = _chain_mean(b_weights / b_dimension)
    character_chain_means = characters.mean(axis=2)
    trace_records, trace_pass = _trace_gate(weights, b_weights, characters, gates)
    stability = _segment_stability(
        labels, weights, b_weights, context["model"].num_qubits, b_dimension, gates,
    )
    return {
        "labels": labels, "weights": weights, "b_weights": b_weights,
        "characters": characters, "character_chain_means": character_chain_means,
        "summary": {
            "q_top": q_top,
            "normalized_weight_mean": weight_mean,
            "normalized_weight_se": weight_se,
            "normalized_b_weight_mean": b_mean,
            "normalized_b_weight_se": b_se,
            "basis_character_means": character_chain_means.mean(axis=0).tolist(),
            "trace_gates": trace_records,
            "trace_pass": trace_pass,
            "early_late": stability,
            "pass": bool(q_top["jackknife_se"] <= gates["max_q_top_se"]
                         and trace_pass and stability["pass"]),
        },
    }


def _compare_families(left, right, gates, num_qubits, b_dimension):
    left_summary, right_summary = left["summary"], right["summary"]
    q_delta = abs(left_summary["q_top"]["q_top"] - right_summary["q_top"]["q_top"])
    q_se = math.hypot(
        left_summary["q_top"]["jackknife_se"], right_summary["q_top"]["jackknife_se"],
    )

    def scalar(name, absolute, slack):
        delta = abs(left_summary[f"normalized_{name}_mean"]
                    - right_summary[f"normalized_{name}_mean"])
        se = math.hypot(left_summary[f"normalized_{name}_se"],
                        right_summary[f"normalized_{name}_se"])
        return {
            "absolute_difference": delta, "difference_se": se,
            "pass": bool(delta <= absolute and delta <= gates["se_multiple"] * se + slack),
        }

    weight = scalar("weight", gates["max_abs_weight_delta"], 1.0 / num_qubits)
    b_weight = scalar("b_weight", gates["max_abs_b_weight_delta"], 1.0 / b_dimension)
    left_char = np.asarray(left_summary["basis_character_means"], dtype=np.float64)
    right_char = np.asarray(right_summary["basis_character_means"], dtype=np.float64)
    deltas = np.abs(left_char - right_char)
    character = {
        "max_absolute_difference": float(deltas.max()),
        "mean_squared_difference": float(np.mean(deltas * deltas)),
    }
    character["pass"] = bool(
        character["max_absolute_difference"] <= gates["character_max_abs_delta"]
        and character["mean_squared_difference"] <= gates["character_d2_max"]
    )
    d2 = _distribution_d2(left["labels"], right["labels"])
    d2["three_se_upper"] = max(0.0, d2["d2_norm"]) + gates["se_multiple"] * d2[
        "jackknife_se"
    ]
    d2["pass"] = bool(d2["three_se_upper"] <= gates["character_d2_max"])
    q_result = {
        "absolute_difference": q_delta, "difference_se": q_se,
        "pass": bool(q_delta <= gates["max_abs_q_top_delta"]
                     and q_delta <= gates["se_multiple"] * q_se + gates["q_top_se_slack"]),
    }
    return {
        "q_top": q_result, "physical_weight": weight, "b_weight": b_weight,
        "basis_characters": character, "full_label_d2": d2,
        "pass": bool(q_result["pass"] and weight["pass"]
                     and b_weight["pass"] and character["pass"] and d2["pass"]),
    }


def _kernel_gates(results, context, config):
    gates = config["gates"]
    records = []
    passed = True
    for task, raw in results:
        burn_changes = int(np.asarray(raw["burn_state_changed"]).sum())
        measurement_changes = int(np.asarray(raw["measurement_state_changed"]).sum())
        measurement_rate = measurement_changes / config["resource"]["measurement_steps"]
        outer = np.bincount(raw["measurement_proposal_source_indices"], minlength=2)
        components = np.zeros((2, 3), dtype=np.int64)
        for source, component in zip(
                raw["measurement_proposal_source_indices"],
                raw["measurement_proposal_component_indices"], strict=True):
            components[int(source), int(component)] += 1
        record_pass = bool(
            measurement_changes >= gates["min_measurement_state_changes"]
            and measurement_rate >= gates["min_measurement_state_change_rate"]
            and int(outer.min()) >= gates["min_outer_source_draws_per_trajectory"]
            and int(components.min()) >= gates["min_each_internal_component_draws_per_trajectory"]
        )
        if task["init_family"] == "U":
            record_pass &= bool(
                burn_changes >= gates["min_burn_state_changes_u"]
                and int(raw["burn_end_weight"]) / context["model"].num_qubits
                <= gates["max_burn_end_normalized_weight_u"]
            )
        records.append({
            "family": task["init_family"], "trajectory_index": task["trajectory_index"],
            "burn_state_changes": burn_changes,
            "burn_end_weight": int(raw["burn_end_weight"]),
            "measurement_state_changes": measurement_changes,
            "measurement_state_change_rate": measurement_rate,
            "measurement_outer_source_counts": outer.tolist(),
            "measurement_source_component_counts": components.tolist(),
            "pass": record_pass,
        })
        passed &= record_pass
    return records, bool(passed)


def _common_freeze_gate(results):
    measurement = np.stack([
        np.asarray(raw["measurement_labels"], dtype=np.uint64) for _task, raw in results
    ])
    initial = np.asarray([int(raw["initial_label"]) for _task, raw in results], dtype=np.uint64)
    failures = []
    for bit in range(64):
        measured_sign = 1 - 2 * ((measurement >> np.uint64(bit)) & np.uint64(1)).astype(np.int8)
        if np.unique(measured_sign).size != 1:
            continue
        common = int(measured_sign.flat[0])
        initial_sign = 1 - 2 * ((initial >> np.uint64(bit)) & np.uint64(1)).astype(np.int8)
        for index in np.flatnonzero(initial_sign != common):
            burn = np.asarray(results[int(index)][1]["burn_labels"], dtype=np.uint64)
            burn_sign = 1 - 2 * ((burn >> np.uint64(bit)) & np.uint64(1)).astype(np.int8)
            if not np.any(burn_sign == common):
                failures.append({
                    "bit": bit,
                    "family": results[int(index)][0]["init_family"],
                    "trajectory_index": results[int(index)][0]["trajectory_index"],
                })
    return {"failure_count": len(failures), "failures": failures, "pass": not failures}


def _analyze(results, context, config):
    gates = config["gates"]
    by_family = {
        family: [(task, raw) for task, raw in results if task["init_family"] == family]
        for family in FAMILIES
    }
    _require(all(len(items) == 8 for items in by_family.values()),
             "BP-IMH analysis family count changed")
    family_data = {
        family: _family_data(items, context, gates) for family, items in by_family.items()
    }
    comparisons = {}
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1:]:
            comparisons[f"{left}_vs_{right}"] = _compare_families(
                family_data[left], family_data[right], gates,
                context["model"].num_qubits, context["H"].shape[0] ** 2,
            )
    kernel_records, kernel_pass = _kernel_gates(results, context, config)
    common_freeze = _common_freeze_gate(results)
    family_summaries = {family: data["summary"] for family, data in family_data.items()}
    passed = bool(
        kernel_pass and common_freeze["pass"]
        and all(summary["pass"] for summary in family_summaries.values())
        and all(comparison["pass"] for comparison in comparisons.values())
    )
    return {
        "family_summaries": family_summaries,
        "cross_family_comparisons": comparisons,
        "kernel_gates": kernel_records,
        "kernel_pass": kernel_pass,
        "common_freeze_gate": common_freeze,
        "pass": passed,
        "terminal_status": (
            "LOCAL_BP_IMH_TRANSPORT_VIABLE_FOR_HARD2"
            if passed else "LOCAL_BP_IMH_TRANSPORT_UNRESOLVED"
        ),
    }


def run(config_path, output_dir):
    output_dir = Path(output_dir)
    _require(output_dir.resolve() == ROOT.resolve(), "BP-IMH output directory changed")
    report_path = output_dir / "bp_imh_report.json"
    receipt_path = output_dir / "run_receipt.json"
    manifest_path = output_dir / "task_manifest.json"
    raw_dir = output_dir / "raw"
    _require(not report_path.exists() and not receipt_path.exists() and not manifest_path.exists(),
             "refusing to overwrite a BP-IMH run")
    _require(not raw_dir.exists(), "refusing a pre-existing BP-IMH raw directory")

    config, config_sha256 = _load_config(config_path)
    source_identity = _source_identity(config_path)
    context = _context(config)
    manifest = _task_manifest(context, config, config_sha256, source_identity)
    atomic_json(manifest_path, manifest)

    generated = []
    timings = []
    for task in manifest["tasks"]:
        initial = _initial_state(context, task)
        started = time.perf_counter()
        raw = run_bp_imh_trajectory(
            context["model"], context["frame"], context["syndrome"],
            config["cell"]["p"], context["combined"], initial, task["sampler_seed"],
            burn_steps=config["resource"]["burn_steps"],
            measurement_steps=config["resource"]["measurement_steps"],
        )
        elapsed = time.perf_counter() - started
        validate_bp_imh_transcript(
            context["model"], context["frame"], context["syndrome"],
            config["cell"]["p"], context["combined"], raw,
            burn_steps=config["resource"]["burn_steps"],
            measurement_steps=config["resource"]["measurement_steps"],
        )
        replay_bp_imh_trajectory(
            context["model"], context["frame"], context["syndrome"],
            config["cell"]["p"], context["combined"], initial,
            task["sampler_seed"], raw,
            burn_steps=config["resource"]["burn_steps"],
            measurement_steps=config["resource"]["measurement_steps"],
        )
        generated.append((task, raw))
        timings.append({
            "family": task["init_family"], "trajectory_index": task["trajectory_index"],
            "wall_seconds": elapsed,
        })

    raw_dir.mkdir(parents=False, exist_ok=False)
    raw_records = []
    for task, raw in generated:
        filename = f"{task['init_family']}_{task['trajectory_index']:02d}.npz"
        path = raw_dir / filename
        payload = _raw_payload(raw, task, context, config_sha256, source_identity)
        atomic_npz(path, **payload)
        with load_npz_no_pickle(path) as stored:
            _require(set(stored.files) == set(payload), "BP-IMH stored raw field set changed")
            for key, value in payload.items():
                _require(np.array_equal(stored[key], np.asarray(value)),
                         f"BP-IMH stored raw mismatch: {filename}:{key}")
        raw_records.append({
            "family": task["init_family"], "trajectory_index": task["trajectory_index"],
            "path": path.relative_to(EXP102_ROOT).as_posix(),
            "raw_sha256": sha256_file(path), "task_fingerprint": task["task_fingerprint"],
        })
    receipt_core = {
        "contract_version": CONTRACT_VERSION,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_records": raw_records,
        "raw_set_sha256": sha256_json(raw_records),
        "replay_count": len(generated),
        "timings": timings,
    }
    receipt = {**receipt_core, "receipt_sha256": sha256_json(receipt_core)}
    atomic_json(receipt_path, receipt)

    analysis = _analyze(generated, context, config)
    report_core = {
        "analysis": analysis,
        "authority": "local_diagnostic_only_not_formal_or_reportable_q_top",
        "config_sha256": config_sha256,
        "contract_version": CONTRACT_VERSION,
        "does_not_establish": [
            "A reportable posterior q_top or a parameter-point physics result.",
            "A global tail/normalizer certificate or an independent confirmation mechanism.",
            "Any remote, formal, held-out, or production authorization.",
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "proposal": {
            "combined_proposal_sha256": context["combined"].proposal_sha256,
            "source_proposal_sha256": [
                proposal.proposal_sha256 for proposal in context["proposals"]
            ],
            "logical_start_sha256": context["logical_start_sha256"],
        },
        "raw_set_sha256": receipt["raw_set_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "report_version": REPORT_VERSION,
        "source_identity": source_identity,
        "terminal_status": analysis["terminal_status"],
    }
    report = {**report_core, "report_sha256": sha256_json(report_core)}
    atomic_json(report_path, report)
    print(report["terminal_status"])
    print(report["report_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
