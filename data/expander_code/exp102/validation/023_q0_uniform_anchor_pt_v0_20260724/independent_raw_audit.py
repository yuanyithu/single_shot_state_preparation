"""Raw-only audit for the UARE V0 local screen.

This intentionally does not import the UARE sampler or its runner.  It reads
pickle-free NPZ data, rebuilds all hard-coset algebra and frozen gates, and
independently derives the terminal local status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, canonical_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import uniform_hard_coset_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _state_label,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_uniform_anchor_pt.v0"
LOCAL_RAW_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.raw.v0"
SAMPLER_RAW_VERSION = "exp102.q0_hgp_uniform_anchor_pt.raw.v0"
AUDIT_VERSION = "exp102.q0_hgp_uniform_anchor_pt.independent_raw_audit.v2"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
INIT_FAMILIES = ("P", "U", "L")

SAMPLER_FIELDS = frozenset({
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "plan_json", "plan_sha256", "initial_state_packed",
    "burn_state_packed", "final_state_packed", "measurement_states_packed",
    "measurement_b_columns", "measurement_a_syndromes", "burn_labels",
    "measurement_labels", "measurement_weights", "measurement_residual_weights",
    "measurement_block", "burn_complete_scores", "measurement_complete_scores",
    "burn_b_weights", "measurement_b_weights", "burn_row_counters",
    "measurement_row_counters", "burn_hot_refresh_changed_bits",
    "measurement_hot_refresh_changed_bits", "burn_cold_a_column_draws",
    "measurement_cold_a_column_draws", "burn_swap_attempts", "burn_swap_accepts",
    "measurement_swap_attempts", "measurement_swap_accepts", "lambda_values",
    "lambda_sha256", "mass_sha256", "initial_label", "burn_label", "final_label",
    "engine",
})
RAW_FIELDS = SAMPLER_FIELDS | {
    "sampler_raw_version", "contract_version", "local_raw_version",
    "task_fingerprint", "task_json", "manifest_sha256", "config_sha256",
    "registry_sha256", "source_binding_sha256", "cell_json", "uniform_seed",
    "model_fingerprint", "frame_fingerprint", "logical_masks", "b_masks",
    "character_sha256", "core_seconds", "wall_seconds", "init_family",
    "trajectory_index",
}


class AuditConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditConflict(message)


def _scalar(raw, name):
    value = raw.get(name)
    _require(value is not None and value.shape == (), f"missing raw scalar: {name}")
    return value.item()


def _array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _float_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=">f8").tobytes()).hexdigest()


def _cosine_ladder(replicas):
    values = np.asarray([
        0.5 * (1.0 - math.cos(math.pi * index / (replicas - 1)))
        for index in range(replicas)
    ], dtype=np.float64)
    values[0], values[-1] = 0.0, 1.0
    return values


def _frozen_masks(config_sha256, registry_sha256, rows):
    from exp101_certified_src.prng import PortablePrng

    seed = derive_seed(
        "exp102.q0_hgp_uniform_anchor_pt.v0", "fixed_characters",
        config_sha256, registry_sha256, "uint64_bit63_safe_v1",
    )
    rng = PortablePrng(seed)
    basis = np.asarray([np.uint64(1) << np.uint64(bit) for bit in range(64)], dtype=np.uint64)
    seen = {int(value) for value in basis}
    nonbasis = []
    while len(nonbasis) < 64:
        value = int(rng.next_uint64())
        if value and value not in seen:
            seen.add(value)
            nonbasis.append(np.uint64(value))
    masks = np.zeros((64, rows), dtype=np.uint32)
    row_mask = (1 << rows) - 1
    for index in range(64):
        while not masks[index].any():
            for column in range(rows):
                masks[index, column] = np.uint32(int(rng.next_uint64()) & row_mask)
    return np.concatenate((basis, np.asarray(nonbasis, dtype=np.uint64))), masks


def _uniform_seed(registry, code, cell):
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _seed_for_uniform_start(identity):
    return derive_seed(
        "exp102.q0_hgp_uniform_anchor_pt.v0", identity["trajectory_namespace"],
        identity["source_commit"], identity["config_sha256"], identity["registry_sha256"],
        identity["cell_fingerprint"], identity["method_id"], identity["resource_tier"],
        identity["init_family"], int(identity["trajectory_index"]), "initialize",
        "hard_coset", 0,
    )


def _unpack_state(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)


def _complete_score(b_columns, a_syndromes, log_mass, log_odds):
    result = 0.0
    for value in np.asarray(b_columns, dtype=np.uint32):
        result += int(value).bit_count() * log_odds
    for value in np.asarray(a_syndromes, dtype=np.uint32):
        result += float(log_mass[int(value)])
    return result


def _expected_swap_attempts(replicas, offset, rounds):
    values = np.zeros(replicas - 1, dtype=np.int64)
    for round_index in range(offset, offset + rounds):
        for lower in range(round_index & 1, replicas - 1, 2):
            values[lower] += 1
    return values


def _load_manifest(root):
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="ascii"))
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest.get("manifest_sha256") == sha256_json(core), "manifest hash mismatch")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "manifest contract mismatch")
    _require(manifest["raw_version"] == LOCAL_RAW_VERSION, "manifest raw version mismatch")
    _require(len(manifest["tasks"]) == 48, "manifest task count mismatch")
    _require(len({item["task_fingerprint"] for item in manifest["tasks"]}) == 48,
             "manifest tasks are not unique")
    for task in manifest["tasks"]:
        task_core = {key: value for key, value in task.items() if key != "task_fingerprint"}
        _require(task["task_fingerprint"] == sha256_json(task_core), "task fingerprint mismatch")
    return manifest


def _validate_source_binding(manifest):
    binding = manifest["source_binding"]
    core = {"source_commit": binding["source_commit"], "files": binding["files"]}
    _require(binding.get("source_binding_sha256") == sha256_json(core), "source binding hash mismatch")
    current_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(current_commit == binding["source_commit"], "source commit changed after manifest")
    expected_files = {
        "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
        "run_local_viability.py": ROOT / "run_local_viability.py",
        "config": EXP102_ROOT / "config" / "q0_hgp_uniform_anchor_pt.v0.json",
    }
    _require(set(binding["files"]) == set(expected_files), "source binding file set mismatch")
    for name, path in expected_files.items():
        _require(binding["files"][name] == sha256_file(path), f"source file changed: {name}")


def _load_context(manifest, control):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == manifest["registry_sha256"], "registry mismatch")
    config = manifest["config"]
    _require(config["contract_version"] == CONTRACT_VERSION, "config contract mismatch")
    _require(sha256_json(config) == manifest["config_sha256"], "config hash mismatch")
    _require(config["registry_sha256"] == registry["registry_sha256"], "config registry mismatch")
    _require(config["init_families"] == list(INIT_FAMILIES), "config init families mismatch")
    _require(config["resource"] == {"burn_rounds": 256, "measurement_rounds": 2048, "name": "V0"},
             "config resource mismatch")
    _require(config["methods"] == [
        {"id": "UARE32-R1", "num_replicas": 32, "positive_row_updates_per_round": 1},
        {"id": "UARE64-R1", "num_replicas": 64, "positive_row_updates_per_round": 1},
    ], "config method mismatch")
    _, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    _require(model.k == 64, "logical dimension mismatch")
    seed = _uniform_seed(registry, code, config["cell"])
    epsilon = (
        np.random.Generator(np.random.PCG64(seed)).random(model.num_qubits)
        < float(config["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(np.array_equal(control["epsilon"], epsilon), "control epsilon mismatch")
    _require(np.array_equal(control["syndrome"], syndrome), "control syndrome mismatch")
    l_move = np.asarray(control["l_move"], dtype=np.uint8)
    residual = (model.H_check.astype(np.int64) @ l_move.astype(np.int64) % 2).astype(np.uint8)
    _require(not residual.any() and _state_label(frame, l_move) != 0, "control L move mismatch")
    logical_masks, b_masks = _frozen_masks(manifest["config_sha256"], registry["registry_sha256"], H.shape[0])
    _require(np.array_equal(control["logical_masks"], logical_masks), "control logical masks mismatch")
    _require(np.array_equal(control["b_masks"], b_masks), "control B masks mismatch")
    # This invokes the separately tested collapsed mass DP, never the UARE
    # transition or its runner.  The reference implementation is deliberately
    # too slow at r=24 for a terminal raw audit.
    mass = build_classical_coset_mass(H, config["cell"]["p"], engine="numba")
    _require(np.array_equal(control["classical_mass"], mass), "control classical mass mismatch")
    _require(manifest["mass_sha256"] == _float_sha256(mass), "manifest mass hash mismatch")
    _require(manifest["H_sha256"] == _array_sha256(H), "manifest H mismatch")
    _require(manifest["syndrome_sha256"] == _array_sha256(syndrome), "manifest syndrome mismatch")
    _require(manifest["epsilon_sha256"] == _array_sha256(epsilon), "manifest epsilon mismatch")
    return {
        "registry": registry, "config": config, "H": np.ascontiguousarray(H, dtype=np.uint8),
        "model": model, "frame": frame, "epsilon": epsilon, "syndrome": syndrome,
        "l_move": l_move, "mass": mass, "seed": seed,
    }


def _initial_state(task, context):
    family = task["init_family"]
    if family == "P":
        state = context["epsilon"].copy()
    elif family == "L":
        state = context["epsilon"] ^ context["l_move"]
    elif family == "U":
        state = uniform_hard_coset_state(
            context["model"], context["syndrome"], _seed_for_uniform_start(task["seed_identity"]),
        )
    else:  # pragma: no cover
        raise AuditConflict("unknown task family")
    return np.ascontiguousarray(state, dtype=np.uint8)


def _sampler_config(method, config):
    replicas = int(method["num_replicas"])
    ladder = _cosine_ladder(replicas)
    return {
        "method_id": method["id"], "p": float(config["cell"]["p"]),
        "burn_rounds": int(config["resource"]["burn_rounds"]),
        "measurement_rounds": int(config["resource"]["measurement_rounds"]),
        "num_replicas": replicas,
        "positive_row_updates_per_round": int(method["positive_row_updates_per_round"]),
        "lambda_values": ladder.tolist(), "lambda_schedule": "cosine_endpoint_cluster_v1",
        "kernel": "uniform_endpoint_full_collapsed_energy_replica_exchange.v1",
        "hot_endpoint": "exact_uniform_B_refresh",
        "tempered_term": "complete_collapsed_log_density",
    }


def _validate_raw(path, task, manifest, context):
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    _require(set(raw) == RAW_FIELDS, f"raw schema mismatch: {path.name}")
    _require(not any(value.dtype.hasobject for value in raw.values()), "object raw field")
    method = next(item for item in context["config"]["methods"] if item["id"] == task["method_id"])
    sampler = _sampler_config(method, context["config"])
    expected = {
        "raw_version": LOCAL_RAW_VERSION, "sampler_raw_version": SAMPLER_RAW_VERSION,
        "contract_version": CONTRACT_VERSION, "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"], "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": manifest["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "method_id": task["method_id"], "init_family": task["init_family"],
        "trajectory_index": int(task["trajectory_index"]), "engine": "numba",
        "lambda_sha256": _float_sha256(np.asarray(sampler["lambda_values"], dtype=np.float64)),
        "mass_sha256": _float_sha256(context["mass"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
    }
    for name, value in expected.items():
        _require(str(_scalar(raw, name)) == str(value), f"raw identity mismatch {name}: {path.name}")
    _require(int(_scalar(raw, "uniform_seed")) == context["seed"], "raw uniform seed mismatch")
    _require(str(_scalar(raw, "sampler_config_json")) == canonical_json(sampler), "sampler config mismatch")
    _require(str(_scalar(raw, "sampler_config_sha256")) == sha256_json(sampler), "sampler config hash mismatch")
    _require(str(_scalar(raw, "seed_identity_json")) == canonical_json(task["seed_identity"]), "seed identity mismatch")
    _require(np.array_equal(raw["lambda_values"], np.asarray(sampler["lambda_values"])), "ladder mismatch")
    _require(np.array_equal(raw["logical_masks"], control_global["logical_masks"]), "raw logical masks mismatch")
    _require(np.array_equal(raw["b_masks"], control_global["b_masks"]), "raw B masks mismatch")
    _require(str(_scalar(raw, "character_sha256")) == manifest["character_sha256"], "character hash mismatch")
    initial = _initial_state(task, context)
    _require(np.array_equal(_unpack_state(raw["initial_state_packed"], context["model"].num_qubits), initial),
             "initial state mismatch")
    for name in ("initial_state_packed", "burn_state_packed", "final_state_packed"):
        state = _unpack_state(raw[name], context["model"].num_qubits)
        residual = (context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8) ^ context["syndrome"]
        _require(not residual.any(), f"hard-coset violation in {name}")
    count = int(context["config"]["resource"]["measurement_rounds"])
    measurement = np.unpackbits(
        raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    _require(measurement.shape == (count, context["model"].num_qubits), "measurement shape mismatch")
    residual = (context["model"].H_check.astype(np.int64) @ measurement.T.astype(np.int64) % 2).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(not residual.any() and not raw["measurement_residual_weights"].any(), "measurement hard-coset violation")
    labels = np.asarray([_state_label(context["frame"], state) for state in measurement], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "label cache mismatch")
    _require(np.array_equal(measurement.sum(axis=1).astype(np.int32), raw["measurement_weights"]), "weight cache mismatch")
    log_mass = np.log(context["mass"])
    log_odds = math.log(float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"])))
    scores, b_weights = np.empty(count), np.empty(count, dtype=np.int32)
    for index, state in enumerate(measurement):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, context["syndrome"], context["H"])
        _require(np.array_equal(b_columns, raw["measurement_b_columns"][index]), "B trace mismatch")
        _require(np.array_equal(a_syndromes, raw["measurement_a_syndromes"][index]), "A trace mismatch")
        scores[index] = _complete_score(b_columns, a_syndromes, log_mass, log_odds)
        b_weights[index] = sum(int(value).bit_count() for value in b_columns)
    _require(np.array_equal(scores, raw["measurement_complete_scores"]), "score trace mismatch")
    _require(np.array_equal(b_weights, raw["measurement_b_weights"]), "B weight trace mismatch")
    for phase, rounds, offset in (("burn", int(context["config"]["resource"]["burn_rounds"]), 0),
                                  ("measurement", count, int(context["config"]["resource"]["burn_rounds"]))):
        counters = raw[f"{phase}_row_counters"]
        replicas = int(method["num_replicas"])
        updates = int(method["positive_row_updates_per_round"])
        _require(counters.shape == (replicas, 3) and not counters[0].any(), "row counter shape mismatch")
        _require(int(counters[:, 0].sum()) == rounds * (replicas - 1) * updates, "row attempt mismatch")
        _require(np.all(counters[:, 1] <= counters[:, 0]) and np.all(counters[:, 2] >= counters[:, 1]),
                 "row counter values mismatch")
        hot = raw[f"{phase}_hot_refresh_changed_bits"]
        _require(hot.shape == (rounds,) and np.all((hot >= 0) & (hot <= context["H"].shape[0] ** 2)),
                 "hot refresh mismatch")
        _require(int(_scalar(raw, f"{phase}_cold_a_column_draws")) == rounds * context["H"].shape[1],
                 "A redraw count mismatch")
        _require(np.array_equal(raw[f"{phase}_swap_attempts"], _expected_swap_attempts(replicas, offset, rounds)),
                 "swap attempt mismatch")
        _require(np.all(raw[f"{phase}_swap_accepts"] <= raw[f"{phase}_swap_attempts"]), "swap accept mismatch")
    return raw


def _parity_uint64(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        values ^= values >> np.uint64(shift)
    return values & np.uint64(1)


def _parity_uint32(values):
    values = np.asarray(values, dtype=np.uint32).copy()
    for shift in (16, 8, 4, 2, 1):
        values ^= values >> np.uint32(shift)
    return values & np.uint32(1)


def _trajectory_values(raw, context):
    labels = np.asarray(raw["measurement_labels"], dtype=np.uint64)
    logical = 1.0 - 2.0 * _parity_uint64(labels[:, None] & control_global["logical_masks"][None, :])
    b_columns = np.asarray(raw["measurement_b_columns"], dtype=np.uint32)
    b_signs = np.empty((b_columns.shape[0], control_global["b_masks"].shape[0]), dtype=np.float64)
    for mask_index, mask in enumerate(control_global["b_masks"]):
        parity = np.zeros(b_columns.shape[0], dtype=np.uint32)
        for column in range(b_columns.shape[1]):
            parity ^= _parity_uint32(b_columns[:, column] & mask[column])
        b_signs[:, mask_index] = 1.0 - 2.0 * parity
    half = labels.size // 2
    series = {
        "normalized_weight": np.asarray(raw["measurement_weights"], dtype=np.float64) / context["model"].num_qubits,
        "normalized_B_weight": np.asarray(raw["measurement_b_weights"], dtype=np.float64) / (context["H"].shape[0] ** 2),
        "complete_score_per_factor": np.asarray(raw["measurement_complete_scores"], dtype=np.float64) / context["H"].shape[1],
        "logical_characters": logical,
        "B_masks": b_signs,
    }
    return {name: np.asarray((value.mean(axis=0), value[:half].mean(axis=0), value[half:].mean(axis=0)))
            for name, value in series.items()}


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def _compare(a, b, absolute, gates):
    ma, sea = _mean_se(a)
    mb, seb = _mean_se(b)
    delta = np.abs(ma - mb)
    allowance = gates["sigma_multiplier"] * np.sqrt(sea * sea + seb * seb) + gates["sigma_slack"]
    return {
        "pass": bool(np.all(delta <= absolute) and np.all(delta <= allowance)),
        "max_abs_delta": float(np.max(delta)),
        "max_three_sigma_allowance": float(np.max(allowance)),
        "failed_components": int(np.count_nonzero((delta > absolute) | (delta > allowance))),
    }


def _bundle(a, b, gates):
    return {
        "normalized_weight": _compare(a["normalized_weight"], b["normalized_weight"], gates["max_abs_normalized_weight_delta"], gates),
        "normalized_B_weight": _compare(a["normalized_B_weight"], b["normalized_B_weight"], gates["max_abs_normalized_B_weight_delta"], gates),
        "complete_score_per_factor": _compare(a["complete_score_per_factor"], b["complete_score_per_factor"], gates["max_abs_complete_score_delta_per_factor"], gates),
        "logical_characters": _compare(a["logical_characters"], b["logical_characters"], gates["max_abs_logical_character_mean_delta"], gates),
        "B_masks": _compare(a["B_masks"], b["B_masks"], gates["max_abs_B_mask_mean_delta"], gates),
    }


def _all_pass(bundle):
    return all(value["pass"] for value in bundle.values())


def _method_summary(records, context):
    from exp101_certified_src.gf2 import gf2_rank

    by_family = {name: [] for name in INIT_FAMILIES}
    for raw in records:
        by_family[str(_scalar(raw, "init_family"))].append(raw)
    means, split = {}, {}
    for family in INIT_FAMILIES:
        _require(len(by_family[family]) == 8, "incomplete method family")
        values = [_trajectory_values(raw, context) for raw in by_family[family]]
        means[family] = {name: np.stack([value[name][0] for value in values]) for name in values[0]}
        split[family] = {
            name: (np.stack([value[name][1] for value in values]), np.stack([value[name][2] for value in values]))
            for name in values[0]
        }
    gates = context["config"]["gates"]
    pairs = {f"{a}_{b}": _bundle(means[a], means[b], gates) for a, b in (("P", "U"), ("P", "L"), ("U", "L"))}
    # ``split[family]`` is keyed by observable, while ``_bundle`` expects two
    # complete observable dictionaries (the first and second time halves).
    stability = {
        family: _bundle(
            {observable: halves[0] for observable, halves in split[family].items()},
            {observable: halves[1] for observable, halves in split[family].items()},
            gates,
        )
        for family in INIT_FAMILIES
    }
    p = float(context["config"]["cell"]["p"])
    log_odds = math.log(p / (1.0 - p))
    w0 = int(context["epsilon"].sum())
    dimension = int(context["model"].num_qubits - gf2_rank(context["model"].H_check))
    support = []
    for raw in by_family["U"]:
        weight = int(raw["measurement_weights"].min())
        log_bound = dimension * math.log(2.0) + (weight - w0) * log_odds
        bound = 0.0 if log_bound < -745.0 else min(1.0, math.exp(log_bound))
        support.append({"minimum_measurement_weight": weight, "target_support_upper_bound": bound,
                        "trapped_negligible_support": bool(bound <= gates["max_negligible_support_upper_bound"])})
    passed = (all(_all_pass(item) for item in pairs.values())
              and all(_all_pass(item) for item in stability.values())
              and not any(item["trapped_negligible_support"] for item in support))
    return {
        "pass": bool(passed), "independent_trajectory_count": {name: len(by_family[name]) for name in INIT_FAMILIES},
        "target_support_gate": {"reference_legal_weight": w0, "hard_coset_dimension": dimension,
                                "per_trajectory": support,
                                "pass": not any(item["trapped_negligible_support"] for item in support)},
        "pairwise": pairs, "time_stability": stability,
        "q_top_read_or_computed": False,
    }


control_global = None


def audit(root):
    global control_global
    root = Path(root)
    manifest = _load_manifest(root)
    _validate_source_binding(manifest)
    _require((root / "RUN_COMPLETE.json").is_file(), "run is incomplete")
    with np.load(root / "CONTROL.npz", allow_pickle=False) as archive:
        control_global = {name: archive[name].copy() for name in archive.files}
    _require(set(control_global) == {"epsilon", "syndrome", "l_move", "logical_masks", "b_masks", "classical_mass"},
             "control schema mismatch")
    _require(sha256_file(root / "CONTROL.npz") == manifest["control_npz_sha256"], "control hash mismatch")
    context = _load_context(manifest, control_global)
    methods = {method["id"]: [] for method in context["config"]["methods"]}
    raw_hashes = {}
    for task in manifest["tasks"]:
        filename = f'{task["method_id"]}_{task["init_family"]}_{int(task["trajectory_index"]):02d}.npz'
        path = root / "raw" / filename
        _require(path.is_file(), f"missing raw: {filename}")
        raw = _validate_raw(path, task, manifest, context)
        methods[task["method_id"]].append(raw)
        raw_hashes[filename] = sha256_file(path)
    summaries = {name: _method_summary(records, context) for name, records in methods.items()}
    passing = [name for name, value in summaries.items() if value["pass"]]
    if not passing:
        selected, status = None, context["config"]["selection"]["zero_pass"]
    elif len(passing) == 1:
        selected, status = passing[0], "LOCAL_UNIFORM_ANCHOR_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    else:
        selected, status = "UARE32-R1", "LOCAL_UNIFORM_ANCHOR_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    core = {
        "audit_version": AUDIT_VERSION, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"], "status": status,
        "selected_method": selected, "formal_authorization": False,
        "posterior_estimation": False, "q_top_read_or_computed": False,
        "methods": summaries, "raw_sha256": raw_hashes,
        "audit_source_sha256": sha256_file(Path(__file__)),
    }
    report = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(root / "INDEPENDENT_AUDIT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_hard_viability")
    args = parser.parse_args()
    print(audit(args.root)["status"])
