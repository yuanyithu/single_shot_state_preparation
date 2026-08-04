"""Frozen local adversarial screen for uniform-anchored collapsed-B exchange.

The runner intentionally has only prepare, run, and analyze phases.  The
manifest is immutable once raw exists; failed raw is never extended or pooled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import uniform_hard_coset_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _pack_state,
    _state_label,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_L_START_RULE,
    build_full_row_elimination_plan,
    select_low_energy_logical_start,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_uniform_anchor_pt import (
    UNIFORM_ANCHOR_PT_KERNEL,
    UNIFORM_ANCHOR_PT_RAW_VERSION,
    UniformAnchorReplicaExchangeConfig,
    UniformAnchorReplicaExchangeSeedIdentity,
    collapsed_complete_score,
    run_uniform_anchor_replica_exchange_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_uniform_anchor_pt.v0"
MANIFEST_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.tasks.v1"
LOCAL_RAW_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.raw.v0"
REPORT_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.report.v1"
RUNNER_VERSION = "exp102.q0_hgp_uniform_anchor_pt.local.runner.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = EXP102_ROOT / "config" / "q0_hgp_uniform_anchor_pt.v0.json"
DEFAULT_REGISTRY = EXP102_ROOT / "registry" / "registry.json"
INIT_FAMILIES = ("P", "U", "L")
LOGICAL_MASK_COUNT = 128
B_MASK_COUNT = 64

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


class LocalUareConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalUareConflict(message)


def _scalar(raw, name):
    value = raw.get(name)
    _require(value is not None and value.shape == (), f"raw scalar {name} is missing")
    return value.item()


def _array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit),
             "source commit is invalid")
    files = {
        name: sha256_file(path)
        for name, path in {
            "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
            "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
            "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
            "run_local_viability.py": Path(__file__),
            "config": Path(config_path),
        }.items()
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _load_runtime_probe(config):
    path = ROOT / "runtime_probe.json"
    _require(path.is_file(), "uniform-anchor runtime probe is missing")
    try:
        report = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalUareConflict(f"cannot parse uniform-anchor runtime probe: {exc}") from exc
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    _require(report.get("report_sha256") == sha256_json(core), "runtime probe hash changed")
    _require(report["report_sha256"] == config["runtime_selection"]["probe_report_sha256"],
             "runtime probe is not the frozen evidence")
    _require(report["purpose"] == "runtime_only_no_raw_no_estimator_no_method_selection_by_physics",
             "runtime probe purpose changed")
    _require(report["kernel"] == UNIFORM_ANCHOR_PT_KERNEL, "runtime probe kernel changed")
    _require(report["cell"] == config["cell"], "runtime probe cell changed")
    methods = {item["method_id"]: item for item in report["timings"]}
    selected = [method["id"] for method in config["methods"]]
    _require(set(selected) <= set(methods), "runtime probe lacks a configured method")
    rounds = config["resource"]["burn_rounds"] + config["resource"]["measurement_rounds"]
    worst = max(float(methods[name]["seconds_per_round"]) for name in selected)
    projected = worst * rounds * float(config["runtime_selection"]["safety_factor"])
    _require(projected <= float(config["runtime_selection"]["max_predicted_trajectory_seconds"]),
             "frozen local resource exceeds runtime cap")
    return report, projected


def load_config(path, registry):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalUareConflict(f"cannot read UARE config: {exc}") from exc
    expected = {
        "cell", "config_version", "contract_version", "gates", "init_families",
        "l_start_rule", "methods", "raw_version", "registry_sha256", "resource",
        "runtime_selection", "scope", "selection", "trajectories_per_family",
        "trajectory_namespace",
    }
    _require(set(config) == expected, "UARE config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "UARE contract changed")
    _require(config["config_version"] == "exp102.q0_hgp_uniform_anchor_pt.v0.config.v1",
             "UARE config version changed")
    _require(config["raw_version"] == LOCAL_RAW_VERSION, "UARE raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "registry SHA changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "UARE hard cell changed")
    _require(config["init_families"] == list(INIT_FAMILIES), "UARE starts changed")
    _require(config["l_start_rule"] == FULL_ROW_L_START_RULE, "UARE L rule changed")
    _require(config["resource"] == {
        "burn_rounds": 256, "measurement_rounds": 2048, "name": "V0",
    }, "UARE resource changed")
    _require(config["trajectories_per_family"] == 8, "UARE trajectory count changed")
    _require(config["methods"] == [
        {"id": "UARE32-R1", "num_replicas": 32, "positive_row_updates_per_round": 1},
        {"id": "UARE64-R1", "num_replicas": 64, "positive_row_updates_per_round": 1},
    ], "UARE candidate methods changed")
    _require(config["selection"] == {
        "both_pass": "select_UARE32_R1",
        "one_pass": "select_the_only_passing_method",
        "selection_observable": "predeclared_convergence_gates_only_no_q_top",
        "zero_pass": "LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT",
    }, "UARE selection changed")
    _require(config["scope"] == {
        "formal_authorization": False, "posterior_estimation": False,
        "purpose": "local_adversarial_convergence_preflight_only",
        "remote_authorization": False,
    }, "UARE scope changed")
    gates = config["gates"]
    _require(gates == {
        "max_abs_B_mask_mean_delta": 0.12,
        "max_abs_complete_score_delta_per_factor": 0.12,
        "max_abs_logical_character_mean_delta": 0.12,
        "max_abs_normalized_B_weight_delta": 0.08,
        "max_abs_normalized_weight_delta": 0.08,
        "max_negligible_support_upper_bound": 0.001,
        "minimum_effective_trajectory_count": 4,
        "sigma_multiplier": 3.0,
        "sigma_slack": 0.01,
    }, "UARE gates changed")
    _require(config["runtime_selection"].get("basis") == "runtime_only", "runtime basis changed")
    _require(config["runtime_selection"].get("safety_factor") == 2.0,
             "runtime safety factor changed")
    _require(config["runtime_selection"].get("max_predicted_trajectory_seconds") == 1200.0,
             "runtime cap changed")
    _require(isinstance(config["trajectory_namespace"], str) and config["trajectory_namespace"],
             "trajectory namespace is invalid")
    runtime_probe, _ = _load_runtime_probe(config)
    source_paths = {
        "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
    }
    for name, expected_path in source_paths.items():
        _require(runtime_probe["source_binding"]["files"].get(name) == sha256_file(expected_path),
                 "runtime probe source binding is stale")
    return config, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _frozen_masks(config_sha256, registry_sha256, rows, logical_k):
    from exp101_certified_src.prng import PortablePrng

    _require(logical_k == 64, "hard m8 logical dimension changed")
    seed = derive_seed(
        "exp102.q0_hgp_uniform_anchor_pt.v0", "fixed_characters",
        config_sha256, registry_sha256, "uint64_bit63_safe_v1",
    )
    rng = PortablePrng(seed)
    basis = np.asarray([np.uint64(1) << np.uint64(bit) for bit in range(64)], dtype=np.uint64)
    seen = {int(value) for value in basis}
    nonbasis = []
    while len(nonbasis) < LOGICAL_MASK_COUNT - basis.size:
        value = int(rng.next_uint64())
        if value and value not in seen:
            seen.add(value)
            nonbasis.append(np.uint64(value))
    b_masks = np.zeros((B_MASK_COUNT, rows), dtype=np.uint32)
    row_mask = (1 << rows) - 1
    for index in range(B_MASK_COUNT):
        while not b_masks[index].any():
            for column in range(rows):
                b_masks[index, column] = np.uint32(int(rng.next_uint64()) & row_mask)
    return np.concatenate((basis, np.asarray(nonbasis, dtype=np.uint64))), b_masks


def _context(registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG,
             *, frozen_l_move=None, frozen_l_metadata=None):
    registry = load_registry(registry_path)
    config, config_sha256 = load_config(config_path, registry)
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = _attempt022_uniform_seed(registry, code, config["cell"])
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(config["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(syndrome.any(), "hard sentinel syndrome unexpectedly vanishes")
    samplers = {
        method["id"]: UniformAnchorReplicaExchangeConfig(
            p=config["cell"]["p"], burn_rounds=config["resource"]["burn_rounds"],
            measurement_rounds=config["resource"]["measurement_rounds"],
            num_replicas=method["num_replicas"],
            positive_row_updates_per_round=method["positive_row_updates_per_round"],
            method_id=method["id"],
        )
        for method in config["methods"]
    }
    if frozen_l_move is None:
        l_start, l_metadata = select_low_energy_logical_start(epsilon, model, frame)
    else:
        l_move = np.ascontiguousarray(frozen_l_move, dtype=np.uint8)
        _require(l_move.shape == epsilon.shape and np.all((l_move == 0) | (l_move == 1)),
                 "frozen L move is invalid")
        l_start = np.ascontiguousarray(epsilon ^ l_move, dtype=np.uint8)
        residual = (
            model.H_check.astype(np.int64) @ l_move.astype(np.int64) % 2
        ).astype(np.uint8)
        _require(not residual.any() and _state_label(frame, l_move) != 0,
                 "frozen L move lost logical kernel membership")
        _require(isinstance(frozen_l_metadata, dict), "frozen L metadata is invalid")
        _require(frozen_l_metadata.get("rule") == FULL_ROW_L_START_RULE
                 and frozen_l_metadata.get("selected_absolute_weight") == int(l_start.sum())
                 and frozen_l_metadata.get("selected_move_weight") == int(l_move.sum())
                 and frozen_l_metadata.get("selected_signature") == int(_state_label(frame, l_move))
                 and frozen_l_metadata.get("selected_move_sha256") == hashlib.sha256(l_move.tobytes()).hexdigest(),
                 "frozen L metadata changed")
        l_metadata = dict(frozen_l_metadata)
    mass = build_classical_coset_mass(H, config["cell"]["p"], engine="numba")
    logical_masks, b_masks = _frozen_masks(
        config_sha256, registry["registry_sha256"], int(H.shape[0]), model.k,
    )
    return {
        "registry": registry, "config": config, "config_sha256": config_sha256,
        "code": code, "H": np.ascontiguousarray(H, dtype=np.uint8), "model": model,
        "frame": frame, "uniform_seed": uniform_seed, "epsilon": epsilon,
        "syndrome": syndrome, "samplers": samplers, "l_start": l_start,
        "l_metadata": l_metadata, "plan": build_full_row_elimination_plan(H),
        "mass": mass, "logical_masks": logical_masks, "b_masks": b_masks,
        "source_binding": _source_binding(config_path),
    }


def _seed_identity(context, method_id, family, trajectory):
    sampler = context["samplers"][method_id]
    return UniformAnchorReplicaExchangeSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=sha256_json(sampler.as_dict()),
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(context["config"]["cell"]), method_id=method_id,
        resource_tier=context["config"]["resource"]["name"], init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=context["config"]["trajectory_namespace"],
    )


def _task(context, method_id, family, trajectory):
    identity = _seed_identity(context, method_id, family, trajectory)
    core = {
        "task_version": TASK_VERSION, "raw_version": LOCAL_RAW_VERSION,
        "cell": context["config"]["cell"], "method_id": method_id,
        "resource": context["config"]["resource"], "init_family": family,
        "trajectory_index": int(trajectory), "seed_identity": identity.as_dict(),
    }
    return {**core, "task_fingerprint": sha256_json(core)}


def _control_arrays(context):
    return {
        "epsilon": context["epsilon"], "syndrome": context["syndrome"],
        "l_move": context["l_start"] ^ context["epsilon"],
        "logical_masks": context["logical_masks"], "b_masks": context["b_masks"],
        "classical_mass": context["mass"],
    }


def _manifest_core(context, tasks, control_sha256):
    return {
        "manifest_version": MANIFEST_VERSION, "contract_version": CONTRACT_VERSION,
        "raw_version": LOCAL_RAW_VERSION, "config": context["config"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding": context["source_binding"], "cell": context["config"]["cell"],
        "uniform_seed": int(context["uniform_seed"]), "H_sha256": _array_sha256(context["H"]),
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "epsilon_sha256": _array_sha256(context["epsilon"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "l_start": context["l_metadata"], "plan_sha256": context["plan"].sha256,
        "plan_json": context["plan"].as_dict(),
        "mass_sha256": hashlib.sha256(np.asarray(context["mass"], dtype=">f8").tobytes()).hexdigest(),
        "character_sha256": sha256_json({
            "logical": _array_sha256(context["logical_masks"]),
            "b": _array_sha256(context["b_masks"]),
        }),
        "control_npz_sha256": control_sha256, "tasks": tasks,
    }


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not output_root.exists(), "UARE output root already exists")
    context = _context(registry_path, config_path)
    tasks = [
        _task(context, method["id"], family, trajectory)
        for method in context["config"]["methods"]
        for family in INIT_FAMILIES
        for trajectory in range(context["config"]["trajectories_per_family"])
    ]
    _require(len(tasks) == 48 and len({task["task_fingerprint"] for task in tasks}) == 48,
             "UARE task plan changed")
    control_path = output_root / "CONTROL.npz"
    atomic_npz(control_path, **_control_arrays(context))
    control_sha256 = sha256_file(control_path)
    core = _manifest_core(context, tasks, control_sha256)
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(output_root / "MANIFEST.json", manifest)
    return manifest


def _load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalUareConflict(f"cannot load UARE manifest: {exc}") from exc
    required = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "epsilon_sha256", "model_fingerprint", "frame_fingerprint",
        "l_start", "plan_sha256", "plan_json", "mass_sha256", "character_sha256",
        "control_npz_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == required, "UARE manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "UARE manifest hash changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "UARE manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "UARE manifest contract changed")
    _require(len(manifest["tasks"]) == 48, "UARE manifest task count changed")
    _require(len({task["task_fingerprint"] for task in manifest["tasks"]}) == 48,
             "UARE tasks are not unique")
    return manifest


def _load_control(path, context, manifest):
    try:
        with np.load(path, allow_pickle=False) as archive:
            control = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise LocalUareConflict(f"cannot load UARE control: {exc}") from exc
    _require(set(control) == {"epsilon", "syndrome", "l_move", "logical_masks", "b_masks", "classical_mass"},
             "UARE control schema changed")
    expected = _control_arrays(context)
    for name, value in expected.items():
        _require(np.array_equal(control[name], value), f"UARE control {name} changed")
    _require(sha256_file(path) == manifest["control_npz_sha256"], "UARE control hash changed")
    return control


def _validate_manifest_context(manifest, context, control_path):
    core = _manifest_core(context, manifest["tasks"], manifest["control_npz_sha256"])
    expected = {**core, "manifest_sha256": sha256_json(core)}
    _require(manifest == expected, "UARE manifest/context binding changed")
    _require(sha256_file(control_path) == manifest["control_npz_sha256"], "UARE control changed")


def _initial_state(task, model, syndrome, control):
    family = task["init_family"]
    if family == "P":
        state = control["epsilon"].copy()
    elif family == "U":
        identity = UniformAnchorReplicaExchangeSeedIdentity(**task["seed_identity"])
        state = uniform_hard_coset_state(model, syndrome, identity.seed("initialize", "hard_coset"))
    elif family == "L":
        state = control["epsilon"] ^ control["l_move"]
    else:  # pragma: no cover
        raise LocalUareConflict("unknown UARE initialization family")
    residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    _require(not residual.any(), "UARE initial state left the hard coset")
    return np.ascontiguousarray(state, dtype=np.uint8)


def _task_context(manifest_path, task):
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    control_path = manifest_path.parent / "CONTROL.npz"
    with np.load(control_path, allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = _context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    _validate_manifest_context(manifest, context, control_path)
    _require(task in manifest["tasks"], "task is not in frozen UARE manifest")
    expected = _task(context, task["method_id"], task["init_family"], task["trajectory_index"])
    _require(task == expected, "UARE task identity changed")
    return context, manifest, _load_control(control_path, context, manifest)


def _task_output_path(output_root, task):
    return Path(output_root) / "raw" / (
        f'{task["method_id"]}_{task["init_family"]}_{int(task["trajectory_index"]):02d}.npz'
    )


def _execute_task(manifest_path, task):
    context, manifest, control = _task_context(manifest_path, task)
    output_path = _task_output_path(Path(manifest_path).parent, task)
    _require(not output_path.exists(), f"UARE raw already exists: {output_path.name}")
    sampler = context["samplers"][task["method_id"]]
    identity = UniformAnchorReplicaExchangeSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context["model"], context["syndrome"], control)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    result = run_uniform_anchor_replica_exchange_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, identity, initial, engine="numba", mass=control["classical_mass"],
    )
    wall_seconds, core_seconds = time.perf_counter() - started_wall, time.process_time() - started_cpu
    _require(result["raw_version"] == UNIFORM_ANCHOR_PT_RAW_VERSION, "sampler raw version changed")
    _require(result["method_id"] == task["method_id"], "sampler method changed")
    _require(result["plan_sha256"] == context["plan"].sha256, "sampler plan changed")
    _require(not result["measurement_residual_weights"].any(), "sampler left hard coset")
    _require(wall_seconds <= context["config"]["runtime_selection"]["max_predicted_trajectory_seconds"],
             "UARE trajectory exceeded frozen wall cap")
    raw = {
        **result,
        "raw_version": LOCAL_RAW_VERSION,
        "sampler_raw_version": UNIFORM_ANCHOR_PT_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"], "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "uniform_seed": np.int64(context["uniform_seed"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "logical_masks": control["logical_masks"], "b_masks": control["b_masks"],
        "character_sha256": manifest["character_sha256"],
        "core_seconds": np.float64(core_seconds), "wall_seconds": np.float64(wall_seconds),
        "init_family": task["init_family"], "trajectory_index": np.int16(task["trajectory_index"]),
    }
    _require(set(raw) == RAW_FIELDS, "UARE local raw schema drifted")
    atomic_npz(output_path, **raw)
    return {
        "filename": output_path.name, "sha256": sha256_file(output_path),
        "wall_seconds": wall_seconds, "core_seconds": core_seconds,
        "task_fingerprint": task["task_fingerprint"],
    }


def run(manifest_path, workers):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    _require(not (output_root / "RUNNING.json").exists(), "UARE run already marked running")
    _require(not (output_root / "RUN_COMPLETE.json").exists(), "UARE run already completed")
    workers = int(workers)
    _require(1 <= workers <= 6, "UARE local workers must lie in [1, 6]")
    atomic_json(output_root / "RUNNING.json", {
        "runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "workers": workers, "pid": os.getpid(),
    })
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_execute_task, str(manifest_path), task)
                       for task in manifest["tasks"]]
            results = [future.result() for future in futures]
    except Exception as exc:
        atomic_json(output_root / "FAILED.json", {
            "runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"],
            "failure": repr(exc),
        })
        raise
    results.sort(key=lambda value: value["filename"])
    core = {
        "runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "raw_count": len(results), "raw": results,
    }
    atomic_json(output_root / "RUN_COMPLETE.json", {**core, "run_sha256": sha256_json(core)})
    return results


def _unpack_state(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)


def _expected_swap_attempts(replicas, offset, rounds):
    result = np.zeros(replicas - 1, dtype=np.int64)
    for round_index in range(int(offset), int(offset) + int(rounds)):
        for lower in range(round_index & 1, replicas - 1, 2):
            result[lower] += 1
    return result


def _validate_one_raw(path, context, manifest, control, task):
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise LocalUareConflict(f"cannot load UARE raw {path}: {exc}") from exc
    _require(set(raw) == RAW_FIELDS, f"UARE raw schema changed: {path.name}")
    _require(not any(value.dtype.hasobject for value in raw.values()), "UARE object dtype is forbidden")
    sampler = context["samplers"][task["method_id"]]
    identity = {
        "raw_version": LOCAL_RAW_VERSION, "sampler_raw_version": UNIFORM_ANCHOR_PT_RAW_VERSION,
        "contract_version": CONTRACT_VERSION, "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"], "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "character_sha256": manifest["character_sha256"], "init_family": task["init_family"],
        "trajectory_index": int(task["trajectory_index"]), "method_id": task["method_id"],
        "engine": "numba", "plan_sha256": context["plan"].sha256,
        "mass_sha256": manifest["mass_sha256"], "lambda_sha256": sampler.lambda_sha256,
    }
    for name, expected in identity.items():
        _require(str(_scalar(raw, name)) == str(expected), f"raw identity {name} changed: {path.name}")
    _require(int(_scalar(raw, "uniform_seed")) == int(context["uniform_seed"]), "raw uniform seed changed")
    _require(np.array_equal(raw["logical_masks"], control["logical_masks"]), "logical masks changed")
    _require(np.array_equal(raw["b_masks"], control["b_masks"]), "B masks changed")
    _require(str(_scalar(raw, "sampler_config_json")) == canonical_json(sampler.as_dict()),
             "sampler config changed")
    _require(str(_scalar(raw, "seed_identity_json")) == canonical_json(task["seed_identity"]),
             "seed identity changed")
    _require(str(_scalar(raw, "plan_json")) == canonical_json(context["plan"].as_dict()), "plan changed")
    _require(np.array_equal(raw["lambda_values"], sampler.lambda_values), "lambda values changed")
    expected_initial = _initial_state(task, context["model"], context["syndrome"], control)
    _require(np.array_equal(_unpack_state(raw["initial_state_packed"], context["model"].num_qubits), expected_initial),
             "initial state changed")
    for name in ("initial_state_packed", "burn_state_packed", "final_state_packed"):
        state = _unpack_state(raw[name], context["model"].num_qubits)
        residual = (
            context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
        ).astype(np.uint8) ^ context["syndrome"]
        _require(not residual.any(), f"{name} left hard coset")
    count = sampler.measurement_rounds
    measurement = np.unpackbits(
        raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    _require(measurement.shape == (count, context["model"].num_qubits), "measurement shape changed")
    residuals = (
        context["model"].H_check.astype(np.int64) @ measurement.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(not residuals.any() and not raw["measurement_residual_weights"].any(),
             "measurement left hard coset")
    labels = np.asarray([_state_label(context["frame"], value) for value in measurement], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "measurement labels changed")
    _require(np.array_equal(measurement.sum(axis=1).astype(np.int32), raw["measurement_weights"]),
             "measurement weights changed")
    _require(np.array_equal(raw["measurement_block"], np.repeat(np.arange(8, dtype=np.int8), count // 8)),
             "measurement blocks changed")
    _require(raw["measurement_b_columns"].shape == (count, context["H"].shape[0]), "B trace shape changed")
    _require(raw["measurement_a_syndromes"].shape == (count, context["H"].shape[1]), "A trace shape changed")
    log_mass = np.log(control["classical_mass"])
    log_odds = math.log(float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"])))
    expected_score = np.empty(count, dtype=np.float64)
    expected_b_weight = np.empty(count, dtype=np.int32)
    for index, state in enumerate(measurement):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, context["syndrome"], context["H"])
        _require(np.array_equal(b_columns, raw["measurement_b_columns"][index]), "B trace changed")
        _require(np.array_equal(a_syndromes, raw["measurement_a_syndromes"][index]), "A trace changed")
        expected_score[index] = collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds)
        expected_b_weight[index] = sum(int(value).bit_count() for value in b_columns)
    _require(np.array_equal(expected_score, raw["measurement_complete_scores"]), "measurement score changed")
    _require(np.array_equal(expected_b_weight, raw["measurement_b_weights"]), "measurement B weights changed")
    for phase, rounds in (("burn", sampler.burn_rounds), ("measurement", count)):
        counters = raw[f"{phase}_row_counters"]
        _require(counters.shape == (sampler.num_replicas, 3), f"{phase} row counter shape changed")
        _require(not counters[0].any(), f"{phase} hot rung performed a row update")
        expected_attempts = rounds * (sampler.num_replicas - 1) * sampler.positive_row_updates_per_round
        _require(int(counters[:, 0].sum()) == expected_attempts, f"{phase} row attempt count changed")
        _require(np.all(counters[:, 1] <= counters[:, 0]) and np.all(counters[:, 2] >= counters[:, 1]),
                 f"{phase} row counters are invalid")
        hot = raw[f"{phase}_hot_refresh_changed_bits"]
        _require(hot.shape == (rounds,) and np.all((hot >= 0) & (hot <= context["H"].shape[0] ** 2)),
                 f"{phase} hot refresh trace changed")
        _require(int(_scalar(raw, f"{phase}_cold_a_column_draws")) == rounds * context["H"].shape[1],
                 f"{phase} A redraw count changed")
        _require(np.array_equal(raw[f"{phase}_swap_attempts"], _expected_swap_attempts(
            sampler.num_replicas, 0 if phase == "burn" else sampler.burn_rounds, rounds)),
                 f"{phase} swap attempts changed")
        _require(np.all(raw[f"{phase}_swap_accepts"] <= raw[f"{phase}_swap_attempts"]),
                 f"{phase} swap accepts changed")
    _require(float(_scalar(raw, "wall_seconds")) <= context["config"]["runtime_selection"]["max_predicted_trajectory_seconds"],
             "raw wall time exceeded frozen cap")
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


def _trajectory_observables(raw, context, control):
    labels = np.asarray(raw["measurement_labels"], dtype=np.uint64)
    logical = 1.0 - 2.0 * _parity_uint64(labels[:, None] & control["logical_masks"][None, :])
    b_columns = np.asarray(raw["measurement_b_columns"], dtype=np.uint32)
    b_signs = np.empty((b_columns.shape[0], control["b_masks"].shape[0]), dtype=np.float64)
    for mask_index, mask in enumerate(control["b_masks"]):
        parity = np.zeros(b_columns.shape[0], dtype=np.uint32)
        for column in range(b_columns.shape[1]):
            parity ^= _parity_uint32(b_columns[:, column] & mask[column])
        b_signs[:, mask_index] = 1.0 - 2.0 * parity
    num_qubits = context["model"].num_qubits
    b_size = context["H"].shape[0] ** 2
    factor_count = context["H"].shape[1]
    half = labels.size // 2
    values = {
        "normalized_weight": np.asarray(raw["measurement_weights"], dtype=np.float64) / num_qubits,
        "normalized_B_weight": np.asarray(raw["measurement_b_weights"], dtype=np.float64) / b_size,
        "complete_score_per_factor": np.asarray(raw["measurement_complete_scores"], dtype=np.float64) / factor_count,
        "logical_characters": logical,
        "B_masks": b_signs,
    }
    return {
        name: np.asarray((value.mean(axis=0), value[:half].mean(axis=0), value[half:].mean(axis=0)))
        for name, value in values.items()
    }


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def _compare(values_a, values_b, absolute_bound, gates):
    mean_a, se_a = _mean_se(values_a)
    mean_b, se_b = _mean_se(values_b)
    delta = np.abs(mean_a - mean_b)
    uncertainty = float(gates["sigma_multiplier"]) * np.sqrt(se_a * se_a + se_b * se_b) + float(gates["sigma_slack"])
    passed = bool(np.all(delta <= float(absolute_bound)) and np.all(delta <= uncertainty))
    return {
        "pass": passed,
        "max_abs_delta": float(np.max(delta)),
        "max_three_sigma_allowance": float(np.max(uncertainty)),
        "failed_components": int(np.count_nonzero((delta > float(absolute_bound)) | (delta > uncertainty))),
    }


def _family_arrays(records, context, control):
    per_trajectory = [_trajectory_observables(record, context, control) for record in records]
    return {
        name: np.stack([item[name][0] for item in per_trajectory], axis=0)
        for name in per_trajectory[0]
    }, {
        name: (np.stack([item[name][1] for item in per_trajectory], axis=0),
               np.stack([item[name][2] for item in per_trajectory], axis=0))
        for name in per_trajectory[0]
    }


def _comparison_bundle(left, right, gates):
    return {
        "normalized_weight": _compare(left["normalized_weight"], right["normalized_weight"],
                                        gates["max_abs_normalized_weight_delta"], gates),
        "normalized_B_weight": _compare(left["normalized_B_weight"], right["normalized_B_weight"],
                                          gates["max_abs_normalized_B_weight_delta"], gates),
        "complete_score_per_factor": _compare(left["complete_score_per_factor"], right["complete_score_per_factor"],
                                               gates["max_abs_complete_score_delta_per_factor"], gates),
        "logical_characters": _compare(left["logical_characters"], right["logical_characters"],
                                        gates["max_abs_logical_character_mean_delta"], gates),
        "B_masks": _compare(left["B_masks"], right["B_masks"], gates["max_abs_B_mask_mean_delta"], gates),
    }


def _bundle_pass(bundle):
    return bool(all(item["pass"] for item in bundle.values()))


def _hard_coset_dimension(model):
    from exp101_certified_src.gf2 import gf2_rank
    return int(model.num_qubits - gf2_rank(model.H_check))


def _support_gate(u_records, context, control):
    p = float(context["config"]["cell"]["p"])
    log_odds = math.log(p / (1.0 - p))
    w0 = int(control["epsilon"].sum())
    dimension = _hard_coset_dimension(context["model"])
    entries = []
    for record in u_records:
        minimum = int(np.asarray(record["measurement_weights"]).min())
        log_bound = dimension * math.log(2.0) + (minimum - w0) * log_odds
        bound = 0.0 if log_bound < -745.0 else min(1.0, math.exp(log_bound))
        entries.append({
            "minimum_measurement_weight": minimum,
            "target_support_upper_bound": bound,
            "trapped_negligible_support": bool(bound <= context["config"]["gates"]["max_negligible_support_upper_bound"]),
        })
    return {
        "reference_legal_weight": w0, "hard_coset_dimension": dimension,
        "per_trajectory": entries,
        "pass": not any(item["trapped_negligible_support"] for item in entries),
    }


def _method_summary(records, context, control):
    gates = context["config"]["gates"]
    by_family = {family: [] for family in INIT_FAMILIES}
    for record in records:
        by_family[str(_scalar(record, "init_family"))].append(record)
    arrays, split = {}, {}
    for family in INIT_FAMILIES:
        _require(len(by_family[family]) == context["config"]["trajectories_per_family"],
                 f"missing UARE family trajectory: {family}")
        arrays[family], split[family] = _family_arrays(by_family[family], context, control)
    pairwise = {
        f"{left}_{right}": _comparison_bundle(arrays[left], arrays[right], gates)
        for left, right in (("P", "U"), ("P", "L"), ("U", "L"))
    }
    time_stability = {
        family: _comparison_bundle(split[family][0], split[family][1], gates)
        for family in INIT_FAMILIES
    }
    support = _support_gate(by_family["U"], context, control)
    independent_counts = {family: len(by_family[family]) for family in INIT_FAMILIES}
    complete = all(value >= gates["minimum_effective_trajectory_count"] for value in independent_counts.values())
    passed = complete and support["pass"] and all(_bundle_pass(value) for value in pairwise.values()) and all(
        _bundle_pass(value) for value in time_stability.values()
    )
    b_variation = {
        family: int(np.count_nonzero(np.ptp(arrays[family]["B_masks"], axis=0) > 0.0))
        for family in INIT_FAMILIES
    }
    return {
        "pass": bool(passed), "independent_trajectory_count": independent_counts,
        "target_support_gate": support, "pairwise": pairwise,
        "time_stability": time_stability, "nonconstant_B_mask_diagnostic": b_variation,
        "q_top_read_or_computed": False,
    }


def _replay_one(context, manifest, control, task, raw):
    sampler = context["samplers"][task["method_id"]]
    identity = UniformAnchorReplicaExchangeSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context["model"], context["syndrome"], control)
    replay = run_uniform_anchor_replica_exchange_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, identity, initial, engine="numba", mass=control["classical_mass"],
    )
    for name in SAMPLER_FIELDS - {"raw_version"}:
        _require(np.array_equal(np.asarray(raw[name]), np.asarray(replay[name])),
                 f"replay mismatch for {task['method_id']}_{task['init_family']}_{task['trajectory_index']:02d}: {name}")


def analyze(manifest_path, *, replay):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    with np.load(output_root / "CONTROL.npz", allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = _context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    _validate_manifest_context(manifest, context, output_root / "CONTROL.npz")
    control = _load_control(output_root / "CONTROL.npz", context, manifest)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "UARE run is incomplete")
    by_method = {method["id"]: [] for method in context["config"]["methods"]}
    raw_hashes = {}
    for task in manifest["tasks"]:
        path = _task_output_path(output_root, task)
        _require(path.is_file(), f"missing UARE raw: {path.name}")
        raw = _validate_one_raw(path, context, manifest, control, task)
        if replay:
            _replay_one(context, manifest, control, task, raw)
        by_method[task["method_id"]].append(raw)
        raw_hashes[path.name] = sha256_file(path)
    methods = {name: _method_summary(records, context, control) for name, records in by_method.items()}
    passing = [name for name, item in methods.items() if item["pass"]]
    if not passing:
        selected, status = None, context["config"]["selection"]["zero_pass"]
    elif len(passing) == 1:
        selected, status = passing[0], "LOCAL_UNIFORM_ANCHOR_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    else:
        selected, status = "UARE32-R1", "LOCAL_UNIFORM_ANCHOR_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    core = {
        "report_version": REPORT_VERSION, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"], "status": status,
        "selected_method": selected, "formal_authorization": False,
        "remote_authorization": False, "posterior_estimation": False,
        "q_top_read_or_computed": False, "replay_performed": bool(replay),
        "methods": methods, "raw_sha256": raw_hashes,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(output_root / "REPORT.json", report)
    if replay:
        atomic_json(output_root / "REPLAY.json", {
            "manifest_sha256": manifest["manifest_sha256"], "raw_sha256": raw_hashes,
            "report_sha256": report["report_sha256"],
        })
    atomic_json(output_root / "SUCCESS.json", {
        "manifest_sha256": manifest["manifest_sha256"], "report_sha256": report["report_sha256"],
        "status": status,
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=ROOT / "local_hard_viability")
    prepare_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    run_parser.add_argument("--workers", type=int, default=6)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    analyze_parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, args.registry, args.config)
        print(result["manifest_sha256"])
    elif args.command == "run":
        result = run(args.manifest, args.workers)
        print(len(result))
    else:
        result = analyze(args.manifest, replay=not args.no_replay)
        print(result["status"])


if __name__ == "__main__":
    main()
