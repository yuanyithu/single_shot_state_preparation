"""Frozen local P/U/L screen for UASRE auxiliary stabilizer transport.

The runner is deliberately local-only.  A passing candidate has no formal or
remote authority; a failed raw set is never extended, pooled, or reused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

# Permit the frozen runner to be invoked by file path as well as ``python -m``.
# Remote launchers normally set PYTHONPATH, but a direct local invocation must
# not silently depend on that external shell detail.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    reduce_logical_basis,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer_pt import (
    AUX_STABILIZER_PT_KERNEL,
    AUX_STABILIZER_PT_RAW_VERSION,
    AuxiliaryStabilizerReplicaExchangeConfig,
    AuxiliaryStabilizerReplicaExchangeSeedIdentity,
    run_auxiliary_stabilizer_replica_exchange_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _pack_state,
    _qubit_signatures,
    _state_label,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_L_START_RULE,
    build_full_row_elimination_plan,
    select_low_energy_logical_start,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_uniform_anchor_pt import (
    collapsed_complete_score,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.v0"
MANIFEST_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.tasks.v1"
LOCAL_RAW_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.raw.v0"
REPORT_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.report.v1"
RUNNER_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.runner.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = EXP102_ROOT / "config" / "q0_hgp_aux_stabilizer_pt.v0.json"
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
    "measurement_hot_refresh_changed_bits", "burn_auxiliary_counters",
    "measurement_auxiliary_counters", "measurement_auxiliary_assignments",
    "burn_cold_a_column_draws", "measurement_cold_a_column_draws",
    "burn_swap_attempts", "burn_swap_accepts", "measurement_swap_attempts",
    "measurement_swap_accepts", "lambda_values", "lambda_sha256", "mass_sha256",
    "initial_label", "burn_label", "final_label", "engine",
})
RAW_FIELDS = SAMPLER_FIELDS | {
    "sampler_raw_version", "contract_version", "local_raw_version",
    "task_fingerprint", "task_json", "manifest_sha256", "config_sha256",
    "registry_sha256", "source_binding_sha256", "cell_json", "uniform_seed",
    "model_fingerprint", "frame_fingerprint", "logical_masks", "b_masks",
    "character_sha256", "core_seconds", "wall_seconds", "init_family",
    "trajectory_index",
}


class LocalAuxiliaryConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalAuxiliaryConflict(message)


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


def _float_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=">f8").tobytes()).hexdigest()


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(len(source_commit) == 40 and all(value in "0123456789abcdef" for value in source_commit),
             "source commit is invalid")
    files = {
        name: sha256_file(path)
        for name, path in {
            "config.py": EXP102_ROOT / "exp102_pipeline" / "config.py",
            "diagnostics.py": EXP102_ROOT / "exp102_pipeline" / "diagnostics.py",
            "exp101_bridge.py": EXP102_ROOT / "exp102_pipeline" / "exp101_bridge.py",
            "io.py": EXP102_ROOT / "exp102_pipeline" / "io.py",
            "labels.py": EXP102_ROOT / "exp102_pipeline" / "labels.py",
            "q0_global.py": EXP102_ROOT / "exp102_pipeline" / "q0_global.py",
            "q0_hgp_aux_stabilizer.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py",
            "q0_hgp_aux_stabilizer_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer_pt.py",
            "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
            "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
            "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
            "q0_pa.py": EXP102_ROOT / "exp102_pipeline" / "q0_pa.py",
            "q0_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_pt.py",
            "registry.py": EXP102_ROOT / "exp102_pipeline" / "registry.py",
            "seeds.py": EXP102_ROOT / "exp102_pipeline" / "seeds.py",
            "worker.py": EXP102_ROOT / "exp102_pipeline" / "worker.py",
            "independent_raw_audit.py": ROOT / "independent_raw_audit.py",
            "run_local_viability.py": Path(__file__),
            "config": Path(config_path),
            "registry.json": DEFAULT_REGISTRY,
        }.items()
    }
    exp101_src = EXP102_ROOT.parent / "exp101" / "src"
    exp101_files = {
        path.relative_to(exp101_src).as_posix(): sha256_file(path)
        for path in sorted(exp101_src.rglob("*.py"))
    }
    _require(bool(exp101_files), "exp101 source tree is unexpectedly empty")
    try:
        import numba
        numba_version = str(numba.__version__)
    except ImportError:  # pragma: no cover - the accelerated runner requires Numba
        numba_version = "missing"
    runtime = {
        "numpy": str(np.__version__), "numba": numba_version,
        "python": sys.version.split()[0],
    }
    core = {
        "source_commit": source_commit, "files": files,
        "exp101_src_files": exp101_files, "runtime": runtime,
    }
    return {**core, "source_binding_sha256": sha256_json(core)}


def _load_runtime_probe(config):
    path = ROOT / "uasre_runtime_probe.json"
    _require(path.is_file(), "UASRE runtime probe is missing")
    report = json.loads(path.read_text(encoding="ascii"))
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    _require(report.get("report_sha256") == sha256_json(core), "UASRE runtime probe hash changed")
    _require(report["report_sha256"] == config["runtime_selection"]["probe_report_sha256"],
             "UASRE runtime probe is not frozen evidence")
    _require(report["purpose"] == "runtime_only_no_raw_no_estimator_no_method_selection_by_physics",
             "UASRE runtime probe purpose changed")
    _require(report["kernel"] == AUX_STABILIZER_PT_KERNEL, "UASRE runtime probe kernel changed")
    _require(report["cell"] == config["cell"], "UASRE runtime probe cell changed")
    paths = {
        "auxiliary": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py",
        "sampler": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer_pt.py",
        "uniform_anchor": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
    }
    for name, source in paths.items():
        _require(report["source"].get(name) == sha256_file(source), f"UASRE probe source changed: {name}")
    timings = {item["method_id"]: item for item in report["timings"]}
    selected = [item["id"] for item in config["methods"]]
    _require(set(selected) <= set(timings), "UASRE runtime probe lacks a configured method")
    rounds = config["resource"]["burn_rounds"] + config["resource"]["measurement_rounds"]
    projected = max(float(timings[name]["seconds_per_round"]) for name in selected)
    projected *= rounds * float(config["runtime_selection"]["safety_factor"])
    _require(projected <= float(config["runtime_selection"]["max_predicted_trajectory_seconds"]),
             "UASRE local resource exceeds its runtime cap")
    return report, projected


def load_config(path, registry):
    config = json.loads(Path(path).read_text(encoding="ascii"))
    expected = {
        "cell", "config_version", "contract_version", "gates", "init_families",
        "l_start_rule", "methods", "raw_version", "registry_sha256", "resource",
        "runtime_selection", "scope", "selection", "trajectories_per_family",
        "trajectory_namespace",
    }
    _require(set(config) == expected, "UASRE config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "UASRE contract changed")
    _require(config["config_version"] == "exp102.q0_hgp_aux_stabilizer_pt.v0.config.v1",
             "UASRE config version changed")
    _require(config["raw_version"] == LOCAL_RAW_VERSION, "UASRE raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "UASRE registry SHA changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "UASRE hard cell changed")
    _require(config["init_families"] == list(INIT_FAMILIES), "UASRE starts changed")
    _require(config["l_start_rule"] == FULL_ROW_L_START_RULE, "UASRE L rule changed")
    _require(config["resource"] == {"burn_rounds": 256, "measurement_rounds": 2048, "name": "V0"},
             "UASRE resource changed")
    _require(config["trajectories_per_family"] == 8, "UASRE trajectory count changed")
    _require(config["methods"] == [
        {"id": "UASRE32-R1-A1", "num_replicas": 32,
         "positive_row_updates_per_round": 1, "cold_auxiliary_rows_per_round": 1},
        {"id": "UASRE64-R1-A1", "num_replicas": 64,
         "positive_row_updates_per_round": 1, "cold_auxiliary_rows_per_round": 1},
    ], "UASRE candidate methods changed")
    _require(config["selection"] == {
        "both_pass": "select_UASRE32_R1_A1",
        "one_pass": "select_the_only_passing_method",
        "selection_observable": "predeclared_convergence_gates_only_no_q_top",
        "zero_pass": "LOCAL_AUXILIARY_STABILIZER_TRANSPORT_UNRESOLVED",
    }, "UASRE selection changed")
    _require(config["scope"] == {
        "formal_authorization": False, "posterior_estimation": False,
        "purpose": "local_adversarial_convergence_preflight_only",
        "remote_authorization": False,
    }, "UASRE scope changed")
    _require(config["gates"] == {
        "max_abs_B_mask_mean_delta": 0.12,
        "max_abs_complete_score_delta_per_factor": 0.12,
        "max_abs_logical_character_mean_delta": 0.12,
        "max_abs_normalized_B_weight_delta": 0.08,
        "max_abs_normalized_weight_delta": 0.08,
        "max_negligible_support_upper_bound": 0.001,
        "minimum_effective_trajectory_count": 4,
        "sigma_multiplier": 3.0,
        "sigma_slack": 0.01,
    }, "UASRE gates changed")
    _require(config["runtime_selection"].get("basis") == "runtime_only", "UASRE runtime basis changed")
    _require(config["runtime_selection"].get("safety_factor") == 2.0, "UASRE safety factor changed")
    _require(config["runtime_selection"].get("max_predicted_trajectory_seconds") == 1200.0,
             "UASRE runtime cap changed")
    _require(isinstance(config["trajectory_namespace"], str) and config["trajectory_namespace"],
             "UASRE trajectory namespace is invalid")
    _load_runtime_probe(config)
    return config, sha256_json(config)


def _uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "UASRE disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _frozen_masks(config_sha256, registry_sha256, rows, logical_k):
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    _require(logical_k == 64, "UASRE hard m8 logical dimension changed")
    seed = derive_seed(
        CONTRACT_VERSION, "fixed_characters", config_sha256, registry_sha256, "uint64_bit63_safe_v1",
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


def _context(registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *,
             frozen_l_move=None, frozen_l_metadata=None):
    registry = load_registry(registry_path)
    config, config_sha256 = load_config(config_path, registry)
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = _uniform_seed(registry, code, config["cell"])
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(config["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(syndrome.any(), "UASRE hard sentinel syndrome unexpectedly vanishes")
    samplers = {
        method["id"]: AuxiliaryStabilizerReplicaExchangeConfig(
            p=config["cell"]["p"], burn_rounds=config["resource"]["burn_rounds"],
            measurement_rounds=config["resource"]["measurement_rounds"],
            num_replicas=method["num_replicas"],
            positive_row_updates_per_round=method["positive_row_updates_per_round"],
            cold_auxiliary_rows_per_round=method["cold_auxiliary_rows_per_round"],
            method_id=method["id"],
        )
        for method in config["methods"]
    }
    if frozen_l_move is None:
        l_start, l_metadata = select_low_energy_logical_start(epsilon, model, frame)
    else:
        l_move = np.ascontiguousarray(frozen_l_move, dtype=np.uint8)
        _require(l_move.shape == epsilon.shape and np.all((l_move == 0) | (l_move == 1)),
                 "UASRE frozen L move is invalid")
        l_start = np.ascontiguousarray(epsilon ^ l_move, dtype=np.uint8)
        residual = (model.H_check.astype(np.int64) @ l_move.astype(np.int64) % 2).astype(np.uint8)
        _require(not residual.any() and _state_label(frame, l_move) != 0,
                 "UASRE frozen L move lost kernel membership")
        _require(isinstance(frozen_l_metadata, dict), "UASRE frozen L metadata is invalid")
        _require(frozen_l_metadata.get("rule") == FULL_ROW_L_START_RULE
                 and frozen_l_metadata.get("selected_absolute_weight") == int(l_start.sum())
                 and frozen_l_metadata.get("selected_move_weight") == int(l_move.sum())
                 and frozen_l_metadata.get("selected_signature") == int(_state_label(frame, l_move))
                 and frozen_l_metadata.get("selected_move_sha256") == hashlib.sha256(l_move.tobytes()).hexdigest(),
                 "UASRE frozen L metadata changed")
        l_metadata = dict(frozen_l_metadata)
    mass = build_classical_coset_mass(H, config["cell"]["p"], engine="numba")
    logical_masks, b_masks = _frozen_masks(config_sha256, registry["registry_sha256"], int(H.shape[0]), model.k)
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
    return AuxiliaryStabilizerReplicaExchangeSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=sha256_json(sampler.as_dict()),
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(context["config"]["cell"]), method_id=method_id,
        resource_tier=context["config"]["resource"]["name"], init_family=family,
        trajectory_index=trajectory, trajectory_namespace=context["config"]["trajectory_namespace"],
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
        "mass_sha256": _float_sha256(context["mass"]),
        "character_sha256": sha256_json({
            "logical": _array_sha256(context["logical_masks"]),
            "b": _array_sha256(context["b_masks"]),
        }),
        "control_npz_sha256": control_sha256, "tasks": tasks,
    }


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not output_root.exists(), "UASRE output root already exists")
    context = _context(registry_path, config_path)
    tasks = [
        _task(context, method["id"], family, trajectory)
        for method in context["config"]["methods"]
        for family in INIT_FAMILIES
        for trajectory in range(context["config"]["trajectories_per_family"])
    ]
    _require(len(tasks) == 48 and len({task["task_fingerprint"] for task in tasks}) == 48,
             "UASRE task plan changed")
    control_path = output_root / "CONTROL.npz"
    atomic_npz(control_path, **_control_arrays(context))
    core = _manifest_core(context, tasks, sha256_file(control_path))
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(output_root / "MANIFEST.json", manifest)
    return manifest


def _load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="ascii"))
    required = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "epsilon_sha256", "model_fingerprint", "frame_fingerprint",
        "l_start", "plan_sha256", "plan_json", "mass_sha256", "character_sha256",
        "control_npz_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == required, "UASRE manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "UASRE manifest hash changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "UASRE manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION and manifest["raw_version"] == LOCAL_RAW_VERSION,
             "UASRE manifest contract changed")
    _require(len(manifest["tasks"]) == 48 and len({task["task_fingerprint"] for task in manifest["tasks"]}) == 48,
             "UASRE manifest task count changed")
    for task in manifest["tasks"]:
        task_core = {key: value for key, value in task.items() if key != "task_fingerprint"}
        _require(task["task_fingerprint"] == sha256_json(task_core), "UASRE task fingerprint changed")
    return manifest


def _validate_manifest_context(manifest, context, control_path):
    _require(manifest["config"] == context["config"] and manifest["config_sha256"] == context["config_sha256"],
             "UASRE config identity changed")
    _require(manifest["registry_sha256"] == context["registry"]["registry_sha256"], "UASRE registry changed")
    _require(manifest["cell"] == context["config"]["cell"], "UASRE cell changed")
    _require(manifest["source_binding"] == context["source_binding"], "UASRE source binding changed")
    _require(manifest["uniform_seed"] == int(context["uniform_seed"]), "UASRE uniform seed changed")
    _require(manifest["H_sha256"] == _array_sha256(context["H"]), "UASRE H changed")
    _require(manifest["syndrome_sha256"] == _array_sha256(context["syndrome"]), "UASRE syndrome changed")
    _require(manifest["epsilon_sha256"] == _array_sha256(context["epsilon"]), "UASRE epsilon changed")
    _require(manifest["model_fingerprint"] == context["model"].fingerprint(), "UASRE model changed")
    _require(manifest["frame_fingerprint"] == context["frame"].fingerprint(), "UASRE frame changed")
    _require(manifest["l_start"] == context["l_metadata"], "UASRE L start changed")
    _require(manifest["plan_json"] == context["plan"].as_dict() and manifest["plan_sha256"] == context["plan"].sha256,
             "UASRE elimination plan changed")
    _require(manifest["mass_sha256"] == _float_sha256(context["mass"]), "UASRE mass changed")
    _require(manifest["character_sha256"] == sha256_json({
        "logical": _array_sha256(context["logical_masks"]),
        "b": _array_sha256(context["b_masks"]),
    }), "UASRE character masks changed")
    _require(manifest["control_npz_sha256"] == sha256_file(control_path), "UASRE control changed")


def _load_control(path, context, manifest):
    with np.load(path, allow_pickle=False) as archive:
        control = {name: archive[name].copy() for name in archive.files}
    _require(set(control) == {"epsilon", "syndrome", "l_move", "logical_masks", "b_masks", "classical_mass"},
             "UASRE control schema changed")
    expected = _control_arrays(context)
    for name, value in expected.items():
        _require(np.array_equal(control[name], value), f"UASRE control {name} changed")
    _require(_float_sha256(control["classical_mass"]) == manifest["mass_sha256"], "UASRE control mass changed")
    return control


def _uniform_start_seed(identity):
    return derive_seed(
        CONTRACT_VERSION, identity["trajectory_namespace"], identity["source_commit"],
        identity["config_sha256"], identity["registry_sha256"], identity["cell_fingerprint"],
        identity["method_id"], identity["resource_tier"], identity["init_family"],
        int(identity["trajectory_index"]), "initialize", "hard_coset", 0,
    )


def _initial_state(task, context, control):
    family = task["init_family"]
    if family == "P":
        return np.ascontiguousarray(control["epsilon"], dtype=np.uint8).copy()
    if family == "L":
        return np.ascontiguousarray(control["epsilon"] ^ control["l_move"], dtype=np.uint8)
    if family == "U":
        return uniform_hard_coset_state(
            context["model"], context["syndrome"], _uniform_start_seed(task["seed_identity"]),
        )
    raise LocalAuxiliaryConflict("UASRE task has an unknown initialization family")


def _task_output_path(output_root, task):
    return Path(output_root) / "raw" / (
        f"{task['method_id']}_{task['init_family']}_{int(task['trajectory_index']):02d}.npz"
    )


def _validate_run_complete(output_root, manifest):
    path = Path(output_root) / "RUN_COMPLETE.json"
    complete = json.loads(path.read_text(encoding="ascii"))
    required = {"runner_version", "manifest_sha256", "raw_count", "raw", "run_sha256"}
    _require(set(complete) == required, "UASRE completion marker schema changed")
    core = {name: value for name, value in complete.items() if name != "run_sha256"}
    _require(complete["run_sha256"] == sha256_json(core), "UASRE completion marker hash changed")
    _require(complete["runner_version"] == RUNNER_VERSION, "UASRE completion runner changed")
    _require(complete["manifest_sha256"] == manifest["manifest_sha256"], "UASRE completion manifest changed")
    _require(complete["raw_count"] == len(manifest["tasks"]), "UASRE completion raw count changed")
    expected = {
        _task_output_path(output_root, task).name: task["task_fingerprint"]
        for task in manifest["tasks"]
    }
    listed = complete["raw"]
    _require(isinstance(listed, list) and len(listed) == len(expected), "UASRE completion raw list changed")
    result = {}
    for item in listed:
        _require(isinstance(item, dict) and set(item) == {"filename", "sha256", "task_fingerprint"},
                 "UASRE completion raw item changed")
        name = item["filename"]
        _require(name in expected and name not in result, "UASRE completion raw item is unexpected")
        _require(item["task_fingerprint"] == expected[name], "UASRE completion task changed")
        path = Path(output_root) / "raw" / name
        _require(path.is_file() and item["sha256"] == sha256_file(path),
                 "UASRE completion raw digest changed")
        result[name] = item["sha256"]
    _require(set(result) == set(expected), "UASRE completion raw set changed")
    return result


def _execute_task(manifest_path, task):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    with np.load(output_root / "CONTROL.npz", allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = _context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    _validate_manifest_context(manifest, context, output_root / "CONTROL.npz")
    control = _load_control(output_root / "CONTROL.npz", context, manifest)
    sampler = context["samplers"][task["method_id"]]
    identity = AuxiliaryStabilizerReplicaExchangeSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context, control)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    raw = run_auxiliary_stabilizer_replica_exchange_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"], sampler,
        identity, initial, engine="numba", mass=context["mass"],
    )
    wall_seconds = time.perf_counter() - started_wall
    core_seconds = time.process_time() - started_cpu
    _require(wall_seconds <= context["config"]["runtime_selection"]["max_predicted_trajectory_seconds"],
             "UASRE raw task exceeded the frozen runtime cap")
    output = {
        **raw,
        "sampler_raw_version": AUX_STABILIZER_PT_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "uniform_seed": np.asarray(context["uniform_seed"], dtype=np.uint64),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "logical_masks": control["logical_masks"],
        "b_masks": control["b_masks"],
        "character_sha256": manifest["character_sha256"],
        "core_seconds": np.asarray(core_seconds, dtype=np.float64),
        "wall_seconds": np.asarray(wall_seconds, dtype=np.float64),
        "init_family": task["init_family"],
        "trajectory_index": np.asarray(task["trajectory_index"], dtype=np.int32),
    }
    _require(set(output) == RAW_FIELDS, "UASRE wrapped raw schema drifted")
    path = _task_output_path(output_root, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), "UASRE task raw already exists")
    atomic_npz(path, **output)
    return {"filename": path.name, "sha256": sha256_file(path), "task_fingerprint": task["task_fingerprint"]}


def run(manifest_path, workers):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    _require(int(workers) > 0, "UASRE workers must be positive")
    _require(not (output_root / "RUN_COMPLETE.json").exists() and not (output_root / "FAILED.json").exists(),
             "UASRE terminal marker already exists")
    atomic_json(output_root / "RUNNING.json", {
        "runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"], "workers": int(workers),
    })
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_execute_task, str(manifest_path), task) for task in manifest["tasks"]]
            results = [future.result() for future in futures]
    except Exception as exc:
        atomic_json(output_root / "FAILED.json", {
            "runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"], "failure": repr(exc),
        })
        raise
    results.sort(key=lambda value: value["filename"])
    core = {"runner_version": RUNNER_VERSION, "manifest_sha256": manifest["manifest_sha256"],
            "raw_count": len(results), "raw": results}
    atomic_json(output_root / "RUN_COMPLETE.json", {**core, "run_sha256": sha256_json(core)})
    return results


def _unpack_state(packed, num_qubits):
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little").astype(np.uint8)


def _expected_swap_attempts(replicas, offset, rounds):
    result = np.zeros(replicas - 1, dtype=np.int64)
    for round_index in range(int(offset), int(offset) + int(rounds)):
        for lower in range(round_index & 1, replicas - 1, 2):
            result[lower] += 1
    return result


def _validate_one_raw(path, context, manifest, control, task):
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    _require(set(raw) == RAW_FIELDS, f"UASRE raw schema changed: {path.name}")
    _require(not any(value.dtype.hasobject for value in raw.values()), "UASRE object dtype is forbidden")
    sampler = context["samplers"][task["method_id"]]
    expected = {
        "raw_version": AUX_STABILIZER_PT_RAW_VERSION, "sampler_raw_version": AUX_STABILIZER_PT_RAW_VERSION,
        "contract_version": CONTRACT_VERSION, "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"], "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "model_fingerprint": context["model"].fingerprint(), "frame_fingerprint": context["frame"].fingerprint(),
        "character_sha256": manifest["character_sha256"], "init_family": task["init_family"],
        "trajectory_index": int(task["trajectory_index"]), "method_id": task["method_id"], "engine": "numba",
        "plan_sha256": context["plan"].sha256, "mass_sha256": manifest["mass_sha256"],
        "lambda_sha256": sampler.lambda_sha256,
    }
    for name, value in expected.items():
        _require(str(_scalar(raw, name)) == str(value), f"UASRE raw identity {name} changed: {path.name}")
    _require(int(_scalar(raw, "uniform_seed")) == int(context["uniform_seed"]), "UASRE raw uniform seed changed")
    for name in ("core_seconds", "wall_seconds"):
        value = float(_scalar(raw, name))
        _require(math.isfinite(value) and value >= 0.0, f"UASRE raw {name} is invalid")
    _require(float(_scalar(raw, "wall_seconds")) <= float(context["config"]["runtime_selection"]["max_predicted_trajectory_seconds"]),
             "UASRE raw wall time exceeds the frozen cap")
    _require(np.array_equal(raw["logical_masks"], control["logical_masks"]), "UASRE logical masks changed")
    _require(np.array_equal(raw["b_masks"], control["b_masks"]), "UASRE B masks changed")
    _require(str(_scalar(raw, "sampler_config_json")) == canonical_json(sampler.as_dict()), "UASRE sampler config changed")
    _require(str(_scalar(raw, "sampler_config_sha256")) == sha256_json(sampler.as_dict()), "UASRE sampler hash changed")
    _require(str(_scalar(raw, "seed_identity_json")) == canonical_json(task["seed_identity"]), "UASRE seed identity changed")
    _require(str(_scalar(raw, "plan_json")) == canonical_json(context["plan"].as_dict()), "UASRE plan changed")
    _require(np.array_equal(raw["lambda_values"], sampler.lambda_values), "UASRE ladder changed")
    expected_initial = _initial_state(task, context, control)
    _require(np.array_equal(_unpack_state(raw["initial_state_packed"], context["model"].num_qubits), expected_initial),
             "UASRE initial state changed")
    for name, label_name in (("initial_state_packed", "initial_label"),
                             ("burn_state_packed", "burn_label"),
                             ("final_state_packed", "final_label")):
        state = _unpack_state(raw[name], context["model"].num_qubits)
        residual = (context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8) ^ context["syndrome"]
        _require(not residual.any(), f"UASRE {name} left hard coset")
        _require(np.uint64(_state_label(context["frame"], state)) == np.uint64(_scalar(raw, label_name)),
                 f"UASRE {label_name} changed")
    count = sampler.measurement_rounds
    _require(raw["burn_labels"].shape == (sampler.burn_rounds,), "UASRE burn label shape changed")
    _require(raw["burn_complete_scores"].shape == (sampler.burn_rounds,)
             and raw["burn_b_weights"].shape == (sampler.burn_rounds,),
             "UASRE burn trace shape changed")
    _require(np.all(np.isfinite(raw["burn_complete_scores"])), "UASRE burn score is non-finite")
    measurement = np.unpackbits(raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
                                bitorder="little").astype(np.uint8, copy=False)
    _require(measurement.shape == (count, context["model"].num_qubits), "UASRE measurement shape changed")
    residuals = (context["model"].H_check.astype(np.int64) @ measurement.T.astype(np.int64) % 2).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(not residuals.any() and not raw["measurement_residual_weights"].any(), "UASRE measurement left hard coset")
    labels = np.asarray([_state_label(context["frame"], value) for value in measurement], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "UASRE labels changed")
    _require(np.array_equal(measurement.sum(axis=1).astype(np.int32), raw["measurement_weights"]), "UASRE weights changed")
    _require(raw["measurement_complete_scores"].shape == (count,)
             and raw["measurement_b_weights"].shape == (count,)
             and np.all(np.isfinite(raw["measurement_complete_scores"])),
             "UASRE measurement score trace changed")
    _require(np.array_equal(raw["measurement_block"], np.repeat(np.arange(8, dtype=np.int8), count // 8)),
             "UASRE measurement blocks changed")
    _require(raw["measurement_b_columns"].shape == (count, context["H"].shape[0]), "UASRE B trace shape changed")
    _require(raw["measurement_a_syndromes"].shape == (count, context["H"].shape[1]), "UASRE A trace shape changed")
    log_mass = np.log(control["classical_mass"])
    log_odds = math.log(float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"])))
    scores, b_weights = np.empty(count), np.empty(count, dtype=np.int32)
    for index, state in enumerate(measurement):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, context["syndrome"], context["H"])
        _require(np.array_equal(b_columns, raw["measurement_b_columns"][index]), "UASRE B trace changed")
        _require(np.array_equal(a_syndromes, raw["measurement_a_syndromes"][index]), "UASRE A trace changed")
        scores[index] = collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds)
        b_weights[index] = sum(int(value).bit_count() for value in b_columns)
    _require(np.array_equal(scores, raw["measurement_complete_scores"]), "UASRE score trace changed")
    _require(np.array_equal(b_weights, raw["measurement_b_weights"]), "UASRE B weights changed")
    for phase, rounds, offset in (("burn", sampler.burn_rounds, 0),
                                  ("measurement", count, sampler.burn_rounds)):
        counters = raw[f"{phase}_row_counters"]
        _require(counters.shape == (sampler.num_replicas, 3) and not counters[0].any(),
                 "UASRE row counter shape changed")
        _require(int(counters[:, 0].sum()) == rounds * (sampler.num_replicas - 1) * sampler.positive_row_updates_per_round,
                 "UASRE row attempt count changed")
        _require(np.all(counters[:, 1] <= counters[:, 0]) and np.all(counters[:, 2] >= counters[:, 1]),
                 "UASRE row counters are invalid")
        auxiliary = raw[f"{phase}_auxiliary_counters"]
        _require(auxiliary.shape == (3,) and int(auxiliary[0]) == rounds * sampler.cold_auxiliary_rows_per_round,
                 "UASRE auxiliary attempt count changed")
        _require(0 <= int(auxiliary[1]) <= int(auxiliary[0]) and int(auxiliary[2]) >= int(auxiliary[1]),
                 "UASRE auxiliary counters are invalid")
        if phase == "measurement":
            assignments = raw["measurement_auxiliary_assignments"]
            _require(assignments.shape == (rounds, sampler.cold_auxiliary_rows_per_round),
                     "UASRE auxiliary assignment shape changed")
            _require(np.all(assignments < (1 << context["H"].shape[0])), "UASRE auxiliary assignment range changed")
            _require(int(sum(int(value).bit_count() for value in assignments.flat)) == int(auxiliary[2]),
                     "UASRE auxiliary assignment counter changed")
        hot = raw[f"{phase}_hot_refresh_changed_bits"]
        _require(hot.shape == (rounds,) and np.all((hot >= 0) & (hot <= context["H"].shape[0] ** 2)),
                 "UASRE hot refresh trace changed")
        _require(int(_scalar(raw, f"{phase}_cold_a_column_draws")) == rounds * context["H"].shape[1],
                 "UASRE A redraw count changed")
        _require(np.array_equal(raw[f"{phase}_swap_attempts"], _expected_swap_attempts(sampler.num_replicas, offset, rounds)),
                 "UASRE swap attempts changed")
        _require(np.all(raw[f"{phase}_swap_accepts"] <= raw[f"{phase}_swap_attempts"]), "UASRE swap accepts changed")
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
    half = labels.size // 2
    values = {
        "normalized_weight": np.asarray(raw["measurement_weights"], dtype=np.float64) / context["model"].num_qubits,
        "normalized_B_weight": np.asarray(raw["measurement_b_weights"], dtype=np.float64) / (context["H"].shape[0] ** 2),
        "complete_score_per_factor": np.asarray(raw["measurement_complete_scores"], dtype=np.float64) / context["H"].shape[1],
        "logical_characters": logical,
        "B_masks": b_signs,
    }
    return {name: np.asarray((value.mean(axis=0), value[:half].mean(axis=0), value[half:].mean(axis=0)))
            for name, value in values.items()}


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def _compare(values_a, values_b, absolute_bound, gates):
    mean_a, se_a = _mean_se(values_a)
    mean_b, se_b = _mean_se(values_b)
    delta = np.abs(mean_a - mean_b)
    allowance = float(gates["sigma_multiplier"]) * np.sqrt(se_a * se_a + se_b * se_b) + float(gates["sigma_slack"])
    return {
        "pass": bool(np.all(delta <= float(absolute_bound)) and np.all(delta <= allowance)),
        "max_abs_delta": float(np.max(delta)), "max_three_sigma_allowance": float(np.max(allowance)),
        "failed_components": int(np.count_nonzero((delta > float(absolute_bound)) | (delta > allowance))),
    }


def _comparison_bundle(left, right, gates):
    return {
        "normalized_weight": _compare(left["normalized_weight"], right["normalized_weight"], gates["max_abs_normalized_weight_delta"], gates),
        "normalized_B_weight": _compare(left["normalized_B_weight"], right["normalized_B_weight"], gates["max_abs_normalized_B_weight_delta"], gates),
        "complete_score_per_factor": _compare(left["complete_score_per_factor"], right["complete_score_per_factor"], gates["max_abs_complete_score_delta_per_factor"], gates),
        "logical_characters": _compare(left["logical_characters"], right["logical_characters"], gates["max_abs_logical_character_mean_delta"], gates),
        "B_masks": _compare(left["B_masks"], right["B_masks"], gates["max_abs_B_mask_mean_delta"], gates),
    }


def _bundle_pass(bundle):
    return bool(all(item["pass"] for item in bundle.values()))


def _family_arrays(records, context, control):
    per_trajectory = [_trajectory_observables(record, context, control) for record in records]
    means = {name: np.stack([item[name][0] for item in per_trajectory], axis=0) for name in per_trajectory[0]}
    halves = {
        name: (np.stack([item[name][1] for item in per_trajectory], axis=0),
               np.stack([item[name][2] for item in per_trajectory], axis=0))
        for name in per_trajectory[0]
    }
    return means, halves


def _support_gate(u_records, context, control):
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    p = float(context["config"]["cell"]["p"])
    log_odds = math.log(p / (1.0 - p))
    reference_weight = int(control["epsilon"].sum())
    dimension = int(context["model"].num_qubits - gf2_rank(context["model"].H_check))
    entries = []
    for record in u_records:
        minimum = int(np.asarray(record["measurement_weights"]).min())
        log_bound = dimension * math.log(2.0) + (minimum - reference_weight) * log_odds
        bound = 0.0 if log_bound < -745.0 else min(1.0, math.exp(log_bound))
        entries.append({"minimum_measurement_weight": minimum, "target_support_upper_bound": bound,
                        "trapped_negligible_support": bool(bound <= context["config"]["gates"]["max_negligible_support_upper_bound"])})
    return {"reference_legal_weight": reference_weight, "hard_coset_dimension": dimension,
            "per_trajectory": entries, "pass": not any(item["trapped_negligible_support"] for item in entries)}


def _method_summary(records, context, control):
    gates = context["config"]["gates"]
    by_family = {family: [] for family in INIT_FAMILIES}
    for record in records:
        by_family[str(_scalar(record, "init_family"))].append(record)
    arrays, halves = {}, {}
    for family in INIT_FAMILIES:
        _require(len(by_family[family]) == context["config"]["trajectories_per_family"],
                 f"UASRE family is incomplete: {family}")
        arrays[family], halves[family] = _family_arrays(by_family[family], context, control)
    pairwise = {f"{left}_{right}": _comparison_bundle(arrays[left], arrays[right], gates)
                for left, right in (("P", "U"), ("P", "L"), ("U", "L"))}
    time_stability = {
        family: _comparison_bundle(
            {name: pair[0] for name, pair in halves[family].items()},
            {name: pair[1] for name, pair in halves[family].items()},
            gates,
        )
        for family in INIT_FAMILIES
    }
    support = _support_gate(by_family["U"], context, control)
    counts = {family: len(by_family[family]) for family in INIT_FAMILIES}
    complete = all(value >= gates["minimum_effective_trajectory_count"] for value in counts.values())
    passed = complete and support["pass"] and all(_bundle_pass(value) for value in pairwise.values()) and all(
        _bundle_pass(value) for value in time_stability.values()
    )
    variation = {family: int(np.count_nonzero(np.ptp(arrays[family]["B_masks"], axis=0) > 0.0))
                 for family in INIT_FAMILIES}
    return {"pass": bool(passed), "independent_trajectory_count": counts,
            "target_support_gate": support, "pairwise": pairwise, "time_stability": time_stability,
            "nonconstant_B_mask_diagnostic": variation, "q_top_read_or_computed": False}


def _replay_one(context, manifest, control, task, raw):
    sampler = context["samplers"][task["method_id"]]
    identity = AuxiliaryStabilizerReplicaExchangeSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context, control)
    replay = run_auxiliary_stabilizer_replica_exchange_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"], sampler,
        identity, initial, engine="numba", mass=control["classical_mass"],
    )
    for name in SAMPLER_FIELDS - {"raw_version"}:
        _require(np.array_equal(np.asarray(raw[name]), np.asarray(replay[name])),
                 f"UASRE replay mismatch for {task['method_id']}_{task['init_family']}_{task['trajectory_index']:02d}: {name}")


def analyze(manifest_path, *, replay):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    with np.load(output_root / "CONTROL.npz", allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = _context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    _validate_manifest_context(manifest, context, output_root / "CONTROL.npz")
    control = _load_control(output_root / "CONTROL.npz", context, manifest)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "UASRE run is incomplete")
    completed_hashes = _validate_run_complete(output_root, manifest)
    by_method = {method["id"]: [] for method in context["config"]["methods"]}
    raw_hashes = {}
    for task in manifest["tasks"]:
        path = _task_output_path(output_root, task)
        _require(path.is_file(), f"UASRE raw is missing: {path.name}")
        raw = _validate_one_raw(path, context, manifest, control, task)
        if replay:
            _replay_one(context, manifest, control, task, raw)
        by_method[task["method_id"]].append(raw)
        raw_hashes[path.name] = sha256_file(path)
    _require(raw_hashes == completed_hashes, "UASRE raw files changed after completion")
    methods = {name: _method_summary(records, context, control) for name, records in by_method.items()}
    passing = [name for name, item in methods.items() if item["pass"]]
    if not passing:
        selected, status = None, context["config"]["selection"]["zero_pass"]
    elif len(passing) == 1:
        selected, status = passing[0], "LOCAL_AUXILIARY_STABILIZER_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    else:
        selected, status = "UASRE32-R1-A1", "LOCAL_AUXILIARY_STABILIZER_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    core = {
        "report_version": REPORT_VERSION, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"], "status": status,
        "selected_method": selected, "formal_authorization": False, "remote_authorization": False,
        "posterior_estimation": False, "q_top_read_or_computed": False, "replay_performed": bool(replay),
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
        "manifest_sha256": manifest["manifest_sha256"], "report_sha256": report["report_sha256"], "status": status,
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=ROOT / "local_hard_viability")
    prepare_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    run_parser.add_argument("--workers", type=int, default=6)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    analyze_parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.output, args.registry, args.config)["manifest_sha256"])
    elif args.command == "run":
        print(len(run(args.manifest, args.workers)))
    else:
        print(analyze(args.manifest, replay=not args.no_replay)["status"])


if __name__ == "__main__":
    main()
