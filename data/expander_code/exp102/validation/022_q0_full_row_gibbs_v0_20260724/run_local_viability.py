"""Frozen local adversarial transport screen for exact full-row q=0 Gibbs.

This local-only screen is intentionally narrower than the exp102 formal
discovery contract.  A pass can only say that this kernel merits a new HARD2
comparison; it has no authority for tuning, held-out, or production work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _pack_state,
    _state_label,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_GIBBS_KERNEL,
    FULL_ROW_GIBBS_RAW_VERSION,
    FULL_ROW_GIBBS_VERSION,
    FULL_ROW_L_START_RULE,
    FullRowGibbsConfig,
    FullRowGibbsSeedIdentity,
    build_full_row_elimination_plan,
    run_full_row_gibbs_trajectory,
    select_low_energy_logical_start,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_full_row_gibbs.v0"
MANIFEST_VERSION = "exp102.q0_hgp_full_row_gibbs.local.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_full_row_gibbs.local.tasks.v1"
LOCAL_RAW_VERSION = "exp102.q0_hgp_full_row_gibbs.local.raw.v0"
REPORT_VERSION = "exp102.q0_hgp_full_row_gibbs.local.report.v1"
RUNNER_VERSION = "exp102.q0_hgp_full_row_gibbs.local.runner.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = EXP102_ROOT / "config" / "q0_hgp_full_row_gibbs.v0.json"
DEFAULT_REGISTRY = EXP102_ROOT / "registry" / "registry.json"
INIT_FAMILIES = ("P", "U", "L")
RESULT_FIELDS = frozenset({
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "plan_json", "plan_sha256", "mass_sha256",
    "initial_state_packed", "burn_state_packed", "final_state_packed",
    "measurement_states_packed", "measurement_b_columns",
    "measurement_a_syndromes", "burn_labels", "measurement_labels",
    "measurement_weights", "measurement_residual_weights", "measurement_block",
    "burn_counters", "measurement_counters", "burn_basis_seen", "initial_label",
    "burn_label", "final_label", "engine",
})
RAW_FIELDS = RESULT_FIELDS | {
    "contract_version", "local_raw_version", "task_fingerprint", "task_json",
    "manifest_sha256", "config_sha256", "registry_sha256", "source_binding_sha256",
    "cell_json", "uniform_seed", "model_fingerprint", "frame_fingerprint",
    "character_masks", "character_sha256", "core_seconds", "wall_seconds",
    "sampler_raw_version", "init_family", "trajectory_index",
}


class LocalViabilityConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalViabilityConflict(message)


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
        "q0_hgp_full_row_gibbs.py": sha256_file(
            EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        ),
        "q0_hgp_collapsed.py": sha256_file(
            EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        ),
        "run_local_viability.py": sha256_file(__file__),
        "config": sha256_file(config_path),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def load_config(path, registry):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalViabilityConflict(f"cannot read full-row config: {exc}") from exc
    expected = {
        "cell", "config_version", "contract_version", "gates", "init_families",
        "l_start_rule", "method", "raw_version", "registry_sha256", "resource",
        "runtime_selection", "scope", "trajectory_namespace",
        "trajectories_per_family",
    }
    _require(set(config) == expected, "full-row config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "full-row contract changed")
    _require(config["config_version"] == "exp102.q0_hgp_full_row_gibbs.v0.config.v1",
             "full-row config version changed")
    _require(config["raw_version"] == LOCAL_RAW_VERSION, "full-row raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "registry SHA changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "full-row cell changed")
    _require(config["method"] == {
        "id": "FRG-VE1", "kernel": FULL_ROW_GIBBS_KERNEL,
        "row_schedule": "ascending",
    }, "full-row method changed")
    _require(config["init_families"] == list(INIT_FAMILIES), "full-row starts changed")
    _require(config["l_start_rule"] == FULL_ROW_L_START_RULE, "full-row L rule changed")
    _require(config["resource"] == {
        "burn_sweeps": 64, "measurement_sweeps": 512, "name": "V0",
    }, "full-row resource changed")
    _require(config["trajectories_per_family"] == 8, "full-row trajectory count changed")
    _require(config["gates"] == {
        "minimum_chains_with_four_measurement_label_changes_per_family": 4,
        "minimum_measurement_label_changes_per_family": 64,
        "minimum_measurement_label_delta_rank_per_family": 64,
        "require_all_basis_character_leave_returns": True,
    }, "full-row gates changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_adversarial_transport_preflight_only",
        "remote_authorization": False,
    }, "full-row scope changed")
    runtime = config["runtime_selection"]
    _require(set(runtime) == {
        "basis", "probe_report_sha256", "safety_factor",
        "trajectory_wall_seconds_upper_bound",
    }, "full-row runtime selection changed")
    _require(runtime["basis"] == "runtime_only" and runtime["safety_factor"] == 2.0,
             "full-row runtime basis changed")
    _require(isinstance(config["trajectory_namespace"], str) and config["trajectory_namespace"],
             "full-row trajectory namespace is invalid")
    return config, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _context(registry_path, config_path, *, frozen_l_move=None, frozen_l_metadata=None):
    registry = load_registry(registry_path)
    config, config_sha256 = load_config(config_path, registry)
    probe_path = ROOT / "runtime_probe.json"
    _require(probe_path.is_file(), "runtime probe is missing")
    probe = json.loads(probe_path.read_text(encoding="ascii"))
    _require(probe.get("report_sha256") == config["runtime_selection"]["probe_report_sha256"],
             "runtime probe binding changed")
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
    sampler = FullRowGibbsConfig(
        config["cell"]["p"], config["resource"]["burn_sweeps"],
        config["resource"]["measurement_sweeps"], method_id=config["method"]["id"],
        row_schedule=config["method"]["row_schedule"],
    )
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
                 "frozen L move lost its logical kernel property")
        _require(_state_label(frame, l_start) != _state_label(frame, epsilon),
                 "frozen L start lost its logical separation")
        _require(isinstance(frozen_l_metadata, dict), "frozen L metadata is invalid")
        _require(frozen_l_metadata.get("rule") == FULL_ROW_L_START_RULE
                 and frozen_l_metadata.get("selected_absolute_weight") == int(l_start.sum())
                 and frozen_l_metadata.get("selected_move_weight") == int(l_move.sum())
                 and frozen_l_metadata.get("selected_signature") == int(_state_label(frame, l_move))
                 and frozen_l_metadata.get("selected_move_sha256") == hashlib.sha256(l_move.tobytes()).hexdigest(),
                 "frozen L metadata changed")
        l_metadata = dict(frozen_l_metadata)
    plan = build_full_row_elimination_plan(H)
    mass = build_classical_coset_mass(H, sampler.p, engine="numba")
    characters = np.asarray([np.uint64(1) << np.uint64(bit) for bit in range(model.k)], dtype=np.uint64)
    _require(model.k == 64 and characters.size == 64, "hard m8 logical dimension changed")
    return {
        "registry": registry, "config": config, "config_sha256": config_sha256,
        "code": code, "H": np.ascontiguousarray(H, dtype=np.uint8), "model": model,
        "frame": frame, "uniform_seed": uniform_seed, "epsilon": epsilon,
        "syndrome": syndrome, "sampler": sampler, "l_start": l_start,
        "l_metadata": l_metadata, "plan": plan, "mass": mass,
        "characters": characters, "source_binding": _source_binding(config_path),
    }


def _seed_identity(context, family, trajectory):
    config = context["config"]
    return FullRowGibbsSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=context["config_sha256"],
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(config["cell"]),
        method_id=config["method"]["id"], resource_tier=config["resource"]["name"],
        init_family=family, trajectory_index=trajectory,
        trajectory_namespace=config["trajectory_namespace"],
    )


def _task(context, family, trajectory):
    identity = _seed_identity(context, family, trajectory)
    core = {
        "task_version": TASK_VERSION, "raw_version": LOCAL_RAW_VERSION,
        "cell": context["config"]["cell"], "method_id": context["sampler"].method_id,
        "resource": context["config"]["resource"], "init_family": family,
        "trajectory_index": trajectory, "seed_identity": identity.as_dict(),
    }
    return {**core, "task_fingerprint": sha256_json(core)}


def _control_arrays(context):
    return {
        "epsilon": context["epsilon"], "syndrome": context["syndrome"],
        "l_move": context["l_start"] ^ context["epsilon"],
        "character_masks": context["characters"],
        "classical_mass": context["mass"],
    }


def _manifest_core(context, tasks, control_sha256):
    return {
        "manifest_version": MANIFEST_VERSION, "contract_version": CONTRACT_VERSION,
        "raw_version": LOCAL_RAW_VERSION, "config": context["config"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding": context["source_binding"], "cell": context["config"]["cell"],
        "uniform_seed": int(context["uniform_seed"]),
        "H_sha256": _array_sha256(context["H"]),
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "epsilon_sha256": _array_sha256(context["epsilon"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "character_masks": [int(value) for value in context["characters"]],
        "character_sha256": _array_sha256(context["characters"]),
        "l_start": context["l_metadata"], "plan_sha256": context["plan"].sha256,
        "plan_json": context["plan"].as_dict(),
        "mass_sha256": hashlib.sha256(np.asarray(context["mass"], dtype=">f8").tobytes()).hexdigest(),
        "control_npz_sha256": control_sha256, "tasks": tasks,
    }


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not (output_root / "MANIFEST.json").exists(), "manifest already exists")
    _require(not (output_root / "raw").exists(), "raw directory already exists")
    context = _context(registry_path, config_path)
    tasks = [
        _task(context, family, trajectory)
        for family in INIT_FAMILIES
        for trajectory in range(context["config"]["trajectories_per_family"])
    ]
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
        raise LocalViabilityConflict(f"cannot load manifest: {exc}") from exc
    required = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "epsilon_sha256", "model_fingerprint", "frame_fingerprint",
        "character_masks", "character_sha256", "l_start", "plan_sha256", "plan_json",
        "mass_sha256", "control_npz_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == required, "manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "manifest hash changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "manifest contract changed")
    _require(len(manifest["tasks"]) == 24, "manifest task count changed")
    _require(len({task["task_fingerprint"] for task in manifest["tasks"]}) == 24,
             "manifest tasks are not unique")
    return manifest


def _validate_manifest_context(manifest, context, control_path):
    _require(sha256_file(control_path) == manifest["control_npz_sha256"], "control NPZ changed")
    expected_core = _manifest_core(context, manifest["tasks"], manifest["control_npz_sha256"])
    expected = {**expected_core, "manifest_sha256": sha256_json(expected_core)}
    _require(manifest == expected, "manifest/context binding changed")


def _load_control(path, context, manifest):
    with np.load(path, allow_pickle=False) as data:
        _require(set(data.files) == {"epsilon", "syndrome", "l_move", "character_masks", "classical_mass"},
                 "control NPZ schema changed")
        control = {name: data[name].copy() for name in data.files}
    _require(np.array_equal(control["epsilon"], context["epsilon"]), "control epsilon changed")
    _require(np.array_equal(control["syndrome"], context["syndrome"]), "control syndrome changed")
    _require(np.array_equal(control["l_move"], context["l_start"] ^ context["epsilon"]),
             "control L move changed")
    _require(np.array_equal(control["character_masks"], context["characters"]), "control characters changed")
    _require(np.array_equal(control["classical_mass"], context["mass"]), "control mass changed")
    return control


def _initial_state(task, model, syndrome, control):
    family = task["init_family"]
    if family == "P":
        state = control["epsilon"].copy()
    elif family == "U":
        identity = FullRowGibbsSeedIdentity(**task["seed_identity"])
        state = uniform_hard_coset_state(
            model, syndrome, identity.seed("initialize", "hard_coset"),
        )
    elif family == "L":
        state = control["epsilon"] ^ control["l_move"]
    else:  # pragma: no cover
        raise LocalViabilityConflict("unknown initialization family")
    residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    _require(not residual.any(), "initial state left the hard coset")
    return np.ascontiguousarray(state, dtype=np.uint8)


def _task_context(manifest_path, task):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    control_path = output_root / "CONTROL.npz"
    with np.load(control_path, allow_pickle=False) as data:
        _require(set(data.files) == {"epsilon", "syndrome", "l_move", "character_masks", "classical_mass"},
                 "control NPZ schema changed")
        frozen_l_move = data["l_move"].copy()
    context = _context(
        DEFAULT_REGISTRY, DEFAULT_CONFIG, frozen_l_move=frozen_l_move,
        frozen_l_metadata=manifest["l_start"],
    )
    _validate_manifest_context(manifest, context, control_path)
    _require(task in manifest["tasks"], "task is not in frozen manifest")
    expected = _task(context, task["init_family"], int(task["trajectory_index"]))
    _require(task == expected, "task identity changed")
    control = _load_control(control_path, context, manifest)
    return context, manifest, control


def _task_output_path(output_root, task):
    return Path(output_root) / "raw" / f'{task["init_family"]}_{int(task["trajectory_index"]):02d}.npz'


def _execute_task(manifest_path, task):
    context, manifest, control = _task_context(manifest_path, task)
    output_path = _task_output_path(Path(manifest_path).parent, task)
    _require(not output_path.exists(), f"raw already exists: {output_path.name}")
    identity = FullRowGibbsSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context["model"], context["syndrome"], control)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_full_row_gibbs_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        context["sampler"], identity, initial, engine="numba", mass=control["classical_mass"],
        plan=context["plan"],
    )
    wall_seconds = time.perf_counter() - started_wall
    core_seconds = time.process_time() - started_cpu
    _require(result["raw_version"] == FULL_ROW_GIBBS_RAW_VERSION, "sampler raw version changed")
    _require(result["method_id"] == context["sampler"].method_id, "sampler method changed")
    _require(result["plan_sha256"] == context["plan"].sha256, "sampler plan changed")
    _require(not result["measurement_residual_weights"].any(), "sampler left hard coset")
    raw = {
        **result,
        "raw_version": LOCAL_RAW_VERSION,
        "sampler_raw_version": FULL_ROW_GIBBS_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "uniform_seed": np.int64(context["uniform_seed"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "character_masks": control["character_masks"],
        "character_sha256": manifest["character_sha256"],
        "core_seconds": np.float64(core_seconds),
        "wall_seconds": np.float64(wall_seconds),
        "init_family": task["init_family"],
        "trajectory_index": np.int16(task["trajectory_index"]),
    }
    _require(set(raw) == RAW_FIELDS, "local raw schema drifted")
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
    _require(not (output_root / "RUNNING.json").exists(), "run already marked running")
    _require(not (output_root / "RUN_COMPLETE.json").exists(), "run already completed")
    workers = int(workers)
    _require(1 <= workers <= 8, "worker count must lie in [1, 8]")
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
    results.sort(key=lambda row: row["filename"])
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


def _rebuild_b_columns(state, H):
    rows, columns = H.shape
    B = np.asarray(state, dtype=np.uint8)[columns * columns:].reshape(rows, rows)
    values = np.zeros(rows, dtype=np.uint32)
    for column in range(rows):
        for row in range(rows):
            values[column] |= np.uint32(B[row, column]) << np.uint32(row)
    return values


def _validate_one_raw(path, context, manifest, control, task):
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise LocalViabilityConflict(f"cannot load raw {path}: {exc}") from exc
    _require(set(raw) == RAW_FIELDS, f"raw schema changed: {path.name}")
    _require(not any(value.dtype.hasobject for value in raw.values()), "raw object dtype is forbidden")
    identity = {
        "raw_version": LOCAL_RAW_VERSION,
        "sampler_raw_version": FULL_ROW_GIBBS_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": context["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]),
        "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(),
        "character_sha256": manifest["character_sha256"],
        "init_family": task["init_family"],
        "trajectory_index": int(task["trajectory_index"]),
        "method_id": context["sampler"].method_id,
        "engine": "numba",
        "plan_sha256": context["plan"].sha256,
        "mass_sha256": manifest["mass_sha256"],
    }
    for name, expected in identity.items():
        actual = _scalar(raw, name)
        _require(str(actual) == str(expected), f"raw identity {name} changed: {path.name}")
    _require(int(_scalar(raw, "uniform_seed")) == int(context["uniform_seed"]), "raw uniform seed changed")
    _require(np.array_equal(raw["character_masks"], control["character_masks"]), "raw character masks changed")
    _require(str(_scalar(raw, "sampler_config_json")) == canonical_json(context["sampler"].as_dict()),
             "raw sampler config changed")
    _require(str(_scalar(raw, "seed_identity_json")) == canonical_json(task["seed_identity"]),
             "raw seed identity changed")
    _require(str(_scalar(raw, "plan_json")) == canonical_json(context["plan"].as_dict()), "raw plan changed")
    expected_initial = _initial_state(task, context["model"], context["syndrome"], control)
    _require(np.array_equal(_unpack_state(raw["initial_state_packed"], context["model"].num_qubits), expected_initial),
             "raw initial state changed")
    for name in ("initial_state_packed", "burn_state_packed", "final_state_packed"):
        state = _unpack_state(raw[name], context["model"].num_qubits)
        residual = (
            context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
        ).astype(np.uint8) ^ context["syndrome"]
        _require(not residual.any(), f"raw {name} left hard coset")
    measurement = np.unpackbits(
        raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    measurement_count = context["sampler"].measurement_sweeps
    _require(measurement.shape == (measurement_count, context["model"].num_qubits), "measurement shape changed")
    residuals = (
        context["model"].H_check.astype(np.int64) @ measurement.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(not residuals.any(), "measurement left hard coset")
    labels = np.asarray([_state_label(context["frame"], state) for state in measurement], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "measurement labels changed")
    _require(np.array_equal(measurement.sum(axis=1).astype(np.int32), raw["measurement_weights"]),
             "measurement weights changed")
    _require(not raw["measurement_residual_weights"].any(), "cached residual weights changed")
    _require(np.array_equal(raw["measurement_block"], np.repeat(np.arange(8, dtype=np.int8), measurement_count // 8)),
             "measurement blocks changed")
    _require(raw["measurement_b_columns"].shape == (measurement_count, context["H"].shape[0]),
             "B trace shape changed")
    _require(raw["measurement_a_syndromes"].shape == (measurement_count, context["H"].shape[1]),
             "A syndrome trace shape changed")
    for index, state in enumerate(measurement):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, context["syndrome"], context["H"])
        _require(np.array_equal(b_columns, raw["measurement_b_columns"][index]), "B trace changed")
        _require(np.array_equal(a_syndromes, raw["measurement_a_syndromes"][index]), "A syndrome trace changed")
    _require(raw["burn_counters"].shape == (5,) and raw["measurement_counters"].shape == (5,),
             "counter shape changed")
    _require(int(raw["burn_counters"][0]) == context["sampler"].burn_sweeps * context["H"].shape[0],
             "burn full-row counter changed")
    _require(int(raw["measurement_counters"][0]) == measurement_count * context["H"].shape[0],
             "measurement full-row counter changed")
    return raw


def _gf2_rank_uint64(values):
    pivots = {}
    for raw in np.asarray(values, dtype=np.uint64):
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def _basis_leave_return(labels, bit):
    signs = ((np.asarray(labels, dtype=np.uint64) >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
    initial = int(signs[0])
    opposite_seen = False
    for sign in signs[1:]:
        if int(sign) != initial:
            opposite_seen = True
        elif opposite_seen:
            return True
    return False


def _family_summary(records, config):
    labels = [np.asarray(record["measurement_labels"], dtype=np.uint64) for record in records]
    changes = [int(np.count_nonzero(trace[1:] != trace[:-1])) for trace in labels]
    deltas = np.concatenate([trace[1:] ^ trace[:-1] for trace in labels])
    leave_returns = np.asarray([
        any(_basis_leave_return(trace, bit) for trace in labels)
        for bit in range(64)
    ], dtype=np.uint8)
    b_changes = [int(np.count_nonzero(
        np.any(trace[1:] != trace[:-1], axis=1)
    )) for trace in [record["measurement_b_columns"] for record in records]]
    gates = config["gates"]
    valid = (
        sum(changes) >= gates["minimum_measurement_label_changes_per_family"]
        and sum(value >= 4 for value in changes) >= gates[
            "minimum_chains_with_four_measurement_label_changes_per_family"
        ]
        and _gf2_rank_uint64(deltas) >= gates["minimum_measurement_label_delta_rank_per_family"]
        and bool(leave_returns.all())
    )
    return {
        "measurement_label_changes": int(sum(changes)),
        "per_chain_label_changes": changes,
        "chains_with_at_least_four_changes": int(sum(value >= 4 for value in changes)),
        "label_delta_rank": _gf2_rank_uint64(deltas),
        "basis_leave_return_count": int(leave_returns.sum()),
        "basis_leave_returns": leave_returns.tolist(),
        "measurement_B_state_changes": int(sum(b_changes)),
        "per_chain_B_state_changes": b_changes,
        "transport_gate_pass": bool(valid),
    }


def _replay_one(context, manifest, control, task, raw):
    identity = FullRowGibbsSeedIdentity(**task["seed_identity"])
    initial = _initial_state(task, context["model"], context["syndrome"], control)
    replay = run_full_row_gibbs_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        context["sampler"], identity, initial, engine="numba", mass=control["classical_mass"],
        plan=context["plan"],
    )
    for name in RESULT_FIELDS:
        if name == "raw_version":
            continue
        _require(np.array_equal(np.asarray(raw[name]), np.asarray(replay[name])),
                 f"replay mismatch for {task['init_family']}_{task['trajectory_index']:02d}: {name}")


def analyze(manifest_path, *, replay):
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    context = _context(DEFAULT_REGISTRY, DEFAULT_CONFIG)
    _validate_manifest_context(manifest, context, output_root / "CONTROL.npz")
    control = _load_control(output_root / "CONTROL.npz", context, manifest)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "run is incomplete")
    by_family = {family: [] for family in INIT_FAMILIES}
    raw_hashes = {}
    for task in manifest["tasks"]:
        path = _task_output_path(output_root, task)
        _require(path.is_file(), f"missing raw: {path.name}")
        raw = _validate_one_raw(path, context, manifest, control, task)
        if replay:
            _replay_one(context, manifest, control, task, raw)
        by_family[task["init_family"]].append(raw)
        raw_hashes[path.name] = sha256_file(path)
    family = {name: _family_summary(records, context["config"])
              for name, records in by_family.items()}
    status = (
        "LOCAL_LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
        if all(value["transport_gate_pass"] for value in family.values())
        else "LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE"
    )
    core = {
        "report_version": REPORT_VERSION, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": status, "formal_authorization": False,
        "remote_authorization": False, "posterior_estimation": False,
        "replay_performed": bool(replay), "family": family,
        "raw_sha256": raw_hashes,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(output_root / "REPORT.json", report)
    if replay:
        atomic_json(output_root / "REPLAY.json", {
            "manifest_sha256": manifest["manifest_sha256"],
            "raw_sha256": raw_hashes, "report_sha256": report["report_sha256"],
        })
    atomic_json(output_root / "SUCCESS.json", {
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"], "status": status,
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
    run_parser.add_argument("--workers", type=int, default=8)
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
