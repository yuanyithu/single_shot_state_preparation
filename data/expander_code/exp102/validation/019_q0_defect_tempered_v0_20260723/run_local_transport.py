#!/usr/bin/env python3
"""Frozen local D=0 transport preflight for defect-tempered sampling.

This V0 run asks one narrow question: can a finite-syndrome replica ladder
leave and return to the hard coset while changing more than a tiny logical
subgroup?  It never estimates q_top.  In particular, every analysis below
filters fixed-clock cold-rung states by D=0 before interpreting a label.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_defect_tempered import (
    DEFECT_TEMPERED_KERNEL,
    DEFECT_TEMPERED_RAW_VERSION,
    DefectTemperedConfig,
    DefectTemperedSeedIdentity,
    run_defect_tempered_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    _signature_rank_masks,
    character_values,
    frozen_character_set,
    reduce_logical_basis,
    state_label,
    uniform_hard_coset_state,
    unpack_states,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_defect_tempered.v0"
CONFIG_VERSION = "exp102.q0_defect_tempered.v0.config.v1"
MANIFEST_VERSION = "exp102.q0_defect_tempered.v0.manifest.v1"
TASK_VERSION = "exp102.q0_defect_tempered.v0.tasks.v1"
RAW_VERSION = "exp102.q0_defect_tempered.v0.raw.v1"
REPORT_VERSION = "exp102.q0_defect_tempered.v0.report.v1"
L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
ROOT = Path("data/expander_code/exp102")
DEFAULT_REGISTRY = ROOT / "registry/registry.json"
DEFAULT_CONFIG = ROOT / "config/q0_defect_tempered.v0.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_m8_d0_transport_v0"
SOURCE_FILES = (
    "data/expander_code/exp102/exp102_pipeline/q0_defect_tempered.py",
    "data/expander_code/exp102/validation/019_q0_defect_tempered_v0_20260723/run_local_transport.py",
)


class LocalDtcConflict(RuntimeError):
    """The immutable local diagnostic cannot be trusted or reused."""


def _require(condition, message):
    if not condition:
        raise LocalDtcConflict(message)


def _scalar(value, name):
    array = np.asarray(value)
    _require(array.shape == (), f"{name} must be scalar")
    return array.item()


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _result_digest(result):
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_defect_tempered.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        _require(not value.dtype.hasobject, f"DTC result {name} has object dtype")
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        dtype = value.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(
        len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit),
        "DTC source commit is invalid",
    )
    files = {name: sha256_file(name) for name in SOURCE_FILES}
    core = {"source_commit": source_commit, "source_files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _load_config(path, registry):
    try:
        config = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalDtcConflict(f"cannot load DTC config: {exc}") from exc
    expected = {
        "character_seed_namespace", "cell", "config_version", "contract_version",
        "gates", "init_families", "l_start_rule", "method",
        "num_nonbasis_characters", "raw_version", "registry_sha256", "resource",
        "scope", "trajectories_per_family", "trajectory_namespace",
    }
    _require(set(config) == expected, "DTC config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "DTC contract changed")
    _require(config["config_version"] == CONFIG_VERSION, "DTC config version changed")
    _require(config["raw_version"] == RAW_VERSION, "DTC raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "DTC registry changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "DTC cell changed")
    _require(config["init_families"] == ["P", "U", "L"], "DTC starts changed")
    _require(config["l_start_rule"] == L_START_RULE, "DTC L rule changed")
    _require(config["trajectories_per_family"] == 8, "DTC trajectory count changed")
    _require(config["num_nonbasis_characters"] == 64, "DTC character count changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_d0_transport_preflight_only",
        "remote_authorization": False,
    }, "DTC scope changed")
    _require(config["resource"] == {
        "burn_rounds": 256, "measurement_rounds": 2048, "name": "V0",
    }, "DTC resource changed")
    _require(config["method"]["id"] == "DTC21-S1", "DTC method changed")
    _require(config["method"]["kernel"] == DEFECT_TEMPERED_KERNEL, "DTC kernel changed")
    _require(config["method"]["sweeps_per_round"] == 1, "DTC sweep schedule changed")
    sampler = DefectTemperedConfig(
        method_id=config["method"]["id"],
        p=float(config["cell"]["p"]),
        kq_values=tuple(config["method"]["kq_values"]),
        burn_rounds=config["resource"]["burn_rounds"],
        measurement_rounds=config["resource"]["measurement_rounds"],
        sweeps_per_round=config["method"]["sweeps_per_round"],
    )
    _require(sampler.num_replicas == 21, "DTC ladder length changed")
    _require(config["gates"] == {
        "minimum_d0_label_delta_rank_per_family": 16,
        "minimum_d0_leave_returns_per_trajectory": 8,
        "minimum_d0_observations_per_trajectory": 256,
        "minimum_d0_sign_leave_returns_per_character_set_per_family": 16,
        "minimum_family_d0_label_changes": 64,
        "minimum_trajectories_with_eight_d0_label_changes_per_family": 6,
    }, "DTC gates changed")
    for name in ("character_seed_namespace", "trajectory_namespace"):
        _require(isinstance(config[name], str) and config[name], f"DTC {name} is invalid")
    return config, sampler, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "DTC disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22",
        registry["registry_sha256"], code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _select_l_move(epsilon, model, frame):
    """Freeze one legal low-energy nontrivial start before any chain exists."""
    reduced = reduce_logical_basis(model.logical_move_basis)
    seen = set()
    selected = None
    candidate_count = 0
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(state_label(frame, move))
            _require(signature != 0, "DTC L candidate lost logical signature")
            residual = (
                model.H_check.astype(np.int64) @ move.astype(np.int64) % 2
            ).astype(np.uint8)
            _require(not residual.any(), "DTC L candidate left kernel")
            candidate_count += 1
            key = (
                int(np.count_nonzero(epsilon ^ move)), int(move.sum()), signature, packed,
            )
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    _require(selected is not None, "DTC has no legal L start")
    key, move = selected
    return move, {
        "rule": L_START_RULE,
        "candidate_orders": [1, 2, 3],
        "candidate_count": candidate_count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": _array_sha256(move),
    }


def _context(registry_path, config_path, *, manifest=None):
    registry = load_registry(registry_path)
    config, sampler, config_sha256 = _load_config(config_path, registry)
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = _attempt022_uniform_seed(registry, code, config["cell"])
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < float(config["cell"]["p"])).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(bool(syndrome.any()), "DTC planted syndrome unexpectedly vanishes")
    character_seed = derive_seed(
        config["character_seed_namespace"], registry["registry_sha256"],
        code["code_id"], "d0_transport_characters",
    )
    characters = frozen_character_set(model.k, character_seed, config["num_nonbasis_characters"])
    l_move, l_start = _select_l_move(epsilon, model, frame)
    context = {
        "registry": registry,
        "config": config,
        "config_sha256": config_sha256,
        "sampler": sampler,
        "code": code,
        "H": np.ascontiguousarray(H, dtype=np.uint8),
        "model": model,
        "frame": frame,
        "uniform_seed": int(uniform_seed),
        "epsilon": epsilon,
        "syndrome": syndrome,
        "characters": characters,
        "l_move": l_move,
        "l_start": l_start,
        "source_binding": _source_binding(),
    }
    if manifest is not None:
        _validate_manifest_context(manifest, context)
    return context


def _task(context, family, index):
    identity = DefectTemperedSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=context["config_sha256"],
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(context["config"]["cell"]),
        method_id=context["sampler"].method_id,
        resource_tier=context["config"]["resource"]["name"],
        init_family=family,
        trajectory_index=int(index),
        trajectory_namespace=context["config"]["trajectory_namespace"],
    )
    return {
        "task_version": TASK_VERSION,
        "raw_version": RAW_VERSION,
        "method_id": context["sampler"].method_id,
        "resource": context["config"]["resource"],
        "cell": context["config"]["cell"],
        "init_family": family,
        "trajectory_index": int(index),
        "sampler_config": context["sampler"].as_dict(),
        "seed_identity": identity.as_dict(),
        "engine": "numba",
    }


def _tasks(context):
    return [
        _task(context, family, index)
        for family in context["config"]["init_families"]
        for index in range(context["config"]["trajectories_per_family"])
    ]


def _manifest_core(context, tasks):
    return {
        "manifest_version": MANIFEST_VERSION,
        "contract_version": CONTRACT_VERSION,
        "raw_version": RAW_VERSION,
        "config": context["config"],
        "config_sha256": context["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding": context["source_binding"],
        "cell": context["config"]["cell"],
        "uniform_seed": context["uniform_seed"],
        "H_sha256": _array_sha256(context["H"]),
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "model_fingerprint": context["model"].fingerprint(),
        "logical_frame_fingerprint": context["frame"].fingerprint(),
        "character_sha256": context["characters"].character_sha256,
        "character_masks": [int(value) for value in context["characters"].masks],
        "l_start": context["l_start"],
        "ladder_sha256": context["sampler"].ladder_sha256,
        "tasks": tasks,
    }


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not output_root.exists(), "DTC output root already exists")
    context = _context(registry_path, config_path)
    tasks = _tasks(context)
    core = _manifest_core(context, tasks)
    atomic_json(output_root / "MANIFEST.json", {**core, "manifest_sha256": sha256_json(core)})
    return output_root / "MANIFEST.json"


def _load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalDtcConflict(f"cannot load DTC manifest: {exc}") from exc
    expected = {
        "manifest_version", "contract_version", "raw_version", "config",
        "config_sha256", "registry_sha256", "source_binding", "cell",
        "uniform_seed", "H_sha256", "syndrome_sha256", "model_fingerprint",
        "logical_frame_fingerprint", "character_sha256", "character_masks",
        "l_start", "ladder_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == expected, "DTC manifest fields changed")
    core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "DTC manifest SHA changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "DTC manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "DTC manifest contract changed")
    _require(manifest["raw_version"] == RAW_VERSION, "DTC manifest raw version changed")
    _require(isinstance(manifest["tasks"], list) and len(manifest["tasks"]) == 24,
             "DTC manifest task count changed")
    _require(len({sha256_json(task) for task in manifest["tasks"]}) == 24,
             "DTC manifest task duplication")
    return manifest


def _validate_manifest_context(manifest, context):
    core = _manifest_core(context, manifest["tasks"])
    expected = {**core, "manifest_sha256": sha256_json(core)}
    _require(manifest == expected, "DTC manifest/context binding changed")


def _initial_state(context, task):
    family = task["init_family"]
    identity = DefectTemperedSeedIdentity(**task["seed_identity"])
    if family == "P":
        state = context["epsilon"].copy()
    elif family == "U":
        state = uniform_hard_coset_state(
            context["model"], context["syndrome"], identity.seed("initialize", "hard_coset"),
        )
    elif family == "L":
        state = context["epsilon"] ^ context["l_move"]
    else:
        raise LocalDtcConflict("DTC task has an unknown initialization family")
    residual = (
        context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ context["syndrome"]
    _require(not residual.any(), "DTC initialization left hard coset")
    if family == "L":
        _require(
            state_label(context["frame"], state) != state_label(context["frame"], context["epsilon"]),
            "DTC L start lost logical separation",
        )
        _require(int(state.sum()) == context["l_start"]["selected_absolute_weight"],
                 "DTC L start energy changed")
    return np.ascontiguousarray(state, dtype=np.uint8)


def _execute_task(context, task):
    _require(task in _tasks(context), "DTC task is not canonical")
    identity = DefectTemperedSeedIdentity(**task["seed_identity"])
    initial = _initial_state(context, task)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_defect_tempered_trajectory(
        context["model"], context["frame"], context["syndrome"],
        context["sampler"], identity, initial, engine="numba",
    )
    core_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    _require(result["raw_version"] == DEFECT_TEMPERED_RAW_VERSION, "DTC engine raw changed")
    _require(result["kernel"] == DEFECT_TEMPERED_KERNEL, "DTC engine kernel changed")
    _require(_scalar(result["ladder_sha256"], "ladder_sha256") == context["sampler"].ladder_sha256,
             "DTC ladder changed")
    _require(np.array_equal(result["ladder_kq"], np.asarray(context["sampler"].kq_values)),
             "DTC ladder values changed")
    for name, value in result.items():
        array = np.asarray(value)
        if array.dtype.kind == "f":
            _require(np.all(np.isfinite(array)), f"DTC result {name} is non-finite")
    return result, core_seconds, wall_seconds


def _raw_payload(context, manifest, task, result, core_seconds, wall_seconds):
    payload = {
        "raw_version": np.array(RAW_VERSION),
        "contract_version": np.array(CONTRACT_VERSION),
        "manifest_sha256": np.array(manifest["manifest_sha256"]),
        "task_fingerprint": np.array(sha256_json(task)),
        "task_json": np.array(canonical_json(task)),
        "source_binding_json": np.array(canonical_json(context["source_binding"])),
        "config_sha256": np.array(context["config_sha256"]),
        "registry_sha256": np.array(context["registry"]["registry_sha256"]),
        "uniform_seed": np.array(context["uniform_seed"], dtype=np.int64),
        "syndrome": np.asarray(context["syndrome"], dtype=np.uint8),
        "character_masks": np.asarray(context["characters"].masks, dtype=np.uint64),
        "character_sha256": np.array(context["characters"].character_sha256),
        "l_start_json": np.array(canonical_json(context["l_start"])),
        "l_move_sha256": np.array(_array_sha256(context["l_move"])),
        "trajectory_digest": np.array(_result_digest(result)),
        "core_seconds": np.array(float(core_seconds), dtype=np.float64),
        "wall_seconds": np.array(float(wall_seconds), dtype=np.float64),
    }
    payload.update({f"dtc_{name}": np.asarray(value) for name, value in result.items()})
    return payload


def _raw_path(output_root, task):
    return Path(output_root) / "raw" / task["init_family"] / f"t{task['trajectory_index']:02d}.npz"


def _load_raw(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise LocalDtcConflict(f"cannot load DTC raw {path}: {exc}") from exc
    _require(not any(value.dtype.hasobject for value in raw.values()), "DTC raw has object dtype")
    return raw


def _run_worker(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require(task in manifest["tasks"], "DTC worker task absent from manifest")
    output_path = _raw_path(output_root, task)
    _require(not output_path.exists(), "DTC raw already exists")
    result, core_seconds, wall_seconds = _execute_task(context, task)
    atomic_npz(output_path, **_raw_payload(
        context, manifest, task, result, core_seconds, wall_seconds,
    ))
    return {
        "task_fingerprint": sha256_json(task), "path": str(output_path),
        "core_seconds": core_seconds, "wall_seconds": wall_seconds,
    }


def run(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *, workers=1):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    _context(registry_path, config_path, manifest=manifest)
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers > 0,
             "DTC worker count is invalid")
    for name in ("RUNNING.json", "SUCCESS.json", "FAILED.json"):
        _require(not (output_root / name).exists(), "DTC run marker already exists")
    for task in manifest["tasks"]:
        _require(not _raw_path(output_root, task).exists(), "DTC raw already exists")
    atomic_json(output_root / "RUNNING.json", {
        "stage": "run", "manifest_sha256": manifest["manifest_sha256"], "workers": workers,
    })
    values = (str(manifest_path), str(registry_path), str(config_path), str(output_root))
    try:
        if workers == 1:
            completed = [_run_worker(*values, task) for task in manifest["tasks"]]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_worker, *values, task) for task in manifest["tasks"]]
                completed = [future.result() for future in futures]
    except Exception as exc:
        atomic_json(output_root / "FAILED.json", {
            "stage": "run", "manifest_sha256": manifest["manifest_sha256"],
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    atomic_json(output_root / "SUCCESS.json", {
        "stage": "run", "manifest_sha256": manifest["manifest_sha256"],
        "task_count": len(completed),
        "task_fingerprints": sorted(row["task_fingerprint"] for row in completed),
    })
    return completed


def _leave_return(labels, masks):
    if labels.size < 3:
        return np.zeros(masks.size, dtype=bool)
    signs = character_values(labels, masks)
    origin = signs[0]
    left = np.zeros(signs.shape[1], dtype=bool)
    returned = np.zeros(signs.shape[1], dtype=bool)
    for values in signs[1:]:
        changed = values != origin
        returned |= left & ~changed
        left |= changed
    return returned


def _validate_raw_task(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    raw = _load_raw(_raw_path(output_root, task))
    timing_fields = ("core_seconds", "wall_seconds")
    for name in timing_fields:
        value = float(_scalar(raw.get(name), name))
        _require(math.isfinite(value) and value >= 0.0, f"DTC {name} invalid")
    result, _, _ = _execute_task(context, task)
    expected = _raw_payload(
        context, manifest, task, result,
        float(_scalar(raw["core_seconds"], "core_seconds")),
        float(_scalar(raw["wall_seconds"], "wall_seconds")),
    )
    _require(set(raw) == set(expected), "DTC raw schema changed")
    for name, value in expected.items():
        if name not in timing_fields:
            _require(np.array_equal(raw[name], value), f"DTC raw seed replay changed: {name}")
    _require(_scalar(raw["dtc_raw_version"], "dtc raw version") == DEFECT_TEMPERED_RAW_VERSION,
             "DTC embedded raw version changed")
    _require(_scalar(raw["dtc_kernel"], "dtc kernel") == DEFECT_TEMPERED_KERNEL,
             "DTC embedded kernel changed")
    states = unpack_states(raw["dtc_measurement_states_packed"], context["model"].num_qubits)
    residuals = (
        context["model"].H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    defects = residuals.sum(axis=1).astype(np.int32)
    _require(np.array_equal(defects, raw["dtc_measurement_defects"]), "DTC defect replay changed")
    _require(np.array_equal(defects, raw["dtc_measurement_residual_weights"]),
             "DTC residual-weight replay changed")
    _require(np.array_equal(states.sum(axis=1).astype(np.int32), raw["dtc_measurement_weights"]),
             "DTC measurement weights changed")
    labels = np.asarray([state_label(context["frame"], state) for state in states], dtype=np.uint64)
    _require(np.array_equal(labels, raw["dtc_measurement_labels"]), "DTC labels changed")
    d0_mask = (defects == 0).astype(np.uint8)
    _require(np.array_equal(d0_mask, raw["dtc_measurement_d0_mask"]), "DTC D0 mask changed")
    _require(np.array_equal(
        raw["dtc_measurement_block"],
        np.repeat(np.arange(8, dtype=np.int8), labels.size // 8),
    ), "DTC measurement blocks changed")
    d0_labels = labels[d0_mask.astype(bool)]
    deltas = d0_labels[1:] ^ d0_labels[:-1] if d0_labels.size > 1 else np.empty(0, dtype=np.uint64)
    changed = deltas[deltas != 0]
    leave_return = _leave_return(d0_labels, context["characters"].masks)
    return {
        "task": task,
        "d0_observations": int(d0_labels.size),
        "d0_label_changes": int(changed.size),
        "d0_deltas": [int(value) for value in changed],
        "d0_leave_return": leave_return.astype(np.uint8),
        "d0_returns": int(_scalar(raw["dtc_measurement_cold_d0_returns"], "D0 returns")),
        "d0_leaves": int(_scalar(raw["dtc_measurement_cold_d0_leaves"], "D0 leaves")),
        "min_swap_rate": float(np.min(
            np.asarray(raw["dtc_measurement_swap_accepts"], dtype=np.float64)
            / np.maximum(np.asarray(raw["dtc_measurement_swap_attempts"], dtype=np.float64), 1.0)
        )),
        "wall_seconds": float(_scalar(raw["wall_seconds"], "wall_seconds")),
    }


def _family_summary(rows, context):
    gates = context["config"]["gates"]
    k = context["model"].k
    masks = context["characters"].masks
    all_deltas = [np.uint64(value) for row in rows for value in row["d0_deltas"]]
    leave_return = np.asarray([row["d0_leave_return"] for row in rows], dtype=np.uint8)
    failures = []
    for index, row in enumerate(rows):
        if row["d0_observations"] < gates["minimum_d0_observations_per_trajectory"]:
            failures.append(f"t{index:02d}:d0_observations")
        if row["d0_returns"] < gates["minimum_d0_leave_returns_per_trajectory"]:
            failures.append(f"t{index:02d}:d0_returns")
    summary = {
        "chain_count": len(rows),
        "d0_observations_per_trajectory": [row["d0_observations"] for row in rows],
        "d0_leave_returns_per_trajectory": [row["d0_returns"] for row in rows],
        "d0_leaves_per_trajectory": [row["d0_leaves"] for row in rows],
        "d0_label_changes_per_trajectory": [row["d0_label_changes"] for row in rows],
        "family_d0_label_changes": int(sum(row["d0_label_changes"] for row in rows)),
        "chains_with_eight_d0_label_changes": int(sum(
            row["d0_label_changes"] >= 8 for row in rows
        )),
        "d0_label_delta_rank": _signature_rank_masks(all_deltas, k),
        "basis_characters_with_d0_leave_return": int(np.count_nonzero(leave_return[:, :k].sum(axis=0))) if rows else 0,
        "nonbasis_characters_with_d0_leave_return": int(np.count_nonzero(leave_return[:, k:].sum(axis=0))) if rows else 0,
        "minimum_adjacent_swap_rate_diagnostic": float(min(row["min_swap_rate"] for row in rows)) if rows else None,
        "median_wall_seconds": float(np.median([row["wall_seconds"] for row in rows])) if rows else None,
        "transport_gate": "D0 observations, D0 leave-return loops, and D0 label deltas only",
        "not_a_posterior_or_qtop_gate": True,
    }
    if summary["chain_count"] != context["config"]["trajectories_per_family"]:
        failures.append("chain_count")
    if summary["family_d0_label_changes"] < gates["minimum_family_d0_label_changes"]:
        failures.append("family_d0_label_changes")
    if summary["chains_with_eight_d0_label_changes"] < gates[
            "minimum_trajectories_with_eight_d0_label_changes_per_family"]:
        failures.append("chains_with_eight_d0_label_changes")
    if summary["d0_label_delta_rank"] < gates["minimum_d0_label_delta_rank_per_family"]:
        failures.append("d0_label_delta_rank")
    if summary["basis_characters_with_d0_leave_return"] < gates[
            "minimum_d0_sign_leave_returns_per_character_set_per_family"]:
        failures.append("basis_d0_leave_return")
    if summary["nonbasis_characters_with_d0_leave_return"] < gates[
            "minimum_d0_sign_leave_returns_per_character_set_per_family"]:
        failures.append("nonbasis_d0_leave_return")
    summary["failures"] = failures
    summary["passes_transport_gate"] = not failures
    return summary


def analyze(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *, workers=1):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "SUCCESS.json").is_file(), "DTC raw stage did not succeed")
    _require(not (output_root / "FAILED.json").exists(), "DTC raw stage failed")
    _require(not (output_root / "REPORT.json").exists(), "DTC report already exists")
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers > 0,
             "DTC analysis worker count is invalid")
    for task in manifest["tasks"]:
        _require(_raw_path(output_root, task).is_file(), "DTC raw is missing")
    values = (str(manifest_path), str(registry_path), str(config_path), str(output_root))
    if workers == 1:
        rows = [_validate_raw_task(*values, task) for task in manifest["tasks"]]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_validate_raw_task, *values, task) for task in manifest["tasks"]]
            rows = [future.result() for future in futures]
    families = {
        family: _family_summary(
            [row for row in rows if row["task"]["init_family"] == family], context,
        )
        for family in context["config"]["init_families"]
    }
    status = (
        "LOCAL_D0_TRANSPORT_SIGNAL_FOR_SCREEN"
        if all(summary["passes_transport_gate"] for summary in families.values())
        else "LOCAL_D0_TRANSPORT_NOT_VIABLE"
    )
    core = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": status,
        "scope": "d0_transport_only_not_qtop_or_formal_authorization",
        "families": families,
        "raw_count": len(rows),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(output_root / "REPORT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.registry, args.config)
    elif args.command == "run":
        result = run(args.output, args.registry, args.config, workers=args.workers)
    else:
        result = analyze(args.output, args.registry, args.config, workers=args.workers)
    print(canonical_json(result if isinstance(result, dict) else {"path": str(result)}))


if __name__ == "__main__":
    main()
