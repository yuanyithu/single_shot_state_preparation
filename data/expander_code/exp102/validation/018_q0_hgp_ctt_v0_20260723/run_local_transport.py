"""Fail-closed local transport preflight for collapsed tempered transitions.

This runner is deliberately narrower than a posterior experiment.  It freezes
one hard m8 cell, runs independent P/U/L trajectories, and accepts only direct
logical-label transport evidence.  Its output can never authorize a formal
exp102 scan or estimate q_top.
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

# Executing this validation file by path otherwise hides the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
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
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    COLLAPSED_TT_COUNTER_NAMES,
    COLLAPSED_TT_KERNEL,
    COLLAPSED_TT_RAW_VERSION,
    CollapsedTemperedTransitionConfig,
    CollapsedTemperedTransitionSeedIdentity,
    build_classical_coset_mass,
    run_collapsed_tempered_transition_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_ctt.v0"
CONFIG_VERSION = "exp102.q0_hgp_ctt.v0.config.v1"
MANIFEST_VERSION = "exp102.q0_hgp_ctt.v0.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_ctt.v0.tasks.v1"
RAW_VERSION = "exp102.q0_hgp_ctt.v0.raw.v1"
REPORT_VERSION = "exp102.q0_hgp_ctt.v0.report.v1"
L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
SOURCE_FILES = (
    "data/expander_code/exp102/exp102_pipeline/q0_global.py",
    "data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed.py",
    "data/expander_code/exp102/validation/018_q0_hgp_ctt_v0_20260723/run_local_transport.py",
)
ROOT = Path("data/expander_code/exp102")
DEFAULT_REGISTRY = ROOT / "registry/registry.json"
DEFAULT_CONFIG = ROOT / "config/q0_hgp_ctt.v0.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_m8_transport_ctt_v0"


class LocalTransportConflict(RuntimeError):
    """A local diagnostic artifact cannot be trusted or reused."""


def _require(condition, message):
    if not condition:
        raise LocalTransportConflict(message)


def _scalar(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise LocalTransportConflict(f"{name} must be scalar")
    return array.item()


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _float64_big_endian_sha256(values):
    return hashlib.sha256(
        np.asarray(values, dtype=">f8").tobytes(order="C")
    ).hexdigest()


def _trajectory_digest(result):
    """Digest every deterministic CTT field, including path counters."""
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_ctt.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        _require(not value.dtype.hasobject, f"CTT result {name} has object dtype")
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
        len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit),
        "CTT source commit is invalid",
    )
    files = {path: sha256_file(path) for path in SOURCE_FILES}
    return {
        "source_commit": source_commit,
        "source_files": files,
        "source_binding_sha256": sha256_json({
            "source_commit": source_commit,
            "source_files": files,
        }),
    }


def _load_config(config_path, registry):
    try:
        config = json.loads(Path(config_path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalTransportConflict(f"cannot load CTT V0 config: {exc}") from exc
    expected_fields = {
        "character_seed_namespace", "cell", "config_version", "contract_version",
        "gates", "init_families", "l_start_rule", "method",
        "num_nonbasis_characters", "raw_version", "registry_sha256", "resource",
        "scope", "trajectories_per_family", "trajectory_namespace",
    }
    _require(set(config) == expected_fields, "CTT V0 config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "CTT V0 contract changed")
    _require(config["config_version"] == CONFIG_VERSION, "CTT V0 config version changed")
    _require(config["raw_version"] == RAW_VERSION, "CTT V0 raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "CTT V0 registry changed")
    _require(config["init_families"] == ["P", "U", "L"], "CTT V0 starts changed")
    _require(config["l_start_rule"] == L_START_RULE, "CTT V0 L start changed")
    _require(config["trajectories_per_family"] == 8, "CTT V0 trajectory count changed")
    _require(config["num_nonbasis_characters"] == 64, "CTT V0 character count changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_transport_preflight_only",
        "remote_authorization": False,
    }, "CTT V0 scope changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "CTT V0 cell changed")
    _require(config["resource"] == {
        "burn_steps": 512, "measurement_steps": 4096, "name": "V0",
    }, "CTT V0 resource changed")
    _require(config["method"] == {
        "block_size": 8,
        "id": "CTT64-S1",
        "kernel": COLLAPSED_TT_KERNEL,
        "num_levels": 64,
        "prior_endpoint": "exact_iid_bernoulli_B",
        "reversible_sweeps_per_level": 1,
    }, "CTT V0 method changed")
    _require(config["gates"] == {
        "minimum_chains_with_eight_measurement_cross_label_changes_per_family": 6,
        "minimum_measurement_cross_label_changes_per_family": 128,
        "minimum_measurement_label_delta_rank_per_family": 64,
        "require_all_basis_character_leave_returns": True,
        "require_all_nonbasis_character_leave_returns": True,
    }, "CTT V0 gates changed")
    for name in ("character_seed_namespace", "trajectory_namespace"):
        _require(isinstance(config[name], str) and config[name], f"CTT V0 {name} is invalid")
    return config, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "CTT V0 disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22",
        registry["registry_sha256"], code["code_id"],
        int(cell["disorder_index"]), "uniforms",
    )


def _select_low_energy_logical_move(epsilon, model, frame):
    """Freeze a nontrivial logical start before chains exist.

    The old all-directions XOR was a very high-energy state on this cell.  This
    selector instead considers the code-only reduced single/pair/triple set
    and chooses its fixed target-energy minimum with deterministic tie breaks.
    It never reads MCMC output.
    """
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
            label = int(state_label(frame, move))
            _require(label != 0, "CTT V0 logical candidate lost its signature")
            residual = (
                model.H_check.astype(np.int64) @ move.astype(np.int64) % 2
            ).astype(np.uint8)
            _require(not residual.any(), "CTT V0 logical candidate left the kernel")
            candidate_count += 1
            key = (
                int(np.count_nonzero(epsilon ^ move)),
                int(move.sum()),
                label,
                packed,
            )
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    _require(selected is not None, "CTT V0 has no nontrivial logical candidate")
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
    config, config_sha256 = _load_config(config_path, registry)
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = _attempt022_uniform_seed(registry, code, config["cell"])
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < float(config["cell"]["p"])).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(bool(syndrome.any()), "CTT V0 planted syndrome unexpectedly vanishes")
    sampler = CollapsedTemperedTransitionConfig(
        p=config["cell"]["p"],
        burn_steps=config["resource"]["burn_steps"],
        measurement_steps=config["resource"]["measurement_steps"],
        num_levels=config["method"]["num_levels"],
        reversible_sweeps_per_level=config["method"]["reversible_sweeps_per_level"],
        block_size=config["method"]["block_size"],
        method_id=config["method"]["id"],
    )
    character_seed = derive_seed(
        config["character_seed_namespace"], registry["registry_sha256"],
        code["code_id"], "transport_characters",
    )
    characters = frozen_character_set(
        model.k, character_seed, config["num_nonbasis_characters"],
    )
    l_move, l_start = _select_low_energy_logical_move(epsilon, model, frame)
    mass = build_classical_coset_mass(H, config["cell"]["p"], engine="numba")
    context = {
        "registry": registry,
        "config": config,
        "config_sha256": config_sha256,
        "code": code,
        "H": np.ascontiguousarray(H, dtype=np.uint8),
        "model": model,
        "frame": frame,
        "uniform_seed": uniform_seed,
        "epsilon": epsilon,
        "syndrome": syndrome,
        "sampler": sampler,
        "characters": characters,
        "l_move": l_move,
        "l_start": l_start,
        "mass_sha256": _float64_big_endian_sha256(mass),
        "lambda_sha256": _float64_big_endian_sha256(sampler.lambda_values),
        "source_binding": _source_binding(),
    }
    if manifest is not None:
        _validate_manifest_context(manifest, context)
    return context


def _task_identity(context, init_family, trajectory_index):
    config = context["config"]
    seed_identity = CollapsedTemperedTransitionSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=context["config_sha256"],
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(config["cell"]),
        method_id=config["method"]["id"],
        resource_tier=config["resource"]["name"],
        init_family=init_family,
        trajectory_index=int(trajectory_index),
        trajectory_namespace=config["trajectory_namespace"],
    )
    return {
        "task_version": TASK_VERSION,
        "raw_version": RAW_VERSION,
        "method_id": config["method"]["id"],
        "resource": config["resource"],
        "init_family": init_family,
        "trajectory_index": int(trajectory_index),
        "cell": config["cell"],
        "sampler_config": context["sampler"].as_dict(),
        "seed_identity": seed_identity.as_dict(),
        "engine": "numba",
    }


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
        "uniform_seed": int(context["uniform_seed"]),
        "H_sha256": _array_sha256(context["H"]),
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "model_fingerprint": context["model"].fingerprint(),
        "logical_frame_fingerprint": context["frame"].fingerprint(),
        "character_sha256": context["characters"].character_sha256,
        "character_masks": [int(value) for value in context["characters"].masks],
        "l_start": context["l_start"],
        "classical_mass_sha256": context["mass_sha256"],
        "lambda_sha256": context["lambda_sha256"],
        "tasks": tasks,
    }


def _manifest_tasks(context):
    return [
        _task_identity(context, family, index)
        for family in context["config"]["init_families"]
        for index in range(context["config"]["trajectories_per_family"])
    ]


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    _require(not manifest_path.exists(), "CTT V0 manifest already exists")
    _require(not (output_root / "raw").exists(), "CTT V0 raw directory already exists")
    context = _context(registry_path, config_path)
    tasks = _manifest_tasks(context)
    core = _manifest_core(context, tasks)
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(manifest_path, manifest)
    return manifest_path


def _load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalTransportConflict(f"cannot load CTT V0 manifest: {exc}") from exc
    required_fields = {
        "manifest_version", "contract_version", "raw_version", "config",
        "config_sha256", "registry_sha256", "source_binding", "cell",
        "uniform_seed", "H_sha256", "syndrome_sha256", "model_fingerprint",
        "logical_frame_fingerprint", "character_sha256", "character_masks",
        "l_start", "classical_mass_sha256", "lambda_sha256", "tasks",
        "manifest_sha256",
    }
    _require(set(manifest) == required_fields, "CTT V0 manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "CTT V0 manifest SHA changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "CTT V0 manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "CTT V0 manifest contract changed")
    _require(manifest["raw_version"] == RAW_VERSION, "CTT V0 manifest raw version changed")
    _require(isinstance(manifest["tasks"], list) and len(manifest["tasks"]) == 24,
             "CTT V0 task count changed")
    _require(len({sha256_json(task) for task in manifest["tasks"]}) == len(manifest["tasks"]),
             "CTT V0 tasks are duplicated")
    return manifest


def _validate_manifest_context(manifest, context):
    core = _manifest_core(context, manifest["tasks"])
    expected = {**core, "manifest_sha256": sha256_json(core)}
    _require(manifest == expected, "CTT V0 manifest/context binding changed")


def _initial_state(context, task):
    family = task["init_family"]
    if family == "P":
        state = context["epsilon"].copy()
    elif family == "U":
        identity = CollapsedTemperedTransitionSeedIdentity(**task["seed_identity"])
        state = uniform_hard_coset_state(
            context["model"], context["syndrome"],
            identity.seed("initialize", "hard_coset"),
        )
    elif family == "L":
        state = context["epsilon"] ^ context["l_move"]
    else:  # Defensive because manifest validation should prevent this.
        raise LocalTransportConflict("CTT V0 task has an unknown initialization")
    residual = (
        context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ context["syndrome"]
    _require(not residual.any(), "CTT V0 initialization left the hard coset")
    if family == "L":
        _require(
            state_label(context["frame"], state)
            != state_label(context["frame"], context["epsilon"]),
            "CTT V0 L start lost its logical separation",
        )
        _require(
            int(state.sum()) == context["l_start"]["selected_absolute_weight"],
            "CTT V0 L start no longer matches its frozen target energy",
        )
    return np.ascontiguousarray(state, dtype=np.uint8)


def _execute_task(context, task):
    _require(task in _manifest_tasks(context), "CTT V0 task is not canonical")
    identity = CollapsedTemperedTransitionSeedIdentity(**task["seed_identity"])
    initial = _initial_state(context, task)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_collapsed_tempered_transition_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        context["sampler"], identity, initial, engine="numba",
    )
    core_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    _require(result["raw_version"] == COLLAPSED_TT_RAW_VERSION, "CTT V0 engine raw changed")
    _require(result["transition_kernel"] == COLLAPSED_TT_KERNEL, "CTT V0 kernel changed")
    _require(
        _scalar(result["mass_sha256"], "mass_sha256") == context["mass_sha256"],
        "CTT V0 mass table changed",
    )
    _require(
        _scalar(result["lambda_sha256"], "lambda_sha256") == context["lambda_sha256"],
        "CTT V0 lambda schedule changed",
    )
    _require(
        np.array_equal(np.asarray(result["counter_names"]), np.asarray(COLLAPSED_TT_COUNTER_NAMES)),
        "CTT V0 counter schema changed",
    )
    _require(not np.asarray(result["measurement_residual_weights"]).any(),
             "CTT V0 trajectory left the hard coset")
    for name in (
        "burn_tt_log_acceptance", "measurement_tt_log_acceptance",
        "burn_tt_accepted_b_bit_changes", "measurement_tt_accepted_b_bit_changes",
        "burn_tt_prior_refresh_bit_changes", "measurement_tt_prior_refresh_bit_changes",
    ):
        _require(np.all(np.isfinite(np.asarray(result[name]))), f"CTT V0 {name} is non-finite")
    return result, initial, core_seconds, wall_seconds


def _raw_payload(context, manifest, task, result, core_seconds, wall_seconds):
    arrays = {
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
        "trajectory_digest": np.array(_trajectory_digest(result)),
        "core_seconds": np.array(float(core_seconds), dtype=np.float64),
        "wall_seconds": np.array(float(wall_seconds), dtype=np.float64),
    }
    arrays.update({f"ctt_{name}": np.asarray(value) for name, value in result.items()})
    return arrays


def _raw_path(output_root, task):
    return Path(output_root) / "raw" / task["init_family"] / f"t{task['trajectory_index']:02d}.npz"


def _run_worker(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require(task in manifest["tasks"], "CTT V0 worker task is absent from manifest")
    output_path = _raw_path(output_root, task)
    _require(not output_path.exists(), "CTT V0 raw already exists")
    result, _, core_seconds, wall_seconds = _execute_task(context, task)
    atomic_npz(
        output_path,
        **_raw_payload(context, manifest, task, result, core_seconds, wall_seconds),
    )
    return {
        "task_fingerprint": sha256_json(task),
        "path": str(output_path),
        "wall_seconds": wall_seconds,
        "core_seconds": core_seconds,
    }


def run(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *, workers=1):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    _context(registry_path, config_path, manifest=manifest)
    _require(int(workers) > 0 and not isinstance(workers, bool), "CTT V0 workers invalid")
    for marker in ("RUNNING.json", "SUCCESS.json", "FAILED.json"):
        _require(not (output_root / marker).exists(), "CTT V0 run marker already exists")
    for task in manifest["tasks"]:
        _require(not _raw_path(output_root, task).exists(), "CTT V0 raw already exists")
    atomic_json(output_root / "RUNNING.json", {
        "stage": "run", "manifest_sha256": manifest["manifest_sha256"],
        "workers": int(workers),
    })
    try:
        values = (str(manifest_path), str(registry_path), str(config_path), str(output_root))
        if int(workers) == 1:
            completed = [_run_worker(*values, task) for task in manifest["tasks"]]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
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
        "task_fingerprints": sorted(value["task_fingerprint"] for value in completed),
    })
    return completed


def _load_raw(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            raw = {name: data[name].copy() for name in data.files}
    except Exception as exc:
        raise LocalTransportConflict(f"cannot read CTT V0 raw {path}: {exc}") from exc
    _require(not any(value.dtype.hasobject for value in raw.values()), "CTT V0 raw contains object")
    return raw


def _validate_raw_worker(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    path = _raw_path(output_root, task)
    raw = _load_raw(path)
    result, _, _, _ = _execute_task(context, task)
    core_seconds = float(_scalar(raw.get("core_seconds"), "core_seconds"))
    wall_seconds = float(_scalar(raw.get("wall_seconds"), "wall_seconds"))
    _require(math.isfinite(core_seconds) and core_seconds >= 0.0, "CTT V0 core timing invalid")
    _require(math.isfinite(wall_seconds) and wall_seconds >= 0.0, "CTT V0 wall timing invalid")
    expected = _raw_payload(context, manifest, task, result, core_seconds, wall_seconds)
    _require(set(raw) == set(expected), "CTT V0 raw schema changed")
    for name, value in expected.items():
        if name in {"core_seconds", "wall_seconds"}:
            continue
        _require(np.array_equal(raw[name], value), f"CTT V0 raw replay changed: {name}")
    states = unpack_states(raw["ctt_measurement_states_packed"], context["model"].num_qubits)
    residuals = (
        context["model"].H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(
        np.array_equal(
            residuals.sum(axis=1).astype(np.int32),
            raw["ctt_measurement_residual_weights"],
        ),
        "CTT V0 raw residual reconstruction changed",
    )
    _require(
        np.array_equal(states.sum(axis=1).astype(np.int32), raw["ctt_measurement_weights"]),
        "CTT V0 raw weight reconstruction changed",
    )
    labels = np.asarray(
        [state_label(context["frame"], state) for state in states], dtype=np.uint64,
    )
    _require(
        np.array_equal(labels, raw["ctt_measurement_labels"]),
        "CTT V0 raw label reconstruction changed",
    )
    _require(not raw["ctt_measurement_residual_weights"].any(), "CTT V0 raw leaves hard coset")
    counters = np.asarray(raw["ctt_measurement_tt_counters"], dtype=np.int64)
    _require(counters.shape == (len(COLLAPSED_TT_COUNTER_NAMES),), "CTT V0 counters shape changed")
    accepts = np.asarray(raw["ctt_measurement_tt_accepts"], dtype=np.uint8)
    b_changes = np.asarray(raw["ctt_measurement_tt_accepted_b_bit_changes"], dtype=np.int32)
    _require(
        int(counters[0]) == accepts.size and int(counters[1]) == int(accepts.sum()),
        "CTT V0 acceptance counters changed",
    )
    _require(
        int(counters[2]) == int(np.count_nonzero((accepts != 0) & (b_changes > 0))),
        "CTT V0 B-changing counters changed",
    )
    return {
        "task": task,
        "burn_label": int(_scalar(raw["ctt_burn_label"], "ctt_burn_label")),
        "measurement_labels": [int(value) for value in raw["ctt_measurement_labels"]],
        "measurement_tt_counters": [int(value) for value in counters],
        "wall_seconds": wall_seconds,
    }


def _leave_return(labels, masks):
    signs = character_values(labels, masks)
    origin = signs[0]
    left = np.zeros(signs.shape[1], dtype=bool)
    returned = np.zeros(signs.shape[1], dtype=bool)
    for values in signs[1:]:
        changed = values != origin
        returned |= left & ~changed
        left |= changed
    return returned


def _family_summary(rows, context):
    k = context["model"].k
    masks = context["characters"].masks
    records = []
    for row in rows:
        labels = np.asarray(row["measurement_labels"], dtype=np.uint64)
        previous = np.concatenate((np.asarray([row["burn_label"]], dtype=np.uint64), labels[:-1]))
        deltas = labels ^ previous
        changed = deltas != 0
        counters = np.asarray(row["measurement_tt_counters"], dtype=np.int64)
        records.append({
            "cross_label_changes": int(changed.sum()),
            "deltas": [int(value) for value in deltas[changed]],
            "leave_return": _leave_return(
                np.concatenate((np.asarray([row["burn_label"]], dtype=np.uint64), labels)),
                masks,
            ).astype(np.uint8),
            "tt_counters": counters,
            "wall_seconds": float(row["wall_seconds"]),
        })
    all_deltas = [np.uint64(delta) for record in records for delta in record["deltas"]]
    returns = np.asarray([record["leave_return"] for record in records], dtype=np.uint8)
    total_counters = (
        np.sum(np.asarray([record["tt_counters"] for record in records], dtype=np.int64), axis=0)
        if records else np.zeros(len(COLLAPSED_TT_COUNTER_NAMES), dtype=np.int64)
    )
    gates = context["config"]["gates"]
    summary = {
        "chain_count": len(records),
        "measurement_cross_label_changes": int(sum(record["cross_label_changes"] for record in records)),
        "chains_with_eight_measurement_cross_label_changes": int(sum(
            record["cross_label_changes"] >= 8 for record in records
        )),
        "measurement_label_delta_rank": _signature_rank_masks(all_deltas, k),
        "basis_characters_with_leave_return": int(np.count_nonzero(returns[:, :k].sum(axis=0))) if records else 0,
        "nonbasis_characters_with_leave_return": int(np.count_nonzero(returns[:, k:].sum(axis=0))) if records else 0,
        "median_wall_seconds": float(np.median([record["wall_seconds"] for record in records])) if records else None,
        "ctt_path_diagnostic": {
            "attempts": int(total_counters[0]),
            "accepts": int(total_counters[1]),
            "accepted_b_changing_proposals": int(total_counters[2]),
            "prior_refresh_bit_changes": int(total_counters[3]),
            "reversible_block_updates": int(total_counters[4]),
            "reversible_block_changes": int(total_counters[5]),
            "is_not_a_transport_gate": True,
        },
    }
    summary["passes_transport_gate"] = bool(
        summary["chain_count"] == context["config"]["trajectories_per_family"]
        and summary["measurement_cross_label_changes"] >= gates["minimum_measurement_cross_label_changes_per_family"]
        and summary["chains_with_eight_measurement_cross_label_changes"] >= gates["minimum_chains_with_eight_measurement_cross_label_changes_per_family"]
        and summary["measurement_label_delta_rank"] >= gates["minimum_measurement_label_delta_rank_per_family"]
        and (
            not gates["require_all_basis_character_leave_returns"]
            or summary["basis_characters_with_leave_return"] == k
        )
        and (
            not gates["require_all_nonbasis_character_leave_returns"]
            or summary["nonbasis_characters_with_leave_return"] == context["config"]["num_nonbasis_characters"]
        )
    )
    return summary


def analyze(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *, workers=1):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "SUCCESS.json").is_file(), "CTT V0 raw stage did not succeed")
    _require(not (output_root / "FAILED.json").exists(), "CTT V0 raw stage failed")
    _require(not (output_root / "REPORT.json").exists(), "CTT V0 report already exists")
    _require(int(workers) > 0 and not isinstance(workers, bool), "CTT V0 workers invalid")
    for task in manifest["tasks"]:
        _require(_raw_path(output_root, task).is_file(), "CTT V0 raw is missing")
    values = (str(manifest_path), str(registry_path), str(config_path), str(output_root))
    if int(workers) == 1:
        rows = [_validate_raw_worker(*values, task) for task in manifest["tasks"]]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_validate_raw_worker, *values, task) for task in manifest["tasks"]]
            rows = [future.result() for future in futures]
    summaries = {
        family: _family_summary(
            [row for row in rows if row["task"]["init_family"] == family], context,
        )
        for family in context["config"]["init_families"]
    }
    status = (
        "LOCAL_LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
        if all(value["passes_transport_gate"] for value in summaries.values())
        else "LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE"
    )
    core = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": status,
        "scope": "transport_only_not_qtop_or_formal_authorization",
        "ctt_path_diagnostics_do_not_substitute_for_label_transport": True,
        "families": summaries,
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
        result = {"manifest": str(prepare(args.output, args.registry, args.config))}
    elif args.command == "run":
        result = run(args.output, args.registry, args.config, workers=args.workers)
    else:
        result = analyze(args.output, args.registry, args.config, workers=args.workers)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
