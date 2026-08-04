"""Fail-closed local transport preflight for catalog-free MLB8-J16.

This runner is intentionally diagnostic-only.  It creates immutable local raw
for one pre-registered hard m8 cell, then independently replays every raw
trajectory before emitting a transport-only report.  It never estimates
q_top or authorizes a formal exp102 stage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
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
    canonical_global_trajectory_digest,
    character_values,
    frozen_character_set,
    reduce_logical_basis,
    state_label,
    uniform_hard_coset_state,
    unpack_states,
)
from data.expander_code.exp102.exp102_pipeline.q0_multilogical_blocks import (
    MULTILOGICAL_KERNEL_MODE,
    MultiLogicalBlockConfig,
    build_multilogical_empty_catalog,
    build_multilogical_blocks,
    run_multilogical_block_trajectory,
    validate_multilogical_empty_catalog,
    validate_multilogical_blocks,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_mlb8.catalogless.v0"
CONFIG_VERSION = "exp102.q0_mlb8.catalogless.v0.config.v1"
MANIFEST_VERSION = "exp102.q0_mlb8.catalogless.v0.manifest.v1"
TASK_VERSION = "exp102.q0_mlb8.catalogless.v0.tasks.v1"
RAW_VERSION = "exp102.q0_mlb8.catalogless.v0.raw.v1"
REPORT_VERSION = "exp102.q0_mlb8.catalogless.v0.report.v1"
SOURCE_FILES = (
    "data/expander_code/exp102/exp102_pipeline/q0_global.py",
    "data/expander_code/exp102/exp102_pipeline/q0_multilogical_blocks.py",
    "data/expander_code/exp102/validation/017_q0_mlb8_catalogless_v0_20260723/run_local_transport.py",
)
ROOT = Path("data/expander_code/exp102")
DEFAULT_REGISTRY = ROOT / "registry/registry.json"
DEFAULT_CONFIG = ROOT / "config/q0_mlb8.catalogless.v0.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_m8_transport_catalogless_v1"


class LocalTransportConflict(RuntimeError):
    """A local diagnostic artifact cannot be trusted or reused."""


@dataclass(frozen=True)
class MlbSeedIdentity:
    """Separate seed namespace that makes the legal L family explicit."""

    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    resource_tier: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str

    def __post_init__(self):
        _require(len(self.source_commit) == 40 and all(
            char in "0123456789abcdef" for char in self.source_commit
        ), "MLB seed source commit is invalid")
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            value = getattr(self, name)
            _require(len(value) == 64 and all(
                char in "0123456789abcdef" for char in value
            ), f"MLB seed {name} is invalid")
        _require(self.init_family in ("P", "U", "L"), "MLB seed family is invalid")
        _require(not isinstance(self.trajectory_index, bool) and int(self.trajectory_index) >= 0,
                 "MLB seed trajectory index is invalid")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            "q0_mlb8_catalogless_v0",
            self.source_commit,
            self.config_sha256,
            self.registry_sha256,
            self.cell_fingerprint,
            self.method_id,
            self.resource_tier,
            self.init_family,
            int(self.trajectory_index),
            self.trajectory_namespace,
            str(stage),
            str(role),
            int(index),
        )

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": self.method_id,
            "resource_tier": self.resource_tier,
            "init_family": self.init_family,
            "trajectory_index": int(self.trajectory_index),
            "trajectory_namespace": self.trajectory_namespace,
        }


def _scalar(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise LocalTransportConflict(f"{name} must be scalar")
    return array.item()


def _require(condition, message):
    if not condition:
        raise LocalTransportConflict(message)


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise LocalTransportConflict("local source commit is invalid")
    files = {path: sha256_file(path) for path in SOURCE_FILES}
    return {
        "source_commit": source_commit,
        "source_files": files,
        "source_binding_sha256": sha256_json({
            "source_commit": source_commit, "source_files": files,
        }),
    }


def _load_config(config_path, registry):
    config_path = Path(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalTransportConflict(f"cannot load MLB V0 config: {exc}") from exc
    expected = {
        "character_seed_namespace", "cell", "config_version", "contract_version",
        "gates", "init_families", "l_start_rule", "method",
        "num_nonbasis_characters", "raw_version", "registry_sha256", "resource",
        "scope", "trajectories_per_family", "trajectory_namespace",
    }
    _require(set(config) == expected, "MLB V0 config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "MLB V0 contract changed")
    _require(config["config_version"] == CONFIG_VERSION, "MLB V0 config version changed")
    _require(config["raw_version"] == RAW_VERSION, "MLB V0 raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "MLB V0 registry changed")
    _require(config["init_families"] == ["P", "U", "L"], "MLB V0 starts changed")
    _require(config["l_start_rule"] == "planted_xor_all_reduced_logical_directions.v1",
             "MLB V0 L start changed")
    _require(config["trajectories_per_family"] == 8, "MLB V0 trajectory count changed")
    _require(config["num_nonbasis_characters"] == 64, "MLB V0 character count changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_transport_preflight_only",
        "remote_authorization": False,
    }, "MLB V0 scope changed")
    cell = config["cell"]
    _require(cell == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "MLB V0 cell changed")
    resource = config["resource"]
    _require(resource == {"burn_sweeps": 512, "measurement_sweeps": 4096, "name": "V0"},
             "MLB V0 resource changed")
    method = config["method"]
    _require(method == {
        "block_size": 16, "id": "MLB8-J16",
        "kernel_mode": MULTILOGICAL_KERNEL_MODE,
        "logical_catalog_mode": "none", "logicals_per_block": 8,
    }, "MLB V0 method changed")
    gates = config["gates"]
    _require(gates == {
        "minimum_basis_character_leave_return_chains_per_family": 1,
        "minimum_chains_with_eight_measurement_cross_label_changes_per_family": 6,
        "minimum_measurement_accepted_cross_label_changes_per_family": 128,
        "minimum_measurement_label_delta_rank_per_family": 64,
        "minimum_nonbasis_character_leave_return_chains_per_family": 1,
    }, "MLB V0 gates changed")
    for name in ("character_seed_namespace", "trajectory_namespace"):
        _require(isinstance(config[name], str) and config[name], f"MLB V0 {name} invalid")
    return config, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "MLB V0 disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


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
    _require(bool(syndrome.any()), "MLB V0 planted syndrome unexpectedly vanishes")
    sampler = MultiLogicalBlockConfig(
        p=config["cell"]["p"],
        burn_sweeps=config["resource"]["burn_sweeps"],
        measurement_sweeps=config["resource"]["measurement_sweeps"],
        logicals_per_block=config["method"]["logicals_per_block"],
        block_size=config["method"]["block_size"],
        logical_catalog_mode=config["method"]["logical_catalog_mode"],
        method_id=config["method"]["id"],
    )
    catalog = build_multilogical_empty_catalog(model, frame)
    blocks = build_multilogical_blocks(
        model, frame, logicals_per_block=sampler.logicals_per_block,
        block_size=sampler.block_size,
    )
    validate_multilogical_empty_catalog(model, frame, catalog)
    validate_multilogical_blocks(
        model, frame, blocks, logicals_per_block=sampler.logicals_per_block,
        block_size=sampler.block_size,
    )
    character_seed = derive_seed(
        config["character_seed_namespace"], registry["registry_sha256"],
        code["code_id"], "transport_characters",
    )
    characters = frozen_character_set(
        model.k, character_seed, config["num_nonbasis_characters"],
    )
    reduced = reduce_logical_basis(model.logical_move_basis)
    l_move = np.bitwise_xor.reduce(reduced, axis=0).astype(np.uint8, copy=False)
    _require(bool(state_label(frame, l_move)), "MLB V0 L move lost its logical signature")
    _require(not (model.H_check.astype(np.int64) @ l_move.astype(np.int64) % 2).any(),
             "MLB V0 L move left the kernel")
    context = {
        "registry": registry,
        "config": config,
        "config_sha256": config_sha256,
        "code": code,
        "model": model,
        "frame": frame,
        "uniform_seed": uniform_seed,
        "epsilon": epsilon,
        "syndrome": syndrome,
        "sampler": sampler,
        "catalog": catalog,
        "blocks": blocks,
        "characters": characters,
        "l_move": l_move,
        "source_binding": _source_binding(),
    }
    if manifest is not None:
        _validate_manifest_context(manifest, context)
    return context


def _manifest_core(context, tasks):
    model = context["model"]
    frame = context["frame"]
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
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "model_fingerprint": model.fingerprint(),
        "logical_frame_fingerprint": frame.fingerprint(),
        "empty_catalog_sha256": context["catalog"].catalog_sha256,
        "joint_sha256": context["blocks"].joint_sha256,
        "character_sha256": context["characters"].character_sha256,
        "character_masks": [int(value) for value in context["characters"].masks],
        "l_move_sha256": _array_sha256(context["l_move"]),
        "tasks": tasks,
    }


def _task_identity(context, init_family, trajectory_index):
    config = context["config"]
    seed_identity = MlbSeedIdentity(
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


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    _require(not manifest_path.exists(), "MLB V0 manifest already exists")
    _require(not (output_root / "raw").exists(), "MLB V0 raw directory already exists")
    context = _context(registry_path, config_path)
    tasks = [
        _task_identity(context, family, index)
        for family in context["config"]["init_families"]
        for index in range(context["config"]["trajectories_per_family"])
    ]
    core = _manifest_core(context, tasks)
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(manifest_path, manifest)
    return manifest_path


def _load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalTransportConflict(f"cannot load MLB V0 manifest: {exc}") from exc
    required_fields = {
        "manifest_version", "contract_version", "raw_version", "config",
        "config_sha256", "registry_sha256", "source_binding", "cell",
        "uniform_seed", "syndrome_sha256", "model_fingerprint",
        "logical_frame_fingerprint", "empty_catalog_sha256", "joint_sha256",
        "character_sha256", "character_masks", "l_move_sha256", "tasks",
        "manifest_sha256",
    }
    _require(set(manifest) == required_fields, "MLB V0 manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "MLB V0 manifest SHA changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "MLB V0 manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "MLB V0 manifest contract changed")
    _require(manifest["raw_version"] == RAW_VERSION, "MLB V0 manifest raw version changed")
    _require(isinstance(manifest["tasks"], list) and len(manifest["tasks"]) == 24,
             "MLB V0 task count changed")
    _require(len({sha256_json(task) for task in manifest["tasks"]}) == len(manifest["tasks"]),
             "MLB V0 tasks are duplicated")
    return manifest


def _validate_manifest_context(manifest, context):
    core = _manifest_core(context, manifest["tasks"])
    expected = {**core, "manifest_sha256": sha256_json(core)}
    _require(manifest == expected, "MLB V0 manifest/context binding changed")


def _initial_state(context, task):
    family = task["init_family"]
    if family == "P":
        state = context["epsilon"].copy()
    elif family == "U":
        identity = MlbSeedIdentity(**task["seed_identity"])
        state = uniform_hard_coset_state(
            context["model"], context["syndrome"],
            identity.seed("initialize", "hard_coset"),
        )
    elif family == "L":
        state = context["epsilon"] ^ context["l_move"]
    else:  # Defensive because manifest validation should prevent this.
        raise LocalTransportConflict("MLB V0 task has an unknown initialization")
    residual = (
        context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ context["syndrome"]
    _require(not residual.any(), "MLB V0 initialization left the hard coset")
    if family == "L":
        _require(state_label(context["frame"], state) != state_label(context["frame"], context["epsilon"]),
                 "MLB V0 L start lost its logical separation")
    return np.ascontiguousarray(state, dtype=np.uint8)


def _execute_task(context, task):
    _require(task in _manifest_tasks(context), "MLB V0 task is not canonical")
    identity = MlbSeedIdentity(**task["seed_identity"])
    initial = _initial_state(context, task)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_multilogical_block_trajectory(
        context["model"], context["frame"], context["syndrome"], context["sampler"],
        identity, initial, engine="numba", catalog=context["catalog"],
        blocks=context["blocks"],
    )
    core_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    _require(not np.asarray(result["measurement_residual_weights"]).any(),
             "MLB V0 trajectory left the hard coset")
    _require(not np.asarray(result["burn_counters"])[2:4].any(),
             "MLB V0 burn used a logical catalog")
    _require(not np.asarray(result["measurement_counters"])[2:4].any(),
             "MLB V0 measurement used a logical catalog")
    return result, initial, core_seconds, wall_seconds


def _manifest_tasks(context):
    return [
        _task_identity(context, family, index)
        for family in context["config"]["init_families"]
        for index in range(context["config"]["trajectories_per_family"])
    ]


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
        "empty_catalog_sha256": np.array(context["catalog"].catalog_sha256),
        "joint_sha256": np.array(context["blocks"].joint_sha256),
        "l_move_sha256": np.array(_array_sha256(context["l_move"])),
        "kernel_mode": np.array(MULTILOGICAL_KERNEL_MODE),
        "trajectory_digest": np.array(canonical_global_trajectory_digest(result)),
        "core_seconds": np.array(float(core_seconds), dtype=np.float64),
        "wall_seconds": np.array(float(wall_seconds), dtype=np.float64),
    }
    arrays.update({name: np.asarray(value) for name, value in result.items()})
    return arrays


def _raw_path(output_root, task):
    return Path(output_root) / "raw" / task["init_family"] / f"t{task['trajectory_index']:02d}.npz"


def _run_worker(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require(task in manifest["tasks"], "MLB V0 worker task is absent from manifest")
    output_path = _raw_path(output_root, task)
    _require(not output_path.exists(), "MLB V0 raw already exists")
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
    _require(int(workers) > 0 and not isinstance(workers, bool), "MLB V0 workers invalid")
    for marker in ("RUNNING.json", "SUCCESS.json", "FAILED.json"):
        _require(not (output_root / marker).exists(), "MLB V0 run marker already exists")
    for task in manifest["tasks"]:
        _require(not _raw_path(output_root, task).exists(), "MLB V0 raw already exists")
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
            result = {name: data[name].copy() for name in data.files}
    except Exception as exc:
        raise LocalTransportConflict(f"cannot read MLB V0 raw {path}: {exc}") from exc
    _require(not any(value.dtype.hasobject for value in result.values()), "MLB V0 raw contains object")
    return result


def _validate_raw_worker(manifest_path, registry_path, config_path, output_root, task):
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    path = _raw_path(output_root, task)
    raw = _load_raw(path)
    result, _, _, _ = _execute_task(context, task)
    core_seconds = float(_scalar(raw.get("core_seconds"), "core_seconds"))
    wall_seconds = float(_scalar(raw.get("wall_seconds"), "wall_seconds"))
    _require(math.isfinite(core_seconds) and core_seconds >= 0.0, "MLB V0 raw core timing invalid")
    _require(math.isfinite(wall_seconds) and wall_seconds >= 0.0, "MLB V0 raw wall timing invalid")
    expected = _raw_payload(context, manifest, task, result, core_seconds, wall_seconds)
    _require(set(raw) == set(expected), "MLB V0 raw schema changed")
    for name, value in expected.items():
        if name in {"core_seconds", "wall_seconds"}:
            continue
        _require(np.array_equal(raw[name], value), f"MLB V0 raw replay changed: {name}")
    states = unpack_states(raw["measurement_states_packed"], context["model"].num_qubits)
    residuals = (
        context["model"].H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(np.array_equal(residuals.sum(axis=1).astype(np.int32), raw["measurement_residual_weights"]),
             "MLB V0 raw residual reconstruction changed")
    _require(np.array_equal(states.sum(axis=1).astype(np.int32), raw["measurement_weights"]),
             "MLB V0 raw weight reconstruction changed")
    labels = np.asarray([state_label(context["frame"], state) for state in states], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "MLB V0 raw label reconstruction changed")
    _require(not raw["measurement_residual_weights"].any(), "MLB V0 raw leaves hard coset")
    _require(not raw["burn_counters"][2:4].any() and not raw["measurement_counters"][2:4].any(),
             "MLB V0 raw used a logical catalog")
    return {
        "task": task,
        "burn_label": int(_scalar(raw["burn_label"], "burn_label")),
        "measurement_labels": [int(value) for value in raw["measurement_labels"]],
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
        records.append({
            "cross_label_changes": int(changed.sum()),
            "deltas": [int(value) for value in deltas[changed]],
            "leave_return": _leave_return(
                np.concatenate((np.asarray([row["burn_label"]], dtype=np.uint64), labels)), masks,
            ).astype(np.uint8),
            "wall_seconds": float(row["wall_seconds"]),
        })
    all_deltas = [np.uint64(delta) for record in records for delta in record["deltas"]]
    returns = np.asarray([record["leave_return"] for record in records], dtype=np.uint8)
    gates = context["config"]["gates"]
    summary = {
        "chain_count": len(records),
        "measurement_cross_label_changes": int(sum(record["cross_label_changes"] for record in records)),
        "chains_with_eight_measurement_cross_label_changes": int(sum(
            record["cross_label_changes"] >= 8 for record in records
        )),
        "measurement_label_delta_rank": _signature_rank_masks(all_deltas, k),
        "basis_characters_with_leave_return": int(np.count_nonzero(returns[:, :k].sum(axis=0))) if len(records) else 0,
        "nonbasis_characters_with_leave_return": int(np.count_nonzero(returns[:, k:].sum(axis=0))) if len(records) else 0,
        "median_wall_seconds": float(np.median([record["wall_seconds"] for record in records])) if records else None,
    }
    summary["passes_transport_gate"] = bool(
        summary["chain_count"] == context["config"]["trajectories_per_family"]
        and summary["measurement_cross_label_changes"] >= gates["minimum_measurement_accepted_cross_label_changes_per_family"]
        and summary["chains_with_eight_measurement_cross_label_changes"] >= gates["minimum_chains_with_eight_measurement_cross_label_changes_per_family"]
        and summary["measurement_label_delta_rank"] >= gates["minimum_measurement_label_delta_rank_per_family"]
        and summary["basis_characters_with_leave_return"] == k
        and summary["nonbasis_characters_with_leave_return"] == context["config"]["num_nonbasis_characters"]
    )
    return summary


def analyze(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG, *, workers=1):
    output_root = Path(output_root)
    manifest_path = output_root / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "SUCCESS.json").is_file(), "MLB V0 raw stage did not succeed")
    _require(not (output_root / "FAILED.json").exists(), "MLB V0 raw stage failed")
    _require(not (output_root / "REPORT.json").exists(), "MLB V0 report already exists")
    for task in manifest["tasks"]:
        _require(_raw_path(output_root, task).is_file(), "MLB V0 raw is missing")
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
