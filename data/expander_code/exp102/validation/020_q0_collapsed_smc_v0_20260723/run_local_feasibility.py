"""Fail-closed local feasibility test for collapsed q=0 HGP SMC.

The test asks one narrow question: does a frozen exact-base population bridge
avoid immediate incremental-weight and root-genealogy collapse on the hardest
m8 cell?  It does not calculate q_top, reconstruct posterior samples, or
authorize any remote or formal exp102 work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    load_npz_no_pickle,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed_smc import (
    BASE_FAMILIES,
    COLLAPSED_SMC_KERNEL,
    COLLAPSED_SMC_RAW_VERSION,
    CollapsedSmcConfig,
    CollapsedSmcSeedIdentity,
    a_syndromes_from_b,
    collapsed_lambda_schedule,
    collapsed_log_likelihood,
    run_collapsed_smc_population,
    systematic_resampling,
    _float64_big_endian_sha256,
    _normalized_incremental_weights,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_collapsed_smc.v0"
CONFIG_VERSION = "exp102.q0_hgp_collapsed_smc.v0.config.v1"
MANIFEST_VERSION = "exp102.q0_hgp_collapsed_smc.v0.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_collapsed_smc.v0.tasks.v1"
RAW_VERSION = "exp102.q0_hgp_collapsed_smc.raw.v0"
REPLAY_VERSION = "exp102.q0_hgp_collapsed_smc.v0.replay.v1"
REPORT_VERSION = "exp102.q0_hgp_collapsed_smc.v0.report.v1"
ROOT = Path("data/expander_code/exp102")
DEFAULT_REGISTRY = ROOT / "registry/registry.json"
DEFAULT_CONFIG = ROOT / "config/q0_hgp_collapsed_smc.v0.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_m8_smc_v0"
SOURCE_FILES = (
    "data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed.py",
    "data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed_smc.py",
    "data/expander_code/exp102/exp102_pipeline/q0_global.py",
    "data/expander_code/exp102/exp102_pipeline/registry.py",
    "data/expander_code/exp102/exp102_pipeline/seeds.py",
    "data/expander_code/exp102/exp102_pipeline/worker.py",
    "data/expander_code/exp102/validation/020_q0_collapsed_smc_v0_20260723/run_local_feasibility.py",
)
RESULT_FIELDS = {
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "b_columns_by_stage", "log_likelihood_by_stage",
    "roots_by_stage", "normalized_incremental_weights", "parent_indices",
    "resample_offsets", "conditional_ess", "max_normalized_weight",
    "log_normalizer_increments", "mutation_block_changes",
    "mutation_block_attempts_per_particle", "final_root_counts",
    "final_root_family_ess", "final_distinct_roots", "final_max_root_fraction",
    "final_b_bit_weights", "final_collapsed_log_target", "lambda_values",
    "lambda_sha256", "mass_sha256", "kernel", "engine",
}


class LocalSmcConflict(RuntimeError):
    """The local diagnostic cannot be trusted or used as evidence."""


def _require(condition, message):
    if not condition:
        raise LocalSmcConflict(message)


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


def _trajectory_digest(result):
    """Hash all deterministic engine fields, excluding wrapper timing only."""
    _require(set(result) == RESULT_FIELDS, "collapsed SMC result fields changed")
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_collapsed_smc.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        _require(not value.dtype.hasobject, f"SMC result {name} has object dtype")
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
        len(source_commit) == 40 and all(value in "0123456789abcdef" for value in source_commit),
        "SMC source commit is invalid",
    )
    source_files = {path: sha256_file(path) for path in SOURCE_FILES}
    payload = {"source_commit": source_commit, "source_files": source_files}
    return {**payload, "source_binding_sha256": sha256_json(payload)}


def _load_config(path, registry):
    try:
        config = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalSmcConflict(f"cannot load collapsed SMC config: {exc}") from exc
    expected_fields = {
        "cell", "config_version", "contract_version", "gates", "method",
        "population", "raw_version", "registry_sha256", "scope", "trajectory_namespace",
    }
    _require(set(config) == expected_fields, "collapsed SMC config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "collapsed SMC contract changed")
    _require(config["config_version"] == CONFIG_VERSION, "collapsed SMC config version changed")
    _require(config["raw_version"] == RAW_VERSION, "collapsed SMC raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "collapsed SMC registry changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "collapsed SMC cell changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_collapsed_smc_weight_and_genealogy_feasibility_only",
        "remote_authorization": False,
    }, "collapsed SMC scope changed")
    _require(config["population"] == {
        "base_families": ["column_major", "row_major"],
        "num_particles": 128,
        "populations_per_base_family": 4,
    }, "collapsed SMC population layout changed")
    _require(config["method"].keys() == {
        "block_size", "id", "kernel", "lambda_generation", "lambda_sha256",
        "lambda_values", "mutation_sweeps", "prior_endpoint", "resampling",
    }, "collapsed SMC method fields changed")
    _require(config["method"]["id"] == "CSMC64-B8-S1-N128", "collapsed SMC method ID changed")
    _require(config["method"]["kernel"] == COLLAPSED_SMC_KERNEL, "collapsed SMC kernel changed")
    _require(config["method"]["lambda_generation"] == "hp64_quadratic_index_v1",
             "collapsed SMC lambda generator changed")
    _require(config["method"]["prior_endpoint"] == "exact_iid_bernoulli_B",
             "collapsed SMC prior endpoint changed")
    _require(config["method"]["resampling"] == "systematic_every_nonzero_lambda_stage",
             "collapsed SMC resampling changed")
    expected_lambda = collapsed_lambda_schedule(64)
    _require(tuple(config["method"]["lambda_values"]) == expected_lambda,
             "collapsed SMC lambda values changed")
    _require(config["method"]["lambda_sha256"] == _float64_big_endian_sha256(expected_lambda),
             "collapsed SMC lambda SHA changed")
    sampler = CollapsedSmcConfig(
        p=config["cell"]["p"],
        num_particles=config["population"]["num_particles"],
        lambda_values=tuple(config["method"]["lambda_values"]),
        mutation_sweeps=config["method"]["mutation_sweeps"],
        block_size=config["method"]["block_size"],
        method_id=config["method"]["id"],
    )
    _require(config["gates"] == {
        "max_stage_normalized_weight": 0.1,
        "min_final_distinct_roots": 32,
        "min_final_root_family_ess": 16.0,
        "min_stage_conditional_ess_fraction": 0.5,
        "require_all_populations_to_pass": True,
        "max_final_root_fraction": 0.2,
    }, "collapsed SMC gates changed")
    _require(isinstance(config["trajectory_namespace"], str) and config["trajectory_namespace"],
             "collapsed SMC trajectory namespace is invalid")
    return config, sampler, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "collapsed SMC disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22",
        registry["registry_sha256"], code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


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
    _require(bool(syndrome.any()), "collapsed SMC planted syndrome unexpectedly vanishes")
    mass = build_classical_coset_mass(H, config["cell"]["p"], engine="numba")
    result = {
        "registry": registry,
        "config": config,
        "sampler": sampler,
        "config_sha256": config_sha256,
        "code": code,
        "H": np.ascontiguousarray(H, dtype=np.uint8),
        "model": model,
        "frame": frame,
        "uniform_seed": uniform_seed,
        "syndrome": syndrome,
        "mass_sha256": _float64_big_endian_sha256(mass),
        "source_binding": _source_binding(),
    }
    if manifest is not None:
        _validate_manifest_context(manifest, result)
    return result


def _task_identity(context, base_family, population_index):
    identity = CollapsedSmcSeedIdentity(
        source_commit=context["source_binding"]["source_commit"],
        config_sha256=context["config_sha256"],
        registry_sha256=context["registry"]["registry_sha256"],
        cell_fingerprint=sha256_json(context["config"]["cell"]),
        method_id=context["sampler"].method_id,
        base_family=base_family,
        population_index=population_index,
        trajectory_namespace=context["config"]["trajectory_namespace"],
    )
    return {
        "task_version": TASK_VERSION,
        "raw_version": RAW_VERSION,
        "cell": context["config"]["cell"],
        "method_id": context["sampler"].method_id,
        "base_family": base_family,
        "population_index": int(population_index),
        "seed_identity": identity.as_dict(),
        "engine": "numba",
    }


def _manifest_tasks(context):
    return [
        _task_identity(context, base_family, population_index)
        for base_family in context["config"]["population"]["base_families"]
        for population_index in range(context["config"]["population"]["populations_per_base_family"])
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
        "uniform_seed": int(context["uniform_seed"]),
        "H_sha256": _array_sha256(context["H"]),
        "syndrome_sha256": _array_sha256(context["syndrome"]),
        "model_fingerprint": context["model"].fingerprint(),
        "logical_frame_fingerprint": context["frame"].fingerprint(),
        "classical_mass_sha256": context["mass_sha256"],
        "tasks": tasks,
    }


def _load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalSmcConflict(f"cannot load collapsed SMC manifest: {exc}") from exc
    expected_fields = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "model_fingerprint", "logical_frame_fingerprint",
        "classical_mass_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == expected_fields, "collapsed SMC manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "collapsed SMC manifest SHA changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "collapsed SMC manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "collapsed SMC manifest contract changed")
    _require(manifest["raw_version"] == RAW_VERSION, "collapsed SMC manifest raw version changed")
    _require(isinstance(manifest["tasks"], list) and len(manifest["tasks"]) == 8,
             "collapsed SMC task count changed")
    _require(len({sha256_json(task) for task in manifest["tasks"]}) == 8,
             "collapsed SMC tasks are duplicated")
    return manifest


def _validate_manifest_context(manifest, context):
    core = _manifest_core(context, manifest["tasks"])
    expected = {**core, "manifest_sha256": sha256_json(core)}
    _require(manifest == expected, "collapsed SMC manifest/context binding changed")


def _raw_path(output_root, task):
    return (
        Path(output_root) / "raw" / task["base_family"]
        / f"p{int(task['population_index']):02d}.npz"
    )


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not (output_root / "MANIFEST.json").exists(), "collapsed SMC manifest already exists")
    _require(not (output_root / "raw").exists(), "collapsed SMC raw directory already exists")
    context = _context(registry_path, config_path)
    tasks = _manifest_tasks(context)
    core = _manifest_core(context, tasks)
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(output_root / "MANIFEST.json", manifest)
    return output_root / "MANIFEST.json"


def _execute_task(context, task):
    _require(task in _manifest_tasks(context), "collapsed SMC task is not canonical")
    identity = CollapsedSmcSeedIdentity(**task["seed_identity"])
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_collapsed_smc_population(
        context["model"], context["frame"], context["H"], context["syndrome"],
        context["sampler"], identity, engine=task["engine"],
    )
    core_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    _require(result["raw_version"] == COLLAPSED_SMC_RAW_VERSION, "collapsed SMC raw changed")
    _require(result["kernel"] == COLLAPSED_SMC_KERNEL, "collapsed SMC kernel changed")
    _require(_scalar(result["mass_sha256"], "mass_sha256") == context["mass_sha256"],
             "collapsed SMC mass table changed")
    _require(_scalar(result["lambda_sha256"], "lambda_sha256")
             == context["config"]["method"]["lambda_sha256"], "collapsed SMC lambda changed")
    _require(np.all(np.isfinite(np.asarray(result["conditional_ess"], dtype=np.float64))),
             "collapsed SMC CESS is non-finite")
    return result, core_seconds, wall_seconds


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
        "trajectory_digest": np.array(_trajectory_digest(result)),
        "core_seconds": np.array(float(core_seconds), dtype=np.float64),
        "wall_seconds": np.array(float(wall_seconds), dtype=np.float64),
    }
    arrays.update({f"smc_{name}": np.asarray(value) for name, value in result.items()})
    return arrays


def run(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    for marker in ("RUNNING.json", "SUCCESS.json", "FAILED.json"):
        _require(not (output_root / marker).exists(), "collapsed SMC run marker already exists")
    for task in manifest["tasks"]:
        _require(not _raw_path(output_root, task).exists(), "collapsed SMC raw already exists")
    atomic_json(output_root / "RUNNING.json", {
        "stage": "run", "manifest_sha256": manifest["manifest_sha256"], "workers": 1,
    })
    completed = []
    try:
        for task in manifest["tasks"]:
            result, core_seconds, wall_seconds = _execute_task(context, task)
            output = _raw_path(output_root, task)
            atomic_npz(
                output,
                **_raw_payload(context, manifest, task, result, core_seconds, wall_seconds),
            )
            completed.append({
                "task_fingerprint": sha256_json(task),
                "path": str(output), "core_seconds": core_seconds, "wall_seconds": wall_seconds,
            })
    except Exception as exc:
        atomic_json(output_root / "FAILED.json", {
            "stage": "run", "manifest_sha256": manifest["manifest_sha256"],
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    atomic_json(output_root / "RUN_COMPLETE.json", {
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": completed,
    })


def _raw_result(raw):
    return {name: np.asarray(raw[f"smc_{name}"]) for name in RESULT_FIELDS}


def replay(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "collapsed SMC run is incomplete")
    _require(not (output_root / "REPLAY.json").exists(), "collapsed SMC replay already exists")
    values = []
    for task in manifest["tasks"]:
        path = _raw_path(output_root, task)
        _require(path.is_file(), "collapsed SMC raw is missing")
        with load_npz_no_pickle(path) as raw:
            result = _raw_result(raw)
            expected_digest = str(_scalar(raw["trajectory_digest"], "trajectory_digest"))
        replayed, _, _ = _execute_task(context, task)
        actual_digest = _trajectory_digest(replayed)
        _require(actual_digest == expected_digest, "collapsed SMC deterministic replay failed")
        values.append({"task_fingerprint": sha256_json(task), "trajectory_digest": actual_digest})
    core = {
        "replay_version": REPLAY_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": values,
    }
    atomic_json(output_root / "REPLAY.json", {**core, "replay_sha256": sha256_json(core)})


def _audit_raw(context, manifest, task, path):
    expected_top = {
        "raw_version", "contract_version", "manifest_sha256", "task_fingerprint", "task_json",
        "source_binding_json", "config_sha256", "registry_sha256", "uniform_seed", "syndrome",
        "trajectory_digest", "core_seconds", "wall_seconds",
        *(f"smc_{name}" for name in RESULT_FIELDS),
    }
    with load_npz_no_pickle(path) as raw:
        _require(set(raw.files) == expected_top, "collapsed SMC raw schema changed")
        _require(str(_scalar(raw["raw_version"], "raw_version")) == RAW_VERSION,
                 "collapsed SMC raw version changed")
        _require(str(_scalar(raw["contract_version"], "contract_version")) == CONTRACT_VERSION,
                 "collapsed SMC raw contract changed")
        _require(str(_scalar(raw["manifest_sha256"], "manifest_sha256")) == manifest["manifest_sha256"],
                 "collapsed SMC raw manifest changed")
        _require(str(_scalar(raw["task_fingerprint"], "task_fingerprint")) == sha256_json(task),
                 "collapsed SMC raw task changed")
        _require(str(_scalar(raw["task_json"], "task_json")) == canonical_json(task),
                 "collapsed SMC raw task JSON changed")
        _require(str(_scalar(raw["source_binding_json"], "source_binding_json"))
                 == canonical_json(context["source_binding"]), "collapsed SMC source changed")
        _require(str(_scalar(raw["config_sha256"], "config_sha256")) == context["config_sha256"],
                 "collapsed SMC raw config changed")
        _require(str(_scalar(raw["registry_sha256"], "registry_sha256"))
                 == context["registry"]["registry_sha256"], "collapsed SMC raw registry changed")
        _require(int(_scalar(raw["uniform_seed"], "uniform_seed")) == context["uniform_seed"],
                 "collapsed SMC uniform seed changed")
        _require(np.array_equal(raw["syndrome"], context["syndrome"]), "collapsed SMC syndrome changed")
        result = _raw_result(raw)
        _require(_trajectory_digest(result) == str(_scalar(raw["trajectory_digest"], "trajectory_digest")),
                 "collapsed SMC raw digest changed")
        _require(str(_scalar(result["raw_version"], "smc_raw_version")) == COLLAPSED_SMC_RAW_VERSION,
                 "collapsed SMC engine raw changed")
        _require(str(_scalar(result["method_id"], "smc_method_id")) == context["sampler"].method_id,
                 "collapsed SMC engine method changed")
        _require(str(_scalar(result["sampler_config_json"], "smc_sampler_config_json"))
                 == canonical_json(context["sampler"].as_dict()), "collapsed SMC sampler config changed")
        _require(str(_scalar(result["sampler_config_sha256"], "smc_sampler_config_sha256"))
                 == sha256_json(context["sampler"].as_dict()), "collapsed SMC sampler config SHA changed")
        _require(str(_scalar(result["seed_identity_json"], "smc_seed_identity_json"))
                 == canonical_json(task["seed_identity"]), "collapsed SMC seed identity changed")
        _require(str(_scalar(result["kernel"], "smc_kernel")) == COLLAPSED_SMC_KERNEL,
                 "collapsed SMC kernel changed")
        _require(str(_scalar(result["engine"], "smc_engine")) == "numba", "collapsed SMC engine changed")
        _require(str(_scalar(result["mass_sha256"], "smc_mass_sha256")) == context["mass_sha256"],
                 "collapsed SMC mass SHA changed")
        _require(str(_scalar(result["lambda_sha256"], "smc_lambda_sha256"))
                 == context["config"]["method"]["lambda_sha256"], "collapsed SMC lambda SHA changed")

        b_by_stage = np.asarray(result["b_columns_by_stage"], dtype=np.uint32)
        likelihood_by_stage = np.asarray(result["log_likelihood_by_stage"], dtype=np.float64)
        roots_by_stage = np.asarray(result["roots_by_stage"], dtype=np.int32)
        weights_by_stage = np.asarray(result["normalized_incremental_weights"], dtype=np.float64)
        parents_by_stage = np.asarray(result["parent_indices"], dtype=np.int32)
        offsets = np.asarray(result["resample_offsets"], dtype=np.float64)
        cess = np.asarray(result["conditional_ess"], dtype=np.float64)
        maximum = np.asarray(result["max_normalized_weight"], dtype=np.float64)
        increments = np.asarray(result["log_normalizer_increments"], dtype=np.float64)
        changes = np.asarray(result["mutation_block_changes"], dtype=np.int32)
        levels = context["sampler"].num_levels
        particles = context["sampler"].num_particles
        _require(b_by_stage.shape == (levels, particles, context["H"].shape[0]),
                 "collapsed SMC B stage shape changed")
        _require(likelihood_by_stage.shape == (levels, particles), "collapsed SMC likelihood shape changed")
        _require(roots_by_stage.shape == (levels, particles), "collapsed SMC roots shape changed")
        _require(weights_by_stage.shape == (levels - 1, particles), "collapsed SMC weights shape changed")
        _require(parents_by_stage.shape == (levels - 1, particles), "collapsed SMC parents shape changed")
        _require(offsets.shape == cess.shape == maximum.shape == increments.shape == (levels - 1,),
                 "collapsed SMC stage summary shape changed")
        _require(changes.shape == (levels - 1, particles), "collapsed SMC mutation shape changed")
        _require(np.array_equal(roots_by_stage[0], np.arange(particles, dtype=np.int32)),
                 "collapsed SMC initial roots changed")
        mass = build_classical_coset_mass(context["H"], context["sampler"].p, engine="numba")
        log_mass = np.log(mass)
        for stage in range(levels):
            a_syndromes = a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[stage])
            expected_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
            _require(np.array_equal(likelihood_by_stage[stage], expected_likelihood),
                     "collapsed SMC stored likelihood disagrees with B state")
            if stage == 0:
                continue
            expected_weights, expected_cess, expected_maximum, expected_increment = (
                _normalized_incremental_weights(
                    likelihood_by_stage[stage - 1],
                    context["sampler"].lambda_values[stage]
                    - context["sampler"].lambda_values[stage - 1],
                )
            )
            _require(np.array_equal(weights_by_stage[stage - 1], expected_weights),
                     "collapsed SMC incremental weights changed")
            _require(float(offsets[stage - 1]) >= 0.0 and float(offsets[stage - 1]) < 1.0 / particles,
                     "collapsed SMC resample offset changed")
            _require(np.array_equal(
                parents_by_stage[stage - 1],
                systematic_resampling(expected_weights, float(offsets[stage - 1])),
            ), "collapsed SMC parents changed")
            _require(np.array_equal(
                roots_by_stage[stage], roots_by_stage[stage - 1][parents_by_stage[stage - 1]],
            ), "collapsed SMC root ancestry changed")
            _require(float(cess[stage - 1]) == expected_cess, "collapsed SMC CESS changed")
            _require(float(maximum[stage - 1]) == expected_maximum, "collapsed SMC max weight changed")
            _require(float(increments[stage - 1]) == expected_increment,
                     "collapsed SMC log normalizer increment changed")
        root_counts = np.bincount(roots_by_stage[-1], minlength=particles).astype(np.int32)
        root_fractions = root_counts.astype(np.float64) / particles
        root_ess = 1.0 / float(np.square(root_fractions).sum(dtype=np.float64))
        _require(np.array_equal(root_counts, result["final_root_counts"]), "collapsed SMC root counts changed")
        _require(float(_scalar(result["final_root_family_ess"], "final_root_family_ess")) == root_ess,
                 "collapsed SMC root ESS changed")
        _require(int(_scalar(result["final_distinct_roots"], "final_distinct_roots"))
                 == int(np.count_nonzero(root_counts)), "collapsed SMC distinct roots changed")
        _require(float(_scalar(result["final_max_root_fraction"], "final_max_root_fraction"))
                 == float(root_fractions.max()), "collapsed SMC maximum root fraction changed")
        attempts = int(_scalar(result["mutation_block_attempts_per_particle"], "mutation_attempts"))
        _require(np.all(changes >= 0) and np.all(changes <= attempts),
                 "collapsed SMC mutation counters changed")
        _require(np.all(np.isfinite(np.asarray(result["final_collapsed_log_target"], dtype=np.float64))),
                 "collapsed SMC final target is non-finite")
        metrics = {
            "base_family": task["base_family"],
            "population_index": task["population_index"],
            "minimum_conditional_ess_fraction": float(cess.min() / particles),
            "maximum_stage_normalized_weight": float(maximum.max()),
            "final_root_family_ess": root_ess,
            "final_distinct_roots": int(np.count_nonzero(root_counts)),
            "final_max_root_fraction": float(root_fractions.max()),
            "mean_final_b_bit_weight": float(np.asarray(result["final_b_bit_weights"], dtype=np.float64).mean()),
            "mean_mutation_block_changes": float(changes.mean()),
        }
    gates = context["config"]["gates"]
    checks = {
        "conditional_ess": metrics["minimum_conditional_ess_fraction"]
        >= gates["min_stage_conditional_ess_fraction"],
        "maximum_weight": metrics["maximum_stage_normalized_weight"]
        <= gates["max_stage_normalized_weight"],
        "root_ess": metrics["final_root_family_ess"] >= gates["min_final_root_family_ess"],
        "distinct_roots": metrics["final_distinct_roots"] >= gates["min_final_distinct_roots"],
        "maximum_root_fraction": metrics["final_max_root_fraction"] <= gates["max_final_root_fraction"],
    }
    return {**metrics, "checks": checks, "passes": bool(all(checks.values()))}


def analyze(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "REPLAY.json").is_file(), "collapsed SMC replay is missing")
    _require(not (output_root / "REPORT.json").exists(), "collapsed SMC report already exists")
    summaries = [_audit_raw(context, manifest, task, _raw_path(output_root, task))
                 for task in manifest["tasks"]]
    all_pass = bool(all(summary["passes"] for summary in summaries))
    status = (
        "LOCAL_COLLAPSED_SMC_WEIGHT_GENEALOGY_VIABLE"
        if all_pass else "LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE"
    )
    core = {
        "report_version": REPORT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "replay_sha256": json.loads((output_root / "REPLAY.json").read_text(encoding="ascii"))["replay_sha256"],
        "scope": context["config"]["scope"],
        "status": status,
        "all_populations_pass": all_pass,
        "population_summaries": summaries,
        "forbidden_interpretations": [
            "no_q_top_estimate",
            "no_posterior_estimation",
            "no_remote_hard2_authorization",
            "no_ready_for_formal_authorization",
        ],
    }
    atomic_json(output_root / "REPORT.json", {**core, "report_sha256": sha256_json(core)})
    atomic_json(output_root / "SUCCESS.json", {
        "manifest_sha256": manifest["manifest_sha256"], "status": status,
        "report_sha256": sha256_json(core),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "replay", "analyze", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output, args.registry, args.config)
    elif args.command == "run":
        run(args.output, args.registry, args.config)
    elif args.command == "replay":
        replay(args.output, args.registry, args.config)
    elif args.command == "analyze":
        analyze(args.output, args.registry, args.config)
    else:
        prepare(args.output, args.registry, args.config)
        run(args.output, args.registry, args.config)
        replay(args.output, args.registry, args.config)
        analyze(args.output, args.registry, args.config)


if __name__ == "__main__":
    main()
