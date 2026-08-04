"""Fail-closed local path-weight diagnostic for collapsed q=0 HGP AIS.

This runner preserves every iid-prior lineage and audits the complete AIS
weight path. It is not a q_top estimator and cannot authorize remote or formal
exp102 work.
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
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed_ais import (
    BASE_FAMILIES,
    COLLAPSED_AIS_KERNEL,
    COLLAPSED_AIS_RAW_VERSION,
    CollapsedAisConfig,
    CollapsedAisSeedIdentity,
    a_syndromes_from_b,
    collapsed_log_likelihood,
    collapsed_log_target,
    float64_big_endian_sha256,
    normalized_log_weights,
    quadratic_lambda_schedule,
    run_collapsed_ais_population,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_hgp_collapsed_ais.v0"
CONFIG_VERSION = "exp102.q0_hgp_collapsed_ais.v0.config.v1"
MANIFEST_VERSION = "exp102.q0_hgp_collapsed_ais.v0.manifest.v1"
TASK_VERSION = "exp102.q0_hgp_collapsed_ais.v0.tasks.v1"
RAW_VERSION = "exp102.q0_hgp_collapsed_ais.raw.v0"
REPLAY_VERSION = "exp102.q0_hgp_collapsed_ais.v0.replay.v1"
REPORT_VERSION = "exp102.q0_hgp_collapsed_ais.v0.report.v1"
ROOT = Path("data/expander_code/exp102")
DEFAULT_REGISTRY = ROOT / "registry/registry.json"
DEFAULT_CONFIG = ROOT / "config/q0_hgp_collapsed_ais.v0.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_m8_ais_v0"
SOURCE_FILES = (
    "data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed.py",
    "data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed_ais.py",
    "data/expander_code/exp102/exp102_pipeline/q0_global.py",
    "data/expander_code/exp102/exp102_pipeline/registry.py",
    "data/expander_code/exp102/exp102_pipeline/seeds.py",
    "data/expander_code/exp102/exp102_pipeline/worker.py",
    "data/expander_code/exp102/validation/021_q0_collapsed_ais_v0_20260723/run_local_feasibility.py",
)
RESULT_FIELDS = {
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "b_columns_by_stage", "log_likelihood_by_stage",
    "log_weight_increments", "cumulative_log_weights", "incremental_ess",
    "incremental_max_weight", "cumulative_ess", "cumulative_max_weight",
    "mutation_attempts", "mutation_changes", "final_normalized_weights",
    "final_importance_ess", "final_max_normalized_weight", "final_log_mean_weight",
    "final_b_bit_weights", "final_collapsed_log_target", "lambda_values",
    "lambda_sha256", "mass_sha256", "kernel", "engine",
}


class LocalAisConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalAisConflict(message)


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
    _require(set(result) == RESULT_FIELDS, "collapsed AIS result fields changed")
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_collapsed_ais.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        _require(not value.dtype.hasobject, f"AIS result {name} has object dtype")
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
    _require(len(source_commit) == 40 and all(value in "0123456789abcdef" for value in source_commit),
             "AIS source commit is invalid")
    source_files = {path: sha256_file(path) for path in SOURCE_FILES}
    payload = {"source_commit": source_commit, "source_files": source_files}
    return {**payload, "source_binding_sha256": sha256_json(payload)}


def _load_config(path, registry):
    try:
        config = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise LocalAisConflict(f"cannot load collapsed AIS config: {exc}") from exc
    expected_fields = {
        "cell", "config_version", "contract_version", "gates", "method",
        "population", "raw_version", "registry_sha256", "scope", "trajectory_namespace",
    }
    _require(set(config) == expected_fields, "collapsed AIS config fields changed")
    _require(config["contract_version"] == CONTRACT_VERSION, "collapsed AIS contract changed")
    _require(config["config_version"] == CONFIG_VERSION, "collapsed AIS config version changed")
    _require(config["raw_version"] == RAW_VERSION, "collapsed AIS raw version changed")
    _require(config["registry_sha256"] == registry["registry_sha256"], "collapsed AIS registry changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "collapsed AIS cell changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "purpose": "local_collapsed_ais_path_weight_feasibility_only",
        "remote_authorization": False,
    }, "collapsed AIS scope changed")
    _require(config["population"] == {
        "base_families": ["column_major", "row_major"],
        "num_particles": 128,
        "populations_per_base_family": 4,
    }, "collapsed AIS population changed")
    _require(set(config["method"]) == {
        "block_size", "id", "kernel", "lambda_generation", "lambda_sha256",
        "lambda_values", "mutation_sweeps", "prior_endpoint", "resampling",
    }, "collapsed AIS method fields changed")
    _require(config["method"]["id"] == "CAIS64-B8-S1-N128", "collapsed AIS method ID changed")
    _require(config["method"]["kernel"] == COLLAPSED_AIS_KERNEL, "collapsed AIS kernel changed")
    _require(config["method"]["lambda_generation"] == "hp64_quadratic_index_v1",
             "collapsed AIS lambda generator changed")
    _require(config["method"]["prior_endpoint"] == "exact_iid_bernoulli_B",
             "collapsed AIS prior endpoint changed")
    _require(config["method"]["resampling"] == "none", "collapsed AIS resampling changed")
    expected_lambda = quadratic_lambda_schedule(64)
    _require(tuple(config["method"]["lambda_values"]) == expected_lambda,
             "collapsed AIS lambda values changed")
    _require(config["method"]["lambda_sha256"] == float64_big_endian_sha256(expected_lambda),
             "collapsed AIS lambda SHA changed")
    sampler = CollapsedAisConfig(
        p=config["cell"]["p"],
        num_particles=config["population"]["num_particles"],
        lambda_values=tuple(config["method"]["lambda_values"]),
        mutation_sweeps=config["method"]["mutation_sweeps"],
        block_size=config["method"]["block_size"],
        method_id=config["method"]["id"],
    )
    _require(config["gates"] == {
        "max_final_normalized_weight": 0.1,
        "max_incremental_normalized_weight": 0.1,
        "min_final_importance_ess_fraction": 0.25,
        "require_all_populations_to_pass": True,
    }, "collapsed AIS gates changed")
    _require(isinstance(config["trajectory_namespace"], str) and config["trajectory_namespace"],
             "collapsed AIS trajectory namespace is invalid")
    return config, sampler, sha256_json(config)


def _attempt022_uniform_seed(registry, code, cell):
    _require(cell["disorder_source"] == "attempt022", "collapsed AIS disorder source changed")
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
    _require(bool(syndrome.any()), "collapsed AIS planted syndrome unexpectedly vanishes")
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
        "mass_sha256": float64_big_endian_sha256(mass),
        "source_binding": _source_binding(),
    }
    if manifest is not None:
        _validate_manifest_context(manifest, result)
    return result


def _task_identity(context, base_family, population_index):
    identity = CollapsedAisSeedIdentity(
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
        _task_identity(context, family, population_index)
        for family in context["config"]["population"]["base_families"]
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
        raise LocalAisConflict(f"cannot load collapsed AIS manifest: {exc}") from exc
    expected_fields = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "model_fingerprint", "logical_frame_fingerprint",
        "classical_mass_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == expected_fields, "collapsed AIS manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "collapsed AIS manifest SHA changed")
    _require(manifest["manifest_version"] == MANIFEST_VERSION, "collapsed AIS manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION, "collapsed AIS manifest contract changed")
    _require(manifest["raw_version"] == RAW_VERSION, "collapsed AIS manifest raw version changed")
    _require(isinstance(manifest["tasks"], list) and len(manifest["tasks"]) == 8,
             "collapsed AIS task count changed")
    _require(len({sha256_json(task) for task in manifest["tasks"]}) == 8,
             "collapsed AIS tasks are duplicated")
    return manifest


def _validate_manifest_context(manifest, context):
    core = _manifest_core(context, manifest["tasks"])
    _require(manifest == {**core, "manifest_sha256": sha256_json(core)},
             "collapsed AIS manifest/context binding changed")


def _raw_path(output_root, task):
    return Path(output_root) / "raw" / task["base_family"] / f"p{int(task['population_index']):02d}.npz"


def prepare(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(not (output_root / "MANIFEST.json").exists(), "collapsed AIS manifest already exists")
    _require(not (output_root / "raw").exists(), "collapsed AIS raw already exists")
    context = _context(registry_path, config_path)
    tasks = _manifest_tasks(context)
    core = _manifest_core(context, tasks)
    atomic_json(output_root / "MANIFEST.json", {**core, "manifest_sha256": sha256_json(core)})


def _execute_task(context, task):
    _require(task in _manifest_tasks(context), "collapsed AIS task is not canonical")
    identity = CollapsedAisSeedIdentity(**task["seed_identity"])
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_collapsed_ais_population(
        context["model"], context["frame"], context["H"], context["syndrome"],
        context["sampler"], identity, engine=task["engine"],
    )
    core_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    _require(result["raw_version"] == COLLAPSED_AIS_RAW_VERSION, "collapsed AIS raw changed")
    _require(result["kernel"] == COLLAPSED_AIS_KERNEL, "collapsed AIS kernel changed")
    _require(str(_scalar(result["mass_sha256"], "mass_sha256")) == context["mass_sha256"],
             "collapsed AIS mass changed")
    _require(str(_scalar(result["lambda_sha256"], "lambda_sha256"))
             == context["config"]["method"]["lambda_sha256"], "collapsed AIS lambda changed")
    _require(np.all(np.isfinite(np.asarray(result["cumulative_log_weights"], dtype=np.float64))),
             "collapsed AIS cumulative path weights are non-finite")
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
    arrays.update({f"ais_{name}": np.asarray(value) for name, value in result.items()})
    return arrays


def run(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    for marker in ("RUNNING.json", "SUCCESS.json", "FAILED.json"):
        _require(not (output_root / marker).exists(), "collapsed AIS run marker already exists")
    for task in manifest["tasks"]:
        _require(not _raw_path(output_root, task).exists(), "collapsed AIS raw already exists")
    atomic_json(output_root / "RUNNING.json", {
        "stage": "run", "manifest_sha256": manifest["manifest_sha256"], "workers": 1,
    })
    completed = []
    try:
        for task in manifest["tasks"]:
            result, core_seconds, wall_seconds = _execute_task(context, task)
            path = _raw_path(output_root, task)
            atomic_npz(path, **_raw_payload(context, manifest, task, result, core_seconds, wall_seconds))
            completed.append({
                "task_fingerprint": sha256_json(task), "path": str(path),
                "core_seconds": core_seconds, "wall_seconds": wall_seconds,
            })
    except Exception as exc:
        atomic_json(output_root / "FAILED.json", {
            "stage": "run", "manifest_sha256": manifest["manifest_sha256"],
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    atomic_json(output_root / "RUN_COMPLETE.json", {
        "manifest_sha256": manifest["manifest_sha256"], "tasks": completed,
    })


def _raw_result(raw):
    return {name: np.asarray(raw[f"ais_{name}"]) for name in RESULT_FIELDS}


def replay(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "collapsed AIS run is incomplete")
    _require(not (output_root / "REPLAY.json").exists(), "collapsed AIS replay already exists")
    values = []
    for task in manifest["tasks"]:
        with load_npz_no_pickle(_raw_path(output_root, task)) as raw:
            expected_digest = str(_scalar(raw["trajectory_digest"], "trajectory_digest"))
        replayed, _, _ = _execute_task(context, task)
        actual_digest = _trajectory_digest(replayed)
        _require(actual_digest == expected_digest, "collapsed AIS deterministic replay failed")
        values.append({"task_fingerprint": sha256_json(task), "trajectory_digest": actual_digest})
    core = {
        "replay_version": REPLAY_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": values,
    }
    atomic_json(output_root / "REPLAY.json", {**core, "replay_sha256": sha256_json(core)})


def _audit_raw(context, manifest, task):
    path = _raw_path(DEFAULT_OUTPUT, task)
    _require(path.is_file(), "collapsed AIS raw is missing")
    with load_npz_no_pickle(path) as raw:
        expected_fields = {
            "raw_version", "contract_version", "manifest_sha256", "task_fingerprint", "task_json",
            "source_binding_json", "config_sha256", "registry_sha256", "uniform_seed", "syndrome",
            "trajectory_digest", "core_seconds", "wall_seconds",
            *(f"ais_{name}" for name in RESULT_FIELDS),
        }
        _require(set(raw.files) == expected_fields, "collapsed AIS raw schema changed")
        _require(str(_scalar(raw["raw_version"], "raw_version")) == RAW_VERSION, "raw version changed")
        _require(str(_scalar(raw["contract_version"], "contract_version")) == CONTRACT_VERSION,
                 "raw contract changed")
        _require(str(_scalar(raw["manifest_sha256"], "manifest_sha256")) == manifest["manifest_sha256"],
                 "raw manifest changed")
        _require(str(_scalar(raw["task_fingerprint"], "task_fingerprint")) == sha256_json(task),
                 "raw task fingerprint changed")
        _require(str(_scalar(raw["task_json"], "task_json")) == canonical_json(task), "raw task JSON changed")
        _require(str(_scalar(raw["source_binding_json"], "source_binding_json"))
                 == canonical_json(context["source_binding"]), "raw source changed")
        _require(str(_scalar(raw["config_sha256"], "config_sha256")) == context["config_sha256"],
                 "raw config changed")
        _require(str(_scalar(raw["registry_sha256"], "registry_sha256"))
                 == context["registry"]["registry_sha256"], "raw registry changed")
        _require(int(_scalar(raw["uniform_seed"], "uniform_seed")) == context["uniform_seed"],
                 "raw uniform seed changed")
        _require(np.array_equal(raw["syndrome"], context["syndrome"]), "raw syndrome changed")
        result = _raw_result(raw)
        _require(_trajectory_digest(result) == str(_scalar(raw["trajectory_digest"], "trajectory_digest")),
                 "raw trajectory digest changed")
        _require(str(_scalar(result["raw_version"], "engine_raw_version")) == COLLAPSED_AIS_RAW_VERSION,
                 "engine raw version changed")
        _require(str(_scalar(result["method_id"], "engine_method_id")) == context["sampler"].method_id,
                 "engine method changed")
        _require(str(_scalar(result["sampler_config_json"], "sampler_config_json"))
                 == canonical_json(context["sampler"].as_dict()), "sampler config changed")
        _require(str(_scalar(result["sampler_config_sha256"], "sampler_config_sha256"))
                 == sha256_json(context["sampler"].as_dict()), "sampler config SHA changed")
        _require(str(_scalar(result["seed_identity_json"], "seed_identity_json"))
                 == canonical_json(task["seed_identity"]), "seed identity changed")
        _require(str(_scalar(result["kernel"], "kernel")) == COLLAPSED_AIS_KERNEL, "kernel changed")
        _require(str(_scalar(result["engine"], "engine")) == "numba", "engine changed")
        _require(str(_scalar(result["mass_sha256"], "mass_sha256")) == context["mass_sha256"],
                 "mass SHA changed")

        b_by_stage = np.asarray(result["b_columns_by_stage"], dtype=np.uint32)
        likelihood_by_stage = np.asarray(result["log_likelihood_by_stage"], dtype=np.float64)
        increments = np.asarray(result["log_weight_increments"], dtype=np.float64)
        cumulative = np.asarray(result["cumulative_log_weights"], dtype=np.float64)
        incremental_ess = np.asarray(result["incremental_ess"], dtype=np.float64)
        incremental_max = np.asarray(result["incremental_max_weight"], dtype=np.float64)
        cumulative_ess = np.asarray(result["cumulative_ess"], dtype=np.float64)
        cumulative_max = np.asarray(result["cumulative_max_weight"], dtype=np.float64)
        attempts = np.asarray(result["mutation_attempts"], dtype=np.int32)
        changes = np.asarray(result["mutation_changes"], dtype=np.int32)
        levels = context["sampler"].num_levels
        particles = context["sampler"].num_particles
        _require(b_by_stage.shape == (levels, particles, context["H"].shape[0]), "B stage shape changed")
        _require(likelihood_by_stage.shape == cumulative.shape == (levels, particles),
                 "AIS path array shape changed")
        _require(increments.shape == attempts.shape == changes.shape == (levels - 1, particles),
                 "AIS increment/mutation shape changed")
        _require(incremental_ess.shape == incremental_max.shape == cumulative_ess.shape
                 == cumulative_max.shape == (levels - 1,), "AIS summary shape changed")
        _require(np.array_equal(cumulative[0], np.zeros(particles, dtype=np.float64)),
                 "AIS initial weights changed")
        mass = build_classical_coset_mass(context["H"], context["sampler"].p, engine="numba")
        log_mass = np.log(mass)
        lambdas = context["sampler"].lambda_values
        for stage in range(levels):
            a_syndromes = a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[stage])
            expected_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
            _require(np.array_equal(likelihood_by_stage[stage], expected_likelihood),
                     "stored likelihood disagrees with B state")
            if stage == 0:
                continue
            expected_increment = (lambdas[stage] - lambdas[stage - 1]) * likelihood_by_stage[stage - 1]
            _require(np.array_equal(increments[stage - 1], expected_increment), "AIS increment changed")
            _require(np.array_equal(cumulative[stage], cumulative[stage - 1] + expected_increment),
                     "AIS cumulative path weight changed")
            _, expected_inc_ess, expected_inc_max, _ = normalized_log_weights(expected_increment)
            _, expected_cum_ess, expected_cum_max, _ = normalized_log_weights(cumulative[stage])
            _require(float(incremental_ess[stage - 1]) == expected_inc_ess, "incremental ESS changed")
            _require(float(incremental_max[stage - 1]) == expected_inc_max, "incremental max changed")
            _require(float(cumulative_ess[stage - 1]) == expected_cum_ess, "cumulative ESS changed")
            _require(float(cumulative_max[stage - 1]) == expected_cum_max, "cumulative max changed")
        final_weights, final_ess, final_max, final_log_mean = normalized_log_weights(cumulative[-1])
        _require(np.array_equal(final_weights, result["final_normalized_weights"]), "final weights changed")
        _require(float(_scalar(result["final_importance_ess"], "final_importance_ess")) == final_ess,
                 "final ESS changed")
        _require(float(_scalar(result["final_max_normalized_weight"], "final_max_weight")) == final_max,
                 "final max weight changed")
        _require(float(_scalar(result["final_log_mean_weight"], "final_log_mean_weight")) == final_log_mean,
                 "final log mean weight changed")
        expected_target = collapsed_log_target(
            b_by_stage[-1], a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[-1]),
            context["sampler"].p, log_mass,
        )
        _require(np.array_equal(expected_target, result["final_collapsed_log_target"]),
                 "final collapsed target changed")
        _require(np.all(attempts > 0) and np.all(changes >= 0) and np.all(changes <= attempts),
                 "mutation counters changed")
        metrics = {
            "base_family": task["base_family"],
            "population_index": task["population_index"],
            "final_importance_ess_fraction": float(final_ess / particles),
            "final_max_normalized_weight": float(final_max),
            "maximum_incremental_normalized_weight": float(incremental_max.max()),
            "minimum_cumulative_ess_fraction": float(cumulative_ess.min() / particles),
            "mean_final_b_bit_weight": float(np.asarray(result["final_b_bit_weights"], dtype=np.float64).mean()),
            "mean_mutation_block_changes": float(changes.mean()),
            "final_log_mean_weight": float(final_log_mean),
        }
    gates = context["config"]["gates"]
    checks = {
        "final_ess": metrics["final_importance_ess_fraction"] >= gates["min_final_importance_ess_fraction"],
        "final_max_weight": metrics["final_max_normalized_weight"] <= gates["max_final_normalized_weight"],
        "incremental_max_weight": metrics["maximum_incremental_normalized_weight"]
        <= gates["max_incremental_normalized_weight"],
    }
    return {**metrics, "checks": checks, "passes": bool(all(checks.values()))}


def analyze(output_root, registry_path=DEFAULT_REGISTRY, config_path=DEFAULT_CONFIG):
    output_root = Path(output_root)
    _require(output_root == DEFAULT_OUTPUT, "AIS V0 audit output path is frozen")
    manifest = _load_manifest(output_root / "MANIFEST.json")
    context = _context(registry_path, config_path, manifest=manifest)
    _require((output_root / "REPLAY.json").is_file(), "collapsed AIS replay is missing")
    _require(not (output_root / "REPORT.json").exists(), "collapsed AIS report already exists")
    summaries = [_audit_raw(context, manifest, task) for task in manifest["tasks"]]
    all_pass = bool(all(summary["passes"] for summary in summaries))
    status = (
        "LOCAL_COLLAPSED_AIS_PATH_WEIGHT_VIABLE"
        if all_pass else "LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE"
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
