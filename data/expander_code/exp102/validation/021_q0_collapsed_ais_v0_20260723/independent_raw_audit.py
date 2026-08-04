#!/usr/bin/env python3
"""Independent raw-only audit for the frozen collapsed-AIS V0 diagnostic.

This verifier deliberately does not import the AIS engine or call an MCMC
kernel.  It rebuilds the hard cell, exact iid base population, collapsed
coset-mass recurrence, path weights, and terminal gates from immutable NPZ
raw data using ``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


ROOT = Path("data/expander_code/exp102")
RUN_ROOT = Path(__file__).resolve().parent / "local_m8_ais_v0"
CONFIG_PATH = ROOT / "config/q0_hgp_collapsed_ais.v0.json"
REGISTRY_PATH = ROOT / "registry/registry.json"

CONTRACT_VERSION = "exp102.q0_hgp_collapsed_ais.v0"
RAW_VERSION = "exp102.q0_hgp_collapsed_ais.raw.v0"
ENGINE_RAW_VERSION = "exp102.q0_hgp_collapsed_ais.raw.v0"
ENGINE_VERSION = "exp102.q0_hgp_collapsed_ais.v0"
KERNEL = "reversible_random_block_heatbath_ais.v1"
AUDIT_VERSION = "exp102.q0_hgp_collapsed_ais.v0.independent_raw_audit.v1"
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


class AuditConflict(RuntimeError):
    """A raw artifact or claimed result cannot be independently reproduced."""


def require(condition, message):
    if not condition:
        raise AuditConflict(message)


def scalar(value, name):
    array = np.asarray(value)
    require(array.shape == (), f"{name} must be scalar")
    return array.item()


def canonical_sha(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def float64_big_endian_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype=">f8").tobytes(order="C")).hexdigest()


def trajectory_digest(result):
    require(set(result) == RESULT_FIELDS, "AIS result fields changed")
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_collapsed_ais.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        require(not value.dtype.hasobject, f"AIS result {name} has object dtype")
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        encoded_dtype = value.dtype.str.encode("ascii")
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def quadratic_lambdas():
    denominator = float(63 * 63)
    return np.asarray([index * index / denominator for index in range(64)], dtype=np.float64)


def bits_to_mask(bits):
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    require(values.size <= 32, "classical mask exceeds uint32 capacity")
    require(np.all((values == 0) | (values == 1)), "mask bits are not binary")
    value = 0
    for index, bit in enumerate(values):
        value |= int(bit) << index
    return np.uint32(value)


def y_columns(syndrome, H):
    rows, columns = H.shape
    matrix = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    return np.asarray([bits_to_mask(matrix[:, column]) for column in range(columns)], dtype=np.uint32)


def a_syndromes_from_b(syndrome, H, b_columns):
    """Independently apply H A xor B H = Y to every saved B state."""
    H = np.asarray(H, dtype=np.uint8)
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    rows, columns = H.shape
    require(b_columns.ndim == 2 and b_columns.shape[1] == rows, "B state shape changed")
    valid_mask = (1 << rows) - 1
    unused_mask = np.uint32(((1 << 32) - 1) ^ valid_mask)
    require(np.all((b_columns & unused_mask) == 0), "B state has unused bits")
    result = np.repeat(y_columns(syndrome, H)[None, :], b_columns.shape[0], axis=0)
    for h_column in range(columns):
        for b_column in np.flatnonzero(H[:, h_column]):
            result[:, h_column] ^= b_columns[:, int(b_column)]
    return result


def independent_coset_mass(H, p):
    """Build Pr[H x=s] with a separate vectorized GF(2) dynamic program."""
    H = np.asarray(H, dtype=np.uint8)
    rows, columns = H.shape
    require(0 < rows <= 24, "classical rank is outside the frozen mass-table bound")
    require(0.0 < float(p) < 0.5, "p is invalid")
    size = 1 << rows
    current = np.zeros(size, dtype=np.float64)
    scratch = np.empty(size, dtype=np.float64)
    permuted = np.empty(size, dtype=np.float64)
    indices = np.arange(size, dtype=np.uint32)
    current[0] = 1.0
    keep = np.float64(1.0 - float(p))
    flip = np.float64(p)
    for column in range(columns):
        mask = bits_to_mask(H[:, column])
        # XOR is its own inverse, so reuse the index buffer without allocations.
        np.bitwise_xor(indices, mask, out=indices)
        np.take(current, indices, out=permuted)
        np.bitwise_xor(indices, mask, out=indices)
        np.multiply(current, keep, out=scratch)
        np.multiply(permuted, flip, out=permuted)
        np.add(scratch, permuted, out=scratch)
        current, scratch = scratch, current
    require(np.all(np.isfinite(current)) and np.all(current > 0.0), "mass table is invalid")
    require(abs(float(current.sum(dtype=np.float64)) - 1.0) <= 5e-13, "mass table is unnormalized")
    return np.ascontiguousarray(current)


def log_likelihood(a_syndromes, log_mass):
    values = np.asarray(a_syndromes, dtype=np.uint32)
    require(values.ndim == 2 and np.all(values < log_mass.size), "A coset syndromes are invalid")
    result = np.zeros(values.shape[0], dtype=np.float64)
    for factor in range(values.shape[1]):
        result += log_mass[values[:, factor]]
    return result


def normalized_log_weights(log_weights):
    values = np.asarray(log_weights, dtype=np.float64)
    require(values.ndim == 1 and np.all(np.isfinite(values)), "log weights are non-finite")
    maximum = float(values.max())
    shifted = np.exp(values - maximum)
    total = float(shifted.sum(dtype=np.float64))
    require(total > 0.0 and math.isfinite(total), "normalized weights vanished")
    weights = np.ascontiguousarray(shifted / total)
    ess = 1.0 / float(np.square(weights).sum(dtype=np.float64))
    return weights, ess, float(weights.max()), maximum + math.log(total / weights.size)


def close_enough(actual, expected, name, *, atol=2e-11, rtol=2e-13):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, f"{name} shape changed")
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), f"{name} is non-finite")
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), f"{name} disagrees with independent reconstruction")
    return error


def expected_config():
    lambdas = quadratic_lambdas()
    return {
        "cell": {
            "code_id": "m08_c06", "disorder_index": 0,
            "disorder_source": "attempt022", "p": 0.04,
        },
        "config_version": "exp102.q0_hgp_collapsed_ais.v0.config.v1",
        "contract_version": CONTRACT_VERSION,
        "gates": {
            "max_final_normalized_weight": 0.1,
            "max_incremental_normalized_weight": 0.1,
            "min_final_importance_ess_fraction": 0.25,
            "require_all_populations_to_pass": True,
        },
        "method": {
            "block_size": 8,
            "id": "CAIS64-B8-S1-N128",
            "kernel": KERNEL,
            "lambda_generation": "hp64_quadratic_index_v1",
            "lambda_sha256": float64_big_endian_sha256(lambdas),
            "lambda_values": lambdas.tolist(),
            "mutation_sweeps": 1,
            "prior_endpoint": "exact_iid_bernoulli_B",
            "resampling": "none",
        },
        "population": {
            "base_families": ["column_major", "row_major"],
            "num_particles": 128,
            "populations_per_base_family": 4,
        },
        "raw_version": RAW_VERSION,
        "scope": {
            "formal_authorization": False,
            "posterior_estimation": False,
            "purpose": "local_collapsed_ais_path_weight_feasibility_only",
            "remote_authorization": False,
        },
        "trajectory_namespace": "exp102.q0_hgp_collapsed_ais.v0.20260723",
    }


def load_context():
    registry = load_registry(REGISTRY_PATH)
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    except Exception as exc:
        raise AuditConflict(f"cannot load AIS config: {exc}") from exc
    expected = expected_config()
    expected["registry_sha256"] = registry["registry_sha256"]
    require(config == expected, "frozen AIS config changed")
    _, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    H = np.ascontiguousarray(H, dtype=np.uint8)
    rows, columns = H.shape
    require(rows <= 24 and rows < columns, "hard-cell classical dimensions changed")
    uniform_seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(config["cell"]["disorder_index"]), "uniforms",
    )
    num_qubits = columns * columns + rows * rows
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(num_qubits)
        < float(config["cell"]["p"])
    ).astype(np.uint8)
    A = epsilon[:columns * columns].reshape(columns, columns)
    B = epsilon[columns * columns:].reshape(rows, rows)
    syndrome = (
        H.astype(np.int64) @ A.astype(np.int64)
        + B.astype(np.int64) @ H.astype(np.int64)
    ) % 2
    syndrome = np.ascontiguousarray(syndrome.reshape(-1), dtype=np.uint8)
    require(bool(syndrome.any()), "independently reconstructed planted syndrome vanishes")
    mass = independent_coset_mass(H, config["cell"]["p"])
    return {
        "registry": registry,
        "config": config,
        "config_sha256": canonical_sha(config),
        "code": code,
        "H": H,
        "uniform_seed": int(uniform_seed),
        "syndrome": syndrome,
        "mass": mass,
        "mass_sha256": float64_big_endian_sha256(mass),
    }


def load_manifest(context):
    path = RUN_ROOT / "MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise AuditConflict(f"cannot load AIS manifest: {exc}") from exc
    expected_fields = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "model_fingerprint", "logical_frame_fingerprint",
        "classical_mass_sha256", "tasks", "manifest_sha256",
    }
    require(set(manifest) == expected_fields, "manifest schema changed")
    core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    require(manifest["manifest_sha256"] == canonical_sha(core), "manifest SHA changed")
    require(manifest["manifest_version"] == "exp102.q0_hgp_collapsed_ais.v0.manifest.v1",
            "manifest version changed")
    require(manifest["contract_version"] == CONTRACT_VERSION, "manifest contract changed")
    require(manifest["raw_version"] == RAW_VERSION, "manifest raw version changed")
    require(manifest["config"] == context["config"], "manifest config changed")
    require(manifest["config_sha256"] == context["config_sha256"], "manifest config SHA changed")
    require(manifest["registry_sha256"] == context["registry"]["registry_sha256"],
            "manifest registry changed")
    require(manifest["cell"] == context["config"]["cell"], "manifest cell changed")
    require(int(manifest["uniform_seed"]) == context["uniform_seed"], "manifest uniform seed changed")
    require(manifest["H_sha256"] == array_sha256(context["H"]), "manifest H changed")
    require(manifest["syndrome_sha256"] == array_sha256(context["syndrome"]), "manifest syndrome changed")
    require(manifest["classical_mass_sha256"] == context["mass_sha256"], "manifest mass changed")
    check_source_binding(manifest["source_binding"])
    tasks = manifest["tasks"]
    require(isinstance(tasks, list) and len(tasks) == 8, "manifest task count changed")
    require(len({canonical_sha(task) for task in tasks}) == 8, "manifest tasks are duplicated")
    expected_tasks = []
    source_binding = manifest["source_binding"]
    for base_family in context["config"]["population"]["base_families"]:
        for population_index in range(context["config"]["population"]["populations_per_base_family"]):
            seed_identity = {
                "source_commit": source_binding["source_commit"],
                "config_sha256": context["config_sha256"],
                "registry_sha256": context["registry"]["registry_sha256"],
                "cell_fingerprint": canonical_sha(context["config"]["cell"]),
                "method_id": context["config"]["method"]["id"],
                "base_family": base_family,
                "population_index": population_index,
                "trajectory_namespace": context["config"]["trajectory_namespace"],
            }
            expected_tasks.append({
                "task_version": "exp102.q0_hgp_collapsed_ais.v0.tasks.v1",
                "raw_version": RAW_VERSION,
                "cell": context["config"]["cell"],
                "method_id": context["config"]["method"]["id"],
                "base_family": base_family,
                "population_index": population_index,
                "seed_identity": seed_identity,
                "engine": "numba",
            })
    require(tasks == expected_tasks, "manifest task identities changed")
    return manifest


def check_source_binding(binding):
    require(set(binding) == {"source_commit", "source_files", "source_binding_sha256"},
            "source binding schema changed")
    core = {"source_commit": binding["source_commit"], "source_files": binding["source_files"]}
    require(binding["source_binding_sha256"] == canonical_sha(core), "source binding SHA changed")
    require(isinstance(binding["source_files"], dict) and binding["source_files"],
            "source binding file list changed")
    workspace = Path.cwd().resolve()
    for relative, expected_sha in binding["source_files"].items():
        path = (workspace / relative).resolve()
        require(workspace in path.parents and path.is_file(), "bound source file is unavailable")
        require(sha256_file(path) == expected_sha, f"bound source file changed: {relative}")
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception as exc:
        raise AuditConflict(f"cannot inspect source commit: {exc}") from exc
    require(head == binding["source_commit"], "bound source commit changed")


def task_path(task):
    return RUN_ROOT / "raw" / task["base_family"] / f"p{int(task['population_index']):02d}.npz"


def expected_sampler_config(context):
    method = context["config"]["method"]
    population = context["config"]["population"]
    return {
        "method_id": method["id"],
        "p": context["config"]["cell"]["p"],
        "num_particles": population["num_particles"],
        "lambda_values": method["lambda_values"],
        "mutation_sweeps": method["mutation_sweeps"],
        "block_size": method["block_size"],
        "resampling": "none",
        "kernel": KERNEL,
        "prior_endpoint": "exact_iid_bernoulli_B",
    }


def expected_initial_b(context, task):
    """Recreate the exact iid base, independently of any AIS transition."""
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rows = int(context["H"].shape[0])
    particles = int(context["config"]["population"]["num_particles"])
    p = float(context["config"]["cell"]["p"])
    seed_identity = task["seed_identity"]
    if task["base_family"] == "column_major":
        positions = [(column, row) for column in range(rows) for row in range(rows)]
    else:
        positions = [(column, row) for row in range(rows) for column in range(rows)]
    values = np.zeros((particles, rows), dtype=np.uint32)
    for output_slot in range(particles):
        seed = derive_seed(
            ENGINE_VERSION,
            seed_identity["trajectory_namespace"],
            seed_identity["source_commit"],
            seed_identity["config_sha256"],
            seed_identity["registry_sha256"],
            seed_identity["cell_fingerprint"],
            seed_identity["method_id"],
            seed_identity["base_family"],
            seed_identity["population_index"],
            "initialize", 0, 0, output_slot,
        )
        rng = PortablePrng(seed)
        for column, row in positions:
            if rng.random() < p:
                values[output_slot, column] |= np.uint32(1) << np.uint32(row)
    return values


def raw_result(raw):
    return {name: np.asarray(raw[f"ais_{name}"]) for name in RESULT_FIELDS}


def audit_task(context, manifest, task):
    path = task_path(task)
    require(path.is_file(), "raw population is missing")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    require(not any(value.dtype.hasobject for value in raw.values()), "raw contains object dtype")
    expected_fields = {
        "raw_version", "contract_version", "manifest_sha256", "task_fingerprint", "task_json",
        "source_binding_json", "config_sha256", "registry_sha256", "uniform_seed", "syndrome",
        "trajectory_digest", "core_seconds", "wall_seconds",
        *(f"ais_{name}" for name in RESULT_FIELDS),
    }
    require(set(raw) == expected_fields, "raw schema changed")
    require(str(scalar(raw["raw_version"], "raw_version")) == RAW_VERSION, "raw version changed")
    require(str(scalar(raw["contract_version"], "contract_version")) == CONTRACT_VERSION,
            "raw contract changed")
    require(str(scalar(raw["manifest_sha256"], "manifest_sha256")) == manifest["manifest_sha256"],
            "raw manifest changed")
    require(str(scalar(raw["task_fingerprint"], "task_fingerprint")) == canonical_sha(task),
            "raw task fingerprint changed")
    require(str(scalar(raw["task_json"], "task_json")) == canonical_json(task), "raw task JSON changed")
    require(str(scalar(raw["source_binding_json"], "source_binding_json"))
            == canonical_json(manifest["source_binding"]), "raw source binding changed")
    require(str(scalar(raw["config_sha256"], "config_sha256")) == context["config_sha256"],
            "raw config changed")
    require(str(scalar(raw["registry_sha256"], "registry_sha256"))
            == context["registry"]["registry_sha256"], "raw registry changed")
    require(int(scalar(raw["uniform_seed"], "uniform_seed")) == context["uniform_seed"],
            "raw uniform seed changed")
    require(np.array_equal(raw["syndrome"], context["syndrome"]), "raw syndrome changed")

    result = raw_result(raw)
    require(trajectory_digest(result) == str(scalar(raw["trajectory_digest"], "trajectory_digest")),
            "raw trajectory digest changed")
    require(str(scalar(result["raw_version"], "engine raw version")) == ENGINE_RAW_VERSION,
            "engine raw version changed")
    require(str(scalar(result["method_id"], "engine method")) == context["config"]["method"]["id"],
            "engine method changed")
    sampler = expected_sampler_config(context)
    require(str(scalar(result["sampler_config_json"], "sampler config")) == canonical_json(sampler),
            "engine sampler config changed")
    require(str(scalar(result["sampler_config_sha256"], "sampler config SHA")) == canonical_sha(sampler),
            "engine sampler config SHA changed")
    require(str(scalar(result["seed_identity_json"], "seed identity")) == canonical_json(task["seed_identity"]),
            "engine seed identity changed")
    require(str(scalar(result["kernel"], "kernel")) == KERNEL, "engine kernel changed")
    require(str(scalar(result["engine"], "engine")) == "numba", "engine changed")
    require(str(scalar(result["mass_sha256"], "mass SHA")) == context["mass_sha256"],
            "engine mass table SHA changed")

    b_by_stage = np.asarray(result["b_columns_by_stage"])
    require(b_by_stage.dtype == np.dtype(np.uint32), "B state dtype changed")
    b_by_stage = np.asarray(b_by_stage, dtype=np.uint32)
    likelihood = np.asarray(result["log_likelihood_by_stage"], dtype=np.float64)
    increments = np.asarray(result["log_weight_increments"], dtype=np.float64)
    cumulative = np.asarray(result["cumulative_log_weights"], dtype=np.float64)
    incremental_ess = np.asarray(result["incremental_ess"], dtype=np.float64)
    incremental_max = np.asarray(result["incremental_max_weight"], dtype=np.float64)
    cumulative_ess = np.asarray(result["cumulative_ess"], dtype=np.float64)
    cumulative_max = np.asarray(result["cumulative_max_weight"], dtype=np.float64)
    attempts = np.asarray(result["mutation_attempts"], dtype=np.int32)
    changes = np.asarray(result["mutation_changes"], dtype=np.int32)
    lambdas = quadratic_lambdas()
    levels, particles, rows = b_by_stage.shape
    require((levels, particles, rows) == (64, 128, context["H"].shape[0]), "B path shape changed")
    require(likelihood.shape == cumulative.shape == (levels, particles), "AIS path shape changed")
    require(increments.shape == attempts.shape == changes.shape == (levels - 1, particles),
            "AIS transition shape changed")
    require(incremental_ess.shape == incremental_max.shape == cumulative_ess.shape
            == cumulative_max.shape == (levels - 1,), "AIS summary shape changed")
    require(np.array_equal(np.asarray(result["lambda_values"], dtype=np.float64), lambdas),
            "stored lambda schedule changed")
    require(str(scalar(result["lambda_sha256"], "lambda SHA")) == float64_big_endian_sha256(lambdas),
            "stored lambda SHA changed")
    require(np.array_equal(cumulative[0], np.zeros(particles, dtype=np.float64)),
            "initial cumulative weights changed")
    require(np.array_equal(b_by_stage[0], expected_initial_b(context, task)),
            "exact iid base population changed")

    log_mass = np.log(context["mass"])
    likelihood_error = 0.0
    increment_error = 0.0
    cumulative_error = 0.0
    normalized_error = 0.0
    computed_cumulative = np.zeros((levels, particles), dtype=np.float64)
    previous_likelihood = None
    for stage in range(levels):
        a_syndromes = a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[stage])
        reconstructed_likelihood = log_likelihood(a_syndromes, log_mass)
        likelihood_error = max(likelihood_error, close_enough(
            likelihood[stage], reconstructed_likelihood, "stored likelihood",
        ))
        if stage == 0:
            previous_likelihood = reconstructed_likelihood
            continue
        # Standard AIS evaluates the bridge ratio before the stage-t mutation.
        reconstructed_increment = (lambdas[stage] - lambdas[stage - 1]) * previous_likelihood
        increment_error = max(increment_error, close_enough(
            increments[stage - 1], reconstructed_increment, "AIS increment",
        ))
        computed_cumulative[stage] = computed_cumulative[stage - 1] + reconstructed_increment
        cumulative_error = max(cumulative_error, close_enough(
            cumulative[stage], computed_cumulative[stage], "AIS cumulative weight",
        ))
        inc_weights, inc_ess, inc_max, _ = normalized_log_weights(reconstructed_increment)
        del inc_weights
        cumulative_weights, cum_ess, cum_max, _ = normalized_log_weights(computed_cumulative[stage])
        del cumulative_weights
        normalized_error = max(normalized_error, close_enough(
            np.asarray([incremental_ess[stage - 1], incremental_max[stage - 1],
                        cumulative_ess[stage - 1], cumulative_max[stage - 1]]),
            np.asarray([inc_ess, inc_max, cum_ess, cum_max]), "AIS stored summary", atol=2e-10,
        ))
        previous_likelihood = reconstructed_likelihood
    blocks_per_column = (rows + 8 - 1) // 8
    expected_attempts = rows * blocks_per_column * int(context["config"]["method"]["mutation_sweeps"])
    require(np.array_equal(attempts, np.full(attempts.shape, expected_attempts, dtype=np.int32)),
            "mutation attempt count changed")
    require(np.all(changes >= 0) and np.all(changes <= attempts), "mutation changes are invalid")

    final_weights, final_ess, final_max, final_log_mean = normalized_log_weights(computed_cumulative[-1])
    normalized_error = max(normalized_error, close_enough(
        np.asarray(result["final_normalized_weights"], dtype=np.float64), final_weights,
        "final normalized weights", atol=2e-10,
    ))
    normalized_error = max(normalized_error, close_enough(
        np.asarray([
            scalar(result["final_importance_ess"], "final ESS"),
            scalar(result["final_max_normalized_weight"], "final max weight"),
            scalar(result["final_log_mean_weight"], "final log mean weight"),
        ], dtype=np.float64),
        np.asarray([final_ess, final_max, final_log_mean]), "final AIS summary", atol=2e-10,
    ))
    final_bit_weights = np.asarray(
        [sum(int(value).bit_count() for value in row) for row in b_by_stage[-1]], dtype=np.int16,
    )
    require(np.array_equal(np.asarray(result["final_b_bit_weights"], dtype=np.int16), final_bit_weights),
            "final B bit weights changed")
    final_a_syndromes = a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[-1])
    final_likelihood = log_likelihood(final_a_syndromes, log_mass)
    final_target = final_bit_weights.astype(np.float64) * math.log(
        float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"]))
    ) + final_likelihood
    close_enough(result["final_collapsed_log_target"], final_target, "final collapsed target")

    gates = context["config"]["gates"]
    metrics = {
        "base_family": task["base_family"],
        "population_index": task["population_index"],
        "final_importance_ess_fraction": float(final_ess / particles),
        "final_max_normalized_weight": float(final_max),
        "maximum_incremental_normalized_weight": float(incremental_max.max()),
        "minimum_cumulative_ess_fraction": float(cumulative_ess.min() / particles),
        "mean_final_b_bit_weight": float(final_bit_weights.astype(np.float64).mean()),
        "mean_mutation_block_changes": float(changes.mean()),
        "final_log_mean_weight": float(final_log_mean),
        "maximum_likelihood_abs_error": likelihood_error,
        "maximum_increment_abs_error": increment_error,
        "maximum_cumulative_abs_error": cumulative_error,
        "maximum_normalization_abs_error": normalized_error,
        "cumulative_ess_by_stage": [float(particles), *[float(value) for value in cumulative_ess]],
    }
    checks = {
        "final_ess": metrics["final_importance_ess_fraction"] >= gates["min_final_importance_ess_fraction"],
        "final_max_weight": metrics["final_max_normalized_weight"] <= gates["max_final_normalized_weight"],
        "incremental_max_weight": metrics["maximum_incremental_normalized_weight"]
        <= gates["max_incremental_normalized_weight"],
    }
    return {**metrics, "checks": checks, "passes": bool(all(checks.values())),
            "trajectory_digest": str(scalar(raw["trajectory_digest"], "trajectory digest"))}


def load_replay_and_report(manifest, summaries):
    replay_path = RUN_ROOT / "REPLAY.json"
    report_path = RUN_ROOT / "REPORT.json"
    success_path = RUN_ROOT / "SUCCESS.json"
    run_complete_path = RUN_ROOT / "RUN_COMPLETE.json"
    require(replay_path.is_file() and report_path.is_file() and success_path.is_file(),
            "replay, report, or terminal marker is missing")
    require(run_complete_path.is_file() and not (RUN_ROOT / "FAILED.json").exists(),
            "run did not complete cleanly")
    replay = json.loads(replay_path.read_text(encoding="ascii"))
    replay_core = {name: value for name, value in replay.items() if name != "replay_sha256"}
    require(replay["replay_sha256"] == canonical_sha(replay_core), "replay SHA changed")
    require(replay["manifest_sha256"] == manifest["manifest_sha256"], "replay manifest changed")
    expected_replay_tasks = [
        {"task_fingerprint": canonical_sha(task), "trajectory_digest": item["trajectory_digest"]}
        for task, item in zip(manifest["tasks"], summaries)
    ]
    require(replay["tasks"] == expected_replay_tasks, "replay task digest changed")
    report = json.loads(report_path.read_text(encoding="ascii"))
    report_core = {name: value for name, value in report.items() if name != "report_sha256"}
    require(report["report_sha256"] == canonical_sha(report_core), "report SHA changed")
    all_pass = bool(all(item["passes"] for item in summaries))
    expected_status = (
        "LOCAL_COLLAPSED_AIS_PATH_WEIGHT_VIABLE"
        if all_pass else "LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE"
    )
    require(report["manifest_sha256"] == manifest["manifest_sha256"], "report manifest changed")
    require(report["replay_sha256"] == replay["replay_sha256"], "report replay changed")
    require(report["status"] == expected_status and bool(report["all_populations_pass"]) == all_pass,
            "report terminal status changed")
    require(report["forbidden_interpretations"] == [
        "no_q_top_estimate", "no_posterior_estimation", "no_remote_hard2_authorization",
        "no_ready_for_formal_authorization",
    ], "report scope guard changed")
    require(len(report["population_summaries"]) == len(summaries), "report population count changed")
    for reported, independent in zip(report["population_summaries"], summaries):
        require(reported["base_family"] == independent["base_family"], "report family changed")
        require(reported["population_index"] == independent["population_index"], "report index changed")
        for name in (
            "final_importance_ess_fraction", "final_max_normalized_weight",
            "maximum_incremental_normalized_weight", "minimum_cumulative_ess_fraction",
            "mean_final_b_bit_weight", "mean_mutation_block_changes", "final_log_mean_weight",
        ):
            close_enough([reported[name]], [independent[name]], f"report {name}", atol=2e-10)
        require(bool(reported["passes"]) == bool(independent["passes"]), "report gate changed")
    success = json.loads(success_path.read_text(encoding="ascii"))
    require(success == {
        "manifest_sha256": manifest["manifest_sha256"], "status": expected_status,
        "report_sha256": report["report_sha256"],
    }, "terminal marker changed")
    completed = json.loads(run_complete_path.read_text(encoding="ascii"))
    require(completed["manifest_sha256"] == manifest["manifest_sha256"], "run completion manifest changed")
    require([item["task_fingerprint"] for item in completed["tasks"]]
            == [canonical_sha(task) for task in manifest["tasks"]], "run completion tasks changed")
    return replay, report, expected_status


def main():
    output = RUN_ROOT / "INDEPENDENT_AUDIT.json"
    require(not output.exists(), "independent audit already exists")
    context = load_context()
    manifest = load_manifest(context)
    summaries = [audit_task(context, manifest, task) for task in manifest["tasks"]]
    replay, report, terminal_status = load_replay_and_report(manifest, summaries)
    selected_stages = (0, 1, 7, 15, 31, 47, 63)
    ess = np.asarray([item["cumulative_ess_by_stage"] for item in summaries], dtype=np.float64)
    stage_summary = [
        {
            "stage": stage,
            "median_cumulative_importance_ess": float(np.median(ess[:, stage])),
            "minimum_cumulative_importance_ess": float(ess[:, stage].min()),
            "maximum_cumulative_importance_ess": float(ess[:, stage].max()),
        }
        for stage in selected_stages
    ]
    core = {
        "audit_version": AUDIT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "replay_sha256": replay["replay_sha256"],
        "report_sha256": report["report_sha256"],
        "status": "PASS_RAW_ONLY_AUDIT",
        "sampler_called": False,
        "raw_result_status": terminal_status,
        "all_populations_pass": bool(all(item["passes"] for item in summaries)),
        "independent_mass_sha256": context["mass_sha256"],
        "population_summaries": summaries,
        "selected_stage_cumulative_ess_summary": stage_summary,
        "forbidden_interpretations": [
            "no_q_top_estimate", "no_posterior_estimation", "no_remote_hard2_authorization",
            "no_ready_for_formal_authorization",
        ],
    }
    atomic_json(output, {**core, "audit_sha256": canonical_sha(core)})


if __name__ == "__main__":
    main()
