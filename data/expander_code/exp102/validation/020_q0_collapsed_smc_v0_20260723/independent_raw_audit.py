"""Independent raw-only audit for the frozen collapsed-SMC V0 diagnostic.

This verifier does not import or call the SMC sampler.  It derives the hard
cell independently, reads raw NPZ files with allow_pickle=False, and checks
B algebra, likelihoods, weights, systematic parents, and root ancestry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    load_npz_no_pickle,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path("data/expander_code/exp102")
RUN_ROOT = Path(__file__).resolve().parent / "local_m8_smc_v0"
CONFIG_PATH = ROOT / "config/q0_hgp_collapsed_smc.v0.json"
REGISTRY_PATH = ROOT / "registry/registry.json"
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


class AuditConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditConflict(message)


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
    _require(set(result) == RESULT_FIELDS, "engine result fields changed")
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_collapsed_smc.v0.trajectory_digest.v1\0")
    for name in sorted(result):
        value = np.asarray(result[name])
        _require(not value.dtype.hasobject, f"result {name} has object dtype")
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        dtype = value.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _float64_sha(values):
    return hashlib.sha256(np.asarray(values, dtype=">f8").tobytes(order="C")).hexdigest()


def _y_columns(syndrome, H):
    rows, columns = H.shape
    matrix = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    return np.asarray([_bits_to_mask(matrix[:, column]) for column in range(columns)], dtype=np.uint32)


def _a_syndromes_from_b(syndrome, H, b_columns):
    y_columns = _y_columns(syndrome, H)
    result = np.repeat(y_columns[None, :], b_columns.shape[0], axis=0)
    for column in range(H.shape[1]):
        for row in np.flatnonzero(H[:, column]):
            result[:, column] ^= b_columns[:, int(row)]
    return result


def _log_likelihood(a_syndromes, log_mass):
    result = np.zeros(a_syndromes.shape[0], dtype=np.float64)
    for column in range(a_syndromes.shape[1]):
        result += log_mass[a_syndromes[:, column]]
    return result


def _normalized_weights(log_likelihood, delta_lambda):
    log_increment = float(delta_lambda) * np.asarray(log_likelihood, dtype=np.float64)
    maximum = float(log_increment.max())
    values = np.exp(log_increment - maximum)
    total = float(values.sum(dtype=np.float64))
    _require(total > 0.0 and np.isfinite(total), "incremental weights vanished")
    weights = values / total
    cess = 1.0 / float(np.square(weights).sum(dtype=np.float64))
    return weights, cess, float(weights.max()), maximum + np.log(total / weights.size)


def _systematic_parents(weights, offset):
    particles = int(weights.size)
    _require(0.0 <= float(offset) < 1.0 / particles, "resample offset is invalid")
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative[-1] = 1.0
    parents = np.empty(particles, dtype=np.int32)
    parent = 0
    for child in range(particles):
        position = float(offset) + child / particles
        while parent < particles - 1 and position >= cumulative[parent]:
            parent += 1
        parents[child] = parent
    return parents


def _context():
    registry = load_registry(REGISTRY_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    _require(config["registry_sha256"] == registry["registry_sha256"], "registry SHA changed")
    cell = config["cell"]
    _, code, H = load_frozen_code(REGISTRY_PATH, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(cell["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(bool(syndrome.any()), "reconstructed syndrome unexpectedly vanished")
    mass = build_classical_coset_mass(H, cell["p"], engine="numba")
    return {
        "registry": registry,
        "config": config,
        "config_sha256": sha256_json(config),
        "H": np.ascontiguousarray(H, dtype=np.uint8),
        "model": model,
        "frame": frame,
        "uniform_seed": uniform_seed,
        "syndrome": syndrome,
        "mass": mass,
        "mass_sha256": _float64_sha(mass),
    }


def _load_manifest():
    manifest = json.loads((RUN_ROOT / "MANIFEST.json").read_text(encoding="ascii"))
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "manifest SHA changed")
    return manifest


def _task_path(task):
    return RUN_ROOT / "raw" / task["base_family"] / f"p{int(task['population_index']):02d}.npz"


def _audit_task(context, manifest, task):
    path = _task_path(task)
    _require(path.is_file(), "raw task is missing")
    with load_npz_no_pickle(path) as raw:
        expected_fields = {
            "raw_version", "contract_version", "manifest_sha256", "task_fingerprint", "task_json",
            "source_binding_json", "config_sha256", "registry_sha256", "uniform_seed", "syndrome",
            "trajectory_digest", "core_seconds", "wall_seconds",
            *(f"smc_{name}" for name in RESULT_FIELDS),
        }
        _require(set(raw.files) == expected_fields, "raw schema changed")
        _require(str(_scalar(raw["manifest_sha256"], "manifest_sha256")) == manifest["manifest_sha256"],
                 "raw manifest changed")
        _require(str(_scalar(raw["task_fingerprint"], "task_fingerprint")) == sha256_json(task),
                 "raw task fingerprint changed")
        _require(str(_scalar(raw["task_json"], "task_json")) == canonical_json(task),
                 "raw task JSON changed")
        _require(str(_scalar(raw["config_sha256"], "config_sha256")) == context["config_sha256"],
                 "raw config changed")
        _require(str(_scalar(raw["registry_sha256"], "registry_sha256"))
                 == context["registry"]["registry_sha256"], "raw registry changed")
        _require(int(_scalar(raw["uniform_seed"], "uniform_seed")) == context["uniform_seed"],
                 "raw uniform seed changed")
        _require(np.array_equal(raw["syndrome"], context["syndrome"]), "raw syndrome changed")
        result = {name: np.asarray(raw[f"smc_{name}"]) for name in RESULT_FIELDS}
        _require(_trajectory_digest(result) == str(_scalar(raw["trajectory_digest"], "trajectory_digest")),
                 "raw trajectory digest changed")
        _require(str(_scalar(result["mass_sha256"], "mass_sha256")) == context["mass_sha256"],
                 "mass table SHA changed")
        b_by_stage = np.asarray(result["b_columns_by_stage"], dtype=np.uint32)
        likelihood_by_stage = np.asarray(result["log_likelihood_by_stage"], dtype=np.float64)
        roots_by_stage = np.asarray(result["roots_by_stage"], dtype=np.int32)
        weights_by_stage = np.asarray(result["normalized_incremental_weights"], dtype=np.float64)
        parents_by_stage = np.asarray(result["parent_indices"], dtype=np.int32)
        offsets = np.asarray(result["resample_offsets"], dtype=np.float64)
        cess = np.asarray(result["conditional_ess"], dtype=np.float64)
        maximum = np.asarray(result["max_normalized_weight"], dtype=np.float64)
        increments = np.asarray(result["log_normalizer_increments"], dtype=np.float64)
        levels, particles, rows = b_by_stage.shape
        _require((levels, particles, rows) == (64, 128, 24), "B stage shape changed")
        _require(likelihood_by_stage.shape == (64, 128), "likelihood shape changed")
        _require(roots_by_stage.shape == (64, 128), "root shape changed")
        _require(weights_by_stage.shape == parents_by_stage.shape == (63, 128), "stage shape changed")
        _require(offsets.shape == cess.shape == maximum.shape == increments.shape == (63,),
                 "stage summary shape changed")
        _require(np.array_equal(roots_by_stage[0], np.arange(128, dtype=np.int32)),
                 "initial roots changed")
        log_mass = np.log(context["mass"])
        lambdas = tuple(context["config"]["method"]["lambda_values"])
        root_ess_by_stage = []
        for stage in range(levels):
            a_syndromes = _a_syndromes_from_b(context["syndrome"], context["H"], b_by_stage[stage])
            expected_likelihood = _log_likelihood(a_syndromes, log_mass)
            _require(np.array_equal(likelihood_by_stage[stage], expected_likelihood),
                     "stored likelihood changed")
            counts = np.bincount(roots_by_stage[stage], minlength=particles)
            root_ess_by_stage.append(1.0 / float(np.square(counts / particles).sum()))
            if stage == 0:
                continue
            expected_weights, expected_cess, expected_maximum, expected_increment = _normalized_weights(
                likelihood_by_stage[stage - 1], lambdas[stage] - lambdas[stage - 1],
            )
            _require(np.array_equal(weights_by_stage[stage - 1], expected_weights),
                     "incremental weights changed")
            _require(np.array_equal(
                parents_by_stage[stage - 1], _systematic_parents(expected_weights, offsets[stage - 1]),
            ), "systematic parents changed")
            _require(np.array_equal(
                roots_by_stage[stage], roots_by_stage[stage - 1][parents_by_stage[stage - 1]],
            ), "root ancestry changed")
            _require(float(cess[stage - 1]) == expected_cess, "CESS changed")
            _require(float(maximum[stage - 1]) == expected_maximum, "maximum weight changed")
            _require(float(increments[stage - 1]) == expected_increment, "normalizer increment changed")
        final_counts = np.bincount(roots_by_stage[-1], minlength=particles)
        final_fraction = final_counts / particles
        final_ess = 1.0 / float(np.square(final_fraction).sum())
        _require(np.array_equal(final_counts, result["final_root_counts"]), "final root counts changed")
        _require(float(_scalar(result["final_root_family_ess"], "final_root_family_ess")) == final_ess,
                 "final root ESS changed")
        return {
            "base_family": task["base_family"],
            "population_index": task["population_index"],
            "final_root_ess": final_ess,
            "final_distinct_roots": int(np.count_nonzero(final_counts)),
            "final_max_root_fraction": float(final_fraction.max()),
            "minimum_cess_fraction": float(cess.min() / particles),
            "maximum_weight": float(maximum.max()),
            "root_ess_by_stage": root_ess_by_stage,
        }


def main():
    _require(not (RUN_ROOT / "INDEPENDENT_AUDIT.json").exists(), "audit already exists")
    context = _context()
    manifest = _load_manifest()
    _require(manifest["registry_sha256"] == context["registry"]["registry_sha256"], "manifest registry changed")
    _require(manifest["config_sha256"] == context["config_sha256"], "manifest config changed")
    _require(manifest["H_sha256"] == _array_sha256(context["H"]), "manifest H changed")
    _require(manifest["syndrome_sha256"] == _array_sha256(context["syndrome"]), "manifest syndrome changed")
    _require(manifest["classical_mass_sha256"] == context["mass_sha256"], "manifest mass changed")
    summaries = [_audit_task(context, manifest, task) for task in manifest["tasks"]]
    selected_stages = (0, 1, 7, 15, 31, 47, 63)
    root_ess = np.asarray([item["root_ess_by_stage"] for item in summaries], dtype=np.float64)
    stage_summary = [
        {
            "stage": stage,
            "median_root_ess": float(np.median(root_ess[:, stage])),
            "minimum_root_ess": float(root_ess[:, stage].min()),
            "maximum_root_ess": float(root_ess[:, stage].max()),
        }
        for stage in selected_stages
    ]
    core = {
        "audit_version": "exp102.q0_hgp_collapsed_smc.v0.independent_raw_audit.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "status": "PASS_RAW_ONLY_AUDIT",
        "sampler_called": False,
        "population_summaries": summaries,
        "selected_stage_root_ess_summary": stage_summary,
    }
    atomic_json(RUN_ROOT / "INDEPENDENT_AUDIT.json", {**core, "audit_sha256": sha256_json(core)})


if __name__ == "__main__":
    main()
