"""Fail-closed exp102 q=0 population-annealing discovery workflow.

The raw schema, task namespace, schedules, panels, and reports in this module
are discovery-only.  They cannot be consumed by the historical PT pilot or by
the formal production freezer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .labels import bits_to_uint64
from .q0_pa import (
    PA_CONTRACT_VERSION,
    PA_RESAMPLE_ESS_FRACTION,
    PA_SCHEDULE_VERSION,
    PaConflictError,
    PaSeedIdentity,
    Q0PaConfig,
    _family_statistics,
    canonical_population_digest,
    label_distribution_collision,
    pa_config_fingerprint,
    pa_coupling_schedule,
    pa_population_gate,
    population_qtop_jackknife,
    run_q0_pa_population,
    systematic_resampling,
    theta_schedule_q32,
    validate_hard_coset_basis,
    weighted_label_distribution,
)
from .q0_pt import ladder_x_q32_sha256
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


PA_DISCOVERY_VERSION = "exp102.q0_pa.discovery.v1"
PA_RAW_VERSION = "exp102.q0_pa.raw.v1"
PA_REPORT_VERSION = "exp102.q0_pa.report.v1"
PA_TASKS_VERSION = "exp102.q0_pa.tasks.v1"
PA_CONFIRMATION_FREEZE_VERSION = "exp102.q0_pa.confirmation_freeze.v1"
PA_STAGES = ("hard_screen", "rescue", "confirmation", "resolution")
PA_NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
PA_P_VALUES = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
PA_POPULATIONS_PER_CELL = 8
PA_FLOAT_REPLAY_MAX_ULPS = 64
PA_LOG_Z_REPLAY_ULPS_PER_STAGE = 32
PA_PARENT_PT_SOURCE = "da69528b43f4a9d1635083c21d713ba63ccec4ab"
PA_PARENT_PT_RUN = "exp102_discovery_v2_20260720_da69528"

HARD_CELLS = (
    {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
)
CONFIRMATION_CELLS = (
    {"code_id": "m05_c00", "p": 0.04, "disorder_index": 2,
     "disorder_source": "attempt022"},
    {"code_id": "m05_c05", "p": 0.04, "disorder_index": 3,
     "disorder_source": "attempt022"},
    {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m06_c05", "p": 0.04, "disorder_index": 2,
     "disorder_source": "attempt022"},
    {"code_id": "m07_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m07_c00", "p": 0.04, "disorder_index": 1,
     "disorder_source": "attempt022"},
    {"code_id": "m07_c05", "p": 0.04, "disorder_index": 1,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c07", "p": 0.04, "disorder_index": 3,
     "disorder_source": "attempt022"},
    {"code_id": "m05_c04", "p": 0.07, "disorder_index": 1,
     "disorder_source": "fresh_v2"},
    {"code_id": "m05_c06", "p": 0.04, "disorder_index": 1,
     "disorder_source": "fresh_v2"},
    {"code_id": "m06_c06", "p": 0.04, "disorder_index": 2,
     "disorder_source": "fresh_v2"},
    {"code_id": "m06_c05", "p": 0.07, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
    {"code_id": "m07_c01", "p": 0.10, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
    {"code_id": "m07_c07", "p": 0.07, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
    {"code_id": "m08_c02", "p": 0.04, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
    {"code_id": "m08_c03", "p": 0.07, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
)
RESOLUTION_CELLS = (
    *HARD_CELLS,
    {"code_id": "m05_c04", "p": 0.07, "disorder_index": 1,
     "disorder_source": "fresh_v2"},
    {"code_id": "m06_c06", "p": 0.04, "disorder_index": 2,
     "disorder_source": "fresh_v2"},
    {"code_id": "m07_c01", "p": 0.10, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
    {"code_id": "m08_c02", "p": 0.04, "disorder_index": 0,
     "disorder_source": "fresh_v2"},
)
CONFIRMATION_PANEL_SHA256 = "8f2c1a6d60f346ecc5bf703f7e5d0d17d068462f978c78dd937ace0fb98b41be"
RESOLUTION_PANEL_SHA256 = "03f9b16dbc0cc52ee18313cdf57fd25ea4db50f44687971bedac53662b275c22"

BASE_METHODS = (
    {"method_id": "C192-2", "logical_kernel": "coordinate",
     "num_anneal_steps": 192, "rejuvenation_sweeps": 2,
     "num_particles": 256},
    {"method_id": "B96-1", "logical_kernel": "block4",
     "num_anneal_steps": 96, "rejuvenation_sweeps": 1,
     "num_particles": 256},
    {"method_id": "B192-1", "logical_kernel": "block4",
     "num_anneal_steps": 192, "rejuvenation_sweeps": 1,
     "num_particles": 256},
    {"method_id": "B96-2", "logical_kernel": "block4",
     "num_anneal_steps": 96, "rejuvenation_sweeps": 2,
     "num_particles": 256},
)
RESCUE_METHOD = {
    "method_id": "B384-2", "logical_kernel": "block4",
    "num_anneal_steps": 384, "rejuvenation_sweeps": 2,
    "num_particles": 512,
}

PA_RAW_FIELDS = {
    "raw_version", "pa_contract_version", "task_fingerprint", "namespace",
    "stage", "method_id", "population_index", "cell_json", "config_json",
    "seed_identity_json", "source_commit", "registry_sha256",
    "discovery_config_sha256", "trajectory_config_sha256", "model_fingerprint",
    "section_fingerprint", "logical_frame_fingerprint", "population_seed",
    "uniform_seed", "engine", "schedule_q32", "schedule_sha256",
    "schedule_version", "ladder_K", "ladder_p", "final_states_packed",
    "num_qubits", "final_weights", "final_labels", "final_energies",
    "stage_energies", "stage_pre_weights", "stage_post_weights",
    "conditional_ess", "ess_before_decision", "ess_after_decision",
    "max_pre_weight", "resampled", "resampling_offsets", "parents",
    "offspring_counts", "root_ancestry", "mutation_counters",
    "logical_bit_flips", "log_normalizer_increments", "log_z",
    "family_masses", "family_ess", "distinct_initial_families",
    "max_family_mass", "max_hard_coset_residual", "affine_dimension",
    "planted_label", "planted_class_mass", "unique_states", "unique_labels",
    "final_logical_flow", "population_digest", "valid", "failure_reason",
    "core_seconds", "wall_seconds",
}


def _schedule_record(p, steps):
    values = theta_schedule_q32(p, steps)
    return {"q32": list(values), "sha256": ladder_x_q32_sha256(values)}


def _schedule_key(p, steps):
    return f"p={float(p):.2f},G={int(steps)}"


def default_pa_discovery_config(registry):
    schedules = {
        _schedule_key(p, steps): _schedule_record(p, steps)
        for steps in (96, 192, 384) for p in PA_P_VALUES
    }
    config = {
        "discovery_version": PA_DISCOVERY_VERSION,
        "pa_contract_version": PA_CONTRACT_VERSION,
        "raw_version": PA_RAW_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "parent_pt_evidence": {
            "source_commit": PA_PARENT_PT_SOURCE,
            "run_id": PA_PARENT_PT_RUN,
            "attempt": 22,
        },
        "p_values": list(PA_P_VALUES),
        "resample_ess_fraction": PA_RESAMPLE_ESS_FRACTION,
        "population_count": PA_POPULATIONS_PER_CELL,
        "schedule_version": PA_SCHEDULE_VERSION,
        "schedules": schedules,
        "base_methods": [dict(value) for value in BASE_METHODS],
        "rescue_method": dict(RESCUE_METHOD),
        "hard_screen": {
            "cells": [dict(value) for value in HARD_CELLS],
            "max_q_top_mcse": 0.05,
            "max_pair_abs_delta": 0.06,
            "pair_sigma_multiplier": 3.0,
            "pair_sigma_slack": 0.005,
        },
        "confirmation": {
            "cells": [dict(value) for value in CONFIRMATION_CELLS],
            "ordered_panel_sha256": CONFIRMATION_PANEL_SHA256,
            "num_particles": 512,
            "max_q_top_mcse": 0.03,
            "max_abs_delta": 0.04,
            "sigma_multiplier": 3.0,
            "sigma_slack": 0.005,
        },
        "resolution": {
            "cells": [dict(value) for value in RESOLUTION_CELLS],
            "ordered_panel_sha256": RESOLUTION_PANEL_SHA256,
            "num_particles": 256,
            "max_abs_delta": 0.04,
            "sigma_multiplier": 3.0,
            "sigma_slack": 0.005,
        },
        "population_gate": {
            "min_conditional_ess_fraction": 0.70,
            "max_normalized_particle_weight": 0.10,
            "min_post_decision_ess_fraction": 0.50,
            "min_final_family_ess": 4.0,
            "min_distinct_initial_families": 8,
            "max_family_mass": 0.50,
            "min_median_family_ess": 8.0,
            "min_median_distinct_families": 16.0,
        },
        "runtime_gate": {
            "max_m8_particle_sweep_us": 200.0,
            "max_startup_seconds": 120.0,
            "max_population_minutes": 20.0,
            "max_schedule_minutes_with_safety_factor_2": 180.0,
            "hard_pass_deadline_minutes": 120.0,
            "wall_limit_minutes": 240.0,
        },
    }
    if sha256_json(config["confirmation"]["cells"]) != CONFIRMATION_PANEL_SHA256:
        raise AssertionError("confirmation panel SHA does not match the frozen plan")
    if sha256_json(config["resolution"]["cells"]) != RESOLUTION_PANEL_SHA256:
        raise AssertionError("resolution panel SHA does not match the frozen plan")
    return config


def write_default_pa_config(registry_path, output_path):
    config = default_pa_discovery_config(load_registry(registry_path))
    atomic_json(output_path, config)
    return config


def load_pa_discovery_config(path, registry=None):
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    if registry is None:
        if raw.get("discovery_version") != PA_DISCOVERY_VERSION:
            raise ValueError("PA discovery config version mismatch")
        return {**raw, "discovery_config_sha256": sha256_json(raw),
                "config_path": str(Path(path).resolve())}
    expected = default_pa_discovery_config(registry)
    if raw != expected:
        raise ValueError("PA discovery config differs from the frozen protocol")
    return {**raw, "discovery_config_sha256": sha256_json(raw),
            "config_path": str(Path(path).resolve())}


def _method_by_id(config, method_id):
    methods = [*config["base_methods"], config["rescue_method"]]
    matches = [record for record in methods if record["method_id"] == method_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated PA method {method_id!r}")
    return dict(matches[0])


def resolved_pa_config(config, stage, method_id, p):
    method = _method_by_id(config, method_id)
    if stage == "hard_screen" and method_id == config["rescue_method"]["method_id"]:
        raise ValueError("rescue method is not a base hard-screen method")
    if stage == "rescue" and method_id != config["rescue_method"]["method_id"]:
        raise ValueError("rescue stage only permits B384-2")
    if stage in {"confirmation", "resolution"}:
        method["num_particles"] = int(config[stage]["num_particles"])
    schedule = config["schedules"][_schedule_key(p, method["num_anneal_steps"])]
    return Q0PaConfig(
        p_target=float(p),
        num_particles=int(method["num_particles"]),
        num_anneal_steps=int(method["num_anneal_steps"]),
        rejuvenation_sweeps=int(method["rejuvenation_sweeps"]),
        logical_kernel=method["logical_kernel"],
        schedule_q32=tuple(schedule["q32"]),
        schedule_sha256=schedule["sha256"],
    )


def _allowed_cells(config, stage):
    if stage in {"hard_screen", "rescue"}:
        return config["hard_screen"]["cells"]
    return config[stage]["cells"]


def _validate_cell(cell, allowed):
    if not isinstance(cell, dict) or set(cell) != {
            "code_id", "p", "disorder_index", "disorder_source"}:
        raise ValueError("PA cell schema mismatch")
    normalized = {
        "code_id": str(cell["code_id"]),
        "p": float(cell["p"]),
        "disorder_index": int(cell["disorder_index"]),
        "disorder_source": str(cell["disorder_source"]),
    }
    if normalized not in allowed:
        raise ValueError("PA cell is outside the frozen stage panel")
    return normalized


def pa_task_identity(registry, config, source_commit, stage, method_id, cell,
                     population_index):
    if stage not in PA_STAGES:
        raise ValueError("unknown PA discovery stage")
    if re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None:
        raise ValueError("PA source commit must be a full lowercase Git SHA")
    if (isinstance(population_index, bool)
            or not 0 <= int(population_index) < config["population_count"]):
        raise ValueError("PA population index is outside the frozen range")
    cell = _validate_cell(cell, _allowed_cells(config, stage))
    pa_config = resolved_pa_config(config, stage, method_id, cell["p"])
    trajectory_config = {
        "discovery_config_sha256": config["discovery_config_sha256"],
        "stage": stage,
        "method_id": method_id,
        "pa_config": pa_config.as_dict(),
    }
    trajectory_config_sha256 = sha256_json(trajectory_config)
    cell_fingerprint = sha256_json(cell)
    seed_identity = PaSeedIdentity(
        source_commit=source_commit,
        config_sha256=trajectory_config_sha256,
        cell_fingerprint=cell_fingerprint,
        population_index=int(population_index),
        trajectory_namespace=f"q0_pa_discovery_v1_{stage}",
    )
    return {
        "raw_version": PA_RAW_VERSION,
        "pa_contract_version": PA_CONTRACT_VERSION,
        "namespace": f"q0_pa_discovery_v1_{stage}",
        "stage": stage,
        "method_id": method_id,
        "population_index": int(population_index),
        "cell": cell,
        "pa_config": pa_config.as_dict(),
        "pa_config_fingerprint": pa_config_fingerprint(pa_config),
        "seed_identity": seed_identity.as_dict(),
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "trajectory_config_sha256": trajectory_config_sha256,
        "source_commit": source_commit,
        "engine": "numba",
    }


def pa_task_manifest(registry_path, config_path, source_commit, stage, method_ids,
                     output_path):
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    method_ids = [str(value) for value in method_ids]
    if not method_ids or len(method_ids) != len(set(method_ids)):
        raise ValueError("PA task manifest method list is empty or duplicated")
    expected_methods = (
        [record["method_id"] for record in config["base_methods"]]
        if stage == "hard_screen" else
        [config["rescue_method"]["method_id"]] if stage == "rescue" else method_ids
    )
    if method_ids != expected_methods:
        raise ValueError("PA stage method order differs from the frozen protocol")
    tasks = [
        pa_task_identity(registry, config, source_commit, stage, method_id, cell, population)
        for method_id in method_ids
        for cell in _allowed_cells(config, stage)
        for population in range(config["population_count"])
    ]
    fingerprints = [sha256_json(task) for task in tasks]
    if len(fingerprints) != len(set(fingerprints)):
        raise PaConflictError("PA task plan contains duplicate fingerprints")
    manifest = {
        "manifest_version": PA_TASKS_VERSION,
        "stage": stage,
        "method_ids": method_ids,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "tasks": tasks,
    }
    atomic_json(output_path, manifest)
    return manifest


def pa_task_cost(task):
    pa_config = task["pa_config"]
    m = int(task["cell"]["code_id"][1:3])
    return float(
        m * m * int(pa_config["num_particles"])
        * int(pa_config["num_anneal_steps"])
        * max(int(pa_config["rejuvenation_sweeps"]), 1)
    )


def fixed_pa_ownership(tasks, nodes, source_commit, control_sha256, stage):
    nodes = list(nodes)
    if (len(nodes) < 2 or len(nodes) != len(set(nodes))
            or not set(nodes) <= set(PA_NODE_CAPACITY)):
        raise ValueError("PA ownership needs at least two distinct known nodes")
    loads = {node: 0.0 for node in nodes}
    owners = {}
    for task in sorted(tasks, key=lambda value: (-pa_task_cost(value), sha256_json(value))):
        node = min(nodes, key=lambda value: (
            loads[value] / PA_NODE_CAPACITY[value], value,
        ))
        fingerprint = sha256_json(task)
        owners[fingerprint] = node
        loads[node] += pa_task_cost(task)
    identity = {
        "source_commit": source_commit,
        "control_sha256": control_sha256,
        "stage": stage,
        "nodes": nodes,
        "task_owner": owners,
        "method_ids": list(dict.fromkeys(task["method_id"] for task in tasks)),
    }
    return {
        "ownership_version": "exp102.q0_pa.ownership.v1",
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "weighted_load": loads,
        "capacity": {node: PA_NODE_CAPACITY[node] for node in nodes},
    }


def _uniform_seed(registry, code, cell):
    if cell["disorder_source"] == "attempt022":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt22"
    elif cell["disorder_source"] == "fresh_v2":
        namespace = "discovery_v2_fresh_disorder"
    else:
        raise ValueError("unknown PA disorder source")
    return derive_seed(
        namespace, registry["registry_sha256"], code["code_id"],
        cell["disorder_index"], "uniforms",
    )


def _pack_states(states):
    return np.packbits(np.asarray(states, dtype=np.uint8), axis=1, bitorder="little")


def _unpack_states(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=1, count=int(num_qubits),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def run_pa_task(registry_path, config_path, source_commit, task, output_path):
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    expected = pa_task_identity(
        registry, config, source_commit, task.get("stage"), task.get("method_id"),
        task.get("cell"), task.get("population_index"),
    )
    if task != expected:
        raise PaConflictError("PA task identity is noncanonical or tampered")
    fingerprint = sha256_json(expected)
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_pa_raw(
            output_path, registry, config, expected_source_commit=source_commit,
        )
        if record["task_fingerprint"] != fingerprint:
            raise PaConflictError("existing PA raw conflicts with requested task")
        return "reused"

    cell = expected["cell"]
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed = _uniform_seed(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < cell["p"]).astype(np.uint8)
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    pa_config = resolved_pa_config(config, expected["stage"], expected["method_id"], cell["p"])
    seed_identity = PaSeedIdentity(**expected["seed_identity"])
    wall_start = time.monotonic()
    core_start = time.process_time()
    result = run_q0_pa_population(
        model, frame, syndrome, pa_config, seed_identity, engine="numba",
    )
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    valid, failures = pa_population_gate(result)
    planted_label = bits_to_uint64(frame.label_of(epsilon))
    planted_mass = float(result["final_weights"][result["final_labels"] == planted_label].sum())
    packed = _pack_states(result["final_states"])
    unique_states = int(np.unique(packed, axis=0).shape[0])
    unique_labels = int(np.unique(result["final_labels"]).size)
    digest = canonical_population_digest(result)
    atomic_npz(
        output_path,
        raw_version=np.array(PA_RAW_VERSION),
        pa_contract_version=np.array(PA_CONTRACT_VERSION),
        task_fingerprint=np.array(fingerprint),
        namespace=np.array(expected["namespace"]),
        stage=np.array(expected["stage"]),
        method_id=np.array(expected["method_id"]),
        population_index=np.array(expected["population_index"], dtype=np.int16),
        cell_json=np.array(canonical_json(cell)),
        config_json=np.array(canonical_json(pa_config.as_dict())),
        seed_identity_json=np.array(canonical_json(seed_identity.as_dict())),
        source_commit=np.array(source_commit),
        registry_sha256=np.array(registry["registry_sha256"]),
        discovery_config_sha256=np.array(config["discovery_config_sha256"]),
        trajectory_config_sha256=np.array(expected["trajectory_config_sha256"]),
        model_fingerprint=np.array(model.fingerprint()),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
        population_seed=np.array(seed_identity.population_seed, dtype=np.int64),
        uniform_seed=np.array(uniform_seed, dtype=np.int64),
        engine=np.array("numba"),
        schedule_q32=np.asarray(pa_config.schedule_q32, dtype=np.uint64),
        schedule_sha256=np.array(pa_config.schedule_sha256),
        schedule_version=np.array(PA_SCHEDULE_VERSION),
        ladder_K=result["ladder_K"],
        ladder_p=result["ladder_p"],
        final_states_packed=packed,
        num_qubits=np.array(model.num_qubits, dtype=np.int32),
        final_weights=result["final_weights"],
        final_labels=result["final_labels"],
        final_energies=result["final_energies"],
        stage_energies=result["stage_energies"],
        stage_pre_weights=result["stage_pre_weights"],
        stage_post_weights=result["stage_post_weights"],
        conditional_ess=result["conditional_ess"],
        ess_before_decision=result["ess_before_decision"],
        ess_after_decision=result["ess_after_decision"],
        max_pre_weight=result["max_pre_weight"],
        resampled=result["resampled"],
        resampling_offsets=result["resampling_offsets"],
        parents=result["parents"],
        offspring_counts=result["offspring_counts"],
        root_ancestry=result["root_ancestry"],
        mutation_counters=result["mutation_counters"],
        logical_bit_flips=result["logical_bit_flips"],
        log_normalizer_increments=result["log_normalizer_increments"],
        log_z=result["log_z"],
        family_masses=result["family_masses"],
        family_ess=np.array(result["family_ess"]),
        distinct_initial_families=np.array(result["distinct_initial_families"], dtype=np.int32),
        max_family_mass=np.array(result["max_family_mass"]),
        max_hard_coset_residual=np.array(result["max_hard_coset_residual"], dtype=np.int32),
        affine_dimension=np.array(result["affine_dimension"], dtype=np.int32),
        planted_label=np.array(planted_label, dtype=np.uint64),
        planted_class_mass=np.array(planted_mass),
        unique_states=np.array(unique_states, dtype=np.int32),
        unique_labels=np.array(unique_labels, dtype=np.int32),
        final_logical_flow=result["logical_bit_flips"][-1],
        population_digest=np.array(digest),
        valid=np.array(valid),
        failure_reason=np.array(";".join(failures), dtype="U1024"),
        core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds),
    )
    return "computed"


def _scalar(data, field):
    if field not in data or data[field].shape != ():
        raise PaConflictError(f"PA raw scalar is missing or malformed: {field}")
    return data[field].item()


def _array(data, field, shape, dtype=None):
    if field not in data or data[field].shape != tuple(shape):
        raise PaConflictError(f"PA raw array has wrong shape: {field}")
    value = data[field].copy()
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise PaConflictError(f"PA raw array has wrong dtype: {field}")
    return value


def _require_array_equal(name, stored, expected, equal_nan=False):
    if not np.array_equal(stored, expected, equal_nan=equal_nan):
        raise PaConflictError(f"PA raw transcript mismatch: {name}")


def _require_float_replay(
        name, stored, expected, max_ulps=PA_FLOAT_REPLAY_MAX_ULPS):
    """Replay derived floats across libm/NumPy while keeping discrete state exact."""
    stored = np.asarray(stored, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if (stored.shape != expected.shape
            or not np.all(np.isfinite(stored))
            or not np.all(np.isfinite(expected))):
        raise PaConflictError(f"PA raw transcript mismatch: {name}")
    spacing = np.maximum(np.abs(np.spacing(stored)), np.abs(np.spacing(expected)))
    if np.any(np.abs(stored - expected) > float(max_ulps) * spacing):
        raise PaConflictError(f"PA raw transcript mismatch: {name}")


def _validate_transcript(arrays, pa_config, affine_dimension, num_qubits):
    particles = pa_config.num_particles
    steps = pa_config.num_anneal_steps
    K = pa_coupling_schedule(pa_config)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    roots = np.arange(particles, dtype=np.int64)
    _require_array_equal("initial root_ancestry", arrays["root_ancestry"][0], roots)
    if (np.any(arrays["stage_energies"] < 0)
            or np.any(arrays["stage_energies"] > int(num_qubits))):
        raise PaConflictError("PA stage energies are outside the model range")
    expected_log_z = affine_dimension * math.log(2.0)
    _require_float_replay(
        "initial log-Z constant", arrays["log_z"][0], expected_log_z,
    )
    for index in range(steps):
        delta_K = float(K[index + 1] - K[index])
        energy = arrays["stage_energies"][index].astype(np.float64)
        log_factor = -delta_K * energy
        maximum = float(log_factor.max())
        incremental = np.exp(log_factor - maximum)
        mean_factor = float(np.dot(weights, incremental))
        second_factor = float(np.dot(weights, incremental * incremental))
        expected_cess = particles * mean_factor * mean_factor / second_factor
        pre = weights * incremental
        pre /= pre.sum()
        expected_ess = float(1.0 / np.dot(pre, pre))
        expected_max = float(pre.max())
        expected_increment = maximum + math.log(mean_factor)
        expected_log_z += expected_increment
        _require_float_replay(
            "stage_pre_weights", arrays["stage_pre_weights"][index], pre,
        )
        for name, stored, expected in (
                ("conditional_ess", arrays["conditional_ess"][index], expected_cess),
                ("ess_before_decision", arrays["ess_before_decision"][index], expected_ess),
                ("max_pre_weight", arrays["max_pre_weight"][index], expected_max),
                ("log_normalizer_increments", arrays["log_normalizer_increments"][index], expected_increment),
                ("log_z", arrays["log_z"][index + 1], expected_log_z)):
            _require_float_replay(
                name, stored, expected,
                max_ulps=(PA_LOG_Z_REPLAY_ULPS_PER_STAGE * steps
                          if name == "log_z" else PA_FLOAT_REPLAY_MAX_ULPS),
            )
        decision = expected_ess < PA_RESAMPLE_ESS_FRACTION * particles
        if bool(arrays["resampled"][index]) != decision:
            raise PaConflictError("PA resampling decision cannot be replayed")
        if decision:
            offset = float(arrays["resampling_offsets"][index])
            expected_parents = systematic_resampling(pre, offset)
            weights = np.full(particles, 1.0 / particles, dtype=np.float64)
            roots = roots[expected_parents]
        else:
            if float(arrays["resampling_offsets"][index]) != -1.0:
                raise PaConflictError("non-resampled PA stage has an offset")
            expected_parents = np.arange(particles, dtype=np.int64)
            weights = pre.copy()
        _require_array_equal("parents", arrays["parents"][index], expected_parents)
        expected_offspring = np.bincount(expected_parents, minlength=particles)
        _require_array_equal("offspring_counts", arrays["offspring_counts"][index], expected_offspring)
        _require_array_equal("root_ancestry", arrays["root_ancestry"][index + 1], roots)
        _require_float_replay(
            "stage_post_weights", arrays["stage_post_weights"][index], weights,
        )
        expected_after = float(1.0 / np.dot(weights, weights))
        _require_float_replay(
            "ess_after_decision", arrays["ess_after_decision"][index],
            expected_after,
        )


def validate_pa_raw(path, registry, config, expected_source_commit=None):
    path = Path(path)
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise PaConflictError(f"cannot load PA raw {path}: {exc}") from exc
    with context as data:
        if set(data.files) != PA_RAW_FIELDS:
            raise PaConflictError("PA raw schema mismatch")
        if (_scalar(data, "raw_version") != PA_RAW_VERSION
                or _scalar(data, "pa_contract_version") != PA_CONTRACT_VERSION):
            raise PaConflictError("PA raw version mismatch")
        source_commit = str(_scalar(data, "source_commit"))
        if expected_source_commit is not None and source_commit != expected_source_commit:
            raise PaConflictError("PA raw source commit mismatch")
        stage = str(_scalar(data, "stage"))
        method_id = str(_scalar(data, "method_id"))
        population_index = int(_scalar(data, "population_index"))
        cell = json.loads(str(_scalar(data, "cell_json")))
        identity = pa_task_identity(
            registry, config, source_commit, stage, method_id, cell, population_index,
        )
        fingerprint = sha256_json(identity)
        pa_config = resolved_pa_config(config, stage, method_id, cell["p"])
        seed_identity = PaSeedIdentity(**identity["seed_identity"])
        expected_scalars = {
            "task_fingerprint": fingerprint,
            "namespace": identity["namespace"],
            "config_json": canonical_json(pa_config.as_dict()),
            "seed_identity_json": canonical_json(seed_identity.as_dict()),
            "registry_sha256": registry["registry_sha256"],
            "discovery_config_sha256": config["discovery_config_sha256"],
            "trajectory_config_sha256": identity["trajectory_config_sha256"],
            "population_seed": seed_identity.population_seed,
            "engine": "numba",
            "schedule_sha256": pa_config.schedule_sha256,
            "schedule_version": PA_SCHEDULE_VERSION,
        }
        for field, expected in expected_scalars.items():
            if str(_scalar(data, field)) != str(expected):
                raise PaConflictError(f"PA raw identity mismatch: {field}")
        code_id = cell["code_id"]
        if code_id not in code_by_id:
            raise PaConflictError("PA raw references an unknown code")
        code = code_by_id[code_id]
        _, _, H = load_frozen_code(config.get("registry_path", "") or Path(
            config["config_path"]
        ).parents[1] / "registry" / "registry.json", code_id)
        model, frame = build_model(H)
        expected_affine_dimension = validate_hard_coset_basis(model)
        for field, expected in {
                "model_fingerprint": model.fingerprint(),
                "section_fingerprint": code["section_fingerprint"],
                "logical_frame_fingerprint": code["logical_frame_fingerprint"],
                "num_qubits": model.num_qubits,
        }.items():
            if str(_scalar(data, field)) != str(expected):
                raise PaConflictError(f"PA raw model identity mismatch: {field}")
        uniform_seed = _uniform_seed(registry, code, cell)
        if int(_scalar(data, "uniform_seed")) != uniform_seed:
            raise PaConflictError("PA raw disorder seed mismatch")
        if not np.array_equal(
                _array(data, "schedule_q32", (pa_config.num_anneal_steps + 1,), np.uint64),
                np.asarray(pa_config.schedule_q32, dtype=np.uint64)):
            raise PaConflictError("PA raw schedule was tampered")
        K = pa_coupling_schedule(pa_config)
        _require_float_replay(
            "ladder_K", _array(data, "ladder_K", K.shape, np.float64), K,
            max_ulps=8,
        )
        _require_float_replay(
            "ladder_p", _array(data, "ladder_p", K.shape, np.float64),
            1.0 / (1.0 + np.exp(K)),
            max_ulps=8,
        )

        N, G, n, k = (
            pa_config.num_particles, pa_config.num_anneal_steps,
            model.num_qubits, model.k,
        )
        packed = _array(data, "final_states_packed", (N, (n + 7) // 8), np.uint8)
        states = _unpack_states(packed, n)
        if n % 8:
            unused_mask = np.uint8((0xFF << (n % 8)) & 0xFF)
            if np.any(packed[:, -1] & unused_mask):
                raise PaConflictError("PA packed states have nonzero padding bits")
        arrays = {
            "final_weights": _array(data, "final_weights", (N,), np.float64),
            "final_labels": _array(data, "final_labels", (N,), np.uint64),
            "final_energies": _array(data, "final_energies", (N,), np.int64),
            "stage_energies": _array(data, "stage_energies", (G + 1, N), np.int64),
            "stage_pre_weights": _array(data, "stage_pre_weights", (G, N), np.float64),
            "stage_post_weights": _array(data, "stage_post_weights", (G, N), np.float64),
            "conditional_ess": _array(data, "conditional_ess", (G,), np.float64),
            "ess_before_decision": _array(data, "ess_before_decision", (G,), np.float64),
            "ess_after_decision": _array(data, "ess_after_decision", (G,), np.float64),
            "max_pre_weight": _array(data, "max_pre_weight", (G,), np.float64),
            "resampled": _array(data, "resampled", (G,), np.bool_),
            "resampling_offsets": _array(data, "resampling_offsets", (G,), np.float64),
            "parents": _array(data, "parents", (G, N), np.int64),
            "offspring_counts": _array(data, "offspring_counts", (G, N), np.int64),
            "root_ancestry": _array(data, "root_ancestry", (G + 1, N), np.int64),
            "mutation_counters": _array(data, "mutation_counters", (G, 4), np.int64),
            "logical_bit_flips": _array(data, "logical_bit_flips", (G, k), np.int64),
            "log_normalizer_increments": _array(data, "log_normalizer_increments", (G,), np.float64),
            "log_z": _array(data, "log_z", (G + 1,), np.float64),
            "family_masses": _array(data, "family_masses", (N,), np.float64),
        }
        if int(_scalar(data, "affine_dimension")) != expected_affine_dimension:
            raise PaConflictError("PA affine dimension disagrees with the frozen model")
        for name, value in arrays.items():
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise PaConflictError(f"PA raw contains non-finite values: {name}")
        weights = arrays["final_weights"]
        if (np.any(weights < 0.0)
                or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-13)):
            raise PaConflictError("PA final weights are not finite normalized probabilities")
        if not np.array_equal(arrays["final_energies"], states.sum(axis=1, dtype=np.int64)):
            raise PaConflictError("PA final energies disagree with packed states")
        uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(n)
        epsilon = (uniforms < cell["p"]).astype(np.uint8)
        syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
        residual = (model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2).T
        residual ^= syndrome[None, :]
        max_residual = int(residual.sum(axis=1).max(initial=0))
        if int(_scalar(data, "max_hard_coset_residual")) != max_residual:
            raise PaConflictError("PA hard-coset residual was tampered")
        labels = np.asarray([bits_to_uint64(frame.label_of(state)) for state in states])
        _require_array_equal("final_labels", arrays["final_labels"], labels)
        _require_array_equal("final stage energies", arrays["stage_energies"][-1], arrays["final_energies"])
        _validate_transcript(
            arrays, pa_config, expected_affine_dimension, model.num_qubits,
        )

        expected_stabilizer_attempts = (
            N * pa_config.rejuvenation_sweeps * model.stabilizer_rows.shape[0]
        )
        expected_logical_attempts = N * pa_config.rejuvenation_sweeps * (
            k if pa_config.logical_kernel == "coordinate" else (k + 3) // 4
        )
        counters = arrays["mutation_counters"]
        if (np.any(counters < 0)
                or np.any(counters[:, 0] != expected_stabilizer_attempts)
                or np.any(counters[:, 2] != expected_logical_attempts)
                or np.any(counters[:, 1] > counters[:, 0])
                or np.any(counters[:, 3] > counters[:, 2])):
            raise PaConflictError("PA mutation counters are algebraically inconsistent")
        if np.any(arrays["logical_bit_flips"] < 0):
            raise PaConflictError("PA logical flow counters are negative")
        _require_array_equal(
            "final_logical_flow", _array(data, "final_logical_flow", (k,)),
            arrays["logical_bit_flips"][-1],
        )
        family = _family_statistics(arrays["root_ancestry"][-1], weights, N)
        _require_float_replay(
            "family_masses", arrays["family_masses"], family["family_masses"],
        )
        if int(_scalar(data, "distinct_initial_families")) != family[
                "distinct_initial_families"]:
            raise PaConflictError(
                "PA genealogy summary mismatch: distinct_initial_families"
            )
        for field in ("family_ess", "max_family_mass"):
            _require_float_replay(
                f"genealogy summary {field}", _scalar(data, field), family[field],
            )
        planted_label = bits_to_uint64(frame.label_of(epsilon))
        planted_mass = float(weights[labels == planted_label].sum())
        diagnostics = {
            "planted_label": planted_label,
            "unique_states": int(np.unique(packed, axis=0).shape[0]),
            "unique_labels": int(np.unique(labels).size),
        }
        for field, expected in diagnostics.items():
            if _scalar(data, field) != expected:
                raise PaConflictError(f"PA diagnostic mismatch: {field}")
        _require_float_replay(
            "diagnostic planted_class_mass",
            _scalar(data, "planted_class_mass"), planted_mass,
        )
        diagnostics["planted_class_mass"] = planted_mass
        result_for_gate = {
            **arrays,
            "final_states": states,
            "max_hard_coset_residual": max_residual,
            **family,
        }
        valid, failures = pa_population_gate(result_for_gate)
        if bool(_scalar(data, "valid")) != valid or str(_scalar(data, "failure_reason")) != ";".join(failures):
            raise PaConflictError("PA stored population gate was tampered")
        digest = canonical_population_digest(result_for_gate)
        if str(_scalar(data, "population_digest")) != digest:
            raise PaConflictError("PA population digest mismatch")
        for field in ("core_seconds", "wall_seconds"):
            value = float(_scalar(data, field))
            if not np.isfinite(value) or value < 0.0:
                raise PaConflictError("PA timing is invalid")
        distribution = weighted_label_distribution(labels, weights)
        record = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "task_fingerprint": fingerprint,
            "stage": stage,
            "method_id": method_id,
            "population_index": population_index,
            "cell": cell,
            "cell_key": canonical_json(cell),
            "valid": valid,
            "failure_reason": ";".join(failures),
            "family_ess": family["family_ess"],
            "distinct_initial_families": family["distinct_initial_families"],
            "max_family_mass": family["max_family_mass"],
            "planted_class_mass": planted_mass,
            "log_z": float(arrays["log_z"][-1]),
            "unique_states": diagnostics["unique_states"],
            "unique_labels": diagnostics["unique_labels"],
            "core_seconds": float(_scalar(data, "core_seconds")),
            "wall_seconds": float(_scalar(data, "wall_seconds")),
            "distribution": distribution,
            "population_digest": digest,
        }
    return record


def _load_manifest(path, registry, config, expected_stage=None, expected_source=None):
    manifest = json.loads(Path(path).read_text(encoding="ascii"))
    if set(manifest) != {
            "manifest_version", "stage", "method_ids", "source_commit",
            "registry_sha256", "discovery_config_sha256", "tasks"}:
        raise PaConflictError("PA task manifest schema mismatch")
    if (manifest["manifest_version"] != PA_TASKS_VERSION
            or manifest["stage"] not in PA_STAGES
            or (expected_stage is not None and manifest["stage"] != expected_stage)
            or (expected_source is not None and manifest["source_commit"] != expected_source)
            or manifest["registry_sha256"] != registry["registry_sha256"]
            or manifest["discovery_config_sha256"] != config["discovery_config_sha256"]):
        raise PaConflictError("PA task manifest identity mismatch")
    seen = set()
    for task in manifest["tasks"]:
        expected = pa_task_identity(
            registry, config, manifest["source_commit"], manifest["stage"],
            task.get("method_id"), task.get("cell"), task.get("population_index"),
        )
        if task != expected:
            raise PaConflictError("PA task manifest contains a noncanonical task")
        fingerprint = sha256_json(task)
        if fingerprint in seen:
            raise PaConflictError("PA task manifest contains duplicate tasks")
        seen.add(fingerprint)
    return manifest, seen


def _source_identity(value, source_commit):
    fields = {
        "source_commit", "mode", "archive_sha256", "manifest_sha256", "file_count",
    }
    if (not isinstance(value, dict) or set(value) != fields
            or value["source_commit"] != source_commit or value["mode"] != "archive"
            or re.fullmatch(r"[0-9a-f]{64}", str(value["archive_sha256"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(value["manifest_sha256"])) is None
            or isinstance(value["file_count"], bool)
            or not isinstance(value["file_count"], int) or value["file_count"] <= 0):
        raise PaConflictError("PA source archive identity is invalid")
    return dict(value)


def _load_ownership(path, control, control_sha256):
    ownership = json.loads(Path(path).read_text(encoding="ascii"))
    fields = {
        "ownership_version", "source_commit", "control_sha256", "stage", "nodes",
        "task_owner", "method_ids", "stage_fingerprint", "weighted_load", "capacity",
    }
    if set(ownership) != fields:
        raise PaConflictError("PA ownership schema mismatch")
    expected = fixed_pa_ownership(
        control["tasks"], ownership["nodes"], control["source_commit"],
        control_sha256, control["stage"],
    )
    if ownership != expected:
        raise PaConflictError("PA ownership is not the canonical frozen LPT assignment")
    return ownership


def _verified_pa_paths(raw_root, manifest_path, registry, config,
                       expected_stage=None, expected_source=None):
    """Verify remote node manifests, ownership, hashes, and exclusive markers."""
    raw_root = Path(raw_root).resolve()
    control, expected_tasks = _load_manifest(
        manifest_path, registry, config, expected_stage, expected_source,
    )
    control_sha256 = sha256_file(manifest_path)
    evidence = {}
    for path in raw_root.rglob("*.json"):
        digest = sha256_file(path)
        evidence.setdefault(digest, path.resolve())
    evidence[control_sha256] = Path(manifest_path).resolve()
    node_manifests = []
    for path in raw_root.rglob("raw_manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except Exception as exc:
            raise PaConflictError(f"cannot read PA node manifest {path}") from exc
        if value.get("raw_manifest_version") == PA_RAW_VERSION and value.get("stage") == control["stage"]:
            node_manifests.append((path.resolve(), value))
    if not node_manifests:
        return None

    listed = {}
    source_identity = None
    ownership = None
    ownership_sha256 = None
    seen_nodes = set()
    manifest_evidence = []
    for manifest_path_on_disk, manifest in sorted(node_manifests):
        if set(manifest) != {
                "raw_manifest_version", "node", "stage", "stage_fingerprint",
                "source_commit", "control_sha256", "ownership_sha256",
                "source_identity", "files"}:
            raise PaConflictError("PA node raw-manifest schema mismatch")
        node = manifest["node"]
        if (manifest["raw_manifest_version"] != PA_RAW_VERSION
                or node not in PA_NODE_CAPACITY or node in seen_nodes
                or manifest["stage"] != control["stage"]
                or manifest["source_commit"] != control["source_commit"]
                or manifest["control_sha256"] != control_sha256):
            raise PaConflictError("PA node raw-manifest identity mismatch")
        seen_nodes.add(node)
        current_source = _source_identity(manifest["source_identity"], control["source_commit"])
        if source_identity is None:
            source_identity = current_source
        elif source_identity != current_source:
            raise PaConflictError("PA nodes used inconsistent source archives")
        if ownership_sha256 is None:
            ownership_sha256 = manifest["ownership_sha256"]
            ownership_path = evidence.get(ownership_sha256)
            if ownership_path is None:
                raise PaConflictError("PA ownership evidence is missing")
            ownership = _load_ownership(ownership_path, control, control_sha256)
        elif manifest["ownership_sha256"] != ownership_sha256:
            raise PaConflictError("PA nodes reference inconsistent ownership evidence")
        if (manifest["stage_fingerprint"] != ownership["stage_fingerprint"]
                or node not in ownership["nodes"]):
            raise PaConflictError("PA node is outside the frozen stage ownership")

        node_root = manifest_path_on_disk.parent
        status_path = node_root / "stage_status.json"
        success_path = node_root / "SUCCESS"
        if (not status_path.is_file() or not success_path.is_file()
                or (node_root / "RUNNING").exists() or (node_root / "FAILED").exists()):
            raise PaConflictError("PA stage lacks an exclusive SUCCESS marker")
        status = json.loads(status_path.read_text(encoding="ascii"))
        success = json.loads(success_path.read_text(encoding="ascii"))
        manifest_sha = sha256_file(manifest_path_on_disk)
        if (set(status) != {
                "status", "node", "stage_fingerprint", "expected", "computed",
                "reused", "raw_manifest_sha256"}
                or status["status"] != "SUCCESS" or status["node"] != node
                or status["stage_fingerprint"] != ownership["stage_fingerprint"]
                or status["raw_manifest_sha256"] != manifest_sha
                or status["expected"] != len(manifest["files"])
                or status["computed"] + status["reused"] != status["expected"]):
            raise PaConflictError("PA stage status does not bind its raw manifest")
        if (set(success) != {"stage_fingerprint", "completed_utc"}
                or success["stage_fingerprint"] != ownership["stage_fingerprint"]
                or not isinstance(success["completed_utc"], str)
                or not success["completed_utc"]):
            raise PaConflictError("PA SUCCESS marker identity mismatch")

        assigned = {
            fingerprint for fingerprint, owner in ownership["task_owner"].items()
            if owner == node
        }
        manifest_fingerprints = set()
        for item in manifest["files"]:
            if not isinstance(item, dict) or set(item) != {
                    "task_fingerprint", "path", "sha256"}:
                raise PaConflictError("PA node file entry schema mismatch")
            fingerprint = item["task_fingerprint"]
            relative = Path(item["path"])
            raw_path = (node_root / relative).resolve()
            if (fingerprint in manifest_fingerprints or fingerprint not in assigned
                    or relative.is_absolute() or ".." in relative.parts
                    or node_root not in raw_path.parents or raw_path.suffix != ".npz"
                    or not raw_path.is_file()
                    or re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is None
                    or sha256_file(raw_path) != item["sha256"]):
                raise PaConflictError("PA node file coverage/hash mismatch")
            if raw_path in listed:
                raise PaConflictError("PA raw file is listed by multiple node manifests")
            manifest_fingerprints.add(fingerprint)
            listed[raw_path] = fingerprint
        if manifest_fingerprints != assigned:
            raise PaConflictError("PA node manifest does not cover its assigned tasks")
        manifest_evidence.append({
            "path": str(manifest_path_on_disk), "sha256": manifest_sha,
            "node": node, "stage_fingerprint": ownership["stage_fingerprint"],
        })
    if set(seen_nodes) != set(ownership["nodes"]):
        raise PaConflictError("PA stage is missing a frozen ownership node")
    if set(listed.values()) != expected_tasks:
        raise PaConflictError("PA completed stage does not cover the exact task manifest")
    actual = {
        path.resolve() for manifest_path_on_disk, _ in node_manifests
        for path in manifest_path_on_disk.parent.rglob("*.npz")
    }
    if actual != set(listed):
        raise PaConflictError("PA node directories contain unmanifested raw files")
    return {
        "paths": sorted(listed),
        "fingerprints": listed,
        "source_identity": source_identity,
        "ownership_sha256": ownership_sha256,
        "stage_fingerprint": ownership["stage_fingerprint"],
        "manifest_evidence": manifest_evidence,
    }


def load_pa_stage(raw_dir, manifest_path, registry_path, config_path,
                  expected_stage=None, expected_source=None):
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    manifest, expected = _load_manifest(
        manifest_path, registry, config, expected_stage, expected_source,
    )
    verified = _verified_pa_paths(
        raw_dir, manifest_path, registry, config, expected_stage, expected_source,
    )
    paths = (
        verified["paths"] if verified is not None
        else sorted(Path(raw_dir).rglob("*.npz"))
    )
    records = []
    seen = set()
    for path in paths:
        record = validate_pa_raw(path, registry, config, manifest["source_commit"])
        fingerprint = record["task_fingerprint"]
        if fingerprint in seen:
            raise PaConflictError("PA stage contains duplicate task fingerprints")
        if fingerprint not in expected:
            raise PaConflictError("PA stage contains an unexpected task")
        seen.add(fingerprint)
        records.append(record)
    return {
        "manifest": manifest,
        "records": records,
        "expected": len(expected),
        "present": len(seen),
        "missing_fingerprints": sorted(expected - seen),
        "complete": seen == expected,
        "remote_evidence": verified,
    }


def _cell_summary(rows, expected_populations, k, max_mcse=None):
    populations = {row["population_index"]: row for row in rows}
    duplicate = len(populations) != len(rows)
    complete = (
        not duplicate and set(populations) == set(range(expected_populations))
    )
    public = {
        "expected_populations": expected_populations,
        "present_populations": len(populations),
        "complete": complete,
        "all_populations_valid": complete and all(row["valid"] for row in rows),
        "median_family_ess": float(np.median([row["family_ess"] for row in rows])) if rows else math.nan,
        "median_distinct_initial_families": float(np.median([
            row["distinct_initial_families"] for row in rows
        ])) if rows else math.nan,
        "core_seconds": float(sum(row["core_seconds"] for row in rows)),
        "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
    }
    genealogy_pass = (
        complete
        and public["median_family_ess"] >= 8.0
        and public["median_distinct_initial_families"] >= 16.0
    )
    public["median_genealogy_pass"] = genealogy_pass
    if complete:
        estimate = population_qtop_jackknife(
            [populations[index]["distribution"] for index in range(expected_populations)], k,
        )
        public.update({
            "collision_mass": estimate["collision_mass"],
            "q_top": estimate["q_top"],
            "q_top_mcse": estimate["q_top_mcse"],
            "delete_one_q_top": estimate["delete_one_q_top"].tolist(),
            "pair_count": estimate["pair_count"],
            "planted_class_mass": float(np.mean([
                populations[index]["planted_class_mass"]
                for index in range(expected_populations)
            ])),
            "mean_log_z": float(np.mean([
                populations[index]["log_z"] for index in range(expected_populations)
            ])),
            "mean_unique_states": float(np.mean([
                populations[index]["unique_states"] for index in range(expected_populations)
            ])),
            "mean_unique_labels": float(np.mean([
                populations[index]["unique_labels"] for index in range(expected_populations)
            ])),
        })
    else:
        public.update({
            "collision_mass": math.nan, "q_top": math.nan,
            "q_top_mcse": math.nan, "delete_one_q_top": [], "pair_count": 0,
            "planted_class_mass": math.nan, "mean_log_z": math.nan,
            "mean_unique_states": math.nan, "mean_unique_labels": math.nan,
        })
    public["pass"] = (
        public["all_populations_valid"] and genealogy_pass
        and (max_mcse is None or public["q_top_mcse"] <= max_mcse)
    )
    public["status"] = (
        "PASS" if public["pass"] else
        "INCOMPLETE" if not complete else
        "FAILED_CANDIDATE"
    )
    return public


def _summarize_records(stage_data, registry, config, max_mcse=None):
    code_k = {row["code_id"]: int(row["k"]) for row in registry["codes"]}
    grouped = defaultdict(list)
    for row in stage_data["records"]:
        grouped[(row["method_id"], row["cell_key"])].append(row)
    summaries = {}
    for method_id in stage_data["manifest"]["method_ids"]:
        method_cells = {}
        for cell in _allowed_cells(config, stage_data["manifest"]["stage"]):
            key = canonical_json(cell)
            method_cells[key] = {
                "cell": cell,
                **_cell_summary(
                    grouped.get((method_id, key), []), config["population_count"],
                    code_k[cell["code_id"]], max_mcse,
                ),
            }
        summaries[method_id] = method_cells
    return summaries


def _consistent(left, right, absolute_limit, sigma_multiplier, sigma_slack):
    delta = abs(float(left["q_top"]) - float(right["q_top"]))
    sigma_limit = sigma_multiplier * math.sqrt(
        float(left["q_top_mcse"]) ** 2 + float(right["q_top_mcse"]) ** 2
    ) + sigma_slack
    return delta <= absolute_limit and delta <= sigma_limit, delta, sigma_limit


def _method_tiebreak(method, core_seconds):
    return (
        float(core_seconds),
        int(method["num_particles"]) * int(method["num_anneal_steps"])
        * int(method["rejuvenation_sweeps"]),
        int(method["num_anneal_steps"]),
        int(method["rejuvenation_sweeps"]),
        0 if method["logical_kernel"] == "coordinate" else 1,
        method["method_id"],
    )


def _portable_raw_evidence(records, raw_root):
    raw_root = Path(raw_root).resolve()
    evidence = []
    for record in records:
        path = Path(record["path"]).resolve()
        try:
            relative = path.relative_to(raw_root)
        except ValueError as exc:
            raise PaConflictError("PA raw evidence lies outside its stage root") from exc
        evidence.append({"path": relative.as_posix(), "sha256": record["sha256"]})
    return evidence


def analyze_hard_screen(base_raw_dir, base_manifest_path, registry_path, config_path,
                        rescue_raw_dir=None, rescue_manifest_path=None, output_path=None):
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    base = load_pa_stage(
        base_raw_dir, base_manifest_path, registry_path, config_path, "hard_screen",
    )
    summaries = _summarize_records(
        base, registry, config, config["hard_screen"]["max_q_top_mcse"],
    )
    base_pass = [
        method for method, cells in summaries.items()
        if all(cell["pass"] for cell in cells.values())
    ]
    rescue = None
    if rescue_raw_dir is not None or rescue_manifest_path is not None:
        if rescue_raw_dir is None or rescue_manifest_path is None:
            raise ValueError("both rescue raw and manifest must be supplied")
        rescue = load_pa_stage(
            rescue_raw_dir, rescue_manifest_path, registry_path, config_path,
            "rescue", base["manifest"]["source_commit"],
        )
        rescue_summaries = _summarize_records(
            rescue, registry, config, config["hard_screen"]["max_q_top_mcse"],
        )
        summaries.update(rescue_summaries)
    if not base["complete"] or (rescue is not None and not rescue["complete"]):
        status = "INCOMPLETE"
        candidates = []
    elif len(base_pass) == 0:
        if rescue is not None:
            raise PaConflictError("rescue is forbidden when zero base methods pass")
        status = "EXHAUSTED"
        candidates = []
    elif len(base_pass) == 1 and rescue is None:
        status = "RESCUE_REQUIRED"
        candidates = base_pass
    elif len(base_pass) >= 2 and rescue is not None:
        raise PaConflictError("rescue is forbidden when at least two base methods pass")
    else:
        candidates = list(base_pass)
        if rescue is not None:
            rescue_id = config["rescue_method"]["method_id"]
            if all(cell["pass"] for cell in summaries[rescue_id].values()):
                candidates.append(rescue_id)
        status = "CANDIDATES_AVAILABLE" if len(candidates) >= 2 else "EXHAUSTED"

    pair_evidence = []
    compatible_pairs = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            per_cell = []
            all_consistent = True
            for key in summaries[first]:
                passed, delta, sigma_limit = _consistent(
                    summaries[first][key], summaries[second][key],
                    config["hard_screen"]["max_pair_abs_delta"],
                    config["hard_screen"]["pair_sigma_multiplier"],
                    config["hard_screen"]["pair_sigma_slack"],
                )
                per_cell.append({"cell": summaries[first][key]["cell"],
                                 "abs_delta": delta, "sigma_limit": sigma_limit,
                                 "pass": passed})
                all_consistent &= passed
            pair_evidence.append({"methods": [first, second], "cells": per_cell,
                                  "pass": all_consistent})
            if all_consistent:
                compatible_pairs.append((first, second))

    primary = backup = None
    if status == "CANDIDATES_AVAILABLE":
        if not compatible_pairs:
            status = "EXHAUSTED"
        else:
            core_by_method = {
                method: sum(cell["core_seconds"] for cell in summaries[method].values())
                for method in candidates
            }
            ordered = sorted(
                candidates,
                key=lambda value: _method_tiebreak(
                    _method_by_id(config, value), core_by_method[value],
                ),
            )
            for first_index, first in enumerate(ordered):
                for second in ordered[first_index + 1:]:
                    if ((first, second) in compatible_pairs
                            or (second, first) in compatible_pairs):
                        primary, backup = first, second
                        break
                if primary is not None:
                    break
            status = "READY_FOR_CONFIRMATION"

    report = {
        "report_version": PA_REPORT_VERSION,
        "report_kind": "hard_screen",
        "status": status,
        "source_commit": base["manifest"]["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "base_complete": base["complete"],
        "rescue_complete": None if rescue is None else rescue["complete"],
        "base_passing_methods": base_pass,
        "candidate_methods": candidates,
        "method_cells": summaries,
        "pair_evidence": pair_evidence,
        "primary": primary,
        "backup": backup,
        "raw_evidence": [
            *_portable_raw_evidence(base["records"], base_raw_dir),
            *([] if rescue is None else _portable_raw_evidence(
                rescue["records"], rescue_raw_dir,
            )),
        ],
    }
    report["analysis_sha256"] = sha256_json({
        key: report[key] for key in report if key not in {"raw_evidence"}
    })
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def freeze_confirmation_manifests(hard_report_path, registry_path, config_path,
                                  confirmation_manifest_path, resolution_manifest_path,
                                  freeze_path):
    frozen_outputs = [
        Path(confirmation_manifest_path), Path(resolution_manifest_path), Path(freeze_path),
    ]
    if any(path.exists() for path in frozen_outputs):
        raise FileExistsError("confirmation/resolution freeze outputs must not pre-exist")
    hard_path = Path(hard_report_path)
    report = json.loads(hard_path.read_text(encoding="ascii"))
    expected_report_fields = {
        "report_version", "report_kind", "status", "source_commit",
        "registry_sha256", "discovery_config_sha256", "base_complete",
        "rescue_complete", "base_passing_methods", "candidate_methods",
        "method_cells", "pair_evidence", "primary", "backup", "raw_evidence",
        "analysis_sha256",
    }
    analysis_payload = {
        key: report[key] for key in report
        if key not in {"raw_evidence", "analysis_sha256"}
    }
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    selected = [report.get("primary"), report.get("backup")]
    selected_pair = next((
        pair for pair in report.get("pair_evidence", [])
        if set(pair.get("methods", [])) == set(selected)
    ), None)
    expected_cell_keys = {
        canonical_json(cell) for cell in config["hard_screen"]["cells"]
    }
    selected_cells_valid = all(
        set(report.get("method_cells", {}).get(method, {})) == expected_cell_keys
        and all(
            cell.get("pass")
            for cell in report["method_cells"][method].values()
        )
        for method in selected
    )
    selected_pair_valid = (
        selected_pair is not None and selected_pair.get("pass")
        and {
            canonical_json(cell.get("cell"))
            for cell in selected_pair.get("cells", []) if isinstance(cell, dict)
        } == expected_cell_keys
        and all(cell.get("pass") for cell in selected_pair.get("cells", []))
    )
    if (set(report) != expected_report_fields
            or report.get("report_version") != PA_REPORT_VERSION
            or report.get("report_kind") != "hard_screen"
            or report.get("status") != "READY_FOR_CONFIRMATION"
            or re.fullmatch(r"[0-9a-f]{40}", str(report.get("source_commit"))) is None
            or report.get("registry_sha256") != registry["registry_sha256"]
            or report.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or report.get("analysis_sha256") != sha256_json(analysis_payload)
            or not selected[0] or not selected[1] or selected[0] == selected[1]
            or not set(selected) <= set(report.get("candidate_methods", []))
            or not selected_cells_valid or not selected_pair_valid):
        raise ValueError("hard-screen report is not eligible for confirmation freeze")
    methods = selected
    confirmation = pa_task_manifest(
        registry_path, config_path, report["source_commit"], "confirmation",
        methods, confirmation_manifest_path,
    )
    resolution = pa_task_manifest(
        registry_path, config_path, report["source_commit"], "resolution",
        methods, resolution_manifest_path,
    )
    freeze = {
        "freeze_version": PA_CONFIRMATION_FREEZE_VERSION,
        "source_commit": report["source_commit"],
        "hard_report_sha256": sha256_file(hard_path),
        "hard_analysis_sha256": report["analysis_sha256"],
        "primary": report["primary"],
        "backup": report["backup"],
        "confirmation_panel_sha256": CONFIRMATION_PANEL_SHA256,
        "resolution_panel_sha256": RESOLUTION_PANEL_SHA256,
        "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path),
        "resolution_manifest_sha256": sha256_file(resolution_manifest_path),
        "confirmation_task_count": len(confirmation["tasks"]),
        "resolution_task_count": len(resolution["tasks"]),
    }
    atomic_json(freeze_path, freeze)
    return freeze


def analyze_confirmation(confirmation_raw_dir, confirmation_manifest_path,
                         resolution_raw_dir, resolution_manifest_path, freeze_path,
                         registry_path, config_path, output_path=None):
    registry = load_registry(registry_path)
    config = load_pa_discovery_config(config_path, registry)
    freeze = json.loads(Path(freeze_path).read_text(encoding="ascii"))
    if set(freeze) != {
            "freeze_version", "source_commit", "hard_report_sha256",
            "hard_analysis_sha256", "primary", "backup",
            "confirmation_panel_sha256", "resolution_panel_sha256",
            "confirmation_manifest_sha256", "resolution_manifest_sha256",
            "confirmation_task_count", "resolution_task_count"}:
        raise PaConflictError("PA confirmation freeze schema mismatch")
    if (freeze["freeze_version"] != PA_CONFIRMATION_FREEZE_VERSION
            or re.fullmatch(r"[0-9a-f]{40}", str(freeze["source_commit"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(freeze["hard_report_sha256"])) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(freeze["hard_analysis_sha256"])) is None
            or not freeze["primary"] or not freeze["backup"]
            or freeze["primary"] == freeze["backup"]
            or freeze["confirmation_panel_sha256"] != CONFIRMATION_PANEL_SHA256
            or freeze["resolution_panel_sha256"] != RESOLUTION_PANEL_SHA256
            or sha256_file(confirmation_manifest_path) != freeze["confirmation_manifest_sha256"]
            or sha256_file(resolution_manifest_path) != freeze["resolution_manifest_sha256"]):
        raise PaConflictError("PA confirmation freeze identity mismatch")
    confirmation = load_pa_stage(
        confirmation_raw_dir, confirmation_manifest_path, registry_path, config_path,
        "confirmation", freeze["source_commit"],
    )
    resolution = load_pa_stage(
        resolution_raw_dir, resolution_manifest_path, registry_path, config_path,
        "resolution", freeze["source_commit"],
    )
    expected_methods = [freeze["primary"], freeze["backup"]]
    if (confirmation["manifest"]["method_ids"] != expected_methods
            or resolution["manifest"]["method_ids"] != expected_methods
            or freeze["confirmation_task_count"]
            != len(confirmation["manifest"]["tasks"])
            or freeze["resolution_task_count"] != len(resolution["manifest"]["tasks"])
            or freeze["confirmation_task_count"] != 272
            or freeze["resolution_task_count"] != 96):
        raise PaConflictError("PA confirmation method order differs from the freeze")
    confirmation_cells = _summarize_records(
        confirmation, registry, config, config["confirmation"]["max_q_top_mcse"],
    )
    resolution_cells = _summarize_records(resolution, registry, config, None)
    complete = confirmation["complete"] and resolution["complete"]
    numerical_pass = complete and all(
        cell["pass"] for method in expected_methods
        for cell in confirmation_cells[method].values()
    ) and all(
        cell["pass"] for method in expected_methods
        for cell in resolution_cells[method].values()
    )
    cross_method = []
    for key in confirmation_cells[freeze["primary"]]:
        passed, delta, sigma_limit = _consistent(
            confirmation_cells[freeze["primary"]][key],
            confirmation_cells[freeze["backup"]][key],
            config["confirmation"]["max_abs_delta"],
            config["confirmation"]["sigma_multiplier"],
            config["confirmation"]["sigma_slack"],
        )
        cross_method.append({
            "cell": confirmation_cells[freeze["primary"]][key]["cell"],
            "abs_delta": delta, "sigma_limit": sigma_limit, "pass": passed,
        })
        numerical_pass &= passed
    cross_resolution = []
    for method in expected_methods:
        for key, low in resolution_cells[method].items():
            high = confirmation_cells[method][key]
            passed, delta, sigma_limit = _consistent(
                low, high, config["resolution"]["max_abs_delta"],
                config["resolution"]["sigma_multiplier"],
                config["resolution"]["sigma_slack"],
            )
            cross_resolution.append({
                "method_id": method, "cell": low["cell"], "abs_delta": delta,
                "sigma_limit": sigma_limit, "pass": passed,
            })
            numerical_pass &= passed
    status = "INCOMPLETE" if not complete else (
        "READY_FOR_FORMAL" if numerical_pass else "EXHAUSTED"
    )
    report = {
        "report_version": PA_REPORT_VERSION,
        "report_kind": "confirmation",
        "status": status,
        "source_commit": freeze["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "confirmation_freeze_sha256": sha256_file(freeze_path),
        "primary": freeze["primary"], "backup": freeze["backup"],
        "confirmation_complete": confirmation["complete"],
        "resolution_complete": resolution["complete"],
        "confirmation_cells": confirmation_cells,
        "resolution_cells": resolution_cells,
        "cross_method_consistency": cross_method,
        "cross_resolution_consistency": cross_resolution,
        "formal_config_ready": status == "READY_FOR_FORMAL",
        "frozen_held_out_pass": False,
        "raw_evidence": [
            *_portable_raw_evidence(
                confirmation["records"], confirmation_raw_dir,
            ),
            *_portable_raw_evidence(resolution["records"], resolution_raw_dir),
        ],
    }
    report["analysis_sha256"] = sha256_json({
        key: report[key] for key in report if key != "raw_evidence"
    })
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-config")
    make.add_argument("registry"); make.add_argument("output")
    plan = sub.add_parser("plan")
    plan.add_argument("registry"); plan.add_argument("config"); plan.add_argument("source_commit")
    plan.add_argument("stage", choices=PA_STAGES); plan.add_argument("methods_json")
    plan.add_argument("output")
    run = sub.add_parser("run-task")
    run.add_argument("registry"); run.add_argument("config"); run.add_argument("source_commit")
    run.add_argument("task_json"); run.add_argument("output")
    hard = sub.add_parser("analyze-hard")
    hard.add_argument("base_raw"); hard.add_argument("base_manifest")
    hard.add_argument("registry"); hard.add_argument("config"); hard.add_argument("output")
    hard.add_argument("--rescue-raw"); hard.add_argument("--rescue-manifest")
    freeze = sub.add_parser("freeze-confirmation")
    freeze.add_argument("hard_report"); freeze.add_argument("registry"); freeze.add_argument("config")
    freeze.add_argument("confirmation_manifest"); freeze.add_argument("resolution_manifest")
    freeze.add_argument("output")
    confirm = sub.add_parser("analyze-confirmation")
    confirm.add_argument("confirmation_raw"); confirm.add_argument("confirmation_manifest")
    confirm.add_argument("resolution_raw"); confirm.add_argument("resolution_manifest")
    confirm.add_argument("freeze"); confirm.add_argument("registry"); confirm.add_argument("config")
    confirm.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "make-config":
        result = write_default_pa_config(args.registry, args.output)
    elif args.command == "plan":
        methods = json.loads(Path(args.methods_json).read_text(encoding="ascii"))
        result = pa_task_manifest(
            args.registry, args.config, args.source_commit, args.stage, methods, args.output,
        )
    elif args.command == "run-task":
        task = json.loads(Path(args.task_json).read_text(encoding="ascii"))
        result = run_pa_task(
            args.registry, args.config, args.source_commit, task, args.output,
        )
    elif args.command == "analyze-hard":
        result = analyze_hard_screen(
            args.base_raw, args.base_manifest, args.registry, args.config,
            args.rescue_raw, args.rescue_manifest, args.output,
        )
    elif args.command == "freeze-confirmation":
        result = freeze_confirmation_manifests(
            args.hard_report, args.registry, args.config, args.confirmation_manifest,
            args.resolution_manifest, args.output,
        )
    else:
        result = analyze_confirmation(
            args.confirmation_raw, args.confirmation_manifest, args.resolution_raw,
            args.resolution_manifest, args.freeze, args.registry, args.config, args.output,
        )
    print(sha256_json(result) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
