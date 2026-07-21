"""Isolated HARD2+EASY3 diagnostic screen for the exp102 q=0 samplers.

This module deliberately has no route to ``READY_FOR_FORMAL``.  It reuses the
frozen q=0 kernels, but owns its config, task, raw, seed, manifest, report, and
terminal-decision namespaces.  Consequently none of its artifacts can be
mistaken for evidence from ``exp102.q0_global.discovery.v1``.
"""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

import numpy as np

from . import global_discovery as _global
from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .q0_global import (
    CHARACTER_SET_VERSION,
    DEFECT_METHODS,
    HARD_METHODS,
    DefectTraceConfig,
    GlobalConflictError,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    frozen_character_set,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
    uniform_hard_coset_state,
)
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


SCREEN_DIAGNOSTIC_VERSION = "exp102.q0_global.screen_diagnostic.v1"
SCREEN_CONFIG_VERSION = "exp102.q0_global.screen_diagnostic.config.v1"
SCREEN_TASKS_VERSION = "exp102.q0_global.screen_diagnostic.tasks.v1"
SCREEN_HARD_RAW_VERSION = "exp102.q0_global.screen_diagnostic.hardcoset.raw.v1"
SCREEN_DEFECT_RAW_VERSION = "exp102.q0_global.screen_diagnostic.defect_trace.raw.v1"
SCREEN_BIAS_RAW_VERSION = "exp102.q0_global.screen_diagnostic.defect_bias.raw.v1"
SCREEN_REPORT_VERSION = "exp102.q0_global.screen_diagnostic.report.v1"
SCREEN_DECISION_VERSION = "exp102.q0_global.screen_diagnostic.decision.v1"
SCREEN_SCHEDULE_VERSION = "exp102.q0_global.screen_diagnostic.schedule.v1"
SCREEN_OWNERSHIP_VERSION = "exp102.q0_global.screen_diagnostic.ownership.v1"
SCREEN_SEED_ROOT = "q0_global_screen_diagnostic_v1"
SCREEN_STAGE = "screen"
PARENT_DISCOVERY_CONFIG_SHA256 = "1d0a453f2bf8445ad6587c612c2eabb3049e76e2d73b59c230b8b1358b06e565"
TERMINATED_PARENT_RUN = "exp102_q0_global_20260721_204b37d"
TERMINATED_PARENT_SOURCE = "204b37d8e00e7d11ffa2b6766b90d947892e179d"
INIT_FAMILIES = ("P", "U")
TRAJECTORIES_PER_FAMILY = 16
NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
RESOURCE_TIERS = {
    "T1": {"burn_sweeps": 2048, "measurement_sweeps": 8192},
    "T2": {"burn_sweeps": 4096, "measurement_sweeps": 16384},
    "T3": {"burn_sweeps": 8192, "measurement_sweeps": 32768},
}

HARD_CELLS = (
    {"code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
    {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
     "disorder_source": "attempt022"},
)
EASY_CELLS = (
    {"code_id": "m03_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m04_c00", "p": 0.07, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m05_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
)
HARD_PANEL_SHA256 = "32c3407c1483af1f9848d8beb2fd51498ff8f915cf0e28f4703f3f2a388ffbef"
EASY_PANEL_SHA256 = "ec110e8550b18064c747fd2418c5134b594d693e70edcef83b093b74cdf162b2"

HARD_RAW_FIELDS = {
    "raw_version", "contract_version", "task_fingerprint", "task_json",
    "cell_json", "sampler_config_json", "seed_identity_json", "source_commit",
    "registry_sha256", "screen_config_sha256", "uniform_seed", "engine",
    "model_fingerprint", "section_fingerprint", "logical_frame_fingerprint",
    "catalog_sha256", "joint_sha256", "character_version", "character_masks",
    "character_sha256", "num_qubits", "k", "initial_state_packed",
    "burn_state_packed", "final_state_packed", "measurement_states_packed",
    "burn_labels", "measurement_labels", "measurement_weights",
    "measurement_residual_weights", "measurement_block", "burn_counters",
    "measurement_counters", "burn_basis_seen", "initial_label", "burn_label",
    "final_label", "trajectory_digest", "core_seconds", "wall_seconds",
}
DEFECT_RAW_FIELDS = {
    *(HARD_RAW_FIELDS - {
        "raw_version", "catalog_sha256", "joint_sha256",
        "measurement_residual_weights",
    }),
    "raw_version", "bias_task_fingerprint", "bias_raw_sha256", "bias_sha256",
    "bias", "measurement_defect_counts", "fixed_clock_d0_mask",
    "boundary_occupancy",
}
BIAS_RAW_FIELDS = {
    "raw_version", "contract_version", "task_fingerprint", "task_json",
    "cell_json", "sampler_config_json", "source_commit", "registry_sha256",
    "screen_config_sha256", "uniform_seed", "engine", "model_fingerprint",
    "num_qubits", "tuning_seed_identities_json", "bias", "bias_trace",
    "tuning_histogram", "tuning_final_states_packed", "tuning_final_residuals",
    "tuning_final_defects", "gammas", "bias_sha256", "core_seconds",
    "wall_seconds",
}


def _cell(code_id, p, disorder_index, source):
    return {
        "code_id": str(code_id),
        "p": float(p),
        "disorder_index": int(disorder_index),
        "disorder_source": str(source),
    }


def _normalize_cell(cell):
    if not isinstance(cell, dict) or set(cell) != {
            "code_id", "p", "disorder_index", "disorder_source"}:
        raise ValueError("screen diagnostic cell schema mismatch")
    return _cell(
        cell["code_id"], cell["p"], cell["disorder_index"],
        cell["disorder_source"],
    )


def _screen_cells(config):
    return [
        *config["panels"]["HARD2"]["cells"],
        *config["panels"]["EASY3"]["cells"],
    ]


def _validate_source_commit(source_commit):
    if re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None:
        raise ValueError("diagnostic source commit must be a full lowercase Git SHA")


def _validate_sha256(name, value):
    if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
        raise ValueError(f"diagnostic {name} must be a lowercase SHA256")


def _uniform_seed_for_cell(registry, code, cell):
    """Retain the frozen disorder itself while isolating every sampler seed."""
    cell = _normalize_cell(cell)
    if cell["disorder_source"] == "attempt022":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt22"
    elif cell["disorder_source"] == "global_fresh_v1":
        namespace = "q0_global_discovery_fresh_v1"
    else:
        raise ValueError("unknown screen diagnostic disorder source")
    return derive_seed(
        namespace, registry["registry_sha256"], code["code_id"],
        cell["disorder_index"], "uniforms",
    )


uniform_seed_for_cell = _uniform_seed_for_cell


def default_screen_diagnostic_config(registry):
    hard = [dict(value) for value in HARD_CELLS]
    easy = [dict(value) for value in EASY_CELLS]
    if sha256_json(hard) != HARD_PANEL_SHA256:
        raise AssertionError("HARD2 ordered panel changed")
    if sha256_json(easy) != EASY_PANEL_SHA256:
        raise AssertionError("EASY3 ordered panel changed")
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    uniform_seeds = []
    for value in [*hard, *easy]:
        uniform_seeds.append({
            "cell_fingerprint": sha256_json(value),
            "uniform_seed": _uniform_seed_for_cell(
                registry, code_by_id[value["code_id"]], value,
            ),
        })
    config = {
        "config_version": SCREEN_CONFIG_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "task_version": SCREEN_TASKS_VERSION,
        "hard_raw_version": SCREEN_HARD_RAW_VERSION,
        "defect_raw_version": SCREEN_DEFECT_RAW_VERSION,
        "defect_bias_raw_version": SCREEN_BIAS_RAW_VERSION,
        "report_version": SCREEN_REPORT_VERSION,
        "decision_version": SCREEN_DECISION_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "scope": {
            "purpose": "diagnostic_sampler_screen_only",
            "formal_authorization": False,
            "formal_readiness_authorized": False,
            "ti_in_scope": False,
            "held_out_in_scope": False,
            "production_authorization": False,
            "production_authorized": False,
            "maximum_terminal_status": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
            "excluded_work": [
                "full_sector_ti", "method_selection", "confirmation",
                "resolution", "held_out", "production",
            ],
        },
        "parent_exhausted_discovery": {
            "config_sha256": PARENT_DISCOVERY_CONFIG_SHA256,
            "run_id": TERMINATED_PARENT_RUN,
            "source_commit": TERMINATED_PARENT_SOURCE,
            "status": "RUNTIME_EXHAUSTED",
            "raw_reuse_allowed": False,
        },
        "seed_root": SCREEN_SEED_ROOT,
        "catalog": {"max_multiple": 8, "max_count": 512},
        "characters": {
            "version": CHARACTER_SET_VERSION,
            "full_max_k": 10,
            "num_nonbasis": 4096,
            "seed_namespace": "q0_global_screen_diagnostic_characters_v1",
        },
        "hard_methods": [
            {"method_id": "RC8-QC1", "kernel": "cluster", "cluster_repeats": 1},
            {"method_id": "RC8-QC4", "kernel": "cluster", "cluster_repeats": 4},
            {"method_id": "RC8-J08", "kernel": "joint", "block_size": 8},
            {"method_id": "RC8-J12", "kernel": "joint", "block_size": 12},
            {"method_id": "RC8-J16", "kernel": "joint", "block_size": 16,
             "requires_runtime_gate": True},
        ],
        "defect_methods": [
            {"method_id": "DT16", "dmax": 16},
            {"method_id": "DT32", "dmax": 32},
            {"method_id": "DT64", "dmax": 64},
        ],
        "defect_tuning": {
            "num_chains": 8,
            "num_sweeps": 4096,
            "target_d0": 0.25,
            "target_tail_total": 0.75,
            "gamma": "min(.1,.5/(t+10)^.6)",
            "K_q": 0.0,
        },
        "resource_tiers": RESOURCE_TIERS,
        "resource_selection": {
            "sampler_timing_only": True,
            "max_trajectory_hours": 2.0,
            "safety_factor": 2.0,
            "capacity_nodes": ["nd-1", "nd-3"],
            "capacity": NODE_CAPACITY["nd-1"] + NODE_CAPACITY["nd-3"],
        },
        "trajectory_count_per_init_family": TRAJECTORIES_PER_FAMILY,
        "init_families": list(INIT_FAMILIES),
        "panels": {
            "HARD2": {"cells": hard, "ordered_panel_sha256": sha256_json(hard)},
            "EASY3": {"cells": easy, "ordered_panel_sha256": sha256_json(easy)},
        },
        "screen_panel_sha256": sha256_json([*hard, *easy]),
        "uniform_seeds": uniform_seeds,
        "uniform_seeds_sha256": sha256_json(uniform_seeds),
        "gates": {
            "max_q_top_se": 0.03,
            "max_abs_delta_q_top": 0.04,
            "delta_sigma_multiplier": 3.0,
            "delta_sigma_slack": 0.005,
            "max_d2_upper": 0.04,
            "max_normalized_weight_delta": 0.01,
            "max_rhat": 1.05,
            "min_bulk_ess": 400.0,
            "diagnostic_nonbasis_characters": 64,
            "worm_min_d0_per_trajectory": 200,
            "worm_min_excursions_per_trajectory": 50,
            "worm_min_median_d0_ess": 50.0,
            "worm_min_family_d0_ess": 800.0,
            "worm_max_boundary_occupancy": 0.10,
        },
        "schedule": {
            "preflight_deadline_hour": 8,
            "bias_deadline_hour": 12,
            "measurement_deadline_hour": 22,
            "analysis_deadline_hour": 24,
            "wall_limit_hours": 24,
        },
    }
    return config


def write_default_screen_diagnostic_config(registry_path, output_path):
    config = default_screen_diagnostic_config(load_registry(registry_path))
    atomic_json(output_path, config)
    return config


def load_screen_diagnostic_config(path, registry=None):
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    if registry is None:
        if (raw.get("config_version") != SCREEN_CONFIG_VERSION
                or raw.get("contract_version") != SCREEN_DIAGNOSTIC_VERSION):
            raise ValueError("screen diagnostic config version mismatch")
    else:
        expected = default_screen_diagnostic_config(registry)
        if raw != expected:
            raise ValueError("screen diagnostic config differs from frozen protocol")
    return {
        **raw,
        "screen_config_sha256": sha256_json(raw),
        "config_path": str(Path(path).resolve()),
    }


def _resource_values(config, tier):
    tier = str(tier)
    if tier not in config["resource_tiers"]:
        raise ValueError("unknown screen diagnostic resource tier")
    values = config["resource_tiers"][tier]
    return int(values["burn_sweeps"]), int(values["measurement_sweeps"])


def resolved_sampler_config(config, method_id, p, resource_tier):
    burn, measurement = _resource_values(config, resource_tier)
    if method_id in HARD_METHODS:
        return HardCosetConfig(method_id, p, burn, measurement)
    if method_id in DEFECT_METHODS:
        tuning = config["defect_tuning"]
        return DefectTraceConfig(
            method_id, p, burn, measurement,
            tuning_chains=tuning["num_chains"],
            tuning_sweeps=tuning["num_sweeps"],
        )
    raise ValueError("unknown screen diagnostic sampler method")


@dataclass(frozen=True)
class ScreenSeedIdentity:
    seed_root: str
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
        if self.seed_root != SCREEN_SEED_ROOT:
            raise ValueError("screen diagnostic seed root mismatch")
        _validate_source_commit(self.source_commit)
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            _validate_sha256(name, getattr(self, name))
        if self.init_family not in (*INIT_FAMILIES, "TUNE"):
            raise ValueError("unknown screen diagnostic initialization family")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("screen diagnostic trajectory index is invalid")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            self.seed_root, self.source_commit, self.config_sha256,
            self.registry_sha256, self.cell_fingerprint, self.method_id,
            self.resource_tier, self.init_family, int(self.trajectory_index),
            self.trajectory_namespace, str(stage), str(role), int(index),
        )

    def as_dict(self):
        return {
            "seed_root": self.seed_root,
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


def _validate_task_cell(config, stage, cell):
    if stage != SCREEN_STAGE:
        raise ValueError("screen diagnostic has only the screen stage")
    cell = _normalize_cell(cell)
    if cell not in _screen_cells(config):
        raise ValueError("task cell is outside HARD2+EASY3")
    return cell


def character_seed(config, registry_sha256, code_id):
    return derive_seed(
        config["characters"]["seed_namespace"], registry_sha256, code_id,
    )


def diagnostic_task_identity(registry, config, source_commit, stage, method_id,
                             resource_tier, cell, init_family,
                             trajectory_index, *, bias_binding=None):
    _validate_source_commit(source_commit)
    cell = _validate_task_cell(config, stage, cell)
    method_id = str(method_id)
    resource_tier = str(resource_tier)
    if method_id not in (*HARD_METHODS, *DEFECT_METHODS):
        raise ValueError("unknown screen diagnostic task method")
    if init_family not in INIT_FAMILIES:
        raise ValueError("invalid screen diagnostic initialization family")
    if (isinstance(trajectory_index, bool)
            or not 0 <= int(trajectory_index) < TRAJECTORIES_PER_FAMILY):
        raise ValueError("screen diagnostic trajectory index is out of range")
    sampler = resolved_sampler_config(config, method_id, cell["p"], resource_tier)
    if method_id in DEFECT_METHODS:
        if not isinstance(bias_binding, dict) or set(bias_binding) != {
                "bias_task_fingerprint", "bias_raw_sha256", "bias_sha256"}:
            raise ValueError("defect task requires an exact diagnostic bias binding")
        for name, value in bias_binding.items():
            _validate_sha256(name, value)
    elif bias_binding is not None:
        raise ValueError("hard-coset task cannot bind a defect bias")
    seed_identity = ScreenSeedIdentity(
        seed_root=SCREEN_SEED_ROOT,
        source_commit=source_commit,
        config_sha256=config["screen_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method_id,
        resource_tier=resource_tier,
        init_family=init_family,
        trajectory_index=int(trajectory_index),
        trajectory_namespace="q0_global_screen_measurement_v1",
    )
    return {
        "task_version": SCREEN_TASKS_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "raw_version": (
            SCREEN_HARD_RAW_VERSION if method_id in HARD_METHODS
            else SCREEN_DEFECT_RAW_VERSION
        ),
        "stage": SCREEN_STAGE,
        "method_id": method_id,
        "resource_tier": resource_tier,
        "init_family": init_family,
        "trajectory_index": int(trajectory_index),
        "cell": cell,
        "sampler_config": sampler.as_dict(),
        "seed_identity": seed_identity.as_dict(),
        "bias_binding": bias_binding,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "engine": "numba",
    }


def diagnostic_bias_task_identity(registry, config, source_commit, stage,
                                  method_id, resource_tier, cell):
    _validate_source_commit(source_commit)
    cell = _validate_task_cell(config, stage, cell)
    method_id = str(method_id)
    resource_tier = str(resource_tier)
    if method_id not in DEFECT_METHODS:
        raise ValueError("diagnostic bias task requires a defect method")
    sampler = resolved_sampler_config(config, method_id, cell["p"], resource_tier)
    seeds = [ScreenSeedIdentity(
        seed_root=SCREEN_SEED_ROOT,
        source_commit=source_commit,
        config_sha256=config["screen_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method_id,
        resource_tier=resource_tier,
        init_family="TUNE",
        trajectory_index=index,
        trajectory_namespace="q0_global_screen_bias_v1",
    ).as_dict() for index in range(sampler.tuning_chains)]
    return {
        "task_version": SCREEN_TASKS_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "raw_version": SCREEN_BIAS_RAW_VERSION,
        "stage": SCREEN_STAGE,
        "method_id": method_id,
        "resource_tier": resource_tier,
        "cell": cell,
        "sampler_config": sampler.as_dict(),
        "tuning_seed_identities": seeds,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "engine": "numba",
    }


# Compatibility names keep stage runners small while retaining new identities.
global_task_identity = diagnostic_task_identity
bias_task_identity = diagnostic_bias_task_identity


def _canonical_method_tiers(config, method_tiers, *, defect_only=False):
    values = [(str(method), str(tier)) for method, tier in method_tiers]
    expected_methods = (
        [value["method_id"] for value in config["defect_methods"]]
        if defect_only else [
            *[value["method_id"] for value in config["hard_methods"]],
            *[value["method_id"] for value in config["defect_methods"]],
        ]
    )
    if ([method for method, _ in values] != expected_methods
            or len(values) != len(set(values))
            or len({tier for _, tier in values}) != 1
            or any(tier not in config["resource_tiers"] for _, tier in values)):
        raise ValueError("diagnostic method/tier list is not canonical")
    return values


def build_bias_manifest(registry_path, config_path, source_commit, stage,
                        method_tiers, output_path):
    registry = load_registry(registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    values = _canonical_method_tiers(config, method_tiers, defect_only=True)
    entries = []
    for method, tier in values:
        for cell in _screen_cells(config):
            task = diagnostic_bias_task_identity(
                registry, config, source_commit, stage, method, tier, cell,
            )
            fingerprint = sha256_json(task)
            entries.append({
                "task": task,
                "task_fingerprint": fingerprint,
                "output_relpath": f"bias/{fingerprint}.npz",
            })
    if len(entries) != 15 or len({row["task_fingerprint"] for row in entries}) != 15:
        raise GlobalConflictError("diagnostic bias manifest is not exactly 15 tasks")
    manifest = {
        "manifest_version": SCREEN_TASKS_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "kind": "diagnostic_defect_bias",
        "stage": SCREEN_STAGE,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "method_tiers": [list(value) for value in values],
        "tasks": entries,
    }
    atomic_json(output_path, manifest)
    return manifest


def _validate_bias_manifest_structure(manifest, registry, config,
                                      source_commit=None):
    source_commit = manifest.get("source_commit") if source_commit is None else source_commit
    _validate_source_commit(source_commit)
    if (manifest.get("manifest_version") != SCREEN_TASKS_VERSION
            or manifest.get("contract_version") != SCREEN_DIAGNOSTIC_VERSION
            or manifest.get("kind") != "diagnostic_defect_bias"
            or manifest.get("stage") != SCREEN_STAGE
            or manifest.get("source_commit") != source_commit
            or manifest.get("registry_sha256") != registry["registry_sha256"]
            or manifest.get("screen_config_sha256")
            != config["screen_config_sha256"]):
        raise GlobalConflictError("diagnostic bias manifest identity mismatch")
    try:
        values = _canonical_method_tiers(
            config, manifest.get("method_tiers", []), defect_only=True,
        )
    except (TypeError, ValueError) as exc:
        raise GlobalConflictError("diagnostic bias method tiers are noncanonical") from exc
    expected = []
    for method, tier in values:
        for cell in _screen_cells(config):
            task = diagnostic_bias_task_identity(
                registry, config, source_commit, SCREEN_STAGE, method, tier, cell,
            )
            fingerprint = sha256_json(task)
            expected.append({
                "task": task,
                "task_fingerprint": fingerprint,
                "output_relpath": f"bias/{fingerprint}.npz",
            })
    if manifest.get("tasks") != expected or len(expected) != 15:
        raise GlobalConflictError("diagnostic bias manifest task order changed")
    return values


def _bias_index_from_manifest(raw_root, manifest, registry, config,
                              source_commit):
    _validate_bias_manifest_structure(manifest, registry, config, source_commit)
    index = {}
    for entry in manifest["tasks"]:
        path = Path(raw_root) / entry["output_relpath"]
        record = validate_bias_raw(path, registry, config, source_commit)
        if record["task_fingerprint"] != entry["task_fingerprint"]:
            raise GlobalConflictError("diagnostic bias manifest/raw mismatch")
        task = entry["task"]
        key = (task["method_id"], task["resource_tier"], sha256_json(task["cell"]))
        index[key] = {
            "binding": {
                "bias_task_fingerprint": record["task_fingerprint"],
                "bias_raw_sha256": record["sha256"],
                "bias_sha256": record["bias_sha256"],
            },
            "relpath": entry["output_relpath"],
        }
    if len(index) != 15:
        raise GlobalConflictError("diagnostic bias index is incomplete")
    return index


def build_measurement_manifest(registry_path, config_path, source_commit,
                               stage, method_tiers, output_path, *,
                               bias_manifest_path, bias_raw_root):
    registry = load_registry(registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    values = _canonical_method_tiers(config, method_tiers)
    bias_manifest = json.loads(Path(bias_manifest_path).read_text(encoding="ascii"))
    bias_manifest_sha = sha256_json(bias_manifest)
    bias_index = _bias_index_from_manifest(
        bias_raw_root, bias_manifest, registry, config, source_commit,
    )
    entries = []
    for method, tier in values:
        for cell in _screen_cells(config):
            bias = bias_index.get((method, tier, sha256_json(cell)))
            if method in DEFECT_METHODS and bias is None:
                raise GlobalConflictError("diagnostic measurement lacks a bound bias")
            for family in INIT_FAMILIES:
                for trajectory in range(TRAJECTORIES_PER_FAMILY):
                    task = diagnostic_task_identity(
                        registry, config, source_commit, stage, method, tier,
                        cell, family, trajectory,
                        bias_binding=None if bias is None else bias["binding"],
                    )
                    fingerprint = sha256_json(task)
                    entries.append({
                        "task": task,
                        "task_fingerprint": fingerprint,
                        "output_relpath": f"trajectories/{fingerprint}.npz",
                        "bias_relpath": None if bias is None else bias["relpath"],
                    })
    if (len(entries) != 1280
            or len({row["task_fingerprint"] for row in entries}) != 1280):
        raise GlobalConflictError("diagnostic measurement is not exactly 1280 tasks")
    manifest = {
        "manifest_version": SCREEN_TASKS_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "kind": "diagnostic_measurement",
        "stage": SCREEN_STAGE,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "method_tiers": [list(value) for value in values],
        "bias_manifest_sha256": bias_manifest_sha,
        "tasks": entries,
    }
    atomic_json(output_path, manifest)
    return manifest


def _validate_measurement_manifest_structure(manifest, registry, config):
    if (manifest.get("manifest_version") != SCREEN_TASKS_VERSION
            or manifest.get("contract_version") != SCREEN_DIAGNOSTIC_VERSION
            or manifest.get("kind") != "diagnostic_measurement"
            or manifest.get("stage") != SCREEN_STAGE
            or manifest.get("registry_sha256") != registry["registry_sha256"]
            or manifest.get("screen_config_sha256")
            != config["screen_config_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(
                manifest.get("bias_manifest_sha256"))) is None):
        raise GlobalConflictError("diagnostic measurement manifest identity mismatch")
    _validate_source_commit(manifest.get("source_commit"))
    try:
        values = _canonical_method_tiers(config, manifest.get("method_tiers", []))
    except (TypeError, ValueError) as exc:
        raise GlobalConflictError("diagnostic measurement method tiers changed") from exc
    expected_coordinates = [
        (method, tier, cell, family, trajectory)
        for method, tier in values
        for cell in _screen_cells(config)
        for family in INIT_FAMILIES
        for trajectory in range(TRAJECTORIES_PER_FAMILY)
    ]
    entries = manifest.get("tasks")
    if not isinstance(entries, list) or len(entries) != 1280:
        raise GlobalConflictError("diagnostic measurement task count is noncanonical")
    bindings = {}
    for entry, coordinate in zip(entries, expected_coordinates):
        if set(entry) != {
                "task", "task_fingerprint", "output_relpath", "bias_relpath"}:
            raise GlobalConflictError("diagnostic measurement entry schema mismatch")
        method, tier, cell, family, trajectory = coordinate
        task = entry["task"]
        expected = diagnostic_task_identity(
            registry, config, manifest["source_commit"], SCREEN_STAGE, method,
            tier, cell, family, trajectory,
            bias_binding=task.get("bias_binding"),
        )
        fingerprint = sha256_json(task)
        if (task != expected
                or entry["task_fingerprint"] != fingerprint
                or entry["output_relpath"] != f"trajectories/{fingerprint}.npz"):
            raise GlobalConflictError("diagnostic measurement task/order changed")
        key = (method, tier, sha256_json(cell))
        if method in DEFECT_METHODS:
            binding = task["bias_binding"]
            if entry["bias_relpath"] != f"bias/{binding['bias_task_fingerprint']}.npz":
                raise GlobalConflictError("diagnostic bias relative path mismatch")
            if key in bindings and bindings[key] != binding:
                raise GlobalConflictError("diagnostic trajectories bind different biases")
            bindings[key] = binding
        elif entry["bias_relpath"] is not None:
            raise GlobalConflictError("hard diagnostic task unexpectedly binds bias")
    return values


def validate_control_manifest(manifest, registry, config):
    if not isinstance(manifest, dict):
        raise GlobalConflictError("diagnostic control is not a JSON object")
    if manifest.get("kind") == "diagnostic_defect_bias":
        _validate_bias_manifest_structure(manifest, registry, config)
    elif manifest.get("kind") == "diagnostic_measurement":
        _validate_measurement_manifest_structure(manifest, registry, config)
    else:
        raise GlobalConflictError("unknown diagnostic control kind")
    return True


validate_screen_control_manifest = validate_control_manifest


def freeze_screen_schedule(registry_path, config_path, source_commit,
                           archive_sha256, source_manifest_sha256, output_path,
                           *, started_unix=None):
    _validate_source_commit(source_commit)
    _validate_sha256("archive_sha256", archive_sha256)
    _validate_sha256("source_manifest_sha256", source_manifest_sha256)
    registry = load_registry(registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    started = time.time() if started_unix is None else float(started_unix)
    if not np.isfinite(started) or started <= 0.0:
        raise ValueError("diagnostic schedule start is invalid")
    hours = config["schedule"]
    deadlines = {
        "preflight": started + hours["preflight_deadline_hour"] * 3600.0,
        "bias": started + hours["bias_deadline_hour"] * 3600.0,
        "measurement": started + hours["measurement_deadline_hour"] * 3600.0,
        "analysis": started + hours["analysis_deadline_hour"] * 3600.0,
    }
    identity = {
        "schedule_version": SCREEN_SCHEDULE_VERSION,
        "status": "FROZEN_24H_DIAGNOSTIC",
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "started_unix": started,
        "deadlines_unix": deadlines,
        "wall_limit_hours": hours["wall_limit_hours"],
        "production_authorized": False,
    }
    schedule = {**identity, "schedule_sha256": sha256_json(identity)}
    output_path = Path(output_path)
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="ascii")) != schedule:
            raise GlobalConflictError("existing diagnostic schedule conflicts")
    else:
        atomic_json(output_path, schedule)
    return schedule


def validate_screen_schedule(path, registry, config, source_commit=None):
    schedule = json.loads(Path(path).read_text(encoding="ascii"))
    expected_keys = {
        "schedule_version", "status", "source_commit", "archive_sha256",
        "source_manifest_sha256", "registry_sha256",
        "screen_config_sha256", "started_unix", "deadlines_unix",
        "wall_limit_hours", "production_authorized", "schedule_sha256",
    }
    identity = {key: value for key, value in schedule.items()
                if key != "schedule_sha256"}
    try:
        _validate_source_commit(schedule.get("source_commit"))
        _validate_sha256("archive_sha256", schedule.get("archive_sha256"))
        _validate_sha256(
            "source_manifest_sha256", schedule.get("source_manifest_sha256"),
        )
        started = float(schedule.get("started_unix"))
    except (TypeError, ValueError) as exc:
        raise GlobalConflictError("diagnostic schedule identity is malformed") from exc
    if (set(schedule) != expected_keys
            or schedule.get("schedule_version") != SCREEN_SCHEDULE_VERSION
            or schedule.get("status") != "FROZEN_24H_DIAGNOSTIC"
            or schedule.get("schedule_sha256") != sha256_json(identity)
            or schedule.get("registry_sha256") != registry["registry_sha256"]
            or schedule.get("screen_config_sha256")
            != config["screen_config_sha256"]
            or schedule.get("production_authorized") is not False
            or (source_commit is not None
                and schedule.get("source_commit") != source_commit)):
        raise GlobalConflictError("diagnostic schedule identity mismatch")
    expected_deadlines = {"preflight", "bias", "measurement", "analysis"}
    values = schedule.get("deadlines_unix", {})
    hours = config["schedule"]
    exact_deadlines = {
        "preflight": started + hours["preflight_deadline_hour"] * 3600.0,
        "bias": started + hours["bias_deadline_hour"] * 3600.0,
        "measurement": started + hours["measurement_deadline_hour"] * 3600.0,
        "analysis": started + hours["analysis_deadline_hour"] * 3600.0,
    }
    if (not np.isfinite(started) or started <= 0.0
            or set(values) != expected_deadlines
            or any(not np.isfinite(float(value)) for value in values.values())
            or values != exact_deadlines
            or schedule.get("wall_limit_hours") != hours["wall_limit_hours"]):
        raise GlobalConflictError("diagnostic schedule deadlines are malformed")
    return schedule


def _task_cost(task):
    m = int(task["cell"]["code_id"][1:3])
    sampler = task["sampler_config"]
    work = int(sampler["burn_sweeps"]) + int(sampler["measurement_sweeps"])
    if task["method_id"].startswith("RC8-QC"):
        multiplier = 1.0 + float(sampler["cluster_repeats"])
    elif task["method_id"].startswith("RC8-J"):
        multiplier = 1.0 + (1 << int(sampler["joint_block_size"])) / 256.0
    else:
        multiplier = 1.0
    return float(m * m * work * multiplier)


def fixed_screen_ownership(tasks, nodes, source_commit, control_sha256,
                           stage=SCREEN_STAGE):
    _validate_source_commit(source_commit)
    _validate_sha256("control_sha256", control_sha256)
    nodes = list(nodes)
    if (stage != SCREEN_STAGE or len(nodes) < 2
            or len(nodes) != len(set(nodes))
            or not set(nodes) <= set(NODE_CAPACITY)):
        raise ValueError("diagnostic ownership nodes/stage are invalid")
    loads = {node: 0.0 for node in nodes}
    owners = {}
    for task in sorted(tasks, key=lambda row: (-_task_cost(row), sha256_json(row))):
        owner = min(nodes, key=lambda node: (loads[node] / NODE_CAPACITY[node], node))
        fingerprint = sha256_json(task)
        if fingerprint in owners:
            raise GlobalConflictError("diagnostic ownership received duplicate tasks")
        owners[fingerprint] = owner
        loads[owner] += _task_cost(task)
    identity = {
        "ownership_version": SCREEN_OWNERSHIP_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "source_commit": source_commit,
        "control_sha256": control_sha256,
        "stage": SCREEN_STAGE,
        "nodes": nodes,
        "task_owner": owners,
    }
    return {
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "weighted_load": loads,
        "capacity": {node: NODE_CAPACITY[node] for node in nodes},
    }


fixed_ownership = fixed_screen_ownership


def validate_screen_ownership(ownership, tasks, nodes, source_commit,
                              control_sha256):
    expected = fixed_screen_ownership(
        tasks, nodes, source_commit, control_sha256, SCREEN_STAGE,
    )
    if ownership != expected:
        raise GlobalConflictError("diagnostic ownership is noncanonical")
    return True


def _registry_with_path(registry, registry_path):
    value = dict(registry)
    value["_registry_path"] = str(Path(registry_path).resolve())
    return value


def _cell_disorder(registry, code, model, cell):
    uniform_seed = _uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(
        model.num_qubits,
    )
    epsilon = (uniforms < float(cell["p"])).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    return uniform_seed, epsilon, syndrome


def _initial_state(model, epsilon, syndrome, seed_identity):
    if seed_identity.init_family == "P":
        return epsilon.copy()
    if seed_identity.init_family == "U":
        return uniform_hard_coset_state(
            model, syndrome, seed_identity.seed("initialize", "hard_coset"),
        )
    raise ValueError("diagnostic measurement cannot use TUNE initialization")


def _raw_common(task, registry, config, code, model, uniform_seed, characters,
                sampler, seed_identity, core_seconds, wall_seconds):
    return {
        "contract_version": np.array(SCREEN_DIAGNOSTIC_VERSION),
        "task_fingerprint": np.array(sha256_json(task)),
        "task_json": np.array(canonical_json(task)),
        "cell_json": np.array(canonical_json(task["cell"])),
        "sampler_config_json": np.array(canonical_json(sampler.as_dict())),
        "seed_identity_json": np.array(canonical_json(seed_identity.as_dict())),
        "source_commit": np.array(task["source_commit"]),
        "registry_sha256": np.array(registry["registry_sha256"]),
        "screen_config_sha256": np.array(config["screen_config_sha256"]),
        "uniform_seed": np.array(uniform_seed, dtype=np.int64),
        "engine": np.array("numba"),
        "model_fingerprint": np.array(model.fingerprint()),
        "section_fingerprint": np.array(code["section_fingerprint"]),
        "logical_frame_fingerprint": np.array(code["logical_frame_fingerprint"]),
        "character_version": np.array(CHARACTER_SET_VERSION),
        "character_masks": characters.masks,
        "character_sha256": np.array(characters.character_sha256),
        "num_qubits": np.array(model.num_qubits, dtype=np.int32),
        "k": np.array(model.k, dtype=np.int16),
        "core_seconds": np.array(core_seconds),
        "wall_seconds": np.array(wall_seconds),
    }


def run_screen_hard_task(registry_path, config_path, source_commit, task,
                         output_path):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    expected = diagnostic_task_identity(
        registry, config, source_commit, task.get("stage"),
        task.get("method_id"), task.get("resource_tier"), task.get("cell"),
        task.get("init_family"), task.get("trajectory_index"),
        bias_binding=None,
    )
    if task != expected or task["method_id"] not in HARD_METHODS:
        raise GlobalConflictError("diagnostic hard task is noncanonical")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_screen_hard_raw(
            output_path, registry, config, source_commit,
        )
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing diagnostic hard raw conflicts")
        return "reused"
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    sampler = resolved_sampler_config(
        config, task["method_id"], task["cell"]["p"], task["resource_tier"],
    )
    seed_identity = ScreenSeedIdentity(**task["seed_identity"])
    initial = _initial_state(model, epsilon, syndrome, seed_identity)
    catalog = build_logical_proposal_catalog(
        model, frame, max_multiple=config["catalog"]["max_multiple"],
        max_count=config["catalog"]["max_count"],
    )
    joint = (
        build_joint_blocks(model, frame, catalog, sampler.joint_block_size)
        if sampler.joint_block_size else None
    )
    characters = frozen_character_set(
        model.k,
        character_seed(config, registry["registry_sha256"], code["code_id"]),
        config["characters"]["num_nonbasis"],
    )
    wall_start, core_start = time.monotonic(), time.process_time()
    result = run_hardcoset_trajectory(
        model, frame, syndrome, sampler, seed_identity, initial,
        engine="numba", catalog=catalog, joint=joint,
    )
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    arrays = _raw_common(
        task, registry, config, code, model, uniform_seed, characters, sampler,
        seed_identity, core_seconds, wall_seconds,
    )
    arrays.update(_global._result_arrays(result))
    arrays.update({
        "raw_version": np.array(SCREEN_HARD_RAW_VERSION),
        "catalog_sha256": np.array(catalog.catalog_sha256),
        "joint_sha256": np.array(result["joint_sha256"]),
        "measurement_residual_weights": result["measurement_residual_weights"],
    })
    atomic_npz(output_path, **arrays)
    return "computed"


def run_screen_bias_task(registry_path, config_path, source_commit, task,
                         output_path):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    expected = diagnostic_bias_task_identity(
        registry, config, source_commit, task.get("stage"),
        task.get("method_id"), task.get("resource_tier"), task.get("cell"),
    )
    if task != expected:
        raise GlobalConflictError("diagnostic bias task is noncanonical")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_screen_bias_raw(
            output_path, registry, config, source_commit,
        )
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing diagnostic bias raw conflicts")
        return "reused"
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, _ = build_model(H)
    uniform_seed, _, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    sampler = resolved_sampler_config(
        config, task["method_id"], task["cell"]["p"], task["resource_tier"],
    )
    identities = [
        ScreenSeedIdentity(**value) for value in task["tuning_seed_identities"]
    ]
    wall_start, core_start = time.monotonic(), time.process_time()
    result = tune_defect_bias(
        model, syndrome, sampler, identities, engine="numba",
    )
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    atomic_npz(
        output_path,
        raw_version=np.array(SCREEN_BIAS_RAW_VERSION),
        contract_version=np.array(SCREEN_DIAGNOSTIC_VERSION),
        task_fingerprint=np.array(sha256_json(task)),
        task_json=np.array(canonical_json(task)),
        cell_json=np.array(canonical_json(task["cell"])),
        sampler_config_json=np.array(canonical_json(sampler.as_dict())),
        source_commit=np.array(source_commit),
        registry_sha256=np.array(registry["registry_sha256"]),
        screen_config_sha256=np.array(config["screen_config_sha256"]),
        uniform_seed=np.array(uniform_seed, dtype=np.int64),
        engine=np.array("numba"),
        model_fingerprint=np.array(model.fingerprint()),
        num_qubits=np.array(model.num_qubits, dtype=np.int32),
        tuning_seed_identities_json=np.array(canonical_json(
            task["tuning_seed_identities"],
        )),
        bias=result["bias"],
        bias_trace=result["bias_trace"],
        tuning_histogram=result["tuning_histogram"],
        tuning_final_states_packed=result["tuning_final_states_packed"],
        tuning_final_residuals=result["tuning_final_residuals"],
        tuning_final_defects=result["tuning_final_defects"],
        gammas=result["gammas"],
        bias_sha256=np.array(result["bias_sha256"]),
        core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds),
    )
    return "computed"


def _checked_cached_bias_record(path, registry, config, source_commit, record):
    path = Path(path).resolve(strict=True)
    required = {
        "path", "sha256", "task", "task_fingerprint", "bias", "bias_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise GlobalConflictError("cached diagnostic bias record schema mismatch")
    task = record["task"]
    if not isinstance(task, dict):
        raise GlobalConflictError("cached diagnostic bias task is malformed")
    expected = diagnostic_bias_task_identity(
        registry, config, task.get("source_commit"), task.get("stage"),
        task.get("method_id"), task.get("resource_tier"), task.get("cell"),
    )
    file_sha = sha256_file(path)
    if (record["path"] != str(path)
            or record["sha256"] != file_sha
            or task != expected
            or record["task_fingerprint"] != sha256_json(task)
            or task.get("registry_sha256") != registry["registry_sha256"]
            or task.get("screen_config_sha256")
            != config["screen_config_sha256"]
            or (source_commit is not None
                and task.get("source_commit") != source_commit)
            or re.fullmatch(r"[0-9a-f]{64}", str(
                record["bias_sha256"])) is None):
        raise GlobalConflictError("cached diagnostic bias record is stale or mismatched")
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot reopen cached diagnostic bias {path}: {exc}") from exc
    with context as data:
        if (set(data.files) != BIAS_RAW_FIELDS
                or str(_scalar(data, "task_fingerprint"))
                != record["task_fingerprint"]
                or str(_scalar(data, "bias_sha256")) != record["bias_sha256"]
                or not np.array_equal(
                    np.asarray(data["bias"]), np.asarray(record["bias"]),
                    equal_nan=True,
                )):
            raise GlobalConflictError("cached diagnostic bias differs from bound raw")
    return record


def _bias_record_from_raw(path, registry, config, source_commit=None, *,
                          _validated_bias_record=None):
    if _validated_bias_record is None:
        return validate_screen_bias_raw(path, registry, config, source_commit)
    return _checked_cached_bias_record(
        path, registry, config, source_commit, _validated_bias_record,
    )


def screen_bias_binding_from_raw(path, registry, config,
                                 source_commit=None, *,
                                 _validated_bias_record=None):
    record = _bias_record_from_raw(
        path, registry, config, source_commit,
        _validated_bias_record=_validated_bias_record,
    )
    return {
        "bias_task_fingerprint": record["task_fingerprint"],
        "bias_raw_sha256": sha256_file(path),
        "bias_sha256": record["bias_sha256"],
    }


def run_screen_defect_task(registry_path, config_path, source_commit, task,
                           bias_path, output_path, *,
                           _validated_bias_record=None):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    bias_record = _bias_record_from_raw(
        bias_path, registry, config, source_commit,
        _validated_bias_record=_validated_bias_record,
    )
    binding = {
        "bias_task_fingerprint": bias_record["task_fingerprint"],
        "bias_raw_sha256": bias_record["sha256"],
        "bias_sha256": bias_record["bias_sha256"],
    }
    expected = diagnostic_task_identity(
        registry, config, source_commit, task.get("stage"),
        task.get("method_id"), task.get("resource_tier"), task.get("cell"),
        task.get("init_family"), task.get("trajectory_index"),
        bias_binding=binding,
    )
    if task != expected or task["method_id"] not in DEFECT_METHODS:
        raise GlobalConflictError("diagnostic defect task is noncanonical")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_screen_defect_raw(
            output_path, registry, config, source_commit, bias_path,
            _validated_bias_record=bias_record,
        )
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing diagnostic defect raw conflicts")
        return "reused"
    bias = np.asarray(bias_record["bias"], dtype=np.float64).copy()
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    sampler = resolved_sampler_config(
        config, task["method_id"], task["cell"]["p"], task["resource_tier"],
    )
    seed_identity = ScreenSeedIdentity(**task["seed_identity"])
    initial = _initial_state(model, epsilon, syndrome, seed_identity)
    characters = frozen_character_set(
        model.k,
        character_seed(config, registry["registry_sha256"], code["code_id"]),
        config["characters"]["num_nonbasis"],
    )
    wall_start, core_start = time.monotonic(), time.process_time()
    result = run_defect_trace_trajectory(
        model, frame, syndrome, sampler, seed_identity, initial, bias,
        binding["bias_sha256"], engine="numba",
    )
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    arrays = _raw_common(
        task, registry, config, code, model, uniform_seed, characters, sampler,
        seed_identity, core_seconds, wall_seconds,
    )
    arrays.update(_global._result_arrays(result))
    arrays.update({
        "raw_version": np.array(SCREEN_DEFECT_RAW_VERSION),
        "bias_task_fingerprint": np.array(binding["bias_task_fingerprint"]),
        "bias_raw_sha256": np.array(binding["bias_raw_sha256"]),
        "bias_sha256": np.array(binding["bias_sha256"]),
        "bias": bias,
        "measurement_defect_counts": result["measurement_defect_counts"],
        "fixed_clock_d0_mask": result["fixed_clock_d0_mask"],
        "boundary_occupancy": np.array(result["boundary_occupancy"]),
    })
    atomic_npz(output_path, **arrays)
    return "computed"


run_hard_task = run_screen_hard_task
run_bias_task = run_screen_bias_task
run_defect_task = run_screen_defect_task


def _scalar(data, field):
    if field not in data or data[field].shape != ():
        raise GlobalConflictError(
            f"diagnostic raw scalar is missing or malformed: {field}",
        )
    return data[field].item()


def _require_equal(field, stored, expected):
    if not np.array_equal(
            np.asarray(stored), np.asarray(expected), equal_nan=True):
        raise GlobalConflictError(f"diagnostic raw replay mismatch: {field}")


def _read_task(data):
    try:
        task = json.loads(str(_scalar(data, "task_json")))
    except Exception as exc:
        raise GlobalConflictError("diagnostic raw task JSON is malformed") from exc
    if str(_scalar(data, "task_fingerprint")) != sha256_json(task):
        raise GlobalConflictError("diagnostic raw task fingerprint mismatch")
    return task


def _validate_common_raw(data, fields, registry, config,
                         expected_source_commit, expected_raw_version):
    if set(data.files) != fields:
        raise GlobalConflictError(
            "diagnostic raw schema mismatch; "
            f"missing={sorted(fields-set(data.files))}, "
            f"extra={sorted(set(data.files)-fields)}",
        )
    task = _read_task(data)
    if (str(_scalar(data, "raw_version")) != expected_raw_version
            or str(_scalar(data, "contract_version"))
            != SCREEN_DIAGNOSTIC_VERSION
            or task.get("contract_version") != SCREEN_DIAGNOSTIC_VERSION
            or task.get("task_version") != SCREEN_TASKS_VERSION
            or (expected_source_commit is not None
                and task.get("source_commit") != expected_source_commit)
            or task.get("registry_sha256") != registry["registry_sha256"]
            or task.get("screen_config_sha256")
            != config["screen_config_sha256"]):
        raise GlobalConflictError("diagnostic raw identity/version mismatch")
    expected_scalars = {
        "source_commit": task["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "engine": "numba",
    }
    for field, expected in expected_scalars.items():
        if str(_scalar(data, field)) != str(expected):
            raise GlobalConflictError(f"diagnostic raw identity mismatch: {field}")
    for field in ("core_seconds", "wall_seconds"):
        value = float(_scalar(data, field))
        if not np.isfinite(value) or value < 0.0:
            raise GlobalConflictError(f"diagnostic raw timing is invalid: {field}")
    return task


def _rebuild_task_context(registry, config, task):
    registry_path = registry.get("_registry_path")
    if registry_path is None:
        registry_path = Path(config["config_path"]).parents[1] / "registry/registry.json"
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    sampler = resolved_sampler_config(
        config, task["method_id"], task["cell"]["p"], task["resource_tier"],
    )
    characters = frozen_character_set(
        model.k,
        character_seed(config, registry["registry_sha256"], code["code_id"]),
        config["characters"]["num_nonbasis"],
    )
    return code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters


def _validate_stored_context(data, task, code, model, uniform_seed, sampler,
                             characters):
    expected_scalars = {
        "cell_json": canonical_json(task["cell"]),
        "sampler_config_json": canonical_json(sampler.as_dict()),
        "uniform_seed": uniform_seed,
        "model_fingerprint": model.fingerprint(),
        "section_fingerprint": code["section_fingerprint"],
        "logical_frame_fingerprint": code["logical_frame_fingerprint"],
        "character_version": CHARACTER_SET_VERSION,
        "character_sha256": characters.character_sha256,
        "num_qubits": model.num_qubits,
        "k": model.k,
    }
    for field, expected in expected_scalars.items():
        if str(_scalar(data, field)) != str(expected):
            raise GlobalConflictError(
                f"diagnostic reconstructed identity mismatch: {field}",
            )
    _require_equal("character_masks", data["character_masks"], characters.masks)


def _record_common(path, task, labels, weights, valid_mask, burn_labels,
                   initial_label, model, characters, core_seconds):
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "task": task,
        "task_fingerprint": sha256_json(task),
        "cell": task["cell"],
        "method_id": task["method_id"],
        "resource_tier": task["resource_tier"],
        "init_family": task["init_family"],
        "trajectory_index": task["trajectory_index"],
        "labels": labels,
        "weights": weights,
        "valid_mask": valid_mask,
        "burn_labels": burn_labels,
        "initial_label": int(initial_label),
        "num_qubits": model.num_qubits,
        "k": model.k,
        "character_masks": characters.masks,
        "core_seconds": core_seconds,
    }


def validate_screen_hard_raw(path, registry, config,
                             expected_source_commit=None):
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open diagnostic hard raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, HARD_RAW_FIELDS, registry, config, expected_source_commit,
            SCREEN_HARD_RAW_VERSION,
        )
        expected = diagnostic_task_identity(
            registry, config, task["source_commit"], task["stage"],
            task["method_id"], task["resource_tier"], task["cell"],
            task["init_family"], task["trajectory_index"], bias_binding=None,
        )
        if task != expected or task["method_id"] not in HARD_METHODS:
            raise GlobalConflictError("diagnostic hard raw embeds invalid task")
        code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters = (
            _rebuild_task_context(registry, config, task)
        )
        _validate_stored_context(
            data, task, code, model, uniform_seed, sampler, characters,
        )
        seed_identity = ScreenSeedIdentity(**task["seed_identity"])
        if str(_scalar(data, "seed_identity_json")) != canonical_json(
                seed_identity.as_dict()):
            raise GlobalConflictError("diagnostic hard seed identity mismatch")
        initial = _initial_state(model, epsilon, syndrome, seed_identity)
        catalog = build_logical_proposal_catalog(
            model, frame, max_multiple=config["catalog"]["max_multiple"],
            max_count=config["catalog"]["max_count"],
        )
        if str(_scalar(data, "catalog_sha256")) != catalog.catalog_sha256:
            raise GlobalConflictError("diagnostic hard catalog SHA mismatch")
        joint = (
            build_joint_blocks(model, frame, catalog, sampler.joint_block_size)
            if sampler.joint_block_size else None
        )
        expected_joint = "none" if joint is None else joint.joint_sha256
        if str(_scalar(data, "joint_sha256")) != expected_joint:
            raise GlobalConflictError("diagnostic joint-block SHA mismatch")
        replay = run_hardcoset_trajectory(
            model, frame, syndrome, sampler, seed_identity, initial,
            engine="numba", catalog=catalog, joint=joint,
        )
        _global._compare_result(data, replay, defect=False)
        record = _record_common(
            path, task, data["measurement_labels"].copy(),
            data["measurement_weights"].copy(),
            np.ones(data["measurement_labels"].size, dtype=bool),
            data["burn_labels"].copy(), replay["initial_label"], model,
            characters, float(_scalar(data, "core_seconds")),
        )
    return record


def validate_screen_bias_raw(path, registry, config,
                             expected_source_commit=None):
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open diagnostic bias raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, BIAS_RAW_FIELDS, registry, config, expected_source_commit,
            SCREEN_BIAS_RAW_VERSION,
        )
        expected = diagnostic_bias_task_identity(
            registry, config, task["source_commit"], task["stage"],
            task["method_id"], task["resource_tier"], task["cell"],
        )
        if task != expected:
            raise GlobalConflictError("diagnostic bias raw embeds invalid task")
        _, model, _, uniform_seed, _, syndrome, sampler, _ = (
            _rebuild_task_context(registry, config, task)
        )
        expected_scalars = {
            "cell_json": canonical_json(task["cell"]),
            "sampler_config_json": canonical_json(sampler.as_dict()),
            "uniform_seed": uniform_seed,
            "model_fingerprint": model.fingerprint(),
            "num_qubits": model.num_qubits,
            "tuning_seed_identities_json": canonical_json(
                task["tuning_seed_identities"],
            ),
        }
        for field, expected_value in expected_scalars.items():
            if str(_scalar(data, field)) != str(expected_value):
                raise GlobalConflictError(
                    f"diagnostic bias identity mismatch: {field}",
                )
        identities = [
            ScreenSeedIdentity(**value)
            for value in task["tuning_seed_identities"]
        ]
        replay = tune_defect_bias(
            model, syndrome, sampler, identities, engine="numba",
        )
        for field in (
                "bias", "bias_trace", "tuning_histogram",
                "tuning_final_states_packed", "tuning_final_residuals",
                "tuning_final_defects", "gammas"):
            _require_equal(field, data[field], replay[field])
        if str(_scalar(data, "bias_sha256")) != replay["bias_sha256"]:
            raise GlobalConflictError("diagnostic bias SHA replay mismatch")
        bias = data["bias"].copy()
        bias_sha = str(_scalar(data, "bias_sha256"))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "task": task,
        "task_fingerprint": sha256_json(task),
        "bias": bias,
        "bias_sha256": bias_sha,
    }


def validate_screen_defect_raw(path, registry, config,
                               expected_source_commit=None, bias_path=None,
                               *, _validated_bias_record=None):
    path = Path(path)
    if bias_path is None:
        raise ValueError("diagnostic defect validation requires bias raw")
    bias_record = _bias_record_from_raw(
        bias_path, registry, config, expected_source_commit,
        _validated_bias_record=_validated_bias_record,
    )
    binding = {
        "bias_task_fingerprint": bias_record["task_fingerprint"],
        "bias_raw_sha256": bias_record["sha256"],
        "bias_sha256": bias_record["bias_sha256"],
    }
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open diagnostic defect raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, DEFECT_RAW_FIELDS, registry, config, expected_source_commit,
            SCREEN_DEFECT_RAW_VERSION,
        )
        expected = diagnostic_task_identity(
            registry, config, task["source_commit"], task["stage"],
            task["method_id"], task["resource_tier"], task["cell"],
            task["init_family"], task["trajectory_index"],
            bias_binding=binding,
        )
        if task != expected or task["method_id"] not in DEFECT_METHODS:
            raise GlobalConflictError("diagnostic defect raw embeds invalid task")
        code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters = (
            _rebuild_task_context(registry, config, task)
        )
        _validate_stored_context(
            data, task, code, model, uniform_seed, sampler, characters,
        )
        for field, expected_value in binding.items():
            if str(_scalar(data, field)) != str(expected_value):
                raise GlobalConflictError(
                    f"diagnostic defect bias mismatch: {field}",
                )
        _require_equal("bias", data["bias"], bias_record["bias"])
        seed_identity = ScreenSeedIdentity(**task["seed_identity"])
        if str(_scalar(data, "seed_identity_json")) != canonical_json(
                seed_identity.as_dict()):
            raise GlobalConflictError("diagnostic defect seed identity mismatch")
        initial = _initial_state(model, epsilon, syndrome, seed_identity)
        replay = run_defect_trace_trajectory(
            model, frame, syndrome, sampler, seed_identity, initial,
            bias_record["bias"], bias_record["bias_sha256"], engine="numba",
        )
        _global._compare_result(data, replay, defect=True)
        record = _record_common(
            path, task, data["measurement_labels"].copy(),
            data["measurement_weights"].copy(),
            data["fixed_clock_d0_mask"].copy(), data["burn_labels"].copy(),
            replay["initial_label"], model, characters,
            float(_scalar(data, "core_seconds")),
        )
        record.update({
            "defects": data["measurement_defect_counts"].copy(),
            "measurement_counters": data["measurement_counters"].copy(),
            "boundary_occupancy": float(_scalar(data, "boundary_occupancy")),
        })
    return record


validate_hard_raw = validate_screen_hard_raw
validate_bias_raw = validate_screen_bias_raw
validate_defect_raw = validate_screen_defect_raw


_PARALLEL_REPLAY_CONTEXT = {}


def _validated_bias_cache(raw_root, manifest, registry, config):
    cache = {}
    raw_root = Path(raw_root)
    for entry in manifest["tasks"]:
        if entry["task"]["method_id"] not in DEFECT_METHODS:
            continue
        bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
        cache_key = str(bias_path)
        if cache_key not in cache:
            cache[cache_key] = validate_screen_bias_raw(
                bias_path, registry, config, manifest["source_commit"],
            )
    expected = len(DEFECT_METHODS) * len(_screen_cells(config))
    if len(cache) != expected:
        raise GlobalConflictError(
            "diagnostic measurement does not bind all 15 unique biases",
        )
    return cache


def _initialize_parallel_replay(registry_path, config_path, source_commit,
                                bias_cache):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    checked = {
        str(Path(path).resolve(strict=True)): _checked_cached_bias_record(
            path, registry, config, source_commit, record,
        )
        for path, record in bias_cache.items()
    }
    key = (str(registry_path), str(config_path), str(source_commit))
    _PARALLEL_REPLAY_CONTEXT.clear()
    _PARALLEL_REPLAY_CONTEXT[key] = {
        "registry": registry,
        "config": config,
        "bias_cache": checked,
    }


def _parallel_validate_entry(payload):
    raw_root, entry, registry_path, config_path, source_commit = payload
    key = (str(registry_path), str(config_path), str(source_commit))
    context = _PARALLEL_REPLAY_CONTEXT.get(key)
    if context is None:
        raise GlobalConflictError("diagnostic replay worker was not initialized")
    registry = context["registry"]
    config = context["config"]
    raw_root = Path(raw_root)
    path = raw_root / entry["output_relpath"]
    task = entry["task"]
    if task["method_id"] in HARD_METHODS:
        record = validate_screen_hard_raw(
            path, registry, config, source_commit,
        )
    else:
        bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
        cache_key = str(bias_path)
        if cache_key not in context["bias_cache"]:
            raise GlobalConflictError("diagnostic replay worker lacks bound bias")
        record = validate_screen_defect_raw(
            path, registry, config, source_commit, bias_path,
            _validated_bias_record=context["bias_cache"][cache_key],
        )
    if record["task_fingerprint"] != entry["task_fingerprint"]:
        raise GlobalConflictError("diagnostic manifest/raw fingerprint mismatch")
    return record


def _load_measurement_records(raw_root, manifest, registry, config,
                              num_workers=1):
    raw_root = Path(raw_root)
    expected_paths = {entry["output_relpath"] for entry in manifest["tasks"]}
    actual_paths = {
        str(path.relative_to(raw_root))
        for path in (raw_root / "trajectories").glob("*.npz")
    } if (raw_root / "trajectories").exists() else set()
    if actual_paths != expected_paths:
        raise GlobalConflictError(
            "diagnostic measurement raw set mismatch; "
            f"missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}",
        )
    if isinstance(num_workers, bool) or int(num_workers) <= 0:
        raise ValueError("diagnostic replay worker count must be positive")
    num_workers = int(num_workers)
    bias_cache = _validated_bias_cache(
        raw_root, manifest, registry, config,
    )
    if num_workers > 1:
        registry_path = registry.get("_registry_path")
        if registry_path is None:
            raise ValueError("parallel diagnostic replay needs registry path")
        payloads = [
            (
                str(raw_root), entry, registry_path, config["config_path"],
                manifest["source_commit"],
            )
            for entry in manifest["tasks"]
        ]
        workers = min(num_workers, len(payloads))
        chunksize = max(1, len(payloads) // max(1, workers * 8))
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_replay,
                initargs=(
                    str(registry_path), str(config["config_path"]),
                    manifest["source_commit"], bias_cache,
                )) as pool:
            return list(pool.map(
                _parallel_validate_entry, payloads, chunksize=chunksize,
            ))
    records = []
    for entry in manifest["tasks"]:
        path = raw_root / entry["output_relpath"]
        task = entry["task"]
        if task["method_id"] in HARD_METHODS:
            record = validate_screen_hard_raw(
                path, registry, config, manifest["source_commit"],
            )
        else:
            bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
            cache_key = str(bias_path)
            if cache_key not in bias_cache:
                raise GlobalConflictError("diagnostic serial replay lacks bound bias")
            record = validate_screen_defect_raw(
                path, registry, config, manifest["source_commit"], bias_path,
                _validated_bias_record=bias_cache[cache_key],
            )
        if record["task_fingerprint"] != entry["task_fingerprint"]:
            raise GlobalConflictError("diagnostic manifest/raw fingerprint mismatch")
        records.append(record)
    return records


def _public_summary(summary):
    return {key: value for key, value in summary.items()
            if not key.startswith("_")}


def _screen_status(valid_hard, valid_defect, selected_pair_valid):
    if not valid_hard:
        return "NO_HARD_COSET_PASS"
    if not valid_defect:
        return "NO_DEFECT_TRACE_PASS"
    if selected_pair_valid:
        return "PAIR_FOUND"
    return "NO_CROSS_MECHANISM_AGREEMENT"


def _hard_selection_key(value):
    method = value["method_id"]
    return (
        float(value["core_seconds"]),
        1 if method == "RC8-QC4" else 0,
        int(method[-2:]) if "-J" in method else 0,
        method,
    )


def _defect_selection_key(value):
    method = value["method_id"]
    return (
        -float(value["d0_ess_per_core_second"]),
        int(method[2:]),
        method,
    )


def analyze_screen_measurement_stage(raw_root, manifest_path, registry_path,
                                     config_path, output_path=None, *,
                                     num_workers=1):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    manifest = json.loads(Path(manifest_path).read_text(encoding="ascii"))
    method_tiers = _validate_measurement_manifest_structure(
        manifest, registry, config,
    )
    records = _load_measurement_records(
        raw_root, manifest, registry, config, num_workers=num_workers,
    )
    grouped = defaultdict(list)
    for record in records:
        grouped[(
            sha256_json(record["cell"]), record["method_id"],
            record["resource_tier"],
        )].append(record)
    summaries = [
        _global._cell_method_summary(values, config)
        for _, values in sorted(grouped.items())
    ]
    if len(summaries) != 40:
        raise GlobalConflictError("diagnostic analyzer lacks 40 cell/method groups")

    by_method_tier = defaultdict(list)
    for summary in summaries:
        by_method_tier[(summary["method_id"], summary["resource_tier"])].append(
            summary,
        )
    method_status = []
    for method, tier in method_tiers:
        values = by_method_tier[(method, tier)]
        if len(values) != 5:
            raise GlobalConflictError("diagnostic method does not cover five cells")
        worm_efficiency = None
        if method in DEFECT_METHODS:
            ess_values = [
                family["worm"]["aggregate_d0_ess"]
                for value in values
                for family in value["families"].values()
                if family["worm"] is not None
            ]
            d0_ess = (
                0.0 if any(value is None for value in ess_values)
                else float(sum(ess_values))
            )
            worm_efficiency = d0_ess / max(
                sum(value["core_seconds"] for value in values), 1e-300,
            )
        method_status.append({
            "method_id": method,
            "resource_tier": tier,
            "cells_passed": sum(bool(value["valid"]) for value in values),
            "cells_total": 5,
            "core_seconds": float(sum(
                value["core_seconds"] for value in values
            )),
            "d0_ess_per_core_second": worm_efficiency,
            "valid": all(value["valid"] for value in values),
        })

    summary_index = {
        (sha256_json(value["cell"]), value["method_id"], value["resource_tier"]): value
        for value in summaries
    }
    comparisons = []
    pair_status = []
    hard_tiers = [value for value in method_tiers if value[0] in HARD_METHODS]
    defect_tiers = [value for value in method_tiers if value[0] in DEFECT_METHODS]
    cells = _screen_cells(config)
    for hard_method, hard_tier in hard_tiers:
        for defect_method, defect_tier in defect_tiers:
            cell_comparisons = []
            for cell in cells:
                cell_key = sha256_json(cell)
                comparison = _global.compare_cell_summaries(
                    summary_index[(cell_key, hard_method, hard_tier)],
                    summary_index[(cell_key, defect_method, defect_tier)],
                    config,
                )
                comparison["cell"] = cell
                comparisons.append(comparison)
                cell_comparisons.append(comparison)
            pair_status.append({
                "hard_method_id": hard_method,
                "hard_resource_tier": hard_tier,
                "defect_method_id": defect_method,
                "defect_resource_tier": defect_tier,
                "cells_passed": sum(
                    bool(value["valid"]) for value in cell_comparisons
                ),
                "cells_total": 5,
                "valid": all(value["valid"] for value in cell_comparisons),
            })

    status_index = {
        (value["method_id"], value["resource_tier"]): value
        for value in method_status
    }
    valid_hard = [
        value for value in hard_tiers if status_index[value]["valid"]
    ]
    valid_defect = [
        value for value in defect_tiers if status_index[value]["valid"]
    ]
    valid_hard.sort(key=lambda value: _hard_selection_key(status_index[value]))
    valid_defect.sort(key=lambda value: _defect_selection_key(status_index[value]))
    primary_pair = None
    selected_pair = None
    selected_pair_valid = False
    if valid_hard and valid_defect:
        selected_hard = valid_hard[0]
        selected_defect = valid_defect[0]
        selected_status = next(
            value for value in pair_status
            if (value["hard_method_id"], value["hard_resource_tier"])
            == selected_hard
            and (value["defect_method_id"], value["defect_resource_tier"])
            == selected_defect
        )
        primary_pair = {
            "hard_method_id": selected_hard[0],
            "hard_resource_tier": selected_hard[1],
            "defect_method_id": selected_defect[0],
            "defect_resource_tier": selected_defect[1],
            "agreement_valid": bool(selected_status["valid"]),
        }
        selected_pair_valid = bool(selected_status["valid"])
        if selected_pair_valid:
            selected_pair = dict(primary_pair)
    status = _screen_status(valid_hard, valid_defect, selected_pair_valid)
    identity = {
        "report_version": SCREEN_REPORT_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "stage": SCREEN_STAGE,
        "source_commit": manifest["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "manifest_sha256": sha256_json(manifest),
        "bias_manifest_sha256": manifest["bias_manifest_sha256"],
        "method_tiers": [list(value) for value in method_tiers],
        "screen_panel_sha256": config["screen_panel_sha256"],
        "raw_count": len(records),
        "cell_summaries": [_public_summary(value) for value in summaries],
        "method_status": method_status,
        "comparisons": comparisons,
        "pair_status": pair_status,
        "primary_pair": primary_pair,
        "selected_pair": selected_pair,
        "status": status,
        "formal_authorization": False,
        "production_authorization": False,
    }
    report = {**identity, "report_sha256": sha256_json(identity)}
    if output_path is not None:
        atomic_json(output_path, report)
    return report


analyze_screen = analyze_screen_measurement_stage
analyze_measurement_stage = analyze_screen_measurement_stage


def _report_number(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise GlobalConflictError(f"diagnostic report {name} is not numeric")
    number = float(value)
    if not np.isfinite(number) or (nonnegative and number < 0.0):
        raise GlobalConflictError(f"diagnostic report {name} is invalid")
    return number


def _report_close(left, right):
    return np.isclose(
        float(left), float(right), rtol=1e-12, atol=1e-12,
    )


def _validate_report_worm(worm, no_observations, config):
    fields = {
        "d0_counts", "excursions", "per_chain_d0_ess", "median_d0_ess",
        "aggregate_d0_ess", "boundary_occupancy",
    }
    if not isinstance(worm, dict) or set(worm) != fields:
        raise GlobalConflictError("diagnostic report worm schema mismatch")
    counts = worm["d0_counts"]
    excursions = worm["excursions"]
    boundary = worm["boundary_occupancy"]
    if (not all(isinstance(value, list) and len(value) == 16
                for value in (counts, excursions, boundary))
            or any(_report_number(value, "worm count", nonnegative=True) < 0
                   for value in counts)
            or any(_report_number(value, "worm excursion", nonnegative=True) < 0
                   for value in excursions)
            or any(not 0.0 <= _report_number(
                value, "worm boundary occupancy",
            ) <= 1.0 for value in boundary)):
        raise GlobalConflictError("diagnostic report worm values are malformed")
    failures = []
    gates = config["gates"]
    if no_observations:
        if (worm["per_chain_d0_ess"] is not None
                or worm["median_d0_ess"] is not None
                or worm["aggregate_d0_ess"] is not None):
            raise GlobalConflictError("diagnostic empty worm ESS is not null")
        failures.append("worm_d0_count")
        return failures
    ess = worm["per_chain_d0_ess"]
    if not isinstance(ess, list) or len(ess) != 16:
        raise GlobalConflictError("diagnostic report worm ESS schema mismatch")
    ess = [_report_number(value, "worm ESS", nonnegative=True) for value in ess]
    median = _report_number(worm["median_d0_ess"], "worm median ESS", nonnegative=True)
    aggregate = _report_number(
        worm["aggregate_d0_ess"], "worm aggregate ESS", nonnegative=True,
    )
    if (not _report_close(median, float(np.median(ess)))
            or not _report_close(aggregate, float(sum(ess)))):
        raise GlobalConflictError("diagnostic report worm ESS is inconsistent")
    if any(float(value) < gates["worm_min_d0_per_trajectory"]
           for value in counts):
        failures.append("worm_d0_count")
    if any(float(value) < gates["worm_min_excursions_per_trajectory"]
           for value in excursions):
        failures.append("worm_excursions")
    if median < gates["worm_min_median_d0_ess"]:
        failures.append("worm_median_d0_ess")
    if aggregate < gates["worm_min_family_d0_ess"]:
        failures.append("worm_family_d0_ess")
    if any(float(value) > gates["worm_max_boundary_occupancy"]
           for value in boundary):
        failures.append("worm_boundary")
    return failures


def _validate_report_family(family, family_name, method, config):
    fields = {
        "init_family", "q_top", "q_top_total_se", "q_top_trajectory_se",
        "q_top_character_se", "label_collision_mass_diagnostic",
        "label_collision_q_top_diagnostic", "normalized_mean_weight",
        "normalized_mean_weight_se", "max_rhat",
        "min_nondegenerate_bulk_ess", "constant_failures", "worm",
        "core_seconds", "valid", "failures",
    }
    if (not isinstance(family, dict) or set(family) != fields
            or family.get("init_family") != family_name
            or not isinstance(family.get("valid"), bool)
            or not isinstance(family.get("failures"), list)
            or family["failures"] != sorted(set(family["failures"]))
            or not isinstance(family.get("constant_failures"), list)
            or len(family["constant_failures"])
            != len(set(family["constant_failures"]))):
        raise GlobalConflictError("diagnostic report family schema mismatch")
    _report_number(family["core_seconds"], "family core seconds", nonnegative=True)
    no_observations = family["q_top"] is None
    failures = []
    worm = family["worm"]
    is_defect = method in DEFECT_METHODS
    if is_defect:
        failures.extend(_validate_report_worm(
            worm, no_observations, config,
        ))
    elif worm is not None:
        raise GlobalConflictError("hard-coset report unexpectedly has worm metrics")
    if no_observations:
        nullable = (
            "q_top_total_se", "q_top_trajectory_se", "q_top_character_se",
            "label_collision_mass_diagnostic",
            "label_collision_q_top_diagnostic", "normalized_mean_weight",
            "normalized_mean_weight_se", "max_rhat",
            "min_nondegenerate_bulk_ess",
        )
        if any(family[name] is not None for name in nullable):
            raise GlobalConflictError("empty diagnostic family has numeric results")
        failures.insert(0, "no_valid_observations")
    else:
        for name in (
            "q_top", "label_collision_mass_diagnostic",
            "label_collision_q_top_diagnostic", "normalized_mean_weight",
            "max_rhat",
        ):
            _report_number(family[name], f"family {name}")
        total = _report_number(
            family["q_top_total_se"], "family total q_top SE", nonnegative=True,
        )
        trajectory = _report_number(
            family["q_top_trajectory_se"], "family trajectory q_top SE",
            nonnegative=True,
        )
        character = _report_number(
            family["q_top_character_se"], "family character q_top SE",
            nonnegative=True,
        )
        weight_se = _report_number(
            family["normalized_mean_weight_se"], "family weight SE",
            nonnegative=True,
        )
        if (not _report_close(total, np.hypot(trajectory, character))
                or weight_se < 0.0):
            raise GlobalConflictError("diagnostic family SE is inconsistent")
        if total > config["gates"]["max_q_top_se"]:
            failures.append("q_top_se")
        if float(family["max_rhat"]) > config["gates"]["max_rhat"]:
            failures.append("rhat")
        minimum_ess = family["min_nondegenerate_bulk_ess"]
        if (minimum_ess is not None
                and _report_number(
                    minimum_ess, "family minimum ESS", nonnegative=True,
                ) < config["gates"]["min_bulk_ess"]):
            failures.append("bulk_ess")
        if family["constant_failures"]:
            failures.append("constant_common_freeze")
    expected = sorted(set(failures))
    if (family["failures"] != expected
            or family["valid"] is not (not expected)):
        raise GlobalConflictError("diagnostic report family status is inconsistent")


def _validate_report_summary(summary, expected_cell, method, tier, config):
    fields = {
        "cell", "method_id", "resource_tier", "num_qubits", "families",
        "q_top", "q_top_total_se", "label_collision_mass_diagnostic",
        "label_collision_q_top_diagnostic", "initialization_delta", "d2",
        "normalized_weight_delta", "normalized_weight_delta_se",
        "ti_anchor_payload", "core_seconds", "valid", "failures",
    }
    if (not isinstance(summary, dict) or set(summary) != fields
            or summary.get("cell") != expected_cell
            or summary.get("method_id") != method
            or summary.get("resource_tier") != tier
            or isinstance(summary.get("num_qubits"), bool)
            or not isinstance(summary.get("num_qubits"), int)
            or summary["num_qubits"] <= 0
            or set(summary.get("families", {})) != set(INIT_FAMILIES)
            or not isinstance(summary.get("valid"), bool)
            or not isinstance(summary.get("failures"), list)
            or summary["failures"] != sorted(set(summary["failures"]))):
        raise GlobalConflictError("diagnostic report cell summary schema mismatch")
    for family_name in INIT_FAMILIES:
        _validate_report_family(
            summary["families"][family_name], family_name, method, config,
        )
    family_core = sum(
        float(summary["families"][name]["core_seconds"])
        for name in INIT_FAMILIES
    )
    core = _report_number(
        summary["core_seconds"], "cell core seconds", nonnegative=True,
    )
    if not _report_close(core, family_core):
        raise GlobalConflictError("diagnostic report cell core time is inconsistent")
    families = summary["families"]
    delta = _global._delta_gate(families["P"], families["U"], config)
    if summary.get("initialization_delta") != delta:
        raise GlobalConflictError("diagnostic report initialization delta changed")
    missing = any(families[name]["q_top"] is None for name in INIT_FAMILIES)
    failures = []
    if missing:
        for name in (
            "q_top", "q_top_total_se", "label_collision_mass_diagnostic",
            "label_collision_q_top_diagnostic", "d2",
            "normalized_weight_delta", "normalized_weight_delta_se",
        ):
            if summary[name] is not None:
                raise GlobalConflictError("empty diagnostic summary has estimates")
        failures = ["family_gate", "no_valid_observations"]
    else:
        if not all(families[name]["valid"] for name in INIT_FAMILIES):
            failures.append("family_gate")
        for name in (
            "q_top", "q_top_total_se", "label_collision_mass_diagnostic",
            "label_collision_q_top_diagnostic",
        ):
            _report_number(summary[name], f"cell {name}")
        d2 = summary["d2"]
        d2_fields = {
            "d2_norm", "d2_trajectory_se", "d2_character_se", "d2_total_se",
        }
        if not isinstance(d2, dict) or set(d2) != d2_fields:
            raise GlobalConflictError("diagnostic report D2 schema mismatch")
        d2_norm = _report_number(d2["d2_norm"], "cell D2")
        d2_total = _report_number(
            d2["d2_total_se"], "cell D2 total SE", nonnegative=True,
        )
        d2_trajectory = _report_number(
            d2["d2_trajectory_se"], "cell D2 trajectory SE", nonnegative=True,
        )
        d2_character = _report_number(
            d2["d2_character_se"], "cell D2 character SE", nonnegative=True,
        )
        if not _report_close(d2_total, np.hypot(d2_trajectory, d2_character)):
            raise GlobalConflictError("diagnostic report D2 SE is inconsistent")
        weight_delta = _report_number(
            summary["normalized_weight_delta"], "cell weight delta",
            nonnegative=True,
        )
        weight_se = _report_number(
            summary["normalized_weight_delta_se"], "cell weight delta SE",
            nonnegative=True,
        )
        expected_weight_delta = abs(
            float(families["P"]["normalized_mean_weight"])
            - float(families["U"]["normalized_mean_weight"])
        )
        expected_weight_se = np.hypot(
            float(families["P"]["normalized_mean_weight_se"]),
            float(families["U"]["normalized_mean_weight_se"]),
        )
        if (not _report_close(weight_delta, expected_weight_delta)
                or not _report_close(weight_se, expected_weight_se)):
            raise GlobalConflictError("diagnostic report weight delta is inconsistent")
        if not delta["absolute_pass"] or not delta["sigma_pass"]:
            failures.append("initialization_q_top")
        if max(0.0, d2_norm) + 3.0 * d2_total > config["gates"]["max_d2_upper"]:
            failures.append("initialization_d2")
        if (weight_delta > config["gates"]["max_normalized_weight_delta"]
                or weight_delta > 3.0 * weight_se + 1.0 / summary["num_qubits"]):
            failures.append("initialization_weight")
    payload = summary["ti_anchor_payload"]
    if payload is not None:
        if (not isinstance(payload, dict)
                or set(payload) != {
                    "character_masks", "trajectory_character_means",
                    "payload_sha256",
                }
                or payload["payload_sha256"] != sha256_json({
                    "character_masks": payload["character_masks"],
                    "trajectory_character_means": payload[
                        "trajectory_character_means"
                    ],
                })):
            raise GlobalConflictError("diagnostic TI payload hash mismatch")
    expected_failures = sorted(set(failures))
    if (summary["failures"] != expected_failures
            or summary["valid"] is not (not expected_failures)):
        raise GlobalConflictError("diagnostic report cell status is inconsistent")


def validate_screen_report(report, registry, config):
    if not isinstance(report, dict):
        raise GlobalConflictError("diagnostic report is not a JSON object")
    identity = {key: value for key, value in report.items()
                if key != "report_sha256"}
    try:
        expected_method_tiers = [list(value) for value in _canonical_method_tiers(
            config, report.get("method_tiers", []),
        )]
        _validate_source_commit(report.get("source_commit"))
        _validate_sha256("manifest_sha256", report.get("manifest_sha256"))
        _validate_sha256(
            "bias_manifest_sha256", report.get("bias_manifest_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise GlobalConflictError("diagnostic report identity is malformed") from exc
    expected_fields = {
        "report_version", "contract_version", "stage", "source_commit",
        "registry_sha256", "screen_config_sha256", "manifest_sha256",
        "bias_manifest_sha256", "method_tiers", "screen_panel_sha256",
        "raw_count", "cell_summaries", "method_status", "comparisons",
        "pair_status", "primary_pair", "selected_pair", "status",
        "formal_authorization", "production_authorization", "report_sha256",
    }
    if (set(report) != expected_fields
            or report.get("report_version") != SCREEN_REPORT_VERSION
            or report.get("contract_version") != SCREEN_DIAGNOSTIC_VERSION
            or report.get("stage") != SCREEN_STAGE
            or report.get("registry_sha256") != registry["registry_sha256"]
            or report.get("screen_config_sha256")
            != config["screen_config_sha256"]
            or report.get("screen_panel_sha256")
            != config["screen_panel_sha256"]
            or report.get("report_sha256") != sha256_json(identity)
            or report.get("formal_authorization") is not False
            or report.get("production_authorization") is not False
            or expected_method_tiers != report.get("method_tiers")
            or report.get("raw_count") != 1280
            or len(report.get("cell_summaries", [])) != 40
            or len(report.get("method_status", [])) != 8
            or len(report.get("pair_status", [])) != 15
            or len(report.get("comparisons", [])) != 75):
        raise GlobalConflictError("diagnostic report identity/shape mismatch")
    method_tiers = [tuple(value) for value in expected_method_tiers]
    cells = _screen_cells(config)
    summaries = report["cell_summaries"]
    expected_summary_coordinates = {
        (sha256_json(cell), method, tier)
        for cell in cells for method, tier in method_tiers
    }
    summary_index = {
        (sha256_json(value.get("cell")), value.get("method_id"),
         value.get("resource_tier")): value
        for value in summaries if isinstance(value, dict)
    }
    if (set(summary_index) != expected_summary_coordinates
            or len(summary_index) != len(summaries)):
        raise GlobalConflictError("diagnostic report cell coordinates changed")
    for cell in cells:
        cell_key = sha256_json(cell)
        for method, tier in method_tiers:
            _validate_report_summary(
                summary_index[(cell_key, method, tier)], cell, method, tier,
                config,
            )

    expected_statuses = []
    for method, tier in method_tiers:
        values = [
            value for value in summaries
            if value["method_id"] == method and value["resource_tier"] == tier
        ]
        efficiency = None
        if method in DEFECT_METHODS:
            ess = [
                family["worm"]["aggregate_d0_ess"]
                for value in values
                for family in value["families"].values()
            ]
            d0_ess = 0.0 if any(value is None for value in ess) else sum(ess)
            efficiency = d0_ess / max(
                sum(value["core_seconds"] for value in values), 1e-300,
            )
        expected_statuses.append({
            "method_id": method,
            "resource_tier": tier,
            "cells_passed": sum(bool(value["valid"]) for value in values),
            "cells_total": 5,
            "core_seconds": float(sum(value["core_seconds"] for value in values)),
            "d0_ess_per_core_second": efficiency,
            "valid": all(value["valid"] for value in values),
        })
    method_status_fields = {
        "method_id", "resource_tier", "cells_passed", "cells_total",
        "core_seconds", "d0_ess_per_core_second", "valid",
    }
    if len(report["method_status"]) != len(expected_statuses):
        raise GlobalConflictError("diagnostic report method status is incomplete")
    for actual, expected in zip(report["method_status"], expected_statuses):
        if not isinstance(actual, dict):
            raise GlobalConflictError("diagnostic report method status is malformed")
        actual_efficiency = actual.get("d0_ess_per_core_second")
        expected_efficiency = expected["d0_ess_per_core_second"]
        if (set(actual) != method_status_fields
                or {key: actual[key] for key in method_status_fields
                    if key not in {"core_seconds", "d0_ess_per_core_second"}}
                != {key: expected[key] for key in method_status_fields
                    if key not in {"core_seconds", "d0_ess_per_core_second"}}
                or not _report_close(actual["core_seconds"], expected["core_seconds"])
                or ((actual_efficiency is None) != (expected_efficiency is None))
                or (actual_efficiency is not None and not _report_close(
                    actual_efficiency, expected_efficiency,
                ))):
            raise GlobalConflictError(
                "diagnostic report method status is inconsistent"
            )

    hard_tiers = [value for value in method_tiers if value[0] in HARD_METHODS]
    defect_tiers = [value for value in method_tiers if value[0] in DEFECT_METHODS]
    comparison_fields = {
        "left", "right", "q_top", "d2_norm", "d2_total_se", "d2_pass",
        "normalized_weight_delta", "normalized_weight_delta_se",
        "weight_pass", "valid", "cell",
    }
    expected_comparison_coordinates = [
        (sha256_json(cell), hard_method, hard_tier, defect_method, defect_tier)
        for hard_method, hard_tier in hard_tiers
        for defect_method, defect_tier in defect_tiers
        for cell in cells
    ]
    actual_comparison_coordinates = []
    for comparison in report["comparisons"]:
        if not isinstance(comparison, dict) or set(comparison) != comparison_fields:
            raise GlobalConflictError("diagnostic comparison schema mismatch")
        left_identity = comparison["left"]
        right_identity = comparison["right"]
        if (not isinstance(left_identity, dict)
                or set(left_identity) != {"method_id", "resource_tier"}
                or not isinstance(right_identity, dict)
                or set(right_identity) != {"method_id", "resource_tier"}):
            raise GlobalConflictError("diagnostic comparison identity is malformed")
        coordinate = (
            sha256_json(comparison["cell"]), left_identity["method_id"],
            left_identity["resource_tier"], right_identity["method_id"],
            right_identity["resource_tier"],
        )
        actual_comparison_coordinates.append(coordinate)
        cell_key, hard_method, hard_tier, defect_method, defect_tier = coordinate
        left = summary_index.get((cell_key, hard_method, hard_tier))
        right = summary_index.get((cell_key, defect_method, defect_tier))
        if left is None or right is None:
            raise GlobalConflictError("diagnostic comparison refers to unknown summary")
        expected_q = _global._delta_gate(left, right, config)
        if comparison["q_top"] != expected_q:
            raise GlobalConflictError("diagnostic comparison q_top gate changed")
        missing = left["q_top"] is None or right["q_top"] is None
        if missing:
            expected_values = (None, None, False, None, None, False, False)
        else:
            d2_norm = _report_number(comparison["d2_norm"], "comparison D2")
            d2_se = _report_number(
                comparison["d2_total_se"], "comparison D2 SE",
                nonnegative=True,
            )
            weight_delta = _report_number(
                comparison["normalized_weight_delta"],
                "comparison weight delta", nonnegative=True,
            )
            weight_se = _report_number(
                comparison["normalized_weight_delta_se"],
                "comparison weight SE", nonnegative=True,
            )
            d2_pass = (
                max(0.0, d2_norm) + 3.0 * d2_se
                <= config["gates"]["max_d2_upper"]
            )
            weight_pass = (
                weight_delta <= config["gates"]["max_normalized_weight_delta"]
                and weight_delta <= 3.0 * weight_se + 1.0 / left["num_qubits"]
            )
            valid = bool(
                expected_q["absolute_pass"] and expected_q["sigma_pass"]
                and d2_pass and weight_pass
            )
            expected_values = (
                d2_norm, d2_se, d2_pass, weight_delta, weight_se,
                weight_pass, valid,
            )
        actual_values = (
            comparison["d2_norm"], comparison["d2_total_se"],
            comparison["d2_pass"], comparison["normalized_weight_delta"],
            comparison["normalized_weight_delta_se"],
            comparison["weight_pass"], comparison["valid"],
        )
        if actual_values != expected_values:
            raise GlobalConflictError("diagnostic comparison status is inconsistent")
    if actual_comparison_coordinates != expected_comparison_coordinates:
        raise GlobalConflictError("diagnostic comparison coordinates changed")

    expected_pair_status = []
    offset = 0
    for hard_method, hard_tier in hard_tiers:
        for defect_method, defect_tier in defect_tiers:
            values = report["comparisons"][offset:offset + len(cells)]
            offset += len(cells)
            expected_pair_status.append({
                "hard_method_id": hard_method,
                "hard_resource_tier": hard_tier,
                "defect_method_id": defect_method,
                "defect_resource_tier": defect_tier,
                "cells_passed": sum(bool(value["valid"]) for value in values),
                "cells_total": 5,
                "valid": all(value["valid"] for value in values),
            })
    if report["pair_status"] != expected_pair_status:
        raise GlobalConflictError("diagnostic report pair status is inconsistent")
    valid_hard = [
        value for value in report["method_status"]
        if value["method_id"] in HARD_METHODS and value.get("valid") is True
    ]
    valid_defect = [
        value for value in report["method_status"]
        if value["method_id"] in DEFECT_METHODS and value.get("valid") is True
    ]
    valid_hard.sort(key=_hard_selection_key)
    valid_defect.sort(key=_defect_selection_key)
    expected_primary = None
    expected_selected = None
    selected_pair_valid = False
    if valid_hard and valid_defect:
        hard = valid_hard[0]
        defect = valid_defect[0]
        pair = next((
            value for value in report["pair_status"]
            if value["hard_method_id"] == hard["method_id"]
            and value["hard_resource_tier"] == hard["resource_tier"]
            and value["defect_method_id"] == defect["method_id"]
            and value["defect_resource_tier"] == defect["resource_tier"]
        ), None)
        if pair is None:
            raise GlobalConflictError("diagnostic primary pair status is missing")
        selected_pair_valid = pair.get("valid") is True
        expected_primary = {
            "hard_method_id": hard["method_id"],
            "hard_resource_tier": hard["resource_tier"],
            "defect_method_id": defect["method_id"],
            "defect_resource_tier": defect["resource_tier"],
            "agreement_valid": selected_pair_valid,
        }
        if selected_pair_valid:
            expected_selected = dict(expected_primary)
    expected_status = _screen_status(
        valid_hard, valid_defect, selected_pair_valid,
    )
    if report.get("status") != expected_status:
        raise GlobalConflictError("diagnostic report status is inconsistent")
    if (report.get("primary_pair") != expected_primary
            or report.get("selected_pair") != expected_selected):
        raise GlobalConflictError("diagnostic report selected-pair state is inconsistent")
    return True


FORMAL_BLOCKERS = [
    "NO_T_VS_2T",
    "NO_FRESH_HARD2_CONFIRMATION",
    "NO_CONF17_RES6_GAP8_SMALL6",
    "NO_TI_OR_REVIEWED_INDEPENDENT_ORACLE",
    "NO_HELD_OUT",
]


def terminal_decision(report_path, registry_path, config_path,
                      output_path=None):
    registry = load_registry(registry_path)
    config = load_screen_diagnostic_config(config_path, registry)
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="ascii"))
    validate_screen_report(report, registry, config)
    terminal_status = {
        "PAIR_FOUND": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        "NO_HARD_COSET_PASS": "UNRESOLVED_NO_HARD_COSET_PASS",
        "NO_DEFECT_TRACE_PASS": "UNRESOLVED_NO_DEFECT_TRACE_PASS",
        "NO_CROSS_MECHANISM_AGREEMENT": "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT",
    }[report["status"]]
    identity = {
        "decision_version": SCREEN_DECISION_VERSION,
        "contract_version": SCREEN_DIAGNOSTIC_VERSION,
        "status": terminal_status,
        "maximum_possible_status": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        "source_commit": report["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "screen_config_sha256": config["screen_config_sha256"],
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "selected_pair": report["selected_pair"],
        "formal_authorization": False,
        "production_authorization": False,
        "formal_blockers": FORMAL_BLOCKERS,
    }
    decision = {**identity, "decision_sha256": sha256_json(identity)}
    if output_path is not None:
        atomic_json(output_path, decision)
    return decision


build_terminal_decision = terminal_decision


default_config = default_screen_diagnostic_config
load_config = load_screen_diagnostic_config
task_identity = diagnostic_task_identity
bias_identity = diagnostic_bias_task_identity
build_screen_bias_manifest = build_bias_manifest
build_screen_measurement_manifest = build_measurement_manifest
analyze_screen_measurement = analyze_screen_measurement_stage
screen_uniform_seed_for_cell = uniform_seed_for_cell
validate_diagnostic_control_manifest = validate_control_manifest
SCREEN_HARD_METHODS = HARD_METHODS
SCREEN_DEFECT_METHODS = DEFECT_METHODS
SCREEN_RESOURCE_TIERS = RESOURCE_TIERS
DIAGNOSTIC_TASKS_VERSION = SCREEN_TASKS_VERSION
