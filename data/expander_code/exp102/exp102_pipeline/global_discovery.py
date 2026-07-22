"""Fail-closed workflow for the exp102 q=0 global-sampling discovery.

This workflow is isolated from every PT/PA pilot, raw file, namespace, freezer,
and production loader.  Its strongest successful status is
``READY_FOR_FORMAL``; it cannot create a held-out production marker.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .diagnostics import split_rhat
from .exp101_bridge import load_exp101
from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .pa_discovery import (
    CONFIRMATION_CELLS,
    CONFIRMATION_PANEL_SHA256,
    HARD_CELLS,
    RESOLUTION_CELLS,
    RESOLUTION_PANEL_SHA256,
)
from .q0_global import (
    CHARACTER_SET_VERSION,
    DEFECT_BIAS_RAW_VERSION,
    DEFECT_METHODS,
    DEFECT_TRACE_RAW_VERSION,
    GLOBAL_DISCOVERY_VERSION,
    HARD_COSET_RAW_VERSION,
    HARD_METHODS,
    DefectTraceConfig,
    CharacterSet,
    GlobalConflictError,
    GlobalSeedIdentity,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    canonical_global_trajectory_digest,
    character_d2_estimate,
    character_means,
    character_qtop_estimate,
    character_values,
    frozen_character_set,
    label_collision_diagnostic,
    pack_state,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    state_label,
    trajectory_mean_and_se,
    tune_defect_bias,
    uniform_hard_coset_state,
    unpack_states,
)
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


GLOBAL_CONFIG_VERSION = "exp102.q0_global.config.v1"
GLOBAL_TASKS_VERSION = "exp102.q0_global.tasks.v1"
GLOBAL_SELECTION_VERSION = "exp102.q0_global.selection.v1"
GLOBAL_REPORT_VERSION = "exp102.q0_global.report.v1"
TI_ANCHOR_RAW_VERSION = "exp102.q0_global.ti_anchor.raw.v1"
TI_ANCHOR_REPORT_VERSION = "exp102.q0_global.ti_anchor.report.v1"
TI_COMPARISON_VERSION = "exp102.q0_global.ti_comparison.v1"
GLOBAL_READINESS_VERSION = "exp102.q0_global.readiness.v1"
GLOBAL_SCHEDULE_VERSION = "exp102.q0_global.schedule.v1"
GLOBAL_POSTSELECTION_VERSION = "exp102.q0_global.postselection_plan.v1"
GLOBAL_CONTROL_FREEZE_VERSION = "exp102.q0_global.control_freeze.v1"
NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
INIT_FAMILIES = ("P", "U")
TRAJECTORIES_PER_FAMILY = 16
RESOURCE_TIERS = {
    "T1": {"burn_sweeps": 2048, "measurement_sweeps": 8192},
    "T2": {"burn_sweeps": 4096, "measurement_sweeps": 16384},
    "T3": {"burn_sweeps": 8192, "measurement_sweeps": 32768},
}

EASY_CELLS = (
    {"code_id": "m03_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m04_c00", "p": 0.07, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
    {"code_id": "m05_c00", "p": 0.10, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"},
)
GAP_CELLS = tuple(
    {"code_id": code_id, "p": p, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"}
    for code_id in ("m06_c00", "m08_c06")
    for p in (0.05, 0.06, 0.08, 0.09)
)
SMALL_CELLS = tuple(
    {"code_id": code_id, "p": p, "disorder_index": 0,
     "disorder_source": "global_fresh_v1"}
    for code_id in ("m03_c00", "m04_c00")
    for p in (0.04, 0.07, 0.10)
)

# These values are checked again while constructing the frozen config.
GAP_PANEL_SHA256 = "25c14dd7b5ddfc1725a6fdcd6629a70319ef97f020eaa583ae67d78a598b8aae"
SMALL_PANEL_SHA256 = "018a52aa41153b36d9fc869d2f7f7308fa00258166b43f6404b713b117efe484"
EASY_PANEL_SHA256 = "ec110e8550b18064c747fd2418c5134b594d693e70edcef83b093b74cdf162b2"

HARD_RAW_FIELDS = {
    "raw_version", "discovery_version", "task_fingerprint", "task_json",
    "cell_json", "config_json", "seed_identity_json", "source_commit",
    "registry_sha256", "discovery_config_sha256", "uniform_seed", "engine",
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
    *(HARD_RAW_FIELDS - {"raw_version", "catalog_sha256", "joint_sha256",
                         "measurement_residual_weights"}),
    "raw_version", "bias_task_fingerprint", "bias_raw_sha256", "bias_sha256",
    "bias", "measurement_defect_counts", "fixed_clock_d0_mask",
    "boundary_occupancy",
}
BIAS_RAW_FIELDS = {
    "raw_version", "discovery_version", "task_fingerprint", "task_json",
    "cell_json", "config_json", "source_commit", "registry_sha256",
    "discovery_config_sha256", "uniform_seed", "engine", "model_fingerprint",
    "num_qubits", "tuning_seed_identities_json", "bias", "bias_trace",
    "tuning_histogram", "tuning_final_states_packed",
    "tuning_final_residuals", "tuning_final_defects", "gammas", "bias_sha256",
    "core_seconds", "wall_seconds",
}
TI_ANCHOR_RAW_FIELDS = {
    "raw_version", "discovery_version", "task_fingerprint", "task_json",
    "cell_json", "ti_config_json", "source_commit", "registry_sha256",
    "discovery_config_sha256", "uniform_seed", "engine_seed", "engine",
    "model_fingerprint", "section_fingerprint", "logical_frame_fingerprint",
    "num_qubits", "k", "labels", "kp_grid", "delta_f",
    "delta_f_infinite_mask", "delta_f_stderr", "acceptance_per_label",
    "weights_absolute", "characters_absolute", "q_top", "q_top_stderr",
    "grid_tv", "grid_q_top_abs_diff", "flags", "valid_for_aggregation",
    "proposal_summary_json", "trajectory_digest", "core_seconds",
    "wall_seconds",
}


def _cell(code_id, p, disorder_index, source):
    return {
        "code_id": str(code_id), "p": float(p),
        "disorder_index": int(disorder_index), "disorder_source": str(source),
    }


def _normalize_cell(cell):
    if not isinstance(cell, dict) or set(cell) != {
            "code_id", "p", "disorder_index", "disorder_source"}:
        raise ValueError("global discovery cell schema mismatch")
    return _cell(
        cell["code_id"], cell["p"], cell["disorder_index"], cell["disorder_source"],
    )


def uniform_seed_for_cell(registry, code, cell):
    cell = _normalize_cell(cell)
    source = cell["disorder_source"]
    if source == "attempt022":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt22"
    elif source == "fresh_v2":
        namespace = "discovery_v2_fresh_disorder"
    elif source == "global_fresh_v1":
        namespace = "q0_global_discovery_fresh_v1"
    else:
        raise ValueError("unknown global disorder source")
    return derive_seed(
        namespace, registry["registry_sha256"], code["code_id"],
        cell["disorder_index"], "uniforms",
    )


def _all_panels():
    return {
        "HARD2": [dict(value) for value in HARD_CELLS],
        "EASY3": [dict(value) for value in EASY_CELLS],
        "CONF17": [dict(value) for value in CONFIRMATION_CELLS],
        "RES6": [dict(value) for value in RESOLUTION_CELLS],
        "GAP8": [dict(value) for value in GAP_CELLS],
        "SMALL6": [dict(value) for value in SMALL_CELLS],
    }


def default_global_discovery_config(registry):
    panels = _all_panels()
    expected_hashes = {
        "HARD2": sha256_json(panels["HARD2"]),
        "EASY3": EASY_PANEL_SHA256,
        "CONF17": CONFIRMATION_PANEL_SHA256,
        "RES6": RESOLUTION_PANEL_SHA256,
        "GAP8": GAP_PANEL_SHA256,
        "SMALL6": SMALL_PANEL_SHA256,
    }
    for name, expected in expected_hashes.items():
        if expected == "TO_BE_FROZEN":
            continue
        if sha256_json(panels[name]) != expected:
            raise AssertionError(f"{name} ordered panel SHA256 changed")

    unique_cells = []
    seen = set()
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    for name in ("HARD2", "EASY3", "CONF17", "RES6", "GAP8", "SMALL6"):
        for value in panels[name]:
            fingerprint = sha256_json(value)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_cells.append({
                "cell_fingerprint": fingerprint,
                "uniform_seed": uniform_seed_for_cell(
                    registry, code_by_id[value["code_id"]], value,
                ),
            })
    panel_records = {
        name: {
            "cells": cells,
            "ordered_panel_sha256": sha256_json(cells),
        }
        for name, cells in panels.items()
    }
    config = {
        "config_version": GLOBAL_CONFIG_VERSION,
        "discovery_version": GLOBAL_DISCOVERY_VERSION,
        "hard_raw_version": HARD_COSET_RAW_VERSION,
        "defect_raw_version": DEFECT_TRACE_RAW_VERSION,
        "defect_bias_raw_version": DEFECT_BIAS_RAW_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "historical_formal_versions": {
            "physics": "exp102.physics.v1",
            "pt": "exp102.q0_pt.v1",
            "scan": "exp102.scan.v1",
        },
        "catalog": {"max_multiple": 8, "max_count": 512},
        "characters": {
            "version": CHARACTER_SET_VERSION,
            "full_max_k": 10,
            "num_nonbasis": 4096,
            "seed_namespace": "q0_global_characters_v1",
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
        "ti_anchor": {
            "method_id": "FULL-SECTOR-TI49",
            "num_kp_grid_points": 49,
            "num_burn_in_sweeps": 400,
            "num_measurements": 1200,
            "num_sweeps_between_measurements": 1,
            "block_count": 10,
            "num_bootstrap": 400,
            "full_max_k": 10,
            "grid_tv_warning": 0.02,
            "grid_q_top_warning": 0.01,
            "seed_namespace": "q0_global_ti_anchor_v1",
        },
        "resource_selection": {
            "capacity_nodes": ["nd-2", "nd-3"],
            "capacity": NODE_CAPACITY["nd-2"] + NODE_CAPACITY["nd-3"],
            "safety_factor": 2.0,
            "max_projected_hours": 58.0,
            "max_trajectory_hours": 2.0,
            "strict_multiplier": 2,
        },
        "trajectory_count_per_init_family": TRAJECTORIES_PER_FAMILY,
        "init_families": list(INIT_FAMILIES),
        "panels": panel_records,
        "uniform_seeds": unique_cells,
        "uniform_seeds_sha256": sha256_json(unique_cells),
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
            "linux_digest_and_runtime_hours": [0, 8],
            "candidate_screen_hours": [8, 20],
            "freeze_deadline_hour": 20,
            "hard_fresh_hours": [20, 44],
            "confirmation_hours": [44, 66],
            "analysis_hours": [66, 72],
            "wall_limit_hours": 72,
            "diagnostic_boundary_max_hours": 12,
        },
    }
    return config


def write_default_global_config(registry_path, output_path):
    config = default_global_discovery_config(load_registry(registry_path))
    atomic_json(output_path, config)
    return config


def freeze_global_schedule(registry_path, config_path, source_commit,
                           archive_sha256, source_manifest_sha256, output_path,
                           *, started_unix=None):
    _validate_source_commit(source_commit)
    for name, value in (
            ("archive_sha256", archive_sha256),
            ("source_manifest_sha256", source_manifest_sha256)):
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
            raise ValueError(f"global schedule {name} is malformed")
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    started = time.time() if started_unix is None else float(started_unix)
    if not np.isfinite(started) or started <= 0.0:
        raise ValueError("global schedule start time is invalid")
    hours = config["schedule"]
    deadlines = {
        "digest_runtime": started + hours["linux_digest_and_runtime_hours"][1] * 3600.0,
        "screen": started + hours["candidate_screen_hours"][1] * 3600.0,
        "freeze": started + hours["freeze_deadline_hour"] * 3600.0,
        "hard_fresh": started + hours["hard_fresh_hours"][1] * 3600.0,
        "confirmation": started + hours["confirmation_hours"][1] * 3600.0,
        "resolution": started + hours["confirmation_hours"][1] * 3600.0,
        "ti_anchors": started + hours["confirmation_hours"][1] * 3600.0,
        "analysis": started + hours["analysis_hours"][1] * 3600.0,
    }
    identity = {
        "schedule_version": GLOBAL_SCHEDULE_VERSION,
        "status": "FROZEN_72H",
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "started_unix": started,
        "deadlines_unix": deadlines,
        "wall_limit_hours": hours["wall_limit_hours"],
    }
    schedule = {**identity, "schedule_sha256": sha256_json(identity)}
    output_path = Path(output_path)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="ascii"))
        if existing != schedule:
            raise GlobalConflictError("existing global 72-hour schedule conflicts")
    else:
        atomic_json(output_path, schedule)
    return schedule


def validate_global_schedule(path, registry, config, source_commit=None):
    schedule = json.loads(Path(path).read_text(encoding="ascii"))
    identity = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    if (schedule.get("schedule_version") != GLOBAL_SCHEDULE_VERSION
            or schedule.get("status") != "FROZEN_72H"
            or schedule.get("schedule_sha256") != sha256_json(identity)
            or schedule.get("registry_sha256") != registry["registry_sha256"]
            or schedule.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or (source_commit is not None
                and schedule.get("source_commit") != source_commit)):
        raise GlobalConflictError("global 72-hour schedule identity mismatch")
    expected_keys = {
        "digest_runtime", "screen", "freeze", "hard_fresh",
        "confirmation", "resolution", "ti_anchors", "analysis",
    }
    if (set(schedule.get("deadlines_unix", {})) != expected_keys
            or any(not np.isfinite(float(value))
                   for value in schedule["deadlines_unix"].values())
            or float(schedule["started_unix"])
            >= min(float(value) for value in schedule["deadlines_unix"].values())):
        raise GlobalConflictError("global 72-hour schedule deadlines are malformed")
    return schedule


def freeze_postselection_plan(selection_path, registry_path, config_path,
                              schedule_path, output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    schedule = validate_global_schedule(schedule_path, registry, config)
    selection = json.loads(Path(selection_path).read_text(encoding="ascii"))
    selection_identity = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    if (selection.get("selection_version") != GLOBAL_SELECTION_VERSION
            or selection.get("selection_sha256") != sha256_json(selection_identity)
            or selection.get("schedule_sha256") != schedule["schedule_sha256"]
            or selection.get("schedule_file_sha256") != sha256_file(schedule_path)):
        raise GlobalConflictError("postselection plan received an invalid selection")
    frozen_unix = time.time()
    if frozen_unix > float(schedule["deadlines_unix"]["freeze"]):
        raise GlobalConflictError("postselection plan missed the hour-20 deadline")
    selected = [
        (value["method_id"], value["resource_tier"])
        for value in selection["selected"]
    ]
    base_tier = selected[0][1]
    strict_tier = "2" + base_tier
    stage_method_tiers = {
        "hard_fresh": [
            [method, tier]
            for method, _ in selected for tier in (base_tier, strict_tier)
        ],
        "confirmation": [[method, strict_tier] for method, _ in selected],
        "resolution": [[method, base_tier] for method, _ in selected],
        "ti_anchors": [[config["ti_anchor"]["method_id"], "TI49"]],
    }
    identity = {
        "plan_version": GLOBAL_POSTSELECTION_VERSION,
        "status": "FROZEN_POSTSELECTION",
        "source_commit": selection["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "frozen_unix": frozen_unix,
        "stage_method_tiers": stage_method_tiers,
        "stage_panel_sha256": {
            "hard_fresh": config["panels"]["HARD2"]["ordered_panel_sha256"],
            "confirmation": sha256_json(_stage_cells(config, "confirmation")),
            "resolution": config["panels"]["RES6"]["ordered_panel_sha256"],
            "ti_anchors": sha256_json(_stage_cells(config, "ti_anchors")),
        },
    }
    plan = {**identity, "plan_sha256": sha256_json(identity)}
    output_path = Path(output_path)
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="ascii")) != plan:
            raise GlobalConflictError("existing postselection plan conflicts")
    else:
        atomic_json(output_path, plan)
    return plan


def load_global_discovery_config(path, registry=None):
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    if registry is None:
        if raw.get("discovery_version") != GLOBAL_DISCOVERY_VERSION:
            raise ValueError("global discovery config version mismatch")
    else:
        expected = default_global_discovery_config(registry)
        if raw != expected:
            raise ValueError("global discovery config differs from the frozen protocol")
    return {
        **raw,
        "discovery_config_sha256": sha256_json(raw),
        "config_path": str(Path(path).resolve()),
    }


def _resource_values(config, tier):
    strict = str(tier).startswith("2")
    base = str(tier)[1:] if strict else str(tier)
    if base not in config["resource_tiers"]:
        raise ValueError("unknown global resource tier")
    multiplier = 2 if strict else 1
    values = config["resource_tiers"][base]
    return base, multiplier * int(values["burn_sweeps"]), multiplier * int(values["measurement_sweeps"])


def resolved_sampler_config(config, method_id, p, resource_tier):
    _, burn, measurement = _resource_values(config, resource_tier)
    if method_id in HARD_METHODS:
        return HardCosetConfig(method_id, p, burn, measurement)
    if method_id in DEFECT_METHODS:
        tuning = config["defect_tuning"]
        return DefectTraceConfig(
            method_id, p, burn, measurement,
            tuning_chains=tuning["num_chains"],
            tuning_sweeps=tuning["num_sweeps"],
        )
    raise ValueError("unknown global sampler method")


def _stage_cells(config, stage):
    panels = config["panels"]
    if stage == "screen":
        return [*panels["HARD2"]["cells"], *panels["EASY3"]["cells"]]
    if stage == "hard_fresh":
        return panels["HARD2"]["cells"]
    if stage == "confirmation":
        return [
            *panels["CONF17"]["cells"], *panels["GAP8"]["cells"],
            *panels["SMALL6"]["cells"],
        ]
    if stage == "resolution":
        return panels["RES6"]["cells"]
    if stage == "diagnostic_boundary":
        return panels["GAP8"]["cells"]
    if stage == "ti_anchors":
        return [
            cell for cell in panels["SMALL6"]["cells"]
            if cell["code_id"] == "m03_c00"
        ]
    raise ValueError("unknown global discovery stage")


def _validate_source_commit(source_commit):
    if re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None:
        raise ValueError("global source commit must be a full lowercase Git SHA")


def _validate_task_cell(config, stage, cell):
    cell = _normalize_cell(cell)
    if cell not in _stage_cells(config, stage):
        raise ValueError("global task cell is outside the frozen stage panel")
    return cell


def character_seed(registry_sha256, code_id):
    return derive_seed("q0_global_characters_v1", registry_sha256, code_id)


def global_task_identity(registry, config, source_commit, stage, method_id,
                         resource_tier, cell, init_family, trajectory_index,
                         *, bias_binding=None):
    _validate_source_commit(source_commit)
    cell = _validate_task_cell(config, stage, cell)
    if method_id not in (*HARD_METHODS, *DEFECT_METHODS):
        raise ValueError("unknown global task method")
    if init_family not in INIT_FAMILIES:
        raise ValueError("global task initialization family is invalid")
    if not 0 <= int(trajectory_index) < TRAJECTORIES_PER_FAMILY:
        raise ValueError("global trajectory index is outside the frozen range")
    sampler = resolved_sampler_config(config, method_id, cell["p"], resource_tier)
    if method_id in DEFECT_METHODS:
        if not isinstance(bias_binding, dict) or set(bias_binding) != {
                "bias_task_fingerprint", "bias_raw_sha256", "bias_sha256"}:
            raise ValueError("defect measurement task requires an exact bias binding")
    elif bias_binding is not None:
        raise ValueError("hard-coset task must not have a defect bias binding")
    cell_fingerprint = sha256_json(cell)
    seed_identity = GlobalSeedIdentity(
        source_commit=source_commit,
        config_sha256=config["discovery_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=cell_fingerprint,
        method_id=method_id,
        resource_tier=str(resource_tier),
        init_family=init_family,
        trajectory_index=int(trajectory_index),
        trajectory_namespace=f"q0_global_{stage}_v1",
    )
    return {
        "task_version": GLOBAL_TASKS_VERSION,
        "raw_version": HARD_COSET_RAW_VERSION if method_id in HARD_METHODS else DEFECT_TRACE_RAW_VERSION,
        "stage": stage,
        "method_id": method_id,
        "resource_tier": str(resource_tier),
        "init_family": init_family,
        "trajectory_index": int(trajectory_index),
        "cell": cell,
        "sampler_config": sampler.as_dict(),
        "seed_identity": seed_identity.as_dict(),
        "bias_binding": bias_binding,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "engine": "numba",
    }


def bias_task_identity(registry, config, source_commit, stage, method_id,
                       resource_tier, cell):
    _validate_source_commit(source_commit)
    cell = _validate_task_cell(config, stage, cell)
    if method_id not in DEFECT_METHODS:
        raise ValueError("bias task requires a defect-trace method")
    sampler = resolved_sampler_config(config, method_id, cell["p"], resource_tier)
    seeds = [GlobalSeedIdentity(
        source_commit=source_commit,
        config_sha256=config["discovery_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method_id,
        resource_tier=str(resource_tier),
        init_family="TUNE",
        trajectory_index=index,
        trajectory_namespace=f"q0_global_{stage}_bias_v1",
    ).as_dict() for index in range(sampler.tuning_chains)]
    return {
        "task_version": GLOBAL_TASKS_VERSION,
        "raw_version": DEFECT_BIAS_RAW_VERSION,
        "stage": stage,
        "method_id": method_id,
        "resource_tier": str(resource_tier),
        "cell": cell,
        "sampler_config": sampler.as_dict(),
        "tuning_seed_identities": seeds,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "engine": "numba",
    }


def ti_anchor_task_identity(registry, config, source_commit, cell):
    _validate_source_commit(source_commit)
    cell = _validate_task_cell(config, "ti_anchors", cell)
    ti_config = dict(config["ti_anchor"])
    engine_seed = derive_seed(
        ti_config["seed_namespace"], source_commit,
        config["discovery_config_sha256"], registry["registry_sha256"],
        sha256_json(cell), "full_sector_ti",
    )
    return {
        "task_version": GLOBAL_TASKS_VERSION,
        "raw_version": TI_ANCHOR_RAW_VERSION,
        "stage": "ti_anchors",
        "method_id": ti_config["method_id"],
        "resource_tier": "TI49",
        "cell": cell,
        "ti_config": ti_config,
        "engine_seed": engine_seed,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "engine": "exp101_certified_full_sector_ti",
    }


def _cell_disorder(registry, code, model, cell):
    uniform_seed = uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
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
    raise ValueError("measurement trajectory cannot use TUNE initialization")


def _raw_common(task, registry, config, code, model, frame, uniform_seed,
                characters, sampler, seed_identity, core_seconds, wall_seconds):
    return {
        "discovery_version": np.array(GLOBAL_DISCOVERY_VERSION),
        "task_fingerprint": np.array(sha256_json(task)),
        "task_json": np.array(canonical_json(task)),
        "cell_json": np.array(canonical_json(task["cell"])),
        "config_json": np.array(canonical_json(sampler.as_dict())),
        "seed_identity_json": np.array(canonical_json(seed_identity.as_dict())),
        "source_commit": np.array(task["source_commit"]),
        "registry_sha256": np.array(registry["registry_sha256"]),
        "discovery_config_sha256": np.array(config["discovery_config_sha256"]),
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


def _result_arrays(result):
    names = (
        "initial_state_packed", "burn_state_packed", "final_state_packed",
        "measurement_states_packed", "burn_labels", "measurement_labels",
        "measurement_weights", "measurement_block", "burn_counters",
        "measurement_counters", "burn_basis_seen", "initial_label", "burn_label",
        "final_label",
    )
    arrays = {name: np.asarray(result[name]) for name in names}
    arrays["trajectory_digest"] = np.array(canonical_global_trajectory_digest(result))
    return arrays


def run_hard_task(registry_path, config_path, source_commit, task, output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    expected = global_task_identity(
        registry, config, source_commit, task.get("stage"), task.get("method_id"),
        task.get("resource_tier"), task.get("cell"), task.get("init_family"),
        task.get("trajectory_index"), bias_binding=None,
    )
    if task != expected or expected["method_id"] not in HARD_METHODS:
        raise GlobalConflictError("hard-coset task identity is noncanonical or tampered")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_hard_raw(output_path, registry, config, source_commit)
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing hard-coset raw conflicts with task")
        return "reused"
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(registry, code, model, task["cell"])
    sampler = resolved_sampler_config(config, task["method_id"], task["cell"]["p"], task["resource_tier"])
    seed_identity = GlobalSeedIdentity(**task["seed_identity"])
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
        model.k, character_seed(registry["registry_sha256"], code["code_id"]),
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
        task, registry, config, code, model, frame, uniform_seed, characters,
        sampler, seed_identity, core_seconds, wall_seconds,
    )
    arrays.update(_result_arrays(result))
    arrays.update({
        "raw_version": np.array(HARD_COSET_RAW_VERSION),
        "catalog_sha256": np.array(catalog.catalog_sha256),
        "joint_sha256": np.array(result["joint_sha256"]),
        "measurement_residual_weights": result["measurement_residual_weights"],
    })
    atomic_npz(output_path, **arrays)
    return "computed"


def run_bias_task(registry_path, config_path, source_commit, task, output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    expected = bias_task_identity(
        registry, config, source_commit, task.get("stage"), task.get("method_id"),
        task.get("resource_tier"), task.get("cell"),
    )
    if task != expected:
        raise GlobalConflictError("defect bias task identity is noncanonical or tampered")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_bias_raw(output_path, registry, config, source_commit)
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing defect bias raw conflicts with task")
        return "reused"
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, _, syndrome = _cell_disorder(registry, code, model, task["cell"])
    sampler = resolved_sampler_config(config, task["method_id"], task["cell"]["p"], task["resource_tier"])
    identities = [GlobalSeedIdentity(**value) for value in task["tuning_seed_identities"]]
    wall_start, core_start = time.monotonic(), time.process_time()
    result = tune_defect_bias(model, syndrome, sampler, identities, engine="numba")
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    atomic_npz(
        output_path,
        raw_version=np.array(DEFECT_BIAS_RAW_VERSION),
        discovery_version=np.array(GLOBAL_DISCOVERY_VERSION),
        task_fingerprint=np.array(sha256_json(task)),
        task_json=np.array(canonical_json(task)),
        cell_json=np.array(canonical_json(task["cell"])),
        config_json=np.array(canonical_json(sampler.as_dict())),
        source_commit=np.array(source_commit),
        registry_sha256=np.array(registry["registry_sha256"]),
        discovery_config_sha256=np.array(config["discovery_config_sha256"]),
        uniform_seed=np.array(uniform_seed, dtype=np.int64),
        engine=np.array("numba"),
        model_fingerprint=np.array(model.fingerprint()),
        num_qubits=np.array(model.num_qubits, dtype=np.int32),
        tuning_seed_identities_json=np.array(canonical_json(task["tuning_seed_identities"])),
        bias=result["bias"], bias_trace=result["bias_trace"],
        tuning_histogram=result["tuning_histogram"],
        tuning_final_states_packed=result["tuning_final_states_packed"],
        tuning_final_residuals=result["tuning_final_residuals"],
        tuning_final_defects=result["tuning_final_defects"],
        gammas=result["gammas"], bias_sha256=np.array(result["bias_sha256"]),
        core_seconds=np.array(core_seconds), wall_seconds=np.array(wall_seconds),
    )
    return "computed"


def bias_binding_from_raw(path, registry, config, source_commit=None):
    record = validate_bias_raw(path, registry, config, source_commit)
    return {
        "bias_task_fingerprint": record["task_fingerprint"],
        "bias_raw_sha256": sha256_file(path),
        "bias_sha256": record["bias_sha256"],
    }


def run_defect_task(registry_path, config_path, source_commit, task, bias_path,
                    output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    binding = bias_binding_from_raw(bias_path, registry, config, source_commit)
    expected = global_task_identity(
        registry, config, source_commit, task.get("stage"), task.get("method_id"),
        task.get("resource_tier"), task.get("cell"), task.get("init_family"),
        task.get("trajectory_index"), bias_binding=binding,
    )
    if task != expected or task["method_id"] not in DEFECT_METHODS:
        raise GlobalConflictError("defect task identity is noncanonical or tampered")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_defect_raw(output_path, registry, config, source_commit, bias_path)
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing defect raw conflicts with task")
        return "reused"
    with np.load(bias_path, allow_pickle=False) as bias_data:
        bias = bias_data["bias"].copy()
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(registry, code, model, task["cell"])
    sampler = resolved_sampler_config(config, task["method_id"], task["cell"]["p"], task["resource_tier"])
    seed_identity = GlobalSeedIdentity(**task["seed_identity"])
    initial = _initial_state(model, epsilon, syndrome, seed_identity)
    characters = frozen_character_set(
        model.k, character_seed(registry["registry_sha256"], code["code_id"]),
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
        task, registry, config, code, model, frame, uniform_seed, characters,
        sampler, seed_identity, core_seconds, wall_seconds,
    )
    arrays.update(_result_arrays(result))
    arrays.update({
        "raw_version": np.array(DEFECT_TRACE_RAW_VERSION),
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


def _ti_runtime_config(config):
    load_exp101()
    from exp101_certified_src.sector_ti import SectorTiConfig

    values = dict(config["ti_anchor"])
    values.pop("method_id")
    values.pop("seed_namespace")
    return SectorTiConfig(**values)


def _execute_ti_anchor(registry, config, registry_path, task):
    load_exp101()
    from exp101_certified_src.model import DisorderRealization, wire_ensemble
    from exp101_certified_src.sector_ti import run_sector_ti

    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    disorder = DisorderRealization(
        epsilon_data_true=epsilon,
        measurement_error=np.zeros(model.num_checks, dtype=np.uint8),
        effective_syndrome=syndrome,
        p=task["cell"]["p"],
        q=0.0,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    result = run_sector_ti(
        model, frame, wiring, _ti_runtime_config(config),
        seed=int(task["engine_seed"]),
    )
    return code, model, frame, uniform_seed, result


def _ti_result_arrays(result):
    labels = np.asarray(result["labels"], dtype=np.int32)
    acceptance = np.asarray([
        result["acceptance_per_label"][int(label)] for label in labels
    ], dtype=np.float64)
    return {
        "labels": labels,
        "kp_grid": np.asarray(result["kp_grid"], dtype=np.float64),
        "delta_f": np.asarray(result["delta_f"], dtype=np.float64),
        "delta_f_infinite_mask": np.asarray(
            result["delta_f_infinite_mask"], dtype=np.bool_,
        ),
        "delta_f_stderr": np.asarray(result["delta_f_stderr"], dtype=np.float64),
        "acceptance_per_label": acceptance,
        "weights_absolute": np.asarray(result["weights_absolute"], dtype=np.float64),
        "characters_absolute": np.asarray(
            result["characters_absolute"], dtype=np.float64,
        ),
        "q_top": np.array(float(result["q_top"])),
        "q_top_stderr": np.array(float(result["q_top_stderr"])),
        "grid_tv": np.array(float(result["grid_tv"])),
        "grid_q_top_abs_diff": np.array(float(result["grid_q_top_abs_diff"])),
        "flags": np.array(str(result["flags"])),
        "valid_for_aggregation": np.array(bool(result["valid_for_aggregation"])),
        "proposal_summary_json": np.array(canonical_json(result["proposal_summary"])),
    }


def _ti_result_digest(arrays):
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_global.ti_anchor.digest.v1\0")
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.kind == "f":
            value = np.round(value.astype(np.float64), decimals=12)
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=">u8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def run_ti_anchor_task(registry_path, config_path, source_commit, task,
                       output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    expected = ti_anchor_task_identity(
        registry, config, source_commit, task.get("cell"),
    )
    if task != expected:
        raise GlobalConflictError("TI anchor task identity is noncanonical or tampered")
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_ti_anchor_raw(
            output_path, registry, config, registry_path, source_commit,
        )
        if record["task_fingerprint"] != sha256_json(task):
            raise GlobalConflictError("existing TI anchor raw conflicts with task")
        return "reused"
    wall_start, core_start = time.monotonic(), time.process_time()
    code, model, frame, uniform_seed, result = _execute_ti_anchor(
        registry, config, registry_path, task,
    )
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    result_arrays = _ti_result_arrays(result)
    atomic_npz(
        output_path,
        raw_version=np.array(TI_ANCHOR_RAW_VERSION),
        discovery_version=np.array(GLOBAL_DISCOVERY_VERSION),
        task_fingerprint=np.array(sha256_json(task)),
        task_json=np.array(canonical_json(task)),
        cell_json=np.array(canonical_json(task["cell"])),
        ti_config_json=np.array(canonical_json(config["ti_anchor"])),
        source_commit=np.array(source_commit),
        registry_sha256=np.array(registry["registry_sha256"]),
        discovery_config_sha256=np.array(config["discovery_config_sha256"]),
        uniform_seed=np.array(uniform_seed, dtype=np.int64),
        engine_seed=np.array(task["engine_seed"], dtype=np.int64),
        engine=np.array("exp101_certified_full_sector_ti"),
        model_fingerprint=np.array(model.fingerprint()),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
        num_qubits=np.array(model.num_qubits, dtype=np.int32),
        k=np.array(model.k, dtype=np.int16),
        **result_arrays,
        trajectory_digest=np.array(_ti_result_digest(result_arrays)),
        core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds),
    )
    return "computed"


def _scalar(data, field):
    if field not in data or data[field].shape != ():
        raise GlobalConflictError(f"global raw scalar is missing or malformed: {field}")
    return data[field].item()


def _require_equal(field, stored, expected):
    if not np.array_equal(np.asarray(stored), np.asarray(expected), equal_nan=True):
        raise GlobalConflictError(f"global raw replay mismatch: {field}")


def _read_task(data):
    try:
        task = json.loads(str(_scalar(data, "task_json")))
    except Exception as exc:
        raise GlobalConflictError("global raw task JSON is malformed") from exc
    if str(_scalar(data, "task_fingerprint")) != sha256_json(task):
        raise GlobalConflictError("global raw task fingerprint mismatch")
    return task


def _validate_common_raw(data, expected_fields, registry, config,
                         expected_source_commit, expected_raw_version):
    if set(data.files) != expected_fields:
        missing = sorted(expected_fields - set(data.files))
        extra = sorted(set(data.files) - expected_fields)
        raise GlobalConflictError(f"global raw schema mismatch; missing={missing}, extra={extra}")
    task = _read_task(data)
    if str(_scalar(data, "raw_version")) != expected_raw_version:
        raise GlobalConflictError("global raw version mismatch")
    if str(_scalar(data, "discovery_version")) != GLOBAL_DISCOVERY_VERSION:
        raise GlobalConflictError("global discovery version mismatch")
    if expected_source_commit is not None and task.get("source_commit") != expected_source_commit:
        raise GlobalConflictError("global raw source commit mismatch")
    if task.get("registry_sha256") != registry["registry_sha256"]:
        raise GlobalConflictError("global raw registry mismatch")
    if task.get("discovery_config_sha256") != config["discovery_config_sha256"]:
        raise GlobalConflictError("global raw config mismatch")
    scalar_expected = {
        "source_commit": task["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "engine": "numba",
    }
    for field, expected in scalar_expected.items():
        if str(_scalar(data, field)) != str(expected):
            raise GlobalConflictError(f"global raw identity mismatch: {field}")
    for field in ("core_seconds", "wall_seconds"):
        value = float(_scalar(data, field))
        if not np.isfinite(value) or value < 0.0:
            raise GlobalConflictError(f"global raw timing is invalid: {field}")
    return task


def _rebuild_task_context(registry, config, task):
    registry_path = Path(config["config_path"]).parents[1] / "registry/registry.json"
    # Tests may use a copied config with the canonical registry elsewhere; use
    # the caller's registry code files through its private path when attached.
    if "_registry_path" in registry:
        registry_path = Path(registry["_registry_path"])
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _cell_disorder(registry, code, model, task["cell"])
    sampler = resolved_sampler_config(
        config, task["method_id"], task["cell"]["p"], task["resource_tier"],
    )
    characters = frozen_character_set(
        model.k, character_seed(registry["registry_sha256"], code["code_id"]),
        config["characters"]["num_nonbasis"],
    )
    return code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters


def _registry_with_path(registry, registry_path):
    value = dict(registry)
    value["_registry_path"] = str(Path(registry_path).resolve())
    return value


def _validate_stored_context(data, task, code, model, frame, uniform_seed,
                             sampler, characters):
    scalar_expected = {
        "cell_json": canonical_json(task["cell"]),
        "config_json": canonical_json(sampler.as_dict()),
        "uniform_seed": uniform_seed,
        "model_fingerprint": model.fingerprint(),
        "section_fingerprint": code["section_fingerprint"],
        "logical_frame_fingerprint": code["logical_frame_fingerprint"],
        "character_version": CHARACTER_SET_VERSION,
        "character_sha256": characters.character_sha256,
        "num_qubits": model.num_qubits,
        "k": model.k,
    }
    for field, expected in scalar_expected.items():
        if str(_scalar(data, field)) != str(expected):
            raise GlobalConflictError(f"global raw reconstructed identity mismatch: {field}")
    _require_equal("character_masks", data["character_masks"], characters.masks)


def _compare_result(data, result, defect=False):
    fields = (
        "initial_state_packed", "burn_state_packed", "final_state_packed",
        "measurement_states_packed", "burn_labels", "measurement_labels",
        "measurement_weights", "measurement_block", "burn_counters",
        "measurement_counters", "burn_basis_seen", "initial_label", "burn_label",
        "final_label",
    )
    for field in fields:
        _require_equal(field, data[field], result[field])
    if defect:
        for field in ("measurement_defect_counts", "fixed_clock_d0_mask"):
            _require_equal(field, data[field], result[field])
        if float(_scalar(data, "boundary_occupancy")) != float(result["boundary_occupancy"]):
            raise GlobalConflictError("defect boundary occupancy replay mismatch")
    else:
        _require_equal(
            "measurement_residual_weights", data["measurement_residual_weights"],
            result["measurement_residual_weights"],
        )
    if str(_scalar(data, "trajectory_digest")) != canonical_global_trajectory_digest(result):
        raise GlobalConflictError("global trajectory digest replay mismatch")


def validate_hard_raw(path, registry, config, expected_source_commit=None):
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open hard-coset raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, HARD_RAW_FIELDS, registry, config, expected_source_commit,
            HARD_COSET_RAW_VERSION,
        )
        expected = global_task_identity(
            registry, config, task["source_commit"], task["stage"], task["method_id"],
            task["resource_tier"], task["cell"], task["init_family"],
            task["trajectory_index"], bias_binding=None,
        )
        if task != expected or task["method_id"] not in HARD_METHODS:
            raise GlobalConflictError("hard-coset raw embeds a noncanonical task")
        code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters = (
            _rebuild_task_context(registry, config, task)
        )
        _validate_stored_context(
            data, task, code, model, frame, uniform_seed, sampler, characters,
        )
        seed_identity = GlobalSeedIdentity(**task["seed_identity"])
        if str(_scalar(data, "seed_identity_json")) != canonical_json(seed_identity.as_dict()):
            raise GlobalConflictError("hard-coset seed identity mismatch")
        initial = _initial_state(model, epsilon, syndrome, seed_identity)
        catalog = build_logical_proposal_catalog(
            model, frame, max_multiple=config["catalog"]["max_multiple"],
            max_count=config["catalog"]["max_count"],
        )
        if str(_scalar(data, "catalog_sha256")) != catalog.catalog_sha256:
            raise GlobalConflictError("hard-coset catalog SHA mismatch")
        joint = (
            build_joint_blocks(model, frame, catalog, sampler.joint_block_size)
            if sampler.joint_block_size else None
        )
        expected_joint = "none" if joint is None else joint.joint_sha256
        if str(_scalar(data, "joint_sha256")) != expected_joint:
            raise GlobalConflictError("hard-coset joint-block SHA mismatch")
        replay = run_hardcoset_trajectory(
            model, frame, syndrome, sampler, seed_identity, initial,
            engine="numba", catalog=catalog, joint=joint,
        )
        _compare_result(data, replay, defect=False)
        labels = data["measurement_labels"].copy()
        weights = data["measurement_weights"].copy()
        burn_labels = data["burn_labels"].copy()
        core_seconds = float(_scalar(data, "core_seconds"))
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "task": task, "task_fingerprint": sha256_json(task),
        "cell": task["cell"], "method_id": task["method_id"],
        "resource_tier": task["resource_tier"], "init_family": task["init_family"],
        "trajectory_index": task["trajectory_index"], "labels": labels,
        "weights": weights, "valid_mask": np.ones(labels.size, dtype=bool),
        "burn_labels": burn_labels, "initial_label": int(replay["initial_label"]),
        "num_qubits": model.num_qubits, "k": model.k,
        "character_masks": characters.masks,
        "core_seconds": core_seconds,
    }


def validate_bias_raw(path, registry, config, expected_source_commit=None):
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open defect bias raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, BIAS_RAW_FIELDS, registry, config, expected_source_commit,
            DEFECT_BIAS_RAW_VERSION,
        )
        expected = bias_task_identity(
            registry, config, task["source_commit"], task["stage"], task["method_id"],
            task["resource_tier"], task["cell"],
        )
        if task != expected:
            raise GlobalConflictError("defect bias raw embeds a noncanonical task")
        code, model, frame, uniform_seed, _, syndrome, sampler, _ = (
            _rebuild_task_context(registry, config, task)
        )
        scalar_expected = {
            "cell_json": canonical_json(task["cell"]),
            "config_json": canonical_json(sampler.as_dict()),
            "uniform_seed": uniform_seed,
            "model_fingerprint": model.fingerprint(),
            "num_qubits": model.num_qubits,
            "tuning_seed_identities_json": canonical_json(task["tuning_seed_identities"]),
        }
        for field, value in scalar_expected.items():
            if str(_scalar(data, field)) != str(value):
                raise GlobalConflictError(f"defect bias identity mismatch: {field}")
        identities = [GlobalSeedIdentity(**value) for value in task["tuning_seed_identities"]]
        replay = tune_defect_bias(model, syndrome, sampler, identities, engine="numba")
        for field in (
                "bias", "bias_trace", "tuning_histogram", "tuning_final_states_packed",
                "tuning_final_residuals", "tuning_final_defects", "gammas"):
            _require_equal(field, data[field], replay[field])
        if str(_scalar(data, "bias_sha256")) != replay["bias_sha256"]:
            raise GlobalConflictError("defect bias SHA replay mismatch")
        bias = data["bias"].copy()
        bias_sha = str(_scalar(data, "bias_sha256"))
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "task": task, "task_fingerprint": sha256_json(task),
        "bias": bias, "bias_sha256": bias_sha,
    }


def validate_defect_raw(path, registry, config, expected_source_commit=None,
                        bias_path=None, *, _validated_bias_record=None):
    path = Path(path)
    if bias_path is None:
        raise ValueError("defect raw validation requires its bound bias raw path")
    if _validated_bias_record is None:
        bias_record = validate_bias_raw(
            bias_path, registry, config, expected_source_commit,
        )
    else:
        bias_record = _validated_bias_record
        resolved_bias = str(Path(bias_path).resolve(strict=True))
        if (bias_record.get("path") != resolved_bias
                or bias_record.get("sha256") != sha256_file(bias_path)
                or bias_record.get("task", {}).get("registry_sha256")
                != registry["registry_sha256"]
                or bias_record.get("task", {}).get("discovery_config_sha256")
                != config["discovery_config_sha256"]
                or (expected_source_commit is not None
                    and bias_record.get("task", {}).get("source_commit")
                    != expected_source_commit)):
            raise GlobalConflictError("cached defect bias validation is stale or mismatched")
    binding = {
        "bias_task_fingerprint": bias_record["task_fingerprint"],
        "bias_raw_sha256": bias_record["sha256"],
        "bias_sha256": bias_record["bias_sha256"],
    }
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open defect raw {path}: {exc}") from exc
    with context as data:
        task = _validate_common_raw(
            data, DEFECT_RAW_FIELDS, registry, config, expected_source_commit,
            DEFECT_TRACE_RAW_VERSION,
        )
        expected = global_task_identity(
            registry, config, task["source_commit"], task["stage"], task["method_id"],
            task["resource_tier"], task["cell"], task["init_family"],
            task["trajectory_index"], bias_binding=binding,
        )
        if task != expected or task["method_id"] not in DEFECT_METHODS:
            raise GlobalConflictError("defect raw embeds a noncanonical task")
        code, model, frame, uniform_seed, epsilon, syndrome, sampler, characters = (
            _rebuild_task_context(registry, config, task)
        )
        _validate_stored_context(
            data, task, code, model, frame, uniform_seed, sampler, characters,
        )
        for field, value in binding.items():
            if str(_scalar(data, field)) != str(value):
                raise GlobalConflictError(f"defect bias binding mismatch: {field}")
        _require_equal("bias", data["bias"], bias_record["bias"])
        seed_identity = GlobalSeedIdentity(**task["seed_identity"])
        if str(_scalar(data, "seed_identity_json")) != canonical_json(seed_identity.as_dict()):
            raise GlobalConflictError("defect seed identity mismatch")
        initial = _initial_state(model, epsilon, syndrome, seed_identity)
        replay = run_defect_trace_trajectory(
            model, frame, syndrome, sampler, seed_identity, initial,
            bias_record["bias"], bias_record["bias_sha256"], engine="numba",
        )
        _compare_result(data, replay, defect=True)
        labels = data["measurement_labels"].copy()
        weights = data["measurement_weights"].copy()
        valid_mask = data["fixed_clock_d0_mask"].copy()
        defects = data["measurement_defect_counts"].copy()
        counters = data["measurement_counters"].copy()
        burn_labels = data["burn_labels"].copy()
        boundary = float(_scalar(data, "boundary_occupancy"))
        core_seconds = float(_scalar(data, "core_seconds"))
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "task": task, "task_fingerprint": sha256_json(task),
        "cell": task["cell"], "method_id": task["method_id"],
        "resource_tier": task["resource_tier"], "init_family": task["init_family"],
        "trajectory_index": task["trajectory_index"], "labels": labels,
        "weights": weights, "valid_mask": valid_mask, "defects": defects,
        "measurement_counters": counters, "boundary_occupancy": boundary,
        "burn_labels": burn_labels, "initial_label": int(replay["initial_label"]),
        "num_qubits": model.num_qubits, "k": model.k,
        "character_masks": characters.masks,
        "core_seconds": core_seconds,
    }


def validate_ti_anchor_raw(path, registry, config, registry_path,
                           expected_source_commit=None):
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise GlobalConflictError(f"cannot open TI anchor raw {path}: {exc}") from exc
    with context as data:
        if set(data.files) != TI_ANCHOR_RAW_FIELDS:
            missing = sorted(TI_ANCHOR_RAW_FIELDS - set(data.files))
            extra = sorted(set(data.files) - TI_ANCHOR_RAW_FIELDS)
            raise GlobalConflictError(
                f"TI anchor raw schema mismatch; missing={missing}, extra={extra}"
            )
        task = _read_task(data)
        source_commit = task.get("source_commit")
        if (str(_scalar(data, "raw_version")) != TI_ANCHOR_RAW_VERSION
                or str(_scalar(data, "discovery_version"))
                != GLOBAL_DISCOVERY_VERSION
                or (expected_source_commit is not None
                    and source_commit != expected_source_commit)):
            raise GlobalConflictError("TI anchor raw version/source mismatch")
        expected = ti_anchor_task_identity(
            registry, config, source_commit, task.get("cell"),
        )
        if task != expected:
            raise GlobalConflictError("TI anchor raw embeds a noncanonical task")
        code, model, frame, uniform_seed, replay = _execute_ti_anchor(
            registry, config, registry_path, task,
        )
        scalar_expected = {
            "cell_json": canonical_json(task["cell"]),
            "ti_config_json": canonical_json(config["ti_anchor"]),
            "source_commit": source_commit,
            "registry_sha256": registry["registry_sha256"],
            "discovery_config_sha256": config["discovery_config_sha256"],
            "uniform_seed": uniform_seed,
            "engine_seed": task["engine_seed"],
            "engine": "exp101_certified_full_sector_ti",
            "model_fingerprint": model.fingerprint(),
            "section_fingerprint": code["section_fingerprint"],
            "logical_frame_fingerprint": code["logical_frame_fingerprint"],
            "num_qubits": model.num_qubits,
            "k": model.k,
        }
        for field, expected_value in scalar_expected.items():
            if str(_scalar(data, field)) != str(expected_value):
                raise GlobalConflictError(f"TI anchor identity mismatch: {field}")
        for field in ("core_seconds", "wall_seconds"):
            value = float(_scalar(data, field))
            if not np.isfinite(value) or value < 0.0:
                raise GlobalConflictError(f"TI anchor timing is invalid: {field}")
        stored = {
            name: data[name].copy()
            for name in _ti_result_arrays(replay)
        }
        replay_arrays = _ti_result_arrays(replay)
        for field, expected_value in replay_arrays.items():
            actual = stored[field]
            if np.asarray(actual).dtype.kind == "f":
                if not np.allclose(
                        actual, expected_value, rtol=0.0, atol=1e-12,
                        equal_nan=False):
                    raise GlobalConflictError(f"TI anchor replay mismatch: {field}")
            elif not np.array_equal(actual, expected_value):
                raise GlobalConflictError(f"TI anchor replay mismatch: {field}")
        stored_digest = str(_scalar(data, "trajectory_digest"))
        if stored_digest != _ti_result_digest(stored):
            raise GlobalConflictError("TI anchor stored digest mismatch")
        weights = np.asarray(stored["weights_absolute"], dtype=np.float64)
        if (not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
                or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)):
            raise GlobalConflictError("TI anchor sector weights are invalid")
        record = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "task": task,
            "task_fingerprint": sha256_json(task),
            "cell": task["cell"],
            "num_qubits": model.num_qubits,
            "k": model.k,
            "q_top": float(_scalar(data, "q_top")),
            "q_top_stderr": float(_scalar(data, "q_top_stderr")),
            "characters_absolute": stored["characters_absolute"],
            "weights_absolute": weights,
            "grid_tv": float(_scalar(data, "grid_tv")),
            "grid_q_top_abs_diff": float(_scalar(data, "grid_q_top_abs_diff")),
            "flags": str(_scalar(data, "flags")),
            "valid_for_aggregation": bool(_scalar(data, "valid_for_aggregation")),
            "core_seconds": float(_scalar(data, "core_seconds")),
        }
    return record


def _fft_bulk_ess(chains):
    """Deterministic multi-chain initial-positive-sequence ESS."""
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2 or chains.shape[1] < 4:
        return 0.0
    centered = chains - chains.mean(axis=1, keepdims=True)
    variance = float(np.mean(centered * centered))
    if variance == 0.0:
        return float(chains.size)
    length = chains.shape[1]
    fft_length = 1 << (2 * length - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_length, axis=1)
    autocov = np.fft.irfft(
        transformed * transformed.conjugate(), n=fft_length, axis=1,
    )[:, :length]
    autocov /= np.arange(length, 0, -1, dtype=np.float64)[None, :]
    rho = autocov.mean(axis=0) / variance
    rho_sum = 0.0
    previous = float("inf")
    for lag in range(1, length - 1, 2):
        pair = min(float(rho[lag] + rho[lag + 1]), previous)
        if pair <= 0.0:
            break
        rho_sum += pair
        previous = pair
    return float(min(chains.size, chains.size / max(1.0 + 2.0 * rho_sum, 1.0)))


def _diagnostic_masks(character_set, count):
    basis = character_set.masks[character_set.basis_positions]
    basis_set = {int(value) for value in basis}
    nonbasis = [value for value in character_set.masks if int(value) not in basis_set]
    return np.asarray([*basis, *nonbasis[:int(count)]], dtype=np.uint64)


def _constant_transport_ok(records, mask, common_value):
    opposite = []
    for record in records:
        initial = int(character_values([record["initial_label"]], [mask])[0, 0])
        if initial != common_value:
            opposite.append(record)
    if not opposite:
        return True
    return all(
        np.any(character_values(record["burn_labels"], [mask])[:, 0] == common_value)
        for record in opposite
    )


def _family_summary(records, config):
    records = sorted(records, key=lambda value: value["trajectory_index"])
    if len(records) != TRAJECTORIES_PER_FAMILY:
        raise GlobalConflictError("global family does not contain 16 trajectories")
    if [row["trajectory_index"] for row in records] != list(range(TRAJECTORIES_PER_FAMILY)):
        raise GlobalConflictError("global family trajectory indices are incomplete")
    first = records[0]
    if any(not np.array_equal(row["character_masks"], first["character_masks"])
           for row in records[1:]):
        raise GlobalConflictError("global family character masks differ")
    k = int(first["k"])
    masks = np.asarray(first["character_masks"], dtype=np.uint64)
    basis_positions = np.asarray([
        int(np.flatnonzero(masks == (np.uint64(1) << np.uint64(bit)))[0])
        for bit in range(k)
    ], dtype=np.int32)
    tier = "full" if k <= 10 else "sampled"
    character_set = CharacterSet(
        masks=masks, basis_positions=basis_positions, tier=tier, k=k,
        random_seed=None, character_sha256="validated-in-raw",
    )
    traces = [row["labels"] for row in records]
    valid_masks = [row["valid_mask"] for row in records]
    valid_counts = np.asarray([int(value.sum()) for value in valid_masks], dtype=np.int64)
    if np.any(valid_counts == 0):
        failures = ["no_valid_observations"]
        worm_metrics = None
        if first["method_id"] in DEFECT_METHODS:
            failures.append("worm_d0_count")
            worm_metrics = {
                "d0_counts": valid_counts.tolist(),
                "excursions": [
                    min(int(row["measurement_counters"][2]),
                        int(row["measurement_counters"][3]))
                    for row in records
                ],
                "per_chain_d0_ess": None,
                "median_d0_ess": None,
                "aggregate_d0_ess": None,
                "boundary_occupancy": [float(row["boundary_occupancy"]) for row in records],
            }
        return {
            "init_family": first["init_family"],
            "q_top": None,
            "q_top_total_se": None,
            "q_top_trajectory_se": None,
            "q_top_character_se": None,
            "label_collision_mass_diagnostic": None,
            "label_collision_q_top_diagnostic": None,
            "normalized_mean_weight": None,
            "normalized_mean_weight_se": None,
            "max_rhat": None,
            "min_nondegenerate_bulk_ess": None,
            "constant_failures": [],
            "worm": worm_metrics,
            "core_seconds": float(sum(row["core_seconds"] for row in records)),
            "valid": False,
            "failures": failures,
            "_means": None,
            "_character_set": character_set,
            "_trajectory_weight_means": None,
        }
    means, counts = character_means(traces, masks, valid_masks)
    qtop = character_qtop_estimate(character_set, means)
    collision = label_collision_diagnostic(traces, k, valid_masks)
    trajectory_weight_means = np.asarray([
        float(np.mean(row["weights"][row["valid_mask"]])) for row in records
    ])
    weight_mean, weight_se = trajectory_mean_and_se(
        trajectory_weight_means, first["num_qubits"],
    )

    diagnostic_masks = _diagnostic_masks(
        character_set, config["gates"]["diagnostic_nonbasis_characters"],
    )
    minimum = min(int(row["valid_mask"].sum()) for row in records)
    diagnostic_rhat = []
    diagnostic_ess = []
    diagnostic_nondegenerate = []
    constant_failures = []
    energy_chains = np.asarray([
        row["weights"][row["valid_mask"]][:minimum] for row in records
    ], dtype=np.float64)
    diagnostic_rhat.append(split_rhat(energy_chains))
    diagnostic_ess.append(_fft_bulk_ess(energy_chains))
    diagnostic_nondegenerate.append(np.unique(energy_chains).size > 1)
    for mask in diagnostic_masks:
        chains = np.asarray([
            character_values(row["labels"][row["valid_mask"]][:minimum], [mask])[:, 0]
            for row in records
        ], dtype=np.float64)
        unique = np.unique(chains)
        if unique.size == 1:
            common = int(unique[0])
            if not _constant_transport_ok(records, mask, common):
                constant_failures.append(f"0x{int(mask):016x}")
            diagnostic_rhat.append(1.0)
            diagnostic_ess.append(float(chains.size))
            diagnostic_nondegenerate.append(False)
        else:
            diagnostic_rhat.append(split_rhat(chains))
            diagnostic_ess.append(_fft_bulk_ess(chains))
            diagnostic_nondegenerate.append(True)

    failures = []
    gates = config["gates"]
    if not np.isfinite(qtop["q_top_total_se"]) or qtop["q_top_total_se"] > gates["max_q_top_se"]:
        failures.append("q_top_se")
    if max(diagnostic_rhat) > gates["max_rhat"]:
        failures.append("rhat")
    nondegenerate_ess = [
        value for value, nondegenerate in zip(
            diagnostic_ess, diagnostic_nondegenerate,
        ) if nondegenerate
    ]
    if nondegenerate_ess and min(nondegenerate_ess) < gates["min_bulk_ess"]:
        failures.append("bulk_ess")
    if constant_failures:
        failures.append("constant_common_freeze")

    worm = first["method_id"] in DEFECT_METHODS
    worm_metrics = None
    if worm:
        per_chain_ess = np.asarray([
            _fft_bulk_ess(row["weights"][row["valid_mask"]][None, :])
            for row in records
        ])
        excursions = np.asarray([
            min(
                int(row["measurement_counters"][2]),
                int(row["measurement_counters"][3]),
            ) for row in records
        ])
        boundary = np.asarray([row["boundary_occupancy"] for row in records])
        if np.any(counts < gates["worm_min_d0_per_trajectory"]):
            failures.append("worm_d0_count")
        if np.any(excursions < gates["worm_min_excursions_per_trajectory"]):
            failures.append("worm_excursions")
        if float(np.median(per_chain_ess)) < gates["worm_min_median_d0_ess"]:
            failures.append("worm_median_d0_ess")
        if float(per_chain_ess.sum()) < gates["worm_min_family_d0_ess"]:
            failures.append("worm_family_d0_ess")
        if np.any(boundary > gates["worm_max_boundary_occupancy"]):
            failures.append("worm_boundary")
        worm_metrics = {
            "d0_counts": counts.tolist(),
            "excursions": excursions.tolist(),
            "per_chain_d0_ess": per_chain_ess.tolist(),
            "median_d0_ess": float(np.median(per_chain_ess)),
            "aggregate_d0_ess": float(per_chain_ess.sum()),
            "boundary_occupancy": boundary.tolist(),
        }
    return {
        "init_family": first["init_family"],
        "q_top": qtop["q_top"],
        "q_top_total_se": qtop["q_top_total_se"],
        "q_top_trajectory_se": qtop["q_top_trajectory_se"],
        "q_top_character_se": qtop["q_top_character_se"],
        "label_collision_mass_diagnostic": collision["collision_mass"],
        "label_collision_q_top_diagnostic": collision["q_top"],
        "normalized_mean_weight": weight_mean,
        "normalized_mean_weight_se": weight_se,
        "max_rhat": float(max(diagnostic_rhat)),
        "min_nondegenerate_bulk_ess": (
            None if not nondegenerate_ess else float(min(nondegenerate_ess))
        ),
        "constant_failures": constant_failures,
        "worm": worm_metrics,
        "core_seconds": float(sum(row["core_seconds"] for row in records)),
        "valid": not failures,
        "failures": sorted(set(failures)),
        "_means": means,
        "_character_set": character_set,
        "_trajectory_weight_means": trajectory_weight_means / first["num_qubits"],
    }


def _delta_gate(left, right, config):
    if (left.get("q_top") is None or right.get("q_top") is None
            or left.get("q_top_total_se") is None
            or right.get("q_top_total_se") is None):
        return {
            "delta_q_top": None,
            "se_delta_q_top": None,
            "absolute_pass": False,
            "sigma_pass": False,
        }
    delta = abs(float(left["q_top"]) - float(right["q_top"]))
    se = math.sqrt(left["q_top_total_se"] ** 2 + right["q_top_total_se"] ** 2)
    gates = config["gates"]
    return {
        "delta_q_top": delta,
        "se_delta_q_top": se,
        "absolute_pass": delta <= gates["max_abs_delta_q_top"],
        "sigma_pass": delta <= gates["delta_sigma_multiplier"] * se + gates["delta_sigma_slack"],
    }


def _cell_method_summary(records, config):
    by_family = defaultdict(list)
    for record in records:
        by_family[record["init_family"]].append(record)
    if set(by_family) != set(INIT_FAMILIES):
        raise GlobalConflictError("cell/method is missing an initialization family")
    families = {name: _family_summary(by_family[name], config) for name in INIT_FAMILIES}
    character_set = families["P"]["_character_set"]
    if any(value["_means"] is None for value in families.values()):
        public_families = {
            name: {key: value for key, value in summary.items() if not key.startswith("_")}
            for name, summary in families.items()
        }
        return {
            "cell": records[0]["cell"],
            "method_id": records[0]["method_id"],
            "resource_tier": records[0]["resource_tier"],
            "num_qubits": int(records[0]["num_qubits"]),
            "families": public_families,
            "q_top": None,
            "q_top_total_se": None,
            "label_collision_mass_diagnostic": None,
            "label_collision_q_top_diagnostic": None,
            "initialization_delta": _delta_gate(families["P"], families["U"], config),
            "d2": None,
            "normalized_weight_delta": None,
            "normalized_weight_delta_se": None,
            "ti_anchor_payload": None,
            "core_seconds": sum(value["core_seconds"] for value in families.values()),
            "valid": False,
            "failures": ["family_gate", "no_valid_observations"],
            "_combined_means": None,
            "_character_set": character_set,
            "_weight_samples": None,
        }
    d2 = character_d2_estimate(
        character_set, families["P"]["_means"], families["U"]["_means"],
    )
    delta = _delta_gate(families["P"], families["U"], config)
    weight_delta = abs(
        families["P"]["normalized_mean_weight"]
        - families["U"]["normalized_mean_weight"]
    )
    weight_se = math.sqrt(
        families["P"]["normalized_mean_weight_se"] ** 2
        + families["U"]["normalized_mean_weight_se"] ** 2
    )
    n = records[0]["num_qubits"]
    gates = config["gates"]
    failures = []
    if not all(value["valid"] for value in families.values()):
        failures.append("family_gate")
    if not delta["absolute_pass"] or not delta["sigma_pass"]:
        failures.append("initialization_q_top")
    if max(0.0, d2["d2_norm"]) + 3.0 * d2["d2_total_se"] > gates["max_d2_upper"]:
        failures.append("initialization_d2")
    if (weight_delta > gates["max_normalized_weight_delta"]
            or weight_delta > 3.0 * weight_se + 1.0 / n):
        failures.append("initialization_weight")
    combined_means = np.vstack((families["P"]["_means"], families["U"]["_means"]))
    combined_q = character_qtop_estimate(character_set, combined_means)
    combined_collision = label_collision_diagnostic(
        [record["labels"] for record in records],
        int(records[0]["k"]),
        [record["valid_mask"] for record in records],
    )
    public_families = {
        name: {key: value for key, value in summary.items() if not key.startswith("_")}
        for name, summary in families.items()
    }
    ti_anchor_payload = None
    if character_set.tier == "full":
        masks_list = [int(value) for value in character_set.masks]
        means_list = combined_means.tolist()
        ti_anchor_payload = {
            "character_masks": masks_list,
            "trajectory_character_means": means_list,
            "payload_sha256": sha256_json({
                "character_masks": masks_list,
                "trajectory_character_means": means_list,
            }),
        }
    return {
        "cell": records[0]["cell"],
        "method_id": records[0]["method_id"],
        "resource_tier": records[0]["resource_tier"],
        "num_qubits": int(n),
        "families": public_families,
        "q_top": combined_q["q_top"],
        "q_top_total_se": combined_q["q_top_total_se"],
        "label_collision_mass_diagnostic": combined_collision["collision_mass"],
        "label_collision_q_top_diagnostic": combined_collision["q_top"],
        "initialization_delta": delta,
        "d2": {key: value for key, value in d2.items() if not key.startswith("per_") and not key.startswith("delete_")},
        "normalized_weight_delta": weight_delta,
        "normalized_weight_delta_se": weight_se,
        "ti_anchor_payload": ti_anchor_payload,
        "core_seconds": sum(value["core_seconds"] for value in families.values()),
        "valid": not failures,
        "failures": sorted(set(failures)),
        "_combined_means": combined_means,
        "_character_set": character_set,
        "_weight_samples": np.concatenate([
            families["P"]["_trajectory_weight_means"],
            families["U"]["_trajectory_weight_means"],
        ]),
    }


def compare_cell_summaries(left, right, config):
    if left["cell"] != right["cell"]:
        raise ValueError("cannot compare summaries from different cells")
    delta = _delta_gate(left, right, config)
    if (left.get("_combined_means") is None
            or right.get("_combined_means") is None):
        return {
            "left": {"method_id": left["method_id"], "resource_tier": left["resource_tier"]},
            "right": {"method_id": right["method_id"], "resource_tier": right["resource_tier"]},
            "q_top": delta,
            "d2_norm": None,
            "d2_total_se": None,
            "d2_pass": False,
            "normalized_weight_delta": None,
            "normalized_weight_delta_se": None,
            "weight_pass": False,
            "valid": False,
        }
    d2 = character_d2_estimate(
        left["_character_set"], left["_combined_means"], right["_combined_means"],
    )
    left_weight, left_se = trajectory_mean_and_se(left["_weight_samples"])
    right_weight, right_se = trajectory_mean_and_se(right["_weight_samples"])
    weight_delta = abs(left_weight - right_weight)
    weight_se = math.sqrt(left_se**2 + right_se**2)
    gates = config["gates"]
    if left["num_qubits"] != right["num_qubits"]:
        raise GlobalConflictError("comparison code dimensions differ")
    weight_pass = (
        weight_delta <= gates["max_normalized_weight_delta"]
        and weight_delta <= 3.0 * weight_se + 1.0 / left["num_qubits"]
    )
    d2_pass = max(0.0, d2["d2_norm"]) + 3.0 * d2["d2_total_se"] <= gates["max_d2_upper"]
    return {
        "left": {"method_id": left["method_id"], "resource_tier": left["resource_tier"]},
        "right": {"method_id": right["method_id"], "resource_tier": right["resource_tier"]},
        "q_top": delta,
        "d2_norm": d2["d2_norm"], "d2_total_se": d2["d2_total_se"],
        "d2_pass": d2_pass,
        "normalized_weight_delta": weight_delta,
        "normalized_weight_delta_se": weight_se,
        "weight_pass": weight_pass,
        "valid": delta["absolute_pass"] and delta["sigma_pass"] and d2_pass and weight_pass,
    }


def _task_cost(task):
    m = int(task["cell"]["code_id"][1:3])
    if task.get("raw_version") == TI_ANCHOR_RAW_VERSION:
        ti = task["ti_config"]
        work = (
            int(ti["num_kp_grid_points"])
            * (int(ti["num_burn_in_sweeps"])
               + int(ti["num_measurements"]))
        )
        return float((1 << (m * m)) * m * m * work)
    sampler = task["sampler_config"]
    work = int(sampler["burn_sweeps"]) + int(sampler["measurement_sweeps"])
    if task["method_id"].startswith("RC8-QC"):
        multiplier = 1.0 + float(sampler["cluster_repeats"])
    elif task["method_id"].startswith("RC8-J"):
        multiplier = 1.0 + (1 << int(sampler["joint_block_size"])) / 256.0
    else:
        multiplier = 1.0
    return float(m * m * work * multiplier)


def fixed_global_ownership(tasks, nodes, source_commit, control_sha256, stage):
    nodes = list(nodes)
    if len(nodes) < 2 or len(nodes) != len(set(nodes)) or not set(nodes) <= set(NODE_CAPACITY):
        raise ValueError("global ownership needs at least two distinct known nodes")
    loads = {node: 0.0 for node in nodes}
    owners = {}
    for task in sorted(tasks, key=lambda value: (-_task_cost(value), sha256_json(value))):
        owner = min(nodes, key=lambda node: (loads[node] / NODE_CAPACITY[node], node))
        fingerprint = sha256_json(task)
        owners[fingerprint] = owner
        loads[owner] += _task_cost(task)
    identity = {
        "ownership_version": "exp102.q0_global.ownership.v1",
        "source_commit": source_commit,
        "control_sha256": control_sha256,
        "stage": stage,
        "nodes": nodes,
        "task_owner": owners,
    }
    return {
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "weighted_load": loads,
        "capacity": {node: NODE_CAPACITY[node] for node in nodes},
    }


def build_bias_manifest(registry_path, config_path, source_commit, stage,
                        method_tiers, output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    method_tiers = [(str(method), str(tier)) for method, tier in method_tiers]
    if not method_tiers or any(method not in DEFECT_METHODS for method, _ in method_tiers):
        raise ValueError("bias manifest requires defect methods")
    tasks = []
    for method, tier in method_tiers:
        for cell in _stage_cells(config, stage):
            task = bias_task_identity(
                registry, config, source_commit, stage, method, tier, cell,
            )
            fingerprint = sha256_json(task)
            tasks.append({
                "task": task,
                "task_fingerprint": fingerprint,
                "output_relpath": f"bias/{fingerprint}.npz",
            })
    fingerprints = [value["task_fingerprint"] for value in tasks]
    if len(fingerprints) != len(set(fingerprints)):
        raise GlobalConflictError("bias manifest has duplicate tasks")
    manifest = {
        "manifest_version": GLOBAL_TASKS_VERSION,
        "kind": "defect_bias",
        "stage": stage,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "method_tiers": [list(value) for value in method_tiers],
        "tasks": tasks,
    }
    atomic_json(output_path, manifest)
    return manifest


def _canonical_ti_anchor_manifest(registry, config, source_commit):
    entries = []
    for cell in _stage_cells(config, "ti_anchors"):
        task = ti_anchor_task_identity(registry, config, source_commit, cell)
        fingerprint = sha256_json(task)
        entries.append({
            "task": task,
            "task_fingerprint": fingerprint,
            "output_relpath": f"ti_anchors/{fingerprint}.npz",
            "bias_relpath": None,
        })
    return {
        "manifest_version": GLOBAL_TASKS_VERSION,
        "kind": "ti_anchor",
        "stage": "ti_anchors",
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "method_tiers": [[config["ti_anchor"]["method_id"], "TI49"]],
        "tasks": entries,
    }


def build_ti_anchor_manifest(registry_path, config_path, source_commit,
                             output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    manifest = _canonical_ti_anchor_manifest(
        registry, config, source_commit,
    )
    atomic_json(output_path, manifest)
    return manifest


def _validate_bias_manifest_structure(bias_manifest, registry, config,
                                      source_commit):
    if (bias_manifest.get("manifest_version") != GLOBAL_TASKS_VERSION
            or bias_manifest.get("kind") != "defect_bias"
            or bias_manifest.get("source_commit") != source_commit
            or bias_manifest.get("registry_sha256") != registry["registry_sha256"]
            or bias_manifest.get("discovery_config_sha256")
            != config["discovery_config_sha256"]):
        raise GlobalConflictError("defect bias manifest identity mismatch")
    method_tiers = [tuple(value) for value in bias_manifest.get("method_tiers", [])]
    if (not method_tiers or len(method_tiers) != len(set(method_tiers))
            or any(method not in DEFECT_METHODS for method, _ in method_tiers)):
        raise GlobalConflictError("defect bias manifest method list is noncanonical")
    expected_entries = []
    for method, tier in method_tiers:
        for cell in _stage_cells(config, bias_manifest.get("stage")):
            task = bias_task_identity(
                registry, config, source_commit, bias_manifest["stage"], method,
                tier, cell,
            )
            fingerprint = sha256_json(task)
            expected_entries.append({
                "task": task,
                "task_fingerprint": fingerprint,
                "output_relpath": f"bias/{fingerprint}.npz",
            })
    if bias_manifest.get("tasks") != expected_entries:
        raise GlobalConflictError("defect bias manifest task set/order is noncanonical")
    return method_tiers


def _bias_index_from_manifest(raw_root, bias_manifest, registry, config,
                              source_commit):
    _validate_bias_manifest_structure(
        bias_manifest, registry, config, source_commit,
    )
    index = {}
    for entry in bias_manifest["tasks"]:
        path = Path(raw_root) / entry["output_relpath"]
        record = validate_bias_raw(path, registry, config, source_commit)
        task = entry["task"]
        if record["task_fingerprint"] != entry["task_fingerprint"]:
            raise GlobalConflictError("bias manifest fingerprint/raw mismatch")
        key = (
            task["stage"], task["method_id"], task["resource_tier"],
            sha256_json(task["cell"]),
        )
        index[key] = {
            "binding": {
                "bias_task_fingerprint": record["task_fingerprint"],
                "bias_raw_sha256": record["sha256"],
                "bias_sha256": record["bias_sha256"],
            },
            "relpath": entry["output_relpath"],
        }
    return index


def validate_global_control_manifest(manifest, registry, config):
    """Reject a noncanonical bias, measurement, or TI stage control."""
    if not isinstance(manifest, dict):
        raise GlobalConflictError("global control is not a JSON object")
    source_commit = manifest.get("source_commit")
    _validate_source_commit(source_commit)
    kind = manifest.get("kind")
    if kind == "defect_bias":
        _validate_bias_manifest_structure(
            manifest, registry, config, source_commit,
        )
    elif kind == "measurement":
        _validate_measurement_manifest_structure(manifest, registry, config)
    elif kind == "ti_anchor":
        if manifest != _canonical_ti_anchor_manifest(
                registry, config, source_commit):
            raise GlobalConflictError("TI anchor control is noncanonical")
    else:
        raise GlobalConflictError("global control kind is unknown")
    return True


def build_measurement_manifest(registry_path, config_path, source_commit, stage,
                               method_tiers, output_path, *, bias_manifest_path=None,
                               bias_raw_root=None):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    method_tiers = [(str(method), str(tier)) for method, tier in method_tiers]
    if not method_tiers or len(method_tiers) != len(set(method_tiers)):
        raise ValueError("measurement manifest method/tier list is empty or duplicated")
    bias_index = {}
    bias_manifest_sha = None
    if any(method in DEFECT_METHODS for method, _ in method_tiers):
        if bias_manifest_path is None or bias_raw_root is None:
            raise ValueError("defect measurement manifest requires frozen bias evidence")
        bias_manifest = json.loads(Path(bias_manifest_path).read_text(encoding="ascii"))
        bias_manifest_sha = sha256_json(bias_manifest)
        bias_index = _bias_index_from_manifest(
            bias_raw_root, bias_manifest, registry, config, source_commit,
        )
    entries = []
    for method, tier in method_tiers:
        for cell in _stage_cells(config, stage):
            key = (stage, method, tier, sha256_json(cell))
            bias = bias_index.get(key)
            if method in DEFECT_METHODS and bias is None:
                raise GlobalConflictError("measurement manifest is missing a bound bias")
            for family in INIT_FAMILIES:
                for trajectory in range(TRAJECTORIES_PER_FAMILY):
                    task = global_task_identity(
                        registry, config, source_commit, stage, method, tier, cell,
                        family, trajectory,
                        bias_binding=None if bias is None else bias["binding"],
                    )
                    fingerprint = sha256_json(task)
                    entries.append({
                        "task": task,
                        "task_fingerprint": fingerprint,
                        "output_relpath": f"trajectories/{fingerprint}.npz",
                        "bias_relpath": None if bias is None else bias["relpath"],
                    })
    fingerprints = [value["task_fingerprint"] for value in entries]
    if len(fingerprints) != len(set(fingerprints)):
        raise GlobalConflictError("measurement manifest has duplicate tasks")
    manifest = {
        "manifest_version": GLOBAL_TASKS_VERSION,
        "kind": "measurement",
        "stage": stage,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "method_tiers": [list(value) for value in method_tiers],
        "bias_manifest_sha256": bias_manifest_sha,
        "tasks": entries,
    }
    atomic_json(output_path, manifest)
    return manifest


def prepare_postselection_controls(selection_path, postselection_plan_path,
                                   registry_path, config_path, schedule_path,
                                   output_dir, output_index):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    schedule = validate_global_schedule(schedule_path, registry, config)
    selection = json.loads(Path(selection_path).read_text(encoding="ascii"))
    plan = json.loads(Path(postselection_plan_path).read_text(encoding="ascii"))
    selection_identity = {
        key: value for key, value in selection.items()
        if key != "selection_sha256"
    }
    plan_identity = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (selection.get("selection_version") != GLOBAL_SELECTION_VERSION
            or selection.get("status") != "FROZEN_DISCOVERY_METHODS"
            or selection.get("selection_sha256") != sha256_json(selection_identity)
            or selection.get("registry_sha256") != registry["registry_sha256"]
            or selection.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or selection.get("schedule_file_sha256") != sha256_file(schedule_path)
            or selection.get("schedule_sha256") != schedule["schedule_sha256"]
            or plan.get("plan_version") != GLOBAL_POSTSELECTION_VERSION
            or plan.get("status") != "FROZEN_POSTSELECTION"
            or plan.get("plan_sha256") != sha256_json(plan_identity)
            or plan.get("selection_sha256") != selection.get("selection_sha256")
            or plan.get("source_commit") != selection.get("source_commit")
            or plan.get("registry_sha256") != registry["registry_sha256"]
            or plan.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or plan.get("schedule_file_sha256") != sha256_file(schedule_path)
            or plan.get("schedule_sha256") != schedule["schedule_sha256"]
            or time.time() > float(schedule["deadlines_unix"]["freeze"])):
        raise GlobalConflictError("cannot freeze controls from an invalid/late plan")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_index = Path(output_index)
    if output_index.parent.resolve() != output_dir.resolve():
        raise ValueError("frozen control index must live beside its controls")
    bias_controls = {}
    for stage in ("hard_fresh", "confirmation", "resolution"):
        method_tiers = [
            tuple(value) for value in plan["stage_method_tiers"][stage]
            if value[0] in DEFECT_METHODS
        ]
        path = output_dir / f"{stage}_bias.json"
        manifest = build_bias_manifest(
            registry_path, config_path, selection["source_commit"], stage,
            method_tiers, path,
        )
        bias_controls[stage] = {
            "filename": path.name,
            "file_sha256": sha256_file(path),
            "manifest_sha256": sha256_json(manifest),
        }
    ti_path = output_dir / "ti_anchors.json"
    ti_manifest = build_ti_anchor_manifest(
        registry_path, config_path, selection["source_commit"], ti_path,
    )
    identity = {
        "control_freeze_version": GLOBAL_CONTROL_FREEZE_VERSION,
        "status": "FROZEN_CONTROLS",
        "source_commit": selection["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "postselection_plan_sha256": plan["plan_sha256"],
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "frozen_unix": time.time(),
        "bias_controls": bias_controls,
        "ti_control": {
            "filename": ti_path.name,
            "file_sha256": sha256_file(ti_path),
            "manifest_sha256": sha256_json(ti_manifest),
        },
        "measurement_templates": {
            stage: {
                "method_tiers": plan["stage_method_tiers"][stage],
                "ordered_panel_sha256": plan["stage_panel_sha256"][stage],
            }
            for stage in ("hard_fresh", "confirmation", "resolution")
        },
    }
    if identity["frozen_unix"] > float(schedule["deadlines_unix"]["freeze"]):
        raise GlobalConflictError("control freeze crossed the hour-20 deadline")
    freeze = {**identity, "control_freeze_sha256": sha256_json(identity)}
    if output_index.exists():
        if json.loads(output_index.read_text(encoding="ascii")) != freeze:
            raise GlobalConflictError("existing frozen control index conflicts")
    else:
        atomic_json(output_index, freeze)
    return freeze


def materialize_postselection_measurement(stage, control_index_path,
                                          postselection_plan_path,
                                          registry_path, config_path,
                                          bias_raw_root, output_path):
    if stage not in ("hard_fresh", "confirmation", "resolution"):
        raise ValueError("postselection measurement stage is invalid")
    index_path = Path(control_index_path)
    freeze = json.loads(index_path.read_text(encoding="ascii"))
    identity = {
        key: value for key, value in freeze.items()
        if key != "control_freeze_sha256"
    }
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    plan = json.loads(Path(postselection_plan_path).read_text(encoding="ascii"))
    plan_identity = {
        key: value for key, value in plan.items()
        if key != "plan_sha256"
    }
    if (freeze.get("control_freeze_version") != GLOBAL_CONTROL_FREEZE_VERSION
            or freeze.get("status") != "FROZEN_CONTROLS"
            or freeze.get("control_freeze_sha256") != sha256_json(identity)
            or freeze.get("registry_sha256") != registry["registry_sha256"]
            or freeze.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or plan.get("plan_version") != GLOBAL_POSTSELECTION_VERSION
            or plan.get("status") != "FROZEN_POSTSELECTION"
            or plan.get("plan_sha256") != sha256_json(plan_identity)
            or freeze.get("postselection_plan_sha256") != plan.get("plan_sha256")
            or freeze.get("measurement_templates", {}).get(stage) != {
                "method_tiers": plan.get("stage_method_tiers", {}).get(stage),
                "ordered_panel_sha256": plan.get("stage_panel_sha256", {}).get(stage),
            }):
        raise GlobalConflictError("postselection control index is invalid")
    control = freeze["bias_controls"][stage]
    bias_manifest_path = index_path.parent / control["filename"]
    if (sha256_file(bias_manifest_path) != control["file_sha256"]
            or sha256_json(json.loads(bias_manifest_path.read_text(encoding="ascii")))
            != control["manifest_sha256"]):
        raise GlobalConflictError("frozen bias control changed before materialization")
    manifest = build_measurement_manifest(
        registry_path, config_path, freeze["source_commit"], stage,
        [tuple(value) for value in freeze["measurement_templates"][stage]["method_tiers"]],
        output_path, bias_manifest_path=bias_manifest_path,
        bias_raw_root=bias_raw_root,
    )
    if manifest.get("bias_manifest_sha256") != control["manifest_sha256"]:
        raise GlobalConflictError("materialized measurement used the wrong bias control")
    manifest.update({
        "postselection_plan_sha256": plan["plan_sha256"],
        "control_freeze_sha256": freeze["control_freeze_sha256"],
    })
    atomic_json(output_path, manifest)
    return manifest


def _validate_measurement_manifest_structure(manifest, registry, config):
    if (manifest.get("manifest_version") != GLOBAL_TASKS_VERSION
            or manifest.get("kind") != "measurement"
            or manifest.get("registry_sha256") != registry["registry_sha256"]
            or manifest.get("discovery_config_sha256")
            != config["discovery_config_sha256"]):
        raise GlobalConflictError("global measurement manifest identity mismatch")
    _validate_source_commit(manifest.get("source_commit"))
    stage = manifest.get("stage")
    cells = _stage_cells(config, stage)
    method_tiers = [tuple(value) for value in manifest.get("method_tiers", [])]
    if not method_tiers or len(method_tiers) != len(set(method_tiers)):
        raise GlobalConflictError("global measurement method list is noncanonical")
    if any(method not in (*HARD_METHODS, *DEFECT_METHODS) for method, _ in method_tiers):
        raise GlobalConflictError("global measurement manifest has an unknown method")
    has_defect = any(method in DEFECT_METHODS for method, _ in method_tiers)
    bias_manifest_sha = manifest.get("bias_manifest_sha256")
    if has_defect:
        if not isinstance(bias_manifest_sha, str) or re.fullmatch(
                r"[0-9a-f]{64}", bias_manifest_sha) is None:
            raise GlobalConflictError("defect measurement manifest lacks a bias manifest SHA")
    elif bias_manifest_sha is not None:
        raise GlobalConflictError("hard-only measurement manifest binds an irrelevant bias manifest")

    entries = manifest.get("tasks")
    expected_count = (
        len(method_tiers) * len(cells) * len(INIT_FAMILIES)
        * TRAJECTORIES_PER_FAMILY
    )
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise GlobalConflictError("global measurement manifest task count is noncanonical")
    expected_coordinates = [
        (method, tier, cell, family, trajectory)
        for method, tier in method_tiers
        for cell in cells
        for family in INIT_FAMILIES
        for trajectory in range(TRAJECTORIES_PER_FAMILY)
    ]
    bindings = {}
    for entry, coordinate in zip(entries, expected_coordinates):
        if set(entry) != {
                "task", "task_fingerprint", "output_relpath", "bias_relpath"}:
            raise GlobalConflictError("global measurement manifest entry schema mismatch")
        method, tier, cell, family, trajectory = coordinate
        task = entry["task"]
        binding = task.get("bias_binding")
        expected = global_task_identity(
            registry, config, manifest["source_commit"], stage, method, tier,
            cell, family, trajectory, bias_binding=binding,
        )
        if task != expected:
            raise GlobalConflictError("global measurement manifest task/order is noncanonical")
        fingerprint = sha256_json(task)
        if (entry["task_fingerprint"] != fingerprint
                or entry["output_relpath"] != f"trajectories/{fingerprint}.npz"):
            raise GlobalConflictError("global measurement manifest fingerprint/path mismatch")
        key = (method, tier, sha256_json(cell))
        if method in DEFECT_METHODS:
            expected_bias_path = f"bias/{binding['bias_task_fingerprint']}.npz"
            if entry["bias_relpath"] != expected_bias_path:
                raise GlobalConflictError("global defect task has a noncanonical bias path")
            if key in bindings and bindings[key] != binding:
                raise GlobalConflictError("global defect trajectories bind different biases")
            bindings[key] = binding
        elif entry["bias_relpath"] is not None:
            raise GlobalConflictError("global hard-coset task unexpectedly binds a bias")
    return True


_PARALLEL_REPLAY_CONTEXT = {}


def _parallel_validate_measurement_entry(payload):
    raw_root, entry, registry_path, config_path, source_commit = payload
    key = (str(registry_path), str(config_path), str(source_commit))
    context = _PARALLEL_REPLAY_CONTEXT.get(key)
    if context is None:
        registry = _registry_with_path(load_registry(registry_path), registry_path)
        config = load_global_discovery_config(config_path, registry)
        context = {
            "registry": registry,
            "config": config,
            "bias_cache": {},
        }
        _PARALLEL_REPLAY_CONTEXT.clear()
        _PARALLEL_REPLAY_CONTEXT[key] = context
    registry = context["registry"]
    config = context["config"]
    raw_root = Path(raw_root)
    path = raw_root / entry["output_relpath"]
    task = entry["task"]
    if task["method_id"] in HARD_METHODS:
        record = validate_hard_raw(path, registry, config, source_commit)
    else:
        if entry["bias_relpath"] is None:
            raise GlobalConflictError("defect manifest entry lacks bias path")
        bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
        cache_key = str(bias_path)
        bias_cache = context["bias_cache"]
        if cache_key not in bias_cache:
            bias_cache[cache_key] = validate_bias_raw(
                bias_path, registry, config, source_commit,
            )
        record = validate_defect_raw(
            path, registry, config, source_commit, bias_path,
            _validated_bias_record=bias_cache[cache_key],
        )
    if record["task_fingerprint"] != entry["task_fingerprint"]:
        raise GlobalConflictError("measurement manifest fingerprint/raw mismatch")
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
            f"measurement raw set mismatch; missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}"
        )
    if isinstance(num_workers, bool) or int(num_workers) <= 0:
        raise ValueError("measurement replay worker count must be positive")
    num_workers = int(num_workers)
    if num_workers > 1:
        registry_path = registry.get("_registry_path")
        if registry_path is None:
            raise ValueError("parallel measurement replay needs the registry path")
        payloads = [
            (
                str(raw_root), entry, str(registry_path),
                str(config["config_path"]), manifest["source_commit"],
            )
            for entry in manifest["tasks"]
        ]
        workers = min(num_workers, len(payloads))
        chunksize = max(1, len(payloads) // max(1, workers * 8))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(
                _parallel_validate_measurement_entry, payloads,
                chunksize=chunksize,
            ))

    records = []
    bias_cache = {}
    for entry in manifest["tasks"]:
        path = raw_root / entry["output_relpath"]
        task = entry["task"]
        if task["method_id"] in HARD_METHODS:
            record = validate_hard_raw(
                path, registry, config, manifest["source_commit"],
            )
        else:
            if entry["bias_relpath"] is None:
                raise GlobalConflictError("defect manifest entry lacks bias path")
            bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
            cache_key = str(bias_path)
            if cache_key not in bias_cache:
                bias_cache[cache_key] = validate_bias_raw(
                    bias_path, registry, config, manifest["source_commit"],
                )
            record = validate_defect_raw(
                path, registry, config, manifest["source_commit"],
                bias_path,
                _validated_bias_record=bias_cache[cache_key],
            )
        if record["task_fingerprint"] != entry["task_fingerprint"]:
            raise GlobalConflictError("measurement manifest fingerprint/raw mismatch")
        records.append(record)
    return records


def verify_global_remote_evidence(run_root, control_path, ownership_path,
                                  deployment_root, schedule_path):
    """Verify source/control/ownership/markers/node manifests before analysis."""
    run_root = Path(run_root).resolve(strict=True)
    control_path = Path(control_path).resolve(strict=True)
    ownership_path = Path(ownership_path).resolve(strict=True)
    deployment_root = Path(deployment_root).resolve(strict=True)
    schedule_path = Path(schedule_path).resolve(strict=True)
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    control_sha = sha256_file(control_path)
    ownership_sha = sha256_file(ownership_path)
    schedule_file_sha = sha256_file(schedule_path)
    schedule = json.loads(schedule_path.read_text(encoding="ascii"))
    schedule_identity = {
        key: value for key, value in schedule.items() if key != "schedule_sha256"
    }
    if (schedule.get("schedule_version") != GLOBAL_SCHEDULE_VERSION
            or schedule.get("status") != "FROZEN_72H"
            or schedule.get("schedule_sha256") != sha256_json(schedule_identity)
            or schedule.get("source_commit") != control.get("source_commit")
            or schedule.get("registry_sha256") != control.get("registry_sha256")
            or schedule.get("discovery_config_sha256")
            != control.get("discovery_config_sha256")):
        raise GlobalConflictError("remote global schedule identity mismatch")
    tasks = [entry["task"] for entry in control["tasks"]]
    expected_ownership = fixed_global_ownership(
        tasks, ownership.get("nodes", []), control["source_commit"], control_sha,
        f"{control['stage']}:{control['kind']}",
    )
    if ownership != expected_ownership:
        raise GlobalConflictError("remote global ownership is noncanonical")
    source_commit = (deployment_root / "SOURCE_COMMIT").read_text(encoding="ascii").strip()
    archive_sha = (deployment_root / "ARCHIVE_SHA256").read_text(encoding="ascii").strip()
    source_manifest = json.loads(
        (deployment_root / "SOURCE_MANIFEST.json").read_text(encoding="ascii")
    )
    if (source_commit != control["source_commit"]
            or sha256_file(deployment_root / "SOURCE.tar") != archive_sha
            or source_manifest.get("source_commit") != source_commit
            or source_manifest.get("archive_sha256") != archive_sha
            or schedule.get("archive_sha256") != archive_sha
            or schedule.get("source_manifest_sha256")
            != sha256_file(deployment_root / "SOURCE_MANIFEST.json")):
        raise GlobalConflictError("remote global source archive identity mismatch")
    expected_by_node = defaultdict(set)
    entry_by_fingerprint = {
        entry["task_fingerprint"]: entry for entry in control["tasks"]
    }
    for fingerprint, node in ownership["task_owner"].items():
        expected_by_node[node].add(fingerprint)
    seen = set()
    node_evidence = []
    completion_times = []
    evidence_root = run_root / "global" / control["stage"]
    for node in ownership["nodes"]:
        marker_root = evidence_root / "markers" / control_sha[:12] / node
        present = [name for name in ("RUNNING", "SUCCESS", "FAILED") if (marker_root / name).exists()]
        if present != ["SUCCESS"]:
            raise GlobalConflictError(f"remote global marker is not exclusive SUCCESS on {node}")
        success = json.loads((marker_root / "SUCCESS").read_text(encoding="ascii"))
        if success.get("stage_fingerprint") != ownership["stage_fingerprint"]:
            raise GlobalConflictError("remote global SUCCESS fingerprint mismatch")
        try:
            completed = datetime.strptime(
                success["completed_utc"], "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc).timestamp()
        except (KeyError, TypeError, ValueError) as exc:
            raise GlobalConflictError("remote global completion time is malformed") from exc
        if completed > float(schedule["deadlines_unix"][control["stage"]]):
            raise GlobalConflictError("remote global stage completed after its deadline")
        completion_times.append(completed)
        node_root = evidence_root / "node_manifests" / control_sha[:12] / node
        status_path = node_root / "stage_status.json"
        raw_manifest_path = node_root / "raw_manifest.json"
        status = json.loads(status_path.read_text(encoding="ascii"))
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="ascii"))
        if (status.get("status") != "SUCCESS"
                or status.get("stage_fingerprint") != ownership["stage_fingerprint"]
                or status.get("raw_manifest_sha256") != sha256_file(raw_manifest_path)
                or raw_manifest.get("stage_fingerprint") != ownership["stage_fingerprint"]
                or raw_manifest.get("source_commit") != source_commit
                or raw_manifest.get("control_sha256") != control_sha
                or raw_manifest.get("ownership_sha256") != ownership_sha
                or raw_manifest.get("schedule_file_sha256") != schedule_file_sha
                or raw_manifest.get("schedule_sha256") != schedule["schedule_sha256"]
                or raw_manifest.get("node") != node):
            raise GlobalConflictError("remote global node status/manifest identity mismatch")
        identity = raw_manifest.get("source_identity")
        if (not isinstance(identity, dict)
                or identity.get("source_commit") != source_commit
                or identity.get("mode") != "archive"
                or identity.get("archive_sha256") != archive_sha
                or identity.get("manifest_sha256")
                != sha256_file(deployment_root / "SOURCE_MANIFEST.json")):
            raise GlobalConflictError("remote global node did not use verified source")
        node_fingerprints = {value["task_fingerprint"] for value in raw_manifest["files"]}
        if node_fingerprints != expected_by_node[node]:
            raise GlobalConflictError("remote global node raw task set mismatch")
        for value in raw_manifest["files"]:
            fingerprint = value["task_fingerprint"]
            if fingerprint in seen:
                raise GlobalConflictError("remote global raw task appears on multiple nodes")
            seen.add(fingerprint)
            entry = entry_by_fingerprint[fingerprint]
            if value["path"] != entry["output_relpath"]:
                raise GlobalConflictError("remote global raw path differs from manifest")
            raw_path = evidence_root / value["path"]
            if sha256_file(raw_path) != value["sha256"]:
                raise GlobalConflictError("remote global raw SHA256 mismatch")
        node_evidence.append({
            "node": node,
            "completed_unix": completed,
            "success_sha256": sha256_file(marker_root / "SUCCESS"),
            "status_sha256": sha256_file(status_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
        })
    if seen != set(entry_by_fingerprint):
        raise GlobalConflictError("remote global evidence is incomplete")
    return {
        "source_commit": source_commit,
        "archive_sha256": archive_sha,
        "source_manifest_sha256": sha256_file(deployment_root / "SOURCE_MANIFEST.json"),
        "control_sha256": control_sha,
        "ownership_sha256": ownership_sha,
        "stage_fingerprint": ownership["stage_fingerprint"],
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "completed_unix_max": max(completion_times),
        "nodes": node_evidence,
    }


def _public_summary(summary):
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def analyze_measurement_stage(raw_root, manifest_path, registry_path, config_path,
                              output_path=None, *, verified_evidence=None,
                              num_workers=1):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_global_discovery_config(config_path, registry)
    manifest = json.loads(Path(manifest_path).read_text(encoding="ascii"))
    _validate_measurement_manifest_structure(manifest, registry, config)
    records = _load_measurement_records(
        raw_root, manifest, registry, config, num_workers=num_workers,
    )
    evidence = None
    if verified_evidence is not None:
        evidence = verify_global_remote_evidence(**verified_evidence)
        if evidence["control_sha256"] != sha256_file(manifest_path):
            raise GlobalConflictError("verified remote control differs from analyzer manifest")
    grouped = defaultdict(list)
    for record in records:
        key = (
            sha256_json(record["cell"]), record["method_id"], record["resource_tier"],
        )
        grouped[key].append(record)
    summaries = [_cell_method_summary(value, config) for _, value in sorted(grouped.items())]

    by_method_tier = defaultdict(list)
    for summary in summaries:
        by_method_tier[(summary["method_id"], summary["resource_tier"])].append(summary)
    method_status = []
    for (method, tier), values in sorted(by_method_tier.items()):
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
            worm_efficiency = d0_ess / max(sum(value["core_seconds"] for value in values), 1e-300)
        method_status.append({
            "method_id": method, "resource_tier": tier,
            "cells_passed": sum(value["valid"] for value in values),
            "cells_total": len(values),
            "core_seconds": float(sum(value["core_seconds"] for value in values)),
            "d0_ess_per_core_second": worm_efficiency,
            "valid": all(value["valid"] for value in values),
        })

    comparisons = []
    by_cell = defaultdict(list)
    for summary in summaries:
        by_cell[sha256_json(summary["cell"])].append(summary)
    for values in by_cell.values():
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                comparison = compare_cell_summaries(
                    values[left_index], values[right_index], config,
                )
                comparison["cell"] = values[left_index]["cell"]
                comparisons.append(comparison)
    all_summaries_valid = all(value["valid"] for value in summaries)
    all_comparisons_valid = all(value["valid"] for value in comparisons)
    report = {
        "report_version": GLOBAL_REPORT_VERSION,
        "stage": manifest["stage"],
        "source_commit": manifest["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "manifest_sha256": sha256_json(manifest),
        "bias_manifest_sha256": manifest.get("bias_manifest_sha256"),
        "postselection_plan_sha256": manifest.get("postselection_plan_sha256"),
        "control_freeze_sha256": manifest.get("control_freeze_sha256"),
        "raw_count": len(records),
        "cell_summaries": [_public_summary(value) for value in summaries],
        "method_status": method_status,
        "comparisons": comparisons,
        "verified_remote_evidence": evidence,
        "status": (
            "PASS" if all_summaries_valid and all_comparisons_valid
            else "SAMPLING_INSUFFICIENT"
        ),
    }
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def analyze_ti_anchor_stage(raw_root, manifest_path, registry_path, config_path,
                            output_path=None, *, verified_evidence=None):
    raw_root = Path(raw_root)
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    expected = _canonical_ti_anchor_manifest(
        registry, config, manifest.get("source_commit"),
    )
    if manifest != expected:
        raise GlobalConflictError("TI anchor manifest is noncanonical")
    expected_paths = {entry["output_relpath"] for entry in manifest["tasks"]}
    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in (raw_root / "ti_anchors").glob("*.npz")
    } if (raw_root / "ti_anchors").exists() else set()
    if actual_paths != expected_paths:
        raise GlobalConflictError("TI anchor raw set differs from its manifest")
    records = []
    for entry in manifest["tasks"]:
        record = validate_ti_anchor_raw(
            raw_root / entry["output_relpath"], registry, config,
            registry_path, manifest["source_commit"],
        )
        if record["task_fingerprint"] != entry["task_fingerprint"]:
            raise GlobalConflictError("TI anchor manifest/raw fingerprint mismatch")
        records.append(record)
    evidence = None
    if verified_evidence is not None:
        evidence = verify_global_remote_evidence(**verified_evidence)
        if evidence["control_sha256"] != sha256_file(manifest_path):
            raise GlobalConflictError("verified TI remote control differs from manifest")
    public_records = [{
        "cell": record["cell"],
        "task_fingerprint": record["task_fingerprint"],
        "raw_sha256": record["sha256"],
        "num_qubits": record["num_qubits"],
        "k": record["k"],
        "q_top": record["q_top"],
        "q_top_stderr": record["q_top_stderr"],
        "characters_absolute": record["characters_absolute"].tolist(),
        "weights_absolute": record["weights_absolute"].tolist(),
        "grid_tv": record["grid_tv"],
        "grid_q_top_abs_diff": record["grid_q_top_abs_diff"],
        "flags": record["flags"],
        "valid_for_aggregation": record["valid_for_aggregation"],
        "core_seconds": record["core_seconds"],
    } for record in records]
    report = {
        "report_version": TI_ANCHOR_REPORT_VERSION,
        "stage": "ti_anchors",
        "source_commit": manifest["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "manifest_sha256": sha256_json(manifest),
        "raw_count": len(records),
        "anchors": public_records,
        "verified_remote_evidence": evidence,
        "status": (
            "PASS" if all(record["valid_for_aggregation"] for record in records)
            else "SAMPLING_INSUFFICIENT"
        ),
    }
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def compare_ti_anchors(confirmation_report_path, ti_report_path,
                       registry_path, config_path, output_path=None):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    confirmation = json.loads(
        Path(confirmation_report_path).read_text(encoding="ascii")
    )
    ti_report = json.loads(Path(ti_report_path).read_text(encoding="ascii"))
    for report, version, stage in (
            (confirmation, GLOBAL_REPORT_VERSION, "confirmation"),
            (ti_report, TI_ANCHOR_REPORT_VERSION, "ti_anchors")):
        if (report.get("report_version") != version
                or report.get("stage") != stage
                or report.get("registry_sha256") != registry["registry_sha256"]
                or report.get("discovery_config_sha256")
                != config["discovery_config_sha256"]
                or report.get("source_commit") != confirmation.get("source_commit")
                or report.get("verified_remote_evidence") is None):
            raise GlobalConflictError("TI comparison report identity/evidence mismatch")
    anchors = {sha256_json(value["cell"]): value for value in ti_report["anchors"]}
    expected_cells = _stage_cells(config, "ti_anchors")
    if set(anchors) != {sha256_json(value) for value in expected_cells}:
        raise GlobalConflictError("TI comparison anchor panel is incomplete")
    method_summaries = [
        value for value in confirmation.get("cell_summaries", [])
        if value.get("cell") in expected_cells
    ]
    method_keys = {
        (value["method_id"], value["resource_tier"])
        for value in method_summaries
    }
    if (len(method_keys) != 2
            or len(method_summaries) != 2 * len(expected_cells)):
        raise GlobalConflictError("TI comparison requires two methods on all m3 anchors")
    gates = config["gates"]
    comparisons = []
    for summary in method_summaries:
        anchor = anchors[sha256_json(summary["cell"])]
        payload = summary.get("ti_anchor_payload")
        if not isinstance(payload, dict):
            raise GlobalConflictError("m3 method summary lacks its TI payload")
        payload_identity = {
            "character_masks": payload.get("character_masks"),
            "trajectory_character_means": payload.get("trajectory_character_means"),
        }
        if payload.get("payload_sha256") != sha256_json(payload_identity):
            raise GlobalConflictError("m3 TI payload SHA mismatch")
        masks = np.asarray(payload_identity["character_masks"], dtype=np.uint64)
        means = np.asarray(
            payload_identity["trajectory_character_means"], dtype=np.float64,
        )
        k = int(anchor["k"])
        expected_masks = np.arange(1, 1 << k, dtype=np.uint64)
        if (not np.array_equal(masks, expected_masks)
                or means.shape != (2 * TRAJECTORIES_PER_FAMILY, masks.size)):
            raise GlobalConflictError("m3 TI payload character shape/order mismatch")
        character_set = CharacterSet(
            masks=masks,
            basis_positions=np.asarray(
                [int(np.flatnonzero(masks == (np.uint64(1) << np.uint64(bit)))[0])
                 for bit in range(k)], dtype=np.int32,
            ),
            tier="full", k=k, random_seed=None,
            character_sha256="full-ti-comparison",
        )
        recomputed_q = character_qtop_estimate(character_set, means)
        if (not np.isclose(recomputed_q["q_top"], summary["q_top"], atol=1e-12, rtol=0.0)
                or not np.isclose(
                    recomputed_q["q_top_total_se"], summary["q_top_total_se"],
                    atol=1e-12, rtol=0.0,
                )):
            raise GlobalConflictError("m3 method TI payload disagrees with its summary")
        ti_characters = np.asarray(anchor["characters_absolute"], dtype=np.float64)
        if ti_characters.shape != masks.shape or not np.all(np.isfinite(ti_characters)):
            raise GlobalConflictError("TI anchor character vector is malformed")
        ti_means = np.repeat(ti_characters[None, :], 3, axis=0)
        d2 = character_d2_estimate(character_set, means, ti_means)
        delta = abs(float(summary["q_top"]) - float(anchor["q_top"]))
        delta_se = math.hypot(
            float(summary["q_top_total_se"]), float(anchor["q_top_stderr"]),
        )
        q_pass = (
            delta <= gates["max_abs_delta_q_top"]
            and delta <= gates["delta_sigma_multiplier"] * delta_se
                         + gates["delta_sigma_slack"]
        )
        d2_upper = max(0.0, d2["d2_norm"]) + 3.0 * d2["d2_total_se"]
        d2_pass = d2_upper <= gates["max_d2_upper"]
        comparisons.append({
            "cell": summary["cell"],
            "method_id": summary["method_id"],
            "resource_tier": summary["resource_tier"],
            "method_q_top": summary["q_top"],
            "method_q_top_total_se": summary["q_top_total_se"],
            "ti_q_top": anchor["q_top"],
            "ti_q_top_stderr": anchor["q_top_stderr"],
            "delta_q_top": delta,
            "se_delta_q_top": delta_se,
            "q_top_pass": q_pass,
            "d2_norm": d2["d2_norm"],
            "d2_total_se": d2["d2_total_se"],
            "d2_upper_3se": d2_upper,
            "d2_pass": d2_pass,
            "valid": bool(
                summary["valid"] and anchor["valid_for_aggregation"]
                and q_pass and d2_pass
            ),
        })
    report = {
        "report_version": TI_COMPARISON_VERSION,
        "source_commit": confirmation["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "confirmation_report_sha256": sha256_json(confirmation),
        "ti_report_sha256": sha256_json(ti_report),
        "method_tiers": [list(value) for value in sorted(method_keys)],
        "comparisons": comparisons,
        "status": "PASS" if all(value["valid"] for value in comparisons)
                  else "SAMPLING_INSUFFICIENT",
    }
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def _validate_stage_report_shape(report, stage, method_tiers, cells,
                                 registry, config, source_commit):
    if (report.get("report_version") != GLOBAL_REPORT_VERSION
            or report.get("stage") != stage
            or report.get("source_commit") != source_commit
            or report.get("registry_sha256") != registry["registry_sha256"]
            or report.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or report.get("verified_remote_evidence") is None):
        raise GlobalConflictError(f"{stage} report identity/evidence mismatch")
    expected = {
        (sha256_json(cell), method, tier)
        for cell in cells for method, tier in method_tiers
    }
    actual = {
        (sha256_json(value["cell"]), value["method_id"], value["resource_tier"])
        for value in report.get("cell_summaries", [])
    }
    if actual != expected or len(report.get("cell_summaries", [])) != len(expected):
        raise GlobalConflictError(f"{stage} report cell/method set is noncanonical")
    expected_raw = len(expected) * len(INIT_FAMILIES) * TRAJECTORIES_PER_FAMILY
    if report.get("raw_count") != expected_raw:
        raise GlobalConflictError(f"{stage} report raw count is noncanonical")
    statuses = {
        (value["method_id"], value["resource_tier"]): value
        for value in report.get("method_status", [])
    }
    if set(statuses) != set(method_tiers):
        raise GlobalConflictError(f"{stage} method status set is noncanonical")
    if any(value.get("cells_total") != len(cells) for value in statuses.values()):
        raise GlobalConflictError(f"{stage} method status cell count is wrong")
    expected_comparisons = {
        (sha256_json(cell), frozenset((left, right)))
        for cell in cells
        for left_index, left in enumerate(method_tiers)
        for right in method_tiers[left_index + 1:]
    }
    actual_comparisons = {
        (
            sha256_json(value["cell"]),
            frozenset((
                (value["left"]["method_id"], value["left"]["resource_tier"]),
                (value["right"]["method_id"], value["right"]["resource_tier"]),
            )),
        )
        for value in report.get("comparisons", [])
    }
    if (actual_comparisons != expected_comparisons
            or len(report.get("comparisons", [])) != len(expected_comparisons)):
        raise GlobalConflictError(f"{stage} comparison set is noncanonical")
    recomputed_status = (
        "PASS" if all(value.get("valid") for value in report["cell_summaries"])
        and all(value.get("valid") for value in report.get("comparisons", []))
        else "SAMPLING_INSUFFICIENT"
    )
    if report.get("status") != recomputed_status:
        raise GlobalConflictError(f"{stage} report status is internally inconsistent")
    return True


def combine_global_readiness(selection_path, hard_fresh_report_path,
                             confirmation_report_path, resolution_report_path,
                             ti_report_path, ti_comparison_path, registry_path,
                             config_path, schedule_path,
                             postselection_plan_path, control_index_path,
                             output_path=None):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    schedule = validate_global_schedule(
        schedule_path, registry, config,
    )
    schedule_file_sha = sha256_file(schedule_path)
    selection = json.loads(Path(selection_path).read_text(encoding="ascii"))
    identity = {
        key: value for key, value in selection.items()
        if key != "selection_sha256"
    }
    if (selection.get("selection_version") != GLOBAL_SELECTION_VERSION
            or selection.get("status") != "FROZEN_DISCOVERY_METHODS"
            or selection.get("selection_sha256") != sha256_json(identity)
            or selection.get("registry_sha256") != registry["registry_sha256"]
            or selection.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or selection.get("schedule_file_sha256") != schedule_file_sha
            or selection.get("schedule_sha256") != schedule["schedule_sha256"]):
        raise GlobalConflictError("global method selection freeze is invalid")
    selected = [
        (value["method_id"], value["resource_tier"])
        for value in selection.get("selected", [])
    ]
    if (len(selected) != 2 or len(set(selected)) != 2
            or sum(method in HARD_METHODS for method, _ in selected) != 1
            or sum(method in DEFECT_METHODS for method, _ in selected) != 1
            or len({tier for _, tier in selected}) != 1):
        raise GlobalConflictError("global method selection pair is noncanonical")
    source_commit = selection["source_commit"]
    if schedule["source_commit"] != source_commit:
        raise GlobalConflictError("global readiness schedule/source mismatch")
    base_tier = selected[0][1]
    strict_tier = "2" + base_tier
    postselection = json.loads(
        Path(postselection_plan_path).read_text(encoding="ascii")
    )
    postselection_identity = {
        key: value for key, value in postselection.items()
        if key != "plan_sha256"
    }
    expected_stage_method_tiers = {
        "hard_fresh": [
            [method, tier]
            for method, _ in selected for tier in (base_tier, strict_tier)
        ],
        "confirmation": [[method, strict_tier] for method, _ in selected],
        "resolution": [[method, base_tier] for method, _ in selected],
        "ti_anchors": [[config["ti_anchor"]["method_id"], "TI49"]],
    }
    expected_panel_sha = {
        "hard_fresh": config["panels"]["HARD2"]["ordered_panel_sha256"],
        "confirmation": sha256_json(_stage_cells(config, "confirmation")),
        "resolution": config["panels"]["RES6"]["ordered_panel_sha256"],
        "ti_anchors": sha256_json(_stage_cells(config, "ti_anchors")),
    }
    if (postselection.get("plan_version") != GLOBAL_POSTSELECTION_VERSION
            or postselection.get("status") != "FROZEN_POSTSELECTION"
            or postselection.get("plan_sha256") != sha256_json(postselection_identity)
            or postselection.get("selection_sha256") != selection["selection_sha256"]
            or postselection.get("schedule_file_sha256") != schedule_file_sha
            or postselection.get("schedule_sha256") != schedule["schedule_sha256"]
            or postselection.get("stage_method_tiers")
            != expected_stage_method_tiers
            or postselection.get("stage_panel_sha256") != expected_panel_sha
            or float(postselection.get("frozen_unix", float("inf")))
            > float(schedule["deadlines_unix"]["freeze"])):
        raise GlobalConflictError("global postselection plan is invalid")
    control_freeze = json.loads(
        Path(control_index_path).read_text(encoding="ascii")
    )
    control_identity = {
        key: value for key, value in control_freeze.items()
        if key != "control_freeze_sha256"
    }
    if (control_freeze.get("control_freeze_version")
            != GLOBAL_CONTROL_FREEZE_VERSION
            or control_freeze.get("status") != "FROZEN_CONTROLS"
            or control_freeze.get("control_freeze_sha256")
            != sha256_json(control_identity)
            or control_freeze.get("postselection_plan_sha256")
            != postselection["plan_sha256"]
            or control_freeze.get("source_commit") != source_commit
            or control_freeze.get("registry_sha256") != registry["registry_sha256"]
            or control_freeze.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or control_freeze.get("selection_sha256")
            != selection["selection_sha256"]
            or control_freeze.get("schedule_file_sha256") != schedule_file_sha
            or control_freeze.get("schedule_sha256") != schedule["schedule_sha256"]
            or control_freeze.get("measurement_templates") != {
                stage: {
                    "method_tiers": expected_stage_method_tiers[stage],
                    "ordered_panel_sha256": expected_panel_sha[stage],
                }
                for stage in ("hard_fresh", "confirmation", "resolution")
            }
            or float(control_freeze.get("frozen_unix", float("inf")))
            > float(schedule["deadlines_unix"]["freeze"])):
        raise GlobalConflictError("global frozen control index is invalid")
    control_root = Path(control_index_path).parent
    for stage in ("hard_fresh", "confirmation", "resolution"):
        control = control_freeze.get("bias_controls", {}).get(stage)
        if not isinstance(control, dict):
            raise GlobalConflictError("global frozen bias control index is incomplete")
        control_path = control_root / str(control.get("filename", ""))
        try:
            control_json = json.loads(control_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GlobalConflictError("global frozen bias control is unreadable") from exc
        if (sha256_file(control_path) != control.get("file_sha256")
                or sha256_json(control_json) != control.get("manifest_sha256")):
            raise GlobalConflictError("global frozen bias control changed")
    ti_control = control_freeze.get("ti_control")
    if not isinstance(ti_control, dict):
        raise GlobalConflictError("global frozen TI control index is incomplete")
    ti_control_path = control_root / str(ti_control.get("filename", ""))
    try:
        ti_control_json = json.loads(ti_control_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalConflictError("global frozen TI control is unreadable") from exc
    if (sha256_file(ti_control_path) != ti_control.get("file_sha256")
            or sha256_json(ti_control_json) != ti_control.get("manifest_sha256")):
        raise GlobalConflictError("global frozen TI control changed")
    hard_report = json.loads(
        Path(hard_fresh_report_path).read_text(encoding="ascii")
    )
    confirmation = json.loads(
        Path(confirmation_report_path).read_text(encoding="ascii")
    )
    resolution = json.loads(
        Path(resolution_report_path).read_text(encoding="ascii")
    )
    ti_report = json.loads(Path(ti_report_path).read_text(encoding="ascii"))
    ti_comparison = json.loads(
        Path(ti_comparison_path).read_text(encoding="ascii")
    )
    _validate_stage_report_shape(
        hard_report, "hard_fresh",
        [(method, tier) for method, _ in selected for tier in (base_tier, strict_tier)],
        _stage_cells(config, "hard_fresh"), registry, config, source_commit,
    )
    _validate_stage_report_shape(
        confirmation, "confirmation",
        [(method, strict_tier) for method, _ in selected],
        _stage_cells(config, "confirmation"), registry, config, source_commit,
    )
    _validate_stage_report_shape(
        resolution, "resolution",
        [(method, base_tier) for method, _ in selected],
        _stage_cells(config, "resolution"), registry, config, source_commit,
    )
    for name, report in (
            ("hard_fresh", hard_report),
            ("confirmation", confirmation),
            ("resolution", resolution)):
        if (report.get("postselection_plan_sha256")
                != postselection["plan_sha256"]
                or report.get("control_freeze_sha256")
                != control_freeze["control_freeze_sha256"]
                or report.get("bias_manifest_sha256")
                != control_freeze["bias_controls"][name]["manifest_sha256"]):
            raise GlobalConflictError(
                f"{name} report is not bound to the frozen controls"
            )
    if (ti_report.get("report_version") != TI_ANCHOR_REPORT_VERSION
            or ti_report.get("source_commit") != source_commit
            or ti_report.get("registry_sha256") != registry["registry_sha256"]
            or ti_report.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or ti_report.get("manifest_sha256")
            != control_freeze["ti_control"]["manifest_sha256"]
            or ti_report.get("verified_remote_evidence") is None
            or ti_report.get("raw_count") != len(_stage_cells(config, "ti_anchors"))):
        raise GlobalConflictError("TI anchor report identity/evidence is invalid")
    expected_ti_status = (
        "PASS" if all(value.get("valid_for_aggregation")
                      for value in ti_report.get("anchors", []))
        else "SAMPLING_INSUFFICIENT"
    )
    if ti_report.get("status") != expected_ti_status:
        raise GlobalConflictError("TI anchor report status is inconsistent")
    expected_strict_methods = sorted((method, strict_tier) for method, _ in selected)
    if (ti_comparison.get("report_version") != TI_COMPARISON_VERSION
            or ti_comparison.get("source_commit") != source_commit
            or ti_comparison.get("registry_sha256") != registry["registry_sha256"]
            or ti_comparison.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or ti_comparison.get("confirmation_report_sha256")
            != sha256_json(confirmation)
            or ti_comparison.get("ti_report_sha256") != sha256_json(ti_report)
            or sorted(tuple(value) for value in ti_comparison.get("method_tiers", []))
            != expected_strict_methods
            or len(ti_comparison.get("comparisons", []))
            != len(expected_strict_methods) * len(_stage_cells(config, "ti_anchors"))):
        raise GlobalConflictError("TI comparison report identity/task set is invalid")
    expected_ti_comparison_status = (
        "PASS" if all(value.get("valid")
                      for value in ti_comparison.get("comparisons", []))
        else "SAMPLING_INSUFFICIENT"
    )
    if ti_comparison.get("status") != expected_ti_comparison_status:
        raise GlobalConflictError("TI comparison report status is inconsistent")
    stage_reports = {
        "hard_fresh": hard_report,
        "confirmation": confirmation,
        "resolution": resolution,
        "ti_anchors": ti_report,
        "ti_comparison": ti_comparison,
    }
    failed_stages = [
        name for name, report in stage_reports.items()
        if report.get("status") != "PASS"
    ]
    for name, report in (
            ("hard_fresh", hard_report),
            ("confirmation", confirmation),
            ("resolution", resolution),
            ("ti_anchors", ti_report)):
        evidence = report["verified_remote_evidence"]
        if (evidence.get("schedule_file_sha256") != schedule_file_sha
                or evidence.get("schedule_sha256") != schedule["schedule_sha256"]
                or float(evidence.get("completed_unix_max", float("inf")))
                > float(schedule["deadlines_unix"][name])):
            raise GlobalConflictError(f"{name} report is outside the frozen schedule")
    if time.time() > float(schedule["deadlines_unix"]["analysis"]):
        failed_stages.append("analysis_deadline")
    all_cell_failures = []
    for name, report in (
            ("hard_fresh", hard_report),
            ("confirmation", confirmation),
            ("resolution", resolution)):
        for summary in report["cell_summaries"]:
            if not summary["valid"]:
                all_cell_failures.append({
                    "stage": name,
                    "cell": summary["cell"],
                    "method_id": summary["method_id"],
                    "resource_tier": summary["resource_tier"],
                    "failures": summary["failures"],
                })
    status = (
        "READY_FOR_FORMAL" if not failed_stages
        else "UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET"
    )
    result = {
        "readiness_version": GLOBAL_READINESS_VERSION,
        "status": status,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "postselection_plan_sha256": postselection["plan_sha256"],
        "control_freeze_sha256": control_freeze["control_freeze_sha256"],
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "selected": selection["selected"],
        "report_sha256": {
            "hard_fresh": sha256_json(hard_report),
            "confirmation": sha256_json(confirmation),
            "resolution": sha256_json(resolution),
            "ti_anchors": sha256_json(ti_report),
            "ti_comparison": sha256_json(ti_comparison),
        },
        "failed_stages": failed_stages,
        "failed_cell_methods": all_cell_failures,
        "formal_production_authorized": False,
        "interpretation": (
            "Discovery passed; fresh exp102.q0_global.v1 tuning and held-out "
            "are still required before FROZEN_HELD_OUT_PASS."
            if status == "READY_FOR_FORMAL" else
            "The complete requested range remains unresolved within the frozen "
            "algorithm and 72-hour budget; this is not an impossibility claim."
        ),
    }
    if output_path is not None:
        atomic_json(output_path, result)
    return result


def freeze_method_selection(screen_report_path, runtime_report_path,
                            digest_report_path, preflight_report_path,
                            registry_path, config_path, schedule_path,
                            output_path):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    schedule = validate_global_schedule(schedule_path, registry, config)
    schedule_file_sha = sha256_file(schedule_path)
    if time.time() > float(schedule["deadlines_unix"]["freeze"]):
        raise GlobalConflictError("method selection missed the frozen hour-20 deadline")
    screen = json.loads(Path(screen_report_path).read_text(encoding="ascii"))
    runtime = json.loads(Path(runtime_report_path).read_text(encoding="ascii"))
    digest = json.loads(Path(digest_report_path).read_text(encoding="ascii"))
    preflight = json.loads(Path(preflight_report_path).read_text(encoding="ascii"))
    if schedule["source_commit"] != screen.get("source_commit"):
        raise GlobalConflictError("method selection schedule/source mismatch")
    for report, name in ((screen, "screen"), (runtime, "runtime"), (digest, "digest")):
        if report.get("registry_sha256") != registry["registry_sha256"] or report.get("discovery_config_sha256") != config["discovery_config_sha256"]:
            raise GlobalConflictError(f"global {name} report identity mismatch")
    if (preflight.get("report_version") != "exp102.q0_global.preflight.v1"
            or preflight.get("status") != "PASS"
            or preflight.get("source_commit") != screen.get("source_commit")
            or preflight.get("registry_sha256") != registry["registry_sha256"]
            or preflight.get("discovery_config_sha256")
            != config["discovery_config_sha256"]
            or preflight.get("schedule_file_sha256") != schedule_file_sha
            or preflight.get("schedule_sha256") != schedule["schedule_sha256"]
            or preflight.get("runtime_consensus_sha256")
            != sha256_file(runtime_report_path)
            or preflight.get("digest_consensus_sha256")
            != sha256_file(digest_report_path)
            or float(preflight.get("completed_unix", float("inf")))
            > float(schedule["deadlines_unix"]["digest_runtime"])):
        raise GlobalConflictError("method selection preflight evidence is invalid")
    if (screen.get("report_version") != GLOBAL_REPORT_VERSION
            or screen.get("stage") != "screen"
            or screen.get("verified_remote_evidence") is None):
        raise GlobalConflictError("method selection requires verified remote screen evidence")
    screen_evidence = screen["verified_remote_evidence"]
    if (screen_evidence.get("schedule_file_sha256") != schedule_file_sha
            or screen_evidence.get("schedule_sha256") != schedule["schedule_sha256"]
            or float(screen_evidence.get("completed_unix_max", float("inf")))
            > float(schedule["deadlines_unix"]["screen"])):
        raise GlobalConflictError("screen evidence is outside the frozen schedule")
    if runtime.get("status") != "PASS":
        raise GlobalConflictError("cannot freeze methods without a passing runtime report")
    if runtime.get("benchmark_version") != "exp102.q0_global.runtime_consensus.v1":
        raise GlobalConflictError("method selection requires three-node runtime consensus")
    if (digest.get("report_version") != "exp102.q0_global.digest_consensus.v1"
            or digest.get("status") != "PASS"
            or digest.get("source_commit") != screen.get("source_commit")
            or runtime.get("source_commit") != screen.get("source_commit")
            or float(runtime.get("completed_unix_max", float("inf")))
            > float(schedule["deadlines_unix"]["digest_runtime"])
            or float(digest.get("completed_unix_max", float("inf")))
            > float(schedule["deadlines_unix"]["digest_runtime"])):
        raise GlobalConflictError("three-node digest/runtime missed the frozen initial gate")
    for report, name in ((runtime, "runtime"), (digest, "digest")):
        source_identity = report.get("source_identity")
        if (not isinstance(source_identity, dict)
                or source_identity.get("mode") != "archive"
                or source_identity.get("source_commit") != schedule["source_commit"]
                or source_identity.get("archive_sha256")
                != schedule["archive_sha256"]
                or source_identity.get("manifest_sha256")
                != schedule["source_manifest_sha256"]):
            raise GlobalConflictError(f"{name} consensus used the wrong source archive")
    if runtime.get("source_commit") != screen.get("source_commit"):
        raise GlobalConflictError("screen/runtime source commits differ")
    selected_tier = runtime.get("selected_resource_tier")
    eligible = list(runtime.get("selected_eligible_methods", []))
    if (selected_tier not in RESOURCE_TIERS or not eligible
            or len(eligible) != len(set(eligible))
            or any(method not in (*HARD_METHODS, *DEFECT_METHODS) for method in eligible)):
        raise GlobalConflictError("runtime report has a noncanonical selected method set")
    actual_method_tiers = {
        (value.get("method_id"), value.get("resource_tier"))
        for value in screen.get("method_status", [])
    }
    expected_method_tiers = {(method, selected_tier) for method in eligible}
    _validate_stage_report_shape(
        screen, "screen",
        [(method, selected_tier) for method in eligible],
        _stage_cells(config, "screen"), registry, config,
        screen["source_commit"],
    )
    if (actual_method_tiers != expected_method_tiers
            or any(value.get("cells_total") != len(_stage_cells(config, "screen"))
                   for value in screen.get("method_status", []))
            or screen.get("raw_count")
            != len(eligible) * len(_stage_cells(config, "screen"))
               * len(INIT_FAMILIES) * TRAJECTORIES_PER_FAMILY):
        raise GlobalConflictError("screen did not run every runtime-eligible candidate")
    statuses = [value for value in screen["method_status"] if value["valid"]]
    hard = [value for value in statuses if value["method_id"] in HARD_METHODS]
    defect = [value for value in statuses if value["method_id"] in DEFECT_METHODS]
    if not hard or not defect:
        raise GlobalConflictError("screen lacks one passing hard-coset and one defect method")
    hard.sort(key=lambda value: (
        value["core_seconds"],
        1 if value["method_id"] == "RC8-QC4" else 0,
        int(value["method_id"][-2:]) if "-J" in value["method_id"] else 0,
        value["method_id"],
    ))
    defect.sort(key=lambda value: (
        -float(value["d0_ess_per_core_second"]),
        int(value["method_id"][2:]), value["method_id"],
    ))
    selected = [hard[0], defect[0]]
    selected_keys = {
        (value["method_id"], value["resource_tier"]) for value in selected
    }
    mutual = [
        value for value in screen.get("comparisons", [])
        if {
            (value["left"]["method_id"], value["left"]["resource_tier"]),
            (value["right"]["method_id"], value["right"]["resource_tier"]),
        } == selected_keys
    ]
    if (len(mutual) != len(_stage_cells(config, "screen"))
            or not all(value.get("valid") for value in mutual)):
        raise GlobalConflictError("selected hard/defect methods disagree on the screen panel")
    identity = {
        "selection_version": GLOBAL_SELECTION_VERSION,
        "status": "FROZEN_DISCOVERY_METHODS",
        "source_commit": screen["source_commit"],
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "screen_report_sha256": sha256_json(screen),
        "runtime_report_sha256": sha256_json(runtime),
        "digest_report_sha256": sha256_json(digest),
        "preflight_report_sha256": sha256_json(preflight),
        "selected": [
            {"method_id": value["method_id"], "resource_tier": value["resource_tier"]}
            for value in selected
        ],
        "panel_sha256": {
            name: config["panels"][name]["ordered_panel_sha256"]
            for name in ("HARD2", "CONF17", "RES6", "GAP8", "SMALL6")
        },
        "uniform_seeds_sha256": config["uniform_seeds_sha256"],
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
    }
    freeze = {**identity, "selection_sha256": sha256_json(identity)}
    output_path = Path(output_path)
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="ascii")) != freeze:
            raise GlobalConflictError("existing global method selection conflicts")
    else:
        atomic_json(output_path, freeze)
    return freeze


def _parse_method_tiers(values):
    result = []
    for value in values:
        if ":" not in value:
            raise ValueError("method tier must have METHOD:TIER form")
        method, tier = value.split(":", 1)
        result.append((method, tier))
    return result


def _cli_verified_evidence(args):
    names = ("run_root", "ownership", "deployment_root", "schedule")
    values = [getattr(args, name, None) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise ValueError(
            "verified analysis requires run root, ownership, deployment, and schedule"
        )
    return {
        "run_root": args.run_root,
        "control_path": args.manifest,
        "ownership_path": args.ownership,
        "deployment_root": args.deployment_root,
        "schedule_path": args.schedule,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write-config")
    write.add_argument("registry")
    write.add_argument("output")

    schedule = subparsers.add_parser("freeze-schedule")
    schedule.add_argument("registry")
    schedule.add_argument("config")
    schedule.add_argument("source_commit")
    schedule.add_argument("archive_sha256")
    schedule.add_argument("source_manifest_sha256")
    schedule.add_argument("output")

    bias = subparsers.add_parser("bias-manifest")
    bias.add_argument("registry")
    bias.add_argument("config")
    bias.add_argument("source_commit")
    bias.add_argument("stage")
    bias.add_argument("output")
    bias.add_argument("--method-tier", action="append", required=True)

    measurement = subparsers.add_parser("measurement-manifest")
    measurement.add_argument("registry")
    measurement.add_argument("config")
    measurement.add_argument("source_commit")
    measurement.add_argument("stage")
    measurement.add_argument("output")
    measurement.add_argument("--method-tier", action="append", required=True)
    measurement.add_argument("--bias-manifest")
    measurement.add_argument("--bias-raw-root")

    ti_manifest = subparsers.add_parser("ti-manifest")
    ti_manifest.add_argument("registry")
    ti_manifest.add_argument("config")
    ti_manifest.add_argument("source_commit")
    ti_manifest.add_argument("output")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("raw_root")
    analyze.add_argument("manifest")
    analyze.add_argument("registry")
    analyze.add_argument("config")
    analyze.add_argument("output")
    analyze.add_argument("--run-root")
    analyze.add_argument("--ownership")
    analyze.add_argument("--deployment-root")
    analyze.add_argument("--schedule")
    analyze.add_argument("--num-workers", type=int, default=1)

    analyze_ti = subparsers.add_parser("analyze-ti")
    analyze_ti.add_argument("raw_root")
    analyze_ti.add_argument("manifest")
    analyze_ti.add_argument("registry")
    analyze_ti.add_argument("config")
    analyze_ti.add_argument("output")
    analyze_ti.add_argument("--run-root")
    analyze_ti.add_argument("--ownership")
    analyze_ti.add_argument("--deployment-root")
    analyze_ti.add_argument("--schedule")

    compare_ti = subparsers.add_parser("compare-ti")
    compare_ti.add_argument("confirmation_report")
    compare_ti.add_argument("ti_report")
    compare_ti.add_argument("registry")
    compare_ti.add_argument("config")
    compare_ti.add_argument("output")

    freeze = subparsers.add_parser("freeze-selection")
    freeze.add_argument("screen_report")
    freeze.add_argument("runtime_report")
    freeze.add_argument("digest_report")
    freeze.add_argument("preflight_report")
    freeze.add_argument("registry")
    freeze.add_argument("config")
    freeze.add_argument("schedule")
    freeze.add_argument("output")

    freeze_plan = subparsers.add_parser("freeze-postselection")
    freeze_plan.add_argument("selection")
    freeze_plan.add_argument("registry")
    freeze_plan.add_argument("config")
    freeze_plan.add_argument("schedule")
    freeze_plan.add_argument("output")

    prepare_controls = subparsers.add_parser("prepare-postselection-controls")
    prepare_controls.add_argument("selection")
    prepare_controls.add_argument("postselection_plan")
    prepare_controls.add_argument("registry")
    prepare_controls.add_argument("config")
    prepare_controls.add_argument("schedule")
    prepare_controls.add_argument("output_dir")
    prepare_controls.add_argument("output_index")

    materialize = subparsers.add_parser("materialize-measurement")
    materialize.add_argument("stage")
    materialize.add_argument("control_index")
    materialize.add_argument("postselection_plan")
    materialize.add_argument("registry")
    materialize.add_argument("config")
    materialize.add_argument("bias_raw_root")
    materialize.add_argument("output")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("selection")
    finalize.add_argument("hard_fresh_report")
    finalize.add_argument("confirmation_report")
    finalize.add_argument("resolution_report")
    finalize.add_argument("ti_report")
    finalize.add_argument("ti_comparison")
    finalize.add_argument("registry")
    finalize.add_argument("config")
    finalize.add_argument("schedule")
    finalize.add_argument("postselection_plan")
    finalize.add_argument("control_index")
    finalize.add_argument("output")

    args = parser.parse_args(argv)
    if args.command == "write-config":
        result = write_default_global_config(args.registry, args.output)
    elif args.command == "freeze-schedule":
        result = freeze_global_schedule(
            args.registry, args.config, args.source_commit,
            args.archive_sha256, args.source_manifest_sha256, args.output,
        )
    elif args.command == "bias-manifest":
        result = build_bias_manifest(
            args.registry, args.config, args.source_commit, args.stage,
            _parse_method_tiers(args.method_tier), args.output,
        )
    elif args.command == "measurement-manifest":
        result = build_measurement_manifest(
            args.registry, args.config, args.source_commit, args.stage,
            _parse_method_tiers(args.method_tier), args.output,
            bias_manifest_path=args.bias_manifest,
            bias_raw_root=args.bias_raw_root,
        )
    elif args.command == "ti-manifest":
        result = build_ti_anchor_manifest(
            args.registry, args.config, args.source_commit, args.output,
        )
    elif args.command == "analyze":
        result = analyze_measurement_stage(
            args.raw_root, args.manifest, args.registry, args.config, args.output,
            verified_evidence=_cli_verified_evidence(args),
            num_workers=args.num_workers,
        )
    elif args.command == "analyze-ti":
        result = analyze_ti_anchor_stage(
            args.raw_root, args.manifest, args.registry, args.config, args.output,
            verified_evidence=_cli_verified_evidence(args),
        )
    elif args.command == "compare-ti":
        result = compare_ti_anchors(
            args.confirmation_report, args.ti_report, args.registry,
            args.config, args.output,
        )
    elif args.command == "freeze-selection":
        result = freeze_method_selection(
            args.screen_report, args.runtime_report, args.digest_report,
            args.preflight_report, args.registry, args.config,
            args.schedule, args.output,
        )
    elif args.command == "freeze-postselection":
        result = freeze_postselection_plan(
            args.selection, args.registry, args.config, args.schedule,
            args.output,
        )
    elif args.command == "prepare-postselection-controls":
        result = prepare_postselection_controls(
            args.selection, args.postselection_plan, args.registry,
            args.config, args.schedule, args.output_dir,
            args.output_index,
        )
    elif args.command == "materialize-measurement":
        result = materialize_postselection_measurement(
            args.stage, args.control_index, args.postselection_plan,
            args.registry, args.config, args.bias_raw_root, args.output,
        )
    else:
        result = combine_global_readiness(
            args.selection, args.hard_fresh_report,
            args.confirmation_report, args.resolution_report,
            args.ti_report, args.ti_comparison, args.registry,
            args.config, args.schedule, args.postselection_plan,
            args.control_index, args.output,
        )
    print(json.dumps({
        "command": args.command,
        "status": result.get("status", "OK"),
        "sha256": sha256_json(result),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
