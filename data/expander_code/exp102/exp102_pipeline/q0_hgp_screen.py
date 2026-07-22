"""Isolated HARD2+EASY3 screen for the collapsed-HGP and MAP samplers.

This module can only produce a diagnostic sampler-pair decision.  Its task,
seed, raw, manifest, and report namespaces are disjoint from every exp102
formal, PT, PA, QC/JB, and defect-trace contract.
"""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from . import global_discovery as _statistics
from .diagnostics import split_rhat
from .exp101_bridge import load_exp101
from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .q0_global import (
    character_d2_estimate,
    frozen_character_set,
    trajectory_mean_and_se,
    uniform_hard_coset_state,
)
from .q0_hgp_collapsed import (
    COLLAPSED_RAW_VERSION,
    CollapsedPowerPtConfig,
    build_classical_coset_mass,
    run_collapsed_power_pt_trajectory,
)
from .q0_map_mixture import (
    MAP_ANCHOR_VERSION,
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_THETA_LOGICAL,
    DEFAULT_THETA_STABILIZER,
    MILP_OPTIONS,
    MAP_METHOD_ID,
    MAP_PROPOSAL_VERSION,
    MAP_RAW_VERSION,
    AffineCoordinateSystem,
    MapAnchorCatalog,
    MapMixtureConfig,
    MapMixtureProposal,
    build_map_mixture_proposal,
    build_milp_map_anchors,
    run_map_mixture_trajectory,
    validate_map_mixture_proposal,
)
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


HGP_SCREEN_VERSION = "exp102.q0_hgp_global.screen.v1"
HGP_SCREEN_CONFIG_VERSION = "exp102.q0_hgp_global.screen.config.v1"
HGP_SCREEN_TASK_VERSION = "exp102.q0_hgp_global.screen.tasks.v1"
HGP_POWER_RAW_VERSION = "exp102.q0_hgp_power.raw.v2"
HGP_MAP_RAW_VERSION = "exp102.q0_map_mixture.raw.v3"
HGP_SCREEN_REPORT_VERSION = "exp102.q0_hgp_global.screen.report.v1"
HGP_SCREEN_MANIFEST_VERSION = "exp102.q0_hgp_global.screen.manifest.v1"
HGP_SCREEN_SEED_ROOT = "q0_hgp_global_screen_v1"
HGP_SCREEN_CHARACTER_ROOT = "q0_hgp_global_screen_characters_v1"
HGP_SCREEN_B_CHARACTER_ROOT = "q0_hgp_global_screen_b_characters_v1"
HGP_SCREEN_HP_TRAJECTORY_ROOT = "q0_hgp_global_screen_hp_trajectory_v1"
HGP_SCREEN_MAP_TRAJECTORY_ROOT = "q0_hgp_global_screen_map_trajectory_v1"
HGP_SCREEN_MAP_ANCHOR_ROOT = "q0_hgp_global_screen_map_anchor_v1"
HGP_SCREEN_IS_ROOT = "q0_hgp_global_screen_is_diagnostic_v1"
HGP_SCREEN_PREFLIGHT_DIGEST_ROOT = (
    "q0_hgp_global_screen_preflight_digest_v1"
)
HGP_SCREEN_RUNTIME_WARMUP_ROOT = "q0_hgp_global_screen_runtime_warmup_v1"
HGP_SCREEN_RUNTIME_TIMED_ROOT = "q0_hgp_global_screen_runtime_timed_v1"
HGP_SCREEN_PREFLIGHT_IS_ROOT = "q0_hgp_global_screen_is_preflight_v1"
HGP_SCREEN_RUNTIME_IS_ROOT = "q0_hgp_global_screen_is_runtime_v1"
HGP_MAP_ARTIFACT_VERSION = "exp102.q0_hgp_global.screen.map_artifact.v1"
HGP_MAP_IS_RAW_VERSION = "exp102.q0_hgp_global.screen.is_diagnostic.v1"
HGP_MAP_IS_SAMPLES = 50_000
HGP_SCREEN_CONFIG_SHA256 = "163a5cc87486beabf453f3d4a57bc63f0c4e0b2f54619c60268ee7f0c9b2a341"
B_CHARACTER_VERSION = "exp102.q0_hgp_b_characters.v1"
B_DENSE_CHARACTER_COUNT = 64
B_MIN_NONDEGENERATE_DENSE = 48

HP_METHODS = ("HP32", "HP64")
SCREEN_METHODS = (*HP_METHODS, MAP_METHOD_ID)
METHOD_PANELS = {
    "HP32": ("HARD2", "EASY3"),
    "HP64": ("HARD2", "EASY3"),
    MAP_METHOD_ID: ("HARD2",),
}
CROSS_MECHANISM_PANELS = ("HARD2",)
INIT_FAMILIES = ("P", "U")
TRAJECTORIES_PER_FAMILY = 16
EXECUTION_NODES = ("nd-2", "nd-3")
ANALYSIS_NODE = "nd-3"
ANALYSIS_CAPACITY = 91
RESOURCE_TIERS = {
    "T1": {"burn": 2048, "measurement": 8192},
    "T2": {"burn": 4096, "measurement": 16384},
    "T3": {"burn": 8192, "measurement": 32768},
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

_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_COMMON_RAW_FIELDS = {
    "raw_version", "sampler_raw_version", "contract_version", "task_json",
    "task_fingerprint", "source_commit", "archive_sha256",
    "source_manifest_sha256", "registry_sha256",
    "hgp_screen_config_sha256", "cell_json", "uniform_seed",
    "syndrome_packed", "syndrome_sha256", "model_fingerprint",
    "section_fingerprint", "logical_frame_fingerprint", "character_masks",
    "character_sha256", "num_qubits", "k", "trajectory_digest",
    "b_character_masks_packed", "b_character_sha256", "b_character_count",
    "b_dimension", "b_dense_character_count",
    "core_seconds", "wall_seconds",
}

_MAP_ARTIFACT_RAW_FIELDS = {
    "map_artifact_descriptor_json", "map_artifact_file_sha256",
    "map_artifact_content_sha256",
}


class HgpScreenConflictError(ValueError):
    pass


@dataclass(frozen=True)
class BCharacterSet:
    masks_packed: np.ndarray
    r: int
    dense_count: int
    random_seed: int
    character_sha256: str

    @property
    def dimension(self):
        return int(self.r) * int(self.r)

    @property
    def single_count(self):
        return self.dimension

    @property
    def row_column_count(self):
        return 2 * int(self.r)

    @property
    def dense_start(self):
        return self.single_count + self.row_column_count

    @property
    def size(self):
        return int(self.masks_packed.shape[0])


_BYTE_PARITY = (
    np.bitwise_count(np.arange(256, dtype=np.uint8)) & np.uint8(1)
).astype(np.uint8)


def _b_character_digest(masks_packed, r, dense_count, seed):
    metadata = canonical_json({
        "version": B_CHARACTER_VERSION,
        "r": int(r),
        "dense_count": int(dense_count),
        "random_seed": int(seed),
        "ordering": "row_major_singles_rows_columns_dense_v1",
    }).encode("ascii")
    masks = np.ascontiguousarray(masks_packed, dtype=np.uint8)
    return hashlib.sha256(
        metadata + b"\0" + np.asarray(masks.shape, dtype=">u8").tobytes()
        + masks.tobytes(order="C")
    ).hexdigest()


def frozen_b_character_set(r, seed, dense_count=B_DENSE_CHARACTER_COUNT):
    """Freeze B-bit, row/column-parity, and dense diagnostic characters."""
    r = int(r)
    dense_count = int(dense_count)
    if r < 2 or dense_count < B_DENSE_CHARACTER_COUNT:
        raise ValueError("B character set requires r>=2 and at least 64 dense masks")
    dimension = r * r
    width = (dimension + 7) // 8
    masks = []
    seen = set()

    def add_positions(positions):
        packed = np.zeros(width, dtype=np.uint8)
        for position in positions:
            packed[int(position) // 8] |= np.uint8(1 << (int(position) & 7))
        key = packed.tobytes()
        if key in seen:
            raise HgpScreenConflictError("B character construction produced a duplicate mask")
        seen.add(key)
        masks.append(packed)

    for position in range(dimension):
        add_positions((position,))
    for row in range(r):
        add_positions(range(row * r, (row + 1) * r))
    for column in range(r):
        add_positions(range(column, dimension, r))

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    dense = []
    minimum_weight = math.ceil(dimension / 3)
    maximum_weight = math.floor(2 * dimension / 3)
    last_mask = np.uint8((1 << (dimension & 7)) - 1) if dimension & 7 else np.uint8(255)
    while len(dense) < dense_count:
        candidate = np.empty(width, dtype=np.uint8)
        for start in range(0, width, 8):
            word = int(rng.next_uint64()).to_bytes(8, "little")
            stop = min(start + 8, width)
            candidate[start:stop] = np.frombuffer(word[:stop - start], dtype=np.uint8)
        candidate[-1] &= last_mask
        weight = int(np.bitwise_count(candidate).sum())
        key = candidate.tobytes()
        if minimum_weight <= weight <= maximum_weight and key not in seen:
            seen.add(key)
            dense.append(candidate.copy())
    masks.extend(sorted(dense, key=lambda value: value.tobytes()))
    packed = np.ascontiguousarray(np.stack(masks), dtype=np.uint8)
    digest = _b_character_digest(packed, r, dense_count, seed)
    return BCharacterSet(packed, r, dense_count, int(seed), digest)


def _extract_b_states_packed(full_states_packed, n, r):
    """Extract row-major B bytes from packed HGP states without pickle/object data."""
    full = np.asarray(full_states_packed, dtype=np.uint8)
    scalar = full.ndim == 1
    if scalar:
        full = full[None, :]
    total = int(n) * int(n) + int(r) * int(r)
    if full.ndim != 2 or full.shape[1] != (total + 7) // 8:
        raise HgpScreenConflictError("packed HGP state width changed")
    bits = np.unpackbits(full, axis=1, count=total, bitorder="little")
    result = np.packbits(bits[:, int(n) * int(n):], axis=1, bitorder="little")
    return result[0] if scalar else result


def _b_weights(b_states_packed):
    states = np.asarray(b_states_packed, dtype=np.uint8)
    if states.ndim != 2:
        raise HgpScreenConflictError("packed B trace must be two-dimensional")
    return np.bitwise_count(states).sum(axis=1).astype(np.int32)


def _b_log_likelihood(b_states_packed, H, syndrome, log_mass):
    """Recompute L(B)=sum_j log Pr_p[H A_j=(Y+B H)_j]."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    states = np.asarray(b_states_packed, dtype=np.uint8)
    bits = np.unpackbits(states, axis=1, count=r * r, bitorder="little")
    B = bits.reshape(-1, r, r).astype(np.int64, copy=False)
    bh = np.einsum("tij,jk->tik", B, H.astype(np.int64), optimize=False) & 1
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    a_syndromes = bh ^ Y[None, :, :]
    powers = np.left_shift(np.int64(1), np.arange(r, dtype=np.int64))
    indices = np.einsum(
        "trn,r->tn", a_syndromes.astype(np.int64, copy=False), powers,
        optimize=False,
    )
    result = np.zeros(states.shape[0], dtype=np.float64)
    for factor in range(n):
        result += np.asarray(log_mass, dtype=np.float64)[indices[:, factor]]
    return result


def _b_character_bits(b_states_packed, masks_packed):
    states = np.asarray(b_states_packed, dtype=np.uint8)
    masks = np.asarray(masks_packed, dtype=np.uint8)
    if states.ndim != 2 or masks.ndim != 2 or states.shape[1] != masks.shape[1]:
        raise HgpScreenConflictError("packed B character dimensions changed")
    result = np.zeros((states.shape[0], masks.shape[0]), dtype=np.uint8)
    for byte in range(states.shape[1]):
        result ^= _BYTE_PARITY[
            states[:, byte, None] & masks[None, :, byte]
        ]
    return result


def _b_single_bits(b_states_packed, positions):
    states = np.asarray(b_states_packed, dtype=np.uint8)
    positions = np.asarray(positions, dtype=np.int64)
    return (
        (states[:, positions // 8] >> (positions & 7)[None, :])
        & np.uint8(1)
    ).astype(np.uint8, copy=False)


def _split_rhat_columns(chains):
    """Vectorized split-Rhat for (chain,time,observable) arrays."""
    values = np.asarray(chains, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("B diagnostic chains must be chain x time x observable")
    half = values.shape[1] // 2
    if half < 2:
        return np.full(values.shape[2], np.inf, dtype=np.float64)
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    result = np.empty(values.shape[2], dtype=np.float64)
    nonzero = within > 0.0
    result[nonzero] = np.sqrt(
        ((half - 1) / half * within[nonzero] + between[nonzero] / half)
        / within[nonzero]
    )
    for index in np.flatnonzero(~nonzero):
        result[index] = (
            1.0 if np.all(split[:, :, index] == split[0, 0, index]) else np.inf
        )
    return result


def _require_source_commit(value):
    if _FULL_SHA_RE.fullmatch(str(value)) is None:
        raise ValueError("HGP screen source commit must be a full lowercase SHA")


def _require_sha256(name, value):
    if _SHA256_RE.fullmatch(str(value)) is None:
        raise ValueError(f"HGP screen {name} must be a lowercase SHA256")


def _cell_fingerprint(cell):
    return sha256_json(cell)


def _screen_cells(config):
    return [
        *config["panels"]["HARD2"]["cells"],
        *config["panels"]["EASY3"]["cells"],
    ]


def _panel_cells(config, panel_names):
    return [
        cell
        for panel_name in panel_names
        for cell in config["panels"][panel_name]["cells"]
    ]


def _method_cells(config, method):
    if method not in SCREEN_METHODS:
        raise ValueError("unknown HGP screen method")
    panels = tuple(config["method_panels"][method])
    if panels != METHOD_PANELS[method]:
        raise ValueError("HGP screen method-specific panel changed")
    return _panel_cells(config, panels)


def _map_cells(config):
    return _method_cells(config, MAP_METHOD_ID)


def _cross_mechanism_cells(config):
    panels = tuple(config["selection"]["cross_mechanism_panels"])
    if panels != CROSS_MECHANISM_PANELS:
        raise ValueError("HGP screen comparison panel changed")
    return _panel_cells(config, panels)


def _method_schedule_identity(config):
    return {
        "method_panels": config["method_panels"],
        "ordered_panel_sha256": {
            name: config["panels"][name]["ordered_panel_sha256"]
            for name in ("HARD2", "EASY3")
        },
        "cross_mechanism_panels": config["selection"][
            "cross_mechanism_panels"
        ],
        "importance_sampling_panels": config["importance_sampling"]["panels"],
        "task_counts": config["task_counts"],
    }


def _map_requested_max_anchors(config, cell):
    """Resolve a preregistered per-cell cap, with legacy global-8 failover."""
    cell = _validate_cell(cell)
    map_spec = config["map_method"]
    by_cell = map_spec.get("requested_max_anchors_by_cell")
    if by_cell is not None:
        expected_keys = {value["code_id"] for value in _map_cells(config)}
        if set(by_cell) != expected_keys:
            raise ValueError("HGP screen per-cell MAP anchor caps changed")
        value = by_cell[cell["code_id"]]
    else:
        value = map_spec.get("max_anchors")
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= 8):
        raise ValueError("HGP screen MAP anchor cap is invalid")
    return int(value)


def _uniform_seed_for_cell(registry, code, cell):
    if cell["disorder_source"] == "attempt022":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt22"
    elif cell["disorder_source"] == "global_fresh_v1":
        namespace = "q0_global_discovery_fresh_v1"
    else:
        raise ValueError("unknown HGP screen disorder source")
    return derive_seed(
        namespace, registry["registry_sha256"], code["code_id"],
        int(cell["disorder_index"]), "uniforms",
    )


def _registry_with_path(registry, path):
    result = dict(registry)
    result["_registry_path"] = str(Path(path).resolve())
    return result


def _validate_cell(cell):
    if not isinstance(cell, dict) or set(cell) != {
            "code_id", "p", "disorder_index", "disorder_source"}:
        raise ValueError("HGP screen cell schema mismatch")
    if (not isinstance(cell["code_id"], str)
            or not 0.0 < float(cell["p"]) < 0.5
            or isinstance(cell["disorder_index"], bool)
            or int(cell["disorder_index"]) < 0):
        raise ValueError("HGP screen cell value is invalid")
    return {
        "code_id": cell["code_id"], "p": float(cell["p"]),
        "disorder_index": int(cell["disorder_index"]),
        "disorder_source": str(cell["disorder_source"]),
    }


def load_hgp_screen_config(path, registry=None):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    raw = json.loads(serialized)
    if serialized != canonical_json(raw) + "\n":
        raise ValueError("HGP screen config is not canonical JSON")
    if sha256_file(path) != HGP_SCREEN_CONFIG_SHA256:
        raise ValueError("HGP screen canonical config SHA changed")
    execution = raw.get("execution", {})
    if (raw.get("contract_version") != HGP_SCREEN_VERSION
            or raw.get("config_version") != HGP_SCREEN_CONFIG_VERSION
            or raw.get("raw_versions", {}).get("tasks")
            != HGP_SCREEN_TASK_VERSION
            or raw.get("raw_versions", {}).get("hp") != HGP_POWER_RAW_VERSION
            or raw.get("raw_versions", {}).get("map") != HGP_MAP_RAW_VERSION
            or raw.get("raw_versions", {}).get("anchor_catalog")
            != MAP_ANCHOR_VERSION
            or raw.get("raw_versions", {}).get("proposal")
            != MAP_PROPOSAL_VERSION
            or raw.get("raw_versions", {}).get("importance_sampling")
            != HGP_MAP_IS_RAW_VERSION
            or tuple(raw.get("init_families", ())) != INIT_FAMILIES
            or int(raw.get("trajectory_count_per_init_family", -1))
            != TRAJECTORIES_PER_FAMILY
            or tuple(execution.get("execution_nodes", ()))
            != EXECUTION_NODES
            or execution.get("analysis") != {
                "node": ANALYSIS_NODE,
                "capacity": ANALYSIS_CAPACITY,
                "num_workers": ANALYSIS_CAPACITY,
            }):
        raise ValueError("HGP screen config identity changed")
    methods = (
        *(value.get("method_id") for value in raw.get("hp_methods", ())),
        raw.get("map_method", {}).get("method_id"),
    )
    if methods != SCREEN_METHODS:
        raise ValueError("HGP screen method order changed")
    if raw.get("method_panels") != {
            method: list(METHOD_PANELS[method]) for method in SCREEN_METHODS}:
        raise ValueError("HGP screen method-specific panels changed")
    for entry, method in zip(raw["hp_methods"], HP_METHODS):
        expected_lambdas = CollapsedPowerPtConfig(method, 0.10, 8, 8).lambda_values
        expected_lambda_sha = hashlib.sha256(
            np.asarray(expected_lambdas, dtype=">f8").tobytes()
        ).hexdigest()
        if (entry.get("num_replicas") != int(method[2:])
                or entry.get("block_size") != 8
                or entry.get("lambda_generation") != "quadratic_index_v1"
                or entry.get("lambda_sha256") != expected_lambda_sha
                or entry.get("tempered_term")
                != "collapsed_syndrome_log_likelihood_only"):
            raise ValueError("HGP screen HP protocol changed")
    map_spec = raw["map_method"]
    if (tuple(map_spec.get("theta_stabilizer", ()))
            != DEFAULT_THETA_STABILIZER
            or tuple(map_spec.get("theta_logical", ())) != DEFAULT_THETA_LOGICAL
            or tuple(map_spec.get("component_weights", ()))
            != DEFAULT_COMPONENT_WEIGHTS
            or int(map_spec.get("max_anchors", -1)) != 8
            or map_spec.get("solver_options") != dict(MILP_OPTIONS)
            or map_spec.get("proposal_full_support_required") is not True):
        raise ValueError("HGP screen MAP protocol changed")
    importance = raw.get("importance_sampling", {})
    if (importance.get("num_samples_per_cell") != HGP_MAP_IS_SAMPLES
            or importance.get("used_for_gate_or_selection") is not False
            or tuple(importance.get("panels", ())) != ("HARD2",)
            or importance.get("role")
            != "auxiliary_proposal_overlap_diagnostic_only"):
        raise ValueError("HGP screen importance-sampling protocol changed")
    if raw.get("b_character_spec") != {
            "version": B_CHARACTER_VERSION,
            "dense_count": B_DENSE_CHARACTER_COUNT,
            "dense_generation": "portable_prng_weight_thirds_v1",
            "minimum_nondegenerate_dense": B_MIN_NONDEGENERATE_DENSE,
            "single_diagnostic": "split_rhat_only",
            "row_column_dense_diagnostic": "split_rhat_and_bulk_ess",
    }:
        raise ValueError("HGP screen B-character protocol changed")
    if raw.get("resource_tiers") != RESOURCE_TIERS:
        raise ValueError("HGP screen resource tiers changed")
    resource_selection = raw.get("resource_selection", {})
    if (resource_selection.get("staging_validation_mode") != "structure_only"
            or resource_selection.get("final_analysis_validation_mode")
            != "full_bit_exact_replay"
            or resource_selection.get("full_sampler_passes_per_task") != 2
            or resource_selection.get("full_is_passes_per_cell") != 2):
        raise ValueError("HGP screen replay/runtime accounting changed")
    hard = [_validate_cell(value) for value in raw["panels"]["HARD2"]["cells"]]
    easy = [_validate_cell(value) for value in raw["panels"]["EASY3"]["cells"]]
    if (hard != list(HARD_CELLS) or easy != list(EASY_CELLS)
            or sha256_json(hard) != HARD_PANEL_SHA256
            or sha256_json(easy) != EASY_PANEL_SHA256
            or raw["panels"]["HARD2"]["ordered_panel_sha256"]
            != HARD_PANEL_SHA256
            or raw["panels"]["EASY3"]["ordered_panel_sha256"]
            != EASY_PANEL_SHA256):
        raise ValueError("HGP screen panel changed")
    for cell in _map_cells(raw):
        _map_requested_max_anchors(raw, cell)
    scope = raw.get("scope", {})
    if (scope.get("maximum_terminal_status")
            != "DIAGNOSTIC_HARD_PAIR_FOUND"
            or scope.get("formal_authorization") is not False
            or scope.get("production_authorization") is not False
            or (scope.get("held_out_in_scope") is not False
                and "held_out" not in scope.get("excluded_work", ()))):
        raise ValueError("HGP screen scope exceeds diagnostic authority")
    if (raw.get("task_counts") != {
            "hp_measurement": 320,
            "map_measurement": 64,
            "total_measurement": 384,
        } or tuple(raw.get("selection", {}).get(
            "cross_mechanism_panels", ())) != CROSS_MECHANISM_PANELS):
        raise ValueError("HGP screen task/comparison scope changed")
    if raw.get("screen_panel_sha256") != sha256_json(
            _method_schedule_identity(raw)):
        raise ValueError("HGP screen method schedule SHA changed")
    nested_gates = raw.get("gates", {})
    gates = dict(nested_gates.get("common", {}))
    hp_gates = nested_gates.get("hp_per_trajectory", {})
    map_gates = nested_gates.get("map_per_trajectory", {})
    gates.update({
        "diagnostic_nonbasis_characters": raw.get(
            "character_spec", {},
        ).get("diagnostic_nonbasis_count"),
        "hp_min_edge_swap_rate": hp_gates.get("min_adjacent_swap_rate"),
        "hp_min_edge_swap_accepts": hp_gates.get("min_adjacent_swap_accepts"),
        "hp_min_round_trips_per_trajectory": hp_gates.get(
            "min_cold_hot_cold_round_trips",
        ),
        "hp_min_cold_origin_fraction": hp_gates.get(
            "min_cold_origin_fraction",
        ),
        "map_min_burn_state_changes": map_gates.get(
            "min_burn_state_changes",
        ),
        "map_min_measurement_state_change_rate": map_gates.get(
            "min_measurement_state_change_rate",
        ),
        "map_min_measurement_state_changes": map_gates.get(
            "min_measurement_state_changes",
        ),
    })
    required_gates = {
        "delta_sigma_multiplier", "delta_sigma_slack",
        "diagnostic_nonbasis_characters", "max_abs_delta_q_top",
        "max_d2_upper", "max_normalized_weight_delta", "max_q_top_se",
        "max_rhat", "min_bulk_ess", "hp_min_edge_swap_rate",
        "hp_min_edge_swap_accepts",
        "hp_min_round_trips_per_trajectory", "hp_min_cold_origin_fraction",
        "map_min_burn_state_changes",
        "map_min_measurement_state_change_rate",
        "map_min_measurement_state_changes",
        "max_b_character_d2_upper", "max_b_normalized_weight_delta",
        "max_b_log_likelihood_delta_per_factor",
        "max_abs_b_character_mean_delta", "b_character_delta_sigma_slack",
    }
    if not required_gates.issubset(gates):
        raise ValueError("HGP screen statistical gates are incomplete")
    if (any(gates[name] is None or not math.isfinite(float(gates[name]))
            for name in required_gates)
            or gates["hp_min_edge_swap_rate"] < 0.0
            or gates["map_min_measurement_state_change_rate"] < 0.0):
        raise ValueError("HGP screen statistical gate value is invalid")
    namespaces = raw.get("seed_namespaces", {})
    if (namespaces.get("root") != HGP_SCREEN_SEED_ROOT
            or namespaces.get("characters") != HGP_SCREEN_CHARACTER_ROOT
            or namespaces.get("b_characters")
            != HGP_SCREEN_B_CHARACTER_ROOT
            or namespaces.get("hp_trajectory")
            != HGP_SCREEN_HP_TRAJECTORY_ROOT
            or namespaces.get("map_trajectory")
            != HGP_SCREEN_MAP_TRAJECTORY_ROOT
            or namespaces.get("map_anchor") != HGP_SCREEN_MAP_ANCHOR_ROOT
            or namespaces.get("importance_sampling") != HGP_SCREEN_IS_ROOT
            or namespaces.get("preflight_digest")
            != HGP_SCREEN_PREFLIGHT_DIGEST_ROOT
            or namespaces.get("runtime_warmup")
            != HGP_SCREEN_RUNTIME_WARMUP_ROOT
            or namespaces.get("runtime_timed")
            != HGP_SCREEN_RUNTIME_TIMED_ROOT
            or namespaces.get("importance_sampling_preflight")
            != HGP_SCREEN_PREFLIGHT_IS_ROOT
            or namespaces.get("importance_sampling_runtime")
            != HGP_SCREEN_RUNTIME_IS_ROOT):
        raise ValueError("HGP screen seed namespace changed")
    if registry is None:
        registry = load_registry(path.parents[1] / "registry/registry.json")
    if raw.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("HGP screen registry SHA changed")
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    expected_uniforms = [
        {
            "cell_fingerprint": _cell_fingerprint(cell),
            "uniform_seed": _uniform_seed_for_cell(
                registry, code_by_id[cell["code_id"]], cell,
            ),
        }
        for cell in [*hard, *easy]
    ]
    if (raw.get("uniform_seeds") != expected_uniforms
            or raw.get("uniform_seeds_sha256") != sha256_json(expected_uniforms)):
        raise ValueError("HGP screen frozen disorder seeds changed")
    result = dict(raw)
    result["gates"] = gates
    result["characters"] = {
        "num_nonbasis": int(raw["character_spec"]["frozen_nonbasis_count"]),
    }
    result["hgp_screen_config_sha256"] = sha256_file(path)
    result["config_path"] = str(path.resolve())
    return result


@dataclass(frozen=True)
class HgpScreenSeedIdentity:
    source_commit: str
    archive_sha256: str
    source_manifest_sha256: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    resource_tier: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str = HGP_SCREEN_HP_TRAJECTORY_ROOT

    def __post_init__(self):
        _require_source_commit(self.source_commit)
        for name in (
            "archive_sha256", "source_manifest_sha256", "config_sha256",
            "registry_sha256", "cell_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))
        if self.method_id not in SCREEN_METHODS or self.resource_tier not in RESOURCE_TIERS:
            raise ValueError("HGP screen seed method/tier changed")
        if self.init_family not in INIT_FAMILIES:
            raise ValueError("HGP screen seed family changed")
        if (isinstance(self.trajectory_index, bool)
                or not isinstance(self.trajectory_index, (int, np.integer))
                or not 0 <= int(self.trajectory_index) < TRAJECTORIES_PER_FAMILY):
            raise ValueError("HGP screen seed trajectory changed")
        formal_namespace = (
            HGP_SCREEN_HP_TRAJECTORY_ROOT if self.method_id in HP_METHODS
            else HGP_SCREEN_MAP_TRAJECTORY_ROOT
        )
        allowed_namespaces = {
            formal_namespace, HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
            HGP_SCREEN_RUNTIME_WARMUP_ROOT, HGP_SCREEN_RUNTIME_TIMED_ROOT,
        }
        if self.trajectory_namespace not in allowed_namespaces:
            raise ValueError("HGP screen seed namespace changed")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            HGP_SCREEN_SEED_ROOT, self.source_commit, self.archive_sha256,
            self.source_manifest_sha256, self.config_sha256,
            self.registry_sha256, self.cell_fingerprint, self.method_id,
            self.resource_tier, self.init_family, int(self.trajectory_index),
            self.trajectory_namespace, str(stage), str(role), int(index),
        )

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "archive_sha256": self.archive_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": self.method_id,
            "resource_tier": self.resource_tier,
            "init_family": self.init_family,
            "trajectory_index": int(self.trajectory_index),
            "trajectory_namespace": self.trajectory_namespace,
        }


def _seed_identity(config, registry, source_commit, archive_sha256,
                   source_manifest_sha256, method, tier, cell, family,
                   trajectory):
    return HgpScreenSeedIdentity(
        source_commit=source_commit,
        archive_sha256=archive_sha256,
        source_manifest_sha256=source_manifest_sha256,
        config_sha256=config["hgp_screen_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=_cell_fingerprint(cell), method_id=method,
        resource_tier=tier, init_family=family,
        trajectory_index=int(trajectory),
        trajectory_namespace=(
            HGP_SCREEN_HP_TRAJECTORY_ROOT if method in HP_METHODS
            else HGP_SCREEN_MAP_TRAJECTORY_ROOT
        ),
    )


def _aux_seed_identity(config, registry, source_commit, archive_sha256,
                       source_manifest_sha256, method, tier, cell, family,
                       trajectory, namespace):
    if namespace not in {
            HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
            HGP_SCREEN_RUNTIME_WARMUP_ROOT,
            HGP_SCREEN_RUNTIME_TIMED_ROOT}:
        raise ValueError("HGP auxiliary trajectory namespace changed")
    return HgpScreenSeedIdentity(
        source_commit=source_commit,
        archive_sha256=archive_sha256,
        source_manifest_sha256=source_manifest_sha256,
        config_sha256=config["hgp_screen_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=_cell_fingerprint(cell), method_id=method,
        resource_tier=tier, init_family=family,
        trajectory_index=int(trajectory), trajectory_namespace=namespace,
    )


def _task_identity(config, registry, source_commit, archive_sha256,
                   source_manifest_sha256, method, tier, cell, family,
                   trajectory, map_artifact_descriptor=None):
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    cell = _validate_cell(cell)
    seed = _seed_identity(
        config, registry, source_commit, archive_sha256,
        source_manifest_sha256, method, tier, cell, family, trajectory,
    )
    task = {
        "contract_version": HGP_SCREEN_VERSION,
        "task_version": HGP_SCREEN_TASK_VERSION,
        "stage": "screen",
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "method_id": method, "resource_tier": tier, "cell": cell,
        "init_family": family, "trajectory_index": int(trajectory),
        "seed_identity": seed.as_dict(),
    }
    if method == MAP_METHOD_ID:
        if not isinstance(map_artifact_descriptor, dict):
            raise ValueError("MAP task requires a frozen artifact descriptor")
        _validate_map_artifact_descriptor(
            map_artifact_descriptor, source_commit=source_commit,
            archive_sha256=archive_sha256,
            source_manifest_sha256=source_manifest_sha256, config=config,
            registry=registry, cell=cell,
        )
        task["map_artifact"] = dict(map_artifact_descriptor)
    elif map_artifact_descriptor is not None:
        raise ValueError("HP task cannot bind a MAP artifact")
    return task


def _task_output_relpath(task):
    return f"trajectories/{sha256_json(task)}.npz"


def _map_artifact_relpath(cell):
    return f"map_artifacts/{_cell_fingerprint(_validate_cell(cell))}.npz"


def _classical_matrix_sha256(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    return hashlib.sha256(
        np.asarray(H.shape, dtype=">u8").tobytes()
        + np.packbits(H, axis=1, bitorder="little").tobytes()
    ).hexdigest()


def _artifact_content_sha256(metadata, arrays, *, version=HGP_MAP_ARTIFACT_VERSION):
    digest = hashlib.sha256(str(version).encode("ascii") + b"\0")
    digest.update(canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        if value.dtype.hasobject:
            raise HgpScreenConflictError("MAP artifact cannot contain objects")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class HgpMapArtifact:
    descriptor: dict
    catalog: MapAnchorCatalog
    proposal: MapMixtureProposal


def _map_artifact_arrays(catalog, proposal, syndrome):
    coordinates = proposal.coordinates
    return {
        "syndrome_packed": np.packbits(syndrome, bitorder="little"),
        "anchors": np.asarray(catalog.anchors, dtype=np.uint8),
        "coordinate_H_check": np.asarray(coordinates.H_check, dtype=np.uint8),
        "coordinate_reference_anchor": np.asarray(
            coordinates.reference_anchor, dtype=np.uint8,
        ),
        "coordinate_basis": np.asarray(coordinates.basis, dtype=np.uint8),
        "coordinate_pivot_columns": np.asarray(
            coordinates.pivot_columns, dtype=np.int32,
        ),
        "coordinate_pivot_inverse": np.asarray(
            coordinates.pivot_inverse, dtype=np.uint8,
        ),
        "coordinate_packed_reference": np.asarray(
            coordinates.packed_reference, dtype=np.uint8,
        ),
        "coordinate_packed_basis": np.asarray(
            coordinates.packed_basis, dtype=np.uint8,
        ),
        "proposal_anchor_centers": np.asarray(
            proposal.anchor_centers, dtype=np.uint8,
        ),
        "proposal_anchor_weights": np.asarray(
            proposal.anchor_weights, dtype=np.float64,
        ),
        "proposal_theta_stabilizer": np.asarray(
            proposal.theta_stabilizer, dtype=np.float64,
        ),
        "proposal_theta_logical": np.asarray(
            proposal.theta_logical, dtype=np.float64,
        ),
        "proposal_component_weights": np.asarray(
            proposal.component_weights, dtype=np.float64,
        ),
    }


def _map_artifact_metadata(registry, config, source_commit, archive_sha256,
                           source_manifest_sha256, cell, code, H, model, frame,
                           uniform_seed, syndrome, catalog, proposal,
                           generation_wall_seconds, generation_core_seconds):
    return {
        "artifact_version": HGP_MAP_ARTIFACT_VERSION,
        "contract_version": HGP_SCREEN_VERSION,
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "cell": _validate_cell(cell),
        "cell_fingerprint": _cell_fingerprint(_validate_cell(cell)),
        "uniform_seed": int(uniform_seed),
        "syndrome_sha256": _syndrome_sha256(syndrome),
        "classical_matrix_sha256": _classical_matrix_sha256(H),
        "model_fingerprint": model.fingerprint(),
        "frame_fingerprint": frame.fingerprint(),
        "section_fingerprint": code["section_fingerprint"],
        "logical_frame_fingerprint": code["logical_frame_fingerprint"],
        "num_qubits": int(model.num_qubits),
        "num_checks": int(model.num_checks),
        "k": int(model.k),
        "generation_wall_seconds": float(generation_wall_seconds),
        "generation_core_seconds": float(generation_core_seconds),
        "anchor_version": MAP_ANCHOR_VERSION,
        "anchor_sha256": catalog.anchor_sha256,
        "anchor_state_sha256": list(catalog.anchor_state_sha256),
        "anchor_objective_sha256": list(catalog.objective_sha256),
        "anchor_tie_break_seeds": [int(value) for value in catalog.tie_break_seeds],
        "anchor_optimum_weight": int(catalog.optimum_weight),
        "anchor_count": int(catalog.size),
        "anchor_requested_max": int(catalog.requested_max_anchors),
        "anchor_p": float(catalog.p),
        "anchor_solver_identity": catalog.solver_identity,
        "anchor_solver_options": [list(value) for value in catalog.solver_options],
        "anchor_solver": catalog.solver,
        "anchor_optimality_evidence": catalog.optimality_evidence,
        "anchor_seed_namespace": catalog.seed_namespace,
        "coordinate_sha256": proposal.coordinates.coordinate_sha256,
        "coordinate_stabilizer_dimension": int(
            proposal.coordinates.stabilizer_dimension,
        ),
        "coordinate_logical_dimension": int(
            proposal.coordinates.logical_dimension,
        ),
        "proposal_version": MAP_PROPOSAL_VERSION,
        "proposal_sha256": proposal.proposal_sha256,
    }


def _map_artifact_descriptor(metadata, content_sha256, file_sha256):
    return {
        "artifact_version": HGP_MAP_ARTIFACT_VERSION,
        "artifact_relpath": _map_artifact_relpath(metadata["cell"]),
        "artifact_file_sha256": file_sha256,
        "artifact_content_sha256": content_sha256,
        "source_commit": metadata["source_commit"],
        "archive_sha256": metadata["archive_sha256"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "registry_sha256": metadata["registry_sha256"],
        "hgp_screen_config_sha256": metadata["hgp_screen_config_sha256"],
        "cell_fingerprint": metadata["cell_fingerprint"],
        "model_fingerprint": metadata["model_fingerprint"],
        "syndrome_sha256": metadata["syndrome_sha256"],
        "generation_wall_seconds": metadata["generation_wall_seconds"],
        "generation_core_seconds": metadata["generation_core_seconds"],
        "requested_max_anchors": metadata["anchor_requested_max"],
        "anchor_count": metadata["anchor_count"],
        "anchor_sha256": metadata["anchor_sha256"],
        "anchor_solver_identity": metadata["anchor_solver_identity"],
        "coordinate_sha256": metadata["coordinate_sha256"],
        "proposal_sha256": metadata["proposal_sha256"],
    }


def _validate_map_artifact_descriptor(descriptor, *, source_commit,
                                      archive_sha256,
                                      source_manifest_sha256, config,
                                      registry, cell):
    required = {
        "artifact_version", "artifact_relpath", "artifact_file_sha256",
        "artifact_content_sha256", "source_commit", "archive_sha256",
        "source_manifest_sha256", "registry_sha256",
        "hgp_screen_config_sha256", "cell_fingerprint",
        "model_fingerprint", "syndrome_sha256", "requested_max_anchors",
        "anchor_count", "anchor_sha256", "anchor_solver_identity",
        "coordinate_sha256", "generation_wall_seconds",
        "generation_core_seconds",
        "proposal_sha256",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != required:
        raise HgpScreenConflictError("MAP artifact descriptor schema changed")
    cell = _validate_cell(cell)
    expected_scalars = {
        "artifact_version": HGP_MAP_ARTIFACT_VERSION,
        "artifact_relpath": _map_artifact_relpath(cell),
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "cell_fingerprint": _cell_fingerprint(cell),
        "requested_max_anchors": _map_requested_max_anchors(config, cell),
    }
    if any(descriptor[name] != expected for name, expected in expected_scalars.items()):
        raise HgpScreenConflictError("MAP artifact descriptor identity changed")
    for name in (
            "artifact_file_sha256", "artifact_content_sha256",
            "model_fingerprint", "syndrome_sha256", "anchor_sha256",
            "coordinate_sha256", "proposal_sha256"):
        _require_sha256(name, descriptor[name])
    if (isinstance(descriptor["anchor_count"], bool)
            or not isinstance(descriptor["anchor_count"], int)
            or not 1 <= descriptor["anchor_count"]
            <= descriptor["requested_max_anchors"]):
        raise HgpScreenConflictError("MAP artifact anchor count changed")
    if (not isinstance(descriptor["anchor_solver_identity"], str)
            or not descriptor["anchor_solver_identity"]
            or "highs=unknown" in descriptor["anchor_solver_identity"]):
        raise HgpScreenConflictError("MAP artifact solver identity changed")
    for name in ("generation_wall_seconds", "generation_core_seconds"):
        value = descriptor[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise HgpScreenConflictError("MAP artifact generation timing changed")
    return True


def build_hgp_map_artifact(registry_path, config_path, source_commit,
                           archive_sha256, source_manifest_sha256, cell,
                           artifact_root):
    """Construct exactly one immutable MAP catalog/proposal artifact per cell."""
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    cell = _validate_cell(cell)
    if cell not in _map_cells(config):
        raise ValueError("MAP artifact cell is outside the frozen screen panel")
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed, _, syndrome = _disorder(registry, code, model, cell)
    requested_max_anchors = _map_requested_max_anchors(config, cell)
    generation_wall_start = time.monotonic()
    generation_core_start = time.process_time()
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, cell["p"],
        max_anchors=requested_max_anchors,
    )
    proposal = build_map_mixture_proposal(model, catalog)
    validate_map_mixture_proposal(
        model, syndrome, cell["p"], catalog, proposal,
        requested_max_anchors=requested_max_anchors,
    )
    generation_wall_seconds = time.monotonic() - generation_wall_start
    generation_core_seconds = time.process_time() - generation_core_start
    metadata = _map_artifact_metadata(
        registry, config, source_commit, archive_sha256,
        source_manifest_sha256, cell, code, H, model, frame, uniform_seed,
        syndrome, catalog, proposal, generation_wall_seconds,
        generation_core_seconds,
    )
    arrays = _map_artifact_arrays(catalog, proposal, syndrome)
    content_sha256 = _artifact_content_sha256(metadata, arrays)
    path = Path(artifact_root) / _map_artifact_relpath(cell)
    if path.exists():
        raise FileExistsError(f"MAP artifact already exists: {path}")
    atomic_npz(
        path, metadata_json=np.array(canonical_json(metadata)),
        artifact_content_sha256=np.array(content_sha256), **arrays,
    )
    descriptor = _map_artifact_descriptor(
        metadata, content_sha256, sha256_file(path),
    )
    _validate_map_artifact_descriptor(
        descriptor, source_commit=source_commit,
        archive_sha256=archive_sha256,
        source_manifest_sha256=source_manifest_sha256, config=config,
        registry=registry, cell=cell,
    )
    return descriptor


def build_hgp_map_artifacts(registry_path, config_path, source_commit,
                            archive_sha256, source_manifest_sha256,
                            artifact_root):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    return [
        build_hgp_map_artifact(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root,
        )
        for cell in _map_cells(config)
    ]


def _reconstruct_map_artifact(metadata, arrays):
    catalog = MapAnchorCatalog(
        anchors=arrays["anchors"],
        optimum_weight=int(metadata["anchor_optimum_weight"]),
        requested_max_anchors=int(metadata["anchor_requested_max"]),
        tie_break_seeds=tuple(metadata["anchor_tie_break_seeds"]),
        p=float(metadata["anchor_p"]),
        anchor_sha256=metadata["anchor_sha256"],
        anchor_state_sha256=tuple(metadata["anchor_state_sha256"]),
        objective_sha256=tuple(metadata["anchor_objective_sha256"]),
        solver_identity=metadata["anchor_solver_identity"],
        solver_options=tuple(
            (str(name), value)
            for name, value in metadata["anchor_solver_options"]
        ),
        solver=metadata["anchor_solver"],
        optimality_evidence=metadata["anchor_optimality_evidence"],
        seed_namespace=metadata["anchor_seed_namespace"],
    )
    coordinates = AffineCoordinateSystem(
        H_check=arrays["coordinate_H_check"],
        reference_anchor=arrays["coordinate_reference_anchor"],
        basis=arrays["coordinate_basis"],
        stabilizer_dimension=int(metadata["coordinate_stabilizer_dimension"]),
        logical_dimension=int(metadata["coordinate_logical_dimension"]),
        pivot_columns=arrays["coordinate_pivot_columns"],
        pivot_inverse=arrays["coordinate_pivot_inverse"],
        packed_reference=arrays["coordinate_packed_reference"],
        packed_basis=arrays["coordinate_packed_basis"],
        coordinate_sha256=metadata["coordinate_sha256"],
    )
    proposal = MapMixtureProposal(
        coordinates=coordinates,
        anchor_catalog=catalog,
        anchor_centers=arrays["proposal_anchor_centers"],
        anchor_weights=arrays["proposal_anchor_weights"],
        theta_stabilizer=arrays["proposal_theta_stabilizer"],
        theta_logical=arrays["proposal_theta_logical"],
        component_weights=arrays["proposal_component_weights"],
        proposal_sha256=metadata["proposal_sha256"],
    )
    return catalog, proposal


def load_hgp_map_artifact(registry_path, config_path, source_commit,
                          archive_sha256, source_manifest_sha256, cell,
                          artifact_root, expected_descriptor=None):
    """Load and independently replay a pickle-free MAP artifact."""
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    cell = _validate_cell(cell)
    if cell not in _map_cells(config):
        raise HgpScreenConflictError("MAP artifact cell is outside the panel")
    path = Path(artifact_root) / _map_artifact_relpath(cell)
    try:
        file_sha256 = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            array_names = {
                "syndrome_packed", "anchors", "coordinate_H_check",
                "coordinate_reference_anchor", "coordinate_basis",
                "coordinate_pivot_columns", "coordinate_pivot_inverse",
                "coordinate_packed_reference", "coordinate_packed_basis",
                "proposal_anchor_centers", "proposal_anchor_weights",
                "proposal_theta_stabilizer", "proposal_theta_logical",
                "proposal_component_weights",
            }
            if set(data.files) != {
                    "metadata_json", "artifact_content_sha256", *array_names}:
                raise HgpScreenConflictError("MAP artifact schema changed")
            metadata_json = str(_scalar(data, "metadata_json"))
            metadata = json.loads(metadata_json)
            if canonical_json(metadata) != metadata_json:
                raise HgpScreenConflictError("MAP artifact metadata is noncanonical")
            arrays = {name: data[name].copy() for name in array_names}
            if any(value.dtype.hasobject for value in arrays.values()):
                raise HgpScreenConflictError("MAP artifact contains an object array")
            content_sha256 = str(_scalar(data, "artifact_content_sha256"))
    except HgpScreenConflictError:
        raise
    except Exception as exc:
        raise HgpScreenConflictError(f"MAP artifact cannot be loaded: {exc}") from exc
    if _artifact_content_sha256(metadata, arrays) != content_sha256:
        raise HgpScreenConflictError("MAP artifact content SHA changed")
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed, _, syndrome = _disorder(registry, code, model, cell)
    catalog, proposal = _reconstruct_map_artifact(metadata, arrays)
    validate_map_mixture_proposal(
        model, syndrome, cell["p"], catalog, proposal,
        requested_max_anchors=_map_requested_max_anchors(config, cell),
        require_current_solver_identity=False,
    )
    expected_metadata = _map_artifact_metadata(
        registry, config, source_commit, archive_sha256,
        source_manifest_sha256, cell, code, H, model, frame, uniform_seed,
        syndrome, catalog, proposal, metadata.get("generation_wall_seconds"),
        metadata.get("generation_core_seconds"),
    )
    if metadata != expected_metadata:
        raise HgpScreenConflictError("MAP artifact identity/content binding changed")
    if not np.array_equal(
            arrays["syndrome_packed"],
            np.packbits(syndrome, bitorder="little")):
        raise HgpScreenConflictError("MAP artifact syndrome bytes changed")
    canonical_arrays = _map_artifact_arrays(catalog, proposal, syndrome)
    if any(not np.array_equal(arrays[name], canonical_arrays[name])
           for name in canonical_arrays):
        raise HgpScreenConflictError("MAP artifact reconstructed arrays changed")
    descriptor = _map_artifact_descriptor(metadata, content_sha256, file_sha256)
    _validate_map_artifact_descriptor(
        descriptor, source_commit=source_commit,
        archive_sha256=archive_sha256,
        source_manifest_sha256=source_manifest_sha256, config=config,
        registry=registry, cell=cell,
    )
    if expected_descriptor is not None and descriptor != expected_descriptor:
        raise HgpScreenConflictError("MAP artifact descriptor changed")
    return HgpMapArtifact(descriptor, catalog, proposal)


def load_hgp_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    return [
        load_hgp_map_artifact(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root,
        ).descriptor
        for cell in _map_cells(config)
    ]


def _map_is_seed(source_commit, archive_sha256, source_manifest_sha256,
                 config, registry, cell, artifact_descriptor,
                 seed_namespace=HGP_SCREEN_IS_ROOT):
    if seed_namespace not in {
            HGP_SCREEN_IS_ROOT, HGP_SCREEN_PREFLIGHT_IS_ROOT,
            HGP_SCREEN_RUNTIME_IS_ROOT}:
        raise ValueError("HGP screen IS seed namespace changed")
    return derive_seed(
        seed_namespace, source_commit, archive_sha256,
        source_manifest_sha256, config["hgp_screen_config_sha256"],
        registry["registry_sha256"], _cell_fingerprint(cell), MAP_METHOD_ID,
        artifact_descriptor["artifact_content_sha256"],
        HGP_MAP_IS_SAMPLES, "iid_proposal_draws",
    )


def _map_is_transcript(proposal, p, num_samples, seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    num_samples = int(num_samples)
    if num_samples != HGP_MAP_IS_SAMPLES:
        raise ValueError("HGP screen IS sample count is not frozen")
    rng = PortablePrng(int(seed))
    state_width = (proposal.coordinates.num_qubits + 7) // 8
    coordinate_width = (proposal.coordinates.dimension + 7) // 8
    arrays = {
        "sample_states_packed": np.empty(
            (num_samples, state_width), dtype=np.uint8,
        ),
        "sample_coordinates_packed": np.empty(
            (num_samples, coordinate_width), dtype=np.uint8,
        ),
        "sample_physical_weights": np.empty(num_samples, dtype=np.int32),
        "sample_log_q": np.empty(num_samples, dtype=np.float64),
        "sample_log_importance_weight": np.empty(
            num_samples, dtype=np.float64,
        ),
        "sample_anchor_index": np.empty(num_samples, dtype=np.int16),
        "sample_component_index": np.empty(num_samples, dtype=np.int8),
    }
    log_b = math.log(float(p) / (1.0 - float(p)))
    for index in range(num_samples):
        draw = proposal.sample(rng)
        state = draw["state"]
        coordinate = draw["coordinate"]
        physical_weight = int(state.sum())
        log_q = float(draw["log_q"])
        log_importance = physical_weight * log_b - log_q
        if not math.isfinite(log_q) or not math.isfinite(log_importance):
            raise HgpScreenConflictError("HGP screen IS produced nonfinite weights")
        arrays["sample_states_packed"][index] = np.packbits(
            state, bitorder="little",
        )
        arrays["sample_coordinates_packed"][index] = np.packbits(
            coordinate, bitorder="little",
        )
        arrays["sample_physical_weights"][index] = physical_weight
        arrays["sample_log_q"][index] = log_q
        arrays["sample_log_importance_weight"][index] = log_importance
        arrays["sample_anchor_index"][index] = draw["anchor_index"]
        arrays["sample_component_index"][index] = draw["component_index"]
    log_weights = arrays["sample_log_importance_weight"]
    maximum = float(log_weights.max())
    shifted = np.exp(log_weights - maximum)
    shifted_sum = float(shifted.sum())
    normalized = shifted / shifted_sum
    ordered = np.sort(shifted)
    multiplicity = 1.0 + 2.0 * np.arange(
        num_samples - 1, -1, -1, dtype=np.float64,
    )
    diagnostics = {
        "num_samples": num_samples,
        "importance_ess": 1.0 / float(np.dot(normalized, normalized)),
        "importance_ess_fraction": (
            1.0 / float(np.dot(normalized, normalized)) / num_samples
        ),
        "max_normalized_weight": float(normalized.max()),
        "top10_normalized_weight": float(np.sort(normalized)[-10:].sum()),
        "weighted_mean_physical_weight": float(np.dot(
            normalized, arrays["sample_physical_weights"],
        )),
        "stationary_imh_acceptance": float(
            np.dot(ordered, multiplicity) / (num_samples * shifted_sum)
        ),
        "minimum_sampled_physical_weight": int(
            arrays["sample_physical_weights"].min(),
        ),
        "log_unnormalized_normalization_estimate": (
            maximum + math.log(shifted_sum) - math.log(num_samples)
        ),
    }
    if any(not math.isfinite(float(value)) for value in diagnostics.values()):
        raise HgpScreenConflictError("HGP screen IS diagnostic is nonfinite")
    return arrays, diagnostics


def _map_is_identity(registry, config, source_commit, archive_sha256,
                     source_manifest_sha256, cell, artifact, seed,
                     diagnostics, seed_namespace):
    roles = {
        HGP_SCREEN_IS_ROOT: "auxiliary_proposal_overlap_diagnostic_only",
        HGP_SCREEN_PREFLIGHT_IS_ROOT: (
            "preflight_portability_transcript_only"
        ),
    }
    if seed_namespace not in roles:
        raise ValueError("HGP screen replayable IS role changed")
    return {
        "raw_version": HGP_MAP_IS_RAW_VERSION,
        "contract_version": HGP_SCREEN_VERSION,
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "cell": _validate_cell(cell),
        "cell_fingerprint": _cell_fingerprint(cell),
        "method_id": MAP_METHOD_ID,
        "seed_namespace": seed_namespace,
        "seed": int(seed),
        "num_samples": HGP_MAP_IS_SAMPLES,
        "artifact_descriptor": artifact.descriptor,
        "role": roles[seed_namespace],
        "used_for_gate_or_selection": False,
        "diagnostics": diagnostics,
    }


def run_hgp_map_is_diagnostic(registry_path, config_path, source_commit,
                              archive_sha256, source_manifest_sha256, cell,
                              artifact_root, output_path,
                              seed_namespace=HGP_SCREEN_IS_ROOT):
    """Write a replayable 50k-draw auxiliary diagnostic; never a gate input."""
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    cell = _validate_cell(cell)
    artifact = load_hgp_map_artifact(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root,
    )
    seed = _map_is_seed(
        source_commit, archive_sha256, source_manifest_sha256, config,
        registry, cell, artifact.descriptor, seed_namespace,
    )
    arrays, diagnostics = _map_is_transcript(
        artifact.proposal, cell["p"], HGP_MAP_IS_SAMPLES, seed,
    )
    identity = _map_is_identity(
        registry, config, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact, seed, diagnostics,
        seed_namespace,
    )
    transcript_sha256 = _artifact_content_sha256(
        identity, arrays, version=HGP_MAP_IS_RAW_VERSION,
    )
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"HGP screen IS raw already exists: {output_path}")
    atomic_npz(
        output_path, identity_json=np.array(canonical_json(identity)),
        transcript_sha256=np.array(transcript_sha256), **arrays,
    )
    return {
        "output": str(output_path), "sha256": sha256_file(output_path),
        "transcript_sha256": transcript_sha256,
        "cell_fingerprint": _cell_fingerprint(cell),
        "seed_namespace": seed_namespace,
        "diagnostics": diagnostics,
        "used_for_gate_or_selection": False,
    }


def validate_hgp_map_is_diagnostic(
        path, registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root,
        seed_namespace=HGP_SCREEN_IS_ROOT):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    cell = _validate_cell(cell)
    artifact = load_hgp_map_artifact(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root,
    )
    seed = _map_is_seed(
        source_commit, archive_sha256, source_manifest_sha256, config,
        registry, cell, artifact.descriptor, seed_namespace,
    )
    replay_arrays, diagnostics = _map_is_transcript(
        artifact.proposal, cell["p"], HGP_MAP_IS_SAMPLES, seed,
    )
    expected_identity = _map_is_identity(
        registry, config, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact, seed, diagnostics,
        seed_namespace,
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_fields = {
                "identity_json", "transcript_sha256", *replay_arrays,
            }
            if set(data.files) != expected_fields:
                raise HgpScreenConflictError("HGP screen IS raw schema changed")
            identity_json = str(_scalar(data, "identity_json"))
            if (identity_json != canonical_json(expected_identity)
                    or json.loads(identity_json) != expected_identity):
                raise HgpScreenConflictError("HGP screen IS identity changed")
            for name, expected in replay_arrays.items():
                if not np.array_equal(data[name], expected):
                    raise HgpScreenConflictError(
                        f"HGP screen IS replay mismatch: {name}",
                    )
            transcript_sha256 = str(_scalar(data, "transcript_sha256"))
    except HgpScreenConflictError:
        raise
    except Exception as exc:
        raise HgpScreenConflictError(f"HGP screen IS cannot be loaded: {exc}") from exc
    expected_sha256 = _artifact_content_sha256(
        expected_identity, replay_arrays, version=HGP_MAP_IS_RAW_VERSION,
    )
    if transcript_sha256 != expected_sha256:
        raise HgpScreenConflictError("HGP screen IS transcript SHA changed")
    return {
        "path": str(Path(path).resolve()), "sha256": sha256_file(path),
        "cell": cell, "diagnostics": diagnostics,
        "seed_namespace": seed_namespace,
        "transcript_sha256": expected_sha256,
        "used_for_gate_or_selection": False,
    }


def build_hgp_screen_manifest(registry_path, config_path, source_commit,
                              archive_sha256, source_manifest_sha256,
                              resource_tier, artifact_root, output_path=None):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    if resource_tier not in RESOURCE_TIERS:
        raise ValueError("unknown HGP screen resource tier")
    artifact_descriptors = load_hgp_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )
    descriptor_by_cell = {
        value["cell_fingerprint"]: value for value in artifact_descriptors
    }
    if len(descriptor_by_cell) != len(_map_cells(config)):
        raise HgpScreenConflictError("MAP artifact descriptor set changed")
    tasks = []
    for method in SCREEN_METHODS:
        for cell in _method_cells(config, method):
            for family in INIT_FAMILIES:
                for trajectory in range(TRAJECTORIES_PER_FAMILY):
                    task = _task_identity(
                        config, registry, source_commit, archive_sha256,
                        source_manifest_sha256, method, resource_tier, cell,
                        family, trajectory,
                        map_artifact_descriptor=(
                            descriptor_by_cell[_cell_fingerprint(cell)]
                            if method == MAP_METHOD_ID else None
                        ),
                    )
                    tasks.append({
                        "task": task,
                        "task_fingerprint": sha256_json(task),
                        "output_relpath": _task_output_relpath(task),
                        "owner": EXECUTION_NODES[trajectory % len(EXECUTION_NODES)],
                    })
    identity = {
        "manifest_version": HGP_SCREEN_MANIFEST_VERSION,
        "contract_version": HGP_SCREEN_VERSION,
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "resource_tier": resource_tier,
        "map_artifacts": artifact_descriptors,
        "importance_sampling": {
            "raw_version": HGP_MAP_IS_RAW_VERSION,
            "num_samples_per_cell": HGP_MAP_IS_SAMPLES,
            "seed_namespace": HGP_SCREEN_IS_ROOT,
            "used_for_gate_or_selection": False,
            "outputs": [
                f"importance_sampling/{_cell_fingerprint(cell)}.npz"
                for cell in _map_cells(config)
            ],
        },
        "execution_nodes": list(EXECUTION_NODES),
        "analysis": dict(config["execution"]["analysis"]),
        "task_count": len(tasks), "tasks": tasks,
    }
    if len(tasks) != 384:
        raise AssertionError("HGP screen task count changed")
    manifest = {**identity, "manifest_sha256": sha256_json(identity)}
    if output_path is not None:
        atomic_json(output_path, manifest)
    return manifest


def validate_hgp_screen_manifest(manifest, registry, config, artifact_root):
    expected = build_hgp_screen_manifest(
        registry.get("_registry_path", Path(config["config_path"]).parents[1]
                     / "registry/registry.json"),
        config["config_path"], manifest.get("source_commit", ""),
        manifest.get("archive_sha256", ""),
        manifest.get("source_manifest_sha256", ""),
        manifest.get("resource_tier", ""), artifact_root, None,
    )
    if manifest != expected:
        raise HgpScreenConflictError("HGP screen manifest is noncanonical")
    return True


def _disorder(registry, code, model, cell):
    uniform_seed = _uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(
        model.num_qubits,
    )
    epsilon = (uniforms < float(cell["p"])).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    return uniform_seed, epsilon, syndrome


def _character_seed(registry_sha256, code_id):
    return derive_seed(
        HGP_SCREEN_CHARACTER_ROOT, registry_sha256, code_id, "characters",
    )


def _b_character_seed(registry_sha256, code_id):
    return derive_seed(
        HGP_SCREEN_B_CHARACTER_ROOT, registry_sha256, code_id, "B-characters",
    )


def _sampler_config(method, p, tier, *, max_anchors=8):
    resource = RESOURCE_TIERS[tier]
    if method in HP_METHODS:
        return CollapsedPowerPtConfig(
            method, p, resource["burn"], resource["measurement"],
        )
    if method == MAP_METHOD_ID:
        return MapMixtureConfig(
            p, resource["burn"], resource["measurement"],
            max_anchors=max_anchors,
        )
    raise ValueError("unknown HGP screen method")


def _task_context(registry_path, config_path, source_commit, archive_sha256,
                  source_manifest_sha256, task, artifact_root):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    expected = _task_identity(
        config, registry, source_commit, archive_sha256,
        source_manifest_sha256, task["method_id"], task["resource_tier"],
        task["cell"], task["init_family"], task["trajectory_index"],
        map_artifact_descriptor=task.get("map_artifact"),
    )
    if task != expected:
        raise HgpScreenConflictError("HGP screen task identity changed")
    _, code, H = load_frozen_code(registry_path, task["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _disorder(
        registry, code, model, task["cell"],
    )
    seed = HgpScreenSeedIdentity(**task["seed_identity"])
    initial = (
        epsilon.copy() if task["init_family"] == "P" else
        uniform_hard_coset_state(
            model, syndrome, seed.seed("initialize", "hard_coset"),
        )
    )
    characters = frozen_character_set(
        model.k,
        _character_seed(registry["registry_sha256"], code["code_id"]),
        config["characters"]["num_nonbasis"],
    )
    b_characters = frozen_b_character_set(
        H.shape[0],
        _b_character_seed(registry["registry_sha256"], code["code_id"]),
        config["b_character_spec"]["dense_count"],
    )
    sampler = _sampler_config(
        task["method_id"], task["cell"]["p"], task["resource_tier"],
        max_anchors=_map_requested_max_anchors(config, task["cell"]),
    )
    map_artifact = None
    if task["method_id"] == MAP_METHOD_ID:
        map_artifact = load_hgp_map_artifact(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, task["cell"], artifact_root,
            expected_descriptor=task["map_artifact"],
        )
    return (registry, config, code, H, model, frame, uniform_seed, syndrome,
            seed, initial, characters, b_characters, sampler, map_artifact)


def _run_sampler(method, model, frame, H, syndrome, sampler, seed, initial,
                 map_artifact=None):
    if method in HP_METHODS:
        return run_collapsed_power_pt_trajectory(
            model, frame, H, syndrome, sampler, seed, initial, engine="numba",
        )
    if map_artifact is None:
        raise HgpScreenConflictError(
            "MAP task requires a loaded frozen artifact",
        )
    return run_map_mixture_trajectory(
        model, frame, syndrome, sampler, seed, initial,
        anchor_catalog=map_artifact.catalog, proposal=map_artifact.proposal,
        frozen_artifact_replay=True,
    )


def _value_digest(values):
    digest = hashlib.sha256(b"exp102.q0_hgp_global.screen.trajectory.v1\0")
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _syndrome_sha256(syndrome):
    return hashlib.sha256(
        np.asarray(syndrome.shape, dtype=">u8").tobytes()
        + np.packbits(syndrome, bitorder="little").tobytes()
    ).hexdigest()


def run_hgp_screen_task(registry_path, config_path, source_commit,
                        archive_sha256, source_manifest_sha256, task,
                        artifact_root, output_path):
    (registry, config, code, H, model, frame, uniform_seed, syndrome, seed,
     initial, characters, b_characters, sampler, map_artifact) = _task_context(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, task, artifact_root,
    )
    start_wall, start_core = time.monotonic(), time.process_time()
    result = _run_sampler(
        task["method_id"], model, frame, H, syndrome, sampler, seed, initial,
        map_artifact,
    )
    wall_seconds = time.monotonic() - start_wall
    core_seconds = time.process_time() - start_core
    sampler_raw_version = str(result["raw_version"])
    expected_sampler_raw = (
        COLLAPSED_RAW_VERSION if task["method_id"] in HP_METHODS
        else MAP_RAW_VERSION
    )
    if sampler_raw_version != expected_sampler_raw:
        raise HgpScreenConflictError("sampler raw version changed")
    payload = {
        "raw_version": np.array(
            HGP_POWER_RAW_VERSION if task["method_id"] in HP_METHODS
            else HGP_MAP_RAW_VERSION
        ),
        "sampler_raw_version": np.array(sampler_raw_version),
        "contract_version": np.array(HGP_SCREEN_VERSION),
        "task_json": np.array(canonical_json(task)),
        "task_fingerprint": np.array(sha256_json(task)),
        "source_commit": np.array(source_commit),
        "archive_sha256": np.array(archive_sha256),
        "source_manifest_sha256": np.array(source_manifest_sha256),
        "registry_sha256": np.array(registry["registry_sha256"]),
        "hgp_screen_config_sha256": np.array(
            config["hgp_screen_config_sha256"],
        ),
        "cell_json": np.array(canonical_json(task["cell"])),
        "uniform_seed": np.array(uniform_seed, dtype=np.uint64),
        "syndrome_packed": np.packbits(syndrome, bitorder="little"),
        "syndrome_sha256": np.array(_syndrome_sha256(syndrome)),
        "model_fingerprint": np.array(model.fingerprint()),
        "section_fingerprint": np.array(code["section_fingerprint"]),
        "logical_frame_fingerprint": np.array(
            code["logical_frame_fingerprint"],
        ),
        "character_masks": characters.masks,
        "character_sha256": np.array(characters.character_sha256),
        "b_character_masks_packed": b_characters.masks_packed,
        "b_character_sha256": np.array(b_characters.character_sha256),
        "b_character_count": np.array(b_characters.size, dtype=np.int32),
        "b_dimension": np.array(b_characters.dimension, dtype=np.int32),
        "b_dense_character_count": np.array(
            b_characters.dense_count, dtype=np.int16,
        ),
        "num_qubits": np.array(model.num_qubits, dtype=np.int32),
        "k": np.array(model.k, dtype=np.int16),
        "trajectory_digest": np.array(_value_digest(result)),
        "core_seconds": np.array(core_seconds, dtype=np.float64),
        "wall_seconds": np.array(wall_seconds, dtype=np.float64),
    }
    if map_artifact is not None:
        payload.update({
            "map_artifact_descriptor_json": np.array(
                canonical_json(map_artifact.descriptor),
            ),
            "map_artifact_file_sha256": np.array(
                map_artifact.descriptor["artifact_file_sha256"],
            ),
            "map_artifact_content_sha256": np.array(
                map_artifact.descriptor["artifact_content_sha256"],
            ),
        })
    for name, value in result.items():
        payload[f"sampler_{name}"] = np.asarray(value)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"HGP screen raw already exists: {output_path}")
    atomic_npz(output_path, **payload)
    return {
        "output": str(output_path), "sha256": sha256_file(output_path),
        "task_fingerprint": sha256_json(task),
        "wall_seconds": wall_seconds, "core_seconds": core_seconds,
    }


def _scalar(data, name):
    value = np.asarray(data[name])
    if value.shape != ():
        raise HgpScreenConflictError(f"HGP screen raw scalar changed: {name}")
    return value.item()


def _compare_result(stored, replay, *, has_map_artifact):
    expected_fields = {f"sampler_{name}" for name in replay}
    actual_fields = {name for name in stored.files if name.startswith("sampler_")}
    if actual_fields != expected_fields:
        raise HgpScreenConflictError("HGP screen sampler raw schema changed")
    for name, expected in replay.items():
        if not np.array_equal(stored[f"sampler_{name}"], np.asarray(expected)):
            raise HgpScreenConflictError(
                f"HGP screen sampler replay mismatch: {name}",
            )
    outer_fields = _COMMON_RAW_FIELDS | (
        _MAP_ARTIFACT_RAW_FIELDS if has_map_artifact else set()
    )
    if set(stored.files) != outer_fields | expected_fields:
        raise HgpScreenConflictError("HGP screen outer raw schema changed")


def _validate_map_transition_counters(data):
    for stage, start_field in (
            ("burn", "sampler_initial_state_packed"),
            ("measurement", "sampler_burn_state_packed")):
        proposals = np.asarray(
            data[f"sampler_{stage}_proposal_states_packed"], dtype=np.uint8,
        )
        states = np.asarray(
            data[f"sampler_{stage}_states_packed"], dtype=np.uint8,
        )
        accepted = np.asarray(
            data[f"sampler_{stage}_accepted"], dtype=np.uint8,
        )
        changed = np.asarray(
            data[f"sampler_{stage}_state_changed"], dtype=np.uint8,
        )
        start = np.asarray(data[start_field], dtype=np.uint8)
        if (proposals.ndim != 2 or states.shape != proposals.shape
                or accepted.shape != (states.shape[0],)
                or changed.shape != accepted.shape
                or start.shape != (states.shape[1],)
                or np.any(accepted > 1) or np.any(changed > 1)):
            raise HgpScreenConflictError("MAP transition transcript shape changed")
        before = np.vstack((start[None, :], states[:-1]))
        accepted_mask = accepted.astype(np.bool_)
        expected_changed = accepted_mask & np.any(proposals != before, axis=1)
        if (not np.array_equal(changed.astype(np.bool_), expected_changed)
                or not np.array_equal(states[accepted_mask], proposals[accepted_mask])
                or not np.array_equal(states[~accepted_mask], before[~accepted_mask])
                or int(_scalar(data, f"sampler_{stage}_attempts"))
                != states.shape[0]
                or int(_scalar(data, f"sampler_{stage}_accepts"))
                != int(accepted.sum())
                or int(_scalar(data, f"sampler_{stage}_state_changes"))
                != int(expected_changed.sum())):
            raise HgpScreenConflictError("MAP transition counters changed")


def _b_record_from_replay(replay, H, syndrome, p, b_characters, method_id,
                          burn_count):
    r, n = np.asarray(H).shape
    initial_b = _extract_b_states_packed(replay["initial_state_packed"], n, r)
    burn_b = _extract_b_states_packed(replay["burn_state_packed"], n, r)
    measurement_b = _extract_b_states_packed(
        replay["measurement_states_packed"], n, r,
    )
    mass = build_classical_coset_mass(H, p)
    log_mass = np.log(mass)
    likelihood = _b_log_likelihood(measurement_b, H, syndrome, log_mass)
    burn_likelihood = _b_log_likelihood(
        burn_b[None, :], H, syndrome, log_mass,
    )[0]
    if method_id in HP_METHODS:
        stored_likelihood = np.asarray(
            replay["cold_log_likelihood"], dtype=np.float64,
        )
        expected_length = int(burn_count) + measurement_b.shape[0]
        if (stored_likelihood.shape != (expected_length,)
                or not np.array_equal(stored_likelihood[int(burn_count):], likelihood)
                or (int(burn_count) > 0 and stored_likelihood[int(burn_count) - 1]
                    != burn_likelihood)):
            raise HgpScreenConflictError(
                "HP cold likelihood disagrees with reconstructed B trace",
            )
    masks = b_characters.masks_packed
    return {
        "b_initial_state_packed": initial_b,
        "b_burn_state_packed": burn_b,
        "b_measurement_states_packed": measurement_b,
        "b_measurement_weights": _b_weights(measurement_b),
        "b_measurement_log_likelihood": likelihood,
        "b_initial_character_bits": _b_character_bits(
            initial_b[None, :], masks,
        )[0],
        "b_burn_character_bits": _b_character_bits(
            burn_b[None, :], masks,
        )[0],
        "b_character_set": b_characters,
        "b_r": int(r),
        "b_a_factor_count": int(n),
    }


def validate_hgp_screen_raw(path, registry, config, source_commit,
                            archive_sha256, source_manifest_sha256,
                            artifact_root):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        task = json.loads(str(_scalar(data, "task_json")))
        expected_task = _task_identity(
            config, registry, source_commit, archive_sha256,
            source_manifest_sha256, task["method_id"], task["resource_tier"],
            task["cell"], task["init_family"], task["trajectory_index"],
            map_artifact_descriptor=task.get("map_artifact"),
        )
        if task != expected_task or str(_scalar(data, "task_fingerprint")) != sha256_json(task):
            raise HgpScreenConflictError("HGP screen raw task changed")
        registry_path = registry.get("_registry_path")
        if registry_path is None:
            registry_path = str(
                Path(config["config_path"]).parents[1] / "registry/registry.json"
            )
        (rebuilt_registry, rebuilt_config, code, H, model, frame, uniform_seed,
         syndrome, seed, initial, characters, b_characters, sampler,
         map_artifact) = _task_context(
            registry_path, config["config_path"],
            source_commit, archive_sha256, source_manifest_sha256, task,
            artifact_root,
        )
        expected_scalars = {
            "contract_version": HGP_SCREEN_VERSION,
            "source_commit": source_commit,
            "archive_sha256": archive_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "registry_sha256": rebuilt_registry["registry_sha256"],
            "hgp_screen_config_sha256": rebuilt_config["hgp_screen_config_sha256"],
            "cell_json": canonical_json(task["cell"]),
            "uniform_seed": uniform_seed,
            "syndrome_sha256": _syndrome_sha256(syndrome),
            "model_fingerprint": model.fingerprint(),
            "section_fingerprint": code["section_fingerprint"],
            "logical_frame_fingerprint": code["logical_frame_fingerprint"],
            "character_sha256": characters.character_sha256,
            "b_character_sha256": b_characters.character_sha256,
            "b_character_count": b_characters.size,
            "b_dimension": b_characters.dimension,
            "b_dense_character_count": b_characters.dense_count,
            "num_qubits": model.num_qubits,
            "k": model.k,
        }
        expected_outer_raw = (
            HGP_POWER_RAW_VERSION if task["method_id"] in HP_METHODS
            else HGP_MAP_RAW_VERSION
        )
        expected_sampler_raw = (
            COLLAPSED_RAW_VERSION if task["method_id"] in HP_METHODS
            else MAP_RAW_VERSION
        )
        expected_scalars.update({
            "raw_version": expected_outer_raw,
            "sampler_raw_version": expected_sampler_raw,
        })
        for name, expected in expected_scalars.items():
            if str(_scalar(data, name)) != str(expected):
                raise HgpScreenConflictError(
                    f"HGP screen raw identity mismatch: {name}",
                )
        if not np.array_equal(
                data["syndrome_packed"], np.packbits(syndrome, bitorder="little")):
            raise HgpScreenConflictError("HGP screen syndrome bytes changed")
        if (not np.array_equal(data["character_masks"], characters.masks)
                or not np.array_equal(
                    data["b_character_masks_packed"],
                    b_characters.masks_packed,
                )
                or not np.isfinite(float(_scalar(data, "core_seconds")))
                or not np.isfinite(float(_scalar(data, "wall_seconds")))
                or float(_scalar(data, "core_seconds")) < 0.0
                or float(_scalar(data, "wall_seconds")) < 0.0):
            raise HgpScreenConflictError("HGP screen raw context/timing changed")
        replay = _run_sampler(
            task["method_id"], model, frame, H, syndrome, sampler, seed, initial,
            map_artifact,
        )
        if map_artifact is not None:
            expected_artifact_scalars = {
                "map_artifact_descriptor_json": canonical_json(
                    map_artifact.descriptor,
                ),
                "map_artifact_file_sha256": map_artifact.descriptor[
                    "artifact_file_sha256"
                ],
                "map_artifact_content_sha256": map_artifact.descriptor[
                    "artifact_content_sha256"
                ],
            }
            for name, expected in expected_artifact_scalars.items():
                if str(_scalar(data, name)) != expected:
                    raise HgpScreenConflictError(
                        f"HGP screen MAP artifact mismatch: {name}",
                    )
        _compare_result(
            data, replay, has_map_artifact=map_artifact is not None,
        )
        if map_artifact is not None:
            _validate_map_transition_counters(data)
        if str(_scalar(data, "trajectory_digest")) != _value_digest(replay):
            raise HgpScreenConflictError("HGP screen trajectory digest changed")
        b_record = _b_record_from_replay(
            replay, H, syndrome, task["cell"]["p"], b_characters,
            task["method_id"],
            sampler.burn_rounds if task["method_id"] in HP_METHODS
            else sampler.burn_steps,
        )
        labels = data["sampler_measurement_labels"].copy()
        weights = data["sampler_measurement_weights"].copy()
        burn_labels = data["sampler_burn_labels"].copy()
        record = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "task": task, "task_fingerprint": sha256_json(task),
            "cell": task["cell"], "method_id": task["method_id"],
            "resource_tier": task["resource_tier"],
            "init_family": task["init_family"],
            "trajectory_index": int(task["trajectory_index"]),
            "labels": labels, "weights": weights,
            "valid_mask": np.ones(labels.size, dtype=np.bool_),
            "burn_labels": burn_labels,
            "initial_label": int(_scalar(data, "sampler_initial_label")),
            "num_qubits": model.num_qubits, "k": model.k,
            "character_masks": characters.masks,
            "core_seconds": float(_scalar(data, "core_seconds")),
            **b_record,
        }
        if task["method_id"] in HP_METHODS:
            attempts = data["sampler_swap_attempts"].astype(np.float64)
            accepts = data["sampler_swap_accepts"].astype(np.float64)
            rates = accepts / np.maximum(attempts, 1.0)
            record["algorithm_metrics"] = {
                "min_edge_swap_rate": float(rates.min()),
                "min_edge_swap_accepts": int(accepts.min()),
                "round_trips": int(data["sampler_round_trips_by_origin"].sum()),
                "cold_origin_fraction": float(
                    np.count_nonzero(data["sampler_cold_visits_by_origin"])
                    / data["sampler_cold_visits_by_origin"].size
                ),
            }
        else:
            measurement_attempts = int(
                _scalar(data, "sampler_measurement_attempts")
            )
            record["algorithm_metrics"] = {
                "burn_accepts": int(_scalar(data, "sampler_burn_accepts")),
                "burn_state_changes": int(
                    _scalar(data, "sampler_burn_state_changes")
                ),
                "measurement_accepts": int(
                    _scalar(data, "sampler_measurement_accepts")
                ),
                "measurement_acceptance": float(
                    _scalar(data, "sampler_measurement_accepts")
                    / max(measurement_attempts, 1)
                ),
                "measurement_state_changes": int(
                    _scalar(data, "sampler_measurement_state_changes")
                ),
                "measurement_state_change_rate": float(
                    _scalar(data, "sampler_measurement_state_changes")
                    / max(measurement_attempts, 1)
                ),
            }
    return record


_REPLAY_CONTEXT = {}


def _initialize_replay(registry_path, config_path, source_commit,
                       archive_sha256, source_manifest_sha256, artifact_root):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    _REPLAY_CONTEXT.clear()
    _REPLAY_CONTEXT["registry"] = registry
    _REPLAY_CONTEXT["config"] = load_hgp_screen_config(config_path, registry)
    _REPLAY_CONTEXT["source_commit"] = source_commit
    _REPLAY_CONTEXT["archive_sha256"] = archive_sha256
    _REPLAY_CONTEXT["source_manifest_sha256"] = source_manifest_sha256
    _REPLAY_CONTEXT["artifact_root"] = str(artifact_root)


def _validate_raw_worker(path):
    return validate_hgp_screen_raw(
        path, _REPLAY_CONTEXT["registry"], _REPLAY_CONTEXT["config"],
        _REPLAY_CONTEXT["source_commit"],
        _REPLAY_CONTEXT["archive_sha256"],
        _REPLAY_CONTEXT["source_manifest_sha256"],
        _REPLAY_CONTEXT["artifact_root"],
    )


def _public_summary(value):
    return {name: item for name, item in value.items() if not name.startswith("_")}


def _algorithm_failures(records, method, gates):
    failures = []
    if method in HP_METHODS:
        if any(value["algorithm_metrics"]["min_edge_swap_rate"]
               < gates["hp_min_edge_swap_rate"] for value in records):
            failures.append("hp_edge_swap")
        if any(value["algorithm_metrics"]["min_edge_swap_accepts"]
               < gates["hp_min_edge_swap_accepts"] for value in records):
            failures.append("hp_edge_swap_attempts")
        if any(value["algorithm_metrics"]["round_trips"]
               < gates["hp_min_round_trips_per_trajectory"] for value in records):
            failures.append("hp_round_trip")
        if any(value["algorithm_metrics"]["cold_origin_fraction"]
               < gates["hp_min_cold_origin_fraction"] for value in records):
            failures.append("hp_origin_transport")
    else:
        if any(value["algorithm_metrics"]["burn_state_changes"]
               < gates["map_min_burn_state_changes"] for value in records):
            failures.append("map_burn_state_changes")
        if any(value["algorithm_metrics"]["measurement_state_change_rate"]
               < gates["map_min_measurement_state_change_rate"]
               for value in records):
            failures.append("map_measurement_state_change_rate")
        if any(value["algorithm_metrics"]["measurement_state_changes"]
               < gates["map_min_measurement_state_changes"]
               for value in records):
            failures.append("map_measurement_state_changes")
    return failures


def _b_character_name(index, character_set):
    index = int(index)
    r = int(character_set.r)
    if index < r * r:
        return f"single_r{index // r:02d}_c{index % r:02d}"
    index -= r * r
    if index < r:
        return f"row_{index:02d}"
    index -= r
    if index < r:
        return f"column_{index:02d}"
    return f"dense_{index - r:02d}"


def _b_character_chains(records, character_set, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((len(records), 0, 0), dtype=np.uint8)
    if np.all(indices < character_set.single_count):
        return np.stack([
            _b_single_bits(record["b_measurement_states_packed"], indices)
            for record in records
        ])
    masks = character_set.masks_packed[indices]
    return np.stack([
        _b_character_bits(record["b_measurement_states_packed"], masks)
        for record in records
    ])


def _constant_b_character_failures(records, indices, common_bits,
                                   character_set):
    failures = []
    for index, common in zip(indices, common_bits):
        opposite = [
            record for record in records
            if int(record["b_initial_character_bits"][int(index)]) != int(common)
        ]
        if any(
                int(record["b_burn_character_bits"][int(index)]) != int(common)
                for record in opposite):
            failures.append(_b_character_name(index, character_set))
    return failures


def _b_character_diagnostics(records, character_set, *, calculate_ess):
    count = character_set.size
    means = np.empty((len(records), count), dtype=np.float64)
    rhats = np.empty(count, dtype=np.float64)
    nondegenerate = np.zeros(count, dtype=np.bool_)
    constant_failures = []
    ess_values = []
    dense_nondegenerate = 0
    chunks = (
        (range(0, character_set.single_count, 128), 128,
         character_set.single_count),
        (range(character_set.single_count, count, 16), 16, count),
    )
    for starts, width, stop_limit in chunks:
        for start in starts:
            stop = min(start + width, stop_limit)
            indices = np.arange(start, stop, dtype=np.int64)
            chains = _b_character_chains(records, character_set, indices)
            means[:, start:stop] = 1.0 - 2.0 * chains.mean(axis=1)
            rhats[start:stop] = _split_rhat_columns(chains)
            flat_first = chains[0, 0, :]
            varying = np.any(chains != flat_first[None, None, :], axis=(0, 1))
            nondegenerate[start:stop] = varying
            constant_local = np.flatnonzero(~varying)
            if constant_local.size:
                constant_failures.extend(_constant_b_character_failures(
                    records, indices[constant_local], flat_first[constant_local],
                    character_set,
                ))
            if calculate_ess and start >= character_set.single_count:
                for offset in np.flatnonzero(varying):
                    ess_values.append(_statistics._fft_bulk_ess(
                        chains[:, :, int(offset)],
                    ))
            dense_start = max(start, character_set.dense_start)
            if dense_start < stop:
                dense_nondegenerate += int(
                    nondegenerate[dense_start:stop].sum()
                )
    return {
        "means": means,
        "rhats": rhats,
        "nondegenerate": nondegenerate,
        "constant_failures": constant_failures,
        "min_nonsingle_bulk_ess": (
            None if not ess_values else float(min(ess_values))
        ),
        "dense_nondegenerate": dense_nondegenerate,
    }


def _b_family_summary(records, config):
    records = sorted(records, key=lambda value: value["trajectory_index"])
    if (len(records) != TRAJECTORIES_PER_FAMILY
            or [row["trajectory_index"] for row in records]
            != list(range(TRAJECTORIES_PER_FAMILY))):
        raise HgpScreenConflictError("B family does not contain trajectories 0..15")
    first = records[0]
    character_set = first["b_character_set"]
    if any(
            row["b_character_set"].character_sha256
            != character_set.character_sha256 for row in records[1:]):
        raise HgpScreenConflictError("B family character masks differ")
    measurement_lengths = {
        int(row["b_measurement_weights"].size) for row in records
    }
    if len(measurement_lengths) != 1:
        raise HgpScreenConflictError("B family measurement clocks differ")
    weight_chains = np.stack([
        row["b_measurement_weights"] for row in records
    ]).astype(np.float64)
    likelihood_chains = np.stack([
        row["b_measurement_log_likelihood"] for row in records
    ]).astype(np.float64)
    character = _b_character_diagnostics(
        records, character_set, calculate_ess=True,
    )
    weight_rhat = split_rhat(weight_chains)
    likelihood_rhat = split_rhat(likelihood_chains)
    weight_ess = _statistics._fft_bulk_ess(weight_chains)
    likelihood_ess = _statistics._fft_bulk_ess(likelihood_chains)
    constant_weight_chains = [
        index for index, chain in enumerate(weight_chains)
        if np.all(chain == chain[0])
    ]
    constant_likelihood_chains = [
        index for index, chain in enumerate(likelihood_chains)
        if np.all(chain == chain[0])
    ]
    max_rhat = float(max(
        weight_rhat, likelihood_rhat, float(np.max(character["rhats"])),
    ))
    nonsingle_ess = character["min_nonsingle_bulk_ess"]
    min_ess = float(min(
        weight_ess, likelihood_ess,
        float("inf") if nonsingle_ess is None else nonsingle_ess,
    ))
    failures = []
    gates = config["gates"]
    if max_rhat > gates["max_rhat"]:
        failures.append("b_rhat")
    if min_ess < gates["min_bulk_ess"]:
        failures.append("b_bulk_ess")
    if constant_weight_chains:
        failures.append("b_weight_constant")
    if constant_likelihood_chains:
        failures.append("b_likelihood_constant")
    if character["constant_failures"]:
        failures.append("b_constant_character_no_burn_crossing")
    if character["dense_nondegenerate"] < B_MIN_NONDEGENERATE_DENSE:
        failures.append("b_dense_characters_uninformative")

    dimension = character_set.dimension
    a_factors = int(first["b_a_factor_count"])
    weight_trajectory_means = weight_chains.mean(axis=1) / dimension
    likelihood_trajectory_means = likelihood_chains.mean(axis=1) / a_factors
    weight_mean, weight_se = trajectory_mean_and_se(weight_trajectory_means)
    likelihood_mean, likelihood_se = trajectory_mean_and_se(
        likelihood_trajectory_means,
    )
    return {
        "init_family": first["init_family"],
        "character_sha256": character_set.character_sha256,
        "character_count": character_set.size,
        "single_count": character_set.single_count,
        "row_column_count": character_set.row_column_count,
        "dense_count": character_set.dense_count,
        "dense_nondegenerate": character["dense_nondegenerate"],
        "max_rhat": max_rhat,
        "min_nonsingle_bulk_ess": min_ess,
        "weight_rhat": float(weight_rhat),
        "weight_bulk_ess": float(weight_ess),
        "likelihood_rhat": float(likelihood_rhat),
        "likelihood_bulk_ess": float(likelihood_ess),
        "constant_weight_chains": constant_weight_chains,
        "constant_likelihood_chains": constant_likelihood_chains,
        "constant_character_failures": character["constant_failures"],
        "normalized_mean_b_weight": weight_mean,
        "normalized_mean_b_weight_se": weight_se,
        "mean_b_log_likelihood_per_factor": likelihood_mean,
        "mean_b_log_likelihood_per_factor_se": likelihood_se,
        "valid": not failures,
        "failures": sorted(set(failures)),
        "_character_means": character["means"],
        "_weight_trajectory_means": weight_trajectory_means,
        "_likelihood_trajectory_means": likelihood_trajectory_means,
        "_records": records,
        "_character_set": character_set,
    }


def _b_d2_estimate(left_means, right_means):
    left = np.asarray(left_means, dtype=np.float64)
    right = np.asarray(right_means, dtype=np.float64)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]
            or left.shape[0] < 3 or right.shape[0] < 3):
        raise HgpScreenConflictError("B-character D2 needs independent trajectories")

    def estimate(a, b):
        a_square = (
            np.square(a.sum(axis=0)) - np.square(a).sum(axis=0)
        ) / (a.shape[0] * (a.shape[0] - 1))
        b_square = (
            np.square(b.sum(axis=0)) - np.square(b).sum(axis=0)
        ) / (b.shape[0] * (b.shape[0] - 1))
        return float(np.mean(a_square + b_square - 2.0 * a.mean(axis=0) * b.mean(axis=0)))

    value = estimate(left, right)
    left_delete = np.asarray([
        estimate(np.delete(left, omitted, axis=0), right)
        for omitted in range(left.shape[0])
    ])
    right_delete = np.asarray([
        estimate(left, np.delete(right, omitted, axis=0))
        for omitted in range(right.shape[0])
    ])
    left_variance = (
        (left.shape[0] - 1) / left.shape[0]
        * float(np.square(left_delete - left_delete.mean()).sum())
    )
    right_variance = (
        (right.shape[0] - 1) / right.shape[0]
        * float(np.square(right_delete - right_delete.mean()).sum())
    )
    return {
        "mean_square_character_delta": value,
        "trajectory_jackknife_se": math.sqrt(left_variance + right_variance),
    }


def _b_character_delta_gate(left_means, right_means, character_set, config):
    left = np.asarray(left_means, dtype=np.float64)
    right = np.asarray(right_means, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise HgpScreenConflictError("B-character delta needs matched trajectories")
    delta = np.abs(left.mean(axis=0) - right.mean(axis=0))
    se = np.sqrt(
        left.var(axis=0, ddof=1) / left.shape[0]
        + right.var(axis=0, ddof=1) / right.shape[0]
    )
    gates = config["gates"]
    absolute = delta <= gates["max_abs_b_character_mean_delta"]
    sigma_bound = (
        gates["delta_sigma_multiplier"] * se
        + gates["b_character_delta_sigma_slack"]
    )
    sigma = delta <= sigma_bound
    failed = np.flatnonzero(~(absolute & sigma))
    return {
        "max_abs_character_mean_delta": float(delta.max(initial=0.0)),
        "max_character_mean_delta_se": float(se.max(initial=0.0)),
        "max_sigma_bound_excess": float(
            np.maximum(delta - sigma_bound, 0.0).max(initial=0.0)
        ),
        "failed_character_count": int(failed.size),
        "failed_characters": [
            _b_character_name(index, character_set) for index in failed
        ],
        "absolute_pass": bool(np.all(absolute)),
        "sigma_pass": bool(np.all(sigma)),
    }


def _b_pooled_rhat(left, right):
    records = [*left["_records"], *right["_records"]]
    character_set = left["_character_set"]
    if (right["_character_set"].character_sha256
            != character_set.character_sha256):
        raise HgpScreenConflictError("compared B character sets differ")
    character = _b_character_diagnostics(
        records, character_set, calculate_ess=False,
    )
    weight_chains = np.stack([
        row["b_measurement_weights"] for row in records
    ]).astype(np.float64)
    likelihood_chains = np.stack([
        row["b_measurement_log_likelihood"] for row in records
    ]).astype(np.float64)
    return {
        "max_rhat": float(max(
            split_rhat(weight_chains), split_rhat(likelihood_chains),
            float(np.max(character["rhats"])),
        )),
        "constant_character_failures": character["constant_failures"],
    }


def _compare_b_summaries(left, right, config):
    d2 = _b_d2_estimate(left["_character_means"], right["_character_means"])
    character_delta = _b_character_delta_gate(
        left["_character_means"], right["_character_means"],
        left["_character_set"], config,
    )
    weight_left, weight_left_se = trajectory_mean_and_se(
        left["_weight_trajectory_means"],
    )
    weight_right, weight_right_se = trajectory_mean_and_se(
        right["_weight_trajectory_means"],
    )
    likelihood_left, likelihood_left_se = trajectory_mean_and_se(
        left["_likelihood_trajectory_means"],
    )
    likelihood_right, likelihood_right_se = trajectory_mean_and_se(
        right["_likelihood_trajectory_means"],
    )
    weight_delta = abs(weight_left - weight_right)
    weight_se = math.hypot(weight_left_se, weight_right_se)
    likelihood_delta = abs(likelihood_left - likelihood_right)
    likelihood_se = math.hypot(likelihood_left_se, likelihood_right_se)
    pooled = _b_pooled_rhat(left, right)
    gates = config["gates"]
    d2_upper = max(0.0, d2["mean_square_character_delta"]) + 3.0 * d2[
        "trajectory_jackknife_se"
    ]
    r = int(left["_character_set"].r)
    n = int(left["_records"][0]["b_a_factor_count"])
    failures = []
    if d2_upper > gates["max_b_character_d2_upper"]:
        failures.append("b_character_d2")
    if (not character_delta["absolute_pass"]
            or not character_delta["sigma_pass"]):
        failures.append("b_character_mean_delta")
    if (weight_delta > gates["max_b_normalized_weight_delta"]
            or weight_delta > 3.0 * weight_se + 1.0 / (r * r)):
        failures.append("b_weight_delta")
    if (likelihood_delta
            > gates["max_b_log_likelihood_delta_per_factor"]
            or likelihood_delta > 3.0 * likelihood_se + 1.0 / n):
        failures.append("b_likelihood_delta")
    if pooled["max_rhat"] > gates["max_rhat"]:
        failures.append("b_pooled_rhat")
    if pooled["constant_character_failures"]:
        failures.append("b_pooled_constant_character_no_burn_crossing")
    return {
        "character_d2": d2,
        "character_d2_upper": d2_upper,
        "character_mean_delta": character_delta,
        "normalized_b_weight_delta": weight_delta,
        "normalized_b_weight_delta_se": weight_se,
        "b_log_likelihood_delta_per_factor": likelihood_delta,
        "b_log_likelihood_delta_per_factor_se": likelihood_se,
        "max_pooled_rhat": pooled["max_rhat"],
        "constant_character_failures": pooled["constant_character_failures"],
        "valid": not failures,
        "failures": sorted(set(failures)),
    }


def _b_cell_summary(families, config):
    comparison = _compare_b_summaries(families["P"], families["U"], config)
    public_families = {
        family: _public_summary(summary) for family, summary in families.items()
    }
    failures = []
    if not all(summary["valid"] for summary in families.values()):
        failures.append("b_family_gate")
    if not comparison["valid"]:
        failures.append("b_initialization_comparison")
    return {
        "families": public_families,
        "initialization_comparison": comparison,
        "valid": not failures,
        "failures": failures,
    }


def _compare_family_summaries(left, right, config, family, num_qubits):
    if not np.array_equal(
            left["_character_set"].masks, right["_character_set"].masks):
        raise HgpScreenConflictError("cross-mechanism character sets changed")
    delta = _statistics._delta_gate(left, right, config)
    failures = []
    if not left["valid"] or not right["valid"]:
        failures.append("family_gate")
    if not delta["absolute_pass"] or not delta["sigma_pass"]:
        failures.append("q_top")
    if left["_means"] is None or right["_means"] is None:
        return {
            "init_family": family, "q_top": delta, "d2": None,
            "normalized_weight_delta": None,
            "normalized_weight_delta_se": None,
            "valid": False,
            "failures": sorted(set(failures + ["no_valid_observations"])),
        }
    d2 = character_d2_estimate(
        left["_character_set"], left["_means"], right["_means"],
    )
    weight_delta = abs(
        float(left["normalized_mean_weight"])
        - float(right["normalized_mean_weight"])
    )
    weight_se = math.hypot(
        float(left["normalized_mean_weight_se"]),
        float(right["normalized_mean_weight_se"]),
    )
    gates = config["gates"]
    if max(0.0, d2["d2_norm"]) + 3.0 * d2["d2_total_se"] > gates["max_d2_upper"]:
        failures.append("d2")
    if (weight_delta > gates["max_normalized_weight_delta"]
            or weight_delta > 3.0 * weight_se + 1.0 / num_qubits):
        failures.append("weight")
    b_comparison = _compare_b_summaries(
        left["_b_summary"], right["_b_summary"], config,
    )
    if not b_comparison["valid"]:
        failures.append("b_marginal")
    return {
        "init_family": family, "q_top": delta,
        "d2": {name: value for name, value in d2.items()
               if not name.startswith("per_") and not name.startswith("delete_")},
        "normalized_weight_delta": weight_delta,
        "normalized_weight_delta_se": weight_se,
        "b_marginal": b_comparison,
        "valid": not failures, "failures": sorted(set(failures)),
    }


def analyze_hgp_screen(raw_root, manifest_path, registry_path, config_path,
                       artifact_root, output_path=None, num_workers=1):
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    manifest = json.loads(Path(manifest_path).read_text(encoding="ascii"))
    validate_hgp_screen_manifest(manifest, registry, config, artifact_root)
    raw_root = Path(raw_root)
    expected_paths = {entry["output_relpath"] for entry in manifest["tasks"]}
    actual_paths = {
        str(path.relative_to(raw_root))
        for path in (raw_root / "trajectories").glob("*.npz")
    } if (raw_root / "trajectories").exists() else set()
    if actual_paths != expected_paths:
        raise HgpScreenConflictError(
            f"HGP screen raw set mismatch; missing={len(expected_paths-actual_paths)} "
            f"extra={len(actual_paths-expected_paths)}",
        )
    expected_is_paths = set(manifest["importance_sampling"]["outputs"])
    actual_is_paths = {
        str(path.relative_to(raw_root))
        for path in (raw_root / "importance_sampling").glob("*.npz")
    } if (raw_root / "importance_sampling").exists() else set()
    if actual_is_paths != expected_is_paths:
        raise HgpScreenConflictError(
            "HGP screen IS raw set mismatch; "
            f"missing={len(expected_is_paths-actual_is_paths)} "
            f"extra={len(actual_is_paths-expected_is_paths)}",
        )
    is_diagnostics = []
    for cell, relative_path in zip(
            _map_cells(config), manifest["importance_sampling"]["outputs"]):
        validated_is = validate_hgp_map_is_diagnostic(
            raw_root / relative_path, registry_path, config_path,
            manifest["source_commit"], manifest["archive_sha256"],
            manifest["source_manifest_sha256"], cell, artifact_root,
        )
        is_diagnostics.append({
            "output_relpath": relative_path,
            "sha256": validated_is["sha256"],
            "cell": validated_is["cell"],
            "diagnostics": validated_is["diagnostics"],
            "transcript_sha256": validated_is["transcript_sha256"],
            "used_for_gate_or_selection": False,
        })
    paths = [str(raw_root / entry["output_relpath"]) for entry in manifest["tasks"]]
    if int(num_workers) > 1:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(int(num_workers), len(paths)),
                initializer=_initialize_replay,
                initargs=(
                    registry_path, config_path, manifest["source_commit"],
                    manifest["archive_sha256"],
                    manifest["source_manifest_sha256"], artifact_root,
                )) as pool:
            records = list(pool.map(_validate_raw_worker, paths, chunksize=1))
    else:
        _initialize_replay(
            registry_path, config_path, manifest["source_commit"],
            manifest["archive_sha256"],
            manifest["source_manifest_sha256"], artifact_root,
        )
        records = [_validate_raw_worker(path) for path in paths]
    grouped = defaultdict(list)
    for record in records:
        grouped[(_cell_fingerprint(record["cell"]), record["method_id"])].append(record)
    summaries = []
    internal = {}
    family_internal = {}
    for method in SCREEN_METHODS:
        for cell in _method_cells(config, method):
            values = grouped[(_cell_fingerprint(cell), method)]
            b_families = {}
            for family in INIT_FAMILIES:
                family_records = [
                    row for row in values if row["init_family"] == family
                ]
                b_family = _b_family_summary(family_records, config)
                family_summary = _statistics._family_summary(
                    family_records, config,
                )
                family_summary["_b_summary"] = b_family
                b_families[family] = b_family
                family_internal[(_cell_fingerprint(cell), method, family)] = (
                    family_summary
                )
            summary = _statistics._cell_method_summary(values, config)
            b_summary = _b_cell_summary(b_families, config)
            algorithm_failures = _algorithm_failures(values, method, config["gates"])
            summary["algorithm_metrics"] = {
                "per_trajectory": [value["algorithm_metrics"] for value in sorted(
                    values, key=lambda row: (
                        row["init_family"], row["trajectory_index"],
                    ),
                )],
                "failures": algorithm_failures,
            }
            summary["failures"] = sorted(set(summary["failures"] + algorithm_failures))
            summary["b_marginal"] = b_summary
            if not b_summary["valid"]:
                summary["failures"] = sorted(set(
                    summary["failures"] + ["b_marginal_gate"]
                ))
            summary["valid"] = not summary["failures"]
            internal[(_cell_fingerprint(cell), method)] = summary
            summaries.append(_public_summary(summary))
    method_status = []
    for method in SCREEN_METHODS:
        values = [internal[(_cell_fingerprint(cell), method)]
                  for cell in _method_cells(config, method)]
        method_status.append({
            "method_id": method,
            "cells_passed": sum(bool(value["valid"]) for value in values),
            "cells_total": len(values),
            "core_seconds": float(sum(value["core_seconds"] for value in values)),
            "valid": all(value["valid"] for value in values),
        })
    comparisons = []
    pair_status = []
    for hp in HP_METHODS:
        cell_comparisons = []
        for cell in _cross_mechanism_cells(config):
            cell_key = _cell_fingerprint(cell)
            num_qubits = int(internal[(cell_key, hp)]["num_qubits"])
            for family in INIT_FAMILIES:
                comparison = _compare_family_summaries(
                    family_internal[(cell_key, hp, family)],
                    family_internal[(cell_key, MAP_METHOD_ID, family)],
                    config, family, num_qubits,
                )
                comparison["cell"] = cell
                comparison["hp_method_id"] = hp
                comparison["map_method_id"] = MAP_METHOD_ID
                comparisons.append(comparison)
                cell_comparisons.append(comparison)
        pair_status.append({
            "hp_method_id": hp, "map_method_id": MAP_METHOD_ID,
            "family_cells_passed": sum(
                bool(value["valid"]) for value in cell_comparisons
            ),
            "family_cells_total": len(cell_comparisons),
            "valid": all(value["valid"] for value in cell_comparisons),
        })
    status_by_method = {value["method_id"]: value for value in method_status}
    valid_hp = [method for method in HP_METHODS if status_by_method[method]["valid"]]
    valid_hp.sort(key=lambda method: (status_by_method[method]["core_seconds"], method))
    selected_pair = None
    if not valid_hp:
        status = "UNRESOLVED_NO_HP_PASS"
    elif not status_by_method[MAP_METHOD_ID]["valid"]:
        status = "UNRESOLVED_MAP_MIXTURE_FAIL"
    else:
        selected_hp = valid_hp[0]
        agreement = next(value for value in pair_status
                         if value["hp_method_id"] == selected_hp)
        if agreement["valid"]:
            status = "DIAGNOSTIC_HARD_PAIR_FOUND"
            selected_pair = {
                "hp_method_id": selected_hp,
                "map_method_id": MAP_METHOD_ID,
                "resource_tier": manifest["resource_tier"],
                "agreement_valid": True,
                "agreement_panels": list(CROSS_MECHANISM_PANELS),
                "easy3_scope": "hp_runtime_and_false_negative_control_only",
            }
        else:
            status = "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT"
    identity = {
        "report_version": HGP_SCREEN_REPORT_VERSION,
        "contract_version": HGP_SCREEN_VERSION,
        "source_commit": manifest["source_commit"],
        "archive_sha256": manifest["archive_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "resource_tier": manifest["resource_tier"], "raw_count": len(records),
        # These proposal-overlap diagnostics are evidence only.  No decision
        # branch below or above reads their numerical values.
        "importance_sampling_diagnostics": is_diagnostics,
        "cell_summaries": summaries, "method_status": method_status,
        "comparisons": comparisons, "pair_status": pair_status,
        "selected_pair": selected_pair, "status": status,
        "formal_authorization": False, "production_authorization": False,
        "remaining_required_stages": [
            "FRESH_T_AND_2T_HARD2", "CONF17_RES6_GAP8_SMALL6",
            "M3_TI_ORACLE", "FORMAL_TUNING", "HELD_OUT",
        ],
    }
    report = {**identity, "report_sha256": sha256_json(identity)}
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def _digest_result(result):
    return _value_digest({
        name: value for name, value in result.items()
        if name not in {"engine"}
    })


def hgp_screen_preflight_digest(registry_path, config_path, source_commit,
                                archive_sha256, source_manifest_sha256,
                                artifact_root):
    """Exercise libm, MILP, k=64, and reference/Numba paths before raw exists."""
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    payload = {
        "contract_version": HGP_SCREEN_VERSION,
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["hgp_screen_config_sha256"],
        "cells": [], "tiny_oracles": [], "auxiliary_seed_catalog": [],
    }
    for oracle_index, classical in enumerate((
        np.asarray([[1, 1, 1]], dtype=np.uint8),
        np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    )):
        model, frame = build_model(classical)
        syndrome = np.zeros(model.num_checks, dtype=np.uint8)
        initial = np.zeros(model.num_qubits, dtype=np.uint8)
        cell = {"code_id": "tiny", "p": 0.10, "disorder_index": 0,
                "disorder_source": "preflight"}
        seed = _aux_seed_identity(
            config, registry, source_commit, archive_sha256,
            source_manifest_sha256, "HP32", "T1", cell, "P",
            oracle_index, HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
        )
        payload["auxiliary_seed_catalog"].append({
            "purpose": "tiny_oracle_reference_numba",
            "oracle_index": oracle_index,
            "seed_identity": seed.as_dict(),
        })
        sampler = CollapsedPowerPtConfig("HP32", 0.10, 2, 8)
        reference = run_collapsed_power_pt_trajectory(
            model, frame, classical, syndrome, sampler, seed, initial,
            engine="reference",
        )
        accelerated = run_collapsed_power_pt_trajectory(
            model, frame, classical, syndrome, sampler, seed, initial,
            engine="numba",
        )
        for name in reference:
            if name != "engine" and not np.array_equal(
                    np.asarray(reference[name]), np.asarray(accelerated[name])):
                raise HgpScreenConflictError(
                    f"preflight reference/Numba mismatch: {name}",
                )
        payload["tiny_oracles"].append(_digest_result(reference))
    for cell in _screen_cells(config):
        _, code, H = load_frozen_code(registry_path, cell["code_id"])
        model, frame = build_model(H)
        uniform_seed, epsilon, syndrome = _disorder(registry, code, model, cell)
        row = {
            "cell": cell, "uniform_seed": uniform_seed,
            "syndrome_sha256": _syndrome_sha256(syndrome),
            "mass_sha256": hashlib.sha256(
                build_classical_coset_mass(H, cell["p"]).astype(">f8").tobytes()
            ).hexdigest(),
        }
        b_characters = frozen_b_character_set(
            H.shape[0],
            _b_character_seed(registry["registry_sha256"], code["code_id"]),
            config["b_character_spec"]["dense_count"],
        )
        row["b_character_sha256"] = b_characters.character_sha256
        row["b_character_count"] = b_characters.size
        if cell["code_id"] == "m08_c06":
            for method in HP_METHODS:
                seed = _aux_seed_identity(
                    config, registry, source_commit, archive_sha256,
                    source_manifest_sha256, method, "T1", cell, "P", 0,
                    HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
                )
                payload["auxiliary_seed_catalog"].append({
                    "purpose": "hard_cell_digest",
                    "seed_identity": seed.as_dict(),
                })
                short = CollapsedPowerPtConfig(method, cell["p"], 2, 8)
                row[f"{method}_transcript_sha256"] = _digest_result(
                    run_collapsed_power_pt_trajectory(
                        model, frame, H, syndrome, short, seed, epsilon,
                        engine="numba",
                    )
                )
        if cell in _map_cells(config):
            map_artifact = load_hgp_map_artifact(
                registry_path, config_path, source_commit, archive_sha256,
                source_manifest_sha256, cell, artifact_root,
            )
            catalog = map_artifact.catalog
            proposal = map_artifact.proposal
            row["map_artifact_file_sha256"] = map_artifact.descriptor[
                "artifact_file_sha256"
            ]
            row["map_artifact_content_sha256"] = map_artifact.descriptor[
                "artifact_content_sha256"
            ]
            row["map_anchor_sha256"] = catalog.anchor_sha256
            row["map_proposal_sha256"] = proposal.proposal_sha256
            row["map_solver_identity"] = catalog.solver_identity
        if cell["code_id"] == "m08_c06":
            seed = _aux_seed_identity(
                config, registry, source_commit, archive_sha256,
                source_manifest_sha256, MAP_METHOD_ID, "T1", cell, "P", 0,
                HGP_SCREEN_PREFLIGHT_DIGEST_ROOT,
            )
            payload["auxiliary_seed_catalog"].append({
                "purpose": "hard_cell_digest",
                "seed_identity": seed.as_dict(),
            })
            short = MapMixtureConfig(cell["p"], 8, 8)
            row["map_transcript_sha256"] = _digest_result(
                run_map_mixture_trajectory(
                    model, frame, syndrome, short, seed, epsilon,
                    anchor_catalog=catalog, proposal=proposal,
                )
            )
        payload["cells"].append(row)
    payload["auxiliary_seed_catalog_sha256"] = sha256_json(
        payload["auxiliary_seed_catalog"],
    )
    return {**payload, "canonical_digest": sha256_json(payload)}


def _lpt_makespan(durations, capacity):
    durations = sorted((float(value) for value in durations), reverse=True)
    if (not durations or isinstance(capacity, bool) or int(capacity) <= 0
            or any(not math.isfinite(value) or value < 0.0 for value in durations)):
        if durations:
            raise ValueError("invalid HGP runtime workload")
        return 0.0
    lanes = [0.0] * min(int(capacity), len(durations))
    for duration in durations:
        lane = min(range(len(lanes)), key=lanes.__getitem__)
        lanes[lane] += duration
    return float(max(lanes))


def _owned_task_counts(config):
    result = {
        node: {method: 0 for method in SCREEN_METHODS}
        for node in EXECUTION_NODES
    }
    for method in SCREEN_METHODS:
        for _cell in _method_cells(config, method):
            for _family in INIT_FAMILIES:
                for trajectory in range(TRAJECTORIES_PER_FAMILY):
                    owner = EXECUTION_NODES[trajectory % len(EXECUTION_NODES)]
                    result[owner][method] += 1
    return result


def _b_analysis_counts(config):
    cell_methods = sum(
        len(_method_cells(config, method)) for method in SCREEN_METHODS
    )
    cross = (
        len(HP_METHODS) * len(_cross_mechanism_cells(config))
        * len(INIT_FAMILIES)
    )
    return {
        "family_count": cell_methods * len(INIT_FAMILIES),
        "comparison_count": cell_methods + cross,
    }


def _validate_b_analysis_timings(timings):
    required = {
        "benchmark_measurement_rounds", "trace_benchmark_seconds",
        "trace_seconds_per_round", "family_benchmark_seconds",
        "comparison_benchmark_seconds",
    }
    if not isinstance(timings, dict) or set(timings) != required:
        raise ValueError("B-analysis benchmark schema changed")
    rounds = timings["benchmark_measurement_rounds"]
    values = [
        timings["trace_benchmark_seconds"],
        timings["trace_seconds_per_round"],
        timings["family_benchmark_seconds"],
        timings["comparison_benchmark_seconds"],
    ]
    if (isinstance(rounds, bool) or int(rounds) <= 0
            or any(not math.isfinite(float(value)) or float(value) < 0.0
                   for value in values)
            or not math.isclose(
                float(timings["trace_seconds_per_round"]),
                float(timings["trace_benchmark_seconds"]) / int(rounds),
                rel_tol=1e-12, abs_tol=1e-12,
            )):
        raise ValueError("B-analysis benchmark timing is invalid")
    return {key: timings[key] for key in required}


def _b_analysis_scale(rounds, benchmark_rounds):
    rounds = int(rounds)
    benchmark_rounds = int(benchmark_rounds)
    return (
        rounds * math.log2(max(rounds, 2))
        / (benchmark_rounds * math.log2(max(benchmark_rounds, 2)))
    )


def _runtime_tier_projections(config, timings, is_seconds_by_cell,
                              artifact_generation_seconds,
                              b_analysis_timings):
    safety = float(config["resource_selection"]["safety_factor"])
    sampler_passes = int(
        config["resource_selection"]["full_sampler_passes_per_task"],
    )
    is_passes = int(config["resource_selection"]["full_is_passes_per_cell"])
    owner_counts = _owned_task_counts(config)
    b_analysis_timings = _validate_b_analysis_timings(b_analysis_timings)
    if int(b_analysis_timings["benchmark_measurement_rounds"]) != max(
            value["measurement"] for value in RESOURCE_TIERS.values()):
        raise ValueError("B-analysis benchmark is not the full worst-tier clock")
    b_counts = _b_analysis_counts(config)
    is_owner = {
        node: [
            cell for index, cell in enumerate(_map_cells(config))
            if EXECUTION_NODES[index % len(EXECUTION_NODES)] == node
        ]
        for node in EXECUTION_NODES
    }
    analysis = config["execution"]["analysis"]
    if analysis != {
            "node": ANALYSIS_NODE,
            "capacity": ANALYSIS_CAPACITY,
            "num_workers": ANALYSIS_CAPACITY,
    }:
        raise ValueError("HGP screen analysis placement changed")
    tiers = {}
    for tier, resource in RESOURCE_TIERS.items():
        total_steps = resource["burn"] + resource["measurement"]
        projected = {
            method: (
                timings[method]["seconds_per_step"] * total_steps
                + timings[method]["setup_seconds_per_task"]
            )
            for method in SCREEN_METHODS
        }
        per_node = {}
        for node in EXECUTION_NODES:
            sampler_durations = [
                projected[method]
                for method in SCREEN_METHODS
                for _ in range(owner_counts[node][method])
            ]
            is_durations = [
                is_seconds_by_cell[_cell_fingerprint(value)]
                for value in is_owner[node]
            ]
            capacity = int(config["execution"]["capacities"][node])
            sampler_lpt = _lpt_makespan(sampler_durations, capacity)
            is_lpt = _lpt_makespan(is_durations, capacity)
            generation_wall = sampler_lpt + is_lpt
            per_node[node] = {
                "capacity": capacity,
                "owned_task_counts": owner_counts[node],
                "owned_is_cell_fingerprints": [
                    _cell_fingerprint(value) for value in is_owner[node]
                ],
                "sampler_generation_lpt_seconds": sampler_lpt,
                "sampler_generation_passes_per_task": 1,
                "is_generation_lpt_seconds": is_lpt,
                "is_generation_passes_per_cell": 1,
                "projected_generation_wall_seconds": generation_wall,
            }
        screen_generation_wall = max(
            value["projected_generation_wall_seconds"]
            for value in per_node.values()
        )
        all_sampler_durations = [
            projected[method]
            for node in EXECUTION_NODES
            for method in SCREEN_METHODS
            for _ in range(owner_counts[node][method])
        ]
        analysis_sampler_lpt = _lpt_makespan(
            [
                duration + b_analysis_timings["trace_seconds_per_round"]
                * resource["measurement"]
                for duration in all_sampler_durations
            ],
            int(analysis["capacity"]),
        )
        analysis_is_seconds = float(sum(
            is_seconds_by_cell[_cell_fingerprint(cell)]
            for cell in _map_cells(config)
        ))
        diagnostic_scale = _b_analysis_scale(
            resource["measurement"],
            b_analysis_timings["benchmark_measurement_rounds"],
        )
        b_family_seconds = (
            b_analysis_timings["family_benchmark_seconds"]
            * diagnostic_scale * b_counts["family_count"]
        )
        b_comparison_seconds = (
            b_analysis_timings["comparison_benchmark_seconds"]
            * diagnostic_scale * b_counts["comparison_count"]
        )
        b_diagnostic_seconds = b_family_seconds + b_comparison_seconds
        analysis_wall = (
            analysis_sampler_lpt + analysis_is_seconds + b_diagnostic_seconds
        )
        unsafetied_schedule_seconds = (
            float(artifact_generation_seconds)
            + screen_generation_wall
            + analysis_wall
        )
        schedule_seconds = safety * unsafetied_schedule_seconds
        tiers[tier] = {
            "projected_worst_trajectory_seconds": max(projected.values()),
            "projected_complete_schedule_seconds": schedule_seconds,
            "eligible": bool(
                max(projected.values())
                <= float(config["resource_selection"]["max_trajectory_seconds"])
                and schedule_seconds
                <= float(config["resource_selection"]["screen_budget_seconds"])
            ),
            "per_method_projected_seconds": projected,
            "per_node_generation_workload": per_node,
            "screen_generation_wall_seconds": screen_generation_wall,
            "analysis_workload": {
                "node": analysis["node"],
                "capacity": int(analysis["capacity"]),
                "num_workers": int(analysis["num_workers"]),
                "sampler_task_count": len(all_sampler_durations),
                "sampler_replay_mode": "process_pool_lpt",
                "sampler_replay_lpt_seconds": analysis_sampler_lpt,
                "sampler_replay_passes_per_task": 1,
                "b_trace_postprocess_included_in_sampler_replay": True,
                "b_trace_seconds_per_task": (
                    b_analysis_timings["trace_seconds_per_round"]
                    * resource["measurement"]
                ),
                "is_cell_count": len(_map_cells(config)),
                "is_replay_mode": "serial",
                "is_replay_seconds": analysis_is_seconds,
                "is_replay_passes_per_cell": 1,
                "b_statistical_diagnostics_mode": "single_node_serial",
                "b_family_count": b_counts["family_count"],
                "b_comparison_count": b_counts["comparison_count"],
                "b_diagnostic_scale": diagnostic_scale,
                "b_family_diagnostics_seconds": b_family_seconds,
                "b_comparison_diagnostics_seconds": b_comparison_seconds,
                "b_statistical_diagnostics_seconds": b_diagnostic_seconds,
                "projected_analysis_wall_seconds": analysis_wall,
            },
            "artifact_generation_wall_seconds": float(
                artifact_generation_seconds,
            ),
            "artifact_generation_mode": "single_serial_stage",
            "projected_unsafetied_schedule_seconds": (
                unsafetied_schedule_seconds
            ),
            "full_sampler_passes_per_task": sampler_passes,
            "full_is_passes_per_cell": is_passes,
            "safety_factor": safety,
        }
    return tiers, owner_counts


def _benchmark_b_analysis(H, syndrome, p, character_set, config,
                          benchmark_rounds=None):
    """Time worst-m8 B reconstruction and the serial convergence diagnostics."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    rounds = (
        max(value["measurement"] for value in RESOURCE_TIERS.values())
        if benchmark_rounds is None else int(benchmark_rounds)
    )
    if rounds <= 0:
        raise ValueError("B-analysis benchmark clock must be positive")
    b_width = (r * r + 7) // 8
    total_width = (n * n + r * r + 7) // 8
    if n * n % 8:
        raise HgpScreenConflictError("worst-code A/B packed boundary is not byte aligned")
    byte_offset = n * n // 8
    rng = np.random.Generator(np.random.PCG64(0xB10220260722))
    trace_b = rng.integers(0, 256, size=(rounds, b_width), dtype=np.uint8)
    if r * r % 8:
        trace_b[:, -1] &= np.uint8((1 << (r * r % 8)) - 1)
    full_trace = np.zeros((rounds, total_width), dtype=np.uint8)
    full_trace[:, byte_offset:byte_offset + b_width] = trace_b
    log_mass = np.log(build_classical_coset_mass(H, p))
    start = time.monotonic()
    reconstructed = _extract_b_states_packed(full_trace, n, r)
    trace_weights = _b_weights(reconstructed)
    trace_likelihood = _b_log_likelihood(
        reconstructed, H, syndrome, log_mass,
    )
    _b_character_bits(reconstructed[:1], character_set.masks_packed)
    _b_character_bits(reconstructed[-1:], character_set.masks_packed)
    trace_seconds = time.monotonic() - start
    if (not np.array_equal(reconstructed, trace_b)
            or trace_weights.shape != (rounds,)
            or trace_likelihood.shape != (rounds,)):
        raise HgpScreenConflictError("B-analysis benchmark reconstruction failed")

    families = {}
    family_seconds = []
    for family_index, family in enumerate(INIT_FAMILIES):
        records = []
        for trajectory in range(TRAJECTORIES_PER_FAMILY):
            packed = rng.integers(
                0, 256, size=(rounds, b_width), dtype=np.uint8,
            )
            if r * r % 8:
                packed[:, -1] &= np.uint8((1 << (r * r % 8)) - 1)
            weights = _b_weights(packed)
            # This synthetic trace is used only to time the FFT/Rhat machinery;
            # the real L(B) reconstruction cost is measured separately above.
            likelihood = (
                -0.125 * weights.astype(np.float64)
                + ((np.arange(rounds) + trajectory + family_index) % 17) * 1e-4
            )
            records.append({
                "trajectory_index": trajectory,
                "init_family": family,
                "b_character_set": character_set,
                "b_measurement_states_packed": packed,
                "b_measurement_weights": weights,
                "b_measurement_log_likelihood": likelihood,
                "b_initial_character_bits": _b_character_bits(
                    packed[:1], character_set.masks_packed,
                )[0],
                "b_burn_character_bits": _b_character_bits(
                    packed[-1:], character_set.masks_packed,
                )[0],
                "b_a_factor_count": n,
            })
        start = time.monotonic()
        families[family] = _b_family_summary(records, config)
        family_seconds.append(time.monotonic() - start)
    start = time.monotonic()
    _compare_b_summaries(families["P"], families["U"], config)
    comparison_seconds = time.monotonic() - start
    result = {
        "benchmark_measurement_rounds": rounds,
        "trace_benchmark_seconds": float(trace_seconds),
        "trace_seconds_per_round": float(trace_seconds / rounds),
        "family_benchmark_seconds": float(max(family_seconds)),
        "comparison_benchmark_seconds": float(comparison_seconds),
    }
    return _validate_b_analysis_timings(result)


def benchmark_hgp_screen(registry_path, config_path, source_commit,
                         archive_sha256, source_manifest_sha256,
                         artifact_root):
    """Project all tiers from the worst frozen m8 cell without reading q_top."""
    _require_source_commit(source_commit)
    _require_sha256("archive_sha256", archive_sha256)
    _require_sha256("source_manifest_sha256", source_manifest_sha256)
    registry = _registry_with_path(load_registry(registry_path), registry_path)
    config = load_hgp_screen_config(config_path, registry)
    cell = dict(HARD_CELLS[-1])
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    _, epsilon, syndrome = _disorder(registry, code, model, cell)
    load_start = time.monotonic()
    benchmark_map_artifact = load_hgp_map_artifact(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root,
    )
    map_artifact_load_seconds = time.monotonic() - load_start
    b_characters = frozen_b_character_set(
        H.shape[0],
        _b_character_seed(registry["registry_sha256"], code["code_id"]),
        config["b_character_spec"]["dense_count"],
    )
    b_analysis_timings = _benchmark_b_analysis(
        H, syndrome, cell["p"], b_characters, config,
    )
    timings = {}
    auxiliary_seed_catalog = []
    for method in SCREEN_METHODS:
        warm_seed = _aux_seed_identity(
            config, registry, source_commit, archive_sha256,
            source_manifest_sha256, method, "T1", cell, "P", 1,
            HGP_SCREEN_RUNTIME_WARMUP_ROOT,
        )
        timed_seed = _aux_seed_identity(
            config, registry, source_commit, archive_sha256,
            source_manifest_sha256, method, "T1", cell, "P", 1,
            HGP_SCREEN_RUNTIME_TIMED_ROOT,
        )
        auxiliary_seed_catalog.extend((
            {"purpose": "runtime_warmup", "seed_identity": warm_seed.as_dict()},
            {"purpose": "runtime_timed", "seed_identity": timed_seed.as_dict()},
        ))
        if method in HP_METHODS:
            warm = CollapsedPowerPtConfig(method, cell["p"], 1, 8)
            measured = CollapsedPowerPtConfig(method, cell["p"], 32, 128)
        else:
            warm = MapMixtureConfig(cell["p"], 8, 8)
            measured = MapMixtureConfig(cell["p"], 32, 128)
        map_artifact = (
            benchmark_map_artifact if method == MAP_METHOD_ID else None
        )
        _run_sampler(
            method, model, frame, H, syndrome, warm, warm_seed, epsilon,
            map_artifact,
        )
        start = time.monotonic()
        _run_sampler(
            method, model, frame, H, syndrome, measured, timed_seed, epsilon,
            map_artifact,
        )
        elapsed = time.monotonic() - start
        timings[method] = {
            "benchmark_seconds": elapsed,
            "benchmark_steps": 160,
            "seconds_per_step": elapsed / 160.0,
            "setup_seconds_per_task": (
                map_artifact_load_seconds if method == MAP_METHOD_ID else 0.0
            ),
        }
    artifact_descriptors = load_hgp_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )
    artifact_generation_seconds = float(sum(
        value["generation_wall_seconds"] for value in artifact_descriptors
    ))
    is_seconds_by_cell = {}
    for is_cell in _map_cells(config):
        artifact = load_hgp_map_artifact(
            registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, is_cell, artifact_root,
        )
        is_seed = _map_is_seed(
            source_commit, archive_sha256, source_manifest_sha256, config,
            registry, is_cell, artifact.descriptor, HGP_SCREEN_RUNTIME_IS_ROOT,
        )
        auxiliary_seed_catalog.append({
            "purpose": "importance_sampling_runtime",
            "cell_fingerprint": _cell_fingerprint(is_cell),
            "seed_namespace": HGP_SCREEN_RUNTIME_IS_ROOT,
            "seed": int(is_seed),
        })
        start = time.monotonic()
        _map_is_transcript(
            artifact.proposal, is_cell["p"], HGP_MAP_IS_SAMPLES, is_seed,
        )
        is_seconds_by_cell[_cell_fingerprint(is_cell)] = (
            time.monotonic() - start
        )
    tiers, owner_counts = _runtime_tier_projections(
        config, timings, is_seconds_by_cell, artifact_generation_seconds,
        b_analysis_timings,
    )
    return {
        "version": "exp102.q0_hgp_global.screen.runtime_node.v3",
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["hgp_screen_config_sha256"],
        "benchmark_cell": cell, "timings": timings,
        "map_artifact_load_seconds": map_artifact_load_seconds,
        "artifact_generation_wall_seconds": artifact_generation_seconds,
        "b_analysis_timings": b_analysis_timings,
        "is_seconds_by_cell": is_seconds_by_cell,
        "auxiliary_seed_catalog": auxiliary_seed_catalog,
        "auxiliary_seed_catalog_sha256": sha256_json(auxiliary_seed_catalog),
        "owner_task_counts": owner_counts,
        "tiers": tiers,
    }
