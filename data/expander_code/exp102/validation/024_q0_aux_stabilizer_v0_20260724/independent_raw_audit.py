"""Independent pickle-free raw audit for the UASRE V0 local screen.

This module deliberately never imports the UASRE sampler or its runner.  It
rebuilds the HGP algebra, starts, collapsed traces, score, fixed gates, and
completion digests directly from the frozen manifest and raw NPZ files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

# The audit is also intended to run by file path after a frozen local run.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed

try:
    from numba import njit
except ImportError:  # pragma: no cover - frozen UASRE runs require Numba
    njit = None


CONTRACT_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.v0"
LOCAL_RAW_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.raw.v0"
SAMPLER_RAW_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.raw.v0"
AUDIT_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.independent_raw_audit.v1"
RUNNER_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.local.runner.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
INIT_FAMILIES = ("P", "U", "L")
LOGICAL_MASK_COUNT = 128
B_MASK_COUNT = 64
FULL_ROW_PLAN_VERSION = "exp102.q0_hgp_full_row_plan.v1"

SAMPLER_FIELDS = frozenset({
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "plan_json", "plan_sha256", "initial_state_packed",
    "burn_state_packed", "final_state_packed", "measurement_states_packed",
    "measurement_b_columns", "measurement_a_syndromes", "burn_labels",
    "measurement_labels", "measurement_weights", "measurement_residual_weights",
    "measurement_block", "burn_complete_scores", "measurement_complete_scores",
    "burn_b_weights", "measurement_b_weights", "burn_row_counters",
    "measurement_row_counters", "burn_hot_refresh_changed_bits",
    "measurement_hot_refresh_changed_bits", "burn_auxiliary_counters",
    "measurement_auxiliary_counters", "measurement_auxiliary_assignments",
    "burn_cold_a_column_draws", "measurement_cold_a_column_draws",
    "burn_swap_attempts", "burn_swap_accepts", "measurement_swap_attempts",
    "measurement_swap_accepts", "lambda_values", "lambda_sha256", "mass_sha256",
    "initial_label", "burn_label", "final_label", "engine",
})
RAW_FIELDS = SAMPLER_FIELDS | {
    "sampler_raw_version", "contract_version", "local_raw_version",
    "task_fingerprint", "task_json", "manifest_sha256", "config_sha256",
    "registry_sha256", "source_binding_sha256", "cell_json", "uniform_seed",
    "model_fingerprint", "frame_fingerprint", "logical_masks", "b_masks",
    "character_sha256", "core_seconds", "wall_seconds", "init_family",
    "trajectory_index",
}


class AuditConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditConflict(message)


def _scalar(raw, name):
    value = raw.get(name)
    _require(value is not None and value.shape == (), f"missing scalar raw field: {name}")
    return value.item()


def _array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _float_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=">f8").tobytes()).hexdigest()


def _bits_to_mask(bits):
    value = 0
    for index, bit in enumerate(np.asarray(bits, dtype=np.uint8)):
        value |= int(bit) << index
    return np.uint32(value)


def _unpack_state(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=int(num_qubits), bitorder="little",
    ).astype(np.uint8, copy=False)


def _hgp_check_matrix(H):
    """Build the H_Z matrix for row-major ``H A + B H`` independently."""
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    rows, columns = matrix.shape
    result = np.zeros((rows * columns, columns * columns + rows * rows), dtype=np.uint8)
    for check_row in range(rows):
        for check_column in range(columns):
            target = check_row * columns + check_column
            for a_row in np.flatnonzero(matrix[check_row]):
                result[target, int(a_row) * columns + check_column] ^= 1
            for b_column in np.flatnonzero(matrix[:, check_column]):
                result[target, columns * columns + check_row * rows + int(b_column)] ^= 1
    return result


def _hgp_syndrome(A, B, H):
    return ((np.asarray(H, dtype=np.int64) @ np.asarray(A, dtype=np.int64)
             + np.asarray(B, dtype=np.int64) @ np.asarray(H, dtype=np.int64)) % 2).astype(np.uint8)


def _split_state(state, H):
    rows, columns = np.asarray(H).shape
    state = np.asarray(state, dtype=np.uint8)
    _require(state.shape == (columns * columns + rows * rows,), "HGP state length changed")
    return state[:columns * columns].reshape(columns, columns), state[columns * columns:].reshape(rows, rows)


def _gf2_rank(matrix):
    work = np.asarray(matrix, dtype=np.uint8).copy()
    _require(work.ndim == 2 and np.all((work == 0) | (work == 1)), "GF(2) input is invalid")
    pivot_row = 0
    for column in range(work.shape[1]):
        pivot = next((row for row in range(pivot_row, work.shape[0]) if work[row, column]), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]
        for row in range(work.shape[0]):
            if row != pivot_row and work[row, column]:
                work[row] ^= work[pivot_row]
        pivot_row += 1
        if pivot_row == work.shape[0]:
            break
    return pivot_row


def _label(W_basis, state):
    result = np.uint64(0)
    for bit, row in enumerate(np.asarray(W_basis, dtype=np.uint8)):
        if int(np.bitwise_and(row, state).sum()) & 1:
            result |= np.uint64(1) << np.uint64(bit)
    return result


def _qubit_signatures(W_basis):
    W_basis = np.asarray(W_basis, dtype=np.uint8)
    result = np.zeros(W_basis.shape[1], dtype=np.uint64)
    for qubit in range(W_basis.shape[1]):
        result[qubit] = _label(W_basis, np.eye(1, W_basis.shape[1], qubit, dtype=np.uint8)[0])
    return result


def _reduce_logical_basis(rows):
    result = np.ascontiguousarray(rows, dtype=np.uint8).copy()
    while True:
        best = None
        for left in range(result.shape[0]):
            old_weight = int(result[left].sum())
            for right in range(result.shape[0]):
                if left == right:
                    continue
                improvement = old_weight - int(np.count_nonzero(result[left] ^ result[right]))
                if improvement > 0:
                    candidate = (-improvement, left, right)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            return result
        _, left, right = best
        result[left] ^= result[right]


def _expected_l_start(epsilon, model, W_basis):
    reduced = _reduce_logical_basis(model.logical_move_basis)
    residual = (np.asarray(model.H_check, dtype=np.int64) @ reduced.T.astype(np.int64) % 2).astype(np.uint8)
    _require(not residual.any() and _gf2_rank(reduced) == int(model.k), "reduced L basis is invalid")
    selected = None
    seen = set()
    count = 0
    for order in (1, 2, 3):
        for indices in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(indices)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(_label(W_basis, move))
            _require(signature != 0, "L candidate has a zero signature")
            count += 1
            key = (int((epsilon ^ move).sum()), int(move.sum()), signature, packed)
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    _require(selected is not None, "no L candidate exists")
    key, move = selected
    metadata = {
        "rule": "planted_xor_minimum_energy_reduced_logical_1to3.v1",
        "candidate_orders": [1, 2, 3],
        "candidate_count": count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": hashlib.sha256(move.tobytes()).hexdigest(),
    }
    return np.ascontiguousarray(epsilon ^ move), move, metadata


def _uniform_start(model, syndrome, seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    state = model.logical_sector_section.apply(syndrome, strict=True).astype(np.uint8)
    rng = PortablePrng(int(seed))
    for row in np.asarray(model.stabilizer_rows, dtype=np.uint8):
        if rng.randbelow(2):
            state ^= row
    for row in np.asarray(model.logical_move_basis, dtype=np.uint8):
        if rng.randbelow(2):
            state ^= row
    return np.ascontiguousarray(state)


def _uniform_seed(registry, code, cell):
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(cell["disorder_index"]), "uniforms",
    )


def _uniform_start_seed(identity):
    return derive_seed(
        CONTRACT_VERSION, identity["trajectory_namespace"], identity["source_commit"],
        identity["config_sha256"], identity["registry_sha256"], identity["cell_fingerprint"],
        identity["method_id"], identity["resource_tier"], identity["init_family"],
        int(identity["trajectory_index"]), "initialize", "hard_coset", 0,
    )


def _frozen_masks(config_sha256, registry_sha256, rows):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    seed = derive_seed(
        CONTRACT_VERSION, "fixed_characters", config_sha256, registry_sha256,
        "uint64_bit63_safe_v1",
    )
    rng = PortablePrng(seed)
    basis = np.asarray([np.uint64(1) << np.uint64(bit) for bit in range(64)], dtype=np.uint64)
    seen = {int(value) for value in basis}
    nonbasis = []
    while len(nonbasis) < LOGICAL_MASK_COUNT - basis.size:
        value = int(rng.next_uint64())
        if value and value not in seen:
            seen.add(value)
            nonbasis.append(np.uint64(value))
    b_masks = np.zeros((B_MASK_COUNT, rows), dtype=np.uint32)
    row_mask = (1 << rows) - 1
    for index in range(B_MASK_COUNT):
        while not b_masks[index].any():
            for column in range(rows):
                b_masks[index, column] = np.uint32(int(rng.next_uint64()) & row_mask)
    return np.concatenate((basis, np.asarray(nonbasis, dtype=np.uint64))), b_masks


def _column_masks(H):
    return np.asarray([_bits_to_mask(np.asarray(H)[:, column]) for column in range(H.shape[1])], dtype=np.uint32)


if njit is not None:
    @njit(cache=False)
    def _mass_recurrence(column_masks, rows, p):
        size = 1 << rows
        current = np.zeros(size, dtype=np.float64)
        scratch = np.empty(size, dtype=np.float64)
        current[0] = 1.0
        keep = 1.0 - p
        for column in range(column_masks.size):
            mask = int(column_masks[column])
            for syndrome in range(size):
                scratch[syndrome] = keep * current[syndrome] + p * current[syndrome ^ mask]
            temporary = current
            current = scratch
            scratch = temporary
        return current
else:  # pragma: no cover
    _mass_recurrence = None


def _classical_mass(H, p):
    _require(_mass_recurrence is not None, "Numba is unavailable for independent mass audit")
    result = _mass_recurrence(_column_masks(H), int(H.shape[0]), float(p))
    _require(np.all(np.isfinite(result)) and np.all(result > 0.0), "independent mass is invalid")
    _require(abs(float(result.sum()) - 1.0) <= 5e-13, "independent mass is unnormalized")
    return np.ascontiguousarray(result)


def _collapsed_trace(state, syndrome, H):
    A, B = _split_state(state, H)
    rows, columns = H.shape
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    _require(np.array_equal(_hgp_syndrome(A, B, H), Y), "state is outside the HGP hard coset")
    b_columns = np.asarray([_bits_to_mask(B[:, column]) for column in range(rows)], dtype=np.uint32)
    a_syndromes = np.asarray([_bits_to_mask(Y[:, column]) for column in range(columns)], dtype=np.uint32)
    for b_row, b_column in enumerate(b_columns):
        for factor in np.flatnonzero(H[b_row]):
            a_syndromes[int(factor)] ^= b_column
    for column in range(columns):
        actual = _bits_to_mask((H.astype(np.int64) @ A[:, column].astype(np.int64) % 2).astype(np.uint8))
        _require(actual == a_syndromes[column], "collapsed A syndrome is inconsistent")
    return b_columns, a_syndromes


def _complete_score(b_columns, a_syndromes, log_mass, log_odds):
    result = 0.0
    for value in np.asarray(b_columns, dtype=np.uint32):
        result += int(value).bit_count() * float(log_odds)
    for value in np.asarray(a_syndromes, dtype=np.uint32):
        result += float(log_mass[int(value)])
    _require(math.isfinite(result), "independent complete score is non-finite")
    return result


def _cosine_ladder(replicas):
    values = np.asarray([
        0.5 * (1.0 - math.cos(math.pi * index / (int(replicas) - 1)))
        for index in range(int(replicas))
    ], dtype=np.float64)
    values[0], values[-1] = 0.0, 1.0
    return values


def _expected_swap_attempts(replicas, offset, rounds):
    values = np.zeros(int(replicas) - 1, dtype=np.int64)
    for round_index in range(int(offset), int(offset) + int(rounds)):
        for lower in range(round_index & 1, int(replicas) - 1, 2):
            values[lower] += 1
    return values


def _h_matrix_sha256(H):
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    return sha256_json({
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "bits_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    })


def _plan_dict(H):
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    rows, columns = matrix.shape
    factor_scopes = [list(map(int, np.flatnonzero(matrix[:, column]))) for column in range(columns)]
    adjacency = [set() for _ in range(rows)]
    for scope in factor_scopes:
        for index, left in enumerate(scope):
            for right in scope[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    remaining = set(range(rows))
    order, widths, buckets = [], [], []
    while remaining:
        choices = []
        for variable in sorted(remaining):
            neighbors = sorted(adjacency[variable] & remaining)
            fill = sum(
                int(right not in adjacency[left])
                for index, left in enumerate(neighbors) for right in neighbors[index + 1:]
            )
            choices.append((fill, len(neighbors), variable, neighbors))
        _, _, variable, neighbors = min(choices, key=lambda item: item[:3])
        order.append(variable)
        widths.append(len(neighbors))
        buckets.append(sorted((variable, *neighbors)))
        for index, left in enumerate(neighbors):
            adjacency[left].discard(variable)
            for right in neighbors[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        adjacency[variable].clear()
        remaining.remove(variable)
    return {
        "plan_version": FULL_ROW_PLAN_VERSION,
        "tie_break": "min_fill_then_degree_then_variable_index",
        "matrix_sha256": _h_matrix_sha256(matrix),
        "rows": int(rows), "columns": int(columns), "factor_scopes": factor_scopes,
        "order": order, "widths": widths, "bucket_scopes": buckets,
        "max_width": int(max(widths, default=0)),
        "max_table_entries": int(1 << (max(widths, default=0) + 1)),
        "structural_table_cells": int(sum(1 << len(scope) for scope in buckets)),
        "output_table_cells": int(sum(1 << (len(scope) - 1) for scope in buckets)),
    }


def _sampler_config(method, config):
    replicas = int(method["num_replicas"])
    return {
        "method_id": method["id"], "p": float(config["cell"]["p"]),
        "burn_rounds": int(config["resource"]["burn_rounds"]),
        "measurement_rounds": int(config["resource"]["measurement_rounds"]),
        "num_replicas": replicas,
        "positive_row_updates_per_round": int(method["positive_row_updates_per_round"]),
        "cold_auxiliary_rows_per_round": int(method["cold_auxiliary_rows_per_round"]),
        "lambda_values": _cosine_ladder(replicas).tolist(),
        "lambda_schedule": "cosine_endpoint_cluster_v1",
        "kernel": "uniform_anchor_complete_energy_with_cold_auxiliary_stabilizer_heatbath.v1",
        "hot_endpoint": "exact_uniform_B_refresh",
        "tempered_term": "complete_collapsed_log_density",
        "cold_block_kernel": "exact_auxiliary_A_row_stabilizer_heatbath.v1",
        "cold_block_schedule": "post_swap_cyclic_A_rows.v1",
    }


def _source_paths(config_path):
    return {
        "config.py": EXP102_ROOT / "exp102_pipeline" / "config.py",
        "diagnostics.py": EXP102_ROOT / "exp102_pipeline" / "diagnostics.py",
        "exp101_bridge.py": EXP102_ROOT / "exp102_pipeline" / "exp101_bridge.py",
        "io.py": EXP102_ROOT / "exp102_pipeline" / "io.py",
        "labels.py": EXP102_ROOT / "exp102_pipeline" / "labels.py",
        "q0_global.py": EXP102_ROOT / "exp102_pipeline" / "q0_global.py",
        "q0_hgp_aux_stabilizer.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py",
        "q0_hgp_aux_stabilizer_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer_pt.py",
        "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
        "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        "q0_pa.py": EXP102_ROOT / "exp102_pipeline" / "q0_pa.py",
        "q0_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_pt.py",
        "registry.py": EXP102_ROOT / "exp102_pipeline" / "registry.py",
        "seeds.py": EXP102_ROOT / "exp102_pipeline" / "seeds.py",
        "worker.py": EXP102_ROOT / "exp102_pipeline" / "worker.py",
        "independent_raw_audit.py": ROOT / "independent_raw_audit.py",
        "run_local_viability.py": ROOT / "run_local_viability.py",
        "config": Path(config_path),
        "registry.json": REGISTRY_PATH,
    }


def _validate_source_binding(manifest):
    binding = manifest["source_binding"]
    _require(set(binding) == {"source_commit", "files", "exp101_src_files", "runtime", "source_binding_sha256"},
             "source binding schema changed")
    core = {name: binding[name] for name in ("source_commit", "files", "exp101_src_files", "runtime")}
    _require(binding["source_binding_sha256"] == sha256_json(core), "source binding hash changed")
    current_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(current_commit == binding["source_commit"], "source commit changed after freeze")
    expected_files = {name: sha256_file(path) for name, path in _source_paths(
        EXP102_ROOT / "config" / "q0_hgp_aux_stabilizer_pt.v0.json",
    ).items()}
    _require(binding["files"] == expected_files, "bound source files changed after freeze")
    exp101_root = EXP102_ROOT.parent / "exp101" / "src"
    expected_exp101 = {
        path.relative_to(exp101_root).as_posix(): sha256_file(path)
        for path in sorted(exp101_root.rglob("*.py"))
    }
    _require(binding["exp101_src_files"] == expected_exp101, "bound exp101 source changed after freeze")
    numba_version = "missing" if njit is None else __import__("numba").__version__
    expected_runtime = {
        "numpy": str(np.__version__), "numba": str(numba_version), "python": sys.version.split()[0],
    }
    _require(binding["runtime"] == expected_runtime, "runtime identity changed after freeze")


def _load_manifest(root):
    manifest = json.loads((Path(root) / "MANIFEST.json").read_text(encoding="ascii"))
    required = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "epsilon_sha256", "model_fingerprint", "frame_fingerprint",
        "l_start", "plan_sha256", "plan_json", "mass_sha256", "character_sha256",
        "control_npz_sha256", "tasks", "manifest_sha256",
    }
    _require(set(manifest) == required, "manifest schema changed")
    core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    _require(manifest["manifest_sha256"] == sha256_json(core), "manifest hash changed")
    _require(manifest["manifest_version"] == "exp102.q0_hgp_aux_stabilizer_pt.local.manifest.v1",
             "manifest version changed")
    _require(manifest["contract_version"] == CONTRACT_VERSION and manifest["raw_version"] == LOCAL_RAW_VERSION,
             "manifest contract changed")
    _require(len(manifest["tasks"]) == 48, "manifest task count changed")
    seen = set()
    for task in manifest["tasks"]:
        core_task = {name: value for name, value in task.items() if name != "task_fingerprint"}
        _require(task.get("task_fingerprint") == sha256_json(core_task), "task fingerprint changed")
        _require(task["task_fingerprint"] not in seen, "task fingerprint is duplicated")
        seen.add(task["task_fingerprint"])
    return manifest


def _validate_completion(root, manifest):
    complete = json.loads((Path(root) / "RUN_COMPLETE.json").read_text(encoding="ascii"))
    required = {"runner_version", "manifest_sha256", "raw_count", "raw", "run_sha256"}
    _require(set(complete) == required, "completion schema changed")
    core = {name: value for name, value in complete.items() if name != "run_sha256"}
    _require(complete["run_sha256"] == sha256_json(core), "completion hash changed")
    _require(complete["runner_version"] == RUNNER_VERSION, "completion runner changed")
    _require(complete["manifest_sha256"] == manifest["manifest_sha256"], "completion manifest changed")
    _require(complete["raw_count"] == len(manifest["tasks"]), "completion raw count changed")
    expected = {
        f'{task["method_id"]}_{task["init_family"]}_{int(task["trajectory_index"]):02d}.npz': task["task_fingerprint"]
        for task in manifest["tasks"]
    }
    hashes = {}
    _require(isinstance(complete["raw"], list) and len(complete["raw"]) == len(expected),
             "completion raw list changed")
    for item in complete["raw"]:
        _require(isinstance(item, dict) and set(item) == {"filename", "sha256", "task_fingerprint"},
                 "completion raw item changed")
        name = item["filename"]
        _require(name in expected and name not in hashes and item["task_fingerprint"] == expected[name],
                 "completion raw identity changed")
        path = Path(root) / "raw" / name
        _require(path.is_file() and sha256_file(path) == item["sha256"], "completion raw digest changed")
        hashes[name] = item["sha256"]
    _require(set(hashes) == set(expected), "completion raw set changed")
    return hashes


def _build_model(H):
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame

    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def _load_context(manifest, control):
    registry = load_registry(REGISTRY_PATH)
    config = manifest["config"]
    expected_config = {
        "cell", "config_version", "contract_version", "gates", "init_families", "l_start_rule",
        "methods", "raw_version", "registry_sha256", "resource", "runtime_selection", "scope",
        "selection", "trajectories_per_family", "trajectory_namespace",
    }
    _require(set(config) == expected_config and sha256_json(config) == manifest["config_sha256"],
             "config identity changed")
    _require(config["contract_version"] == CONTRACT_VERSION and config["raw_version"] == LOCAL_RAW_VERSION,
             "config contract changed")
    _require(config["registry_sha256"] == registry["registry_sha256"] == manifest["registry_sha256"],
             "registry identity changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0, "disorder_source": "attempt022", "p": 0.04,
    }, "hard cell changed")
    _require(config["init_families"] == list(INIT_FAMILIES)
             and config["l_start_rule"] == "planted_xor_minimum_energy_reduced_logical_1to3.v1",
             "initialization contract changed")
    _require(config["resource"] == {"burn_rounds": 256, "measurement_rounds": 2048, "name": "V0"},
             "resource tier changed")
    _require(config["trajectories_per_family"] == 8 and config["methods"] == [
        {"id": "UASRE32-R1-A1", "num_replicas": 32,
         "positive_row_updates_per_round": 1, "cold_auxiliary_rows_per_round": 1},
        {"id": "UASRE64-R1-A1", "num_replicas": 64,
         "positive_row_updates_per_round": 1, "cold_auxiliary_rows_per_round": 1},
    ], "method roster changed")
    _require(config["gates"] == {
        "max_abs_B_mask_mean_delta": 0.12,
        "max_abs_complete_score_delta_per_factor": 0.12,
        "max_abs_logical_character_mean_delta": 0.12,
        "max_abs_normalized_B_weight_delta": 0.08,
        "max_abs_normalized_weight_delta": 0.08,
        "max_negligible_support_upper_bound": 0.001,
        "minimum_effective_trajectory_count": 4,
        "sigma_multiplier": 3.0,
        "sigma_slack": 0.01,
    }, "gate values changed")
    _, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    H = np.ascontiguousarray(H, dtype=np.uint8)
    model, frame = _build_model(H)
    expected_check = _hgp_check_matrix(H)
    _require(np.array_equal(model.H_check, expected_check), "HGP check wiring changed")
    _require(model.sector == "x_error" and model.k == 64, "model sector or logical dimension changed")
    _require(_gf2_rank(np.vstack((model.stabilizer_rows, model.logical_move_basis)))
             == model.num_qubits - _gf2_rank(expected_check), "hard-coset basis is not bijective")
    uniform_seed = _uniform_seed(registry, code, config["cell"])
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits) < float(config["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (expected_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    _require(syndrome.any(), "hard sentinel syndrome vanished")
    _require(set(control) == {"epsilon", "syndrome", "l_move", "logical_masks", "b_masks", "classical_mass"},
             "control schema changed")
    _require(np.array_equal(control["epsilon"], epsilon) and np.array_equal(control["syndrome"], syndrome),
             "control disorder changed")
    l_start, l_move, l_metadata = _expected_l_start(epsilon, model, frame.W_basis)
    _require(np.array_equal(control["l_move"], l_move) and manifest["l_start"] == l_metadata,
             "control L start changed")
    logical_masks, b_masks = _frozen_masks(manifest["config_sha256"], registry["registry_sha256"], H.shape[0])
    _require(np.array_equal(control["logical_masks"], logical_masks)
             and np.array_equal(control["b_masks"], b_masks), "control character masks changed")
    mass = _classical_mass(H, config["cell"]["p"])
    _require(np.array_equal(control["classical_mass"], mass), "control mass table changed")
    _require(manifest["H_sha256"] == _array_sha256(H)
             and manifest["syndrome_sha256"] == _array_sha256(syndrome)
             and manifest["epsilon_sha256"] == _array_sha256(epsilon), "manifest algebra hashes changed")
    plan = _plan_dict(H)
    _require(manifest["plan_json"] == plan and manifest["plan_sha256"] == sha256_json(plan),
             "manifest elimination plan changed")
    _require(manifest["mass_sha256"] == _float_sha256(mass), "manifest mass hash changed")
    character_sha = sha256_json({"logical": _array_sha256(logical_masks), "b": _array_sha256(b_masks)})
    _require(manifest["character_sha256"] == character_sha, "manifest character hash changed")
    _require(manifest["uniform_seed"] == int(uniform_seed), "manifest uniform seed changed")
    _require(manifest["model_fingerprint"] == model.fingerprint() and manifest["frame_fingerprint"] == frame.fingerprint(),
             "manifest model frame changed")
    return {
        "registry": registry, "config": config, "H": H, "model": model, "frame": frame,
        "epsilon": epsilon, "syndrome": syndrome, "l_start": l_start, "l_move": l_move,
        "mass": mass, "uniform_seed": uniform_seed, "logical_masks": logical_masks, "b_masks": b_masks,
        "plan": plan, "character_sha": character_sha,
    }


def _initial_state(task, context):
    family = task["init_family"]
    if family == "P":
        return context["epsilon"].copy()
    if family == "L":
        return np.ascontiguousarray(context["epsilon"] ^ context["l_move"])
    if family == "U":
        return _uniform_start(context["model"], context["syndrome"], _uniform_start_seed(task["seed_identity"]))
    raise AuditConflict("unknown initialization family")


def _validate_raw(path, task, manifest, context):
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    _require(set(raw) == RAW_FIELDS, f"raw schema changed: {path.name}")
    _require(not any(value.dtype.hasobject for value in raw.values()), "object raw field is forbidden")
    method = next((item for item in context["config"]["methods"] if item["id"] == task["method_id"]), None)
    _require(method is not None, "raw method is not in the frozen config")
    sampler = _sampler_config(method, context["config"])
    expected = {
        "raw_version": SAMPLER_RAW_VERSION, "sampler_raw_version": SAMPLER_RAW_VERSION,
        "contract_version": CONTRACT_VERSION, "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"], "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
        "registry_sha256": context["registry"]["registry_sha256"],
        "source_binding_sha256": manifest["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(context["config"]["cell"]), "method_id": task["method_id"],
        "init_family": task["init_family"], "trajectory_index": int(task["trajectory_index"]),
        "engine": "numba", "model_fingerprint": context["model"].fingerprint(),
        "frame_fingerprint": context["frame"].fingerprint(), "character_sha256": context["character_sha"],
        "plan_sha256": sha256_json(context["plan"]), "mass_sha256": _float_sha256(context["mass"]),
        "lambda_sha256": _float_sha256(np.asarray(sampler["lambda_values"], dtype=np.float64)),
    }
    for name, value in expected.items():
        _require(str(_scalar(raw, name)) == str(value), f"raw identity changed: {path.name}:{name}")
    _require(int(_scalar(raw, "uniform_seed")) == context["uniform_seed"], "raw uniform seed changed")
    _require(str(_scalar(raw, "sampler_config_json")) == canonical_json(sampler)
             and str(_scalar(raw, "sampler_config_sha256")) == sha256_json(sampler),
             "raw sampler configuration changed")
    _require(str(_scalar(raw, "seed_identity_json")) == canonical_json(task["seed_identity"]),
             "raw seed identity changed")
    _require(str(_scalar(raw, "plan_json")) == canonical_json(context["plan"]), "raw elimination plan changed")
    _require(np.array_equal(raw["lambda_values"], np.asarray(sampler["lambda_values"], dtype=np.float64)),
             "raw lambda ladder changed")
    _require(np.array_equal(raw["logical_masks"], context["logical_masks"])
             and np.array_equal(raw["b_masks"], context["b_masks"]), "raw masks changed")
    for name in ("core_seconds", "wall_seconds"):
        value = float(_scalar(raw, name))
        _require(math.isfinite(value) and value >= 0.0, f"raw timing is invalid: {name}")
    _require(float(_scalar(raw, "wall_seconds")) <= float(context["config"]["runtime_selection"]["max_predicted_trajectory_seconds"]),
             "raw wall time exceeds the frozen cap")
    initial = _initial_state(task, context)
    _require(np.array_equal(_unpack_state(raw["initial_state_packed"], context["model"].num_qubits), initial),
             "raw initial state changed")
    for name, label_name in (("initial_state_packed", "initial_label"), ("burn_state_packed", "burn_label"),
                             ("final_state_packed", "final_label")):
        state = _unpack_state(raw[name], context["model"].num_qubits)
        residual = (context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8) ^ context["syndrome"]
        _require(not residual.any(), f"hard-coset violation: {name}")
        _require(_label(context["frame"].W_basis, state) == np.uint64(_scalar(raw, label_name)),
                 f"state label changed: {label_name}")
    burn = int(context["config"]["resource"]["burn_rounds"])
    count = int(context["config"]["resource"]["measurement_rounds"])
    _require(raw["burn_labels"].shape == (burn,) and raw["burn_complete_scores"].shape == (burn,)
             and raw["burn_b_weights"].shape == (burn,) and np.all(np.isfinite(raw["burn_complete_scores"])),
             "burn trace shape changed")
    measurement = np.unpackbits(raw["measurement_states_packed"], axis=1, count=context["model"].num_qubits,
                                bitorder="little").astype(np.uint8, copy=False)
    _require(measurement.shape == (count, context["model"].num_qubits), "measurement state shape changed")
    residuals = (context["model"].H_check.astype(np.int64) @ measurement.T.astype(np.int64) % 2).T.astype(np.uint8) ^ context["syndrome"][None, :]
    _require(not residuals.any() and not raw["measurement_residual_weights"].any(), "measurement left hard coset")
    labels = np.asarray([_label(context["frame"].W_basis, state) for state in measurement], dtype=np.uint64)
    _require(np.array_equal(labels, raw["measurement_labels"]), "measurement label trace changed")
    _require(np.array_equal(measurement.sum(axis=1).astype(np.int32), raw["measurement_weights"]),
             "measurement weight trace changed")
    _require(raw["measurement_complete_scores"].shape == (count,)
             and raw["measurement_b_weights"].shape == (count,)
             and np.all(np.isfinite(raw["measurement_complete_scores"])), "measurement score shape changed")
    _require(np.array_equal(raw["measurement_block"], np.repeat(np.arange(8, dtype=np.int8), count // 8)),
             "measurement block schedule changed")
    rows, columns = context["H"].shape
    _require(raw["measurement_b_columns"].shape == (count, rows)
             and raw["measurement_a_syndromes"].shape == (count, columns), "collapsed trace shape changed")
    log_mass = np.log(context["mass"])
    log_odds = math.log(float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"])))
    scores = np.empty(count, dtype=np.float64)
    b_weights = np.empty(count, dtype=np.int32)
    for index, state in enumerate(measurement):
        b_columns, a_syndromes = _collapsed_trace(state, context["syndrome"], context["H"])
        _require(np.array_equal(raw["measurement_b_columns"][index], b_columns)
                 and np.array_equal(raw["measurement_a_syndromes"][index], a_syndromes),
                 "collapsed trace changed")
        scores[index] = _complete_score(b_columns, a_syndromes, log_mass, log_odds)
        b_weights[index] = sum(int(value).bit_count() for value in b_columns)
    _require(np.array_equal(scores, raw["measurement_complete_scores"])
             and np.array_equal(b_weights, raw["measurement_b_weights"]), "complete score trace changed")
    replicas = int(method["num_replicas"])
    updates = int(method["positive_row_updates_per_round"])
    auxiliary_rows = int(method["cold_auxiliary_rows_per_round"])
    for phase, rounds, offset in (("burn", burn, 0), ("measurement", count, burn)):
        counters = raw[f"{phase}_row_counters"]
        _require(counters.shape == (replicas, 3) and not counters[0].any(), "row counter shape changed")
        _require(int(counters[:, 0].sum()) == rounds * (replicas - 1) * updates,
                 "row attempt counter changed")
        _require(np.all(counters[:, 1] <= counters[:, 0]) and np.all(counters[:, 2] >= counters[:, 1]),
                 "row counter values changed")
        auxiliary = raw[f"{phase}_auxiliary_counters"]
        _require(auxiliary.shape == (3,) and int(auxiliary[0]) == rounds * auxiliary_rows,
                 "auxiliary attempt counter changed")
        _require(0 <= int(auxiliary[1]) <= int(auxiliary[0]) and int(auxiliary[2]) >= int(auxiliary[1]),
                 "auxiliary counter values changed")
        hot = raw[f"{phase}_hot_refresh_changed_bits"]
        _require(hot.shape == (rounds,) and np.all((hot >= 0) & (hot <= rows * rows)),
                 "hot refresh trace changed")
        _require(int(_scalar(raw, f"{phase}_cold_a_column_draws")) == rounds * columns,
                 "cold A redraw count changed")
        _require(np.array_equal(raw[f"{phase}_swap_attempts"], _expected_swap_attempts(replicas, offset, rounds)),
                 "swap attempts changed")
        _require(np.all(raw[f"{phase}_swap_accepts"] <= raw[f"{phase}_swap_attempts"]), "swap accepts changed")
    assignments = raw["measurement_auxiliary_assignments"]
    _require(assignments.shape == (count, auxiliary_rows) and np.all(assignments < (1 << rows)),
             "auxiliary assignment trace changed")
    _require(int(sum(int(value).bit_count() for value in assignments.flat))
             == int(raw["measurement_auxiliary_counters"][2]), "auxiliary assignment count changed")
    return raw


def _parity_uint64(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        values ^= values >> np.uint64(shift)
    return values & np.uint64(1)


def _parity_uint32(values):
    values = np.asarray(values, dtype=np.uint32).copy()
    for shift in (16, 8, 4, 2, 1):
        values ^= values >> np.uint32(shift)
    return values & np.uint32(1)


def _trajectory_values(raw, context):
    labels = np.asarray(raw["measurement_labels"], dtype=np.uint64)
    logical = 1.0 - 2.0 * _parity_uint64(labels[:, None] & context["logical_masks"][None, :])
    b_columns = np.asarray(raw["measurement_b_columns"], dtype=np.uint32)
    b_signs = np.empty((b_columns.shape[0], context["b_masks"].shape[0]), dtype=np.float64)
    for mask_index, mask in enumerate(context["b_masks"]):
        parity = np.zeros(b_columns.shape[0], dtype=np.uint32)
        for column in range(b_columns.shape[1]):
            parity ^= _parity_uint32(b_columns[:, column] & mask[column])
        b_signs[:, mask_index] = 1.0 - 2.0 * parity
    half = labels.size // 2
    series = {
        "normalized_weight": np.asarray(raw["measurement_weights"], dtype=np.float64) / context["model"].num_qubits,
        "normalized_B_weight": np.asarray(raw["measurement_b_weights"], dtype=np.float64) / (context["H"].shape[0] ** 2),
        "complete_score_per_factor": np.asarray(raw["measurement_complete_scores"], dtype=np.float64) / context["H"].shape[1],
        "logical_characters": logical,
        "B_masks": b_signs,
    }
    return {
        name: np.asarray((value.mean(axis=0), value[:half].mean(axis=0), value[half:].mean(axis=0)))
        for name, value in series.items()
    }


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def _compare(left, right, absolute_bound, gates):
    mean_left, se_left = _mean_se(left)
    mean_right, se_right = _mean_se(right)
    delta = np.abs(mean_left - mean_right)
    allowance = float(gates["sigma_multiplier"]) * np.sqrt(se_left * se_left + se_right * se_right) + float(gates["sigma_slack"])
    return {
        "pass": bool(np.all(delta <= float(absolute_bound)) and np.all(delta <= allowance)),
        "max_abs_delta": float(np.max(delta)), "max_three_sigma_allowance": float(np.max(allowance)),
        "failed_components": int(np.count_nonzero((delta > float(absolute_bound)) | (delta > allowance))),
    }


def _bundle(left, right, gates):
    return {
        "normalized_weight": _compare(left["normalized_weight"], right["normalized_weight"], gates["max_abs_normalized_weight_delta"], gates),
        "normalized_B_weight": _compare(left["normalized_B_weight"], right["normalized_B_weight"], gates["max_abs_normalized_B_weight_delta"], gates),
        "complete_score_per_factor": _compare(left["complete_score_per_factor"], right["complete_score_per_factor"], gates["max_abs_complete_score_delta_per_factor"], gates),
        "logical_characters": _compare(left["logical_characters"], right["logical_characters"], gates["max_abs_logical_character_mean_delta"], gates),
        "B_masks": _compare(left["B_masks"], right["B_masks"], gates["max_abs_B_mask_mean_delta"], gates),
    }


def _bundle_pass(bundle):
    return bool(all(item["pass"] for item in bundle.values()))


def _method_summary(records, context):
    by_family = {family: [] for family in INIT_FAMILIES}
    for record in records:
        by_family[str(_scalar(record, "init_family"))].append(record)
    means, halves = {}, {}
    for family in INIT_FAMILIES:
        _require(len(by_family[family]) == context["config"]["trajectories_per_family"],
                 f"incomplete initialization family: {family}")
        values = [_trajectory_values(record, context) for record in by_family[family]]
        means[family] = {name: np.stack([item[name][0] for item in values], axis=0) for name in values[0]}
        halves[family] = {
            name: (np.stack([item[name][1] for item in values], axis=0),
                   np.stack([item[name][2] for item in values], axis=0))
            for name in values[0]
        }
    gates = context["config"]["gates"]
    pairwise = {
        f"{left}_{right}": _bundle(means[left], means[right], gates)
        for left, right in (("P", "U"), ("P", "L"), ("U", "L"))
    }
    time_stability = {
        family: _bundle(
            {name: pair[0] for name, pair in halves[family].items()},
            {name: pair[1] for name, pair in halves[family].items()},
            gates,
        )
        for family in INIT_FAMILIES
    }
    log_odds = math.log(float(context["config"]["cell"]["p"]) / (1.0 - float(context["config"]["cell"]["p"])))
    reference_weight = int(context["epsilon"].sum())
    dimension = int(context["model"].num_qubits - _gf2_rank(context["model"].H_check))
    support = []
    for record in by_family["U"]:
        minimum = int(np.asarray(record["measurement_weights"]).min())
        log_bound = dimension * math.log(2.0) + (minimum - reference_weight) * log_odds
        bound = 0.0 if log_bound < -745.0 else min(1.0, math.exp(log_bound))
        support.append({
            "minimum_measurement_weight": minimum, "target_support_upper_bound": bound,
            "trapped_negligible_support": bool(bound <= gates["max_negligible_support_upper_bound"]),
        })
    counts = {family: len(by_family[family]) for family in INIT_FAMILIES}
    passed = (
        all(value >= gates["minimum_effective_trajectory_count"] for value in counts.values())
        and not any(item["trapped_negligible_support"] for item in support)
        and all(_bundle_pass(item) for item in pairwise.values())
        and all(_bundle_pass(item) for item in time_stability.values())
    )
    return {
        "pass": bool(passed), "independent_trajectory_count": counts,
        "target_support_gate": {
            "reference_legal_weight": reference_weight, "hard_coset_dimension": dimension,
            "per_trajectory": support,
            "pass": not any(item["trapped_negligible_support"] for item in support),
        },
        "pairwise": pairwise, "time_stability": time_stability,
        "q_top_read_or_computed": False,
    }


def audit(root):
    root = Path(root)
    output = root / "INDEPENDENT_AUDIT.json"
    _require(not output.exists(), "independent audit output already exists")
    manifest = _load_manifest(root)
    _validate_source_binding(manifest)
    _require((root / "RUN_COMPLETE.json").is_file(), "run is incomplete")
    completion_hashes = _validate_completion(root, manifest)
    with np.load(root / "CONTROL.npz", allow_pickle=False) as archive:
        control = {name: archive[name].copy() for name in archive.files}
    _require(sha256_file(root / "CONTROL.npz") == manifest["control_npz_sha256"], "control file hash changed")
    context = _load_context(manifest, control)
    methods = {method["id"]: [] for method in context["config"]["methods"]}
    raw_hashes = {}
    for task in manifest["tasks"]:
        name = f'{task["method_id"]}_{task["init_family"]}_{int(task["trajectory_index"]):02d}.npz'
        path = root / "raw" / name
        _require(path.is_file(), f"raw file is missing: {name}")
        methods[task["method_id"]].append(_validate_raw(path, task, manifest, context))
        raw_hashes[name] = sha256_file(path)
    _require(raw_hashes == completion_hashes, "raw hashes changed after completion")
    summaries = {name: _method_summary(records, context) for name, records in methods.items()}
    passing = [name for name, value in summaries.items() if value["pass"]]
    if not passing:
        selected, status = None, context["config"]["selection"]["zero_pass"]
    elif len(passing) == 1:
        selected, status = passing[0], "LOCAL_AUXILIARY_STABILIZER_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    else:
        selected, status = "UASRE32-R1-A1", "LOCAL_AUXILIARY_STABILIZER_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
    core = {
        "audit_version": AUDIT_VERSION, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"], "status": status,
        "selected_method": selected, "formal_authorization": False, "remote_authorization": False,
        "posterior_estimation": False, "q_top_read_or_computed": False,
        "methods": summaries, "raw_sha256": raw_hashes,
        "audit_source_sha256": sha256_file(Path(__file__)),
    }
    report = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_hard_viability")
    args = parser.parse_args(argv)
    print(audit(args.root)["status"])


if __name__ == "__main__":
    main()
