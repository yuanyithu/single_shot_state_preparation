#!/usr/bin/env python3
"""Independent raw-only audit for the frozen full-row Gibbs V0 screen.

This verifier intentionally imports neither the full-row Gibbs sampler nor the
local runner.  It rebuilds the HGP algebra, P/U/L starts, labels, packed-state
traces, collapsed B/A traces, counters, manifests, and the transport gate from
saved NPZ files opened with ``allow_pickle=False``.  It does not generate a
single new MCMC transition.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
PROJECT_ROOT = ROOT.parents[4]
RUN_ROOT = ROOT / "local_hard_viability_001"
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
CONFIG_PATH = EXP102_ROOT / "config" / "q0_hgp_full_row_gibbs.v0.json"

sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


CONTRACT_VERSION = "exp102.q0_hgp_full_row_gibbs.v0"
LOCAL_RAW_VERSION = "exp102.q0_hgp_full_row_gibbs.local.raw.v0"
SAMPLER_RAW_VERSION = "exp102.q0_hgp_full_row_gibbs.raw.v0"
FULL_ROW_VERSION = "exp102.q0_hgp_full_row_gibbs.v0"
FULL_ROW_KERNEL = "exact_collapsed_full_row_variable_elimination.v1"
FULL_ROW_METHOD = "FRG-VE1"
L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
AUDIT_VERSION = "exp102.q0_hgp_full_row_gibbs.v0.independent_raw_audit.v1"
EXPECTED_MANIFEST_SHA256 = "430659be5aac3b1fe099b2c15eadda194878beba663fb7b11874fc05b4bf69a7"

RAW_FIELDS = {
    "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
    "seed_identity_json", "plan_json", "plan_sha256", "mass_sha256",
    "initial_state_packed", "burn_state_packed", "final_state_packed",
    "measurement_states_packed", "measurement_b_columns",
    "measurement_a_syndromes", "burn_labels", "measurement_labels",
    "measurement_weights", "measurement_residual_weights", "measurement_block",
    "burn_counters", "measurement_counters", "burn_basis_seen", "initial_label",
    "burn_label", "final_label", "engine", "contract_version",
    "local_raw_version", "task_fingerprint", "task_json", "manifest_sha256",
    "config_sha256", "registry_sha256", "source_binding_sha256", "cell_json",
    "uniform_seed", "model_fingerprint", "frame_fingerprint", "character_masks",
    "character_sha256", "core_seconds", "wall_seconds", "sampler_raw_version",
    "init_family", "trajectory_index",
}


class AuditConflict(RuntimeError):
    """A frozen artifact or a claimed result is inconsistent."""


def require(condition, message):
    if not condition:
        raise AuditConflict(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scalar(value, name):
    array = np.asarray(value)
    require(array.shape == (), f"{name} must be a scalar")
    return array.item()


def array_sha256(values):
    """Match the frozen runner's unambiguous ndarray digest exactly."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def big_endian_float64_sha256(values):
    """Hash the large mass table without materializing a second full copy."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    digest = hashlib.sha256()
    block = 1 << 20
    for start in range(0, array.size, block):
        digest.update(np.asarray(array[start:start + block], dtype=">f8").tobytes())
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise AuditConflict(f"cannot read {path}: {exc}") from exc


def read_npz(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise AuditConflict(f"cannot load {path}: {exc}") from exc
    require(not any(value.dtype.hasobject for value in values.values()),
            f"object dtype is forbidden in {Path(path).name}")
    return values


def build_model(H):
    """Build the certified HGP sector directly, without the experiment worker."""
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame

    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def direct_hgp_syndrome(state, H):
    """Evaluate H_Z e through the tensor formula, not model.H_check."""
    H = np.asarray(H, dtype=np.uint8)
    rows, columns = H.shape
    state = np.asarray(state, dtype=np.uint8)
    require(state.shape == (columns * columns + rows * rows,), "state length changed")
    A = state[:columns * columns].reshape(columns, columns)
    B = state[columns * columns:].reshape(rows, rows)
    return ((H.astype(np.int64) @ A.astype(np.int64)
             + B.astype(np.int64) @ H.astype(np.int64)) % 2).astype(np.uint8)


def manual_label(W_basis, state):
    bits = (
        np.asarray(W_basis, dtype=np.int64) @ np.asarray(state, dtype=np.int64)
        % 2
    ).astype(np.uint8)
    result = 0
    for bit, entry in enumerate(bits):
        if int(entry):
            result |= 1 << bit
    return np.uint64(result)


def pack_state(state):
    return np.packbits(np.asarray(state, dtype=np.uint8), bitorder="little")


def unpack_one(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)


def unpack_states(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=1, count=num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)


def reduce_logical_basis(rows):
    """Reproduce the frozen deterministic local reduction rule independently."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8).copy()
    while True:
        best = None
        for left in range(rows.shape[0]):
            old_weight = int(rows[left].sum())
            for right in range(rows.shape[0]):
                if left == right:
                    continue
                improvement = old_weight - int(np.count_nonzero(rows[left] ^ rows[right]))
                if improvement <= 0:
                    continue
                candidate = (-improvement, left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return rows
        _, left, right = best
        rows[left] ^= rows[right]


def select_l_move(epsilon, model, W_basis, H):
    """Recompute the frozen logical adversarial start without chain output."""
    reduced = reduce_logical_basis(model.logical_move_basis)
    seen = set()
    selected = None
    candidate_count = 0
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = pack_state(move).tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(manual_label(W_basis, move))
            require(signature != 0, "L-start candidate has zero logical signature")
            require(not direct_hgp_syndrome(move, H).any(),
                    "L-start candidate is outside the HGP kernel")
            candidate_count += 1
            key = (int((epsilon ^ move).sum()), int(move.sum()), signature, packed)
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    require(selected is not None, "no L-start candidate exists")
    key, move = selected
    return move, {
        "rule": L_START_RULE,
        "candidate_orders": [1, 2, 3],
        "candidate_count": candidate_count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": hashlib.sha256(move.tobytes()).hexdigest(),
    }


def h_matrix_sha256(H):
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    return sha256_json({
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "bits_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    })


def independent_plan_json(H):
    """Rebuild the state-independent min-fill plan from the base matrix."""
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    rows, columns = matrix.shape
    factor_scopes = [
        [int(value) for value in np.flatnonzero(matrix[:, column])]
        for column in range(columns)
    ]
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
                right not in adjacency[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
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
    max_width = max(widths, default=0)
    return {
        "plan_version": "exp102.q0_hgp_full_row_plan.v1",
        "tie_break": "min_fill_then_degree_then_variable_index",
        "matrix_sha256": h_matrix_sha256(matrix),
        "rows": rows,
        "columns": columns,
        "factor_scopes": factor_scopes,
        "order": order,
        "widths": widths,
        "bucket_scopes": buckets,
        "max_width": max_width,
        "max_table_entries": 1 << (max_width + 1),
        "structural_table_cells": int(sum(1 << len(scope) for scope in buckets)),
        "output_table_cells": int(sum(1 << (len(scope) - 1) for scope in buckets)),
    }


def seed_from_identity(identity, stage, role="stream", index=0):
    return derive_seed(
        FULL_ROW_VERSION,
        identity["trajectory_namespace"], identity["source_commit"],
        identity["config_sha256"], identity["registry_sha256"],
        identity["cell_fingerprint"], identity["method_id"],
        identity["resource_tier"], identity["init_family"],
        int(identity["trajectory_index"]), str(stage), str(role), int(index),
    )


def uniform_hard_coset_state(model, syndrome, seed):
    """Recreate the K=0 affine draw directly from its certified basis."""
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    state = model.logical_sector_section.apply(syndrome, strict=True).astype(np.uint8)
    rng = PortablePrng(int(seed))
    for row in model.stabilizer_rows:
        if rng.randbelow(2):
            state ^= row
    for row in model.logical_move_basis:
        if rng.randbelow(2):
            state ^= row
    return np.ascontiguousarray(state, dtype=np.uint8)


def initial_state(task, context):
    family = task["init_family"]
    if family == "P":
        result = context["epsilon"].copy()
    elif family == "U":
        seed = seed_from_identity(task["seed_identity"], "initialize", "hard_coset")
        result = uniform_hard_coset_state(context["model"], context["syndrome"], seed)
    elif family == "L":
        result = context["epsilon"] ^ context["l_move"]
    else:
        raise AuditConflict(f"unknown initialization family {family}")
    require(np.array_equal(
        direct_hgp_syndrome(result, context["H"]).reshape(-1), context["syndrome"],
    ), f"{family} initial state leaves the hard coset")
    return result


def labels_to_basis_seen(labels, k):
    seen = np.zeros((k, 2), dtype=np.uint8)
    for label in np.asarray(labels, dtype=np.uint64):
        for bit in range(k):
            seen[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = 1
    return seen


def gf2_rank_uint64(values):
    pivots = {}
    for raw in np.asarray(values, dtype=np.uint64):
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def basis_leave_return(labels, bit):
    values = ((np.asarray(labels, dtype=np.uint64) >> np.uint64(bit)) & np.uint64(1))
    initial = int(values[0])
    left = False
    for value in values[1:]:
        if int(value) != initial:
            left = True
        elif left:
            return True
    return False


def b_columns_and_a_syndromes(state, syndrome, H):
    """Recover the collapsed coordinates using the HGP tensor identities."""
    H = np.asarray(H, dtype=np.uint8)
    rows, columns = H.shape
    state = np.asarray(state, dtype=np.uint8)
    B = state[columns * columns:].reshape(rows, rows)
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    b_columns = np.zeros(rows, dtype=np.uint32)
    a_syndromes = np.zeros(columns, dtype=np.uint32)
    for column in range(rows):
        for row in range(rows):
            b_columns[column] |= np.uint32(B[row, column]) << np.uint32(row)
    for column in range(columns):
        value = np.uint32(0)
        for row in range(rows):
            value |= np.uint32(Y[row, column]) << np.uint32(row)
        for row in np.flatnonzero(H[:, column]):
            value ^= b_columns[int(row)]
        a_syndromes[column] = value
    A = state[:columns * columns].reshape(columns, columns)
    for column in range(columns):
        value = np.uint32(0)
        syndrome_column = (H.astype(np.int64) @ A[:, column].astype(np.int64) % 2)
        for row in range(rows):
            value |= np.uint32(syndrome_column[row]) << np.uint32(row)
        require(value == a_syndromes[column], "A/B collapsed factorization changed")
    return b_columns, a_syndromes


def validate_source_binding(manifest):
    source = manifest["source_binding"]
    expected_files = {
        "q0_hgp_full_row_gibbs.py": sha256_file(
            EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        ),
        "q0_hgp_collapsed.py": sha256_file(
            EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        ),
        "run_local_viability.py": sha256_file(ROOT / "run_local_viability.py"),
        "config": sha256_file(CONFIG_PATH),
    }
    core = {"source_commit": source["source_commit"], "files": expected_files}
    expected = {**core, "source_binding_sha256": sha256_json(core)}
    require(source == expected, "frozen source binding changed")
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    require(commit == source["source_commit"], "source commit changed after raw generation")


def load_context(manifest):
    config = read_json(CONFIG_PATH)
    require(manifest["config"] == config, "manifest config diverges from frozen config")
    require(manifest["config_sha256"] == sha256_json(config), "config SHA changed")
    validate_source_binding(manifest)
    registry = load_registry(REGISTRY_PATH)
    require(registry["registry_sha256"] == manifest["registry_sha256"], "registry SHA changed")
    _, code, H = load_frozen_code(REGISTRY_PATH, manifest["cell"]["code_id"])
    H = np.ascontiguousarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    require(model.num_qubits == int(code["n"]) and model.k == int(code["k"]),
            "registry/model dimensions changed")
    uniform_seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(manifest["cell"]["disorder_index"]), "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(manifest["cell"]["p"])
    ).astype(np.uint8)
    syndrome_matrix = direct_hgp_syndrome(epsilon, H)
    syndrome = syndrome_matrix.reshape(-1)
    model_syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    require(np.array_equal(syndrome, model_syndrome), "HGP H_Z orientation changed")
    require(bool(syndrome.any()), "planted syndrome unexpectedly vanishes")
    require(array_sha256(H) == manifest["H_sha256"], "H matrix changed")
    require(array_sha256(epsilon) == manifest["epsilon_sha256"], "planted error changed")
    require(array_sha256(syndrome) == manifest["syndrome_sha256"], "syndrome changed")
    require(int(uniform_seed) == int(manifest["uniform_seed"]), "uniform seed changed")
    require(model.fingerprint() == manifest["model_fingerprint"], "model fingerprint changed")
    require(frame.fingerprint() == manifest["frame_fingerprint"], "frame fingerprint changed")
    expected_masks = np.asarray(
        [np.uint64(1) << np.uint64(bit) for bit in range(model.k)], dtype=np.uint64,
    )
    require(model.k == 64 and expected_masks.size == 64, "logical dimension changed")
    require(manifest["character_masks"] == [int(value) for value in expected_masks],
            "character masks changed")
    require(manifest["character_sha256"] == array_sha256(expected_masks),
            "character SHA changed")
    l_move, l_start = select_l_move(epsilon, model, frame.W_basis, H)
    require(l_start == manifest["l_start"], "L-start construction changed")
    require(int(manual_label(frame.W_basis, epsilon ^ l_move)) != int(manual_label(frame.W_basis, epsilon)),
            "L start is not logically separated")
    plan = independent_plan_json(H)
    require(plan == manifest["plan_json"], "min-fill plan changed")
    require(sha256_json(plan) == manifest["plan_sha256"], "min-fill plan SHA changed")
    return {
        "config": config,
        "registry": registry,
        "code": code,
        "H": H,
        "model": model,
        "frame": frame,
        "epsilon": epsilon,
        "syndrome": syndrome,
        "uniform_seed": uniform_seed,
        "l_move": l_move,
        "characters": expected_masks,
    }


def validate_control(manifest, context):
    path = RUN_ROOT / "CONTROL.npz"
    require(sha256_file(path) == manifest["control_npz_sha256"], "control NPZ changed")
    control = read_npz(path)
    require(set(control) == {"epsilon", "syndrome", "l_move", "character_masks", "classical_mass"},
            "control NPZ schema changed")
    require(np.array_equal(control["epsilon"], context["epsilon"]), "control epsilon changed")
    require(np.array_equal(control["syndrome"], context["syndrome"]), "control syndrome changed")
    require(np.array_equal(control["l_move"], context["l_move"]), "control L move changed")
    require(np.array_equal(control["character_masks"], context["characters"]), "control characters changed")
    mass = np.asarray(control["classical_mass"], dtype=np.float64)
    rows = context["H"].shape[0]
    require(mass.shape == (1 << rows,) and np.isfinite(mass).all() and np.all(mass > 0.0),
            "control mass table changed")
    require(big_endian_float64_sha256(mass) == manifest["mass_sha256"], "control mass SHA changed")


def validate_manifest(manifest):
    expected = {
        "manifest_version", "contract_version", "raw_version", "config", "config_sha256",
        "registry_sha256", "source_binding", "cell", "uniform_seed", "H_sha256",
        "syndrome_sha256", "epsilon_sha256", "model_fingerprint", "frame_fingerprint",
        "character_masks", "character_sha256", "l_start", "plan_sha256", "plan_json",
        "mass_sha256", "control_npz_sha256", "tasks", "manifest_sha256",
    }
    require(set(manifest) == expected, "manifest schema changed")
    core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(core), "manifest SHA changed")
    require(manifest["manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "foreign manifest")
    require(manifest["contract_version"] == CONTRACT_VERSION, "manifest contract changed")
    require(manifest["raw_version"] == LOCAL_RAW_VERSION, "manifest raw version changed")
    require(len(manifest["tasks"]) == 24, "manifest task count changed")
    require(len({task["task_fingerprint"] for task in manifest["tasks"]}) == 24,
            "manifest task identities are not unique")
    families = [task["init_family"] for task in manifest["tasks"]]
    require(families.count("P") == families.count("U") == families.count("L") == 8,
            "manifest initialization families changed")


def validate_task(task, context, manifest):
    path = RUN_ROOT / "raw" / f"{task['init_family']}_{int(task['trajectory_index']):02d}.npz"
    require(path.is_file(), f"raw is missing: {path.name}")
    raw = read_npz(path)
    require(set(raw) == RAW_FIELDS, f"raw schema changed: {path.name}")
    core_task = {name: value for name, value in task.items() if name != "task_fingerprint"}
    require(task["task_fingerprint"] == sha256_json(core_task), f"task SHA changed: {path.name}")
    expected_scalars = {
        "raw_version": LOCAL_RAW_VERSION,
        "sampler_raw_version": SAMPLER_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "local_raw_version": LOCAL_RAW_VERSION,
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"],
        "registry_sha256": manifest["registry_sha256"],
        "source_binding_sha256": manifest["source_binding"]["source_binding_sha256"],
        "cell_json": canonical_json(manifest["cell"]),
        "model_fingerprint": manifest["model_fingerprint"],
        "frame_fingerprint": manifest["frame_fingerprint"],
        "character_sha256": manifest["character_sha256"],
        "method_id": FULL_ROW_METHOD,
        "engine": "numba",
        "plan_json": canonical_json(manifest["plan_json"]),
        "plan_sha256": manifest["plan_sha256"],
        "mass_sha256": manifest["mass_sha256"],
        "init_family": task["init_family"],
        "trajectory_index": int(task["trajectory_index"]),
    }
    for name, expected in expected_scalars.items():
        require(str(scalar(raw[name], name)) == str(expected),
                f"raw {name} changed: {path.name}")
    require(int(scalar(raw["uniform_seed"], "uniform_seed")) == int(context["uniform_seed"]),
            f"raw uniform seed changed: {path.name}")
    sampler_config = {
        "method_id": FULL_ROW_METHOD,
        "p": float(manifest["cell"]["p"]),
        "burn_sweeps": int(manifest["config"]["resource"]["burn_sweeps"]),
        "measurement_sweeps": int(manifest["config"]["resource"]["measurement_sweeps"]),
        "row_schedule": "ascending",
        "kernel": FULL_ROW_KERNEL,
        "a_update": "exact_column_conditional_draw_after_each_B_sweep",
    }
    require(str(scalar(raw["sampler_config_json"], "sampler_config_json")) == canonical_json(sampler_config),
            f"sampler config changed: {path.name}")
    require(str(scalar(raw["sampler_config_sha256"], "sampler_config_sha256")) == sha256_json(sampler_config),
            f"sampler config SHA changed: {path.name}")
    require(str(scalar(raw["seed_identity_json"], "seed_identity_json"))
            == canonical_json(task["seed_identity"]), f"seed identity changed: {path.name}")
    require(np.array_equal(raw["character_masks"], context["characters"]),
            f"raw characters changed: {path.name}")
    expected_initial = initial_state(task, context)
    require(np.array_equal(raw["initial_state_packed"], pack_state(expected_initial)),
            f"raw initial state changed: {path.name}")
    require(int(scalar(raw["initial_label"], "initial_label"))
            == int(manual_label(context["frame"].W_basis, expected_initial)),
            f"raw initial label changed: {path.name}")

    num_qubits = context["model"].num_qubits
    for name in ("initial_state_packed", "burn_state_packed", "final_state_packed"):
        state = unpack_one(raw[name], num_qubits)
        require(np.array_equal(direct_hgp_syndrome(state, context["H"]).reshape(-1), context["syndrome"]),
                f"{name} leaves hard coset: {path.name}")
    burn_state = unpack_one(raw["burn_state_packed"], num_qubits)
    final_state = unpack_one(raw["final_state_packed"], num_qubits)
    require(int(scalar(raw["burn_label"], "burn_label"))
            == int(manual_label(context["frame"].W_basis, burn_state)),
            f"burn label changed: {path.name}")
    require(int(scalar(raw["final_label"], "final_label"))
            == int(manual_label(context["frame"].W_basis, final_state)),
            f"final label changed: {path.name}")

    states = unpack_states(raw["measurement_states_packed"], num_qubits)
    measurement_sweeps = int(manifest["config"]["resource"]["measurement_sweeps"])
    require(states.shape == (measurement_sweeps, num_qubits), f"measurement shape changed: {path.name}")
    labels = np.asarray([
        manual_label(context["frame"].W_basis, state) for state in states
    ], dtype=np.uint64)
    require(np.array_equal(labels, raw["measurement_labels"]), f"labels changed: {path.name}")
    require(np.array_equal(states.sum(axis=1).astype(np.int32), raw["measurement_weights"]),
            f"weights changed: {path.name}")
    direct_residuals = np.asarray([
        np.count_nonzero(direct_hgp_syndrome(state, context["H"]).reshape(-1) ^ context["syndrome"])
        for state in states
    ], dtype=np.int32)
    require(np.array_equal(direct_residuals, raw["measurement_residual_weights"]),
            f"residual weights changed: {path.name}")
    require(not direct_residuals.any(), f"measurement leaves hard coset: {path.name}")
    require(np.array_equal(raw["measurement_block"], np.repeat(
        np.arange(8, dtype=np.int8), measurement_sweeps // 8,
    )), f"measurement blocks changed: {path.name}")

    b_trace = np.asarray(raw["measurement_b_columns"], dtype=np.uint32)
    a_trace = np.asarray(raw["measurement_a_syndromes"], dtype=np.uint32)
    rows, columns = context["H"].shape
    require(b_trace.shape == (measurement_sweeps, rows), f"B trace shape changed: {path.name}")
    require(a_trace.shape == (measurement_sweeps, columns), f"A trace shape changed: {path.name}")
    for index, state in enumerate(states):
        b_columns, a_syndromes = b_columns_and_a_syndromes(state, context["syndrome"], context["H"])
        require(np.array_equal(b_columns, b_trace[index]), f"B trace changed: {path.name}")
        require(np.array_equal(a_syndromes, a_trace[index]), f"A trace changed: {path.name}")

    burn_labels = np.asarray(raw["burn_labels"], dtype=np.uint64)
    require(burn_labels.shape == (int(manifest["config"]["resource"]["burn_sweeps"]),),
            f"burn label trace shape changed: {path.name}")
    require(np.array_equal(raw["burn_basis_seen"], labels_to_basis_seen(burn_labels, context["model"].k)),
            f"burn character coverage changed: {path.name}")
    burn_counters = np.asarray(raw["burn_counters"], dtype=np.int64)
    measurement_counters = np.asarray(raw["measurement_counters"], dtype=np.int64)
    require(burn_counters.shape == measurement_counters.shape == (5,),
            f"counter shape changed: {path.name}")
    burn_sweeps = int(manifest["config"]["resource"]["burn_sweeps"])
    require(int(burn_counters[0]) == burn_sweeps * rows,
            f"burn row-update counter changed: {path.name}")
    require(int(measurement_counters[0]) == measurement_sweeps * rows,
            f"measurement row-update counter changed: {path.name}")
    require(int(burn_counters[3]) == burn_sweeps * columns,
            f"burn A-draw counter changed: {path.name}")
    require(int(measurement_counters[3]) == measurement_sweeps * columns,
            f"measurement A-draw counter changed: {path.name}")
    for counters in (burn_counters, measurement_counters):
        require(0 <= int(counters[1]) <= int(counters[0]), f"row-change count invalid: {path.name}")
        require(int(counters[1]) <= int(counters[2]) <= rows * int(counters[1]),
                f"row-bit counter invalid: {path.name}")
        require(0 <= int(counters[4]) <= int(counters[3]), f"A-change counter invalid: {path.name}")
    b_change_steps = int(np.count_nonzero(np.any(b_trace[1:] != b_trace[:-1], axis=1)))
    b_changed_bits = int(np.count_nonzero(b_trace[1:] != b_trace[:-1]))
    require(b_change_steps <= int(measurement_counters[1]), f"B changes exceed row changes: {path.name}")
    require(b_changed_bits <= int(measurement_counters[2]),
            f"B bit changes exceed row-bit counter: {path.name}")
    require(np.isfinite(float(scalar(raw["core_seconds"], "core_seconds")))
            and np.isfinite(float(scalar(raw["wall_seconds"], "wall_seconds"))),
            f"timing is non-finite: {path.name}")
    return {
        "path": path,
        "labels": labels,
        "b_trace": b_trace,
        "raw_sha256": sha256_file(path),
    }


def family_summary(records, gates):
    label_traces = [record["labels"] for record in records]
    changes = [int(np.count_nonzero(trace[1:] != trace[:-1])) for trace in label_traces]
    deltas = np.concatenate([trace[1:] ^ trace[:-1] for trace in label_traces])
    returns = np.asarray([
        any(basis_leave_return(trace, bit) for trace in label_traces)
        for bit in range(64)
    ], dtype=np.uint8)
    b_changes = [int(np.count_nonzero(np.any(trace[1:] != trace[:-1], axis=1)))
                 for trace in [record["b_trace"] for record in records]]
    valid = (
        sum(changes) >= gates["minimum_measurement_label_changes_per_family"]
        and sum(value >= 4 for value in changes)
        >= gates["minimum_chains_with_four_measurement_label_changes_per_family"]
        and gf2_rank_uint64(deltas) >= gates["minimum_measurement_label_delta_rank_per_family"]
        and bool(returns.all())
    )
    return {
        "measurement_label_changes": int(sum(changes)),
        "per_chain_label_changes": changes,
        "chains_with_at_least_four_changes": int(sum(value >= 4 for value in changes)),
        "label_delta_rank": gf2_rank_uint64(deltas),
        "basis_leave_return_count": int(returns.sum()),
        "basis_leave_returns": returns.tolist(),
        "measurement_B_state_changes": int(sum(b_changes)),
        "per_chain_B_state_changes": b_changes,
        "transport_gate_pass": bool(valid),
    }


def main():
    output = RUN_ROOT / "INDEPENDENT_AUDIT.json"
    require(not output.exists(), "independent audit already exists")
    manifest = read_json(RUN_ROOT / "MANIFEST.json")
    report = read_json(RUN_ROOT / "REPORT.json")
    run_complete = read_json(RUN_ROOT / "RUN_COMPLETE.json")
    validate_manifest(manifest)
    require(report["manifest_sha256"] == manifest["manifest_sha256"], "report uses foreign manifest")
    report_core = {name: value for name, value in report.items() if name != "report_sha256"}
    require(report["report_sha256"] == sha256_json(report_core), "report SHA changed")
    require(run_complete["manifest_sha256"] == manifest["manifest_sha256"], "run completion uses foreign manifest")
    run_core = {name: value for name, value in run_complete.items() if name != "run_sha256"}
    require(run_complete["run_sha256"] == sha256_json(run_core), "run completion SHA changed")
    context = load_context(manifest)
    validate_control(manifest, context)
    expected_paths = {
        RUN_ROOT / "raw" / f"{task['init_family']}_{int(task['trajectory_index']):02d}.npz"
        for task in manifest["tasks"]
    }
    require(set((RUN_ROOT / "raw").glob("*.npz")) == expected_paths,
            "raw files are missing or unexpected")
    records = [validate_task(task, context, manifest) for task in manifest["tasks"]]
    complete_hashes = {entry["filename"]: entry["sha256"] for entry in run_complete["raw"]}
    require(len(complete_hashes) == len(records), "completion raw count changed")
    raw_hashes = {record["path"].name: record["raw_sha256"] for record in records}
    require(raw_hashes == complete_hashes, "completion raw hashes changed")
    require(raw_hashes == report["raw_sha256"], "report raw hashes changed")
    gates = manifest["config"]["gates"]
    summaries = {
        family: family_summary(
            [record for record, task in zip(records, manifest["tasks"])
             if task["init_family"] == family], gates,
        )
        for family in manifest["config"]["init_families"]
    }
    require(summaries == report["family"], "independent family summary disagrees with report")
    status = (
        "LOCAL_LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN"
        if all(summary["transport_gate_pass"] for summary in summaries.values())
        else "LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE"
    )
    require(status == report["status"], "transport status disagrees with report")
    core = {
        "audit_version": AUDIT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "status": "INDEPENDENT_RAW_AUDIT_PASS",
        "sampler_imported": False,
        "sampler_called": False,
        "raw_count": len(records),
        "transport_status": status,
        "family": summaries,
    }
    atomic_json(output, {**core, "audit_sha256": sha256_json(core)})
    print(canonical_json({**core, "audit_sha256": sha256_json(core)}))


if __name__ == "__main__":
    main()
