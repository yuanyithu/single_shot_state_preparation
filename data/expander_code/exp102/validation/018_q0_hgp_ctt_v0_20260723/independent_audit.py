#!/usr/bin/env python3
"""Independent raw-only audit for the frozen CTT V0 local preflight.

This does not invoke the CTT transition kernel. The runner already reruns each
trajectory from seed; this second implementation independently rebuilds the
cell, initial states, labels, raw schema, deterministic digests, and transport
gate directly from saved NPZ files with allow_pickle=False.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    frozen_character_set,
    reduce_logical_basis,
    state_label,
    uniform_hard_coset_state,
    unpack_states,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    COLLAPSED_TT_COUNTER_NAMES,
    COLLAPSED_TT_KERNEL,
    COLLAPSED_TT_RAW_VERSION,
    CollapsedTemperedTransitionSeedIdentity,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "local_m8_transport_ctt_v0"
MANIFEST_SHA256 = "f77add0a8b1825b117ac49ed85b3a3a138045cb233bed43fb691cac9bd31ff85"
RAW_VERSION = "exp102.q0_hgp_ctt.v0.raw.v1"
CONTRACT_VERSION = "exp102.q0_hgp_ctt.v0"
L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
AUDIT_VERSION = "exp102.q0_hgp_ctt.v0.independent_raw_audit.v1"


class AuditConflict(RuntimeError):
    """A raw artifact or its reported conclusion cannot be trusted."""


def require(condition, message):
    if not condition:
        raise AuditConflict(message)


def scalar(value, name):
    array = np.asarray(value)
    require(array.shape == (), f"{name} must be scalar")
    return array.item()


def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def engine_digest(raw):
    """Recompute the runner digest without importing the runner itself."""
    fields = {name[4:]: raw[name] for name in raw if name.startswith("ctt_")}
    digest = hashlib.sha256()
    digest.update(b"exp102.q0_hgp_ctt.v0.trajectory_digest.v1\0")
    for name in sorted(fields):
        value = np.asarray(fields[name])
        require(not value.dtype.hasobject, f"CTT engine field {name} has object dtype")
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        dtype = value.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def rank_masks(masks, k):
    pivots = {}
    for raw in masks:
        value = int(np.uint64(raw))
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    require(len(pivots) <= k, "label-delta rank exceeds logical dimension")
    return len(pivots)


def character_signs(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    result = np.empty((labels.size, masks.size), dtype=np.int8)
    for row, label in enumerate(labels):
        for column, mask in enumerate(masks):
            result[row, column] = 1 if (int(label & mask).bit_count() & 1) == 0 else -1
    return result


def leave_return(labels, masks):
    signs = character_signs(labels, masks)
    origin = signs[0]
    left = np.zeros(signs.shape[1], dtype=bool)
    returned = np.zeros(signs.shape[1], dtype=bool)
    for values in signs[1:]:
        changed = values != origin
        returned |= left & ~changed
        left |= changed
    return returned


def pack_state(state):
    return np.packbits(np.asarray(state, dtype=np.uint8), bitorder="little")


def unpack_one(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)


def float64_big_endian_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype=">f8").tobytes(order="C")).hexdigest()


def basis_seen(labels, k):
    seen = np.zeros((k, 2), dtype=np.uint8)
    for label in np.asarray(labels, dtype=np.uint64):
        for bit in range(k):
            seen[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = 1
    return seen


def select_l_move(epsilon, model, frame):
    """Independently reproduce the frozen deterministic L-start rule."""
    reduced = reduce_logical_basis(model.logical_move_basis)
    seen = set()
    selected = None
    candidate_count = 0
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(state_label(frame, move))
            require(signature != 0, "L-start candidate has zero signature")
            residual = (
                model.H_check.astype(np.int64) @ move.astype(np.int64) % 2
            ).astype(np.uint8)
            require(not residual.any(), "L-start candidate leaves the kernel")
            candidate_count += 1
            key = (
                int(np.count_nonzero(epsilon ^ move)),
                int(move.sum()),
                signature,
                packed,
            )
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    require(selected is not None, "no L-start candidate exists")
    key, move = selected
    metadata = {
        "rule": L_START_RULE,
        "candidate_orders": [1, 2, 3],
        "candidate_count": candidate_count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": array_sha256(move),
    }
    return move, metadata


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise AuditConflict(f"cannot load {path}: {exc}") from exc


def load_raw(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise AuditConflict(f"cannot read raw {path}: {exc}") from exc
    require(not any(value.dtype.hasobject for value in raw.values()), "raw contains object dtype")
    return raw


def expected_raw_fields():
    outer = {
        "raw_version", "contract_version", "manifest_sha256", "task_fingerprint",
        "task_json", "source_binding_json", "config_sha256", "registry_sha256",
        "uniform_seed", "syndrome", "character_masks", "character_sha256",
        "l_start_json", "l_move_sha256", "trajectory_digest", "core_seconds",
        "wall_seconds",
    }
    engine = {
        "raw_version", "method_id", "sampler_config_json", "sampler_config_sha256",
        "seed_identity_json", "initial_state_packed", "burn_state_packed",
        "final_state_packed", "measurement_states_packed", "burn_labels",
        "measurement_labels", "measurement_weights", "measurement_residual_weights",
        "measurement_block", "burn_basis_seen", "initial_label", "burn_label",
        "final_label", "burn_tt_counters", "measurement_tt_counters",
        "burn_tt_accepts", "measurement_tt_accepts", "burn_tt_log_acceptance",
        "measurement_tt_log_acceptance", "burn_tt_accepted_b_bit_changes",
        "measurement_tt_accepted_b_bit_changes", "burn_tt_prior_refresh_bit_changes",
        "measurement_tt_prior_refresh_bit_changes", "lambda_values", "lambda_sha256",
        "mass_sha256", "transition_kernel", "counter_names", "engine",
    }
    return outer | {f"ctt_{name}" for name in engine}


def raw_path(task):
    return OUTPUT / "raw" / task["init_family"] / f"t{task['trajectory_index']:02d}.npz"


def initial_state(task, model, syndrome, epsilon, l_move):
    family = task["init_family"]
    if family == "P":
        value = epsilon.copy()
    elif family == "U":
        identity = CollapsedTemperedTransitionSeedIdentity(**task["seed_identity"])
        value = uniform_hard_coset_state(model, syndrome, identity.seed("initialize", "hard_coset"))
    elif family == "L":
        value = epsilon ^ l_move
    else:
        raise AuditConflict(f"unknown initial family {family}")
    residual = (
        model.H_check.astype(np.int64) @ value.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    require(not residual.any(), f"{family} initial state leaves hard coset")
    return value


def validate_raw(raw, task, manifest, context):
    require(set(raw) == expected_raw_fields(), "raw schema changed")
    expected_scalars = {
        "raw_version": RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"],
        "registry_sha256": manifest["registry_sha256"],
    }
    for name, expected in expected_scalars.items():
        require(scalar(raw[name], name) == expected, f"raw {name} changed")
    require(scalar(raw["task_fingerprint"], "task_fingerprint") == sha256_json(task),
            "raw task fingerprint changed")
    require(json.loads(scalar(raw["task_json"], "task_json")) == task, "raw task JSON changed")
    require(
        json.loads(scalar(raw["source_binding_json"], "source_binding_json"))
        == manifest["source_binding"],
        "raw source binding changed",
    )
    require(int(scalar(raw["uniform_seed"], "uniform_seed")) == context["uniform_seed"],
            "raw uniform seed changed")
    require(np.array_equal(raw["syndrome"], context["syndrome"]), "raw syndrome changed")
    require(np.array_equal(raw["character_masks"], context["characters"].masks),
            "raw character masks changed")
    require(scalar(raw["character_sha256"], "character_sha256") == context["characters"].character_sha256,
            "raw character digest changed")
    require(json.loads(scalar(raw["l_start_json"], "l_start_json")) == context["l_start"],
            "raw L-start metadata changed")
    require(scalar(raw["l_move_sha256"], "l_move_sha256") == array_sha256(context["l_move"]),
            "raw L move changed")
    require(scalar(raw["ctt_raw_version"], "ctt_raw_version") == COLLAPSED_TT_RAW_VERSION,
            "engine raw version changed")
    require(scalar(raw["ctt_method_id"], "ctt_method_id") == task["method_id"],
            "engine method changed")
    require(scalar(raw["ctt_transition_kernel"], "ctt_transition_kernel") == COLLAPSED_TT_KERNEL,
            "engine kernel changed")
    require(json.loads(scalar(raw["ctt_sampler_config_json"], "ctt_sampler_config_json")) == task["sampler_config"],
            "engine sampler config changed")
    require(json.loads(scalar(raw["ctt_seed_identity_json"], "ctt_seed_identity_json")) == task["seed_identity"],
            "engine seed identity changed")
    require(
        scalar(raw["ctt_lambda_sha256"], "ctt_lambda_sha256") == manifest["lambda_sha256"],
        "engine lambda digest changed",
    )
    require(
        scalar(raw["ctt_mass_sha256"], "ctt_mass_sha256") == manifest["classical_mass_sha256"],
        "engine mass digest changed",
    )
    require(np.array_equal(raw["ctt_counter_names"], np.asarray(COLLAPSED_TT_COUNTER_NAMES)),
            "engine counter schema changed")
    require(scalar(raw["trajectory_digest"], "trajectory_digest") == engine_digest(raw),
            "raw trajectory digest changed")

    expected_initial = initial_state(
        task, context["model"], context["syndrome"], context["epsilon"], context["l_move"],
    )
    require(np.array_equal(raw["ctt_initial_state_packed"], pack_state(expected_initial)),
            "raw initial state changed")
    require(
        int(scalar(raw["ctt_initial_label"], "ctt_initial_label"))
        == int(state_label(context["frame"], expected_initial)),
        "initial label changed",
    )
    states = unpack_states(raw["ctt_measurement_states_packed"], context["model"].num_qubits)
    residuals = (
        context["model"].H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ context["syndrome"][None, :]
    require(not residuals.any(), "measurement state leaves hard coset")
    require(np.array_equal(
        residuals.sum(axis=1).astype(np.int32), raw["ctt_measurement_residual_weights"],
    ), "measurement residual weights changed")
    require(np.array_equal(
        states.sum(axis=1).astype(np.int32), raw["ctt_measurement_weights"],
    ), "measurement weights changed")
    labels = np.asarray(
        [state_label(context["frame"], state) for state in states], dtype=np.uint64,
    )
    require(np.array_equal(labels, raw["ctt_measurement_labels"]), "measurement labels changed")
    burn_state = unpack_one(raw["ctt_burn_state_packed"], context["model"].num_qubits)
    final_state = unpack_one(raw["ctt_final_state_packed"], context["model"].num_qubits)
    for name, state in (("burn", burn_state), ("final", final_state)):
        residual = (
            context["model"].H_check.astype(np.int64) @ state.astype(np.int64) % 2
        ).astype(np.uint8) ^ context["syndrome"]
        require(not residual.any(), f"{name} state leaves hard coset")
    require(int(scalar(raw["ctt_burn_label"], "ctt_burn_label")) == int(state_label(context["frame"], burn_state)),
            "burn label changed")
    require(int(scalar(raw["ctt_final_label"], "ctt_final_label")) == int(state_label(context["frame"], final_state)),
            "final label changed")
    require(np.array_equal(raw["ctt_measurement_block"], np.repeat(
        np.arange(8, dtype=np.int8), labels.size // 8,
    )), "measurement block changed")
    require(
        np.array_equal(
            raw["ctt_burn_basis_seen"],
            basis_seen(raw["ctt_burn_labels"], context["model"].k),
        ),
        "burn basis coverage changed",
    )
    counters = np.asarray(raw["ctt_measurement_tt_counters"], dtype=np.int64)
    accepts = np.asarray(raw["ctt_measurement_tt_accepts"], dtype=np.uint8)
    b_changes = np.asarray(raw["ctt_measurement_tt_accepted_b_bit_changes"], dtype=np.int32)
    require(counters.shape == (len(COLLAPSED_TT_COUNTER_NAMES),), "counter shape changed")
    require(int(counters[0]) == labels.size and int(counters[1]) == int(accepts.sum()),
            "acceptance counters changed")
    require(int(counters[2]) == int(np.count_nonzero((accepts != 0) & (b_changes > 0))),
            "B-changing counters changed")
    require(
        np.all(np.isfinite(np.asarray(raw["ctt_measurement_tt_log_acceptance"], dtype=np.float64))),
        "measurement log acceptance is non-finite",
    )
    return {
        "task": task,
        "burn_label": int(scalar(raw["ctt_burn_label"], "ctt_burn_label")),
        "labels": labels,
        "counters": counters,
        "wall_seconds": float(scalar(raw["wall_seconds"], "wall_seconds")),
    }


def family_summary(rows, context):
    k = context["model"].k
    masks = context["characters"].masks
    change_count = 0
    chains_with_eight = 0
    deltas = []
    returns = []
    wall_seconds = []
    counters = np.zeros(len(COLLAPSED_TT_COUNTER_NAMES), dtype=np.int64)
    for row in rows:
        labels = row["labels"]
        previous = np.concatenate((np.asarray([row["burn_label"]], dtype=np.uint64), labels[:-1]))
        nonzero = labels ^ previous
        changed = nonzero != 0
        changes = int(changed.sum())
        change_count += changes
        chains_with_eight += int(changes >= 8)
        deltas.extend(np.uint64(value) for value in nonzero[changed])
        returns.append(leave_return(
            np.concatenate((np.asarray([row["burn_label"]], dtype=np.uint64), labels)),
            masks,
        ))
        wall_seconds.append(row["wall_seconds"])
        counters += row["counters"]
    summary = {
        "chain_count": len(rows),
        "measurement_cross_label_changes": change_count,
        "chains_with_eight_measurement_cross_label_changes": chains_with_eight,
        "measurement_label_delta_rank": rank_masks(deltas, k),
        "basis_characters_with_leave_return": int(np.count_nonzero(
            np.asarray(returns, dtype=np.uint8)[:, :k].sum(axis=0)
        )),
        "nonbasis_characters_with_leave_return": int(np.count_nonzero(
            np.asarray(returns, dtype=np.uint8)[:, k:].sum(axis=0)
        )),
        "median_wall_seconds": float(np.median(wall_seconds)),
        "ctt_path_diagnostic": {
            "attempts": int(counters[0]),
            "accepts": int(counters[1]),
            "accepted_b_changing_proposals": int(counters[2]),
            "prior_refresh_bit_changes": int(counters[3]),
            "reversible_block_updates": int(counters[4]),
            "reversible_block_changes": int(counters[5]),
            "is_not_a_transport_gate": True,
        },
    }
    gates = context["config"]["gates"]
    summary["passes_transport_gate"] = bool(
        summary["chain_count"] == 8
        and summary["measurement_cross_label_changes"] >= gates[
            "minimum_measurement_cross_label_changes_per_family"
        ]
        and summary["chains_with_eight_measurement_cross_label_changes"] >= gates[
            "minimum_chains_with_eight_measurement_cross_label_changes_per_family"
        ]
        and summary["measurement_label_delta_rank"] >= gates[
            "minimum_measurement_label_delta_rank_per_family"
        ]
        and (
            not gates["require_all_basis_character_leave_returns"]
            or summary["basis_characters_with_leave_return"] == k
        )
        and (
            not gates["require_all_nonbasis_character_leave_returns"]
            or summary["nonbasis_characters_with_leave_return"]
            == context["config"]["num_nonbasis_characters"]
        )
    )
    return summary


def main():
    manifest = load_json(OUTPUT / "MANIFEST.json")
    report = load_json(OUTPUT / "REPORT.json")
    require(manifest["manifest_sha256"] == MANIFEST_SHA256, "manifest SHA changed")
    require(report["manifest_sha256"] == MANIFEST_SHA256, "report uses foreign manifest")
    registry_path = Path("data/expander_code/exp102/registry/registry.json")
    registry = load_registry(registry_path)
    require(registry["registry_sha256"] == manifest["registry_sha256"], "registry changed")
    _, code, H = load_frozen_code(registry_path, manifest["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = int(manifest["uniform_seed"])
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < float(manifest["cell"]["p"])
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    require(array_sha256(H) == manifest["H_sha256"], "H matrix changed")
    require(array_sha256(syndrome) == manifest["syndrome_sha256"], "syndrome changed")
    character_seed = derive_seed(
        manifest["config"]["character_seed_namespace"], registry["registry_sha256"],
        code["code_id"], "transport_characters",
    )
    characters = frozen_character_set(model.k, character_seed, manifest["config"]["num_nonbasis_characters"])
    require(characters.character_sha256 == manifest["character_sha256"], "character set changed")
    require([int(value) for value in characters.masks] == manifest["character_masks"], "character masks changed")
    l_move, l_start = select_l_move(epsilon, model, frame)
    require(l_start == manifest["l_start"], "L-start selection changed")
    expected_files = {raw_path(task) for task in manifest["tasks"]}
    observed_files = set((OUTPUT / "raw").glob("*/*.npz"))
    require(observed_files == expected_files, "raw files are missing or unexpected")
    context = {
        "uniform_seed": uniform_seed,
        "syndrome": syndrome,
        "epsilon": epsilon,
        "model": model,
        "frame": frame,
        "characters": characters,
        "l_move": l_move,
        "l_start": l_start,
        "config": manifest["config"],
    }
    rows = [validate_raw(load_raw(raw_path(task)), task, manifest, context) for task in manifest["tasks"]]
    summaries = {
        family: family_summary(
            [row for row in rows if row["task"]["init_family"] == family], context,
        )
        for family in manifest["config"]["init_families"]
    }
    require(summaries == report["families"], "independent transport summary differs from report")
    require(report["status"] == "LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE", "unexpected report status")
    core = {
        "audit_version": AUDIT_VERSION,
        "manifest_sha256": MANIFEST_SHA256,
        "report_sha256": report["report_sha256"],
        "status": "INDEPENDENT_RAW_AUDIT_PASS",
        "raw_count": len(rows),
        "families": summaries,
    }
    audit = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(OUTPUT / "INDEPENDENT_AUDIT.json", audit)
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
