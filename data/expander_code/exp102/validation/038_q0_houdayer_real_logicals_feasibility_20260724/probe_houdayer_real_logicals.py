"""Frozen real-low-energy component preflight for the Houdayer pair kernel."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import reduce_logical_basis
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer import (
    build_sparse_hgp_coordinate_basis,
    component_logical_mask,
    coordinate_factor_scopes,
    coordinates_from_kernel_delta,
    coordinates_to_state,
    houdayer_components,
    prepare_coordinate_readout,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_houdayer_real_logicals.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class RealLogicalHoudayerError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise RealLogicalHoudayerError(message)


def _as_bits(value, *, ndim, name):
    result = np.asarray(value)
    _require(result.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.all((result == 0) | (result == 1)), f"{name} must be binary")
    return np.ascontiguousarray(result, dtype=np.uint8)


def _signature(bits):
    bits = _as_bits(bits, ndim=1, name="logical signature bits")
    _require(bits.size <= 64, "logical signature exceeds uint64")
    result = 0
    for position in np.flatnonzero(bits):
        result |= 1 << int(position)
    return result


def _label(frame, state):
    state = _as_bits(state, ndim=1, name="physical state")
    return _signature(
        (frame.W_basis.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8),
    )


def _mask_rank(values):
    pivots = {}
    for raw in values:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = value
                break
            value ^= previous
    return len(pivots)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer.py"),
        "probe": sha256_file(Path(__file__)),
        "q0_global": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_global.py"),
        "q0_hgp_screen": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_screen.py"),
        "registry": sha256_file(REGISTRY_PATH),
        "worker": sha256_file(EXP102_ROOT / "exp102_pipeline" / "worker.py"),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise RealLogicalHoudayerError("real-logical Houdayer config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "real-logical Houdayer config is not canonical")
    expected_keys = {
        "catalog", "cell", "config_version", "contract_version", "coordinate_basis",
        "exhaustive_component_limit", "pair_families", "registry_sha256", "scope", "version",
    }
    _require(set(config) == expected_keys and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"]
             == "exp102.q0_houdayer_real_logicals.feasibility.config.v0",
             "real-logical Houdayer config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "real-logical Houdayer cell changed")
    _require(config["coordinate_basis"]
             == "h_x_plus_code_only_tensor_logical_complement.v0",
             "real-logical Houdayer coordinate basis changed")
    _require(config["catalog"] == {
        "candidate_orders": [1, 2, 3],
        "deduplication": "one_minimum_p_derived_state_per_nonzero_logical_signature",
        "low_energy_label_count": 16,
        "rank_complete_label_count": 64,
        "selection_order": "state_weight,move_weight,signature,packed_move",
    }, "real-logical Houdayer catalog definition changed")
    _require(config["pair_families"] == [
        "P_vs_each_low_energy", "all_pairs_within_low_energy", "P_vs_each_rank_complete",
    ], "real-logical Houdayer pair schedule changed")
    _require(config["exhaustive_component_limit"] == 12,
             "real-logical Houdayer exhaustive component cap changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "real-logical Houdayer registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "real_low_energy_houdayer_component_feasibility_only",
        "remote_authorization": False,
    }, "real-logical Houdayer scope changed")
    return config, sha256_file(path)


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "real-logical Houdayer registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64,
             "real-logical Houdayer dimensions changed")
    return registry, code, H, model, frame, int(uniform_seed), planted, syndrome


def _candidate_catalog(model, frame, planted, config):
    """Build fixed P-derived starts without querying a sampler or its output."""
    reduced = _as_bits(reduce_logical_basis(model.logical_move_basis), ndim=2,
                       name="reduced logical basis")
    _require(reduced.shape == (model.k, model.num_qubits),
             "real-logical reduced basis dimensions changed")
    _require(not (model.H_check.astype(np.int64) @ reduced.T.astype(np.int64) % 2).any(),
             "real-logical reduced basis leaves the hard kernel")
    candidates = {}
    for order in config["catalog"]["candidate_orders"]:
        for combination in itertools.combinations(range(model.k), int(order)):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            signature = _label(frame, move)
            _require(signature, "real-logical candidate has zero logical signature")
            state = np.ascontiguousarray(planted ^ move, dtype=np.uint8)
            candidates[packed] = {
                "move": move,
                "state": state,
                "move_weight": int(move.sum()),
                "state_weight": int(state.sum()),
                "signature": signature,
                "packed": packed,
            }
    _require(candidates, "real-logical catalog has no candidates")
    key = lambda record: (
        record["state_weight"], record["move_weight"], record["signature"], record["packed"],
    )
    per_signature = {}
    for record in sorted(candidates.values(), key=key):
        per_signature.setdefault(record["signature"], record)
    ordered = tuple(sorted(per_signature.values(), key=key))
    _require(len(ordered) >= config["catalog"]["low_energy_label_count"],
             "real-logical catalog has too few low-energy labels")
    low_energy = ordered[:config["catalog"]["low_energy_label_count"]]
    rank_complete = []
    previous_rank = 0
    for record in ordered:
        current_rank = _mask_rank([item["signature"] for item in rank_complete]
                                  + [record["signature"]])
        if current_rank > previous_rank:
            rank_complete.append(record)
            previous_rank = current_rank
        if len(rank_complete) == config["catalog"]["rank_complete_label_count"]:
            break
    _require(len(rank_complete) == model.k and previous_rank == model.k,
             "real-logical catalog does not span every logical direction")
    return {
        "candidate_move_count": len(candidates),
        "distinct_nonzero_signature_count": len(ordered),
        "reduced_basis_sha256": hashlib.sha256(reduced.tobytes()).hexdigest(),
        "low_energy": tuple(low_energy),
        "rank_complete": tuple(rank_complete),
    }


def _catalog_record(record, index):
    return {
        "index": int(index),
        "logical_signature": int(record["signature"]),
        "move_weight": int(record["move_weight"]),
        "state_weight": int(record["state_weight"]),
        "move_sha256": hashlib.sha256(record["move"].tobytes()).hexdigest(),
        "state_sha256": hashlib.sha256(record["state"].tobytes()).hexdigest(),
    }


def _coordinates_for_state(state, planted, generators, readout):
    state = _as_bits(state, ndim=1, name="real-logical state")
    coordinates = coordinates_from_kernel_delta(state ^ planted, generators, readout=readout)
    _require(np.array_equal(coordinates_to_state(planted, generators, coordinates), state),
             "real-logical coordinate inverse does not round-trip")
    return coordinates


def _residual(model, state):
    return (model.H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)


def _unordered_pair_key(left, right):
    left_bytes = left.tobytes()
    right_bytes = right.tobytes()
    return left_bytes + right_bytes if left_bytes <= right_bytes else right_bytes + left_bytes


def _pair_record(family, left_descriptor, right_descriptor, left_state, right_state,
                 left_coordinates, right_coordinates, basis, factors, model, frame,
                 syndrome, exhaustive_limit):
    difference = np.ascontiguousarray(left_coordinates ^ right_coordinates, dtype=np.uint8)
    components = houdayer_components(left_coordinates, right_coordinates, factors)
    _require(sum(int(component.size) for component in components) == int(difference.sum()),
             "Houdayer components do not partition the coordinate disagreement")
    generators = basis["generators"]
    component_deltas = []
    component_masks = []
    component_records = []
    reconstructed_difference = np.zeros(model.num_qubits, dtype=np.uint8)
    for component in components:
        coordinate_delta = np.zeros(generators.shape[0], dtype=np.uint8)
        coordinate_delta[component] = difference[component]
        physical_delta = (
            coordinate_delta.astype(np.int64) @ generators.astype(np.int64) % 2
        ).astype(np.uint8)
        mask = component_logical_mask(component, difference, basis["logical_masks"])
        _require(mask == _label(frame, physical_delta),
                 "component logical mask disagrees with the physical label")
        reconstructed_difference ^= physical_delta
        component_deltas.append(np.ascontiguousarray(physical_delta, dtype=np.uint8))
        component_masks.append(mask)
        component_records.append({
            "coordinate_count": int(component.size),
            "stabilizer_coordinate_count": int(np.count_nonzero(
                component < basis["stabilizer_count"],
            )),
            "logical_coordinate_count": int(np.count_nonzero(
                component >= basis["stabilizer_count"],
            )),
            "is_complete_disagreement": bool(component.size == difference.sum()),
            "logical_delta_mask": int(mask),
            "physical_delta_weight": int(physical_delta.sum()),
        })
    _require(np.array_equal(reconstructed_difference, left_state ^ right_state),
             "Houdayer component deltas do not reconstruct the replica difference")
    _require(np.array_equal(_residual(model, left_state), syndrome)
             and np.array_equal(_residual(model, right_state), syndrome),
             "real-logical pair leaves the hard coset")
    total_weight = int(left_state.sum() + right_state.sum())
    if len(components) <= exhaustive_limit:
        subset_indices = tuple(range(1 << len(components)))
        subset_mode = "EXHAUSTIVE_ALL_COMPONENT_SUBSETS"
    else:
        subset_indices = (0,) + tuple(1 << index for index in range(len(components)))
        subset_mode = "ORIGINAL_PLUS_EACH_SINGLE_COMPONENT_ONLY"
    unordered_pairs = set()
    label_pairs = set()
    left_weights = []
    right_weights = []
    for subset in subset_indices:
        delta = np.zeros(model.num_qubits, dtype=np.uint8)
        for index, physical_delta in enumerate(component_deltas):
            if (subset >> index) & 1:
                delta ^= physical_delta
        new_left = left_state ^ delta
        new_right = right_state ^ delta
        _require(np.array_equal(_residual(model, new_left), syndrome)
                 and np.array_equal(_residual(model, new_right), syndrome),
                 "Houdayer subset leaves the hard coset")
        _require(int(new_left.sum() + new_right.sum()) == total_weight,
                 "Houdayer subset changes the pair energy")
        unordered_pairs.add(_unordered_pair_key(new_left, new_right))
        labels = tuple(sorted((_label(frame, new_left), _label(frame, new_right))))
        label_pairs.add(labels)
        left_weights.append(int(new_left.sum()))
        right_weights.append(int(new_right.sum()))
    initial_key = _unordered_pair_key(left_state, right_state)
    _require(initial_key in unordered_pairs, "Houdayer subset schedule lost the original pair")
    evaluated_maxima = [max(left, right) for left, right in zip(left_weights, right_weights)]
    evaluated_minima = [min(left, right) for left, right in zip(left_weights, right_weights)]
    return {
        "family": family,
        "left": left_descriptor,
        "right": right_descriptor,
        "coordinate_disagreement_count": int(difference.sum()),
        "left_weight": int(left_state.sum()),
        "right_weight": int(right_state.sum()),
        "pair_total_weight": total_weight,
        "component_count": len(components),
        "largest_component_coordinates": int(max((component.size for component in components), default=0)),
        "component_logical_delta_rank": _mask_rank(component_masks),
        "component_records": component_records,
        "subset_evaluation": {
            "mode": subset_mode,
            "component_subset_exponent": len(components),
            "evaluated_subset_count": len(subset_indices),
            "unique_unordered_physical_pair_count": len(unordered_pairs),
            "novel_unordered_physical_pair_count": len(unordered_pairs - {initial_key}),
            "unique_unordered_logical_label_pair_count": len(label_pairs),
            "left_weight_min_max": [min(left_weights), max(left_weights)],
            "right_weight_min_max": [min(right_weights), max(right_weights)],
            "minimum_evaluated_max_replica_weight": min(evaluated_maxima),
            "maximum_evaluated_min_replica_weight": max(evaluated_minima),
        },
    }


def _summary(records, family):
    selected = [record for record in records if record["family"] == family]
    _require(selected, f"real-logical Houdayer has no {family} records")
    return {
        "record_count": len(selected),
        "component_count_min_max": [
            min(record["component_count"] for record in selected),
            max(record["component_count"] for record in selected),
        ],
        "component_logical_rank_min_max": [
            min(record["component_logical_delta_rank"] for record in selected),
            max(record["component_logical_delta_rank"] for record in selected),
        ],
        "whole_pair_exchange_only_record_count": sum(
            record["component_count"] == 1
            and record["component_records"][0]["is_complete_disagreement"]
            for record in selected
        ),
        "records_with_novel_unordered_pair": sum(
            record["subset_evaluation"]["novel_unordered_physical_pair_count"] > 0
            for record in selected
        ),
        "records_with_nontrivial_logical_component_rank": sum(
            record["component_logical_delta_rank"] > 1 for record in selected
        ),
        "exhaustive_record_count": sum(
            record["subset_evaluation"]["mode"] == "EXHAUSTIVE_ALL_COMPONENT_SUBSETS"
            for record in selected
        ),
    }


def run_probe(config):
    registry, code, H, model, frame, uniform_seed, planted, syndrome = _context(config)
    basis = build_sparse_hgp_coordinate_basis(H, model, frame)
    factors = coordinate_factor_scopes(basis["generators"])
    readout = prepare_coordinate_readout(basis["generators"])
    catalog = _candidate_catalog(model, frame, planted, config)
    low_states = tuple({
        **record,
        "coordinates": _coordinates_for_state(
            record["state"], planted, basis["generators"], readout,
        ),
    } for record in catalog["low_energy"])
    rank_states = tuple({
        **record,
        "coordinates": _coordinates_for_state(
            record["state"], planted, basis["generators"], readout,
        ),
    } for record in catalog["rank_complete"])
    planted_coordinates = np.zeros(basis["generators"].shape[0], dtype=np.uint8)
    records = []
    if "P_vs_each_low_energy" in config["pair_families"]:
        for index, record in enumerate(low_states):
            records.append(_pair_record(
                "P_vs_each_low_energy", {"kind": "P"},
                {"kind": "low_energy", "catalog_index": index,
                 "logical_signature": int(record["signature"])},
                planted, record["state"], planted_coordinates, record["coordinates"],
                basis, factors, model, frame, syndrome, config["exhaustive_component_limit"],
            ))
    if "all_pairs_within_low_energy" in config["pair_families"]:
        for left_index, right_index in itertools.combinations(range(len(low_states)), 2):
            left = low_states[left_index]
            right = low_states[right_index]
            records.append(_pair_record(
                "all_pairs_within_low_energy",
                {"kind": "low_energy", "catalog_index": left_index,
                 "logical_signature": int(left["signature"])},
                {"kind": "low_energy", "catalog_index": right_index,
                 "logical_signature": int(right["signature"])},
                left["state"], right["state"], left["coordinates"], right["coordinates"],
                basis, factors, model, frame, syndrome, config["exhaustive_component_limit"],
            ))
    if "P_vs_each_rank_complete" in config["pair_families"]:
        for index, record in enumerate(rank_states):
            records.append(_pair_record(
                "P_vs_each_rank_complete", {"kind": "P"},
                {"kind": "rank_complete", "catalog_index": index,
                 "logical_signature": int(record["signature"])},
                planted, record["state"], planted_coordinates, record["coordinates"],
                basis, factors, model, frame, syndrome, config["exhaustive_component_limit"],
            ))
    expected_count = (
        len(low_states) + len(low_states) * (len(low_states) - 1) // 2 + len(rank_states)
    )
    _require(len(records) == expected_count, "real-logical Houdayer pair schedule is incomplete")
    summaries = {family: _summary(records, family) for family in config["pair_families"]}
    low_novel = sum(
        summaries[family]["records_with_novel_unordered_pair"]
        for family in ("P_vs_each_low_energy", "all_pairs_within_low_energy")
    )
    all_novel = sum(summary["records_with_novel_unordered_pair"] for summary in summaries.values())
    if low_novel:
        status = "NONTRIVIAL_LOW_ENERGY_RECOMBINATION_REQUIRES_EXACT_SAMPLER_TEST"
    elif all_novel:
        status = "NONTRIVIAL_RANK_COMPLETE_ONLY_RECOMBINATION_NOT_A_LOW_ENERGY_PASS"
    else:
        status = "FROZEN_REAL_LOGICAL_CATALOG_ONLY_WHOLE_PAIR_EXCHANGES"
    return {
        "status": status,
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "disorder_uniform_seed": uniform_seed,
        "dimensions": {
            "classical_shape": [int(H.shape[0]), int(H.shape[1])],
            "physical_qubits": int(model.num_qubits),
            "hard_coset_dimension": int(basis["generators"].shape[0]),
            "logical_dimension": int(model.k),
            "planted_weight_diagnostic_only": int(planted.sum()),
            "syndrome_weight": int(syndrome.sum()),
        },
        "coordinate_basis": {
            "version": basis["version"],
            "stabilizer_coordinate_count": int(basis["stabilizer_count"]),
            "logical_coordinate_count": int(basis["logical_count"]),
            "readout_pivot_columns_sha256": hashlib.sha256(
                readout.pivot_columns.astype(">i4").tobytes(),
            ).hexdigest(),
        },
        "catalog": {
            "candidate_move_count": catalog["candidate_move_count"],
            "distinct_nonzero_signature_count": catalog["distinct_nonzero_signature_count"],
            "reduced_basis_sha256": catalog["reduced_basis_sha256"],
            "low_energy": [_catalog_record(record, index)
                           for index, record in enumerate(low_states)],
            "rank_complete": [_catalog_record(record, index)
                              for index, record in enumerate(rank_states)],
            "rank_complete_signature_rank": _mask_rank(
                [record["signature"] for record in rank_states],
            ),
        },
        "identity": {
            "pair_target": "pi(e_left|y)*pi(e_right|y)",
            "coordinate_swap": "one complete disagreement component",
            "required_invariant": "hard-coset membership and total pair Hamming weight",
            "not_counted_as_mixing": "a swap that only exchanges the original unordered pair",
        },
        "family_summaries": summaries,
        "pair_records": records,
        "does_not_establish": [
            "A constructed or run MCMC trajectory.",
            "A posterior, q_top, logical-sector probability, or convergence claim.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "houdayer_real_logicals.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace real-logical Houdayer report: {args.output}")
    config, config_sha256 = _load_config(args.config)
    core = {
        "probe_version": PROBE_VERSION,
        "config_sha256": config_sha256,
        "scope": config["scope"],
        "source_binding": _source_binding(args.config),
        "probe": run_probe(config),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
