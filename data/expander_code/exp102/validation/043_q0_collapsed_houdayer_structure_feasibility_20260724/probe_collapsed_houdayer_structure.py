"""Frozen structural feasibility probe for collapsed-B Houdayer components."""

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
from data.expander_code.exp102.exp102_pipeline.q0_collapsed_houdayer import (
    b_bits_to_masks,
    build_collapsed_b_houdayer_kernel,
    collapsed_b_component_delta,
    collapsed_b_factor_masks,
    collapsed_b_houdayer_components,
    collapsed_b_pair_invariants,
    hgp_syndrome_to_columns,
    initialize_collapsed_b_houdayer_pair,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    reduce_logical_basis,
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import _initial_collapsed_masks
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_collapsed_houdayer.structure.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class CollapsedHoudayerProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise CollapsedHoudayerProbeError(message)


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
        "collapsed_houdayer": sha256_file(
            EXP102_ROOT / "exp102_pipeline" / "q0_collapsed_houdayer.py",
        ),
        "collapsed_hgp": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py"),
        "q0_global": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_global.py"),
        "q0_hgp_screen": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_screen.py"),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer.py"),
        "probe": sha256_file(Path(__file__)),
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
        raise CollapsedHoudayerProbeError("collapsed HCA config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "collapsed HCA config is not canonical")
    expected_keys = {
        "catalog", "cell", "config_version", "contract_version",
        "exhaustive_component_limit", "pair_families", "registry_sha256", "scope",
        "uniform_pair_seeds", "version",
    }
    _require(set(config) == expected_keys and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"]
             == "exp102.q0_collapsed_houdayer.structure.feasibility.config.v0",
             "collapsed HCA config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "collapsed HCA cell changed")
    _require(config["catalog"] == {
        "candidate_orders": [1, 2, 3],
        "deduplication": "one_minimum_p_derived_state_per_nonzero_logical_signature",
        "low_energy_label_count": 16,
        "rank_complete_label_count": 64,
        "selection_order": "state_weight,move_weight,signature,packed_move",
    }, "collapsed HCA catalog definition changed")
    _require(config["pair_families"] == [
        "P_vs_each_low_energy", "all_pairs_within_low_energy",
        "P_vs_each_rank_complete", "U_vs_U",
    ], "collapsed HCA pair schedule changed")
    _require(config["exhaustive_component_limit"] == 12,
             "collapsed HCA component cap changed")
    _require(config["uniform_pair_seeds"] == [
        1056985142298559599, 7506801422450599911,
    ], "collapsed HCA uniform-pair seeds changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "collapsed HCA registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "collapsed_b_houdayer_component_feasibility_only",
        "remote_authorization": False,
    }, "collapsed HCA scope changed")
    return config, sha256_file(path)


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "collapsed HCA registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64,
             "collapsed HCA dimensions changed")
    y_columns = hgp_syndrome_to_columns(syndrome, H)
    return registry, code, H, model, frame, int(uniform_seed), planted, syndrome, y_columns


def _b_record(state, model, syndrome, H, y_columns, kernel):
    state = np.ascontiguousarray(state, dtype=np.uint8)
    b_columns, a_syndromes, _unused = _initial_collapsed_masks(state, syndrome, H)
    expected = collapsed_b_factor_masks(H, b_columns, y_columns)
    _require(np.array_equal(a_syndromes, expected),
             "collapsed HCA B extraction disagrees with the HGP factorization")
    _require(np.array_equal(collapsed_b_factor_masks(H, b_columns, kernel.y_columns), expected),
             "collapsed HCA kernel syndrome binding drifted")
    return np.ascontiguousarray(b_columns, dtype=np.uint32)


def _candidate_catalog(model, frame, planted, config):
    """Rebuild the frozen P-derived catalog without consulting sampler output."""
    reduced = np.ascontiguousarray(reduce_logical_basis(model.logical_move_basis), dtype=np.uint8)
    _require(reduced.shape == (model.k, model.num_qubits)
             and np.all((reduced == 0) | (reduced == 1)),
             "collapsed HCA reduced logical basis changed")
    _require(not (model.H_check.astype(np.int64) @ reduced.T.astype(np.int64) % 2).any(),
             "collapsed HCA reduced logical basis leaves the hard kernel")
    candidates = {}
    for order in config["catalog"]["candidate_orders"]:
        for combination in itertools.combinations(range(model.k), int(order)):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            signature = int(state_label(frame, move))
            _require(signature, "collapsed HCA candidate has zero logical signature")
            state = np.ascontiguousarray(planted ^ move, dtype=np.uint8)
            candidates[packed] = {
                "move": move,
                "state": state,
                "move_weight": int(move.sum()),
                "state_weight": int(state.sum()),
                "signature": signature,
                "packed": packed,
            }
    _require(candidates, "collapsed HCA catalog has no candidates")
    key = lambda record: (
        record["state_weight"], record["move_weight"], record["signature"], record["packed"],
    )
    per_signature = {}
    for record in sorted(candidates.values(), key=key):
        per_signature.setdefault(record["signature"], record)
    ordered = tuple(sorted(per_signature.values(), key=key))
    low_count = int(config["catalog"]["low_energy_label_count"])
    _require(len(ordered) >= low_count, "collapsed HCA low-energy catalog is too small")
    rank_complete = []
    previous_rank = 0
    for record in ordered:
        current_rank = _mask_rank([item["signature"] for item in rank_complete]
                                  + [record["signature"]])
        if current_rank > previous_rank:
            rank_complete.append(record)
            previous_rank = current_rank
        if len(rank_complete) == int(config["catalog"]["rank_complete_label_count"]):
            break
    _require(len(rank_complete) == model.k and previous_rank == model.k,
             "collapsed HCA catalog does not span every logical direction")
    return {
        "candidate_move_count": len(candidates),
        "distinct_nonzero_signature_count": len(ordered),
        "reduced_basis_sha256": hashlib.sha256(reduced.tobytes()).hexdigest(),
        "low_energy": tuple(ordered[:low_count]),
        "rank_complete": tuple(rank_complete),
    }


def _b_pair_key(left, right):
    left_bytes = np.asarray(left, dtype=np.uint32).tobytes()
    right_bytes = np.asarray(right, dtype=np.uint32).tobytes()
    return left_bytes + right_bytes if left_bytes <= right_bytes else right_bytes + left_bytes


def _pair_record(family, left_descriptor, right_descriptor, left_b, right_b, kernel,
                 exhaustive_limit):
    initial = initialize_collapsed_b_houdayer_pair(kernel, left_b, right_b)
    components = collapsed_b_houdayer_components(initial, kernel)
    difference = np.bitwise_xor(
        np.asarray(left_b, dtype=np.uint32), np.asarray(right_b, dtype=np.uint32),
    )
    differing_bits = int(sum(int(value).bit_count() for value in difference))
    _require(sum(int(component.size) for component in components) == differing_bits,
             "collapsed HCA components do not partition the B disagreement")
    deltas = [collapsed_b_component_delta(initial, kernel, component) for component in components]
    component_records = [{
        "b_variable_count": int(component.size),
        "changed_b_bit_count": int(sum(int(value).bit_count() for value in delta)),
        "is_complete_disagreement": bool(component.size == differing_bits),
        "delta_sha256": hashlib.sha256(np.asarray(delta, dtype=np.uint32).tobytes()).hexdigest(),
    } for component, delta in zip(components, deltas)]
    before = collapsed_b_pair_invariants(initial, kernel)
    if len(components) <= int(exhaustive_limit):
        subset_indices = tuple(range(1 << len(components)))
        subset_mode = "EXHAUSTIVE_ALL_COMPONENT_SUBSETS"
    else:
        subset_indices = (0,) + tuple(1 << index for index in range(len(components)))
        subset_mode = "ORIGINAL_PLUS_EACH_SINGLE_COMPONENT_ONLY"
    unordered_pairs = set()
    left_weights = []
    right_weights = []
    for subset in subset_indices:
        candidate = initial.copy()
        for index, delta in enumerate(deltas):
            if (subset >> index) & 1:
                candidate.left ^= delta
                candidate.right ^= delta
        after = collapsed_b_pair_invariants(candidate, kernel)
        _require(after["pair_b_weight"] == before["pair_b_weight"],
                 "collapsed HCA subset changed the B unary pair weight")
        _require(np.array_equal(after["factor_pairs"], before["factor_pairs"]),
                 "collapsed HCA subset changed a collapsed factor pair")
        unordered_pairs.add(_b_pair_key(candidate.left, candidate.right))
        left_weights.append(sum(int(value).bit_count() for value in candidate.left))
        right_weights.append(sum(int(value).bit_count() for value in candidate.right))
    initial_key = _b_pair_key(initial.left, initial.right)
    _require(initial_key in unordered_pairs, "collapsed HCA subset schedule lost its initial pair")
    evaluated_maxima = [max(left, right) for left, right in zip(left_weights, right_weights)]
    evaluated_minima = [min(left, right) for left, right in zip(left_weights, right_weights)]
    return {
        "family": family,
        "left": left_descriptor,
        "right": right_descriptor,
        "left_b_weight": sum(int(value).bit_count() for value in initial.left),
        "right_b_weight": sum(int(value).bit_count() for value in initial.right),
        "pair_b_weight": int(before["pair_b_weight"]),
        "b_disagreement_count": differing_bits,
        "component_count": len(components),
        "largest_component_b_variables": int(max((component.size for component in components), default=0)),
        "component_records": component_records,
        "subset_evaluation": {
            "mode": subset_mode,
            "component_subset_exponent": len(components),
            "evaluated_subset_count": len(subset_indices),
            "unique_unordered_b_pair_count": len(unordered_pairs),
            "novel_unordered_b_pair_count": len(unordered_pairs - {initial_key}),
            "left_b_weight_min_max": [min(left_weights), max(left_weights)],
            "right_b_weight_min_max": [min(right_weights), max(right_weights)],
            "minimum_evaluated_max_b_weight": min(evaluated_maxima),
            "maximum_evaluated_min_b_weight": max(evaluated_minima),
        },
    }


def _summary(records, family):
    selected = [record for record in records if record["family"] == family]
    _require(selected, f"collapsed HCA has no {family} records")
    return {
        "record_count": len(selected),
        "component_count_min_max": [
            min(record["component_count"] for record in selected),
            max(record["component_count"] for record in selected),
        ],
        "whole_pair_exchange_only_record_count": sum(
            record["component_count"] == 1
            and record["component_records"][0]["is_complete_disagreement"]
            for record in selected if record["component_records"]
        ),
        "records_with_novel_unordered_b_pair": sum(
            record["subset_evaluation"]["novel_unordered_b_pair_count"] > 0
            for record in selected
        ),
        "exhaustive_record_count": sum(
            record["subset_evaluation"]["mode"] == "EXHAUSTIVE_ALL_COMPONENT_SUBSETS"
            for record in selected
        ),
    }


def _descriptor(kind, state, frame, **extra):
    return {
        "kind": kind,
        "physical_state_sha256": hashlib.sha256(np.asarray(state, dtype=np.uint8).tobytes()).hexdigest(),
        "physical_logical_label": int(state_label(frame, state)),
        "physical_weight_diagnostic_only": int(np.asarray(state, dtype=np.uint8).sum()),
        **extra,
    }


def run_probe(config):
    registry, _code, H, model, frame, uniform_seed, planted, syndrome, y_columns = _context(config)
    kernel = build_collapsed_b_houdayer_kernel(H, y_columns)
    p_b = _b_record(planted, model, syndrome, H, y_columns, kernel)
    catalog = _candidate_catalog(model, frame, planted, config)
    low_energy = catalog["low_energy"]
    rank_complete = catalog["rank_complete"]
    low_b = tuple(_b_record(record["state"], model, syndrome, H, y_columns, kernel)
                  for record in low_energy)
    rank_b = tuple(_b_record(record["state"], model, syndrome, H, y_columns, kernel)
                   for record in rank_complete)
    uniform_left = uniform_hard_coset_state(model, syndrome, config["uniform_pair_seeds"][0])
    uniform_right = uniform_hard_coset_state(model, syndrome, config["uniform_pair_seeds"][1])
    uniform_b = (
        _b_record(uniform_left, model, syndrome, H, y_columns, kernel),
        _b_record(uniform_right, model, syndrome, H, y_columns, kernel),
    )
    records = []
    p_descriptor = _descriptor("P", planted, frame)
    for index, record in enumerate(low_energy):
        records.append(_pair_record(
            "P_vs_each_low_energy", p_descriptor,
            _descriptor("low_energy", record["state"], frame, catalog_index=index,
                        logical_signature=int(record["signature"]),
                        move_weight=int(record["move_weight"])),
            p_b, low_b[index], kernel, config["exhaustive_component_limit"],
        ))
    for left_index, right_index in itertools.combinations(range(len(low_energy)), 2):
        left = low_energy[left_index]
        right = low_energy[right_index]
        records.append(_pair_record(
            "all_pairs_within_low_energy",
            _descriptor("low_energy", left["state"], frame, catalog_index=left_index,
                        logical_signature=int(left["signature"]), move_weight=int(left["move_weight"])),
            _descriptor("low_energy", right["state"], frame, catalog_index=right_index,
                        logical_signature=int(right["signature"]), move_weight=int(right["move_weight"])),
            low_b[left_index], low_b[right_index], kernel, config["exhaustive_component_limit"],
        ))
    for index, record in enumerate(rank_complete):
        records.append(_pair_record(
            "P_vs_each_rank_complete", p_descriptor,
            _descriptor("rank_complete", record["state"], frame, catalog_index=index,
                        logical_signature=int(record["signature"]), move_weight=int(record["move_weight"])),
            p_b, rank_b[index], kernel, config["exhaustive_component_limit"],
        ))
    records.append(_pair_record(
        "U_vs_U",
        _descriptor("U", uniform_left, frame, uniform_seed=config["uniform_pair_seeds"][0]),
        _descriptor("U", uniform_right, frame, uniform_seed=config["uniform_pair_seeds"][1]),
        uniform_b[0], uniform_b[1], kernel, config["exhaustive_component_limit"],
    ))
    expected_count = (len(low_energy) + len(low_energy) * (len(low_energy) - 1) // 2
                      + len(rank_complete) + 1)
    _require(len(records) == expected_count, "collapsed HCA pair schedule is incomplete")
    summaries = {family: _summary(records, family) for family in config["pair_families"]}
    low_energy_novel = summaries["all_pairs_within_low_energy"]["records_with_novel_unordered_b_pair"]
    status = (
        "COLLAPSED_B_HCA_LOW_ENERGY_SIGNAL_REQUIRES_EXACT_PAIR_KERNEL"
        if low_energy_novel else "COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION"
    )
    return {
        "status": status,
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "disorder_uniform_seed": uniform_seed,
        "dimensions": {
            "classical_shape": [int(H.shape[0]), int(H.shape[1])],
            "collapsed_b_variables": int(H.shape[0] ** 2),
            "collapsed_factor_count": int(H.shape[1]),
            "physical_qubits": int(model.num_qubits),
            "logical_dimension": int(model.k),
            "syndrome_weight": int(syndrome.sum()),
        },
        "identity": {
            "collapsed_target": "b**|B| * product_j M_p(Y[:,j] xor B H[:,j])",
            "pair_target": "pi_B(B_left)*pi_B(B_right)",
            "component_graph": "differing B variables linked by a common collapsed column factor",
            "required_invariant": "unordered factor-mask pairs and total B unary weight",
            "not_counted_as_transport": "whole-pair exchange, A-only redraw, or physical-only movement",
        },
        "catalog": {
            "candidate_move_count": catalog["candidate_move_count"],
            "distinct_nonzero_signature_count": catalog["distinct_nonzero_signature_count"],
            "reduced_basis_sha256": catalog["reduced_basis_sha256"],
            "low_energy": [{
                "index": int(index),
                "logical_signature": int(record["signature"]),
                "move_weight": int(record["move_weight"]),
                "state_weight": int(record["state_weight"]),
                "state_sha256": hashlib.sha256(record["state"].tobytes()).hexdigest(),
            } for index, record in enumerate(low_energy)],
            "rank_complete_signature_rank": _mask_rank(
                [record["signature"] for record in rank_complete],
            ),
        },
        "uniform_pair": {
            "seeds": list(config["uniform_pair_seeds"]),
            "left_state_sha256": hashlib.sha256(uniform_left.tobytes()).hexdigest(),
            "right_state_sha256": hashlib.sha256(uniform_right.tobytes()).hexdigest(),
        },
        "family_summaries": summaries,
        "pair_records": records,
        "does_not_establish": [
            "A constructed or run MCMC trajectory.",
            "A posterior, q_top, logical-sector probability, or convergence claim.",
            "A future HP64-plus-HCA pair-kernel's stationarity or viability.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "collapsed_houdayer_structure.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace collapsed HCA report: {args.output}")
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
