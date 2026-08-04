"""Bounded local feasibility probe for fixed-sector q=0 free-energy bridges.

This is not a posterior experiment.  It checks three prerequisites for a
future sector-stratified estimator: the bridge identity on exact small HGP
codes, complete auxiliary-stabilizer coordinates on the m8 hard cell, and
short same-sector P/S-tail agreement along one frozen low-weight logical path.
It intentionally does not calculate q_top, choose a production method, or
authorize remote, held-out, or formal work.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    build_logical_proposal_catalog,
    state_label,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer import (
    auxiliary_stabilizer_delta,
    auxiliary_stabilizer_sweep,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    join_hgp_state,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_sector_bridge import (
    SECTOR_BRIDGE_VERSION,
    bridge_step_ratio,
    exact_fixed_sector_bridge,
    logical_bridge_prefixes,
    reverse_bridge_step_ratio,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_sector_bridge.feasibility.v2"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
DEFAULT_OUTPUT = ROOT / "sector_bridge_probe_v2.json"
SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


class ProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, generator in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generator
    packed = np.packbits(states, axis=1, bitorder="little")
    _require(len({row.tobytes() for row in packed}) == states.shape[0],
             "small HGP hard-coset coordinates are not bijective")
    return states


def _small_syndrome(model, nonzero):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)


def _exact_oracle():
    cases = []
    maximum_step_error = 0.0
    maximum_product_error = 0.0
    maximum_endpoint_sector_error = 0.0
    for matrix_index, H in enumerate(SMALL_H):
        model, frame = build_model(H)
        for p in (0.04, 0.10, 0.25):
            for nonzero in (False, True):
                syndrome = _small_syndrome(model, nonzero)
                states = _hard_coset_states(model, syndrome)
                labels = np.asarray(
                    [state_label(frame, state) for state in states], dtype=np.uint64,
                )
                base = model.logical_sector_section.apply(syndrome, strict=True)
                base_label = state_label(frame, base)
                move = np.asarray(model.logical_move_basis[0], dtype=np.uint8)
                report = exact_fixed_sector_bridge(
                    states, labels, base_label, move, p,
                )
                delta = state_label(frame, move)
                endpoint_label = np.uint64(base_label) ^ np.uint64(delta)
                odds = p / (1.0 - p)
                start_mass = (odds ** states[labels == base_label].sum(axis=1)).sum(dtype=np.float64)
                endpoint_mass = (
                    odds ** states[labels == endpoint_label].sum(axis=1)
                ).sum(dtype=np.float64)
                step_error = float(np.max(np.abs(
                    report["expected_step_ratios"] - report["actual_step_ratios"],
                )))
                product_error = abs(report["product_ratio"] - report["endpoint_ratio"])
                endpoint_error = abs(report["endpoint_ratio"] - endpoint_mass / start_mass)
                maximum_step_error = max(maximum_step_error, step_error)
                maximum_product_error = max(maximum_product_error, product_error)
                maximum_endpoint_sector_error = max(maximum_endpoint_sector_error, endpoint_error)
                cases.append({
                    "classical_shape": [int(H.shape[0]), int(H.shape[1])],
                    "p": p,
                    "nonzero_syndrome": nonzero,
                    "support_size": int(report["support"].size),
                    "step_error": step_error,
                    "product_error": product_error,
                    "endpoint_sector_error": endpoint_error,
                })
    _require(maximum_step_error <= 2e-13, "exact bridge step identity failed")
    _require(maximum_product_error <= 3e-13, "exact bridge product identity failed")
    _require(maximum_endpoint_sector_error <= 3e-13,
             "exact bridge endpoint sector identity failed")
    return {
        "case_count": len(cases),
        "maximum_step_error": maximum_step_error,
        "maximum_product_error": maximum_product_error,
        "maximum_endpoint_sector_error": maximum_endpoint_sector_error,
        "cases": cases,
    }


def _auxiliary_generators(H, model, frame):
    rows, columns = H.shape
    generators = []
    for a_row in range(columns):
        for bit in range(rows):
            delta_a, delta_b = auxiliary_stabilizer_delta(H, a_row, 1 << bit)
            generators.append(join_hgp_state(delta_a, delta_b))
    result = np.ascontiguousarray(generators, dtype=np.uint8)
    residual = model.H_check.astype(np.int64) @ result.T.astype(np.int64) % 2
    labels = np.asarray([state_label(frame, row) for row in result], dtype=np.uint64)
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    rank = int(gf2_rank(result))
    _require(not residual.any(), "auxiliary coordinate left the HGP kernel")
    _require(not labels.any(), "auxiliary coordinate changed a logical label")
    _require(rank == int(model.stabilizer_rows.shape[0]),
             "auxiliary coordinates do not span the stabilizer subgroup")
    return result, {
        "generator_count": int(result.shape[0]),
        "rank": rank,
        "stabilizer_rank": int(model.stabilizer_rows.shape[0]),
        "kernel": True,
        "logical_label_zero": True,
        "sha256": hashlib.sha256(result.tobytes()).hexdigest(),
    }


def _lowest_same_sector_tail(epsilon, generators):
    candidates = []
    for generator in generators:
        if not generator.any():
            continue
        packed = np.packbits(generator, bitorder="little").tobytes()
        candidates.append((int((epsilon ^ generator).sum()), packed, generator))
    _require(candidates, "auxiliary stabilizer basis unexpectedly has no nonzero row")
    _, _, selected = min(candidates, key=lambda value: (value[0], value[1]))
    return np.ascontiguousarray(selected, dtype=np.uint8)


def _sorted_catalog_indices(catalog):
    def key(index):
        return (
            int(catalog.weights[index]), int(catalog.signatures[index]),
            np.packbits(catalog.moves[index], bitorder="little").tobytes(),
        )

    return sorted(range(catalog.size), key=key)


def _residual(model, state):
    return (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8)


def _run_fixed_sector_chain(H, model, frame, p, initial, expected_syndrome,
                            observed_bits, burn_sweeps, measurement_sweeps, seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    state = np.ascontiguousarray(initial, dtype=np.uint8).copy()
    _require(np.array_equal(_residual(model, state), expected_syndrome),
             "bridge chain initial state has the wrong intermediate syndrome")
    expected_label = state_label(frame, state)
    A, B = split_hgp_state(state, H)
    rng = PortablePrng(int(seed))
    start = time.perf_counter()
    for _ in range(int(burn_sweeps)):
        A, B, _ = auxiliary_stabilizer_sweep(
            H, A, B, p, rng, row_order=rng.permutation(H.shape[1]),
        )
    burn_state = join_hgp_state(A, B)
    observed_bits = tuple(sorted({int(bit) for bit in observed_bits}))
    _require(observed_bits, "bridge chain has no observed path bit")
    bit_values = {str(bit): [] for bit in observed_bits}
    weights = []
    for _ in range(int(measurement_sweeps)):
        A, B, _ = auxiliary_stabilizer_sweep(
            H, A, B, p, rng, row_order=rng.permutation(H.shape[1]),
        )
        state = join_hgp_state(A, B)
        _require(np.array_equal(_residual(model, state), expected_syndrome),
                 "auxiliary chain left its intermediate hard coset")
        _require(state_label(frame, state) == expected_label,
                 "auxiliary chain left its fixed logical sector")
        for bit in observed_bits:
            bit_values[str(bit)].append(int(state[bit]))
        weights.append(int(state.sum()))
    final_state = join_hgp_state(A, B)
    return {
        "seed": int(seed),
        "initial_weight": int(initial.sum()),
        "burn_weight": int(burn_state.sum()),
        "final_weight": int(final_state.sum()),
        "initial_label": int(expected_label),
        "measurement_bit_values": bit_values,
        "measurement_weights": weights,
        "mean_bit_one": {
            bit: float(np.mean(values)) for bit, values in bit_values.items()
        },
        "mean_weight": float(np.mean(weights)),
        "wall_seconds": time.perf_counter() - start,
    }


def _m8_context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], 0, "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < 0.04
    ).astype(np.uint8)
    syndrome = _residual(model, epsilon)
    _require(syndrome.any(), "m8 hard sentinel syndrome unexpectedly vanished")
    return registry, code, np.ascontiguousarray(H), model, frame, epsilon, syndrome


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "probe": Path(__file__),
        "sector_bridge": EXP102_ROOT / "exp102_pipeline" / "q0_sector_bridge.py",
        "auxiliary": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py",
        "collapsed": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "global": EXP102_ROOT / "exp102_pipeline" / "q0_global.py",
        "registry": REGISTRY_PATH,
    }
    files = {name: sha256_file(path) for name, path in paths.items()}
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _probe_m8(args):
    registry, code, H, model, frame, epsilon, syndrome = _m8_context()
    generators, auxiliary = _auxiliary_generators(H, model, frame)
    tail = _lowest_same_sector_tail(epsilon, generators)
    _require(state_label(frame, tail) == 0, "selected S-tail changed label")
    _require(not _residual(model, tail).any(), "selected S-tail left kernel")
    starts = {
        "P": epsilon,
        "S": np.ascontiguousarray(epsilon ^ tail, dtype=np.uint8),
    }
    _require(state_label(frame, starts["P"]) == state_label(frame, starts["S"]),
             "P/S starts are not in the same logical sector")
    catalog = build_logical_proposal_catalog(model, frame)
    selected = _sorted_catalog_indices(catalog)[:int(args.moves)]
    _require(selected, "no logical bridge move was selected")
    p = 0.04
    odds = p / (1.0 - p)
    move_reports = []
    for move_order, catalog_index in enumerate(selected):
        move = np.ascontiguousarray(catalog.moves[catalog_index], dtype=np.uint8)
        support, prefixes = logical_bridge_prefixes(move)
        stage_reports = []
        for stage, prefix in enumerate(prefixes):
            prefix = prefixes[stage]
            expected_syndrome = syndrome ^ _residual(model, prefix)
            family_reports = {}
            for family, start in starts.items():
                chains = []
                for chain in range(int(args.chains)):
                    seed = derive_seed(
                        PROBE_VERSION, "m08_c06", int(catalog_index), int(stage),
                        family, int(chain), int(args.burn), int(args.measurement),
                    )
                    chains.append(_run_fixed_sector_chain(
                        H, model, frame, p, start ^ prefix, expected_syndrome,
                        support, args.burn, args.measurement, seed,
                    ))
                per_bit_means = {
                    str(bit): float(np.mean([
                        row["mean_bit_one"][str(int(bit))] for row in chains
                    ]))
                    for bit in support
                }
                family_reports[family] = {
                    "mean_bit_one_by_path_bit": per_bit_means,
                    "chain_mean_bit_one_by_path_bit": {
                        str(bit): [
                            row["mean_bit_one"][str(int(bit))] for row in chains
                        ]
                        for bit in support
                    },
                    "chains": chains,
                }
            stage_reports.append({
                "stage": int(stage),
                "prefix_weight": int(prefix.sum()),
                "intermediate_syndrome_weight": int(expected_syndrome.sum()),
                "families": family_reports,
            })
        step_reports = []
        forward_products = {family: 1.0 for family in starts}
        reverse_products = {family: 1.0 for family in starts}
        for stage, bit in enumerate(support):
            family_steps = {}
            for family in starts:
                source_probability = stage_reports[stage]["families"][family][
                    "mean_bit_one_by_path_bit"
                ][str(int(bit))]
                target_probability = stage_reports[stage + 1]["families"][family][
                    "mean_bit_one_by_path_bit"
                ][str(int(bit))]
                forward_ratio = bridge_step_ratio(p, source_probability)
                reverse_ratio = reverse_bridge_step_ratio(p, target_probability)
                forward_products[family] *= forward_ratio
                reverse_products[family] *= reverse_ratio
                family_steps[family] = {
                    "source_bit_one": source_probability,
                    "target_bit_one": target_probability,
                    "forward_ratio": forward_ratio,
                    "reverse_ratio": reverse_ratio,
                    "forward_minus_reverse": forward_ratio - reverse_ratio,
                }
            step_reports.append({
                "stage": int(stage), "path_bit": int(bit),
                "P_minus_S_source_bit_mean": (
                    family_steps["P"]["source_bit_one"]
                    - family_steps["S"]["source_bit_one"]
                ),
                "families": family_steps,
            })
        move_reports.append({
            "catalog_index": int(catalog_index),
            "logical_move_weight": int(move.sum()),
            "logical_signature": int(catalog.signatures[catalog_index]),
            "P_endpoint_weight": int((epsilon ^ move).sum()),
            "S_endpoint_weight": int((starts["S"] ^ move).sum()),
            "path_support": support.astype(int).tolist(),
            "per_family_forward_product_ratio": forward_products,
            "per_family_reverse_product_ratio": reverse_products,
            "stages": stage_reports,
            "steps": step_reports,
        })
    return {
        "cell": {
            "code_id": code["code_id"], "p": p, "disorder_index": 0,
            "disorder_source": "attempt022",
        },
        "dimensions": {
            "num_qubits": int(model.num_qubits), "logical_dimension": int(model.k),
            "stabilizer_dimension": int(model.stabilizer_rows.shape[0]),
        },
        "registry_sha256": registry["registry_sha256"],
        "planted_weight": int(epsilon.sum()),
        "syndrome_weight": int(syndrome.sum()),
        "p_label": int(state_label(frame, starts["P"])),
        "s_label": int(state_label(frame, starts["S"])),
        "s_tail_weight": int(tail.sum()),
        "s_tail_endpoint_weight": int(starts["S"].sum()),
        "auxiliary_span": auxiliary,
        "catalog_sha256": catalog.catalog_sha256,
        "moves": move_reports,
        "odds": odds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--moves", type=int, default=1)
    parser.add_argument("--burn", type=int, default=4)
    parser.add_argument("--measurement", type=int, default=12)
    parser.add_argument("--chains", type=int, default=1)
    args = parser.parse_args(argv)
    for name in ("moves", "burn", "measurement", "chains"):
        _require(int(getattr(args, name)) > 0, f"{name} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    source_binding = _source_binding()
    core = {
        "probe_version": PROBE_VERSION,
        "bridge_version": SECTOR_BRIDGE_VERSION,
        "status": "EXPLORATORY_NO_QTOP_OR_READINESS_CLAIM",
        "scope": {
            "posterior_estimation": False,
            "formal_authorization": False,
            "remote_authorization": False,
            "production_authorization": False,
        },
        "fixed_schedule": {
            "moves": int(args.moves), "burn_sweeps": int(args.burn),
            "measurement_sweeps": int(args.measurement), "chains_per_family": int(args.chains),
            "families": ["P", "S"],
            "logical_path_order": "increasing_physical_bit_index",
        },
        "exact_oracle": _exact_oracle(),
        "m8_probe": _probe_m8(args),
        "source_binding": source_binding,
        "unresolved_requirements": [
            "No proof that the short within-sector trajectories are stationary.",
            "Forward and reverse bridge identities are only diagnostics at this short clock.",
            "No enumeration or rigorous bound for omitted logical sectors.",
            "No q_top, sector-mass, or physical estimate is produced.",
            "A future truth-free anchor construction must not use planted epsilon.",
        ],
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(sha256_json({"output": str(args.output), "report_sha256": report["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
