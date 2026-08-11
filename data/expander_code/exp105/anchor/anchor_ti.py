"""Track B: the transport-free `q_top` anchor at m = 2 and m = 3.

Track A measures what a frozen decoder does. This measures what the posterior
itself says, for the two sizes where it can be measured at all: full-sector
thermodynamic integration is limited to `k <= 10`, and `k = m²`, so `m <= 3`.

It is called transport-free because it never asks a trajectory to move between
logical sectors. Each sector gets its own chain at fixed label, and the barrier
between sectors appears as a difference of independently computed free energies.
That is exactly why this route survives where the parallel-tempering route does
not: the gate that kills `m >= 4` is a requirement on cold-end logical moves, and
there are none here.

Boundaries, from EXPERIMENT_CONTRACT.md section 8:

- Track B is a preregistered **secondary**. It cannot change Track A's terminal
  status in either direction.
- It writes its own `exp105.anchor.raw.v1` schema and its own loader. It never
  produces `scan_results.npz`, never claims `exp101.scan.v3`, and is never passed
  to `src.scan_results.load_publication_q_top`.
- It asserts no threshold and performs no finite-size scaling. Two sizes are two
  sizes.

Aggregation is fail-closed with the same semantics as scan v3: a disorder whose
TI grid-refinement gate fails is `INVALID`, and one `INVALID` or missing disorder
makes the whole `(m, p)` point `NaN`. Raw estimates are stored unclipped.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data.expander_code.exp105.exp105_pipeline import (
    ANCHOR_AGGREGATE_SCHEMA,
    ANCHOR_RAW_SCHEMA,
    EXPERIMENT_ID,
)
from data.expander_code.exp105.exp105_pipeline.config import (
    ANCHOR_M_VALUES,
    MASTER_SEED_HEX,
    NAMESPACES,
    P_TOKENS,
    Q_TOKEN,
)
from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp105.exp105_pipeline.io import atomic_json, sha256_json
from .sector_ti_fast import NUMBA_AVAILABLE, fast_chain_installed


# Frozen before the anchor runs. Eight codes is the same panel width exp102
# froze for its registry, and eight disorders per (code, p) is what keeps the
# whole anchor inside a couple of local core-hours at m = 3.
ANCHOR_CODES_PER_M = 8
ANCHOR_DISORDERS = 8
ANCHOR_SEED_NAMESPACE = NAMESPACES["anchor"]


def _digest_seed(payload):
    import hashlib

    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8], "big",
    ) & ((1 << 63) - 1)


def disorder_seed(m, code_index, p_token, disorder_index):
    return _digest_seed(":".join([
        MASTER_SEED_HEX, ANCHOR_SEED_NAMESPACE, "disorder",
        str(int(m)), str(int(code_index)), str(p_token), Q_TOKEN,
        str(int(disorder_index)),
    ]))


def chain_seed(m, code_index, p_token, disorder_index):
    return _digest_seed(":".join([
        MASTER_SEED_HEX, ANCHOR_SEED_NAMESPACE, "chain",
        str(int(m)), str(int(code_index)), str(p_token), Q_TOKEN,
        str(int(disorder_index)),
    ]))


def _build_model(m, code_index):
    """Build one anchor code from the production ensemble namespace.

    m = 3 anchor codes are literally production codes 0..7, so the anchor and
    the Track A curve refer to the same objects at the size they share.
    """
    from data.expander_code.exp105.exp105_pipeline.ensemble import generate_codes

    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame
    from exp101_certified_src.graphs import random_biregular_graph_from_m
    from exp101_certified_src.hgp import classical_parity_check_matrix

    panel, _ = generate_codes(
        m, code_index + 1, namespace=NAMESPACES["ensemble"], with_distance=True,
    )
    row = panel[code_index]
    graph = random_biregular_graph_from_m(m, 3, 4, int(row["graph_seed"]))
    H = classical_parity_check_matrix(graph)
    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    return model, frame, row


def run_one_cell(payload):
    """One (m, code, p) cell: every disorder, fail-closed."""
    m, code_index, p_token = payload
    load_exp101()
    from exp101_certified_src import sector_ti as ti
    from exp101_certified_src.model import disorder_from_uniforms, wire_ensemble

    model, frame, row = _build_model(m, code_index)
    config = ti.SectorTiConfig()
    q = float(Q_TOKEN)
    p = float(p_token)

    q_top = np.full(ANCHOR_DISORDERS, np.nan)
    q_top_stderr = np.full(ANCHOR_DISORDERS, np.nan)
    map_success = np.full(ANCHOR_DISORDERS, np.nan)
    planted_mass = np.full(ANCHOR_DISORDERS, np.nan)
    grid_tv = np.full(ANCHOR_DISORDERS, np.nan)
    grid_q_top_diff = np.full(ANCHOR_DISORDERS, np.nan)
    valid = np.zeros(ANCHOR_DISORDERS, dtype=np.bool_)
    flags = []

    started = time.perf_counter()
    with fast_chain_installed():
        for index in range(ANCHOR_DISORDERS):
            rng = np.random.default_rng(disorder_seed(m, code_index, p_token, index))
            disorder = disorder_from_uniforms(
                model, p, q,
                data_uniforms=rng.random(model.num_qubits),
                syndrome_uniforms=rng.random(model.num_checks),
            )
            wiring = wire_ensemble(model, disorder, "true_posterior", frame)
            result = ti.run_sector_ti(
                model, frame, wiring, config,
                chain_seed(m, code_index, p_token, index),
            )
            # Stored raw and unclipped, whatever the gate says.
            q_top[index] = float(result["q_top"])
            q_top_stderr[index] = float(result["q_top_stderr"])
            map_success[index] = float(result["map_success_probability"])
            planted_mass[index] = float(result["posterior_mass_on_planted_class"])
            grid_tv[index] = float(result.get("grid_tv", np.nan))
            grid_q_top_diff[index] = float(result.get("grid_q_top_abs_diff", np.nan))
            valid[index] = bool(result.get("valid_for_aggregation", False))
            flags.append(str(result.get("flags", "")))

    return {
        "m": int(m),
        "code_index": int(code_index),
        "graph_seed": int(row["graph_seed"]),
        "classical_distance": int(row["classical_distance"]),
        "n": int(model.num_qubits),
        "k": int(model.k),
        "p_token": p_token,
        "q_token": Q_TOKEN,
        "q_top": q_top.tolist(),
        "q_top_stderr": q_top_stderr.tolist(),
        "map_success_probability": map_success.tolist(),
        "posterior_mass_on_planted_class": planted_mass.tolist(),
        "grid_tv": grid_tv.tolist(),
        "grid_q_top_abs_diff": grid_q_top_diff.tolist(),
        "valid_for_aggregation": valid.tolist(),
        "flags": flags,
        "wall_seconds": time.perf_counter() - started,
    }


def planned_cells(m_values=None, p_tokens=None):
    m_values = list(ANCHOR_M_VALUES if m_values is None else m_values)
    p_tokens = list(P_TOKENS if p_tokens is None else p_tokens)
    return [
        (m, code_index, token)
        for m in m_values
        for code_index in range(ANCHOR_CODES_PER_M)
        for token in p_tokens
    ]


def run_anchor(output_path, num_workers, m_values=None, p_tokens=None):
    cells = planned_cells(m_values, p_tokens)
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        for result in executor.map(run_one_cell, cells):
            results.append(result)
            print(
                f"m={result['m']} c{result['code_index']} p={result['p_token']} "
                f"valid={sum(result['valid_for_aggregation'])}/{ANCHOR_DISORDERS} "
                f"{result['wall_seconds']:.1f}s",
                flush=True,
            )
    core = {
        "schema_version": ANCHOR_RAW_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "role": "preregistered_secondary_anchor",
        "authority": (
            "cannot change Track A's terminal status; never passed to "
            "exp101 load_publication_q_top; asserts no threshold and no "
            "finite-size scaling"
        ),
        "numba_fast_path": bool(NUMBA_AVAILABLE),
        "master_seed_hex": MASTER_SEED_HEX,
        "seed_namespace": ANCHOR_SEED_NAMESPACE,
        "ensemble_namespace": NAMESPACES["ensemble"],
        "q_token": Q_TOKEN,
        "p_tokens": list(P_TOKENS if p_tokens is None else p_tokens),
        "m_values": list(ANCHOR_M_VALUES if m_values is None else m_values),
        "codes_per_m": ANCHOR_CODES_PER_M,
        "disorders_per_cell": ANCHOR_DISORDERS,
        "wall_seconds": time.perf_counter() - started,
        "cells": results,
    }
    report = dict(core, raw_sha256=sha256_json(core))
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def aggregate_anchor(raw):
    """Fail-closed aggregation: one invalid or missing disorder voids the point."""
    if raw["schema_version"] != ANCHOR_RAW_SCHEMA:
        raise ValueError("unexpected anchor raw schema")
    p_tokens = list(raw["p_tokens"])
    per_m = {}
    for m in raw["m_values"]:
        rows = {}
        for token in p_tokens:
            cells = [
                cell for cell in raw["cells"]
                if cell["m"] == m and cell["p_token"] == token
            ]
            planned = int(raw["codes_per_m"])
            present = len(cells)
            values = []
            invalid = 0
            for cell in cells:
                for index, ok in enumerate(cell["valid_for_aggregation"]):
                    if ok:
                        values.append(float(cell["q_top"][index]))
                    else:
                        invalid += 1
            if present < planned:
                status = "INCOMPLETE"
            elif invalid:
                status = "SAMPLING_INSUFFICIENT"
            else:
                status = "REPORTABLE"
            reportable = status == "REPORTABLE"
            array = np.asarray(values, dtype=np.float64)
            rows[token] = {
                "status": status,
                "planned_codes": planned,
                "present_codes": present,
                "invalid_disorders": invalid,
                "samples": int(array.size),
                "mean_q_top": float(array.mean()) if reportable else float("nan"),
                "sem_q_top": (
                    float(array.std(ddof=1) / np.sqrt(array.size))
                    if reportable and array.size > 1 else float("nan")
                ),
                "min_q_top": float(array.min()) if reportable else float("nan"),
                "max_q_top": float(array.max()) if reportable else float("nan"),
            }
        per_m[str(m)] = rows
    core = {
        "schema_version": ANCHOR_AGGREGATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "raw_sha256": raw["raw_sha256"],
        "q_token": raw["q_token"],
        "p_tokens": p_tokens,
        "aggregation_policy": {
            "point_eligibility": "all_planned_disorders_valid",
            "maximum_invalid_disorders": 0,
            "maximum_missing_codes": 0,
            "raw_estimates_are_clipped": False,
        },
        "per_m": per_m,
    }
    return dict(core, aggregate_sha256=sha256_json(core))


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp105 Track B q_top anchor")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--m-values", type=int, nargs="+", default=None)
    parser.add_argument("--aggregate-output", default=None)
    args = parser.parse_args(argv)

    raw = run_anchor(args.output, args.num_workers, m_values=args.m_values)
    aggregate = aggregate_anchor(raw)
    if args.aggregate_output:
        atomic_json(args.aggregate_output, aggregate)
    print(json.dumps({
        "raw_sha256": raw["raw_sha256"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "wall_seconds": round(raw["wall_seconds"], 1),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
