"""Measure why q_top is not available at m >= 4, rather than asserting it.

EXPERIMENT_CONTRACT.md section 2 rests on a quantitative claim: the exp101
parallel-tempering validity gate requires a pooled worst-basis cold logical
acceptance of at least 1e-4, and for this family that acceptance is smaller by
tens of orders of magnitude. This script measures the inputs to that claim from
the certified exp101 source and writes them down, so the contract cites a file
rather than an argument.

What it measures, per m:

- the weights of the exp101 logical_X move basis, which are what a cold L-move
  actually flips (`exp101/src/reference_mcmc.py`: dE = K_p * d|v|);
- the resulting acceptance bound exp(-K_p (1 - 2p) w) at the contract's p grid;
- the engine the exp101 router would resolve for (k, q), and the gate threshold.

It runs no Markov chain and produces no physical result.
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp105.exp105_pipeline.io import atomic_json, sha256_json


M_VALUES = [3, 4, 5, 6, 7, 8]
P_VALUES = [0.02, 0.05, 0.10, 0.20]
Q = 0.05


def main():
    load_exp101()
    from exp101_certified_src.gates import GateThresholds
    from exp101_certified_src.run_scan import build_code, resolve_engine
    from exp101_certified_src.model import assemble_sector_model

    thresholds = GateThresholds()
    rows = []
    for m in M_VALUES:
        H_Z, H_X, logicals, _ = build_code("expander34", m, "full_rank", None)
        model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
        weights = np.asarray(model.logical_move_basis).sum(axis=1)
        engine = resolve_engine("auto", model.k, Q)
        acceptance = {}
        for p in P_VALUES:
            K_p = math.log((1.0 - p) / p)
            # A cold L-move flips a whole basis logical; the syndrome term is
            # unchanged because logicals are in ker(H_check), so only the weight
            # term survives. The typical weight change of the flipped support is
            # w(1 - 2p), which makes this an upper bound on the acceptance.
            acceptance[str(p)] = {
                "K_p": K_p,
                "log10_acceptance_upper_bound_worst_basis": float(
                    -K_p * (1.0 - 2.0 * p) * int(weights.max()) / math.log(10.0)
                ),
                "log10_acceptance_upper_bound_lightest_basis": float(
                    -K_p * (1.0 - 2.0 * p) * int(weights.min()) / math.log(10.0)
                ),
            }
        rows.append({
            "m": m,
            "n": int(model.num_qubits),
            "n_checks": int(model.num_checks),
            "k": int(model.k),
            "resolved_engine_at_q": engine,
            "full_sector_ti_available": bool(model.k <= 10),
            "logical_move_basis_weight_min": int(weights.min()),
            "logical_move_basis_weight_median": int(np.median(weights)),
            "logical_move_basis_weight_max": int(weights.max()),
            "acceptance_upper_bounds": acceptance,
        })

    core = {
        "schema_version": "exp105.pt_gate_infeasibility.v1",
        "purpose": (
            "outcome-blind measurement of the exp101 PT validity gate inputs; "
            "no Markov chain is run and no physical result is produced"
        ),
        "q": Q,
        "gate_min_cold_logical_acceptance": float(
            thresholds.min_cold_logical_acceptance
        ),
        "gate_min_round_trips": int(thresholds.min_round_trips),
        "engine_routing_rule": "k<=10 full-sector TI; k>10 and q>0 parallel tempering",
        "per_m": rows,
    }
    report = dict(core, report_sha256=sha256_json(core))
    output = Path(__file__).resolve().parent / "pt_gate_infeasibility.json"
    atomic_json(output, report)
    print(f"wrote {output.name} sha256={report['report_sha256']}")
    for row in rows:
        worst = row["acceptance_upper_bounds"]["0.05"][
            "log10_acceptance_upper_bound_worst_basis"
        ]
        print(
            f"m={row['m']} k={row['k']} engine={row['resolved_engine_at_q']} "
            f"weights {row['logical_move_basis_weight_min']}/"
            f"{row['logical_move_basis_weight_median']}/"
            f"{row['logical_move_basis_weight_max']}  "
            f"log10 acceptance bound at p=0.05: {worst:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
