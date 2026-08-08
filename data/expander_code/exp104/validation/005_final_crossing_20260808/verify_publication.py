"""Independently re-verify the published exp104 aggregate on macmini.

The aggregate is produced on nd-3. This script reloads it here through the frozen
loader, which recomputes rates, Wilson intervals, pooled means, cluster standard
errors, the distance-strata table, the cluster bootstrap, the simultaneous band,
the terminal decision and the crossing location from the stored per-code counts.
An aggregate this refuses is not a result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[5])
sys.path.insert(0, ROOT)

from data.expander_code.exp104.exp104_pipeline.aggregate import DISTANCE_STRATA
from data.expander_code.exp104.exp104_pipeline.config import M_VALUES, load_config
from data.expander_code.exp104.exp104_pipeline.io import atomic_json, sha256_file
from data.expander_code.exp104.exp104_pipeline.loader import load_exp104_crossing

REMOTE_CONFIG = f"{ROOT}/data/expander_code/exp104/config/ensemble_mc.remote.v1.json"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = load_config(REMOTE_CONFIG)
    payload = load_exp104_crossing(args.aggregate, config)

    p_values = payload["p_values"]
    print("per-m ensemble mean block logical failure rate")
    print("   p    " + "".join(f"{m:>9d}" for m in M_VALUES) + "     Delta38    simultaneous band")
    for index, p in enumerate(p_values):
        line = f"  {p:.2f} " + "".join(
            f"{payload['primary_mean'][j, index]:9.4f}" for j in range(len(M_VALUES))
        )
        mark = ""
        if payload["delta38_band_high"][index] < 0:
            mark = "  certified negative"
        elif payload["delta38_band_low"][index] > 0:
            mark = "  certified positive"
        line += (
            f"  {payload['delta38'][index]:+9.4f}"
            f"  [{payload['delta38_band_low'][index]:+.4f},"
            f"{payload['delta38_band_high'][index]:+.4f}]{mark}"
        )
        print(line)

    print()
    print(f"terminal status          : {payload['terminal_status']}")
    print(f"simultaneous half-width  : {payload['bootstrap_half_width']:.4f}")
    print(
        f"certified bracket        : "
        f"[{payload['crossing_bracket_low']:.2f}, {payload['crossing_bracket_high']:.2f}]"
    )
    print(
        f"p_cross                  : {payload['p_cross']:.5f}  "
        f"95% CI [{payload['p_cross_low']:.5f}, {payload['p_cross_high']:.5f}]  "
        f"defined {payload['p_cross_defined_fraction']:.4f}"
    )
    print(f"replay                   : {payload['replay_status']} ({payload['replay_scope']})")
    print(
        f"cells                    : "
        f"{int((payload['code_status'] == 'REPORTABLE').sum())} of "
        f"{payload['code_status'].size} REPORTABLE"
    )
    print(
        f"precision                : largest cluster SE "
        f"{np.nanmax(payload['cluster_se']):.5f}, largest between-code std "
        f"{np.nanmax(payload['between_code_std']):.4f}"
    )

    print("\nensemble composition actually drawn (fraction of 2000 codes per m)")
    print("   m  " + "".join(f"{'d=' + str(d):>9s}" for d in DISTANCE_STRATA))
    for j, m in enumerate(M_VALUES):
        print(f"   {m}  " + "".join(
            f"{payload['strata_code_counts'][j, i] / 2000.0:9.4f}"
            for i in range(len(DISTANCE_STRATA))
        ))

    print("\ndistance-stratified failure rate at m=8 (preregistered secondary)")
    print("   p    " + "".join(f"{'d=' + str(d):>9s}" for d in DISTANCE_STRATA))
    last = len(M_VALUES) - 1
    for index, p in enumerate(p_values):
        print(f"  {p:.2f} " + "".join(
            f"{payload['strata_rate'][last, i, index]:9.4f}"
            for i in range(len(DISTANCE_STRATA))
        ))

    report = {
        "schema_version": "exp104.local_verification.v1",
        "status": "PASS",
        "aggregate_path": str(Path(args.aggregate).name),
        "aggregate_sha256": sha256_file(args.aggregate),
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "terminal_status": payload["terminal_status"],
        "overall_status": payload["overall_status"],
        "replay_status": payload["replay_status"],
        "bootstrap_half_width": float(payload["bootstrap_half_width"]),
        "crossing_bracket": [
            float(payload["crossing_bracket_low"]),
            float(payload["crossing_bracket_high"]),
        ],
        "p_cross": float(payload["p_cross"]),
        "p_cross_low": float(payload["p_cross_low"]),
        "p_cross_high": float(payload["p_cross_high"]),
        "p_cross_defined_fraction": float(payload["p_cross_defined_fraction"]),
        "reportable_cells": int((payload["code_status"] == "REPORTABLE").sum()),
        "total_cells": int(payload["code_status"].size),
        "max_cluster_se": float(np.nanmax(payload["cluster_se"])),
        "max_between_code_std": float(np.nanmax(payload["between_code_std"])),
        "delta38": [float(v) for v in payload["delta38"]],
        "delta38_band_low": [float(v) for v in payload["delta38_band_low"]],
        "delta38_band_high": [float(v) for v in payload["delta38_band_high"]],
        "primary_mean": [
            [float(v) for v in payload["primary_mean"][j]] for j in range(len(M_VALUES))
        ],
    }
    atomic_json(args.output, report)
    print("\nloader accepted the aggregate; verification written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
