#!/usr/bin/env python3
"""Combine accepted Stage D TI records after a targeted rerun.

The six-point run passed D1/D3 for every point but missed one D2 CI by a
small amount on record 1.  This script replaces only that record with the
longer targeted rerun and recomputes the Stage D gates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
STAGE_DIR = SCRIPT_PATH.parent
FULL_DIR = STAGE_DIR / "full_linear_m1024"
RERUN_DIR = STAGE_DIR / "rerun_r1_linear_m2048"
OUTPUT_DIR = STAGE_DIR / "accepted_combined"

TV_THRESHOLD = 0.02
QTOP_THRESHOLD = 0.02
GRID_TV_THRESHOLD = 0.02
GRID_QTOP_THRESHOLD = 0.02


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def compute_gates(records: list[dict]) -> dict:
    d1_passed = all(record["d1_passed"] for record in records)
    d2_passed = all(record["d2_passed"] for record in records)
    d3_passed = all(record["d3_passed"] for record in records)
    return {
        "D1": {
            "passed": bool(d1_passed),
            "max_tv": float(max(record["tv"] for record in records)),
        },
        "D2": {
            "passed": bool(d2_passed),
            "max_q_top_abs_diff": float(
                max(record["q_top_abs_diff"] for record in records)
            ),
            "num_ci_misses": int(
                sum(not record["ci_covers_exact"] for record in records)
            ),
        },
        "D3": {
            "passed": bool(d3_passed),
            "max_grid_tv": float(max(record["grid_tv"] for record in records)),
            "max_grid_q_top_abs_diff": float(
                max(record["grid_q_top_abs_diff"] for record in records)
            ),
        },
    }


def write_comparison_csv(records: list[dict]) -> None:
    with (OUTPUT_DIR / "ti_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_id",
                "p_value",
                "q_value",
                "exact_q_top",
                "ti_q_top",
                "q_top_ci_low",
                "q_top_ci_high",
                "q_top_abs_diff",
                "tv",
                "grid_tv",
                "grid_q_top_abs_diff",
                "ci_covers_exact",
                "d1_passed",
                "d2_passed",
                "d3_passed",
                "acceptance_mean",
                "sector_reject_mean",
                "wall_time_seconds",
                "source_run",
            ),
        )
        writer.writeheader()
        for record in records:
            q_ci = record["q_top_ci95"]
            writer.writerow({
                "record_id": record["record_id"],
                "p_value": record["p_value"],
                "q_value": record["q_value"],
                "exact_q_top": record["exact_q_top"],
                "ti_q_top": record["q_top"],
                "q_top_ci_low": q_ci[0],
                "q_top_ci_high": q_ci[1],
                "q_top_abs_diff": record["q_top_abs_diff"],
                "tv": record["tv"],
                "grid_tv": record["grid_tv"],
                "grid_q_top_abs_diff": record["grid_q_top_abs_diff"],
                "ci_covers_exact": record["ci_covers_exact"],
                "d1_passed": record["d1_passed"],
                "d2_passed": record["d2_passed"],
                "d3_passed": record["d3_passed"],
                "acceptance_mean": record["acceptance_mean"],
                "sector_reject_mean": record["sector_reject_mean"],
                "wall_time_seconds": record["wall_time_seconds"],
                "source_run": record["source_run"],
            })


def write_npz(records: list[dict]) -> None:
    np.savez_compressed(
        OUTPUT_DIR / "ti_results.npz",
        record_id=np.asarray([record["record_id"] for record in records], dtype=np.int64),
        p_value=np.asarray([record["p_value"] for record in records], dtype=np.float64),
        q_value=np.asarray([record["q_value"] for record in records], dtype=np.float64),
        exact_weights=np.stack([as_array(record["exact_weights"]) for record in records]),
        ti_weights=np.stack([as_array(record["weights"]) for record in records]),
        coarse_weights=np.stack([as_array(record["coarse_weights"]) for record in records]),
        exact_q_top=np.asarray([record["exact_q_top"] for record in records], dtype=np.float64),
        ti_q_top=np.asarray([record["q_top"] for record in records], dtype=np.float64),
        q_top_ci95=np.stack([as_array(record["q_top_ci95"]) for record in records]),
        tv=np.asarray([record["tv"] for record in records], dtype=np.float64),
        q_top_abs_diff=np.asarray(
            [record["q_top_abs_diff"] for record in records],
            dtype=np.float64,
        ),
        grid_tv=np.asarray([record["grid_tv"] for record in records], dtype=np.float64),
        grid_q_top_abs_diff=np.asarray(
            [record["grid_q_top_abs_diff"] for record in records],
            dtype=np.float64,
        ),
        source_run=np.asarray([record["source_run"] for record in records]),
    )


def write_summary(payload: dict) -> None:
    gates = payload["gates"]
    records = payload["records"]
    lines = [
        "# Stage D sector-resolved TI summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        (
            "Estimator: sector-resolved thermodynamic integration with the "
            "Stage C fixed-sector sampler.  The accepted benchmark uses "
            "`linear_kernel`, which preserves `P_L x` exactly and is equivalent "
            "to the corrected decoder-sector labels for this L=2 zero-disorder "
            "linear-section reference."
        ),
        "Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.",
        (
            "Composition: records 0,2,3,4,5 from `full_linear_m1024`; "
            "record 1 from `rerun_r1_linear_m2048`."
        ),
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| D1 | TV(w_TI,w_exact) <= {TV_THRESHOLD:.3f} | "
            f"max TV={gates['D1']['max_tv']:.4g} | "
            f"{'PASS' if gates['D1']['passed'] else 'FAIL'} |"
        ),
        (
            f"| D2 | abs dq_top <= {QTOP_THRESHOLD:.3f} and CI covers exact | "
            f"max abs dq={gates['D2']['max_q_top_abs_diff']:.4g}, "
            f"CI misses={gates['D2']['num_ci_misses']} | "
            f"{'PASS' if gates['D2']['passed'] else 'FAIL'} |"
        ),
        (
            f"| D3 | coarse/fine grid TV and abs dq <= {GRID_TV_THRESHOLD:.3f} | "
            f"max grid TV={gates['D3']['max_grid_tv']:.4g}, "
            f"max grid dq={gates['D3']['max_grid_q_top_abs_diff']:.4g} | "
            f"{'PASS' if gates['D3']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Point Comparison",
        "",
        "| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | source | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        gate_text = "/".join([
            "D1" if record["d1_passed"] else "d1FAIL",
            "D2" if record["d2_passed"] else "d2FAIL",
            "D3" if record["d3_passed"] else "d3FAIL",
        ])
        q_ci = record["q_top_ci95"]
        lines.append(
            f"| {record['record_id']} | {record['p_value']:.6f} | "
            f"{record['q_value']:.6f} | {record['exact_q_top']:.6f} | "
            f"{record['q_top']:.6f} | [{q_ci[0]:.6f}, {q_ci[1]:.6f}] | "
            f"{record['tv']:.5f} | {record['grid_tv']:.5f} | "
            f"{record['grid_q_top_abs_diff']:.5f} | "
            f"{record['source_run']} | {gate_text} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `stageD_results.json`",
        "- `ti_results.npz`",
        "- `ti_comparison.csv`",
    ])
    (OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_payload = load_payload(FULL_DIR / "stageD_results.json")
    rerun_payload = load_payload(RERUN_DIR / "stageD_results.json")

    full_records = {
        int(record["record_id"]): record
        for record in full_payload["records"]
    }
    rerun_records = {
        int(record["record_id"]): record
        for record in rerun_payload["records"]
    }
    full_records[1] = rerun_records[1]

    records = []
    for record_id in sorted(full_records):
        record = dict(full_records[record_id])
        record["source_run"] = (
            "rerun_r1_linear_m2048"
            if record_id == 1
            else "full_linear_m1024"
        )
        records.append(record)

    gates = compute_gates(records)
    overall_passed = bool(
        gates["D1"]["passed"]
        and gates["D2"]["passed"]
        and gates["D3"]["passed"]
    )
    payload = {
        "stage": "D",
        "overall_passed": overall_passed,
        "code_family": full_payload["code_family"],
        "lattice_size": full_payload["lattice_size"],
        "projection_mode": full_payload.get("projection_mode"),
        "sector_observable": full_payload.get("sector_observable"),
        "sampler_mode": "linear_kernel",
        "section_prefer_bplsd": full_payload.get("section_prefer_bplsd"),
        "reference_path": full_payload.get("reference_path"),
        "disorder": full_payload["disorder"],
        "thresholds": {
            "tv": TV_THRESHOLD,
            "q_top_abs_diff": QTOP_THRESHOLD,
            "grid_tv": GRID_TV_THRESHOLD,
            "grid_q_top_abs_diff": GRID_QTOP_THRESHOLD,
        },
        "source_runs": {
            "records_0_2_3_4_5": str(FULL_DIR.relative_to(STAGE_DIR)),
            "record_1": str(RERUN_DIR.relative_to(STAGE_DIR)),
        },
        "gates": gates,
        "records": records,
    }

    write_comparison_csv(records)
    write_npz(records)
    (OUTPUT_DIR / "stageD_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(payload)

    print(json.dumps({
        "overall_passed": overall_passed,
        "D1_max_tv": gates["D1"]["max_tv"],
        "D2_max_q_top_abs_diff": gates["D2"]["max_q_top_abs_diff"],
        "D2_num_ci_misses": gates["D2"]["num_ci_misses"],
        "D3_max_grid_tv": gates["D3"]["max_grid_tv"],
        "D3_max_grid_q_top_abs_diff": gates["D3"]["max_grid_q_top_abs_diff"],
    }, indent=2, sort_keys=True))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
