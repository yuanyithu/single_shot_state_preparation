#!/usr/bin/env python3
"""exp38 P3 second-method subset check.

This wraps the exp37 Stage F stochastic bidirectional logical-loop bridge with
BAR, but reads the exact disorder seeds from the exp38 P2 TI NPZ.  exp38 P2
uses ``disorder_seed_scope=disorder_index`` so the old exp37 L-dependent seed
formula would compare the bridge against the wrong disorder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
EXP37_RUNNER = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "exp37"
    / "038_stageF_ti_grid_20260603"
    / "run_stageF_second_method_subset.py"
)

DEFAULT_SUBSET = "3:0.22:0,4:0.22:0,5:0.22:0"
TV_THRESHOLD = 0.03
QTOP_THRESHOLD = 0.02
FULL_PATH_BIDIRECTIONAL_GAP_THRESHOLD = 0.20
BAR_RESIDUAL_THRESHOLD = 1.0e-8


def _format_q(value: float) -> float:
    return float(round(float(value), 6))


def _load_exp37_runner():
    spec = importlib.util.spec_from_file_location(
        "exp37_stageF_second_method_subset_for_exp38_p3",
        EXP37_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import exp37 second-method runner: {EXP37_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ti_records_with_exp38_seed(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int)
        q_values = data["q_values"].astype(float)
        q_top = data["q_top_per_disorder"].astype(float)
        weights = data["weights_per_disorder"].astype(float)
        delta_f = data["delta_f_per_disorder"].astype(float)
        flags = data["flags_per_disorder"].astype("<U128")
        disorder_seed = data["disorder_seed_per_disorder"].astype(np.int64)
        sample_seed = data["sample_seed_per_disorder"].astype(np.int64)
        manifest_json = str(data["manifest_json"].item())
    records = {}
    seed_lookup = {}
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            for disorder_index in range(q_top.shape[2]):
                key = (int(lattice_size), _format_q(q_value), int(disorder_index))
                records[key] = {
                    "q_top": float(q_top[li, qi, disorder_index]),
                    "weights": weights[li, qi, disorder_index],
                    "delta_f": delta_f[li, qi, disorder_index],
                    "flags": str(flags[li, qi, disorder_index]),
                    "disorder_seed": int(disorder_seed[li, qi, disorder_index]),
                    "sample_seed": int(sample_seed[li, qi, disorder_index]),
                }
                seed_lookup[key] = int(disorder_seed[li, qi, disorder_index])
    records["_exp38_seed_lookup"] = seed_lookup
    records["_exp38_ti_manifest_json"] = manifest_json
    return records


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "lattice_size",
        "p_value",
        "q_value",
        "disorder_index",
        "seed",
        "ti_q_top",
        "bridge_q_top",
        "tv_vs_ti",
        "q_top_abs_diff_vs_ti",
        "max_adjacent_bidirectional_gap",
        "max_full_path_bidirectional_gap",
        "max_bar_residual",
        "mean_acceptance_rate",
        "min_acceptance_rate",
        "ti_flags",
        "p3a_passed",
        "p3b_passed",
        "passed",
        "wall_time_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def _write_summary(path: Path, payload: dict) -> None:
    gates = payload["gates"]
    lines = [
        "# exp38 P3 second-method subset summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        (
            "Estimator: stochastic bidirectional logical-loop bridge with BAR "
            "on adjacent lambda intervals, reusing the exp37 validated "
            "second-method path.  The only exp38-specific change is that "
            "disorder seeds are read from the P2 TI NPZ so the bridge compares "
            "against the same disorder realization."
        ),
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            "| P3a | sampled subset TI vs second method: TV <= 0.03 and |dq_top| <= 0.02 | "
            f"checks={gates['P3a']['num_checks']}, max TV={gates['P3a']['max_tv_vs_ti']:.6f}, "
            f"max |dq|={gates['P3a']['max_q_top_abs_diff_vs_ti']:.6f} | "
            f"{'PASS' if gates['P3a']['passed'] else 'FAIL'} |"
        ),
        (
            "| P3b | bidirectional consistency diagnostic within recorded stochastic threshold | "
            f"max full-path gap={gates['P3b']['max_full_path_bidirectional_gap']:.6f}, "
            f"max BAR residual={gates['P3b']['max_bar_residual']:.3e} | "
            f"{'PASS' if gates['P3b']['passed'] else 'FAIL'} |"
        ),
        (
            "| Coverage | at least one crossing-region check for each L=3,4,5 | "
            f"lattice_sizes={gates['coverage']['lattice_sizes']}, "
            f"num_checks={gates['coverage']['num_checks']} | "
            f"{'PASS' if gates['coverage']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Point Comparison",
        "",
        "| L | q | d | TI q_top | bridge q_top | TV | dq_top | full-path gap | seed | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in payload["records"]:
        lines.append(
            f"| {record['lattice_size']} | {record['q_value']:.3f} | "
            f"{record['disorder_index']} | {record['ti_q_top']:.6f} | "
            f"{record['bridge_q_top']:.6f} | {record['tv_vs_ti']:.6f} | "
            f"{record['q_top_abs_diff_vs_ti']:.6f} | "
            f"{record['max_full_path_bidirectional_gap']:.6f} | "
            f"{record['seed']} | {'PASS' if record['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `p3_second_method_subset.json`",
        "- `p3_second_method_subset.csv`",
        "- `stageF_second_method_subset.json` (raw exp37 runner output)",
        "- `stageF_second_method_subset.csv` (raw exp37 runner output)",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_p3_payload(raw_payload: dict, ti_results_path: Path) -> dict:
    records = []
    for record in raw_payload["records"]:
        p3a = bool(
            float(record["tv_vs_ti"]) <= TV_THRESHOLD
            and float(record["q_top_abs_diff_vs_ti"]) <= QTOP_THRESHOLD
        )
        p3b = bool(
            float(record["max_full_path_bidirectional_gap"])
            <= FULL_PATH_BIDIRECTIONAL_GAP_THRESHOLD
            and float(record["max_bar_residual"]) <= BAR_RESIDUAL_THRESHOLD
        )
        updated = dict(record)
        updated["p3a_passed"] = p3a
        updated["p3b_passed"] = p3b
        updated["passed"] = bool(p3a and p3b)
        records.append(updated)

    lattice_sizes = sorted({int(record["lattice_size"]) for record in records})
    p3a_passed = bool(records and all(record["p3a_passed"] for record in records))
    p3b_passed = bool(records and all(record["p3b_passed"] for record in records))
    coverage_passed = bool(set(lattice_sizes) == {3, 4, 5} and len(records) >= 3)
    payload = {
        "stage": "P3",
        "method": "stochastic_bidirectional_logical_loop_bridge_bar",
        "overall_passed": bool(p3a_passed and p3b_passed and coverage_passed),
        "ti_results_path": str(ti_results_path),
        "raw_stageF_payload_source": "stageF_second_method_subset.json",
        "thresholds": {
            "tv_vs_ti": TV_THRESHOLD,
            "q_top_abs_diff_vs_ti": QTOP_THRESHOLD,
            "max_full_path_bidirectional_gap": FULL_PATH_BIDIRECTIONAL_GAP_THRESHOLD,
            "max_bar_residual": BAR_RESIDUAL_THRESHOLD,
        },
        "config": raw_payload.get("config", {}),
        "numba_available": bool(raw_payload.get("numba_available")),
        "used_python_fallback": bool(raw_payload.get("used_python_fallback")),
        "wall_time_seconds": float(raw_payload.get("wall_time_seconds", 0.0)),
        "gates": {
            "P3a": {
                "passed": p3a_passed,
                "num_checks": int(len(records)),
                "max_tv_vs_ti": float(max(record["tv_vs_ti"] for record in records)),
                "max_q_top_abs_diff_vs_ti": float(
                    max(record["q_top_abs_diff_vs_ti"] for record in records)
                ),
            },
            "P3b": {
                "passed": p3b_passed,
                "max_full_path_bidirectional_gap": float(
                    max(record["max_full_path_bidirectional_gap"] for record in records)
                ),
                "max_bar_residual": float(max(record["max_bar_residual"] for record in records)),
                "threshold_full_path_bidirectional_gap": FULL_PATH_BIDIRECTIONAL_GAP_THRESHOLD,
                "threshold_bar_residual": BAR_RESIDUAL_THRESHOLD,
            },
            "coverage": {
                "passed": coverage_passed,
                "lattice_sizes": lattice_sizes,
                "num_checks": int(len(records)),
            },
        },
        "records": records,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--code-family", default="3d_toric")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--seed-base", type=int, default=639000)
    parser.add_argument("--common-disorder-across-q", action="store_true")
    parser.add_argument("--num-lambda-points", type=int, default=65)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=512)
    parser.add_argument("--num-measurements", type=int, default=16384)
    parser.add_argument("--num-sweeps-between-measurements", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--allow-python-fallback", action="store_true")
    args = parser.parse_args()

    module = _load_exp37_runner()
    original_load_ti_records = module._load_ti_records
    del original_load_ti_records
    seed_lookup_holder = {}

    def patched_load_ti_records(path: Path) -> dict:
        records = _load_ti_records_with_exp38_seed(Path(path))
        seed_lookup_holder.clear()
        seed_lookup_holder.update(records.pop("_exp38_seed_lookup"))
        records.pop("_exp38_ti_manifest_json", None)
        return records

    def patched_disorder_seed(
            seed_base: int,
            lattice_size: int,
            q_value: float,
            disorder_index: int,
            common_disorder_across_q: bool) -> int:
        del seed_base, common_disorder_across_q
        key = (int(lattice_size), _format_q(q_value), int(disorder_index))
        if key not in seed_lookup_holder:
            raise KeyError(f"missing exp38 P2 disorder seed for {key}")
        return int(seed_lookup_holder[key])

    module.DEFAULT_SUBSET = DEFAULT_SUBSET
    module.TV_THRESHOLD = TV_THRESHOLD
    module.QTOP_THRESHOLD = QTOP_THRESHOLD
    module._load_ti_records = patched_load_ti_records
    module._disorder_seed = patched_disorder_seed

    argv = [
        str(EXP37_RUNNER),
        "--ti-results",
        str(args.ti_results),
        "--output-dir",
        str(args.output_dir),
        "--subset",
        str(args.subset),
        "--code-family",
        str(args.code_family),
        "--p",
        str(args.p),
        "--seed-base",
        str(args.seed_base),
        "--num-lambda-points",
        str(args.num_lambda_points),
        "--num-burn-in-sweeps",
        str(args.num_burn_in_sweeps),
        "--num-measurements",
        str(args.num_measurements),
        "--num-sweeps-between-measurements",
        str(args.num_sweeps_between_measurements),
        "--seed-offset",
        str(args.seed_offset),
    ]
    if bool(args.common_disorder_across_q):
        argv.append("--common-disorder-across-q")
    if bool(args.allow_python_fallback):
        argv.append("--allow-python-fallback")

    old_argv = sys.argv
    sys.argv = argv
    try:
        raw_exit_code = int(module.main())
    finally:
        sys.argv = old_argv

    raw_json_path = args.output_dir / "stageF_second_method_subset.json"
    raw_csv_path = args.output_dir / "stageF_second_method_subset.csv"
    raw_payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    payload = _build_p3_payload(raw_payload, args.ti_results)

    p3_json_path = args.output_dir / "p3_second_method_subset.json"
    p3_csv_path = args.output_dir / "p3_second_method_subset.csv"
    p3_summary_path = args.output_dir / "summary.md"
    p3_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(p3_csv_path, payload["records"])
    _write_summary(p3_summary_path, payload)

    if raw_csv_path.exists() and not (args.output_dir / "raw_stageF_second_method_subset.csv").exists():
        shutil.copyfile(raw_csv_path, args.output_dir / "raw_stageF_second_method_subset.csv")

    print(json.dumps({
        "raw_exit_code": raw_exit_code,
        "overall_passed": payload["overall_passed"],
        "P3a_passed": payload["gates"]["P3a"]["passed"],
        "P3b_passed": payload["gates"]["P3b"]["passed"],
        "coverage_passed": payload["gates"]["coverage"]["passed"],
        "num_checks": len(payload["records"]),
        "max_tv_vs_ti": payload["gates"]["P3a"]["max_tv_vs_ti"],
        "max_q_top_abs_diff_vs_ti": payload["gates"]["P3a"]["max_q_top_abs_diff_vs_ti"],
        "max_full_path_bidirectional_gap": payload["gates"]["P3b"]["max_full_path_bidirectional_gap"],
        "json_path": str(p3_json_path),
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
