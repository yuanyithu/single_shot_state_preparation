#!/usr/bin/env python3
"""Build the exp38 P2 sector-TI failure-map draft."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_LATTICE_SIZES = (3, 4, 5)
EXPECTED_Q_VALUES = (0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23)

GRID_TV_THRESHOLD = 0.02
GRID_QTOP_THRESHOLD = 0.02
TAIL_QTOP_NEAR_ONE = 0.98
TAIL_RESOLUTION_SIGMA = 2.0
TAIL_ABSOLUTE_FLOOR = 1.0e-8


def _read_scalar_text(array: np.ndarray) -> str:
    value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _format_q(value: float) -> float:
    return float(round(float(value), 6))


def _flag_summary(flags: list[str]) -> str:
    counts: dict[str, int] = {}
    for flag in flags:
        counts[flag] = counts.get(flag, 0) + 1
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _load_ti_grid(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(_read_scalar_text(data["manifest_json"]))
        arrays = {
            "lattice_sizes": data["lattice_size_list"].astype(int),
            "q_values": data["q_values"].astype(float),
            "q_top": data["q_top_per_disorder"].astype(float),
            "q_top_stderr": data["q_top_stderr_per_disorder"].astype(float),
            "q_top_ci95": data["q_top_ci95_per_disorder"].astype(float),
            "grid_tv": data["grid_tv_per_disorder"].astype(float),
            "grid_q_top_abs_diff": data["grid_q_top_abs_diff_per_disorder"].astype(float),
            "weights": data["weights_per_disorder"].astype(float),
            "weights_stderr": data["weights_stderr_per_disorder"].astype(float),
            "delta_f": data["delta_f_per_disorder"].astype(float),
            "delta_f_stderr": data["delta_f_stderr_per_disorder"].astype(float),
            "flags": data["flags_per_disorder"].astype("<U128"),
            "mean_q_top": data["mean_q_top"].astype(float),
            "total_sem_q_top": data["total_sem_q_top"].astype(float),
            "disorder_sem_q_top": data["disorder_sem_q_top"].astype(float),
            "mcmc_sem_q_top": data["mcmc_sem_q_top"].astype(float),
            "pass_fraction": data["pass_fraction"].astype(float),
        }
        for name in ("seed_per_disorder", "disorder_seed_per_disorder", "sample_seed_per_disorder"):
            if name in data.files:
                arrays[name] = data[name].astype(np.int64)
    return manifest, arrays


def _coverage_gate(lattice_sizes: np.ndarray, q_values: np.ndarray) -> dict:
    present_l = {int(value) for value in lattice_sizes.tolist()}
    present_q = {_format_q(value) for value in q_values.tolist()}
    expected_l = set(EXPECTED_LATTICE_SIZES)
    expected_q = {_format_q(value) for value in EXPECTED_Q_VALUES}
    return {
        "passed": present_l == expected_l and present_q == expected_q,
        "missing_lattice_sizes": sorted(expected_l - present_l),
        "missing_q_values": sorted(expected_q - present_q),
        "extra_lattice_sizes": sorted(present_l - expected_l),
        "extra_q_values": sorted(present_q - expected_q),
    }


def _common_seed_gate(arrays: dict[str, np.ndarray]) -> dict:
    seeds = arrays.get("disorder_seed_per_disorder")
    if seeds is None:
        return {
            "passed": False,
            "reason": "disorder_seed_per_disorder missing",
            "num_mismatches": None,
            "examples": [],
        }
    mismatches = []
    q_values = arrays["q_values"]
    lattice_sizes = arrays["lattice_sizes"]
    for qi, q_value in enumerate(q_values):
        for disorder_index in range(seeds.shape[2]):
            values = seeds[:, qi, disorder_index]
            if np.any(values < 0) or len(set(int(value) for value in values.tolist())) != 1:
                mismatches.append({
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "seeds_by_lattice_size": {
                        str(int(lattice_sizes[li])): int(values[li])
                        for li in range(len(lattice_sizes))
                    },
                })
    return {
        "passed": len(mismatches) == 0,
        "reason": "same disorder_seed across L for every (q, disorder_index)",
        "num_mismatches": int(len(mismatches)),
        "examples": mismatches[:10],
    }


def _tail_record(weights: np.ndarray, weights_stderr: np.ndarray, q_top: float, q_top_ci95: np.ndarray) -> dict:
    if not np.isfinite(q_top) or not np.all(np.isfinite(weights)):
        return {
            "dominant_sector": -1,
            "max_subdominant_weight": math.nan,
            "max_subdominant_weight_stderr": math.nan,
            "max_subdominant_snr": math.nan,
            "all_subdominant_below_resolution": False,
            "unresolved_tail": False,
            "q_top_lower_bound": math.nan,
        }
    dominant = int(np.nanargmax(weights))
    mask = np.ones(weights.shape[0], dtype=bool)
    mask[dominant] = False
    sub_weights = np.asarray(weights[mask], dtype=np.float64)
    sub_stderr = np.asarray(weights_stderr[mask], dtype=np.float64)
    denominator = np.maximum(np.nan_to_num(sub_stderr, nan=np.inf), TAIL_ABSOLUTE_FLOOR)
    resolution_floor = np.maximum(TAIL_ABSOLUTE_FLOOR, TAIL_RESOLUTION_SIGMA * denominator)
    all_sub_below_resolution = bool(np.all(sub_weights <= resolution_floor))
    return {
        "dominant_sector": dominant,
        "max_subdominant_weight": float(np.nanmax(sub_weights)),
        "max_subdominant_weight_stderr": float(np.nanmax(sub_stderr)),
        "max_subdominant_snr": float(np.nanmax(sub_weights / denominator)),
        "all_subdominant_below_resolution": all_sub_below_resolution,
        "unresolved_tail": bool(float(q_top) >= TAIL_QTOP_NEAR_ONE and all_sub_below_resolution),
        "q_top_lower_bound": float(np.asarray(q_top_ci95, dtype=np.float64)[0]),
    }


def _build_rows(arrays: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    lattice_sizes = arrays["lattice_sizes"]
    q_values = arrays["q_values"]
    q_top = arrays["q_top"]
    disorder_rows = []
    point_rows = []

    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            statuses = []
            point_flags = []
            for disorder_index in range(q_top.shape[2]):
                raw_flag = str(arrays["flags"][li, qi, disorder_index])
                row_flags = []
                missing = (
                    raw_flag == "MISSING"
                    or not np.isfinite(arrays["q_top"][li, qi, disorder_index])
                    or not np.all(np.isfinite(arrays["weights"][li, qi, disorder_index]))
                    or not np.all(np.isfinite(arrays["delta_f"][li, qi, disorder_index][1:]))
                )
                if missing:
                    row_flags.append("MISSING_FAIL")
                if arrays["grid_tv"][li, qi, disorder_index] > GRID_TV_THRESHOLD:
                    row_flags.append("TI_GRID_TV_WARN")
                if arrays["grid_q_top_abs_diff"][li, qi, disorder_index] > GRID_QTOP_THRESHOLD:
                    row_flags.append("TI_GRID_QTOP_WARN")
                tail = _tail_record(
                    weights=arrays["weights"][li, qi, disorder_index],
                    weights_stderr=arrays["weights_stderr"][li, qi, disorder_index],
                    q_top=float(arrays["q_top"][li, qi, disorder_index]),
                    q_top_ci95=arrays["q_top_ci95"][li, qi, disorder_index],
                )
                if tail["unresolved_tail"]:
                    row_flags.append("UNRESOLVED_TAIL_FAIL")

                if any(flag.endswith("_FAIL") for flag in row_flags):
                    status = "FAIL"
                elif row_flags:
                    status = "WARN"
                else:
                    status = "PASS"
                statuses.append(status)
                point_flags.extend(row_flags)
                disorder_rows.append({
                    "lattice_size": int(lattice_size),
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "q_top": float(arrays["q_top"][li, qi, disorder_index]),
                    "q_top_ci_low": float(arrays["q_top_ci95"][li, qi, disorder_index, 0]),
                    "q_top_ci_high": float(arrays["q_top_ci95"][li, qi, disorder_index, 1]),
                    "grid_tv": float(arrays["grid_tv"][li, qi, disorder_index]),
                    "grid_q_top_abs_diff": float(arrays["grid_q_top_abs_diff"][li, qi, disorder_index]),
                    "max_subdominant_weight": tail["max_subdominant_weight"],
                    "max_subdominant_weight_stderr": tail["max_subdominant_weight_stderr"],
                    "max_subdominant_snr": tail["max_subdominant_snr"],
                    "q_top_lower_bound": tail["q_top_lower_bound"],
                    "raw_flags": raw_flag,
                    "p2_flags": ";".join(row_flags) if row_flags else "PASS",
                    "status": status,
                })
            if any(status == "FAIL" for status in statuses):
                point_status = "FAIL"
            elif any(status == "WARN" for status in statuses):
                point_status = "WARN"
            else:
                point_status = "PASS"
            point_rows.append({
                "lattice_size": int(lattice_size),
                "q_value": float(q_value),
                "mean_q_top": float(arrays["mean_q_top"][li, qi]),
                "total_sem_q_top": float(arrays["total_sem_q_top"][li, qi]),
                "disorder_sem_q_top": float(arrays["disorder_sem_q_top"][li, qi]),
                "mcmc_sem_q_top": float(arrays["mcmc_sem_q_top"][li, qi]),
                "num_disorder": int(q_top.shape[2]),
                "num_pass_disorder": int(sum(status == "PASS" for status in statuses)),
                "num_warn_disorder": int(sum(status == "WARN" for status in statuses)),
                "num_fail_disorder": int(sum(status == "FAIL" for status in statuses)),
                "max_grid_tv": float(np.nanmax(arrays["grid_tv"][li, qi])),
                "max_grid_q_top_abs_diff": float(np.nanmax(arrays["grid_q_top_abs_diff"][li, qi])),
                "flags": _flag_summary(point_flags) if point_flags else "PASS",
                "status": point_status,
            })
    return disorder_rows, point_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_failure_map(path: Path, point_rows: list[dict]) -> None:
    lines = [
        "# exp38 P2 failure map draft",
        "",
        "| L | q | status | mean q_top | total SEM | pass/warn/fail disorder | max grid TV | max grid dq | flags |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in point_rows:
        lines.append(
            "| "
            f"{row['lattice_size']} | "
            f"{row['q_value']:.3f} | "
            f"{row['status']} | "
            f"{row['mean_q_top']:.6f} | "
            f"{row['total_sem_q_top']:.6f} | "
            f"{row['num_pass_disorder']}/{row['num_warn_disorder']}/{row['num_fail_disorder']} | "
            f"{row['max_grid_tv']:.6f} | "
            f"{row['max_grid_q_top_abs_diff']:.6f} | "
            f"{row['flags']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, payload: dict, point_rows: list[dict]) -> None:
    status_counts: dict[str, int] = {}
    for row in point_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    status_text = ", ".join(f"{key}:{status_counts[key]}" for key in sorted(status_counts))
    gates = payload["gates"]
    lines = [
        "# exp38 P2 sector-TI acceptance draft",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'DOING/FAIL'}",
        "",
        f"Source TI result: `{payload['ti_results_path']}`",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            "| P2a | full L x q x disorder coverage with weights, delta_f, q_top, stderr and explicit flags | "
            f"coverage={gates['P2a']['coverage_passed']}, missing={gates['P2a']['num_missing_disorder']}, "
            f"statuses={status_text or 'none'} | "
            f"{'PASS' if gates['P2a']['passed'] else 'FAIL'} |"
        ),
        (
            "| P2b | unresolved high-q_top tails are marked FAIL, never PASS | "
            f"unresolved_tail_fail={gates['P2b']['num_unresolved_tail_fail']}, "
            f"pass_violations={gates['P2b']['num_unresolved_tail_pass_violations']} | "
            f"{'PASS' if gates['P2b']['passed'] else 'FAIL'} |"
        ),
        (
            "| P2c | every PASS disorder has grid TV and |dq_top| <= 0.02 | "
            f"PASS-disorder grid failures={gates['P2c']['num_pass_disorder_grid_failures']} | "
            f"{'PASS' if gates['P2c']['passed'] else 'FAIL'} |"
        ),
        (
            "| Common disorder | disorder_seed_per_disorder identical across L for every (q, disorder) | "
            f"mismatches={gates['common_disorder']['num_mismatches']} | "
            f"{'PASS' if gates['common_disorder']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Artifacts",
        "",
        "- `p2_acceptance.json`",
        "- `p2_point_status.csv`",
        "- `p2_disorder_status.csv`",
        "- `failure_map.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arrays = _load_ti_grid(Path(args.ti_results))
    disorder_rows, point_rows = _build_rows(arrays)
    coverage = _coverage_gate(arrays["lattice_sizes"], arrays["q_values"])
    seed_gate = _common_seed_gate(arrays)

    num_missing = sum("MISSING_FAIL" in row["p2_flags"] for row in disorder_rows)
    num_unresolved = sum("UNRESOLVED_TAIL_FAIL" in row["p2_flags"] for row in disorder_rows)
    num_unresolved_pass = sum(
        "UNRESOLVED_TAIL_FAIL" in row["p2_flags"] and row["status"] == "PASS"
        for row in disorder_rows
    )
    num_pass_grid_failures = sum(
        row["status"] == "PASS"
        and (
            row["grid_tv"] > GRID_TV_THRESHOLD
            or row["grid_q_top_abs_diff"] > GRID_QTOP_THRESHOLD
        )
        for row in disorder_rows
    )

    p2a_passed = bool(coverage["passed"] and num_missing == 0)
    p2b_passed = bool(num_unresolved_pass == 0)
    p2c_passed = bool(num_pass_grid_failures == 0)
    payload = {
        "stage": "P2",
        "overall_passed": bool(p2a_passed and p2b_passed and p2c_passed and seed_gate["passed"]),
        "ti_results_path": str(Path(args.ti_results)),
        "manifest": manifest,
        "coverage": coverage,
        "thresholds": {
            "grid_tv": GRID_TV_THRESHOLD,
            "grid_q_top_abs_diff": GRID_QTOP_THRESHOLD,
            "tail_q_top_near_one": TAIL_QTOP_NEAR_ONE,
            "tail_resolution_sigma": TAIL_RESOLUTION_SIGMA,
            "tail_absolute_floor": TAIL_ABSOLUTE_FLOOR,
        },
        "gates": {
            "P2a": {
                "passed": p2a_passed,
                "coverage_passed": bool(coverage["passed"]),
                "num_missing_disorder": int(num_missing),
            },
            "P2b": {
                "passed": p2b_passed,
                "num_unresolved_tail_fail": int(num_unresolved),
                "num_unresolved_tail_pass_violations": int(num_unresolved_pass),
            },
            "P2c": {
                "passed": p2c_passed,
                "num_pass_disorder_grid_failures": int(num_pass_grid_failures),
            },
            "common_disorder": seed_gate,
        },
        "point_rows": point_rows,
        "disorder_rows": disorder_rows,
    }

    (output_dir / "p2_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "p2_point_status.csv", point_rows)
    _write_csv(output_dir / "p2_disorder_status.csv", disorder_rows)
    _write_failure_map(output_dir / "failure_map.md", point_rows)
    _write_summary(output_dir / "summary.md", payload, point_rows)

    print(json.dumps({
        "overall_passed": payload["overall_passed"],
        "P2a_passed": p2a_passed,
        "P2b_passed": p2b_passed,
        "P2c_passed": p2c_passed,
        "common_disorder_passed": seed_gate["passed"],
        "num_points": len(point_rows),
        "num_disorder_rows": len(disorder_rows),
        "num_unresolved_tail_fail": num_unresolved,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
