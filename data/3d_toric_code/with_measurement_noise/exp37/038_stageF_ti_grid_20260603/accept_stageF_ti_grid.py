#!/usr/bin/env python3
"""Stage F acceptance and failure-map builder for sector-TI production grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_LATTICE_SIZES = (3, 4, 5)
EXPECTED_Q_VALUES = tuple(round(0.08 + 0.01 * index, 2) for index in range(16))

GRID_TV_THRESHOLD = 0.02
GRID_QTOP_THRESHOLD = 0.02
SECOND_METHOD_TV_THRESHOLD = 0.03
SECOND_METHOD_QTOP_THRESHOLD = 0.02
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


def _flag_summary(flags: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for value in flags.ravel():
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _load_second_method(path: Path | None) -> dict[tuple[int, float, int], dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    mapping = {}
    for record in records:
        lattice_size = int(record["lattice_size"])
        q_value = _format_q(record["q_value"])
        disorder_index = int(record.get("disorder_index", 0))
        tv = float(
            record.get(
                "tv_vs_ti",
                record.get("tv_vs_stage_d_ti", record.get("tv", math.nan)),
            )
        )
        dq = float(
            record.get(
                "q_top_abs_diff_vs_ti",
                record.get(
                    "q_top_abs_diff_vs_stage_d_ti",
                    record.get("q_top_abs_diff", math.nan),
                ),
            )
        )
        mapping[(lattice_size, q_value, disorder_index)] = {
            "tv_vs_ti": tv,
            "q_top_abs_diff_vs_ti": dq,
            "passed": bool(
                tv <= SECOND_METHOD_TV_THRESHOLD
                and dq <= SECOND_METHOD_QTOP_THRESHOLD
            ),
        }
    return mapping


def _tail_resolution_record(
    weights: np.ndarray,
    weights_stderr: np.ndarray,
    q_top: float,
    q_top_ci95: np.ndarray,
) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    weights_stderr = np.asarray(weights_stderr, dtype=np.float64)
    dominant = int(np.nanargmax(weights))
    mask = np.ones(weights.shape[0], dtype=bool)
    mask[dominant] = False
    sub_weights = weights[mask]
    sub_stderr = weights_stderr[mask]
    resolution_floor = np.maximum(
        TAIL_ABSOLUTE_FLOOR,
        TAIL_RESOLUTION_SIGMA * np.nan_to_num(sub_stderr, nan=np.inf),
    )
    max_sub_weight = float(np.nanmax(sub_weights))
    max_sub_stderr = float(np.nanmax(sub_stderr))
    max_sub_snr = float(
        np.nanmax(
            sub_weights
            / np.maximum(np.nan_to_num(sub_stderr, nan=np.inf), TAIL_ABSOLUTE_FLOOR)
        )
    )
    all_sub_below_resolution = bool(np.all(sub_weights <= resolution_floor))
    unresolved = bool(
        float(q_top) >= TAIL_QTOP_NEAR_ONE
        and all_sub_below_resolution
    )
    return {
        "dominant_sector": dominant,
        "max_subdominant_weight": max_sub_weight,
        "max_subdominant_weight_stderr": max_sub_stderr,
        "max_subdominant_snr": max_sub_snr,
        "all_subdominant_below_resolution": all_sub_below_resolution,
        "unresolved_tail": unresolved,
        "q_top_lower_bound": float(np.asarray(q_top_ci95, dtype=np.float64)[0]),
    }


def _load_ti_grid(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(_read_scalar_text(data["manifest_json"]))
        arrays = {
            "lattice_sizes": data["lattice_size_list"].astype(int),
            "q_values": data["q_values"].astype(float),
            "q_top": data["q_top_per_disorder"].astype(float),
            "q_top_stderr": data["q_top_stderr_per_disorder"].astype(float),
            "q_top_ci95": data["q_top_ci95_per_disorder"].astype(float),
            "grid_tv": data["grid_tv_per_disorder"].astype(float),
            "grid_q_top_abs_diff": data[
                "grid_q_top_abs_diff_per_disorder"
            ].astype(float),
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
    return {"manifest": manifest, "arrays": arrays}


def _coverage_gate(lattice_sizes: np.ndarray, q_values: np.ndarray) -> dict:
    present_l = {int(value) for value in lattice_sizes.tolist()}
    present_q = {_format_q(value) for value in q_values.tolist()}
    expected_l = set(EXPECTED_LATTICE_SIZES)
    expected_q = {_format_q(value) for value in EXPECTED_Q_VALUES}
    missing_l = sorted(expected_l - present_l)
    missing_q = sorted(expected_q - present_q)
    extra_l = sorted(present_l - expected_l)
    extra_q = sorted(present_q - expected_q)
    return {
        "passed": not missing_l and not missing_q and not extra_l and not extra_q,
        "missing_lattice_sizes": missing_l,
        "missing_q_values": missing_q,
        "extra_lattice_sizes": extra_l,
        "extra_q_values": extra_q,
    }


def _build_rows(
    arrays: dict[str, np.ndarray],
    second_method: dict[tuple[int, float, int], dict],
) -> tuple[list[dict], list[dict]]:
    lattice_sizes = arrays["lattice_sizes"]
    q_values = arrays["q_values"]
    q_top = arrays["q_top"]
    q_top_ci95 = arrays["q_top_ci95"]
    weights = arrays["weights"]
    weights_stderr = arrays["weights_stderr"]
    grid_tv = arrays["grid_tv"]
    grid_q_top_abs_diff = arrays["grid_q_top_abs_diff"]
    raw_flags = arrays["flags"]

    disorder_rows = []
    point_rows = []
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            point_statuses = []
            point_reasons = []
            point_second_method_checks = []
            for disorder_index in range(q_top.shape[2]):
                raw_flag = str(raw_flags[li, qi, disorder_index])
                row_flags = []
                if raw_flag == "MISSING" or not np.isfinite(q_top[li, qi, disorder_index]):
                    row_flags.append("MISSING_FAIL")
                if grid_tv[li, qi, disorder_index] > GRID_TV_THRESHOLD:
                    row_flags.append("TI_GRID_TV_WARN")
                if grid_q_top_abs_diff[li, qi, disorder_index] > GRID_QTOP_THRESHOLD:
                    row_flags.append("TI_GRID_QTOP_WARN")
                tail = _tail_resolution_record(
                    weights=weights[li, qi, disorder_index],
                    weights_stderr=weights_stderr[li, qi, disorder_index],
                    q_top=float(q_top[li, qi, disorder_index]),
                    q_top_ci95=q_top_ci95[li, qi, disorder_index],
                )
                if tail["unresolved_tail"]:
                    row_flags.append("UNRESOLVED_TAIL_FAIL")

                second_key = (
                    int(lattice_size),
                    _format_q(q_value),
                    int(disorder_index),
                )
                second = second_method.get(second_key)
                second_status = "NOT_SAMPLED"
                second_tv = math.nan
                second_dq = math.nan
                if second is not None:
                    second_tv = second["tv_vs_ti"]
                    second_dq = second["q_top_abs_diff_vs_ti"]
                    second_status = "PASS" if second["passed"] else "FAIL"
                    point_second_method_checks.append(second)
                    if not second["passed"]:
                        row_flags.append("SECOND_METHOD_FAIL")

                if any(flag.endswith("_FAIL") for flag in row_flags):
                    status = "FAIL"
                elif row_flags:
                    status = "WARN"
                else:
                    status = "PASS"
                point_statuses.append(status)
                if row_flags:
                    point_reasons.extend(row_flags)
                disorder_rows.append({
                    "lattice_size": int(lattice_size),
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "q_top": float(q_top[li, qi, disorder_index]),
                    "q_top_ci_low": float(q_top_ci95[li, qi, disorder_index, 0]),
                    "q_top_ci_high": float(q_top_ci95[li, qi, disorder_index, 1]),
                    "grid_tv": float(grid_tv[li, qi, disorder_index]),
                    "grid_q_top_abs_diff": float(
                        grid_q_top_abs_diff[li, qi, disorder_index]
                    ),
                    "max_subdominant_weight": tail["max_subdominant_weight"],
                    "max_subdominant_weight_stderr": (
                        tail["max_subdominant_weight_stderr"]
                    ),
                    "max_subdominant_snr": tail["max_subdominant_snr"],
                    "q_top_lower_bound": tail["q_top_lower_bound"],
                    "second_method_status": second_status,
                    "second_method_tv": second_tv,
                    "second_method_q_top_abs_diff": second_dq,
                    "raw_flags": raw_flag,
                    "stageF_flags": ";".join(row_flags) if row_flags else "PASS",
                    "status": status,
                })

            if any(status == "FAIL" for status in point_statuses):
                point_status = "FAIL"
            elif any(status == "WARN" for status in point_statuses):
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
                "num_pass_disorder": int(sum(status == "PASS" for status in point_statuses)),
                "num_warn_disorder": int(sum(status == "WARN" for status in point_statuses)),
                "num_fail_disorder": int(sum(status == "FAIL" for status in point_statuses)),
                "num_second_method_checks": int(len(point_second_method_checks)),
                "max_grid_tv": float(np.nanmax(grid_tv[li, qi])),
                "max_grid_q_top_abs_diff": float(
                    np.nanmax(grid_q_top_abs_diff[li, qi])
                ),
                "flags": _flag_summary(np.asarray(point_reasons, dtype="<U128"))
                if point_reasons else "PASS",
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
        "# Stage F failure map",
        "",
        "| L | q | status | mean q_top | total SEM | pass/warn/fail disorder | max grid TV | max grid dq | second checks | flags |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in point_rows:
        lines.append(
            "| "
            f"{row['lattice_size']} | "
            f"{row['q_value']:.3f} | "
            f"{row['status']} | "
            f"{row['mean_q_top']:.6f} | "
            f"{row['total_sem_q_top']:.6f} | "
            f"{row['num_pass_disorder']}/"
            f"{row['num_warn_disorder']}/"
            f"{row['num_fail_disorder']} | "
            f"{row['max_grid_tv']:.6f} | "
            f"{row['max_grid_q_top_abs_diff']:.6f} | "
            f"{row['num_second_method_checks']} | "
            f"{row['flags']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, payload: dict, point_rows: list[dict]) -> None:
    gates = payload["gates"]
    status_counts: dict[str, int] = {}
    for row in point_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    status_text = ", ".join(
        f"{key}:{status_counts[key]}" for key in sorted(status_counts)
    )
    lines = [
        "# Stage F sector-TI acceptance summary",
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
            "| F1 | full grid present and every point has explicit PASS/WARN/FAIL; "
            "unresolved high-q_top tails are FAIL | "
            f"coverage={gates['F1']['coverage_passed']}, "
            f"unresolved={gates['F1']['num_unresolved_tail_fail']}, "
            f"statuses={status_text or 'none'} | "
            f"{'PASS' if gates['F1']['passed'] else 'FAIL'} |"
        ),
        (
            "| F2 | every PASS disorder satisfies coarse/fine grid TV and dq <= 0.02 | "
            f"PASS-disorder grid failures={gates['F2']['num_pass_disorder_grid_failures']} | "
            f"{'PASS' if gates['F2']['passed'] else 'FAIL'} |"
        ),
        (
            "| F3 | sampled subset second method agrees with TI | "
            f"checks={gates['F3']['num_second_method_checks']}, "
            f"failures={gates['F3']['num_second_method_failures']} | "
            f"{'PASS' if gates['F3']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Artifacts",
        "",
        "- `stageF_acceptance.json`",
        "- `stageF_point_status.csv`",
        "- `stageF_disorder_status.csv`",
        "- `failure_map.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--second-method-results", type=Path)
    parser.add_argument("--min-second-method-points", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ti_payload = _load_ti_grid(Path(args.ti_results))
    manifest = ti_payload["manifest"]
    arrays = ti_payload["arrays"]
    second_method = _load_second_method(args.second_method_results)
    disorder_rows, point_rows = _build_rows(arrays, second_method)
    coverage = _coverage_gate(arrays["lattice_sizes"], arrays["q_values"])

    num_unresolved = sum(
        "UNRESOLVED_TAIL_FAIL" in row["stageF_flags"]
        for row in disorder_rows
    )
    num_missing = sum(row["status"] == "FAIL" and "MISSING_FAIL" in row["stageF_flags"] for row in disorder_rows)
    num_pass_grid_failures = sum(
        row["status"] == "PASS"
        and (
            row["grid_tv"] > GRID_TV_THRESHOLD
            or row["grid_q_top_abs_diff"] > GRID_QTOP_THRESHOLD
        )
        for row in disorder_rows
    )
    second_checks = [
        row for row in disorder_rows
        if row["second_method_status"] != "NOT_SAMPLED"
    ]
    num_second_failures = sum(
        row["second_method_status"] == "FAIL"
        for row in second_checks
    )
    f1_passed = bool(coverage["passed"] and num_missing == 0)
    f2_passed = bool(num_pass_grid_failures == 0)
    f3_passed = bool(
        len(second_checks) >= int(args.min_second_method_points)
        and num_second_failures == 0
    )
    payload = {
        "stage": "F",
        "overall_passed": bool(f1_passed and f2_passed and f3_passed),
        "ti_results_path": str(Path(args.ti_results)),
        "second_method_results_path": (
            None if args.second_method_results is None
            else str(args.second_method_results)
        ),
        "manifest": manifest,
        "thresholds": {
            "grid_tv": GRID_TV_THRESHOLD,
            "grid_q_top_abs_diff": GRID_QTOP_THRESHOLD,
            "second_method_tv": SECOND_METHOD_TV_THRESHOLD,
            "second_method_q_top_abs_diff": SECOND_METHOD_QTOP_THRESHOLD,
            "tail_q_top_near_one": TAIL_QTOP_NEAR_ONE,
            "tail_resolution_sigma": TAIL_RESOLUTION_SIGMA,
            "tail_absolute_floor": TAIL_ABSOLUTE_FLOOR,
            "min_second_method_points": int(args.min_second_method_points),
        },
        "coverage": coverage,
        "gates": {
            "F1": {
                "passed": f1_passed,
                "coverage_passed": bool(coverage["passed"]),
                "num_missing_disorder": int(num_missing),
                "num_unresolved_tail_fail": int(num_unresolved),
            },
            "F2": {
                "passed": f2_passed,
                "num_pass_disorder_grid_failures": int(num_pass_grid_failures),
            },
            "F3": {
                "passed": f3_passed,
                "num_second_method_checks": int(len(second_checks)),
                "num_second_method_failures": int(num_second_failures),
            },
        },
        "point_rows": point_rows,
        "disorder_rows": disorder_rows,
    }

    (output_dir / "stageF_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "stageF_point_status.csv", point_rows)
    _write_csv(output_dir / "stageF_disorder_status.csv", disorder_rows)
    _write_failure_map(output_dir / "failure_map.md", point_rows)
    _write_summary(output_dir / "summary.md", payload, point_rows)

    print(json.dumps({
        "overall_passed": payload["overall_passed"],
        "F1_passed": f1_passed,
        "F2_passed": f2_passed,
        "F3_passed": f3_passed,
        "coverage_passed": coverage["passed"],
        "num_points": len(point_rows),
        "num_disorder_rows": len(disorder_rows),
        "num_unresolved_tail_fail": num_unresolved,
        "num_second_method_checks": len(second_checks),
        "num_second_method_failures": num_second_failures,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
