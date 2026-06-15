#!/usr/bin/env python3
"""Build exp38 P4 acceptance, failure map, and paired-difference table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_LATTICE_SIZES = (3, 4, 5)
EXPECTED_Q_VALUES = (
    0.08,
    0.10,
    0.12,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
    0.21,
    0.22,
    0.23,
)
LATTICE_PAIRS = ((3, 4), (3, 5), (4, 5))
CROSSING_REGION_Q_MIN = 0.15
DEFAULT_BOOTSTRAP_REPS = 10000
DEFAULT_BOOTSTRAP_SEED = 20260605

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
EXP38_DIR = REPO_ROOT / "data/3d_toric_code/with_measurement_noise/exp38"
DEFAULT_P2_ACCEPTANCE = (
    EXP38_DIR
    / "003_p2_production_grid_20260605"
    / "accepted_exp38_p2_ti_grid_20260605_0145"
    / "p2_acceptance.json"
)
DEFAULT_P2_TI_RESULTS = (
    EXP38_DIR
    / "003_p2_production_grid_20260605"
    / "merged_exp38_p2_ti_grid_20260605_0145"
    / "sector_ti_results.npz"
)
DEFAULT_P3_RESULTS = (
    EXP38_DIR
    / "004_p3_second_method_subset_20260605"
    / "remote_collected_exp38_p3_second_method_20260605_0610"
    / "exp38_p3_second_method_20260605_0610"
    / "second_method_subset"
    / "p3_second_method_subset.json"
)


def _round_q(value: Any) -> float:
    return float(round(float(value), 6))


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _bool_gate(value: Any) -> bool:
    return bool(value)


def _p2_gate_passed(p2_payload: dict[str, Any]) -> dict[str, Any]:
    gates = p2_payload.get("gates", {})
    point_rows = list(p2_payload.get("point_rows", []))
    disorder_rows = list(p2_payload.get("disorder_rows", []))
    num_fail_points = sum(row.get("status") == "FAIL" for row in point_rows)
    num_fail_disorder = sum(row.get("status") == "FAIL" for row in disorder_rows)
    p2_gate_names = ("P2a", "P2b", "P2c", "common_disorder")
    gate_passes = {
        name: _bool_gate(gates.get(name, {}).get("passed", False))
        for name in p2_gate_names
    }
    return {
        "passed": bool(all(gate_passes.values()) and num_fail_points == 0 and num_fail_disorder == 0),
        "p2_overall_passed": bool(p2_payload.get("overall_passed", False)),
        "gate_passes": gate_passes,
        "num_fail_points": int(num_fail_points),
        "num_fail_disorder": int(num_fail_disorder),
        "point_status_counts": _status_counts(point_rows),
        "disorder_status_counts": _status_counts(disorder_rows),
        "num_points": int(len(point_rows)),
        "num_disorder_rows": int(len(disorder_rows)),
    }


def _p3_gate_passed(p3_payload: dict[str, Any]) -> dict[str, Any]:
    gates = p3_payload.get("gates", {})
    gate_names = ("P3a", "P3b", "coverage")
    gate_passes = {
        name: _bool_gate(gates.get(name, {}).get("passed", False))
        for name in gate_names
    }
    records = list(p3_payload.get("records", []))
    num_failed_records = sum(not bool(record.get("passed", False)) for record in records)
    return {
        "passed": bool(bool(p3_payload.get("overall_passed", False)) and all(gate_passes.values()) and num_failed_records == 0),
        "p3_overall_passed": bool(p3_payload.get("overall_passed", False)),
        "gate_passes": gate_passes,
        "num_checks": int(len(records)),
        "num_failed_records": int(num_failed_records),
        "max_tv_vs_ti": float(gates.get("P3a", {}).get("max_tv_vs_ti", math.nan)),
        "max_q_top_abs_diff_vs_ti": float(gates.get("P3a", {}).get("max_q_top_abs_diff_vs_ti", math.nan)),
        "max_full_path_bidirectional_gap": float(gates.get("P3b", {}).get("max_full_path_bidirectional_gap", math.nan)),
        "max_bar_residual": float(gates.get("P3b", {}).get("max_bar_residual", math.nan)),
        "coverage_lattice_sizes": list(gates.get("coverage", {}).get("lattice_sizes", [])),
    }


def _load_seed_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int)
        q_values = data["q_values"].astype(float)
        disorder_seeds = data["disorder_seed_per_disorder"].astype(np.int64)
        sample_seeds = data["sample_seed_per_disorder"].astype(np.int64)

    mismatches: list[dict[str, Any]] = []
    for qi, q_value in enumerate(q_values):
        for disorder_index in range(disorder_seeds.shape[2]):
            values = disorder_seeds[:, qi, disorder_index]
            unique_values = sorted({int(value) for value in values.tolist()})
            if len(unique_values) != 1:
                mismatches.append(
                    {
                        "q_value": float(q_value),
                        "disorder_index": int(disorder_index),
                        "seeds_by_lattice_size": {
                            str(int(lattice_sizes[li])): int(values[li])
                            for li in range(len(lattice_sizes))
                        },
                    }
                )
    return {
        "lattice_sizes": [int(value) for value in lattice_sizes.tolist()],
        "q_values": [float(value) for value in q_values.tolist()],
        "num_disorder": int(disorder_seeds.shape[2]),
        "num_common_disorder_seed_mismatches": int(len(mismatches)),
        "common_disorder_seed_mismatch_examples": mismatches[:10],
        "num_sample_seed_collisions": int(sample_seeds.size - np.unique(sample_seeds).size),
    }


def _paired_difference_rows(
    p2_payload: dict[str, Any],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    disorder_rows = list(p2_payload["disorder_rows"])
    by_key = {
        (int(row["lattice_size"]), _round_q(row["q_value"]), int(row["disorder_index"])): row
        for row in disorder_rows
    }
    q_values = sorted({_round_q(row["q_value"]) for row in disorder_rows})
    disorder_indices = sorted({int(row["disorder_index"]) for row in disorder_rows})
    rng = np.random.default_rng(bootstrap_seed)
    output_rows: list[dict[str, Any]] = []

    for lattice_size_a, lattice_size_b in LATTICE_PAIRS:
        for q_value in q_values:
            deltas: list[float] = []
            values_a: list[float] = []
            values_b: list[float] = []
            excluded_a_not_pass = 0
            excluded_b_not_pass = 0
            missing_pairs = 0
            for disorder_index in disorder_indices:
                row_a = by_key.get((lattice_size_a, q_value, disorder_index))
                row_b = by_key.get((lattice_size_b, q_value, disorder_index))
                if row_a is None or row_b is None:
                    missing_pairs += 1
                    continue
                pass_a = row_a.get("status") == "PASS"
                pass_b = row_b.get("status") == "PASS"
                if not pass_a:
                    excluded_a_not_pass += 1
                if not pass_b:
                    excluded_b_not_pass += 1
                if not (pass_a and pass_b):
                    continue
                q_top_a = float(row_a["q_top"])
                q_top_b = float(row_b["q_top"])
                values_a.append(q_top_a)
                values_b.append(q_top_b)
                deltas.append(q_top_a - q_top_b)

            delta_array = np.asarray(deltas, dtype=np.float64)
            n = int(delta_array.size)
            if n == 0:
                delta_mean = math.nan
                paired_sem = math.nan
                ci_low = math.nan
                ci_high = math.nan
                ci_excludes_zero = False
                paired_std = math.nan
            elif n == 1:
                delta_mean = float(delta_array[0])
                paired_sem = math.nan
                ci_low = float(delta_array[0])
                ci_high = float(delta_array[0])
                ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
                paired_std = math.nan
            else:
                delta_mean = float(np.mean(delta_array))
                paired_std = float(np.std(delta_array, ddof=1))
                paired_sem = float(paired_std / math.sqrt(n))
                boot_indices = rng.integers(0, n, size=(bootstrap_reps, n))
                boot_means = delta_array[boot_indices].mean(axis=1)
                ci_low, ci_high = [float(x) for x in np.quantile(boot_means, [0.025, 0.975])]
                ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)

            output_rows.append(
                {
                    "lattice_size_a": int(lattice_size_a),
                    "lattice_size_b": int(lattice_size_b),
                    "q_value": float(q_value),
                    "delta_definition": f"q_top_L{lattice_size_a}_minus_L{lattice_size_b}",
                    "delta_mean": delta_mean,
                    "paired_sem": paired_sem,
                    "paired_std": paired_std,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "ci_excludes_zero": ci_excludes_zero,
                    "effective_paired_disorder_count": n,
                    "num_possible_disorder": int(len(disorder_indices)),
                    "num_missing_pairs": int(missing_pairs),
                    "num_excluded_not_pass_a": int(excluded_a_not_pass),
                    "num_excluded_not_pass_b": int(excluded_b_not_pass),
                    "mean_q_top_a_paired": float(np.mean(values_a)) if values_a else math.nan,
                    "mean_q_top_b_paired": float(np.mean(values_b)) if values_b else math.nan,
                    "crossing_region": bool(q_value >= CROSSING_REGION_Q_MIN),
                    "bootstrap_reps": int(bootstrap_reps if n > 1 else 0),
                    "bootstrap_seed": int(bootstrap_seed),
                    "pair_status_rule": "PASS-only on both lattice sizes for the same disorder_index",
                }
            )
    return output_rows


def _paired_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = len(LATTICE_PAIRS) * len(EXPECTED_Q_VALUES)
    completed_rows = len(rows)
    rows_with_pairs = [row for row in rows if int(row["effective_paired_disorder_count"]) > 0]
    crossing_sig_rows = [
        row
        for row in rows
        if bool(row["crossing_region"]) and bool(row["ci_excludes_zero"])
    ]
    return {
        "passed": bool(completed_rows == expected_rows and len(rows_with_pairs) == expected_rows and len(crossing_sig_rows) > 0),
        "expected_rows": int(expected_rows),
        "completed_rows": int(completed_rows),
        "rows_with_nonzero_pair_count": int(len(rows_with_pairs)),
        "min_effective_paired_disorder_count": int(min(row["effective_paired_disorder_count"] for row in rows)) if rows else 0,
        "num_crossing_region_ci_excludes_zero": int(len(crossing_sig_rows)),
        "q_values_with_crossing_region_ci_excludes_zero": sorted(
            {float(row["q_value"]) for row in crossing_sig_rows}
        ),
        "lattice_pairs_with_crossing_region_ci_excludes_zero": sorted(
            {
                f"{int(row['lattice_size_a'])}-{int(row['lattice_size_b'])}"
                for row in crossing_sig_rows
            }
        ),
        "criterion": (
            "all q/L-pair paired rows recorded with nonzero PASS-only pair counts; "
            "at least one crossing-region paired CI excludes zero"
        ),
    }


def _build_p4_point_rows(
    p2_payload: dict[str, Any],
    p3_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    p3_by_point: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for record in p3_payload.get("records", []):
        key = (int(record["lattice_size"]), _round_q(record["q_value"]))
        p3_by_point.setdefault(key, []).append(record)

    point_rows: list[dict[str, Any]] = []
    for row in p2_payload.get("point_rows", []):
        lattice_size = int(row["lattice_size"])
        q_value = _round_q(row["q_value"])
        p2_status = str(row["status"])
        p3_records = p3_by_point.get((lattice_size, q_value), [])
        p3_failed = any(not bool(record.get("passed", False)) for record in p3_records)
        if p2_status == "FAIL" or p3_failed:
            p4_status = "FAIL"
        elif p2_status == "WARN":
            p4_status = "WARN"
        else:
            p4_status = "PASS"
        p3_text = "not checked"
        if p3_records:
            p3_bits = []
            for record in p3_records:
                p3_bits.append(
                    "d={d}: {status}, TV={tv:.6f}, dq={dq:.6f}, gap={gap:.6f}".format(
                        d=int(record["disorder_index"]),
                        status="PASS" if record.get("passed", False) else "FAIL",
                        tv=float(record.get("tv_vs_ti", math.nan)),
                        dq=float(record.get("q_top_abs_diff_vs_ti", math.nan)),
                        gap=float(record.get("max_full_path_bidirectional_gap", math.nan)),
                    )
                )
            p3_text = "; ".join(p3_bits)
        point_rows.append(
            {
                "lattice_size": lattice_size,
                "q_value": float(q_value),
                "p4_status": p4_status,
                "p2_status": p2_status,
                "mean_q_top": float(row["mean_q_top"]),
                "total_sem_q_top": float(row["total_sem_q_top"]),
                "num_pass_disorder": int(row["num_pass_disorder"]),
                "num_warn_disorder": int(row["num_warn_disorder"]),
                "num_fail_disorder": int(row["num_fail_disorder"]),
                "max_grid_tv": float(row["max_grid_tv"]),
                "max_grid_q_top_abs_diff": float(row["max_grid_q_top_abs_diff"]),
                "p2_flags": str(row["flags"]),
                "p3_cross_check": p3_text,
            }
        )
    return point_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_failure_map(
    path: Path,
    point_rows: list[dict[str, Any]],
    p3_payload: dict[str, Any],
) -> None:
    lines = [
        "# exp38 P4 failure map",
        "",
        "P4 status starts from P2 point/disorder acceptance and adds the P3 second-method subset check.",
        "WARN rows remain context; paired differences are computed separately from PASS-only disorder pairs.",
        "",
        "## Point Map",
        "",
        "| L | q | P4 status | P2 status | mean q_top | total SEM | pass/warn/fail disorder | max grid TV | max grid dq | P2 flags | P3 cross-check |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in point_rows:
        lines.append(
            "| "
            f"{row['lattice_size']} | "
            f"{row['q_value']:.3f} | "
            f"{row['p4_status']} | "
            f"{row['p2_status']} | "
            f"{row['mean_q_top']:.6f} | "
            f"{row['total_sem_q_top']:.6f} | "
            f"{row['num_pass_disorder']}/{row['num_warn_disorder']}/{row['num_fail_disorder']} | "
            f"{row['max_grid_tv']:.6f} | "
            f"{row['max_grid_q_top_abs_diff']:.6f} | "
            f"{row['p2_flags']} | "
            f"{row['p3_cross_check']} |"
        )

    lines.extend(
        [
            "",
            "## P3 Records",
            "",
            "| L | q | d | status | TV vs TI | abs dq_top | full-path gap | BAR residual | seed |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in p3_payload.get("records", []):
        lines.append(
            "| "
            f"{int(record['lattice_size'])} | "
            f"{float(record['q_value']):.3f} | "
            f"{int(record['disorder_index'])} | "
            f"{'PASS' if record.get('passed', False) else 'FAIL'} | "
            f"{float(record.get('tv_vs_ti', math.nan)):.6f} | "
            f"{float(record.get('q_top_abs_diff_vs_ti', math.nan)):.6f} | "
            f"{float(record.get('max_full_path_bidirectional_gap', math.nan)):.6f} | "
            f"{float(record.get('max_bar_residual', math.nan)):.3e} | "
            f"{int(record.get('seed', -1))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, payload: dict[str, Any], paired_rows: list[dict[str, Any]]) -> None:
    gates = payload["gates"]
    p4a = gates["P4a"]
    p4b = gates["P4b"]
    p4c = gates["P4c"]
    point_counts = p4a["point_status_counts"]
    disorder_counts = p4a["disorder_status_counts"]
    significant_rows = [
        row
        for row in paired_rows
        if bool(row["crossing_region"]) and bool(row["ci_excludes_zero"])
    ]
    strongest = sorted(significant_rows, key=lambda row: (row["q_value"], row["lattice_size_a"], row["lattice_size_b"]))
    pair_summaries: list[dict[str, Any]] = []
    for lattice_size_a, lattice_size_b in LATTICE_PAIRS:
        pair_rows = [
            row
            for row in paired_rows
            if int(row["lattice_size_a"]) == lattice_size_a
            and int(row["lattice_size_b"]) == lattice_size_b
        ]
        crossing_rows = [row for row in pair_rows if bool(row["crossing_region"])]
        crossing_sig = [row for row in crossing_rows if bool(row["ci_excludes_zero"])]
        pair_summaries.append(
            {
                "pair": f"L{lattice_size_a}-L{lattice_size_b}",
                "min_pair_count": min(int(row["effective_paired_disorder_count"]) for row in pair_rows),
                "crossing_region_rows": len(crossing_rows),
                "crossing_region_ci_excludes_zero": len(crossing_sig),
                "q_values": sorted({float(row["q_value"]) for row in crossing_sig}),
                "resolution": "resolved at listed q" if crossing_sig else "unresolved by paired CI",
            }
        )

    lines = [
        "# exp38 P4 acceptance and paired differences",
        "",
        f"Status: `{'PASS' if payload['overall_passed'] else 'DOING/FAIL'}`",
        "",
        "## Inputs",
        "",
        f"- P2 acceptance: `{payload['inputs']['p2_acceptance']}`",
        f"- P2 TI NPZ: `{payload['inputs']['p2_ti_results']}`",
        f"- P3 second method: `{payload['inputs']['p3_results']}`",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            "| P4a | P2 acceptance gates passed, common disorder verified, no FAIL point/disorder rows | "
            f"P2 gates={p4a['gate_passes']}, point statuses={point_counts}, disorder statuses={disorder_counts} | "
            f"{'PASS' if p4a['passed'] else 'FAIL'} |"
        ),
        (
            "| P4b | P3 second-method subset gates passed | "
            f"checks={p4b['num_checks']}, max TV={p4b['max_tv_vs_ti']:.6f}, "
            f"max |dq_top|={p4b['max_q_top_abs_diff_vs_ti']:.6f}, "
            f"max full-path gap={p4b['max_full_path_bidirectional_gap']:.6f} | "
            f"{'PASS' if p4b['passed'] else 'FAIL'} |"
        ),
        (
            "| P4c | PASS-only paired differences recorded for every q/L-pair; crossing-region CIs include some nonzero separations | "
            f"rows={p4c['completed_rows']}/{p4c['expected_rows']}, min paired N={p4c['min_effective_paired_disorder_count']}, "
            f"CI excludes zero rows={p4c['num_crossing_region_ci_excludes_zero']} at q={p4c['q_values_with_crossing_region_ci_excludes_zero']} | "
            f"{'PASS' if p4c['passed'] else 'FAIL'} |"
        ),
        "",
        "## Pair Resolution Context",
        "",
        "| L pair | min paired N | crossing-region rows | crossing-region CI excludes zero | q values | resolution |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in pair_summaries:
        lines.append(
            "| "
            f"{row['pair']} | "
            f"{row['min_pair_count']} | "
            f"{row['crossing_region_rows']} | "
            f"{row['crossing_region_ci_excludes_zero']} | "
            f"{row['q_values']} | "
            f"{row['resolution']} |"
        )

    lines.extend(
        [
            "",
        "## Paired Difference Highlights",
        "",
        ]
    )
    if strongest:
        lines.extend(
            [
                "| L pair | q | delta mean | paired SEM | 95% paired bootstrap CI | N paired |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in strongest:
            lines.append(
                "| "
                f"L{int(row['lattice_size_a'])}-L{int(row['lattice_size_b'])} | "
                f"{float(row['q_value']):.3f} | "
                f"{float(row['delta_mean']):.6f} | "
                f"{float(row['paired_sem']):.6f} | "
                f"[{float(row['bootstrap_ci95_low']):.6f}, {float(row['bootstrap_ci95_high']):.6f}] | "
                f"{int(row['effective_paired_disorder_count'])} |"
            )
    else:
        lines.append("No crossing-region paired CI excludes zero; P4c is unresolved.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `p4_acceptance.json`",
            "- `failure_map.md`",
            "- `paired_difference.csv`",
            "- `p4_point_status.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-acceptance", type=Path, default=DEFAULT_P2_ACCEPTANCE)
    parser.add_argument("--p2-ti-results", type=Path, default=DEFAULT_P2_TI_RESULTS)
    parser.add_argument("--p3-results", type=Path, default=DEFAULT_P3_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    p2_payload = _json_load(args.p2_acceptance)
    p3_payload = _json_load(args.p3_results)
    seed_audit = _load_seed_arrays(args.p2_ti_results)
    p4_point_rows = _build_p4_point_rows(p2_payload, p3_payload)
    paired_rows = _paired_difference_rows(
        p2_payload,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )

    p4a = _p2_gate_passed(p2_payload)
    p4b = _p3_gate_passed(p3_payload)
    p4c = _paired_gate(paired_rows)
    seed_gate_passed = seed_audit["num_common_disorder_seed_mismatches"] == 0
    p4a["seed_npz_audit_passed"] = bool(seed_gate_passed)
    p4a["seed_npz_audit"] = seed_audit
    p4a["passed"] = bool(p4a["passed"] and seed_gate_passed)

    payload = {
        "stage": "P4",
        "overall_passed": bool(p4a["passed"] and p4b["passed"] and p4c["passed"]),
        "inputs": {
            "p2_acceptance": str(args.p2_acceptance),
            "p2_ti_results": str(args.p2_ti_results),
            "p3_results": str(args.p3_results),
        },
        "thresholds": {
            "crossing_region_q_min": CROSSING_REGION_Q_MIN,
            "bootstrap_reps": int(args.bootstrap_reps),
            "bootstrap_seed": int(args.bootstrap_seed),
            "paired_rule": "status == PASS on both lattice sizes for the same disorder_index",
        },
        "gates": {
            "P4a": p4a,
            "P4b": p4b,
            "P4c": p4c,
        },
        "p4_point_rows": p4_point_rows,
        "paired_difference_rows": paired_rows,
    }

    (output_dir / "p4_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(output_dir / "p4_point_status.csv", p4_point_rows)
    _write_csv(output_dir / "paired_difference.csv", paired_rows)
    _write_failure_map(output_dir / "failure_map.md", p4_point_rows, p3_payload)
    _write_summary(output_dir / "summary.md", payload, paired_rows)

    print(
        json.dumps(
            {
                "overall_passed": payload["overall_passed"],
                "P4a_passed": p4a["passed"],
                "P4b_passed": p4b["passed"],
                "P4c_passed": p4c["passed"],
                "min_effective_paired_disorder_count": p4c["min_effective_paired_disorder_count"],
                "num_crossing_region_ci_excludes_zero": p4c["num_crossing_region_ci_excludes_zero"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
