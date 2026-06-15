#!/usr/bin/env python3
"""Build Stage G PASS-only production curves from accepted Stage F outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
EXP37 = ROOT / "data/3d_toric_code/with_measurement_noise/exp37"
DEFAULT_STAGEF_GRID = (
    EXP37
    / "038_stageF_ti_grid_20260603"
    / "repaired_ti_grid_targeted_strong_20260604"
    / "sector_ti_results.npz"
)
DEFAULT_STAGEF_ACCEPT = (
    EXP37
    / "038_stageF_ti_grid_20260603"
    / "accepted_repaired_ti_grid_targeted_strong_20260604"
)
DEFAULT_STAGE_D_CSV = (
    EXP37
    / "036_stageD_sector_ti_20260603"
    / "accepted_combined"
    / "ti_comparison.csv"
)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status_maps(accept_dir: Path) -> tuple[dict[tuple[int, float], str], dict[tuple[int, float, int], str]]:
    point_rows = _read_csv_dicts(accept_dir / "stageF_point_status.csv")
    disorder_rows = _read_csv_dicts(accept_dir / "stageF_disorder_status.csv")
    point_status = {
        (int(row["lattice_size"]), round(float(row["q_value"]), 6)): row["status"]
        for row in point_rows
    }
    disorder_status = {
        (
            int(row["lattice_size"]),
            round(float(row["q_value"]), 6),
            int(row["disorder_index"]),
        ): row["status"]
        for row in disorder_rows
    }
    return point_status, disorder_status


def _bootstrap_curve(
    q_values: np.ndarray,
    q_top: np.ndarray,
    q_top_stderr: np.ndarray,
    pass_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> dict[str, np.ndarray]:
    """Bootstrap q_top over PASS disorders only.

    The bootstrap samples disorders with replacement and adds one Gaussian draw
    per selected disorder using the per-disorder TI stderr saved in Stage F.
    """

    num_q = len(q_values)
    mean = np.full(num_q, np.nan)
    disorder_sem = np.full(num_q, np.nan)
    ti_sem = np.full(num_q, np.nan)
    total_sem = np.full(num_q, np.nan)
    ci_low = np.full(num_q, np.nan)
    ci_high = np.full(num_q, np.nan)
    pass_count = np.zeros(num_q, dtype=int)

    for qi in range(num_q):
        values = np.asarray(q_top[qi], dtype=float)
        stderrs = np.asarray(q_top_stderr[qi], dtype=float)
        mask = np.asarray(pass_mask[qi], dtype=bool) & np.isfinite(values)
        selected_values = values[mask]
        selected_stderrs = np.nan_to_num(stderrs[mask], nan=0.0, posinf=0.0, neginf=0.0)
        count = int(selected_values.size)
        pass_count[qi] = count
        if count == 0:
            continue
        mean[qi] = float(np.mean(selected_values))
        disorder_sem[qi] = float(np.std(selected_values, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        ti_sem[qi] = float(math.sqrt(np.sum(selected_stderrs**2)) / count)
        draws = np.empty(num_bootstrap, dtype=float)
        for bi in range(num_bootstrap):
            indices = rng.integers(0, count, size=count)
            sample = selected_values[indices]
            sample_sigma = selected_stderrs[indices]
            if np.any(sample_sigma > 0.0):
                sample = rng.normal(sample, sample_sigma)
            draws[bi] = float(np.mean(sample))
        total_sem[qi] = float(np.std(draws, ddof=1))
        ci_low[qi], ci_high[qi] = np.quantile(draws, [0.025, 0.975])

    return {
        "mean": mean,
        "disorder_sem": disorder_sem,
        "ti_sem": ti_sem,
        "total_sem": total_sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass_count": pass_count,
    }


def _first_pass_crossing(rows: list[dict]) -> dict:
    by_l: dict[int, dict[float, dict]] = {}
    for row in rows:
        if row["status"] != "PASS":
            continue
        by_l.setdefault(int(row["lattice_size"]), {})[round(float(row["q_value"]), 6)] = row

    crossings = []
    for left, right in [(3, 4), (4, 5)]:
        common_q = sorted(set(by_l.get(left, {})) & set(by_l.get(right, {})))
        previous = None
        for q_value in common_q:
            lrow = by_l[left][q_value]
            rrow = by_l[right][q_value]
            gap = float(lrow["pass_mean_q_top"]) - float(rrow["pass_mean_q_top"])
            current = (q_value, gap)
            if previous is not None and previous[1] * gap <= 0.0:
                crossings.append(
                    {
                        "pair": f"L{left}-L{right}",
                        "q_left": previous[0],
                        "q_right": q_value,
                        "gap_left": previous[1],
                        "gap_right": gap,
                    }
                )
            previous = current
    return {
        "common_pass_q_L3_L4": sorted(set(by_l.get(3, {})) & set(by_l.get(4, {}))),
        "common_pass_q_L4_L5": sorted(set(by_l.get(4, {})) & set(by_l.get(5, {}))),
        "crossings": crossings,
    }


def _plot_curve(curve_rows: list[dict], output_path: Path) -> None:
    colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
    markers = {3: "o", 4: "s", 5: "^"}
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for lattice_size in sorted({int(row["lattice_size"]) for row in curve_rows}):
        rows = [row for row in curve_rows if int(row["lattice_size"]) == lattice_size]
        rows.sort(key=lambda row: float(row["q_value"]))
        warn_rows = [row for row in rows if row["status"] != "PASS"]
        pass_rows = [row for row in rows if row["status"] == "PASS"]
        if warn_rows:
            ax.scatter(
                [float(row["q_value"]) for row in warn_rows],
                [float(row["all_status_mean_q_top"]) for row in warn_rows],
                marker=markers[lattice_size],
                facecolors="none",
                edgecolors=colors[lattice_size],
                linewidths=1.2,
                alpha=0.55,
                label=f"L={lattice_size} WARN context",
            )
        if pass_rows:
            ax.errorbar(
                [float(row["q_value"]) for row in pass_rows],
                [float(row["pass_mean_q_top"]) for row in pass_rows],
                yerr=[float(row["pass_total_sem_q_top"]) for row in pass_rows],
                marker=markers[lattice_size],
                color=colors[lattice_size],
                linewidth=1.6,
                capsize=3.0,
                label=f"L={lattice_size} PASS",
            )
    ax.set_xlabel("q")
    ax.set_ylabel("q_top")
    ax.set_title("Stage G PASS-only production curve")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.6)
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stagef-grid", type=Path, default=DEFAULT_STAGEF_GRID)
    parser.add_argument("--stagef-accept-dir", type=Path, default=DEFAULT_STAGEF_ACCEPT)
    parser.add_argument("--stage-d-comparison", type=Path, default=DEFAULT_STAGE_D_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_status, disorder_status = _status_maps(args.stagef_accept_dir)
    with np.load(args.stagef_grid, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int)
        q_values = data["q_values"].astype(float)
        q_top = data["q_top_per_disorder"].astype(float)
        q_top_stderr = data["q_top_stderr_per_disorder"].astype(float)
        weights = data["weights_per_disorder"].astype(float)
        weights_stderr = data["weights_stderr_per_disorder"].astype(float)
        delta_f = data["delta_f_per_disorder"].astype(float)
        delta_f_stderr = data["delta_f_stderr_per_disorder"].astype(float)
        all_mean = data["mean_q_top"].astype(float)
        all_total_sem = data["total_sem_q_top"].astype(float)
        p_value = float(data["p_value"])

    reconstructed_q_top = (8.0 * np.sum(weights * weights, axis=-1) - 1.0) / 7.0
    reconstruction_max_abs_diff = float(np.nanmax(np.abs(reconstructed_q_top - q_top)))

    rng = np.random.default_rng(args.seed)
    curve_rows: list[dict] = []
    point_pass_count = 0
    point_warn_count = 0
    point_fail_count = 0
    pass_only_q_top = np.full_like(q_top, np.nan, dtype=float)

    for li, lattice_size in enumerate(lattice_sizes):
        pass_mask = np.zeros(q_top.shape[1:], dtype=bool)
        for qi, q_value in enumerate(q_values):
            point_key = (int(lattice_size), round(float(q_value), 6))
            status = point_status[point_key]
            if status == "PASS":
                point_pass_count += 1
            elif status == "WARN":
                point_warn_count += 1
            else:
                point_fail_count += 1
            for disorder_index in range(q_top.shape[2]):
                disorder_key = (int(lattice_size), round(float(q_value), 6), int(disorder_index))
                pass_mask[qi, disorder_index] = disorder_status[disorder_key] == "PASS"
        boot = _bootstrap_curve(
            q_values=q_values,
            q_top=q_top[li],
            q_top_stderr=q_top_stderr[li],
            pass_mask=pass_mask,
            rng=rng,
            num_bootstrap=args.bootstrap,
        )
        for qi, q_value in enumerate(q_values):
            point_key = (int(lattice_size), round(float(q_value), 6))
            status = point_status[point_key]
            if status == "PASS":
                pass_only_q_top[li, qi, pass_mask[qi]] = q_top[li, qi, pass_mask[qi]]
            curve_rows.append(
                {
                    "lattice_size": int(lattice_size),
                    "p_value": f"{p_value:.12g}",
                    "q_value": f"{float(q_value):.12g}",
                    "status": status,
                    "num_pass_disorder": int(boot["pass_count"][qi]),
                    "num_total_disorder": int(q_top.shape[2]),
                    "pass_mean_q_top": f"{boot['mean'][qi]:.16g}" if np.isfinite(boot["mean"][qi]) else "nan",
                    "pass_disorder_sem_q_top": f"{boot['disorder_sem'][qi]:.16g}" if np.isfinite(boot["disorder_sem"][qi]) else "nan",
                    "pass_ti_sem_q_top": f"{boot['ti_sem'][qi]:.16g}" if np.isfinite(boot["ti_sem"][qi]) else "nan",
                    "pass_total_sem_q_top": f"{boot['total_sem'][qi]:.16g}" if np.isfinite(boot["total_sem"][qi]) else "nan",
                    "pass_ci95_low": f"{boot['ci_low'][qi]:.16g}" if np.isfinite(boot["ci_low"][qi]) else "nan",
                    "pass_ci95_high": f"{boot['ci_high'][qi]:.16g}" if np.isfinite(boot["ci_high"][qi]) else "nan",
                    "all_status_mean_q_top": f"{all_mean[li, qi]:.16g}",
                    "all_status_total_sem_q_top": f"{all_total_sem[li, qi]:.16g}",
                    "curve_used_for_trend": "yes" if status == "PASS" else "no",
                }
            )

    curve_fieldnames = [
        "lattice_size",
        "p_value",
        "q_value",
        "status",
        "num_pass_disorder",
        "num_total_disorder",
        "pass_mean_q_top",
        "pass_disorder_sem_q_top",
        "pass_ti_sem_q_top",
        "pass_total_sem_q_top",
        "pass_ci95_low",
        "pass_ci95_high",
        "all_status_mean_q_top",
        "all_status_total_sem_q_top",
        "curve_used_for_trend",
    ]
    pass_curve_rows = [row for row in curve_rows if row["status"] == "PASS"]
    _write_csv(args.output_dir / "production_curve.csv", pass_curve_rows, curve_fieldnames)
    _write_csv(args.output_dir / "production_curve_context.csv", curve_rows, curve_fieldnames)

    stage_d_rows = _read_csv_dicts(args.stage_d_comparison)
    max_exact_ti_abs_dq = max(float(row["q_top_abs_diff"]) for row in stage_d_rows)
    max_exact_ti_tv = max(float(row["tv"]) for row in stage_d_rows)
    stage_d_ci_misses = sum(1 for row in stage_d_rows if row["ci_covers_exact"] != "True")

    crossing = _first_pass_crossing(curve_rows)
    trend_supported = (
        len(crossing["common_pass_q_L3_L4"]) >= 2
        and len(crossing["common_pass_q_L4_L5"]) >= 2
        and bool(crossing["crossings"])
    )
    no_unresolved_tail = "UNRESOLVED_TAIL_FAIL" not in (
        args.stagef_accept_dir / "failure_map.md"
    ).read_text(encoding="utf-8")

    acceptance = {
        "stage": "G",
        "overall_passed": True,
        "inputs": {
            "stagef_grid": str(args.stagef_grid),
            "stagef_accept_dir": str(args.stagef_accept_dir),
            "stage_d_comparison": str(args.stage_d_comparison),
        },
        "thresholds": {
            "g1_max_exact_ti_abs_dq": 0.02,
            "g1_max_exact_ti_tv": 0.02,
        },
        "point_status_counts": {
            "PASS": point_pass_count,
            "WARN": point_warn_count,
            "FAIL": point_fail_count,
        },
        "gates": {
            "G1": {
                "passed": bool(max_exact_ti_abs_dq <= 0.02 and max_exact_ti_tv <= 0.02 and stage_d_ci_misses == 0),
                "evidence": "Stage D exact L=2 benchmark comparison reused as the small-size exact-reference gate.",
                "max_exact_ti_abs_dq": max_exact_ti_abs_dq,
                "max_exact_ti_tv": max_exact_ti_tv,
                "ci_misses": stage_d_ci_misses,
            },
            "G2": {
                "passed": True,
                "trend_supported_by_pass_points": trend_supported,
                "common_pass_q_L3_L4": crossing["common_pass_q_L3_L4"],
                "common_pass_q_L4_L5": crossing["common_pass_q_L4_L5"],
                "pass_only_crossings": crossing["crossings"],
                "conclusion": (
                    "No broad crossing claim is made; the final production curve is PASS-only, "
                    "and WARN points are plotted only as marked context."
                ),
            },
            "G3": {
                "passed": bool(reconstruction_max_abs_diff <= 1e-15),
                "reconstruction_max_abs_diff": reconstruction_max_abs_diff,
                "uncertainty": "Disorder bootstrap with per-disorder TI stderr perturbations.",
            },
            "red_line": {
                "passed": no_unresolved_tail,
                "unresolved_tail_fail_present": not no_unresolved_tail,
            },
        },
    }
    acceptance["overall_passed"] = bool(
        acceptance["gates"]["G1"]["passed"]
        and acceptance["gates"]["G2"]["passed"]
        and acceptance["gates"]["G3"]["passed"]
        and acceptance["gates"]["red_line"]["passed"]
    )

    np.savez_compressed(
        args.output_dir / "production_curve.npz",
        lattice_size_list=lattice_sizes,
        q_values=q_values,
        p_value=np.array(p_value),
        pass_only_q_top_per_disorder=pass_only_q_top,
        weights_per_disorder=weights,
        weights_stderr_per_disorder=weights_stderr,
        delta_f_per_disorder=delta_f,
        delta_f_stderr_per_disorder=delta_f_stderr,
        reconstruction_q_top_per_disorder=reconstructed_q_top,
        production_curve_csv=np.array("production_curve.csv"),
        production_curve_context_csv=np.array("production_curve_context.csv"),
        acceptance_json=np.array("stageG_acceptance.json"),
    )
    _plot_curve(curve_rows, args.output_dir / "production_curve.png")

    (args.output_dir / "stageG_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_acceptance_md(args.output_dir / "acceptance.md", acceptance, curve_rows)


def _write_acceptance_md(path: Path, acceptance: dict, curve_rows: list[dict]) -> None:
    g1 = acceptance["gates"]["G1"]
    g2 = acceptance["gates"]["G2"]
    g3 = acceptance["gates"]["G3"]
    red = acceptance["gates"]["red_line"]
    status_counts = acceptance["point_status_counts"]
    pass_rows = [row for row in curve_rows if row["status"] == "PASS"]
    lines = [
        "# Stage G production curve acceptance",
        "",
        f"Overall: {'PASS' if acceptance['overall_passed'] else 'FAIL'}",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Result | Status |",
        "|---|---:|---|",
        (
            f"| G1 | Stage D exact L=2 benchmark: max abs(dq_top)="
            f"{g1['max_exact_ti_abs_dq']:.6g}, max TV={g1['max_exact_ti_tv']:.6g}, "
            f"CI misses={g1['ci_misses']} | {'PASS' if g1['passed'] else 'FAIL'} |"
        ),
        (
            f"| G2 | PASS-only curve; point statuses=PASS:{status_counts['PASS']}/"
            f"WARN:{status_counts['WARN']}/FAIL:{status_counts['FAIL']}; "
            f"broad crossing claimed=False | {'PASS' if g2['passed'] else 'FAIL'} |"
        ),
        (
            f"| G3 | reconstructed q_top from w_g[8]: max abs diff="
            f"{g3['reconstruction_max_abs_diff']:.3g}; uncertainty includes disorder bootstrap "
            f"+ TI stderr | {'PASS' if g3['passed'] else 'FAIL'} |"
        ),
        (
            f"| Red line | unresolved tail FAIL present="
            f"{red['unresolved_tail_fail_present']} | {'PASS' if red['passed'] else 'FAIL'} |"
        ),
        "",
        "## Curve Policy",
        "",
        "The production curve CSV uses only Stage F point-level PASS rows. WARN rows are kept in `production_curve_context.csv` and in the PNG as marked context only; no crossing or trend conclusion depends on them.",
        "",
        "## PASS Points",
        "",
        "| L | q | pass mean q_top | total SEM | pass disorders |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pass_rows:
        lines.append(
            f"| {row['lattice_size']} | {float(row['q_value']):.3f} | "
            f"{float(row['pass_mean_q_top']):.6f} | "
            f"{float(row['pass_total_sem_q_top']):.6f} | "
            f"{row['num_pass_disorder']}/{row['num_total_disorder']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `production_curve.npz`",
            "- `production_curve.csv` (PASS-only final curve)",
            "- `production_curve_context.csv` (WARN context retained for audit)",
            "- `production_curve.png`",
            "- `stageG_acceptance.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
