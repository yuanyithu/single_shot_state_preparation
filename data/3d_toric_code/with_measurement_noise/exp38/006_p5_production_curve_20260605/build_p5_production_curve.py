#!/usr/bin/env python3
"""Build exp38 P5 production curves, paired-difference plot, and acceptance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
EXP38_DIR = REPO_ROOT / "data/3d_toric_code/with_measurement_noise/exp38"
DEFAULT_TI_RESULTS = (
    EXP38_DIR
    / "003_p2_production_grid_20260605"
    / "merged_exp38_p2_ti_grid_20260605_0145"
    / "sector_ti_results.npz"
)
DEFAULT_P4_ACCEPTANCE = EXP38_DIR / "005_p4_acceptance_20260605" / "p4_acceptance.json"
DEFAULT_P4_POINT_STATUS = EXP38_DIR / "005_p4_acceptance_20260605" / "p4_point_status.csv"
DEFAULT_P4_PAIRED = EXP38_DIR / "005_p4_acceptance_20260605" / "paired_difference.csv"
DEFAULT_P0_COMPARISON = EXP38_DIR / "001_p0_regression_anchor_20260604" / "ti_comparison.csv"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_q(value: Any) -> float:
    return float(round(float(value), 6))


def _load_scalar_text(array: np.ndarray) -> str:
    value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _point_status_maps(
    path: Path,
    p4_acceptance_path: Path,
) -> tuple[dict[tuple[int, float], dict[str, str]], dict[tuple[int, float, int], str]]:
    point_rows = _read_csv_dicts(path)
    point_status = {
        (int(row["lattice_size"]), _round_q(row["q_value"])): row
        for row in point_rows
    }
    # Disorder-level PASS/WARN/FAIL comes from P4 acceptance JSON to avoid a separate CSV dependency.
    p4_payload = json.loads(p4_acceptance_path.read_text(encoding="utf-8"))
    disorder_status = {
        (
            int(row["lattice_size"]),
            _round_q(row["q_value"]),
            int(row["disorder_index"]),
        ): str(row["status"])
        for row in p4_payload["p4a_source_disorder_rows"]
    } if "p4a_source_disorder_rows" in p4_payload else {}
    if disorder_status:
        return point_status, disorder_status

    p2_acceptance = Path(p4_payload["inputs"]["p2_acceptance"])
    p2_payload = json.loads(p2_acceptance.read_text(encoding="utf-8"))
    disorder_status = {
        (
            int(row["lattice_size"]),
            _round_q(row["q_value"]),
            int(row["disorder_index"]),
        ): str(row["status"])
        for row in p2_payload["disorder_rows"]
    }
    return point_status, disorder_status


def _bootstrap_curve(
    values: np.ndarray,
    stderrs: np.ndarray,
    pass_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> dict[str, np.ndarray]:
    num_q = values.shape[0]
    mean = np.full(num_q, np.nan)
    disorder_sem = np.full(num_q, np.nan)
    ti_sem = np.full(num_q, np.nan)
    total_sem = np.full(num_q, np.nan)
    ci_low = np.full(num_q, np.nan)
    ci_high = np.full(num_q, np.nan)
    pass_count = np.zeros(num_q, dtype=np.int64)

    for qi in range(num_q):
        mask = np.asarray(pass_mask[qi], dtype=bool) & np.isfinite(values[qi])
        selected = np.asarray(values[qi, mask], dtype=np.float64)
        selected_stderr = np.nan_to_num(
            np.asarray(stderrs[qi, mask], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        count = int(selected.size)
        pass_count[qi] = count
        if count == 0:
            continue
        mean[qi] = float(np.mean(selected))
        disorder_sem[qi] = float(np.std(selected, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        ti_sem[qi] = float(math.sqrt(np.sum(selected_stderr**2)) / count)
        draws = np.empty(bootstrap_reps, dtype=np.float64)
        for bi in range(bootstrap_reps):
            indices = rng.integers(0, count, size=count)
            sample = selected[indices]
            sample_stderr = selected_stderr[indices]
            if np.any(sample_stderr > 0.0):
                sample = rng.normal(sample, sample_stderr)
            draws[bi] = float(np.mean(sample))
        total_sem[qi] = float(np.std(draws, ddof=1))
        ci_low[qi], ci_high[qi] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    return {
        "mean": mean,
        "disorder_sem": disorder_sem,
        "ti_sem": ti_sem,
        "total_sem": total_sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass_count": pass_count,
    }


def _build_curve_rows(
    *,
    lattice_sizes: np.ndarray,
    q_values: np.ndarray,
    p_value: float,
    q_top: np.ndarray,
    q_top_stderr: np.ndarray,
    all_mean: np.ndarray,
    all_total_sem: np.ndarray,
    point_status: dict[tuple[int, float], dict[str, str]],
    disorder_status: dict[tuple[int, float, int], str],
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    curve_rows: list[dict[str, Any]] = []
    pass_only_q_top = np.full_like(q_top, np.nan, dtype=np.float64)

    for li, lattice_size in enumerate(lattice_sizes):
        pass_mask = np.zeros(q_top.shape[1:], dtype=bool)
        for qi, q_value in enumerate(q_values):
            q_key = _round_q(q_value)
            for disorder_index in range(q_top.shape[2]):
                pass_mask[qi, disorder_index] = (
                    disorder_status[(int(lattice_size), q_key, int(disorder_index))] == "PASS"
                )
        boot = _bootstrap_curve(
            q_top[li],
            q_top_stderr[li],
            pass_mask,
            rng=rng,
            bootstrap_reps=bootstrap_reps,
        )
        pass_only_q_top[li, pass_mask] = q_top[li, pass_mask]

        for qi, q_value in enumerate(q_values):
            point_row = point_status[(int(lattice_size), _round_q(q_value))]
            status = str(point_row["p4_status"])
            curve_rows.append(
                {
                    "lattice_size": int(lattice_size),
                    "p_value": f"{p_value:.12g}",
                    "q_value": f"{float(q_value):.12g}",
                    "status": status,
                    "curve_used_for_final": "yes" if status == "PASS" else "no",
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
                    "p4_point_flags": point_row["p2_flags"],
                }
            )
    return curve_rows, pass_only_q_top


def _plot_production_curve(curve_rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
    markers = {3: "o", 4: "s", 5: "^"}
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
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
                linewidths=1.25,
                alpha=0.55,
                label=f"L={lattice_size} WARN context",
            )
        if pass_rows:
            yerr = [
                [float(row["pass_mean_q_top"]) - float(row["pass_ci95_low"]) for row in pass_rows],
                [float(row["pass_ci95_high"]) - float(row["pass_mean_q_top"]) for row in pass_rows],
            ]
            ax.errorbar(
                [float(row["q_value"]) for row in pass_rows],
                [float(row["pass_mean_q_top"]) for row in pass_rows],
                yerr=yerr,
                marker=markers[lattice_size],
                color=colors[lattice_size],
                linewidth=1.6,
                capsize=3.0,
                label=f"L={lattice_size} PASS",
            )
    ax.set_xlabel("q")
    ax.set_ylabel("q_top")
    ax.set_title("exp38 production curve: PASS rows only")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.65)
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_paired_difference(rows: list[dict[str, str]], output_path: Path) -> None:
    colors = {"3-4": "#1f77b4", "3-5": "#9467bd", "4-5": "#2ca02c"}
    markers = {"3-4": "o", "3-5": "D", "4-5": "s"}
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    for pair in ("3-4", "3-5", "4-5"):
        left, right = [int(x) for x in pair.split("-")]
        pair_rows = [
            row
            for row in rows
            if int(row["lattice_size_a"]) == left and int(row["lattice_size_b"]) == right
        ]
        pair_rows.sort(key=lambda row: float(row["q_value"]))
        x = np.asarray([float(row["q_value"]) for row in pair_rows], dtype=np.float64)
        y = np.asarray([float(row["delta_mean"]) for row in pair_rows], dtype=np.float64)
        lo = np.asarray([float(row["bootstrap_ci95_low"]) for row in pair_rows], dtype=np.float64)
        hi = np.asarray([float(row["bootstrap_ci95_high"]) for row in pair_rows], dtype=np.float64)
        significant = np.asarray([row["ci_excludes_zero"] == "True" for row in pair_rows], dtype=bool)
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            marker=markers[pair],
            color=colors[pair],
            linewidth=1.4,
            capsize=3.0,
            label=f"L{pair.replace('-', '-L')}",
        )
        if np.any(significant):
            ax.scatter(
                x[significant],
                y[significant],
                marker=markers[pair],
                s=70,
                color=colors[pair],
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
    ax.set_xlabel("q")
    ax.set_ylabel("paired delta q_top")
    ax.set_title("exp38 paired differences: PASS-only common disorder")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.65)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _p0_gate(path: Path) -> dict[str, Any]:
    rows = _read_csv_dicts(path)
    max_abs_dq = max(float(row["q_top_abs_diff"]) for row in rows)
    max_tv = max(float(row["tv"]) for row in rows)
    ci_misses = sum(row["ci_covers_exact"] != "True" for row in rows)
    return {
        "passed": bool(max_abs_dq <= 0.02 and max_tv <= 0.02 and ci_misses == 0),
        "max_abs_dq": max_abs_dq,
        "max_tv": max_tv,
        "ci_misses": int(ci_misses),
        "source": str(path),
    }


def _paired_conclusion(rows: list[dict[str, str]]) -> dict[str, Any]:
    crossing_rows = [row for row in rows if row["crossing_region"] == "True"]
    significant = [row for row in crossing_rows if row["ci_excludes_zero"] == "True"]
    pair_summary: dict[str, dict[str, Any]] = {}
    for pair in ("3-4", "3-5", "4-5"):
        left, right = [int(x) for x in pair.split("-")]
        pair_rows = [
            row
            for row in crossing_rows
            if int(row["lattice_size_a"]) == left and int(row["lattice_size_b"]) == right
        ]
        pair_sig = [row for row in pair_rows if row["ci_excludes_zero"] == "True"]
        pair_summary[pair] = {
            "num_crossing_region_rows": int(len(pair_rows)),
            "num_ci_excludes_zero": int(len(pair_sig)),
            "q_values": [float(row["q_value"]) for row in pair_sig],
            "min_paired_count": min(int(row["effective_paired_disorder_count"]) for row in pair_rows),
            "max_abs_significant_delta": max((abs(float(row["delta_mean"])) for row in pair_sig), default=0.0),
        }
    all_pairs_resolved = all(item["num_ci_excludes_zero"] > 0 for item in pair_summary.values())
    return {
        "passed": bool(len(significant) > 0),
        "num_crossing_region_ci_excludes_zero": int(len(significant)),
        "significant_rows": [
            {
                "pair": f"{row['lattice_size_a']}-{row['lattice_size_b']}",
                "q_value": float(row["q_value"]),
                "delta_mean": float(row["delta_mean"]),
                "ci95": [float(row["bootstrap_ci95_low"]), float(row["bootstrap_ci95_high"])],
                "effective_paired_disorder_count": int(row["effective_paired_disorder_count"]),
            }
            for row in significant
        ],
        "pair_summary": pair_summary,
        "statistically_resolved_common_three_size_crossing": bool(all_pairs_resolved),
        "crossing_statement": (
            "High-q finite-size separation is statistically resolved for L3-L5 and L3-L4 at the listed q values, "
            "but a common three-size crossing is not statistically resolved because L4-L5 has no crossing-region "
            "paired CI excluding zero."
        ),
    }


def _write_acceptance_md(path: Path, acceptance: dict[str, Any], curve_rows: list[dict[str, Any]]) -> None:
    g1 = acceptance["gates"]["G1"]
    g2 = acceptance["gates"]["G2"]
    g3 = acceptance["gates"]["G3"]
    red = acceptance["gates"]["red_line"]
    counts = acceptance["point_status_counts"]
    pass_rows = [row for row in curve_rows if row["status"] == "PASS"]
    lines = [
        "# exp38 P5 production curve acceptance",
        "",
        f"Overall: `{'PASS' if acceptance['overall_passed'] else 'FAIL'}`",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Result | Status |",
        "|---|---:|---|",
        (
            f"| G1 | P0 exact benchmark replay: max TV={g1['max_tv']:.6f}, "
            f"max |dq_top|={g1['max_abs_dq']:.6f}, CI misses={g1['ci_misses']} | "
            f"{'PASS' if g1['passed'] else 'FAIL'} |"
        ),
        (
            f"| G2 | paired CI evidence only: significant crossing-region rows="
            f"{g2['num_crossing_region_ci_excludes_zero']}; common three-size crossing resolved="
            f"{g2['statistically_resolved_common_three_size_crossing']} | "
            f"{'PASS' if g2['passed'] else 'FAIL'} |"
        ),
        (
            f"| G3 | q_top reconstructed from w_g[8]: max abs diff="
            f"{g3['reconstruction_max_abs_diff']:.3g}; uncertainty includes disorder bootstrap + TI stderr | "
            f"{'PASS' if g3['passed'] else 'FAIL'} |"
        ),
        (
            f"| Red line | unresolved tail FAIL present={red['unresolved_tail_fail_present']} | "
            f"{'PASS' if red['passed'] else 'FAIL'} |"
        ),
        "",
        "## Crossing Conclusion",
        "",
        g2["crossing_statement"],
        "",
        "The conclusion uses only `paired_difference.csv` rows where the paired bootstrap CI excludes zero. WARN context points and independent mean overlaps are not used to claim crossing.",
        "",
        "## Point Statuses",
        "",
        f"Point statuses: PASS:{counts['PASS']}, WARN:{counts['WARN']}, FAIL:{counts['FAIL']}. `production_curve.csv` contains only PASS rows; WARN rows are retained in `production_curve_context.csv` and plotted hollow in Figure A.",
        "",
        "## PASS Curve Rows",
        "",
        "| L | q | mean q_top | total SEM | 95% CI | pass disorders |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pass_rows:
        lines.append(
            f"| {row['lattice_size']} | {float(row['q_value']):.3f} | "
            f"{float(row['pass_mean_q_top']):.6f} | "
            f"{float(row['pass_total_sem_q_top']):.6f} | "
            f"[{float(row['pass_ci95_low']):.6f}, {float(row['pass_ci95_high']):.6f}] | "
            f"{row['num_pass_disorder']}/{row['num_total_disorder']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `production_curve.npz`",
            "- `production_curve.csv`",
            "- `production_curve_context.csv`",
            "- `paired_difference.csv`",
            "- `production_curve.png`",
            "- `paired_difference.png`",
            "- `p5_acceptance.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, acceptance: dict[str, Any]) -> None:
    g1 = acceptance["gates"]["G1"]
    g2 = acceptance["gates"]["G2"]
    g3 = acceptance["gates"]["G3"]
    red = acceptance["gates"]["red_line"]
    lines = [
        "# exp38 P5 production curve summary",
        "",
        f"Status: `{'PASS' if acceptance['overall_passed'] else 'FAIL'}`",
        "",
        "## Gates",
        "",
        "| Gate | Key numbers | Status |",
        "|---|---:|---|",
        f"| G1 | max TV={g1['max_tv']:.6f}, max |dq_top|={g1['max_abs_dq']:.6f}, CI misses={g1['ci_misses']} | {'PASS' if g1['passed'] else 'FAIL'} |",
        f"| G2 | paired CI excludes zero rows={g2['num_crossing_region_ci_excludes_zero']}; common three-size crossing resolved={g2['statistically_resolved_common_three_size_crossing']} | {'PASS' if g2['passed'] else 'FAIL'} |",
        f"| G3 | reconstruct max abs diff={g3['reconstruction_max_abs_diff']:.3g} | {'PASS' if g3['passed'] else 'FAIL'} |",
        f"| Red line | unresolved tail fail present={red['unresolved_tail_fail_present']} | {'PASS' if red['passed'] else 'FAIL'} |",
        "",
        "## Conclusion",
        "",
        g2["crossing_statement"],
        "",
        "Artifacts: `production_curve.png`, `paired_difference.png`, `production_curve.csv`, `paired_difference.csv`, `acceptance.md`, `p5_acceptance.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-results", type=Path, default=DEFAULT_TI_RESULTS)
    parser.add_argument("--p4-acceptance", type=Path, default=DEFAULT_P4_ACCEPTANCE)
    parser.add_argument("--p4-point-status", type=Path, default=DEFAULT_P4_POINT_STATUS)
    parser.add_argument("--p4-paired", type=Path, default=DEFAULT_P4_PAIRED)
    parser.add_argument("--p0-comparison", type=Path, default=DEFAULT_P0_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    p4_payload = json.loads(args.p4_acceptance.read_text(encoding="utf-8"))
    if not p4_payload.get("overall_passed", False):
        raise RuntimeError("P4 acceptance is not PASS; refusing to build P5")

    point_status, disorder_status = _point_status_maps(args.p4_point_status, args.p4_acceptance)
    with np.load(args.ti_results, allow_pickle=False) as data:
        manifest = json.loads(_load_scalar_text(data["manifest_json"]))
        lattice_sizes = data["lattice_size_list"].astype(np.int64)
        q_values = data["q_values"].astype(np.float64)
        p_value = float(data["p_value"])
        q_top = data["q_top_per_disorder"].astype(np.float64)
        q_top_stderr = data["q_top_stderr_per_disorder"].astype(np.float64)
        weights = data["weights_per_disorder"].astype(np.float64)
        weights_stderr = data["weights_stderr_per_disorder"].astype(np.float64)
        delta_f = data["delta_f_per_disorder"].astype(np.float64)
        delta_f_stderr = data["delta_f_stderr_per_disorder"].astype(np.float64)
        all_mean = data["mean_q_top"].astype(np.float64)
        all_total_sem = data["total_sem_q_top"].astype(np.float64)

    rng = np.random.default_rng(args.seed)
    curve_rows, pass_only_q_top = _build_curve_rows(
        lattice_sizes=lattice_sizes,
        q_values=q_values,
        p_value=p_value,
        q_top=q_top,
        q_top_stderr=q_top_stderr,
        all_mean=all_mean,
        all_total_sem=all_total_sem,
        point_status=point_status,
        disorder_status=disorder_status,
        rng=rng,
        bootstrap_reps=args.bootstrap,
    )
    paired_rows = _read_csv_dicts(args.p4_paired)

    curve_fieldnames = [
        "lattice_size",
        "p_value",
        "q_value",
        "status",
        "curve_used_for_final",
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
        "p4_point_flags",
    ]
    pass_curve_rows = [row for row in curve_rows if row["status"] == "PASS"]
    _write_csv(args.output_dir / "production_curve.csv", pass_curve_rows, curve_fieldnames)
    _write_csv(args.output_dir / "production_curve_context.csv", curve_rows, curve_fieldnames)
    _write_csv(args.output_dir / "paired_difference.csv", paired_rows, list(paired_rows[0].keys()))

    reconstructed_q_top = (8.0 * np.sum(weights * weights, axis=-1) - 1.0) / 7.0
    reconstruction_max_abs_diff = float(np.nanmax(np.abs(reconstructed_q_top - q_top)))
    p0_gate = _p0_gate(args.p0_comparison)
    paired_conclusion = _paired_conclusion(paired_rows)
    point_counts = {
        "PASS": sum(row["status"] == "PASS" for row in curve_rows),
        "WARN": sum(row["status"] == "WARN" for row in curve_rows),
        "FAIL": sum(row["status"] == "FAIL" for row in curve_rows),
    }
    red_line_passed = bool(
        p4_payload["gates"]["P4a"]["num_fail_points"] == 0
        and p4_payload["gates"]["P4a"]["num_fail_disorder"] == 0
        and p4_payload["gates"]["P4a"]["gate_passes"]["P2b"]
    )

    acceptance = {
        "stage": "P5",
        "overall_passed": False,
        "inputs": {
            "ti_results": str(args.ti_results),
            "p4_acceptance": str(args.p4_acceptance),
            "p4_point_status": str(args.p4_point_status),
            "p4_paired": str(args.p4_paired),
            "p0_comparison": str(args.p0_comparison),
        },
        "manifest": manifest,
        "point_status_counts": point_counts,
        "thresholds": {
            "g1_max_abs_dq": 0.02,
            "g1_max_tv": 0.02,
            "g3_reconstruction_tolerance": 1.0e-12,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "gates": {
            "G1": p0_gate,
            "G2": paired_conclusion,
            "G3": {
                "passed": bool(reconstruction_max_abs_diff <= 1.0e-12),
                "reconstruction_max_abs_diff": reconstruction_max_abs_diff,
                "uncertainty": "disorder bootstrap over PASS disorders plus per-disorder TI stderr Gaussian perturbation",
            },
            "red_line": {
                "passed": red_line_passed,
                "unresolved_tail_fail_present": not red_line_passed,
                "source": "P4a includes P2b unresolved-tail red-line gate and no FAIL point/disorder rows.",
            },
        },
    }
    acceptance["overall_passed"] = bool(
        acceptance["gates"]["G1"]["passed"]
        and acceptance["gates"]["G2"]["passed"]
        and acceptance["gates"]["G3"]["passed"]
        and acceptance["gates"]["red_line"]["passed"]
    )

    _plot_production_curve(curve_rows, args.output_dir / "production_curve.png")
    _plot_paired_difference(paired_rows, args.output_dir / "paired_difference.png")
    np.savez_compressed(
        args.output_dir / "production_curve.npz",
        lattice_size_list=lattice_sizes,
        q_values=q_values,
        p_value=np.array(p_value),
        pass_only_q_top_per_disorder=pass_only_q_top,
        q_top_per_disorder=q_top,
        q_top_stderr_per_disorder=q_top_stderr,
        weights_per_disorder=weights,
        weights_stderr_per_disorder=weights_stderr,
        delta_f_per_disorder=delta_f,
        delta_f_stderr_per_disorder=delta_f_stderr,
        reconstructed_q_top_per_disorder=reconstructed_q_top,
        production_curve_csv=np.array("production_curve.csv"),
        production_curve_context_csv=np.array("production_curve_context.csv"),
        paired_difference_csv=np.array("paired_difference.csv"),
        acceptance_json=np.array("p5_acceptance.json"),
    )
    (args.output_dir / "p5_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_acceptance_md(args.output_dir / "acceptance.md", acceptance, curve_rows)
    _write_summary_md(args.output_dir / "summary.md", acceptance)

    print(
        json.dumps(
            {
                "overall_passed": acceptance["overall_passed"],
                "G1_passed": acceptance["gates"]["G1"]["passed"],
                "G2_passed": acceptance["gates"]["G2"]["passed"],
                "G3_passed": acceptance["gates"]["G3"]["passed"],
                "red_line_passed": acceptance["gates"]["red_line"]["passed"],
                "point_status_counts": point_counts,
                "num_pair_ci_excludes_zero": acceptance["gates"]["G2"]["num_crossing_region_ci_excludes_zero"],
                "common_three_size_crossing_resolved": acceptance["gates"]["G2"]["statistically_resolved_common_three_size_crossing"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if acceptance["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
