import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CI95_Z_SCORE = 1.96
SIGN_TOLERANCE = 1.0e-12


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _format_probability_tag(value):
    return f"{float(value):0.4f}".replace(".", "p")


def _load_q_runs(run_root):
    run_root = Path(run_root)
    items = []
    for q_dir in sorted(run_root.glob("q_*")):
        npz_paths = sorted(q_dir.glob("*.npz"))
        if not npz_paths:
            continue
        npz_path = npz_paths[0]
        with np.load(npz_path, allow_pickle=True) as loaded:
            q_value = float(loaded["syndrome_error_probability"])
            p_values = np.asarray(
                loaded["data_error_probability_list"],
                dtype=np.float64,
            )
            lattice_sizes = np.asarray(
                loaded["lattice_size_list"],
                dtype=np.int64,
            )
            q_top = np.asarray(loaded["q_top_curve_matrix"], dtype=np.float64)
            q_top_std = np.asarray(
                loaded["q_top_std_curve_matrix"],
                dtype=np.float64,
            )
            num_disorder_samples = int(loaded["num_disorder_samples"])
            fields = set(loaded.files)
            diagnostics = {}
            for key in (
                    "mean_q_top_spread_curve_matrix",
                    "max_r_hat_curve_matrix",
                    "min_effective_sample_size_curve_matrix",
                    "mean_pt_min_swap_acceptance_rate_curve_matrix",
                    "converged_mask_matrix",
                    "mean_cold_winding_acceptance_rate_curve_matrix"):
                if key in fields:
                    diagnostics[key] = np.asarray(loaded[key])
            if "q0_mean_q_top_spread_curve_matrix" in fields:
                diagnostics["q0_mean_q_top_spread_curve_matrix"] = np.asarray(
                    loaded["q0_mean_q_top_spread_curve_matrix"]
                )
        items.append({
            "q_dir": q_dir,
            "npz_path": npz_path,
            "q": q_value,
            "p_values": p_values,
            "lattice_sizes": lattice_sizes,
            "q_top": q_top,
            "q_top_std": q_top_std,
            "num_disorder_samples": num_disorder_samples,
            "diagnostics": diagnostics,
        })

    if not items:
        raise ValueError(f"No q_*/*.npz files found under {run_root}")
    items.sort(key=lambda item: item["q"])

    reference_p_values = items[0]["p_values"]
    reference_lattice_sizes = items[0]["lattice_sizes"]
    reference_num_samples = items[0]["num_disorder_samples"]
    for item in items[1:]:
        if not np.allclose(item["p_values"], reference_p_values):
            raise ValueError("data_error_probability_list mismatch across q runs")
        if not np.array_equal(item["lattice_sizes"], reference_lattice_sizes):
            raise ValueError("lattice_size_list mismatch across q runs")
        if item["num_disorder_samples"] != reference_num_samples:
            raise ValueError("num_disorder_samples mismatch across q runs")

    return items


def _linear_crossing(x0, x1, y0, y1):
    denominator = y1 - y0
    if abs(denominator) <= SIGN_TOLERANCE:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / denominator)


def _sign(value):
    if abs(float(value)) <= SIGN_TOLERANCE:
        return 0
    return -1 if value < 0.0 else 1


def _detect_crossings(q_values, gap_values):
    crossings = []
    for index in range(len(q_values) - 1):
        left_q = float(q_values[index])
        right_q = float(q_values[index + 1])
        left_gap = float(gap_values[index])
        right_gap = float(gap_values[index + 1])
        left_sign = _sign(left_gap)
        right_sign = _sign(right_gap)
        if left_sign == 0 and right_sign == 0:
            continue
        if left_sign == 0:
            direction = (
                "zero_to_positive" if right_sign > 0 else "zero_to_negative"
            )
            crossings.append({
                "left_index": int(index),
                "right_index": int(index),
                "left_q": left_q,
                "right_q": left_q,
                "crossing_estimate_q": left_q,
                "direction": direction,
            })
            continue
        if right_sign == 0:
            direction = (
                "negative_to_zero" if left_sign < 0 else "positive_to_zero"
            )
            crossings.append({
                "left_index": int(index + 1),
                "right_index": int(index + 1),
                "left_q": right_q,
                "right_q": right_q,
                "crossing_estimate_q": right_q,
                "direction": direction,
            })
            continue
        if left_sign * right_sign < 0:
            if left_sign < 0 and right_sign > 0:
                direction = "negative_to_positive"
            else:
                direction = "positive_to_negative"
            crossings.append({
                "left_index": int(index),
                "right_index": int(index + 1),
                "left_q": left_q,
                "right_q": right_q,
                "crossing_estimate_q": _linear_crossing(
                    x0=left_q,
                    x1=right_q,
                    y0=left_gap,
                    y1=right_gap,
                ),
                "direction": direction,
            })
    return crossings


def _primary_crossing(crossings):
    upward = [
        crossing for crossing in crossings
        if crossing["direction"] == "negative_to_positive"
    ]
    if upward:
        return upward[0]
    nonzero_crossings = [
        crossing for crossing in crossings
        if crossing["crossing_estimate_q"] > SIGN_TOLERANCE
    ]
    if nonzero_crossings:
        return nonzero_crossings[0]
    return crossings[0] if crossings else None


def _classify_gap(gap_values, crossings):
    signs = np.array([_sign(value) for value in gap_values], dtype=np.int8)
    nonzero = signs[signs != 0]
    if len(crossings) > 1:
        return "nonmonotonic_or_noisy"
    if len(crossings) == 1:
        return "single_crossing"
    if nonzero.size == 0:
        return "flat_zero"
    if np.all(nonzero < 0):
        return "below_threshold_through_q_window"
    if np.all(nonzero > 0):
        return "above_threshold_through_q_window"
    return "mixed_without_detected_crossing"


def _build_summary(items):
    q_values = np.array([item["q"] for item in items], dtype=np.float64)
    p_values = items[0]["p_values"]
    lattice_sizes = items[0]["lattice_sizes"]
    num_disorder_samples = items[0]["num_disorder_samples"]

    q_top_by_q = np.stack([item["q_top"] for item in items], axis=0)
    q_top_std_by_q = np.stack([item["q_top_std"] for item in items], axis=0)
    q_top = np.transpose(q_top_by_q, (1, 2, 0))
    q_top_std = np.transpose(q_top_std_by_q, (1, 2, 0))
    q_top_sem = q_top_std / math.sqrt(float(num_disorder_samples))
    q_top_ci95 = CI95_Z_SCORE * q_top_sem

    pair_specs = [
        (0, 1, f"L{int(lattice_sizes[0])}-L{int(lattice_sizes[1])}"),
        (1, 2, f"L{int(lattice_sizes[1])}-L{int(lattice_sizes[2])}"),
    ]
    pair_summaries = []
    csv_rows = []
    for p_index, p_value in enumerate(p_values):
        per_pair = []
        for left_index, right_index, label in pair_specs:
            gap_values = (
                q_top[left_index, p_index, :]
                - q_top[right_index, p_index, :]
            )
            gap_sem = np.sqrt(
                q_top_sem[left_index, p_index, :] ** 2
                + q_top_sem[right_index, p_index, :] ** 2
            )
            gap_ci95 = CI95_Z_SCORE * gap_sem
            crossings = _detect_crossings(q_values, gap_values)
            primary = _primary_crossing(crossings)
            classification = _classify_gap(gap_values, crossings)
            record = {
                "pair": label,
                "p": float(p_value),
                "classification": classification,
                "crossings": crossings,
                "primary_crossing": primary,
                "gap_values": gap_values.tolist(),
                "gap_ci95": gap_ci95.tolist(),
                "min_abs_gap_q": float(q_values[int(np.argmin(np.abs(gap_values)))]),
                "min_abs_gap": float(gap_values[int(np.argmin(np.abs(gap_values)))]),
            }
            per_pair.append(record)
            csv_rows.append({
                "p": float(p_value),
                "pair": label,
                "classification": classification,
                "primary_crossing_q": (
                    None if primary is None else primary["crossing_estimate_q"]
                ),
                "primary_crossing_q_left": (
                    None if primary is None else primary["left_q"]
                ),
                "primary_crossing_q_right": (
                    None if primary is None else primary["right_q"]
                ),
                "num_crossings": len(crossings),
                "min_abs_gap_q": record["min_abs_gap_q"],
                "min_abs_gap": record["min_abs_gap"],
            })
        stable_records = [
            record for record in per_pair
            if (
                record["primary_crossing"] is not None
                and record["classification"] == "single_crossing"
            )
        ]
        primary_estimates = [
            record["primary_crossing"]["crossing_estimate_q"]
            for record in stable_records
        ]
        if len(primary_estimates) == len(per_pair):
            common = {
                "p": float(p_value),
                "q_min": float(min(primary_estimates)),
                "q_max": float(max(primary_estimates)),
                "representative_q": float(np.mean(primary_estimates)),
                "reason": "both_adjacent_pair_crossings_stable",
            }
        else:
            common = None
        pair_summaries.append({
            "p": float(p_value),
            "pairs": per_pair,
            "common_crossing_window": common,
        })

    diagnostics = _build_diagnostic_summary(items, p_values, q_values)
    return {
        "q_values": q_values,
        "p_values": p_values,
        "lattice_sizes": lattice_sizes,
        "num_disorder_samples": num_disorder_samples,
        "q_top": q_top,
        "q_top_ci95": q_top_ci95,
        "pair_summaries": pair_summaries,
        "csv_rows": csv_rows,
        "diagnostics": diagnostics,
    }


def _build_diagnostic_summary(items, p_values, q_values):
    metric_values = {
        "worst_mean_q_top_spread": np.full(
            (len(q_values), len(p_values)), np.nan
        ),
        "worst_max_r_hat": np.full((len(q_values), len(p_values)), np.nan),
        "worst_min_effective_sample_size": np.full(
            (len(q_values), len(p_values)), np.nan
        ),
        "worst_mean_pt_min_swap_acceptance_rate": np.full(
            (len(q_values), len(p_values)), np.nan
        ),
        "converged_fraction": np.full((len(q_values), len(p_values)), np.nan),
    }
    q0_spread = np.full((len(q_values), len(p_values)), np.nan)

    for q_index, item in enumerate(items):
        diagnostics = item["diagnostics"]
        if "mean_q_top_spread_curve_matrix" in diagnostics:
            metric_values["worst_mean_q_top_spread"][q_index, :] = np.max(
                diagnostics["mean_q_top_spread_curve_matrix"],
                axis=0,
            )
        if "max_r_hat_curve_matrix" in diagnostics:
            metric_values["worst_max_r_hat"][q_index, :] = np.nanmax(
                diagnostics["max_r_hat_curve_matrix"],
                axis=0,
            )
        if "min_effective_sample_size_curve_matrix" in diagnostics:
            metric_values["worst_min_effective_sample_size"][q_index, :] = np.nanmin(
                diagnostics["min_effective_sample_size_curve_matrix"],
                axis=0,
            )
        if "mean_pt_min_swap_acceptance_rate_curve_matrix" in diagnostics:
            metric_values["worst_mean_pt_min_swap_acceptance_rate"][q_index, :] = (
                np.nanmin(
                    diagnostics["mean_pt_min_swap_acceptance_rate_curve_matrix"],
                    axis=0,
                )
            )
        if "converged_mask_matrix" in diagnostics:
            metric_values["converged_fraction"][q_index, :] = np.mean(
                diagnostics["converged_mask_matrix"],
                axis=0,
            )
        if "q0_mean_q_top_spread_curve_matrix" in diagnostics:
            q0_spread[q_index, :] = np.max(
                diagnostics["q0_mean_q_top_spread_curve_matrix"],
                axis=0,
            )

    return {
        "metric_values": metric_values,
        "q0_worst_mean_q_top_spread": q0_spread,
        "num_q_positive_points": int(
            np.count_nonzero(~np.isnan(metric_values["converged_fraction"]))
        ),
        "num_q_positive_passed_lattice_points": int(
            np.nansum(metric_values["converged_fraction"] * 3.0)
        ),
    }


def _plot_boundary(summary, output_path):
    p_values = summary["p_values"]
    q_values = summary["q_values"]
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    pair_markers = {
        f"L{int(summary['lattice_sizes'][0])}-L{int(summary['lattice_sizes'][1])}": "o",
        f"L{int(summary['lattice_sizes'][1])}-L{int(summary['lattice_sizes'][2])}": "s",
    }

    for pair_label, marker in pair_markers.items():
        stable_xs = []
        stable_ys = []
        stable_yerr_low = []
        stable_yerr_high = []
        noisy_xs = []
        noisy_ys = []
        lower_bound_xs = []
        lower_bound_ys = []
        for p_summary in summary["pair_summaries"]:
            pair_record = next(
                record for record in p_summary["pairs"]
                if record["pair"] == pair_label
            )
            primary = pair_record["primary_crossing"]
            if primary is None:
                if (
                        pair_record["classification"]
                        == "below_threshold_through_q_window"):
                    lower_bound_xs.append(p_summary["p"])
                    lower_bound_ys.append(float(np.max(q_values)))
                continue
            estimate = float(primary["crossing_estimate_q"])
            if pair_record["classification"] == "single_crossing":
                stable_xs.append(p_summary["p"])
                stable_ys.append(estimate)
                stable_yerr_low.append(
                    max(0.0, estimate - float(primary["left_q"]))
                )
                stable_yerr_high.append(
                    max(0.0, float(primary["right_q"]) - estimate)
                )
            else:
                noisy_xs.append(p_summary["p"])
                noisy_ys.append(estimate)
        if stable_xs:
            axis.errorbar(
                stable_xs,
                stable_ys,
                yerr=np.vstack([stable_yerr_low, stable_yerr_high]),
                marker=marker,
                linewidth=1.4,
                capsize=3,
                label=f"{pair_label} single crossing",
            )
        if noisy_xs:
            axis.scatter(
                noisy_xs,
                noisy_ys,
                marker="x",
                s=64,
                linewidths=1.6,
                label=f"{pair_label} nonmonotonic crossing",
            )
        if lower_bound_xs:
            axis.scatter(
                lower_bound_xs,
                lower_bound_ys,
                marker="^",
                s=58,
                facecolors="none",
                linewidths=1.4,
                label=f"{pair_label} no crossing below q_max",
            )

    common_p = []
    common_low = []
    common_high = []
    common_rep = []
    for p_summary in summary["pair_summaries"]:
        common = p_summary["common_crossing_window"]
        if common is None:
            continue
        common_p.append(common["p"])
        common_low.append(common["q_min"])
        common_high.append(common["q_max"])
        common_rep.append(common["representative_q"])
    if common_p:
        axis.fill_between(
            common_p,
            common_low,
            common_high,
            alpha=0.18,
            color="C2",
            label="two-pair crossing span",
        )
        axis.plot(
            common_p,
            common_rep,
            color="C2",
            marker="^",
            linewidth=1.6,
            label="two-pair representative",
        )

    axis.set_xlim(float(np.min(p_values)) - 0.005, float(np.max(p_values)) + 0.005)
    axis.set_ylim(float(np.min(q_values)) - 0.002, float(np.max(q_values)) + 0.004)
    axis.set_xlabel("data error probability p")
    axis.set_ylabel("measurement error probability q")
    axis.set_title("3D toric code with measurement noise: pairwise threshold boundary")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_q_scan(summary, output_path):
    p_values = summary["p_values"]
    q_values = summary["q_values"]
    lattice_sizes = summary["lattice_sizes"]
    q_top = summary["q_top"]
    q_top_ci95 = summary["q_top_ci95"]
    num_cols = 3
    num_rows = int(math.ceil(len(p_values) / num_cols))
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12.0, 3.6 * num_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for p_index, p_value in enumerate(p_values):
        axis = axes[p_index]
        for lattice_index, lattice_size in enumerate(lattice_sizes):
            axis.errorbar(
                q_values,
                q_top[lattice_index, p_index, :],
                yerr=q_top_ci95[lattice_index, p_index, :],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=f"L={int(lattice_size)}",
            )
        axis.set_title(f"p={p_value:0.4f}")
        axis.set_xlabel("measurement error probability q")
        axis.set_ylabel("q_top")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[len(p_values):]:
        axis.axis("off")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_gap_scan(summary, output_path):
    p_values = summary["p_values"]
    q_values = summary["q_values"]
    num_cols = 3
    num_rows = int(math.ceil(len(p_values) / num_cols))
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12.0, 3.6 * num_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    pair_colors = ["C0", "C1"]
    for p_index, p_value in enumerate(p_values):
        axis = axes[p_index]
        p_summary = summary["pair_summaries"][p_index]
        for pair_index, pair_record in enumerate(p_summary["pairs"]):
            gap = np.asarray(pair_record["gap_values"], dtype=np.float64)
            ci95 = np.asarray(pair_record["gap_ci95"], dtype=np.float64)
            color = pair_colors[pair_index]
            axis.plot(
                q_values,
                gap,
                marker="o",
                linewidth=1.2,
                color=color,
                label=pair_record["pair"],
            )
            axis.fill_between(
                q_values,
                gap - ci95,
                gap + ci95,
                alpha=0.16,
                color=color,
            )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(f"p={p_value:0.4f}")
        axis.set_xlabel("measurement error probability q")
        axis.set_ylabel("pairwise gap")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[len(p_values):]:
        axis.axis("off")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_diagnostics(summary, output_path):
    p_values = summary["p_values"]
    q_values = summary["q_values"]
    diagnostics = summary["diagnostics"]["metric_values"]
    metric_plot_specs = [
        ("worst_mean_q_top_spread", "max_L mean q_top spread"),
        ("worst_max_r_hat", "max_L R-hat"),
        ("worst_min_effective_sample_size", "min_L ESS"),
        ("worst_mean_pt_min_swap_acceptance_rate", "min_L PT min swap"),
        ("converged_fraction", "converged fraction over L"),
    ]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.6),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    extent = [
        float(np.min(p_values)),
        float(np.max(p_values)),
        float(np.min(q_values)),
        float(np.max(q_values)),
    ]
    for axis, (metric_key, title) in zip(axes, metric_plot_specs):
        values = np.asarray(diagnostics[metric_key], dtype=np.float64)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("p")
        axis.set_ylabel("q")
        figure.colorbar(image, ax=axis, shrink=0.85)
    axes[-1].axis("off")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _write_summary_outputs(summary, output_dir, output_stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{output_stem}_boundary_summary.json"
    csv_path = output_dir / f"{output_stem}_boundary_points.csv"

    serializable_summary = {
        "p_values": summary["p_values"],
        "q_values": summary["q_values"],
        "lattice_sizes": summary["lattice_sizes"],
        "num_disorder_samples": summary["num_disorder_samples"],
        "pair_summaries": summary["pair_summaries"],
        "diagnostics": {
            "num_q_positive_points": summary["diagnostics"][
                "num_q_positive_points"
            ],
            "num_q_positive_passed_lattice_points": summary["diagnostics"][
                "num_q_positive_passed_lattice_points"
            ],
            "q_positive_lattice_point_pass_fraction": (
                summary["diagnostics"][
                    "num_q_positive_passed_lattice_points"
                ]
                / max(1, 3 * 10 * len(summary["p_values"]))
            ),
        },
        "interpretation": {
            "threshold_direction": (
                "below threshold: larger L has larger q_top, so pairwise "
                "gap q_top(L_small)-q_top(L_large) is negative; above "
                "threshold the gap becomes positive"
            ),
            "main_caveat": (
                "Adjacent-pair crossings are finite-size estimates. A clean "
                "boundary requires L3-L4 and L4-L5 crossings to agree within "
                "the scan resolution and pass diagnostics."
            ),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            serializable_summary,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "p",
            "pair",
            "classification",
            "primary_crossing_q",
            "primary_crossing_q_left",
            "primary_crossing_q_right",
            "num_crossings",
            "min_abs_gap_q",
            "min_abs_gap",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in summary["csv_rows"]:
            writer.writerow(row)

    plot_paths = {
        "boundary_plot": output_dir / f"{output_stem}_pq_boundary.png",
        "q_scan_plot": output_dir / f"{output_stem}_q_scan_sem95.png",
        "gap_scan_plot": output_dir / f"{output_stem}_q_gap_scan.png",
        "diagnostic_plot": output_dir / f"{output_stem}_diagnostic_heatmaps.png",
    }
    _plot_boundary(summary, plot_paths["boundary_plot"])
    _plot_q_scan(summary, plot_paths["q_scan_plot"])
    _plot_gap_scan(summary, plot_paths["gap_scan_plot"])
    _plot_diagnostics(summary, plot_paths["diagnostic_plot"])

    return {
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        **{key: str(value) for key, value in plot_paths.items()},
    }


def analyze_pq_threshold_boundary(run_root, output_dir, output_stem):
    items = _load_q_runs(run_root)
    summary = _build_summary(items)
    return _write_summary_outputs(
        summary=summary,
        output_dir=output_dir,
        output_stem=output_stem,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze fixed-p q-scans for a p-q threshold boundary in 3D "
            "toric code runs."
        )
    )
    parser.add_argument("run_root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="pq_threshold_boundary")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    outputs = analyze_pq_threshold_boundary(
        run_root=args.run_root,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
