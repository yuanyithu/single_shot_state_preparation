import argparse
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
DISORDER_AXIS = 2


def _as_float(value):
    return float(np.asarray(value).item())


def _as_bool(value):
    return bool(np.asarray(value).item())


def _load_run(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _format_q_tag(q_value):
    return f"q{q_value:0.4f}".replace(".", "p")


def _sem95(std_values, num_samples):
    if num_samples <= 0:
        return np.zeros_like(std_values, dtype=np.float64)
    return CI95_Z_SCORE * np.asarray(std_values, dtype=np.float64) / math.sqrt(
        float(num_samples)
    )


def _linear_crossing(x0, y0, x1, y1):
    if y0 == y1:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _find_crossing_windows(x_values, y_values):
    windows = []
    for index in range(len(x_values) - 1):
        x0 = float(x_values[index])
        x1 = float(x_values[index + 1])
        y0 = float(y_values[index])
        y1 = float(y_values[index + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            windows.append(
                {
                    "left": x0,
                    "right": x0,
                    "estimate": x0,
                    "left_value": y0,
                    "right_value": y0,
                }
            )
        elif y0 * y1 < 0.0:
            windows.append(
                {
                    "left": x0,
                    "right": x1,
                    "estimate": _linear_crossing(x0, y0, x1, y1),
                    "left_value": y0,
                    "right_value": y1,
                }
            )
    if len(x_values) and float(y_values[-1]) == 0.0:
        x_last = float(x_values[-1])
        windows.append(
            {
                "left": x_last,
                "right": x_last,
                "estimate": x_last,
                "left_value": 0.0,
                "right_value": 0.0,
            }
        )
    return windows


def _load_fixed_p_q_scan(input_paths, fixed_p, p_tolerance):
    runs = []
    for path in input_paths:
        result = _load_run(path)
        p_values = np.asarray(result["data_error_probability_list"], dtype=np.float64)
        if p_values.size != 1:
            raise ValueError(f"{path} must contain exactly one p value")
        p_value = float(p_values[0])
        if abs(p_value - fixed_p) > p_tolerance:
            raise ValueError(f"{path} has p={p_value}, expected {fixed_p}")
        q_value = _as_float(result["syndrome_error_probability"])
        runs.append((q_value, Path(path), result))

    runs.sort(key=lambda item: item[0])
    if not runs:
        raise ValueError("at least one input NPZ is required")

    lattice_size_list = np.asarray(runs[0][2]["lattice_size_list"], dtype=np.int64)
    for q_value, path, result in runs[1:]:
        if not np.array_equal(
            lattice_size_list,
            np.asarray(result["lattice_size_list"], dtype=np.int64),
        ):
            raise ValueError(f"lattice_size_list differs in {path}")

    q_values = np.asarray([item[0] for item in runs], dtype=np.float64)
    q_top_matrix = np.stack(
        [
            np.asarray(result["q_top_curve_matrix"], dtype=np.float64)[:, 0]
            for _, _, result in runs
        ],
        axis=1,
    )
    q_top_ci95_matrix = np.stack(
        [
            _sem95(
                np.asarray(result["q_top_std_curve_matrix"], dtype=np.float64)[:, 0],
                int(result["num_disorder_samples"]),
            )
            for _, _, result in runs
        ],
        axis=1,
    )
    disorder_values = [
        np.asarray(result["disorder_q_top_values_tensor"], dtype=np.float64)[:, 0, :]
        for _, _, result in runs
    ]
    num_disorder_samples = np.asarray(
        [int(result["num_disorder_samples"]) for _, _, result in runs],
        dtype=np.int64,
    )
    return {
        "fixed_p": float(fixed_p),
        "q_values": q_values,
        "lattice_size_list": lattice_size_list,
        "q_top_matrix": q_top_matrix,
        "q_top_ci95_matrix": q_top_ci95_matrix,
        "disorder_values": disorder_values,
        "num_disorder_samples": num_disorder_samples,
        "runs": runs,
    }


def _compute_gap_series(scan):
    lattice_sizes = scan["lattice_size_list"]
    disorder_values = scan["disorder_values"]
    gap_rows = []
    for lattice_index in range(len(lattice_sizes) - 1):
        gap_values = []
        gap_ci95 = []
        for values in disorder_values:
            per_disorder_gap = values[lattice_index] - values[lattice_index + 1]
            gap_values.append(float(np.mean(per_disorder_gap)))
            if per_disorder_gap.size <= 1:
                gap_ci95.append(0.0)
            else:
                gap_ci95.append(
                    float(
                        _sem95(
                            np.std(per_disorder_gap, ddof=1),
                            per_disorder_gap.size,
                        )
                    )
                )
        gap_rows.append(
            {
                "label": f"L{int(lattice_sizes[lattice_index])}-L{int(lattice_sizes[lattice_index + 1])}",
                "values": np.asarray(gap_values, dtype=np.float64),
                "ci95": np.asarray(gap_ci95, dtype=np.float64),
            }
        )
    return gap_rows


def _summarize_diagnostics(scan):
    diagnostics = []
    for q_value, path, result in scan["runs"]:
        item = {
            "q": float(q_value),
            "source_npz": str(path),
            "num_disorder_samples": int(result["num_disorder_samples"]),
            "pt_enabled": _as_bool(result.get("pt_enabled", np.array(False))),
        }
        if "converged_mask_matrix" in result:
            mask = np.asarray(result["converged_mask_matrix"], dtype=bool)
            item["num_converged_points"] = int(np.count_nonzero(mask))
            item["num_total_points"] = int(mask.size)
        if "mean_q_top_spread_curve_matrix" in result:
            item["max_mean_q_top_spread"] = float(
                np.nanmax(result["mean_q_top_spread_curve_matrix"])
            )
        if "max_r_hat_curve_matrix" in result:
            finite_values = np.asarray(
                result["max_r_hat_curve_matrix"],
                dtype=np.float64,
            )
            finite_values = finite_values[np.isfinite(finite_values)]
            item["max_r_hat"] = (
                None if finite_values.size == 0 else float(np.max(finite_values))
            )
        if "min_effective_sample_size_curve_matrix" in result:
            item["min_effective_sample_size"] = float(
                np.nanmin(result["min_effective_sample_size_curve_matrix"])
            )
        if "mean_pt_min_swap_acceptance_rate_curve_matrix" in result:
            item["mean_pt_min_swap_acceptance_rate"] = float(
                np.nanmean(result["mean_pt_min_swap_acceptance_rate_curve_matrix"])
            )
        diagnostics.append(item)
    return diagnostics


def _write_summary(scan, gap_rows, output_path):
    q_values = scan["q_values"]
    summary = {
        "fixed_p": scan["fixed_p"],
        "q_values": q_values.tolist(),
        "lattice_size_list": scan["lattice_size_list"].astype(int).tolist(),
        "num_disorder_samples_by_q": scan["num_disorder_samples"].astype(int).tolist(),
        "q_top_matrix": scan["q_top_matrix"].tolist(),
        "q_top_ci95_matrix": scan["q_top_ci95_matrix"].tolist(),
        "gap_rows": [],
        "diagnostics": _summarize_diagnostics(scan),
    }
    for gap_row in gap_rows:
        summary["gap_rows"].append(
            {
                "label": gap_row["label"],
                "values": gap_row["values"].tolist(),
                "ci95": gap_row["ci95"].tolist(),
                "crossing_windows": _find_crossing_windows(
                    q_values,
                    gap_row["values"],
                ),
            }
        )

    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _plot_q_top(scan, output_path):
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(8.0, 4.8),
        constrained_layout=True,
    )
    q_values = scan["q_values"]
    for lattice_index, lattice_size in enumerate(scan["lattice_size_list"]):
        axis.errorbar(
            q_values,
            scan["q_top_matrix"][lattice_index],
            yerr=scan["q_top_ci95_matrix"][lattice_index],
            marker="o",
            linewidth=1.6,
            capsize=3.0,
            label=f"L={int(lattice_size)}",
        )
    axis.set_xlabel("syndrome error probability q")
    axis.set_ylabel("q_top")
    axis.set_title(f"3D toric fixed p={scan['fixed_p']:0.4f}")
    axis.grid(True, alpha=0.3)
    axis.legend(title="95% CI of disorder mean")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_gaps(scan, gap_rows, output_path):
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(8.0, 4.8),
        constrained_layout=True,
    )
    q_values = scan["q_values"]
    for gap_row in gap_rows:
        axis.errorbar(
            q_values,
            gap_row["values"],
            yerr=gap_row["ci95"],
            marker="o",
            linewidth=1.6,
            capsize=3.0,
            label=gap_row["label"],
        )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.75)
    axis.set_xlabel("syndrome error probability q")
    axis.set_ylabel("q_top(L_small) - q_top(L_large)")
    axis.set_title(f"3D toric fixed p={scan['fixed_p']:0.4f} pairwise gaps")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Gap; negative means larger L is better")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_fixed_p_q_scan(input_paths, output_dir, output_stem, fixed_p, p_tolerance):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = _load_fixed_p_q_scan(
        input_paths=input_paths,
        fixed_p=fixed_p,
        p_tolerance=p_tolerance,
    )
    gap_rows = _compute_gap_series(scan)

    q_top_path = output_dir / f"{output_stem}_sem95.png"
    gap_path = output_dir / f"{output_stem}_gap_ci95.png"
    summary_path = output_dir / f"{output_stem}_summary.json"

    _plot_q_top(scan, q_top_path)
    _plot_gaps(scan, gap_rows, gap_path)
    summary = _write_summary(scan, gap_rows, summary_path)
    return {
        "q_top_plot_path": str(q_top_path),
        "gap_plot_path": str(gap_path),
        "summary_path": str(summary_path),
        "num_q_values": len(summary["q_values"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot a fixed-p 3D toric scan whose x-axis is syndrome q."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input pooled NPZ for one q. Repeat for every q value.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="fixed_p_q_scan")
    parser.add_argument("--fixed-p", type=float, required=True)
    parser.add_argument("--p-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    result = plot_fixed_p_q_scan(
        input_paths=args.input,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        fixed_p=args.fixed_p,
        p_tolerance=args.p_tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
