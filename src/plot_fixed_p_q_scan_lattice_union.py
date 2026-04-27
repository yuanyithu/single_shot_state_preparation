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


def _load_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _as_float(value):
    return float(np.asarray(value).item())


def _sem95_from_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(CI95_Z_SCORE * np.std(values, ddof=1) / math.sqrt(values.size))


def _sem95_from_std(std_value, num_samples):
    if num_samples <= 1:
        return 0.0
    return float(CI95_Z_SCORE * float(std_value) / math.sqrt(float(num_samples)))


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


def _parse_runs(input_paths, fixed_p, p_tolerance):
    by_q = {}
    for input_path in input_paths:
        path = Path(input_path).resolve()
        result = _load_npz(path)
        p_values = np.asarray(result["data_error_probability_list"], dtype=np.float64)
        if p_values.size != 1:
            raise ValueError(f"{path} must contain exactly one p value")
        p_value = float(p_values[0])
        if abs(p_value - fixed_p) > p_tolerance:
            raise ValueError(f"{path} has p={p_value}, expected {fixed_p}")
        q_value = _as_float(result["syndrome_error_probability"])
        by_q.setdefault(q_value, []).append((path, result))
    if not by_q:
        raise ValueError("at least one input NPZ is required")
    return by_q


def _build_scan(input_paths, fixed_p, p_tolerance):
    by_q = _parse_runs(input_paths, fixed_p, p_tolerance)
    q_values = np.asarray(sorted(by_q), dtype=np.float64)
    lattice_sizes = sorted(
        {
            int(lattice_size)
            for runs in by_q.values()
            for _, result in runs
            for lattice_size in np.asarray(result["lattice_size_list"], dtype=np.int64)
        }
    )
    lattice_to_index = {lattice_size: index for index, lattice_size in enumerate(lattice_sizes)}

    q_top_matrix = np.full((len(lattice_sizes), len(q_values)), np.nan, dtype=np.float64)
    q_top_ci95_matrix = np.full_like(q_top_matrix, np.nan)
    num_disorder_matrix = np.zeros_like(q_top_matrix, dtype=np.int64)
    source_paths = {}
    source_group = {}
    disorder_values = {}

    for q_index, q_value in enumerate(q_values):
        for group_index, (path, result) in enumerate(by_q[float(q_value)]):
            result_lattices = np.asarray(result["lattice_size_list"], dtype=np.int64)
            q_top_curve = np.asarray(result["q_top_curve_matrix"], dtype=np.float64)
            q_top_std = np.asarray(result["q_top_std_curve_matrix"], dtype=np.float64)
            disorder_tensor = np.asarray(
                result["disorder_q_top_values_tensor"],
                dtype=np.float64,
            )
            num_disorder = int(result["num_disorder_samples"])
            for local_index, lattice_size in enumerate(result_lattices):
                global_index = lattice_to_index[int(lattice_size)]
                if np.isfinite(q_top_matrix[global_index, q_index]):
                    raise ValueError(
                        f"duplicate L={int(lattice_size)} at q={float(q_value)}"
                    )
                q_top_matrix[global_index, q_index] = float(q_top_curve[local_index, 0])
                q_top_ci95_matrix[global_index, q_index] = _sem95_from_std(
                    q_top_std[local_index, 0],
                    num_disorder,
                )
                num_disorder_matrix[global_index, q_index] = num_disorder
                source_paths[(global_index, q_index)] = str(path)
                source_group[(global_index, q_index)] = group_index
                disorder_values[(global_index, q_index)] = disorder_tensor[
                    local_index,
                    0,
                    :,
                ]

    if np.any(~np.isfinite(q_top_matrix)):
        missing = []
        for lattice_index, lattice_size in enumerate(lattice_sizes):
            for q_index, q_value in enumerate(q_values):
                if not np.isfinite(q_top_matrix[lattice_index, q_index]):
                    missing.append({"L": int(lattice_size), "q": float(q_value)})
        raise ValueError(f"missing lattice/q points: {missing}")

    return {
        "fixed_p": float(fixed_p),
        "q_values": q_values,
        "lattice_size_list": np.asarray(lattice_sizes, dtype=np.int64),
        "q_top_matrix": q_top_matrix,
        "q_top_ci95_matrix": q_top_ci95_matrix,
        "num_disorder_matrix": num_disorder_matrix,
        "source_paths": source_paths,
        "source_group": source_group,
        "disorder_values": disorder_values,
    }


def _compute_gap_rows(scan):
    q_values = scan["q_values"]
    lattice_sizes = scan["lattice_size_list"]
    gap_rows = []
    for lattice_index in range(len(lattice_sizes) - 1):
        values = []
        ci95 = []
        methods = []
        for q_index, _q_value in enumerate(q_values):
            small_values = scan["disorder_values"][(lattice_index, q_index)]
            large_values = scan["disorder_values"][(lattice_index + 1, q_index)]
            same_source = (
                scan["source_group"][(lattice_index, q_index)]
                == scan["source_group"][(lattice_index + 1, q_index)]
            )
            if same_source and small_values.size == large_values.size:
                gap_samples = small_values - large_values
                gap_mean = float(np.mean(gap_samples))
                gap_ci95 = _sem95_from_values(gap_samples)
                method = "paired_disorder"
            else:
                small_mean = float(np.mean(small_values))
                large_mean = float(np.mean(large_values))
                gap_mean = small_mean - large_mean
                small_se = (
                    0.0
                    if small_values.size <= 1
                    else np.std(small_values, ddof=1) / math.sqrt(small_values.size)
                )
                large_se = (
                    0.0
                    if large_values.size <= 1
                    else np.std(large_values, ddof=1) / math.sqrt(large_values.size)
                )
                gap_ci95 = float(CI95_Z_SCORE * math.sqrt(small_se**2 + large_se**2))
                method = "independent_disorder"
            values.append(gap_mean)
            ci95.append(gap_ci95)
            methods.append(method)
        gap_rows.append(
            {
                "label": f"L{int(lattice_sizes[lattice_index])}-L{int(lattice_sizes[lattice_index + 1])}",
                "values": np.asarray(values, dtype=np.float64),
                "ci95": np.asarray(ci95, dtype=np.float64),
                "ci95_methods": methods,
            }
        )
    return gap_rows


def _plot_q_top(scan, output_path):
    figure, axis = plt.subplots(1, 1, figsize=(8.4, 5.0), constrained_layout=True)
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
    figure, axis = plt.subplots(1, 1, figsize=(8.4, 5.0), constrained_layout=True)
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


def _write_summary(scan, gap_rows, output_path):
    q_values = scan["q_values"]
    summary = {
        "fixed_p": scan["fixed_p"],
        "q_values": q_values.tolist(),
        "lattice_size_list": scan["lattice_size_list"].astype(int).tolist(),
        "q_top_matrix": scan["q_top_matrix"].tolist(),
        "q_top_ci95_matrix": scan["q_top_ci95_matrix"].tolist(),
        "num_disorder_matrix": scan["num_disorder_matrix"].astype(int).tolist(),
        "source_paths_by_lattice_and_q": [
            [
                scan["source_paths"][(lattice_index, q_index)]
                for q_index in range(len(q_values))
            ]
            for lattice_index in range(len(scan["lattice_size_list"]))
        ],
        "gap_rows": [],
    }
    for gap_row in gap_rows:
        summary["gap_rows"].append(
            {
                "label": gap_row["label"],
                "values": gap_row["values"].tolist(),
                "ci95": gap_row["ci95"].tolist(),
                "ci95_methods": list(gap_row["ci95_methods"]),
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


def plot_fixed_p_q_scan_lattice_union(
    input_paths,
    output_dir,
    output_stem,
    fixed_p,
    p_tolerance,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = _build_scan(
        input_paths=input_paths,
        fixed_p=fixed_p,
        p_tolerance=p_tolerance,
    )
    gap_rows = _compute_gap_rows(scan)
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
        "lattice_size_list": summary["lattice_size_list"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a fixed-p q scan from NPZ files whose lattice-size sets can "
            "come from separate independent runs."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input pooled NPZ. Repeat for every q/source group.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="fixed_p_q_scan_lattice_union")
    parser.add_argument("--fixed-p", type=float, required=True)
    parser.add_argument("--p-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    result = plot_fixed_p_q_scan_lattice_union(
        input_paths=args.input,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        fixed_p=args.fixed_p,
        p_tolerance=args.p_tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
