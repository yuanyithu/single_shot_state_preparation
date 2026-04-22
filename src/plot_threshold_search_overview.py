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
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter


CI95_Z_SCORE = 1.96


def _load_run_data(run_root):
    run_root = Path(run_root)
    result_list = []
    for summary_path in sorted(run_root.glob("q_*/threshold_summary.json")):
        q_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        npz_path = next(q_dir.glob("*.npz"))
        with np.load(npz_path, allow_pickle=True) as data:
            probability_list = np.asarray(
                data["data_error_probability_list"],
                dtype=np.float64,
            )
            lattice_size_list = np.asarray(
                data["lattice_size_list"],
                dtype=np.int64,
            )
            q_top_curve_matrix = np.asarray(
                data["q_top_curve_matrix"],
                dtype=np.float64,
            )
            q_top_std_curve_matrix = np.asarray(
                data["q_top_std_curve_matrix"],
                dtype=np.float64,
            )
            num_disorder_samples = int(data["num_disorder_samples"])
            syndrome_error_probability = float(
                data["syndrome_error_probability"]
            )
        q_top_ci95_curve_matrix = (
            CI95_Z_SCORE
            * q_top_std_curve_matrix
            / math.sqrt(float(num_disorder_samples))
        )
        result_list.append({
            "summary": summary,
            "q_dir": q_dir,
            "syndrome_error_probability": syndrome_error_probability,
            "probability_list": probability_list,
            "lattice_size_list": lattice_size_list,
            "q_top_curve_matrix": q_top_curve_matrix,
            "q_top_ci95_curve_matrix": q_top_ci95_curve_matrix,
        })
    return result_list


def _plot_sem95_overview(result_list, output_path):
    figure, axes = plt.subplots(
        len(result_list),
        1,
        figsize=(8.0, 3.4 * len(result_list)),
        constrained_layout=True,
    )
    if len(result_list) == 1:
        axes = np.array([axes], dtype=object)

    for axis, item in zip(axes, result_list):
        probability_list = item["probability_list"]
        lattice_size_list = item["lattice_size_list"]
        q_top_curve_matrix = item["q_top_curve_matrix"]
        q_top_ci95_curve_matrix = item["q_top_ci95_curve_matrix"]
        q_value = item["syndrome_error_probability"]
        plotted_group_list = []
        for lattice_index, lattice_size in enumerate(lattice_size_list):
            curve = q_top_curve_matrix[lattice_index]
            ci95_curve = q_top_ci95_curve_matrix[lattice_index]
            matching_group = None
            for group in plotted_group_list:
                if np.array_equal(curve, group["curve"]):
                    matching_group = group
                    break
            if matching_group is None:
                matching_group = {
                    "curve": curve,
                    "ci95_curve_list": [ci95_curve],
                    "lattice_size_list": [int(lattice_size)],
                }
                plotted_group_list.append(matching_group)
            else:
                matching_group["ci95_curve_list"].append(ci95_curve)
                matching_group["lattice_size_list"].append(int(lattice_size))

        for group_index, group in enumerate(plotted_group_list):
            size_label = ",".join(str(size) for size in group["lattice_size_list"])
            if len(group["lattice_size_list"]) > 1:
                curve_label = f"L={size_label} (same mean)"
            else:
                curve_label = f"L={size_label}"
            ci95_envelope = np.max(
                np.vstack(group["ci95_curve_list"]),
                axis=0,
            )
            axis.errorbar(
                probability_list,
                group["curve"],
                yerr=ci95_envelope,
                marker="o",
                linewidth=1.3,
                capsize=2.5,
                label=curve_label,
                color=f"C{group_index}",
            )
        y_min = float(np.min(q_top_curve_matrix - q_top_ci95_curve_matrix))
        y_max = float(np.max(q_top_curve_matrix + q_top_ci95_curve_matrix))
        y_range = y_max - y_min
        if y_max <= 1.001 and y_min >= 0.97 and y_range < 0.02:
            y_padding = max(5e-4, 0.15 * max(y_range, 1e-6))
            axis.set_ylim(max(0.0, y_min - y_padding), min(1.001, y_max + y_padding))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        else:
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.get_offset_text().set_visible(False)
        axis.set_title(
            "Threshold search overview: "
            f"measurement error q={q_value:0.4f}, "
            f"Pauli error p in [{probability_list[0]:0.4f}, "
            f"{probability_list[-1]:0.4f}]"
        )
        axis.set_xlabel("data error probability p")
        axis.set_ylabel("q_top")
        axis.grid(True, alpha=0.3)
        axis.legend(title="Linear size L, error bar: 95% CI")

    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_gap_summary(result_list, output_path):
    lattice_size_list = result_list[0]["lattice_size_list"]
    lower_pair_label = f"L{int(lattice_size_list[0])}-L{int(lattice_size_list[1])}"
    upper_pair_label = f"L{int(lattice_size_list[1])}-L{int(lattice_size_list[2])}"
    q_value_list = np.array(
        [item["syndrome_error_probability"] for item in result_list],
        dtype=np.float64,
    )
    delta_35_min_abs_list = np.array([
        abs(
            item["summary"]["pairwise_gap_analysis"]["delta_35"][
                "min_abs_delta_value"
            ]
        )
        for item in result_list
    ], dtype=np.float64)
    delta_57_min_abs_list = np.array([
        abs(
            item["summary"]["pairwise_gap_analysis"]["delta_57"][
                "min_abs_delta_value"
            ]
        )
        for item in result_list
    ], dtype=np.float64)
    delta_35_min_abs_p_list = np.array([
        item["summary"]["pairwise_gap_analysis"]["delta_35"]["min_abs_delta_p"]
        for item in result_list
    ], dtype=np.float64)
    delta_57_min_abs_p_list = np.array([
        item["summary"]["pairwise_gap_analysis"]["delta_57"]["min_abs_delta_p"]
        for item in result_list
    ], dtype=np.float64)
    recommended_p_min_list = np.array([
        (
            item["summary"]["recommended_server_window"]["p_min"]
            if item["summary"]["recommended_server_window"] is not None
            else np.nan
        )
        for item in result_list
    ], dtype=np.float64)
    recommended_p_max_list = np.array([
        (
            item["summary"]["recommended_server_window"]["p_max"]
            if item["summary"]["recommended_server_window"] is not None
            else np.nan
        )
        for item in result_list
    ], dtype=np.float64)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        constrained_layout=True,
    )
    top_axis = axes[0]
    bottom_axis = axes[1]

    top_axis.plot(
        q_value_list,
        delta_35_min_abs_list,
        marker="o",
        linewidth=1.5,
        label=f"min |{lower_pair_label}|",
    )
    top_axis.plot(
        q_value_list,
        delta_57_min_abs_list,
        marker="o",
        linewidth=1.5,
        label=f"min |{upper_pair_label}|",
    )
    top_axis.set_title(
        "Pairwise gap summary: "
        f"measurement error q in [{q_value_list[0]:0.4f}, "
        f"{q_value_list[-1]:0.4f}], Pauli error p windows vary by q"
    )
    top_axis.set_xlabel("measurement error probability q")
    top_axis.set_ylabel("min absolute pairwise gap")
    top_axis.grid(True, alpha=0.3)
    top_axis.legend()

    bottom_axis.plot(
        q_value_list,
        delta_35_min_abs_p_list,
        marker="o",
        linewidth=1.5,
        label=f"argmin |{lower_pair_label}|",
    )
    bottom_axis.plot(
        q_value_list,
        delta_57_min_abs_p_list,
        marker="o",
        linewidth=1.5,
        label=f"argmin |{upper_pair_label}|",
    )
    finite_window_mask = np.isfinite(recommended_p_min_list) & np.isfinite(
        recommended_p_max_list
    )
    if np.any(finite_window_mask):
        bottom_axis.fill_between(
            q_value_list,
            recommended_p_min_list,
            recommended_p_max_list,
            where=finite_window_mask,
            alpha=0.2,
            label="recommended next p window",
        )
    bottom_axis.set_title(
        "Recommended next windows: "
        f"measurement error q in [{q_value_list[0]:0.4f}, "
        f"{q_value_list[-1]:0.4f}], Pauli error p shown on y-axis"
    )
    bottom_axis.set_xlabel("measurement error probability q")
    bottom_axis.set_ylabel("Pauli error probability p")
    bottom_axis.grid(True, alpha=0.3)
    bottom_axis.legend()

    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_threshold_search_overview(run_root):
    run_root = Path(run_root)
    result_list = _load_run_data(run_root)
    sem95_output_path = (
        run_root / "measurement_noise_threshold_search_sem95_overview.png"
    )
    gap_output_path = (
        run_root / "measurement_noise_threshold_search_gap_summary.png"
    )
    _plot_sem95_overview(result_list, sem95_output_path)
    _plot_gap_summary(result_list, gap_output_path)
    return sem95_output_path, gap_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot overview figures for a threshold-search run root."
    )
    parser.add_argument(
        "run_root",
        help="Path to a threshold-search run root under data/2d_toric_code/with_measurement_noise/",
    )
    args = parser.parse_args()

    sem95_output_path, gap_output_path = plot_threshold_search_overview(
        args.run_root
    )
    print(f"Saved overview to {sem95_output_path}")
    print(f"Saved gap summary to {gap_output_path}")


if __name__ == "__main__":
    main()
