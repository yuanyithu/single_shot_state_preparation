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
        for lattice_index, lattice_size in enumerate(lattice_size_list):
            axis.errorbar(
                probability_list,
                q_top_curve_matrix[lattice_index],
                yerr=q_top_ci95_curve_matrix[lattice_index],
                marker="o",
                linewidth=1.3,
                capsize=2.5,
                label=f"L={int(lattice_size)}",
            )
        axis.set_title(
            "Threshold search overview: "
            f"measurement error q={q_value:0.4f}, "
            f"Pauli error p in [{probability_list[0]:0.4f}, "
            f"{probability_list[-1]:0.4f}]"
        )
        axis.set_xlabel("data error probability p")
        axis.set_ylabel("q_top")
        axis.grid(True, alpha=0.3)
        axis.legend(title="Error bar: 95% CI")

    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_gap_summary(result_list, output_path):
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
        item["summary"]["recommended_server_window"]["p_min"]
        for item in result_list
    ], dtype=np.float64)
    recommended_p_max_list = np.array([
        item["summary"]["recommended_server_window"]["p_max"]
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
        label="min |L3-L5|",
    )
    top_axis.plot(
        q_value_list,
        delta_57_min_abs_list,
        marker="o",
        linewidth=1.5,
        label="min |L5-L7|",
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
        label="argmin |L3-L5|",
    )
    bottom_axis.plot(
        q_value_list,
        delta_57_min_abs_p_list,
        marker="o",
        linewidth=1.5,
        label="argmin |L5-L7|",
    )
    bottom_axis.fill_between(
        q_value_list,
        recommended_p_min_list,
        recommended_p_max_list,
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
