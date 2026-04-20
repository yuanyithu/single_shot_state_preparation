import argparse
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


def _build_output_path(input_path, output_path):
    if output_path is not None:
        return Path(output_path)
    return Path(input_path).with_suffix(".png")


def _plot_single_size_result(loaded_result, figure):
    data_error_probability_list = loaded_result["data_error_probability_list"]
    q_top_curve = loaded_result["q_top_curve"]
    q_top_std_error_curve = loaded_result["q_top_std_error_curve"]
    average_acceptance_rate_curve = loaded_result[
        "average_acceptance_rate_curve"
    ]
    lattice_size = int(loaded_result["lattice_size"])

    axes = figure.subplots(2, 1, sharex=True)
    axes[0].errorbar(
        data_error_probability_list,
        q_top_curve,
        yerr=q_top_std_error_curve,
        marker="o",
        linewidth=1.5,
        capsize=3.0,
        label=f"L={lattice_size}",
    )
    axes[0].set_ylabel("q_top")
    axes[0].set_title("Toric code scan result")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        data_error_probability_list,
        average_acceptance_rate_curve,
        marker="o",
        linewidth=1.5,
        label=f"L={lattice_size}",
    )
    axes[1].set_xlabel("data error probability p")
    axes[1].set_ylabel("acceptance rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return axes


def _plot_multi_size_result(loaded_result, figure):
    data_error_probability_list = loaded_result["data_error_probability_list"]
    lattice_size_list = loaded_result["lattice_size_list"]
    q_top_curve_matrix = loaded_result["q_top_curve_matrix"]
    q_top_std_error_curve_matrix = loaded_result["q_top_std_error_curve_matrix"]
    average_acceptance_rate_curve_matrix = loaded_result[
        "average_acceptance_rate_curve_matrix"
    ]

    axes = figure.subplots(2, 1, sharex=True)

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        label = f"L={int(lattice_size)}"
        axes[0].errorbar(
            data_error_probability_list,
            q_top_curve_matrix[lattice_index],
            yerr=q_top_std_error_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=label,
        )
        axes[1].plot(
            data_error_probability_list,
            average_acceptance_rate_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            label=label,
        )

    axes[0].set_ylabel("q_top")
    axes[0].set_title("Toric code multi-size scan result")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("data error probability p")
    axes[1].set_ylabel("acceptance rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return axes


def plot_scan_result(input_path, output_path=None):
    output_path = _build_output_path(input_path, output_path)

    with np.load(input_path, allow_pickle=True) as loaded_result:
        figure = plt.figure(figsize=(8.0, 8.0), constrained_layout=True)

        if "q_top_curve_matrix" in loaded_result.files:
            _plot_multi_size_result(loaded_result, figure)
        elif "q_top_curve" in loaded_result.files:
            _plot_single_size_result(loaded_result, figure)
        else:
            raise ValueError(
                "Unsupported result format: missing q_top curve fields."
            )

        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot toric code scan results from an NPZ file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="scan_result_multi_L.npz",
        help="Path to scan_result.npz or scan_result_multi_L.npz",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to replacing .npz with .png",
    )
    args = parser.parse_args()

    output_path = plot_scan_result(
        input_path=args.input_path,
        output_path=args.output,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
