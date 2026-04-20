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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "scan_result_multi_L_q0_geometric_multistart.npz"
)


def _build_output_path(input_path, output_path):
    if output_path is not None:
        return Path(output_path)
    return Path(input_path).with_suffix(".png")


def _load_q_top_std_curve(loaded_result):
    if "q_top_std_curve" in loaded_result.files:
        return loaded_result["q_top_std_curve"]
    return loaded_result["q_top_std_error_curve"]


def _load_q_top_std_curve_matrix(loaded_result):
    if "q_top_std_curve_matrix" in loaded_result.files:
        return loaded_result["q_top_std_curve_matrix"]
    return loaded_result["q_top_std_error_curve_matrix"]


def _has_q0_spread_diagnostics(loaded_result):
    return (
        "q0_mean_m_u_spread_linf_curve" in loaded_result.files
        or "q0_mean_m_u_spread_linf_curve_matrix" in loaded_result.files
    )


def _plot_single_size_result(loaded_result, axes):
    data_error_probability_list = loaded_result["data_error_probability_list"]
    q_top_curve = loaded_result["q_top_curve"]
    q_top_std_curve = _load_q_top_std_curve(loaded_result)
    lattice_size = int(loaded_result["lattice_size"])

    axis = axes[0]
    axis.errorbar(
        data_error_probability_list,
        q_top_curve,
        yerr=q_top_std_curve,
        marker="o",
        linewidth=1.5,
        capsize=3.0,
        label=f"L={lattice_size}",
    )
    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.set_title("Toric code scan result")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Error bar: disorder std dev")
    if len(axes) > 1:
        diagnostic_axis = axes[1]
        diagnostic_axis.plot(
            data_error_probability_list,
            loaded_result["q0_mean_m_u_spread_linf_curve"],
            marker="o",
            linewidth=1.5,
            label=f"L={lattice_size}",
        )
        diagnostic_axis.set_xlabel("data error probability p")
        diagnostic_axis.set_ylabel("q=0 mean start spread")
        diagnostic_axis.grid(True, alpha=0.3)
        diagnostic_axis.legend()
    return axes


def _plot_multi_size_result(loaded_result, axes):
    data_error_probability_list = loaded_result["data_error_probability_list"]
    lattice_size_list = loaded_result["lattice_size_list"]
    q_top_curve_matrix = loaded_result["q_top_curve_matrix"]
    q_top_std_curve_matrix = _load_q_top_std_curve_matrix(loaded_result)

    axis = axes[0]

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        label = f"L={int(lattice_size)}"
        axis.errorbar(
            data_error_probability_list,
            q_top_curve_matrix[lattice_index],
            yerr=q_top_std_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=label,
        )

    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.set_title("Toric code multi-size scan result")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Error bar: disorder std dev")
    if len(axes) > 1:
        diagnostic_axis = axes[1]
        q0_mean_m_u_spread_linf_curve_matrix = loaded_result[
            "q0_mean_m_u_spread_linf_curve_matrix"
        ]
        for lattice_index, lattice_size in enumerate(lattice_size_list):
            label = f"L={int(lattice_size)}"
            diagnostic_axis.plot(
                data_error_probability_list,
                q0_mean_m_u_spread_linf_curve_matrix[lattice_index],
                marker="o",
                linewidth=1.5,
                label=label,
            )
        diagnostic_axis.set_xlabel("data error probability p")
        diagnostic_axis.set_ylabel("q=0 mean start spread")
        diagnostic_axis.grid(True, alpha=0.3)
        diagnostic_axis.legend()
    return axes


def plot_scan_result(input_path, output_path=None):
    output_path = _build_output_path(input_path, output_path)

    with np.load(input_path, allow_pickle=True) as loaded_result:
        has_q0_spread_diagnostics = _has_q0_spread_diagnostics(loaded_result)
        if has_q0_spread_diagnostics:
            figure, axes = plt.subplots(
                2,
                1,
                figsize=(8.0, 8.0),
                sharex=True,
                constrained_layout=True,
            )
        else:
            figure, axis = plt.subplots(
                1,
                1,
                figsize=(8.0, 4.8),
                constrained_layout=True,
            )
            axes = np.array([axis], dtype=object)

        if "q_top_curve_matrix" in loaded_result.files:
            _plot_multi_size_result(loaded_result, axes)
        elif "q_top_curve" in loaded_result.files:
            _plot_single_size_result(loaded_result, axes)
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
        default=str(DEFAULT_INPUT_PATH),
        help="Path to a scan_result*.npz file",
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
