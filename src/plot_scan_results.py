import argparse
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


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "2d_toric_code"
    / "without_measurement_noise"
    / "q0_geometric_multistart_local"
    / "scan_result_multi_L_q0_geometric_multistart.npz"
)
CI95_Z_SCORE = 1.96


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


def _format_probability_range(probability_list):
    probability_list = np.asarray(probability_list, dtype=np.float64)
    return (
        f"measurement error q={{q_value}}, "
        f"Pauli error p in [{probability_list[0]:0.4f}, {probability_list[-1]:0.4f}]"
    )


def _format_q_value(loaded_result):
    if "syndrome_error_probability" not in loaded_result.files:
        return None
    return float(loaded_result["syndrome_error_probability"])


def _format_code_family(loaded_result):
    if "code_family" in loaded_result.files:
        return str(loaded_result["code_family"]).replace("_", " ")
    if "code_type" in loaded_result.files:
        return str(loaded_result["code_type"]).replace("_", " ")
    return "toric code"


def _build_main_title(loaded_result):
    probability_list = loaded_result["data_error_probability_list"]
    q_value = _format_q_value(loaded_result)
    code_family_label = _format_code_family(loaded_result)
    probability_range_template = _format_probability_range(probability_list)
    if q_value is None:
        return (
            f"{code_family_label} scan "
            + probability_range_template.format(q_value="unknown")
        )
    return (
        f"{code_family_label} scan "
        + probability_range_template.format(q_value=f"{q_value:0.4f}")
    )


def _build_error_bar_values(std_curve, loaded_result):
    std_curve = np.asarray(std_curve, dtype=np.float64)
    if "num_disorder_samples" not in loaded_result.files:
        return std_curve, "disorder std dev"
    num_disorder_samples = int(loaded_result["num_disorder_samples"])
    if num_disorder_samples <= 0:
        return std_curve, "disorder std dev"
    sem_curve = std_curve / math.sqrt(float(num_disorder_samples))
    return CI95_Z_SCORE * sem_curve, "95% CI of disorder mean"


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
    yerr_curve, yerr_label = _build_error_bar_values(
        q_top_std_curve,
        loaded_result=loaded_result,
    )

    axis = axes[0]
    axis.errorbar(
        data_error_probability_list,
        q_top_curve,
        yerr=yerr_curve,
        marker="o",
        linewidth=1.5,
        capsize=3.0,
        label=f"L={lattice_size}",
    )
    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.set_title(_build_main_title(loaded_result))
    axis.grid(True, alpha=0.3)
    axis.legend(title=f"Error bar: {yerr_label}")
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
    yerr_curve_matrix, yerr_label = _build_error_bar_values(
        q_top_std_curve_matrix,
        loaded_result=loaded_result,
    )

    axis = axes[0]

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        label = f"L={int(lattice_size)}"
        axis.errorbar(
            data_error_probability_list,
            q_top_curve_matrix[lattice_index],
            yerr=yerr_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=label,
        )

    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.set_title(_build_main_title(loaded_result))
    axis.grid(True, alpha=0.3)
    axis.legend(title=f"Error bar: {yerr_label}")
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
