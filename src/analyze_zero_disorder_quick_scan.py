import argparse
import json
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
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "exp06_zero_disorder_quick_scan"
)


def _load_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _round_float(value, digits=8):
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _gap_summary(q_top_curve_matrix):
    q_top_curve_matrix = np.asarray(q_top_curve_matrix, dtype=np.float64)
    gap_34 = q_top_curve_matrix[0] - q_top_curve_matrix[1]
    gap_45 = q_top_curve_matrix[1] - q_top_curve_matrix[2]
    return {
        "gap_L3_minus_L4": [_round_float(value) for value in gap_34],
        "gap_L4_minus_L5": [_round_float(value) for value in gap_45],
        "max_abs_gap_L3_minus_L4": _round_float(np.max(np.abs(gap_34))),
        "max_abs_gap_L4_minus_L5": _round_float(np.max(np.abs(gap_45))),
        "has_L3_L4_sign_change": bool(
            np.any(gap_34[:-1] * gap_34[1:] < 0.0)
        ),
        "has_L4_L5_sign_change": bool(
            np.any(gap_45[:-1] * gap_45[1:] < 0.0)
        ),
    }


def _curve_summary(data):
    q_top = np.asarray(data["q_top_curve_matrix"], dtype=np.float64)
    deficit = 1.0 - q_top
    acceptance = np.asarray(
        data["average_acceptance_rate_curve_matrix"],
        dtype=np.float64,
    )
    observed_syndrome_weight = np.asarray(
        data["observed_syndrome_weight_matrix"],
        dtype=np.int64,
    )
    disorder_data_weight = np.asarray(
        data["disorder_data_weight_matrix"],
        dtype=np.int64,
    )
    summary = {
        "x_axis_name": str(data["x_axis_name"]),
        "x_axis_values": [
            _round_float(value, 6)
            for value in np.asarray(data["x_axis_values"], dtype=np.float64)
        ],
        "lattice_size_list": [
            int(value)
            for value in np.asarray(data["lattice_size_list"], dtype=np.int64)
        ],
        "q_top_min": _round_float(np.min(q_top)),
        "q_top_max": _round_float(np.max(q_top)),
        "max_deficit_1_minus_q_top": _round_float(np.max(deficit)),
        "mean_deficit_1_minus_q_top": _round_float(np.mean(deficit)),
        "max_acceptance_rate": _round_float(np.max(acceptance)),
        "min_acceptance_rate": _round_float(np.min(acceptance)),
        "observed_syndrome_weight_all_zero": bool(
            np.all(observed_syndrome_weight == 0)
        ),
        "disorder_data_weight_all_zero": bool(
            np.all(disorder_data_weight == 0)
        ),
    }
    if "q_top_spread_curve_matrix" in data:
        q_top_spread = np.asarray(
            data["q_top_spread_curve_matrix"],
            dtype=np.float64,
        )
        summary["max_q_top_spread"] = _round_float(np.nanmax(q_top_spread))
    if "max_r_hat_curve_matrix" in data:
        max_r_hat = np.asarray(data["max_r_hat_curve_matrix"], dtype=np.float64)
        summary["max_finite_r_hat"] = _round_float(np.nanmax(max_r_hat))
    if "min_effective_sample_size_curve_matrix" in data:
        ess = np.asarray(
            data["min_effective_sample_size_curve_matrix"],
            dtype=np.float64,
        )
        summary["min_finite_effective_sample_size"] = _round_float(
            np.nanmin(ess)
        )
    summary.update(_gap_summary(q_top))
    return summary


def _plot_scan(axis, data, ylabel, value_matrix, title):
    x_values = np.asarray(data["x_axis_values"], dtype=np.float64)
    lattice_size_list = np.asarray(data["lattice_size_list"], dtype=np.int64)
    for lattice_index, lattice_size in enumerate(lattice_size_list):
        axis.plot(
            x_values,
            value_matrix[lattice_index],
            marker="o",
            linewidth=1.6,
            label=f"L={int(lattice_size)}",
        )
    axis.set_title(title)
    axis.set_xlabel(str(data["x_axis_name"]).replace("_", " "))
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()


def _plot_gap(axis, data, title):
    x_values = np.asarray(data["x_axis_values"], dtype=np.float64)
    q_top = np.asarray(data["q_top_curve_matrix"], dtype=np.float64)
    axis.plot(
        x_values,
        q_top[0] - q_top[1],
        marker="o",
        linewidth=1.6,
        label="L3-L4",
    )
    axis.plot(
        x_values,
        q_top[1] - q_top[2],
        marker="s",
        linewidth=1.6,
        label="L4-L5",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel(str(data["x_axis_name"]).replace("_", " "))
    axis.set_ylabel("q_top gap")
    axis.grid(True, alpha=0.3)
    axis.legend()


def _plot_combined(fixed_q_data, fixed_p_data, output_path):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12.0, 11.0),
        constrained_layout=True,
    )
    _plot_scan(
        axis=axes[0, 0],
        data=fixed_q_data,
        ylabel="q_top",
        value_matrix=np.asarray(fixed_q_data["q_top_curve_matrix"], dtype=float),
        title="fixed q=0.005, all-zero disorder",
    )
    _plot_scan(
        axis=axes[0, 1],
        data=fixed_p_data,
        ylabel="q_top",
        value_matrix=np.asarray(fixed_p_data["q_top_curve_matrix"], dtype=float),
        title="fixed p=0.005, all-zero disorder",
    )
    _plot_scan(
        axis=axes[1, 0],
        data=fixed_q_data,
        ylabel="1 - q_top",
        value_matrix=1.0 - np.asarray(
            fixed_q_data["q_top_curve_matrix"],
            dtype=float,
        ),
        title="fixed q=0.005 deficit",
    )
    _plot_scan(
        axis=axes[1, 1],
        data=fixed_p_data,
        ylabel="1 - q_top",
        value_matrix=1.0 - np.asarray(
            fixed_p_data["q_top_curve_matrix"],
            dtype=float,
        ),
        title="fixed p=0.005 deficit",
    )
    _plot_gap(axes[2, 0], fixed_q_data, "fixed q=0.005 pairwise gaps")
    _plot_gap(axes[2, 1], fixed_p_data, "fixed p=0.005 pairwise gaps")
    figure.suptitle(
        "3D toric all-zero-disorder quick scan "
        "(num_disorder_samples=1, no disorder averaging)",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze all-zero-disorder quick scan outputs.",
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--output-stem", default="zero_disorder_combined")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    fixed_q_path = run_root / "fixed_q" / "fixed_q_q0p0050.npz"
    fixed_p_path = run_root / "fixed_p" / "fixed_p_p0p0050.npz"
    fixed_q_data = _load_npz(fixed_q_path)
    fixed_p_data = _load_npz(fixed_p_path)

    output_plot_path = run_root / f"{args.output_stem}_analysis.png"
    output_summary_path = run_root / f"{args.output_stem}_summary.json"
    _plot_combined(
        fixed_q_data=fixed_q_data,
        fixed_p_data=fixed_p_data,
        output_path=output_plot_path,
    )

    summary = {
        "run_root": str(run_root),
        "disorder_mode": "all_zero_single_sample",
        "num_disorder_samples": 1,
        "fixed_q": _curve_summary(fixed_q_data),
        "fixed_p": _curve_summary(fixed_p_data),
        "analysis_plot_path": str(output_plot_path),
    }
    with output_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"analysis_plot_path={output_plot_path}")
    print(f"summary_path={output_summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
