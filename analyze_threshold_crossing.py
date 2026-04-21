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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TOLERANCE = 1.0e-12
CI95_Z_SCORE = 1.96


def _load_q_top_std_curve_matrix(loaded_result):
    if "q_top_std_curve_matrix" in loaded_result.files:
        return loaded_result["q_top_std_curve_matrix"]
    return loaded_result["q_top_std_error_curve_matrix"]


def _safe_float_list(array):
    return [float(value) for value in np.asarray(array, dtype=np.float64)]


def _sign_with_tolerance(values, tolerance=DEFAULT_TOLERANCE):
    signs = np.sign(values)
    signs[np.abs(values) <= tolerance] = 0.0
    return signs.astype(np.int8, copy=False)


def _sem_from_std(std_curve, num_disorder_samples):
    if num_disorder_samples <= 0:
        raise ValueError("num_disorder_samples must be positive")
    return np.asarray(std_curve, dtype=np.float64) / math.sqrt(
        float(num_disorder_samples)
    )


def _linear_crossing(left_p, right_p, left_delta, right_delta):
    denominator = right_delta - left_delta
    if abs(denominator) <= DEFAULT_TOLERANCE:
        return float(0.5 * (left_p + right_p))
    return float(
        left_p - left_delta * (right_p - left_p) / denominator
    )


def _detect_crossings(probability_list, delta_curve):
    crossings = []
    signs = _sign_with_tolerance(delta_curve)
    for index in range(len(delta_curve) - 1):
        left_p = float(probability_list[index])
        right_p = float(probability_list[index + 1])
        left_delta = float(delta_curve[index])
        right_delta = float(delta_curve[index + 1])
        left_sign = int(signs[index])
        right_sign = int(signs[index + 1])
        if left_sign == 0:
            crossings.append({
                "left_index": int(index),
                "right_index": int(index),
                "left_p": left_p,
                "right_p": left_p,
                "crossing_estimate_p": left_p,
            })
            continue
        if right_sign == 0:
            crossings.append({
                "left_index": int(index + 1),
                "right_index": int(index + 1),
                "left_p": right_p,
                "right_p": right_p,
                "crossing_estimate_p": right_p,
            })
            continue
        if left_sign * right_sign < 0:
            crossings.append({
                "left_index": int(index),
                "right_index": int(index + 1),
                "left_p": left_p,
                "right_p": right_p,
                "crossing_estimate_p": _linear_crossing(
                    left_p=left_p,
                    right_p=right_p,
                    left_delta=left_delta,
                    right_delta=right_delta,
                ),
            })
    return crossings


def _classify_crossing_location(signs):
    unique_signs = set(int(value) for value in signs)
    if unique_signs == {1} or unique_signs == {0, 1}:
        return "left_of_window"
    if unique_signs == {-1} or unique_signs == {-1, 0}:
        return "right_of_window"
    return "inside_or_touching_window"


def _build_recommended_window(
        probability_list,
        delta_35_curve,
        delta_57_curve,
        pair_35_crossings,
        pair_57_crossings):
    step = float(np.median(np.diff(probability_list)))
    half_width = max(0.0125, 4.0 * step)

    if pair_35_crossings and pair_57_crossings:
        crossing_estimates = [
            pair_35_crossings[0]["crossing_estimate_p"],
            pair_57_crossings[0]["crossing_estimate_p"],
        ]
        center = float(np.mean(crossing_estimates))
        reason = "both_pairwise_gaps_cross_zero"
    else:
        pair_35_min_index = int(np.argmin(np.abs(delta_35_curve)))
        pair_57_min_index = int(np.argmin(np.abs(delta_57_curve)))
        center = float(np.mean([
            probability_list[pair_35_min_index],
            probability_list[pair_57_min_index],
        ]))
        signs_35 = _sign_with_tolerance(delta_35_curve)
        signs_57 = _sign_with_tolerance(delta_57_curve)
        location_35 = _classify_crossing_location(signs_35)
        location_57 = _classify_crossing_location(signs_57)
        if (
                location_35 == "left_of_window"
                and location_57 == "left_of_window"):
            center = float(probability_list[0] - 4.0 * step)
            reason = "both_pairwise_gaps_positive_shift_left"
        elif (
                location_35 == "right_of_window"
                and location_57 == "right_of_window"):
            center = float(probability_list[-1] + 4.0 * step)
            reason = "both_pairwise_gaps_negative_shift_right"
        else:
            reason = "mixed_pairwise_signatures_center_on_min_abs_gap"

    return {
        "p_min": float(center - half_width),
        "p_max": float(center + half_width),
        "step_reference": step,
        "reason": reason,
    }


def _build_pair_summary(
        probability_list,
        delta_curve,
        pooled_sem_curve,
        label):
    delta_curve = np.asarray(delta_curve, dtype=np.float64)
    pooled_sem_curve = np.asarray(pooled_sem_curve, dtype=np.float64)
    min_abs_index = int(np.argmin(np.abs(delta_curve)))
    sign_array = _sign_with_tolerance(delta_curve)
    crossings = _detect_crossings(probability_list, delta_curve)
    min_abs_delta_value = float(delta_curve[min_abs_index])
    pooled_sem_at_min_abs = float(pooled_sem_curve[min_abs_index])
    return {
        "pair_label": label,
        "delta_curve": _safe_float_list(delta_curve),
        "pooled_sem_curve": _safe_float_list(pooled_sem_curve),
        "pooled_ci95_curve": _safe_float_list(CI95_Z_SCORE * pooled_sem_curve),
        "min_abs_delta_index": min_abs_index,
        "min_abs_delta_p": float(probability_list[min_abs_index]),
        "min_abs_delta_value": min_abs_delta_value,
        "pooled_sem_at_min_abs": pooled_sem_at_min_abs,
        "secondary_proximity_hit": bool(
            abs(min_abs_delta_value) <= 2.0 * pooled_sem_at_min_abs
        ),
        "sign_flip_detected": bool(len(crossings) > 0),
        "sign_pattern": [int(value) for value in sign_array.tolist()],
        "crossing_location_classification": (
            _classify_crossing_location(sign_array)
        ),
        "crossing_intervals": crossings,
    }


def _default_output_paths(input_path, output_dir=None, output_stem=None):
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    if output_stem is None:
        output_stem = input_path.stem
    return {
        "summary_path": output_dir / "threshold_summary.json",
        "sem95_plot_path": output_dir / f"{output_stem}_sem95.png",
        "gap_plot_path": output_dir / f"{output_stem}_gap_crossing.png",
    }


def _format_q_p_title(syndrome_error_probability, probability_list):
    probability_list = np.asarray(probability_list, dtype=np.float64)
    return (
        f"measurement error q={syndrome_error_probability:0.4f}, "
        f"Pauli error p in [{probability_list[0]:0.4f}, "
        f"{probability_list[-1]:0.4f}]"
    )


def _plot_sem95(probability_list, lattice_size_list, q_top_curve_matrix,
                q_top_ci95_curve_matrix, syndrome_error_probability,
                output_path):
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(8.0, 4.8),
        constrained_layout=True,
    )
    for lattice_index, lattice_size in enumerate(lattice_size_list):
        axis.errorbar(
            probability_list,
            q_top_curve_matrix[lattice_index],
            yerr=q_top_ci95_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=f"L={int(lattice_size)}",
        )
    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.set_title(
        "Threshold search (95% CI): "
        + _format_q_p_title(syndrome_error_probability, probability_list)
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Error bar: 95% CI")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_gap_crossing(
        probability_list,
        lattice_size_list,
        q_top_curve_matrix,
        q_top_ci95_curve_matrix,
        delta_35_curve,
        delta_57_curve,
        pooled_sem_35_curve,
        pooled_sem_57_curve,
        pair_35_crossings,
        pair_57_crossings,
        syndrome_error_probability,
        output_path):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    top_axis = axes[0]
    bottom_axis = axes[1]

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        top_axis.errorbar(
            probability_list,
            q_top_curve_matrix[lattice_index],
            yerr=q_top_ci95_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=f"L={int(lattice_size)}",
        )
    top_axis.set_ylabel("q_top")
    top_axis.set_title(
        "Threshold search with pairwise gap diagnostics: "
        + _format_q_p_title(syndrome_error_probability, probability_list)
    )
    top_axis.grid(True, alpha=0.3)
    top_axis.legend(title="Error bar: 95% CI")

    ci95_35_curve = CI95_Z_SCORE * np.asarray(pooled_sem_35_curve)
    ci95_57_curve = CI95_Z_SCORE * np.asarray(pooled_sem_57_curve)
    bottom_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    bottom_axis.plot(
        probability_list,
        delta_35_curve,
        marker="o",
        linewidth=1.5,
        label=f"L={int(lattice_size_list[0])}-L={int(lattice_size_list[1])}",
    )
    bottom_axis.fill_between(
        probability_list,
        delta_35_curve - ci95_35_curve,
        delta_35_curve + ci95_35_curve,
        alpha=0.2,
    )
    bottom_axis.plot(
        probability_list,
        delta_57_curve,
        marker="o",
        linewidth=1.5,
        label=f"L={int(lattice_size_list[1])}-L={int(lattice_size_list[2])}",
    )
    bottom_axis.fill_between(
        probability_list,
        delta_57_curve - ci95_57_curve,
        delta_57_curve + ci95_57_curve,
        alpha=0.2,
    )
    for crossing_entry in pair_35_crossings:
        bottom_axis.axvline(
            crossing_entry["crossing_estimate_p"],
            color="C0",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )
    for crossing_entry in pair_57_crossings:
        bottom_axis.axvline(
            crossing_entry["crossing_estimate_p"],
            color="C1",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )
    bottom_axis.set_xlabel("data error probability p")
    bottom_axis.set_ylabel("pairwise gap")
    bottom_axis.grid(True, alpha=0.3)
    bottom_axis.legend(title="Band: pairwise 95% CI")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def analyze_threshold_crossing(
        input_path,
        output_dir=None,
        output_stem=None,
        summary_path=None,
        sem95_plot_path=None,
        gap_plot_path=None):
    input_path = Path(input_path).resolve()
    default_paths = _default_output_paths(
        input_path=input_path,
        output_dir=output_dir,
        output_stem=output_stem,
    )
    if summary_path is None:
        summary_path = default_paths["summary_path"]
    else:
        summary_path = Path(summary_path)
    if sem95_plot_path is None:
        sem95_plot_path = default_paths["sem95_plot_path"]
    else:
        sem95_plot_path = Path(sem95_plot_path)
    if gap_plot_path is None:
        gap_plot_path = default_paths["gap_plot_path"]
    else:
        gap_plot_path = Path(gap_plot_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=True) as loaded_result:
        probability_list = np.asarray(
            loaded_result["data_error_probability_list"],
            dtype=np.float64,
        )
        lattice_size_list = np.asarray(
            loaded_result["lattice_size_list"],
            dtype=np.int64,
        )
        q_top_curve_matrix = np.asarray(
            loaded_result["q_top_curve_matrix"],
            dtype=np.float64,
        )
        q_top_std_curve_matrix = np.asarray(
            _load_q_top_std_curve_matrix(loaded_result),
            dtype=np.float64,
        )
        num_disorder_samples = int(loaded_result["num_disorder_samples"])
        syndrome_error_probability = float(
            loaded_result["syndrome_error_probability"]
        )
        common_random_disorder_across_p = bool(
            loaded_result["common_random_disorder_across_p"]
        )

    if q_top_curve_matrix.shape[0] != 3:
        raise ValueError(
            "threshold crossing analysis currently expects exactly 3 sizes"
        )

    q_top_sem_curve_matrix = _sem_from_std(
        q_top_std_curve_matrix,
        num_disorder_samples=num_disorder_samples,
    )
    q_top_ci95_curve_matrix = CI95_Z_SCORE * q_top_sem_curve_matrix

    delta_35_curve = q_top_curve_matrix[0] - q_top_curve_matrix[1]
    delta_57_curve = q_top_curve_matrix[1] - q_top_curve_matrix[2]
    pooled_sem_35_curve = np.sqrt(
        q_top_sem_curve_matrix[0] ** 2 + q_top_sem_curve_matrix[1] ** 2
    )
    pooled_sem_57_curve = np.sqrt(
        q_top_sem_curve_matrix[1] ** 2 + q_top_sem_curve_matrix[2] ** 2
    )

    pair_35_summary = _build_pair_summary(
        probability_list=probability_list,
        delta_curve=delta_35_curve,
        pooled_sem_curve=pooled_sem_35_curve,
        label=f"{int(lattice_size_list[0])}-{int(lattice_size_list[1])}",
    )
    pair_57_summary = _build_pair_summary(
        probability_list=probability_list,
        delta_curve=delta_57_curve,
        pooled_sem_curve=pooled_sem_57_curve,
        label=f"{int(lattice_size_list[1])}-{int(lattice_size_list[2])}",
    )
    recommended_window = _build_recommended_window(
        probability_list=probability_list,
        delta_35_curve=delta_35_curve,
        delta_57_curve=delta_57_curve,
        pair_35_crossings=pair_35_summary["crossing_intervals"],
        pair_57_crossings=pair_57_summary["crossing_intervals"],
    )

    primary_hit = bool(
        pair_35_summary["sign_flip_detected"]
        and pair_57_summary["sign_flip_detected"]
    )
    crossing_estimate_list = [
        crossing_entry["crossing_estimate_p"]
        for crossing_entry in pair_35_summary["crossing_intervals"]
        + pair_57_summary["crossing_intervals"]
    ]
    if crossing_estimate_list:
        crossing_window = {
            "p_min": float(np.min(crossing_estimate_list)),
            "p_max": float(np.max(crossing_estimate_list)),
        }
    else:
        crossing_window = None

    summary = {
        "input_path": str(input_path),
        "summary_path": str(summary_path),
        "sem95_plot_path": str(sem95_plot_path),
        "gap_plot_path": str(gap_plot_path),
        "syndrome_error_probability": syndrome_error_probability,
        "common_random_disorder_across_p": common_random_disorder_across_p,
        "num_disorder_samples": num_disorder_samples,
        "lattice_size_list": [int(value) for value in lattice_size_list.tolist()],
        "data_error_probability_list": _safe_float_list(probability_list),
        "q_top_curve_matrix": [
            _safe_float_list(row) for row in q_top_curve_matrix
        ],
        "q_top_sem_curve_matrix": [
            _safe_float_list(row) for row in q_top_sem_curve_matrix
        ],
        "q_top_ci95_curve_matrix": [
            _safe_float_list(row) for row in q_top_ci95_curve_matrix
        ],
        "pairwise_gap_analysis": {
            "delta_35": pair_35_summary,
            "delta_57": pair_57_summary,
        },
        "primary_crossing_window_hit": primary_hit,
        "secondary_proximity_hit": bool(
            pair_35_summary["secondary_proximity_hit"]
            or pair_57_summary["secondary_proximity_hit"]
        ),
        "crossing_window": crossing_window,
        "recommended_server_window": recommended_window,
    }

    _plot_sem95(
        probability_list=probability_list,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        q_top_ci95_curve_matrix=q_top_ci95_curve_matrix,
        syndrome_error_probability=syndrome_error_probability,
        output_path=sem95_plot_path,
    )
    _plot_gap_crossing(
        probability_list=probability_list,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        q_top_ci95_curve_matrix=q_top_ci95_curve_matrix,
        delta_35_curve=delta_35_curve,
        delta_57_curve=delta_57_curve,
        pooled_sem_35_curve=pooled_sem_35_curve,
        pooled_sem_57_curve=pooled_sem_57_curve,
        pair_35_crossings=pair_35_summary["crossing_intervals"],
        pair_57_crossings=pair_57_summary["crossing_intervals"],
        syndrome_error_probability=syndrome_error_probability,
        output_path=gap_plot_path,
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-size threshold scan NPZ results."
    )
    parser.add_argument("input_path", help="Path to merged scan_result*.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--sem95-plot-path", default=None)
    parser.add_argument("--gap-plot-path", default=None)
    args = parser.parse_args()

    summary = analyze_threshold_crossing(
        input_path=args.input_path,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        summary_path=args.summary_path,
        sem95_plot_path=args.sem95_plot_path,
        gap_plot_path=args.gap_plot_path,
    )
    print(json.dumps({
        "summary_path": summary["summary_path"],
        "sem95_plot_path": summary["sem95_plot_path"],
        "gap_plot_path": summary["gap_plot_path"],
        "primary_crossing_window_hit": (
            summary["primary_crossing_window_hit"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
