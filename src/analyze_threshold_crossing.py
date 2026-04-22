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


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_TOLERANCE = 1.0e-12
CI95_Z_SCORE = 1.96
SATURATION_Q_TOP_TOLERANCE = 1.0e-9


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


def _is_q_top_saturated_at_unity(q_top_values, q_top_ci95_values):
    q_top_values = np.asarray(q_top_values, dtype=np.float64)
    q_top_ci95_values = np.asarray(q_top_ci95_values, dtype=np.float64)
    return bool(np.all(
        q_top_values >= 1.0 - q_top_ci95_values - SATURATION_Q_TOP_TOLERANCE
    ))


def _build_left_saturation_profile(q_top_curve_matrix, q_top_ci95_curve_matrix):
    saturated_point_flags = np.array([
        _is_q_top_saturated_at_unity(
            q_top_curve_matrix[:, point_index],
            q_top_ci95_curve_matrix[:, point_index],
        )
        for point_index in range(q_top_curve_matrix.shape[1])
    ], dtype=bool)
    left_edge_end_index = -1
    for point_index, is_saturated in enumerate(saturated_point_flags.tolist()):
        if not is_saturated:
            break
        left_edge_end_index = point_index
    return saturated_point_flags, left_edge_end_index


def _group_consecutive_indices(index_list):
    if not index_list:
        return []
    grouped = [[int(index_list[0])]]
    for index in index_list[1:]:
        if index == grouped[-1][-1] + 1:
            grouped[-1].append(int(index))
        else:
            grouped.append([int(index)])
    return grouped


def _build_window_record(
        p_min,
        p_max,
        representative_p,
        reason,
        index_min=None,
        index_max=None,
        representative_index=None):
    window = {
        "p_min": float(min(p_min, p_max)),
        "p_max": float(max(p_min, p_max)),
        "representative_p": float(representative_p),
        "reason": str(reason),
    }
    if index_min is not None:
        window["index_min"] = int(index_min)
    if index_max is not None:
        window["index_max"] = int(index_max)
    if representative_index is not None:
        window["representative_index"] = int(representative_index)
    return window


def _window_from_crossing_entry(crossing_entry, reason):
    return _build_window_record(
        p_min=crossing_entry["left_p"],
        p_max=crossing_entry["right_p"],
        representative_p=crossing_entry["crossing_estimate_p"],
        reason=reason,
        index_min=crossing_entry["left_index"],
        index_max=crossing_entry["right_index"],
        representative_index=crossing_entry["left_index"],
    )


def _window_from_near_group(probability_list, delta_curve, index_group, reason):
    representative_index = min(
        index_group,
        key=lambda index: abs(float(delta_curve[index])),
    )
    return _build_window_record(
        p_min=probability_list[index_group[0]],
        p_max=probability_list[index_group[-1]],
        representative_p=probability_list[representative_index],
        reason=reason,
        index_min=index_group[0],
        index_max=index_group[-1],
        representative_index=representative_index,
    )


def _build_common_crossing_window(pair_35_summary, pair_57_summary):
    if not (
            pair_35_summary["interior_crossing_intervals"]
            and pair_57_summary["interior_crossing_intervals"]):
        return None
    pair_35_crossing = pair_35_summary["interior_crossing_intervals"][0]
    pair_57_crossing = pair_57_summary["interior_crossing_intervals"][0]
    estimate_values = [
        pair_35_crossing["crossing_estimate_p"],
        pair_57_crossing["crossing_estimate_p"],
    ]
    return _build_window_record(
        p_min=min(estimate_values),
        p_max=max(estimate_values),
        representative_p=float(np.mean(estimate_values)),
        reason="both_pairwise_gaps_cross_zero_interior",
    )


def _build_common_near_crossing_window(pair_35_summary, pair_57_summary):
    pair_target_windows = []
    target_reasons = []
    for pair_summary in (pair_35_summary, pair_57_summary):
        if pair_summary["interior_crossing_intervals"]:
            pair_target_windows.append(_window_from_crossing_entry(
                pair_summary["interior_crossing_intervals"][0],
                reason="pairwise_gap_cross_zero_interior",
            ))
            target_reasons.append("crossing")
        elif pair_summary["interior_near_crossing_windows"]:
            pair_target_windows.append(
                pair_summary["interior_near_crossing_windows"][0]
            )
            target_reasons.append("near")
        else:
            return None

    representative_values = [
        window["representative_p"] for window in pair_target_windows
    ]
    if target_reasons == ["near", "near"]:
        reason = "both_pairwise_gaps_within_pooled_ci95_interior"
    else:
        reason = "one_pair_crosses_zero_other_within_pooled_ci95_interior"
    return _build_window_record(
        p_min=min(representative_values),
        p_max=max(representative_values),
        representative_p=float(np.mean(representative_values)),
        reason=reason,
    )


def _build_recommended_window(
        probability_list,
        interior_crossing_window,
        interior_near_crossing_window):
    step = float(np.median(np.diff(probability_list)))
    if interior_crossing_window is not None:
        window = dict(interior_crossing_window)
        window["step_reference"] = step
        return window
    if interior_near_crossing_window is not None:
        window = dict(interior_near_crossing_window)
        window["step_reference"] = step
        return window
    return None


def _build_pair_summary(
        probability_list,
        delta_curve,
        pooled_sem_curve,
        label,
        left_saturation_end_index):
    delta_curve = np.asarray(delta_curve, dtype=np.float64)
    pooled_sem_curve = np.asarray(pooled_sem_curve, dtype=np.float64)
    pooled_ci95_curve = CI95_Z_SCORE * pooled_sem_curve
    min_abs_index = int(np.argmin(np.abs(delta_curve)))
    sign_array = _sign_with_tolerance(delta_curve)
    crossings = _detect_crossings(probability_list, delta_curve)
    boundary_crossings = []
    interior_crossings = []
    for crossing_entry in crossings:
        if (
                left_saturation_end_index >= 0
                and crossing_entry["right_index"] <= left_saturation_end_index):
            boundary_crossings.append(crossing_entry)
        else:
            interior_crossings.append(crossing_entry)
    candidate_index_list = [
        index
        for index in range(left_saturation_end_index + 1, len(delta_curve))
        if abs(float(delta_curve[index])) <= float(pooled_ci95_curve[index])
    ]
    interior_near_windows = [
        _window_from_near_group(
            probability_list=probability_list,
            delta_curve=delta_curve,
            index_group=index_group,
            reason="abs_gap_within_pooled_ci95_interior",
        )
        for index_group in _group_consecutive_indices(candidate_index_list)
    ]
    if left_saturation_end_index + 1 < len(delta_curve):
        non_saturated_slice = np.abs(delta_curve[left_saturation_end_index + 1:])
        non_saturated_min_abs_index = int(
            np.argmin(non_saturated_slice) + left_saturation_end_index + 1
        )
    else:
        non_saturated_min_abs_index = None
    min_abs_delta_value = float(delta_curve[min_abs_index])
    pooled_sem_at_min_abs = float(pooled_sem_curve[min_abs_index])
    return {
        "pair_label": label,
        "delta_curve": _safe_float_list(delta_curve),
        "pooled_sem_curve": _safe_float_list(pooled_sem_curve),
        "pooled_ci95_curve": _safe_float_list(pooled_ci95_curve),
        "min_abs_delta_index": min_abs_index,
        "min_abs_delta_p": float(probability_list[min_abs_index]),
        "min_abs_delta_value": min_abs_delta_value,
        "pooled_sem_at_min_abs": pooled_sem_at_min_abs,
        "secondary_proximity_hit": bool(len(interior_near_windows) > 0),
        "sign_flip_detected": bool(len(interior_crossings) > 0),
        "sign_pattern": [int(value) for value in sign_array.tolist()],
        "crossing_location_classification": (
            _classify_crossing_location(sign_array)
        ),
        "crossing_intervals": interior_crossings,
        "raw_crossing_intervals": crossings,
        "boundary_crossing_intervals": boundary_crossings,
        "boundary_artifact_crossing_detected": bool(len(boundary_crossings) > 0),
        "interior_crossing_intervals": interior_crossings,
        "interior_sign_flip_detected": bool(len(interior_crossings) > 0),
        "interior_near_crossing_windows": interior_near_windows,
        "interior_near_crossing_hit": bool(len(interior_near_windows) > 0),
        "non_saturated_min_abs_delta_index": non_saturated_min_abs_index,
        "non_saturated_min_abs_delta_p": (
            None if non_saturated_min_abs_index is None
            else float(probability_list[non_saturated_min_abs_index])
        ),
        "non_saturated_min_abs_delta_value": (
            None if non_saturated_min_abs_index is None
            else float(delta_curve[non_saturated_min_abs_index])
        ),
        "right_edge_sign": int(sign_array[-1]),
        "right_edge_delta": float(delta_curve[-1]),
        "right_edge_pooled_sem": float(pooled_sem_curve[-1]),
        "right_edge_pooled_ci95": float(pooled_ci95_curve[-1]),
        "right_edge_near_hit": bool(
            abs(float(delta_curve[-1])) <= float(pooled_ci95_curve[-1])
        ),
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


def _format_code_family_label(code_family):
    return str(code_family).replace("_", " ")


def _format_q_p_title(
        code_family,
        syndrome_error_probability,
        probability_list):
    probability_list = np.asarray(probability_list, dtype=np.float64)
    return (
        f"{_format_code_family_label(code_family)}, "
        f"measurement error q={syndrome_error_probability:0.4f}, "
        f"Pauli error p in [{probability_list[0]:0.4f}, "
        f"{probability_list[-1]:0.4f}]"
    )


def _plot_sem95(probability_list, lattice_size_list, q_top_curve_matrix,
                q_top_ci95_curve_matrix, code_family,
                syndrome_error_probability,
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
        + _format_q_p_title(
            code_family,
            syndrome_error_probability,
            probability_list,
        )
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
        pair_35_boundary_crossings,
        pair_57_boundary_crossings,
        code_family,
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
        + _format_q_p_title(
            code_family,
            syndrome_error_probability,
            probability_list,
        )
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
    for crossing_entry in pair_35_boundary_crossings:
        bottom_axis.axvline(
            crossing_entry["crossing_estimate_p"],
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            alpha=0.5,
        )
    for crossing_entry in pair_57_boundary_crossings:
        bottom_axis.axvline(
            crossing_entry["crossing_estimate_p"],
            color="0.65",
            linestyle=":",
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
        if "code_family" in loaded_result.files:
            code_family = str(loaded_result["code_family"])
        elif "code_type" in loaded_result.files:
            code_family = str(loaded_result["code_type"])
        else:
            code_family = "2d_toric"

    if q_top_curve_matrix.shape[0] != 3:
        raise ValueError(
            "threshold crossing analysis currently expects exactly 3 sizes"
        )

    q_top_sem_curve_matrix = _sem_from_std(
        q_top_std_curve_matrix,
        num_disorder_samples=num_disorder_samples,
    )
    q_top_ci95_curve_matrix = CI95_Z_SCORE * q_top_sem_curve_matrix
    saturated_point_flags, left_saturation_end_index = (
        _build_left_saturation_profile(
            q_top_curve_matrix=q_top_curve_matrix,
            q_top_ci95_curve_matrix=q_top_ci95_curve_matrix,
        )
    )

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
        left_saturation_end_index=left_saturation_end_index,
    )
    pair_57_summary = _build_pair_summary(
        probability_list=probability_list,
        delta_curve=delta_57_curve,
        pooled_sem_curve=pooled_sem_57_curve,
        label=f"{int(lattice_size_list[1])}-{int(lattice_size_list[2])}",
        left_saturation_end_index=left_saturation_end_index,
    )
    interior_crossing_window = _build_common_crossing_window(
        pair_35_summary=pair_35_summary,
        pair_57_summary=pair_57_summary,
    )
    interior_near_crossing_window = None
    if interior_crossing_window is None:
        interior_near_crossing_window = _build_common_near_crossing_window(
            pair_35_summary=pair_35_summary,
            pair_57_summary=pair_57_summary,
        )
    recommended_window = _build_recommended_window(
        probability_list=probability_list,
        interior_crossing_window=interior_crossing_window,
        interior_near_crossing_window=interior_near_crossing_window,
    )

    raw_crossing_estimate_list = [
        crossing_entry["crossing_estimate_p"]
        for crossing_entry in pair_35_summary["raw_crossing_intervals"]
        + pair_57_summary["raw_crossing_intervals"]
    ]
    if raw_crossing_estimate_list:
        raw_crossing_window = {
            "p_min": float(np.min(raw_crossing_estimate_list)),
            "p_max": float(np.max(raw_crossing_estimate_list)),
        }
    else:
        raw_crossing_window = None
    boundary_saturation_artifact = bool(
        pair_35_summary["boundary_artifact_crossing_detected"]
        or pair_57_summary["boundary_artifact_crossing_detected"]
    )
    right_edge_gap_signs = {
        pair_35_summary["pair_label"]: pair_35_summary["right_edge_sign"],
        pair_57_summary["pair_label"]: pair_57_summary["right_edge_sign"],
    }
    right_edge_gap_values = {
        pair_35_summary["pair_label"]: {
            "delta": pair_35_summary["right_edge_delta"],
            "pooled_ci95": pair_35_summary["right_edge_pooled_ci95"],
        },
        pair_57_summary["pair_label"]: {
            "delta": pair_57_summary["right_edge_delta"],
            "pooled_ci95": pair_57_summary["right_edge_pooled_ci95"],
        },
    }

    summary = {
        "input_path": str(input_path),
        "summary_path": str(summary_path),
        "sem95_plot_path": str(sem95_plot_path),
        "gap_plot_path": str(gap_plot_path),
        "code_family": code_family,
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
        "left_edge_saturated_point_flags": [
            bool(value) for value in saturated_point_flags.tolist()
        ],
        "left_edge_saturation_end_index": int(left_saturation_end_index),
        "left_edge_saturation_end_p": (
            None if left_saturation_end_index < 0
            else float(probability_list[left_saturation_end_index])
        ),
        "pairwise_gap_analysis": {
            "delta_35": pair_35_summary,
            "delta_57": pair_57_summary,
        },
        "pair_labels": [
            pair_35_summary["pair_label"],
            pair_57_summary["pair_label"],
        ],
        "boundary_saturation_artifact": boundary_saturation_artifact,
        "primary_crossing_window_hit": bool(interior_crossing_window is not None),
        "secondary_proximity_hit": bool(
            interior_near_crossing_window is not None
        ),
        "raw_crossing_window": raw_crossing_window,
        "crossing_window": interior_crossing_window,
        "interior_crossing_window": interior_crossing_window,
        "interior_near_crossing_window": interior_near_crossing_window,
        "right_edge_gap_signs": right_edge_gap_signs,
        "right_edge_gap_values": right_edge_gap_values,
        "recommended_server_window": recommended_window,
    }

    _plot_sem95(
        probability_list=probability_list,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        q_top_ci95_curve_matrix=q_top_ci95_curve_matrix,
        code_family=code_family,
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
        pair_35_boundary_crossings=pair_35_summary["boundary_crossing_intervals"],
        pair_57_boundary_crossings=pair_57_summary["boundary_crossing_intervals"],
        code_family=code_family,
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
