import argparse
import csv
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
CI95_Z_SCORE = 1.96
DEFAULT_LEVEL_LIST = [0.55, 0.50]
DEFAULT_TOLERANCE = 1.0e-12


def _load_q_top_std_curve_matrix(loaded_result):
    if "q_top_std_curve_matrix" in loaded_result.files:
        return loaded_result["q_top_std_curve_matrix"]
    return loaded_result["q_top_std_error_curve_matrix"]


def _read_npz_result(npz_path):
    with np.load(npz_path, allow_pickle=True) as loaded_result:
        return {
            "npz_path": str(npz_path),
            "lattice_size_list": np.asarray(
                loaded_result["lattice_size_list"],
                dtype=np.int64,
            ),
            "data_error_probability_list": np.asarray(
                loaded_result["data_error_probability_list"],
                dtype=np.float64,
            ),
            "q_top_curve_matrix": np.asarray(
                loaded_result["q_top_curve_matrix"],
                dtype=np.float64,
            ),
            "q_top_std_curve_matrix": np.asarray(
                _load_q_top_std_curve_matrix(loaded_result),
                dtype=np.float64,
            ),
            "num_disorder_samples": int(
                loaded_result["num_disorder_samples"]
            ),
            "syndrome_error_probability": float(
                loaded_result["syndrome_error_probability"]
            ),
        }


def _load_measurement_run_root(run_root):
    run_root = Path(run_root)
    result_by_q = {}
    for summary_path in sorted(run_root.glob("q_*/threshold_summary.json")):
        q_dir = summary_path.parent
        npz_path = next(q_dir.glob("*.npz"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = _read_npz_result(npz_path)
        result["threshold_summary"] = summary
        q_value = float(result["syndrome_error_probability"])
        result_by_q[q_value] = result
    return result_by_q


def _merge_q0_results(base_result, extension_result):
    extension_size_set = set(
        int(value) for value in extension_result["lattice_size_list"]
    )
    base_size_set = set(int(value) for value in base_result["lattice_size_list"])
    if base_size_set & extension_size_set:
        raise ValueError("q=0 base and extension sizes must be disjoint")
    if not np.allclose(
            base_result["data_error_probability_list"],
            extension_result["data_error_probability_list"]):
        raise ValueError("q=0 base and extension p grids must match")

    size_to_row = {}
    for row_index, lattice_size in enumerate(base_result["lattice_size_list"]):
        size_to_row[int(lattice_size)] = {
            "curve": base_result["q_top_curve_matrix"][row_index],
            "std": base_result["q_top_std_curve_matrix"][row_index],
        }
    for row_index, lattice_size in enumerate(extension_result["lattice_size_list"]):
        size_to_row[int(lattice_size)] = {
            "curve": extension_result["q_top_curve_matrix"][row_index],
            "std": extension_result["q_top_std_curve_matrix"][row_index],
        }

    merged_sizes = np.asarray(sorted(size_to_row.keys()), dtype=np.int64)
    merged_curve_matrix = np.vstack([
        size_to_row[int(lattice_size)]["curve"]
        for lattice_size in merged_sizes
    ])
    merged_std_matrix = np.vstack([
        size_to_row[int(lattice_size)]["std"]
        for lattice_size in merged_sizes
    ])
    return {
        "npz_path": None,
        "lattice_size_list": merged_sizes,
        "data_error_probability_list": base_result["data_error_probability_list"],
        "q_top_curve_matrix": merged_curve_matrix,
        "q_top_std_curve_matrix": merged_std_matrix,
        "num_disorder_samples": int(base_result["num_disorder_samples"]),
        "syndrome_error_probability": 0.0,
    }


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _safe_float_list(values):
    return [float(value) for value in values]


def _pair_label(left_size, right_size):
    return f"L{int(left_size)}-L{int(right_size)}"


def _linear_interpolate_zero(x_left, x_right, y_left, y_right):
    denominator = y_right - y_left
    if abs(denominator) <= DEFAULT_TOLERANCE:
        return float(0.5 * (x_left + x_right))
    return float(x_left - y_left * (x_right - x_left) / denominator)


def _linear_interpolate_level(x_left, x_right, y_left, y_right, level):
    denominator = y_right - y_left
    if abs(denominator) <= DEFAULT_TOLERANCE:
        return float(0.5 * (x_left + x_right))
    return float(
        x_left + (level - y_left) * (x_right - x_left) / denominator
    )


def _sem_matrix_from_std(std_matrix, num_disorder_samples):
    return np.asarray(std_matrix, dtype=np.float64) / math.sqrt(
        float(num_disorder_samples)
    )


def _classify_pair_crossing(probability_list, delta_curve):
    probability_list = np.asarray(probability_list, dtype=np.float64)
    delta_curve = np.asarray(delta_curve, dtype=np.float64)
    sign_array = np.sign(delta_curve)
    sign_array[np.abs(delta_curve) <= DEFAULT_TOLERANCE] = 0.0

    crossing_list = []
    for index in range(len(delta_curve) - 1):
        left_value = float(delta_curve[index])
        right_value = float(delta_curve[index + 1])
        left_sign = int(sign_array[index])
        right_sign = int(sign_array[index + 1])
        if left_sign == 0:
            crossing_list.append(float(probability_list[index]))
            continue
        if right_sign == 0:
            crossing_list.append(float(probability_list[index + 1]))
            continue
        if left_sign * right_sign < 0:
            crossing_list.append(_linear_interpolate_zero(
                x_left=float(probability_list[index]),
                x_right=float(probability_list[index + 1]),
                y_left=left_value,
                y_right=right_value,
            ))

    if crossing_list:
        return {
            "status": "crossing",
            "crossing_estimate_p": float(crossing_list[0]),
            "bound_p": None,
            "min_abs_delta_p": float(
                probability_list[np.argmin(np.abs(delta_curve))]
            ),
            "min_abs_delta_value": float(np.min(np.abs(delta_curve))),
        }

    if np.all(delta_curve > 0.0):
        status = "upper_bound"
        bound_p = float(probability_list[0])
    elif np.all(delta_curve < 0.0):
        status = "lower_bound"
        bound_p = float(probability_list[-1])
    else:
        status = "ambiguous"
        bound_p = None
    min_abs_index = int(np.argmin(np.abs(delta_curve)))
    return {
        "status": status,
        "crossing_estimate_p": None,
        "bound_p": _safe_float(bound_p),
        "min_abs_delta_p": float(probability_list[min_abs_index]),
        "min_abs_delta_value": float(delta_curve[min_abs_index]),
    }


def _estimate_levelset(probability_list, q_top_curve, target_level):
    probability_list = np.asarray(probability_list, dtype=np.float64)
    q_top_curve = np.asarray(q_top_curve, dtype=np.float64)
    if np.all(q_top_curve > target_level):
        return {
            "status": "upper_bound",
            "p_value": float(probability_list[-1]),
        }
    if np.all(q_top_curve < target_level):
        return {
            "status": "lower_bound",
            "p_value": float(probability_list[0]),
        }
    for index in range(len(q_top_curve) - 1):
        left_value = float(q_top_curve[index])
        right_value = float(q_top_curve[index + 1])
        if abs(left_value - target_level) <= DEFAULT_TOLERANCE:
            return {
                "status": "interpolated",
                "p_value": float(probability_list[index]),
            }
        if abs(right_value - target_level) <= DEFAULT_TOLERANCE:
            return {
                "status": "interpolated",
                "p_value": float(probability_list[index + 1]),
            }
        if (
                (left_value - target_level) * (right_value - target_level)
                < 0.0):
            return {
                "status": "interpolated",
                "p_value": _linear_interpolate_level(
                    x_left=float(probability_list[index]),
                    x_right=float(probability_list[index + 1]),
                    y_left=left_value,
                    y_right=right_value,
                    level=float(target_level),
                ),
            }
    min_abs_index = int(np.argmin(np.abs(q_top_curve - target_level)))
    return {
        "status": "ambiguous",
        "p_value": float(probability_list[min_abs_index]),
    }


def _write_csv(path, fieldnames, row_list):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def _plot_q0_control(
        q0_result,
        q0_control_plot_path,
        q0_crossing_plot_path,
        crossing_row_list):
    probability_list = q0_result["data_error_probability_list"]
    lattice_size_list = q0_result["lattice_size_list"]
    q_top_curve_matrix = q0_result["q_top_curve_matrix"]
    q_top_ci95_curve_matrix = (
        CI95_Z_SCORE
        * _sem_matrix_from_std(
            q0_result["q_top_std_curve_matrix"],
            q0_result["num_disorder_samples"],
        )
    )

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
            linewidth=1.4,
            capsize=3.0,
            label=f"L={int(lattice_size)}",
        )
    axis.set_title("q=0 control with larger sizes (95% CI)")
    axis.set_xlabel("data error probability p")
    axis.set_ylabel("q_top")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Error bar: 95% CI")
    figure.savefig(q0_control_plot_path, dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(6.5, 4.6),
        constrained_layout=True,
    )
    valid_row_list = [
        row for row in crossing_row_list
        if row["q"] == 0.0 and row["status"] == "crossing"
    ]
    x_value_list = []
    y_value_list = []
    label_list = []
    for row in valid_row_list:
        x_value_list.append(float(row["inverse_l_eff"]))
        y_value_list.append(float(row["crossing_estimate_p"]))
        label_list.append(row["pair_label"])
    axis.plot(x_value_list, y_value_list, marker="o", linewidth=1.4)
    for x_value, y_value, label in zip(x_value_list, y_value_list, label_list):
        axis.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(4, 4))
    axis.set_title("q=0 pairwise crossing drift vs 1/L_eff")
    axis.set_xlabel("1 / L_eff")
    axis.set_ylabel("pairwise crossing p_cross")
    axis.grid(True, alpha=0.3)
    figure.savefig(q0_crossing_plot_path, dpi=220)
    plt.close(figure)


def _plot_q_positive_pseudocritical_drift(
        crossing_row_list,
        output_path):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 8.0),
        constrained_layout=True,
    )
    axes = axes.ravel()
    q_value_list = sorted({
        float(row["q"])
        for row in crossing_row_list
        if float(row["q"]) > 0.0
    })
    pair_label_list = sorted({
        row["pair_label"]
        for row in crossing_row_list
        if float(row["q"]) > 0.0
    })
    for axis, q_value in zip(axes, q_value_list):
        q_row_list = [
            row for row in crossing_row_list
            if abs(float(row["q"]) - q_value) <= DEFAULT_TOLERANCE
        ]
        for pair_label in pair_label_list:
            pair_rows = [
                row for row in q_row_list
                if row["pair_label"] == pair_label
            ]
            if not pair_rows:
                continue
            x_value_list = []
            y_value_list = []
            linestyle = "-"
            for row in pair_rows:
                x_value_list.append(float(row["inverse_l_eff"]))
                if row["status"] == "crossing":
                    y_value_list.append(float(row["crossing_estimate_p"]))
                elif row["status"] == "upper_bound":
                    y_value_list.append(float(row["bound_p"]))
                    linestyle = "--"
                elif row["status"] == "lower_bound":
                    y_value_list.append(float(row["bound_p"]))
                    linestyle = ":"
                else:
                    y_value_list.append(float(row["min_abs_delta_p"]))
                    linestyle = "-."
            axis.plot(
                x_value_list,
                y_value_list,
                marker="o",
                linewidth=1.3,
                linestyle=linestyle,
                label=pair_label,
            )
        axis.set_title(f"q={q_value:0.4f}")
        axis.set_xlabel("1 / L_eff")
        axis.set_ylabel("p_cross or bound")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Pseudo-threshold drift for q>0", fontsize=12)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_levelset_drift(levelset_row_list, output_path):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 8.0),
        constrained_layout=True,
    )
    axes = axes.ravel()
    q_value_list = sorted({
        float(row["q"])
        for row in levelset_row_list
        if float(row["q"]) > 0.0
    })
    level_list = sorted({
        float(row["target_level"]) for row in levelset_row_list
    }, reverse=True)
    for axis, q_value in zip(axes, q_value_list):
        q_row_list = [
            row for row in levelset_row_list
            if abs(float(row["q"]) - q_value) <= DEFAULT_TOLERANCE
        ]
        for target_level in level_list:
            level_rows = [
                row for row in q_row_list
                if abs(float(row["target_level"]) - target_level)
                <= DEFAULT_TOLERANCE
            ]
            x_value_list = [float(row["inverse_L"]) for row in level_rows]
            y_value_list = [float(row["p_value"]) for row in level_rows]
            axis.plot(
                x_value_list,
                y_value_list,
                marker="o",
                linewidth=1.3,
                label=f"p_{target_level:0.2f}(L,q)",
            )
        axis.set_title(f"q={q_value:0.4f}")
        axis.set_xlabel("1 / L")
        axis.set_ylabel("level-set p")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Level-set drift for q>0", fontsize=12)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_fixed_p_size_trends(fixed_p_summary, output_path):
    q_value_list = sorted(
        float(q_value) for q_value in fixed_p_summary.keys()
        if float(q_value) > 0.0
    )
    figure, axes = plt.subplots(
        len(q_value_list),
        1,
        figsize=(8.0, 3.0 * max(1, len(q_value_list))),
        constrained_layout=True,
    )
    if len(q_value_list) == 1:
        axes = np.array([axes], dtype=object)
    for axis, q_value in zip(axes, q_value_list):
        q_summary = fixed_p_summary[f"{q_value:0.4f}"]
        for selection_entry in q_summary["selected_p_points"]:
            axis.plot(
                selection_entry["lattice_size_list"],
                selection_entry["q_top_values"],
                marker="o",
                linewidth=1.3,
                label=f"p={selection_entry['selected_p']:0.4f}",
            )
        axis.set_title(f"Fixed-p size trends for q={q_value:0.4f}")
        axis.set_xlabel("L")
        axis.set_ylabel("q_top")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_pairwise_gap_supplement(q_positive_result_by_q, output_path):
    q_value_list = sorted(q_positive_result_by_q.keys())
    figure, axes = plt.subplots(
        len(q_value_list),
        1,
        figsize=(8.0, 3.2 * max(1, len(q_value_list))),
        constrained_layout=True,
    )
    if len(q_value_list) == 1:
        axes = np.array([axes], dtype=object)
    for axis, q_value in zip(axes, q_value_list):
        result = q_positive_result_by_q[q_value]
        probability_list = result["data_error_probability_list"]
        lattice_size_list = result["lattice_size_list"]
        q_top_curve_matrix = result["q_top_curve_matrix"]
        for index in range(len(lattice_size_list) - 1):
            delta_curve = (
                q_top_curve_matrix[index]
                - q_top_curve_matrix[index + 1]
            )
            axis.plot(
                probability_list,
                delta_curve,
                marker="o",
                linewidth=1.2,
                label=_pair_label(
                    lattice_size_list[index],
                    lattice_size_list[index + 1],
                ),
            )
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        axis.set_title(f"Pairwise gap supplement for q={q_value:0.4f}")
        axis.set_xlabel("p")
        axis.set_ylabel("pairwise q_top gap")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def analyze_no_threshold_evidence(
        q0_baseline_npz_path,
        q0_extension_npz_path,
        q_positive_run_root,
        output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q0_baseline_result = _read_npz_result(q0_baseline_npz_path)
    q0_extension_result = _read_npz_result(q0_extension_npz_path)
    q0_result = _merge_q0_results(q0_baseline_result, q0_extension_result)
    q_positive_result_by_q = _load_measurement_run_root(q_positive_run_root)

    crossing_row_list = []
    levelset_row_list = []
    fixed_p_summary = {}

    def append_crossing_rows(result, q_value):
        probability_list = result["data_error_probability_list"]
        lattice_size_list = result["lattice_size_list"]
        q_top_curve_matrix = result["q_top_curve_matrix"]
        for index in range(len(lattice_size_list) - 1):
            left_size = int(lattice_size_list[index])
            right_size = int(lattice_size_list[index + 1])
            delta_curve = q_top_curve_matrix[index] - q_top_curve_matrix[index + 1]
            crossing_entry = _classify_pair_crossing(
                probability_list=probability_list,
                delta_curve=delta_curve,
            )
            l_eff = 0.5 * (left_size + right_size)
            crossing_row_list.append({
                "q": float(q_value),
                "pair_label": _pair_label(left_size, right_size),
                "left_size": left_size,
                "right_size": right_size,
                "l_eff": float(l_eff),
                "inverse_l_eff": float(1.0 / l_eff),
                "status": crossing_entry["status"],
                "crossing_estimate_p": _safe_float(
                    crossing_entry["crossing_estimate_p"]
                ),
                "bound_p": _safe_float(crossing_entry["bound_p"]),
                "min_abs_delta_p": crossing_entry["min_abs_delta_p"],
                "min_abs_delta_value": crossing_entry["min_abs_delta_value"],
            })

    append_crossing_rows(q0_result, q_value=0.0)
    for q_value, result in sorted(q_positive_result_by_q.items()):
        append_crossing_rows(result, q_value=q_value)

        lattice_size_list = result["lattice_size_list"]
        probability_list = result["data_error_probability_list"]
        q_top_curve_matrix = result["q_top_curve_matrix"]
        for lattice_index, lattice_size in enumerate(lattice_size_list):
            q_top_curve = q_top_curve_matrix[lattice_index]
            for target_level in DEFAULT_LEVEL_LIST:
                levelset_entry = _estimate_levelset(
                    probability_list=probability_list,
                    q_top_curve=q_top_curve,
                    target_level=target_level,
                )
                levelset_row_list.append({
                    "q": float(q_value),
                    "target_level": float(target_level),
                    "lattice_size": int(lattice_size),
                    "inverse_L": float(1.0 / float(lattice_size)),
                    "status": levelset_entry["status"],
                    "p_value": float(levelset_entry["p_value"]),
                })

        selected_indices = [
            0,
            len(probability_list) // 2,
            len(probability_list) - 1,
        ]
        selected_p_points = []
        for selected_index in selected_indices:
            selected_p = float(probability_list[selected_index])
            q_top_values = _safe_float_list(
                q_top_curve_matrix[:, selected_index]
            )
            monotone_descending = bool(np.all(np.diff(q_top_values) < 0.0))
            tail_values = np.asarray(q_top_values[2:], dtype=np.float64)
            tail_monotone_descending = bool(np.all(np.diff(tail_values) < 0.0))
            selected_p_points.append({
                "selected_p": selected_p,
                "lattice_size_list": [int(v) for v in lattice_size_list.tolist()],
                "q_top_values": q_top_values,
                "monotone_descending_all_sizes": monotone_descending,
                "monotone_descending_tail_sizes": tail_monotone_descending,
            })
        fixed_p_summary[f"{q_value:0.4f}"] = {
            "selected_p_points": selected_p_points
        }

    crossing_csv_path = output_dir / "pseudocritical_crossing_table.csv"
    levelset_csv_path = output_dir / "levelset_drift_table.csv"
    fixed_p_summary_path = output_dir / "fixed_p_size_trend_summary.json"
    summary_path = output_dir / "no_threshold_evidence_summary.json"

    _write_csv(
        crossing_csv_path,
        fieldnames=[
            "q",
            "pair_label",
            "left_size",
            "right_size",
            "l_eff",
            "inverse_l_eff",
            "status",
            "crossing_estimate_p",
            "bound_p",
            "min_abs_delta_p",
            "min_abs_delta_value",
        ],
        row_list=crossing_row_list,
    )
    _write_csv(
        levelset_csv_path,
        fieldnames=[
            "q",
            "target_level",
            "lattice_size",
            "inverse_L",
            "status",
            "p_value",
        ],
        row_list=levelset_row_list,
    )
    with fixed_p_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(fixed_p_summary, handle, indent=2, sort_keys=True)

    q_positive_no_stable_common_crossing = True
    q_positive_drift_left = True
    q_positive_levelset_drift = True
    q_positive_fixed_p_monotone = True
    q_summary_rows = []

    for q_value, result in sorted(q_positive_result_by_q.items()):
        q_crossing_rows = [
            row for row in crossing_row_list
            if abs(float(row["q"]) - q_value) <= DEFAULT_TOLERANCE
        ]
        no_stable_common_crossing = not all(
            row["status"] == "crossing" for row in q_crossing_rows
        )
        exact_crossings = [
            row for row in q_crossing_rows if row["status"] == "crossing"
        ]
        drift_reference_values = [
            float(row["crossing_estimate_p"])
            for row in exact_crossings
        ]
        if len(drift_reference_values) >= 2:
            drift_left = bool(np.all(np.diff(drift_reference_values) <= 0.0))
        else:
            upper_bound_rows = [
                row for row in q_crossing_rows if row["status"] == "upper_bound"
            ]
            drift_left = bool(len(upper_bound_rows) > 0 or len(exact_crossings) > 0)

        q_level_rows = [
            row for row in levelset_row_list
            if abs(float(row["q"]) - q_value) <= DEFAULT_TOLERANCE
        ]
        level_drift_flags = []
        for target_level in DEFAULT_LEVEL_LIST:
            rows = [
                row for row in q_level_rows
                if abs(float(row["target_level"]) - target_level) <= DEFAULT_TOLERANCE
            ]
            p_value_list = np.asarray(
                [float(row["p_value"]) for row in rows],
                dtype=np.float64,
            )
            level_drift_flags.append(bool(np.all(np.diff(p_value_list) <= 0.0)))
        levelset_drift = bool(all(level_drift_flags))

        fixed_entries = fixed_p_summary[f"{q_value:0.4f}"]["selected_p_points"]
        fixed_p_monotone = bool(all(
            entry["monotone_descending_tail_sizes"]
            for entry in fixed_entries
        ))

        q_positive_no_stable_common_crossing &= no_stable_common_crossing
        q_positive_drift_left &= drift_left
        q_positive_levelset_drift &= levelset_drift
        q_positive_fixed_p_monotone &= fixed_p_monotone
        q_summary_rows.append({
            "q": float(q_value),
            "no_stable_common_crossing": no_stable_common_crossing,
            "drift_left": drift_left,
            "levelset_drift": levelset_drift,
            "fixed_p_tail_monotone": fixed_p_monotone,
        })

    q0_crossing_rows = [
        row for row in crossing_row_list
        if float(row["q"]) == 0.0
    ]
    q0_control_recovers_threshold = bool(all(
        row["status"] == "crossing" for row in q0_crossing_rows
    ))

    summary = {
        "q0_control_recovers_threshold": q0_control_recovers_threshold,
        "q_positive_no_stable_common_crossing": q_positive_no_stable_common_crossing,
        "q_positive_crossing_or_bound_drift_left": q_positive_drift_left,
        "q_positive_levelset_drift_left": q_positive_levelset_drift,
        "q_positive_fixed_p_tail_monotone": q_positive_fixed_p_monotone,
        "paper_claim_supported": bool(
            q0_control_recovers_threshold
            and q_positive_no_stable_common_crossing
            and q_positive_drift_left
            and q_positive_levelset_drift
            and q_positive_fixed_p_monotone
        ),
        "q_summary_rows": q_summary_rows,
        "output_files": {
            "crossing_csv": str(crossing_csv_path),
            "levelset_csv": str(levelset_csv_path),
            "fixed_p_summary_json": str(fixed_p_summary_path),
            "summary_json": str(summary_path),
            "q0_control_plot": str(output_dir / "q0_control_sem95.png"),
            "q0_crossing_plot": str(output_dir / "q0_control_crossing_drift.png"),
            "q_positive_pseudocritical_plot": str(
                output_dir / "q_positive_pseudocritical_drift.png"
            ),
            "q_positive_levelset_plot": str(
                output_dir / "q_positive_levelset_drift.png"
            ),
            "fixed_p_size_trend_plot": str(
                output_dir / "fixed_p_size_trends.png"
            ),
            "pairwise_gap_supplement_plot": str(
                output_dir / "q_positive_pairwise_gap_supplement.png"
            ),
        },
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    _plot_q0_control(
        q0_result=q0_result,
        q0_control_plot_path=output_dir / "q0_control_sem95.png",
        q0_crossing_plot_path=output_dir / "q0_control_crossing_drift.png",
        crossing_row_list=crossing_row_list,
    )
    _plot_q_positive_pseudocritical_drift(
        crossing_row_list=crossing_row_list,
        output_path=output_dir / "q_positive_pseudocritical_drift.png",
    )
    _plot_levelset_drift(
        levelset_row_list=levelset_row_list,
        output_path=output_dir / "q_positive_levelset_drift.png",
    )
    _plot_fixed_p_size_trends(
        fixed_p_summary=fixed_p_summary,
        output_path=output_dir / "fixed_p_size_trends.png",
    )
    _plot_pairwise_gap_supplement(
        q_positive_result_by_q=q_positive_result_by_q,
        output_path=output_dir / "q_positive_pairwise_gap_supplement.png",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze final no-threshold evidence for 2D measurement-noise runs."
    )
    parser.add_argument("--q0-baseline", required=True)
    parser.add_argument("--q0-extension", required=True)
    parser.add_argument("--q-positive-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = analyze_no_threshold_evidence(
        q0_baseline_npz_path=args.q0_baseline,
        q0_extension_npz_path=args.q0_extension,
        q_positive_run_root=args.q_positive_run_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
