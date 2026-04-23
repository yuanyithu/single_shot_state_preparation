import json
from pathlib import Path

import numpy as np


DEFAULT_MAX_R_HAT = 1.05
DEFAULT_MIN_EFFECTIVE_SAMPLE_SIZE = 200.0
DEFAULT_MEAN_Q_TOP_SPREAD = 0.03
DEFAULT_MIN_COLD_WINDING_ACCEPTANCE_RATE = 1e-4


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_convergence_summary(
        merged_result,
        lattice_size_list,
        data_error_probability_list,
        syndrome_error_probability):
    q_top_spread_per_disorder_tensor = np.asarray(
        merged_result["q_top_spread_per_disorder_tensor"],
        dtype=np.float64,
    )
    max_r_hat_per_disorder_tensor = np.asarray(
        merged_result["max_r_hat_per_disorder_tensor"],
        dtype=np.float64,
    )
    min_effective_sample_size_per_disorder_tensor = np.asarray(
        merged_result["min_effective_sample_size_per_disorder_tensor"],
        dtype=np.float64,
    )
    chain_winding_acceptance_rate_tensor = np.asarray(
        merged_result[
            "chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor"
        ],
        dtype=np.float64,
    )
    pt_enabled = bool(merged_result.get("pt_enabled", False))
    pt_min_swap_acceptance_rate_per_disorder_tensor = None
    if pt_enabled:
        pt_min_swap_acceptance_rate_per_disorder_tensor = np.asarray(
            merged_result["pt_min_swap_acceptance_rate_per_disorder_tensor"],
            dtype=np.float64,
        )

    num_sizes = len(lattice_size_list)
    num_points = len(data_error_probability_list)
    converged_mask_matrix = np.zeros((num_sizes, num_points), dtype=bool)
    point_summaries = []

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        for point_index, data_error_probability in enumerate(
                data_error_probability_list):
            point_q_top_spread_values = q_top_spread_per_disorder_tensor[
                lattice_index,
                point_index,
            ]
            point_max_r_hat_values = max_r_hat_per_disorder_tensor[
                lattice_index,
                point_index,
            ]
            point_min_effective_sample_size_values = (
                min_effective_sample_size_per_disorder_tensor[
                    lattice_index,
                    point_index,
                ]
            )
            point_winding_acceptance_values = chain_winding_acceptance_rate_tensor[
                lattice_index,
                point_index,
            ]
            mean_q_top_spread = float(np.mean(point_q_top_spread_values))
            finite_r_hat_values = point_max_r_hat_values[
                np.isfinite(point_max_r_hat_values)
            ]
            if finite_r_hat_values.size == 0:
                max_r_hat = np.nan
            else:
                max_r_hat = float(np.max(finite_r_hat_values))
            min_effective_sample_size = float(
                np.min(point_min_effective_sample_size_values)
            )
            mean_cold_winding_acceptance_rate = float(
                np.mean(point_winding_acceptance_values)
            )
            failed_checks = []
            if not np.isfinite(max_r_hat) or max_r_hat >= DEFAULT_MAX_R_HAT:
                failed_checks.append(
                    f"max_r_hat>={DEFAULT_MAX_R_HAT:0.2f}"
                )
            if (
                    not np.isfinite(min_effective_sample_size)
                    or min_effective_sample_size <= DEFAULT_MIN_EFFECTIVE_SAMPLE_SIZE):
                failed_checks.append(
                    f"min_effective_sample_size<={DEFAULT_MIN_EFFECTIVE_SAMPLE_SIZE:0.0f}"
                )
            if mean_q_top_spread >= DEFAULT_MEAN_Q_TOP_SPREAD:
                failed_checks.append(
                    f"mean_q_top_spread>={DEFAULT_MEAN_Q_TOP_SPREAD:0.3f}"
                )

            mean_pt_min_swap_acceptance_rate = None
            if pt_enabled:
                mean_pt_min_swap_acceptance_rate = float(np.mean(
                    pt_min_swap_acceptance_rate_per_disorder_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                if (
                        mean_cold_winding_acceptance_rate
                        <= DEFAULT_MIN_COLD_WINDING_ACCEPTANCE_RATE
                        and mean_pt_min_swap_acceptance_rate <= 0.0):
                    failed_checks.append(
                        "pt_transport_insufficient"
                    )

            passed = len(failed_checks) == 0
            converged_mask_matrix[lattice_index, point_index] = passed
            point_summary = {
                "lattice_index": int(lattice_index),
                "point_index": int(point_index),
                "lattice_size": int(lattice_size),
                "data_error_probability": float(data_error_probability),
                "syndrome_error_probability": float(
                    syndrome_error_probability
                ),
                "passed": bool(passed),
                "failed_checks": failed_checks,
                "metrics": {
                    "mean_q_top_spread": mean_q_top_spread,
                    "max_r_hat": max_r_hat,
                    "min_effective_sample_size": (
                        min_effective_sample_size
                    ),
                    "mean_cold_winding_acceptance_rate": (
                        mean_cold_winding_acceptance_rate
                    ),
                },
            }
            if pt_enabled:
                point_summary["metrics"][
                    "mean_pt_min_swap_acceptance_rate"
                ] = mean_pt_min_swap_acceptance_rate
            point_summaries.append(point_summary)

    return {
        "thresholds": {
            "max_r_hat": DEFAULT_MAX_R_HAT,
            "min_effective_sample_size": (
                DEFAULT_MIN_EFFECTIVE_SAMPLE_SIZE
            ),
            "mean_q_top_spread": DEFAULT_MEAN_Q_TOP_SPREAD,
            "min_cold_winding_acceptance_rate": (
                DEFAULT_MIN_COLD_WINDING_ACCEPTANCE_RATE
            ),
        },
        "pt_enabled": pt_enabled,
        "syndrome_error_probability": float(syndrome_error_probability),
        "converged_mask_matrix": converged_mask_matrix,
        "num_passed_points": int(np.count_nonzero(converged_mask_matrix)),
        "num_total_points": int(converged_mask_matrix.size),
        "points": point_summaries,
    }


def write_convergence_summary_json(output_path, convergence_summary):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            convergence_summary,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
