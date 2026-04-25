import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

import numpy as np

from analyze_threshold_crossing import analyze_threshold_crossing
from mcmc_convergence_gate import (
    build_convergence_summary,
    write_convergence_summary_json,
)
from plot_scan_results import plot_scan_result


DISORDER_AXIS = 2
CONCATENATED_FIELD_SUFFIXES = (
    "_per_disorder_tensor",
    "_per_disorder_per_start_replica_tensor",
    "_per_disorder_per_start_tensor",
)


def _load_npz_as_dict(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _as_python_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _require_same_array(name, first, current, path):
    if not np.array_equal(np.asarray(first), np.asarray(current)):
        raise ValueError(f"{name} differs in {path}")


def _require_same_scalar(name, first, current, path):
    if _as_python_scalar(first) != _as_python_scalar(current):
        raise ValueError(f"{name} differs in {path}")


def _is_disorder_tensor(name, value, num_sizes, num_points, num_disorder):
    if not name.endswith(CONCATENATED_FIELD_SUFFIXES):
        return False
    array = np.asarray(value)
    return (
        array.ndim >= 3
        and array.shape[0] == num_sizes
        and array.shape[1] == num_points
        and array.shape[DISORDER_AXIS] == num_disorder
    )


def _std_over_disorder(values):
    if values.shape[DISORDER_AXIS] <= 1:
        return np.zeros(values.shape[:2], dtype=np.float64)
    return np.std(values, axis=DISORDER_AXIS, ddof=1)


def _recompute_curve_fields(pooled):
    disorder_q_top_values = np.asarray(
        pooled["disorder_q_top_values_tensor"],
        dtype=np.float64,
    )
    acceptance_values = np.asarray(
        pooled["average_acceptance_rate_per_disorder_tensor"],
        dtype=np.float64,
    )
    pooled["q_top_curve_matrix"] = np.mean(
        disorder_q_top_values,
        axis=DISORDER_AXIS,
    )
    pooled["q_top_std_curve_matrix"] = _std_over_disorder(disorder_q_top_values)
    pooled["average_acceptance_rate_curve_matrix"] = np.mean(
        acceptance_values,
        axis=DISORDER_AXIS,
    )

    if "q_top_spread_per_disorder_tensor" in pooled:
        pooled["mean_q_top_spread_curve_matrix"] = np.mean(
            pooled["q_top_spread_per_disorder_tensor"],
            axis=DISORDER_AXIS,
        )
        pooled["mean_m_u_spread_linf_curve_matrix"] = np.mean(
            pooled["m_u_spread_linf_per_disorder_tensor"],
            axis=DISORDER_AXIS,
        )
        finite_r_hat = np.asarray(
            pooled["max_r_hat_per_disorder_tensor"],
            dtype=np.float64,
        )
        max_r_hat = np.empty(finite_r_hat.shape[:2], dtype=np.float64)
        for lattice_index in range(finite_r_hat.shape[0]):
            for point_index in range(finite_r_hat.shape[1]):
                point_values = finite_r_hat[lattice_index, point_index]
                point_values = point_values[np.isfinite(point_values)]
                max_r_hat[lattice_index, point_index] = (
                    np.nan if point_values.size == 0 else np.max(point_values)
                )
        pooled["max_r_hat_curve_matrix"] = max_r_hat
        pooled["min_effective_sample_size_curve_matrix"] = np.min(
            pooled["min_effective_sample_size_per_disorder_tensor"],
            axis=DISORDER_AXIS,
        )
        pooled["mean_num_chains_that_never_flipped_sector_curve_matrix"] = (
            np.mean(
                pooled[
                    "num_chains_that_never_flipped_sector_per_disorder_tensor"
                ],
                axis=DISORDER_AXIS,
            )
        )
        pooled["mean_cold_winding_acceptance_rate_curve_matrix"] = np.mean(
            pooled[
                "chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor"
            ],
            axis=(DISORDER_AXIS, 3, 4),
        )

    if bool(_as_python_scalar(pooled.get("pt_enabled", False))):
        pooled["mean_pt_min_swap_acceptance_rate_curve_matrix"] = np.mean(
            pooled["pt_min_swap_acceptance_rate_per_disorder_tensor"],
            axis=DISORDER_AXIS,
        )
        pooled["mean_pt_mean_swap_acceptance_rate_curve_matrix"] = np.mean(
            pooled["pt_mean_swap_acceptance_rate_per_disorder_tensor"],
            axis=DISORDER_AXIS,
        )


def _build_pooled_result(input_paths):
    if len(input_paths) == 0:
        raise ValueError("at least one input is required")
    loaded_runs = [_load_npz_as_dict(path) for path in input_paths]
    first = loaded_runs[0]

    lattice_size_list = np.asarray(first["lattice_size_list"], dtype=np.int64)
    probability_list = np.asarray(
        first["data_error_probability_list"],
        dtype=np.float64,
    )
    num_sizes = lattice_size_list.size
    num_points = probability_list.size
    first_num_disorder = int(first["num_disorder_samples"])

    array_invariants = [
        "lattice_size_list",
        "num_qubits_list",
        "num_logical_qubits_list",
        "effective_num_burn_in_sweeps_list",
        "data_error_probability_list",
        "start_sector_labels",
    ]
    scalar_invariants = [
        "code_family",
        "code_type",
        "syndrome_error_probability",
        "num_burn_in_sweeps",
        "num_sweeps_between_measurements",
        "num_measurements_per_disorder",
        "q0_num_start_chains",
        "num_start_chains",
        "num_replicas_per_start",
        "num_zero_syndrome_sweeps_per_cycle",
        "winding_repeat_factor",
        "common_random_disorder_across_p",
        "burn_in_scaling_reference_num_qubits",
        "pt_enabled",
        "pt_p_hot",
        "pt_num_temperatures",
    ]

    for path, current in zip(input_paths[1:], loaded_runs[1:]):
        for name in array_invariants:
            if name in first or name in current:
                if name not in first or name not in current:
                    raise ValueError(f"{name} missing from one run")
                _require_same_array(name, first[name], current[name], path)
        for name in scalar_invariants:
            if name in first or name in current:
                if name not in first or name not in current:
                    raise ValueError(f"{name} missing from one run")
                _require_same_scalar(name, first[name], current[name], path)

    pooled = {}
    for key, value in first.items():
        if key in {
            "converged_mask_matrix",
            "convergence_summary_path",
            "source_chunks_dir",
            "seed_base",
            "git_commit_sha",
            "chunk_size",
            "num_chunks_per_point",
            "num_disorder_samples",
        }:
            continue
        if _is_disorder_tensor(key, value, num_sizes, num_points, first_num_disorder):
            tensor_list = []
            for path, current in zip(input_paths, loaded_runs):
                current_num_disorder = int(current["num_disorder_samples"])
                if not _is_disorder_tensor(
                    key,
                    current[key],
                    num_sizes,
                    num_points,
                    current_num_disorder,
                ):
                    raise ValueError(f"{key} has incompatible shape in {path}")
                tensor_list.append(np.asarray(current[key]))
            pooled[key] = np.concatenate(tensor_list, axis=DISORDER_AXIS)
        else:
            pooled[key] = np.asarray(value)

    total_num_disorder = sum(int(run["num_disorder_samples"]) for run in loaded_runs)
    pooled["num_disorder_samples"] = np.int64(total_num_disorder)
    pooled["seed_base"] = np.array(
        ",".join(str(_as_python_scalar(run["seed_base"])) for run in loaded_runs)
    )
    pooled["git_commit_sha"] = np.array(
        ",".join(str(_as_python_scalar(run["git_commit_sha"])) for run in loaded_runs)
    )
    pooled["source_npz_paths"] = np.asarray([str(path) for path in input_paths])
    pooled["pooled_num_source_runs"] = np.int64(len(input_paths))
    pooled["chunk_size"] = np.array(
        ",".join(str(_as_python_scalar(run["chunk_size"])) for run in loaded_runs)
    )
    pooled["num_chunks_per_point"] = np.array(
        ",".join(
            str(_as_python_scalar(run["num_chunks_per_point"]))
            for run in loaded_runs
        )
    )
    _recompute_curve_fields(pooled)
    return pooled


def pool_independent_runs(input_paths, output_dir, output_stem):
    input_paths = [Path(path).resolve() for path in input_paths]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.npz"

    pooled = _build_pooled_result(input_paths)
    convergence_summary_path = output_dir / f"{output_stem}_convergence.json"
    convergence_summary = build_convergence_summary(
        merged_result=pooled,
        lattice_size_list=np.asarray(pooled["lattice_size_list"], dtype=np.int64),
        data_error_probability_list=np.asarray(
            pooled["data_error_probability_list"],
            dtype=np.float64,
        ),
        syndrome_error_probability=float(pooled["syndrome_error_probability"]),
    )
    pooled["converged_mask_matrix"] = convergence_summary["converged_mask_matrix"]
    pooled["convergence_summary_path"] = np.array(str(convergence_summary_path))
    np.savez(output_path, **pooled)
    write_convergence_summary_json(convergence_summary_path, convergence_summary)

    plot_path = output_dir / f"{output_stem}.png"
    plot_scan_result(output_path, plot_path)
    threshold_summary = analyze_threshold_crossing(
        input_path=output_path,
        output_dir=output_dir,
        output_stem=output_stem,
        summary_path=output_dir / f"{output_stem}_threshold_summary.json",
    )
    return {
        "output_path": str(output_path),
        "plot_path": str(plot_path),
        "convergence_summary_path": str(convergence_summary_path),
        "threshold_summary_path": threshold_summary["summary_path"],
        "num_disorder_samples": int(pooled["num_disorder_samples"]),
        "primary_crossing_window_hit": threshold_summary[
            "primary_crossing_window_hit"
        ],
        "recommended_server_window": threshold_summary[
            "recommended_server_window"
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pool independent multi-size threshold scan NPZ files."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input merged scan_result*.npz. Repeat once per independent run.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    summary = pool_independent_runs(
        input_paths=args.input,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
