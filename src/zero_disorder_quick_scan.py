import argparse
import json
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_toric_code_examples import (
    SUPPORTED_CODE_FAMILIES,
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from main import run_disorder_average_simulation


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS = 18
DEFAULT_Q_POSITIVE_NUM_START_CHAINS = 8
DEFAULT_Q0_NUM_START_CHAINS = 8
DEFAULT_NUM_REPLICAS_PER_START = 1
DEFAULT_PT_P_HOT = 0.44
DEFAULT_PT_NUM_TEMPERATURES = 7
DEFAULT_PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS = 1
DEFAULT_NUM_BURN_IN_SWEEPS = 600
DEFAULT_NUM_SWEEPS_BETWEEN_MEASUREMENTS = 4
DEFAULT_NUM_MEASUREMENTS_PER_POINT = 256
DEFAULT_LOCAL_WORKERS = 6
DEFAULT_ND3_WORKERS = 24


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamp_tag():
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _log(message):
    print(f"[{_timestamp()}] {message}", flush=True)


def _parse_int_csv(csv_value):
    return [int(value.strip()) for value in csv_value.split(",") if value.strip()]


def _parse_float_csv(csv_value):
    return [
        float(value.strip())
        for value in csv_value.split(",")
        if value.strip()
    ]


def _csv_from_float_values(values):
    return ",".join(f"{float(value):0.4f}" for value in values)


def _csv_from_int_values(values):
    return ",".join(str(int(value)) for value in values)


def _effective_num_burn_in_sweeps(num_burn_in_sweeps, num_qubits):
    return int(np.ceil(
        num_burn_in_sweeps
        * (num_qubits / DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS)
    ))


def _build_default_run_root():
    return (
        PROJECT_ROOT
        / "data"
        / "3d_toric_code"
        / "with_measurement_noise"
        / "zero_disorder_quick_scan_local"
        / f"zero_disorder_quick_scan_{_timestamp_tag()}"
    )


def _build_default_output_stem(scan_kind, fixed_q, fixed_p):
    if scan_kind == "fixed-q-scan":
        return f"zero_disorder_fixed_q_q{float(fixed_q):0.4f}".replace(".", "p")
    return f"zero_disorder_fixed_p_p{float(fixed_p):0.4f}".replace(".", "p")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Quick q_top scans for a single all-zero disorder sample.",
    )
    parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        default="3d_toric",
    )
    parser.add_argument(
        "--lattice-sizes",
        default="3,4,5",
    )
    parser.add_argument(
        "--scan-kind",
        choices=("fixed-q-scan", "fixed-p-scan"),
        required=True,
    )
    parser.add_argument("--fixed-q", type=float, default=None)
    parser.add_argument("--fixed-p", type=float, default=None)
    parser.add_argument(
        "--p-values",
        default=_csv_from_float_values(np.arange(0.10, 0.30 + 0.0001, 0.02)),
    )
    parser.add_argument(
        "--q-values",
        default=_csv_from_float_values(np.arange(0.00, 0.20 + 0.0001, 0.02)),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--num-burn-in-sweeps",
        type=int,
        default=DEFAULT_NUM_BURN_IN_SWEEPS,
    )
    parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        default=DEFAULT_NUM_SWEEPS_BETWEEN_MEASUREMENTS,
    )
    parser.add_argument(
        "--num-measurements-per-point",
        type=int,
        default=DEFAULT_NUM_MEASUREMENTS_PER_POINT,
    )
    parser.add_argument(
        "--q0-num-start-chains",
        type=int,
        default=DEFAULT_Q0_NUM_START_CHAINS,
    )
    parser.add_argument(
        "--num-start-chains",
        type=int,
        default=DEFAULT_Q_POSITIVE_NUM_START_CHAINS,
    )
    parser.add_argument(
        "--num-replicas-per-start",
        type=int,
        default=DEFAULT_NUM_REPLICAS_PER_START,
    )
    parser.add_argument("--pt-p-hot", type=float, default=DEFAULT_PT_P_HOT)
    parser.add_argument(
        "--pt-num-temperatures",
        type=int,
        default=DEFAULT_PT_NUM_TEMPERATURES,
    )
    parser.add_argument(
        "--pt-swap-attempt-every-num-sweeps",
        type=int,
        default=DEFAULT_PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS,
    )
    parser.add_argument("--seed-base", type=int, default=20260423)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def _resolve_scan_axis(args):
    lattice_size_list = _parse_int_csv(args.lattice_sizes)
    if not lattice_size_list:
        raise ValueError("lattice_sizes must be non-empty")
    if args.code_family != "3d_toric":
        raise ValueError("zero_disorder_quick_scan currently supports 3d_toric only")

    if args.scan_kind == "fixed-q-scan":
        if args.fixed_q is None:
            raise ValueError("--fixed-q is required for fixed-q-scan")
        x_axis_name = "data_error_probability"
        x_axis_values = _parse_float_csv(args.p_values)
        fixed_q = float(args.fixed_q)
        fixed_p = None
    else:
        if args.fixed_p is None:
            raise ValueError("--fixed-p is required for fixed-p-scan")
        x_axis_name = "syndrome_error_probability"
        x_axis_values = _parse_float_csv(args.q_values)
        fixed_q = None
        fixed_p = float(args.fixed_p)

    if not x_axis_values:
        raise ValueError("scan axis values must be non-empty")
    return lattice_size_list, x_axis_name, x_axis_values, fixed_q, fixed_p


def _resolve_workers(requested_workers):
    if requested_workers is not None:
        return int(requested_workers)
    hostname = os.uname().nodename.lower()
    if hostname.startswith("nd-3") or hostname == "nd-3":
        return DEFAULT_ND3_WORKERS
    return DEFAULT_LOCAL_WORKERS


def _build_point_tasks(
        lattice_size_list,
        x_axis_values,
        x_axis_name,
        fixed_q,
        fixed_p,
        seed_base,
        num_burn_in_sweeps):
    task_data_list = []
    effective_num_burn_in_sweeps_list = np.empty(
        len(lattice_size_list),
        dtype=np.int64,
    )
    num_qubits_list = np.empty(len(lattice_size_list), dtype=np.int64)
    num_logical_qubits_list = np.empty(len(lattice_size_list), dtype=np.int64)

    for lattice_index, lattice_size in enumerate(lattice_size_list):
        parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
            code_family="3d_toric",
            lattice_size=lattice_size,
        )
        num_checks, num_qubits = parity_check_matrix.shape
        num_qubits_list[lattice_index] = num_qubits
        num_logical_qubits_list[lattice_index] = dual_logical_z_basis.shape[0]
        effective_num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            num_burn_in_sweeps=num_burn_in_sweeps,
            num_qubits=num_qubits,
        )
        effective_num_burn_in_sweeps_list[lattice_index] = (
            effective_num_burn_in_sweeps
        )

        for point_index, axis_value in enumerate(x_axis_values):
            if x_axis_name == "data_error_probability":
                data_error_probability = float(axis_value)
                syndrome_error_probability = float(fixed_q)
            else:
                data_error_probability = float(fixed_p)
                syndrome_error_probability = float(axis_value)
            task_data_list.append({
                "lattice_index": int(lattice_index),
                "point_index": int(point_index),
                "lattice_size": int(lattice_size),
                "num_checks": int(num_checks),
                "num_qubits": int(num_qubits),
                "data_error_probability": data_error_probability,
                "syndrome_error_probability": syndrome_error_probability,
                "effective_num_burn_in_sweeps": int(
                    effective_num_burn_in_sweeps
                ),
                "seed": int(
                    seed_base
                    + 1000003 * lattice_index
                    + 1009 * point_index
                ),
            })

    return (
        task_data_list,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list,
    )


def _build_multiprocessing_context():
    available_start_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_start_methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _compute_worker_count(requested_workers, num_tasks):
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(int(requested_workers), int(num_tasks), int(cpu_count)))


def _run_single_point_task(task_data):
    parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
        code_family="3d_toric",
        lattice_size=task_data["lattice_size"],
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family="3d_toric",
        lattice_size=task_data["lattice_size"],
    )
    syndrome_uniform_values = np.ones(
        (1, task_data["num_checks"]),
        dtype=np.float64,
    )
    data_uniform_values = np.ones(
        (1, task_data["num_qubits"]),
        dtype=np.float64,
    )
    observed_syndrome_bits = (
        syndrome_uniform_values[0] < task_data["syndrome_error_probability"]
    )
    disorder_data_error_bits = (
        data_uniform_values[0] < task_data["data_error_probability"]
    )
    if np.any(observed_syndrome_bits):
        raise AssertionError("all-zero disorder path produced non-zero syndrome bits")
    if np.any(disorder_data_error_bits):
        raise AssertionError("all-zero disorder path produced non-zero disorder bits")

    use_parallel_tempering = task_data["syndrome_error_probability"] > 0.0
    simulation_result = run_disorder_average_simulation(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        syndrome_error_probability=task_data["syndrome_error_probability"],
        data_error_probability=task_data["data_error_probability"],
        num_disorder_samples=1,
        num_burn_in_sweeps=task_data["effective_num_burn_in_sweeps"],
        num_sweeps_between_measurements=(
            task_data["num_sweeps_between_measurements"]
        ),
        num_measurements_per_disorder=(
            task_data["num_measurements_per_point"]
        ),
        seed=task_data["seed"],
        zero_syndrome_move_data=zero_syndrome_move_data,
        q0_num_start_chains=task_data["q0_num_start_chains"],
        num_start_chains=task_data["num_start_chains"],
        num_replicas_per_start=task_data["num_replicas_per_start"],
        pt_p_hot=(
            task_data["pt_p_hot"] if use_parallel_tempering else None
        ),
        pt_num_temperatures=(
            task_data["pt_num_temperatures"]
            if use_parallel_tempering else None
        ),
        pt_swap_attempt_every_num_sweeps=(
            task_data["pt_swap_attempt_every_num_sweeps"]
        ),
        precomputed_syndrome_uniform_values_per_disorder=(
            syndrome_uniform_values
        ),
        precomputed_data_uniform_values_per_disorder=(
            data_uniform_values
        ),
    )

    point_result = {
        "lattice_index": int(task_data["lattice_index"]),
        "point_index": int(task_data["point_index"]),
        "q_top": float(simulation_result["disorder_average_q_top"]),
        "average_acceptance_rate": float(
            np.mean(simulation_result["average_acceptance_rate_per_disorder"])
        ),
        "observed_syndrome_weight": 0,
        "disorder_data_weight": 0,
    }
    if use_parallel_tempering:
        point_result["q_top_spread"] = float(
            simulation_result["q_top_spread_per_disorder"][0]
        )
        point_result["max_r_hat"] = float(
            simulation_result["max_r_hat_per_disorder"][0]
        )
        point_result["min_effective_sample_size"] = float(
            simulation_result["min_effective_sample_size_per_disorder"][0]
        )
        point_result["mean_cold_winding_acceptance_rate"] = float(
            np.mean(
                simulation_result[
                    "chain_winding_acceptance_rate_per_disorder_per_start_replica"
                ][0]
            )
        )
        if bool(simulation_result["pt_enabled"]):
            point_result["pt_min_swap_acceptance_rate"] = float(
                simulation_result["pt_min_swap_acceptance_rate_per_disorder"][0]
            )
            point_result["pt_mean_swap_acceptance_rate"] = float(
                simulation_result["pt_mean_swap_acceptance_rate_per_disorder"][0]
            )
    else:
        point_result["q0_q_top_spread"] = float(
            simulation_result["q0_q_top_spread_per_disorder"][0]
        )
        point_result["q0_m_u_spread_linf"] = float(
            simulation_result["q0_m_u_spread_linf_per_disorder"][0]
        )
    return point_result


def _build_output_paths(run_root, output_stem):
    output_root = Path(run_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return (
        output_root / f"{output_stem}.npz",
        output_root / f"{output_stem}.json",
        output_root / f"{output_stem}.png",
    )


def _plot_results(
        output_path,
        x_axis_values,
        x_axis_name,
        lattice_size_list,
        q_top_curve_matrix,
        fixed_q,
        fixed_p,
        disorder_mode):
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(8.0, 5.2),
        constrained_layout=True,
    )
    for lattice_index, lattice_size in enumerate(lattice_size_list):
        axis.plot(
            x_axis_values,
            q_top_curve_matrix[lattice_index],
            marker="o",
            linewidth=1.5,
            label=f"L={int(lattice_size)}",
        )

    if x_axis_name == "data_error_probability":
        xlabel = "data error probability p"
        fixed_axis_line = f"fixed q={fixed_q:0.4f}"
    else:
        xlabel = "syndrome error probability q"
        fixed_axis_line = f"fixed p={fixed_p:0.4f}"
    axis.set_xlabel(xlabel)
    axis.set_ylabel("q_top")
    axis.set_title(
        "3d toric zero-disorder quick scan\n"
        f"{fixed_axis_line}, {disorder_mode}, num_disorder_samples=1"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _build_summary_json(
        args,
        x_axis_name,
        x_axis_values,
        lattice_size_list,
        q_top_curve_matrix,
        average_acceptance_rate_curve_matrix,
        q_top_spread_curve_matrix,
        max_r_hat_curve_matrix,
        min_effective_sample_size_curve_matrix,
        q0_q_top_spread_curve_matrix,
        q0_m_u_spread_linf_curve_matrix,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list):
    summary = {
        "created_at": _timestamp(),
        "code_family": args.code_family,
        "scan_kind": args.scan_kind,
        "disorder_mode": "all_zero_single_sample",
        "disorder_description": "all-zero disorder, num_disorder_samples = 1, no disorder averaging",
        "num_disorder_samples": 1,
        "lattice_size_list": [int(value) for value in lattice_size_list],
        "num_qubits_list": [int(value) for value in num_qubits_list],
        "num_logical_qubits_list": [
            int(value) for value in num_logical_qubits_list
        ],
        "effective_num_burn_in_sweeps_list": [
            int(value) for value in effective_num_burn_in_sweeps_list
        ],
        "x_axis_name": x_axis_name,
        "x_axis_values": [float(value) for value in x_axis_values],
        "q_top_curve_matrix": q_top_curve_matrix.tolist(),
        "average_acceptance_rate_curve_matrix": (
            average_acceptance_rate_curve_matrix.tolist()
        ),
        "fixed_q": None if args.fixed_q is None else float(args.fixed_q),
        "fixed_p": None if args.fixed_p is None else float(args.fixed_p),
        "num_burn_in_sweeps_base": int(args.num_burn_in_sweeps),
        "num_sweeps_between_measurements": int(
            args.num_sweeps_between_measurements
        ),
        "num_measurements_per_point": int(args.num_measurements_per_point),
        "q0_num_start_chains": int(args.q0_num_start_chains),
        "num_start_chains": int(args.num_start_chains),
        "num_replicas_per_start": int(args.num_replicas_per_start),
        "pt_p_hot": float(args.pt_p_hot),
        "pt_num_temperatures": int(args.pt_num_temperatures),
        "pt_swap_attempt_every_num_sweeps": int(
            args.pt_swap_attempt_every_num_sweeps
        ),
    }
    if q_top_spread_curve_matrix is not None:
        summary["q_top_spread_curve_matrix"] = (
            q_top_spread_curve_matrix.tolist()
        )
        summary["max_r_hat_curve_matrix"] = max_r_hat_curve_matrix.tolist()
        summary["min_effective_sample_size_curve_matrix"] = (
            min_effective_sample_size_curve_matrix.tolist()
        )
    if q0_q_top_spread_curve_matrix is not None:
        summary["q0_q_top_spread_curve_matrix"] = (
            q0_q_top_spread_curve_matrix.tolist()
        )
        summary["q0_m_u_spread_linf_curve_matrix"] = (
            q0_m_u_spread_linf_curve_matrix.tolist()
        )
    return summary


def _save_npz(
        output_path,
        args,
        x_axis_name,
        x_axis_values,
        lattice_size_list,
        q_top_curve_matrix,
        average_acceptance_rate_curve_matrix,
        q_top_spread_curve_matrix,
        max_r_hat_curve_matrix,
        min_effective_sample_size_curve_matrix,
        q0_q_top_spread_curve_matrix,
        q0_m_u_spread_linf_curve_matrix,
        observed_syndrome_weight_matrix,
        disorder_data_weight_matrix,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list):
    npz_payload = {
        "code_family": np.array(args.code_family),
        "code_type": np.array(args.code_family),
        "scan_kind": np.array(args.scan_kind),
        "disorder_mode": np.array("all_zero_single_sample"),
        "disorder_description": np.array(
            "all-zero disorder, num_disorder_samples = 1, no disorder averaging"
        ),
        "num_disorder_samples": np.int64(1),
        "lattice_size_list": np.asarray(lattice_size_list, dtype=np.int64),
        "num_qubits_list": np.asarray(num_qubits_list, dtype=np.int64),
        "num_logical_qubits_list": np.asarray(
            num_logical_qubits_list,
            dtype=np.int64,
        ),
        "effective_num_burn_in_sweeps_list": np.asarray(
            effective_num_burn_in_sweeps_list,
            dtype=np.int64,
        ),
        "x_axis_name": np.array(x_axis_name),
        "x_axis_values": np.asarray(x_axis_values, dtype=np.float64),
        "q_top_curve_matrix": np.asarray(q_top_curve_matrix, dtype=np.float64),
        "q_top_std_curve_matrix": np.zeros_like(
            q_top_curve_matrix,
            dtype=np.float64,
        ),
        "average_acceptance_rate_curve_matrix": np.asarray(
            average_acceptance_rate_curve_matrix,
            dtype=np.float64,
        ),
        "observed_syndrome_weight_matrix": np.asarray(
            observed_syndrome_weight_matrix,
            dtype=np.int64,
        ),
        "disorder_data_weight_matrix": np.asarray(
            disorder_data_weight_matrix,
            dtype=np.int64,
        ),
        "fixed_q": np.float64(np.nan if args.fixed_q is None else args.fixed_q),
        "fixed_p": np.float64(np.nan if args.fixed_p is None else args.fixed_p),
        "num_burn_in_sweeps": np.int64(args.num_burn_in_sweeps),
        "num_sweeps_between_measurements": np.int64(
            args.num_sweeps_between_measurements
        ),
        "num_measurements_per_point": np.int64(
            args.num_measurements_per_point
        ),
        "q0_num_start_chains": np.int64(args.q0_num_start_chains),
        "num_start_chains": np.int64(args.num_start_chains),
        "num_replicas_per_start": np.int64(args.num_replicas_per_start),
        "pt_p_hot": np.float64(args.pt_p_hot),
        "pt_num_temperatures": np.int64(args.pt_num_temperatures),
        "pt_swap_attempt_every_num_sweeps": np.int64(
            args.pt_swap_attempt_every_num_sweeps
        ),
    }
    if x_axis_name == "data_error_probability":
        npz_payload["data_error_probability_list"] = np.asarray(
            x_axis_values,
            dtype=np.float64,
        )
        npz_payload["syndrome_error_probability"] = np.float64(args.fixed_q)
    else:
        npz_payload["syndrome_error_probability_list"] = np.asarray(
            x_axis_values,
            dtype=np.float64,
        )
        npz_payload["data_error_probability"] = np.float64(args.fixed_p)
    if q_top_spread_curve_matrix is not None:
        npz_payload["q_top_spread_curve_matrix"] = np.asarray(
            q_top_spread_curve_matrix,
            dtype=np.float64,
        )
        npz_payload["max_r_hat_curve_matrix"] = np.asarray(
            max_r_hat_curve_matrix,
            dtype=np.float64,
        )
        npz_payload["min_effective_sample_size_curve_matrix"] = np.asarray(
            min_effective_sample_size_curve_matrix,
            dtype=np.float64,
        )
    if q0_q_top_spread_curve_matrix is not None:
        npz_payload["q0_q_top_spread_curve_matrix"] = np.asarray(
            q0_q_top_spread_curve_matrix,
            dtype=np.float64,
        )
        npz_payload["q0_m_u_spread_linf_curve_matrix"] = np.asarray(
            q0_m_u_spread_linf_curve_matrix,
            dtype=np.float64,
        )
    np.savez(output_path, **npz_payload)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    lattice_size_list, x_axis_name, x_axis_values, fixed_q, fixed_p = (
        _resolve_scan_axis(args)
    )
    workers = _resolve_workers(args.workers)
    run_root = (
        _build_default_run_root()
        if args.run_root is None
        else Path(args.run_root).expanduser().resolve()
    )
    output_stem = (
        _build_default_output_stem(args.scan_kind, fixed_q, fixed_p)
        if args.output_stem is None
        else str(args.output_stem)
    )
    npz_output_path, json_output_path, png_output_path = _build_output_paths(
        run_root=run_root,
        output_stem=output_stem,
    )
    (
        task_data_list,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list,
    ) = _build_point_tasks(
        lattice_size_list=lattice_size_list,
        x_axis_values=x_axis_values,
        x_axis_name=x_axis_name,
        fixed_q=fixed_q,
        fixed_p=fixed_p,
        seed_base=args.seed_base,
        num_burn_in_sweeps=args.num_burn_in_sweeps,
    )

    for task_data in task_data_list:
        task_data["num_sweeps_between_measurements"] = int(
            args.num_sweeps_between_measurements
        )
        task_data["num_measurements_per_point"] = int(
            args.num_measurements_per_point
        )
        task_data["q0_num_start_chains"] = int(args.q0_num_start_chains)
        task_data["num_start_chains"] = int(args.num_start_chains)
        task_data["num_replicas_per_start"] = int(
            args.num_replicas_per_start
        )
        task_data["pt_p_hot"] = float(args.pt_p_hot)
        task_data["pt_num_temperatures"] = int(args.pt_num_temperatures)
        task_data["pt_swap_attempt_every_num_sweeps"] = int(
            args.pt_swap_attempt_every_num_sweeps
        )

    num_sizes = len(lattice_size_list)
    num_points = len(x_axis_values)
    q_top_curve_matrix = np.empty((num_sizes, num_points), dtype=np.float64)
    average_acceptance_rate_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    observed_syndrome_weight_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.int64,
    )
    disorder_data_weight_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.int64,
    )
    q_top_spread_curve_matrix = None
    max_r_hat_curve_matrix = None
    min_effective_sample_size_curve_matrix = None
    q0_q_top_spread_curve_matrix = None
    q0_m_u_spread_linf_curve_matrix = None

    if args.scan_kind == "fixed-q-scan":
        q_top_spread_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        max_r_hat_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        min_effective_sample_size_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
    else:
        q0_q_top_spread_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        q0_m_u_spread_linf_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )

    worker_count = _compute_worker_count(workers, len(task_data_list))
    multiprocessing_context = _build_multiprocessing_context()
    _log(
        f"Starting zero-disorder quick scan: {args.scan_kind}, "
        f"{len(task_data_list)} points, workers={worker_count}"
    )

    completed_count = 0
    with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing_context) as executor:
        future_map = {
            executor.submit(_run_single_point_task, task_data): task_data
            for task_data in task_data_list
        }
        for future in as_completed(future_map):
            task_result = future.result()
            lattice_index = task_result["lattice_index"]
            point_index = task_result["point_index"]
            q_top_curve_matrix[lattice_index, point_index] = (
                task_result["q_top"]
            )
            average_acceptance_rate_curve_matrix[lattice_index, point_index] = (
                task_result["average_acceptance_rate"]
            )
            observed_syndrome_weight_matrix[lattice_index, point_index] = (
                task_result["observed_syndrome_weight"]
            )
            disorder_data_weight_matrix[lattice_index, point_index] = (
                task_result["disorder_data_weight"]
            )
            if q_top_spread_curve_matrix is not None:
                q_top_spread_curve_matrix[lattice_index, point_index] = (
                    task_result["q_top_spread"]
                )
                max_r_hat_curve_matrix[lattice_index, point_index] = (
                    task_result["max_r_hat"]
                )
                min_effective_sample_size_curve_matrix[
                    lattice_index,
                    point_index,
                ] = task_result["min_effective_sample_size"]
            if q0_q_top_spread_curve_matrix is not None:
                q0_q_top_spread_curve_matrix[lattice_index, point_index] = (
                    task_result["q0_q_top_spread"]
                )
                q0_m_u_spread_linf_curve_matrix[
                    lattice_index,
                    point_index,
                ] = task_result["q0_m_u_spread_linf"]
            completed_count += 1
            _log(
                f"Completed point {completed_count}/{len(task_data_list)} "
                f"(L={lattice_size_list[lattice_index]}, "
                f"x={x_axis_values[point_index]:0.4f})"
            )

    if np.any(observed_syndrome_weight_matrix != 0):
        raise AssertionError("observed_syndrome_weight_matrix must be all zero")
    if np.any(disorder_data_weight_matrix != 0):
        raise AssertionError("disorder_data_weight_matrix must be all zero")

    _save_npz(
        output_path=npz_output_path,
        args=args,
        x_axis_name=x_axis_name,
        x_axis_values=x_axis_values,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        average_acceptance_rate_curve_matrix=(
            average_acceptance_rate_curve_matrix
        ),
        q_top_spread_curve_matrix=q_top_spread_curve_matrix,
        max_r_hat_curve_matrix=max_r_hat_curve_matrix,
        min_effective_sample_size_curve_matrix=(
            min_effective_sample_size_curve_matrix
        ),
        q0_q_top_spread_curve_matrix=q0_q_top_spread_curve_matrix,
        q0_m_u_spread_linf_curve_matrix=q0_m_u_spread_linf_curve_matrix,
        observed_syndrome_weight_matrix=observed_syndrome_weight_matrix,
        disorder_data_weight_matrix=disorder_data_weight_matrix,
        num_qubits_list=num_qubits_list,
        num_logical_qubits_list=num_logical_qubits_list,
        effective_num_burn_in_sweeps_list=(
            effective_num_burn_in_sweeps_list
        ),
    )
    _plot_results(
        output_path=png_output_path,
        x_axis_values=x_axis_values,
        x_axis_name=x_axis_name,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        fixed_q=fixed_q,
        fixed_p=fixed_p,
        disorder_mode="all-zero disorder / no disorder averaging",
    )
    summary = _build_summary_json(
        args=args,
        x_axis_name=x_axis_name,
        x_axis_values=x_axis_values,
        lattice_size_list=lattice_size_list,
        q_top_curve_matrix=q_top_curve_matrix,
        average_acceptance_rate_curve_matrix=(
            average_acceptance_rate_curve_matrix
        ),
        q_top_spread_curve_matrix=q_top_spread_curve_matrix,
        max_r_hat_curve_matrix=max_r_hat_curve_matrix,
        min_effective_sample_size_curve_matrix=(
            min_effective_sample_size_curve_matrix
        ),
        q0_q_top_spread_curve_matrix=q0_q_top_spread_curve_matrix,
        q0_m_u_spread_linf_curve_matrix=q0_m_u_spread_linf_curve_matrix,
        num_qubits_list=num_qubits_list,
        num_logical_qubits_list=num_logical_qubits_list,
        effective_num_burn_in_sweeps_list=(
            effective_num_burn_in_sweeps_list
        ),
    )
    with json_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    _log(f"Final NPZ: {npz_output_path}")
    _log(f"Final JSON: {json_output_path}")
    _log(f"Final PNG: {png_output_path}")


if __name__ == "__main__":
    main()
