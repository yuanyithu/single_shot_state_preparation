import argparse
import json
import multiprocessing
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from build_toric_code_examples import (
    SUPPORTED_CODE_FAMILIES,
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from cluster_update import combine_cluster_summaries
from main import run_disorder_average_simulation
from mcmc_convergence_gate import (
    build_convergence_summary,
    write_convergence_summary_json,
)
from mcmc_diagnostics import (
    DATA_ONLY_PT_LADDER_SEMANTICS,
    SYNC_PT_LADDER_SEMANTICS,
    sync_pt_enlarge_ladder,
    sync_pt_ladders_from_enlarge,
)
from plot_scan_results import plot_scan_result


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_Q0_OUTPUT_STEM = (
    "scan_result_multi_L_q0_geometric_multistart_threshold_deep"
)
DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS = 18


def _pt_ladder_semantics(pt_ladder_mode):
    pt_ladder_mode = str(pt_ladder_mode)
    if pt_ladder_mode == "sync_enlarge":
        return SYNC_PT_LADDER_SEMANTICS
    return DATA_ONLY_PT_LADDER_SEMANTICS


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _format_probability_tag(probability):
    probability_string = f"{float(probability):0.4f}"
    return probability_string.replace(".", "p")


def _build_default_output_stem(
        code_family,
        syndrome_error_probability,
        common_random_disorder_across_p):
    code_family_tag = code_family
    if syndrome_error_probability == 0.0:
        if code_family == "2d_toric":
            output_stem = DEFAULT_Q0_OUTPUT_STEM
        else:
            output_stem = (
                f"scan_result_multi_L_{code_family_tag}_"
                "q0_geometric_multistart_threshold_deep"
            )
    else:
        output_stem = (
            "scan_result_multi_L_"
            f"{code_family_tag}_"
            f"q{_format_probability_tag(syndrome_error_probability)}_"
            "measurement_noise_threshold_deep"
        )
    if common_random_disorder_across_p:
        output_stem += "_common_random"
    return output_stem


def _ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(output_path, data):
    output_path = Path(output_path)
    _ensure_parent_dir(output_path)
    temporary_output_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    temporary_output_path.replace(output_path)


def _load_json(input_path):
    with Path(input_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_python_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _npz_scalar_string(value):
    return str(_as_python_scalar(value))


def _extract_section_stats(result):
    if "section_apply_count" not in result:
        return None
    return {
        "backend_name": _npz_scalar_string(result["section_backend_name"]),
        "apply_count": int(_as_python_scalar(result["section_apply_count"])),
        "cache_hit_count": int(
            _as_python_scalar(result["section_cache_hit_count"])
        ),
        "cache_size": int(_as_python_scalar(result["section_cache_size"])),
        "fallback_count": int(
            _as_python_scalar(result["section_fallback_count"])
        ),
        "decoder_failure_count": int(
            _as_python_scalar(result["section_decoder_failure_count"])
        ),
        "ldpc_import_error": _npz_scalar_string(
            result["section_ldpc_import_error"]
        ),
    }


def _combine_section_stats(section_summary_list):
    if not section_summary_list:
        return {}
    backend_names = sorted(
        {item["backend_name"] for item in section_summary_list}
    )
    ldpc_import_errors = sorted(
        {
            item["ldpc_import_error"]
            for item in section_summary_list
            if item["ldpc_import_error"]
        }
    )
    apply_count = sum(item["apply_count"] for item in section_summary_list)
    cache_hit_count = sum(
        item["cache_hit_count"] for item in section_summary_list
    )
    cache_size = sum(item["cache_size"] for item in section_summary_list)
    fallback_count = sum(item["fallback_count"] for item in section_summary_list)
    decoder_failure_count = sum(
        item["decoder_failure_count"] for item in section_summary_list
    )
    cache_hit_rate = (
        0.0 if apply_count == 0 else float(cache_hit_count / apply_count)
    )
    backend_name = (
        backend_names[0]
        if len(backend_names) == 1
        else ",".join(backend_names)
    )
    return {
        "section_backend_name": np.array(backend_name),
        "section_backend_names": np.asarray(backend_names),
        "section_apply_count": np.int64(apply_count),
        "section_cache_hit_count": np.int64(cache_hit_count),
        "section_cache_hit_rate": np.float64(cache_hit_rate),
        "section_cache_size": np.int64(cache_size),
        "section_fallback_count": np.int64(fallback_count),
        "section_decoder_failure_count": np.int64(decoder_failure_count),
        "section_ldpc_import_error": np.array(
            "; ".join(ldpc_import_errors)
        ),
        "section_num_chunks": np.int64(len(section_summary_list)),
    }


def _section_summary_for_manifest(merged_result):
    if "section_apply_count" not in merged_result:
        return None
    return {
        "backend_name": _npz_scalar_string(
            merged_result["section_backend_name"]
        ),
        "apply_count": int(
            _as_python_scalar(merged_result["section_apply_count"])
        ),
        "cache_hit_count": int(
            _as_python_scalar(merged_result["section_cache_hit_count"])
        ),
        "cache_hit_rate": float(
            _as_python_scalar(merged_result["section_cache_hit_rate"])
        ),
        "cache_size": int(_as_python_scalar(merged_result["section_cache_size"])),
        "fallback_count": int(
            _as_python_scalar(merged_result["section_fallback_count"])
        ),
        "decoder_failure_count": int(
            _as_python_scalar(merged_result["section_decoder_failure_count"])
        ),
        "ldpc_import_error": _npz_scalar_string(
            merged_result["section_ldpc_import_error"]
        ),
        "num_chunks": int(_as_python_scalar(merged_result["section_num_chunks"])),
    }


def _effective_num_burn_in_sweeps(
        num_burn_in_sweeps,
        num_qubits,
        burn_in_scaling_reference_num_qubits,
        max_effective_num_burn_in_sweeps=None):
    effective_num_burn_in_sweeps = int(np.ceil(
        num_burn_in_sweeps
        * (num_qubits / burn_in_scaling_reference_num_qubits)
    ))
    if max_effective_num_burn_in_sweeps is None:
        return effective_num_burn_in_sweeps
    return min(
        effective_num_burn_in_sweeps,
        int(max_effective_num_burn_in_sweeps),
    )


def _sort_tasks_for_execution(task_data_list):
    """
    Run larger lattice sizes first to avoid long L-tail chunks at the end.
    Seeds and output paths are assigned before this sort, so results are unchanged.
    """
    return sorted(
        task_data_list,
        key=lambda task_data: (
            -int(task_data["lattice_size"]),
            int(task_data["point_index"]),
            int(task_data["chunk_index"]),
        ),
    )


def _resolve_cli_num_start_chains(
        raw_num_start_chains,
        q0_num_start_chains,
        syndrome_error_probability):
    if raw_num_start_chains is not None:
        return int(raw_num_start_chains)
    if float(syndrome_error_probability) == 0.0:
        return int(q0_num_start_chains)
    return 1


def _resolve_git_commit_sha(explicit_git_commit_sha=None):
    if explicit_git_commit_sha is not None:
        return explicit_git_commit_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_chunk_output_path(
        chunks_dir,
        lattice_size,
        point_index,
        chunk_index):
    return (
        Path(chunks_dir)
        / f"L{int(lattice_size):02d}_p{int(point_index):02d}_chunk{int(chunk_index):03d}.npz"
    )


def _task_identifier(task_data):
    return (
        f"L{int(task_data['lattice_size']):02d}_"
        f"p{int(task_data['point_index']):02d}_"
        f"chunk{int(task_data['chunk_index']):03d}"
    )


def _build_submit_config_from_args(args):
    lattice_size_list = _parse_int_csv(args.lattice_sizes)
    data_error_probability_list = _parse_float_csv(
        args.data_error_probabilities
    )
    if not lattice_size_list:
        raise ValueError("lattice_size_list must be non-empty")
    if not data_error_probability_list:
        raise ValueError("data_error_probability_list must be non-empty")
    if args.num_disorder_samples_total % args.chunk_size != 0:
        raise ValueError(
            "num_disorder_samples_total must be divisible by chunk_size"
        )
    if (
            args.max_effective_num_burn_in_sweeps is not None
            and args.max_effective_num_burn_in_sweeps < 1):
        raise ValueError("max_effective_num_burn_in_sweeps must be >= 1")
    if args.cluster_budget_fraction_rho < 0.0:
        raise ValueError("cluster_budget_fraction_rho must be >= 0")
    if not (0.0 < float(args.single_bit_proposal_fraction) <= 1.0):
        raise ValueError("single_bit_proposal_fraction must be in (0, 1]")
    if int(args.pt_sector_diagnostic_stride) < 1:
        raise ValueError("--pt-sector-diagnostic-stride must be >= 1")
    if int(args.pt_swap_sweeps_per_attempt) < 1:
        raise ValueError("--pt-swap-sweeps-per-attempt must be >= 1")
    if int(args.pt_cold_edge_swap_stride) < 1:
        raise ValueError("--pt-cold-edge-swap-stride must be >= 1")
    if int(args.winding_plane_heatbath_sweeps) < 0:
        raise ValueError("--winding-plane-heatbath-sweeps must be >= 0")
    pt_ladder_spacing_power = float(args.pt_ladder_spacing_power)
    if (
            not np.isfinite(pt_ladder_spacing_power)
            or pt_ladder_spacing_power <= 0.0):
        raise ValueError("--pt-ladder-spacing-power must be finite and positive")
    if args.pt_ladder_mode not in ("data_only", "sync_enlarge"):
        raise ValueError("pt_ladder_mode must be data_only or sync_enlarge")
    if args.pt_ladder_mode != "sync_enlarge" and pt_ladder_spacing_power != 1.0:
        raise ValueError(
            "--pt-ladder-spacing-power is only supported with "
            "--pt-ladder-mode sync_enlarge"
        )
    pt_requested = (
        args.pt_p_hot is not None
        or args.pt_num_temperatures is not None
        or args.pt_ladder_mode != "data_only"
        or args.pt_q_hot is not None
        or pt_ladder_spacing_power != 1.0
        or int(args.adaptive_pt_rounds) > 0
    )
    if pt_requested and args.pt_num_temperatures is None:
        raise ValueError("--pt-num-temperatures is required for PT")
    if pt_requested and args.pt_ladder_mode == "data_only":
        if args.pt_p_hot is None:
            raise ValueError("--pt-p-hot is required for data_only PT")
        for data_error_probability in data_error_probability_list:
            if float(args.pt_p_hot) <= float(data_error_probability):
                raise ValueError("--pt-p-hot must be greater than cold p")
    if args.pt_ladder_mode == "sync_enlarge":
        if args.pt_q_hot is None:
            raise ValueError("--pt-q-hot is required for sync_enlarge PT")
        if not (
                float(args.syndrome_error_probability)
                < float(args.pt_q_hot)
                < 0.5):
            raise ValueError("--pt-q-hot must be in (cold q, 0.5)")
        pt_enlarge_ladder = sync_pt_enlarge_ladder(
            q_cold=float(args.syndrome_error_probability),
            q_hot=float(args.pt_q_hot),
            num_temperatures=int(args.pt_num_temperatures),
            spacing_power=pt_ladder_spacing_power,
        )
        for data_error_probability in data_error_probability_list:
            sync_pt_ladders_from_enlarge(
                p_cold=float(data_error_probability),
                q_cold=float(args.syndrome_error_probability),
                pt_enlarge=pt_enlarge_ladder,
            )
    if int(args.adaptive_pt_rounds) < 0:
        raise ValueError("--adaptive-pt-rounds must be >= 0")
    if int(args.adaptive_pt_rounds) > 0 and args.pt_ladder_mode != "sync_enlarge":
        raise ValueError(
            "--adaptive-pt-rounds requires --pt-ladder-mode sync_enlarge"
        )
    if int(args.adaptive_pt_calibration_sweeps) < 1:
        raise ValueError("--adaptive-pt-calibration-sweeps must be >= 1")

    run_root = Path(args.run_root).expanduser().resolve()
    chunks_dir = run_root / "chunks"
    num_chunks_per_point = (
        args.num_disorder_samples_total // args.chunk_size
    )
    git_commit_sha = _resolve_git_commit_sha(args.git_commit_sha)
    syndrome_error_probability = float(args.syndrome_error_probability)
    common_random_disorder_across_p = bool(
        args.common_random_disorder_across_p
    )
    code_family = str(args.code_family)
    output_stem = args.output_stem
    if output_stem is None:
        output_stem = _build_default_output_stem(
            code_family=code_family,
            syndrome_error_probability=syndrome_error_probability,
            common_random_disorder_across_p=(
                common_random_disorder_across_p
            ),
        )

    return {
        "run_root": str(run_root),
        "chunks_dir": str(chunks_dir),
        "manifest_path": str(run_root / "manifest.json"),
        "code_family": code_family,
        "lattice_size_list": lattice_size_list,
        "data_error_probability_list": data_error_probability_list,
        "syndrome_error_probability": syndrome_error_probability,
        "common_random_disorder_across_p": (
            common_random_disorder_across_p
        ),
        "num_disorder_samples_total": int(args.num_disorder_samples_total),
        "chunk_size": int(args.chunk_size),
        "num_chunks_per_point": int(num_chunks_per_point),
        "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
        "num_sweeps_between_measurements": (
            int(args.num_sweeps_between_measurements)
        ),
        "num_measurements_per_disorder": (
            int(args.num_measurements_per_disorder)
        ),
        "q0_num_start_chains": int(args.q0_num_start_chains),
        "num_start_chains": _resolve_cli_num_start_chains(
            raw_num_start_chains=args.num_start_chains,
            q0_num_start_chains=args.q0_num_start_chains,
            syndrome_error_probability=syndrome_error_probability,
        ),
        "num_replicas_per_start": int(args.num_replicas_per_start),
        "pt_p_hot": (
            None if args.pt_p_hot is None else float(args.pt_p_hot)
        ),
        "pt_num_temperatures": (
            None
            if args.pt_num_temperatures is None
            else int(args.pt_num_temperatures)
        ),
        "pt_ladder_mode": str(args.pt_ladder_mode),
        "pt_ladder_semantics": (
            None
            if not pt_requested
            else _pt_ladder_semantics(args.pt_ladder_mode)
        ),
        "pt_q_hot": (
            None if args.pt_q_hot is None else float(args.pt_q_hot)
        ),
        "pt_ladder_spacing_power": pt_ladder_spacing_power,
        "adaptive_pt_rounds": int(args.adaptive_pt_rounds),
        "adaptive_pt_calibration_sweeps": int(
            args.adaptive_pt_calibration_sweeps
        ),
        "pt_swap_attempt_every_num_sweeps": int(
            args.pt_swap_attempt_every_num_sweeps
        ),
        "pt_swap_sweeps_per_attempt": int(args.pt_swap_sweeps_per_attempt),
        "pt_cold_edge_swap_stride": int(args.pt_cold_edge_swap_stride),
        "num_zero_syndrome_sweeps_per_cycle": int(
            args.num_zero_syndrome_sweeps_per_cycle
        ),
        "winding_repeat_factor": int(args.winding_repeat_factor),
        "winding_plane_heatbath_sweeps": int(
            args.winding_plane_heatbath_sweeps
        ),
        "single_bit_proposal_fraction": float(
            args.single_bit_proposal_fraction
        ),
        "observable_temperature_mode": str(args.observable_temperature_mode),
        "track_pt_sector_diagnostics": bool(
            args.track_pt_sector_diagnostics
        ),
        "pt_sector_diagnostic_stride": int(args.pt_sector_diagnostic_stride),
        "cluster_update_enabled": not bool(args.disable_cluster_update),
        "cluster_budget_fraction_rho": float(args.cluster_budget_fraction_rho),
        "cluster_update_debug": bool(args.cluster_debug_assertions),
        "seed_base": int(args.seed_base),
        "burn_in_scaling_reference_num_qubits": int(
            args.burn_in_scaling_reference_num_qubits
        ),
        "max_effective_num_burn_in_sweeps": (
            None
            if args.max_effective_num_burn_in_sweeps is None
            else int(args.max_effective_num_burn_in_sweeps)
        ),
        "workers": int(args.workers),
        "git_commit_sha": git_commit_sha,
        "hostname": socket.gethostname(),
        "created_at": _timestamp(),
        "output_stem": output_stem,
        "final_output_path": str(run_root / f"{output_stem}.npz"),
        "final_plot_path": str(run_root / f"{output_stem}.png"),
    }


def _build_chunk_tasks(config):
    task_data_list = []
    num_qubits_list = []
    num_logical_qubits_list = []
    effective_num_burn_in_sweeps_list = []
    num_points = len(config["data_error_probability_list"])

    for lattice_index, lattice_size in enumerate(config["lattice_size_list"]):
        parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
            code_family=config["code_family"],
            lattice_size=lattice_size,
        )
        num_qubits = parity_check_matrix.shape[1]
        num_logical_qubits = dual_logical_z_basis.shape[0]
        effective_num_burn_in_sweeps = _effective_num_burn_in_sweeps(
            num_burn_in_sweeps=config["num_burn_in_sweeps"],
            num_qubits=num_qubits,
            burn_in_scaling_reference_num_qubits=(
                config["burn_in_scaling_reference_num_qubits"]
            ),
            max_effective_num_burn_in_sweeps=(
                config["max_effective_num_burn_in_sweeps"]
            ),
        )
        num_qubits_list.append(int(num_qubits))
        num_logical_qubits_list.append(int(num_logical_qubits))
        effective_num_burn_in_sweeps_list.append(
            int(effective_num_burn_in_sweeps)
        )

        for point_index, data_error_probability in enumerate(
                config["data_error_probability_list"]):
            for chunk_index in range(config["num_chunks_per_point"]):
                disorder_offset = chunk_index * config["chunk_size"]
                seed_stride_index = (
                    (
                        lattice_index * num_points
                        + point_index
                    ) * config["num_chunks_per_point"]
                    + chunk_index
                )
                task_data = {
                    "lattice_index": int(lattice_index),
                    "point_index": int(point_index),
                    "lattice_size": int(lattice_size),
                    "data_error_probability": float(data_error_probability),
                    "syndrome_error_probability": float(
                        config["syndrome_error_probability"]
                    ),
                    "num_disorder_samples": int(config["chunk_size"]),
                    "disorder_offset": int(disorder_offset),
                    "chunk_index": int(chunk_index),
                    "q0_num_start_chains": int(
                        config["q0_num_start_chains"]
                    ),
                    "num_start_chains": int(config["num_start_chains"]),
                    "num_replicas_per_start": int(
                        config["num_replicas_per_start"]
                    ),
                    "pt_p_hot": config["pt_p_hot"],
                    "pt_num_temperatures": config["pt_num_temperatures"],
                    "pt_ladder_mode": config["pt_ladder_mode"],
                    "pt_ladder_semantics": config["pt_ladder_semantics"],
                    "pt_q_hot": config["pt_q_hot"],
                    "pt_ladder_spacing_power": float(
                        config.get("pt_ladder_spacing_power", 1.0)
                    ),
                    "adaptive_pt_rounds": config["adaptive_pt_rounds"],
                    "adaptive_pt_calibration_sweeps": (
                        config["adaptive_pt_calibration_sweeps"]
                    ),
                    "pt_swap_attempt_every_num_sweeps": int(
                        config["pt_swap_attempt_every_num_sweeps"]
                    ),
                    "pt_swap_sweeps_per_attempt": int(
                        config["pt_swap_sweeps_per_attempt"]
                    ),
                    "pt_cold_edge_swap_stride": int(
                        config.get("pt_cold_edge_swap_stride", 1)
                    ),
                    "num_zero_syndrome_sweeps_per_cycle": int(
                        config["num_zero_syndrome_sweeps_per_cycle"]
                    ),
                    "winding_repeat_factor": int(
                        config["winding_repeat_factor"]
                    ),
                    "winding_plane_heatbath_sweeps": int(
                        config.get("winding_plane_heatbath_sweeps", 0)
                    ),
                    "single_bit_proposal_fraction": float(
                        config["single_bit_proposal_fraction"]
                    ),
                    "observable_temperature_mode": str(
                        config["observable_temperature_mode"]
                    ),
                    "track_pt_sector_diagnostics": bool(
                        config.get("track_pt_sector_diagnostics", False)
                    ),
                    "pt_sector_diagnostic_stride": int(
                        config.get("pt_sector_diagnostic_stride", 1)
                    ),
                    "cluster_update_enabled": bool(
                        config["cluster_update_enabled"]
                    ),
                    "cluster_budget_fraction_rho": float(
                        config["cluster_budget_fraction_rho"]
                    ),
                    "cluster_update_debug": bool(
                        config["cluster_update_debug"]
                    ),
                    "code_family": str(config["code_family"]),
                    "common_random_disorder_across_p": bool(
                        config["common_random_disorder_across_p"]
                    ),
                    "num_burn_in_sweeps": int(config["num_burn_in_sweeps"]),
                    "effective_num_burn_in_sweeps": int(
                        effective_num_burn_in_sweeps
                    ),
                    "num_sweeps_between_measurements": int(
                        config["num_sweeps_between_measurements"]
                    ),
                    "num_measurements_per_disorder": int(
                        config["num_measurements_per_disorder"]
                    ),
                    "burn_in_scaling_reference_num_qubits": int(
                        config["burn_in_scaling_reference_num_qubits"]
                    ),
                    "seed": int(
                        config["seed_base"] + 1000003 * seed_stride_index
                    ),
                    "disorder_seed": int(
                        config["seed_base"]
                        + 7000003 * (
                            lattice_index * config["num_chunks_per_point"]
                            + chunk_index
                        )
                        + 19
                    ),
                }
                task_data["output_path"] = str(
                    _build_chunk_output_path(
                        chunks_dir=config["chunks_dir"],
                        lattice_size=lattice_size,
                        point_index=point_index,
                        chunk_index=chunk_index,
                    )
                )
                task_data["task_id"] = _task_identifier(task_data)
                task_data_list.append(task_data)

    return (
        task_data_list,
        np.asarray(num_qubits_list, dtype=np.int64),
        np.asarray(num_logical_qubits_list, dtype=np.int64),
        np.asarray(effective_num_burn_in_sweeps_list, dtype=np.int64),
    )


def _build_manifest(
        config,
        task_data_list,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list):
    chunk_entries = []
    for task_data in task_data_list:
        chunk_entries.append({
            "task_id": task_data["task_id"],
            "lattice_index": task_data["lattice_index"],
            "point_index": task_data["point_index"],
            "lattice_size": task_data["lattice_size"],
            "data_error_probability": task_data["data_error_probability"],
            "chunk_index": task_data["chunk_index"],
            "disorder_offset": task_data["disorder_offset"],
            "num_disorder_samples": task_data["num_disorder_samples"],
            "output_path": task_data["output_path"],
            "seed": task_data["seed"],
            "disorder_seed": task_data["disorder_seed"],
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "error": None,
            "section_stats": None,
        })

    return {
        "run_root": config["run_root"],
        "created_at": config["created_at"],
        "updated_at": config["created_at"],
        "hostname": config["hostname"],
        "git_commit_sha": config["git_commit_sha"],
        "config": {
            "code_family": config["code_family"],
            "lattice_size_list": config["lattice_size_list"],
            "data_error_probability_list": (
                config["data_error_probability_list"]
            ),
            "syndrome_error_probability": (
                config["syndrome_error_probability"]
            ),
            "common_random_disorder_across_p": (
                config["common_random_disorder_across_p"]
            ),
            "num_disorder_samples_total": (
                config["num_disorder_samples_total"]
            ),
            "chunk_size": config["chunk_size"],
            "num_chunks_per_point": config["num_chunks_per_point"],
            "num_burn_in_sweeps": config["num_burn_in_sweeps"],
            "num_sweeps_between_measurements": (
                config["num_sweeps_between_measurements"]
            ),
            "num_measurements_per_disorder": (
                config["num_measurements_per_disorder"]
            ),
            "q0_num_start_chains": config["q0_num_start_chains"],
            "num_start_chains": config["num_start_chains"],
            "num_replicas_per_start": config["num_replicas_per_start"],
            "pt_p_hot": config["pt_p_hot"],
            "pt_num_temperatures": config["pt_num_temperatures"],
            "pt_ladder_mode": config["pt_ladder_mode"],
            "pt_ladder_semantics": config["pt_ladder_semantics"],
            "pt_q_hot": config["pt_q_hot"],
            "pt_ladder_spacing_power": float(
                config.get("pt_ladder_spacing_power", 1.0)
            ),
            "adaptive_pt_rounds": config["adaptive_pt_rounds"],
            "adaptive_pt_calibration_sweeps": (
                config["adaptive_pt_calibration_sweeps"]
            ),
            "pt_swap_attempt_every_num_sweeps": (
                config["pt_swap_attempt_every_num_sweeps"]
            ),
            "pt_swap_sweeps_per_attempt": (
                config["pt_swap_sweeps_per_attempt"]
            ),
            "pt_cold_edge_swap_stride": int(
                config.get("pt_cold_edge_swap_stride", 1)
            ),
            "num_zero_syndrome_sweeps_per_cycle": (
                config["num_zero_syndrome_sweeps_per_cycle"]
            ),
            "winding_repeat_factor": config["winding_repeat_factor"],
            "winding_plane_heatbath_sweeps": config.get(
                "winding_plane_heatbath_sweeps",
                0,
            ),
            "single_bit_proposal_fraction": (
                config["single_bit_proposal_fraction"]
            ),
            "observable_temperature_mode": config["observable_temperature_mode"],
            "track_pt_sector_diagnostics": bool(
                config.get("track_pt_sector_diagnostics", False)
            ),
            "pt_sector_diagnostic_stride": int(
                config.get("pt_sector_diagnostic_stride", 1)
            ),
            "cluster_update_enabled": config["cluster_update_enabled"],
            "cluster_budget_fraction_rho": (
                config["cluster_budget_fraction_rho"]
            ),
            "cluster_update_debug": config["cluster_update_debug"],
            "seed_base": config["seed_base"],
            "burn_in_scaling_reference_num_qubits": (
                config["burn_in_scaling_reference_num_qubits"]
            ),
            "max_effective_num_burn_in_sweeps": (
                config["max_effective_num_burn_in_sweeps"]
            ),
            "workers": config["workers"],
            "output_stem": config["output_stem"],
            "num_qubits_list": num_qubits_list.tolist(),
            "num_logical_qubits_list": num_logical_qubits_list.tolist(),
            "effective_num_burn_in_sweeps_list": (
                effective_num_burn_in_sweeps_list.tolist()
            ),
            "final_output_path": config["final_output_path"],
            "final_plot_path": config["final_plot_path"],
        },
        "preflight": {
            "completed": False,
            "run_root": str(Path(config["run_root"]) / "preflight"),
            "validated_at": None,
        },
        "summary": {
            "total_chunks": len(chunk_entries),
            "completed_chunks": 0,
            "failed_chunks": 0,
            "pending_chunks": len(chunk_entries),
        },
        "chunks": chunk_entries,
        "final_outputs": {
            "npz_path": config["final_output_path"],
            "png_path": config["final_plot_path"],
            "convergence_json_path": str(
                Path(config["run_root"])
                / f"{config['output_stem']}_convergence.json"
            ),
            "completed_at": None,
            "status": "pending",
        },
    }


def _update_manifest_summary(manifest):
    completed_chunks = 0
    failed_chunks = 0
    pending_chunks = 0
    for chunk_entry in manifest["chunks"]:
        if chunk_entry["status"] == "completed":
            completed_chunks += 1
        elif chunk_entry["status"] == "failed":
            failed_chunks += 1
        else:
            pending_chunks += 1
    manifest["summary"]["completed_chunks"] = completed_chunks
    manifest["summary"]["failed_chunks"] = failed_chunks
    manifest["summary"]["pending_chunks"] = pending_chunks
    manifest["updated_at"] = _timestamp()


def _task_status_index(manifest):
    status_index = {}
    for index, chunk_entry in enumerate(manifest["chunks"]):
        status_index[chunk_entry["task_id"]] = index
    return status_index


def _mark_existing_chunk_outputs_completed(manifest):
    for chunk_entry in manifest["chunks"]:
        if Path(chunk_entry["output_path"]).exists():
            chunk_entry["status"] = "completed"
            if chunk_entry["completed_at"] is None:
                chunk_entry["completed_at"] = _timestamp()
    _update_manifest_summary(manifest)


def _run_chunk_task(task_data):
    output_path = Path(task_data["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(
        output_path.stem + ".tmp.npz"
    )
    start_time = time.time()

    try:
        parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
            code_family=task_data["code_family"],
            lattice_size=task_data["lattice_size"],
        )
        zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
            code_family=task_data["code_family"],
            lattice_size=task_data["lattice_size"],
        )
        precomputed_syndrome_uniform_values_per_disorder = None
        precomputed_data_uniform_values_per_disorder = None
        if task_data["common_random_disorder_across_p"]:
            num_checks, num_qubits = parity_check_matrix.shape
            disorder_rng = np.random.default_rng(task_data["disorder_seed"])
            precomputed_syndrome_uniform_values_per_disorder = (
                disorder_rng.random(
                    (task_data["num_disorder_samples"], num_checks)
                )
            )
            precomputed_data_uniform_values_per_disorder = (
                disorder_rng.random(
                    (task_data["num_disorder_samples"], num_qubits)
                )
            )
        simulation_result = run_disorder_average_simulation(
            parity_check_matrix=parity_check_matrix,
            dual_logical_z_basis=dual_logical_z_basis,
            syndrome_error_probability=task_data["syndrome_error_probability"],
            data_error_probability=task_data["data_error_probability"],
            num_disorder_samples=task_data["num_disorder_samples"],
            num_burn_in_sweeps=task_data["effective_num_burn_in_sweeps"],
            num_sweeps_between_measurements=(
                task_data["num_sweeps_between_measurements"]
            ),
            num_measurements_per_disorder=(
                task_data["num_measurements_per_disorder"]
            ),
            seed=task_data["seed"],
            zero_syndrome_move_data=zero_syndrome_move_data,
            q0_num_start_chains=task_data["q0_num_start_chains"],
            num_start_chains=task_data["num_start_chains"],
            num_replicas_per_start=task_data["num_replicas_per_start"],
            pt_p_hot=task_data["pt_p_hot"],
            pt_num_temperatures=task_data["pt_num_temperatures"],
            pt_ladder_mode=task_data["pt_ladder_mode"],
            pt_q_hot=task_data["pt_q_hot"],
            pt_ladder_spacing_power=float(
                task_data.get("pt_ladder_spacing_power", 1.0)
            ),
            adaptive_pt_rounds=task_data["adaptive_pt_rounds"],
            adaptive_pt_calibration_sweeps=(
                task_data["adaptive_pt_calibration_sweeps"]
            ),
            pt_swap_attempt_every_num_sweeps=(
                task_data["pt_swap_attempt_every_num_sweeps"]
            ),
            pt_swap_sweeps_per_attempt=(
                task_data["pt_swap_sweeps_per_attempt"]
            ),
            pt_cold_edge_swap_stride=int(
                task_data.get("pt_cold_edge_swap_stride", 1)
            ),
            num_zero_syndrome_sweeps_per_cycle=(
                task_data["num_zero_syndrome_sweeps_per_cycle"]
            ),
            winding_repeat_factor=task_data["winding_repeat_factor"],
            winding_plane_heatbath_sweeps=(
                task_data.get("winding_plane_heatbath_sweeps", 0)
            ),
            single_bit_proposal_fraction=(
                task_data["single_bit_proposal_fraction"]
            ),
            observable_temperature_mode=task_data["observable_temperature_mode"],
            track_pt_sector_diagnostics=bool(
                task_data.get("track_pt_sector_diagnostics", False)
            ),
            pt_sector_diagnostic_stride=int(
                task_data.get("pt_sector_diagnostic_stride", 1)
            ),
            cluster_update_enabled=task_data["cluster_update_enabled"],
            cluster_budget_fraction_rho=(
                task_data["cluster_budget_fraction_rho"]
            ),
            cluster_update_debug=task_data["cluster_update_debug"],
            precomputed_syndrome_uniform_values_per_disorder=(
                precomputed_syndrome_uniform_values_per_disorder
            ),
            precomputed_data_uniform_values_per_disorder=(
                precomputed_data_uniform_values_per_disorder
            ),
        )
        simulation_result_for_save = dict(simulation_result)
        simulation_result_for_save.pop(
            "num_zero_syndrome_sweeps_per_cycle",
            None,
        )
        simulation_result_for_save.pop("winding_repeat_factor", None)
        simulation_result_for_save.pop("winding_plane_heatbath_sweeps", None)
        simulation_result_for_save.pop(
            "pt_swap_attempt_every_num_sweeps",
            None,
        )
        simulation_result_for_save.pop("pt_swap_sweeps_per_attempt", None)
        simulation_result_for_save.pop("pt_cold_edge_swap_stride", None)
        simulation_result_for_save.pop("pt_ladder_spacing_power", None)
        section_stats = _extract_section_stats(simulation_result_for_save)

        np.savez(
            temporary_output_path,
            **simulation_result_for_save,
            lattice_index=np.int64(task_data["lattice_index"]),
            point_index=np.int64(task_data["point_index"]),
            lattice_size=np.int64(task_data["lattice_size"]),
            data_error_probability=np.float64(
                task_data["data_error_probability"]
            ),
            syndrome_error_probability=np.float64(
                task_data["syndrome_error_probability"]
            ),
            chunk_index=np.int64(task_data["chunk_index"]),
            disorder_offset=np.int64(task_data["disorder_offset"]),
            num_disorder_samples=np.int64(task_data["num_disorder_samples"]),
            num_burn_in_sweeps=np.int64(task_data["num_burn_in_sweeps"]),
            effective_num_burn_in_sweeps=np.int64(
                task_data["effective_num_burn_in_sweeps"]
            ),
            num_sweeps_between_measurements=np.int64(
                task_data["num_sweeps_between_measurements"]
            ),
            num_measurements_per_disorder=np.int64(
                task_data["num_measurements_per_disorder"]
            ),
            q0_num_start_chains=np.int64(task_data["q0_num_start_chains"]),
            pt_swap_attempt_every_num_sweeps=np.int64(
                task_data["pt_swap_attempt_every_num_sweeps"]
            ),
            pt_swap_sweeps_per_attempt=np.int64(
                task_data["pt_swap_sweeps_per_attempt"]
            ),
            pt_cold_edge_swap_stride=np.int64(
                task_data.get("pt_cold_edge_swap_stride", 1)
            ),
            num_zero_syndrome_sweeps_per_cycle=np.int64(
                task_data["num_zero_syndrome_sweeps_per_cycle"]
            ),
            winding_repeat_factor=np.int64(
                task_data["winding_repeat_factor"]
            ),
            winding_plane_heatbath_sweeps=np.int64(
                task_data.get("winding_plane_heatbath_sweeps", 0)
            ),
            single_bit_proposal_fraction=np.float64(
                task_data["single_bit_proposal_fraction"]
            ),
            observable_temperature_mode=np.array(
                task_data["observable_temperature_mode"]
            ),
            track_pt_sector_diagnostics_config=np.bool_(
                task_data.get("track_pt_sector_diagnostics", False)
            ),
            pt_sector_diagnostic_stride_config=np.int64(
                task_data.get("pt_sector_diagnostic_stride", 1)
            ),
            pt_ladder_mode_config=np.array(task_data["pt_ladder_mode"]),
            pt_ladder_semantics_config=np.array(
                task_data.get(
                    "pt_ladder_semantics",
                    _pt_ladder_semantics(task_data["pt_ladder_mode"]),
                )
            ),
            pt_q_hot_config=np.float64(
                np.nan
                if task_data["pt_q_hot"] is None
                else task_data["pt_q_hot"]
            ),
            pt_ladder_spacing_power=np.float64(
                task_data.get("pt_ladder_spacing_power", 1.0)
            ),
            pt_ladder_spacing_power_config=np.float64(
                task_data.get("pt_ladder_spacing_power", 1.0)
            ),
            adaptive_pt_rounds_config=np.int64(task_data["adaptive_pt_rounds"]),
            adaptive_pt_calibration_sweeps_config=np.int64(
                task_data["adaptive_pt_calibration_sweeps"]
            ),
            cluster_update_config_enabled=np.bool_(
                task_data["cluster_update_enabled"]
            ),
            cluster_budget_fraction_rho_config=np.float64(
                task_data["cluster_budget_fraction_rho"]
            ),
            cluster_update_debug=np.bool_(task_data["cluster_update_debug"]),
            common_random_disorder_across_p=np.bool_(
                task_data["common_random_disorder_across_p"]
            ),
            seed=np.uint64(task_data["seed"]),
            disorder_seed=np.uint64(task_data["disorder_seed"]),
        )
        temporary_output_path.replace(output_path)
        return {
            "task_id": task_data["task_id"],
            "output_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "section_stats": section_stats,
        }
    except Exception:
        if temporary_output_path.exists():
            temporary_output_path.unlink()
        raise


def _run_pending_tasks_sequentially(
        pending_task_data_list,
        manifest,
        status_index,
        manifest_path):
    completed_task_count = 0
    for task_data in pending_task_data_list:
        chunk_entry = manifest["chunks"][status_index[task_data["task_id"]]]
        try:
            chunk_result = _run_chunk_task(task_data)
            chunk_entry["status"] = "completed"
            chunk_entry["completed_at"] = _timestamp()
            chunk_entry["error"] = None
            chunk_entry["section_stats"] = chunk_result.get(
                "section_stats"
            )
            completed_task_count += 1
            _log(
                f"Completed {chunk_result['task_id']} "
                f"({completed_task_count}/"
                f"{len(pending_task_data_list)}) "
                f"in {chunk_result['duration_seconds']:.1f}s"
            )
        except Exception as exc:
            chunk_entry["status"] = "failed"
            chunk_entry["completed_at"] = _timestamp()
            chunk_entry["error"] = repr(exc)
            _update_manifest_summary(manifest)
            _write_json_atomic(manifest_path, manifest)
            raise RuntimeError(
                f"Sequential chunk execution failed for {task_data['task_id']}: "
                f"{exc!r}"
            ) from exc

        _update_manifest_summary(manifest)
        _write_json_atomic(manifest_path, manifest)


def _compute_parallel_worker_count(requested_workers, num_tasks):
    if num_tasks <= 0:
        return 0
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(requested_workers, cpu_count, num_tasks))


def _build_multiprocessing_context():
    available_start_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_start_methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _run_exact_enumeration_validation():
    _log("Running exact_enumeration.py quick validation")
    subprocess.run(
        [sys.executable, str(SOURCE_DIR / "exact_enumeration.py"), "--quick"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _validate_chunk_payload(loaded_chunk_result, task_data):
    expected_num_disorders = task_data["num_disorder_samples"]
    if int(loaded_chunk_result["lattice_index"]) != task_data["lattice_index"]:
        raise ValueError("chunk lattice_index mismatch")
    if int(loaded_chunk_result["point_index"]) != task_data["point_index"]:
        raise ValueError("chunk point_index mismatch")
    if int(loaded_chunk_result["chunk_index"]) != task_data["chunk_index"]:
        raise ValueError("chunk chunk_index mismatch")
    if int(loaded_chunk_result["disorder_offset"]) != task_data["disorder_offset"]:
        raise ValueError("chunk disorder_offset mismatch")
    if int(loaded_chunk_result["num_disorder_samples"]) != expected_num_disorders:
        raise ValueError("chunk num_disorder_samples mismatch")
    if loaded_chunk_result["disorder_q_top_values"].shape[0] != expected_num_disorders:
        raise ValueError("chunk disorder_q_top_values length mismatch")
    if "pt_swap_sweeps_per_attempt" in loaded_chunk_result:
        if (
                int(loaded_chunk_result["pt_swap_sweeps_per_attempt"])
                != int(task_data.get("pt_swap_sweeps_per_attempt", 1))):
            raise ValueError("chunk pt_swap_sweeps_per_attempt mismatch")
    if "pt_cold_edge_swap_stride" in loaded_chunk_result:
        if (
                int(loaded_chunk_result["pt_cold_edge_swap_stride"])
                != int(task_data.get("pt_cold_edge_swap_stride", 1))):
            raise ValueError("chunk pt_cold_edge_swap_stride mismatch")
    if "winding_plane_heatbath_sweeps" in loaded_chunk_result:
        if (
                int(loaded_chunk_result["winding_plane_heatbath_sweeps"])
                != int(task_data.get("winding_plane_heatbath_sweeps", 0))):
            raise ValueError("chunk winding_plane_heatbath_sweeps mismatch")


def _merge_outputs(
        config,
        task_data_list,
        output_path,
        include_plot=False,
        plot_output_path=None):
    output_path = Path(output_path)
    _ensure_parent_dir(output_path)
    chunks_dir = Path(config["chunks_dir"])
    num_sizes = len(config["lattice_size_list"])
    num_points = len(config["data_error_probability_list"])
    num_disorder_samples_total = config["num_disorder_samples_total"]
    q0_num_start_chains = config["q0_num_start_chains"]
    has_q0_diagnostics = config["syndrome_error_probability"] == 0.0
    has_q_positive_diagnostics = config["syndrome_error_probability"] > 0.0
    num_start_chains = int(config["num_start_chains"])
    num_replicas_per_start = int(config["num_replicas_per_start"])
    pt_enabled = (
        config["pt_num_temperatures"] is not None
        and (
            config["pt_p_hot"] is not None
            or config.get("pt_ladder_mode", "data_only") != "data_only"
            or config.get("pt_q_hot") is not None
            or int(config.get("adaptive_pt_rounds", 0)) > 0
        )
    )
    pt_num_temperatures = (
        0 if not pt_enabled else int(config["pt_num_temperatures"])
    )
    adaptive_pt_rounds = int(config.get("adaptive_pt_rounds", 0))
    track_pt_sector_diagnostics = bool(
        config.get("track_pt_sector_diagnostics", False)
    )
    pt_sector_diagnostic_stride = int(
        config.get("pt_sector_diagnostic_stride", 1)
    )
    num_qubits_list = np.empty(num_sizes, dtype=np.int64)
    num_logical_qubits_list = np.empty(num_sizes, dtype=np.int64)
    effective_num_burn_in_sweeps_list = np.empty(num_sizes, dtype=np.int64)

    q_top_curve_matrix = np.empty((num_sizes, num_points), dtype=np.float64)
    q_top_std_curve_matrix = np.empty((num_sizes, num_points), dtype=np.float64)
    average_acceptance_rate_curve_matrix = np.empty(
        (num_sizes, num_points),
        dtype=np.float64,
    )
    q0_mean_m_u_spread_linf_curve_matrix = None
    q0_mean_q_top_spread_curve_matrix = None
    mean_q_top_spread_curve_matrix = None
    mean_m_u_spread_linf_curve_matrix = None
    max_r_hat_curve_matrix = None
    min_effective_sample_size_curve_matrix = None
    mean_num_chains_that_never_flipped_sector_curve_matrix = None
    mean_cold_winding_acceptance_rate_curve_matrix = None
    mean_pt_min_swap_acceptance_rate_curve_matrix = None
    mean_pt_mean_swap_acceptance_rate_curve_matrix = None
    mean_pt_winding_acceptance_rate_per_temperature_curve_tensor = None
    mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor = None
    mean_pt_sector_flip_count_per_temperature_curve_tensor = None
    mean_pt_sector_diagnostic_sample_count_curve_matrix = None
    mean_pt_hot_to_cold_sector_delivery_count_curve_matrix = None
    mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix = None
    mean_pt_cluster_sector_attempted_count_per_temperature_curve_tensor = None
    mean_pt_cluster_sector_nonzero_count_per_temperature_curve_tensor = None
    mean_pt_cluster_sector_changed_count_per_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_arrival_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_survived_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_reverted_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_other_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_curve_tensor = None
    mean_pt_cluster_sector_pending_remaining_count_per_origin_temperature_curve_tensor = None
    if has_q0_diagnostics:
        q0_mean_m_u_spread_linf_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        q0_mean_q_top_spread_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
    if has_q_positive_diagnostics:
        mean_q_top_spread_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        mean_m_u_spread_linf_curve_matrix = np.empty(
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
        mean_num_chains_that_never_flipped_sector_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        mean_cold_winding_acceptance_rate_curve_matrix = np.empty(
            (num_sizes, num_points),
            dtype=np.float64,
        )
        if pt_enabled:
            mean_pt_min_swap_acceptance_rate_curve_matrix = np.empty(
                (num_sizes, num_points),
                dtype=np.float64,
            )
            mean_pt_mean_swap_acceptance_rate_curve_matrix = np.empty(
                (num_sizes, num_points),
                dtype=np.float64,
            )
            mean_pt_winding_acceptance_rate_per_temperature_curve_tensor = (
                np.empty(
                    (num_sizes, num_points, pt_num_temperatures),
                    dtype=np.float64,
                )
            )
            mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor = (
                np.empty(
                    (num_sizes, num_points, pt_num_temperatures),
                    dtype=np.float64,
                )
            )
            if track_pt_sector_diagnostics:
                mean_pt_sector_flip_count_per_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_sector_diagnostic_sample_count_curve_matrix = (
                    np.empty((num_sizes, num_points), dtype=np.float64)
                )
                mean_pt_hot_to_cold_sector_delivery_count_curve_matrix = (
                    np.empty((num_sizes, num_points), dtype=np.float64)
                )
                mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix = (
                    np.empty((num_sizes, num_points), dtype=np.float64)
                )
                mean_pt_cluster_sector_attempted_count_per_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_nonzero_count_per_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_changed_count_per_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_arrival_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_survived_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_reverted_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_other_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )
                mean_pt_cluster_sector_pending_remaining_count_per_origin_temperature_curve_tensor = (
                    np.empty(
                        (num_sizes, num_points, pt_num_temperatures),
                        dtype=np.float64,
                    )
                )

    num_masks = None
    q0_start_sector_labels = None
    q_positive_start_sector_labels = None
    disorder_q_top_values_tensor = None
    average_acceptance_rate_per_disorder_tensor = None
    logical_observable_mean_values_per_disorder_tensor = None
    q0_logical_observable_mean_values_per_disorder_per_start_tensor = None
    q0_q_top_values_per_disorder_per_start_tensor = None
    q0_m_u_spread_linf_per_disorder_tensor = None
    q0_q_top_spread_per_disorder_tensor = None
    chain_logical_observable_mean_values_per_disorder_per_start_replica_tensor = None
    chain_q_top_values_per_disorder_per_start_replica_tensor = None
    chain_average_acceptance_rate_per_disorder_per_start_replica_tensor = None
    chain_contractible_acceptance_rate_per_disorder_per_start_replica_tensor = None
    chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor = None
    chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor = None
    chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor = None
    chain_observable_wall_time_per_disorder_per_start_replica_tensor = None
    chain_measurement_wall_time_per_disorder_per_start_replica_tensor = None
    q_top_spread_per_disorder_tensor = None
    m_u_spread_linf_per_disorder_tensor = None
    max_r_hat_per_disorder_tensor = None
    min_effective_sample_size_per_disorder_tensor = None
    num_chains_that_never_flipped_sector_per_disorder_tensor = None
    pt_min_swap_acceptance_rate_per_disorder_tensor = None
    pt_mean_swap_acceptance_rate_per_disorder_tensor = None
    chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor = None
    chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica_tensor = None
    chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica_tensor = None
    chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor = None
    chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor = None
    chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor = None
    chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor = None
    chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor = None
    pt_data_error_probability_ladder_per_disorder_tensor = None
    pt_syndrome_error_probability_ladder_per_disorder_tensor = None
    pt_enlarge_ladder_per_disorder_tensor = None
    adaptive_pt_num_rounds_completed_per_disorder_tensor = None
    adaptive_pt_round_f_raw_tensor = None
    adaptive_pt_round_f_mono_tensor = None
    adaptive_pt_round_f_target_tensor = None
    cluster_summary_list = []
    section_summary_list = []

    grouped_tasks = {}
    for task_data in task_data_list:
        key = (task_data["lattice_index"], task_data["point_index"])
        grouped_tasks.setdefault(key, []).append(task_data)

    for lattice_index, lattice_size in enumerate(config["lattice_size_list"]):
        parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
            code_family=config["code_family"],
            lattice_size=lattice_size,
        )
        num_qubits = parity_check_matrix.shape[1]
        num_logical_qubits = dual_logical_z_basis.shape[0]
        num_qubits_list[lattice_index] = num_qubits
        num_logical_qubits_list[lattice_index] = num_logical_qubits
        effective_num_burn_in_sweeps_list[lattice_index] = (
            _effective_num_burn_in_sweeps(
                num_burn_in_sweeps=config["num_burn_in_sweeps"],
                num_qubits=num_qubits,
                burn_in_scaling_reference_num_qubits=(
                    config["burn_in_scaling_reference_num_qubits"]
                ),
                max_effective_num_burn_in_sweeps=(
                    config.get("max_effective_num_burn_in_sweeps")
                ),
            )
        )
        current_num_masks = (1 << num_logical_qubits) - 1
        if num_masks is None:
            num_masks = current_num_masks
            disorder_q_top_values_tensor = np.empty(
                (num_sizes, num_points, num_disorder_samples_total),
                dtype=np.float64,
            )
            average_acceptance_rate_per_disorder_tensor = np.empty(
                (num_sizes, num_points, num_disorder_samples_total),
                dtype=np.float64,
            )
            logical_observable_mean_values_per_disorder_tensor = np.empty(
                (
                    num_sizes,
                    num_points,
                    num_disorder_samples_total,
                    num_masks,
                ),
                dtype=np.float64,
            )
            if has_q0_diagnostics:
                q0_logical_observable_mean_values_per_disorder_per_start_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            q0_num_start_chains,
                            num_masks,
                        ),
                        dtype=np.float64,
                    )
                )
                q0_q_top_values_per_disorder_per_start_tensor = np.empty(
                    (
                        num_sizes,
                        num_points,
                        num_disorder_samples_total,
                        q0_num_start_chains,
                    ),
                    dtype=np.float64,
                )
                q0_m_u_spread_linf_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
                q0_q_top_spread_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
            if has_q_positive_diagnostics:
                chain_logical_observable_mean_values_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                            num_masks,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_q_top_values_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_average_acceptance_rate_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_contractible_acceptance_rate_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_observable_wall_time_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                chain_measurement_wall_time_per_disorder_per_start_replica_tensor = (
                    np.empty(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            num_start_chains,
                            num_replicas_per_start,
                        ),
                        dtype=np.float64,
                    )
                )
                q_top_spread_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
                m_u_spread_linf_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
                max_r_hat_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
                min_effective_sample_size_per_disorder_tensor = np.empty(
                    (num_sizes, num_points, num_disorder_samples_total),
                    dtype=np.float64,
                )
                num_chains_that_never_flipped_sector_per_disorder_tensor = (
                    np.empty(
                        (num_sizes, num_points, num_disorder_samples_total),
                        dtype=np.int64,
                    )
                )
                if pt_enabled:
                    pt_min_swap_acceptance_rate_per_disorder_tensor = np.empty(
                        (num_sizes, num_points, num_disorder_samples_total),
                        dtype=np.float64,
                    )
                    pt_mean_swap_acceptance_rate_per_disorder_tensor = np.empty(
                        (num_sizes, num_points, num_disorder_samples_total),
                        dtype=np.float64,
                    )
                    pt_data_error_probability_ladder_per_disorder_tensor = (
                        np.full(
                            (
                                num_sizes,
                                num_points,
                                num_disorder_samples_total,
                                pt_num_temperatures,
                            ),
                            np.nan,
                            dtype=np.float64,
                        )
                    )
                    pt_syndrome_error_probability_ladder_per_disorder_tensor = (
                        np.full(
                            (
                                num_sizes,
                                num_points,
                                num_disorder_samples_total,
                                pt_num_temperatures,
                            ),
                            np.nan,
                            dtype=np.float64,
                        )
                    )
                    pt_enlarge_ladder_per_disorder_tensor = np.full(
                        (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            pt_num_temperatures,
                        ),
                        np.nan,
                        dtype=np.float64,
                    )
                    pt_temperature_chain_shape = (
                        num_sizes,
                        num_points,
                        num_disorder_samples_total,
                        num_start_chains,
                        num_replicas_per_start,
                        pt_num_temperatures,
                    )
                    pt_pair_chain_shape = pt_temperature_chain_shape[:-1] + (
                        max(pt_num_temperatures - 1, 0),
                    )
                    chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.float64,
                    )
                    chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor = np.empty(
                        pt_pair_chain_shape,
                        dtype=np.float64,
                    )
                    chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica_tensor = np.empty(
                        pt_pair_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica_tensor = np.empty(
                        pt_pair_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape[:-1],
                        dtype=np.int64,
                    )
                    chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor = np.empty(
                        pt_temperature_chain_shape,
                        dtype=np.int64,
                    )
                    if track_pt_sector_diagnostics:
                        chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape
                            + (1 << int(num_logical_qubits),),
                            dtype=np.int64,
                        )
                        chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape[:-1],
                            dtype=np.int64,
                        )
                        chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape[:-1],
                            dtype=np.int64,
                        )
                        chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape[:-1],
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                        chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor = np.empty(
                            pt_temperature_chain_shape,
                            dtype=np.int64,
                        )
                    if adaptive_pt_rounds > 0:
                        adaptive_pt_num_rounds_completed_per_disorder_tensor = (
                            np.zeros(
                                (
                                    num_sizes,
                                    num_points,
                                    num_disorder_samples_total,
                                ),
                                dtype=np.int64,
                            )
                        )
                        adaptive_shape = (
                            num_sizes,
                            num_points,
                            num_disorder_samples_total,
                            adaptive_pt_rounds,
                            pt_num_temperatures,
                        )
                        adaptive_pt_round_f_raw_tensor = np.full(
                            adaptive_shape,
                            np.nan,
                            dtype=np.float64,
                        )
                        adaptive_pt_round_f_mono_tensor = np.full(
                            adaptive_shape,
                            np.nan,
                            dtype=np.float64,
                        )
                        adaptive_pt_round_f_target_tensor = np.full(
                            adaptive_shape,
                            np.nan,
                            dtype=np.float64,
                        )
        elif current_num_masks != num_masks:
            raise ValueError("num_masks must be consistent across sizes")

    for key, grouped_task_list in grouped_tasks.items():
        lattice_index, point_index = key
        grouped_task_list.sort(key=lambda task_data: task_data["disorder_offset"])

        total_loaded_num_disorders = 0
        for task_data in grouped_task_list:
            chunk_output_path = Path(task_data["output_path"])
            if not chunk_output_path.exists():
                raise FileNotFoundError(
                    f"Missing chunk output: {chunk_output_path}"
                )
            with np.load(chunk_output_path, allow_pickle=True) as loaded_chunk_result:
                _validate_chunk_payload(loaded_chunk_result, task_data)
                start_index = task_data["disorder_offset"]
                stop_index = start_index + task_data["num_disorder_samples"]
                total_loaded_num_disorders += task_data["num_disorder_samples"]

                if has_q0_diagnostics:
                    if q0_start_sector_labels is None:
                        q0_start_sector_labels = loaded_chunk_result[
                            "q0_start_sector_labels"
                        ]
                    elif not np.array_equal(
                            q0_start_sector_labels,
                            loaded_chunk_result["q0_start_sector_labels"]):
                        raise ValueError("q0_start_sector_labels mismatch")
                if has_q_positive_diagnostics:
                    if q_positive_start_sector_labels is None:
                        q_positive_start_sector_labels = loaded_chunk_result[
                            "start_sector_labels"
                        ]
                    elif not np.array_equal(
                            q_positive_start_sector_labels,
                            loaded_chunk_result["start_sector_labels"]):
                        raise ValueError("start_sector_labels mismatch")

                disorder_q_top_values_tensor[
                    lattice_index,
                    point_index,
                    start_index:stop_index,
                ] = loaded_chunk_result["disorder_q_top_values"]
                average_acceptance_rate_per_disorder_tensor[
                    lattice_index,
                    point_index,
                    start_index:stop_index,
                ] = loaded_chunk_result["average_acceptance_rate_per_disorder"]
                logical_observable_mean_values_per_disorder_tensor[
                    lattice_index,
                    point_index,
                    start_index:stop_index,
                    :,
                ] = loaded_chunk_result[
                    "logical_observable_mean_values_per_disorder"
                ]
                if has_q0_diagnostics:
                    q0_logical_observable_mean_values_per_disorder_per_start_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "q0_logical_observable_mean_values_per_disorder_per_start"
                    ]
                    q0_q_top_values_per_disorder_per_start_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                    ] = loaded_chunk_result[
                        "q0_q_top_values_per_disorder_per_start"
                    ]
                    q0_m_u_spread_linf_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result["q0_m_u_spread_linf_per_disorder"]
                    q0_q_top_spread_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result["q0_q_top_spread_per_disorder"]
                if has_q_positive_diagnostics:
                    chain_logical_observable_mean_values_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_logical_observable_mean_values_per_disorder_per_start_replica"
                    ]
                    chain_q_top_values_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_q_top_values_per_disorder_per_start_replica"
                    ]
                    chain_average_acceptance_rate_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_average_acceptance_rate_per_disorder_per_start_replica"
                    ]
                    chain_contractible_acceptance_rate_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_contractible_acceptance_rate_per_disorder_per_start_replica"
                    ]
                    chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_winding_acceptance_rate_per_disorder_per_start_replica"
                    ]
                    chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_ordinary_update_wall_time_per_disorder_per_start_replica"
                    ]
                    chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_pt_swap_wall_time_per_disorder_per_start_replica"
                    ]
                    chain_observable_wall_time_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_observable_wall_time_per_disorder_per_start_replica"
                    ]
                    chain_measurement_wall_time_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                        :,
                        :,
                    ] = loaded_chunk_result[
                        "chain_measurement_wall_time_per_disorder_per_start_replica"
                    ]
                    q_top_spread_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result["q_top_spread_per_disorder"]
                    m_u_spread_linf_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result["m_u_spread_linf_per_disorder"]
                    max_r_hat_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result["max_r_hat_per_disorder"]
                    min_effective_sample_size_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result[
                        "min_effective_sample_size_per_disorder"
                    ]
                    num_chains_that_never_flipped_sector_per_disorder_tensor[
                        lattice_index,
                        point_index,
                        start_index:stop_index,
                    ] = loaded_chunk_result[
                        "num_chains_that_never_flipped_sector_per_disorder"
                    ]
                    if pt_enabled:
                        pt_min_swap_acceptance_rate_per_disorder_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                        ] = loaded_chunk_result[
                            "pt_min_swap_acceptance_rate_per_disorder"
                        ]
                        pt_mean_swap_acceptance_rate_per_disorder_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                        ] = loaded_chunk_result[
                            "pt_mean_swap_acceptance_rate_per_disorder"
                        ]
                        chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica"
                        ]
                        chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica"
                        ]
                        chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica"
                        ]
                        if "chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica" in loaded_chunk_result:
                            chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica"
                            ]
                        else:
                            chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                        if "chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica" in loaded_chunk_result:
                            chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica"
                            ]
                        else:
                            chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                        chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica"
                        ]
                        chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica"
                        ]
                        chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                            start_index:stop_index,
                            :,
                            :,
                            :,
                        ] = loaded_chunk_result[
                            "chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica"
                        ]
                        transport_shape = (
                            lattice_index,
                            point_index,
                            slice(start_index, stop_index),
                            slice(None),
                            slice(None),
                        )
                        if (
                                "chain_pt_transport_position_sample_count_per_disorder_per_start_replica"
                                in loaded_chunk_result):
                            chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor[
                                transport_shape
                            ] = loaded_chunk_result[
                                "chain_pt_transport_position_sample_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_cold_visit_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_hot_visit_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica"
                            ]
                            chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica"
                            ]
                        else:
                            chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor[
                                transport_shape
                            ] = 0
                            chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                            chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = 0
                        if track_pt_sector_diagnostics:
                            chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica"
                            ]
                            chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica"
                            ]
                            chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica"
                            ]
                            if (
                                    "chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica"
                                    in loaded_chunk_result):
                                chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor[
                                    lattice_index,
                                    point_index,
                                    start_index:stop_index,
                                    :,
                                    :,
                                ] = loaded_chunk_result[
                                    "chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica"
                                ]
                            else:
                                chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor[
                                    lattice_index,
                                    point_index,
                                    start_index:stop_index,
                                    :,
                                    :,
                                ] = int(np.ceil(
                                    config["num_measurements_per_disorder"]
                                    / max(pt_sector_diagnostic_stride, 1)
                                ))
                            chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica"
                            ]
                            if (
                                    "chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica"
                                    in loaded_chunk_result):
                                chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor[
                                    lattice_index,
                                    point_index,
                                    start_index:stop_index,
                                    :,
                                    :,
                                ] = loaded_chunk_result[
                                    "chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica"
                                ]
                            else:
                                chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor[
                                    lattice_index,
                                    point_index,
                                    start_index:stop_index,
                                    :,
                                    :,
                                ] = 0
                            for field_name, destination in (
                                    (
                                        "chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                                    (
                                        "chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica",
                                        chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    ),
                            ):
                                if field_name in loaded_chunk_result:
                                    destination[
                                        lattice_index,
                                        point_index,
                                        start_index:stop_index,
                                        :,
                                        :,
                                        :,
                                    ] = loaded_chunk_result[field_name]
                                else:
                                    destination[
                                        lattice_index,
                                        point_index,
                                        start_index:stop_index,
                                        :,
                                        :,
                                        :,
                                    ] = 0
                        if (
                                "pt_data_error_probability_ladder_per_disorder"
                                in loaded_chunk_result):
                            pt_data_error_probability_ladder_per_disorder_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                            ] = loaded_chunk_result[
                                "pt_data_error_probability_ladder_per_disorder"
                            ]
                        if (
                                "pt_syndrome_error_probability_ladder_per_disorder"
                                in loaded_chunk_result):
                            pt_syndrome_error_probability_ladder_per_disorder_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                            ] = loaded_chunk_result[
                                "pt_syndrome_error_probability_ladder_per_disorder"
                            ]
                        if "pt_enlarge_ladder_per_disorder" in loaded_chunk_result:
                            pt_enlarge_ladder_per_disorder_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                            ] = loaded_chunk_result[
                                "pt_enlarge_ladder_per_disorder"
                            ]
                        if (
                                adaptive_pt_rounds > 0
                                and "adaptive_pt_num_rounds_completed_per_disorder"
                                in loaded_chunk_result):
                            adaptive_pt_num_rounds_completed_per_disorder_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                            ] = loaded_chunk_result[
                                "adaptive_pt_num_rounds_completed_per_disorder"
                            ]
                        if (
                                adaptive_pt_rounds > 0
                                and "adaptive_pt_round_f_raw_per_disorder"
                                in loaded_chunk_result):
                            adaptive_pt_round_f_raw_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "adaptive_pt_round_f_raw_per_disorder"
                            ]
                        if (
                                adaptive_pt_rounds > 0
                                and "adaptive_pt_round_f_mono_per_disorder"
                                in loaded_chunk_result):
                            adaptive_pt_round_f_mono_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "adaptive_pt_round_f_mono_per_disorder"
                            ]
                        if (
                                adaptive_pt_rounds > 0
                                and "adaptive_pt_round_f_target_per_disorder"
                                in loaded_chunk_result):
                            adaptive_pt_round_f_target_tensor[
                                lattice_index,
                                point_index,
                                start_index:stop_index,
                                :,
                                :,
                            ] = loaded_chunk_result[
                                "adaptive_pt_round_f_target_per_disorder"
                            ]
                if "cluster_num_attempts" in loaded_chunk_result:
                    cluster_summary_list.append({
                        key: loaded_chunk_result[key]
                        for key in (
                            "cluster_update_enabled",
                            "cluster_update_requested_enabled",
                            "cluster_update_adaptive",
                            "cluster_budget_fraction_rho",
                            "cluster_num_attempts",
                            "cluster_num_nonzero_moves",
                            "cluster_total_wall_time",
                            "cluster_wall_time_fraction",
                            "cluster_nullity_mean",
                            "cluster_nullity_median",
                            "cluster_nullity_histogram",
                            "cluster_move_fraction_mean",
                            "cluster_by_temperature_attempts",
                            "cluster_by_temperature_nonzero_rate",
                            "cluster_by_temperature_mean_nullity",
                            "cluster_by_temperature_mean_move_fraction",
                            "cluster_by_temperature_wall_time",
                            "cluster_controller_frozen",
                        )
                    })
                section_stats = _extract_section_stats(loaded_chunk_result)
                if section_stats is not None:
                    section_summary_list.append(section_stats)

        if total_loaded_num_disorders != num_disorder_samples_total:
            raise ValueError(
                f"Incomplete disorder coverage for key={key}: "
                f"{total_loaded_num_disorders} != {num_disorder_samples_total}"
            )

    for lattice_index in range(num_sizes):
        for point_index in range(num_points):
            disorder_q_top_values = disorder_q_top_values_tensor[
                lattice_index,
                point_index,
            ]
            acceptance_values = average_acceptance_rate_per_disorder_tensor[
                lattice_index,
                point_index,
            ]
            q_top_curve_matrix[lattice_index, point_index] = float(
                np.mean(disorder_q_top_values)
            )
            if num_disorder_samples_total == 1:
                q_top_std_curve_matrix[lattice_index, point_index] = 0.0
            else:
                q_top_std_curve_matrix[lattice_index, point_index] = float(
                    np.std(disorder_q_top_values, ddof=1)
                )
            average_acceptance_rate_curve_matrix[
                lattice_index,
                point_index,
            ] = float(np.mean(acceptance_values))
            if has_q0_diagnostics:
                q0_m_u_spread_values = q0_m_u_spread_linf_per_disorder_tensor[
                    lattice_index,
                    point_index,
                ]
                q0_q_top_spread_values = q0_q_top_spread_per_disorder_tensor[
                    lattice_index,
                    point_index,
                ]
                q0_mean_m_u_spread_linf_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(q0_m_u_spread_values))
                q0_mean_q_top_spread_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(q0_q_top_spread_values))
            if has_q_positive_diagnostics:
                mean_q_top_spread_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(
                    q_top_spread_per_disorder_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                mean_m_u_spread_linf_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(
                    m_u_spread_linf_per_disorder_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                finite_r_hat_values = max_r_hat_per_disorder_tensor[
                    lattice_index,
                    point_index,
                ]
                finite_r_hat_values = finite_r_hat_values[
                    np.isfinite(finite_r_hat_values)
                ]
                if finite_r_hat_values.size == 0:
                    max_r_hat_curve_matrix[lattice_index, point_index] = (
                        np.nan
                    )
                else:
                    max_r_hat_curve_matrix[lattice_index, point_index] = (
                        float(np.max(finite_r_hat_values))
                    )
                min_effective_sample_size_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.min(
                    min_effective_sample_size_per_disorder_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                mean_num_chains_that_never_flipped_sector_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(
                    num_chains_that_never_flipped_sector_per_disorder_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                mean_cold_winding_acceptance_rate_curve_matrix[
                    lattice_index,
                    point_index,
                ] = float(np.mean(
                    chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor[
                        lattice_index,
                        point_index,
                    ]
                ))
                if pt_enabled:
                    mean_pt_min_swap_acceptance_rate_curve_matrix[
                        lattice_index,
                        point_index,
                    ] = float(np.mean(
                        pt_min_swap_acceptance_rate_per_disorder_tensor[
                            lattice_index,
                            point_index,
                        ]
                    ))
                    mean_pt_mean_swap_acceptance_rate_curve_matrix[
                        lattice_index,
                        point_index,
                    ] = float(np.mean(
                        pt_mean_swap_acceptance_rate_per_disorder_tensor[
                            lattice_index,
                            point_index,
                        ]
                    ))
                    mean_pt_winding_acceptance_rate_per_temperature_curve_tensor[
                        lattice_index,
                        point_index,
                    ] = np.mean(
                        chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                        ],
                        axis=(0, 1, 2),
                    )
                    mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor[
                        lattice_index,
                        point_index,
                    ] = np.mean(
                        chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor[
                            lattice_index,
                            point_index,
                        ],
                        axis=(0, 1, 2),
                    )
                    if track_pt_sector_diagnostics:
                        mean_pt_sector_flip_count_per_temperature_curve_tensor[
                            lattice_index,
                            point_index,
                        ] = np.mean(
                            chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ],
                            axis=(0, 1, 2),
                        )
                        mean_pt_sector_diagnostic_sample_count_curve_matrix[
                            lattice_index,
                            point_index,
                        ] = float(np.mean(
                            chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ]
                        ))
                        mean_pt_hot_to_cold_sector_delivery_count_curve_matrix[
                            lattice_index,
                            point_index,
                        ] = float(np.mean(
                            chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ]
                        ))
                        mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix[
                            lattice_index,
                            point_index,
                        ] = float(np.mean(
                            chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ]
                        ))
                        mean_pt_cluster_sector_attempted_count_per_temperature_curve_tensor[
                            lattice_index,
                            point_index,
                        ] = np.mean(
                            chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ],
                            axis=(0, 1, 2),
                        )
                        mean_pt_cluster_sector_nonzero_count_per_temperature_curve_tensor[
                            lattice_index,
                            point_index,
                        ] = np.mean(
                            chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ],
                            axis=(0, 1, 2),
                        )
                        mean_pt_cluster_sector_changed_count_per_temperature_curve_tensor[
                            lattice_index,
                            point_index,
                        ] = np.mean(
                            chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor[
                                lattice_index,
                                point_index,
                            ],
                            axis=(0, 1, 2),
                        )
                        for source_tensor, destination_tensor in (
                                (
                                    chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_arrival_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_survived_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_reverted_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_other_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_curve_tensor,
                                ),
                                (
                                    chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor,
                                    mean_pt_cluster_sector_pending_remaining_count_per_origin_temperature_curve_tensor,
                                ),
                        ):
                            destination_tensor[
                                lattice_index,
                                point_index,
                            ] = np.mean(
                                source_tensor[lattice_index, point_index],
                                axis=(0, 1, 2),
                            )

    merged_result = {
        "lattice_size_list": np.asarray(
            config["lattice_size_list"],
            dtype=np.int64,
        ),
        "num_qubits_list": num_qubits_list,
        "num_logical_qubits_list": num_logical_qubits_list,
        "effective_num_burn_in_sweeps_list": (
            effective_num_burn_in_sweeps_list
        ),
        "data_error_probability_list": np.asarray(
            config["data_error_probability_list"],
            dtype=np.float64,
        ),
        "q_top_curve_matrix": q_top_curve_matrix,
        "q_top_std_curve_matrix": q_top_std_curve_matrix,
        "average_acceptance_rate_curve_matrix": (
            average_acceptance_rate_curve_matrix
        ),
        "disorder_q_top_values_tensor": disorder_q_top_values_tensor,
        "average_acceptance_rate_per_disorder_tensor": (
            average_acceptance_rate_per_disorder_tensor
        ),
        "logical_observable_mean_values_per_disorder_tensor": (
            logical_observable_mean_values_per_disorder_tensor
        ),
    }
    if cluster_summary_list:
        merged_result.update(combine_cluster_summaries(cluster_summary_list))
    merged_result.update(_combine_section_stats(section_summary_list))
    if has_q0_diagnostics:
        merged_result["q0_start_sector_labels"] = q0_start_sector_labels
        merged_result["q0_mean_m_u_spread_linf_curve_matrix"] = (
            q0_mean_m_u_spread_linf_curve_matrix
        )
        merged_result["q0_mean_q_top_spread_curve_matrix"] = (
            q0_mean_q_top_spread_curve_matrix
        )
        merged_result[
            "q0_logical_observable_mean_values_per_disorder_per_start_tensor"
        ] = q0_logical_observable_mean_values_per_disorder_per_start_tensor
        merged_result["q0_q_top_values_per_disorder_per_start_tensor"] = (
            q0_q_top_values_per_disorder_per_start_tensor
        )
        merged_result["q0_m_u_spread_linf_per_disorder_tensor"] = (
            q0_m_u_spread_linf_per_disorder_tensor
        )
        merged_result["q0_q_top_spread_per_disorder_tensor"] = (
            q0_q_top_spread_per_disorder_tensor
        )
    if has_q_positive_diagnostics:
        merged_result["start_sector_labels"] = q_positive_start_sector_labels
        merged_result["num_start_chains"] = np.int64(num_start_chains)
        merged_result["num_replicas_per_start"] = np.int64(
            num_replicas_per_start
        )
        merged_result[
            "chain_logical_observable_mean_values_per_disorder_per_start_replica_tensor"
        ] = chain_logical_observable_mean_values_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_q_top_values_per_disorder_per_start_replica_tensor"
        ] = chain_q_top_values_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_average_acceptance_rate_per_disorder_per_start_replica_tensor"
        ] = chain_average_acceptance_rate_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_contractible_acceptance_rate_per_disorder_per_start_replica_tensor"
        ] = chain_contractible_acceptance_rate_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor"
        ] = chain_winding_acceptance_rate_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor"
        ] = chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor"
        ] = chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_observable_wall_time_per_disorder_per_start_replica_tensor"
        ] = chain_observable_wall_time_per_disorder_per_start_replica_tensor
        merged_result[
            "chain_measurement_wall_time_per_disorder_per_start_replica_tensor"
        ] = chain_measurement_wall_time_per_disorder_per_start_replica_tensor
        merged_result["q_top_spread_per_disorder_tensor"] = (
            q_top_spread_per_disorder_tensor
        )
        merged_result["m_u_spread_linf_per_disorder_tensor"] = (
            m_u_spread_linf_per_disorder_tensor
        )
        merged_result["max_r_hat_per_disorder_tensor"] = (
            max_r_hat_per_disorder_tensor
        )
        merged_result["min_effective_sample_size_per_disorder_tensor"] = (
            min_effective_sample_size_per_disorder_tensor
        )
        merged_result[
            "num_chains_that_never_flipped_sector_per_disorder_tensor"
        ] = num_chains_that_never_flipped_sector_per_disorder_tensor
        merged_result["mean_q_top_spread_curve_matrix"] = (
            mean_q_top_spread_curve_matrix
        )
        merged_result["mean_m_u_spread_linf_curve_matrix"] = (
            mean_m_u_spread_linf_curve_matrix
        )
        merged_result["max_r_hat_curve_matrix"] = max_r_hat_curve_matrix
        merged_result["min_effective_sample_size_curve_matrix"] = (
            min_effective_sample_size_curve_matrix
        )
        merged_result[
            "mean_num_chains_that_never_flipped_sector_curve_matrix"
        ] = mean_num_chains_that_never_flipped_sector_curve_matrix
        merged_result["mean_cold_winding_acceptance_rate_curve_matrix"] = (
            mean_cold_winding_acceptance_rate_curve_matrix
        )
        merged_result["pt_enabled"] = np.bool_(pt_enabled)
        if pt_enabled:
            merged_result["pt_p_hot"] = np.float64(
                np.nan if config["pt_p_hot"] is None else config["pt_p_hot"]
            )
            merged_result["pt_q_hot"] = np.float64(
                (
                    config["syndrome_error_probability"]
                    if config.get("pt_q_hot") is None
                    else config["pt_q_hot"]
                )
            )
            merged_result["pt_num_temperatures"] = np.int64(
                config["pt_num_temperatures"]
            )
            merged_result["pt_ladder_mode"] = np.array(
                config.get("pt_ladder_mode", "data_only")
            )
            merged_result["pt_ladder_semantics"] = np.array(
                config.get(
                    "pt_ladder_semantics",
                    _pt_ladder_semantics(
                        config.get("pt_ladder_mode", "data_only")
                    ),
                )
            )
            merged_result["pt_ladder_spacing_power"] = np.float64(
                config.get("pt_ladder_spacing_power", 1.0)
            )
            merged_result["adaptive_pt_rounds"] = np.int64(
                adaptive_pt_rounds
            )
            merged_result["adaptive_pt_calibration_sweeps"] = np.int64(
                config.get("adaptive_pt_calibration_sweeps", 128)
            )
            merged_result["pt_min_swap_acceptance_rate_per_disorder_tensor"] = (
                pt_min_swap_acceptance_rate_per_disorder_tensor
            )
            merged_result["pt_mean_swap_acceptance_rate_per_disorder_tensor"] = (
                pt_mean_swap_acceptance_rate_per_disorder_tensor
            )
            merged_result[
                "chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_winding_accepted_count_per_temperature_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_winding_attempted_count_per_temperature_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_swap_accept_count_per_pair_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_swap_attempt_count_per_pair_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_cold_visit_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_hot_visit_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_cold_to_hot_passage_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_hot_to_cold_passage_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_min_temperature_visited_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor"
            ] = (
                chain_pt_replica_max_temperature_visited_per_disorder_per_start_replica_tensor
            )
            merged_result[
                "pt_data_error_probability_ladder_per_disorder_tensor"
            ] = pt_data_error_probability_ladder_per_disorder_tensor
            merged_result[
                "pt_syndrome_error_probability_ladder_per_disorder_tensor"
            ] = pt_syndrome_error_probability_ladder_per_disorder_tensor
            merged_result["pt_enlarge_ladder_per_disorder_tensor"] = (
                pt_enlarge_ladder_per_disorder_tensor
            )
            merged_result[
                "mean_pt_min_swap_acceptance_rate_curve_matrix"
            ] = mean_pt_min_swap_acceptance_rate_curve_matrix
            merged_result[
                "mean_pt_mean_swap_acceptance_rate_curve_matrix"
            ] = mean_pt_mean_swap_acceptance_rate_curve_matrix
            merged_result[
                "mean_pt_winding_acceptance_rate_per_temperature_curve_tensor"
            ] = mean_pt_winding_acceptance_rate_per_temperature_curve_tensor
            merged_result[
                "mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor"
            ] = (
                mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor
            )
            merged_result["pt_track_sector_diagnostics"] = np.bool_(
                track_pt_sector_diagnostics
            )
            merged_result["pt_sector_diagnostic_stride"] = np.int64(
                pt_sector_diagnostic_stride
            )
            if track_pt_sector_diagnostics:
                merged_result[
                    "chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_first_sector_change_index_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_hot_to_cold_sector_delivery_count_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor"
                ] = (
                    chain_pt_cluster_sector_pending_remaining_count_per_origin_temperature_per_disorder_per_start_replica_tensor
                )
                merged_result[
                    "mean_pt_sector_flip_count_per_temperature_curve_tensor"
                ] = mean_pt_sector_flip_count_per_temperature_curve_tensor
                merged_result[
                    "mean_pt_sector_diagnostic_sample_count_curve_matrix"
                ] = mean_pt_sector_diagnostic_sample_count_curve_matrix
                merged_result[
                    "mean_pt_hot_to_cold_sector_delivery_count_curve_matrix"
                ] = mean_pt_hot_to_cold_sector_delivery_count_curve_matrix
                merged_result[
                    "mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix"
                ] = (
                    mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix
                )
                merged_result[
                    "mean_pt_cluster_sector_attempted_count_per_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_attempted_count_per_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_nonzero_count_per_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_nonzero_count_per_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_changed_count_per_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_changed_count_per_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_arrival_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_arrival_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_survived_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_survived_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_reverted_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_reverted_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_other_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_other_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_cold_active_remaining_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_pending_overwritten_count_per_origin_temperature_curve_tensor
                )
                merged_result[
                    "mean_pt_cluster_sector_pending_remaining_count_per_origin_temperature_curve_tensor"
                ] = (
                    mean_pt_cluster_sector_pending_remaining_count_per_origin_temperature_curve_tensor
                )
            if adaptive_pt_rounds > 0:
                merged_result[
                    "adaptive_pt_num_rounds_completed_per_disorder_tensor"
                ] = adaptive_pt_num_rounds_completed_per_disorder_tensor
                merged_result["adaptive_pt_round_f_raw_tensor"] = (
                    adaptive_pt_round_f_raw_tensor
                )
                merged_result["adaptive_pt_round_f_mono_tensor"] = (
                    adaptive_pt_round_f_mono_tensor
                )
                merged_result["adaptive_pt_round_f_target_tensor"] = (
                    adaptive_pt_round_f_target_tensor
                )

    convergence_summary = None
    convergence_summary_path = None
    if has_q_positive_diagnostics:
        convergence_summary = build_convergence_summary(
            merged_result=merged_result,
            lattice_size_list=config["lattice_size_list"],
            data_error_probability_list=config["data_error_probability_list"],
            syndrome_error_probability=config["syndrome_error_probability"],
        )
        merged_result["converged_mask_matrix"] = (
            convergence_summary["converged_mask_matrix"]
        )
        convergence_summary_path = output_path.with_name(
            output_path.stem + "_convergence.json"
        )
        merged_result["convergence_summary_path"] = np.array(
            str(convergence_summary_path)
        )

    np.savez(
        output_path,
        **merged_result,
        code_family=np.array(config["code_family"]),
        code_type=np.array(config["code_family"]),
        syndrome_error_probability=np.float64(
            config["syndrome_error_probability"]
        ),
        num_disorder_samples=np.int64(config["num_disorder_samples_total"]),
        num_burn_in_sweeps=np.int64(config["num_burn_in_sweeps"]),
        num_sweeps_between_measurements=np.int64(
            config["num_sweeps_between_measurements"]
        ),
        num_measurements_per_disorder=np.int64(
            config["num_measurements_per_disorder"]
        ),
        q0_num_start_chains=np.int64(config["q0_num_start_chains"]),
        pt_swap_attempt_every_num_sweeps=np.int64(
            config["pt_swap_attempt_every_num_sweeps"]
        ),
        pt_swap_sweeps_per_attempt=np.int64(
            config["pt_swap_sweeps_per_attempt"]
        ),
        pt_cold_edge_swap_stride=np.int64(
            config.get("pt_cold_edge_swap_stride", 1)
        ),
        num_zero_syndrome_sweeps_per_cycle=np.int64(
            config["num_zero_syndrome_sweeps_per_cycle"]
        ),
        winding_repeat_factor=np.int64(config["winding_repeat_factor"]),
        winding_plane_heatbath_sweeps=np.int64(
            config.get("winding_plane_heatbath_sweeps", 0)
        ),
        single_bit_proposal_fraction=np.float64(
            config["single_bit_proposal_fraction"]
        ),
        observable_temperature_mode=np.array(
            config["observable_temperature_mode"]
        ),
        track_pt_sector_diagnostics_config=np.bool_(
            track_pt_sector_diagnostics
        ),
        pt_sector_diagnostic_stride_config=np.int64(
            pt_sector_diagnostic_stride
        ),
        pt_ladder_mode_config=np.array(config.get("pt_ladder_mode", "data_only")),
        pt_ladder_semantics_config=np.array(
            config.get(
                "pt_ladder_semantics",
                _pt_ladder_semantics(config.get("pt_ladder_mode", "data_only")),
            )
        ),
        pt_q_hot_config=np.float64(
            np.nan
            if config.get("pt_q_hot") is None
            else config["pt_q_hot"]
        ),
        pt_ladder_spacing_power_config=np.float64(
            config.get("pt_ladder_spacing_power", 1.0)
        ),
        adaptive_pt_rounds_config=np.int64(adaptive_pt_rounds),
        adaptive_pt_calibration_sweeps_config=np.int64(
            config.get("adaptive_pt_calibration_sweeps", 128)
        ),
        cluster_update_config_enabled=np.bool_(
            config["cluster_update_enabled"]
        ),
        cluster_budget_fraction_rho_config=np.float64(
            config["cluster_budget_fraction_rho"]
        ),
        cluster_update_debug=np.bool_(config["cluster_update_debug"]),
        common_random_disorder_across_p=np.bool_(
            config["common_random_disorder_across_p"]
        ),
        seed_base=np.int64(config["seed_base"]),
        burn_in_scaling_reference_num_qubits=np.int64(
            config["burn_in_scaling_reference_num_qubits"]
        ),
        chunk_size=np.int64(config["chunk_size"]),
        num_chunks_per_point=np.int64(config["num_chunks_per_point"]),
        git_commit_sha=np.array(config["git_commit_sha"]),
        source_chunks_dir=np.array(str(chunks_dir)),
    )
    if convergence_summary is not None:
        write_convergence_summary_json(
            output_path=convergence_summary_path,
            convergence_summary=convergence_summary,
        )

    if include_plot:
        if plot_output_path is None:
            plot_output_path = output_path.with_suffix(".png")
        plot_scan_result(
            input_path=output_path,
            output_path=plot_output_path,
        )

    return merged_result


def _run_preflight(config):
    preflight_root = Path(config["run_root"]) / "preflight"
    preflight_chunks_dir = preflight_root / "chunks"
    preflight_root.mkdir(parents=True, exist_ok=True)
    preflight_chunks_dir.mkdir(parents=True, exist_ok=True)

    _run_exact_enumeration_validation()

    preflight_config = dict(config)
    preflight_config["run_root"] = str(preflight_root)
    preflight_config["chunks_dir"] = str(preflight_chunks_dir)
    preflight_config["manifest_path"] = str(preflight_root / "manifest.json")
    preflight_config["lattice_size_list"] = [3]
    preflight_config["data_error_probability_list"] = [
        float(config["data_error_probability_list"][0])
    ]
    preflight_config["num_disorder_samples_total"] = 2
    preflight_config["chunk_size"] = 2
    preflight_config["num_chunks_per_point"] = 1
    preflight_config["num_measurements_per_disorder"] = 20
    preflight_config["final_output_path"] = str(
        preflight_root / "preflight_scan_result.npz"
    )
    preflight_config["final_plot_path"] = str(
        preflight_root / "preflight_scan_result.png"
    )
    preflight_config["created_at"] = _timestamp()

    (
        preflight_task_data_list,
        preflight_num_qubits_list,
        preflight_num_logical_qubits_list,
        preflight_effective_num_burn_in_sweeps_list,
    ) = _build_chunk_tasks(preflight_config)
    _log("Running preflight chunk")
    for task_data in preflight_task_data_list:
        _run_chunk_task(task_data)
    _log("Running preflight merge")
    _merge_outputs(
        config=preflight_config,
        task_data_list=preflight_task_data_list,
        output_path=preflight_config["final_output_path"],
        include_plot=False,
    )

    return {
        "completed": True,
        "run_root": str(preflight_root),
        "validated_at": _timestamp(),
    }


def _submit_run(args):
    config = _build_submit_config_from_args(args)
    run_root = Path(config["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    Path(config["chunks_dir"]).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["manifest_path"])

    (
        task_data_list,
        num_qubits_list,
        num_logical_qubits_list,
        effective_num_burn_in_sweeps_list,
    ) = _build_chunk_tasks(config)

    if manifest_path.exists():
        if not args.resume and not args.merge_only:
            raise RuntimeError(
                f"Manifest already exists: {manifest_path}. "
                "Use --resume or a fresh run_root."
            )
        manifest = _load_json(manifest_path)
    else:
        manifest = _build_manifest(
            config=config,
            task_data_list=task_data_list,
            num_qubits_list=num_qubits_list,
            num_logical_qubits_list=num_logical_qubits_list,
            effective_num_burn_in_sweeps_list=(
                effective_num_burn_in_sweeps_list
            ),
        )
        _write_json_atomic(manifest_path, manifest)

    if args.merge_only:
        _log("Merge-only mode")
        _merge_outputs(
            config=config,
            task_data_list=task_data_list,
            output_path=config["final_output_path"],
            include_plot=True,
            plot_output_path=config["final_plot_path"],
        )
        manifest["final_outputs"]["status"] = "completed"
        manifest["final_outputs"]["completed_at"] = _timestamp()
        _update_manifest_summary(manifest)
        _write_json_atomic(manifest_path, manifest)
        return 0

    if not manifest["preflight"]["completed"]:
        manifest["preflight"] = _run_preflight(config)
        _write_json_atomic(manifest_path, manifest)

    _mark_existing_chunk_outputs_completed(manifest)
    status_index = _task_status_index(manifest)
    _write_json_atomic(manifest_path, manifest)

    pending_task_data_list = []
    for task_data in task_data_list:
        chunk_entry = manifest["chunks"][status_index[task_data["task_id"]]]
        if chunk_entry["status"] == "completed":
            continue
        chunk_entry["status"] = "pending"
        chunk_entry["error"] = None
        pending_task_data_list.append(task_data)
    pending_task_data_list = _sort_tasks_for_execution(pending_task_data_list)

    _update_manifest_summary(manifest)
    _write_json_atomic(manifest_path, manifest)

    if pending_task_data_list:
        worker_count = _compute_parallel_worker_count(
            requested_workers=config["workers"],
            num_tasks=len(pending_task_data_list),
        )
        multiprocessing_context = _build_multiprocessing_context()
        _log(
            "Launching chunk workers: "
            f"{worker_count} workers for {len(pending_task_data_list)} chunks"
        )

        for task_data in pending_task_data_list:
            manifest["chunks"][status_index[task_data["task_id"]]][
                "started_at"
            ] = _timestamp()
        _write_json_atomic(manifest_path, manifest)

        encountered_failure = False
        try:
            with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing_context) as executor:
                future_index = {
                    executor.submit(_run_chunk_task, task_data): task_data
                    for task_data in pending_task_data_list
                }
                completed_future_count = 0
                for future in as_completed(future_index):
                    task_data = future_index[future]
                    chunk_entry = manifest["chunks"][
                        status_index[task_data["task_id"]]
                    ]
                    try:
                        chunk_result = future.result()
                        chunk_entry["status"] = "completed"
                        chunk_entry["completed_at"] = _timestamp()
                        chunk_entry["error"] = None
                        chunk_entry["section_stats"] = chunk_result.get(
                            "section_stats"
                        )
                        completed_future_count += 1
                        _log(
                            f"Completed {chunk_result['task_id']} "
                            f"({completed_future_count}/"
                            f"{len(pending_task_data_list)}) "
                            f"in {chunk_result['duration_seconds']:.1f}s"
                        )
                    except Exception as exc:
                        chunk_entry["status"] = "failed"
                        chunk_entry["completed_at"] = _timestamp()
                        chunk_entry["error"] = repr(exc)
                        encountered_failure = True
                        _log(f"FAILED {task_data['task_id']}: {exc!r}")
                    finally:
                        _update_manifest_summary(manifest)
                        _write_json_atomic(manifest_path, manifest)
        except PermissionError:
            _log(
                "ProcessPoolExecutor unavailable in current environment; "
                "falling back to sequential chunk execution"
            )
            _run_pending_tasks_sequentially(
                pending_task_data_list=pending_task_data_list,
                manifest=manifest,
                status_index=status_index,
                manifest_path=manifest_path,
            )
        else:
            if encountered_failure:
                manifest["final_outputs"]["status"] = "failed"
                _update_manifest_summary(manifest)
                _write_json_atomic(manifest_path, manifest)
                raise RuntimeError(
                    "At least one chunk failed. Re-run submit with --resume."
                )
    else:
        _log("All chunk outputs already present; skipping worker launch")

    _log("Merging chunk outputs")
    merged_result = _merge_outputs(
        config=config,
        task_data_list=task_data_list,
        output_path=config["final_output_path"],
        include_plot=True,
        plot_output_path=config["final_plot_path"],
    )
    if "cluster_num_attempts" in merged_result:
        manifest["final_outputs"]["cluster_summary"] = {
            "cluster_update_enabled": bool(
                merged_result["cluster_update_enabled"]
            ),
            "cluster_update_adaptive": bool(
                merged_result["cluster_update_adaptive"]
            ),
            "cluster_budget_fraction_rho": float(
                merged_result["cluster_budget_fraction_rho"]
            ),
            "cluster_num_attempts": int(
                merged_result["cluster_num_attempts"]
            ),
            "cluster_num_nonzero_moves": int(
                merged_result["cluster_num_nonzero_moves"]
            ),
            "cluster_total_wall_time": float(
                merged_result["cluster_total_wall_time"]
            ),
            "cluster_wall_time_fraction": float(
                merged_result["cluster_wall_time_fraction"]
            ),
            "cluster_nullity_mean": float(
                merged_result["cluster_nullity_mean"]
            ),
            "cluster_nullity_median": float(
                merged_result["cluster_nullity_median"]
            ),
            "cluster_move_fraction_mean": float(
                merged_result["cluster_move_fraction_mean"]
            ),
            "cluster_controller_frozen": bool(
                merged_result["cluster_controller_frozen"]
            ),
        }
    if "adaptive_pt_round_f_mono_tensor" in merged_result:
        f_mono = np.asarray(
            merged_result["adaptive_pt_round_f_mono_tensor"],
            dtype=np.float64,
        )
        f_target = np.asarray(
            merged_result["adaptive_pt_round_f_target_tensor"],
            dtype=np.float64,
        )
        finite_delta = np.abs(f_mono - f_target)
        finite_delta = finite_delta[np.isfinite(finite_delta)]
        manifest["final_outputs"]["adaptive_pt_summary"] = {
            "adaptive_pt_rounds": int(
                merged_result.get("adaptive_pt_rounds", 0)
            ),
            "adaptive_pt_calibration_sweeps": int(
                merged_result.get("adaptive_pt_calibration_sweeps", 0)
            ),
            "mean_abs_flow_error": (
                None
                if finite_delta.size == 0
                else float(np.mean(finite_delta))
            ),
            "max_abs_flow_error": (
                None
                if finite_delta.size == 0
                else float(np.max(finite_delta))
            ),
        }
    section_summary = _section_summary_for_manifest(merged_result)
    if section_summary is not None:
        manifest["final_outputs"]["section_summary"] = section_summary
    manifest["final_outputs"]["status"] = "completed"
    manifest["final_outputs"]["completed_at"] = _timestamp()
    _update_manifest_summary(manifest)
    _write_json_atomic(manifest_path, manifest)

    _log(f"Final NPZ: {config['final_output_path']}")
    _log(f"Final PNG: {config['final_plot_path']}")
    _log(
        "Chunk counts: "
        f"completed={manifest['summary']['completed_chunks']} "
        f"failed={manifest['summary']['failed_chunks']} "
        f"pending={manifest['summary']['pending_chunks']}"
    )
    return 0


def _run_chunk_command(args):
    if int(args.pt_swap_sweeps_per_attempt) < 1:
        raise ValueError("--pt-swap-sweeps-per-attempt must be >= 1")
    if int(args.pt_cold_edge_swap_stride) < 1:
        raise ValueError("--pt-cold-edge-swap-stride must be >= 1")
    pt_ladder_spacing_power = float(args.pt_ladder_spacing_power)
    if (
            not np.isfinite(pt_ladder_spacing_power)
            or pt_ladder_spacing_power <= 0.0):
        raise ValueError("--pt-ladder-spacing-power must be finite and positive")
    if args.pt_ladder_mode != "sync_enlarge" and pt_ladder_spacing_power != 1.0:
        raise ValueError(
            "--pt-ladder-spacing-power is only supported with "
            "--pt-ladder-mode sync_enlarge"
        )
    task_data = {
        "lattice_index": int(args.lattice_index),
        "point_index": int(args.point_index),
        "lattice_size": int(args.lattice_size),
        "code_family": str(args.code_family),
        "data_error_probability": float(args.data_error_probability),
        "syndrome_error_probability": float(args.syndrome_error_probability),
        "num_disorder_samples": int(args.num_disorder_samples),
        "disorder_offset": int(args.disorder_offset),
        "chunk_index": int(args.chunk_index),
        "q0_num_start_chains": int(args.q0_num_start_chains),
        "num_start_chains": _resolve_cli_num_start_chains(
            raw_num_start_chains=args.num_start_chains,
            q0_num_start_chains=args.q0_num_start_chains,
            syndrome_error_probability=args.syndrome_error_probability,
        ),
        "num_replicas_per_start": int(args.num_replicas_per_start),
        "pt_p_hot": (
            None if args.pt_p_hot is None else float(args.pt_p_hot)
        ),
        "pt_num_temperatures": (
            None
            if args.pt_num_temperatures is None
            else int(args.pt_num_temperatures)
        ),
        "pt_ladder_mode": str(args.pt_ladder_mode),
        "pt_ladder_semantics": _pt_ladder_semantics(args.pt_ladder_mode),
        "pt_q_hot": (
            None if args.pt_q_hot is None else float(args.pt_q_hot)
        ),
        "pt_ladder_spacing_power": pt_ladder_spacing_power,
        "adaptive_pt_rounds": int(args.adaptive_pt_rounds),
        "adaptive_pt_calibration_sweeps": int(
            args.adaptive_pt_calibration_sweeps
        ),
        "pt_swap_attempt_every_num_sweeps": int(
            args.pt_swap_attempt_every_num_sweeps
        ),
        "pt_swap_sweeps_per_attempt": int(args.pt_swap_sweeps_per_attempt),
        "pt_cold_edge_swap_stride": int(args.pt_cold_edge_swap_stride),
        "num_zero_syndrome_sweeps_per_cycle": int(
            args.num_zero_syndrome_sweeps_per_cycle
        ),
        "winding_repeat_factor": int(args.winding_repeat_factor),
        "winding_plane_heatbath_sweeps": int(
            args.winding_plane_heatbath_sweeps
        ),
        "single_bit_proposal_fraction": float(
            args.single_bit_proposal_fraction
        ),
        "observable_temperature_mode": str(args.observable_temperature_mode),
        "track_pt_sector_diagnostics": bool(
            args.track_pt_sector_diagnostics
        ),
        "pt_sector_diagnostic_stride": int(args.pt_sector_diagnostic_stride),
        "cluster_update_enabled": not bool(args.disable_cluster_update),
        "cluster_budget_fraction_rho": float(args.cluster_budget_fraction_rho),
        "cluster_update_debug": bool(args.cluster_debug_assertions),
        "common_random_disorder_across_p": bool(
            args.common_random_disorder_across_p
        ),
        "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
        "effective_num_burn_in_sweeps": int(args.effective_num_burn_in_sweeps),
        "num_sweeps_between_measurements": int(
            args.num_sweeps_between_measurements
        ),
        "num_measurements_per_disorder": int(
            args.num_measurements_per_disorder
        ),
        "burn_in_scaling_reference_num_qubits": int(
            args.burn_in_scaling_reference_num_qubits
        ),
        "seed": int(args.seed),
        "disorder_seed": int(args.disorder_seed),
        "output_path": str(Path(args.output_path).expanduser().resolve()),
    }
    task_data["task_id"] = _task_identifier(task_data)
    chunk_result = _run_chunk_task(task_data)
    _log(
        f"Completed {chunk_result['task_id']} "
        f"in {chunk_result['duration_seconds']:.1f}s"
    )
    return 0


def _merge_command(args):
    config = _build_submit_config_from_args(args)
    (
        task_data_list,
        _,
        _,
        _,
    ) = _build_chunk_tasks(config)
    _merge_outputs(
        config=config,
        task_data_list=task_data_list,
        output_path=config["final_output_path"],
        include_plot=True,
        plot_output_path=config["final_plot_path"],
    )
    _log(f"Final NPZ: {config['final_output_path']}")
    _log(f"Final PNG: {config['final_plot_path']}")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Chunked production scan runner for toric code scans.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_submit_parser = argparse.ArgumentParser(add_help=False)
    common_submit_parser.add_argument("--run-root", required=True)
    common_submit_parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        default="2d_toric",
    )
    common_submit_parser.add_argument("--workers", type=int, default=1)
    common_submit_parser.add_argument("--chunk-size", type=int, default=32)
    common_submit_parser.add_argument(
        "--num-disorder-samples-total",
        type=int,
        required=True,
    )
    common_submit_parser.add_argument(
        "--data-error-probabilities",
        required=True,
        help="Comma-separated list, e.g. 0.09,0.0925,0.095",
    )
    common_submit_parser.add_argument(
        "--lattice-sizes",
        required=True,
        help="Comma-separated list, e.g. 3,5,7",
    )
    common_submit_parser.add_argument(
        "--syndrome-error-probability",
        type=float,
        default=0.0,
    )
    common_submit_parser.add_argument(
        "--num-burn-in-sweeps",
        type=int,
        required=True,
    )
    common_submit_parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        required=True,
    )
    common_submit_parser.add_argument(
        "--num-measurements-per-disorder",
        type=int,
        required=True,
    )
    common_submit_parser.add_argument(
        "--q0-num-start-chains",
        type=int,
        default=4,
    )
    common_submit_parser.add_argument(
        "--num-start-chains",
        type=int,
        default=None,
    )
    common_submit_parser.add_argument(
        "--num-replicas-per-start",
        type=int,
        default=1,
    )
    common_submit_parser.add_argument(
        "--pt-p-hot",
        type=float,
        default=None,
    )
    common_submit_parser.add_argument(
        "--pt-num-temperatures",
        type=int,
        default=None,
    )
    common_submit_parser.add_argument(
        "--pt-ladder-mode",
        choices=("data_only", "sync_enlarge"),
        default="data_only",
    )
    common_submit_parser.add_argument(
        "--pt-q-hot",
        type=float,
        default=None,
    )
    common_submit_parser.add_argument(
        "--pt-ladder-spacing-power",
        type=float,
        default=1.0,
        help=(
            "Shape sync_enlarge PT heat-scale positions as x**power. Values "
            ">1 place more temperatures near the cold endpoint while keeping "
            "the same q_hot. The default 1.0 preserves uniform log spacing."
        ),
    )
    common_submit_parser.add_argument(
        "--adaptive-pt-rounds",
        type=int,
        default=0,
    )
    common_submit_parser.add_argument(
        "--adaptive-pt-calibration-sweeps",
        type=int,
        default=128,
    )
    common_submit_parser.add_argument(
        "--pt-swap-attempt-every-num-sweeps",
        type=int,
        default=1,
    )
    common_submit_parser.add_argument(
        "--pt-swap-sweeps-per-attempt",
        type=int,
        default=1,
        help=(
            "Number of alternating adjacent PT swap sweeps to run whenever "
            "the swap cadence fires. The default 1 preserves old behavior."
        ),
    )
    common_submit_parser.add_argument(
        "--pt-cold-edge-swap-stride",
        type=int,
        default=1,
        help=(
            "Attempt the cold adjacent PT edge (temperature pair 0-1) only "
            "once every N eligible alternating swap sweeps. The default 1 "
            "preserves old behavior."
        ),
    )
    common_submit_parser.add_argument(
        "--num-zero-syndrome-sweeps-per-cycle",
        type=int,
        default=1,
    )
    common_submit_parser.add_argument(
        "--winding-repeat-factor",
        type=int,
        default=1,
    )
    common_submit_parser.add_argument(
        "--winding-plane-heatbath-sweeps",
        type=int,
        default=0,
        help=(
            "Number of exact heatbath sweeps over parallel nontrivial winding "
            "planes per update cycle. The default 0 preserves old behavior."
        ),
    )
    common_submit_parser.add_argument(
        "--single-bit-proposal-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of qubits attempted in the q>0 single-bit stage per "
            "update cycle. The default 1.0 preserves a full sweep."
        ),
    )
    common_submit_parser.add_argument(
        "--observable-temperature-mode",
        choices=("all", "cold"),
        default="all",
        help=(
            "For parallel tempering, accumulate logical observables at all "
            "temperatures or only at the cold target temperature."
        ),
    )
    common_submit_parser.add_argument(
        "--track-pt-sector-diagnostics",
        action="store_true",
        help=(
            "Record compressed per-temperature logical-sector flip diagnostics "
            "for PT runs. This adds all-temperature logical observable probes."
        ),
    )
    common_submit_parser.add_argument(
        "--pt-sector-diagnostic-stride",
        type=int,
        default=1,
        help=(
            "When PT sector diagnostics are enabled, probe logical-sector "
            "signatures once every N measurements. The default 1 preserves "
            "the historical per-measurement diagnostics."
        ),
    )
    common_submit_parser.add_argument(
        "--disable-cluster-update",
        action="store_true",
        help="Disable the q>0 adaptive cluster update.",
    )
    common_submit_parser.add_argument(
        "--cluster-budget-fraction-rho",
        type=float,
        default=0.05,
    )
    common_submit_parser.add_argument(
        "--cluster-debug-assertions",
        action="store_true",
    )
    common_submit_parser.add_argument(
        "--seed-base",
        type=int,
        required=True,
    )
    common_submit_parser.add_argument(
        "--burn-in-scaling-reference-num-qubits",
        type=int,
        default=DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS,
    )
    common_submit_parser.add_argument(
        "--max-effective-num-burn-in-sweeps",
        type=int,
        default=None,
        help=(
            "Optional cap applied after lattice-size burn-in scaling. "
            "Leave unset to preserve the historical scaling rule."
        ),
    )
    common_submit_parser.add_argument(
        "--git-commit-sha",
        default=None,
    )
    common_submit_parser.add_argument(
        "--output-stem",
        default=None,
    )
    common_submit_parser.add_argument(
        "--common-random-disorder-across-p",
        action="store_true",
    )

    submit_parser = subparsers.add_parser(
        "submit",
        parents=[common_submit_parser],
    )
    submit_parser.add_argument(
        "--merge-only",
        action="store_true",
    )
    submit_parser.add_argument(
        "--resume",
        action="store_true",
    )

    subparsers.add_parser(
        "merge",
        parents=[common_submit_parser],
    )

    run_chunk_parser = subparsers.add_parser("run-chunk")
    run_chunk_parser.add_argument("--output-path", required=True)
    run_chunk_parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        default="2d_toric",
    )
    run_chunk_parser.add_argument("--lattice-index", type=int, required=True)
    run_chunk_parser.add_argument("--point-index", type=int, required=True)
    run_chunk_parser.add_argument("--lattice-size", type=int, required=True)
    run_chunk_parser.add_argument(
        "--data-error-probability",
        type=float,
        required=True,
    )
    run_chunk_parser.add_argument(
        "--syndrome-error-probability",
        type=float,
        default=0.0,
    )
    run_chunk_parser.add_argument(
        "--num-disorder-samples",
        type=int,
        required=True,
    )
    run_chunk_parser.add_argument("--disorder-offset", type=int, required=True)
    run_chunk_parser.add_argument("--chunk-index", type=int, required=True)
    run_chunk_parser.add_argument(
        "--num-burn-in-sweeps",
        type=int,
        required=True,
    )
    run_chunk_parser.add_argument(
        "--effective-num-burn-in-sweeps",
        type=int,
        required=True,
    )
    run_chunk_parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        required=True,
    )
    run_chunk_parser.add_argument(
        "--num-measurements-per-disorder",
        type=int,
        required=True,
    )
    run_chunk_parser.add_argument(
        "--q0-num-start-chains",
        type=int,
        default=4,
    )
    run_chunk_parser.add_argument(
        "--num-start-chains",
        type=int,
        default=None,
    )
    run_chunk_parser.add_argument(
        "--num-replicas-per-start",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--pt-p-hot",
        type=float,
        default=None,
    )
    run_chunk_parser.add_argument(
        "--pt-num-temperatures",
        type=int,
        default=None,
    )
    run_chunk_parser.add_argument(
        "--pt-ladder-mode",
        choices=("data_only", "sync_enlarge"),
        default="data_only",
    )
    run_chunk_parser.add_argument(
        "--pt-q-hot",
        type=float,
        default=None,
    )
    run_chunk_parser.add_argument(
        "--pt-ladder-spacing-power",
        type=float,
        default=1.0,
    )
    run_chunk_parser.add_argument(
        "--adaptive-pt-rounds",
        type=int,
        default=0,
    )
    run_chunk_parser.add_argument(
        "--adaptive-pt-calibration-sweeps",
        type=int,
        default=128,
    )
    run_chunk_parser.add_argument(
        "--pt-swap-attempt-every-num-sweeps",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--pt-swap-sweeps-per-attempt",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--pt-cold-edge-swap-stride",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--num-zero-syndrome-sweeps-per-cycle",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--winding-repeat-factor",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--winding-plane-heatbath-sweeps",
        type=int,
        default=0,
    )
    run_chunk_parser.add_argument(
        "--single-bit-proposal-fraction",
        type=float,
        default=1.0,
    )
    run_chunk_parser.add_argument(
        "--observable-temperature-mode",
        choices=("all", "cold"),
        default="all",
    )
    run_chunk_parser.add_argument(
        "--track-pt-sector-diagnostics",
        action="store_true",
    )
    run_chunk_parser.add_argument(
        "--pt-sector-diagnostic-stride",
        type=int,
        default=1,
    )
    run_chunk_parser.add_argument(
        "--disable-cluster-update",
        action="store_true",
    )
    run_chunk_parser.add_argument(
        "--cluster-budget-fraction-rho",
        type=float,
        default=0.05,
    )
    run_chunk_parser.add_argument(
        "--cluster-debug-assertions",
        action="store_true",
    )
    run_chunk_parser.add_argument(
        "--common-random-disorder-across-p",
        action="store_true",
    )
    run_chunk_parser.add_argument(
        "--burn-in-scaling-reference-num-qubits",
        type=int,
        default=DEFAULT_BURN_IN_SCALING_REFERENCE_NUM_QUBITS,
    )
    run_chunk_parser.add_argument("--seed", type=int, required=True)
    run_chunk_parser.add_argument("--disorder-seed", type=int, required=True)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        return _submit_run(args)
    if args.command == "merge":
        return _merge_command(args)
    if args.command == "run-chunk":
        return _run_chunk_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
