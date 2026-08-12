#!/usr/bin/env python3
"""Run the q=0.001 transport sentinel on exact legacy disorder samples.

This task-local runner deliberately lives outside ``src``.  It reuses the
tracked sampler without changing shared semantics, reconstructs the complete
legacy 32-disorder RNG block before selecting its first 16 rows, and fans the
two chain-length arms out as 32 independent, safely resumable tasks.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np


PER_DISORDER_KEYS = (
    "disorder_q_top_values",
    "logical_observable_mean_values_per_disorder",
    "average_acceptance_rate_per_disorder",
    "chain_logical_observable_mean_values_per_disorder_per_start_replica",
    "chain_q_top_values_per_disorder_per_start_replica",
    "chain_average_acceptance_rate_per_disorder_per_start_replica",
    "chain_contractible_acceptance_rate_per_disorder_per_start_replica",
    "chain_winding_acceptance_rate_per_disorder_per_start_replica",
    "chain_ordinary_update_wall_time_per_disorder_per_start_replica",
    "chain_pt_swap_wall_time_per_disorder_per_start_replica",
    "chain_observable_wall_time_per_disorder_per_start_replica",
    "chain_measurement_wall_time_per_disorder_per_start_replica",
    "chain_q_top_block_m_u_values_per_disorder_per_start_replica",
    "chain_q_top_block_values_per_disorder_per_start_replica",
    "chain_q_top_block_drift_per_disorder_per_start_replica",
    "chain_q_top_block_range_per_disorder_per_start_replica",
    "chain_q_top_last_half_minus_full_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_counts_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_block_counts_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_first_half_counts_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_second_half_counts_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica",
    "chain_cold_sector_histogram_adjacent_block_tv_per_disorder_per_start_replica",
    "q_top_block_m_u_values_per_disorder",
    "q_top_block_values_per_disorder",
    "q_top_block_drift_per_disorder",
    "q_top_block_range_per_disorder",
    "q_top_last_half_minus_full_per_disorder",
    "cold_sector_histogram_counts_per_disorder",
    "cold_sector_histogram_block_counts_per_disorder",
    "q_top_spread_per_disorder",
    "m_u_spread_linf_per_disorder",
    "max_r_hat_per_disorder",
    "min_effective_sample_size_per_disorder",
    "num_chains_that_never_flipped_sector_per_disorder",
)

INVARIANT_RESULT_KEYS = (
    "start_sector_labels",
    "q_positive_initial_chain_mode",
    "num_start_chains",
    "num_replicas_per_start",
    "q_top_block_count",
    "num_zero_syndrome_sweeps_per_cycle",
    "winding_repeat_factor",
    "winding_plane_heatbath_sweeps",
    "pt_enabled",
)


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def reconstruct_legacy_uniforms(config):
    sentinel = config["q_positive_sentinel"]
    legacy = sentinel["legacy_reference"]
    rng = np.random.default_rng(int(legacy["disorder_seed"]))
    full_chunk_size = int(legacy["legacy_chunk_size"])
    syndrome_uniforms = rng.random(
        (full_chunk_size, int(legacy["num_checks"]))
    )
    data_uniforms = rng.random(
        (full_chunk_size, int(legacy["num_qubits"]))
    )
    sample_count = int(sentinel["num_disorder_samples"])
    return (
        syndrome_uniforms[:sample_count].copy(),
        data_uniforms[:sample_count].copy(),
    )


def verify_legacy_reconstruction(config):
    sentinel = config["q_positive_sentinel"]
    legacy = sentinel["legacy_reference"]
    syndrome_uniforms, data_uniforms = reconstruct_legacy_uniforms(config)
    syndrome_bits = (
        syndrome_uniforms < float(sentinel["syndrome_error_probability"])
    )
    data_bits = data_uniforms < float(sentinel["data_error_probability"])
    observed = {
        "syndrome_uniforms_first16_sha256": _sha256_array(
            syndrome_uniforms
        ),
        "data_uniforms_first16_sha256": _sha256_array(data_uniforms),
        "syndrome_bits_first16_sha256": _sha256_array(syndrome_bits),
        "data_bits_first16_sha256": _sha256_array(data_bits),
    }
    mismatches = {
        key: {"expected": legacy[key], "observed": value}
        for key, value in observed.items()
        if value != legacy[key]
    }
    if mismatches:
        raise RuntimeError(
            "legacy disorder reconstruction hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return syndrome_uniforms, data_uniforms, observed


def _configure_repo_imports(repo_root):
    source_dir = str(Path(repo_root).resolve() / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)


def _part_path(run_root, arm, disorder_index):
    return (
        Path(run_root)
        / "parts"
        / arm
        / f"disorder_{int(disorder_index):03d}.npz"
    )


def _part_is_valid(path, arm, disorder_index, source_commit):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["arm"].item()) == str(arm)
                and int(data["disorder_index"].item())
                == int(disorder_index)
                and str(data["source_commit"].item()) == source_commit
                and all(key in data.files for key in PER_DISORDER_KEYS)
            )
    except (OSError, ValueError, KeyError):
        return False


def _run_one_task(task):
    repo_root = Path(task["repo_root"])
    _configure_repo_imports(repo_root)
    from build_toric_code_examples import (
        build_toric_code_by_family,
        build_zero_syndrome_move_data_by_family,
    )
    from main import run_disorder_average_simulation

    code_family = str(task["code_family"])
    lattice_size = int(task["lattice_size"])
    parity_check_matrix, logical_z_basis = build_toric_code_by_family(
        code_family=code_family,
        lattice_size=lattice_size,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family=code_family,
        lattice_size=lattice_size,
    )
    if parity_check_matrix.shape != (
        int(task["num_checks"]),
        int(task["num_qubits"]),
    ):
        raise RuntimeError(
            "2D code shape changed: "
            f"observed={parity_check_matrix.shape}, "
            f"expected={(task['num_checks'], task['num_qubits'])}"
        )

    result = run_disorder_average_simulation(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=logical_z_basis,
        syndrome_error_probability=float(task["q"]),
        data_error_probability=float(task["p"]),
        num_disorder_samples=1,
        num_burn_in_sweeps=int(task["effective_num_burn_in_sweeps"]),
        num_sweeps_between_measurements=int(
            task["num_sweeps_between_measurements"]
        ),
        num_measurements_per_disorder=int(
            task["num_measurements_per_disorder"]
        ),
        seed=int(task["mcmc_seed"]),
        zero_syndrome_move_data=zero_syndrome_move_data,
        q0_num_start_chains=int(task["num_start_chains"]),
        num_start_chains=int(task["num_start_chains"]),
        num_replicas_per_start=int(task["num_replicas_per_start"]),
        pt_p_hot=None,
        pt_num_temperatures=None,
        num_zero_syndrome_sweeps_per_cycle=int(
            task["num_zero_syndrome_sweeps_per_cycle"]
        ),
        winding_repeat_factor=int(task["winding_repeat_factor"]),
        winding_plane_heatbath_sweeps=int(
            task["winding_plane_heatbath_sweeps"]
        ),
        q_top_block_count=int(task["q_top_block_count"]),
        q_positive_initial_chain_mode="sector",
        cluster_update_enabled=False,
        precomputed_syndrome_uniform_values_per_disorder=np.asarray(
            task["syndrome_uniforms"], dtype=np.float64
        )[None, :],
        precomputed_data_uniform_values_per_disorder=np.asarray(
            task["data_uniforms"], dtype=np.float64
        )[None, :],
    )

    payload = {
        "schema_version": np.int64(1),
        "arm": np.array(task["arm"]),
        "disorder_index": np.int64(task["disorder_index"]),
        "mcmc_seed": np.uint64(task["mcmc_seed"]),
        "legacy_disorder_seed": np.uint64(task["legacy_disorder_seed"]),
        "source_commit": np.array(task["source_commit"]),
        "lattice_size": np.int64(lattice_size),
        "data_error_probability": np.float64(task["p"]),
        "syndrome_error_probability": np.float64(task["q"]),
        "base_num_burn_in_sweeps": np.int64(
            task["base_num_burn_in_sweeps"]
        ),
        "effective_num_burn_in_sweeps": np.int64(
            task["effective_num_burn_in_sweeps"]
        ),
        "num_sweeps_between_measurements": np.int64(
            task["num_sweeps_between_measurements"]
        ),
        "num_measurements_per_disorder": np.int64(
            task["num_measurements_per_disorder"]
        ),
        "syndrome_uniforms_sha256": np.array(
            _sha256_array(task["syndrome_uniforms"])
        ),
        "data_uniforms_sha256": np.array(
            _sha256_array(task["data_uniforms"])
        ),
    }
    for key in PER_DISORDER_KEYS + INVARIANT_RESULT_KEYS:
        if key not in result:
            raise KeyError(f"sampler result is missing required field {key!r}")
        payload[key] = result[key]

    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(temporary_path, **payload)
    temporary_path.replace(output_path)
    return {
        "arm": task["arm"],
        "disorder_index": int(task["disorder_index"]),
        "output_path": str(output_path),
    }


def _aggregate_arm(config, run_root, arm, source_commit):
    sentinel = config["q_positive_sentinel"]
    sample_count = int(sentinel["num_disorder_samples"])
    paths = [_part_path(run_root, arm, index) for index in range(sample_count)]
    invalid = [
        str(path)
        for index, path in enumerate(paths)
        if not _part_is_valid(path, arm, index, source_commit)
    ]
    if invalid:
        raise RuntimeError(f"cannot aggregate {arm}; invalid parts: {invalid}")

    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        merged = {
            key: np.concatenate(
                [np.asarray(data[key]) for data in loaded], axis=0
            )
            for key in PER_DISORDER_KEYS
        }
        for key in INVARIANT_RESULT_KEYS:
            first = np.asarray(loaded[0][key])
            if not all(np.array_equal(first, np.asarray(data[key])) for data in loaded[1:]):
                raise RuntimeError(f"invariant field differs across parts: {key}")
            merged[key] = first
        legacy = sentinel["legacy_reference"]
        merged.update({
            "schema_version": np.int64(1),
            "arm": np.array(arm),
            "created_at": np.array(_timestamp()),
            "source_commit": np.array(source_commit),
            "lattice_size": np.int64(sentinel["lattice_size"]),
            "data_error_probability": np.float64(
                sentinel["data_error_probability"]
            ),
            "syndrome_error_probability": np.float64(
                sentinel["syndrome_error_probability"]
            ),
            "num_disorder_samples": np.int64(sample_count),
            "base_num_burn_in_sweeps": np.int64(
                sentinel["base_num_burn_in_sweeps"]
            ),
            "effective_num_burn_in_sweeps": np.int64(
                loaded[0]["effective_num_burn_in_sweeps"].item()
            ),
            "num_sweeps_between_measurements": np.int64(
                sentinel["num_sweeps_between_measurements"]
            ),
            "num_measurements_per_disorder": np.int64(
                sentinel["arms"][arm]["num_measurements_per_disorder"]
            ),
            "mcmc_seed_per_disorder": np.asarray(
                [data["mcmc_seed"].item() for data in loaded],
                dtype=np.uint64,
            ),
            "legacy_disorder_seed": np.uint64(legacy["disorder_seed"]),
            "legacy_source_commit_recorded": np.array(
                legacy["source_commit_recorded"]
            ),
            "legacy_source_npz_sha256": np.array(
                legacy["source_npz_sha256"]
            ),
            "legacy_q_top_first16": np.asarray(
                legacy["q_top_first16"], dtype=np.float64
            ),
            "syndrome_uniforms_first16_sha256": np.array(
                legacy["syndrome_uniforms_first16_sha256"]
            ),
            "data_uniforms_first16_sha256": np.array(
                legacy["data_uniforms_first16_sha256"]
            ),
            "cluster_update_enabled_config": np.bool_(False),
            "parallel_tempering_enabled_config": np.bool_(False),
        })
        output_path = Path(run_root) / f"qpositive_sentinel_{arm}.npz"
        temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
        np.savez_compressed(temporary_path, **merged)
        temporary_path.replace(output_path)
        return output_path
    finally:
        for data in loaded:
            data.close()


def run(config_path, repo_root, run_root, workers, source_commit, resume):
    config = _load_json(config_path)
    if config["source_commit"] != source_commit:
        raise RuntimeError(
            "source commit does not match config: "
            f"{source_commit} != {config['source_commit']}"
        )
    marker = Path(repo_root) / "SOURCE_COMMIT"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != source_commit:
        raise RuntimeError(f"missing or incorrect immutable source marker: {marker}")

    _configure_repo_imports(repo_root)
    from build_toric_code_examples import build_toric_code_by_family
    from production_chunked_scan import _effective_num_burn_in_sweeps

    syndrome_uniforms, data_uniforms, observed_hashes = (
        verify_legacy_reconstruction(config)
    )
    sentinel = config["q_positive_sentinel"]
    legacy = sentinel["legacy_reference"]
    parity_check_matrix, _ = build_toric_code_by_family(
        code_family=config["code_family"],
        lattice_size=int(sentinel["lattice_size"]),
    )
    effective_burn_in = _effective_num_burn_in_sweeps(
        num_burn_in_sweeps=int(sentinel["base_num_burn_in_sweeps"]),
        num_qubits=int(parity_check_matrix.shape[1]),
        burn_in_scaling_reference_num_qubits=int(
            sentinel["burn_in_scaling_reference_num_qubits"]
        ),
    )

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    task_list = []
    for arm, arm_config in sentinel["arms"].items():
        for disorder_index in range(int(sentinel["num_disorder_samples"])):
            output_path = _part_path(run_root, arm, disorder_index)
            if resume and _part_is_valid(
                output_path, arm, disorder_index, source_commit
            ):
                continue
            task_list.append({
                "repo_root": str(Path(repo_root).resolve()),
                "output_path": str(output_path),
                "source_commit": source_commit,
                "code_family": config["code_family"],
                "arm": arm,
                "disorder_index": disorder_index,
                "lattice_size": int(sentinel["lattice_size"]),
                "p": float(sentinel["data_error_probability"]),
                "q": float(sentinel["syndrome_error_probability"]),
                "num_checks": int(legacy["num_checks"]),
                "num_qubits": int(legacy["num_qubits"]),
                "legacy_disorder_seed": int(legacy["disorder_seed"]),
                "mcmc_seed": int(
                    sentinel["mcmc_seed_base"] + 1000003 * disorder_index
                ),
                "base_num_burn_in_sweeps": int(
                    sentinel["base_num_burn_in_sweeps"]
                ),
                "effective_num_burn_in_sweeps": int(effective_burn_in),
                "num_sweeps_between_measurements": int(
                    sentinel["num_sweeps_between_measurements"]
                ),
                "num_measurements_per_disorder": int(
                    arm_config["num_measurements_per_disorder"]
                ),
                "num_start_chains": int(sentinel["num_start_chains"]),
                "num_replicas_per_start": int(
                    sentinel["num_replicas_per_start"]
                ),
                "num_zero_syndrome_sweeps_per_cycle": int(
                    sentinel["num_zero_syndrome_sweeps_per_cycle"]
                ),
                "winding_repeat_factor": int(
                    sentinel["winding_repeat_factor"]
                ),
                "winding_plane_heatbath_sweeps": int(
                    sentinel["winding_plane_heatbath_sweeps"]
                ),
                "q_top_block_count": int(sentinel["q_top_block_count"]),
                "syndrome_uniforms": syndrome_uniforms[disorder_index],
                "data_uniforms": data_uniforms[disorder_index],
            })

    print(
        f"[{_timestamp()}] q-positive sentinel pending_tasks={len(task_list)} "
        f"workers={workers} effective_burn_in={effective_burn_in}",
        flush=True,
    )
    if task_list:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_run_one_task, task) for task in task_list]
            for future in as_completed(futures):
                completed = future.result()
                print(
                    f"[{_timestamp()}] completed arm={completed['arm']} "
                    f"disorder={completed['disorder_index']}",
                    flush=True,
                )

    outputs = {
        arm: str(_aggregate_arm(config, run_root, arm, source_commit))
        for arm in sentinel["arms"]
    }
    summary = {
        "schema_version": 1,
        "created_at": _timestamp(),
        "source_commit": source_commit,
        "legacy_disorder_hashes": observed_hashes,
        "effective_num_burn_in_sweeps": int(effective_burn_in),
        "outputs": outputs,
    }
    summary_path = run_root / "qpositive_sentinel_run_summary.json"
    temporary_summary_path = summary_path.with_name(summary_path.name + ".tmp")
    with temporary_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    temporary_summary_path.replace(summary_path)
    print(f"[{_timestamp()}] sentinel outputs: {outputs}", flush=True)


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    run(
        config_path=args.config,
        repo_root=args.repo_root,
        run_root=args.run_root,
        workers=args.workers,
        source_commit=args.source_commit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
