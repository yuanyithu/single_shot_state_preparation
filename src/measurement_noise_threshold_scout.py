import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from analyze_threshold_crossing import analyze_threshold_crossing
from build_toric_code_examples import SUPPORTED_CODE_FAMILIES
from production_chunked_scan import _format_probability_tag


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_LATTICE_SIZES_BY_CODE_FAMILY = {
    "2d_toric": [3, 5, 7],
    "3d_toric": [3, 4, 5],
}
DEFAULT_Q_VALUES = [0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]
DEFAULT_P_VALUES = np.arange(0.0300, 0.1000 + 0.0001, 0.0050)
DEFAULT_COMMON_RANDOM_DISORDER_ACROSS_P = True
DEFAULT_3D_NUM_START_CHAINS = 8
DEFAULT_3D_NUM_REPLICAS_PER_START = 2
DEFAULT_3D_PT_P_HOT = 0.44
DEFAULT_3D_PT_NUM_TEMPERATURES = 9


def _timestamp_tag():
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _csv_from_float_list(values):
    return ",".join(f"{float(value):0.4f}" for value in values)


def _csv_from_int_list(values):
    return ",".join(str(int(value)) for value in values)


def _build_default_run_root(code_family, run_id):
    if code_family == "2d_toric":
        code_family_dir = "2d_toric_code"
    elif code_family == "3d_toric":
        code_family_dir = "3d_toric_code"
    else:
        raise ValueError(
            f"Unsupported code_family={code_family!r}; "
            f"expected one of {SUPPORTED_CODE_FAMILIES}"
        )
    return (
        PROJECT_ROOT
        / "data"
        / code_family_dir
        / "with_measurement_noise"
        / "measurement_noise_threshold_scout_local"
        / run_id
    )


def _run_command(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def _build_q_output_stem(code_family, q_value):
    q_tag = _format_probability_tag(q_value)
    return (
        f"scan_result_multi_L_{code_family}_q{q_tag}_"
        "measurement_noise_threshold_scout_common_random"
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run local scouting scans for q>0 threshold search."
    )
    parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        default="2d_toric",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Defaults to "
            "data/<code_family>_code/with_measurement_noise/"
            "measurement_noise_threshold_scout_local/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--lattice-sizes",
        default=None,
    )
    parser.add_argument(
        "--q-values",
        default=_csv_from_float_list(DEFAULT_Q_VALUES),
    )
    parser.add_argument(
        "--data-error-probabilities",
        default=_csv_from_float_list(DEFAULT_P_VALUES),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-disorder-samples-total", type=int, default=256)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=1200)
    parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-measurements-per-disorder",
        type=int,
        default=320,
    )
    parser.add_argument("--q0-num-start-chains", type=int, default=4)
    parser.add_argument("--num-start-chains", type=int, default=None)
    parser.add_argument("--num-replicas-per-start", type=int, default=None)
    parser.add_argument("--pt-p-hot", type=float, default=None)
    parser.add_argument("--pt-num-temperatures", type=int, default=None)
    parser.add_argument(
        "--pt-swap-attempt-every-num-sweeps",
        type=int,
        default=1,
    )
    parser.add_argument("--seed-base", type=int, default=20260425)
    parser.add_argument(
        "--burn-in-scaling-reference-num-qubits",
        type=int,
        default=18,
    )
    parser.add_argument("--git-commit-sha", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    code_family = str(args.code_family)
    if args.run_root is None:
        run_id = f"measurement_noise_threshold_scout_local_{_timestamp_tag()}"
        run_root = _build_default_run_root(
            code_family=code_family,
            run_id=run_id,
        )
    else:
        run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    lattice_sizes_csv = args.lattice_sizes
    if lattice_sizes_csv is None:
        lattice_sizes_csv = _csv_from_int_list(
            DEFAULT_LATTICE_SIZES_BY_CODE_FAMILY[code_family]
        )
    q_values = [
        float(value.strip())
        for value in args.q_values.split(",")
        if value.strip()
    ]
    num_start_chains = args.num_start_chains
    num_replicas_per_start = args.num_replicas_per_start
    pt_p_hot = args.pt_p_hot
    pt_num_temperatures = args.pt_num_temperatures
    if code_family == "3d_toric":
        if num_start_chains is None:
            num_start_chains = DEFAULT_3D_NUM_START_CHAINS
        if num_replicas_per_start is None:
            num_replicas_per_start = DEFAULT_3D_NUM_REPLICAS_PER_START
        if pt_p_hot is None:
            pt_p_hot = DEFAULT_3D_PT_P_HOT
        if pt_num_temperatures is None:
            pt_num_temperatures = DEFAULT_3D_PT_NUM_TEMPERATURES
    else:
        if num_start_chains is None:
            num_start_chains = 1
        if num_replicas_per_start is None:
            num_replicas_per_start = 1
    data_error_probabilities_csv = args.data_error_probabilities

    scout_index = {
        "run_root": str(run_root),
        "code_family": code_family,
        "lattice_sizes": lattice_sizes_csv,
        "q_values": q_values,
        "data_error_probabilities": data_error_probabilities_csv,
        "common_random_disorder_across_p": (
            DEFAULT_COMMON_RANDOM_DISORDER_ACROSS_P
        ),
        "num_start_chains": int(num_start_chains),
        "num_replicas_per_start": int(num_replicas_per_start),
        "pt_p_hot": pt_p_hot,
        "pt_num_temperatures": pt_num_temperatures,
        "outputs": [],
    }

    for q_index, q_value in enumerate(q_values):
        q_tag = _format_probability_tag(q_value)
        q_run_root = run_root / f"q_{q_tag}"
        output_stem = _build_q_output_stem(
            code_family=code_family,
            q_value=q_value,
        )
        submit_command = [
            sys.executable,
            str(SOURCE_DIR / "production_chunked_scan.py"),
            "submit",
            "--run-root",
            str(q_run_root),
            "--code-family",
            code_family,
            "--workers",
            str(args.workers),
            "--chunk-size",
            str(args.chunk_size),
            "--num-disorder-samples-total",
            str(args.num_disorder_samples_total),
            "--data-error-probabilities",
            data_error_probabilities_csv,
            "--lattice-sizes",
            lattice_sizes_csv,
            "--syndrome-error-probability",
            f"{q_value:0.4f}",
            "--num-burn-in-sweeps",
            str(args.num_burn_in_sweeps),
            "--num-sweeps-between-measurements",
            str(args.num_sweeps_between_measurements),
            "--num-measurements-per-disorder",
            str(args.num_measurements_per_disorder),
            "--q0-num-start-chains",
            str(args.q0_num_start_chains),
            "--num-start-chains",
            str(num_start_chains),
            "--num-replicas-per-start",
            str(num_replicas_per_start),
            "--seed-base",
            str(args.seed_base + q_index * 1000000000),
            "--burn-in-scaling-reference-num-qubits",
            str(args.burn_in_scaling_reference_num_qubits),
            "--output-stem",
            output_stem,
        ]
        if pt_p_hot is not None:
            submit_command.extend(["--pt-p-hot", f"{pt_p_hot:0.4f}"])
        if pt_num_temperatures is not None:
            submit_command.extend([
                "--pt-num-temperatures",
                str(pt_num_temperatures),
            ])
        submit_command.extend([
            "--pt-swap-attempt-every-num-sweeps",
            str(args.pt_swap_attempt_every_num_sweeps),
        ])
        if DEFAULT_COMMON_RANDOM_DISORDER_ACROSS_P:
            submit_command.append("--common-random-disorder-across-p")
        if args.git_commit_sha is not None:
            submit_command.extend(["--git-commit-sha", args.git_commit_sha])
        if args.resume:
            submit_command.append("--resume")
        if args.merge_only:
            submit_command.append("--merge-only")
        _run_command(submit_command, cwd=PROJECT_ROOT)

        merged_npz_path = q_run_root / f"{output_stem}.npz"
        summary = analyze_threshold_crossing(
            input_path=merged_npz_path,
            output_dir=q_run_root,
            output_stem=output_stem,
            summary_path=q_run_root / "threshold_summary.json",
            sem95_plot_path=q_run_root / f"{output_stem}_sem95.png",
            gap_plot_path=q_run_root / f"{output_stem}_gap_crossing.png",
        )
        scout_index["outputs"].append({
            "q_value": q_value,
            "run_root": str(q_run_root),
            "npz_path": str(merged_npz_path),
            "png_path": str(q_run_root / f"{output_stem}.png"),
            "sem95_plot_path": summary["sem95_plot_path"],
            "gap_plot_path": summary["gap_plot_path"],
            "summary_path": summary["summary_path"],
            "convergence_summary_path": str(
                q_run_root / f"{output_stem}_convergence.json"
            ),
            "boundary_saturation_artifact": (
                summary["boundary_saturation_artifact"]
            ),
            "primary_crossing_window_hit": (
                summary["primary_crossing_window_hit"]
            ),
            "interior_crossing_window": (
                summary["interior_crossing_window"]
            ),
            "recommended_server_window": (
                summary["recommended_server_window"]
            ),
        })

    index_path = run_root / "threshold_scout_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(scout_index, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "run_root": str(run_root),
        "index_path": str(index_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
