import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np

from build_toric_code_examples import (
    SUPPORTED_CODE_FAMILIES,
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from main import (
    _run_parallel_tempering_single_chain,
    _run_single_disorder_measurement,
)
from mcmc import draw_disorder_sample
from mcmc_diagnostics import equal_log_odds_ladder
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)
from linear_section import apply_linear_section, build_linear_section


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _estimate_non_pt_proposals(
        syndrome_error_probability,
        num_qubits,
        zero_syndrome_move_data,
        num_burn_in_sweeps,
        num_measurements,
        num_sweeps_between_measurements,
        num_zero_syndrome_sweeps_per_cycle,
        winding_repeat_factor):
    total_cycles = int(num_burn_in_sweeps + num_measurements * num_sweeps_between_measurements)
    contractible_per_zero_sweep = int(
        zero_syndrome_move_data["contractible_moves"].shape[0]
    )
    winding_per_zero_sweep = int(
        winding_repeat_factor
        * zero_syndrome_move_data["winding_moves"].shape[0]
    )
    zero_syndrome_per_cycle = int(
        num_zero_syndrome_sweeps_per_cycle
        * (contractible_per_zero_sweep + winding_per_zero_sweep)
    )
    single_bit_per_cycle = 0
    if syndrome_error_probability > 0.0:
        single_bit_per_cycle = int(num_qubits)
    return int(total_cycles * (single_bit_per_cycle + zero_syndrome_per_cycle))


def run_benchmark(args):
    if args.pt_num_temperatures is not None and args.pt_num_temperatures <= 0:
        args.pt_num_temperatures = None
    if args.syndrome_error_probability == 0.0 and args.pt_num_temperatures is not None:
        raise ValueError("parallel tempering benchmark requires q>0")

    rng = np.random.default_rng(args.seed)
    parity_check_matrix, dual_logical_z_basis = build_toric_code_by_family(
        code_family=args.code_family,
        lattice_size=args.lattice_size,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family=args.code_family,
        lattice_size=args.lattice_size,
    )
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    num_checks, num_qubits = parity_check_matrix.shape
    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=args.syndrome_error_probability,
        data_error_probability=args.data_error_probability,
        rng=rng,
    )
    initial_chain_bits = None
    if args.syndrome_error_probability == 0.0:
        initial_chain_bits = apply_linear_section(
            observed_syndrome_bits,
            linear_section_data,
        )

    started_at = time.perf_counter()
    if args.pt_num_temperatures is not None:
        data_error_probability_ladder = equal_log_odds_ladder(
            p_cold=args.data_error_probability,
            p_hot=args.pt_p_hot,
            num_temperatures=args.pt_num_temperatures,
        )
        result = _run_parallel_tempering_single_chain(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=args.syndrome_error_probability,
            data_error_probability_ladder=data_error_probability_ladder,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=args.num_burn_in_sweeps,
            num_measurements_per_disorder=args.num_measurements,
            num_sweeps_between_measurements=args.num_sweeps_between_measurements,
            rng=rng,
            zero_syndrome_move_data=zero_syndrome_move_data,
            initial_chain_bits=initial_chain_bits,
            num_zero_syndrome_sweeps_per_cycle=(
                args.num_zero_syndrome_sweeps_per_cycle
            ),
            winding_repeat_factor=args.winding_repeat_factor,
            pt_swap_attempt_every_num_sweeps=(
                args.pt_swap_attempt_every_num_sweeps
            ),
        )
        elapsed_seconds = time.perf_counter() - started_at
        pt_swap_acceptance_rates = np.asarray(
            result["pt_swap_acceptance_rates"],
            dtype=np.float64,
        )
        proposals_attempted = None
        benchmark_result = {
            "mode": "parallel_tempering",
            "m_u_values": result["m_u_values"],
            "q_top_value": result["q_top_value"],
            "acceptance_rate": result["acceptance_rate"],
            "single_bit_acceptance_rate": result["single_bit_acceptance_rate"],
            "contractible_acceptance_rate": result["contractible_acceptance_rate"],
            "winding_acceptance_rate": result["winding_acceptance_rate"],
            "pt_ladder": result["pt_ladder"],
            "pt_min_swap_acceptance_rate": (
                float(np.min(pt_swap_acceptance_rates))
                if pt_swap_acceptance_rates.size
                else 0.0
            ),
            "pt_mean_swap_acceptance_rate": (
                float(np.mean(pt_swap_acceptance_rates))
                if pt_swap_acceptance_rates.size
                else 0.0
            ),
        }
    else:
        result = _run_single_disorder_measurement(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=args.syndrome_error_probability,
            data_error_probability=args.data_error_probability,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=args.num_burn_in_sweeps,
            num_measurements_per_disorder=args.num_measurements,
            num_sweeps_between_measurements=args.num_sweeps_between_measurements,
            rng=rng,
            zero_syndrome_move_data=zero_syndrome_move_data,
            initial_chain_bits=None,
            num_zero_syndrome_sweeps_per_cycle=(
                args.num_zero_syndrome_sweeps_per_cycle
            ),
            winding_repeat_factor=args.winding_repeat_factor,
            return_diagnostics=True,
        )
        elapsed_seconds = time.perf_counter() - started_at
        proposals_attempted = _estimate_non_pt_proposals(
            syndrome_error_probability=args.syndrome_error_probability,
            num_qubits=num_qubits,
            zero_syndrome_move_data=zero_syndrome_move_data,
            num_burn_in_sweeps=args.num_burn_in_sweeps,
            num_measurements=args.num_measurements,
            num_sweeps_between_measurements=args.num_sweeps_between_measurements,
            num_zero_syndrome_sweeps_per_cycle=(
                args.num_zero_syndrome_sweeps_per_cycle
            ),
            winding_repeat_factor=args.winding_repeat_factor,
        )
        benchmark_result = {
            "mode": "single_chain",
            "m_u_values": result["m_u_values"],
            "q_top_value": result["q_top_value"],
            "acceptance_rate": result["acceptance_rate"],
            "single_bit_acceptance_rate": result["single_bit_acceptance_rate"],
            "contractible_acceptance_rate": result["contractible_acceptance_rate"],
            "winding_acceptance_rate": result["winding_acceptance_rate"],
        }

    if proposals_attempted is None:
        # PT wrapper currently exposes cold-chain totals. Use deterministic
        # expected proposal counts for all temperatures to avoid adding
        # benchmark-only fields to production results.
        total_cycles = int(
            args.num_burn_in_sweeps
            + args.num_measurements * args.num_sweeps_between_measurements
        )
        zero_syndrome_per_cycle = int(
            args.num_zero_syndrome_sweeps_per_cycle
            * (
                zero_syndrome_move_data["contractible_moves"].shape[0]
                + args.winding_repeat_factor
                * zero_syndrome_move_data["winding_moves"].shape[0]
            )
        )
        proposals_per_temperature_cycle = int(num_qubits + zero_syndrome_per_cycle)
        swap_attempts = 0
        if args.pt_num_temperatures is not None:
            for sweep_counter in range(1, total_cycles + 1):
                if (
                        args.pt_swap_attempt_every_num_sweeps > 0
                        and sweep_counter % args.pt_swap_attempt_every_num_sweeps == 0):
                    parity_index = (sweep_counter - 1) % 2
                    swap_attempts += len(
                        range(parity_index, args.pt_num_temperatures - 1, 2)
                    )
        proposals_attempted = int(
            total_cycles
            * args.pt_num_temperatures
            * proposals_per_temperature_cycle
            + swap_attempts
        )

    benchmark_result.update({
        "code_family": args.code_family,
        "lattice_size": int(args.lattice_size),
        "num_qubits": int(num_qubits),
        "num_checks": int(num_checks),
        "syndrome_error_probability": float(args.syndrome_error_probability),
        "data_error_probability": float(args.data_error_probability),
        "seed": int(args.seed),
        "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
        "num_measurements": int(args.num_measurements),
        "num_sweeps_between_measurements": int(
            args.num_sweeps_between_measurements
        ),
        "num_zero_syndrome_sweeps_per_cycle": int(
            args.num_zero_syndrome_sweeps_per_cycle
        ),
        "winding_repeat_factor": int(args.winding_repeat_factor),
        "pt_p_hot": (
            None if args.pt_p_hot is None else float(args.pt_p_hot)
        ),
        "pt_num_temperatures": (
            None
            if args.pt_num_temperatures is None
            else int(args.pt_num_temperatures)
        ),
        "pt_swap_attempt_every_num_sweeps": int(
            args.pt_swap_attempt_every_num_sweeps
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "proposals_attempted": int(proposals_attempted),
        "proposals_per_second": float(proposals_attempted / elapsed_seconds),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    })

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                benchmark_result,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
    return benchmark_result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Small isolated MCMC performance benchmark."
    )
    parser.add_argument("--code-family", choices=SUPPORTED_CODE_FAMILIES, default="3d_toric")
    parser.add_argument("--lattice-size", type=int, default=3)
    parser.add_argument("--syndrome-error-probability", type=float, default=0.005)
    parser.add_argument("--data-error-probability", type=float, default=0.21)
    parser.add_argument("--seed", type=int, default=2026042501)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=32)
    parser.add_argument("--num-measurements", type=int, default=128)
    parser.add_argument("--num-sweeps-between-measurements", type=int, default=2)
    parser.add_argument("--num-zero-syndrome-sweeps-per-cycle", type=int, default=1)
    parser.add_argument("--winding-repeat-factor", type=int, default=1)
    parser.add_argument("--pt-p-hot", type=float, default=0.44)
    parser.add_argument("--pt-num-temperatures", type=int, default=7)
    parser.add_argument("--pt-swap-attempt-every-num-sweeps", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    return parser


if __name__ == "__main__":
    parsed_args = _build_parser().parse_args()
    print(json.dumps(
        run_benchmark(parsed_args),
        indent=2,
        sort_keys=True,
        default=_json_default,
    ))
