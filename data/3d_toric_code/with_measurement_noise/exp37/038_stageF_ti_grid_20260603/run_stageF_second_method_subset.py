#!/usr/bin/env python3
"""Stage F production-grid second-method subset check.

This is a stochastic version of the Stage E logical-loop bridge for selected
production disorders.  For each nonzero logical sector g it samples sector-0
configurations y from

    U_lambda(y) = U_0(y) + lambda * (U_g(y xor ell_g) - U_0(y))

on a lambda grid, then combines adjacent intervals with BAR.  Since ell_g is a
zero-syndrome logical representative, only the data term changes along the
bridge.  This is independent of the Stage F Kp thermodynamic-integration path
and is not single-step FEP or flip reweighting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is expected in local/remote envs
    njit = None


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_toric_code_examples import (  # noqa: E402
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from exp37_sector_ti import (  # noqa: E402
    _build_logical_projection_masks,
    _build_sector_preserving_proposals,
    _build_sector_representatives,
    _compute_k,
    _q_top_from_weights,
    _weights_from_delta_f,
)


DEFAULT_SUBSET = "3:0.23:1,4:0.21:3,5:0.19:0"
TV_THRESHOLD = 0.03
QTOP_THRESHOLD = 0.02


if njit is not None:

    @njit(cache=True)
    def _numba_bridge_sweep(
            current_y_bits,
            current_syndrome_term_bits,
            current_data_weight,
            current_syndrome_weight,
            current_loop_delta,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            representative_mask,
            kp_value,
            kq_value,
            lambda_value):
        accepted_count = 0
        num_proposals = data_support_lengths.shape[0]
        for _ in range(num_proposals):
            proposal_index = np.random.randint(0, num_proposals)

            support_length = data_support_lengths[proposal_index]
            ones = 0
            overlap_length = 0
            overlap_ones = 0
            for support_position in range(support_length):
                qubit_index = data_supports[proposal_index, support_position]
                if current_y_bits[qubit_index]:
                    ones += 1
                if representative_mask[qubit_index]:
                    overlap_length += 1
                    if current_y_bits[qubit_index]:
                        overlap_ones += 1
            delta_data_weight = support_length - 2 * ones
            delta_loop_delta = -2 * overlap_length + 4 * overlap_ones

            syndrome_support_length = syndrome_support_lengths[proposal_index]
            syndrome_ones = 0
            for support_position in range(syndrome_support_length):
                check_index = syndrome_supports[proposal_index, support_position]
                if current_syndrome_term_bits[check_index]:
                    syndrome_ones += 1
            delta_syndrome_weight = syndrome_support_length - 2 * syndrome_ones

            log_acceptance = (
                -kp_value * delta_data_weight
                - kq_value * delta_syndrome_weight
                - lambda_value * kp_value * delta_loop_delta
            )
            if log_acceptance >= 0.0 or np.random.random() < math.exp(log_acceptance):
                for support_position in range(support_length):
                    qubit_index = data_supports[proposal_index, support_position]
                    current_y_bits[qubit_index] = not current_y_bits[qubit_index]
                for support_position in range(syndrome_support_length):
                    check_index = syndrome_supports[proposal_index, support_position]
                    current_syndrome_term_bits[check_index] = (
                        not current_syndrome_term_bits[check_index]
                    )
                current_data_weight += delta_data_weight
                current_syndrome_weight += delta_syndrome_weight
                current_loop_delta += delta_loop_delta
                accepted_count += 1
        return (
            accepted_count,
            num_proposals,
            current_data_weight,
            current_syndrome_weight,
            current_loop_delta,
        )


    @njit(cache=True)
    def _numba_run_bridge_lambda_chain(
            initial_y_bits,
            initial_syndrome_term_bits,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            representative_mask,
            kp_value,
            kq_value,
            lambda_value,
            num_burn_in_sweeps,
            num_measurements,
            num_sweeps_between_measurements,
            seed):
        np.random.seed(seed)
        current_y_bits = initial_y_bits.copy()
        current_syndrome_term_bits = initial_syndrome_term_bits.copy()
        current_data_weight = 0
        current_syndrome_weight = 0
        representative_weight = 0
        representative_ones = 0
        for index in range(current_y_bits.shape[0]):
            if current_y_bits[index]:
                current_data_weight += 1
            if representative_mask[index]:
                representative_weight += 1
                if current_y_bits[index]:
                    representative_ones += 1
        for index in range(current_syndrome_term_bits.shape[0]):
            if current_syndrome_term_bits[index]:
                current_syndrome_weight += 1
        current_loop_delta = representative_weight - 2 * representative_ones

        accepted_total = 0
        attempted_total = 0
        for _ in range(num_burn_in_sweeps):
            (
                accepted,
                attempted,
                current_data_weight,
                current_syndrome_weight,
                current_loop_delta,
            ) = _numba_bridge_sweep(
                current_y_bits,
                current_syndrome_term_bits,
                current_data_weight,
                current_syndrome_weight,
                current_loop_delta,
                data_supports,
                data_support_lengths,
                syndrome_supports,
                syndrome_support_lengths,
                representative_mask,
                kp_value,
                kq_value,
                lambda_value,
            )
            accepted_total += accepted
            attempted_total += attempted

        loop_delta_samples = np.empty(num_measurements, dtype=np.float64)
        data_weight_samples = np.empty(num_measurements, dtype=np.float64)
        syndrome_weight_samples = np.empty(num_measurements, dtype=np.float64)
        for measurement_index in range(num_measurements):
            for _ in range(num_sweeps_between_measurements):
                (
                    accepted,
                    attempted,
                    current_data_weight,
                    current_syndrome_weight,
                    current_loop_delta,
                ) = _numba_bridge_sweep(
                    current_y_bits,
                    current_syndrome_term_bits,
                    current_data_weight,
                    current_syndrome_weight,
                    current_loop_delta,
                    data_supports,
                    data_support_lengths,
                    syndrome_supports,
                    syndrome_support_lengths,
                    representative_mask,
                    kp_value,
                    kq_value,
                    lambda_value,
                )
                accepted_total += accepted
                attempted_total += attempted
            loop_delta_samples[measurement_index] = current_loop_delta
            data_weight_samples[measurement_index] = current_data_weight
            syndrome_weight_samples[measurement_index] = current_syndrome_weight

        acceptance_rate = 0.0
        if attempted_total > 0:
            acceptance_rate = accepted_total / attempted_total
        return (
            current_y_bits,
            current_syndrome_term_bits,
            loop_delta_samples,
            data_weight_samples,
            syndrome_weight_samples,
            acceptance_rate,
        )

else:
    _numba_run_bridge_lambda_chain = None


def _python_bridge_sweep(
        current_y_bits: np.ndarray,
        current_syndrome_term_bits: np.ndarray,
        current_data_weight: int,
        current_syndrome_weight: int,
        current_loop_delta: int,
        data_supports: np.ndarray,
        data_support_lengths: np.ndarray,
        syndrome_supports: np.ndarray,
        syndrome_support_lengths: np.ndarray,
        representative_mask: np.ndarray,
        kp_value: float,
        kq_value: float,
        lambda_value: float,
        rng: np.random.Generator) -> tuple[int, int, int, int, int]:
    accepted_count = 0
    num_proposals = int(data_support_lengths.shape[0])
    for proposal_index in rng.integers(0, num_proposals, size=num_proposals):
        support_length = int(data_support_lengths[int(proposal_index)])
        support = data_supports[int(proposal_index), :support_length]
        ones = int(np.count_nonzero(current_y_bits[support]))
        delta_data_weight = support_length - 2 * ones

        overlap_mask = representative_mask[support]
        overlap_length = int(np.count_nonzero(overlap_mask))
        overlap_ones = int(np.count_nonzero(current_y_bits[support][overlap_mask]))
        delta_loop_delta = -2 * overlap_length + 4 * overlap_ones

        syndrome_support_length = int(syndrome_support_lengths[int(proposal_index)])
        if syndrome_support_length:
            syndrome_support = syndrome_supports[
                int(proposal_index),
                :syndrome_support_length,
            ]
            syndrome_ones = int(np.count_nonzero(
                current_syndrome_term_bits[syndrome_support]
            ))
            delta_syndrome_weight = syndrome_support_length - 2 * syndrome_ones
        else:
            syndrome_support = None
            delta_syndrome_weight = 0

        log_acceptance = (
            -float(kp_value) * float(delta_data_weight)
            -float(kq_value) * float(delta_syndrome_weight)
            -float(lambda_value) * float(kp_value) * float(delta_loop_delta)
        )
        if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
            current_y_bits[support] ^= True
            if syndrome_support is not None:
                current_syndrome_term_bits[syndrome_support] ^= True
            current_data_weight += int(delta_data_weight)
            current_syndrome_weight += int(delta_syndrome_weight)
            current_loop_delta += int(delta_loop_delta)
            accepted_count += 1
    return (
        int(accepted_count),
        int(num_proposals),
        int(current_data_weight),
        int(current_syndrome_weight),
        int(current_loop_delta),
    )


def _run_bridge_lambda_chain_python(
        initial_y_bits: np.ndarray,
        initial_syndrome_term_bits: np.ndarray,
        data_supports: np.ndarray,
        data_support_lengths: np.ndarray,
        syndrome_supports: np.ndarray,
        syndrome_support_lengths: np.ndarray,
        representative_mask: np.ndarray,
        kp_value: float,
        kq_value: float,
        lambda_value: float,
        num_burn_in_sweeps: int,
        num_measurements: int,
        num_sweeps_between_measurements: int,
        seed: int):
    rng = np.random.default_rng(int(seed))
    current_y_bits = np.asarray(initial_y_bits, dtype=bool).copy()
    current_syndrome_term_bits = np.asarray(initial_syndrome_term_bits, dtype=bool).copy()
    representative_mask = np.asarray(representative_mask, dtype=bool)
    current_data_weight = int(np.count_nonzero(current_y_bits))
    current_syndrome_weight = int(np.count_nonzero(current_syndrome_term_bits))
    current_loop_delta = int(np.count_nonzero(representative_mask)) - 2 * int(
        np.count_nonzero(current_y_bits[representative_mask])
    )

    accepted_total = 0
    attempted_total = 0
    for _ in range(int(num_burn_in_sweeps)):
        accepted, attempted, current_data_weight, current_syndrome_weight, current_loop_delta = (
            _python_bridge_sweep(
                current_y_bits=current_y_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                current_data_weight=current_data_weight,
                current_syndrome_weight=current_syndrome_weight,
                current_loop_delta=current_loop_delta,
                data_supports=data_supports,
                data_support_lengths=data_support_lengths,
                syndrome_supports=syndrome_supports,
                syndrome_support_lengths=syndrome_support_lengths,
                representative_mask=representative_mask,
                kp_value=float(kp_value),
                kq_value=float(kq_value),
                lambda_value=float(lambda_value),
                rng=rng,
            )
        )
        accepted_total += accepted
        attempted_total += attempted

    loop_delta_samples = np.empty(int(num_measurements), dtype=np.float64)
    data_weight_samples = np.empty(int(num_measurements), dtype=np.float64)
    syndrome_weight_samples = np.empty(int(num_measurements), dtype=np.float64)
    for measurement_index in range(int(num_measurements)):
        for _ in range(int(num_sweeps_between_measurements)):
            accepted, attempted, current_data_weight, current_syndrome_weight, current_loop_delta = (
                _python_bridge_sweep(
                    current_y_bits=current_y_bits,
                    current_syndrome_term_bits=current_syndrome_term_bits,
                    current_data_weight=current_data_weight,
                    current_syndrome_weight=current_syndrome_weight,
                    current_loop_delta=current_loop_delta,
                    data_supports=data_supports,
                    data_support_lengths=data_support_lengths,
                    syndrome_supports=syndrome_supports,
                    syndrome_support_lengths=syndrome_support_lengths,
                    representative_mask=representative_mask,
                    kp_value=float(kp_value),
                    kq_value=float(kq_value),
                    lambda_value=float(lambda_value),
                    rng=rng,
                )
            )
            accepted_total += accepted
            attempted_total += attempted
        loop_delta_samples[measurement_index] = float(current_loop_delta)
        data_weight_samples[measurement_index] = float(current_data_weight)
        syndrome_weight_samples[measurement_index] = float(current_syndrome_weight)

    acceptance_rate = 0.0 if attempted_total == 0 else accepted_total / attempted_total
    return (
        current_y_bits,
        current_syndrome_term_bits,
        loop_delta_samples,
        data_weight_samples,
        syndrome_weight_samples,
        float(acceptance_rate),
    )


def _parse_subset(text: str) -> list[tuple[int, float, int]]:
    subset = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError("subset entries must be L:q:disorder_index")
        subset.append((int(fields[0]), float(fields[1]), int(fields[2])))
    if not subset:
        raise ValueError("empty subset")
    return subset


def _format_q(value: float) -> float:
    return float(round(float(value), 6))


def _stable_sigmoid_neg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = np.exp(-values[positive]) / (
        1.0 + np.exp(-values[positive])
    )
    exp_values = np.exp(values[~positive])
    result[~positive] = 1.0 / (1.0 + exp_values)
    return result


def _logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(values))
    return max_value + math.log(float(np.mean(np.exp(values - max_value))))


def _solve_sample_bar_delta_f(
        work_a: np.ndarray,
        work_b: np.ndarray) -> tuple[float, float]:
    work_a = np.asarray(work_a, dtype=np.float64)
    work_b = np.asarray(work_b, dtype=np.float64)

    def residual(delta_f: float) -> float:
        left = float(np.sum(_stable_sigmoid_neg(work_a - delta_f)))
        right = float(np.sum(_stable_sigmoid_neg(-work_b + delta_f)))
        return left - right

    combined = np.concatenate([work_a, work_b])
    low = float(np.min(combined) - 80.0)
    high = float(np.max(combined) + 80.0)
    residual_low = residual(low)
    residual_high = residual(high)
    if residual_low > 0.0 or residual_high < 0.0:
        raise RuntimeError(
            f"sample BAR root not bracketed: low={residual_low}, high={residual_high}"
        )
    for _ in range(200):
        midpoint = 0.5 * (low + high)
        residual_mid = residual(midpoint)
        if abs(residual_mid) < 1.0e-12:
            return midpoint, abs(residual_mid)
        if residual_mid < 0.0:
            low = midpoint
        else:
            high = midpoint
    midpoint = 0.5 * (low + high)
    return midpoint, abs(residual(midpoint))


def _estimate_bridge_delta_f(
        loop_delta_samples: np.ndarray,
        lambda_grid: np.ndarray,
        kp_value: float) -> dict:
    loop_delta_samples = np.asarray(loop_delta_samples, dtype=np.float64)
    bar_steps = []
    forward_steps = []
    reverse_steps = []
    residuals = []
    gaps = []
    for lambda_index in range(len(lambda_grid) - 1):
        delta_lambda = float(lambda_grid[lambda_index + 1] - lambda_grid[lambda_index])
        work_a = delta_lambda * float(kp_value) * loop_delta_samples[lambda_index]
        work_b = delta_lambda * float(kp_value) * loop_delta_samples[lambda_index + 1]
        forward_delta_f = -_logmeanexp(-work_a)
        reverse_delta_f = _logmeanexp(work_b)
        bar_delta_f, residual = _solve_sample_bar_delta_f(work_a, work_b)
        bar_steps.append(bar_delta_f)
        forward_steps.append(forward_delta_f)
        reverse_steps.append(reverse_delta_f)
        residuals.append(residual)
        gaps.append(
            max(
                abs(bar_delta_f - forward_delta_f),
                abs(bar_delta_f - reverse_delta_f),
                abs(forward_delta_f - reverse_delta_f),
            )
        )
    bar_delta_f = float(np.sum(bar_steps))
    forward_delta_f = float(np.sum(forward_steps))
    reverse_delta_f = float(np.sum(reverse_steps))
    return {
        "delta_f": bar_delta_f,
        "forward_delta_f": forward_delta_f,
        "reverse_delta_f": reverse_delta_f,
        "max_adjacent_bidirectional_gap": float(np.max(gaps)),
        "max_adjacent_bar_residual": float(np.max(residuals)),
        "full_path_bidirectional_gap": float(
            max(
                abs(bar_delta_f - forward_delta_f),
                abs(bar_delta_f - reverse_delta_f),
                abs(forward_delta_f - reverse_delta_f),
            )
        ),
        "bar_steps": np.asarray(bar_steps, dtype=np.float64),
        "forward_steps": np.asarray(forward_steps, dtype=np.float64),
        "reverse_steps": np.asarray(reverse_steps, dtype=np.float64),
    }


def _load_ti_records(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int)
        q_values = data["q_values"].astype(float)
        q_top = data["q_top_per_disorder"].astype(float)
        weights = data["weights_per_disorder"].astype(float)
        delta_f = data["delta_f_per_disorder"].astype(float)
        flags = data["flags_per_disorder"].astype("<U128")
    records = {}
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            for disorder_index in range(q_top.shape[2]):
                records[(int(lattice_size), _format_q(q_value), int(disorder_index))] = {
                    "q_top": float(q_top[li, qi, disorder_index]),
                    "weights": weights[li, qi, disorder_index],
                    "delta_f": delta_f[li, qi, disorder_index],
                    "flags": str(flags[li, qi, disorder_index]),
                }
    return records


def _disorder_seed(
        seed_base: int,
        lattice_size: int,
        q_value: float,
        disorder_index: int,
        common_disorder_across_q: bool) -> int:
    seed = int(seed_base) + 1000003 * int(lattice_size) + int(disorder_index)
    if not common_disorder_across_q:
        seed += 1009 * int(round(10000 * float(q_value)))
    return int(seed)


def _run_record(args: argparse.Namespace, selected: tuple[int, float, int], ti_record: dict) -> dict:
    if _numba_run_bridge_lambda_chain is None and not bool(args.allow_python_fallback):
        raise RuntimeError(
            "Stage F second-method subset requires numba; use "
            "--allow-python-fallback only for tiny local smoke tests"
        )

    lattice_size, q_value, disorder_index = selected
    started_at = time.perf_counter()
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        args.code_family,
        lattice_size,
    )
    logical_projection_masks = _build_logical_projection_masks(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        args.code_family,
        lattice_size,
    )
    proposals = _build_sector_preserving_proposals(
        parity_check_matrix=parity_check_matrix,
        logical_projection_masks=logical_projection_masks,
        zero_syndrome_move_data=zero_syndrome_move_data,
    )
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=logical_projection_masks,
        parity_check_matrix=parity_check_matrix,
    )
    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    seed = _disorder_seed(
        seed_base=int(args.seed_base),
        lattice_size=int(lattice_size),
        q_value=float(q_value),
        disorder_index=int(disorder_index),
        common_disorder_across_q=bool(args.common_disorder_across_q),
    )
    rng_disorder = np.random.default_rng(seed)
    eta_bits = rng_disorder.random(num_qubits) < float(args.p)
    measurement_error_bits = rng_disorder.random(num_checks) < float(q_value)
    del eta_bits

    kp_value = _compute_k(float(args.p))
    kq_value = _compute_k(float(q_value))
    lambda_grid = np.linspace(
        0.0,
        1.0,
        int(args.num_lambda_points),
        dtype=np.float64,
    )
    num_sectors = sector_representatives.shape[0]
    bridge_delta_f = np.zeros(num_sectors, dtype=np.float64)
    forward_delta_f = np.zeros(num_sectors, dtype=np.float64)
    reverse_delta_f = np.zeros(num_sectors, dtype=np.float64)
    adjacent_gaps = np.zeros(num_sectors, dtype=np.float64)
    full_path_gaps = np.zeros(num_sectors, dtype=np.float64)
    bar_residuals = np.zeros(num_sectors, dtype=np.float64)
    acceptance_by_sector_lambda = np.full(
        (num_sectors, len(lambda_grid)),
        np.nan,
        dtype=np.float64,
    )

    zero_y_bits = np.zeros(num_qubits, dtype=bool)
    zero_syndrome_term_bits = measurement_error_bits.astype(bool).copy()
    # Validate the cache convention used by the bridge state.
    recomputed_zero = (
        (parity_check_matrix_uint8 @ zero_y_bits.astype(np.uint8)) % 2
    ).astype(bool) ^ measurement_error_bits
    if not np.array_equal(recomputed_zero, zero_syndrome_term_bits):
        raise AssertionError("initial syndrome-term cache mismatch")

    for sector in range(1, num_sectors):
        representative_mask = np.asarray(sector_representatives[sector], dtype=bool)
        current_y_bits = zero_y_bits.copy()
        current_syndrome_term_bits = zero_syndrome_term_bits.copy()
        loop_delta_samples = np.empty(
            (len(lambda_grid), int(args.num_measurements)),
            dtype=np.float64,
        )
        sector_seed_base = (
            int(seed)
            + 1000000007 * int(sector)
            + 9176 * int(round(10000 * float(q_value)))
        )
        for lambda_index, lambda_value in enumerate(lambda_grid):
            chain_seed = int(
                (
                    sector_seed_base
                    + 104729 * int(lambda_index)
                    + 15485863 * int(args.seed_offset)
                )
                % np.iinfo(np.int32).max
            )
            run_bridge_lambda_chain = (
                _numba_run_bridge_lambda_chain
                if _numba_run_bridge_lambda_chain is not None
                else _run_bridge_lambda_chain_python
            )
            (
                current_y_bits,
                current_syndrome_term_bits,
                samples,
                _data_samples,
                _syndrome_samples,
                acceptance_rate,
            ) = run_bridge_lambda_chain(
                current_y_bits,
                current_syndrome_term_bits,
                proposals["data_supports"],
                proposals["data_support_lengths"],
                proposals["syndrome_supports"],
                proposals["syndrome_support_lengths"],
                representative_mask,
                float(kp_value),
                float(kq_value),
                float(lambda_value),
                int(args.num_burn_in_sweeps),
                int(args.num_measurements),
                int(args.num_sweeps_between_measurements),
                chain_seed,
            )
            loop_delta_samples[lambda_index] = samples
            acceptance_by_sector_lambda[sector, lambda_index] = float(acceptance_rate)

        estimate = _estimate_bridge_delta_f(
            loop_delta_samples=loop_delta_samples,
            lambda_grid=lambda_grid,
            kp_value=float(kp_value),
        )
        bridge_delta_f[sector] = float(estimate["delta_f"])
        forward_delta_f[sector] = float(estimate["forward_delta_f"])
        reverse_delta_f[sector] = float(estimate["reverse_delta_f"])
        adjacent_gaps[sector] = float(estimate["max_adjacent_bidirectional_gap"])
        full_path_gaps[sector] = float(estimate["full_path_bidirectional_gap"])
        bar_residuals[sector] = float(estimate["max_adjacent_bar_residual"])

    bridge_weights = _weights_from_delta_f(bridge_delta_f)
    bridge_q_top = _q_top_from_weights(bridge_weights)
    ti_weights = np.asarray(ti_record["weights"], dtype=np.float64)
    ti_q_top = float(ti_record["q_top"])
    tv_vs_ti = float(0.5 * np.sum(np.abs(bridge_weights - ti_weights)))
    q_top_abs_diff_vs_ti = float(abs(bridge_q_top - ti_q_top))
    return {
        "lattice_size": int(lattice_size),
        "p_value": float(args.p),
        "q_value": float(q_value),
        "disorder_index": int(disorder_index),
        "seed": int(seed),
        "method": "stochastic_bidirectional_logical_loop_bridge_bar",
        "projection_mode": "linear",
        "num_qubits": int(num_qubits),
        "num_checks": int(num_checks),
        "kp_value": float(kp_value),
        "kq_value": float(kq_value),
        "num_lambda_points": int(args.num_lambda_points),
        "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
        "num_measurements": int(args.num_measurements),
        "num_sweeps_between_measurements": int(args.num_sweeps_between_measurements),
        "lambda_grid": lambda_grid.tolist(),
        "ti_q_top": float(ti_q_top),
        "bridge_q_top": float(bridge_q_top),
        "q_top_abs_diff_vs_ti": float(q_top_abs_diff_vs_ti),
        "tv_vs_ti": float(tv_vs_ti),
        "ti_weights": ti_weights.tolist(),
        "bridge_weights": bridge_weights.tolist(),
        "ti_delta_f": np.asarray(ti_record["delta_f"], dtype=np.float64).tolist(),
        "bridge_delta_f": bridge_delta_f.tolist(),
        "forward_delta_f": forward_delta_f.tolist(),
        "reverse_delta_f": reverse_delta_f.tolist(),
        "max_adjacent_bidirectional_gap": float(np.nanmax(adjacent_gaps)),
        "max_full_path_bidirectional_gap": float(np.nanmax(full_path_gaps)),
        "max_bar_residual": float(np.nanmax(bar_residuals)),
        "mean_acceptance_rate": float(np.nanmean(acceptance_by_sector_lambda[1:])),
        "min_acceptance_rate": float(np.nanmin(acceptance_by_sector_lambda[1:])),
        "ti_flags": str(ti_record["flags"]),
        "passed": bool(tv_vs_ti <= TV_THRESHOLD and q_top_abs_diff_vs_ti <= QTOP_THRESHOLD),
        "wall_time_seconds": float(time.perf_counter() - started_at),
    }


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "lattice_size",
        "p_value",
        "q_value",
        "disorder_index",
        "ti_q_top",
        "bridge_q_top",
        "tv_vs_ti",
        "q_top_abs_diff_vs_ti",
        "max_adjacent_bidirectional_gap",
        "max_full_path_bidirectional_gap",
        "max_bar_residual",
        "mean_acceptance_rate",
        "min_acceptance_rate",
        "passed",
        "wall_time_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def _write_summary(path: Path, payload: dict) -> None:
    lines = [
        "# Stage F second-method subset summary",
        "",
        f"Overall subset agreement: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        (
            "Estimator: stochastic bidirectional logical-loop bridge with BAR "
            "on adjacent lambda intervals.  It uses the same production "
            "disorder seeds and sector representatives as the Stage F TI grid, "
            "but not the Kp thermodynamic-integration estimator."
        ),
        "",
        f"Thresholds: TV <= {TV_THRESHOLD:.3f}, |dq_top| <= {QTOP_THRESHOLD:.3f}.",
        "",
        "| L | q | d | TI q_top | bridge q_top | TV | dq_top | bidir gap | passed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in payload["records"]:
        lines.append(
            f"| {record['lattice_size']} | {record['q_value']:.3f} | "
            f"{record['disorder_index']} | {record['ti_q_top']:.6f} | "
            f"{record['bridge_q_top']:.6f} | {record['tv_vs_ti']:.5f} | "
            f"{record['q_top_abs_diff_vs_ti']:.5f} | "
            f"{record['max_full_path_bidirectional_gap']:.5f} | "
            f"{'PASS' if record['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `stageF_second_method_subset.json`",
        "- `stageF_second_method_subset.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--code-family", default="3d_toric")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--seed-base", type=int, default=637000)
    parser.add_argument("--common-disorder-across-q", action="store_true")
    parser.add_argument("--num-lambda-points", type=int, default=33)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=192)
    parser.add_argument("--num-measurements", type=int, default=4096)
    parser.add_argument("--num-sweeps-between-measurements", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--allow-python-fallback",
        action="store_true",
        help="Allow the pure-Python kernel; intended only for tiny local smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_lambda_points) < 3:
        raise ValueError("--num-lambda-points must be at least 3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = _parse_subset(args.subset)
    ti_records = _load_ti_records(args.ti_results)
    records = []
    started_at = time.perf_counter()
    for index, selected in enumerate(selected_records):
        key = (int(selected[0]), _format_q(selected[1]), int(selected[2]))
        if key not in ti_records:
            raise KeyError(f"selected record is absent from TI grid: {key}")
        record = _run_record(args, selected, ti_records[key])
        records.append(record)
        print(
            f"[{index + 1}/{len(selected_records)}] "
            f"L={record['lattice_size']} q={record['q_value']:.3f} "
            f"d={record['disorder_index']} "
            f"bridge_q_top={record['bridge_q_top']:.6f} "
            f"TI_q_top={record['ti_q_top']:.6f} "
            f"TV={record['tv_vs_ti']:.5f} "
            f"dq={record['q_top_abs_diff_vs_ti']:.5f} "
            f"passed={record['passed']} "
            f"wall={record['wall_time_seconds']:.1f}s",
            flush=True,
        )

    overall_passed = bool(all(record["passed"] for record in records))
    payload = {
        "stage": "F",
        "method": "stochastic_bidirectional_logical_loop_bridge_bar",
        "overall_passed": overall_passed,
        "ti_results_path": str(args.ti_results),
        "subset": [
            {
                "lattice_size": int(lattice_size),
                "q_value": float(q_value),
                "disorder_index": int(disorder_index),
            }
            for lattice_size, q_value, disorder_index in selected_records
        ],
        "thresholds": {
            "tv_vs_ti": TV_THRESHOLD,
            "q_top_abs_diff_vs_ti": QTOP_THRESHOLD,
        },
        "config": {
            "code_family": str(args.code_family),
            "p_value": float(args.p),
            "seed_base": int(args.seed_base),
            "common_disorder_across_q": bool(args.common_disorder_across_q),
            "num_lambda_points": int(args.num_lambda_points),
            "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
            "num_measurements": int(args.num_measurements),
            "num_sweeps_between_measurements": int(args.num_sweeps_between_measurements),
            "seed_offset": int(args.seed_offset),
        },
        "numba_available": bool(_numba_run_bridge_lambda_chain is not None),
        "used_python_fallback": bool(_numba_run_bridge_lambda_chain is None),
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "records": records,
    }
    json_path = args.output_dir / "stageF_second_method_subset.json"
    csv_path = args.output_dir / "stageF_second_method_subset.csv"
    summary_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, records)
    _write_summary(summary_path, payload)
    print(json.dumps({
        "overall_passed": overall_passed,
        "num_records": len(records),
        "max_tv_vs_ti": float(max(record["tv_vs_ti"] for record in records)),
        "max_q_top_abs_diff_vs_ti": float(
            max(record["q_top_abs_diff_vs_ti"] for record in records)
        ),
        "wall_time_seconds": payload["wall_time_seconds"],
        "json_path": str(json_path),
    }, indent=2, sort_keys=True))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
