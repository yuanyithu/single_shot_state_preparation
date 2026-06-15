#!/usr/bin/env python3
"""Stage E bidirectional logical-loop bridge cross-check.

This is the exp37 second free-energy path.  For each nontrivial logical
sector g it maps sector-0 configurations y to y xor ell_g and estimates
F_g - F_0 along a lambda bridge

    u_lambda(y) = u_0(y) + lambda * (u_g(y xor ell_g) - u_0(y)).

The accepted Stage E gate uses exact L=2 bridge count tables, then applies
multi-step forward/reverse exponential estimates and BAR on each adjacent
lambda interval.  It is deliberately not the Stage D Kp thermodynamic
integration path and not a single-step FEP / flip-reweight estimator.
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


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_toric_code_examples import (  # noqa: E402
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from exact_enumeration import _iter_chain_bit_chunks, _logsumexp  # noqa: E402
from exp37_sector_ti import (  # noqa: E402
    _build_logical_projection_masks,
    _build_sector_representatives,
    _compute_k,
    _compute_signature,
    _q_top_from_weights,
    _weights_from_delta_f,
)


OUTPUT_DIR = SCRIPT_PATH.parent
STAGE_B_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "exp37"
    / "034_stageB_exact_reference_20260603"
    / "exact_reference.json"
)
STAGE_D_ACCEPTED = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "exp37"
    / "036_stageD_sector_ti_20260603"
    / "accepted_combined"
    / "stageD_results.json"
)

LATTICE_SIZE = 2
NUM_SECTORS = 8
CHUNK_SIZE = 1 << 18
NUM_LAMBDA_POINTS = 17

E1_TV_THRESHOLD = 0.03
E1_QTOP_THRESHOLD = 0.02
E2_TV_THRESHOLD = 0.03
E2_QTOP_THRESHOLD = 0.02
E3_BIDIRECTIONAL_GAP_THRESHOLD = 1.0e-8
E3_BAR_RESIDUAL_THRESHOLD = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage E bidirectional logical-loop bridge benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for Stage E artifacts.",
    )
    parser.add_argument(
        "--record-ids",
        default="",
        help="Comma-separated Stage B record IDs; default runs all.",
    )
    parser.add_argument(
        "--num-lambda-points",
        type=int,
        default=NUM_LAMBDA_POINTS,
        help="Number of lambda bridge points from 0 to 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Enumeration chunk size.",
    )
    return parser.parse_args()


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def stable_sigmoid_neg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = np.exp(-values[positive]) / (
        1.0 + np.exp(-values[positive])
    )
    exp_values = np.exp(values[~positive])
    result[~positive] = 1.0 / (1.0 + exp_values)
    return result


def load_reference_records(selected_record_ids: set[int] | None) -> list[dict]:
    payload = json.loads(STAGE_B_REFERENCE.read_text(encoding="utf-8"))
    if payload.get("lattice_size") != LATTICE_SIZE:
        raise ValueError("Stage B reference lattice size mismatch")
    records = payload["records"]
    if selected_record_ids is not None:
        records = [
            record for record in records
            if int(record["record_id"]) in selected_record_ids
        ]
        if not records:
            raise ValueError("selected --record-ids did not match any records")
    return records


def load_stage_d_records() -> dict[int, dict]:
    payload = json.loads(STAGE_D_ACCEPTED.read_text(encoding="utf-8"))
    if not payload.get("overall_passed"):
        raise ValueError("Stage D accepted result is not PASS")
    return {
        int(record["record_id"]): record
        for record in payload["records"]
    }


def build_bridge_counts(
    parity_check_matrix: np.ndarray,
    logical_projection_masks: np.ndarray,
    sector_representatives: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    """Count sector-0 configurations by |y|, |Hy|, and |y xor ell_g|-|y|."""
    num_checks, num_qubits = parity_check_matrix.shape
    delta_offset = num_qubits
    counts = np.zeros(
        (NUM_SECTORS, num_qubits + 1, num_checks + 1, 2 * num_qubits + 1),
        dtype=np.int64,
    )
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    projection_uint8 = logical_projection_masks.astype(np.uint8)
    bit_weights = 1 << np.arange(logical_projection_masks.shape[0], dtype=np.int64)

    total_seen = 0
    sector0_seen = 0
    rep_supports = [
        np.flatnonzero(sector_representatives[sector]).astype(np.int64)
        for sector in range(NUM_SECTORS)
    ]
    encoded_size = (num_qubits + 1) * (num_checks + 1) * (2 * num_qubits + 1)

    for x_chunk in _iter_chain_bit_chunks(num_qubits, int(chunk_size)):
        x_uint8 = x_chunk.astype(np.uint8)
        signatures = (
            (x_uint8 @ projection_uint8.T) % 2
        ).astype(np.int64) @ bit_weights
        sector0_mask = signatures == 0
        if not np.any(sector0_mask):
            total_seen += int(x_chunk.shape[0])
            continue

        y_chunk = x_chunk[sector0_mask]
        y_uint8 = x_uint8[sector0_mask]
        data_weights = np.count_nonzero(y_chunk, axis=1).astype(np.int64)
        syndrome_bits = (y_uint8 @ parity_check_matrix_uint8.T) % 2
        syndrome_weights = np.count_nonzero(syndrome_bits, axis=1).astype(np.int64)
        sector0_seen += int(y_chunk.shape[0])

        for sector in range(NUM_SECTORS):
            support = rep_supports[sector]
            if support.size == 0:
                delta_data_weights = np.zeros_like(data_weights)
            else:
                ones_on_support = np.count_nonzero(y_chunk[:, support], axis=1)
                delta_data_weights = (
                    int(support.size) - 2 * ones_on_support.astype(np.int64)
                )
            delta_indices = delta_data_weights + delta_offset
            encoded = (
                data_weights * (num_checks + 1) * (2 * num_qubits + 1)
                + syndrome_weights * (2 * num_qubits + 1)
                + delta_indices
            )
            counts[sector] += np.bincount(
                encoded,
                minlength=encoded_size,
            ).reshape(counts.shape[1:])
        total_seen += int(x_chunk.shape[0])

    metadata = {
        "total_configurations": int(total_seen),
        "expected_configurations": int(1 << num_qubits),
        "sector0_configurations": int(sector0_seen),
        "expected_sector0_configurations": int((1 << num_qubits) // NUM_SECTORS),
        "delta_offset": int(delta_offset),
    }
    if total_seen != 1 << num_qubits:
        raise AssertionError("enumeration count mismatch")
    if sector0_seen != (1 << num_qubits) // NUM_SECTORS:
        raise AssertionError("sector-0 count mismatch")
    return counts, metadata


def logsumexp_weighted(
    log_counts: np.ndarray,
    positive_mask: np.ndarray,
    values: np.ndarray,
) -> float:
    terms = log_counts[positive_mask] + values[positive_mask]
    return float(_logsumexp(terms.reshape(-1)))


def solve_bar_delta_f(
    log_prob_a: np.ndarray,
    log_prob_b: np.ndarray,
    positive_mask: np.ndarray,
    work_forward: np.ndarray,
) -> tuple[float, float]:
    log_prob_a_flat = log_prob_a[positive_mask]
    log_prob_b_flat = log_prob_b[positive_mask]
    work_flat = work_forward[positive_mask]
    prob_a = np.exp(log_prob_a_flat)
    prob_b = np.exp(log_prob_b_flat)

    def residual(delta_f: float) -> float:
        left = float(np.sum(prob_a * stable_sigmoid_neg(work_flat - delta_f)))
        right = float(np.sum(prob_b * stable_sigmoid_neg(-work_flat + delta_f)))
        return left - right

    low = float(np.min(work_flat) - 80.0)
    high = float(np.max(work_flat) + 80.0)
    residual_low = residual(low)
    residual_high = residual(high)
    if residual_low > 0.0 or residual_high < 0.0:
        raise RuntimeError(
            f"BAR root not bracketed: low={residual_low}, high={residual_high}"
        )
    for _ in range(200):
        midpoint = 0.5 * (low + high)
        residual_mid = residual(midpoint)
        if abs(residual_mid) < 1.0e-14:
            return midpoint, abs(residual_mid)
        if residual_mid < 0.0:
            low = midpoint
        else:
            high = midpoint
    midpoint = 0.5 * (low + high)
    return midpoint, abs(residual(midpoint))


def bridge_for_sector(
    counts: np.ndarray,
    kp_value: float,
    kq_value: float,
    lambda_grid: np.ndarray,
    delta_offset: int,
) -> dict:
    num_data_weights, num_syndrome_weights, num_delta_values = counts.shape
    data_grid = np.arange(num_data_weights, dtype=np.float64)[:, None, None]
    syndrome_grid = np.arange(num_syndrome_weights, dtype=np.float64)[None, :, None]
    delta_grid = (
        np.arange(num_delta_values, dtype=np.float64)[None, None, :]
        - float(delta_offset)
    )
    positive = counts > 0
    log_counts = np.full(counts.shape, -np.inf, dtype=np.float64)
    log_counts[positive] = np.log(counts[positive].astype(np.float64))
    base_log_values = (
        -kp_value * data_grid
        - kq_value * syndrome_grid
    )
    delta_energy = kp_value * delta_grid

    log_z = np.empty(lambda_grid.shape[0], dtype=np.float64)
    for lambda_index, lambda_value in enumerate(lambda_grid):
        log_z[lambda_index] = logsumexp_weighted(
            log_counts,
            positive,
            base_log_values - float(lambda_value) * delta_energy,
        )

    bar_delta_f_steps = []
    forward_delta_f_steps = []
    reverse_delta_f_steps = []
    bar_residuals = []
    bidirectional_gaps = []
    for lambda_index in range(lambda_grid.shape[0] - 1):
        lambda_a = float(lambda_grid[lambda_index])
        lambda_b = float(lambda_grid[lambda_index + 1])
        work_forward = np.broadcast_to(
            (lambda_b - lambda_a) * delta_energy,
            counts.shape,
        )
        log_prob_a = (
            log_counts
            + base_log_values
            - lambda_a * delta_energy
            - log_z[lambda_index]
        )
        log_prob_b = (
            log_counts
            + base_log_values
            - lambda_b * delta_energy
            - log_z[lambda_index + 1]
        )
        forward_delta_f = -float(
            _logsumexp((log_prob_a[positive] - work_forward[positive]).reshape(-1))
        )
        reverse_delta_f = float(
            _logsumexp((log_prob_b[positive] + work_forward[positive]).reshape(-1))
        )
        bar_delta_f, bar_residual = solve_bar_delta_f(
            log_prob_a=log_prob_a,
            log_prob_b=log_prob_b,
            positive_mask=positive,
            work_forward=work_forward,
        )
        forward_delta_f_steps.append(forward_delta_f)
        reverse_delta_f_steps.append(reverse_delta_f)
        bar_delta_f_steps.append(bar_delta_f)
        bar_residuals.append(bar_residual)
        bidirectional_gaps.append(
            max(
                abs(bar_delta_f - forward_delta_f),
                abs(bar_delta_f - reverse_delta_f),
                abs(forward_delta_f - reverse_delta_f),
            )
        )

    forward_delta_f = float(np.sum(forward_delta_f_steps))
    reverse_delta_f = float(np.sum(reverse_delta_f_steps))
    bar_delta_f = float(np.sum(bar_delta_f_steps))
    true_delta_f = -float(log_z[-1] - log_z[0])
    return {
        "delta_f": bar_delta_f,
        "forward_delta_f": forward_delta_f,
        "reverse_delta_f": reverse_delta_f,
        "true_path_delta_f": true_delta_f,
        "max_adjacent_bidirectional_gap": float(np.max(bidirectional_gaps)),
        "max_adjacent_bar_residual": float(np.max(bar_residuals)),
        "full_path_bidirectional_gap": float(
            max(
                abs(bar_delta_f - forward_delta_f),
                abs(bar_delta_f - reverse_delta_f),
                abs(forward_delta_f - reverse_delta_f),
                abs(bar_delta_f - true_delta_f),
            )
        ),
    }


def evaluate_record(
    reference: dict,
    stage_d_record: dict,
    bridge_counts: np.ndarray,
    lambda_grid: np.ndarray,
    delta_offset: int,
) -> dict:
    kp_value = _compute_k(float(reference["p_value"]))
    kq_value = _compute_k(float(reference["q_value"]))

    delta_f = np.zeros(NUM_SECTORS, dtype=np.float64)
    forward_delta_f = np.zeros(NUM_SECTORS, dtype=np.float64)
    reverse_delta_f = np.zeros(NUM_SECTORS, dtype=np.float64)
    true_path_delta_f = np.zeros(NUM_SECTORS, dtype=np.float64)
    adjacent_gaps = np.zeros(NUM_SECTORS, dtype=np.float64)
    bar_residuals = np.zeros(NUM_SECTORS, dtype=np.float64)
    full_path_gaps = np.zeros(NUM_SECTORS, dtype=np.float64)

    for sector in range(1, NUM_SECTORS):
        sector_result = bridge_for_sector(
            counts=bridge_counts[sector],
            kp_value=kp_value,
            kq_value=kq_value,
            lambda_grid=lambda_grid,
            delta_offset=delta_offset,
        )
        delta_f[sector] = sector_result["delta_f"]
        forward_delta_f[sector] = sector_result["forward_delta_f"]
        reverse_delta_f[sector] = sector_result["reverse_delta_f"]
        true_path_delta_f[sector] = sector_result["true_path_delta_f"]
        adjacent_gaps[sector] = sector_result["max_adjacent_bidirectional_gap"]
        bar_residuals[sector] = sector_result["max_adjacent_bar_residual"]
        full_path_gaps[sector] = sector_result["full_path_bidirectional_gap"]

    weights = _weights_from_delta_f(delta_f)
    q_top = _q_top_from_weights(weights)
    exact_weights = np.asarray(reference["weights"], dtype=np.float64)
    ti_weights = np.asarray(stage_d_record["weights"], dtype=np.float64)
    exact_q_top = float(reference["q_top"])
    ti_q_top = float(stage_d_record["q_top"])
    exact_tv = total_variation(weights, exact_weights)
    ti_tv = total_variation(weights, ti_weights)
    exact_q_top_abs_diff = abs(float(q_top - exact_q_top))
    ti_q_top_abs_diff = abs(float(q_top - ti_q_top))
    max_bidirectional_gap = float(np.max(np.maximum(adjacent_gaps, full_path_gaps)))
    max_bar_residual = float(np.max(bar_residuals))
    return {
        "record_id": int(reference["record_id"]),
        "p_value": float(reference["p_value"]),
        "q_value": float(reference["q_value"]),
        "kp_value": float(kp_value),
        "kq_value": float(kq_value),
        "exact_weights": exact_weights,
        "stage_d_ti_weights": ti_weights,
        "bridge_weights": weights,
        "exact_q_top": exact_q_top,
        "stage_d_ti_q_top": ti_q_top,
        "bridge_q_top": float(q_top),
        "delta_f": delta_f,
        "forward_delta_f": forward_delta_f,
        "reverse_delta_f": reverse_delta_f,
        "true_path_delta_f": true_path_delta_f,
        "exact_delta_f": np.asarray(reference["delta_f"], dtype=np.float64),
        "stage_d_ti_delta_f": np.asarray(stage_d_record["delta_f"], dtype=np.float64),
        "tv_vs_exact": float(exact_tv),
        "q_top_abs_diff_vs_exact": float(exact_q_top_abs_diff),
        "tv_vs_stage_d_ti": float(ti_tv),
        "q_top_abs_diff_vs_stage_d_ti": float(ti_q_top_abs_diff),
        "max_bidirectional_gap": max_bidirectional_gap,
        "max_bar_residual": max_bar_residual,
        "e1_passed": bool(
            exact_tv <= E1_TV_THRESHOLD
            and exact_q_top_abs_diff <= E1_QTOP_THRESHOLD
        ),
        "e2_passed": bool(
            ti_tv <= E2_TV_THRESHOLD
            and ti_q_top_abs_diff <= E2_QTOP_THRESHOLD
        ),
        "e3_passed": bool(
            max_bidirectional_gap <= E3_BIDIRECTIONAL_GAP_THRESHOLD
            and max_bar_residual <= E3_BAR_RESIDUAL_THRESHOLD
        ),
    }


def json_ready(record: dict) -> dict:
    array_keys = {
        "exact_weights",
        "stage_d_ti_weights",
        "bridge_weights",
        "delta_f",
        "forward_delta_f",
        "reverse_delta_f",
        "true_path_delta_f",
        "exact_delta_f",
        "stage_d_ti_delta_f",
    }
    result = {}
    for key, value in record.items():
        if key in array_keys:
            result[key] = np.asarray(value, dtype=np.float64).tolist()
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def write_csv(records: list[dict]) -> None:
    with (OUTPUT_DIR / "stageE_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_id",
                "p_value",
                "q_value",
                "exact_q_top",
                "bridge_q_top",
                "stage_d_ti_q_top",
                "tv_vs_exact",
                "q_top_abs_diff_vs_exact",
                "tv_vs_stage_d_ti",
                "q_top_abs_diff_vs_stage_d_ti",
                "max_bidirectional_gap",
                "max_bar_residual",
                "e1_passed",
                "e2_passed",
                "e3_passed",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record["record_id"],
                "p_value": record["p_value"],
                "q_value": record["q_value"],
                "exact_q_top": record["exact_q_top"],
                "bridge_q_top": record["bridge_q_top"],
                "stage_d_ti_q_top": record["stage_d_ti_q_top"],
                "tv_vs_exact": record["tv_vs_exact"],
                "q_top_abs_diff_vs_exact": record["q_top_abs_diff_vs_exact"],
                "tv_vs_stage_d_ti": record["tv_vs_stage_d_ti"],
                "q_top_abs_diff_vs_stage_d_ti": (
                    record["q_top_abs_diff_vs_stage_d_ti"]
                ),
                "max_bidirectional_gap": record["max_bidirectional_gap"],
                "max_bar_residual": record["max_bar_residual"],
                "e1_passed": record["e1_passed"],
                "e2_passed": record["e2_passed"],
                "e3_passed": record["e3_passed"],
            })


def write_npz(records: list[dict]) -> None:
    np.savez_compressed(
        OUTPUT_DIR / "stageE_results.npz",
        record_id=np.asarray([record["record_id"] for record in records], dtype=np.int64),
        p_value=np.asarray([record["p_value"] for record in records], dtype=np.float64),
        q_value=np.asarray([record["q_value"] for record in records], dtype=np.float64),
        exact_weights=np.stack([record["exact_weights"] for record in records]),
        bridge_weights=np.stack([record["bridge_weights"] for record in records]),
        stage_d_ti_weights=np.stack([record["stage_d_ti_weights"] for record in records]),
        exact_q_top=np.asarray([record["exact_q_top"] for record in records], dtype=np.float64),
        bridge_q_top=np.asarray([record["bridge_q_top"] for record in records], dtype=np.float64),
        stage_d_ti_q_top=np.asarray(
            [record["stage_d_ti_q_top"] for record in records],
            dtype=np.float64,
        ),
        tv_vs_exact=np.asarray([record["tv_vs_exact"] for record in records], dtype=np.float64),
        q_top_abs_diff_vs_exact=np.asarray(
            [record["q_top_abs_diff_vs_exact"] for record in records],
            dtype=np.float64,
        ),
        tv_vs_stage_d_ti=np.asarray(
            [record["tv_vs_stage_d_ti"] for record in records],
            dtype=np.float64,
        ),
        q_top_abs_diff_vs_stage_d_ti=np.asarray(
            [record["q_top_abs_diff_vs_stage_d_ti"] for record in records],
            dtype=np.float64,
        ),
        max_bidirectional_gap=np.asarray(
            [record["max_bidirectional_gap"] for record in records],
            dtype=np.float64,
        ),
        max_bar_residual=np.asarray(
            [record["max_bar_residual"] for record in records],
            dtype=np.float64,
        ),
    )


def write_summary(payload: dict) -> None:
    gates = payload["gates"]
    records = payload["records"]
    lines = [
        "# Stage E bidirectional logical-loop bridge summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        (
            "Estimator: independent multi-step logical-loop bridge.  For each "
            "sector g, sector-0 configurations are annealed to `y xor ell_g` "
            "on a lambda grid and adjacent intervals are combined with BAR."
        ),
        "Reference: Stage B exact L=2 zero-disorder benchmark and Stage D accepted TI.",
        "No single-step FEP, flip-reweighting, or Kp thermodynamic integration is used.",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| E1 | TV vs exact <= {E1_TV_THRESHOLD:.3f}, "
            f"abs dq_top <= {E1_QTOP_THRESHOLD:.3f} | "
            f"max TV={gates['E1']['max_tv_vs_exact']:.4g}, "
            f"max abs dq={gates['E1']['max_q_top_abs_diff_vs_exact']:.4g} | "
            f"{'PASS' if gates['E1']['passed'] else 'FAIL'} |"
        ),
        (
            f"| E2 | TV vs TI <= {E2_TV_THRESHOLD:.3f}, "
            f"abs dq_top <= {E2_QTOP_THRESHOLD:.3f} | "
            f"max TV={gates['E2']['max_tv_vs_stage_d_ti']:.4g}, "
            f"max abs dq={gates['E2']['max_q_top_abs_diff_vs_stage_d_ti']:.4g} | "
            f"{'PASS' if gates['E2']['passed'] else 'FAIL'} |"
        ),
        (
            f"| E3 | bidirectional gap <= {E3_BIDIRECTIONAL_GAP_THRESHOLD:.1e}, "
            f"BAR residual <= {E3_BAR_RESIDUAL_THRESHOLD:.1e} | "
            f"max gap={gates['E3']['max_bidirectional_gap']:.4g}, "
            f"max residual={gates['E3']['max_bar_residual']:.4g} | "
            f"{'PASS' if gates['E3']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Point Comparison",
        "",
        "| id | p | q | exact q_top | bridge q_top | TI q_top | TV exact | dq exact | TV TI | dq TI | bidir gap | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        gates_text = "/".join([
            "E1" if record["e1_passed"] else "e1FAIL",
            "E2" if record["e2_passed"] else "e2FAIL",
            "E3" if record["e3_passed"] else "e3FAIL",
        ])
        lines.append(
            f"| {record['record_id']} | {record['p_value']:.6f} | "
            f"{record['q_value']:.6f} | {record['exact_q_top']:.6f} | "
            f"{record['bridge_q_top']:.6f} | {record['stage_d_ti_q_top']:.6f} | "
            f"{record['tv_vs_exact']:.5g} | "
            f"{record['q_top_abs_diff_vs_exact']:.5g} | "
            f"{record['tv_vs_stage_d_ti']:.5f} | "
            f"{record['q_top_abs_diff_vs_stage_d_ti']:.5f} | "
            f"{record['max_bidirectional_gap']:.3g} | {gates_text} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `stageE_results.json`",
        "- `stageE_results.npz`",
        "- `stageE_comparison.csv`",
    ])
    (OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if int(args.num_lambda_points) < 3:
        raise ValueError("--num-lambda-points must be at least 3")

    selected_record_ids = None
    if str(args.record_ids).strip():
        selected_record_ids = {
            int(part)
            for part in str(args.record_ids).split(",")
            if part.strip()
        }

    started_at = time.perf_counter()
    reference_records = load_reference_records(selected_record_ids)
    stage_d_records = load_stage_d_records()
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    logical_projection_masks = _build_logical_projection_masks(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=logical_projection_masks,
        parity_check_matrix=parity_check_matrix,
    )
    for sector in range(NUM_SECTORS):
        signature = _compute_signature(
            sector_representatives[sector],
            logical_projection_masks,
        )
        if signature != sector:
            raise AssertionError("logical representative signature mismatch")

    bridge_counts, bridge_metadata = build_bridge_counts(
        parity_check_matrix=parity_check_matrix,
        logical_projection_masks=logical_projection_masks,
        sector_representatives=sector_representatives,
        chunk_size=int(args.chunk_size),
    )
    lambda_grid = np.linspace(
        0.0,
        1.0,
        int(args.num_lambda_points),
        dtype=np.float64,
    )

    records = [
        evaluate_record(
            reference=reference,
            stage_d_record=stage_d_records[int(reference["record_id"])],
            bridge_counts=bridge_counts,
            lambda_grid=lambda_grid,
            delta_offset=int(bridge_metadata["delta_offset"]),
        )
        for reference in reference_records
    ]

    e1_passed = all(record["e1_passed"] for record in records)
    e2_passed = all(record["e2_passed"] for record in records)
    e3_passed = all(record["e3_passed"] for record in records)
    gates = {
        "E1": {
            "passed": bool(e1_passed),
            "max_tv_vs_exact": float(max(record["tv_vs_exact"] for record in records)),
            "max_q_top_abs_diff_vs_exact": float(
                max(record["q_top_abs_diff_vs_exact"] for record in records)
            ),
        },
        "E2": {
            "passed": bool(e2_passed),
            "max_tv_vs_stage_d_ti": float(
                max(record["tv_vs_stage_d_ti"] for record in records)
            ),
            "max_q_top_abs_diff_vs_stage_d_ti": float(
                max(record["q_top_abs_diff_vs_stage_d_ti"] for record in records)
            ),
        },
        "E3": {
            "passed": bool(e3_passed),
            "max_bidirectional_gap": float(
                max(record["max_bidirectional_gap"] for record in records)
            ),
            "max_bar_residual": float(
                max(record["max_bar_residual"] for record in records)
            ),
        },
    }
    overall_passed = bool(e1_passed and e2_passed and e3_passed)
    payload = {
        "stage": "E",
        "overall_passed": overall_passed,
        "method": "bidirectional_logical_loop_bridge_bar",
        "code_family": "3d_toric",
        "lattice_size": LATTICE_SIZE,
        "num_lambda_points": int(args.num_lambda_points),
        "lambda_grid": lambda_grid.tolist(),
        "reference_path": str(STAGE_B_REFERENCE.relative_to(PROJECT_ROOT)),
        "stage_d_reference_path": str(STAGE_D_ACCEPTED.relative_to(PROJECT_ROOT)),
        "bridge_count_metadata": bridge_metadata,
        "thresholds": {
            "e1_tv_vs_exact": E1_TV_THRESHOLD,
            "e1_q_top_abs_diff_vs_exact": E1_QTOP_THRESHOLD,
            "e2_tv_vs_stage_d_ti": E2_TV_THRESHOLD,
            "e2_q_top_abs_diff_vs_stage_d_ti": E2_QTOP_THRESHOLD,
            "e3_bidirectional_gap": E3_BIDIRECTIONAL_GAP_THRESHOLD,
            "e3_bar_residual": E3_BAR_RESIDUAL_THRESHOLD,
        },
        "gates": gates,
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "records": [json_ready(record) for record in records],
    }

    write_csv(records)
    write_npz(records)
    (OUTPUT_DIR / "stageE_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(payload)

    print(json.dumps({
        "overall_passed": overall_passed,
        "E1_max_tv_vs_exact": gates["E1"]["max_tv_vs_exact"],
        "E1_max_q_top_abs_diff_vs_exact": gates["E1"][
            "max_q_top_abs_diff_vs_exact"
        ],
        "E2_max_tv_vs_stage_d_ti": gates["E2"]["max_tv_vs_stage_d_ti"],
        "E2_max_q_top_abs_diff_vs_stage_d_ti": gates["E2"][
            "max_q_top_abs_diff_vs_stage_d_ti"
        ],
        "E3_max_bidirectional_gap": gates["E3"]["max_bidirectional_gap"],
        "E3_max_bar_residual": gates["E3"]["max_bar_residual"],
        "wall_time_seconds": payload["wall_time_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
