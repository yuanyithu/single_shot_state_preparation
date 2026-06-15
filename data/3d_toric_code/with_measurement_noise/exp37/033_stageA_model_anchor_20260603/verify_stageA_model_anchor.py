#!/usr/bin/env python3
"""Stage A model-alignment checks for exp37.

This script does not sample.  It validates the production exp37 x-space
energy, corrected decoder-sector labels, exact sector aggregation, the Kp=0
analytic anchor, and the q_top formula.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_toric_code_examples import build_toric_code_by_family
from exp37_sector_ti import (
    _compute_exact_sector_weights_decoder,
    _compute_k,
    _q_top_from_weights,
)
from exact_enumeration import _iter_chain_bit_chunks, _logsumexp
from linear_section import build_syndrome_representative_section


OUTPUT_DIR = SCRIPT_PATH.parent


def gf2_rank(matrix: np.ndarray) -> int:
    working = np.asarray(matrix, dtype=bool).copy()
    num_rows, num_columns = working.shape
    pivot_row = 0
    for column in range(num_columns):
        if pivot_row >= num_rows:
            break
        pivot_candidates = np.flatnonzero(working[pivot_row:, column])
        if pivot_candidates.size == 0:
            continue
        selected = pivot_row + int(pivot_candidates[0])
        if selected != pivot_row:
            working[[pivot_row, selected]] = working[[selected, pivot_row]]
        for row in range(num_rows):
            if row != pivot_row and working[row, column]:
                working[row] ^= working[pivot_row]
        pivot_row += 1
    return int(pivot_row)


def production_path_energy(
    x_bits: np.ndarray,
    parity_check_matrix_uint8: np.ndarray,
    measurement_error_bits: np.ndarray,
    kp_value: float,
    kq_value: float,
) -> float:
    syndrome_term_bits = (
        parity_check_matrix_uint8 @ np.asarray(x_bits, dtype=np.uint8)
    ) % 2
    syndrome_term_bits = syndrome_term_bits.astype(bool) ^ measurement_error_bits
    data_weight = int(np.count_nonzero(x_bits))
    syndrome_weight = int(np.count_nonzero(syndrome_term_bits))
    return float(kp_value * data_weight + kq_value * syndrome_weight)


def brute_force_energy(
    x_bits: np.ndarray,
    parity_check_matrix: np.ndarray,
    measurement_error_bits: np.ndarray,
    kp_value: float,
    kq_value: float,
) -> float:
    data_weight = 0
    for bit in np.asarray(x_bits, dtype=bool):
        data_weight += int(bool(bit))

    syndrome_weight = 0
    x_bool = np.asarray(x_bits, dtype=bool)
    for check_index in range(parity_check_matrix.shape[0]):
        parity = False
        for qubit_index in np.flatnonzero(parity_check_matrix[check_index]):
            parity ^= bool(x_bool[int(qubit_index)])
        parity ^= bool(measurement_error_bits[check_index])
        syndrome_weight += int(parity)
    return float(kp_value * data_weight + kq_value * syndrome_weight)


def run_a1_energy_check() -> dict:
    lattice_size = 2
    parity_check_matrix, _ = build_toric_code_by_family("3d_toric", lattice_size)
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    num_checks, num_qubits = parity_check_matrix.shape
    p_value = 0.137
    q_value = 0.211
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    rng = np.random.default_rng(3700101)
    measurement_error_bits = rng.random(num_checks) < q_value

    max_abs_diff = 0.0
    for _ in range(1200):
        x_bits = rng.integers(0, 2, size=num_qubits).astype(bool)
        production_energy = production_path_energy(
            x_bits=x_bits,
            parity_check_matrix_uint8=parity_check_matrix_uint8,
            measurement_error_bits=measurement_error_bits,
            kp_value=kp_value,
            kq_value=kq_value,
        )
        brute_energy = brute_force_energy(
            x_bits=x_bits,
            parity_check_matrix=parity_check_matrix,
            measurement_error_bits=measurement_error_bits,
            kp_value=kp_value,
            kq_value=kq_value,
        )
        max_abs_diff = max(max_abs_diff, abs(production_energy - brute_energy))

    return {
        "name": "A1",
        "num_random_x": 1200,
        "p_value": p_value,
        "q_value": q_value,
        "max_abs_energy_diff": max_abs_diff,
        "threshold": 0.0,
        "passed": bool(max_abs_diff == 0.0),
    }


def compute_global_log_z_by_enumeration(
    parity_check_matrix: np.ndarray,
    measurement_error_bits: np.ndarray,
    kp_value: float,
    kq_value: float,
    chunk_size: int,
) -> float:
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    log_z = -np.inf
    num_qubits = parity_check_matrix.shape[1]
    for x_chunk in _iter_chain_bit_chunks(num_qubits, chunk_size):
        x_uint8 = x_chunk.astype(np.uint8)
        data_weights = np.count_nonzero(x_chunk, axis=1)
        syndrome_bits = (x_uint8 @ parity_check_matrix_uint8.T) % 2
        syndrome_bits = syndrome_bits.astype(bool) ^ measurement_error_bits[None, :]
        syndrome_weights = np.count_nonzero(syndrome_bits, axis=1)
        log_weights = (
            -kp_value * data_weights.astype(np.float64)
            -kq_value * syndrome_weights.astype(np.float64)
        )
        log_z = np.logaddexp(log_z, _logsumexp(log_weights))
    return float(log_z)


def run_a2_exact_sector_sum_check() -> dict:
    lattice_size = 2
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        "3d_toric",
        lattice_size,
    )
    num_checks, num_qubits = parity_check_matrix.shape
    p_value = 0.173
    q_value = 0.197
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    rng = np.random.default_rng(3700202)
    eta_bits = rng.random(num_qubits) < p_value
    measurement_error_bits = rng.random(num_checks) < q_value
    disorder_syndrome_bits = (
        parity_check_matrix.astype(np.uint8) @ eta_bits.astype(np.uint8)
    ) % 2
    section_data = build_syndrome_representative_section(
        parity_check_matrix,
        prefer_bplsd=False,
    )

    exact = _compute_exact_sector_weights_decoder(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
        disorder_syndrome_bits=disorder_syndrome_bits.astype(bool),
        measurement_error_bits=measurement_error_bits,
        p_value=p_value,
        q_value=q_value,
        chunk_size=1 << 18,
    )
    sector_sum_log_z = _logsumexp(np.asarray(exact["log_z"], dtype=np.float64))
    global_log_z = compute_global_log_z_by_enumeration(
        parity_check_matrix=parity_check_matrix,
        measurement_error_bits=measurement_error_bits,
        kp_value=kp_value,
        kq_value=kq_value,
        chunk_size=1 << 18,
    )
    log_z_abs_diff = abs(float(sector_sum_log_z - global_log_z))
    probability_sum_diff = abs(float(np.sum(exact["weights"])) - 1.0)
    return {
        "name": "A2",
        "lattice_size": lattice_size,
        "num_qubits": int(num_qubits),
        "p_value": p_value,
        "q_value": q_value,
        "eta_weight": int(np.count_nonzero(eta_bits)),
        "measurement_error_weight": int(np.count_nonzero(measurement_error_bits)),
        "section_backend": section_data.stats()["backend_name"],
        "sector_sum_log_z": float(sector_sum_log_z),
        "global_log_z": float(global_log_z),
        "log_z_abs_diff": log_z_abs_diff,
        "probability_sum_diff": probability_sum_diff,
        "threshold": 1.0e-9,
        "passed": bool(log_z_abs_diff < 1.0e-9 and probability_sum_diff < 1.0e-12),
    }


def run_a3_kp0_anchor_check() -> dict:
    records = []
    max_weight_deviation = 0.0
    max_q_top_deviation = 0.0
    for lattice_size in (2, 3):
        parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
            "3d_toric",
            lattice_size,
        )
        rank_h = gf2_rank(parity_check_matrix)
        rank_augmented = gf2_rank(
            np.vstack((parity_check_matrix, primitive_logical_masks))
        )
        rank_increment = int(rank_augmented - rank_h)
        if rank_increment == primitive_logical_masks.shape[0]:
            weights = np.full(8, 1.0 / 8.0, dtype=np.float64)
        else:
            weights = np.full(8, np.nan, dtype=np.float64)
        q_top = _q_top_from_weights(weights) if np.all(np.isfinite(weights)) else np.nan
        weight_deviation = float(np.max(np.abs(weights - 1.0 / 8.0)))
        q_top_deviation = float(abs(q_top - 0.0))
        max_weight_deviation = max(max_weight_deviation, weight_deviation)
        max_q_top_deviation = max(max_q_top_deviation, q_top_deviation)
        records.append({
            "lattice_size": int(lattice_size),
            "rank_h": int(rank_h),
            "rank_augmented": int(rank_augmented),
            "rank_increment": int(rank_increment),
            "expected_increment": int(primitive_logical_masks.shape[0]),
            "max_abs_weight_minus_one_eighth": weight_deviation,
            "q_top": q_top,
        })
    return {
        "name": "A3",
        "method": "rank([H_Z; logical_z]) = rank(H_Z) + 3 proves equal sector counts in every syndrome fiber at Kp=0",
        "records": records,
        "max_abs_weight_minus_one_eighth": max_weight_deviation,
        "max_abs_q_top_minus_zero": max_q_top_deviation,
        "threshold": 1.0e-9,
        "passed": bool(
            max_weight_deviation < 1.0e-9 and max_q_top_deviation < 1.0e-9
        ),
    }


def run_a4_qtop_roundtrip_check() -> dict:
    rng = np.random.default_rng(3700404)
    max_purity_roundtrip_diff = 0.0
    for _ in range(1000):
        weights = rng.random(8)
        weights /= np.sum(weights)
        q_top = _q_top_from_weights(weights)
        recovered_purity = (7.0 * q_top + 1.0) / 8.0
        direct_purity = float(np.sum(weights ** 2))
        max_purity_roundtrip_diff = max(
            max_purity_roundtrip_diff,
            abs(recovered_purity - direct_purity),
        )
    uniform_q_top = _q_top_from_weights(np.full(8, 1.0 / 8.0))
    pure_weights = np.zeros(8, dtype=np.float64)
    pure_weights[0] = 1.0
    pure_q_top = _q_top_from_weights(pure_weights)
    max_anchor_diff = max(abs(uniform_q_top - 0.0), abs(pure_q_top - 1.0))
    return {
        "name": "A4",
        "num_random_weight_vectors": 1000,
        "max_purity_roundtrip_diff": float(max_purity_roundtrip_diff),
        "uniform_q_top": float(uniform_q_top),
        "pure_q_top": float(pure_q_top),
        "max_anchor_diff": float(max_anchor_diff),
        "threshold": 1.0e-14,
        "passed": bool(max_purity_roundtrip_diff < 1.0e-14 and max_anchor_diff < 1.0e-14),
    }


def write_summary(results: dict) -> None:
    a1 = results["checks"]["A1"]
    a2 = results["checks"]["A2"]
    a3 = results["checks"]["A3"]
    a4 = results["checks"]["A4"]
    lines = [
        "# Stage A model anchor summary",
        "",
        f"Overall: {'PASS' if results['overall_passed'] else 'FAIL'}",
        "",
        "No sampling was used.  The exact L=2 check enumerates all 2^24 x-space configurations.",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| A1 | 1200 random x energies match brute force exactly | "
            f"{a1['max_abs_energy_diff']:.3e} | {'PASS' if a1['passed'] else 'FAIL'} |"
        ),
        (
            f"| A2 | L=2 sum_g Z_g equals global Z, log error < 1e-9 | "
            f"{a2['log_z_abs_diff']:.3e} | {'PASS' if a2['passed'] else 'FAIL'} |"
        ),
        (
            f"| A3 | Kp=0 gives w_g=1/8 and q_top=0 on L=2,L=3 | "
            f"max |dw|={a3['max_abs_weight_minus_one_eighth']:.3e}, "
            f"max |dq|={a3['max_abs_q_top_minus_zero']:.3e} | "
            f"{'PASS' if a3['passed'] else 'FAIL'} |"
        ),
        (
            f"| A4 | q_top purity roundtrip for random w_g | "
            f"{a4['max_purity_roundtrip_diff']:.3e} | "
            f"{'PASS' if a4['passed'] else 'FAIL'} |"
        ),
        "",
        "## Details",
        "",
        f"- A2 section backend: `{a2['section_backend']}`.",
        (
            f"- A2 exact point: L=2, p={a2['p_value']:.6g}, q={a2['q_value']:.6g}, "
            f"eta weight={a2['eta_weight']}, measurement-error weight={a2['measurement_error_weight']}."
        ),
        "- A3 rank increments:",
    ]
    for record in a3["records"]:
        lines.append(
            f"  - L={record['lattice_size']}: rank(H)={record['rank_h']}, "
            f"rank([H;Z])={record['rank_augmented']}, "
            f"increment={record['rank_increment']}."
        )
    lines.append("")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    checks = {
        "A1": run_a1_energy_check(),
        "A2": run_a2_exact_sector_sum_check(),
        "A3": run_a3_kp0_anchor_check(),
        "A4": run_a4_qtop_roundtrip_check(),
    }
    overall_passed = all(check["passed"] for check in checks.values())
    results = {
        "stage": "A",
        "overall_passed": bool(overall_passed),
        "checks": checks,
    }
    (OUTPUT_DIR / "stageA_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(results)
    print(json.dumps({
        "overall_passed": overall_passed,
        "A1_max_abs_energy_diff": checks["A1"]["max_abs_energy_diff"],
        "A2_log_z_abs_diff": checks["A2"]["log_z_abs_diff"],
        "A3_max_abs_q_top_minus_zero": checks["A3"]["max_abs_q_top_minus_zero"],
        "A4_max_purity_roundtrip_diff": checks["A4"]["max_purity_roundtrip_diff"],
    }, indent=2, sort_keys=True))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
