#!/usr/bin/env python3
"""Generate the Stage B exact L=2 benchmark table for exp37.

The benchmark uses zero disorder on the L=2 3D toric code and the corrected
decoder-section sector label.  It builds an independent sector/data/syndrome
count table by full enumeration, chooses points with q_top away from 1, and
then cross-checks those points with the production exact helper.
"""

from __future__ import annotations

import csv
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
from exact_enumeration import _iter_chain_bit_chunks, _logsumexp
from exp37_sector_ti import _compute_exact_sector_weights_decoder, _compute_k
from linear_section import apply_section, build_syndrome_representative_section


OUTPUT_DIR = SCRIPT_PATH.parent
NUM_SECTORS = 8
LATTICE_SIZE = 2
CHUNK_SIZE = 1 << 18
TARGET_QTOPS = (0.25, 0.40, 0.55, 0.70, 0.85, 0.92)


def q_top_from_weights(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float((8.0 * np.sum(weights ** 2) - 1.0) / 7.0)


def weights_from_log_z(log_z: np.ndarray) -> np.ndarray:
    log_z = np.asarray(log_z, dtype=np.float64)
    total = _logsumexp(log_z)
    return np.exp(log_z - total)


def delta_f_from_log_z(log_z: np.ndarray) -> np.ndarray:
    log_z = np.asarray(log_z, dtype=np.float64)
    return -(log_z - log_z[0])


def build_signature_lookup(
    parity_check_matrix: np.ndarray,
    primitive_logical_masks: np.ndarray,
    section_data,
) -> np.ndarray:
    num_checks = parity_check_matrix.shape[0]
    syndrome_bit_weights = 1 << np.arange(num_checks, dtype=np.int64)
    signature_lookup = np.full(1 << num_checks, -1, dtype=np.int8)

    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    for x_chunk in _iter_chain_bit_chunks(
        parity_check_matrix.shape[1],
        CHUNK_SIZE,
    ):
        syndrome_bits_chunk = (
            x_chunk.astype(np.uint8) @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_indices = (
            syndrome_bits_chunk.astype(np.int64) @ syndrome_bit_weights
        )
        missing = np.unique(
            syndrome_indices[signature_lookup[syndrome_indices] < 0]
        )
        for syndrome_index in missing:
            syndrome_bits = (
                (
                    int(syndrome_index)
                    >> np.arange(num_checks, dtype=np.int64)
                ) & 1
            ).astype(bool)
            representative_bits = apply_section(syndrome_bits, section_data)
            logical_bits = (
                primitive_logical_masks.astype(np.uint8)
                @ representative_bits.astype(np.uint8)
            ) % 2
            signature = int(
                logical_bits.astype(np.int64)
                @ (1 << np.arange(primitive_logical_masks.shape[0], dtype=np.int64))
            )
            signature_lookup[int(syndrome_index)] = np.int8(signature)
    if np.any(signature_lookup[signature_lookup >= 0] >= NUM_SECTORS):
        raise AssertionError("invalid signature lookup entry")
    return signature_lookup


def build_independent_count_table(
    parity_check_matrix: np.ndarray,
    primitive_logical_masks: np.ndarray,
    section_data,
) -> tuple[np.ndarray, dict]:
    """Enumerate all x and count by corrected sector, |x|, and |Hx|.

    This implementation deliberately does not call exp37's exact benchmark
    helper.  It reduces the full state space into a small integer table, which
    is then used for every (p, q) point.
    """
    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    primitive_masks_uint8 = primitive_logical_masks.astype(np.uint8)
    bit_weights = 1 << np.arange(primitive_logical_masks.shape[0], dtype=np.int64)
    syndrome_bit_weights = 1 << np.arange(num_checks, dtype=np.int64)
    signature_lookup = build_signature_lookup(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
    )

    counts = np.zeros(
        (NUM_SECTORS, num_qubits + 1, num_checks + 1),
        dtype=np.int64,
    )
    encoded_size = (num_qubits + 1) * (num_checks + 1)
    total_seen = 0
    for x_chunk in _iter_chain_bit_chunks(num_qubits, CHUNK_SIZE):
        x_uint8 = x_chunk.astype(np.uint8)
        data_weights = np.count_nonzero(x_chunk, axis=1)
        syndrome_bits_chunk = (x_uint8 @ parity_check_matrix_uint8.T) % 2
        syndrome_weights = np.count_nonzero(syndrome_bits_chunk, axis=1)
        syndrome_indices = (
            syndrome_bits_chunk.astype(np.int64) @ syndrome_bit_weights
        )
        raw_logical_bits = (x_uint8 @ primitive_masks_uint8.T) % 2
        raw_logical_indices = raw_logical_bits.astype(np.int64) @ bit_weights
        sector_indices = np.bitwise_xor(
            raw_logical_indices,
            signature_lookup[syndrome_indices].astype(np.int64),
        )
        if np.any(sector_indices < 0):
            raise AssertionError("unfilled syndrome signature lookup")
        encoded = (
            sector_indices.astype(np.int64) * encoded_size
            + data_weights.astype(np.int64) * (num_checks + 1)
            + syndrome_weights.astype(np.int64)
        )
        counts += np.bincount(
            encoded,
            minlength=NUM_SECTORS * encoded_size,
        ).reshape(counts.shape)
        total_seen += int(x_chunk.shape[0])

    metadata = {
        "total_configurations": int(total_seen),
        "expected_configurations": int(1 << num_qubits),
        "num_reachable_syndromes": int(np.count_nonzero(signature_lookup >= 0)),
        "section_stats": section_data.stats(),
    }
    if total_seen != 1 << num_qubits:
        raise AssertionError("enumeration count mismatch")
    return counts, metadata


def exact_from_count_table(
    counts: np.ndarray,
    p_value: float,
    q_value: float,
) -> dict:
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    _, num_data_weights, num_syndrome_weights = counts.shape
    data_grid = np.arange(num_data_weights, dtype=np.float64)[None, :, None]
    syndrome_grid = np.arange(num_syndrome_weights, dtype=np.float64)[None, None, :]
    log_counts = np.full(counts.shape, -np.inf, dtype=np.float64)
    positive = counts > 0
    log_counts[positive] = np.log(counts[positive].astype(np.float64))
    log_terms = log_counts - kp_value * data_grid - kq_value * syndrome_grid
    log_z = np.array([
        _logsumexp(log_terms[sector].reshape(-1))
        for sector in range(NUM_SECTORS)
    ], dtype=np.float64)
    weights = weights_from_log_z(log_z)
    return {
        "kp_value": float(kp_value),
        "kq_value": float(kq_value),
        "log_z": log_z,
        "weights": weights,
        "delta_f": delta_f_from_log_z(log_z),
        "q_top": q_top_from_weights(weights),
    }


def build_candidate_grid() -> list[tuple[float, float]]:
    p_values = np.unique(np.concatenate([
        np.linspace(0.05, 0.30, 26),
        np.linspace(0.305, 0.485, 37),
    ]))
    q_values = np.unique(np.concatenate([
        np.linspace(0.05, 0.30, 26),
        np.linspace(0.305, 0.485, 37),
    ]))
    return [
        (float(round(p_value, 6)), float(round(q_value, 6)))
        for p_value in p_values
        for q_value in q_values
        if 0.0 < p_value < 0.5 and 0.0 < q_value < 0.5
    ]


def select_reference_points(candidate_rows: list[dict]) -> list[dict]:
    selected = []
    used_pairs = set()
    for target_qtop in TARGET_QTOPS:
        candidates = sorted(
            candidate_rows,
            key=lambda row: (
                abs(float(row["q_top"]) - float(target_qtop)),
                float(row["p_value"]),
                float(row["q_value"]),
            ),
        )
        for candidate in candidates:
            pair = (candidate["p_value"], candidate["q_value"])
            if pair not in used_pairs:
                selected.append(candidate)
                used_pairs.add(pair)
                break
    selected = sorted(
        selected,
        key=lambda row: float(row["q_top"]),
    )
    return selected


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "record_id",
        "p_value",
        "q_value",
        "kp_value",
        "kq_value",
        "q_top",
        "independent_tv",
        "max_abs_weight_diff",
    ]
    fieldnames += [f"w_{sector}" for sector in range(NUM_SECTORS)]
    fieldnames += [f"delta_f_{sector}" for sector in range(NUM_SECTORS)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "record_id": record["record_id"],
                "p_value": record["p_value"],
                "q_value": record["q_value"],
                "kp_value": record["kp_value"],
                "kq_value": record["kq_value"],
                "q_top": record["q_top"],
                "independent_tv": record["independent_check"]["tv"],
                "max_abs_weight_diff": record["independent_check"][
                    "max_abs_weight_diff"
                ],
            }
            for sector, weight in enumerate(record["weights"]):
                row[f"w_{sector}"] = weight
            for sector, delta_f in enumerate(record["delta_f"]):
                row[f"delta_f_{sector}"] = delta_f
            writer.writerow(row)


def write_candidate_scan(path: Path, candidate_rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("p_value", "q_value", "q_top"),
        )
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "q_top": row["q_top"],
            })


def write_summary(payload: dict) -> None:
    records = payload["records"]
    mid_records = [
        record for record in records
        if 0.2 <= float(record["q_top"]) <= 0.8
    ]
    max_tv = max(
        float(record["independent_check"]["tv"])
        for record in records
    )
    max_dq = max(
        abs(
            float(record["q_top"])
            - float(record["independent_check"]["production_q_top"])
        )
        for record in records
    )
    lines = [
        "# Stage B exact reference summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        "Reference: L=2 3D toric code, zero eta, zero measurement error, corrected decoder-section sector label.",
        "Primary table is produced by an independent full-enumeration count table; B2 compares against the production exact helper.",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| B1 | >=3 exact q_top values in [0.2,0.8] | "
            f"{len(mid_records)} / {len(records)} | "
            f"{'PASS' if payload['gates']['B1']['passed'] else 'FAIL'} |"
        ),
        (
            f"| B2 | independent implementation TV < 1e-9 | "
            f"max TV={max_tv:.3e}, max dq={max_dq:.3e} | "
            f"{'PASS' if payload['gates']['B2']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Reference Points",
        "",
        "| id | p | q | q_top | TV(count-table, production) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['record_id']} | {record['p_value']:.6f} | "
            f"{record['q_value']:.6f} | {record['q_top']:.6f} | "
            f"{record['independent_check']['tv']:.3e} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `exact_reference.json`",
        "- `exact_reference.csv`",
        "- `candidate_scan.csv`",
    ])
    (OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    section_data = build_syndrome_representative_section(
        parity_check_matrix,
        prefer_bplsd=False,
    )
    counts, count_metadata = build_independent_count_table(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
    )

    candidate_rows = []
    for p_value, q_value in build_candidate_grid():
        exact = exact_from_count_table(
            counts=counts,
            p_value=p_value,
            q_value=q_value,
        )
        candidate_rows.append({
            "p_value": p_value,
            "q_value": q_value,
            "q_top": float(exact["q_top"]),
        })
    write_candidate_scan(OUTPUT_DIR / "candidate_scan.csv", candidate_rows)

    selected_rows = select_reference_points(candidate_rows)
    if len(selected_rows) < len(TARGET_QTOPS):
        raise RuntimeError("failed to select enough reference points")

    disorder_syndrome_bits = np.zeros(parity_check_matrix.shape[0], dtype=bool)
    measurement_error_bits = np.zeros(parity_check_matrix.shape[0], dtype=bool)
    records = []
    for record_id, selected in enumerate(selected_rows):
        p_value = float(selected["p_value"])
        q_value = float(selected["q_value"])
        count_exact = exact_from_count_table(
            counts=counts,
            p_value=p_value,
            q_value=q_value,
        )
        production_exact = _compute_exact_sector_weights_decoder(
            parity_check_matrix=parity_check_matrix,
            primitive_logical_masks=primitive_logical_masks,
            section_data=section_data,
            disorder_syndrome_bits=disorder_syndrome_bits,
            measurement_error_bits=measurement_error_bits,
            p_value=p_value,
            q_value=q_value,
            chunk_size=CHUNK_SIZE,
        )
        tv = total_variation(count_exact["weights"], production_exact["weights"])
        max_abs_weight_diff = float(
            np.max(np.abs(count_exact["weights"] - production_exact["weights"]))
        )
        q_top_diff = abs(float(count_exact["q_top"]) - float(production_exact["q_top"]))
        records.append({
            "record_id": int(record_id),
            "lattice_size": LATTICE_SIZE,
            "code_family": "3d_toric",
            "disorder": "zero_eta_zero_measurement_error",
            "p_value": p_value,
            "q_value": q_value,
            "kp_value": float(count_exact["kp_value"]),
            "kq_value": float(count_exact["kq_value"]),
            "weights": count_exact["weights"].tolist(),
            "delta_f": count_exact["delta_f"].tolist(),
            "q_top": float(count_exact["q_top"]),
            "independent_check": {
                "method_a": "independent_count_table",
                "method_b": "src.exp37_sector_ti._compute_exact_sector_weights_decoder",
                "tv": float(tv),
                "max_abs_weight_diff": max_abs_weight_diff,
                "q_top_abs_diff": float(q_top_diff),
                "production_q_top": float(production_exact["q_top"]),
            },
        })

    mid_records = [
        record for record in records
        if 0.2 <= float(record["q_top"]) <= 0.8
    ]
    max_tv = max(
        float(record["independent_check"]["tv"])
        for record in records
    )
    b1_passed = len(mid_records) >= 3
    b2_passed = max_tv < 1.0e-9
    payload = {
        "stage": "B",
        "mode": "exact_reference_l2",
        "overall_passed": bool(b1_passed and b2_passed),
        "sector_observable": "corrected_c_eta_section",
        "projection_mode": "decoder_reject",
        "section_prefer_bplsd": False,
        "code_family": "3d_toric",
        "lattice_size": LATTICE_SIZE,
        "num_qubits": int(parity_check_matrix.shape[1]),
        "num_checks": int(parity_check_matrix.shape[0]),
        "disorder": {
            "eta_weight": 0,
            "measurement_error_weight": 0,
            "eta_syndrome_weight": 0,
        },
        "enumeration": count_metadata,
        "selection_targets_q_top": list(TARGET_QTOPS),
        "gates": {
            "B1": {
                "passed": bool(b1_passed),
                "num_mid_q_top_points": int(len(mid_records)),
                "required": 3,
                "interval": [0.2, 0.8],
            },
            "B2": {
                "passed": bool(b2_passed),
                "max_tv": float(max_tv),
                "threshold": 1.0e-9,
            },
        },
        "records": records,
    }
    (OUTPUT_DIR / "exact_reference.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "exact_reference.csv", records)
    write_summary(payload)
    print(json.dumps({
        "overall_passed": payload["overall_passed"],
        "num_mid_q_top_points": len(mid_records),
        "max_tv": max_tv,
        "q_top_values": [record["q_top"] for record in records],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
