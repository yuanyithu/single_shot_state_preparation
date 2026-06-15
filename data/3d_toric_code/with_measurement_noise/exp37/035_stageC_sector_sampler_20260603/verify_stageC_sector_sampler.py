#!/usr/bin/env python3
"""Stage C self-check for the exp37 fixed-sector sampler.

The external reference is a direct L=2 full-enumeration count table of exact
sector-conditional means.  The sampled path reuses the production
decoder-reject fixed-sector proposal machinery from src/exp37_sector_ti.py.
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

from build_toric_code_examples import (  # noqa: E402
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from exact_enumeration import _iter_chain_bit_chunks, _logsumexp  # noqa: E402
from exp37_sector_ti import (  # noqa: E402
    _build_decoder_reject_proposals,
    _build_sector_representatives,
    _compute_corrected_x_decoder_signature,
    _compute_k,
    _run_decoder_reject_sector_sweep,
)
from linear_section import (  # noqa: E402
    apply_section,
    build_syndrome_representative_section,
)


OUTPUT_DIR = SCRIPT_PATH.parent
LATTICE_SIZE = 2
P_VALUE = 0.28
Q_VALUE = 0.305
CHUNK_SIZE = 1 << 18
NUM_SECTORS = 8
NUM_BURN_IN_SWEEPS = 3000
NUM_MEASUREMENTS = 24000
NUM_SWEEPS_BETWEEN_MEASUREMENTS = 2
BLOCK_COUNT = 48
NUM_BOOTSTRAP = 4000
CI_LOW = 0.005
CI_HIGH = 0.995
SEED = 3703501


def signature_from_representative(
    representative_bits: np.ndarray,
    primitive_logical_masks: np.ndarray,
) -> int:
    logical_bits = (
        primitive_logical_masks.astype(np.uint8)
        @ representative_bits.astype(np.uint8)
    ) % 2
    return int(
        logical_bits.astype(np.int64)
        @ (1 << np.arange(primitive_logical_masks.shape[0], dtype=np.int64))
    )


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
            signature_lookup[int(syndrome_index)] = np.int8(
                signature_from_representative(
                    representative_bits=representative_bits,
                    primitive_logical_masks=primitive_logical_masks,
                )
            )
    return signature_lookup


def build_exact_count_table(
    parity_check_matrix: np.ndarray,
    primitive_logical_masks: np.ndarray,
    section_data,
) -> np.ndarray:
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
    counts = np.zeros((NUM_SECTORS, num_qubits + 1, num_checks + 1), dtype=np.int64)
    encoded_size = (num_qubits + 1) * (num_checks + 1)
    for x_chunk in _iter_chain_bit_chunks(num_qubits, CHUNK_SIZE):
        x_uint8 = x_chunk.astype(np.uint8)
        data_weights = np.count_nonzero(x_chunk, axis=1)
        syndrome_bits_chunk = (x_uint8 @ parity_check_matrix_uint8.T) % 2
        syndrome_weights = np.count_nonzero(syndrome_bits_chunk, axis=1)
        syndrome_indices = syndrome_bits_chunk.astype(np.int64) @ syndrome_bit_weights
        raw_logical_bits = (x_uint8 @ primitive_masks_uint8.T) % 2
        raw_logical_indices = raw_logical_bits.astype(np.int64) @ bit_weights
        sector_indices = np.bitwise_xor(
            raw_logical_indices,
            signature_lookup[syndrome_indices].astype(np.int64),
        )
        encoded = (
            sector_indices * encoded_size
            + data_weights.astype(np.int64) * (num_checks + 1)
            + syndrome_weights.astype(np.int64)
        )
        counts += np.bincount(
            encoded,
            minlength=NUM_SECTORS * encoded_size,
        ).reshape(counts.shape)
    return counts


def exact_sector_moments(counts: np.ndarray, p_value: float, q_value: float) -> dict:
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    _, num_data_weights, num_syndrome_weights = counts.shape
    data_grid = np.arange(num_data_weights, dtype=np.float64)[None, :, None]
    syndrome_grid = np.arange(num_syndrome_weights, dtype=np.float64)[None, None, :]
    log_counts = np.full(counts.shape, -np.inf, dtype=np.float64)
    positive = counts > 0
    log_counts[positive] = np.log(counts[positive].astype(np.float64))
    log_terms = log_counts - kp_value * data_grid - kq_value * syndrome_grid

    log_z = np.empty(NUM_SECTORS, dtype=np.float64)
    data_mu = np.empty(NUM_SECTORS, dtype=np.float64)
    syndrome_mu = np.empty(NUM_SECTORS, dtype=np.float64)
    for sector in range(NUM_SECTORS):
        sector_terms = log_terms[sector]
        log_z[sector] = _logsumexp(sector_terms.reshape(-1))
        normalized_weights = np.exp(sector_terms - log_z[sector])
        data_mu[sector] = float(
            np.sum(normalized_weights * data_grid.reshape(num_data_weights, 1))
        )
        syndrome_mu[sector] = float(
            np.sum(normalized_weights * syndrome_grid.reshape(1, num_syndrome_weights))
        )
    return {
        "log_z": log_z,
        "data_mu": data_mu,
        "syndrome_mu": syndrome_mu,
    }


def bootstrap_ci(block_means: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    block_means = np.asarray(block_means, dtype=np.float64)
    num_blocks = int(block_means.shape[0])
    samples = np.empty(NUM_BOOTSTRAP, dtype=np.float64)
    for sample_index in range(NUM_BOOTSTRAP):
        indices = rng.integers(0, num_blocks, size=num_blocks)
        samples[sample_index] = float(np.mean(block_means[indices]))
    return (
        float(np.mean(block_means)),
        float(np.quantile(samples, CI_LOW)),
        float(np.quantile(samples, CI_HIGH)),
    )


def run_sampler(
    parity_check_matrix: np.ndarray,
    primitive_logical_masks: np.ndarray,
    section_data,
    exact: dict,
) -> dict:
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    num_checks, num_qubits = parity_check_matrix.shape
    measurement_error_bits = np.zeros(num_checks, dtype=bool)
    disorder_syndrome_bits = np.zeros(num_checks, dtype=bool)
    disorder_syndrome_representative_bits = apply_section(
        disorder_syndrome_bits,
        section_data,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    proposals = _build_decoder_reject_proposals(
        parity_check_matrix=parity_check_matrix,
        zero_syndrome_move_data=zero_syndrome_move_data,
    )
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=primitive_logical_masks,
        parity_check_matrix=parity_check_matrix,
    )
    kp_value = _compute_k(P_VALUE)
    kq_value = _compute_k(Q_VALUE)
    rng_master = np.random.default_rng(SEED)
    block_indices = np.array_split(np.arange(NUM_MEASUREMENTS), BLOCK_COUNT)

    records = []
    trace_rows = []
    invariant_failure_count = 0
    for sector in range(NUM_SECTORS):
        rng = np.random.default_rng(
            int(rng_master.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        )
        current_x_bits = sector_representatives[sector].copy()
        current_chain_syndrome_bits = (
            parity_check_matrix_uint8 @ current_x_bits.astype(np.uint8)
        ) % 2
        current_chain_syndrome_bits = current_chain_syndrome_bits.astype(bool)
        current_syndrome_term_bits = current_chain_syndrome_bits ^ measurement_error_bits
        sector_trace = []

        def check_signature(context: str) -> int:
            nonlocal invariant_failure_count
            chain_syndrome_bits = current_syndrome_term_bits ^ measurement_error_bits
            signature = _compute_corrected_x_decoder_signature(
                chain_bits=current_x_bits,
                chain_syndrome_bits=chain_syndrome_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
            )
            if int(signature) != int(sector):
                invariant_failure_count += 1
                raise AssertionError(
                    f"sector invariant failed in {context}: "
                    f"target={sector}, got={signature}"
                )
            return int(signature)

        initial_signature = check_signature("initialization")
        accepted_total = 0
        attempted_total = 0
        sector_rejected_total = 0
        for burn_index in range(NUM_BURN_IN_SWEEPS):
            accepted, attempted, sector_rejected = _run_decoder_reject_sector_sweep(
                current_x_bits=current_x_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                measurement_error_bits=measurement_error_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                proposals=proposals,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                target_sector=sector,
                kp_value=kp_value,
                kq_value=kq_value,
                rng=rng,
            )
            accepted_total += accepted
            attempted_total += attempted
            sector_rejected_total += sector_rejected
            if burn_index % 100 == 0:
                check_signature(f"burn_sweep_{burn_index}")

        data_samples = np.empty(NUM_MEASUREMENTS, dtype=np.float64)
        syndrome_samples = np.empty(NUM_MEASUREMENTS, dtype=np.float64)
        for measurement_index in range(NUM_MEASUREMENTS):
            for stride_index in range(NUM_SWEEPS_BETWEEN_MEASUREMENTS):
                accepted, attempted, sector_rejected = _run_decoder_reject_sector_sweep(
                    current_x_bits=current_x_bits,
                    current_syndrome_term_bits=current_syndrome_term_bits,
                    measurement_error_bits=measurement_error_bits,
                    disorder_syndrome_bits=disorder_syndrome_bits,
                    disorder_syndrome_representative_bits=(
                        disorder_syndrome_representative_bits
                    ),
                    proposals=proposals,
                    primitive_logical_masks=primitive_logical_masks,
                    section_data=section_data,
                    target_sector=sector,
                    kp_value=kp_value,
                    kq_value=kq_value,
                    rng=rng,
                )
                accepted_total += accepted
                attempted_total += attempted
                sector_rejected_total += sector_rejected
                if stride_index == NUM_SWEEPS_BETWEEN_MEASUREMENTS - 1:
                    signature = check_signature(
                        f"measurement_{measurement_index}_stride_{stride_index}"
                    )
            sector_trace.append(signature)
            data_samples[measurement_index] = float(np.count_nonzero(current_x_bits))
            syndrome_samples[measurement_index] = float(
                np.count_nonzero(current_syndrome_term_bits)
            )
            if measurement_index % 100 == 0:
                trace_rows.append({
                    "sector": sector,
                    "measurement_index": measurement_index,
                    "signature": signature,
                })

        data_block_means = np.array([
            float(np.mean(data_samples[indices]))
            for indices in block_indices
        ])
        syndrome_block_means = np.array([
            float(np.mean(syndrome_samples[indices]))
            for indices in block_indices
        ])
        bootstrap_rng = np.random.default_rng(
            int(rng_master.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        )
        data_mean, data_ci_low, data_ci_high = bootstrap_ci(
            data_block_means,
            bootstrap_rng,
        )
        syndrome_mean, syndrome_ci_low, syndrome_ci_high = bootstrap_ci(
            syndrome_block_means,
            bootstrap_rng,
        )
        exact_data = float(exact["data_mu"][sector])
        exact_syndrome = float(exact["syndrome_mu"][sector])
        data_passed = data_ci_low <= exact_data <= data_ci_high
        syndrome_passed = syndrome_ci_low <= exact_syndrome <= syndrome_ci_high
        trace_unique = sorted(set(int(value) for value in sector_trace))
        trace_passed = trace_unique == [int(sector)]
        records.append({
            "sector": int(sector),
            "initial_signature": int(initial_signature),
            "trace_unique_signatures": trace_unique,
            "trace_num_samples": int(len(sector_trace)),
            "acceptance_rate": (
                0.0 if attempted_total == 0 else accepted_total / attempted_total
            ),
            "sector_reject_rate": (
                0.0 if attempted_total == 0 else sector_rejected_total / attempted_total
            ),
            "data_exact": exact_data,
            "data_mcmc_mean": float(data_mean),
            "data_ci_99": [float(data_ci_low), float(data_ci_high)],
            "data_abs_diff": float(abs(data_mean - exact_data)),
            "data_relative_diff": float(abs(data_mean - exact_data) / max(1.0, abs(exact_data))),
            "data_passed": bool(data_passed),
            "syndrome_exact": exact_syndrome,
            "syndrome_mcmc_mean": float(syndrome_mean),
            "syndrome_ci_99": [float(syndrome_ci_low), float(syndrome_ci_high)],
            "syndrome_abs_diff": float(abs(syndrome_mean - exact_syndrome)),
            "syndrome_relative_diff": float(
                abs(syndrome_mean - exact_syndrome) / max(1.0, abs(exact_syndrome))
            ),
            "syndrome_passed": bool(syndrome_passed),
            "trace_passed": bool(trace_passed),
        })
    return {
        "records": records,
        "trace_rows": trace_rows,
        "invariant_failure_count": int(invariant_failure_count),
    }


def write_trace_csv(trace_rows: list[dict]) -> None:
    with (OUTPUT_DIR / "sector_trace_sample.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sector", "measurement_index", "signature"),
        )
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)


def write_comparison_csv(records: list[dict]) -> None:
    with (OUTPUT_DIR / "sector_mean_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sector",
                "data_exact",
                "data_mcmc_mean",
                "data_ci_low",
                "data_ci_high",
                "data_abs_diff",
                "data_passed",
                "syndrome_exact",
                "syndrome_mcmc_mean",
                "syndrome_ci_low",
                "syndrome_ci_high",
                "syndrome_abs_diff",
                "syndrome_passed",
                "trace_unique_signatures",
                "acceptance_rate",
                "sector_reject_rate",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "sector": record["sector"],
                "data_exact": record["data_exact"],
                "data_mcmc_mean": record["data_mcmc_mean"],
                "data_ci_low": record["data_ci_99"][0],
                "data_ci_high": record["data_ci_99"][1],
                "data_abs_diff": record["data_abs_diff"],
                "data_passed": record["data_passed"],
                "syndrome_exact": record["syndrome_exact"],
                "syndrome_mcmc_mean": record["syndrome_mcmc_mean"],
                "syndrome_ci_low": record["syndrome_ci_99"][0],
                "syndrome_ci_high": record["syndrome_ci_99"][1],
                "syndrome_abs_diff": record["syndrome_abs_diff"],
                "syndrome_passed": record["syndrome_passed"],
                "trace_unique_signatures": " ".join(
                    str(value) for value in record["trace_unique_signatures"]
                ),
                "acceptance_rate": record["acceptance_rate"],
                "sector_reject_rate": record["sector_reject_rate"],
            })


def write_summary(payload: dict) -> None:
    records = payload["records"]
    max_data_abs_diff = max(record["data_abs_diff"] for record in records)
    max_syndrome_abs_diff = max(record["syndrome_abs_diff"] for record in records)
    trace_passed = payload["gates"]["C1"]["passed"]
    means_passed = payload["gates"]["C2"]["passed"]
    lines = [
        "# Stage C sector sampler summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        (
            "Sampler: production decoder-reject fixed-sector sweep from "
            "`src/exp37_sector_ti.py`."
        ),
        (
            f"Exact reference: L=2 full enumeration at p={P_VALUE}, q={Q_VALUE}, "
            "zero eta and zero measurement error."
        ),
        (
            f"MCMC config: burn={NUM_BURN_IN_SWEEPS}, "
            f"measurements={NUM_MEASUREMENTS}, stride={NUM_SWEEPS_BETWEEN_MEASUREMENTS}, "
            f"blocks={BLOCK_COUNT}, bootstrap={NUM_BOOTSTRAP}, "
            f"CI=[{CI_LOW:.3f},{CI_HIGH:.3f}]."
        ),
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| C1 | sector_trace constant for all sectors | "
            f"violations={payload['gates']['C1']['invariant_failure_count']} | "
            f"{'PASS' if trace_passed else 'FAIL'} |"
        ),
        (
            f"| C2 | exact means inside block-bootstrap CI | "
            f"max abs d_data={max_data_abs_diff:.4g}, "
            f"max abs d_synd={max_syndrome_abs_diff:.4g} | "
            f"{'PASS' if means_passed else 'FAIL'} |"
        ),
        "",
        "## Sector Mean Comparison",
        "",
        "| sector | data exact | data MCMC | data 99% CI | syndrome exact | syndrome MCMC | syndrome 99% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['sector']} | {record['data_exact']:.6f} | "
            f"{record['data_mcmc_mean']:.6f} | "
            f"[{record['data_ci_99'][0]:.6f}, {record['data_ci_99'][1]:.6f}] | "
            f"{record['syndrome_exact']:.6f} | "
            f"{record['syndrome_mcmc_mean']:.6f} | "
            f"[{record['syndrome_ci_99'][0]:.6f}, {record['syndrome_ci_99'][1]:.6f}] |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `stageC_results.json`",
        "- `sector_mean_comparison.csv`",
        "- `sector_trace_sample.csv`",
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
    counts = build_exact_count_table(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
    )
    exact = exact_sector_moments(
        counts=counts,
        p_value=P_VALUE,
        q_value=Q_VALUE,
    )
    sampler_result = run_sampler(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
        exact=exact,
    )
    records = sampler_result["records"]
    write_trace_csv(sampler_result["trace_rows"])
    write_comparison_csv(records)

    c1_passed = (
        sampler_result["invariant_failure_count"] == 0
        and all(record["trace_passed"] for record in records)
    )
    c2_passed = all(
        record["data_passed"] and record["syndrome_passed"]
        for record in records
    )
    payload = {
        "stage": "C",
        "overall_passed": bool(c1_passed and c2_passed),
        "code_family": "3d_toric",
        "lattice_size": LATTICE_SIZE,
        "projection_mode": "decoder_reject",
        "sector_observable": "corrected_c_eta_section",
        "section_prefer_bplsd": False,
        "p_value": P_VALUE,
        "q_value": Q_VALUE,
        "kp_value": float(_compute_k(P_VALUE)),
        "kq_value": float(_compute_k(Q_VALUE)),
        "disorder": {
            "eta_weight": 0,
            "measurement_error_weight": 0,
        },
        "mcmc_config": {
            "num_burn_in_sweeps": NUM_BURN_IN_SWEEPS,
            "num_measurements": NUM_MEASUREMENTS,
            "num_sweeps_between_measurements": NUM_SWEEPS_BETWEEN_MEASUREMENTS,
            "block_count": BLOCK_COUNT,
            "num_bootstrap": NUM_BOOTSTRAP,
            "bootstrap_ci": [CI_LOW, CI_HIGH],
            "seed": SEED,
        },
        "gates": {
            "C1": {
                "passed": bool(c1_passed),
                "invariant_failure_count": sampler_result[
                    "invariant_failure_count"
                ],
                "trace_unique_signatures": [
                    record["trace_unique_signatures"]
                    for record in records
                ],
            },
            "C2": {
                "passed": bool(c2_passed),
                "max_data_abs_diff": float(
                    max(record["data_abs_diff"] for record in records)
                ),
                "max_syndrome_abs_diff": float(
                    max(record["syndrome_abs_diff"] for record in records)
                ),
            },
        },
        "records": records,
    }
    (OUTPUT_DIR / "stageC_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(payload)
    print(json.dumps({
        "overall_passed": payload["overall_passed"],
        "C1_invariant_failures": payload["gates"]["C1"][
            "invariant_failure_count"
        ],
        "C2_max_data_abs_diff": payload["gates"]["C2"]["max_data_abs_diff"],
        "C2_max_syndrome_abs_diff": payload["gates"]["C2"][
            "max_syndrome_abs_diff"
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
