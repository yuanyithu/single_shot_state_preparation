#!/usr/bin/env python3
"""Stage D sector-resolved TI benchmark against Stage B exact references.

This script uses the Stage C decoder-reject fixed-sector chain as the TI
engine.  It does not use AIS, FEP, or flip-reweighting.
"""

from __future__ import annotations

import csv
import argparse
import json
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
from exp37_sector_ti import (  # noqa: E402
    _bootstrap_ti,
    _build_decoder_reject_proposals,
    _build_logical_projection_masks,
    _build_sector_representatives,
    _build_sector_preserving_proposals,
    _coarse_indices_from_fine_grid,
    _compute_k,
    _integrate_mu,
    _q_top_from_weights,
    _run_fixed_sector_chain_decoder_reject,
    _run_fixed_sector_chain,
    _weights_from_delta_f,
)
from linear_section import apply_section, build_syndrome_representative_section  # noqa: E402


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

LATTICE_SIZE = 2
NUM_SECTORS = 8
NUM_KP_GRID_POINTS = 65
NUM_BURN_IN_SWEEPS = 80
NUM_MEASUREMENTS = 768
NUM_SWEEPS_BETWEEN_MEASUREMENTS = 2
BLOCK_COUNT = 24
NUM_BOOTSTRAP = 800
SEED_BASE = 3703601

TV_THRESHOLD = 0.02
QTOP_THRESHOLD = 0.02
GRID_TV_THRESHOLD = 0.02
GRID_QTOP_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage D sector-resolved TI against Stage B exact references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for Stage D artifacts.",
    )
    parser.add_argument(
        "--record-ids",
        default="",
        help="Comma-separated Stage B record IDs to run; default runs all.",
    )
    parser.add_argument("--num-kp-grid-points", type=int, default=NUM_KP_GRID_POINTS)
    parser.add_argument("--num-burn-in-sweeps", type=int, default=NUM_BURN_IN_SWEEPS)
    parser.add_argument("--num-measurements", type=int, default=NUM_MEASUREMENTS)
    parser.add_argument(
        "--num-sweeps-between-measurements",
        type=int,
        default=NUM_SWEEPS_BETWEEN_MEASUREMENTS,
    )
    parser.add_argument("--block-count", type=int, default=BLOCK_COUNT)
    parser.add_argument("--num-bootstrap", type=int, default=NUM_BOOTSTRAP)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument(
        "--sampler-mode",
        choices=("linear_kernel", "decoder_reject"),
        default="linear_kernel",
        help=(
            "linear_kernel uses P_L-preserving proposals. For the Stage B "
            "zero-disorder linear-section benchmark it is equivalent to the "
            "decoder-reject sector labels and is much faster."
        ),
    )
    return parser.parse_args()


def apply_runtime_config(args: argparse.Namespace) -> set[int] | None:
    global OUTPUT_DIR
    global NUM_KP_GRID_POINTS
    global NUM_BURN_IN_SWEEPS
    global NUM_MEASUREMENTS
    global NUM_SWEEPS_BETWEEN_MEASUREMENTS
    global BLOCK_COUNT
    global NUM_BOOTSTRAP
    global SEED_BASE

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NUM_KP_GRID_POINTS = int(args.num_kp_grid_points)
    NUM_BURN_IN_SWEEPS = int(args.num_burn_in_sweeps)
    NUM_MEASUREMENTS = int(args.num_measurements)
    NUM_SWEEPS_BETWEEN_MEASUREMENTS = int(args.num_sweeps_between_measurements)
    BLOCK_COUNT = int(args.block_count)
    NUM_BOOTSTRAP = int(args.num_bootstrap)
    SEED_BASE = int(args.seed_base)

    if not str(args.record_ids).strip():
        return None
    return {
        int(part)
        for part in str(args.record_ids).split(",")
        if part.strip()
    }


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def load_reference_records() -> list[dict]:
    payload = json.loads(STAGE_B_REFERENCE.read_text(encoding="utf-8"))
    records = payload["records"]
    if payload.get("lattice_size") != LATTICE_SIZE:
        raise ValueError("Stage B reference lattice size mismatch")
    if payload.get("disorder", {}).get("eta_weight") != 0:
        raise ValueError("Stage D script expects zero eta reference")
    if payload.get("disorder", {}).get("measurement_error_weight") != 0:
        raise ValueError("Stage D script expects zero measurement-error reference")
    return records


def run_ti_for_record(
    reference: dict,
    parity_check_matrix: np.ndarray,
    primitive_logical_masks: np.ndarray,
    section_data,
    logical_projection_masks: np.ndarray,
    proposals: dict,
    sector_representatives: np.ndarray,
    sampler_mode: str,
    rng_master: np.random.Generator,
) -> dict:
    p_value = float(reference["p_value"])
    q_value = float(reference["q_value"])
    kp_target = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    kp_grid = np.linspace(0.0, kp_target, NUM_KP_GRID_POINTS, dtype=np.float64)

    num_checks = parity_check_matrix.shape[0]
    measurement_error_bits = np.zeros(num_checks, dtype=bool)
    disorder_syndrome_bits = np.zeros(num_checks, dtype=bool)
    disorder_syndrome_representative_bits = None
    if sampler_mode == "decoder_reject":
        disorder_syndrome_representative_bits = apply_section(
            disorder_syndrome_bits,
            section_data,
        )

    num_grid = kp_grid.shape[0]
    block_count = min(BLOCK_COUNT, NUM_MEASUREMENTS)
    mu_by_sector = np.empty((NUM_SECTORS, num_grid), dtype=np.float64)
    syndrome_mu_by_sector = np.empty((NUM_SECTORS, num_grid), dtype=np.float64)
    block_mu_by_sector = np.empty(
        (NUM_SECTORS, num_grid, block_count),
        dtype=np.float64,
    )
    acceptance_by_sector = np.empty((NUM_SECTORS, num_grid), dtype=np.float64)
    sector_reject_by_sector = np.empty((NUM_SECTORS, num_grid), dtype=np.float64)

    started_at = time.perf_counter()
    for sector in range(NUM_SECTORS):
        sector_rng = np.random.default_rng(
            int(rng_master.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        )
        if sampler_mode == "decoder_reject":
            chain_result = _run_fixed_sector_chain_decoder_reject(
                parity_check_matrix=parity_check_matrix,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                sector_representative_bits=sector_representatives[sector],
                target_sector=sector,
                measurement_error_bits=measurement_error_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                proposals=proposals,
                kp_grid=kp_grid,
                kq_value=kq_value,
                num_burn_in_sweeps=NUM_BURN_IN_SWEEPS,
                num_measurements=NUM_MEASUREMENTS,
                num_sweeps_between_measurements=NUM_SWEEPS_BETWEEN_MEASUREMENTS,
                block_count=block_count,
                rng=sector_rng,
                debug_checks=True,
            )
        elif sampler_mode == "linear_kernel":
            chain_result = _run_fixed_sector_chain(
                parity_check_matrix=parity_check_matrix,
                logical_projection_masks=logical_projection_masks,
                sector_representative_bits=sector_representatives[sector],
                target_sector=sector,
                measurement_error_bits=measurement_error_bits,
                proposals=proposals,
                winding_groups=[],
                kp_grid=kp_grid,
                kq_value=kq_value,
                num_burn_in_sweeps=NUM_BURN_IN_SWEEPS,
                num_measurements=NUM_MEASUREMENTS,
                num_sweeps_between_measurements=NUM_SWEEPS_BETWEEN_MEASUREMENTS,
                block_count=block_count,
                winding_heatbath_sweeps=0,
                rng=sector_rng,
                use_numba=False,
                debug_checks=True,
            )
        else:
            raise ValueError(f"unsupported sampler_mode={sampler_mode!r}")
        mu_by_sector[sector] = chain_result["mu"]
        syndrome_mu_by_sector[sector] = chain_result["syndrome_mu"]
        block_mu_by_sector[sector] = chain_result["block_mu"]
        acceptance_by_sector[sector] = chain_result["acceptance_rate"]
        sector_reject_by_sector[sector] = chain_result[
            "winding_heatbath_change_rate"
        ]

    integrals = _integrate_mu(kp_grid, mu_by_sector)
    delta_f = integrals - integrals[0]
    weights = _weights_from_delta_f(delta_f)
    q_top = _q_top_from_weights(weights)

    coarse_indices = _coarse_indices_from_fine_grid(num_grid)
    coarse_integrals = _integrate_mu(
        kp_grid[coarse_indices],
        mu_by_sector[:, coarse_indices],
    )
    coarse_delta_f = coarse_integrals - coarse_integrals[0]
    coarse_weights = _weights_from_delta_f(coarse_delta_f)
    coarse_q_top = _q_top_from_weights(coarse_weights)
    grid_tv = total_variation(weights, coarse_weights)
    grid_q_top_abs_diff = abs(float(q_top - coarse_q_top))

    bootstrap_rng = np.random.default_rng(
        int(rng_master.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    )
    bootstrap = _bootstrap_ti(
        kp_grid=kp_grid,
        block_mu_by_sector=block_mu_by_sector,
        num_bootstrap=NUM_BOOTSTRAP,
        rng=bootstrap_rng,
    )

    exact_weights = np.asarray(reference["weights"], dtype=np.float64)
    exact_q_top = float(reference["q_top"])
    tv = total_variation(weights, exact_weights)
    q_top_abs_diff = abs(float(q_top - exact_q_top))
    q_top_ci95 = np.asarray(bootstrap["q_top_ci95"], dtype=np.float64)
    ci_covers_exact = bool(q_top_ci95[0] <= exact_q_top <= q_top_ci95[1])

    return {
        "record_id": int(reference["record_id"]),
        "p_value": p_value,
        "q_value": q_value,
        "kp_target": float(kp_target),
        "kq_value": float(kq_value),
        "kp_grid": kp_grid,
        "exact_weights": exact_weights,
        "exact_delta_f": np.asarray(reference["delta_f"], dtype=np.float64),
        "exact_q_top": exact_q_top,
        "mu_by_sector": mu_by_sector,
        "syndrome_mu_by_sector": syndrome_mu_by_sector,
        "integrals": integrals,
        "delta_f": delta_f,
        "weights": weights,
        "q_top": float(q_top),
        "q_top_ci95": q_top_ci95,
        "q_top_stderr": float(bootstrap["q_top_stderr"]),
        "weights_stderr": np.asarray(bootstrap["weights_stderr"], dtype=np.float64),
        "delta_f_stderr": np.asarray(
            bootstrap["delta_f_stderr"],
            dtype=np.float64,
        ),
        "coarse_delta_f": coarse_delta_f,
        "coarse_weights": coarse_weights,
        "coarse_q_top": float(coarse_q_top),
        "tv": float(tv),
        "q_top_abs_diff": float(q_top_abs_diff),
        "ci_covers_exact": ci_covers_exact,
        "grid_tv": float(grid_tv),
        "grid_q_top_abs_diff": float(grid_q_top_abs_diff),
        "d1_passed": bool(tv <= TV_THRESHOLD),
        "d2_passed": bool(q_top_abs_diff <= QTOP_THRESHOLD and ci_covers_exact),
        "d3_passed": bool(
            grid_tv <= GRID_TV_THRESHOLD
            and grid_q_top_abs_diff <= GRID_QTOP_THRESHOLD
        ),
        "acceptance_mean": float(np.mean(acceptance_by_sector)),
        "sector_reject_mean": float(np.mean(sector_reject_by_sector)),
        "wall_time_seconds": float(time.perf_counter() - started_at),
    }


def json_ready_record(record: dict) -> dict:
    array_keys = {
        "kp_grid",
        "exact_weights",
        "exact_delta_f",
        "mu_by_sector",
        "syndrome_mu_by_sector",
        "integrals",
        "delta_f",
        "weights",
        "q_top_ci95",
        "weights_stderr",
        "delta_f_stderr",
        "coarse_delta_f",
        "coarse_weights",
    }
    result = {}
    for key, value in record.items():
        if key in array_keys:
            result[key] = np.asarray(value).tolist()
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def write_comparison_csv(records: list[dict]) -> None:
    with (OUTPUT_DIR / "ti_comparison.csv").open(
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
                "ti_q_top",
                "q_top_ci_low",
                "q_top_ci_high",
                "q_top_abs_diff",
                "tv",
                "grid_tv",
                "grid_q_top_abs_diff",
                "ci_covers_exact",
                "d1_passed",
                "d2_passed",
                "d3_passed",
                "acceptance_mean",
                "sector_reject_mean",
                "wall_time_seconds",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record["record_id"],
                "p_value": record["p_value"],
                "q_value": record["q_value"],
                "exact_q_top": record["exact_q_top"],
                "ti_q_top": record["q_top"],
                "q_top_ci_low": record["q_top_ci95"][0],
                "q_top_ci_high": record["q_top_ci95"][1],
                "q_top_abs_diff": record["q_top_abs_diff"],
                "tv": record["tv"],
                "grid_tv": record["grid_tv"],
                "grid_q_top_abs_diff": record["grid_q_top_abs_diff"],
                "ci_covers_exact": record["ci_covers_exact"],
                "d1_passed": record["d1_passed"],
                "d2_passed": record["d2_passed"],
                "d3_passed": record["d3_passed"],
                "acceptance_mean": record["acceptance_mean"],
                "sector_reject_mean": record["sector_reject_mean"],
                "wall_time_seconds": record["wall_time_seconds"],
            })


def write_npz(records: list[dict]) -> None:
    np.savez_compressed(
        OUTPUT_DIR / "ti_results.npz",
        record_id=np.asarray([record["record_id"] for record in records], dtype=np.int64),
        p_value=np.asarray([record["p_value"] for record in records], dtype=np.float64),
        q_value=np.asarray([record["q_value"] for record in records], dtype=np.float64),
        exact_weights=np.stack([record["exact_weights"] for record in records], axis=0),
        ti_weights=np.stack([record["weights"] for record in records], axis=0),
        coarse_weights=np.stack([record["coarse_weights"] for record in records], axis=0),
        exact_q_top=np.asarray([record["exact_q_top"] for record in records], dtype=np.float64),
        ti_q_top=np.asarray([record["q_top"] for record in records], dtype=np.float64),
        q_top_ci95=np.stack([record["q_top_ci95"] for record in records], axis=0),
        tv=np.asarray([record["tv"] for record in records], dtype=np.float64),
        q_top_abs_diff=np.asarray(
            [record["q_top_abs_diff"] for record in records],
            dtype=np.float64,
        ),
        grid_tv=np.asarray([record["grid_tv"] for record in records], dtype=np.float64),
        grid_q_top_abs_diff=np.asarray(
            [record["grid_q_top_abs_diff"] for record in records],
            dtype=np.float64,
        ),
    )


def write_summary(payload: dict) -> None:
    records = payload["records"]
    gates = payload["gates"]
    lines = [
        "# Stage D sector-resolved TI summary",
        "",
        f"Overall: {'PASS' if payload['overall_passed'] else 'FAIL'}",
        "",
        "Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.",
        "Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.",
        (
            f"TI config: grid={NUM_KP_GRID_POINTS}, burn={NUM_BURN_IN_SWEEPS}, "
            f"measurements={NUM_MEASUREMENTS}, stride={NUM_SWEEPS_BETWEEN_MEASUREMENTS}, "
            f"blocks={BLOCK_COUNT}, bootstrap={NUM_BOOTSTRAP}."
        ),
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| D1 | TV(w_TI,w_exact) <= {TV_THRESHOLD:.3f} | "
            f"max TV={gates['D1']['max_tv']:.4g} | "
            f"{'PASS' if gates['D1']['passed'] else 'FAIL'} |"
        ),
        (
            f"| D2 | abs dq_top <= {QTOP_THRESHOLD:.3f} and CI covers exact | "
            f"max abs dq={gates['D2']['max_q_top_abs_diff']:.4g}, "
            f"CI misses={gates['D2']['num_ci_misses']} | "
            f"{'PASS' if gates['D2']['passed'] else 'FAIL'} |"
        ),
        (
            f"| D3 | coarse/fine grid TV and abs dq <= {GRID_TV_THRESHOLD:.3f} | "
            f"max grid TV={gates['D3']['max_grid_tv']:.4g}, "
            f"max grid dq={gates['D3']['max_grid_q_top_abs_diff']:.4g} | "
            f"{'PASS' if gates['D3']['passed'] else 'FAIL'} |"
        ),
        "",
        "## Point Comparison",
        "",
        "| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        gate_text = "".join(
            (
                "D1" if record["d1_passed"] else "d1FAIL",
                "/",
                "D2" if record["d2_passed"] else "d2FAIL",
                "/",
                "D3" if record["d3_passed"] else "d3FAIL",
            )
        )
        lines.append(
            f"| {record['record_id']} | {record['p_value']:.6f} | "
            f"{record['q_value']:.6f} | {record['exact_q_top']:.6f} | "
            f"{record['q_top']:.6f} | "
            f"[{record['q_top_ci95'][0]:.6f}, {record['q_top_ci95'][1]:.6f}] | "
            f"{record['tv']:.5f} | {record['grid_tv']:.5f} | "
            f"{record['grid_q_top_abs_diff']:.5f} | {gate_text} |"
        )
    lines.extend([
        "",
        "Artifacts:",
        "- `stageD_results.json`",
        "- `ti_results.npz`",
        "- `ti_comparison.csv`",
    ])
    (OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    selected_record_ids = apply_runtime_config(args)
    sampler_mode = str(args.sampler_mode)
    started_at = time.perf_counter()
    reference_records = load_reference_records()
    if selected_record_ids is not None:
        reference_records = [
            record for record in reference_records
            if int(record["record_id"]) in selected_record_ids
        ]
        if not reference_records:
            raise ValueError("selected --record-ids did not match any reference record")
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    section_data = build_syndrome_representative_section(
        parity_check_matrix,
        prefer_bplsd=False,
    )
    logical_projection_masks = _build_logical_projection_masks(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        "3d_toric",
        LATTICE_SIZE,
    )
    if sampler_mode == "decoder_reject":
        proposals = _build_decoder_reject_proposals(
            parity_check_matrix=parity_check_matrix,
            zero_syndrome_move_data=zero_syndrome_move_data,
        )
        representative_masks = primitive_logical_masks
    else:
        proposals = _build_sector_preserving_proposals(
            parity_check_matrix=parity_check_matrix,
            logical_projection_masks=logical_projection_masks,
            zero_syndrome_move_data=zero_syndrome_move_data,
        )
        representative_masks = logical_projection_masks
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=representative_masks,
        parity_check_matrix=parity_check_matrix,
    )

    rng_master = np.random.default_rng(SEED_BASE)
    records = []
    for reference in reference_records:
        records.append(
            run_ti_for_record(
                reference=reference,
                parity_check_matrix=parity_check_matrix,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                logical_projection_masks=logical_projection_masks,
                proposals=proposals,
                sector_representatives=sector_representatives,
                sampler_mode=sampler_mode,
                rng_master=rng_master,
            )
        )

    d1_passed = all(record["d1_passed"] for record in records)
    d2_passed = all(record["d2_passed"] for record in records)
    d3_passed = all(record["d3_passed"] for record in records)
    payload = {
        "stage": "D",
        "overall_passed": bool(d1_passed and d2_passed and d3_passed),
        "code_family": "3d_toric",
        "lattice_size": LATTICE_SIZE,
        "projection_mode": "decoder_reject",
        "sector_observable": "corrected_c_eta_section",
        "sampler_mode": sampler_mode,
        "section_prefer_bplsd": False,
        "reference_path": str(STAGE_B_REFERENCE.relative_to(PROJECT_ROOT)),
        "disorder": {
            "eta_weight": 0,
            "measurement_error_weight": 0,
        },
        "ti_config": {
            "num_kp_grid_points": NUM_KP_GRID_POINTS,
            "num_burn_in_sweeps": NUM_BURN_IN_SWEEPS,
            "num_measurements": NUM_MEASUREMENTS,
            "num_sweeps_between_measurements": NUM_SWEEPS_BETWEEN_MEASUREMENTS,
            "block_count": BLOCK_COUNT,
            "num_bootstrap": NUM_BOOTSTRAP,
            "seed_base": SEED_BASE,
        },
        "thresholds": {
            "tv": TV_THRESHOLD,
            "q_top_abs_diff": QTOP_THRESHOLD,
            "grid_tv": GRID_TV_THRESHOLD,
            "grid_q_top_abs_diff": GRID_QTOP_THRESHOLD,
        },
        "gates": {
            "D1": {
                "passed": bool(d1_passed),
                "max_tv": float(max(record["tv"] for record in records)),
            },
            "D2": {
                "passed": bool(d2_passed),
                "max_q_top_abs_diff": float(
                    max(record["q_top_abs_diff"] for record in records)
                ),
                "num_ci_misses": int(
                    sum(not record["ci_covers_exact"] for record in records)
                ),
            },
            "D3": {
                "passed": bool(d3_passed),
                "max_grid_tv": float(
                    max(record["grid_tv"] for record in records)
                ),
                "max_grid_q_top_abs_diff": float(
                    max(record["grid_q_top_abs_diff"] for record in records)
                ),
            },
        },
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "records": [json_ready_record(record) for record in records],
    }

    write_comparison_csv(records)
    write_npz(records)
    (OUTPUT_DIR / "stageD_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(payload)

    print(json.dumps({
        "overall_passed": payload["overall_passed"],
        "D1_max_tv": payload["gates"]["D1"]["max_tv"],
        "D2_max_q_top_abs_diff": payload["gates"]["D2"][
            "max_q_top_abs_diff"
        ],
        "D2_num_ci_misses": payload["gates"]["D2"]["num_ci_misses"],
        "D3_max_grid_tv": payload["gates"]["D3"]["max_grid_tv"],
        "D3_max_grid_q_top_abs_diff": payload["gates"]["D3"][
            "max_grid_q_top_abs_diff"
        ],
        "wall_time_seconds": payload["wall_time_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
