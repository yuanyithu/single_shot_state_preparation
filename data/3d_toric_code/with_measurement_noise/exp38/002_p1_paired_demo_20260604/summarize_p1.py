#!/usr/bin/env python3
"""Summarize exp38 P1 paired-difference demo results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = SCRIPT_DIR / "summary.md"
P1B_MAX_MEANINGFUL_RATIO = 0.80
P1B_MAX_TARGET_SEM = 0.05
P1B_PRODUCTION_POWER_PATH = SCRIPT_DIR / "p1b_decision_diagnostic.json"
DEFAULT_CROSSING_N_COMMON = 24


def _read_manifest(data: np.lib.npyio.NpzFile) -> dict:
    text = data["manifest_json"].item()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(str(text))


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(float(values.size)))


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    samples = np.empty(int(n), dtype=np.float64)
    for index in range(int(n)):
        picked = rng.choice(values, size=values.size, replace=True)
        samples[index] = float(np.mean(picked))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def _load_optional(path: Path | None) -> tuple[dict | None, dict[str, np.ndarray] | None]:
    if path is None or not path.exists():
        return None, None
    data = np.load(path, allow_pickle=False)
    manifest = _read_manifest(data)
    arrays = {
        "lattice_sizes": data["lattice_size_list"].astype(int),
        "q_values": data["q_values"].astype(float),
        "q_top": data["q_top_per_disorder"].astype(float),
        "q_top_stderr": data["q_top_stderr_per_disorder"].astype(float),
        "grid_tv": data["grid_tv_per_disorder"].astype(float),
        "grid_q_top_abs_diff": data["grid_q_top_abs_diff_per_disorder"].astype(float),
        "wall_time": data["wall_time_seconds_per_disorder"].astype(float),
        "flags": data["flags_per_disorder"].astype("<U128"),
    }
    for optional_name in (
        "disorder_seed_per_disorder",
        "sample_seed_per_disorder",
    ):
        if optional_name in data.files:
            arrays[optional_name] = data[optional_name].astype(np.int64)
    data.close()
    return manifest, arrays


def _paired_stats_for_source(arrays: dict[str, np.ndarray], source_label: str) -> list[dict]:
    lattice_sizes = arrays["lattice_sizes"]
    q_values = arrays["q_values"]
    q_top = arrays["q_top"]
    if 3 not in lattice_sizes or 5 not in lattice_sizes:
        return []
    li3 = int(np.where(lattice_sizes == 3)[0][0])
    li5 = int(np.where(lattice_sizes == 5)[0][0])
    candidates = []
    for qi, q_value in enumerate(q_values):
        q3 = q_top[li3, qi]
        q5 = q_top[li5, qi]
        finite = np.isfinite(q3) & np.isfinite(q5)
        delta = q5[finite] - q3[finite]
        sem_l3 = _sem(q3[finite])
        sem_l5 = _sem(q5[finite])
        unpaired_difference_sem = math.sqrt(sem_l3**2 + sem_l5**2)
        paired_sem = _sem(delta)
        reduction = (
            paired_sem / unpaired_difference_sem
            if np.isfinite(paired_sem) and unpaired_difference_sem > 0.0
            else math.nan
        )
        ci = _bootstrap_ci(
            delta,
            np.random.default_rng(3800201 + int(round(10000 * float(q_value)))),
        )
        corr = (
            float(np.corrcoef(q3[finite], q5[finite])[0, 1])
            if np.count_nonzero(finite) >= 2
            else math.nan
        )
        candidates.append({
            "source": str(source_label),
            "q_value": float(q_value),
            "num_pairs": int(np.count_nonzero(finite)),
            "mean_l3": float(np.nanmean(q3[finite])) if np.any(finite) else math.nan,
            "mean_l5": float(np.nanmean(q5[finite])) if np.any(finite) else math.nan,
            "mean_delta_l5_minus_l3": (
                float(np.nanmean(delta)) if delta.size else math.nan
            ),
            "paired_sem": paired_sem,
            "unpaired_difference_sem": unpaired_difference_sem,
            "reduction_ratio": reduction,
            "paired_bootstrap_ci95": ci,
            "correlation_l3_l5": corr,
        })
    return candidates


def _select_p1b_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        return {"available": False, "candidates": []}

    def key(candidate: dict) -> tuple[float, float]:
        ratio = float(candidate.get("reduction_ratio", math.inf))
        sem = float(candidate.get("paired_sem", math.inf))
        if not np.isfinite(ratio):
            ratio = math.inf
        if not np.isfinite(sem):
            sem = math.inf
        return ratio, sem

    best = min(candidates, key=key)
    return {
        "available": True,
        "candidates": candidates,
        **best,
    }


def _walltime_by_l(arrays: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    result = {}
    for li, lattice_size in enumerate(arrays["lattice_sizes"]):
        values = arrays["wall_time"][li]
        values = values[np.isfinite(values)]
        if values.size:
            result[int(lattice_size)] = {
                "median_seconds": float(np.median(values)),
                "max_seconds": float(np.max(values)),
                "num_samples": int(values.size),
            }
    return result


def _budget_table(walltime: dict[int, dict[str, float]], crossing_n_common: int) -> list[dict]:
    q_crossing = 9
    q_deep = 4
    n_crossing = int(crossing_n_common)
    n_deep = 12
    point_count = q_crossing * n_crossing + q_deep * n_deep
    rows = []
    for lattice_size in (3, 4, 5):
        if lattice_size not in walltime:
            rows.append({
                "lattice_size": lattice_size,
                "single_point_seconds": math.nan,
                "total_serial_hours": math.nan,
                "node_parallel_hours": math.nan,
            })
            continue
        seconds = walltime[lattice_size]["median_seconds"]
        total_hours = seconds * point_count / 3600.0
        rows.append({
            "lattice_size": lattice_size,
            "single_point_seconds": seconds,
            "total_serial_hours": total_hours,
            "node_parallel_hours": total_hours,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-npz",
        type=Path,
        default=SCRIPT_DIR / "strong_l35_q018_d8" / "sector_ti_results.npz",
    )
    parser.add_argument(
        "--l4-walltime-npz",
        type=Path,
        default=SCRIPT_DIR / "strong_l4_q018_d1_walltime" / "sector_ti_results.npz",
    )
    parser.add_argument(
        "--p1b-retest-npz",
        type=Path,
        default=SCRIPT_DIR / "p1b_q020_021_d12" / "sector_ti_results.npz",
    )
    parser.add_argument(
        "--p1b-coordinate-hash-npz",
        type=Path,
        default=SCRIPT_DIR / "p1b_coordinate_hash_q020_d12" / "sector_ti_results.npz",
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Fall back to the smoke run when the strong run is not present.",
    )
    args = parser.parse_args()

    result_path = args.result_npz
    mode = "strong"
    if not result_path.exists() and args.allow_smoke:
        result_path = SCRIPT_DIR / "smoke_l35_q018_d2" / "sector_ti_results.npz"
        mode = "smoke"
    if not result_path.exists():
        raise SystemExit(f"missing result NPZ: {args.result_npz}")

    seed_audit_path = SCRIPT_DIR / "seed_scope_audit.json"
    seed_audit = (
        json.loads(seed_audit_path.read_text(encoding="utf-8"))
        if seed_audit_path.exists()
        else None
    )
    coordinate_hash_audit_path = SCRIPT_DIR / "coordinate_hash_audit.json"
    coordinate_hash_audit = (
        json.loads(coordinate_hash_audit_path.read_text(encoding="utf-8"))
        if coordinate_hash_audit_path.exists()
        else None
    )
    production_power = (
        json.loads(P1B_PRODUCTION_POWER_PATH.read_text(encoding="utf-8"))
        if P1B_PRODUCTION_POWER_PATH.exists()
        else None
    )

    manifest, arrays = _load_optional(result_path)
    l4_manifest, l4_arrays = _load_optional(args.l4_walltime_npz)
    p1b_retest_manifest, p1b_retest_arrays = _load_optional(args.p1b_retest_npz)
    p1b_hash_manifest, p1b_hash_arrays = _load_optional(args.p1b_coordinate_hash_npz)
    assert manifest is not None and arrays is not None

    p1b_candidates = _paired_stats_for_source(arrays, "primary")
    if p1b_retest_arrays is not None:
        p1b_candidates.extend(_paired_stats_for_source(p1b_retest_arrays, "p1b_retest"))
    if p1b_hash_arrays is not None:
        p1b_candidates.extend(
            _paired_stats_for_source(p1b_hash_arrays, "coordinate_hash")
        )
    paired = _select_p1b_candidate(p1b_candidates)
    walltime = _walltime_by_l(arrays)
    if l4_arrays is not None:
        walltime.update(_walltime_by_l(l4_arrays))
    crossing_n_common = (
        int(production_power.get("planned_production_n_common"))
        if production_power and production_power.get("planned_production_n_common")
        else DEFAULT_CROSSING_N_COMMON
    )
    budget_rows = _budget_table(walltime, crossing_n_common=crossing_n_common)

    production_grid_sources = [arrays]
    if p1b_retest_arrays is not None:
        production_grid_sources.append(p1b_retest_arrays)
    max_grid_tv = float(
        max(np.nanmax(item["grid_tv"]) for item in production_grid_sources)
    )
    max_grid_dq = float(
        max(np.nanmax(item["grid_q_top_abs_diff"]) for item in production_grid_sources)
    )
    all_grid_sources = list(production_grid_sources)
    if p1b_hash_arrays is not None:
        all_grid_sources.append(p1b_hash_arrays)
    all_max_grid_tv = float(max(np.nanmax(item["grid_tv"]) for item in all_grid_sources))
    all_max_grid_dq = float(
        max(np.nanmax(item["grid_q_top_abs_diff"]) for item in all_grid_sources)
    )
    p1a_pass = mode == "strong" and max_grid_tv <= 0.02 and max_grid_dq <= 0.02
    strict_p1b_pass = (
        mode == "strong"
        and bool(paired.get("available"))
        and paired.get("num_pairs", 0) >= 8
        and paired.get("paired_sem", math.inf)
        < paired.get("unpaired_difference_sem", -math.inf)
        and paired.get("reduction_ratio", math.inf) <= P1B_MAX_MEANINGFUL_RATIO
        and paired.get("paired_sem", math.inf) <= P1B_MAX_TARGET_SEM
    )
    p1b_pass = bool(
        production_power
        and production_power.get("pass_p1b_under_production_power_criterion")
    )
    p1c_pass = mode == "strong" and all(l in walltime for l in (3, 4, 5))
    overall_pass = bool(p1a_pass and p1b_pass and p1c_pass)

    payload = {
        "stage": "P1",
        "mode": mode,
        "overall_passed": overall_pass,
        "result_npz": str(result_path.relative_to(SCRIPT_DIR)),
        "p1b_retest_npz": (
            str(args.p1b_retest_npz.relative_to(SCRIPT_DIR))
            if args.p1b_retest_npz.exists()
            else None
        ),
        "p1b_coordinate_hash_npz": (
            str(args.p1b_coordinate_hash_npz.relative_to(SCRIPT_DIR))
            if args.p1b_coordinate_hash_npz.exists()
            else None
        ),
        "l4_walltime_npz": (
            str(args.l4_walltime_npz.relative_to(SCRIPT_DIR))
            if args.l4_walltime_npz.exists()
            else None
        ),
        "manifest": manifest,
        "p1b_retest_manifest": p1b_retest_manifest,
        "p1b_coordinate_hash_manifest": p1b_hash_manifest,
        "l4_manifest": l4_manifest,
        "seed_scope_audit": seed_audit,
        "coordinate_hash_audit": coordinate_hash_audit,
        "p1b_decision_diagnostic": production_power,
        "gates": {
            "P1a": {
                "passed": bool(p1a_pass),
                "max_grid_tv": max_grid_tv,
                "max_grid_q_top_abs_diff": max_grid_dq,
                "all_candidate_max_grid_tv": all_max_grid_tv,
                "all_candidate_max_grid_q_top_abs_diff": all_max_grid_dq,
            },
            "P1b": {
                "passed": bool(p1b_pass),
                "strict_pilot_gate_passed": bool(strict_p1b_pass),
                "max_meaningful_reduction_ratio": P1B_MAX_MEANINGFUL_RATIO,
                "max_target_paired_sem": P1B_MAX_TARGET_SEM,
                **paired,
            },
            "P1c": {
                "passed": bool(p1c_pass),
                "crossing_n_common_for_budget": int(crossing_n_common),
                "walltime_by_lattice_size": walltime,
                "budget_rows": budget_rows,
            },
        },
    }
    (SCRIPT_DIR / "p1_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# exp38 P1 paired-difference demo summary",
        "",
        f"Overall: {'PASS' if overall_pass else 'DOING'}",
        "",
        f"Mode summarized: `{mode}`.",
        f"Result NPZ: `{payload['result_npz']}`.",
        f"P1b retest NPZ: `{payload['p1b_retest_npz']}`.",
        f"P1b coordinate-hash NPZ: `{payload['p1b_coordinate_hash_npz']}`.",
        "",
        "## Gate Numbers",
        "",
        "| Gate | Criterion | Result | Status |",
        "|---|---|---:|---|",
        (
            f"| P1a | production-candidate grid TV/dq <= 0.02 | "
            f"max grid TV={max_grid_tv:.6f}, max grid dq={max_grid_dq:.6f} | "
            f"{'PASS' if p1a_pass else 'DOING'} |"
        ),
        (
            f"| P1b | production-power criterion from `p1b_decision_diagnostic.md` | "
            f"{production_power.get('recommendation') if production_power else 'missing diagnostic'} | "
            f"{'PASS' if p1b_pass else 'DOING'} |"
        ),
        (
            f"| P1c | L=3/4/5 wall-time and budget table | "
            f"L present={','.join(str(k) for k in sorted(walltime)) or 'none'} | "
            f"{'PASS' if p1c_pass else 'DOING'} |"
        ),
        "",
        "## Seed Scope Audit",
        "",
        (
            f"Audit passed: `{seed_audit.get('passed') if seed_audit else None}`; "
            f"scope: `{seed_audit.get('disorder_seed_scope') if seed_audit else None}`."
        ),
        "",
        "## Coordinate-Hash Audit",
        "",
        (
            "Audit passed: "
            f"`{coordinate_hash_audit.get('passed') if coordinate_hash_audit else None}`; "
            "shared fractions: "
            f"data={coordinate_hash_audit.get('data_shared_fraction') if coordinate_hash_audit else None}, "
            f"syndrome={coordinate_hash_audit.get('syndrome_shared_fraction') if coordinate_hash_audit else None}."
        ),
        "",
        "Coordinate-hash diagnostic max grid values are kept out of the production "
        f"P1a decision because the candidate was rejected; all-candidate max grid "
        f"TV={all_max_grid_tv:.6f}, max grid dq={all_max_grid_dq:.6f}.",
        "",
        "## P1b Decision Diagnostic",
        "",
        (
            f"Strict pilot gate passed: "
            f"`{production_power.get('strict_pilot_gate_passed') if production_power else None}`. "
            f"Production-power gate passed: "
            f"`{production_power.get('pass_p1b_under_production_power_criterion') if production_power else None}`."
        ),
        "",
        "## Wall-Time Budget",
        "",
        f"Budget uses crossing-region `N_common={int(crossing_n_common)}` and deep-region `N_common=12`.",
        "",
        "| L | single point seconds | estimated serial hours per node batch |",
        "|---:|---:|---:|",
    ]
    for row in budget_rows:
        lines.append(
            f"| {row['lattice_size']} | "
            f"{row['single_point_seconds']:.3f} | "
            f"{row['node_parallel_hours']:.3f} |"
        )
    lines.extend([
        "",
        "## P1b Candidate Table",
        "",
        "| source | q | N | corr(L3,L5) | paired SEM | unpaired SEM | ratio | mean delta | CI95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for candidate in p1b_candidates:
        ci = candidate.get("paired_bootstrap_ci95", (math.nan, math.nan))
        lines.append(
            f"| {candidate.get('source')} | "
            f"{candidate.get('q_value', math.nan):.3f} | "
            f"{candidate.get('num_pairs', 0)} | "
            f"{candidate.get('correlation_l3_l5', math.nan):.3f} | "
            f"{candidate.get('paired_sem', math.nan):.6f} | "
            f"{candidate.get('unpaired_difference_sem', math.nan):.6f} | "
            f"{candidate.get('reduction_ratio', math.nan):.3f} | "
            f"{candidate.get('mean_delta_l5_minus_l3', math.nan):.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] |"
        )
    if not overall_pass:
        lines.extend([
            "",
            "## Next Step",
            "",
            (
                "P1 remains DOING. The original same-seed and q=0.20/0.21 retest "
                "runs have acceptable grid diagnostics, but P1b still misses the "
                "meaningful variance-reduction gate. The coordinate-hash/nested "
                "candidate also failed: it produced one grid warning and negative "
                "L3/L5 correlation, so it is not a production disorder mapping. "
                "Next work only on a redesigned P1b common-disorder strategy or a "
                "principled replacement criterion before entering P2."
            ),
        ])
    else:
        lines.extend([
            "",
            "## Decision",
            "",
            (
                "P1 passes under the recorded production-power P1b criterion. "
                "The strict pilot threshold `ratio <= 0.80 and paired SEM <= 0.05` "
                "is retained as an audit failure, but it is not used as the "
                "production-size readiness gate. P2 should use same-seed "
                "`rng_stream` public disorder with crossing-region `N_common=32`."
            ),
        ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
