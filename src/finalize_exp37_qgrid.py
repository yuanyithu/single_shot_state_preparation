#!/usr/bin/env python3
"""Build the final corrected exp37 q-grid from production AIS shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP37_ROOT = (
    PROJECT_ROOT
    / "data"
    / "3d_toric_code"
    / "with_measurement_noise"
    / "exp37"
)

RUN030 = EXP37_ROOT / "030_ais_corrected_flip_grid513_R4_d4_20260603"
RUN031 = EXP37_ROOT / "031_l5_lowq_t16_R4_20260603"

SOURCE_PATHS = [
    RUN030 / "L3_nd1" / "ais_results.npz",
    RUN030 / "L4_nd2" / "ais_results.npz",
    RUN030 / "L5_q008_013_nd3" / "ais_results.npz",
    RUN030 / "L5_q014_018_nd1" / "ais_results.npz",
    RUN030 / "L5_q019_023_nd2" / "ais_results.npz",
    RUN031 / "L5_q008_009_t16_nd3" / "ais_results.npz",
]


def _q_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


def _read_scalar_text(array: np.ndarray) -> str:
    value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _delta_f_stderr_from_weights(
    weights: np.ndarray,
    weights_stderr: np.ndarray,
) -> np.ndarray:
    stderr = np.full_like(weights, np.nan, dtype=np.float64)
    stderr[..., 0] = 0.0
    w0 = weights[..., 0]
    s0 = weights_stderr[..., 0]
    for sector in range(1, weights.shape[-1]):
        wg = weights[..., sector]
        sg = weights_stderr[..., sector]
        mask = (w0 > 0.0) & (wg > 0.0)
        stderr[..., sector][mask] = np.sqrt(
            (s0[mask] / w0[mask]) ** 2 + (sg[mask] / wg[mask]) ** 2
        )
    return stderr


def _flag_summary(flags: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for value in flags.ravel():
        flag = str(value)
        counts[flag] = counts.get(flag, 0) + 1
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _load_sources(source_paths: list[Path]) -> dict[tuple[int, int], dict[str, object]]:
    points: dict[tuple[int, int], dict[str, object]] = {}
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        with np.load(source_path, allow_pickle=False) as data:
            manifest = json.loads(_read_scalar_text(data["manifest_json"]))
            lattice_sizes = data["lattice_size_list"].astype(int).tolist()
            q_values = data["q_values"].astype(float).tolist()
            delta_f_stderr = (
                data["delta_f_stderr_per_disorder"]
                if "delta_f_stderr_per_disorder" in data
                else _delta_f_stderr_from_weights(
                    data["weights_per_disorder"],
                    data["weights_stderr_per_disorder"],
                )
            )
            delta_f_stderr_source = (
                "source_npz"
                if "delta_f_stderr_per_disorder" in data
                else "delta_method_from_weights_stderr"
            )
            for li, lattice_size in enumerate(lattice_sizes):
                for qi, q_value in enumerate(q_values):
                    key = (int(lattice_size), _q_key(q_value))
                    points[key] = {
                        "source_path": source_path,
                        "source_experiment": source_path.parents[1].name,
                        "source_run": source_path.parent.name,
                        "manifest": manifest,
                        "q_top_per_disorder": data[
                            "q_top_per_disorder"
                        ][li, qi].copy(),
                        "q_top_stderr_per_disorder": data[
                            "q_top_stderr_per_disorder"
                        ][li, qi].copy(),
                        "q_top_ci95_per_disorder": data[
                            "q_top_ci95_per_disorder"
                        ][li, qi].copy(),
                        "weights_per_disorder": data[
                            "weights_per_disorder"
                        ][li, qi].copy(),
                        "weights_stderr_per_disorder": data[
                            "weights_stderr_per_disorder"
                        ][li, qi].copy(),
                        "delta_f_per_disorder": data[
                            "delta_f_per_disorder"
                        ][li, qi].copy(),
                        "delta_f_stderr_per_disorder": delta_f_stderr[
                            li, qi
                        ].copy(),
                        "delta_f_stderr_source": delta_f_stderr_source,
                        "ais_ess_per_disorder": data[
                            "ais_ess_per_disorder"
                        ][li, qi].copy(),
                        "ais_ess_fraction_per_disorder": data[
                            "ais_ess_fraction_per_disorder"
                        ][li, qi].copy(),
                        "sector_sample_counts_per_disorder": data[
                            "sector_sample_counts_per_disorder"
                        ][li, qi].copy(),
                        "flags_per_disorder": data[
                            "flags_per_disorder"
                        ][li, qi].astype("<U128").copy(),
                        "wall_time_seconds_per_disorder": data[
                            "wall_time_seconds_per_disorder"
                        ][li, qi].copy(),
                        "num_ais_particles_per_disorder": data[
                            "num_ais_particles_per_disorder"
                        ][li, qi].copy(),
                    }
    return points


def _compute_rows(
    lattice_sizes: np.ndarray,
    q_values: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    q_top = arrays["q_top_per_disorder"]
    q_top_stderr = arrays["q_top_stderr_per_disorder"]
    weights = arrays["weights_per_disorder"]
    ais_ess = arrays["ais_ess_per_disorder"]
    ais_ess_fraction = arrays["ais_ess_fraction_per_disorder"]
    sector_counts = arrays["sector_sample_counts_per_disorder"]
    flags = arrays["flags_per_disorder"]
    particles = arrays["num_ais_particles_per_disorder"]
    source_exp = arrays["source_experiment_per_point"]
    source_run = arrays["source_run_per_point"]

    mean_q_top = np.nanmean(q_top, axis=2)
    if q_top.shape[2] > 1:
        disorder_sem = np.nanstd(q_top, axis=2, ddof=1) / math.sqrt(q_top.shape[2])
    else:
        disorder_sem = np.zeros_like(mean_q_top)
    ais_sem = np.sqrt(np.nanmean(q_top_stderr ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + ais_sem ** 2)
    pass_fraction = np.mean(flags == "PASS", axis=2)

    arrays["mean_q_top"] = mean_q_top
    arrays["disorder_sem_q_top"] = disorder_sem
    arrays["ais_sem_q_top"] = ais_sem
    arrays["total_sem_q_top"] = total_sem
    arrays["pass_fraction"] = pass_fraction

    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            tail_samples = int(np.nansum(sector_counts[li, qi, :, 1:]))
            tail_weight_by_disorder = np.nansum(weights[li, qi, :, 1:], axis=1)
            rows.append(
                {
                    "lattice_size": int(lattice_size),
                    "q_value": float(q_value),
                    "mean_q_top": float(mean_q_top[li, qi]),
                    "disorder_sem_q_top": float(disorder_sem[li, qi]),
                    "ais_sem_q_top": float(ais_sem[li, qi]),
                    "total_sem_q_top": float(total_sem[li, qi]),
                    "pass_fraction": float(pass_fraction[li, qi]),
                    "min_ais_ess": float(np.nanmin(ais_ess[li, qi])),
                    "min_ais_ess_fraction": float(
                        np.nanmin(ais_ess_fraction[li, qi])
                    ),
                    "min_particles": int(np.nanmin(particles[li, qi])),
                    "tail_samples": tail_samples,
                    "mean_tail_weight": float(np.nanmean(tail_weight_by_disorder)),
                    "max_tail_weight": float(np.nanmax(tail_weight_by_disorder)),
                    "flags": _flag_summary(flags[li, qi]),
                    "source_experiment": str(source_exp[li, qi]),
                    "source_run": str(source_run[li, qi]),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "lattice_size",
        "q_value",
        "mean_q_top",
        "disorder_sem_q_top",
        "ais_sem_q_top",
        "total_sem_q_top",
        "pass_fraction",
        "min_ais_ess",
        "min_ais_ess_fraction",
        "min_particles",
        "tail_samples",
        "mean_tail_weight",
        "max_tail_weight",
        "flags",
        "source_experiment",
        "source_run",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# exp37 final corrected q-grid",
        "",
        "Merge rule: use 030 for all rows except L=5, q=0.08 and q=0.09, "
        "which use 031 t16 reinforcement.",
        "",
        "| L | q | mean q_top | total SEM | pass frac | min ESS | min ESS frac | tail samples | max tail w | flags | source |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row['lattice_size'])} | "
            f"{float(row['q_value']):.3f} | "
            f"{float(row['mean_q_top']):.6f} | "
            f"{float(row['total_sem_q_top']):.6f} | "
            f"{float(row['pass_fraction']):.3f} | "
            f"{float(row['min_ais_ess']):.1f} | "
            f"{float(row['min_ais_ess_fraction']):.4f} | "
            f"{int(row['tail_samples'])} | "
            f"{float(row['max_tail_weight']):.6f} | "
            f"{row['flags']} | "
            f"{row['source_run']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_final(output_dir: Path) -> None:
    lattice_sizes = np.asarray([3, 4, 5], dtype=np.int64)
    q_values = np.asarray([round(0.08 + 0.01 * index, 2) for index in range(16)])
    num_disorder = 4
    shape = (len(lattice_sizes), len(q_values), num_disorder)
    arrays: dict[str, np.ndarray] = {
        "q_top_per_disorder": np.full(shape, np.nan, dtype=np.float64),
        "q_top_stderr_per_disorder": np.full(shape, np.nan, dtype=np.float64),
        "q_top_ci95_per_disorder": np.full(shape + (2,), np.nan, dtype=np.float64),
        "weights_per_disorder": np.full(shape + (8,), np.nan, dtype=np.float64),
        "weights_stderr_per_disorder": np.full(
            shape + (8,), np.nan, dtype=np.float64
        ),
        "delta_f_per_disorder": np.full(shape + (8,), np.nan, dtype=np.float64),
        "delta_f_stderr_per_disorder": np.full(
            shape + (8,), np.nan, dtype=np.float64
        ),
        "ais_ess_per_disorder": np.full(shape, np.nan, dtype=np.float64),
        "ais_ess_fraction_per_disorder": np.full(shape, np.nan, dtype=np.float64),
        "sector_sample_counts_per_disorder": np.full(
            shape + (8,), -1, dtype=np.int64
        ),
        "flags_per_disorder": np.full(shape, "MISSING", dtype="<U128"),
        "wall_time_seconds_per_disorder": np.full(shape, np.nan, dtype=np.float64),
        "num_ais_particles_per_disorder": np.full(shape, 0, dtype=np.int64),
        "source_experiment_per_point": np.full(
            (len(lattice_sizes), len(q_values)), "", dtype="<U96"
        ),
        "source_run_per_point": np.full(
            (len(lattice_sizes), len(q_values)), "", dtype="<U64"
        ),
        "source_result_path_per_point": np.full(
            (len(lattice_sizes), len(q_values)), "", dtype="<U256"
        ),
        "delta_f_stderr_source_per_point": np.full(
            (len(lattice_sizes), len(q_values)), "", dtype="<U64"
        ),
    }
    sources = _load_sources(SOURCE_PATHS)
    l_index = {int(value): index for index, value in enumerate(lattice_sizes)}
    q_index = {_q_key(value): index for index, value in enumerate(q_values)}

    for lattice_size in lattice_sizes:
        for q_value in q_values:
            key = (int(lattice_size), _q_key(q_value))
            if int(lattice_size) == 5 and key[1] in {_q_key(0.08), _q_key(0.09)}:
                source_key = key
                source = sources[source_key]
                if source["source_experiment"] != RUN031.name:
                    raise RuntimeError(f"low-q override did not select 031 for {key}")
            else:
                source = sources[key]
                if source["source_experiment"] != RUN030.name:
                    raise RuntimeError(f"default row did not select 030 for {key}")

            li = l_index[int(lattice_size)]
            qi = q_index[_q_key(q_value)]
            for name in [
                "q_top_per_disorder",
                "q_top_stderr_per_disorder",
                "q_top_ci95_per_disorder",
                "weights_per_disorder",
                "weights_stderr_per_disorder",
                "delta_f_per_disorder",
                "delta_f_stderr_per_disorder",
                "ais_ess_per_disorder",
                "ais_ess_fraction_per_disorder",
                "sector_sample_counts_per_disorder",
                "flags_per_disorder",
                "wall_time_seconds_per_disorder",
                "num_ais_particles_per_disorder",
            ]:
                arrays[name][li, qi] = source[name]
            arrays["source_experiment_per_point"][li, qi] = source[
                "source_experiment"
            ]
            arrays["source_run_per_point"][li, qi] = source["source_run"]
            arrays["source_result_path_per_point"][li, qi] = str(
                source["source_path"]
            )
            arrays["delta_f_stderr_source_per_point"][li, qi] = source[
                "delta_f_stderr_source"
            ]

    rows = _compute_rows(lattice_sizes, q_values, arrays)
    if any(row["flags"] != "PASS:4" for row in rows):
        bad = [row for row in rows if row["flags"] != "PASS:4"]
        raise RuntimeError(f"final grid contains non-PASS rows: {bad}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Final corrected exp37 q-grid for p=0.05.",
        "merge_rule": (
            "030 for all rows except L=5 q=0.08/0.09, "
            "which are replaced by 031 transition_sweeps=16 reinforcement."
        ),
        "lattice_sizes": lattice_sizes.tolist(),
        "q_values": q_values.tolist(),
        "p_value": 0.05,
        "num_disorder_samples": num_disorder,
        "source_paths": [str(path) for path in SOURCE_PATHS],
        "all_rows_pass": True,
        "sector_observable": "corrected_c_eta_section",
        "ais_estimator": "flip_reweight",
        "delta_f_stderr_note": (
            "Source AIS NPZ files predate saving delta_f_stderr_per_disorder; "
            "final grid uses a delta-method estimate from weights_stderr_per_disorder."
        ),
    }
    np.savez_compressed(
        output_dir / "final_qgrid.npz",
        manifest_json=np.array(json.dumps(manifest, indent=2)),
        lattice_size_list=lattice_sizes,
        q_values=q_values,
        p_value=np.float64(0.05),
        **arrays,
    )
    _write_csv(rows, output_dir / "final_qgrid.csv")
    _write_markdown(rows, output_dir / "final_qgrid.md")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# exp37 final corrected q-grid",
                "",
                "This directory is the final merged corrected AIS grid.",
                "",
                "- Source 030 supplies all rows except L=5 q=0.08 and q=0.09.",
                "- Source 031 supplies L=5 q=0.08 and q=0.09 after t16 reinforcement.",
                "- All 48 final rows have PASS:4 flags.",
                "- final_qgrid.npz contains per-disorder weights[8], DeltaF[8], "
                "q_top, CI/stderr arrays, ESS diagnostics, flags, and source mapping.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXP37_ROOT / "032_final_corrected_qgrid_20260603",
    )
    args = parser.parse_args()
    _build_final(args.output_dir)
    print(args.output_dir / "final_qgrid.npz")
    print(args.output_dir / "final_qgrid.csv")
    print(args.output_dir / "final_qgrid.md")


if __name__ == "__main__":
    main()
