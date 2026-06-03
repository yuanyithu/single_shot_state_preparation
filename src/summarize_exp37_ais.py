#!/usr/bin/env python3
"""Summarize exp37 AIS result directories into one q-grid table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _flag_summary(flags: np.ndarray) -> str:
    flat = [_as_str(value) for value in flags.ravel()]
    counts = {}
    for flag in flat:
        counts[flag] = counts.get(flag, 0) + 1
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _load_rows(result_path: Path) -> list[dict[str, object]]:
    with np.load(result_path, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int).tolist()
        q_values = data["q_values"].astype(float).tolist()
        mean_q_top = data["mean_q_top"]
        total_sem = data["total_sem_q_top"]
        pass_fraction = data["pass_fraction"]
        weights = data["weights_per_disorder"]
        ais_ess = data["ais_ess_per_disorder"]
        ais_ess_fraction = data["ais_ess_fraction_per_disorder"]
        sector_sample_counts = data["sector_sample_counts_per_disorder"]
        if "num_ais_particles_per_disorder" in data:
            num_particles = data["num_ais_particles_per_disorder"]
        else:
            num_particles = np.full_like(ais_ess, np.nan, dtype=np.float64)
        flags = data["flags_per_disorder"]

        rows = []
        for li, lattice_size in enumerate(lattice_sizes):
            for qi, q_value in enumerate(q_values):
                q_flags = flags[li, qi]
                tail_samples = int(np.nansum(sector_sample_counts[li, qi, :, 1:]))
                tail_weight_by_disorder = np.nansum(
                    weights[li, qi, :, 1:],
                    axis=1,
                )
                rows.append({
                    "source": str(result_path),
                    "run": result_path.parent.name,
                    "lattice_size": int(lattice_size),
                    "q_value": float(q_value),
                    "mean_q_top": float(mean_q_top[li, qi]),
                    "total_sem_q_top": float(total_sem[li, qi]),
                    "pass_fraction": float(pass_fraction[li, qi]),
                    "min_ais_ess": float(np.nanmin(ais_ess[li, qi])),
                    "min_ais_ess_fraction": float(
                        np.nanmin(ais_ess_fraction[li, qi])
                    ),
                    "min_particles": float(np.nanmin(num_particles[li, qi])),
                    "tail_samples": tail_samples,
                    "mean_tail_weight": float(np.nanmean(tail_weight_by_disorder)),
                    "max_tail_weight": float(np.nanmax(tail_weight_by_disorder)),
                    "flags": _flag_summary(q_flags),
                })
    return rows


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# exp37 AIS combined summary",
        "",
        "| run | L | q | mean q_top | total SEM | pass frac | min ESS | min ESS frac | min particles | tail samples | mean tail w | max tail w | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['run']} | "
            f"{row['lattice_size']} | "
            f"{_format_float(float(row['q_value']), 3)} | "
            f"{_format_float(float(row['mean_q_top']))} | "
            f"{_format_float(float(row['total_sem_q_top']))} | "
            f"{_format_float(float(row['pass_fraction']), 3)} | "
            f"{_format_float(float(row['min_ais_ess']), 1)} | "
            f"{_format_float(float(row['min_ais_ess_fraction']), 4)} | "
            f"{_format_float(float(row['min_particles']), 0)} | "
            f"{row['tail_samples']} | "
            f"{_format_float(float(row['mean_tail_weight']))} | "
            f"{_format_float(float(row['max_tail_weight']))} | "
            f"{row['flags']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "source",
        "run",
        "lattice_size",
        "q_value",
        "mean_q_top",
        "total_sem_q_top",
        "pass_fraction",
        "min_ais_ess",
        "min_ais_ess_fraction",
        "min_particles",
        "tail_samples",
        "mean_tail_weight",
        "max_tail_weight",
        "flags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    result_paths = sorted(args.input_dir.glob("*/ais_results.npz"))
    if not result_paths:
        raise SystemExit(f"no ais_results.npz found under {args.input_dir}")

    rows = []
    for result_path in result_paths:
        rows.extend(_load_rows(result_path))
    rows.sort(key=lambda row: (int(row["lattice_size"]), float(row["q_value"])))

    markdown_path = args.markdown or args.input_dir / "combined_ais_summary.md"
    csv_path = args.csv or args.input_dir / "combined_ais_summary.csv"
    _write_markdown(rows, markdown_path)
    _write_csv(rows, csv_path)
    print(markdown_path)
    print(csv_path)


if __name__ == "__main__":
    main()
