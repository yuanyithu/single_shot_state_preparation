#!/usr/bin/env python3
"""Compare Stage F production TI, targeted strong TI, and bridge subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _format_q(value: float) -> float:
    return float(round(float(value), 6))


def _load_ti_npz(path: Path) -> dict[tuple[int, float, int], dict]:
    with np.load(path, allow_pickle=False) as data:
        lattice_sizes = data["lattice_size_list"].astype(int)
        q_values = data["q_values"].astype(float)
        q_top = data["q_top_per_disorder"].astype(float)
        q_top_stderr = data["q_top_stderr_per_disorder"].astype(float)
        q_top_ci95 = data["q_top_ci95_per_disorder"].astype(float)
        weights = data["weights_per_disorder"].astype(float)
        grid_tv = data["grid_tv_per_disorder"].astype(float)
        grid_dq = data["grid_q_top_abs_diff_per_disorder"].astype(float)
        flags = data["flags_per_disorder"].astype("<U128")
    records = {}
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            for disorder_index in range(q_top.shape[2]):
                records[(int(lattice_size), _format_q(q_value), int(disorder_index))] = {
                    "q_top": float(q_top[li, qi, disorder_index]),
                    "q_top_stderr": float(q_top_stderr[li, qi, disorder_index]),
                    "q_top_ci95": q_top_ci95[li, qi, disorder_index],
                    "weights": weights[li, qi, disorder_index],
                    "grid_tv": float(grid_tv[li, qi, disorder_index]),
                    "grid_dq": float(grid_dq[li, qi, disorder_index]),
                    "flags": str(flags[li, qi, disorder_index]),
                }
    return records


def _load_bridge(path: Path) -> dict[tuple[int, float, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for record in payload["records"]:
        records[
            (
                int(record["lattice_size"]),
                _format_q(record["q_value"]),
                int(record["disorder_index"]),
            )
        ] = record
    return records


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Stage F targeted TI vs bridge comparison",
        "",
        "This compares the original production sector-TI grid, the targeted stronger sector-TI rerun, and the stronger bidirectional logical-loop bridge BAR subset.",
        "",
        "| L | q | d | old TI q_top | strong TI q_top | bridge q_top | |strong-old| | |strong-bridge| | TV strong-old | TV strong-bridge | strong grid TV | strong grid dq |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lattice_size']} | {row['q_value']:.3f} | "
            f"{row['disorder_index']} | {row['old_ti_q_top']:.6f} | "
            f"{row['strong_ti_q_top']:.6f} | {row['bridge_q_top']:.6f} | "
            f"{row['strong_old_q_top_abs_diff']:.6f} | "
            f"{row['strong_bridge_q_top_abs_diff']:.6f} | "
            f"{row['strong_old_tv']:.6f} | "
            f"{row['strong_bridge_tv']:.6f} | "
            f"{row['strong_ti_grid_tv']:.6f} | "
            f"{row['strong_ti_grid_q_top_abs_diff']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ti", type=Path, required=True)
    parser.add_argument("--strong-ti", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old_ti = _load_ti_npz(args.old_ti)
    strong_ti = _load_ti_npz(args.strong_ti)
    bridge = _load_bridge(args.bridge)
    rows = []
    for key in sorted(bridge):
        if key not in old_ti or key not in strong_ti:
            continue
        lattice_size, q_value, disorder_index = key
        old_record = old_ti[key]
        strong_record = strong_ti[key]
        bridge_record = bridge[key]
        bridge_weights = np.asarray(bridge_record["bridge_weights"], dtype=np.float64)
        rows.append({
            "lattice_size": int(lattice_size),
            "q_value": float(q_value),
            "disorder_index": int(disorder_index),
            "old_ti_q_top": float(old_record["q_top"]),
            "old_ti_q_top_stderr": float(old_record["q_top_stderr"]),
            "old_ti_ci_low": float(old_record["q_top_ci95"][0]),
            "old_ti_ci_high": float(old_record["q_top_ci95"][1]),
            "old_ti_grid_tv": float(old_record["grid_tv"]),
            "old_ti_grid_q_top_abs_diff": float(old_record["grid_dq"]),
            "old_ti_flags": str(old_record["flags"]),
            "strong_ti_q_top": float(strong_record["q_top"]),
            "strong_ti_q_top_stderr": float(strong_record["q_top_stderr"]),
            "strong_ti_ci_low": float(strong_record["q_top_ci95"][0]),
            "strong_ti_ci_high": float(strong_record["q_top_ci95"][1]),
            "strong_ti_grid_tv": float(strong_record["grid_tv"]),
            "strong_ti_grid_q_top_abs_diff": float(strong_record["grid_dq"]),
            "strong_ti_flags": str(strong_record["flags"]),
            "bridge_q_top": float(bridge_record["bridge_q_top"]),
            "bridge_tv_vs_old_ti": float(bridge_record["tv_vs_ti"]),
            "bridge_q_top_abs_diff_vs_old_ti": float(
                bridge_record["q_top_abs_diff_vs_ti"]
            ),
            "bridge_bidir_gap": float(
                bridge_record["max_full_path_bidirectional_gap"]
            ),
            "strong_old_q_top_abs_diff": float(
                abs(strong_record["q_top"] - old_record["q_top"])
            ),
            "strong_bridge_q_top_abs_diff": float(
                abs(strong_record["q_top"] - bridge_record["bridge_q_top"])
            ),
            "strong_old_tv": _tv(strong_record["weights"], old_record["weights"]),
            "strong_bridge_tv": _tv(strong_record["weights"], bridge_weights),
        })
    if not rows:
        raise SystemExit("no overlapping records to compare")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "old_ti_path": str(args.old_ti),
        "strong_ti_path": str(args.strong_ti),
        "bridge_path": str(args.bridge),
        "records": rows,
    }
    (args.output_dir / "targeted_ti_vs_bridge_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "targeted_ti_vs_bridge_comparison.csv", rows)
    _write_summary(args.output_dir / "summary.md", rows)
    print(json.dumps({
        "num_records": len(rows),
        "max_strong_old_q_top_abs_diff": max(
            row["strong_old_q_top_abs_diff"] for row in rows
        ),
        "max_strong_bridge_q_top_abs_diff": max(
            row["strong_bridge_q_top_abs_diff"] for row in rows
        ),
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
