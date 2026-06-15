#!/usr/bin/env python3
"""Merge exp38 P2 sector-TI shard NPZ files into one full-grid result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FLOAT_ARRAYS_3D = (
    "q_top_per_disorder",
    "q_top_stderr_per_disorder",
    "grid_tv_per_disorder",
    "grid_q_top_abs_diff_per_disorder",
    "wall_time_seconds_per_disorder",
)
FLOAT_ARRAYS_4D = (
    "q_top_ci95_per_disorder",
)
SECTOR_ARRAYS = (
    "weights_per_disorder",
    "weights_stderr_per_disorder",
    "delta_f_per_disorder",
    "delta_f_stderr_per_disorder",
)
SEED_ARRAYS = (
    "seed_per_disorder",
    "disorder_seed_per_disorder",
    "sample_seed_per_disorder",
)


def _read_text_scalar(value: np.ndarray) -> str:
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _discover_npz(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*/sector_ti/sector_ti_results.npz"))


def _require_same_grid(reference: np.ndarray | None, current: np.ndarray, path: Path) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    if reference is None:
        return current
    if reference.shape != current.shape or not np.allclose(reference, current, atol=0.0, rtol=0.0):
        raise ValueError(f"q grid mismatch in {path}")
    return reference


def _load_shards(paths: list[Path]) -> tuple[list[tuple[Path, dict, dict[str, np.ndarray]]], np.ndarray, float, int]:
    loaded = []
    q_values_ref = None
    p_value = None
    num_disorder = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(_read_text_scalar(data["manifest_json"]))
            q_values_ref = _require_same_grid(q_values_ref, data["q_values"], path)
            p_here = float(data["p_value"])
            if p_value is None:
                p_value = p_here
            elif p_value != p_here:
                raise ValueError(f"p mismatch in {path}: {p_here} != {p_value}")
            n_here = int(data["q_top_per_disorder"].shape[2])
            if num_disorder is None:
                num_disorder = n_here
            elif num_disorder != n_here:
                raise ValueError(f"num disorder mismatch in {path}: {n_here} != {num_disorder}")

            arrays = {
                name: data[name].copy()
                for name in data.files
                if name != "manifest_json"
            }
            loaded.append((path, manifest, arrays))
    if q_values_ref is None or p_value is None or num_disorder is None:
        raise ValueError("no usable shard data")
    return loaded, q_values_ref, float(p_value), int(num_disorder)


def _merge_arrays(
    loaded: list[tuple[Path, dict, dict[str, np.ndarray]]],
    q_values: np.ndarray,
    num_disorder: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    all_lattice_sizes = []
    for _, _, arrays in loaded:
        all_lattice_sizes.extend(int(value) for value in arrays["lattice_size_list"].astype(int))
    lattice_sizes = np.asarray(sorted(set(all_lattice_sizes)), dtype=np.int64)
    shape = (len(lattice_sizes), len(q_values), int(num_disorder))

    merged: dict[str, np.ndarray] = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in FLOAT_ARRAYS_3D
    }
    merged["q_top_ci95_per_disorder"] = np.full(shape + (2,), np.nan, dtype=np.float64)
    for name in SECTOR_ARRAYS:
        merged[name] = np.full(shape + (8,), np.nan, dtype=np.float64)
    for name in SEED_ARRAYS:
        merged[name] = np.full(shape, -1, dtype=np.int64)
    merged["flags_per_disorder"] = np.full(shape, "MISSING", dtype="<U128")

    l_index = {int(value): index for index, value in enumerate(lattice_sizes)}
    source_manifests = []
    for path, manifest, arrays in loaded:
        source_manifests.append({"path": str(path), "manifest": manifest})
        for local_li, lattice_size in enumerate(arrays["lattice_size_list"].astype(int)):
            li = l_index[int(lattice_size)]
            for name in FLOAT_ARRAYS_3D:
                if name in arrays:
                    merged[name][li] = arrays[name][local_li].astype(np.float64)
            for name in FLOAT_ARRAYS_4D:
                if name in arrays:
                    merged[name][li] = arrays[name][local_li].astype(np.float64)
            for name in SECTOR_ARRAYS:
                if name in arrays:
                    merged[name][li] = arrays[name][local_li].astype(np.float64)
            for name in SEED_ARRAYS:
                if name in arrays:
                    merged[name][li] = arrays[name][local_li].astype(np.int64)
            merged["flags_per_disorder"][li] = arrays["flags_per_disorder"][local_li].astype("<U128")
    return lattice_sizes, merged, source_manifests


def _add_aggregates(arrays: dict[str, np.ndarray]) -> None:
    q_top = arrays["q_top_per_disorder"]
    finite = np.isfinite(q_top)
    counts = np.sum(finite, axis=2).astype(np.int64)
    mean_q_top = np.nanmean(q_top, axis=2)
    disorder_sem = np.full(mean_q_top.shape, np.nan, dtype=np.float64)
    valid = counts > 1
    disorder_sem[valid] = np.nanstd(q_top, axis=2, ddof=1)[valid] / np.sqrt(counts[valid])
    disorder_sem[counts == 1] = 0.0
    mcmc_sem = np.sqrt(np.nanmean(arrays["q_top_stderr_per_disorder"] ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + mcmc_sem ** 2)
    pass_fraction = np.mean(arrays["flags_per_disorder"] == "PASS", axis=2)
    arrays["mean_q_top"] = mean_q_top
    arrays["disorder_sem_q_top"] = disorder_sem
    arrays["mcmc_sem_q_top"] = mcmc_sem
    arrays["total_sem_q_top"] = total_sem
    arrays["pass_fraction"] = pass_fraction
    arrays["finite_disorder_count"] = counts


def _write_summary(output_dir: Path, lattice_sizes: np.ndarray, q_values: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    lines = [
        "# exp38 P2 merged sector-TI grid",
        "",
        "| L | q | mean q_top | total SEM | disorder SEM | MCMC SEM | finite n | pass fraction | max grid TV | max grid dq |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            lines.append(
                "| "
                f"{int(lattice_size)} | "
                f"{float(q_value):.3f} | "
                f"{arrays['mean_q_top'][li, qi]:.6f} | "
                f"{arrays['total_sem_q_top'][li, qi]:.6f} | "
                f"{arrays['disorder_sem_q_top'][li, qi]:.6f} | "
                f"{arrays['mcmc_sem_q_top'][li, qi]:.6f} | "
                f"{int(arrays['finite_disorder_count'][li, qi])} | "
                f"{arrays['pass_fraction'][li, qi]:.3f} | "
                f"{np.nanmax(arrays['grid_tv_per_disorder'][li, qi]):.6f} | "
                f"{np.nanmax(arrays['grid_q_top_abs_diff_per_disorder'][li, qi]):.6f} |"
            )
    (output_dir / "sector_ti_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = _discover_npz(args.input_dir)
    if not paths:
        raise SystemExit(f"no shard sector_ti_results.npz under {args.input_dir}")

    loaded, q_values, p_value, num_disorder = _load_shards(paths)
    lattice_sizes, arrays, source_manifests = _merge_arrays(loaded, q_values, num_disorder)
    _add_aggregates(arrays)

    manifest = {
        "mode": "exp38_p2_merged_sector_ti",
        "source_manifests": source_manifests,
        "lattice_sizes": lattice_sizes.tolist(),
        "q_values": [float(value) for value in q_values.tolist()],
        "p_value": float(p_value),
        "num_disorder_samples": int(num_disorder),
        "merge_note": "seed arrays are preserved when present so cross-L common-disorder checks remain auditable",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "sector_ti_results.npz",
        manifest_json=np.array(json.dumps(manifest, indent=2, sort_keys=True)),
        lattice_size_list=lattice_sizes,
        q_values=q_values,
        p_value=np.float64(p_value),
        **arrays,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary(args.output_dir, lattice_sizes, q_values, arrays)
    print(args.output_dir / "sector_ti_results.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
