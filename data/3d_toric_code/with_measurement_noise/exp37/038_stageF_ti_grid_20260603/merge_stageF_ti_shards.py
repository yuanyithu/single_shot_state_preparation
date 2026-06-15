#!/usr/bin/env python3
"""Merge Stage F sector-TI shard NPZ files into one full-grid result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_text_scalar(value: np.ndarray) -> str:
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def discover_npz(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*/sector_ti/sector_ti_results.npz"))


def write_summary(output_dir: Path, lattice_sizes: np.ndarray, q_values: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    lines = [
        "# Stage F merged sector-TI grid",
        "",
        "| L | q | mean q_top | total SEM | disorder SEM | MCMC SEM | pass fraction | max grid TV | max grid dq |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{arrays['pass_fraction'][li, qi]:.3f} | "
                f"{np.nanmax(arrays['grid_tv_per_disorder'][li, qi]):.6f} | "
                f"{np.nanmax(arrays['grid_q_top_abs_diff_per_disorder'][li, qi]):.6f} |"
            )
    (output_dir / "sector_ti_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = discover_npz(args.input_dir)
    if not paths:
        raise SystemExit(f"no shard sector_ti_results.npz under {args.input_dir}")
    loaded = []
    all_lattice_sizes = []
    q_values_ref = None
    num_disorder = None
    p_value = None
    for path in paths:
        data = np.load(path, allow_pickle=False)
        manifest = json.loads(read_text_scalar(data["manifest_json"]))
        q_values = data["q_values"].astype(float)
        if q_values_ref is None:
            q_values_ref = q_values
        elif not np.allclose(q_values_ref, q_values, atol=0.0, rtol=0.0):
            raise ValueError(f"q grid mismatch in {path}")
        if num_disorder is None:
            num_disorder = data["q_top_per_disorder"].shape[2]
        elif num_disorder != data["q_top_per_disorder"].shape[2]:
            raise ValueError(f"num disorder mismatch in {path}")
        p_here = float(data["p_value"])
        if p_value is None:
            p_value = p_here
        elif p_value != p_here:
            raise ValueError(f"p mismatch in {path}")
        lattice_sizes = data["lattice_size_list"].astype(int)
        all_lattice_sizes.extend(int(value) for value in lattice_sizes)
        loaded.append((path, manifest, data))

    lattice_sizes_out = np.asarray(sorted(set(all_lattice_sizes)), dtype=np.int64)
    q_values_out = np.asarray(q_values_ref, dtype=np.float64)
    shape = (len(lattice_sizes_out), len(q_values_out), int(num_disorder))
    array_names = [
        "q_top_per_disorder",
        "q_top_stderr_per_disorder",
        "grid_tv_per_disorder",
        "grid_q_top_abs_diff_per_disorder",
        "wall_time_seconds_per_disorder",
    ]
    arrays: dict[str, np.ndarray] = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in array_names
    }
    arrays["q_top_ci95_per_disorder"] = np.full(shape + (2,), np.nan, dtype=np.float64)
    for name in [
        "weights_per_disorder",
        "weights_stderr_per_disorder",
        "delta_f_per_disorder",
        "delta_f_stderr_per_disorder",
    ]:
        arrays[name] = np.full(shape + (8,), np.nan, dtype=np.float64)
    arrays["flags_per_disorder"] = np.full(shape, "MISSING", dtype="<U128")

    l_index = {int(value): index for index, value in enumerate(lattice_sizes_out)}
    source_manifests = []
    for path, manifest, data in loaded:
        source_manifests.append({"path": str(path), "manifest": manifest})
        for local_li, lattice_size in enumerate(data["lattice_size_list"].astype(int)):
            li = l_index[int(lattice_size)]
            for name in array_names:
                arrays[name][li] = data[name][local_li]
            arrays["q_top_ci95_per_disorder"][li] = data["q_top_ci95_per_disorder"][local_li]
            for name in [
                "weights_per_disorder",
                "weights_stderr_per_disorder",
                "delta_f_per_disorder",
                "delta_f_stderr_per_disorder",
            ]:
                arrays[name][li] = data[name][local_li]
            arrays["flags_per_disorder"][li] = data["flags_per_disorder"][local_li].astype("<U128")
        data.close()

    q_top = arrays["q_top_per_disorder"]
    mean_q_top = np.nanmean(q_top, axis=2)
    if int(num_disorder) > 1:
        disorder_sem = np.nanstd(q_top, axis=2, ddof=1) / math.sqrt(float(num_disorder))
    else:
        disorder_sem = np.zeros_like(mean_q_top)
    mcmc_sem = np.sqrt(np.nanmean(arrays["q_top_stderr_per_disorder"] ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + mcmc_sem ** 2)
    pass_fraction = np.empty(mean_q_top.shape, dtype=np.float64)
    for li in range(len(lattice_sizes_out)):
        for qi in range(len(q_values_out)):
            pass_fraction[li, qi] = float(np.mean(arrays["flags_per_disorder"][li, qi] == "PASS"))
    arrays["mean_q_top"] = mean_q_top
    arrays["disorder_sem_q_top"] = disorder_sem
    arrays["mcmc_sem_q_top"] = mcmc_sem
    arrays["total_sem_q_top"] = total_sem
    arrays["pass_fraction"] = pass_fraction

    manifest = {
        "mode": "stageF_merged_sector_ti",
        "source_manifests": source_manifests,
        "lattice_sizes": lattice_sizes_out.tolist(),
        "q_values": q_values_out.tolist(),
        "p_value": float(p_value),
        "num_disorder_samples": int(num_disorder),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "sector_ti_results.npz",
        manifest_json=np.array(json.dumps(manifest, indent=2, sort_keys=True)),
        lattice_size_list=lattice_sizes_out,
        q_values=q_values_out,
        p_value=np.float64(p_value),
        **arrays,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(args.output_dir, lattice_sizes_out, q_values_out, arrays)
    print(args.output_dir / "sector_ti_results.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
