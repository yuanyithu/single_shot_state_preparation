#!/usr/bin/env python3
"""Build a Stage F grid with targeted strong TI records replaced.

The original full Stage F grid is preserved.  This script creates a derived
NPZ where only records present in a targeted strong TI rerun replace the
corresponding production TI records, then regenerates the second-method JSON
so its TV/q_top differences are measured against the repaired grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


REPLACED_ARRAYS = [
    "q_top_per_disorder",
    "q_top_stderr_per_disorder",
    "q_top_ci95_per_disorder",
    "grid_tv_per_disorder",
    "grid_q_top_abs_diff_per_disorder",
    "weights_per_disorder",
    "weights_stderr_per_disorder",
    "delta_f_per_disorder",
    "delta_f_stderr_per_disorder",
    "flags_per_disorder",
    "wall_time_seconds_per_disorder",
]


def _read_text_scalar(value: np.ndarray) -> str:
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _format_q(value: float) -> float:
    return float(round(float(value), 6))


def _load_npz(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(_read_text_scalar(data["manifest_json"]))
        for name in data.files:
            if name == "manifest_json":
                continue
            arrays[name] = data[name].copy()
    return manifest, arrays


def _recompute_aggregates(arrays: dict[str, np.ndarray]) -> None:
    q_top = arrays["q_top_per_disorder"].astype(float)
    flags = arrays["flags_per_disorder"].astype("<U128")
    num_disorder = int(q_top.shape[2])
    mean_q_top = np.nanmean(q_top, axis=2)
    if num_disorder > 1:
        disorder_sem = np.nanstd(q_top, axis=2, ddof=1) / math.sqrt(float(num_disorder))
    else:
        disorder_sem = np.zeros_like(mean_q_top)
    mcmc_sem = np.sqrt(np.nanmean(arrays["q_top_stderr_per_disorder"] ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + mcmc_sem ** 2)
    pass_fraction = np.empty(mean_q_top.shape, dtype=np.float64)
    for li in range(mean_q_top.shape[0]):
        for qi in range(mean_q_top.shape[1]):
            pass_fraction[li, qi] = float(np.mean(flags[li, qi] == "PASS"))
    arrays["mean_q_top"] = mean_q_top
    arrays["disorder_sem_q_top"] = disorder_sem
    arrays["mcmc_sem_q_top"] = mcmc_sem
    arrays["total_sem_q_top"] = total_sem
    arrays["pass_fraction"] = pass_fraction


def _write_summary(output_dir: Path, arrays: dict[str, np.ndarray], replacements: list[dict]) -> None:
    lattice_sizes = arrays["lattice_size_list"].astype(int)
    q_values = arrays["q_values"].astype(float)
    lines = [
        "# Stage F targeted-repaired sector-TI grid",
        "",
        "Only the records listed below were replaced by the targeted strong TI rerun; all other records are copied from the full production TI grid.",
        "",
        "## Replaced Records",
        "",
        "| L | q | disorder | old q_top | strong q_top | old flag | strong flag |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in replacements:
        lines.append(
            f"| {record['lattice_size']} | {record['q_value']:.3f} | "
            f"{record['disorder_index']} | {record['old_q_top']:.6f} | "
            f"{record['new_q_top']:.6f} | {record['old_flag']} | "
            f"{record['new_flag']} |"
        )
    lines.extend([
        "",
        "## Grid Summary",
        "",
        "| L | q | mean q_top | total SEM | pass fraction | max grid TV | max grid dq |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            lines.append(
                f"| {int(lattice_size)} | {float(q_value):.3f} | "
                f"{arrays['mean_q_top'][li, qi]:.6f} | "
                f"{arrays['total_sem_q_top'][li, qi]:.6f} | "
                f"{arrays['pass_fraction'][li, qi]:.3f} | "
                f"{np.nanmax(arrays['grid_tv_per_disorder'][li, qi]):.6f} | "
                f"{np.nanmax(arrays['grid_q_top_abs_diff_per_disorder'][li, qi]):.6f} |"
            )
    (output_dir / "sector_ti_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_rebased_bridge(
        bridge_path: Path,
        repaired_arrays: dict[str, np.ndarray],
        output_path: Path,
        tv_threshold: float,
        dq_threshold: float) -> None:
    bridge_payload = json.loads(bridge_path.read_text(encoding="utf-8"))
    lattice_sizes = repaired_arrays["lattice_size_list"].astype(int)
    q_values = repaired_arrays["q_values"].astype(float)
    l_index = {int(value): index for index, value in enumerate(lattice_sizes)}
    q_index = {_format_q(value): index for index, value in enumerate(q_values)}
    for record in bridge_payload["records"]:
        lattice_size = int(record["lattice_size"])
        q_value = _format_q(record["q_value"])
        disorder_index = int(record["disorder_index"])
        li = l_index[lattice_size]
        qi = q_index[q_value]
        ti_weights = repaired_arrays["weights_per_disorder"][li, qi, disorder_index].astype(float)
        ti_delta_f = repaired_arrays["delta_f_per_disorder"][li, qi, disorder_index].astype(float)
        ti_q_top = float(repaired_arrays["q_top_per_disorder"][li, qi, disorder_index])
        bridge_weights = np.asarray(record["bridge_weights"], dtype=np.float64)
        tv = float(0.5 * np.sum(np.abs(bridge_weights - ti_weights)))
        dq = float(abs(float(record["bridge_q_top"]) - ti_q_top))
        record["ti_q_top"] = ti_q_top
        record["ti_weights"] = ti_weights.tolist()
        record["ti_delta_f"] = ti_delta_f.tolist()
        record["tv_vs_ti"] = tv
        record["q_top_abs_diff_vs_ti"] = dq
        record["passed"] = bool(tv <= tv_threshold and dq <= dq_threshold)
    bridge_payload["overall_passed"] = bool(
        all(record["passed"] for record in bridge_payload["records"])
    )
    bridge_payload["ti_results_path"] = str(output_path.parent / "sector_ti_results.npz")
    bridge_payload["rebased_from_bridge_path"] = str(bridge_path)
    bridge_payload["thresholds"] = {
        **bridge_payload.get("thresholds", {}),
        "tv_vs_ti": float(tv_threshold),
        "q_top_abs_diff_vs_ti": float(dq_threshold),
    }
    output_path.write_text(
        json.dumps(bridge_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ti", type=Path, required=True)
    parser.add_argument("--replacement-ti", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--second-method-output", type=Path, required=True)
    parser.add_argument("--second-tv-threshold", type=float, default=0.03)
    parser.add_argument("--second-dq-threshold", type=float, default=0.02)
    args = parser.parse_args()

    base_manifest, base_arrays = _load_npz(args.base_ti)
    replacement_manifest, replacement_arrays = _load_npz(args.replacement_ti)
    base_lattice = base_arrays["lattice_size_list"].astype(int)
    base_q = base_arrays["q_values"].astype(float)
    repl_lattice = replacement_arrays["lattice_size_list"].astype(int)
    repl_q = replacement_arrays["q_values"].astype(float)
    l_index = {int(value): index for index, value in enumerate(base_lattice)}
    q_index = {_format_q(value): index for index, value in enumerate(base_q)}
    replacements = []
    for local_li, lattice_size in enumerate(repl_lattice):
        li = l_index[int(lattice_size)]
        for local_qi, q_value in enumerate(repl_q):
            qi = q_index[_format_q(q_value)]
            for disorder_index in range(replacement_arrays["q_top_per_disorder"].shape[2]):
                replacements.append({
                    "lattice_size": int(lattice_size),
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "old_q_top": float(base_arrays["q_top_per_disorder"][li, qi, disorder_index]),
                    "new_q_top": float(replacement_arrays["q_top_per_disorder"][local_li, local_qi, disorder_index]),
                    "old_flag": str(base_arrays["flags_per_disorder"][li, qi, disorder_index]),
                    "new_flag": str(replacement_arrays["flags_per_disorder"][local_li, local_qi, disorder_index]),
                })
                for name in REPLACED_ARRAYS:
                    base_arrays[name][li, qi, disorder_index] = (
                        replacement_arrays[name][local_li, local_qi, disorder_index]
                    )

    _recompute_aggregates(base_arrays)
    repaired_manifest = {
        **base_manifest,
        "mode": "stageF_targeted_repaired_sector_ti",
        "base_ti_path": str(args.base_ti),
        "replacement_ti_path": str(args.replacement_ti),
        "replacement_manifest": replacement_manifest,
        "targeted_replacements": replacements,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "sector_ti_results.npz",
        manifest_json=np.array(json.dumps(repaired_manifest, indent=2, sort_keys=True)),
        **base_arrays,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(repaired_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary(args.output_dir, base_arrays, replacements)
    args.second_method_output.parent.mkdir(parents=True, exist_ok=True)
    _write_rebased_bridge(
        bridge_path=args.bridge,
        repaired_arrays=base_arrays,
        output_path=args.second_method_output,
        tv_threshold=float(args.second_tv_threshold),
        dq_threshold=float(args.second_dq_threshold),
    )
    print(json.dumps({
        "num_replaced_records": len(replacements),
        "repaired_ti": str(args.output_dir / "sector_ti_results.npz"),
        "rebased_second_method": str(args.second_method_output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
