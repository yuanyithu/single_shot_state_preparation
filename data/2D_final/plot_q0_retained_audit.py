#!/usr/bin/env python3
"""Rebuild audit-preview plots for the retained q=0 2D toric-code data.

This script intentionally excludes all q>0 legacy data.  It validates the
canonical input hashes, recorded source SHAs, manifest completion, tensor
shapes, and archived aggregate curves before producing any output.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CI_Z = 1.959963984540054


@dataclass(frozen=True)
class RunSpec:
    name: str
    npz_relpath: str
    manifest_relpath: str
    npz_sha256: str
    source_git_sha: str
    expected_lattice_sizes: tuple[int, ...]
    expected_num_p: int
    expected_num_disorder: int
    expected_manifest_chunks: int


RUN_SPECS = (
    RunSpec(
        name="q0_threshold_deep_nd3_20260420_221142",
        npz_relpath=(
            "data/2d_toric_code/without_measurement_noise/"
            "q0_threshold_deep_nd3_20260420_221142/"
            "scan_result_multi_L_q0_geometric_multistart_threshold_deep.npz"
        ),
        manifest_relpath=(
            "data/2d_toric_code/without_measurement_noise/"
            "q0_threshold_deep_nd3_20260420_221142/manifest.json"
        ),
        npz_sha256="f3821be7f779119603f1464b9f201ece184106d515f9f19baf5fc2db9a5f4f61",
        source_git_sha="a15c3326fcc07844e06cc02ff176cf39ab7c0bbb",
        expected_lattice_sizes=(3, 5, 7),
        expected_num_p=15,
        expected_num_disorder=512,
        expected_manifest_chunks=720,
    ),
    RunSpec(
        name="q0_control_extension_nd3_20260421_225303",
        npz_relpath=(
            "data/2d_toric_code/without_measurement_noise/"
            "q0_control_extension_nd3_20260421_225303/"
            "scan_result_multi_L_q0_control_extension.npz"
        ),
        manifest_relpath=(
            "data/2d_toric_code/without_measurement_noise/"
            "q0_control_extension_nd3_20260421_225303/manifest.json"
        ),
        npz_sha256="06254aa73b3e5c4596bdaf94d076e2c26c7427e43e8ba3b70789b49d199094ee",
        source_git_sha="a197215bd18e9ffc160b4864b7f54239ff4e39da",
        expected_lattice_sizes=(9, 11),
        expected_num_p=7,
        expected_num_disorder=1024,
        expected_manifest_chunks=448,
    ),
)


REQUIRED_FIELDS = {
    "data_error_probability_list",
    "disorder_q_top_values_tensor",
    "git_commit_sha",
    "lattice_size_list",
    "num_disorder_samples",
    "q0_num_start_chains",
    "q0_q_top_spread_per_disorder_tensor",
    "q0_q_top_values_per_disorder_per_start_tensor",
    "q0_start_sector_labels",
    "q_top_curve_matrix",
    "q_top_std_curve_matrix",
    "syndrome_error_probability",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: np.ndarray) -> str:
    return str(value.item())


def load_and_validate(spec: RunSpec) -> dict[str, Any]:
    npz_path = REPO_ROOT / spec.npz_relpath
    manifest_path = REPO_ROOT / spec.manifest_relpath
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical input: {spec.npz_relpath}. Restore the retained "
            "raw NPZ at this repository-relative path before plotting."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing canonical manifest: {spec.manifest_relpath}")

    actual_sha256 = sha256_file(npz_path)
    if actual_sha256 != spec.npz_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {spec.npz_relpath}: "
            f"expected {spec.npz_sha256}, got {actual_sha256}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    summary = manifest.get("summary", {})
    if len(chunks) != spec.expected_manifest_chunks:
        raise ValueError(f"Unexpected manifest chunk count for {spec.name}")
    if any(chunk.get("status") != "completed" for chunk in chunks):
        raise ValueError(f"Non-completed manifest chunk found for {spec.name}")
    if summary.get("completed_chunks") != spec.expected_manifest_chunks:
        raise ValueError(f"Manifest summary is incomplete for {spec.name}")
    if summary.get("failed_chunks") != 0 or summary.get("pending_chunks") != 0:
        raise ValueError(f"Manifest records failed or pending work for {spec.name}")
    if manifest.get("git_commit_sha") != spec.source_git_sha:
        raise ValueError(f"Manifest source SHA mismatch for {spec.name}")

    with np.load(npz_path, allow_pickle=False) as archive:
        missing = REQUIRED_FIELDS.difference(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ fields for {spec.name}: {sorted(missing)}")
        data = {key: np.array(archive[key], copy=True) for key in archive.files}

    lattice_sizes = data["lattice_size_list"].astype(int)
    p_values = data["data_error_probability_list"].astype(float)
    num_disorder = int(data["num_disorder_samples"].item())
    expected_curve_shape = (len(spec.expected_lattice_sizes), spec.expected_num_p)
    expected_tensor_shape = (*expected_curve_shape, spec.expected_num_disorder)
    expected_start_shape = (*expected_tensor_shape, 4)

    if tuple(lattice_sizes.tolist()) != spec.expected_lattice_sizes:
        raise ValueError(f"Unexpected lattice sizes for {spec.name}")
    if p_values.shape != (spec.expected_num_p,):
        raise ValueError(f"Unexpected p-grid shape for {spec.name}")
    if num_disorder != spec.expected_num_disorder:
        raise ValueError(f"Unexpected disorder count for {spec.name}")
    if data["disorder_q_top_values_tensor"].shape != expected_tensor_shape:
        raise ValueError(f"Unexpected per-disorder q_top shape for {spec.name}")
    if data["q0_q_top_values_per_disorder_per_start_tensor"].shape != expected_start_shape:
        raise ValueError(f"Unexpected per-start q_top shape for {spec.name}")
    if data["q0_q_top_spread_per_disorder_tensor"].shape != expected_tensor_shape:
        raise ValueError(f"Unexpected start-spread shape for {spec.name}")
    if int(data["q0_num_start_chains"].item()) != 4:
        raise ValueError(f"Expected four q=0 start chains for {spec.name}")
    if not np.isclose(float(data["syndrome_error_probability"].item()), 0.0):
        raise ValueError(f"Expected q=0 input for {spec.name}")
    if scalar_text(data["git_commit_sha"]) != spec.source_git_sha:
        raise ValueError(f"NPZ source SHA mismatch for {spec.name}")

    per_disorder = data["disorder_q_top_values_tensor"].astype(float)
    archived_mean = data["q_top_curve_matrix"].astype(float)
    archived_std = data["q_top_std_curve_matrix"].astype(float)
    rebuilt_mean = per_disorder.mean(axis=2)
    rebuilt_std = per_disorder.std(axis=2, ddof=1)
    if not np.allclose(archived_mean, rebuilt_mean, rtol=0.0, atol=1e-14):
        raise ValueError(f"Archived q_top mean mismatch for {spec.name}")
    if not np.allclose(archived_std, rebuilt_std, rtol=0.0, atol=1e-14):
        raise ValueError(f"Archived q_top std mismatch for {spec.name}")

    per_start = data["q0_q_top_values_per_disorder_per_start_tensor"].astype(float)
    rebuilt_spread = np.ptp(per_start, axis=3)
    archived_spread = data["q0_q_top_spread_per_disorder_tensor"].astype(float)
    if not np.allclose(archived_spread, rebuilt_spread, rtol=0.0, atol=1e-14):
        raise ValueError(f"Archived four-start spread mismatch for {spec.name}")

    return {
        "spec": spec,
        "sha256": actual_sha256,
        "manifest": manifest,
        "p": p_values,
        "lattice_sizes": lattice_sizes,
        "q_top": rebuilt_mean,
        "q_top_std": rebuilt_std,
        "q_top_sem": rebuilt_std / np.sqrt(num_disorder),
        "spread": archived_spread.mean(axis=2),
        "spread_std": archived_spread.std(axis=2, ddof=1),
        "spread_sem": archived_spread.std(axis=2, ddof=1) / np.sqrt(num_disorder),
        "num_disorder": num_disorder,
        "start_labels": data["q0_start_sector_labels"].astype(str).tolist(),
    }


def flatten_by_lattice(runs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for run in runs:
        for index, lattice_size in enumerate(run["lattice_sizes"]):
            result[int(lattice_size)] = {
                "run": run,
                "p": run["p"],
                "q_top": run["q_top"][index],
                "q_top_std": run["q_top_std"][index],
                "q_top_sem": run["q_top_sem"][index],
                "spread": run["spread"][index],
                "spread_std": run["spread_std"][index],
                "spread_sem": run["spread_sem"][index],
                "num_disorder": run["num_disorder"],
            }
    return result


def plot_all_sizes(series: dict[int, dict[str, Any]]) -> None:
    colors = {3: "#4477AA", 5: "#66A61E", 7: "#D95F02", 9: "#7570B3", 11: "#C51B7D"}
    markers = {3: "o", 5: "s", 7: "^", 9: "D", 11: "P"}
    fig, (axis_curve, axis_spread) = plt.subplots(
        2,
        1,
        figsize=(8.3, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (2.15, 1.0), "hspace": 0.08},
        constrained_layout=False,
    )

    for lattice_size in sorted(series):
        item = series[lattice_size]
        p = item["p"]
        mean = item["q_top"]
        half_width = CI_Z * item["q_top_sem"]
        spread = item["spread"]
        spread_half_width = CI_Z * item["spread_sem"]
        label = f"L={lattice_size} (n={item['num_disorder']})"
        color = colors[lattice_size]
        axis_curve.fill_between(
            p,
            mean - half_width,
            mean + half_width,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        axis_curve.plot(
            p,
            mean,
            color=color,
            marker=markers[lattice_size],
            markersize=4.2,
            linewidth=1.45,
            label=label,
        )
        axis_spread.fill_between(
            p,
            spread - spread_half_width,
            spread + spread_half_width,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        axis_spread.plot(
            p,
            spread,
            color=color,
            marker=markers[lattice_size],
            markersize=3.7,
            linewidth=1.3,
        )

    for axis in (axis_curve, axis_spread):
        axis.axvline(0.11, color="0.4", linestyle="--", linewidth=0.9)
        axis.axvspan(0.11125, 0.1255, color="#F2C14E", alpha=0.10)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.set_xlim(0.089, 0.126)
    axis_curve.text(
        0.1182,
        0.704,
        "L=9,11 not sampled",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#7A5C00",
    )
    axis_curve.set_ylabel(r"$q_{\mathrm{top}}$ disorder mean")
    axis_curve.set_ylim(0.30, 0.735)
    axis_curve.legend(ncol=2, loc="lower left", frameon=False, fontsize=8.5)
    axis_spread.set_ylabel("mean four-start\nmax-min spread")
    axis_spread.set_xlabel(r"data-error probability $p$ (syndrome error $q=0$)")
    axis_spread.set_ylim(bottom=0.0)
    fig.suptitle(
        "AUDIT PREVIEW — retained q=0 data only (not a final paper figure)",
        y=0.985,
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.012,
        "Bands are pointwise 95% normal CIs across disorder. "
        "Start sensitivity is retained as a required diagnostic.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.115, top=0.93)
    fig.savefig(
        OUTPUT_DIR / "q0_retained_audit_preview.png",
        dpi=240,
        metadata={"Software": "plot_q0_retained_audit.py"},
    )
    fig.savefig(
        OUTPUT_DIR / "q0_retained_audit_preview.pdf",
        dpi=240,
        metadata={
            "Creator": "plot_q0_retained_audit.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def align_pair(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_common = np.array(sorted(set(np.round(first["p"], 10)).intersection(np.round(second["p"], 10))))
    first_indices = [int(np.flatnonzero(np.isclose(first["p"], value))[0]) for value in p_common]
    second_indices = [int(np.flatnonzero(np.isclose(second["p"], value))[0]) for value in p_common]
    difference = second["q_top"][second_indices] - first["q_top"][first_indices]
    difference_sem = np.sqrt(
        first["q_top_sem"][first_indices] ** 2 + second["q_top_sem"][second_indices] ** 2
    )
    return p_common, difference, difference_sem


def plot_large_lattice_gap(series: dict[int, dict[str, Any]]) -> dict[str, Any]:
    colors = {7: "#D95F02", 9: "#7570B3", 11: "#C51B7D"}
    markers = {7: "^", 9: "D", 11: "P"}
    fig, (axis_curve, axis_difference) = plt.subplots(
        2,
        1,
        figsize=(7.8, 6.4),
        sharex=False,
        gridspec_kw={"height_ratios": (1.7, 1.0), "hspace": 0.25},
    )
    for lattice_size in (7, 9, 11):
        item = series[lattice_size]
        half_width = CI_Z * item["q_top_sem"]
        axis_curve.fill_between(
            item["p"],
            item["q_top"] - half_width,
            item["q_top"] + half_width,
            color=colors[lattice_size],
            alpha=0.14,
            linewidth=0,
        )
        axis_curve.plot(
            item["p"],
            item["q_top"],
            color=colors[lattice_size],
            marker=markers[lattice_size],
            markersize=4.5,
            linewidth=1.5,
            label=f"L={lattice_size}",
        )
    axis_curve.axvspan(0.11125, 0.1255, color="#F2C14E", alpha=0.12)
    axis_curve.axvline(0.11, color="0.4", linestyle="--", linewidth=0.9)
    axis_curve.text(
        0.1182,
        0.666,
        "recommended extension:\nL=9,11 at p=0.1125…0.1250",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#7A5C00",
    )
    axis_curve.set_xlim(0.094, 0.126)
    axis_curve.set_ylim(0.32, 0.72)
    axis_curve.set_xlabel(r"data-error probability $p$")
    axis_curve.set_ylabel(r"$q_{\mathrm{top}}$ disorder mean")
    axis_curve.legend(frameon=False, ncol=3, loc="lower left")
    axis_curve.grid(True, alpha=0.22, linewidth=0.7)

    pair_results: dict[str, Any] = {}
    for lower, upper, color, marker in (
        (7, 9, "#3B6FB6", "o"),
        (9, 11, "#A23B72", "s"),
    ):
        p_common, difference, difference_sem = align_pair(series[lower], series[upper])
        half_width = CI_Z * difference_sem
        label = rf"$q_{{\mathrm{{top}}}}(L={upper})-q_{{\mathrm{{top}}}}(L={lower})$"
        axis_difference.errorbar(
            p_common,
            difference,
            yerr=half_width,
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.2,
            capsize=2.4,
            label=label,
        )
        key = f"L{upper}_minus_L{lower}"
        pair_results[key] = {
            "p": p_common.tolist(),
            "difference": difference.tolist(),
            "ci95_low": (difference - half_width).tolist(),
            "ci95_high": (difference + half_width).tolist(),
            "point_estimate_bracketed": bool(np.any(difference[:-1] * difference[1:] <= 0)),
        }
    axis_difference.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9)
    axis_difference.grid(True, alpha=0.22, linewidth=0.7)
    axis_difference.set_xlim(0.094, 0.111)
    axis_difference.set_xlabel(r"data-error probability $p$")
    axis_difference.set_ylabel("adjacent-size\ndifference")
    axis_difference.legend(frameon=False, fontsize=8.3, loc="lower left")
    fig.suptitle(
        "AUDIT PREVIEW — large-L coverage and crossing diagnostic",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.012,
        "Difference error bars use independent-disorder standard errors. "
        "No finite-size crossing is claimed from this preview.",
        ha="center",
        fontsize=7.8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.92)
    fig.savefig(
        OUTPUT_DIR / "q0_retained_large_L_gap_preview.png",
        dpi=240,
        metadata={"Software": "plot_q0_retained_audit.py"},
    )
    fig.savefig(
        OUTPUT_DIR / "q0_retained_large_L_gap_preview.pdf",
        dpi=240,
        metadata={
            "Creator": "plot_q0_retained_audit.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return pair_results


def write_summary_csv(runs: list[dict[str, Any]]) -> None:
    output_path = OUTPUT_DIR / "q0_retained_summary.csv"
    fieldnames = [
        "run",
        "canonical_source",
        "input_sha256",
        "source_git_sha",
        "lattice_size",
        "p",
        "num_disorder",
        "q_top_mean",
        "q_top_std",
        "q_top_sem",
        "q_top_ci95_low",
        "q_top_ci95_high",
        "four_start_spread_mean",
        "four_start_spread_std",
        "four_start_spread_sem",
        "four_start_spread_ci95_low",
        "four_start_spread_ci95_high",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            spec = run["spec"]
            for lattice_index, lattice_size in enumerate(run["lattice_sizes"]):
                for p_index, p_value in enumerate(run["p"]):
                    q_mean = float(run["q_top"][lattice_index, p_index])
                    q_std = float(run["q_top_std"][lattice_index, p_index])
                    q_sem = float(run["q_top_sem"][lattice_index, p_index])
                    spread_mean = float(run["spread"][lattice_index, p_index])
                    spread_std = float(run["spread_std"][lattice_index, p_index])
                    spread_sem = float(run["spread_sem"][lattice_index, p_index])
                    writer.writerow(
                        {
                            "run": spec.name,
                            "canonical_source": spec.npz_relpath,
                            "input_sha256": run["sha256"],
                            "source_git_sha": spec.source_git_sha,
                            "lattice_size": int(lattice_size),
                            "p": f"{p_value:.10g}",
                            "num_disorder": run["num_disorder"],
                            "q_top_mean": f"{q_mean:.17g}",
                            "q_top_std": f"{q_std:.17g}",
                            "q_top_sem": f"{q_sem:.17g}",
                            "q_top_ci95_low": f"{q_mean - CI_Z * q_sem:.17g}",
                            "q_top_ci95_high": f"{q_mean + CI_Z * q_sem:.17g}",
                            "four_start_spread_mean": f"{spread_mean:.17g}",
                            "four_start_spread_std": f"{spread_std:.17g}",
                            "four_start_spread_sem": f"{spread_sem:.17g}",
                            "four_start_spread_ci95_low": f"{spread_mean - CI_Z * spread_sem:.17g}",
                            "four_start_spread_ci95_high": f"{spread_mean + CI_Z * spread_sem:.17g}",
                        }
                    )


def write_audit_json(
    runs: list[dict[str, Any]],
    series: dict[int, dict[str, Any]],
    pair_results: dict[str, Any],
) -> None:
    audit = {
        "schema_version": 1,
        "scope": "retained q=0 audit preview only",
        "paper_figure": False,
        "q_positive_data_included": False,
        "ci_method": "pointwise normal 95% CI across disorder; z=1.959963984540054",
        "inputs": [
            {
                "run": run["spec"].name,
                "canonical_source": run["spec"].npz_relpath,
                "manifest": run["spec"].manifest_relpath,
                "input_sha256": run["sha256"],
                "source_git_sha": run["spec"].source_git_sha,
                "lattice_sizes": run["lattice_sizes"].tolist(),
                "p": run["p"].tolist(),
                "num_disorder": run["num_disorder"],
                "start_labels": run["start_labels"],
                "manifest_completed_chunks": run["manifest"]["summary"]["completed_chunks"],
                "manifest_failed_chunks": run["manifest"]["summary"]["failed_chunks"],
                "manifest_pending_chunks": run["manifest"]["summary"]["pending_chunks"],
            }
            for run in runs
        ],
        "adjacent_size_differences": pair_results,
        "coverage_gap": {
            "lattice_sizes": [9, 11],
            "missing_p": [0.1125, 0.115, 0.1175, 0.12, 0.1225, 0.125],
            "purpose": "close or explicitly exclude the L7-L9 crossing window",
        },
        "diagnostics": {
            "maximum_mean_four_start_spread_by_lattice_size": {
                str(lattice_size): float(np.max(item["spread"]))
                for lattice_size, item in sorted(series.items())
            }
        },
        "outputs": [
            "data/2D_final/q0_retained_summary.csv",
            "data/2D_final/q0_retained_audit_preview.png",
            "data/2D_final/q0_retained_audit_preview.pdf",
            "data/2D_final/q0_retained_large_L_gap_preview.png",
            "data/2D_final/q0_retained_large_L_gap_preview.pdf",
        ],
    }
    (OUTPUT_DIR / "q0_retained_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    runs = [load_and_validate(spec) for spec in RUN_SPECS]
    series = flatten_by_lattice(runs)
    if set(series) != {3, 5, 7, 9, 11}:
        raise ValueError(f"Unexpected retained lattice-size set: {sorted(series)}")
    write_summary_csv(runs)
    plot_all_sizes(series)
    pair_results = plot_large_lattice_gap(series)
    write_audit_json(runs, series, pair_results)
    print("Validated two retained q=0 inputs and wrote audit-preview assets to data/2D_final/.")


if __name__ == "__main__":
    main()
