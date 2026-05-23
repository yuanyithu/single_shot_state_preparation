#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROJECT_ROOT = Path("/Users/jarvis/Desktop/sync/project D")
DEFAULT_SNAPSHOT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "data/3d_toric_code/with_measurement_noise/"
    "exp34_fixed_p050_q000_080_L34567_corrected_observable_20260524_final_stopped_after_L6q060_nd12"
)
DEFAULT_HOST_TAGS = ("nd1", "nd2")
DEFAULT_Q_VALUES = (
    "0.0000",
    "0.0100",
    "0.0200",
    "0.0300",
    "0.0400",
    "0.0500",
    "0.0600",
    "0.0700",
    "0.0800",
)
DEFAULT_LATTICE_SIZES = (3, 4, 5, 6, 7)
CI95_Z = 1.96


def complete_manifest(path):
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary = manifest.get("summary", {})
    total = int(summary.get("total_chunks", 0))
    return (
        total > 0
        and int(summary.get("completed_chunks", 0)) == total
        and int(summary.get("failed_chunks", 0)) == 0
        and int(summary.get("pending_chunks", 0)) == 0
    )


def find_final_npz(point_dir):
    matches = sorted(point_dir.glob("scan_result_*.npz"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"multiple final NPZ files in {point_dir}: {matches}")
    return matches[0]


def load_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def point_values(path):
    data = load_npz(path)
    values = np.asarray(data["disorder_q_top_values_tensor"], dtype=np.float64)
    if values.shape[0] != 1 or values.shape[1] != 1:
        raise ValueError(f"{path} is not a single L,p point")
    return values.reshape(-1)


def q_from_tag(q_tag):
    return float(q_tag.replace("p", "."))


def probability_tag(value):
    return f"{float(value):.4f}".replace(".", "p")


def parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot pooled exp34 q_top(q) curves from complete per-node final NPZ files."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--host-tags", default=",".join(DEFAULT_HOST_TAGS))
    parser.add_argument("--q-values", default=",".join(DEFAULT_Q_VALUES))
    parser.add_argument(
        "--lattice-sizes",
        default=",".join(str(value) for value in DEFAULT_LATTICE_SIZES),
    )
    parser.add_argument("--fixed-p", type=float, default=0.05)
    parser.add_argument(
        "--stem",
        default="fixed_p050_q000_080_exp34_final_stopped_after_L6q060_nd12_qtop_by_L",
    )
    parser.add_argument("--status-label", default="final_stopped_after_L6q060")
    parser.add_argument(
        "--title",
        default="exp34 fixed p=0.0500, pooled nd1+nd2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    host_tags = parse_csv(args.host_tags)
    q_tags = tuple(probability_tag(value) for value in parse_csv(args.q_values))
    lattice_sizes = tuple(int(value) for value in parse_csv(args.lattice_sizes))

    rows = []
    skipped = []
    for lattice_size in lattice_sizes:
        for q_tag in q_tags:
            source_paths = []
            reasons = []
            for host_tag in host_tags:
                point_dir = (
                    snapshot_dir
                    / "remote_runs"
                    / host_tag
                    / f"L{lattice_size}"
                    / f"q_{q_tag}"
                )
                manifest_path = point_dir / "manifest.json"
                if not complete_manifest(manifest_path):
                    reasons.append(f"{host_tag}:incomplete_or_missing_manifest")
                    continue
                final_npz = find_final_npz(point_dir)
                if final_npz is None:
                    reasons.append(f"{host_tag}:missing_final_npz")
                    continue
                source_paths.append(final_npz)

            if len(source_paths) != len(host_tags):
                skipped.append(
                    {
                        "L": lattice_size,
                        "q": q_from_tag(q_tag),
                        "q_tag": q_tag,
                        "reasons": reasons,
                    }
                )
                continue

            samples = np.concatenate([point_values(path) for path in source_paths])
            q_top = float(np.mean(samples))
            std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
            ci95 = 0.0 if samples.size <= 1 else CI95_Z * std / math.sqrt(samples.size)
            rows.append(
                {
                    "L": lattice_size,
                    "q": q_from_tag(q_tag),
                    "q_top": q_top,
                    "q_top_ci95": ci95,
                    "num_disorder_samples": int(samples.size),
                    "source_npz_paths": [
                        str(path.relative_to(snapshot_dir)) for path in source_paths
                    ],
                }
            )

    analysis_dir = snapshot_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem
    png_path = analysis_dir / f"{stem}.png"
    pdf_path = analysis_dir / f"{stem}.pdf"
    csv_path = analysis_dir / f"{stem}_points.csv"
    summary_path = analysis_dir / f"{stem}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("L", "q", "q_top", "q_top_ci95", "num_disorder_samples"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    summary = {
        "status": args.status_label,
        "fixed_p": float(args.fixed_p),
        "host_tags": list(host_tags),
        "num_pooled_points": len(rows),
        "skipped_points": skipped,
        "note": (
            "Direct pooled q_top over disorder samples from complete per-node final "
            "NPZ files only; running chunks were not copied."
        ),
        "rows": rows,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    colors = {
        3: "#0B5CAD",
        4: "#C46A00",
        5: "#1F8A4C",
        6: "#7A4EAB",
        7: "#A8323E",
    }
    for lattice_size in sorted({row["L"] for row in rows}):
        series = sorted(
            (row for row in rows if row["L"] == lattice_size),
            key=lambda row: row["q"],
        )
        x_values = np.asarray([row["q"] for row in series], dtype=np.float64)
        y_values = np.asarray([row["q_top"] for row in series], dtype=np.float64)
        y_errors = np.asarray(
            [row["q_top_ci95"] for row in series],
            dtype=np.float64,
        )
        label = f"L={lattice_size}"
        if len(series) < len(q_tags):
            label += f" ({len(series)} pts)"
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            markersize=4.4,
            linewidth=1.8,
            capsize=2.8,
            color=colors.get(lattice_size),
            label=label,
        )
    ax.set_xlabel("measurement noise q")
    ax.set_ylabel("q_top")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.28)
    ax.legend(title="lattice size", frameon=False)
    ax.set_xlim(-0.002, 0.082)
    if rows:
        ymin = min(row["q_top"] - row["q_top_ci95"] for row in rows)
        ymax = max(row["q_top"] + row["q_top_ci95"] for row in rows)
        pad = max(0.004, 0.05 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), min(1.005, ymax + pad))
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    readme_path = snapshot_dir / "README.md"
    readme_path.write_text(
        "# exp34 fixed p=0.0500 q scan final stopped snapshot\n\n"
        "- Snapshot: 2026-05-24, after stopping remote production at the L6 q=0.0600 checkpoint.\n"
        "- Safety: copied only manifests, final scan_result NPZ/PNG files, and logs; chunks from running jobs were not copied.\n"
        f"- Plot: `analysis/{png_path.name}`.\n"
        f"- Summary: `analysis/{summary_path.name}`.\n"
        f"- Pooled complete points: `{len(rows)}`.\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "num_pooled_points": len(rows),
                "pdf_path": str(pdf_path),
                "plot_path": str(png_path),
                "summary_path": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
