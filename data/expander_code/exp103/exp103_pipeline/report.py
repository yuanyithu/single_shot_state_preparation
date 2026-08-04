import csv
import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import atomic_json, sha256_file
from .loader import load_exp103_crossing


FINAL_REPORT_FILENAMES = (
    "primary_curves.csv",
    "code_diagnostics.csv",
    "distance_strata_secondary.csv",
    "primary_crossing.png",
    "per_code_secondary.png",
    "report.json",
    "report.md",
)


def _require_new_path(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"final report evidence is immutable: {path}")


def _require_new_report_targets(output_dir):
    output_dir = Path(output_dir)
    for filename in FINAL_REPORT_FILENAMES:
        _require_new_path(output_dir / filename)


def _atomic_csv(path, fieldnames, rows):
    path = Path(path)
    _require_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_figure(path, figure):
    path = Path(path)
    _require_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_final_report(result_path, output_dir):
    output_dir = Path(output_dir)
    _require_new_report_targets(output_dir)
    data = load_exp103_crossing(result_path)
    if data["overall_status"] != "COMPLETE":
        raise ValueError("formal final reports require all 624 code-p cells to be reportable")
    output_dir.mkdir(parents=True, exist_ok=True)
    p_values = data["p_values"]
    primary_rows = []
    for m_index, m in enumerate(data["m_values"]):
        for p_index, p in enumerate(p_values):
            primary_rows.append({
                "m": int(m), "p": f"{p:.2f}",
                "primary_mean": f"{data['primary_mean'][m_index, p_index]:.12g}",
                "simultaneous_low": f"{data['primary_band_low'][m_index, p_index]:.12g}",
                "simultaneous_high": f"{data['primary_band_high'][m_index, p_index]:.12g}",
                "median_secondary": f"{data['primary_median'][m_index, p_index]:.12g}",
                "fixed_panel_mc_se": f"{data['fixed_panel_mc_se'][m_index, p_index]:.12g}",
                "between_code_sem": f"{data['between_code_sem'][m_index, p_index]:.12g}",
            })
    _atomic_csv(output_dir / "primary_curves.csv", list(primary_rows[0]), primary_rows)
    code_rows = []
    for code_index, code_id in enumerate(data["code_ids"]):
        for p_index, p in enumerate(p_values):
            code_rows.append({
                "code_id": str(code_id),
                "m": int(data["code_m"][code_index]),
                "classical_distance": int(data["classical_distance"][code_index]),
                "p": f"{p:.2f}",
                "failures": int(data["failure_counts"][code_index, p_index]),
                "trials": int(data["trial_counts"][code_index, p_index]),
                "rate": f"{data['code_rates'][code_index, p_index]:.12g}",
                "wilson_low": f"{data['wilson_low'][code_index, p_index]:.12g}",
                "wilson_high": f"{data['wilson_high'][code_index, p_index]:.12g}",
                "bp_convergence_rate": f"{data['bp_convergence_rate'][code_index, p_index]:.12g}",
                "mean_bp_iterations": f"{data['mean_bp_iterations'][code_index, p_index]:.12g}",
                "syndrome_mismatch_rate": f"{data['syndrome_mismatch_rate'][code_index, p_index]:.12g}",
                "mean_logical_weight": f"{data['mean_logical_weight'][code_index, p_index]:.12g}",
            })
    _atomic_csv(output_dir / "code_diagnostics.csv", list(code_rows[0]), code_rows)
    strata_rows = []
    for distance in sorted(set(int(value) for value in data["classical_distance"])):
        selected = data["classical_distance"] == distance
        for p_index, p in enumerate(p_values):
            values = data["code_rates"][selected, p_index]
            strata_rows.append({
                "classical_distance": distance,
                "num_codes": int(selected.sum()),
                "p": f"{p:.2f}",
                "mean_secondary": f"{values.mean():.12g}",
                "median_secondary": f"{np.median(values):.12g}",
                "std_secondary": f"{np.std(values, ddof=1):.12g}" if len(values) > 1 else "nan",
            })
    _atomic_csv(output_dir / "distance_strata_secondary.csv", list(strata_rows[0]), strata_rows)

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, 6))
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for m_index, (m, color) in enumerate(zip(data["m_values"], colors)):
        axis.plot(p_values, data["primary_mean"][m_index], marker="o", color=color, label=f"m={m}")
        axis.fill_between(
            p_values, data["primary_band_low"][m_index], data["primary_band_high"][m_index],
            color=color, alpha=0.09,
        )
    if np.isfinite(data["crossing_bracket_low"]):
        axis.axvspan(data["crossing_bracket_low"], data["crossing_bracket_high"], color="#d97706", alpha=0.18)
    axis.set(xlabel="physical X error probability p", ylabel="block logical failure rate",
             title="exp103 BpLSD finite-size decoder crossing")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    _save_figure(output_dir / "primary_crossing.png", figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    for m_index, axis in enumerate(axes.ravel()):
        code_slice = slice(8 * m_index, 8 * (m_index + 1))
        for offset, values in enumerate(data["code_rates"][code_slice]):
            axis.plot(p_values, values, alpha=0.65, linewidth=1.0, label=f"c{offset:02d}")
        axis.plot(p_values, data["primary_mean"][m_index], color="black", linewidth=2.0, label="mean")
        axis.plot(p_values, data["primary_median"][m_index], color="#d97706", linestyle="--", label="median")
        axis.set_title(f"m={data['m_values'][m_index]}")
        axis.grid(alpha=0.15)
    axes[1, 0].set_xlabel("p")
    axes[1, 1].set_xlabel("p")
    axes[1, 2].set_xlabel("p")
    axes[0, 0].set_ylabel("failure rate")
    axes[1, 0].set_ylabel("failure rate")
    axes[0, 0].legend(ncol=3, fontsize=7)
    figure.suptitle("Frozen per-code curves (secondary)")
    _save_figure(output_dir / "per_code_secondary.png", figure)

    summary = {
        "schema_version": "exp103.final_report.v1",
        "aggregate_sha256": sha256_file(result_path),
        "terminal_status": data["terminal_status"],
        "crossing_bracket": None if not np.isfinite(data["crossing_bracket_low"]) else [
            data["crossing_bracket_low"], data["crossing_bracket_high"],
        ],
        "compatible_triple": json.loads(data["compatible_triple_json"]),
        "bootstrap_half_width": data["bootstrap_half_width"],
        "num_code_p": 624,
        "trials_per_code_p": 10000,
        "total_trials": int(data["trial_counts"].sum()),
        "authority": "finite_grid_bplsd_decoder_crossing_only",
        "exp102_blockers_cleared": [],
        "files": [
            "primary_curves.csv", "code_diagnostics.csv", "distance_strata_secondary.csv",
            "primary_crossing.png", "per_code_secondary.png",
        ],
    }
    _require_new_path(output_dir / "report.json")
    atomic_json(output_dir / "report.json", summary)
    bracket_text = "none" if summary["crossing_bracket"] is None else (
        f"[{summary['crossing_bracket'][0]:.2f}, {summary['crossing_bracket'][1]:.2f}]"
    )
    markdown = "\n".join([
        "# exp103 final decoder-MC report", "",
        f"- Terminal status: `{summary['terminal_status']}`",
        f"- Certified finite-grid bracket: `{bracket_text}`",
        f"- Complete panel: `624/624` code-p cells, `{summary['total_trials']}` trials",
        "- Primary: equal-weight mean over all eight frozen codes at each m",
        "- Secondary: medians, per-code curves, classical-distance strata and BP diagnostics",
        "- Scope: BpLSD decoder crossing only; no asymptotic p_c, q_top, MLD or preparation claim",
        "- Authority: clears no exp102 blocker and authorizes no exp102 stage", "",
    ])
    report_md = output_dir / "report.md"
    _require_new_path(report_md)
    report_md.write_text(markdown, encoding="ascii")
    return summary
