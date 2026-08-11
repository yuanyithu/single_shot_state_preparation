"""Publication report generation, outside the identity-bound package.

Corrected after the measurement, so it cannot live in `exp105_pipeline`
without invalidating the frozen `source_tree_sha256` that the configs, the
raw files and the published aggregate are all bound to. It reads the
aggregate through the loader and never touches raw, the decoder or the
seeds.

Original module docstring follows.

Publication artifacts for exp105: curves, contrasts, strata, plots, report."""

import csv
from pathlib import Path

import numpy as np

from data.expander_code.exp105.exp105_pipeline.aggregate import DISTANCE_STRATA
import json

from data.expander_code.exp105.exp105_pipeline.config import ensure_config
from data.expander_code.exp105.exp105_pipeline.crossing import CERTIFIED
from data.expander_code.exp105.exp105_pipeline.io import atomic_json, sha256_file


DECODER_LABEL = "BP+OSD-0"


def panel_counts(payload):
    """Per-m code counts, read from the published aggregate itself."""
    counts = json.loads(str(payload["codes_per_m_json"]))
    return {int(key): int(value) for key, value in counts.items()}


def panel_trials(payload):
    trials = json.loads(str(payload["trials_per_code_p_json"]))
    return {int(key): int(value) for key, value in trials.items()}


def _write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return sha256_file(path)


def write_primary_curves(path, payload):
    rows = []
    for m_index, m in enumerate(payload["m_values"].tolist()):
        for p_index, p in enumerate(payload["p_values"]):
            rows.append([
                m, f"{p:.4f}", payload["m_status"][m_index, p_index],
                f"{payload['primary_mean'][m_index, p_index]:.6f}",
                f"{payload['primary_band_low'][m_index, p_index]:.6f}",
                f"{payload['primary_band_high'][m_index, p_index]:.6f}",
                f"{payload['cluster_se'][m_index, p_index]:.6f}",
                f"{payload['between_code_std'][m_index, p_index]:.6f}",
                f"{payload['pooled_binomial_se'][m_index, p_index]:.6f}",
                int(payload["primary_failures"][m_index, p_index]),
                int(payload["primary_trials"][m_index, p_index]),
            ])
    return _write_csv(path, [
        "m", "p", "status", "primary_mean", "pointwise_low", "pointwise_high",
        "cluster_se", "between_code_std", "pooled_binomial_se", "failures", "trials",
    ], rows)


def write_contrasts(path, payload):
    ms = payload["m_values"].tolist()
    rows = []
    for p_index, p in enumerate(payload["p_values"]):
        row = [
            f"{p:.4f}",
            f"{payload['delta38'][p_index]:.6f}",
            f"{payload['delta38_band_low'][p_index]:.6f}",
            f"{payload['delta38_band_high'][p_index]:.6f}",
        ]
        certified = ""
        if payload["delta38_band_high"][p_index] < 0:
            certified = "certified_negative"
        elif payload["delta38_band_low"][p_index] > 0:
            certified = "certified_positive"
        row.append(certified)
        for adjacent in range(len(ms) - 1):
            row.append(f"{payload['adjacent_delta'][adjacent, p_index]:.6f}")
        rows.append(row)
    header = [
        "p", "delta38", "simultaneous_low", "simultaneous_high", "certification",
    ] + [
        f"delta{ms[i]}{ms[i + 1]}" for i in range(len(ms) - 1)
    ]
    return _write_csv(path, header, rows)


def write_distance_strata(path, payload):
    rows = []
    for m_index, m in enumerate(payload["m_values"].tolist()):
        for d_index, distance in enumerate(DISTANCE_STRATA):
            codes = int(payload["strata_code_counts"][m_index, d_index])
            if codes == 0:
                continue
            for p_index, p in enumerate(payload["p_values"]):
                rows.append([
                    m, distance, codes,
                    f"{codes / panel_counts(payload)[m]:.6f}", f"{p:.4f}",
                    int(payload["strata_failures"][m_index, d_index, p_index]),
                    int(payload["strata_trials"][m_index, d_index, p_index]),
                    f"{payload['strata_rate'][m_index, d_index, p_index]:.6f}",
                ])
    return _write_csv(path, [
        "m", "classical_distance", "codes", "ensemble_fraction", "p",
        "failures", "trials", "rate",
    ], rows)


def write_code_diagnostics(path, payload):
    rows = []
    for code_slot in range(payload["code_m"].shape[0]):
        for p_index, p in enumerate(payload["p_values"]):
            rows.append([
                int(payload["code_m"][code_slot]),
                int(payload["code_index"][code_slot]),
                int(payload["classical_distance"][code_slot]),
                f"{p:.4f}",
                payload["code_status"][code_slot, p_index],
                int(payload["failure_counts"][code_slot, p_index]),
                int(payload["trial_counts"][code_slot, p_index]),
                f"{payload['bp_convergence_rate'][code_slot, p_index]:.6f}",
                f"{payload['mean_bp_iterations'][code_slot, p_index]:.3f}",
                f"{payload['readout_mismatch_rate'][code_slot, p_index]:.6f}",
            ])
    return _write_csv(path, [
        "m", "code_index", "classical_distance", "p", "status", "failures",
        "trials", "bp_convergence_rate", "mean_bp_iterations",
        "readout_mismatch_rate",
    ], rows)


def write_composition(path, payload):
    rows = []
    for m_index, m in enumerate(payload["m_values"].tolist()):
        for d_index, distance in enumerate(DISTANCE_STRATA):
            codes = int(payload["strata_code_counts"][m_index, d_index])
            if codes:
                rows.append([
                    m, distance, codes,
                    f"{codes / panel_counts(payload)[m]:.6f}",
                ])
    return _write_csv(path, ["m", "classical_distance", "codes", "fraction"], rows)


def write_plots(directory, payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p_values = payload["p_values"]
    written = {}

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m_index, m in enumerate(payload["m_values"].tolist()):
        axes[0].plot(p_values, payload["primary_mean"][m_index], marker="o", label=f"m={m}")
        axes[0].fill_between(
            p_values, payload["primary_band_low"][m_index],
            payload["primary_band_high"][m_index], alpha=0.18,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("physical error rate p")
    axes[0].set_ylabel("ensemble mean block logical failure rate")
    axes[0].set_title(
        f"{DECODER_LABEL} ensemble mean, "
        f"{min(panel_counts(payload).values())}-{max(panel_counts(payload).values())} random codes per m"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(p_values, payload["delta38"], marker="o", color="crimson", label="Delta38")
    axes[1].fill_between(
        p_values, payload["delta38_band_low"], payload["delta38_band_high"],
        alpha=0.25, color="crimson", label="95% simultaneous band",
    )
    if np.isfinite(payload["p_cross"]):
        axes[1].axvline(payload["p_cross"], color="navy", linestyle="--", label="p_cross")
        axes[1].axvspan(
            payload["p_cross_low"], payload["p_cross_high"], color="navy", alpha=0.15,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("physical error rate p")
    axes[1].set_ylabel("P_fail(m=8) - P_fail(m=3)")
    axes[1].set_title(f"{DECODER_LABEL} primary contrast and its simultaneous band")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    path = directory / "primary_crossing.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written["primary_crossing.png"] = sha256_file(path)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for d_index, distance in enumerate(DISTANCE_STRATA):
        rates = payload["strata_rate"][:, d_index, :]
        if not np.isfinite(rates).any():
            continue
        for m_index, m in enumerate(payload["m_values"].tolist()):
            if m != payload["m_values"].tolist()[-1]:
                continue
            axes[0].plot(
                p_values, rates[m_index], marker="s", label=f"d={distance} (m=8)",
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("physical error rate p")
    axes[0].set_ylabel("block logical failure rate")
    axes[0].set_title("Distance strata at m=8 (preregistered secondary)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    width = 0.13
    for d_index, distance in enumerate(DISTANCE_STRATA):
        totals = np.asarray(
            [panel_counts(payload)[m] for m in payload["m_values"].tolist()],
            dtype=float,
        )
        fractions = payload["strata_code_counts"][:, d_index] / totals
        if not fractions.any():
            continue
        axes[1].bar(
            np.arange(len(payload["m_values"])) + d_index * width, fractions, width,
            label=f"d={distance}",
        )
    axes[1].set_xticks(np.arange(len(payload["m_values"])) + 2 * width)
    axes[1].set_xticklabels([str(m) for m in payload["m_values"].tolist()])
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("fraction of the ensemble")
    axes[1].set_title("Measured ensemble composition")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    figure.tight_layout()
    path = directory / "distance_strata.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written["distance_strata.png"] = sha256_file(path)
    return written


def write_report(directory, payload, config):
    config = ensure_config(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "primary_curves.csv": write_primary_curves(
            directory / "primary_curves.csv", payload),
        "crossing_contrasts.csv": write_contrasts(
            directory / "crossing_contrasts.csv", payload),
        "distance_strata.csv": write_distance_strata(
            directory / "distance_strata.csv", payload),
        "ensemble_composition.csv": write_composition(
            directory / "ensemble_composition.csv", payload),
        "code_diagnostics.csv": write_code_diagnostics(
            directory / "code_diagnostics.csv", payload),
    }
    files.update(write_plots(directory, payload))

    certified = payload["terminal_status"] == CERTIFIED
    lines = [
        "# exp105 ensemble crossing report",
        "",
        f"Terminal status: `{payload['terminal_status']}`",
        "",
        f"- Decoder: {DECODER_LABEL}, identity frozen byte for byte with "
        "`exp103.decoder_mc.v2`.",
        f"- Ensemble: {panel_counts(payload)} randomly generated codes per m, "
        f"m = {payload['m_values'][0]}..{payload['m_values'][-1]}, no post-selection.",
        f"- {panel_trials(payload)} trials per code and p over "
        f"{len(config['p_tokens'])} grid points.",
        f"- Simultaneous band half-width on the primary contrast: "
        f"`{payload['bootstrap_half_width']:.4f}`.",
        "",
    ]
    if certified:
        lines += [
            f"Certified bracket: `[{payload['crossing_bracket_low']:.2f}, "
            f"{payload['crossing_bracket_high']:.2f}]`.",
            "",
            f"Crossing location: `p_cross = {payload['p_cross']:.5f}` with 95% "
            f"bootstrap interval `[{payload['p_cross_low']:.5f}, "
            f"{payload['p_cross_high']:.5f}]` "
            f"(defined in {payload['p_cross_defined_fraction']:.3f} of replicates).",
        ]
    else:
        low = payload["delta38_band_low"]
        high = payload["delta38_band_high"]
        negative = int((high < 0).sum())
        positive = int((low > 0).sum())
        lines += ["No certified bracket and no crossing location.", ""]
        if positive == len(low) and negative == 0:
            # Worth separating from "nothing was found": every point is
            # certified on the same side, which is a result about the physics
            # rather than an absence of resolution.
            lines += [
                f"The primary contrast is **certified positive at all "
                f"{positive} grid points**: the simultaneous band excludes zero "
                "from below everywhere, so the larger code is worse than the "
                "smaller one at every p in the window. This is a certified "
                "absence of a crossing, not a failure to resolve one.",
            ]
        else:
            lines += [
                f"Certified negative at {negative} grid points and certified "
                f"positive at {positive}, with no negative point preceding a "
                "positive one.",
            ]
    bound = json.loads(str(payload["qtop_lower_bound_json"]))
    ms = payload["m_values"].tolist()
    best = max(
        (value, ms[index], float(payload["p_values"][j]))
        for index, m in enumerate(ms)
        for j, value in enumerate(bound[str(m)])
        if value is not None
    )
    lines += [
        "",
        "## Bound on q_top",
        "",
        "Per disorder the exact posterior satisfies `map_success <= "
        "sqrt(purity)`, and no decoder beats MAP success at its own "
        "observation, so with `S = 1 - P_fail` and `M = 2^k` Jensen gives "
        "`E[q_top] >= (M S^2 - 1)/(M - 1)`. This is a certified one-sided "
        "bound, never an estimate, and it is informative only where `S` is "
        f"large. Its strongest value here is `{best[0]:.5f}` at "
        f"`m = {best[1]}, p = {best[2]:g}`. Full table in `report.json`.",
        "",
        "## Scope",
        "",
        "Finite-grid, decoder-dependent result for one frozen "
        f"{DECODER_LABEL} decoder on one randomly generated expander-code "
        f"ensemble at q = {payload['q_token']}. No asymptotic threshold, no "
        "critical exponent, no finite-size scaling, no q_top *estimate* at "
        "m >= 4, no MLD and no preparation-channel claim. Clears no exp102 "
        "blocker.",
        "",
    ]
    report_md = directory / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="ascii")
    files["report.md"] = sha256_file(report_md)

    report = {
        "schema_version": "exp105.report.v1",
        "experiment_id": payload["experiment_id"],
        "config_sha256": payload["config_sha256"],
        "registry_sha256": payload["registry_sha256"],
        "source_commit": payload["source_commit"],
        "source_tree_sha256": payload["source_tree_sha256"],
        "decoder_binary_sha256": payload["decoder_binary_sha256"],
        "decoder": DECODER_LABEL,
        "overall_status": payload["overall_status"],
        "terminal_status": payload["terminal_status"],
        "crossing_bracket": [
            float(payload["crossing_bracket_low"]),
            float(payload["crossing_bracket_high"]),
        ],
        "bootstrap_half_width": float(payload["bootstrap_half_width"]),
        "p_cross": float(payload["p_cross"]),
        "p_cross_low": float(payload["p_cross_low"]),
        "p_cross_high": float(payload["p_cross_high"]),
        "p_cross_defined_fraction": float(payload["p_cross_defined_fraction"]),
        "p_cross_reason": payload["p_cross_reason"],
        "replay_status": payload["replay_status"],
        "replay_scope": payload["replay_scope"],
        "q_token": payload["q_token"],
        "trials_per_code_p": panel_trials(payload),
        "codes_per_m": panel_counts(payload),
        "qtop_lower_bound_json": payload["qtop_lower_bound_json"],
        "payload_sha256": payload["payload_sha256"],
        "file_sha256": files,
    }
    atomic_json(directory / "report.json", report)
    return report
