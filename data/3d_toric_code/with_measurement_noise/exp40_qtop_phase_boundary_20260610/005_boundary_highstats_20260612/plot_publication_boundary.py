#!/usr/bin/env python3
"""Publication-quality measurement-error phase boundary q_c(p) for the 3D toric code.

Only the validated high-statistics data (exp40/005, 384 disorders/point, L=3,4,5,
TI / projection_mode=linear): q_top L3-L5 crossing (primary) + w0=P(true class)
L3-L5 crossing (sign-aware cross-check), both with 6000-disorder bootstrap CIs.
The q=0 endpoint p_c≈0.227 (3D RBIM Nishimori threshold) is marked as a known
anchor; no schematic closure line is drawn because the finite-size (L≤5) crossing
does not resolve the steep drop near p_c. No 48-disorder (low-biased) points.

Reads exp40/005 qc_table.json. Outputs phase_boundary_publication.{pdf,png}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

SD = Path(__file__).resolve().parent
PC0 = 0.227

rcParams.update({
    "font.size": 14, "axes.labelsize": 17, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3,
    "ytick.minor.size": 3, "xtick.top": True, "ytick.right": True,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})


def series(points, key):
    p, qc, lo, hi = [], [], [], []
    for r in points:
        c = r["crossings"][key]
        if c["q_c"] is None:
            continue
        p.append(r["p"]); qc.append(c["q_c"])
        lo.append(c["q_c"] - c["CI95"][0]); hi.append(c["CI95"][1] - c["q_c"])
    return np.array(p), np.array(qc), np.array([lo, hi])


def main():
    pts = sorted(json.loads((SD / "qc_table.json").read_text())["points"], key=lambda r: r["p"])
    pq, qq, eq = series(pts, "q_top_L35")
    pw, qw, ew = series(pts, "w0_L35")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)

    C_QTOP, C_W0, C_PC = "#0072B2", "#009E73", "#D55E00"

    # primary: q_top
    ax.errorbar(pq, qq, yerr=eq, marker="o", ms=8, mfc=C_QTOP, mec=C_QTOP,
                color=C_QTOP, lw=2.0, capsize=4, capthick=1.4, elinewidth=1.4,
                label=r"$q_{\mathrm{top}}$  ($L{=}3,5$ crossing)", zorder=4)
    # cross-check: w0
    ax.errorbar(pw, qw, yerr=ew, marker="s", ms=7.5, mfc="white", mec=C_W0,
                color=C_W0, lw=1.6, ls="--", capsize=4, capthick=1.2, elinewidth=1.2,
                label=r"$w_0 = P(\mathrm{true\ class})$  ($L{=}3,5$)", zorder=3)
    # known q=0 endpoint (3D RBIM); no connecting line
    ax.plot([PC0], [0.0], marker="*", ms=20, mfc=C_PC, mec="k", mew=0.6, ls="none",
            label=r"$q=0$ endpoint  $p_c\approx0.227$ (3D RBIM)", zorder=5)

    ax.set_xlabel(r"Pauli-$X$ error rate  $p$")
    ax.set_ylabel(r"measurement-error threshold  $q_c$")
    ax.set_xlim(0.0, 0.245)
    ax.set_ylim(0.0, 0.066)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax.grid(True, which="major", alpha=0.25, lw=0.7)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="0.7",
              handletextpad=0.6, borderpad=0.7)

    for ext in ("pdf", "png"):
        fig.savefig(SD / f"phase_boundary_publication.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_boundary_publication.{pdf,png}")
    print(f"q_top L3-L5: p={list(pq)}  q_c={list(qq)}")
    print(f"w0    L3-L5: p={list(pw)}  q_c={list(qw)}")


if __name__ == "__main__":
    main()
