#!/usr/bin/env python3
"""exp41 FINAL phase diagram: q_c(p) for the 3D toric code with measurement noise.

Five L=7-anchored points (w0 sign-aware headline = L3-L7 crossing, 384 disorder,
NBOOT=10000) + even-moment q_W cross-check + q=0 anchor p_c~0.227 (3D RBIM) with
the near-vertical collapse window p in (0.22, 0.227) drawn as a schematic.

Sources: exp41/003 (p=0.11), /004 (p=0.21), /005 (p=0.22), /006 (p=0.05, 0.17);
L3-5(6) reused from exp40/004-005 where applicable.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = Path(__file__).resolve().parent

# w0 L3-L7 headline (q_c, CI95 lo, hi); q_W L3-L7 cross-check.
POINTS = {
    0.05: {"w0": (0.0320, 0.0304, 0.0350), "q_W": (0.0343, 0.0309, 0.0409), "src": "exp41/006"},
    0.11: {"w0": (0.0338, 0.0314, 0.0358), "q_W": (0.0400, 0.0327, 0.0420), "src": "exp41/003"},
    0.17: {"w0": (0.0351, 0.0284, 0.0367), "q_W": (0.0411, 0.0371, 0.0436), "src": "exp41/006"},
    0.21: {"w0": (0.0344, 0.0284, 0.0370), "q_W": (0.0405, 0.0365, 0.0429), "src": "exp41/004"},
    0.22: {"w0": (0.0349, 0.0327, 0.0363), "q_W": (0.0404, 0.0371, 0.0431), "src": "exp41/005"},
}
P_C = 0.227          # 3D RBIM q=0 anchor
BLUE, ORANGE = "#1f77b4", "#ff7f0e"   # validated pair (CVD dE=101)
INK, MUT = "#222222", "#666666"


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    ps = sorted(POINTS)

    for key, col, mk, mfc, lab in (
            ("w0", BLUE, "o", BLUE, "w0 = P(true class)  [sign-aware, headline]"),
            ("q_W", ORANGE, "^", "none", "q_W = mean m_u²  [even-moment, cross-check]")):
        y  = [POINTS[p][key][0] for p in ps]
        lo = [POINTS[p][key][0] - POINTS[p][key][1] for p in ps]
        hi = [POINTS[p][key][2] - POINTS[p][key][0] for p in ps]
        ax.errorbar(ps, y, yerr=[lo, hi], color=col, marker=mk, ms=8, mfc=mfc,
                    mew=1.6, lw=2, ls="-" if key == "w0" else "--",
                    capsize=3.5, label=lab, zorder=5 if key == "w0" else 4)

    # collapse window p in (0.22, p_c): schematic near-vertical drop + anchor
    ax.axvspan(0.22, P_C, color="0.85", alpha=0.55, zorder=1)
    ax.plot([0.22, P_C], [POINTS[0.22]["w0"][0], 0.0], color=MUT, ls=":", lw=2, zorder=3)
    ax.plot([P_C], [0.0], marker="*", ms=15, color=INK, zorder=6)
    ax.annotate("p_c ≈ 0.227  (3D RBIM, q=0)\ncollapse confined to p ∈ (0.22, 0.227)\n— schematic", xy=(P_C, 0.0),
                xytext=(0.168, 0.0075), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUT, lw=1))

    # direct series labels (relief for the orange contrast WARN) in ink
    ax.annotate("q_W (even-moment, +0.006 bias)", xy=(0.075, 0.0435), fontsize=9.5,
                color=INK, ha="left")
    ax.annotate("w0 (headline)", xy=(0.075, 0.0288), fontsize=10, color=INK,
                ha="left", fontweight="bold")

    # region labels
    ax.text(0.075, 0.012, "single-shot correctable\n(ordered)", fontsize=10.5,
            color=MUT, ha="center")
    ax.text(0.10, 0.0505, "not correctable (disordered)", fontsize=10.5,
            color=MUT, ha="center")

    ax.set_xlim(0.0, 0.245); ax.set_ylim(0.0, 0.054)
    ax.set_xlabel("p  (data error rate)", fontsize=11.5)
    ax.set_ylabel("q_c  (measurement error threshold)", fontsize=11.5)
    ax.grid(alpha=0.3, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.set_title("3D toric code, single-shot phase boundary   |   plateau q_c ≈ 0.033–0.035 across p ∈ [0.05, 0.22],\n"
                 "near-vertical collapse only inside p ∈ (0.22, p_c≈0.227)      [L=3..7 × 384 disorder, sector-TI]",
                 fontsize=10.5)

    out = SD / "phase_diagram_final.png"
    fig.savefig(out, dpi=200); plt.close(fig)

    table = {"headline": "w0 L3-L7 crossing, 384 disorder, NBOOT=10000",
             "p_c_anchor": P_C,
             "points": {str(p): {"w0_qc": POINTS[p]["w0"][0], "w0_CI95": list(POINTS[p]["w0"][1:]),
                                 "qW_qc": POINTS[p]["q_W"][0], "qW_CI95": list(POINTS[p]["q_W"][1:]),
                                 "source": POINTS[p]["src"]} for p in ps}}
    (SD / "phase_diagram_final.json").write_text(json.dumps(table, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
