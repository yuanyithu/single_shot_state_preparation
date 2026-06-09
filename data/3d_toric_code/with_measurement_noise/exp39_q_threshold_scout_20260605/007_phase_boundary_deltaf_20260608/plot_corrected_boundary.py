#!/usr/bin/env python3
"""Clean corrected phase-boundary figure: q_c(p) from the sign-aware w0 crossing
(with bootstrap CI), the old biased Δf-gap boundary for contrast, the q=0
endpoint, and ordered/disordered shading. Reads boundary_corrected_summary.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path(__file__).resolve().parent
PC0 = 0.227
S = json.loads((SD / "boundary_corrected_summary.json").read_text())
pts = sorted(S["points"], key=lambda r: r["p"])


def series(est):
    p, qc, lo, hi = [], [], [], []
    for r in pts:
        c = r["crossings"][est]
        if c["q_c"] is None:
            continue
        p.append(r["p"]); qc.append(c["q_c"])
        ci = c["CI95"]
        lo.append(c["q_c"] - ci[0] if ci[0] is not None else 0)
        hi.append(ci[1] - c["q_c"] if ci[1] is not None else 0)
    return np.array(p), np.array(qc), np.array(lo), np.array(hi)


pw, qw, lw, hw = series("w0")
pd, qd, _, _ = series("deltaf_gap")

fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
# ordered/disordered shading from the corrected (w0) boundary + endpoint
cp = np.append(pw, PC0); cq = np.append(qw, 0.0); o = np.argsort(cp); cp, cq = cp[o], cq[o]
pg = np.linspace(0, PC0 + 0.005, 400); qb = np.interp(pg, cp, cq)
ymax = 0.075
ax.fill_between(pg, 0, qb, color="#2ca02c", alpha=0.12, label="ordered / correctable")
ax.fill_between(pg, qb, ymax, color="#d62728", alpha=0.10, label="disordered / uncorrectable")
# biased Δf boundary (for contrast)
ax.plot(pd, qd, "s--", ms=6, lw=1.3, color="#9467bd", alpha=0.8,
        label="Δf-gap crossing (007, even-moment → biased high)")
# corrected w0 boundary with CI
ax.errorbar(pw, qw, yerr=[lw, hw], fmt="o-", ms=8, lw=1.8, color="#2ca02c", capsize=4,
            label="w₀ = P(true class) crossing  [corrected]")
ax.plot([pw[-1], PC0], [qw[-1], 0.0], ":", color="#2ca02c", lw=1.3, alpha=0.7)
ax.scatter([PC0], [0.0], marker="*", s=260, color="#ff7f0e", zorder=6,
           label=f"q=0 endpoint  p_c≈{PC0:g}")
ax.annotate("L=3,4,5 finite-size crossing\n→ still overestimates;\ntrue boundary lower\n(needs larger L + data collapse)",
            xy=(0.012, 0.052), fontsize=8.5, color="0.25",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax.set_xlim(0, PC0 + 0.012); ax.set_ylim(0, ymax)
ax.set_xlabel("p  (Pauli-X error rate)")
ax.set_ylabel("q_c  (measurement-error threshold)")
ax.set_title("3D toric code single-shot phase boundary q_c(p) — corrected (sign-aware w₀)")
ax.grid(alpha=0.35); ax.legend(fontsize=9, loc="upper right")
out = SD / "phase_boundary_corrected.png"
fig.savefig(out, dpi=175); plt.close(fig)
print(f"fig -> {out}")
print("w0 boundary:", [(float(p), float(q)) for p, q in zip(pw, qw)])
