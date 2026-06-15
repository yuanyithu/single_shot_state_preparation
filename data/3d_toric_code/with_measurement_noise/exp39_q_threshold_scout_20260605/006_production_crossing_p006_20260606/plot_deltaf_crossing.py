#!/usr/bin/env python3
"""Plot the sector free-energy gap Δf (dominant vs nearest competing logical
sector) vs q for L=3,4,5 at p=0.06. Δf does NOT saturate, so it shows the
two-sided threshold crossing clearly: ordered side larger L -> larger gap
(more protected), disordered side larger L -> smaller gap."""
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path(__file__).resolve().parent
RNG = np.random.default_rng(11)
NB = 8000

data = {}
for f in sorted(glob.glob(str(SD / "collected" / "*" / "sector_ti_results.npz"))):
    d = np.load(f, allow_pickle=False); L = int(d["lattice_size_list"][0])
    q = d["q_values"].astype(float); o = np.argsort(q)
    df = d["delta_f_per_disorder"][0][o]            # [nq, ndis, 8]
    gap = np.sort(df, axis=2)[:, :, 1] - np.sort(df, axis=2)[:, :, 0]  # [nq, ndis] gap to nearest competitor
    data[L] = (q[o], gap)
q = data[3][0]
Ls = sorted(data)

mean = {L: np.nanmean(data[L][1], axis=1) for L in Ls}
sem = {}
for L in Ls:
    g = data[L][1]; nd = g.shape[1]
    draws = np.array([np.nanmean(g[:, RNG.integers(0, nd, nd)], axis=1) for _ in range(NB)])
    sem[L] = draws.std(axis=0, ddof=1)

# crossing of L3 and L5 gap curves
def cross(qq, dd):
    for i in range(len(qq) - 1):
        if dd[i] * dd[i + 1] < 0:
            t = dd[i] / (dd[i] - dd[i + 1]); return float(qq[i] + t * (qq[i + 1] - qq[i]))
    return None
qc = cross(q, mean[3] - mean[5])

colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
for L in Ls:
    ax.errorbar(q, mean[L], yerr=sem[L], marker="o", ms=6, lw=1.7, color=colors[L], capsize=3, label=f"L={L}")
if qc:
    ax.axvline(qc, color="0.4", ls="--", lw=1.2, label=f"crossing q_c≈{qc:.3f}")
ax.annotate("ordered:\nlarger L → larger gap\n(more protected)", xy=(0.035, 25), fontsize=9, color="0.25")
ax.annotate("disordered:\nlarger L → smaller gap", xy=(0.14, 3.4), fontsize=9, color="0.25")
ax.set_xlabel("q (measurement error rate)")
ax.set_ylabel("sector free-energy gap  Δf  (logical protection)")
ax.set_title("exp39 p=0.06: free-energy gap crossing (non-saturating), 96 disorders")
ax.grid(alpha=0.4); ax.legend(fontsize=9)
fig.savefig(SD / "deltaf_crossing.png", dpi=170); plt.close(fig)
print(f"crossing q_c(Δf, L3-L5) ≈ {qc}")
print("  q     " + "   ".join(f"L{L}" for L in Ls))
for qi, qv in enumerate(q):
    print(f"  {qv:5.3f}  " + "  ".join(f"{mean[L][qi]:5.2f}±{sem[L][qi]:.2f}" for L in Ls))
print("fig -> deltaf_crossing.png")
