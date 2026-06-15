#!/usr/bin/env python3
"""Zoom into the ordered side (q<=0.03) of the p=0.12 crossing, y-axis zoomed,
to inspect the 'q_top increases with L' signature."""
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path(__file__).resolve().parent
QMAX = 0.031
RNG = np.random.default_rng(7)

perL = {}
for f in sorted(glob.glob(str(SD / "collected" / "*" / "sector_ti_results.npz"))):
    d = np.load(f, allow_pickle=False); L = int(d["lattice_size_list"][0])
    perL.setdefault(L, []).append((d["q_values"].astype(float), d["q_top_per_disorder"][0]))
data = {}; q = None
for L in sorted(perL):
    qv = np.concatenate([c[0] for c in perL[L]]); pd = np.concatenate([c[1] for c in perL[L]], axis=0)
    o = np.argsort(qv); qv = qv[o]; pd = pd[o]; _, idx = np.unique(np.round(qv, 6), return_index=True)
    data[L] = pd[idx]; q = np.round(qv[idx], 6) if q is None else q

m = q <= QMAX
print(f"q<=0.03 points: {q[m].tolist()}")
colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
for L in sorted(data):
    mean = np.nanmean(data[L], axis=1)
    sem = np.array([np.std([np.nanmean(data[L][qi, RNG.integers(0, data[L].shape[1], data[L].shape[1])])
                            for _ in range(4000)], ddof=1) for qi in range(len(q))])
    ax.errorbar(q[m], mean[m], yerr=sem[m], marker="o", ms=7, lw=1.6, color=colors[L], capsize=4, label=f"L={L}")
    print(f"L{L}: " + "  ".join(f"q={qq:.3f}:{mm:.4f}±{ss:.4f}" for qq, mm, ss in zip(q[m], mean[m], sem[m])))
ax.set_xlabel("q (measurement error rate)"); ax.set_ylabel("mean q_top (TI/linear)")
ax.set_title("exp39 p=0.12 ordered side (q<0.03), y-zoom")
ax.set_ylim(0.985, 1.003); ax.grid(alpha=0.4); ax.legend(fontsize=10)
fig.savefig(SD / "zoom_lowq.png", dpi=170); plt.close(fig)
print("fig -> zoom_lowq.png")
