#!/usr/bin/env python3
"""Analyze exp39 q-threshold scout: locate q_c(p) and L3-vs-L4 separation.

Loads per-p sector_ti_results.npz (TI/linear), and for each p finds where the
L3 and L4 mean_q_top curves cross (q_c), the crossing q_top value, and the
max |L3-L4| separation in the scanned q-range. Picks the p with the cleanest,
best-separated crossing for the follow-up production run.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def _interp_cross(q, d):
    """First sign-change crossing of d(q)=qtop_L3-qtop_L4 -> (q_c, qtop_at_cross) or None."""
    for i in range(len(q) - 1):
        if d[i] == 0.0:
            return float(q[i]), None
        if d[i] * d[i + 1] < 0.0:
            t = d[i] / (d[i] - d[i + 1])
            return float(q[i] + t * (q[i + 1] - q[i])), None
    return None


def load_shard(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    L = list(d["lattice_size_list"].astype(int))
    q = d["q_values"].astype(float)
    mean = d["mean_q_top"].astype(float)       # [nL, nq]
    sem = d["total_sem_q_top"].astype(float)   # [nL, nq]
    p = float(d["p_value"])
    return p, L, q, mean, sem


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "collected"
    npzs = sorted(glob.glob(str(root / "**" / "sector_ti_results.npz"), recursive=True))
    if not npzs:
        print(f"no sector_ti_results.npz under {root}")
        return 1
    shards = [load_shard(f) for f in npzs]
    shards.sort(key=lambda s: s[0])

    rows = []
    fig, axes = plt.subplots(1, len(shards), figsize=(4.2 * len(shards), 4.0), squeeze=False)
    for ax, (p, L, q, mean, sem) in zip(axes[0], shards):
        order = np.argsort(q)
        q = q[order]; mean = mean[:, order]; sem = sem[:, order]
        li3 = L.index(3) if 3 in L else None
        li4 = L.index(4) if 4 in L else None
        colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
        for li, Lv in enumerate(L):
            ax.errorbar(q, mean[li], yerr=sem[li], marker="o", ms=4, lw=1.3,
                        color=colors.get(Lv, "k"), capsize=2, label=f"L={Lv}")
        qc = None; sep = np.nan
        if li3 is not None and li4 is not None:
            diff = mean[li3] - mean[li4]
            cr = _interp_cross(q, diff)
            qc = cr[0] if cr else None
            sep = float(np.max(np.abs(diff)))
            if qc is not None:
                ax.axvline(qc, color="0.4", ls="--", lw=1)
        ax.set_title(f"p={p:g}  q_c≈{qc if qc is not None else 'NA'}")
        ax.set_xlabel("q"); ax.set_ylim(-0.02, 1.03); ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
        rows.append({"p": p, "q_c_L3L4": qc, "max_abs_L3L4_sep": round(sep, 4),
                     "q_range": [float(q.min()), float(q.max())],
                     "mean_total_sem": round(float(np.nanmean(sem)), 4)})
    axes[0][0].set_ylabel("mean q_top (TI/linear)")
    fig.suptitle("exp39 scout: q_c(p) and L3-L4 separation (correct observable)")
    fig.tight_layout()
    out_png = SCRIPT_DIR / "scout_qc_separation.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # rank by separation (bigger = fewer disorders needed) among shards with a real crossing
    ranked = sorted([r for r in rows if r["q_c_L3L4"] is not None],
                    key=lambda r: r["max_abs_L3L4_sep"], reverse=True)
    summary = {"shards": rows,
               "best_p_by_separation": ranked[0]["p"] if ranked else None,
               "note": "Pick p with a clear crossing in-range AND largest L3-L4 separation; "
                       "that p needs the fewest disorders for the production crossing."}
    (SCRIPT_DIR / "scout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nplot -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
