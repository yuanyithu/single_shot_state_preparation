#!/usr/bin/env python3
"""exp39 production crossing analysis: merge per-L NPZs (L=3,4,5 at p=0.12),
plot q_top(q) with disorder-bootstrap error bars, locate the finite-size
crossing q_c (per pair, with bootstrap CI), and write a summary."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260606)
NBOOT = 20000


def load():
    # Merge all NPZs per L (main run + low-q supplement share the same disorders),
    # concatenating q-points, sorting, and de-duplicating.
    perL = {}
    p = None
    for f in sorted(glob.glob(str(SD / "collected" / "*" / "sector_ti_results.npz"))):
        d = np.load(f, allow_pickle=False)
        L = int(d["lattice_size_list"][0])
        qv = d["q_values"].astype(float)
        pd = d["q_top_per_disorder"][0]            # [nq, ndis]
        perL.setdefault(L, []).append((qv, pd))
        p = float(d["p_value"])
    data = {}
    qref = None
    for L in sorted(perL):
        qv = np.concatenate([c[0] for c in perL[L]])
        pd = np.concatenate([c[1] for c in perL[L]], axis=0)
        o = np.argsort(qv); qv = qv[o]; pd = pd[o]
        _, idx = np.unique(np.round(qv, 6), return_index=True)
        data[L] = pd[idx]                          # [nq, ndis], q-sorted unique
        if qref is None:
            qref = np.round(qv[idx], 6)
    return p, qref, data  # data[L] = [nq, ndis]


def boot_mean_curve(per_dis):
    """per_dis [nq, ndis] -> mean[nq], sem[nq], and bootstrap draws [NBOOT, nq]."""
    nq, nd = per_dis.shape
    mean = np.nanmean(per_dis, axis=1)
    draws = np.empty((NBOOT, nq))
    for b in range(NBOOT):
        idx = RNG.integers(0, nd, nd)
        draws[b] = np.nanmean(per_dis[:, idx], axis=1)
    sem = draws.std(axis=0, ddof=1)
    return mean, sem, draws


def cross_q(q, d):
    for i in range(len(q) - 1):
        if d[i] == 0: return float(q[i])
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1]); return float(q[i] + t * (q[i + 1] - q[i]))
    return None


def main():
    p, q, data = load()
    Ls = sorted(data)
    mean, sem, draws = {}, {}, {}
    for L in Ls:
        mean[L], sem[L], draws[L] = boot_mean_curve(data[L])

    # ---- figure ----
    colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    for L in Ls:
        ax.errorbar(q, mean[L], yerr=sem[L], marker="o", ms=5, lw=1.6,
                    color=colors.get(L), capsize=3, label=f"L={L}")
    # pairwise crossings
    pair_out = {}
    for a, b in [(3, 4), (4, 5), (3, 5)]:
        if a in data and b in data:
            dmean = mean[a] - mean[b]
            qc = cross_q(q, dmean)
            # bootstrap q_c CI
            qcs = []
            for bi in range(NBOOT):
                dd = draws[a][bi] - draws[b][bi]
                c = cross_q(q, dd)
                if c is not None: qcs.append(c)
            qc_lo, qc_hi = (float(np.quantile(qcs, 0.025)), float(np.quantile(qcs, 0.975))) if qcs else (None, None)
            sepmax = float(np.max(np.abs(dmean)))
            isep = int(np.argmax(np.abs(dmean)))
            sd = np.sqrt(sem[a]**2 + sem[b]**2)
            sig = float(abs(dmean[isep]) / sd[isep]) if sd[isep] > 0 else 0.0
            pair_out[f"{a}-{b}"] = {"q_c": qc, "q_c_CI95": [qc_lo, qc_hi],
                                    "max_sep": round(sepmax, 4), "max_sep_q": float(q[isep]),
                                    "max_sep_sigma": round(sig, 1),
                                    "boot_cross_frac": round(len(qcs) / NBOOT, 3)}
    # mark common crossing region (use 3-5 crossing as the anchor)
    qc35 = pair_out.get("3-5", {}).get("q_c")
    if qc35 is not None:
        ci = pair_out["3-5"]["q_c_CI95"]
        ax.axvspan(ci[0], ci[1], color="0.85", alpha=0.6, zorder=0)
        ax.axvline(qc35, color="0.4", ls="--", lw=1.2, label=f"q_c(L3-L5)≈{qc35:.3f}")
    ax.set_xlabel("q (measurement error rate)")
    ax.set_ylabel("q_top  (logical-class purity, TI/linear)")
    ax.set_title(f"exp39 production crossing: p={p:g}, 96 disorders (correct observable)")
    ax.set_ylim(-0.02, 1.04); ax.grid(alpha=0.4); ax.legend(fontsize=9)
    out_png = SD / "production_crossing.png"
    fig.savefig(out_png, dpi=170); plt.close(fig)

    # ---- table + summary ----
    table = []
    for qi, qv in enumerate(q):
        row = {"q": round(float(qv), 4)}
        for L in Ls:
            row[f"L{L}"] = f"{mean[L][qi]:.3f}±{sem[L][qi]:.3f}"
        table.append(row)
    summary = {"p": p, "lattice_sizes": Ls, "num_disorder": int(data[Ls[0]].shape[1]),
               "q_values": [round(float(x), 4) for x in q],
               "pairwise_crossings": pair_out, "per_q": table}
    (SD / "production_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # print compact
    print(f"p={p}  L={Ls}  disorders={data[Ls[0]].shape[1]}")
    print("\nmean q_top ± boot-SEM:")
    print("  q     " + "   ".join(f"L{L}" for L in Ls))
    for qi, qv in enumerate(q):
        print(f"  {qv:5.3f}  " + "  ".join(f"{mean[L][qi]:.3f}±{sem[L][qi]:.3f}" for L in Ls))
    print("\npairwise crossings:")
    for k, v in pair_out.items():
        print(f"  L{k}: q_c={None if v['q_c'] is None else round(v['q_c'],4)} "
              f"CI95={[None if x is None else round(x,4) for x in v['q_c_CI95']]} "
              f"max_sep={v['max_sep']} ({v['max_sep_sigma']}σ @ q={v['max_sep_q']}) "
              f"boot_cross_frac={v['boot_cross_frac']}")
    print(f"\nfig -> {out_png}")


if __name__ == "__main__":
    main()
