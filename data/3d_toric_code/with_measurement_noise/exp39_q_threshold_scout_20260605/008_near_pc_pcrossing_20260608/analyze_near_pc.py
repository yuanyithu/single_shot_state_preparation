#!/usr/bin/env python3
"""exp39/008 — resolve the steep approach to p_c by p-crossings at small q.

The 007 boundary q_c(p) is nearly flat (~0.05) for p in [0.02,0.20]; near p_c it
must turn down to 0. Here we fix small q in {0.01,0.02,0.03,0.04} and scan
p in {0.20..0.25} for L=3,4,5, locating the FINITE-SIZE p-crossing p_c(q):
ordered (p<p_c) larger L => larger Δf gap; disordered (p>p_c) reversed.

All p share seed_base=840000 => common-random-number disorder across p, so the
bootstrap is PAIRED (resample disorder indices once, apply to every p) and the
L3-L5 difference curve is low-variance.

Δf = sector free-energy gap (dominant -> nearest competing logical sector),
TI / projection_mode=linear. Combined with 007 q_c(p) this traces the full
(p,q) phase boundary including the turn.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = Path(__file__).resolve().parent
EXP = SD.parent
RNG = np.random.default_rng(20260608)
NBOOT = 8000
LS = [3, 4, 5]
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
BSUMMARY_007 = EXP / "007_phase_boundary_deltaf_20260608" / "boundary_summary.json"
PC0 = 0.227  # q=0 threshold (anchor)


def gap_of(delta_f):
    s = np.sort(delta_f, axis=-1)
    return s[..., 1] - s[..., 0]


def load():
    """recs[p][L][q] = gap[ndis] (disorder index-aligned across p via common seed)."""
    recs = {}
    for f in sorted(glob.glob(str(SD / "collected" / "p*" / "**" / "sector_ti_results.npz"),
                              recursive=True)):
        d = np.load(f, allow_pickle=False)
        p = round(float(d["p_value"]), 4)
        qv = d["q_values"].astype(float)
        Ls = [int(x) for x in d["lattice_size_list"]]
        for li, L in enumerate(Ls):
            g = gap_of(d["delta_f_per_disorder"][li])  # [nq, ndis]
            for qi, q in enumerate(qv):
                recs.setdefault(p, {}).setdefault(L, {})[round(float(q), 4)] = g[qi]
    return recs


def cross_x(x, d):
    """first x where d changes sign (linear interp); None if no sign change."""
    for i in range(len(x) - 1):
        if d[i] == 0:
            return float(x[i])
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def main():
    recs = load()
    ps = sorted(recs)
    qs = sorted({q for L in recs[ps[0]] for q in recs[ps[0]][L]})
    ndis = recs[ps[0]][LS[0]][qs[0]].shape[0]
    print(f"p values: {ps}\nq values: {qs}\nndisorder: {ndis}")

    # paired bootstrap index sets (shared across all p,q,L)
    boot_idx = [RNG.integers(0, ndis, ndis) for _ in range(NBOOT)]

    results = {}   # q -> dict
    for q in qs:
        # A[L] = [nP, ndis]
        A = {L: np.array([recs[p][L][q] for p in ps]) for L in LS}
        mean = {L: np.nanmean(A[L], axis=1) for L in LS}
        # diff(p) = Δf(L3) - Δf(L5): <0 ordered (L5 higher), >0 disordered
        dmean = mean[3] - mean[5]
        p_c = cross_x(ps, dmean)
        pcs = []
        for idx in boot_idx:
            m3 = np.nanmean(A[3][:, idx], axis=1)
            m5 = np.nanmean(A[5][:, idx], axis=1)
            c = cross_x(ps, m3 - m5)
            if c is not None:
                pcs.append(c)
        ci = ([round(float(np.quantile(pcs, 0.025)), 4),
               round(float(np.quantile(pcs, 0.975)), 4)] if len(pcs) > 10 else [None, None])
        # status if unresolved
        status = "resolved"
        if p_c is None:
            status = "all_ordered(p_c>%.2f)" % ps[-1] if dmean[-1] < 0 else "all_disordered(p_c<%.2f)" % ps[0]
        sem = {L: np.array([np.nanstd([np.nanmean(A[L][pi, idx]) for idx in boot_idx], ddof=1)
                            for pi in range(len(ps))]) for L in LS}
        results[q] = {"mean": mean, "sem": sem, "p_c": p_c, "p_c_CI95": ci,
                      "boot_frac": round(len(pcs) / NBOOT, 3), "status": status}
        print(f" q={q:.3f}: p_c={None if p_c is None else round(p_c,4)} CI={ci} "
              f"boot_frac={results[q]['boot_frac']} status={status}")

    # ---------- Figure A: Δf vs p panels per q ----------
    nq = len(qs)
    fig, axes = plt.subplots(1, nq, figsize=(4.2 * nq, 4.0), constrained_layout=True, squeeze=False)
    for k, q in enumerate(qs):
        ax = axes[0][k]
        r = results[q]
        for L in LS:
            ax.errorbar(ps, r["mean"][L], yerr=r["sem"][L], marker="o", ms=5, lw=1.5,
                        color=COLORS[L], capsize=3, label=f"L={L}")
        if r["p_c"] is not None:
            ax.axvline(r["p_c"], color="0.4", ls="--", lw=1.1)
            ci = r["p_c_CI95"]
            if ci[0] is not None:
                ax.axvspan(ci[0], ci[1], color="0.85", alpha=0.6, zorder=0)
            ttl = f"q={q:g}   p_c≈{r['p_c']:.3f}"
        else:
            ttl = f"q={q:g}   {r['status']}"
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("p  (Pauli-X error rate)")
        ax.set_ylabel("Δf gap")
        ax.grid(alpha=0.35)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("exp39/008 near p_c: Δf gap vs p at fixed small q (L=3,4,5), "
                 f"{ndis} disorders (common across p), TI/linear", fontsize=12)
    outA = SD / "near_pc_pcrossings.png"
    fig.savefig(outA, dpi=160)
    plt.close(fig)

    # ---------- Figure B: (p,q) overview — finite-size ordered region vs asymptotic p_c ----------
    flat_p, flat_q = [], []
    if BSUMMARY_007.exists():
        b = json.loads(BSUMMARY_007.read_text())
        for pt in b["points"]:
            if pt["resolved"] and pt["p"] <= 0.20:
                flat_p.append(pt["p"]); flat_q.append(pt["q_c"])
    figB, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    if flat_p:
        o = np.argsort(flat_p)
        ax.plot(np.array(flat_p)[o], np.array(flat_q)[o], "o-", color="#1f1f1f", ms=6, lw=1.4,
                zorder=3, label="007 finite-size boundary q_c(p) (Δf q-crossing)")
    # 008 grid: ordered (L5>L3) vs disordered, per (p,q)
    ord_p, ord_q, dis_p, dis_q = [], [], [], []
    for q in qs:
        for pi, p in enumerate(ps):
            (ord_p if results[q]["mean"][5][pi] > results[q]["mean"][3][pi] else dis_p).append(p)
            (ord_q if results[q]["mean"][5][pi] > results[q]["mean"][3][pi] else dis_q).append(q)
    if ord_p:
        ax.scatter(ord_p, ord_q, marker="o", s=60, facecolors="none", edgecolors="#2ca02c",
                   linewidths=1.7, zorder=4, label="008 grid: Δf-ordered (L5>L3) at L=3,4,5")
    if dis_p:
        ax.scatter(dis_p, dis_q, marker="v", s=60, color="#d62728", zorder=4,
                   label="008 grid: Δf-disordered (L3>L5)")
    turn_q = [q for q in qs if results[q]["p_c"] is not None]
    if turn_q:
        ax.scatter([results[q]["p_c"] for q in turn_q], turn_q, marker="s", s=80,
                   color="#8c2d04", zorder=5, label="008 p_c(q) finite-size crossing")
    ax.axvline(PC0, color="#ff7f0e", ls="--", lw=1.7, zorder=2,
               label=f"asymptotic p_c≈{PC0:g} (q=0, optimal)")
    ax.annotate("finite-size (L=3,4,5) Δf-ordered region\nextends past the asymptotic p_c "
                "≈0.227\n→ small-L Δf crossing OVERESTIMATES the\nthreshold; true turn needs FSS / larger L",
                xy=(0.012, 0.038), fontsize=9, color="0.2",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(0, 0.265)
    ax.set_ylim(0, (max(flat_q) if flat_q else 0.06) * 1.3)
    ax.set_xlabel("p  (Pauli-X error rate)")
    ax.set_ylabel("q  (measurement error rate)")
    ax.set_title("exp39/008: finite-size Δf-ordered region vs asymptotic p_c (near-axis probe)")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8.5, loc="upper right")
    outB = SD / "near_pc_overview.png"
    figB.savefig(outB, dpi=170)
    plt.close(figB)

    summary = {
        "method": "p-crossing of Δf gap (L3-L5) at fixed q; common-disorder across p (seed 840000); paired bootstrap",
        "pc_endpoint_q0": PC0,
        "p_crossings": {f"{q:.3f}": {"p_c": results[q]["p_c"], "p_c_CI95": results[q]["p_c_CI95"],
                                     "boot_frac": results[q]["boot_frac"], "status": results[q]["status"]}
                        for q in qs},
        "ndisorder": int(ndis),
    }
    (SD / "near_pc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nfigA -> {outA}\nfigB -> {outB}\nsummary -> {SD/'near_pc_summary.json'}")


if __name__ == "__main__":
    main()
