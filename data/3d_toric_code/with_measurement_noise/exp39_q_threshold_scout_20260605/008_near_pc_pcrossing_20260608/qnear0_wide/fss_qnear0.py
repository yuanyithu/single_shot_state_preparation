#!/usr/bin/env python3
"""exp39/008 q≈0 finite-size scaling: does the Δf-gap p-crossing drift toward
the known asymptotic p_c≈0.233 (3D RBIM Nishimori) as L grows?

Loads Δf gap at L=3,4,5 (wide scan) + L=6 (FSS run) at q=0.002 over p, all on
common disorders (seed 840000, scope disorder_index). For each consecutive pair
(L,L') finds the p where the mean Δf(L) and Δf(L') curves cross, then linearly
extrapolates p_cross vs 1/L_eff to 1/L->0. Flat (no drift) => the gap-crossing
locates a value far above 0.233 (method/observable bias, not just finite size);
clear downward drift => finite-size, extrapolating toward p_c.
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
RNG = np.random.default_rng(20260609)
NBOOT = 8000
PC_TRUE = 0.233   # 3D RBIM Nishimori (q=0 optimal X-error threshold)
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c", 6: "#9467bd", 7: "#8c564b"}


def gap_of(df):
    s = np.sort(df, axis=-1)
    return s[..., 1] - s[..., 0]


def load():
    """rec[L][p] = gap[ndis] at q=0.002 (disorder index-aligned across L)."""
    rec = {}
    for f in sorted(glob.glob(str(SD / "collected" / "p*" / "**" / "sector_ti_results.npz"),
                              recursive=True)):
        d = np.load(f, allow_pickle=False)
        p = round(float(d["p_value"]), 4)
        Ls = [int(x) for x in d["lattice_size_list"]]
        for li, L in enumerate(Ls):
            rec.setdefault(L, {})[p] = gap_of(d["delta_f_per_disorder"][li])[0]  # single q -> [ndis]
    return rec


def cross_x(x, dvec):
    for i in range(len(x) - 1):
        if dvec[i] == 0:
            return float(x[i])
        if dvec[i] * dvec[i + 1] < 0:
            t = dvec[i] / (dvec[i] - dvec[i + 1])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def main():
    rec = load()
    Ls = sorted(rec)
    ps = sorted(rec[Ls[0]])
    ndis = rec[Ls[0]][ps[0]].shape[0]
    print(f"L sizes: {Ls}\np values: {ps}\nndisorder: {ndis}")
    # arrays A[L] = [nP, ndis]
    A = {L: np.array([rec[L][p] for p in ps]) for L in Ls}
    mean = {L: np.nanmean(A[L], axis=1) for L in Ls}
    boot_idx = [RNG.integers(0, ndis, ndis) for _ in range(NBOOT)]
    sem = {L: np.array([np.nanstd([np.nanmean(A[L][pi, idx]) for idx in boot_idx], ddof=1)
                        for pi in range(len(ps))]) for L in Ls}

    print("\nmean Δf gap (q=0.002):")
    print("  p     " + "  ".join(f"L{L}" for L in Ls))
    for pi, p in enumerate(ps):
        print(f"  {p:.2f}  " + "  ".join(f"{mean[L][pi]:6.2f}" for L in Ls))

    # consecutive-pair crossings with paired bootstrap
    pairs = [(Ls[i], Ls[i + 1]) for i in range(len(Ls) - 1)]
    pair_cross = {}
    print("\nconsecutive-pair p-crossings:")
    for (a, b) in pairs:
        pc = cross_x(ps, mean[a] - mean[b])
        pcs = []
        for idx in boot_idx:
            c = cross_x(ps, np.nanmean(A[a][:, idx], axis=1) - np.nanmean(A[b][:, idx], axis=1))
            if c is not None:
                pcs.append(c)
        ci = ([float(np.quantile(pcs, 0.025)), float(np.quantile(pcs, 0.975))]
              if len(pcs) > 10 else [None, None])
        Leff = 0.5 * (a + b)
        pair_cross[f"{a}-{b}"] = {"p_cross": pc, "L_eff": Leff, "CI95": ci,
                                  "boot_frac": round(len(pcs) / NBOOT, 3)}
        print(f"  L{a}-L{b} (L_eff={Leff}): p_cross={None if pc is None else round(pc,4)} "
              f"CI={[None if x is None else round(x,4) for x in ci]}")

    # linear extrapolation p_cross vs 1/L_eff -> intercept
    pts = [(pc["L_eff"], pc["p_cross"]) for pc in pair_cross.values() if pc["p_cross"] is not None]
    extrap = None
    if len(pts) >= 2:
        Leff = np.array([t[0] for t in pts]); pc = np.array([t[1] for t in pts])
        x = 1.0 / Leff
        b1, b0 = np.polyfit(x, pc, 1)   # pc = b1*x + b0 ; intercept b0 = p_c(inf)
        extrap = float(b0)
        print(f"\n1/L_eff extrapolation -> p_c(inf) = {extrap:.4f}  (known p_c≈{PC_TRUE})")

    # ---- figure A: Δf vs p for all L ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for L in Ls:
        ax1.errorbar(ps, mean[L], yerr=sem[L], marker="o", ms=5, lw=1.5, color=COLORS.get(L),
                     capsize=3, label=f"L={L}")
    for key, info in pair_cross.items():
        if info["p_cross"] is not None:
            ax1.axvline(info["p_cross"], color="0.7", ls=":", lw=1)
    ax1.axvline(PC_TRUE, color="#ff7f0e", ls="--", lw=1.6, label=f"true p_c≈{PC_TRUE}")
    ax1.set_xlabel("p"); ax1.set_ylabel("Δf gap")
    ax1.set_title(f"q≈0 (q=0.002): Δf gap vs p, L={Ls}, {ndis} disorders")
    ax1.grid(alpha=0.35); ax1.legend(fontsize=9)
    # ---- figure B: p_cross vs 1/L_eff extrapolation ----
    if pts:
        Leff = np.array([t[0] for t in pts]); pcv = np.array([t[1] for t in pts])
        x = 1.0 / Leff
        ax2.errorbar(x, pcv, fmt="s", ms=9, color="#1f1f1f", capsize=3, label="pair crossings")
        if extrap is not None:
            xx = np.linspace(0, x.max() * 1.1, 50)
            ax2.plot(xx, b1 * xx + b0, "-", color="0.5", lw=1.4)
            ax2.scatter([0], [extrap], marker="*", s=240, color="#2ca02c", zorder=5,
                        label=f"extrap 1/L→0: p_c={extrap:.3f}")
        ax2.axhline(PC_TRUE, color="#ff7f0e", ls="--", lw=1.6, label=f"true p_c≈{PC_TRUE}")
        for key, info in pair_cross.items():
            if info["p_cross"] is not None:
                ax2.annotate(key, (1.0 / info["L_eff"], info["p_cross"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax2.set_xlabel("1 / L_eff"); ax2.set_ylabel("p_cross")
        ax2.set_title("finite-size scaling: p_cross vs 1/L_eff")
        ax2.set_xlim(left=-0.01); ax2.grid(alpha=0.35); ax2.legend(fontsize=9)
    out = SD / "fss_qnear0.png"
    fig.savefig(out, dpi=170); plt.close(fig)

    summary = {"q": 0.002, "lattice_sizes": Ls, "ndisorder": int(ndis),
               "pair_crossings": pair_cross, "extrap_pc_inf": extrap, "pc_true": PC_TRUE}
    (SD / "fss_qnear0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nfig -> {out}")


if __name__ == "__main__":
    main()
