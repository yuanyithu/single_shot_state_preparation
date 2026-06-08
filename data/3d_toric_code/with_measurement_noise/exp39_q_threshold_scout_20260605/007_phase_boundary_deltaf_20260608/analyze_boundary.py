#!/usr/bin/env python3
"""exp39/007 — assemble the measurement-error phase boundary q_c(p).

For each p we use the sector free-energy gap Δf (dominant -> nearest competing
logical sector; NON-saturating, unlike q_top) as the finite-size order
parameter. At fixed p the L=3,4,5 Δf(q) curves cross at q_c(p): ordered side
(q<q_c) larger L => larger gap (more protected); disordered side reversed.
Collecting q_c(p) over p traces the boundary in the (p, q) plane.

Path is TI / projection_mode=linear only (correct observable). The two already
finished points (p=0.06 in 006_, p=0.12 in 004_) are reused and re-extracted
with the SAME pipeline so the boundary is uniform.

Loader handles both layouts:
  - new runs: one NPZ per p holding all L (lattice_size_list = [3,4,5])
  - reused runs: separate per-L NPZs, possibly several per L (q-merge)
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
EXP = SD.parent  # exp39_q_threshold_scout_20260605
RNG = np.random.default_rng(20260608)
NBOOT = 8000
PC_ENDPOINT = (0.227, 0.0)  # known q=0 threshold (anchor, not re-run here)
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}

# Source roots: new p-cells under 007/collected/p*, plus the two reused points.
SOURCES = sorted(glob.glob(str(SD / "collected" / "p*")))
SOURCES += [
    str(EXP / "006_production_crossing_p006_20260606" / "collected"),
    str(EXP / "004_production_crossing_p012_20260605" / "collected"),
]


def _gap_from_delta_f(delta_f):
    """delta_f [..., 8] -> gap = F_(2)-F_(1) = sorted[1]-sorted[0] over sectors."""
    s = np.sort(delta_f, axis=-1)
    return s[..., 1] - s[..., 0]


def load_source(root):
    """Return (p, {L: (q_sorted[nq], gap[nq,ndis], qtop[nq,ndis])}) or None."""
    files = sorted(glob.glob(str(Path(root) / "**" / "sector_ti_results.npz"), recursive=True))
    if not files:
        return None
    acc = {}  # L -> list of (q[nq], gap[nq,ndis], qtop[nq,ndis])
    p_value = None
    for f in files:
        d = np.load(f, allow_pickle=False)
        p_value = float(d["p_value"])
        qv = d["q_values"].astype(float)
        Ls = [int(x) for x in d["lattice_size_list"]]
        dfp = d["delta_f_per_disorder"]      # [nL, nq, ndis, 8]
        qtp = d["q_top_per_disorder"]         # [nL, nq, ndis]
        for li, L in enumerate(Ls):
            gap = _gap_from_delta_f(dfp[li])  # [nq, ndis]
            acc.setdefault(L, []).append((qv, gap, qtp[li]))
    data = {}
    for L, chunks in acc.items():
        qv = np.concatenate([c[0] for c in chunks])
        gap = np.concatenate([c[1] for c in chunks], axis=0)
        qtop = np.concatenate([c[2] for c in chunks], axis=0)
        o = np.argsort(qv)
        qv, gap, qtop = qv[o], gap[o], qtop[o]
        _, idx = np.unique(np.round(qv, 6), return_index=True)
        data[L] = (np.round(qv[idx], 6), gap[idx], qtop[idx])
    return p_value, data


def boot_curve(per_dis):
    """per_dis [nq, ndis] -> mean[nq], sem[nq], draws[NBOOT, nq]."""
    nq, nd = per_dis.shape
    mean = np.nanmean(per_dis, axis=1)
    draws = np.empty((NBOOT, nq))
    for b in range(NBOOT):
        idx = RNG.integers(0, nd, nd)
        draws[b] = np.nanmean(per_dis[:, idx], axis=1)
    sem = draws.std(axis=0, ddof=1)
    return mean, sem, draws


def cross_q(q, d):
    """First q where curve-difference d changes sign (linear interp)."""
    for i in range(len(q) - 1):
        if d[i] == 0:
            return float(q[i])
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            return float(q[i] + t * (q[i + 1] - q[i]))
    return None


def pairwise(q, mean, sem, draws, a, b):
    """Crossing of L=a and L=b curves with bootstrap CI."""
    if a not in mean or b not in mean:
        return None
    dmean = mean[a] - mean[b]
    qc = cross_q(q, dmean)
    qcs = []
    for bi in range(NBOOT):
        c = cross_q(q, draws[a][bi] - draws[b][bi])
        if c is not None:
            qcs.append(c)
    ci = ([float(np.quantile(qcs, 0.025)), float(np.quantile(qcs, 0.975))]
          if len(qcs) > 10 else [None, None])
    isep = int(np.argmax(np.abs(dmean)))
    sd = np.hypot(sem[a][isep], sem[b][isep])
    sig = float(abs(dmean[isep]) / sd) if sd > 0 else 0.0
    return {
        "q_c": qc,
        "q_c_CI95": ci,
        "boot_cross_frac": round(len(qcs) / NBOOT, 3),
        "max_sep": round(float(np.max(np.abs(dmean))), 4),
        "max_sep_q": round(float(q[isep]), 4),
        "max_sep_sigma": round(sig, 1),
    }


def analyze_point(p, data):
    Ls = sorted(data)
    q = data[Ls[0]][0]
    mean = {L: None for L in Ls}
    sem = {L: None for L in Ls}
    draws = {L: None for L in Ls}
    qtop_mean = {}
    for L in Ls:
        qv, gap, qtop = data[L]
        mean[L], sem[L], draws[L] = boot_curve(gap)
        qtop_mean[L] = np.nanmean(qtop, axis=1)
    pairs = {f"{a}-{b}": pairwise(q, mean, sem, draws, a, b)
             for a, b in [(3, 4), (3, 5), (4, 5)]}
    # primary q_c: L3-L5 (most resolved); fall back to L3-L4 if absent
    prim = pairs.get("3-5") or pairs.get("3-4")
    q_c = prim["q_c"] if prim else None
    q_c_ci = prim["q_c_CI95"] if prim else [None, None]
    resolved = q_c is not None
    # q_top crossing (saturated secondary cross-check)
    qtcross = cross_q(q, qtop_mean.get(3, np.zeros_like(q)) - qtop_mean.get(5, np.zeros_like(q))) \
        if (3 in qtop_mean and 5 in qtop_mean) else None
    return {
        "p": p, "q": q, "Ls": Ls, "mean": mean, "sem": sem,
        "num_disorder": int(data[Ls[0]][1].shape[1]),
        "pairs": pairs, "q_c": q_c, "q_c_CI95": q_c_ci, "resolved": resolved,
        "q_min": float(q.min()), "q_max": float(q.max()),
        "qtop_crossing": qtcross,
    }


def main():
    points = []
    for root in SOURCES:
        res = load_source(root)
        if res is None:
            print(f"[skip] no NPZ under {root}")
            continue
        p, data = res
        points.append(analyze_point(p, data))
        print(f"[ok] p={p:g}  L={sorted(data)}  ndis={points[-1]['num_disorder']}  "
              f"q_c={'unresolved' if not points[-1]['resolved'] else round(points[-1]['q_c'], 4)}")
    points.sort(key=lambda r: r["p"])

    # ---------- Figure A: Δf-vs-q crossing grid ----------
    n = len(points)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    figA, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                              constrained_layout=True, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(points):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        for L in r["Ls"]:
            ax.errorbar(r["q"], r["mean"][L], yerr=r["sem"][L], marker="o", ms=4,
                        lw=1.4, color=COLORS.get(L), capsize=2, label=f"L={L}")
        if r["resolved"]:
            ax.axvline(r["q_c"], color="0.4", ls="--", lw=1.1)
            ci = r["q_c_CI95"]
            if ci[0] is not None:
                ax.axvspan(ci[0], ci[1], color="0.85", alpha=0.6, zorder=0)
            ttl = f"p={r['p']:g}   q_c≈{r['q_c']:.3f}"
        else:
            ttl = f"p={r['p']:g}   q_c<{r['q_min']:.3f} (unresolved)"
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("Δf gap")
        ax.grid(alpha=0.35)
        if k == 0:
            ax.legend(fontsize=8)
    figA.suptitle("exp39 phase boundary: sector free-energy gap Δf vs q (L=3,4,5), "
                  f"{points[0]['num_disorder']}–96 disorders, TI/linear", fontsize=12)
    outA = SD / "deltaf_crossings_grid.png"
    figA.savefig(outA, dpi=160); plt.close(figA)

    # ---------- Figure B: phase boundary q_c(p) ----------
    res_p = [r["p"] for r in points if r["resolved"]]
    res_qc = [r["q_c"] for r in points if r["resolved"]]
    res_lo = [r["q_c"] - r["q_c_CI95"][0] if r["q_c_CI95"][0] is not None else 0 for r in points if r["resolved"]]
    res_hi = [r["q_c_CI95"][1] - r["q_c"] if r["q_c_CI95"][1] is not None else 0 for r in points if r["resolved"]]
    unq_p = [r["p"] for r in points if not r["resolved"]]
    unq_q = [r["q_min"] for r in points if not r["resolved"]]

    # resolved Δf points sorted by p; endpoint only used to shade the regions
    order = np.argsort(res_p)
    rp = np.array(res_p)[order]
    rq = np.array(res_qc)[order]
    curve_p = np.append(rp, PC_ENDPOINT[0])
    curve_q = np.append(rq, PC_ENDPOINT[1])
    # q_top crossing (saturating estimator) as a lower finite-size reference line
    qt_p = [r["p"] for r in points if r["qtop_crossing"] is not None]
    qt_q = [r["qtop_crossing"] for r in points if r["qtop_crossing"] is not None]
    qo = np.argsort(qt_p)
    qt_p = np.array(qt_p)[qo]
    qt_q = np.array(qt_q)[qo]

    figB, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    pgrid = np.linspace(0, PC_ENDPOINT[0] + 0.01, 400)
    qb = np.interp(pgrid, curve_p, curve_q)
    ymax = max(res_qc) * 1.3 if res_qc else 0.1
    ax.fill_between(pgrid, 0, qb, color="#2ca02c", alpha=0.12, label="ordered / correctable")
    ax.fill_between(pgrid, qb, ymax, color="#d62728", alpha=0.10, label="disordered / uncorrectable")
    # solid line through resolved points; dashed schematic for the unresolved steep approach to p_c
    ax.plot(rp, rq, "-", color="0.3", lw=1.6, zorder=2)
    ax.plot([rp[-1], PC_ENDPOINT[0]], [rq[-1], PC_ENDPOINT[1]], "--", color="0.55", lw=1.3,
            zorder=2, label="schematic approach to p_c (unresolved)")
    if len(qt_p) >= 2:
        ax.plot(qt_p, qt_q, ":", color="#9467bd", lw=1.5, marker="s", ms=4, zorder=2,
                label="q_top crossing (lower FS estimate)")
    ax.errorbar(res_p, res_qc, yerr=[res_lo, res_hi], fmt="o", ms=8, color="#1f1f1f",
                capsize=4, zorder=3, label="q_c(p)  (Δf L3–L5 crossing)")
    if unq_p:
        ax.scatter(unq_p, unq_q, marker="v", s=90, color="0.45", zorder=3,
                   label="unresolved: q_c < q_min")
    ax.scatter([PC_ENDPOINT[0]], [PC_ENDPOINT[1]], marker="*", s=240, color="#ff7f0e",
               zorder=4, label=f"q=0 endpoint p_c≈{PC_ENDPOINT[0]:g}")
    for r in points:
        if r["resolved"]:
            ax.annotate(f"{r['p']:g}", (r["p"], r["q_c"]), textcoords="offset points",
                        xytext=(5, 6), fontsize=8, color="0.3")
    ax.set_xlim(0, PC_ENDPOINT[0] + 0.012)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("p  (Pauli-X error rate)")
    ax.set_ylabel("q_c  (measurement-error threshold)")
    ax.set_title("3D toric code single-shot phase boundary q_c(p)  (sector free-energy gap Δf)")
    ax.grid(alpha=0.35); ax.legend(fontsize=9, loc="upper right")
    outB = SD / "phase_boundary.png"
    figB.savefig(outB, dpi=170); plt.close(figB)

    # ---------- summary ----------
    summary = {
        "method": "sector free-energy gap Δf (TI/linear); q_c = finite-size crossing of L=3,4,5",
        "pc_endpoint": {"p": PC_ENDPOINT[0], "q": PC_ENDPOINT[1], "note": "q=0 threshold, anchor"},
        "points": [
            {"p": r["p"], "num_disorder": r["num_disorder"], "resolved": r["resolved"],
             "q_c": (None if not r["resolved"] else round(r["q_c"], 4)),
             "q_c_CI95": [None if x is None else round(x, 4) for x in r["q_c_CI95"]],
             "q_min": round(r["q_min"], 4), "q_max": round(r["q_max"], 4),
             "qtop_crossing": (None if r["qtop_crossing"] is None else round(r["qtop_crossing"], 4)),
             "pairwise": r["pairs"]}
            for r in points
        ],
    }
    (SD / "boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==== phase boundary q_c(p) (Δf L3-L5 crossing) ====")
    print("   p      q_c        CI95            qtop_xing  ndis")
    for r in points:
        qc = "unresolved" if not r["resolved"] else f"{r['q_c']:.4f}"
        ci = r["q_c_CI95"]
        cis = "        -       " if ci[0] is None else f"[{ci[0]:.4f},{ci[1]:.4f}]"
        qt = "-" if r["qtop_crossing"] is None else f"{r['qtop_crossing']:.4f}"
        print(f"  {r['p']:<6g} {qc:>10}  {cis:>16}   {qt:>7}   {r['num_disorder']}")
    print(f"\n[endpoint] q=0 -> p_c≈{PC_ENDPOINT[0]}")
    print(f"figA -> {outA}\nfigB -> {outB}\nsummary -> {SD/'boundary_summary.json'}")


if __name__ == "__main__":
    main()
