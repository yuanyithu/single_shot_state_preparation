#!/usr/bin/env python3
"""exp40/004 final: p=0.11 single-point q_c with L=3,4,5,6 and 384 disorders.

Merges per-L disorder blocks from heterogeneous NPZ files (the L=3,4,5 run and
the L=6 run share q grid and seed blocks). Produces the publication-grade
single-point figure:
  (a) q_top(q) per L with disorder-SEM error bars,
  (b) crossing zoom,
  (c) pairwise difference curves (consecutive pairs) with bootstrap bands,
  (d) q_c(pair) vs 1/L_mean with linear 1/L -> 0 extrapolation.
Reports all pairwise crossings for q_top and w0 (cross-check).

Usage: python analyze_p011_L3456.py [--nboot N]
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260612)
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c", 6: "#9467bd"}
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])
SOURCE_GLOBS = ["nd1/collected", "nd2/collected", "nd3/collected",
                "nd1_L6/collected", "nd2_L6/collected", "nd3_L6/collected"]


def estimators(delta_f):
    x = delta_f - delta_f.min(axis=2, keepdims=True)
    w = np.exp(-x); w /= w.sum(axis=2, keepdims=True)
    m = w @ SIGN.T
    return {"q_top": np.mean(m ** 2, axis=2), "w0": w[..., 0]}


def load_merged():
    """Merge blocks per L along the disorder axis; q grids must match."""
    files = []
    for g in SOURCE_GLOBS:
        files += sorted(glob.glob(str(SD / g / "**" / "sector_ti_results.npz"), recursive=True))
    if not files:
        raise SystemExit("no NPZ found")
    p_value, qv = None, None
    acc = {}  # L -> {est: [blocks]}
    for f in files:
        d = np.load(f, allow_pickle=False)
        p_value = float(d["p_value"])
        if qv is None:
            qv = d["q_values"].astype(float)
        else:
            assert np.allclose(qv, d["q_values"].astype(float)), f"q grid mismatch: {f}"
        for li, L in enumerate([int(x) for x in d["lattice_size_list"]]):
            e = estimators(d["delta_f_per_disorder"][li])
            slot = acc.setdefault(L, {k: [] for k in e})
            for k in e:
                slot[k].append(e[k])
    data = {L: {k: np.concatenate(v[k], axis=1) for k in v} for L, v in acc.items()}
    return p_value, qv, data, len(files)


def cross_q(q, dvec):
    nz = np.flatnonzero(dvec)
    if len(nz) == 0:
        return None
    prev = nz[0]
    for i in nz[1:]:
        if dvec[prev] * dvec[i] < 0:
            t = dvec[prev] / (dvec[prev] - dvec[i])
            return float(q[prev] + t * (q[i] - q[prev]))
        prev = i
    return None


def crossing_with_ci(q, A_small, A_large, nboot):
    nd = min(A_small.shape[1], A_large.shape[1])
    A_small, A_large = A_small[:, :nd], A_large[:, :nd]
    qc = cross_q(q, np.nanmean(A_small, axis=1) - np.nanmean(A_large, axis=1))
    qcs = []
    for _ in range(nboot):
        idx = RNG.integers(0, nd, nd)
        c = cross_q(q, np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1))
        if c is not None:
            qcs.append(c)
    ci = ([float(np.quantile(qcs, 0.025)), float(np.quantile(qcs, 0.975))]
          if len(qcs) > 10 else [None, None])
    return qc, ci, len(qcs) / nboot, np.asarray(qcs)


def diff_band(q, A_small, A_large, nboot):
    nd = min(A_small.shape[1], A_large.shape[1])
    A_small, A_large = A_small[:, :nd], A_large[:, :nd]
    diffs = np.empty((nboot, len(q)))
    for b in range(nboot):
        idx = RNG.integers(0, nd, nd)
        diffs[b] = np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1)
    return np.quantile(diffs, 0.025, axis=0), np.quantile(diffs, 0.975, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=6000)
    args = ap.parse_args()

    p, q, data, nfiles = load_merged()
    Ls = sorted(data)
    nd_per_L = {L: data[L]["q_top"].shape[1] for L in Ls}
    print(f"p={p}  files={nfiles}  Ls={Ls}  ndis={nd_per_L}")

    pairs = [(a, b) for i, a in enumerate(Ls) for b in Ls[i + 1:]]
    consec = [(Ls[i], Ls[i + 1]) for i in range(len(Ls) - 1)]
    results, boots = {}, {}
    for est_name in ("q_top", "w0"):
        for (a, b) in pairs:
            qc, ci, bf, qcs = crossing_with_ci(q, data[a][est_name], data[b][est_name], args.nboot)
            key = f"{est_name}_L{a}{b}"
            results[key] = {"q_c": None if qc is None else round(qc, 4),
                            "CI95": [None if c is None else round(c, 4) for c in ci],
                            "boot_frac": round(bf, 3)}
            boots[key] = qcs
            if est_name == "q_top":
                print(f"q_top L{a}-L{b}: {results[key]}")

    # ---- 1/L extrapolation over consecutive pairs (q_top), bootstrap-propagated ----
    extrap = None
    xs = np.array([2.0 / (a + b) for (a, b) in consec])     # 1/L_mean
    ys = [results[f"q_top_L{a}{b}"]["q_c"] for (a, b) in consec]
    if all(y is not None for y in ys):
        nb_ex = 2000
        samples = []
        for _ in range(nb_ex):
            yb = []
            for (a, b) in consec:
                arr = boots[f"q_top_L{a}{b}"]
                if len(arr) == 0:
                    break
                yb.append(arr[RNG.integers(0, len(arr))])
            if len(yb) == len(consec):
                coef = np.polyfit(xs, yb, 1)
                samples.append(coef[1])
        fit = np.polyfit(xs, np.array(ys, float), 1)
        extrap = {"q_c_inf": round(float(fit[1]), 4),
                  "CI95": [round(float(np.quantile(samples, 0.025)), 4),
                           round(float(np.quantile(samples, 0.975)), 4)] if samples else [None, None],
                  "pairs": {f"L{a}{b}": y for (a, b), y in zip(consec, ys)},
                  "slope": round(float(fit[0]), 4)}
        print(f"1/L extrapolation (consecutive q_top pairs): q_c(inf)={extrap['q_c_inf']} CI={extrap['CI95']}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.4), constrained_layout=True)
    (axA, axB), (axC, axD) = axes
    mean = {L: data[L]["q_top"].mean(axis=1) for L in Ls}
    sem = {L: data[L]["q_top"].std(axis=1, ddof=1) / np.sqrt(nd_per_L[L]) for L in Ls}
    for ax in (axA, axB):
        for L in Ls:
            ax.errorbar(q, mean[L], yerr=sem[L], marker="o", ms=4.5, lw=1.5,
                        capsize=2.5, color=COLORS[L], label=f"L={L}")
        ax.grid(alpha=0.35); ax.legend(fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("q_top (disorder mean ± SEM)")
    ref = results.get("q_top_L46") or results["q_top_L35"]
    if ref["q_c"] is not None:
        for ax in (axA, axB):
            ax.axvline(ref["q_c"], color="0.3", ls="--", lw=1.2)
            if ref["CI95"][0] is not None:
                ax.axvspan(ref["CI95"][0], ref["CI95"][1], color="0.6", alpha=0.16)
    axA.set_title(f"p={p}: q_top(q), L={','.join(map(str, Ls))}, {max(nd_per_L.values())} disorders")
    zoom_hi = (ref["CI95"][1] or 0.05) + 0.012
    axB.set_xlim(q[0], zoom_hi)
    ymin = min(mean[L][q <= zoom_hi].min() for L in Ls) - 0.004
    axB.set_ylim(ymin, 1.0015)
    axB.set_title("crossing zoom")

    Lmax = Ls[-1]
    for a in Ls[:-1]:
        nd = min(nd_per_L[a], nd_per_L[Lmax])
        d = data[a]["q_top"][:, :nd].mean(axis=1) - data[Lmax]["q_top"][:, :nd].mean(axis=1)
        lo, hi = diff_band(q, data[a]["q_top"], data[Lmax]["q_top"], 2000)
        c = COLORS[a]
        axC.plot(q, d, marker="o", ms=4, lw=1.5, color=c, label=f"L{a} − L{Lmax}")
        axC.fill_between(q, lo, hi, color=c, alpha=0.15)
    axC.axhline(0, color="k", lw=1)
    if ref["q_c"] is not None:
        axC.axvline(ref["q_c"], color="0.3", ls="--", lw=1.2)
    axC.set_xlabel("q"); axC.set_ylabel(f"Δ q_top (L − L{Lmax})")
    axC.set_title(f"differences vs largest L: sign change = q_c")
    axC.grid(alpha=0.35); axC.legend(fontsize=10)

    # ---- pair-convergence panel: all pairs, both estimators, no fit ----
    for est_name, mk, col, lab in (("q_top", "o", "#1f77b4", "q_top"), ("w0", "^", "#2ca02c", "w0")):
        px, py, plo, phi, labels = [], [], [], [], []
        for (a, b) in pairs:
            r = results[f"{est_name}_L{a}{b}"]
            if r["q_c"] is None or r["CI95"][0] is None:
                continue
            px.append(2.0 / (a + b)); py.append(r["q_c"])
            plo.append(r["q_c"] - r["CI95"][0]); phi.append(r["CI95"][1] - r["q_c"])
            labels.append(f"L{a}{b}")
        axD.errorbar(px, py, yerr=[plo, phi], marker=mk, ms=6, ls="none", capsize=3,
                     color=col, label=lab, alpha=0.9)
        if est_name == "q_top":
            for x, y, s in zip(px, py, labels):
                axD.annotate(s, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    hl = results[f"q_top_L{Ls[0]}{Lmax}"]
    if hl["CI95"][0] is not None:
        axD.axhspan(hl["CI95"][0], hl["CI95"][1], color="#1f77b4", alpha=0.10,
                    label=f"q_top L{Ls[0]}-L{Lmax} CI")
    axD.set_xlabel("1 / L_mean"); axD.set_ylabel("q_c")
    axD.set_title("pairwise q_c convergence (all L-pairs)")
    axD.grid(alpha=0.35); axD.legend(fontsize=9); axD.set_xlim(left=0.0)

    out = SD / "p011_L3456_qc.png"
    fig.savefig(out, dpi=170); plt.close(fig)

    summary = {"p": p, "ndis_per_L": nd_per_L, "q_values": [float(x) for x in q],
               "crossings": results, "extrapolation_qtop_consecutive": extrap,
               "note": "L=3,4,5,6 x 384 disorders; q_top primary, w0 cross-check"}
    (SD / "p011_L3456_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nfig -> {out}")
    for k, v in results.items():
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
