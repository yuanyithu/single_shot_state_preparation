#!/usr/bin/env python3
"""exp40/004: single-point high-statistics q_c determination at p=0.11.

Merges the three per-node disorder blocks (seeds 860000/861000/862000, 128 each
-> 384 disorders) along the disorder axis, then produces ONE clear figure:
  (a) q_top(q) per L with disorder-SEM error bars (full range),
  (b) zoom on the crossing region,
  (c) pairwise differences q_top(L_small)-q_top(L_large) with bootstrap CI band
      -> sign change = q_c, with 6000-bootstrap CI.
w0 = P(true class) crossings reported as cross-check.

Usage: python analyze_p011_highstats.py [--sources DIR ...] [--nboot N]
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
RNG = np.random.default_rng(20260611)
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
PAIR_COLORS = {(3, 4): "#9467bd", (3, 5): "#1f77b4", (4, 5): "#8c564b"}
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])


def estimators(delta_f):
    x = delta_f - delta_f.min(axis=2, keepdims=True)
    w = np.exp(-x); w /= w.sum(axis=2, keepdims=True)
    m = w @ SIGN.T
    return {"q_top": np.mean(m ** 2, axis=2), "w0": w[..., 0]}


def load_merged(sources):
    """Merge NPZ blocks with identical q grids along the disorder axis."""
    files = []
    for root in sources:
        files += sorted(glob.glob(str(Path(root) / "**" / "sector_ti_results.npz"), recursive=True))
    if not files:
        raise SystemExit("no NPZ found under: " + ", ".join(map(str, sources)))
    p_value, qv, Ls = None, None, None
    blocks = {"q_top": [], "w0": []}
    for f in files:
        d = np.load(f, allow_pickle=False)
        p_value = float(d["p_value"])
        if qv is None:
            qv = d["q_values"].astype(float)
            Ls = [int(x) for x in d["lattice_size_list"]]
        else:
            assert np.allclose(qv, d["q_values"].astype(float)), f"q grid mismatch in {f}"
            assert Ls == [int(x) for x in d["lattice_size_list"]], f"L list mismatch in {f}"
        est = {k: [] for k in blocks}
        for li in range(len(Ls)):
            e = estimators(d["delta_f_per_disorder"][li])
            for k in blocks:
                est[k].append(e[k])
        for k in blocks:
            blocks[k].append(np.stack(est[k]))            # [nL,nq,ndis_block]
    merged = {k: np.concatenate(blocks[k], axis=2) for k in blocks}   # [nL,nq,ndis]
    return p_value, qv, Ls, merged, len(files)


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
    qc = cross_q(q, np.nanmean(A_small, axis=1) - np.nanmean(A_large, axis=1))
    nd = A_small.shape[1]; qcs = []
    for _ in range(nboot):
        idx = RNG.integers(0, nd, nd)
        c = cross_q(q, np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1))
        if c is not None:
            qcs.append(c)
    ci = ([float(np.quantile(qcs, 0.025)), float(np.quantile(qcs, 0.975))]
          if len(qcs) > 10 else [None, None])
    return qc, ci, len(qcs) / nboot, np.array(qcs)


def diff_band(q, A_small, A_large, nboot):
    """Bootstrap CI band of the mean-difference curve."""
    nd = A_small.shape[1]
    diffs = np.empty((nboot, len(q)))
    for b in range(nboot):
        idx = RNG.integers(0, nd, nd)
        diffs[b] = np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1)
    return np.quantile(diffs, 0.025, axis=0), np.quantile(diffs, 0.975, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--nboot", type=int, default=6000)
    args = ap.parse_args()
    sources = args.sources or [SD / f"nd{k}" / "collected" for k in (1, 2, 3)]

    p, q, Ls, est, nfiles = load_merged(sources)
    qt = est["q_top"]
    ndis = qt.shape[2]
    print(f"p={p}  merged blocks={nfiles}  ndis={ndis}  q={q}")

    pairs = [(a, b) for i, a in enumerate(Ls) for b in Ls[i + 1:]]
    results = {}
    for est_name in ("q_top", "w0"):
        A = est[est_name]
        for (a, b) in pairs:
            ia, ib = Ls.index(a), Ls.index(b)
            qc, ci, bf, qcs = crossing_with_ci(q, A[ia], A[ib], args.nboot)
            results[f"{est_name}_L{a}{b}"] = {
                "q_c": None if qc is None else round(qc, 4),
                "CI95": [None if c is None else round(c, 4) for c in ci],
                "boot_frac": round(bf, 3)}
            if est_name == "q_top":
                print(f"q_top L{a}-L{b}: q_c={qc and round(qc,4)}  CI={ci and [round(c,4) for c in ci]}  bf={bf:.3f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    mean = qt.mean(axis=2); sem = qt.std(axis=2, ddof=1) / np.sqrt(ndis)
    for li, L in enumerate(Ls):
        for ax in axes[:2]:
            ax.errorbar(q, mean[li], yerr=sem[li], marker="o", ms=4.5, lw=1.5,
                        capsize=2.5, color=COLORS[L], label=f"L={L}")
    qc35 = results["q_top_L35"]
    for ax in axes[:2]:
        if qc35["q_c"] is not None:
            ax.axvline(qc35["q_c"], color="0.3", ls="--", lw=1.2)
            if qc35["CI95"][0] is not None:
                ax.axvspan(qc35["CI95"][0], qc35["CI95"][1], color="0.6", alpha=0.18)
        ax.grid(alpha=0.35); ax.legend(fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("q_top (disorder mean ± SEM)")
    axes[0].set_title(f"p={p}: q_top(q), {ndis} disorders/point")
    zoom_hi = (qc35["CI95"][1] or 0.05) + 0.012
    axes[1].set_xlim(q[0], zoom_hi)
    ymin = min(mean[li][q <= zoom_hi].min() for li in range(len(Ls))) - 0.004
    axes[1].set_ylim(ymin, 1.0015)
    axes[1].set_title(f"crossing zoom: q_c(L3-L5)={qc35['q_c']}  CI95={qc35['CI95']}")

    ax = axes[2]
    for (a, b) in pairs:
        ia, ib = Ls.index(a), Ls.index(b)
        d = mean[ia] - mean[ib]
        lo, hi = diff_band(q, qt[ia], qt[ib], min(args.nboot, 2000))
        c = PAIR_COLORS[(a, b)]
        ax.plot(q, d, marker="o", ms=4, lw=1.5, color=c, label=f"L{a} − L{b}")
        ax.fill_between(q, lo, hi, color=c, alpha=0.15)
    ax.axhline(0, color="k", lw=1)
    if qc35["q_c"] is not None:
        ax.axvline(qc35["q_c"], color="0.3", ls="--", lw=1.2)
    ax.set_xlabel("q"); ax.set_ylabel("Δ q_top (small L − large L)")
    ax.set_title("pairwise differences: sign change = q_c")
    ax.grid(alpha=0.35); ax.legend(fontsize=10)
    out = SD / "p011_highstats_qc.png"
    fig.savefig(out, dpi=170); plt.close(fig)

    summary = {"p": p, "ndis": ndis, "n_blocks": nfiles, "q_values": [float(x) for x in q],
               "crossings": results,
               "note": "384-disorder high-stats single point; q_top primary, w0 cross-check"}
    (SD / "p011_highstats_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nfig -> {out}")
    for k in ("q_top_L34", "q_top_L35", "q_top_L45", "w0_L34", "w0_L35", "w0_L45"):
        print(f"  {k:10s} {results[k]}")


if __name__ == "__main__":
    main()
