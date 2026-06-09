#!/usr/bin/env python3
"""Corrected 007 boundary: recompute q_c(p) with the SIGN-AWARE w0 crossing
instead of the biased Δf-gap crossing (no new compute — reuses the per-disorder
sector weights already in the NPZs).

w0 = P(true logical class = sector 0) = softmax(-delta_f)[...,0]. It references
correctness (via η), so its L=3,4,5 crossing tracks the true correctability
threshold; the Δf-gap (even-moment) tracks a higher full-disorder crossover
(see 008/qnear0_wide: at q≈0 Δf-gap→0.40 but w0→0.25≈p_c). Also reports q_top
and signed-magnetization crossings for comparison.
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
RNG = np.random.default_rng(20260610)
NBOOT = 6000
PC0 = 0.227
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])  # [3,8]
SOURCES = sorted(glob.glob(str(SD / "collected" / "p*"))) + [
    str(EXP / "006_production_crossing_p006_20260606" / "collected"),
    str(EXP / "004_production_crossing_p012_20260605" / "collected"),
]


def estimators(delta_f):
    """delta_f [nq,ndis,8] -> dict of per-(q,disorder) estimators."""
    x = delta_f - delta_f.min(axis=2, keepdims=True)
    w = np.exp(-x); w /= w.sum(axis=2, keepdims=True)        # [nq,ndis,8]
    s = np.sort(delta_f, axis=2)
    m = w @ SIGN.T                                            # [nq,ndis,3]
    return {
        "w0": w[..., 0],                                      # [nq,ndis]
        "q_top": np.mean(m ** 2, axis=2),
        "signed_mag": np.mean(m, axis=2),
        "deltaf_gap": s[..., 1] - s[..., 0],
    }


def load_source(root):
    files = sorted(glob.glob(str(Path(root) / "**" / "sector_ti_results.npz"), recursive=True))
    if not files:
        return None
    acc = {}  # L -> list of (q[nq], {est: [nq,ndis]})
    p_value = None
    for f in files:
        d = np.load(f, allow_pickle=False); p_value = float(d["p_value"])
        qv = d["q_values"].astype(float)
        Ls = [int(x) for x in d["lattice_size_list"]]
        dfp = d["delta_f_per_disorder"]
        for li, L in enumerate(Ls):
            est = estimators(dfp[li])
            acc.setdefault(L, []).append((qv, est))
    data = {}
    for L, chunks in acc.items():
        qv = np.concatenate([c[0] for c in chunks])
        merged = {k: np.concatenate([c[1][k] for c in chunks], axis=0) for k in chunks[0][1]}
        o = np.argsort(qv); qv = qv[o]
        _, idx = np.unique(np.round(qv, 6), return_index=True)
        data[L] = (np.round(qv[idx], 6), {k: merged[k][o][idx] for k in merged})
    return p_value, data


def cross_q(q, dvec):
    for i in range(len(q) - 1):
        if dvec[i] == 0:
            return float(q[i])
        if dvec[i] * dvec[i + 1] < 0:
            t = dvec[i] / (dvec[i] - dvec[i + 1]); return float(q[i] + t * (q[i + 1] - q[i]))
    return None


def crossing_with_ci(q, A3, A5):
    """A3,A5 = [nq,ndis]; L3-L5 crossing of disorder-mean curves + bootstrap CI."""
    m3, m5 = np.nanmean(A3, axis=1), np.nanmean(A5, axis=1)
    qc = cross_q(q, m3 - m5)
    nd = A3.shape[1]; qcs = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, nd, nd)
        c = cross_q(q, np.nanmean(A3[:, idx], axis=1) - np.nanmean(A5[:, idx], axis=1))
        if c is not None:
            qcs.append(c)
    ci = ([round(float(np.quantile(qcs, 0.025)), 4), round(float(np.quantile(qcs, 0.975)), 4)]
          if len(qcs) > 10 else [None, None])
    return qc, ci, round(len(qcs) / NBOOT, 3)


def main():
    points = []
    for root in SOURCES:
        res = load_source(root)
        if res is None:
            continue
        p, data = res
        if 3 not in data or 5 not in data:
            continue
        q = data[3][0]
        rec = {"p": p, "q": q, "ndis": data[3][1]["w0"].shape[1], "cross": {}}
        for est in ["w0", "q_top", "signed_mag", "deltaf_gap"]:
            qc, ci, bf = crossing_with_ci(q, data[3][1][est], data[5][1][est])
            rec["cross"][est] = {"q_c": None if qc is None else round(qc, 4), "CI95": ci, "boot_frac": bf}
        rec["w0_curves"] = {L: np.nanmean(data[L][1]["w0"], axis=1) for L in (3, 4, 5)}
        points.append(rec)
        print(f"p={p:<5} ndis={rec['ndis']:>3}  "
              + "  ".join(f"{e}:{rec['cross'][e]['q_c']}" for e in ['w0', 'q_top', 'deltaf_gap']))
    points.sort(key=lambda r: r["p"])

    # ---- boundary comparison figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    style = {"w0": ("#2ca02c", "o", "w0 = P(true class)  [corrected]"),
             "q_top": ("#1f77b4", "^", "q_top"),
             "deltaf_gap": ("#9467bd", "s", "Δf-gap  [007, biased]")}
    for est, (c, mk, lab) in style.items():
        ps = [r["p"] for r in points if r["cross"][est]["q_c"] is not None]
        qc = [r["cross"][est]["q_c"] for r in points if r["cross"][est]["q_c"] is not None]
        ax1.plot(ps, qc, marker=mk, ms=7, lw=1.5, color=c, label=lab)
    ax1.scatter([PC0], [0.0], marker="*", s=220, color="#ff7f0e", zorder=5, label=f"q=0 endpoint p_c≈{PC0:g}")
    ax1.set_xlabel("p  (Pauli-X error rate)"); ax1.set_ylabel("q_c  (measurement threshold)")
    ax1.set_title("007 boundary: corrected (w0) vs biased (Δf-gap)")
    ax1.grid(alpha=0.35); ax1.legend(fontsize=9); ax1.set_ylim(bottom=0)
    # per-p w0(q) curves for a representative p (the one closest to 0.12)
    rp = min(points, key=lambda r: abs(r["p"] - 0.12))
    for L in (3, 4, 5):
        ax2.plot(rp["q"], rp["w0_curves"][L], marker="o", ms=4, color=COLORS[L], label=f"L={L}")
    wc = rp["cross"]["w0"]["q_c"]
    if wc:
        ax2.axvline(wc, color="0.4", ls="--", lw=1.1, label=f"w0 crossing q_c≈{wc:.3f}")
    dc = rp["cross"]["deltaf_gap"]["q_c"]
    if dc:
        ax2.axvline(dc, color="#9467bd", ls=":", lw=1.2, label=f"Δf-gap q_c≈{dc:.3f}")
    ax2.set_xlabel("q"); ax2.set_ylabel("w0 = P(true class)")
    ax2.set_title(f"w0 vs q at p={rp['p']:g} (L=3,4,5)")
    ax2.grid(alpha=0.35); ax2.legend(fontsize=9)
    out = SD / "boundary_corrected_w0.png"
    fig.savefig(out, dpi=170); plt.close(fig)

    summary = {"estimator_note": "w0=P(true class) is sign-aware (correct); deltaf_gap is biased high",
               "pc0_endpoint": PC0,
               "points": [{"p": r["p"], "ndis": r["ndis"], "crossings": r["cross"]} for r in points]}
    (SD / "boundary_corrected_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n  p     w0_q_c   q_top_q_c  deltaf_q_c   (w0 CI95)")
    for r in points:
        c = r["cross"]
        print(f"  {r['p']:<5} {str(c['w0']['q_c']):>7}   {str(c['q_top']['q_c']):>7}    "
              f"{str(c['deltaf_gap']['q_c']):>7}     {c['w0']['CI95']}")
    print(f"\nfig -> {out}")


if __name__ == "__main__":
    main()
