#!/usr/bin/env python3
"""exp40/003: q_c(p) phase boundary from the dense centered q-grid (exp40/002).

Primary estimator: q_top L-pair crossing (user-confirmed observable for the
threshold). Secondary cross-check: sign-aware w0 = P(true logical class)
= softmax(-delta_f)[..., 0] (exp39/008 showed Δf-gap is biased high; w0 and
q_top crossings both converge to the true correctability threshold).

Per p: crossings for L-pairs (3,5) [headline], (3,4), (4,5), each with a
disorder-bootstrap CI. Outputs qc_table.{json,md}, per-p crossing panels and
the final phase_boundary_qtop.png with exp39/007 corrected points overlaid.

Usage:
  python analyze_exp40_boundary.py                # production: ../002_*/nd*/collected/p*
  python analyze_exp40_boundary.py --smoke DIR    # validate code path on a smoke NPZ
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
EXP40 = SD.parent
WMN = EXP40.parent
EXP39_SUMMARY = (WMN / "exp39_q_threshold_scout_20260605"
                 / "007_phase_boundary_deltaf_20260608" / "boundary_corrected_summary.json")
RNG = np.random.default_rng(20260610)
NBOOT = 6000
PC0 = 0.227
COLORS = {2: "#8c564b", 3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])  # [3,8]


def estimators(delta_f):
    """delta_f [nq,ndis,8] -> dict of per-(q,disorder) estimators."""
    x = delta_f - delta_f.min(axis=2, keepdims=True)
    w = np.exp(-x); w /= w.sum(axis=2, keepdims=True)        # [nq,ndis,8]
    m = w @ SIGN.T                                            # [nq,ndis,3]
    return {
        "q_top": np.mean(m ** 2, axis=2),                     # [nq,ndis]
        "w0": w[..., 0],
    }


def load_source(root):
    """One p-cell dir -> (p_value, {L: (q[nq], {est: [nq,ndis]})})."""
    files = sorted(glob.glob(str(Path(root) / "**" / "sector_ti_results.npz"), recursive=True))
    if not files:
        return None
    acc = {}
    p_value = None
    for f in files:
        d = np.load(f, allow_pickle=False); p_value = float(d["p_value"])
        qv = d["q_values"].astype(float)
        Ls = [int(x) for x in d["lattice_size_list"]]
        dfp = d["delta_f_per_disorder"]
        for li, L in enumerate(Ls):
            acc.setdefault(L, []).append((qv, estimators(dfp[li])))
    data = {}
    for L, chunks in acc.items():
        qv = np.concatenate([c[0] for c in chunks])
        merged = {k: np.concatenate([c[1][k] for c in chunks], axis=0) for k in chunks[0][1]}
        o = np.argsort(qv); qv = qv[o]
        _, idx = np.unique(np.round(qv, 6), return_index=True)
        data[L] = (np.round(qv[idx], 6), {k: merged[k][o][idx] for k in merged})
    return p_value, data


def cross_q(q, dvec):
    """First sign change of dvec(q). Exact zeros (saturated region where both
    curves coincide, e.g. q_top=1 for every disorder at small q) are NOT
    crossings by themselves; a zero only counts if flanked by opposite signs."""
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


def crossing_with_ci(q, A_small, A_large, nboot=NBOOT):
    """A_* = [nq,ndis]; crossing of disorder-mean curves + disorder bootstrap CI."""
    qc = cross_q(q, np.nanmean(A_small, axis=1) - np.nanmean(A_large, axis=1))
    nd = A_small.shape[1]; qcs = []
    for _ in range(nboot):
        idx = RNG.integers(0, nd, nd)
        c = cross_q(q, np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1))
        if c is not None:
            qcs.append(c)
    ci = ([round(float(np.quantile(qcs, 0.025)), 4), round(float(np.quantile(qcs, 0.975)), 4)]
          if len(qcs) > 10 else [None, None])
    return qc, ci, round(len(qcs) / nboot, 3)


def load_exp39_overlay():
    if not EXP39_SUMMARY.exists():
        return []
    pts = json.loads(EXP39_SUMMARY.read_text(encoding="utf-8"))["points"]
    out = []
    for r in pts:
        for est in ("q_top", "w0"):
            c = r["crossings"].get(est, {})
            if c.get("q_c") is not None:
                out.append((r["p"], c["q_c"], est))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default=None, help="single NPZ dir to validate the code path")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    args = ap.parse_args()

    if args.smoke:
        sources = [args.smoke]
        out_prefix = "smoke_"
    else:
        sources = sorted(glob.glob(str(EXP40 / "002_production_boundary_20260610" / "nd*" / "collected" / "p*")))
        out_prefix = ""

    points = []
    for root in sources:
        res = load_source(root)
        if res is None:
            continue
        p, data = res
        Ls = sorted(data.keys())
        pairs = [(a, b) for i, a in enumerate(Ls) for b in Ls[i + 1:]]
        q = data[Ls[0]][0]
        rec = {"p": p, "q": q, "ndis": data[Ls[0]][1]["q_top"].shape[1],
               "Ls": Ls, "cross": {}, "curves": {}}
        for est in ("q_top", "w0"):
            rec["curves"][est] = {L: np.nanmean(data[L][1][est], axis=1) for L in Ls}
            for (a, b) in pairs:
                qc, ci, bf = crossing_with_ci(q, data[a][1][est], data[b][1][est], args.nboot)
                rec["cross"][f"{est}_L{a}{b}"] = {
                    "q_c": None if qc is None else round(qc, 4), "CI95": ci, "boot_frac": bf}
        points.append(rec)
        head = [k for k in rec["cross"] if k.startswith("q_top")]
        print(f"p={p:<5} ndis={rec['ndis']:>3} Ls={Ls}  "
              + "  ".join(f"{k}:{rec['cross'][k]['q_c']}" for k in head))
    points.sort(key=lambda r: r["p"])
    if not points:
        raise SystemExit("no sources found")

    headline = None
    for cand in ("q_top_L35", "q_top_L23"):
        if cand in points[0]["cross"]:
            headline = cand; break
    w0_headline = headline.replace("q_top", "w0")

    # ---- qc table (json + md) ----
    table = {"headline_pair": headline, "pc0_endpoint": PC0, "nboot": args.nboot,
             "estimator_note": "q_top primary (user-confirmed); w0 sign-aware cross-check; "
                               "never Δf-gap (biased high, exp39/008)",
             "points": [{"p": r["p"], "ndis": r["ndis"], "Ls": r["Ls"],
                         "crossings": r["cross"]} for r in points]}
    (SD / f"{out_prefix}qc_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    md = ["# exp40 q_c(p) crossing table", "",
          f"headline pair: `{headline}`; bootstrap N={args.nboot} (disorder resampling)", "",
          "| p | ndis | q_c(q_top) | CI95 | boot_frac | q_c(w0) | CI95(w0) |",
          "|---|---|---|---|---|---|---|"]
    for r in points:
        ct, cw = r["cross"][headline], r["cross"][w0_headline]
        md.append(f"| {r['p']:g} | {r['ndis']} | {ct['q_c']} | {ct['CI95']} | {ct['boot_frac']} "
                  f"| {cw['q_c']} | {cw['CI95']} |")
    (SD / f"{out_prefix}qc_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- per-p crossing panels (q_top curves per L) ----
    n = len(points)
    ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, r in zip(axes.flat, points):
        for L in r["Ls"]:
            ax.plot(r["q"], r["curves"]["q_top"][L], marker="o", ms=3.5,
                    color=COLORS.get(L, "0.4"), label=f"L={L}")
        c = r["cross"][headline]
        if c["q_c"] is not None:
            ax.axvline(c["q_c"], color="0.3", ls="--", lw=1.0)
            lo, hi = c["CI95"]
            if lo is not None:
                ax.axvspan(lo, hi, color="0.6", alpha=0.18)
        ax.set_title(f"p={r['p']:g}  q_c={c['q_c']}", fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("q_top"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    panels_out = SD / f"{out_prefix}qtop_crossings_grid.png"
    fig.savefig(panels_out, dpi=160); plt.close(fig)

    # ---- final boundary figure ----
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    ps, qcs, lo_e, hi_e = [], [], [], []
    for r in points:
        c = r["cross"][headline]
        if c["q_c"] is None:
            continue
        ps.append(r["p"]); qcs.append(c["q_c"])
        lo, hi = c["CI95"]
        lo_e.append(c["q_c"] - lo if lo is not None else 0)
        hi_e.append(hi - c["q_c"] if hi is not None else 0)
    ax.errorbar(ps, qcs, yerr=[lo_e, hi_e], marker="o", ms=7, lw=1.8, capsize=3,
                color="#1f77b4", label=f"exp40 q_top crossing ({headline.split('_')[-1]})", zorder=4)
    w0_ps = [r["p"] for r in points if r["cross"][w0_headline]["q_c"] is not None]
    w0_qc = [r["cross"][w0_headline]["q_c"] for r in points if r["cross"][w0_headline]["q_c"] is not None]
    ax.plot(w0_ps, w0_qc, marker="^", ms=6, lw=1.0, ls=":", color="#2ca02c",
            label="exp40 w0 cross-check", zorder=3)
    overlay = load_exp39_overlay()
    o_ps = [p for p, _, e in overlay if e == "q_top"]
    o_qc = [qc for _, qc, e in overlay if e == "q_top"]
    if o_ps:
        ax.scatter(o_ps, o_qc, facecolors="none", edgecolors="#d62728", s=70, marker="o",
                   label="exp39/007 corrected (q_top, coarse grid)", zorder=2)
    ax.scatter([PC0], [0.0], marker="*", s=240, color="#ff7f0e", zorder=5,
               label=f"q=0 endpoint p_c≈{PC0:g}")
    if ps:
        # schematic closure toward the q=0 endpoint
        ax.plot([ps[-1], PC0], [qcs[-1], 0.0], ls="--", lw=1.0, color="0.6", zorder=1)
    ax.set_xlabel("p  (Pauli-X error rate)")
    ax.set_ylabel("q_c  (measurement-error threshold)")
    ax.set_title("3D toric code measurement-error phase boundary q_c(p) — exp40 dense grid")
    ax.grid(alpha=0.35); ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    boundary_out = SD / f"{out_prefix}phase_boundary_qtop.png"
    fig.savefig(boundary_out, dpi=170); plt.close(fig)

    print(f"\nqc_table -> {SD / (out_prefix + 'qc_table.md')}")
    print(f"panels   -> {panels_out}")
    print(f"boundary -> {boundary_out}")


if __name__ == "__main__":
    main()
