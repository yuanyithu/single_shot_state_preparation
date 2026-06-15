#!/usr/bin/env python3
"""exp40/005: high-statistics measurement-error phase boundary q_c(p).

Each p-point = 384 disorders (3 node blocks merged along the disorder axis),
L=3,4,5, dense centered q grid. Auto-discovers which p-points in 005 are
complete (all 3 node blocks present); always includes p=0.11 from 004
(L=3,4,5; the L=6 refinement lives in 004 and is reported separately).

Headline estimator: q_top L3-L5 crossing (6000-disorder bootstrap CI).
Cross-check: w0 = P(true logical class). Never Δf-gap / ais (see CLAUDE.md).

Output: qc_table.{json,md}, qtop_curves_grid.png (per-p errorbar curves),
phase_boundary_highstats.png (boundary vs the biased 48-disorder exp40/002 pts).
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
EXP40 = SD.parent
RNG = np.random.default_rng(20260614)
NBOOT = 6000
PC0 = 0.227
COLORS = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])
P004 = EXP40 / "004_p011_highstats_20260611"


def estimators(delta_f):
    x = delta_f - delta_f.min(axis=2, keepdims=True)
    w = np.exp(-x); w /= w.sum(axis=2, keepdims=True)
    m = w @ SIGN.T
    return {"q_top": np.mean(m ** 2, axis=2), "w0": w[..., 0]}


def merge_blocks(files):
    """List of NPZ files (same p, same q grid) -> (p, q, {L:{est:[nq,ndis]}})."""
    p_value, qv = None, None
    acc = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        p_value = float(d["p_value"])
        if qv is None:
            qv = d["q_values"].astype(float)
        else:
            assert np.allclose(qv, d["q_values"].astype(float)), f"q grid mismatch {f}"
        for li, L in enumerate([int(x) for x in d["lattice_size_list"]]):
            e = estimators(d["delta_f_per_disorder"][li])
            slot = acc.setdefault(L, {k: [] for k in e})
            for k in e:
                slot[k].append(e[k])
    data = {L: {k: np.concatenate(v[k], axis=1) for k in v} for L, v in acc.items()}
    return p_value, qv, data


def discover_points():
    """Return {p_tag: [npz files]} for complete (3-block) points in 005 + 004 p=0.11."""
    points = {}
    # 005: group by ptag, require >=3 node blocks (the 3 nd dirs)
    by_tag = {}
    for f in glob.glob(str(SD / "nd*" / "collected" / "p*" / "sector_ti_results.npz")):
        tag = Path(f).parent.name
        by_tag.setdefault(tag, []).append(f)
    for tag, files in by_tag.items():
        if len(files) >= 3:
            points[tag] = sorted(files)
    # 004 p=0.11 (L=3,4,5 blocks: nd1/nd2/nd3/collected/p0p11)
    p011 = sorted(glob.glob(str(P004 / "nd[123]" / "collected" / "p0p11" / "sector_ti_results.npz")))
    if len(p011) >= 3:
        points["p0p11"] = p011
    return points


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


def crossing_with_ci(q, A_small, A_large, nboot=NBOOT):
    nd = min(A_small.shape[1], A_large.shape[1])
    A_small, A_large = A_small[:, :nd], A_large[:, :nd]
    qc = cross_q(q, np.nanmean(A_small, axis=1) - np.nanmean(A_large, axis=1))
    qcs = []
    for _ in range(nboot):
        idx = RNG.integers(0, nd, nd)
        c = cross_q(q, np.nanmean(A_small[:, idx], axis=1) - np.nanmean(A_large[:, idx], axis=1))
        if c is not None:
            qcs.append(c)
    ci = ([round(float(np.quantile(qcs, 0.025)), 4), round(float(np.quantile(qcs, 0.975)), 4)]
          if len(qcs) > 10 else [None, None])
    return (None if qc is None else round(qc, 4)), ci, round(len(qcs) / nboot, 3)


def load_old48():
    """exp40/002 48-disorder q_top L3-L5 crossings, for greyed comparison."""
    f = EXP40 / "003_boundary_analysis_20260610" / "qc_table.json"
    if not f.exists():
        return []
    pts = json.loads(f.read_text())["points"]
    return [(r["p"], r["crossings"]["q_top_L35"]["q_c"]) for r in pts
            if r["crossings"]["q_top_L35"]["q_c"] is not None]


def main():
    points = discover_points()
    recs = []
    for tag, files in sorted(points.items()):
        p, q, data = merge_blocks(files)
        Ls = sorted(data)
        ndis = data[Ls[0]]["q_top"].shape[1]
        rec = {"p": p, "tag": tag, "ndis": ndis, "Ls": Ls, "q": q, "data": data, "cross": {}}
        for est in ("q_top", "w0"):
            for (a, b) in [(3, 4), (3, 5), (4, 5)]:
                if a in data and b in data:
                    qc, ci, bf = crossing_with_ci(q, data[a][est], data[b][est])
                    rec["cross"][f"{est}_L{a}{b}"] = {"q_c": qc, "CI95": ci, "boot_frac": bf}
        recs.append(rec)
        h = rec["cross"].get("q_top_L35", {})
        print(f"p={p:<5} ndis={ndis:>3} Ls={Ls}  q_top_L35={h.get('q_c')} CI={h.get('CI95')} bf={h.get('boot_frac')}")
    recs.sort(key=lambda r: r["p"])

    # ---- qc table ----
    table = {"pc0_endpoint": PC0, "nboot": NBOOT,
             "note": "high-stats (384 disorder) boundary; q_top L3-L5 headline, w0 cross-check; p=0.11 from 004",
             "points": [{"p": r["p"], "ndis": r["ndis"], "Ls": r["Ls"], "crossings": r["cross"]} for r in recs]}
    (SD / "qc_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    md = ["# exp40/005 high-stats q_c(p) table", "",
          f"384 disorder/point; bootstrap N={NBOOT}; headline = q_top L3-L5", "",
          "| p | ndis | q_c(q_top L35) | CI95 | bf | q_c(w0 L35) | CI95(w0) |",
          "|---|---|---|---|---|---|---|"]
    for r in recs:
        t = r["cross"].get("q_top_L35", {}); w = r["cross"].get("w0_L35", {})
        md.append(f"| {r['p']:g} | {r['ndis']} | {t.get('q_c')} | {t.get('CI95')} | {t.get('boot_frac')} "
                  f"| {w.get('q_c')} | {w.get('CI95')} |")
    (SD / "qc_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- per-p q_top curves ----
    n = len(recs); ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.7 * nrow), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, r in zip(axes.flat, recs):
        for L in r["Ls"]:
            mean = r["data"][L]["q_top"].mean(axis=1)
            sem = r["data"][L]["q_top"].std(axis=1, ddof=1) / np.sqrt(r["ndis"])
            ax.errorbar(r["q"], mean, yerr=sem, marker="o", ms=3.5, lw=1.3, capsize=2,
                        color=COLORS[L], label=f"L={L}")
        h = r["cross"].get("q_top_L35", {})
        if h.get("q_c") is not None:
            ax.axvline(h["q_c"], color="0.3", ls="--", lw=1.0)
            if h["CI95"][0] is not None:
                ax.axvspan(h["CI95"][0], h["CI95"][1], color="0.6", alpha=0.16)
        ax.set_title(f"p={r['p']:g}  q_c={h.get('q_c')}  (n={r['ndis']})", fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("q_top ± SEM"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(SD / "qtop_curves_grid.png", dpi=160); plt.close(fig)

    # ---- boundary ----
    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    old = load_old48()
    if old:
        ax.scatter([p for p, _ in old], [qc for _, qc in old], s=45, color="0.7",
                   marker="x", label="exp40/002 48-dis q_top (biased low)", zorder=1)
    ps, qcs, lo, hi = [], [], [], []
    for r in recs:
        h = r["cross"].get("q_top_L35", {})
        if h.get("q_c") is None:
            continue
        ps.append(r["p"]); qcs.append(h["q_c"])
        lo.append(h["q_c"] - h["CI95"][0] if h["CI95"][0] is not None else 0)
        hi.append(h["CI95"][1] - h["q_c"] if h["CI95"][1] is not None else 0)
    ax.errorbar(ps, qcs, yerr=[lo, hi], marker="o", ms=8, lw=1.8, capsize=4, color="#1f77b4",
                label="exp40/005 q_top L3-L5 (384 dis)", zorder=4)
    wps = [r["p"] for r in recs if r["cross"].get("w0_L35", {}).get("q_c") is not None]
    wqc = [r["cross"]["w0_L35"]["q_c"] for r in recs if r["cross"].get("w0_L35", {}).get("q_c") is not None]
    ax.plot(wps, wqc, marker="^", ms=6, ls=":", lw=1.0, color="#2ca02c", label="w0 cross-check", zorder=3)
    ax.scatter([PC0], [0.0], marker="*", s=240, color="#ff7f0e", zorder=5, label=f"q=0 endpoint p_c≈{PC0:g}")
    if ps:
        ax.plot([ps[-1], PC0], [qcs[-1], 0.0], ls="--", lw=1.0, color="0.6", zorder=1)
    ax.set_xlabel("p  (Pauli-X error rate)"); ax.set_ylabel("q_c  (measurement-error threshold)")
    ax.set_title("3D toric code measurement-error phase boundary q_c(p) — high statistics")
    ax.grid(alpha=0.35); ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    fig.savefig(SD / "phase_boundary_highstats.png", dpi=170); plt.close(fig)
    print(f"\n-> qc_table.md, qtop_curves_grid.png, phase_boundary_highstats.png in {SD.name}/")


if __name__ == "__main__":
    main()
