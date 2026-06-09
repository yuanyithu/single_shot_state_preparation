#!/usr/bin/env python3
"""q≈0 FSS comparison of candidate threshold estimators (L=3,4,5,6).
For each estimator: consecutive-L crossings -> linear 1/L_eff extrapolation.
Δf-gap (nearest competitor) is biased (pinned ~0.40); q_top / w0 / signed
magnetization drift toward the true p_c≈0.233; Binder (even moments) is
degenerate (2/3) in the pure phase. Saves a comparison figure."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

QD = Path(__file__).resolve().parent / "collected"
SD = Path(__file__).resolve().parent
PC = 0.233
SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])


def load():
    rec = {}
    for f in sorted(glob.glob(str(QD / "p*" / "**" / "sector_ti_results.npz"), recursive=True)):
        d = np.load(f, allow_pickle=False); p = round(float(d["p_value"]), 4)
        for li, L in enumerate(int(x) for x in d["lattice_size_list"]):
            x = d["delta_f_per_disorder"][li, 0]
            rec.setdefault(L, {})[p] = x
    return rec


def cross(ps, dv):
    for i in range(len(ps) - 1):
        if dv[i] * dv[i + 1] < 0:
            t = dv[i] / (dv[i] - dv[i + 1]); return ps[i] + t * (ps[i + 1] - ps[i])
    return None


def main():
    rec = load(); Ls = sorted(rec); ps = sorted(rec[Ls[0]])
    Q = {k: {L: [] for L in Ls} for k in ["Δf-gap", "q_top", "w0", "signed-mag"]}
    for L in Ls:
        for p in ps:
            x = rec[L][p]; xs = x - x.min(axis=1, keepdims=True)
            w = np.exp(-xs); w /= w.sum(axis=1, keepdims=True)
            s = np.sort(x, axis=1)
            Q["Δf-gap"][L].append(float(np.mean(s[:, 1] - s[:, 0])))
            m = w @ SIGN.T
            Q["q_top"][L].append(float(np.mean(m ** 2)))
            Q["w0"][L].append(float(np.mean(w[:, 0])))
            Q["signed-mag"][L].append(float(np.mean(m)))

    colors = {"Δf-gap": "#9467bd", "q_top": "#1f77b4", "w0": "#2ca02c", "signed-mag": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    out = {}
    for k in Q:
        pairs = list(zip(Ls[:-1], Ls[1:]))
        xc = []
        for a, b in pairs:
            dv = [Q[k][a][i] - Q[k][b][i] for i in range(len(ps))]
            c = cross(ps, dv); xc.append((0.5 * (a + b), c))
        pts = [(le, c) for le, c in xc if c is not None]
        if len(pts) >= 2:
            le = np.array([t[0] for t in pts]); cc = np.array([t[1] for t in pts])
            b1, b0 = np.polyfit(1.0 / le, cc, 1)
            out[k] = round(float(b0), 4)
            ax.plot(1.0 / le, cc, "o", ms=8, color=colors[k])
            xx = np.linspace(0, (1.0 / le).max() * 1.1, 50)
            ax.plot(xx, b1 * xx + b0, "-", lw=1.5, color=colors[k],
                    label=f"{k}: extrap→{b0:.3f}")
            ax.scatter([0], [b0], marker="*", s=160, color=colors[k], zorder=5)
    ax.axhline(PC, color="0.2", ls="--", lw=1.8, label=f"true p_c≈{PC} (3D RBIM)")
    ax.set_xlabel("1 / L_eff"); ax.set_ylabel("crossing p")
    ax.set_title("q≈0 FSS: which estimator's crossing converges to the true p_c?")
    ax.set_xlim(left=-0.012); ax.grid(alpha=0.35); ax.legend(fontsize=9, loc="center right")
    figpath = SD / "order_param_fss.png"
    fig.savefig(figpath, dpi=170); plt.close(fig)
    (SD / "order_param_fss_summary.json").write_text(
        json.dumps({"true_pc": PC, "extrapolated_pc": out}, indent=2), encoding="utf-8")
    print("1/L→0 extrapolated p_c:")
    for k, v in out.items():
        print(f"  {k:12s} -> {v}   (true {PC})")
    print(f"fig -> {figpath}")


if __name__ == "__main__":
    main()
