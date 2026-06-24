#!/usr/bin/env python3
"""exp41 p=0.11 multi-estimator crossing analysis.

Reuses exp40/004 L=3,4,5,6 NPZ + (when present) exp41 L=7 NPZ, merges per-L along
the disorder axis, and computes ALL estimators uniformly from delta_f_per_disorder
via softmax — identical reconstruction to exp40's analyze_p011_L3456.py, so the
reused L3-6 and the new L7 sit on the same footing.

Estimators (all from w_g = softmax(-delta_f_g), sector 0 = true logical class):
  w0       = w_0                          # sign-aware  [headline]
  msigned  = (1/3) sum_u m_u              # sign-aware  (cross-check). SIGNED, not |.|:
                                          #   |m_u| can't tell true from a dominant
                                          #   wrong class -> behaves even-moment-like.
  q_W      = (1/3) sum_u m_u^2            # even-moment == exp40 published "q_top"
  q_purity = (8 sum_g w_g^2 - 1) / 7      # even-moment  (diagnostic)
with m_u = sum_g (-1)^{bit_u(g)} w_g  (true class g=0 -> every m_u = +w_0 contribution).

NB: NPZ scalar field q_top_per_disorder is a SEPARATE sample/TI estimate and is NOT
used here (differs from softmax reconstruction under finite statistics). Audit only.

Usage:
  python analyze_exp41_p011.py --globs <dir1> <dir2> ... [--nboot 10000] [--out-prefix p011]
Each glob dir is searched recursively for sector_ti_results.npz.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import numpy as np

SIGN = np.array([[1 - 2 * ((g >> i) & 1) for g in range(8)] for i in range(3)])
RNG = np.random.default_rng(20260620)


def estimators(delta_f):
    """delta_f: (..., 8) -> dict of (...) arrays. Softmax over last axis."""
    x = delta_f - delta_f.min(axis=-1, keepdims=True)
    w = np.exp(-x)
    w /= w.sum(axis=-1, keepdims=True)
    m = w @ SIGN.T                       # (..., 3)
    return {
        "w0": w[..., 0],
        "msigned": np.mean(m, axis=-1),                  # sign-aware (signed, not abs)
        "q_W": np.mean(m ** 2, axis=-1),
        "q_purity": (8.0 * np.sum(w ** 2, axis=-1) - 1.0) / 7.0,
    }


def load_merged(globs):
    """Merge blocks per L along disorder. Returns p, q grid, {L:{est:(nq,ndis)}}, nfiles."""
    files = []
    for g in globs:
        files += sorted(glob.glob(str(Path(g) / "**" / "sector_ti_results.npz"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no NPZ found under {globs}")
    p_value, qv = None, None
    acc = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        p_value = float(d["p_value"])
        q = d["q_values"].astype(float)
        if qv is None:
            qv = q
        elif not np.allclose(qv, q):
            raise SystemExit(f"q grid mismatch in {f}: {q} vs {qv}")
        df = d["delta_f_per_disorder"]            # (nL, nq, ndis, 8)
        for li, L in enumerate(int(x) for x in d["lattice_size_list"]):
            e = estimators(df[li])
            slot = acc.setdefault(L, {k: [] for k in e})
            for k in e:
                slot[k].append(e[k])
    data = {L: {k: np.concatenate(v[k], axis=1) for k in v} for L, v in acc.items()}
    return p_value, qv, data, len(files)


def cross_q(q, dvec):
    """First sign change of dvec(q); zero diffs count only when flanked by opposite signs."""
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


def crossing_ci(q, A_small, A_large, nboot):
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
    return qc, ci, len(qcs) / nboot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+", required=True)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p, q, data, nfiles = load_merged(args.globs)
    Ls = sorted(data)
    ndis = {L: data[L]["w0"].shape[1] for L in Ls}
    print(f"p={p}  files={nfiles}  Ls={Ls}  ndis={ndis}  nq={len(q)}")
    print(f"q grid: {list(np.round(q,4))}")

    pairs = [(a, b) for i, a in enumerate(Ls) for b in Ls[i + 1:]]
    ests = ["w0", "msigned", "q_W", "q_purity"]
    results = {}
    for est in ests:
        for (a, b) in pairs:
            qc, ci, bf = crossing_ci(q, data[a][est], data[b][est], args.nboot)
            results[f"{est}_L{a}{b}"] = {
                "q_c": None if qc is None else round(qc, 4),
                "CI95": [None if c is None else round(c, 4) for c in ci],
                "ci_halfwidth": (None if (qc is None or ci[0] is None)
                                 else round((ci[1] - ci[0]) / 2, 4)),
                "boot_frac": round(bf, 3),
            }

    # Print headline-relevant pairs
    print("\n-- crossings (q_c [CI95] halfwidth bf) --")
    for est in ests:
        print(f"  [{est}]")
        for (a, b) in pairs:
            r = results[f"{est}_L{a}{b}"]
            print(f"    L{a}-L{b}: {r['q_c']}  {r['CI95']}  hw={r['ci_halfwidth']}  bf={r['boot_frac']}")

    out = {
        "p": p, "lattice_sizes": Ls, "ndis_per_L": ndis,
        "q_values": [float(x) for x in q], "nboot": args.nboot,
        "estimators_note": "all from softmax(-delta_f); q_W == exp40 published q_top; stored q_top_per_disorder NOT used",
        "crossings": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
