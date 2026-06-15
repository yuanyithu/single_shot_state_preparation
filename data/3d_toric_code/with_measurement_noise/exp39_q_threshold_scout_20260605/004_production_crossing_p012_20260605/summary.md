# exp39 production crossing — p=0.12, L=3,4,5, 96 disorders (with low-q ordered side)

Runs: `exp39_prodp012_20260605_1431` (q=0.02..0.24) + `exp39_prodp012lowq_20260606_0150` (q=0.005/0.01/0.015), merged (same 96 disorders, seed_base 800000). nd1:L3 / nd2:L4 / nd3:L5. Observable: **TI / `projection_mode=linear`** (correct; `ais`/`decoder_reject` is the buggy x+r(Hx) label — see `exp37-decoder-sector-bug`). TI config: grid 129, 8192 measurements, burn 512, `disorder_seed_scope=disorder_index`, `rng_stream`. Disorder-bootstrap error bars (20000 reps).

## Result: clean two-sided finite-size crossing → measurement-error threshold at p=0.12

- **Ordered side anchored:** q ≤ 0.015 → q_top = 1.000 for all of L=3,4,5 (deep-ordered, all sizes correctable). q_top saturates at 1 here, so the ordered side shows convergence-to-1 rather than visible "larger-L-higher" separation (intrinsic to a saturating order parameter).
- **Common crossing at q_c ≈ 0.03**; all three pairs now cross in 100% of bootstraps (anchored ordered side):
  - L3–L5: q_c≈0.031, CI95 [0.020, 0.043], boot-frac 1.000  ← most asymptotic (biggest size gap)
  - L4–L5: q_c≈0.019, CI95 [0.015, 0.039], boot-frac 1.000
  - L3–L4: q_c≈0.010, CI95 [0.010, 0.043], boot-frac 1.000
  - (pairwise q_c spread 0.01–0.03 = finite-size drift; take L3–L5 ≈0.031 as the best estimate.)
- **Disordered fan-out fully resolved** above q_c (larger L → lower q_top, monotonic, significant): L3–L5 max sep **0.181 (5.8σ)** @ q=0.20; L3–L4 0.100 (3.7σ); L4–L5 0.104 (**3.2σ**) — L4 and L5 are distinguishable at 96 disorders.
- **Conclusion: single-shot measurement-error threshold q_c(p=0.12) ≈ 0.03.** For q>q_c the code is in the uncorrectable phase (bigger code → lower q_top).

## Remaining options (optional)
- More *central* crossing (q_c at higher q, both sides with visible spread): smaller p (≈0.06–0.08 → q_c≈0.08–0.10), at higher compute on the deep-ordered low-q side.
- Threshold line q_c(p): repeat at 2–3 p to trace the (p, q_c) boundary = the single-shot threshold curve.

## Files
- `production_crossing.png` — final two-sided q_top(q), L=3,4,5, q_c band.
- `production_summary.json` — per-q means±SEM + pairwise crossings (15 q-points).
- `analyze_production.py` (merges main + low-q per L), `collected/{nd?_L?, nd?_L?_lowq}/`.
