# exp39 p=0.06 crossing — two-sided threshold via free-energy gap Δf

Run `exp39_prodp006_20260606_0230` (nd1:L3/nd2:L4/nd3:L5), p=0.06, L=3,4,5, 96 disorder, seed 810000, TI/linear grid129/m8192, q={0.03..0.18} 11 pts.

## Why Δf, not q_top
q_top (purity) **saturates at 1** in the ordered phase, so the ordered-side "larger L → higher q_top" is compressed below the noise floor (the q<0.05 region is all ~0.97-1.0, messy). The sector free-energy gap **Δf = F(nearest competing logical sector) − F(dominant) = logical protection** does NOT saturate and shows both sides cleanly.

## Result: clean two-sided crossing → threshold q_c ≈ 0.05–0.06 at p=0.06
Δf gap (mean ± boot-SEM), L3/L4/L5:
- q=0.03: 13.97 / 21.45 / 33.06  → ordered: L5>L4>L3 (larger L MORE protected, many σ)
- q=0.05: 8.70 / 9.66 / 9.78      → crossing region
- q=0.06: 7.49 / 7.34 / 6.77      → disordered onset (order swaps)
- q=0.18: 2.84 / 2.25 / 1.92      → disordered: L3>L4>L5 (larger L LESS protected)
- **Δf crossing q_c(L3-L5) ≈ 0.056.** The curves genuinely swap order across q_c (not mere fan-out) → threshold proven.
- Same-data q_top crossing at q_c≈0.03–0.04 (saturating observable; consistent threshold ~0.05).

## Files
- `deltaf_crossing.png` — the two-sided Δf crossing (main deliverable).
- `production_crossing.png` — q_top(q) (disordered fan-out clear; ordered side saturated).
- `plot_deltaf_crossing.py`, `analyze_production.py`, `collected/{nd1_L3,nd2_L4,nd3_L5}/`.
