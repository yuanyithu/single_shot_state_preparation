# Stage E bidirectional logical-loop bridge summary

Overall: PASS

Estimator: independent multi-step logical-loop bridge.  For each sector g, sector-0 configurations are annealed to `y xor ell_g` on a lambda grid and adjacent intervals are combined with BAR.
Reference: Stage B exact L=2 zero-disorder benchmark and Stage D accepted TI.
No single-step FEP, flip-reweighting, or Kp thermodynamic integration is used.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| E1 | TV vs exact <= 0.030, abs dq_top <= 0.020 | max TV=1.261e-14, max abs dq=1.61e-14 | PASS |
| E2 | TV vs TI <= 0.030, abs dq_top <= 0.020 | max TV=0.004957, max abs dq=0.00545 | PASS |
| E3 | bidirectional gap <= 1.0e-08, BAR residual <= 1.0e-10 | max gap=8.793e-14, max residual=9.936e-15 | PASS |

## Point Comparison

| id | p | q | exact q_top | bridge q_top | TI q_top | TV exact | dq exact | TV TI | dq TI | bidir gap | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.250121 | 0.255571 | 1.261e-14 | 6.8279e-15 | 0.00496 | 0.00545 | 5.46e-14 | E1/E2/E3 |
| 1 | 0.250000 | 0.310000 | 0.399815 | 0.399815 | 0.401325 | 9.2747e-15 | 3.0531e-15 | 0.00155 | 0.00151 | 8.79e-14 | E1/E2/E3 |
| 2 | 0.140000 | 0.385000 | 0.549674 | 0.549674 | 0.548363 | 1.2211e-14 | 1.6098e-14 | 0.00138 | 0.00131 | 7.55e-14 | E1/E2/E3 |
| 3 | 0.050000 | 0.450000 | 0.699772 | 0.699772 | 0.702460 | 2.7365e-15 | 5.218e-15 | 0.00166 | 0.00269 | 2.4e-14 | E1/E2/E3 |
| 4 | 0.150000 | 0.300000 | 0.850252 | 0.850252 | 0.855146 | 1.6497e-15 | 3.4417e-15 | 0.00235 | 0.00489 | 6.66e-14 | E1/E2/E3 |
| 5 | 0.210000 | 0.160000 | 0.920286 | 0.920286 | 0.921407 | 5.0529e-16 | 1.5543e-15 | 0.00052 | 0.00112 | 4.97e-14 | E1/E2/E3 |

Artifacts:
- `stageE_results.json`
- `stageE_results.npz`
- `stageE_comparison.csv`
