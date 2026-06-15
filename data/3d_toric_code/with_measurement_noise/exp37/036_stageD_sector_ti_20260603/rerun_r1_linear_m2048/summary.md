# Stage D sector-resolved TI summary

Overall: PASS

Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=65, burn=160, measurements=2048, stride=2, blocks=64, bootstrap=1000.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.001545 | PASS |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.00151, CI misses=0 | PASS |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.002104, max grid dq=0.0007157 | PASS |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.250000 | 0.310000 | 0.399815 | 0.401325 | [0.394494, 0.408547] | 0.00155 | 0.00210 | 0.00072 | D1/D2/D3 |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
