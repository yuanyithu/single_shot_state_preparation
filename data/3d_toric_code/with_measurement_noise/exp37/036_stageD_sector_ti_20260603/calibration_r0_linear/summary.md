# Stage D sector-resolved TI summary

Overall: FAIL

Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=33, burn=80, measurements=512, stride=2, blocks=16, bootstrap=400.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.02465 | FAIL |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.02696, CI misses=1 | FAIL |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.01318, max grid dq=0.01512 | PASS |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.277078 | [0.260707, 0.294369] | 0.02465 | 0.01318 | 0.01512 | d1FAIL/d2FAIL/D3 |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
