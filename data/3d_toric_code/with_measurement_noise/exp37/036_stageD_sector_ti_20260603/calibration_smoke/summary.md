# Stage D sector-resolved TI summary

Overall: FAIL

Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=5, burn=1, measurements=2, stride=1, blocks=2, bootstrap=10.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.4607 | FAIL |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.1681, CI misses=1 | FAIL |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.3232, max grid dq=0.0895 | FAIL |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.082049 | [0.067458, 0.136688] | 0.46072 | 0.32320 | 0.08950 | d1FAIL/d2FAIL/d3FAIL |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
