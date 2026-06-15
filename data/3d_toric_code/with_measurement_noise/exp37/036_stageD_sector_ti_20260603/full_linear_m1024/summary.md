# Stage D sector-resolved TI summary

Overall: FAIL

Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=65, burn=120, measurements=1024, stride=2, blocks=32, bootstrap=600.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.00785 | PASS |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.01074, CI misses=1 | FAIL |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.00951, max grid dq=0.01269 | PASS |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.255571 | [0.247710, 0.263285] | 0.00496 | 0.00434 | 0.00394 | D1/D2/D3 |
| 1 | 0.250000 | 0.310000 | 0.399815 | 0.389075 | [0.378168, 0.398734] | 0.00785 | 0.00951 | 0.01269 | D1/d2FAIL/D3 |
| 2 | 0.140000 | 0.385000 | 0.549674 | 0.548363 | [0.537199, 0.560387] | 0.00138 | 0.00264 | 0.00361 | D1/D2/D3 |
| 3 | 0.050000 | 0.450000 | 0.699772 | 0.702460 | [0.690509, 0.713727] | 0.00166 | 0.00139 | 0.00236 | D1/D2/D3 |
| 4 | 0.150000 | 0.300000 | 0.850252 | 0.855146 | [0.849812, 0.860951] | 0.00235 | 0.00056 | 0.00036 | D1/D2/D3 |
| 5 | 0.210000 | 0.160000 | 0.920286 | 0.921407 | [0.918169, 0.924180] | 0.00052 | 0.00056 | 0.00099 | D1/D2/D3 |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
