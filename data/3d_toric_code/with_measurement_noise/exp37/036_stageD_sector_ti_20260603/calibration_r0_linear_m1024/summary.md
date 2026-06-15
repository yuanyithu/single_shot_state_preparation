# Stage D sector-resolved TI summary

Overall: PASS

Estimator: sector-resolved thermodynamic integration using the Stage C fixed-sector decoder-reject chain.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=65, burn=120, measurements=1024, stride=2, blocks=32, bootstrap=600.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.004957 | PASS |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.00545, CI misses=0 | PASS |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.004336, max grid dq=0.003939 | PASS |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.255571 | [0.247710, 0.263285] | 0.00496 | 0.00434 | 0.00394 | D1/D2/D3 |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
