# exp38 P0 regression anchor summary

Overall: PASS

Estimator: sector-resolved thermodynamic integration via the exp37 Stage D runner (`run_stageD_sector_ti.py`) on three Stage B exact mid-q_top anchors.
Reference: Stage B exact L=2 zero-disorder benchmark. No AIS/FEP/flip-reweighting is used.
TI config: grid=65, burn=80, measurements=2048, stride=2, blocks=64, bootstrap=800.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.00402 | PASS |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.006496, CI misses=0 | PASS |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.004113, max grid dq=0.004373 | PASS |

## Point Comparison

| id | p | q | exact q_top | TI q_top | q_top 95% CI | TV | grid TV | grid dq | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.280000 | 0.305000 | 0.250121 | 0.249827 | [0.243868, 0.255574] | 0.00091 | 0.00411 | 0.00437 | D1/D2/D3 |
| 1 | 0.250000 | 0.310000 | 0.399815 | 0.397244 | [0.389505, 0.404665] | 0.00211 | 0.00103 | 0.00060 | D1/D2/D3 |
| 2 | 0.140000 | 0.385000 | 0.549674 | 0.543178 | [0.534677, 0.551556] | 0.00402 | 0.00163 | 0.00090 | D1/D2/D3 |

Artifacts:
- `stageD_results.json`
- `ti_results.npz`
- `ti_comparison.csv`
