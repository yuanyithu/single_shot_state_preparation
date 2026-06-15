# Stage B exact reference summary

Overall: PASS

Reference: L=2 3D toric code, zero eta, zero measurement error, corrected decoder-section sector label.
Primary table is produced by an independent full-enumeration count table; B2 compares against the production exact helper.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| B1 | >=3 exact q_top values in [0.2,0.8] | 4 / 6 | PASS |
| B2 | independent implementation TV < 1e-9 | max TV=2.090e-16, max dq=3.331e-16 | PASS |

## Reference Points

| id | p | q | q_top | TV(count-table, production) |
|---:|---:|---:|---:|---:|
| 0 | 0.280000 | 0.305000 | 0.250121 | 1.700e-16 |
| 1 | 0.250000 | 0.310000 | 0.399815 | 2.090e-16 |
| 2 | 0.140000 | 0.385000 | 0.549674 | 3.209e-17 |
| 3 | 0.050000 | 0.450000 | 0.699772 | 2.233e-17 |
| 4 | 0.150000 | 0.300000 | 0.850252 | 7.671e-18 |
| 5 | 0.210000 | 0.160000 | 0.920286 | 3.467e-17 |

Artifacts:
- `exact_reference.json`
- `exact_reference.csv`
- `candidate_scan.csv`
