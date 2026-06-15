# Stage G production curve summary

Overall: PASS

Stage G builds the final production curve from the accepted Stage F repaired
sector-TI grid.  The final `production_curve.csv` is strictly PASS-only; WARN
rows are retained only in `production_curve_context.csv` and as open markers in
`production_curve.png` for audit context.  No crossing or broad trend claim is
made from WARN points.

## Gate Numbers

| Gate | Result | Status |
|---|---:|---|
| G1 | Stage D exact L=2 benchmark max abs(dq_top)=0.00544999, max TV=0.00495657, CI misses=0 | PASS |
| G2 | PASS-only curve; point statuses=PASS:10/WARN:38/FAIL:0; broad crossing claimed=False | PASS |
| G3 | q_top reconstructed from saved w_g[8], max abs diff=0; error bars include disorder bootstrap + TI stderr | PASS |
| Red line | unresolved tail FAIL present=False | PASS |

## PASS Curve Points

| L | q | pass mean q_top | total SEM | pass disorders |
|---:|---:|---:|---:|---:|
| 3 | 0.150 | 0.604777 | 0.103444 | 4/4 |
| 3 | 0.160 | 0.564824 | 0.138171 | 4/4 |
| 3 | 0.180 | 0.560648 | 0.131932 | 4/4 |
| 3 | 0.190 | 0.564126 | 0.133504 | 4/4 |
| 3 | 0.200 | 0.573621 | 0.128756 | 4/4 |
| 3 | 0.210 | 0.458747 | 0.093424 | 4/4 |
| 4 | 0.170 | 0.509191 | 0.075216 | 4/4 |
| 4 | 0.210 | 0.471431 | 0.058408 | 4/4 |
| 5 | 0.190 | 0.458835 | 0.115386 | 4/4 |
| 5 | 0.210 | 0.420834 | 0.122179 | 4/4 |

## Artifacts

- `build_stageG_production_curve.py`
- `production_curve.npz`
- `production_curve.csv`
- `production_curve_context.csv`
- `production_curve.png`
- `stageG_acceptance.json`
- `acceptance.md`
