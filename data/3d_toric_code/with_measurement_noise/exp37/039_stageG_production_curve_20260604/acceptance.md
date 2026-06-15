# Stage G production curve acceptance

Overall: PASS

## Gate Numbers

| Gate | Result | Status |
|---|---:|---|
| G1 | Stage D exact L=2 benchmark: max abs(dq_top)=0.00544999, max TV=0.00495657, CI misses=0 | PASS |
| G2 | PASS-only curve; point statuses=PASS:10/WARN:38/FAIL:0; broad crossing claimed=False | PASS |
| G3 | reconstructed q_top from w_g[8]: max abs diff=0; uncertainty includes disorder bootstrap + TI stderr | PASS |
| Red line | unresolved tail FAIL present=False | PASS |

## Curve Policy

The production curve CSV uses only Stage F point-level PASS rows. WARN rows are kept in `production_curve_context.csv` and in the PNG as marked context only; no crossing or trend conclusion depends on them.

## PASS Points

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

- `production_curve.npz`
- `production_curve.csv` (PASS-only final curve)
- `production_curve_context.csv` (WARN context retained for audit)
- `production_curve.png`
- `stageG_acceptance.json`
