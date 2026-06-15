# exp38 P5 production curve acceptance

Overall: `PASS`

## Gate Numbers

| Gate | Result | Status |
|---|---:|---|
| G1 | P0 exact benchmark replay: max TV=0.004020, max |dq_top|=0.006496, CI misses=0 | PASS |
| G2 | paired CI evidence only: significant crossing-region rows=5; common three-size crossing resolved=False | PASS |
| G3 | q_top reconstructed from w_g[8]: max abs diff=0; uncertainty includes disorder bootstrap + TI stderr | PASS |
| Red line | unresolved tail FAIL present=False | PASS |

## Crossing Conclusion

High-q finite-size separation is statistically resolved for L3-L5 and L3-L4 at the listed q values, but a common three-size crossing is not statistically resolved because L4-L5 has no crossing-region paired CI excluding zero.

The conclusion uses only `paired_difference.csv` rows where the paired bootstrap CI excludes zero. WARN context points and independent mean overlaps are not used to claim crossing.

## Point Statuses

Point statuses: PASS:20, WARN:19, FAIL:0. `production_curve.csv` contains only PASS rows; WARN rows are retained in `production_curve_context.csv` and plotted hollow in Figure A.

## PASS Curve Rows

| L | q | mean q_top | total SEM | 95% CI | pass disorders |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.120 | 0.888713 | 0.026068 | [0.834272, 0.936488] | 32/32 |
| 3 | 0.140 | 0.851368 | 0.028078 | [0.793407, 0.902978] | 32/32 |
| 3 | 0.150 | 0.830544 | 0.028389 | [0.771066, 0.882913] | 32/32 |
| 3 | 0.160 | 0.812763 | 0.030921 | [0.749146, 0.870395] | 32/32 |
| 3 | 0.170 | 0.783595 | 0.033473 | [0.714771, 0.846444] | 32/32 |
| 3 | 0.180 | 0.766301 | 0.032447 | [0.701472, 0.827800] | 32/32 |
| 3 | 0.190 | 0.745362 | 0.032167 | [0.680774, 0.806846] | 32/32 |
| 3 | 0.200 | 0.735876 | 0.031323 | [0.673181, 0.795990] | 32/32 |
| 3 | 0.210 | 0.711858 | 0.033249 | [0.646142, 0.775938] | 32/32 |
| 3 | 0.220 | 0.704577 | 0.032579 | [0.638931, 0.766534] | 32/32 |
| 3 | 0.230 | 0.651148 | 0.035820 | [0.579736, 0.720095] | 32/32 |
| 4 | 0.140 | 0.812519 | 0.036675 | [0.735791, 0.879084] | 32/32 |
| 4 | 0.150 | 0.770167 | 0.040357 | [0.686727, 0.843967] | 32/32 |
| 4 | 0.160 | 0.746449 | 0.041560 | [0.662364, 0.825017] | 32/32 |
| 4 | 0.180 | 0.700943 | 0.042118 | [0.614213, 0.778730] | 32/32 |
| 4 | 0.190 | 0.687871 | 0.039770 | [0.607438, 0.762866] | 32/32 |
| 4 | 0.210 | 0.625746 | 0.038622 | [0.547581, 0.699082] | 32/32 |
| 4 | 0.220 | 0.584905 | 0.038298 | [0.507681, 0.658802] | 32/32 |
| 4 | 0.230 | 0.572665 | 0.037999 | [0.497048, 0.646210] | 32/32 |
| 5 | 0.220 | 0.553335 | 0.034967 | [0.483857, 0.621211] | 32/32 |

## Artifacts

- `production_curve.npz`
- `production_curve.csv`
- `production_curve_context.csv`
- `paired_difference.csv`
- `production_curve.png`
- `paired_difference.png`
- `p5_acceptance.json`
