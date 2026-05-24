# 3D q>0 profiling summary

- created_at: `2026-05-23T18:04:49+08:00`
- suite: `optimization`
- q: `0.05`
- completed/skipped tasks: `32` / `0`
- git_commit_sha: `unknown`

## Config summaries

| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.2000 | `opt_noPT_no_cluster_single_0p10` | 4 | 0.5404 | 702.8 | 1.027 | 0.05417 | 0.1914 | 2327 | 0 | 0 | 0 | observable (0.3772) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p05_coldobs` | 4 | 0.5458 | 252.2 | 1.019 | 0.03919 | 0.1484 | 3682 | 3682 | 0 | 0 | contractible (0.3584) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p10_coldobs` | 4 | 0.5344 | 182.7 | 1.004 | 0.03663 | 0.09896 | 3799 | 3799 | 0 | 0 | single_bit (0.352) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p25_coldobs` | 4 | 0.5327 | 177.7 | 1.004 | 0.02698 | 0.07552 | 3772 | 3772 | 0 | 0 | single_bit (0.373) |
| 5 | 0.2000 | `opt_noPT_no_cluster_single_0p10` | 4 | 0.6855 | 75.52 | 1.309 | 0.3033 | 0.4141 | 426 | 0 | 0 | 0 | observable (0.6544) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p05_coldobs` | 4 | 0.7055 | 102.5 | 1.016 | 0.07234 | 0.1276 | 1871 | 1871 | 0 | 0 | contractible (0.3368) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p10_coldobs` | 4 | 0.6874 | 72.96 | 1.026 | 0.08806 | 0.1615 | 1970 | 1970 | 0 | 0 | observable (0.3184) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p25_coldobs` | 4 | 0.6485 | 44.38 | 1.05 | 0.1635 | 0.2142 | 2340 | 2340 | 0 | 0 | observable (0.5523) |

## Stage Wall-Time Rankings

### L4_p0p2000_opt_noPT_no_cluster_single_0p10
- observable: 1.877s, fraction=0.3772
- contractible: 1.262s, fraction=0.2536
- single_bit: 0.9561s, fraction=0.1922
- winding: 0.8299s, fraction=0.1668
- cluster: 0.05064s, fraction=0.01018
- pt_swap: 0s, fraction=0

### L4_p0p2000_opt_no_cluster_PT7_single_0p05_coldobs
- contractible: 8.498s, fraction=0.3584
- single_bit: 5.949s, fraction=0.2509
- winding: 5.704s, fraction=0.2406
- observable: 2.223s, fraction=0.09373
- pt_swap: 1.288s, fraction=0.05432
- cluster: 0.05135s, fraction=0.002165

### L4_p0p2000_opt_no_cluster_PT7_single_0p10_coldobs
- single_bit: 10.68s, fraction=0.352
- contractible: 10.27s, fraction=0.3385
- winding: 5.697s, fraction=0.1879
- observable: 2.356s, fraction=0.0777
- pt_swap: 1.281s, fraction=0.04223
- cluster: 0.05116s, fraction=0.001687

### L4_p0p2000_opt_no_cluster_PT7_single_0p25_coldobs
- single_bit: 12.43s, fraction=0.373
- contractible: 10.4s, fraction=0.3121
- winding: 5.801s, fraction=0.1741
- observable: 3.343s, fraction=0.1003
- pt_swap: 1.297s, fraction=0.03892
- cluster: 0.0514s, fraction=0.001542

### L5_p0p2000_opt_noPT_no_cluster_single_0p10
- observable: 18.24s, fraction=0.6544
- single_bit: 5.186s, fraction=0.186
- contractible: 3.535s, fraction=0.1268
- winding: 0.8611s, fraction=0.03089
- cluster: 0.05235s, fraction=0.001878
- pt_swap: 0s, fraction=0

### L5_p0p2000_opt_no_cluster_PT7_single_0p05_coldobs
- contractible: 13.65s, fraction=0.3368
- single_bit: 10.52s, fraction=0.2597
- observable: 9.044s, fraction=0.2232
- winding: 5.995s, fraction=0.1479
- pt_swap: 1.258s, fraction=0.03104
- cluster: 0.05351s, fraction=0.001321

### L5_p0p2000_opt_no_cluster_PT7_single_0p10_coldobs
- observable: 15.25s, fraction=0.3184
- contractible: 13.64s, fraction=0.2848
- single_bit: 11.7s, fraction=0.2444
- winding: 5.985s, fraction=0.125
- pt_swap: 1.257s, fraction=0.02625
- cluster: 0.05401s, fraction=0.001128

### L5_p0p2000_opt_no_cluster_PT7_single_0p25_coldobs
- observable: 45.01s, fraction=0.5523
- single_bit: 15.18s, fraction=0.1862
- contractible: 13.82s, fraction=0.1695
- winding: 6.154s, fraction=0.0755
- pt_swap: 1.293s, fraction=0.01586
- cluster: 0.05501s, fraction=0.0006749

