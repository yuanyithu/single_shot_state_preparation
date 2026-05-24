# 3D q>0 profiling summary

- created_at: `2026-05-23T15:34:55+08:00`
- suite: `calibration`
- q: `0.005`
- completed/skipped tasks: `2` / `0`
- git_commit_sha: `unknown`

## Config summaries

| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.2600 | `calibration_base_PT5_hot0p44_cluster_rho0p05` | 1 | 0.05134 | 27.17 | 1.031 | 0.07589 | 0.4375 | 37 | 56 | 0 | 1 | single_bit (0.7216) |
| 4 | 0.2600 | `calibration_no_PT` | 1 | 0.2132 | 63.99 | 1.168 | 0.1272 | 0.3125 | 25 | 0 | 0 | 0 | single_bit (0.7867) |

## Stage Wall-Time Rankings

### L4_p0p2600_calibration_base_PT5_hot0p44_cluster_rho0p05
- single_bit: 1.203s, fraction=0.7216
- contractible: 0.3422s, fraction=0.2052
- cluster: 0.08238s, fraction=0.04941
- observable: 0.02889s, fraction=0.01733
- winding: 0.008683s, fraction=0.005208
- pt_swap: 0.001982s, fraction=0.001189

### L4_p0p2600_calibration_no_PT
- single_bit: 1.254s, fraction=0.7867
- contractible: 0.3234s, fraction=0.2029
- cluster: 0.009085s, fraction=0.005701
- observable: 0.005677s, fraction=0.003562
- winding: 0.001711s, fraction=0.001074
- pt_swap: 0s, fraction=0

