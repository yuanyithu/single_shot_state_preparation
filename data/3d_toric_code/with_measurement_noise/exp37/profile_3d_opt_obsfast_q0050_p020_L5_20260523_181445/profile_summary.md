# 3D q>0 profiling summary

- created_at: `2026-05-23T18:15:43+08:00`
- suite: `optimization`
- q: `0.05`
- completed/skipped tasks: `8` / `0`
- git_commit_sha: `unknown`

## Config summaries

| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.2000 | `opt_noPT_no_cluster_single_0p10` | 4 | 0.7018 | 140 | 1.097 | 0.2234 | 0.334 | 308 | 0 | 0 | 0 | observable (0.4202) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p05_coldobs` | 4 | 0.6701 | 91.3 | 1.027 | 0.07765 | 0.1426 | 1993 | 1993 | 0 | 0 | contractible (0.3335) |

## Stage Wall-Time Rankings

### L5_p0p2000_opt_noPT_no_cluster_single_0p10
- observable: 6.808s, fraction=0.4202
- single_bit: 4.975s, fraction=0.307
- contractible: 3.509s, fraction=0.2165
- winding: 0.8599s, fraction=0.05307
- cluster: 0.05158s, fraction=0.003183
- pt_swap: 0s, fraction=0

### L5_p0p2000_opt_no_cluster_PT7_single_0p05_coldobs
- contractible: 13.57s, fraction=0.3335
- single_bit: 10.27s, fraction=0.2524
- observable: 9.555s, fraction=0.2349
- winding: 5.976s, fraction=0.1469
- pt_swap: 1.259s, fraction=0.03094
- cluster: 0.05298s, fraction=0.001302

