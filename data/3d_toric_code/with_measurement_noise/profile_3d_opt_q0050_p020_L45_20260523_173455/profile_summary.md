# 3D q>0 profiling summary

- created_at: `2026-05-23T17:51:31+08:00`
- suite: `optimization`
- q: `0.05`
- completed/skipped tasks: `24` / `0`
- git_commit_sha: `unknown`

## Config summaries

| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.2000 | `opt_noPT_no_cluster_full_single` | 2 | 0.1122 | 21.29 | 1.031 | 0.06172 | 0.3542 | 1220 | 0 | 0 | 0 | single_bit (0.6511) |
| 4 | 0.2000 | `opt_noPT_no_cluster_single_0p10` | 2 | 0.1229 | 22.67 | 1.063 | 0.07999 | 0.3151 | 1061 | 0 | 0 | 0 | single_bit (0.6059) |
| 4 | 0.2000 | `opt_no_cluster_PT7_full_single_coldobs` | 2 | 0.1219 | 9.659 | 1.005 | 0.04224 | 0.1589 | 1760 | 1760 | 49 | 0 | single_bit (0.561) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p05_coldobs` | 2 | 0.1085 | 9.787 | 1.016 | 0.04224 | 0.2604 | 1737 | 1737 | 74 | 0 | single_bit (0.414) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p10_coldobs` | 2 | 0.1078 | 10.89 | 1.011 | 0.06339 | 0.2552 | 1795 | 1795 | 60 | 0 | single_bit (0.4223) |
| 4 | 0.2000 | `opt_no_cluster_PT7_single_0p25_coldobs` | 2 | 0.1095 | 11.44 | 1.008 | 0.03237 | 0.2266 | 1821 | 1821 | 53 | 0 | single_bit (0.4513) |
| 5 | 0.2000 | `opt_noPT_no_cluster_full_single` | 2 | 0.5203 | 2.084 | 1.745 | 0.2562 | 1.286 | 262 | 0 | 0 | 0 | single_bit (0.6196) |
| 5 | 0.2000 | `opt_noPT_no_cluster_single_0p10` | 2 | 0.6224 | 8.56 | 1.176 | 0.4423 | 0.7161 | 107 | 0 | 0 | 0 | single_bit (0.5395) |
| 5 | 0.2000 | `opt_no_cluster_PT7_full_single_coldobs` | 2 | 0.4719 | 0.4097 | 1.029 | 0.1154 | 0.3359 | 845 | 845 | 15 | 0 | single_bit (0.5937) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p05_coldobs` | 2 | 0.5468 | 1.153 | 1.028 | 0.1739 | 0.2656 | 637 | 637 | 18 | 0 | contractible (0.3737) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p10_coldobs` | 2 | 0.5248 | 1.343 | 1.057 | 0.2009 | 0.3542 | 748 | 748 | 19 | 0 | single_bit (0.3943) |
| 5 | 0.2000 | `opt_no_cluster_PT7_single_0p25_coldobs` | 2 | 0.498 | 0.7244 | 1.068 | 0.09173 | 0.3099 | 746 | 746 | 11 | 0 | single_bit (0.4396) |

## Stage Wall-Time Rankings

### L4_p0p2000_opt_noPT_no_cluster_full_single
- single_bit: 2.804s, fraction=0.6511
- contractible: 0.9891s, fraction=0.2297
- observable: 0.2709s, fraction=0.06292
- winding: 0.2276s, fraction=0.05286
- cluster: 0.01475s, fraction=0.003426
- pt_swap: 0s, fraction=0

### L4_p0p2000_opt_noPT_no_cluster_single_0p10
- single_bit: 2.34s, fraction=0.6059
- contractible: 1.011s, fraction=0.2618
- observable: 0.2695s, fraction=0.06979
- winding: 0.2268s, fraction=0.05872
- cluster: 0.01444s, fraction=0.00374
- pt_swap: 0s, fraction=0

### L4_p0p2000_opt_no_cluster_PT7_full_single_coldobs
- single_bit: 6.763s, fraction=0.561
- contractible: 3s, fraction=0.2488
- winding: 1.623s, fraction=0.1346
- pt_swap: 0.3639s, fraction=0.03018
- observable: 0.2882s, fraction=0.0239
- cluster: 0.01738s, fraction=0.001442

### L4_p0p2000_opt_no_cluster_PT7_single_0p05_coldobs
- single_bit: 3.659s, fraction=0.414
- contractible: 2.963s, fraction=0.3352
- winding: 1.571s, fraction=0.1778
- pt_swap: 0.3482s, fraction=0.03939
- observable: 0.2804s, fraction=0.03172
- cluster: 0.01652s, fraction=0.001869

### L4_p0p2000_opt_no_cluster_PT7_single_0p10_coldobs
- single_bit: 3.88s, fraction=0.4223
- contractible: 3.018s, fraction=0.3285
- winding: 1.627s, fraction=0.1771
- pt_swap: 0.3599s, fraction=0.03917
- observable: 0.2861s, fraction=0.03114
- cluster: 0.01715s, fraction=0.001866

### L4_p0p2000_opt_no_cluster_PT7_single_0p25_coldobs
- single_bit: 4.271s, fraction=0.4513
- contractible: 2.966s, fraction=0.3134
- winding: 1.575s, fraction=0.1664
- pt_swap: 0.3521s, fraction=0.0372
- observable: 0.2821s, fraction=0.02981
- cluster: 0.01721s, fraction=0.001818

### L5_p0p2000_opt_noPT_no_cluster_full_single
- single_bit: 3.242s, fraction=0.6196
- contractible: 1.172s, fraction=0.2241
- observable: 0.5444s, fraction=0.104
- winding: 0.2544s, fraction=0.04862
- cluster: 0.01883s, fraction=0.003599
- pt_swap: 0s, fraction=0

### L5_p0p2000_opt_noPT_no_cluster_single_0p10
- single_bit: 2.262s, fraction=0.5395
- contractible: 1.127s, fraction=0.2686
- observable: 0.5408s, fraction=0.129
- winding: 0.248s, fraction=0.05914
- cluster: 0.01577s, fraction=0.00376
- pt_swap: 0s, fraction=0

### L5_p0p2000_opt_no_cluster_PT7_full_single_coldobs
- single_bit: 9.942s, fraction=0.5937
- contractible: 4.047s, fraction=0.2416
- winding: 1.784s, fraction=0.1065
- observable: 0.5641s, fraction=0.03368
- pt_swap: 0.3863s, fraction=0.02306
- cluster: 0.02409s, fraction=0.001438

### L5_p0p2000_opt_no_cluster_PT7_single_0p05_coldobs
- contractible: 3.907s, fraction=0.3737
- single_bit: 3.906s, fraction=0.3736
- winding: 1.707s, fraction=0.1632
- observable: 0.5535s, fraction=0.05294
- pt_swap: 0.3623s, fraction=0.03466
- cluster: 0.0203s, fraction=0.001942

### L5_p0p2000_opt_no_cluster_PT7_single_0p10_coldobs
- single_bit: 4.277s, fraction=0.3943
- contractible: 3.918s, fraction=0.3612
- winding: 1.708s, fraction=0.1575
- observable: 0.5546s, fraction=0.05113
- pt_swap: 0.3682s, fraction=0.03394
- cluster: 0.02176s, fraction=0.002006

### L5_p0p2000_opt_no_cluster_PT7_single_0p25_coldobs
- single_bit: 5.297s, fraction=0.4396
- contractible: 4.012s, fraction=0.333
- winding: 1.773s, fraction=0.1471
- observable: 0.5607s, fraction=0.04653
- pt_swap: 0.3835s, fraction=0.03183
- cluster: 0.02349s, fraction=0.001949

