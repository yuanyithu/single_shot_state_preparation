# 3D q>0 profiling summary

- created_at: `2026-05-23T15:44:25+08:00`
- suite: `default`
- q: `0.005`
- completed/skipped tasks: `95` / `0`
- git_commit_sha: `unknown`

## Config summaries

| L | p | config | disorders | q_top | ESS/sec | R-hat max | q_top spread | m_u spread | cold flips | hot flips | hot->cold deliveries | cluster nonzero | top wall stage |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.2200 | `base_PT7_hot0p44_cluster_rho0p05` | 3 | 0.1913 | 11.5 | 1.002 | 0.03568 | 0.1259 | 5389 | 8046 | 214 | 0.3552 | single_bit (0.3024) |
| 4 | 0.2200 | `cluster_off` | 3 | 0.3264 | 12.5 | 1.002 | 0.05039 | 0.1102 | 4552 | 8070 | 194 | 0 | single_bit (0.3984) |
| 4 | 0.2200 | `no_PT` | 3 | 0.104 | 28.8 | 1.006 | 0.03494 | 0.1832 | 4460 | 0 | 0 | 0.005029 | cluster (0.4366) |
| 4 | 0.2200 | `winding_repeat_4` | 3 | 0.6529 | 7.874 | 1.002 | 0.04809 | 0.07726 | 2106 | 8064 | 130 | 0.4891 | winding (0.3192) |
| 4 | 0.2600 | `PT_K5_hot0p44_swap1` | 3 | 0.03555 | 20.52 | 1.003 | 0.01053 | 0.1215 | 7038 | 8027 | 231 | 0.2889 | single_bit (0.3084) |
| 4 | 0.2600 | `PT_K7_hot0p36_swap1` | 3 | 0.06658 | 16.42 | 1.001 | 0.01459 | 0.1372 | 7028 | 8042 | 545 | 0.1245 | single_bit (0.3031) |
| 4 | 0.2600 | `PT_K7_hot0p44_swap1` | 3 | 0.03774 | 16.74 | 1.001 | 0.01418 | 0.09115 | 7240 | 8031 | 312 | 0.3902 | single_bit (0.3043) |
| 4 | 0.2600 | `PT_K7_hot0p44_swap4` | 3 | 0.08809 | 16.92 | 1.003 | 0.01211 | 0.1354 | 6334 | 8082 | 100 | 0.6781 | single_bit (0.3101) |
| 4 | 0.2600 | `PT_K7_hot0p48_swap1` | 3 | 0.05582 | 18.02 | 1.004 | 0.0147 | 0.1649 | 7076 | 8021 | 226 | 0.07216 | single_bit (0.3073) |
| 4 | 0.2600 | `PT_K9_hot0p44_swap1` | 3 | 0.06541 | 12.81 | 1.003 | 0.02591 | 0.1337 | 7139 | 8058 | 380 | 0.1398 | single_bit (0.3151) |
| 4 | 0.2600 | `base_PT7_hot0p44_cluster_rho0p05` | 3 | 0.0481 | 15.33 | 1.003 | 0.01954 | 0.1406 | 7016 | 8016 | 317 | 0.0568 | single_bit (0.3006) |
| 4 | 0.2600 | `cluster_disabled_sensitivity` | 3 | 0.02934 | 17.5 | 1.003 | 0.01187 | 0.1215 | 7394 | 8042 | 338 | 0 | single_bit (0.4048) |
| 4 | 0.2600 | `cluster_off` | 3 | 0.06544 | 16.93 | 1.002 | 0.01677 | 0.1354 | 6857 | 8082 | 332 | 0 | single_bit (0.4079) |
| 4 | 0.2600 | `cluster_rho0p05` | 3 | 0.02944 | 18.8 | 1.002 | 0.006744 | 0.1155 | 7465 | 8063 | 306 | 0.2994 | cluster (0.2817) |
| 4 | 0.2600 | `cluster_rho0p15` | 3 | 0.07229 | 18.19 | 1.002 | 0.02247 | 0.1276 | 7007 | 8062 | 334 | 0.7295 | cluster (0.3071) |
| 4 | 0.2600 | `no_PT` | 3 | 0.07776 | 97.64 | 1.003 | 0.03563 | 0.1901 | 6336 | 0 | 0 | 0.008278 | cluster (0.4061) |
| 4 | 0.2600 | `starts4_reps1` | 3 | 0.09296 | 15.36 | 1.002 | 0.01673 | 0.151 | 6651 | 8042 | 297 | 0.1545 | cluster (0.2818) |
| 4 | 0.2600 | `starts4_reps2` | 3 | 0.02226 | 20.94 | 1.002 | 0.012 | 0.1762 | 14823 | 16180 | 704 | 0.09009 | single_bit (0.2933) |
| 4 | 0.2600 | `starts8_reps1` | 3 | 0.02771 | 19.83 | 1.002 | 0.008788 | 0.1528 | 14781 | 16065 | 654 | 0.1568 | cluster (0.2845) |
| 4 | 0.2600 | `winding_repeat_4` | 3 | 0.01612 | 8.414 | 1.003 | 0.01045 | 0.1337 | 6692 | 8015 | 332 | 0.5238 | winding (0.3222) |
| 4 | 0.2600 | `zero_sweeps_1_winding_repeat_4` | 3 | 0.1468 | 8.495 | 1.004 | 0.04244 | 0.1424 | 5694 | 8040 | 289 | 0.3046 | winding (0.3238) |
| 4 | 0.2600 | `zero_sweeps_4_winding_repeat_1` | 3 | 0.1224 | 10.7 | 1.002 | 0.01021 | 0.1319 | 7051 | 8051 | 326 | 0.05969 | contractible (0.326) |
| 4 | 0.2600 | `zero_sweeps_4_winding_repeat_4` | 3 | 0.05013 | 5.028 | 1.001 | 0.01468 | 0.1224 | 7422 | 8045 | 379 | 0.02575 | winding (0.535) |
| 4 | 0.3000 | `base_PT7_hot0p44_cluster_rho0p05` | 3 | 0.008793 | 20.61 | 1.002 | 0.004962 | 0.1337 | 7859 | 8069 | 456 | 0.03885 | single_bit (0.3232) |
| 4 | 0.3000 | `cluster_off` | 3 | 0.006122 | 20.14 | 1.001 | 0.00425 | 0.105 | 8022 | 8005 | 491 | 0 | single_bit (0.3976) |
| 4 | 0.3000 | `no_PT` | 3 | 0.003122 | 127.6 | 1.001 | 0.002994 | 0.09983 | 7867 | 0 | 0 | 0.02456 | cluster (0.4116) |
| 4 | 0.3000 | `winding_repeat_4` | 3 | 0.01021 | 12.27 | 1.001 | 0.005255 | 0.125 | 7637 | 8094 | 458 | 0.05202 | winding (0.3338) |
| 5 | 0.2600 | `L5_PT_K9` | 2 | 0.04231 | 3.906 | 1.003 | 0.02025 | 0.1823 | 3784 | 5426 | 148 | 0.1589 | single_bit (0.3545) |
| 5 | 0.2600 | `L5_cluster_off` | 2 | 0.05209 | 4.09 | 1.01 | 0.03736 | 0.2174 | 3539 | 5412 | 103 | 0 | single_bit (0.4157) |
| 5 | 0.2600 | `L5_cluster_rho0p15` | 2 | 0.03546 | 4.033 | 1.008 | 0.01484 | 0.1758 | 3456 | 5403 | 114 | 0.105 | single_bit (0.3181) |
| 5 | 0.2600 | `L5_hot0p48` | 2 | 0.01635 | 4.199 | 1.008 | 0.009868 | 0.194 | 3698 | 5366 | 98 | 0.129 | single_bit (0.3482) |
| 5 | 0.2600 | `L5_no_PT` | 2 | 0.08883 | 8.943 | 1.011 | 0.04246 | 0.2188 | 1992 | 0 | 0 | 0.004651 | single_bit (0.374) |
| 5 | 0.2600 | `L5_winding_repeat_4` | 2 | 0.06617 | 2.16 | 1.005 | 0.03717 | 0.138 | 3371 | 5360 | 113 | 0.2222 | single_bit (0.2941) |
| 5 | 0.2600 | `base_PT7_hot0p44_cluster_rho0p05` | 2 | 0.02535 | 4.746 | 1.003 | 0.005053 | 0.1406 | 3887 | 5373 | 130 | 0.625 | single_bit (0.3552) |

## Stage Wall-Time Rankings

### L4_p0p2200_base_PT7_hot0p44_cluster_rho0p05
- single_bit: 14.8s, fraction=0.3024
- cluster: 12.37s, fraction=0.2526
- contractible: 7.976s, fraction=0.1629
- observable: 7.292s, fraction=0.149
- winding: 5.455s, fraction=0.1114
- pt_swap: 1.06s, fraction=0.02166

### L4_p0p2200_cluster_off
- single_bit: 15.89s, fraction=0.3984
- contractible: 8.836s, fraction=0.2215
- observable: 7.718s, fraction=0.1935
- winding: 6.221s, fraction=0.156
- pt_swap: 1.132s, fraction=0.02838
- cluster: 0.08944s, fraction=0.002242

### L4_p0p2200_no_PT
- cluster: 8.53s, fraction=0.4366
- single_bit: 6.402s, fraction=0.3277
- contractible: 2.366s, fraction=0.1211
- observable: 1.235s, fraction=0.0632
- winding: 1.003s, fraction=0.05135
- pt_swap: 0s, fraction=0

### L4_p0p2200_winding_repeat_4
- winding: 21.3s, fraction=0.3192
- single_bit: 15.7s, fraction=0.2352
- cluster: 13.81s, fraction=0.2069
- contractible: 8.029s, fraction=0.1203
- observable: 6.802s, fraction=0.1019
- pt_swap: 1.1s, fraction=0.01648

### L4_p0p2600_PT_K5_hot0p44_swap1
- single_bit: 12.12s, fraction=0.3084
- cluster: 11.12s, fraction=0.283
- contractible: 6.022s, fraction=0.1532
- observable: 5.293s, fraction=0.1347
- winding: 3.934s, fraction=0.1001
- pt_swap: 0.8056s, fraction=0.0205

### L4_p0p2600_PT_K7_hot0p36_swap1
- single_bit: 14.81s, fraction=0.3031
- cluster: 11.8s, fraction=0.2414
- contractible: 8.146s, fraction=0.1667
- observable: 7.314s, fraction=0.1497
- winding: 5.717s, fraction=0.117
- pt_swap: 1.082s, fraction=0.02214

### L4_p0p2600_PT_K7_hot0p44_swap1
- single_bit: 16.57s, fraction=0.3043
- cluster: 13.07s, fraction=0.24
- contractible: 9.042s, fraction=0.166
- observable: 7.794s, fraction=0.1431
- winding: 6.812s, fraction=0.1251
- pt_swap: 1.172s, fraction=0.02152

### L4_p0p2600_PT_K7_hot0p44_swap4
- single_bit: 14.48s, fraction=0.3101
- cluster: 11.57s, fraction=0.2476
- contractible: 7.855s, fraction=0.1682
- observable: 7.206s, fraction=0.1543
- winding: 5.334s, fraction=0.1142
- pt_swap: 0.2614s, fraction=0.005598

### L4_p0p2600_PT_K7_hot0p48_swap1
- single_bit: 14.62s, fraction=0.3073
- cluster: 11.7s, fraction=0.246
- contractible: 7.786s, fraction=0.1636
- observable: 7.132s, fraction=0.1499
- winding: 5.314s, fraction=0.1117
- pt_swap: 1.027s, fraction=0.02158

### L4_p0p2600_PT_K9_hot0p44_swap1
- single_bit: 19.73s, fraction=0.3151
- cluster: 14.14s, fraction=0.2258
- contractible: 10.45s, fraction=0.1668
- observable: 9.224s, fraction=0.1473
- winding: 7.671s, fraction=0.1225
- pt_swap: 1.403s, fraction=0.02241

### L4_p0p2600_base_PT7_hot0p44_cluster_rho0p05
- single_bit: 15.78s, fraction=0.3006
- cluster: 13.35s, fraction=0.2543
- contractible: 8.351s, fraction=0.159
- observable: 7.641s, fraction=0.1455
- winding: 6.217s, fraction=0.1184
- pt_swap: 1.166s, fraction=0.0222

### L4_p0p2600_cluster_disabled_sensitivity
- single_bit: 15.22s, fraction=0.4048
- contractible: 8.084s, fraction=0.2149
- observable: 7.423s, fraction=0.1974
- winding: 5.694s, fraction=0.1514
- pt_swap: 1.103s, fraction=0.02934
- cluster: 0.08268s, fraction=0.002198

### L4_p0p2600_cluster_off
- single_bit: 15.11s, fraction=0.4079
- contractible: 8.002s, fraction=0.2161
- observable: 7.264s, fraction=0.1962
- winding: 5.495s, fraction=0.1484
- pt_swap: 1.081s, fraction=0.02919
- cluster: 0.08216s, fraction=0.002219

### L4_p0p2600_cluster_rho0p05
- cluster: 12.31s, fraction=0.2817
- single_bit: 11.53s, fraction=0.2638
- contractible: 6.881s, fraction=0.1574
- observable: 6.666s, fraction=0.1525
- winding: 5.233s, fraction=0.1197
- pt_swap: 1.086s, fraction=0.02485

### L4_p0p2600_cluster_rho0p15
- cluster: 13.6s, fraction=0.3071
- single_bit: 10.61s, fraction=0.2395
- observable: 6.991s, fraction=0.1579
- contractible: 6.757s, fraction=0.1526
- winding: 5.287s, fraction=0.1194
- pt_swap: 1.044s, fraction=0.02357

### L4_p0p2600_no_PT
- cluster: 7.034s, fraction=0.4061
- single_bit: 6.382s, fraction=0.3685
- contractible: 2.137s, fraction=0.1234
- observable: 1.006s, fraction=0.05807
- winding: 0.7612s, fraction=0.04395
- pt_swap: 0s, fraction=0

### L4_p0p2600_starts4_reps1
- cluster: 13.64s, fraction=0.2818
- single_bit: 11.82s, fraction=0.2442
- contractible: 7.883s, fraction=0.1628
- observable: 7.489s, fraction=0.1547
- winding: 6.412s, fraction=0.1324
- pt_swap: 1.167s, fraction=0.02411

### L4_p0p2600_starts4_reps2
- single_bit: 26.09s, fraction=0.2933
- cluster: 25.7s, fraction=0.289
- contractible: 13.68s, fraction=0.1539
- observable: 11.63s, fraction=0.1307
- winding: 9.682s, fraction=0.1089
- pt_swap: 2.162s, fraction=0.0243

### L4_p0p2600_starts8_reps1
- cluster: 25.52s, fraction=0.2845
- single_bit: 25.26s, fraction=0.2817
- contractible: 14.04s, fraction=0.1565
- observable: 12.38s, fraction=0.138
- winding: 10.32s, fraction=0.115
- pt_swap: 2.176s, fraction=0.02426

### L4_p0p2600_winding_repeat_4
- winding: 21.73s, fraction=0.3222
- single_bit: 16.24s, fraction=0.2407
- cluster: 13.11s, fraction=0.1943
- contractible: 8.272s, fraction=0.1226
- observable: 6.975s, fraction=0.1034
- pt_swap: 1.125s, fraction=0.01668

### L4_p0p2600_zero_sweeps_1_winding_repeat_4
- winding: 21.8s, fraction=0.3238
- single_bit: 16.55s, fraction=0.2458
- cluster: 13.2s, fraction=0.196
- contractible: 7.859s, fraction=0.1167
- observable: 6.794s, fraction=0.1009
- pt_swap: 1.126s, fraction=0.01673

### L4_p0p2600_zero_sweeps_4_winding_repeat_1
- contractible: 29.2s, fraction=0.326
- winding: 21.6s, fraction=0.2411
- single_bit: 17.28s, fraction=0.1929
- cluster: 13.86s, fraction=0.1547
- observable: 6.507s, fraction=0.07265
- pt_swap: 1.121s, fraction=0.01252

### L4_p0p2600_zero_sweeps_4_winding_repeat_4
- winding: 79.61s, fraction=0.535
- contractible: 28.65s, fraction=0.1925
- single_bit: 17.4s, fraction=0.1169
- cluster: 15.98s, fraction=0.1074
- observable: 6.075s, fraction=0.04082
- pt_swap: 1.092s, fraction=0.007337

### L4_p0p3000_base_PT7_hot0p44_cluster_rho0p05
- single_bit: 15.36s, fraction=0.3232
- cluster: 11.35s, fraction=0.2389
- contractible: 7.506s, fraction=0.1579
- observable: 6.975s, fraction=0.1468
- winding: 5.282s, fraction=0.1111
- pt_swap: 1.045s, fraction=0.02198

### L4_p0p3000_cluster_off
- single_bit: 15.73s, fraction=0.3976
- contractible: 8.785s, fraction=0.222
- observable: 7.598s, fraction=0.192
- winding: 6.23s, fraction=0.1574
- pt_swap: 1.135s, fraction=0.02868
- cluster: 0.09151s, fraction=0.002312

### L4_p0p3000_no_PT
- cluster: 7.329s, fraction=0.4116
- single_bit: 6.277s, fraction=0.3525
- contractible: 2.232s, fraction=0.1254
- observable: 1.134s, fraction=0.06371
- winding: 0.8325s, fraction=0.04676
- pt_swap: 0s, fraction=0

### L4_p0p3000_winding_repeat_4
- winding: 22.6s, fraction=0.3338
- single_bit: 16.26s, fraction=0.2401
- cluster: 12.48s, fraction=0.1843
- contractible: 8.393s, fraction=0.124
- observable: 6.88s, fraction=0.1016
- pt_swap: 1.101s, fraction=0.01625

### L5_p0p2600_L5_PT_K9
- single_bit: 20.5s, fraction=0.3545
- observable: 11.93s, fraction=0.2064
- cluster: 9.991s, fraction=0.1728
- contractible: 9.256s, fraction=0.1601
- winding: 5.199s, fraction=0.08991
- pt_swap: 0.9459s, fraction=0.01636

### L5_p0p2600_L5_cluster_off
- single_bit: 15.19s, fraction=0.4157
- observable: 9.987s, fraction=0.2733
- contractible: 6.618s, fraction=0.1811
- winding: 3.963s, fraction=0.1085
- pt_swap: 0.7314s, fraction=0.02001
- cluster: 0.05399s, fraction=0.001477

### L5_p0p2600_L5_cluster_rho0p15
- single_bit: 16.29s, fraction=0.3181
- cluster: 12.93s, fraction=0.2525
- observable: 9.62s, fraction=0.1878
- contractible: 7.46s, fraction=0.1457
- winding: 4.158s, fraction=0.0812
- pt_swap: 0.7521s, fraction=0.01469

### L5_p0p2600_L5_hot0p48
- single_bit: 16.29s, fraction=0.3482
- observable: 9.342s, fraction=0.1997
- cluster: 8.92s, fraction=0.1906
- contractible: 7.427s, fraction=0.1587
- winding: 4.065s, fraction=0.08688
- pt_swap: 0.7394s, fraction=0.0158

### L5_p0p2600_L5_no_PT
- single_bit: 4.911s, fraction=0.374
- cluster: 3.906s, fraction=0.2974
- contractible: 1.796s, fraction=0.1367
- observable: 1.794s, fraction=0.1366
- winding: 0.7265s, fraction=0.05532
- pt_swap: 0s, fraction=0

### L5_p0p2600_L5_winding_repeat_4
- single_bit: 16.48s, fraction=0.2941
- winding: 14.38s, fraction=0.2566
- cluster: 8.87s, fraction=0.1583
- observable: 8.573s, fraction=0.153
- contractible: 7.02s, fraction=0.1253
- pt_swap: 0.7228s, fraction=0.0129

### L5_p0p2600_base_PT7_hot0p44_cluster_rho0p05
- single_bit: 16.71s, fraction=0.3552
- observable: 9.45s, fraction=0.2008
- cluster: 8.922s, fraction=0.1896
- contractible: 7.07s, fraction=0.1503
- winding: 4.13s, fraction=0.08778
- pt_swap: 0.7659s, fraction=0.01628

