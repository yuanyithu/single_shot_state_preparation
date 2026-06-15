# 012 radical-start high-q probe summary

completed; same disorder seed_base=436000, compare sector/all_zero/random_high_weight at q=0.23

共同参数：`lattice_size=6,p=0.05,q=0.23,q_hot=0.35,num_temperatures=17,num_start_chains=4,cluster_budget_fraction_rho=0.15`。

## 目标指标

| run | q_top | chain q_top | spread | Rhat | ESS | gate | block q_top | block range | last-half-full | wall s | ordinary | swap | observable | cluster |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| run01_sector_q023_m1024_seed436000 | 0.991097 | [0.9911, 0.9911, 0.9955, 0.9867] | 0.008878 | 1.000048 | 1024 | pass | [0.9955, 0.9955, 0.9779, 0.9867, 1, 0.9911, 0.9911, 0.9911] | 0.022103 | 0.002220 | 69.92 | 34.95 | 2.717 | 8.885 | 5.486 |
| run02_allzero_q023_m1024_seed436000 | 0.992207 | [0.9911, 1, 0.9889, 0.9889] | 0.011126 | 1.000028 | 1024 | pass | [0.9911, 0.9955, 0.9955, 0.9867, 1, 0.9823, 0.9955, 0.9911] | 0.017744 | -0.000001 | 68.53 | 35.72 | 2.213 | 9.421 | 5.474 |
| run03_randomhigh_q023_m1024_seed436000 | 0.990540 | [0.9889, 0.9933, 0.9889, 0.9911] | 0.004445 | 1.000141 | 1024 | pass | [0.9911, 1, 0.9734, 0.9911, 1, 0.9955, 0.9911, 0.9822] | 0.026576 | 0.001666 | 176.2 | 84.77 | 6.953 | 18.74 | 13.57 |

## 解释指标

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_sector_q023_m1024_seed436000 | 4 | 1 | 6 | 1024 | 1 | 0.583914 | [] | 796 | 63 | 54 | 29 | 20 | 29 | 0 | 29 | 20 | 5 | 616/56 |
| run02_allzero_q023_m1024_seed436000 | 4 | 1 | 6 | 1024 | 1 | 0.588918 | [] | 774 | 46 | 43 | 23 | 20 | 23 | 0 | 23 | 20 | 0 | 488/40 |
| run03_randomhigh_q023_m1024_seed436000 | 4 | 1 | 6 | 1024 | 1 | 0.587250 | [] | 800 | 85 | 71 | 30 | 28 | 30 | 0 | 30 | 28 | 13 | 776/32 |

结论：

- q=0.23 三种初态最终 q_top 为 0.991097/0.992207/0.990540，最大差约 0.00167，未发现显著初态依赖。
- 8-block range 约 0.0177~0.0266，但 last-half-full 绝对值不超过 0.0023，没有明显系统性 drift。
- random_high_weight wall time 明显高于 sector/all_zero，不适合作为生产初态；后续生产-like 路径使用 sector 或 all_zero 即可。
- radical-start 通过后，下一步已进入 q=0.23 common-disorder A/B。
