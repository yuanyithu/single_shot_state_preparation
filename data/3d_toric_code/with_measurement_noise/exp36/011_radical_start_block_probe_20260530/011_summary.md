# 011 radical-start block probe summary

completed; same disorder seed_base=435000, compare sector/all_zero/random_high_weight at q=0.08

共同参数：`lattice_size=6,p=0.05,q=0.08,q_hot=0.35,num_temperatures=17,num_start_chains=4,cluster_budget_fraction_rho=0.15`。

## 目标指标

| run | q_top | chain q_top | spread | Rhat | ESS | gate | block q_top | block range | last-half-full | wall s | ordinary | swap | observable | cluster |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| run01_sector_m1024_seed435000 | 1.000000 | [1, 1, 1, 1] | 0.000000 | 1.000000 | 1024 | pass | [1, 1, 1, 1, 1, 1, 1, 1] | 0.000000 | 0.000000 | 49.52 | 34.94 | 1.022 | 2.008 | 5.393 |
| run02_allzero_m1024_seed435000 | 1.000000 | [1, 1, 1, 1] | 0.000000 | 1.000000 | 1024 | pass | [1, 1, 1, 1, 1, 1, 1, 1] | 0.000000 | 0.000000 | 49.94 | 35.19 | 1.06 | 2.004 | 5.519 |
| run03_randomhigh_m1024_seed435000 | 1.000000 | [1, 1, 1, 1] | 0.000000 | 1.000000 | 1024 | pass | [1, 1, 1, 1, 1, 1, 1, 1] | 0.000000 | 0.000000 | 130.8 | 83.7 | 3.592 | 3.532 | 13.5 |

## 解释指标

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_sector_m1024_seed435000 | 4 | 1 | 6 | 1024 | 1 | 0.113722 | [] | 157 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run02_allzero_m1024_seed435000 | 4 | 1 | 6 | 1024 | 1 | 0.116768 | [] | 164 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run03_randomhigh_m1024_seed435000 | 4 | 1 | 6 | 1024 | 1 | 0.112054 | [] | 153 | 27 | 20 | 6 | 10 | 6 | 0 | 6 | 10 | 4 | 176/16 |

结论：

- q=0.08 的三种初态最终 q_top、chain q_top 和 8 个 block q_top 全部为 1，按原 q_top 初态一致性判据未发现初态依赖。
- 该结果也显示 q=0.08 已处于 q_top 饱和区：q_top 对固定 logical sector 的符号不敏感，因此不能把全 1 单独解释为充分热化证明。
- 原 sector 初始化曾错误尝试对带 measurement noise 的 observed syndrome 求 section representative，导致 preflight 卡住；已修正为 q>0 sector 只使用 zero-syndrome sector representatives，本地 sector smoke 和远端重跑均通过。
- 下一步应在更不饱和的高 q 点重复 radical-start，或进入 common-disorder A/B 时同时关注 block drift 和 m_u/sector 解释指标。
