# 005 cold-arrival persistence summary

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0`。

| run | swap | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_rho015_swap1_m1024_s4_seed422000 | 1 | 1024 | 4 | 0.142370 | `[0, 2, 0, 0]` | 180 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 4/4 |
| run02_rho015_swap2_m1024_s4_seed423000 | 2 | 1024 | 4 | 0.122715 | `[0, 0, 0, 0]` | 201 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run03_rho015_swap1_m2048_s8_seed424000 | 1 | 2048 | 8 | 0.121453 | `[0, 0, 0, 0]` | 344 | 131 | 74 | 20 | 26 | 2 | 68 | 20 | 26 | 28 | 180/6 |

逐温度 persistence 诊断：

- run01_rho015_swap1_m1024_s4_seed422000: changed by temp = `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run01_rho015_swap1_m1024_s4_seed422000: arrival by origin = `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run01_rho015_swap1_m1024_s4_seed422000: diagnostic survived by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`，missed = `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run01_rho015_swap1_m1024_s4_seed422000: departure survived/reverted/other by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` / `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` / `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run02_rho015_swap2_m1024_s4_seed423000: changed by temp = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run02_rho015_swap2_m1024_s4_seed423000: arrival by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run02_rho015_swap2_m1024_s4_seed423000: diagnostic survived by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`，missed = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run02_rho015_swap2_m1024_s4_seed423000: departure survived/reverted/other by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` / `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` / `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。
- run03_rho015_swap1_m2048_s8_seed424000: changed by temp = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 20, 2, 0, 104, 0, 0]`。
- run03_rho015_swap1_m2048_s8_seed424000: arrival by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 17, 1, 0, 52, 0, 0]`。
- run03_rho015_swap1_m2048_s8_seed424000: diagnostic survived by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]`，missed = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 15, 1, 0, 48, 0, 0]`。
- run03_rho015_swap1_m2048_s8_seed424000: departure survived/reverted/other by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 12, 0, 0]` / `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 1, 0, 16, 0, 0]` / `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 24, 0, 0]`。

结论：

- run01 只有 1 次 origin `k=6` cluster-stage sector change；它到达 cold 时已经 reverted，在 cold 停留 4 个 transport samples，但没有赶上 stride=4 的 sector diagnostic，离开 cold 时仍 reverted。
- run02 没有 cluster-stage sector change，因此无法检验 persistence。
- run03 给出主要信号：131 次 cluster-stage sector change，其中 74 次到达 cold；到达时 20 次 survived、26 次 reverted、28 次 other。
- run03 的 74 次 cold arrival 中，只有 2 次被下一次 sector diagnostic 看到为 survived，68 次 missed；离开 cold 时分布为 20 survived、26 reverted、28 other，dwell sample 总数 180、最大 6。
- 因此 004 里 observed 的 survived-at-arrival 不是纯诊断假象；问题是绝大多数 arrival 在 cold 内停留很短或错过诊断 cadence，即使到达时 survived，也没有形成稳定的 cold-slot logical-sector flip。
