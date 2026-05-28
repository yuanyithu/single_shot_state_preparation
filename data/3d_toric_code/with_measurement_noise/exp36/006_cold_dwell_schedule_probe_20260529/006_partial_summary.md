# 006 cold-dwell schedule partial summary

run03 `run03_measure1_m2048_seed427000` 仍在运行，以下仅为 run01/run02 中间结果。

| run | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_stride1_m1024_seed425000 | 1 | 6 | 1024 | 1 | 0.117131 | `[0, 0, 0, 0]` | 151 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run02_swap6_m1024_seed426000 | 6 | 6 | 1024 | 1 | 0.121739 | `[0, 0, 0, 0]` | 30 | 4 | 2 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 4/2 |

中间结论：

- run01 把 sector diagnostic stride 降到 1，但没有 cluster-stage sector change，也没有 cold flips。
- run02 把 swap cadence 降到 6，roundtrip 从百级降到 30；2 个 arrival 都被 diagnostic 捕获，missed=0，但 cold flips 仍为 0。
- run02 的 dwell max 只有 2，说明简单降低 swap cadence 没有显著拉长 cold 停留；它也明显损伤 PT transport。
