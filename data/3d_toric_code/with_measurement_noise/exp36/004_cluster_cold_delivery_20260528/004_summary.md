# 004 cluster cold-delivery summary

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0`。

| run | swap | m | stride | min swap | cold flips | hot flips mean | roundtrip | cluster nonzero | changed | arrival | survived | reverted | other | remaining |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run01 swap1 m512 | 1 | 512 | 4 | 0.104526 | `[0, 0, 0, 0]` | 95.75 | 59 | 90 | 0 | 0 | 0 | 0 | 0 | 0 |
| run02 swap2 m512 | 2 | 512 | 4 | 0.120552 | `[0, 0, 0, 0]` | 93.50 | 82 | 98 | 0 | 0 | 0 | 0 | 0 | 0 |
| run03 swap1 m1024 | 1 | 1024 | 8 | 0.109806 | `[0, 0, 0, 0]` | 94.50 | 143 | 215 | 38 | 30 | 13 | 10 | 7 | 2 |

逐温度 cold-delivery 诊断：

- run01/run02: cluster-stage sector change 为 0，因此 arrival/survival 均为 0。
- run03: changed by temp = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 30, 2, 2, 2, 0]`。
- run03: arrival by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 24, 2, 1, 1, 0]`。
- run03: survived by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 1, 0]`。
- run03: reverted by origin = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 7, 1, 0, 0, 0]`。

结论：

- 三条 run 的 cold flips 都是 `[0,0,0,0]`。
- `swap_sweeps=2` 短链没有产生 cluster-stage sector change，因此不能改善 cold flips；它只提高了 roundtrip。
- `m=1024` 长链产生 38 次 cluster-stage sector change，其中 30 次在 run 内到达 cold，13 次到达 cold 时仍保持 cluster 后 signature，10 次已经回到 cluster 前 signature，7 次变成其他 signature，2 次截至 run 结束仍 pending。
- 这说明中温 sector change 可以被 PT 带到 cold，且有一部分在首次 cold 接触时并未立刻消失；但这些事件没有形成按当前 cold-slot 诊断可见的持久 cold logical-sector flip。下一步应追踪 cold arrival 后的驻留时间和离开 cold 前的 persistence，而不是只看首次到达分类。
