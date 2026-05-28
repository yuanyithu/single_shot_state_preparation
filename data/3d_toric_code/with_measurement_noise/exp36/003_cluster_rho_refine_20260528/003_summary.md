# 003 cluster-rho refinement summary

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,m=512,stride=4,num_start_chains=4,adaptive_pt_rounds=0`。

| run | rho | min swap | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip | cluster nonzero | cluster-stage changes | cluster wall frac |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| run01 rho010 | 0.10 | 0.141549 | `[0, 0, 0, 0]` | 97.00 | 0 | 17 | 74 | 94 | 1 | 0.095 |
| run02 rho010 | 0.10 | 0.129513 | `[0, 0, 0, 0]` | 91.50 | 0 | 14 | 67 | 66 | 0 | 0.096 |
| run03 rho015 | 0.15 | 0.151361 | `[0, 0, 0, 0]` | 96.00 | 0 | 13 | 65 | 102 | 19 | 0.145 |

逐温度 cluster-stage sector change：

- run01 rho=0.10: `[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]`。
- run02 rho=0.10: 全 0。
- run03 rho=0.15: `[0,0,0,0,0,0,0,0,0,1,3,2,13,0,0,0,0]`，集中在 `k=9..12`。

结论：

- 两条新的 `rho=0.10` repeat 均没有 cold logical-sector flip；002 run03 的 `[10,4,2,2]` 不是稳定可复现效果。
- `rho=0.15` 产生 19 次 cluster-stage sector change，但 cold flips 仍为 `[0,0,0,0]`，说明中温 sector change 大多在冷却回 cold 前丢失。
- 单纯增大 cluster 预算不是稳健修复；下一步应改 PT/cooling 机制，增强中温 sector change 向 cold 的保留，而不是继续盲目提高 rho。
