# 002 cluster-stage repeats summary

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,m=512,stride=4,num_start_chains=4,adaptive_pt_rounds=0`。

| run | rho | min swap | bottleneck | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip | cluster moves | cluster sector changes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| AD previous | 0.05 | 0.153061 | 10 | `[2, 0, 0, 0]` | 100.50 | 0 | 19 | 87 | 47 | None |
| run01 rho005 | 0.05 | 0.129513 | 11 | `[0, 0, 0, 0]` | 94.75 | 0 | 14 | 79 | 38 | 0 |
| run02 rho005 | 0.05 | 0.121795 | 10 | `[0, 0, 0, 0]` | 96.50 | 0 | 12 | 67 | 26 | 0 |
| run03 rho010 | 0.10 | 0.167321 | 10 | `[10, 4, 2, 2]` | 100.50 | 0 | 18 | 76 | 77 | 9 |

结论：

- 两条新的 `rho=0.05` repeat 均没有 cold logical-sector flip，也没有 cluster-stage sector change；AD 的 `[2,0,0,0]` 不是稳定可复现信号。
- `rho=0.10` 的 run03 出现 cold flips `[10,4,2,2]`，并且 cluster-stage 诊断记录到 9 次 logical-sector 改变。
- 四条链 strict hot-to-cold delivery 仍为 0，因此这不是热端 sector change 简单输运回 cold，而更像是 cluster 在中温区直接造成 sector 改变后影响 cold。
- 下一轮应围绕 `q_hot=0.35,rho≈0.10` 做独立 seed 重复，并小幅扫描 rho。
