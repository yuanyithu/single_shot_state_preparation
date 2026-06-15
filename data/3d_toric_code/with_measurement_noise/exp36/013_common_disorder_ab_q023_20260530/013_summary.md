# 013 common-disorder A/B q=0.23 summary

completed; 3 common disorder samples, reference is candidate config with m=2048

## Runs

| run | config | q_top mean | q_top by disorder | drift max | block range max | half max | spread max | Rhat max | ESS min | wall s | ordinary | swap | observable | cluster |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run01_candidate_qhot035_rho015_coldedge4_m1024_seed437000 | qhot=0.35,rho=0.15,edge=4,m=1024,cluster=on | 0.994247 | [0.998327, 0.989429, 0.994985] | 0.00890241 | 0.0177264 | 0.00167315 | 0.0110975 | 1.00047 | 1024 | 164.252 | 105.027 | 3.11616 | 22.8229 | 16.475 |
| run02_cheap_qhot032_nocluster_m1024_seed437000 | qhot=0.32,rho=0.00,edge=1,m=1024,cluster=off | 0.99332 | [0.998327, 0.991095, 0.990539] | 0.0133057 | 0.0177699 | 0.00277846 | 0.00666373 | 1.00039 | 1024 | 138.616 | 106.603 | 3.7787 | 23.1633 | 0 |
| run03_reference_qhot035_rho015_coldedge4_m2048_seed437000 | qhot=0.35,rho=0.15,edge=4,m=2048,cluster=on | 0.994063 | [0.999442, 0.988317, 0.994429] | 0.0110888 | 0.0133384 | 0.00111164 | 0.00998306 | 1.00031 | 2048 | 741.311 | 477.643 | 20.2951 | 81.747 | 75.2859 |

## Reference Deltas

| run | reference | delta q_top by disorder | mean abs delta | max abs delta | wall/reference |
|---|---|---|---:|---:|---:|
| run01_candidate_qhot035_rho015_coldedge4_m1024_seed437000 | run03_reference_qhot035_rho015_coldedge4_m2048_seed437000 | [-0.001115, 0.001112, 0.000556] | 0.000927667 | 0.001115 | 0.221569 |
| run02_cheap_qhot032_nocluster_m1024_seed437000 | run03_reference_qhot035_rho015_coldedge4_m2048_seed437000 | [-0.001115, 0.002778, -0.00389] | 0.00259433 | 0.00389 | 0.186988 |
| run03_reference_qhot035_rho015_coldedge4_m2048_seed437000 | run03_reference_qhot035_rho015_coldedge4_m2048_seed437000 | [0, 0, 0] | 0 | 0 | 1 |

## Conclusion

- 候选 m=1024 与 2x reference 的逐 disorder q_top 差异最大 0.001115，mean abs 0.000928；cheap baseline 的最大差异 0.00389，mean abs 0.00259，绝对偏差仍很小。
- cheap baseline 关闭 cluster、q_hot=0.32、cold_edge_stride=1，wall time 为 138.6s；候选为 164.3s，cheap 约快 16%。2x reference 为 741.3s，但 q_top 没有给出明显更稳定的新信息。
- 三种配置的 Rhat/ESS/spread 均通过；block drift/range 没有系统性恶化。当前证据更支持把 cheap baseline 作为下一轮 production-like 小矩阵候选，而不是继续为 cluster/cold-edge hold 付费。
