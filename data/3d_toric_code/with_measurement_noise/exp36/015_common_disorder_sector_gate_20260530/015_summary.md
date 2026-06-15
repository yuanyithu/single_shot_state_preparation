# 015 common-disorder cold-sector gate summary

## 结论

本轮在 `L=6,p=0.05,q=0.23` 上使用 3 个 common disorder 复核 014 的单 disorder 结论。三种初态 `sector / all_zero / random_high_weight` 对每个 disorder 都给出几乎相同的 cold-sector 驻留分布。

这说明 014 不是单个 disorder 的偶然现象。当前配置在 high-q 端通过了“不同初态是否收敛到同一个热态 sector 分布”的 gate。

## 实验参数

- `L=6`
- `p=0.05`
- `q=0.23`
- disorder seeds: `515019, 7515022, 14515025`
- `num_disorder_samples_total=3`
- `num_measurements_per_disorder=2048`
- `num_start_chains=4`
- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`

## Sector-Space Gate

逐 disorder 三初态 cold-sector TV：

- disorder 0: max TV `0.0009`
- disorder 1: max TV `0.0007`
- disorder 2: max TV `0.0006`

这些数值远小于计划中的工程 gate `0.05`。

每条链前半/后半 cold-sector TV：

- `sector`: mean `0.0018`, max `0.0039`
- `all_zero`: mean `0.0015`, max `0.0039`
- `random_high_weight`: mean `0.0013`, max `0.0049`

所有 run 的 top sector 都是 `+++++++`，概率约 `0.9972` 到 `0.9996`，低概率 sector 的集合也一致。

## q_top 一致性

逐 disorder 的三初态 `q_top`：

- disorder 0: `0.993872, 0.993594, 0.994150`，spread `0.000556`
- disorder 1: `0.997769, 0.999163, 0.999163`，spread `0.001394`
- disorder 2: `0.998884, 0.998048, 0.999163`，spread `0.001115`

三初态平均后的 disorder 统计：

- per-disorder q_top: `0.993872, 0.998699, 0.998698`
- mean: `0.997090`
- sample std: `0.002787`
- SEM for 3 disorder: `0.001609`

这 3 个 disorder 只用于正确性 gate，不作为最终 disorder average 精度估计。但它说明 high-q 端的初态依赖误差远小于 disorder-to-disorder fluctuation。

## 物理解释

本轮检查的不是“`q_top` 是否碰巧相近”，而是每个时间片的 Wilson-loop sector 驻留分布是否相同。三初态在每个 disorder 上都驻留在同一批 sector，且概率几乎完全一致。因此没有看到“不同初态困在不同 sector，但 `q_top` 平方平均掩盖问题”的现象。

## 代价

- `sector`: 3 个 disorder chunk 总计约 353 秒。
- `all_zero`: 总计约 368 秒。
- `random_high_weight`: 总计约 882 秒。

`random_high_weight` 明显更慢，但作为极端初态诊断是有价值的。生产扫描不需要保留三初态；这是 gate 诊断，不是 production 配置。

## 下一步

按 014/015 的计划，下一步做 `q=0.08` 低 q 饱和点 sanity。若低 q 也从极端初态收敛到同一个 sector 分布，则进入小矩阵 production-like 试跑。

完整自动报告见：

`remote_results/015_common_disorder_sector_summary.md`
