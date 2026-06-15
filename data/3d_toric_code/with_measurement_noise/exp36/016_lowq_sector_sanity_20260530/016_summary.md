# 016 low-q sector sanity summary

## 结论

本轮检查 `L=6,p=0.05,q=0.08` 低 q 饱和区是否存在伪正确：不同初态各自卡在不同 sector，但 `q_top` 都因为平方平均接近 1。

结果：没有看到这种问题。两 个 common disorder 上，三种初态 `sector / all_zero / random_high_weight` 都驻留在同一个 top sector `+++++++`。三初态 cold-sector TV 最大只有 `0.0002`。

## 实验参数

- `L=6`
- `p=0.05`
- `q=0.08`
- disorder seeds 由 `seed_base=516000` 生成
- `num_disorder_samples_total=2`
- `num_measurements_per_disorder=1024`
- `num_start_chains=4`
- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`

## 关键数值

整体：

- `sector`: `q_top mean = 1.000000`
- `all_zero`: `q_top mean = 0.999721`
- `random_high_weight`: `q_top mean = 1.000000`

三初态整体 pairwise cold-sector TV：

- `sector` vs `all_zero`: `0.0001`
- `sector` vs `random_high_weight`: `0.0000`
- `all_zero` vs `random_high_weight`: `0.0001`

逐 disorder：

- disorder 0:
  - 三初态 `q_top = 1.000000, 1.000000, 1.000000`
  - pairwise TV 全为 `0.0000`
  - top sector 全为 `+++++++`
- disorder 1:
  - 三初态 `q_top = 1.000000, 0.999442, 1.000000`
  - max pairwise TV `0.0002`
  - top sector 全为 `+++++++`

## 物理解释

低 q 下 `q_top=1` 的确对应同一个 Wilson-loop sector，而不是不同初态各自卡在不同 sector。`all_zero` 在第二个 disorder 中只有一次低概率 sector 访问，概率约 `0.0002`，不影响结论。

## 下一步

014/015/016 已覆盖 high-q 和 low-q 的 sector-space correctness gate。下一步应做小矩阵 production-like 试跑：

- `L=3,4,5,6`
- `q=0.08,0.15,0.23`
- 少量 disorder
- 单生产初态配置
- 检查 `q_top` 曲线方向、disorder 方差和 wall time

完整自动报告见：

`remote_results/016_lowq_sector_summary.md`
