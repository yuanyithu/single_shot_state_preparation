# 015 common-disorder cold-sector gate plan

## 目标

014 已经在 `L=6,p=0.05,q=0.23` 的一个固定 disorder seed 上证明：三种初态最终给出几乎相同的 cold-sector 驻留分布。本轮 015 的目标是检查这不是单个 disorder 的偶然现象。

本轮仍然只回答正确性问题：固定 disorder 后，MCMC 是否采到同一个热态分布，而不是只让 `q_top` 看起来接近。

## 物理判据

每个冷端测量时间片有一个 Wilson loop sector：

`sector_t = (W_1(c_t), ..., W_7(c_t))`

对同一个 disorder，如果 MCMC 已经热化，则从 `sector / all_zero / random_high_weight` 三种初态出发，长时间后的 sector 直方图应一致。

我们比较：

- 同一 disorder 内三初态之间的 cold-sector TV 距离。
- 每条链前半段 vs 后半段的 cold-sector TV。
- 同一 disorder 内三初态的 `q_top`、`m_u` 是否一致。
- 三个 disorder 上 `q_top` 的样本方差，估计 production 需要的 disorder 数。

## 实验参数

固定参数：

- `L=6`
- `p=0.05`
- `q=0.23`
- 三个 common disorder seeds，由同一个 `seed_base` 和 disorder index 生成
- 三种初态：
  - `sector`
  - `all_zero`
  - `random_high_weight`

采样配置沿用 014 已通过的版本：

- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=2048`
- `num_start_chains=4`
- `num_replicas_per_start=1`
- `q_top_block_count=8`

运行方式：

- 每个初态单独一个 run。
- `num_disorder_samples_total=3`
- `chunk_size=1`
- `common_random_disorder_across_p` 打开。
- 三个 run 使用相同 `seed_base=515000`，确保 disorder seed 对齐。

## 通过标准

对每个 disorder：

- 三初态 pairwise cold-sector TV 均小于 `0.05`。
- 每条链前半/后半 TV 的 max 小于 `0.10`。
- 三初态 `q_top` 差异不超过同 disorder 内 sector sampling noise 的量级；工程 gate 先取 max spread `< 0.01`。

整体：

- 3 个 disorder 中至少 3 个都通过上述 gate，才进入低 q sanity。
- 如果只有 2/3 通过，不进入生产扫描；先看失败 disorder 的 top sectors 和 block TV，再决定加长链还是改 PT。
- 如果有任一 disorder 出现三初态 TV `>0.20`，说明存在明确未热化风险，停止速度优化。

## 预期输出

- 每个初态的 run 目录和 NPZ。
- `015_common_disorder_sector_summary.md`：
  - 每个 disorder 的三初态 q_top 表。
  - 每个 disorder 的三初态 pairwise cold-sector TV 矩阵。
  - 每个 disorder 的 top sectors。
  - 前半/后半 TV 统计。
  - disorder 间 q_top mean/std。

## 下一步决策

如果 015 通过：

1. 在 `q=0.08` 做一个低 q 饱和点 sanity，只需 1 到 2 个 disorder。
2. 若也通过，开始生产扫描前的小矩阵：`L=3,4,5,6`，抽 `q=0.08,0.15,0.23`，少量 disorder，检查趋势和方差。

如果 015 不通过：

1. 不启动生产扫描。
2. 优先调整跨 sector 混合机制，例如提高热端、温度数、burn-in 或引入更强的 sector-changing update。

## 外部顾问评估记录

按流程分别尝试调用 DeepSeek-V4-Pro 和 Kimi-K2.6 评估本计划；两个模型调用均在 45 秒内无有效输出并 timeout。输出记录见：

- `deepseek_plan_review.txt`
- `kimi_plan_review.txt`

因此本轮不等待外部顾问，避免工具故障阻塞实验。

我的批判性修正：

- 3 个 common disorder 只作为进入生产扫描前的正确性 gate，不作为最终 disorder average 方差估计。
- 每个 disorder 必须单独报告三初态 sector TV；不能把 3 个 disorder 混合成一个直方图后下结论。
- `TV<0.05` 是工程 gate，不是严格数学证明；若结果接近阈值，优先加长链或增加 disorder，而不是直接生产。
- 若 3 个 disorder 都像 014 一样给出 TV 约 `1e-3`，则说明当前配置在 high-q 端通过初态无关性检查，下一步转向低 q sanity 和小矩阵 production-like 试跑。
