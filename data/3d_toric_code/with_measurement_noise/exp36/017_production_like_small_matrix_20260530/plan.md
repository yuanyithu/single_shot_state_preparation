# 017 production-like small matrix plan

## 目标

014/015/016 已经用三初态 cold-sector histogram gate 检查了 high-q 和 low-q 两端：

- `q=0.23`: 1 个 disorder 长链 + 3 个 common disorder gate 均通过。
- `q=0.08`: 2 个 common disorder low-q sanity 通过，确认 `q_top≈1` 不是不同初态卡在不同 sector 的伪正确。

本轮 017 不再重复三初态 gate，而是用通过 gate 的配置跑一个小矩阵 production-like 试验，回答：

1. `q_top(L,q)` 曲线方向是否物理合理。
2. 少量 disorder 下方差大概是多少。
3. 当前配置的 wall time 是否可接受，是否能进入正式生产扫描。

## 物理判据

固定 `p=0.05` 时，如果系统在 threshold 以下，较大 L 的 `q_top` 应不低于小 L，至少不应出现 exp35 那种明显反物理的大尺寸系统性下降。

这不是最终 threshold 扫描；它是进入全区间生产前的小矩阵 gate。

## 实验矩阵

- `L = 3,4,5,6`
- `p = 0.05`
- `q = 0.08,0.15,0.23`
- 每个 `(L,q)` 使用 `4` 个 disorder
- 每个 q 单独一个 run，三个 q 分别放到不同节点
- 三个 q run 使用相同 `seed_base=517000`，确保同一个 L 和 disorder index 在不同 q 之间使用同一批 disorder seeds

## 采样配置

沿用已经通过 sector gate 的生产候选：

- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=1024`
- `num_start_chains=4`
- `num_replicas_per_start=1`
- `q_positive_initial_chain_mode=sector`
- `q_top_block_count=8`

本轮不启用额外 PT cluster-sector diagnostics；cold-sector histogram 和 q_top block summary 已经由主路径保存，足够做小矩阵 gate。

## 通过标准

进入正式生产扫描前，017 至少应满足：

1. 每个 q 上 `q_top` 随 L 不出现明显反物理趋势。
2. 同一 `(L,q)` 的 4 个 disorder `q_top` 标准差可接受；若 SEM 大到无法分辨 L 趋势，则正式扫描需要更多 disorder。
3. block `q_top` drift 不显示系统性漂移；若高 q/L6 drift 明显，则回到更长链或更强混合。
4. wall time 不出现不可接受的长尾；若 L6/q0.23 单 chunk 时间远超 014/015 经验，应重新评估配置。

## 输出

- 三个 q run 的 NPZ/manifest/log。
- `017_summary.md`：
  - 每个 q 的 `q_top(L)` mean/std/SEM。
  - 每个 q 的 L 趋势检查。
  - 每个 q/L 的 block drift/range 概览。
  - wall time 汇总。
  - 是否建议进入正式生产扫描。

## 下一步

若 017 通过，正式生产可以采用同一配置，并扩大 disorder 数；若 017 某个中间 q 或 L 出现异常，再针对那个点回到三初态 sector gate 或加长链复核。

## 外部顾问评估记录

按流程尝试调用 DeepSeek-V4-Pro 和 Kimi-K2.6 评估本计划；两个模型调用均在 45 秒内无有效输出并 timeout。输出记录见：

- `deepseek_plan_review.txt`
- `kimi_plan_review.txt`

批判性修正：

- 保留 017 为 production-like 小矩阵，而不是再对每个点重复三初态 gate；014/015/016 已覆盖 high-q 和 low-q 的初态无关性检查。
- 保留 `q=0.15` 作为中间点。如果 017 中 `q=0.15` 出现异常趋势或 block drift，再追加三初态 sector gate，而不是预先付费。
- `4` 个 disorder 只用于估计方差量级和曲线方向，不作为最终 disorder average。
- `m=1024` 是省时试跑设置；若 L6/q0.23 drift 或方差不可接受，再局部加长到 `m=2048`。
