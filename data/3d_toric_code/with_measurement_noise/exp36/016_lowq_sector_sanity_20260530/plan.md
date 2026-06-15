# 016 low-q sector sanity plan

## 目标

014/015 已经证明当前配置在 high-q 端 `q=0.23` 通过 cold-sector 初态无关性 gate。本轮检查低 q 饱和区 `q=0.08`。

低 q 区域的危险点是：`q_top` 很容易等于 1。如果不同初态各自困在不同固定 sector，`q_top` 仍可能都是 1。因此本轮不以 `q_top=1` 作为正确性证据，而是检查三初态的 cold-sector 直方图是否一致。

## 物理判据

固定 disorder 后，每个冷端测量时间片有

`sector_t = (W_1(c_t), ..., W_7(c_t))`

若热态在低 q 下确实几乎完全位于某个 sector，那么三种初态最终应驻留在同一个 sector，尤其 top sector 应一致。

若三种初态分别驻留在不同 sector，但 `q_top` 都等于 1，则说明之前的 `q_top` 饱和是伪正确，不能进入生产扫描。

## 实验参数

- `L=6`
- `p=0.05`
- `q=0.08`
- `num_disorder_samples_total=2`
- 三种初态：
  - `sector`
  - `all_zero`
  - `random_high_weight`
- `seed_base=516000`

采样配置：

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
- `q_top_block_count=8`

选择 `m=1024` 是为了省服务器时间；低 q 若真的饱和，sector histogram 应很快稳定。如果结果不明确，再加到 `m=2048`。

## 通过标准

对每个 disorder：

- 三初态 pairwise cold-sector TV `< 0.05`。
- 三初态 top sector 一致。
- 若 `q_top=1`，必须确认是同一个 sector 导致，而不是不同初态各自卡住。

若任一 disorder 出现不同初态 top sector 不同，或 pairwise TV `>0.20`，立即停止进入生产扫描，回到混合算法。

## 下一步

若 016 通过：

1. 当前配置通过 high-q gate 和 low-q sanity。
2. 下一步进入小矩阵 production-like 试跑：`L=3,4,5,6`，`q=0.08,0.15,0.23`，少量 disorder，检查曲线趋势与方差。

## 外部顾问评估记录

按流程尝试调用 DeepSeek-V4-Pro 和 Kimi-K2.6 评估本计划；两个模型调用均在 45 秒内无有效输出并 timeout。输出记录见：

- `deepseek_plan_review.txt`
- `kimi_plan_review.txt`

批判性修正：

- 保留 `q=0.08`，因为它正是最容易让 `q_top=1` 掩盖问题的低 q 饱和点。
- `m=1024` 只作为 sanity；如果三初态 top sector 不一致，立刻判失败；如果 TV 接近阈值，再加长到 `m=2048`。
- 不把本轮用于估计最终 disorder 方差，只用于确认低 q 端不会出现“不同 sector 但 q_top 都为 1”的伪正确。
