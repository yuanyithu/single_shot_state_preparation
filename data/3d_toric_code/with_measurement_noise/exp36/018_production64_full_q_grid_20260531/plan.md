# 018 production64 full-q-grid plan

## 目标

014/015/016 已经用 cold-sector histogram gate 验证当前配置在 `L=6,p=0.05` 的 high-q 和 low-q 两端没有明显初态依赖；017 小矩阵也显示 `q_top(L,q)` 没有出现 exp35 那种反物理趋势。

本轮 018 开始第一批正式 disorder average：

- `L=3,4,5,6`
- `p=0.05`
- `q=0.08,0.09,...,0.23`
- 每个 `(L,q)` 先跑 `64` disorder

这不是最终最大样本数，而是第一批 production。目标是用可接受服务器时间得到一版完整 q-grid 曲线，并估计哪些点需要补样本。

## 生产配置

沿用 014-017 已通过 gate 的配置：

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

## q 分组

为了平衡三台节点的 wall time，把高 q 分散到不同节点：

- nd-1: `q=0.08,0.09,0.10,0.22,0.23`
- nd-2: `q=0.11,0.12,0.13,0.20,0.21`
- nd-3: `q=0.14,0.15,0.16,0.17,0.18,0.19`

每个 q 是一个独立 run，节点内顺序执行，便于失败后定位和补跑。

## 随机数策略

所有 q run 使用同一个 `seed_base=518000`。`production_chunked_scan.py` 的 disorder seed 对同一 `(L, disorder_index)` 不依赖 q，因此不同 q 间使用 common disorder random stream，有利于降低 q 曲线差分噪声。

## 通过/补样本标准

第一批完成后检查：

1. 每个 `(L,q)` 的 `q_top` mean/std/SEM。
2. `q_top(L,q)` 是否有明显反物理趋势。
3. `q_top_block_drift/range` 是否提示链长不足。
4. wall time 是否和 017 估计一致，是否有异常长尾。

补样本规则：

- 若某点 SEM 明显大于相邻 L 间差值，优先局部补 disorder。
- 若某点 block drift 明显偏大，优先局部加长 `m`，而不是盲目补 disorder。
- 若中间 q 出现异常 L 趋势，再针对该点追加三初态 cold-sector gate。

## 预期成本

017 的 4-disorder 小矩阵给出粗略估计：

- `q=0.08` 全 L 4 disorder 约 `512s`
- `q=0.15` 全 L 4 disorder 约 `548s`
- `q=0.23` 全 L 4 disorder 约 `1318s`

放大到 64 disorder 约为 16 倍。三节点分组后，预期 wall time 是数十小时量级，不是交互式短 run。

## 输出

每个 q run 保存到 `remote_results/run_qXXXX.../`，本地回收后生成：

- `018_summary.md`
- q-grid mean/std/SEM 表
- `q_top(L,q)` 曲线图
- block drift/range gate 表
- 需要补样本的点列表

## 外部顾问评估记录

按流程尝试调用 DeepSeek-V4-Pro 和 Kimi-K2.6 评估本计划；两个模型调用均在 45 秒内无有效输出并 timeout。输出记录见：

- `deepseek_plan_review.txt`
- `kimi_plan_review.txt`

批判性修正：

- 保留 `64` disorder/point，原因是 017 显示 `q=0.23,L=3/4` 的 4-disorder SEM 较大；32 disorder 可能不足以看清 L 趋势。
- 不再缩小 q 范围，因为 017 已完成 `q=0.08,0.15,0.23` 三点 sanity；018 的价值正是补齐完整 q-grid。
- 保留 `m=1024`；若 block drift gate 发现局部链长不足，再针对该点加长到 `m=2048`。
- 节点分组把 `q=0.22/0.23`、`q=0.20/0.21`、`q=0.14..0.19` 分散，避免一个节点独占所有高 q 长尾。
