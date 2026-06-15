# exp36 下一步建议：以机器时间和 q_top 热态可信性为目标

日期：2026-05-29

本文是 Codex 在两轮 Claude Code 咨询后给出的最终建议。Claude 只作为咨询顾问；这里的取舍以本项目总目标为准：**降低关心参数区间内 MCMC 所需机器时间，同时给出接近热平衡采样的 `q_top`**。

## 目标重述

exp36 后续不再把 sector flip、arrival、roundtrip、PT swap、cluster changed 等诊断指标当成目标。它们只用于解释为什么 `q_top` 可信或不可信、为什么机器时间高或低。

后续每个实验必须回答两个问题：

1. 在同等或更低机器时间下，`q_top` 是否更接近热平衡采样？
2. 现有证据是否足够把该参数作为生产扫描候选？

## 当前结论

exp36 已经证明：

- 旧 convergence gate 会漏判严重冷端冻结；不能单靠旧 gate 判断 `q_top` 可信。
- 热端会频繁换 sector，不等于 cold 端 `q_top` 已热化。
- `q_hot≈0.32/0.35` 比 `q_hot=0.44` 更合理；继续升热端会制造中间温度 swap bottleneck，并增加机器时间。
- repeated swap sweeps、winding-plane heatbath、near-cold ladder spacing power>1、继续升 `q_hot` 都没有稳定改善 cold 端 `q_top` 可信性，且通常增加机器时间或降低 roundtrip。
- cluster update 可以在中温制造 sector change，并且这些 change 有时能到达 cold；但 004-007 显示它们多数不能转化为稳定 cold-slot sampling。
- 006 的高频 measurement 只看到很少 transient cold flip；007 的 cold-edge hold 能让 arrival 被诊断捕获、增加 dwell，但 cold flips 仍为 0。因此问题不是单纯“没看到 arrival”。

尚未证明：

- 当前候选参数在 `L=6,p=0.05,q=0.08` 下能以可接受机器时间给出热态可信的 `q_top`。
- 少量 cold flip 或 arrival survived 能说明 `q_top` 正确。
- 相同初始化分布下的多 seed 收敛能排除共同冻结；如果所有链冻在同一个错误区域，`q_top_spread` 也可能很小。

## 立即停止的方向

除非 009 给出强烈反证，否则停止继续投入以下方向：

- `pt_ladder_spacing_power > 1`。008 已显示它主要制造 hot-side bottleneck，roundtrip 降低，没有改善 cold flips。
- `q_hot > 0.40`。此前结果显示机器时间和 transport 都变差。
- 增加 `pt_swap_sweeps_per_attempt`。此前没有带来稳定 cold sampling，`swap_sweeps=4` 成本明显变高。
- winding-plane heatbath。热端 changed 很多但没有改善 cold `q_top` 可信性。
- 单纯继续增加 `cluster rho` 或 measurement 数量。若没有 `q_top` 稳定性证据，只会增加机器时间。
- full PT sector histogram 诊断作为默认长链设置。它适合定位机制，不适合生产-like 性价比评估。

## 009 完成后的处理

009 只作为当前调度类优化的收尾检验：`m=2048,cold_edge_stride=4,q_hot=0.35,rho=0.15,K=17,lightdiag` 两个独立 seed。

009 完成后必须先生成目标导向 summary，不只看 arrival/roundtrip。summary 主表必须包含：

- `q_top_curve_matrix` 或最终 `q_top`
- `chain_q_top_values_per_disorder_per_start_replica_tensor`
- `mean_q_top_spread_curve_matrix` / `q_top_spread_per_disorder_tensor`
- `max_r_hat_curve_matrix`
- `min_effective_sample_size_curve_matrix`
- ordinary update / PT swap / observable / measurement / cluster wall time
- 每个 run 的总 wall time 与相对成本

临时 go/no-go 标准：

- 若 `max_r_hat > 1.05`、`min ESS < 100`、或 `q_top_spread > 0.02`，则不能把该配置视为热态可信。
- 若两个 009 seed 的 `q_top` 差异大于统计误差或 `0.02` 量级，也不能作为生产候选。
- 若 wall time 明显高于便宜基线而 `q_top` 稳定性没有明显改善，应停止该方向。

这些阈值是为了防止继续无止境调参；后续可随更多数据微调，但不能在没有硬标准的情况下继续扩实验。

## 009 后的下一步实验顺序

### 1. 先做 radical-start q_top 收敛测试

这是最便宜、最直接的热化压力测试，优先级高于继续扫参数。

做法：

- 固定同一个 disorder realization。
- 使用当前最佳候选配置，例如 `q_hot=0.35,rho=0.15,cold_edge_stride=4`，关闭 full sector diagnostics，只保留必要轻量记录。
- 从明显不同的初态启动：例如 all-zero、随机高权重初态、非平庸 sector 初态或现有 start-sector 机制能构造的极端 sector。
- 每个初态用相同 wall-time 预算运行，并记录 `q_top` block/window 轨迹。

停止条件：

- 若不同初态的 `q_top` 在相同 wall-time 后仍差异 `>0.02`，或 block 末端仍有系统 drift，则当前候选不能视为热化，停止用它做生产扫描。
- 若不同初态的 `q_top` 在多个 block 内收敛到同一值，且 `Rhat<=1.05`、`ESS>=100`、`q_top_spread<=0.02`，才进入共同 disorder A/B。

理由：相同初始化分布的多 seed 可能共同冻结；radical-start 能更直接暴露这一点。

### 2. 增加低开销 q_top block/window summary

当前 production NPZ 有最终 chain-level `q_top`、Rhat、ESS、wall time，但没有低开销的 `q_top` 分窗轨迹。full measurement trajectory 太贵，不适合作为默认长链诊断。

建议新增一个轻量输出：

- 将 cold 端 measurement 分成 4 或 8 个 block。
- 每个 block 只保存 cold 端 `m_u` 均值和 `q_top`，不保存全温度全 measurement trace。
- merge 后输出每条 start chain 的 block `q_top`、block drift、末半段 vs 全段差异。

用途：判断 `q_top` 是否随时间稳定，直接服务热化判据。这个改动应优先于继续新增复杂物理 kernel。

### 3. 共同 disorder A/B 小矩阵

radical-start 通过后，再做小矩阵比较机器时间和 `q_top` 可信性。

设计：

- 使用 3-5 个共同 disorder，避免 disorder 噪声掩盖算法差异。
- 至少比较三类配置：
  - 当前候选：`q_hot=0.35,rho=0.15,cold_edge_stride=4,minimal diagnostics`
  - 便宜基线：例如 `q_hot=0.32,rho=0` 或低 rho、无 cold-edge hold、minimal diagnostics
  - 长链参考：相同候选参数但 2x 或 4x wall-time，用作近似 reference
- 每个配置记录 `q_top`、block drift、chain spread、Rhat、ESS、wall time。

停止条件：

- 若候选相比便宜基线 wall time 增加明显，但共同 disorder 上 `q_top` 与长链参考没有更接近，停止该候选。
- 若便宜基线和候选在共同 disorder 上的 `q_top` 已一致，优先选择便宜基线。
- 若长链参考本身仍有 drift 或 radical-start 不一致，则该参数区间暂时没有可信生产配置，应回到 kernel 设计而不是扩大扫描。

### 4. 小尺寸 exact / production-path 校准

小 L exact 不是大 L 热化证明，但可以排除 observable 公式、section、production merge 或 PT marginal 的代码级偏差。

建议：

- 在可 exact 的小尺寸上，用相同 `p=0.05,q=0.08` 或附近参数比较 exact `q_top` 与 production path MCMC。
- 用它校准 `q_top` bias 和 block/window summary 的判读方式。
- 不要用“小 L 正确”直接证明 `L=6` 正确；它只证明代码路径没有明显错误。

## 需要修正的汇总口径

`src/summarize_exp36_probe.py` 当前主表偏向 sector/arrival/roundtrip。后续应改成两张表：

1. 目标表：`q_top`、chain spread、Rhat、ESS、wall time、单位 wall-time 有效样本/稳定性。
2. 解释表：cold flips、arrival、dwell、roundtrip、min swap、cluster changed。

解释表不能替代目标表。若二者冲突，以 `q_top` 稳定性和机器时间为准。

## 最终建议

009 完成后，不应继续围绕调度和诊断指标做小步试错。正确顺序是：

1. 用目标导向 summary 评估 009。
2. 若 009 不满足硬标准，停止当前调度类优化。
3. 实现低开销 `q_top` block/window summary。
4. 做 radical-start q_top 收敛测试。
5. 只有 radical-start 通过后，才做共同 disorder A/B 并考虑生产-like 参数。

这一路线把 exp36 从“追诊断事件”转回用户真正关心的目标：更少机器时间下可信的 `q_top`。
