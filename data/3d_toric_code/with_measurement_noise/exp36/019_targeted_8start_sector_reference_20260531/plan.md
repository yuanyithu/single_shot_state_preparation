# 019 targeted 8-start sector-reference plan

## 目标

018 已完成的 final gate 中，大多数 `(q,L)` 没有不同初态 cold-sector
histogram 分裂。用 10000-bootstrap 复核后，稳定保留的 final marginal
fail 只有：

- `q=0.13, L=3, disorder=52`: observed start-TV `0.0400`,
  bootstrap p99 `0.0342`, `q_top=0.865445`, `q_top spread=0.079274`.

本轮 019 不扩大生产扫描，而是只做 targeted correctness reference：
判断这个 marginal fail 是生产链长度/4-start 统计噪声，还是固定 disorder
下不同初态真的没有收敛到同一个 Wilson-loop sector 热分布。

## 物理检验

固定 disorder 后，冷端每个 measurement 对应 Wilson-loop sector 向量
`S_t=(W_1(t),...,W_7(t))`。正确热化时，不同拓扑初态长时间后应给出同一个
sector 驻留分布 `h(S)`。

019 比较：

1. 8 个 zero-syndrome sector 初态 `000..111` 的 cold-sector histogram。
2. 前半/后半链 histogram 是否稳定。
3. 8-start 的 `q_top` spread 和 block range。
4. 019 的 8-start 结果是否和 018 中同一 disorder 的 4-start 结果相容。

若 8-start/长链仍出现不同初态 sector histogram 分裂，则当前生产配置不能被
视为在该点热化；若 8-start/长链收敛到同一 histogram，则 018 的 marginal
fail 可解释为有限 measurement / bootstrap p99 边界事件。

## 复查点

最小必跑：

- `q=0.13, L=3, disorder=52`

低成本对照：

- `q=0.22, L=3, disorder=36`：1000-bootstrap 曾 marginal fail，
  10000-bootstrap 已通过，用来确认 near-threshold high-q 行为。
- `q=0.23, L=3, disorder=6` 和 `15`：018 当前 partial gate 中 high-q
  L3 出现 marginal flags，虽然 q=0.23 final 尚未完成；加入这两个点可以用
  很少成本提前检查高 q 小尺寸最容易波动的情形。

## 顾问意见与批判性修订

DeepSeek-V4-Pro 调用在 120 秒内没有返回有效输出，记录见
`deepseek_plan_review.txt`。

Kimi-K2.6 返回了有效评估，记录见 `kimi_plan_review.txt`。可采纳意见：

- 019 作为 stop-loss targeted test 是合理的；若失败，足以说明当前生产配置
  不能直接当作最终热化版本。
- `m=2048` 对 8 个初态、128 个 full Wilson-loop sector bin 来说仍偏短；
  核心点应加长。
- 019 通过不能证明整个 018 全区间已经完成，只能清除当前 marginal fail；
  后续仍要等待 018 final gate，并对新出现的 final fail 做同类 targeted 复查。

不采纳意见：

- Kimi 建议对 `L=3` 做 exact enumeration gold standard。这个建议不适用于
  当前 3D `L=3`：本项目 exact 路径是枚举 `2^n` 条 chain，而 3D `L=3`
  有 `n=81`，不可作为低成本 exact reference。exact 只适合更小尺寸或专门
  降维校验，不能放入 019。

正式修订：

- 核心点和对照点统一使用 `m=8192`，`q_top_block_count=32`，确保每个初态
  的 sector histogram 统计误差显著低于 018。
- burn-in 不改，仍用 018 的 `max_effective_num_burn_in_sweeps=750`；这样复查
  主要检验长时驻留分布和初态依赖，而不把问题混入不同 burn-in 设定。
- 若 019 通过，不声明全区间完成；只说明当前 stable final marginal fail 被清除，
  然后继续等待 018 全 q-grid。
- 若 019 失败，下一步先做同一 disorder 的更长链/更强更新 A/B，而不是补
  disorder average。

## 配置

沿用 018 的物理更新配置，只加强诊断覆盖：

- `L=3`
- `p=0.05`
- `q=0.13,0.22,0.23` 中指定 disorder
- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=8192`
- `num_start_chains=8`
- `num_replicas_per_start=1`
- `q_positive_initial_chain_mode=sector`
- `q_top_block_count=32`

## 实施方式

使用 `production_chunked_scan.py run-chunk` 直接跑指定 chunk：

- 018 使用 `chunk_size=1`，所以 `disorder_offset == chunk_index == disorder`。
- 对 L3，018 的原始 disorder seed 公式为
  `disorder_seed = 518000 + 7000003 * disorder + 19`。
- 019 为避免与 018 完全复用链随机数，MCMC seed 使用新的 `519000` base；
  disorder seed 保持 018 的原始值，以确保是同一个 disorder realization。

## 通过标准

对每个 targeted disorder：

- 8-start 最大 pairwise cold-sector TV 不超过 pooled-bootstrap p99。
- first/second TV 与 start-TV 同量级，没有明显单向 drift。
- `q_top` spread 明显小于 018 的 suspicious spread；若仍大，必须看是否对应
  sector histogram 分裂。

整体判据：

- 若 `q=0.13,L=3,d52` 通过，且 q=0.22/q=0.23 对照点没有系统分裂，则继续等待
  018 全 q-grid 完成，不追加大规模诊断。
- 若 `q=0.13,L=3,d52` 仍失败，则停止把当前 018 配置当作最终热化版本，
  下一步应改为更长链或替代 update kernel 的小矩阵 A/B，而不是补 disorder。

## 启动条件

本计划已经完成外部顾问咨询和批判性修订，可以进入执行。
