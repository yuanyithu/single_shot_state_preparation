# 020 q=0.18 L3 d10 targeted 8-start reference plan

## 目标

018 的 `q=0.18` final sector-histogram gate 已完整覆盖 `L=3,4,5,6`
各 64 个 disorder。`L=4/5/6` 均通过，但 `L=3,disorder=10` 在
10000-bootstrap 下仍保留一个边界 fail：

- observed start-TV: `0.0195`
- bootstrap p99: `0.0176`
- `q_top=0.964710`
- `q_top spread=0.038982`
- top sectors: `+++++++:0.984, +++----:0.010, -+-+-+-:0.005`

020 不扩大生产扫描，不补 disorder average。它只复查这个固定 disorder：
判断 018 的 4-start、`m=1024` 边界 fail 是有限测量统计波动，还是不同
初态真的没有收敛到同一个 cold Wilson-loop sector 热分布。

## 物理检验

固定 disorder 后，冷端每个 measurement 对应 Wilson-loop sector 向量
`S_t=(W_1(t),...,W_7(t))`。若 MCMC 已经采到该 disorder 下的热态，不同
logical sector 初态长时间后应给出同一个 sector 驻留分布 `h(S)`。

020 比较：

1. 8 个 zero-syndrome sector 初态 `000,100,010,110,001,101,011,111`
   的 cold-sector histogram。
2. 8-start 最大 pairwise TV 是否低于 pooled-bootstrap p99。
3. first/second half histogram 是否同量级稳定。
4. 020 的 8-start 长链 `q_top`、sector probabilities 是否与 018 的同一
   disorder 相容，但 start spread 显著下降。

## 顾问意见与批判性修订

DeepSeek-V4-Pro 调用超过约 150 秒无有效输出，记录见
`deepseek_plan_review.txt`，不作为本轮决策依据。

Kimi-K2.6 返回了有效评估，记录见 `kimi_plan_review.txt`。可采纳意见：

- `8/8` 初态、`m=8192` 足以判断 018 中这个只有约 11% 超出 p99 的边界
  fail 是否主要来自有限测量波动。
- 建议加入一个同 `q=0.18,L=3` 下 018 已知通过的 sanity disorder，用来确认
  本轮 seed、chunk、环境和 bootstrap 流程没有配置错误。L3 单点代价约
  4.5 分钟，加入 sanity 点的额外服务器时间可接受，且可降低重跑风险。

不采纳或暂缓的意见：

- 现在不直接加到 `m=16384/32768`。若 `m=8192` 仍失败，再新开更长链或更强
  update A/B；过早加长会浪费服务器时间。
- 现在不提高 burn-in 到 1000。020 的目标是复查 018/019 同一更新配置下的
  初态依赖，先保持 `effective_num_burn_in_sweeps=675`。若 first/second
  half 显示漂移，再把 burn-in 作为下一轮变量。
- 现在不提高 `q_top_block_count` 到 64/128。`m=8192`、32 blocks 已能看
  block drift；过细 block 会让单 block 噪声升高。

正式修订：020 跑 2 个 targeted points：核心 fail `d10` 和 sanity pass `d1`。

## 最小实验

mandatory point：

- `L=3`
- `p=0.05`
- `q=0.18`
- `disorder=10`
- 018 原始 `disorder_seed=70518049`

sanity point：

- `L=3`
- `p=0.05`
- `q=0.18`
- `disorder=1`
- 018 原始 `disorder_seed=7518022`
- 018 中该点已经干净通过：`q_top=0.999442`，observed TV `0.0010`，
  boot p99 `0.0020`

理由：019 已经证明同类 L3 边界 fail 可由 8-start 长链清除；020 的目标是
尽量少服务器时间内判断这个新 fail 是否同类。加入一个 sanity pass 点会使
服务器时间约翻倍，但仍只是约 9 分钟量级，可避免单点结果因配置错误而不可解释。
若 `d10` 通过且 `d1` 正常通过，不需要再为 `q=0.18` 追加 targeted reference；
若 `d10` 失败，再启动下一轮更强 A/B，而不是先补更多 disorder。

## 初始配置建议

沿用 018/019 已验证的物理更新配置，只加强初态覆盖和测量长度：

- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `effective_num_burn_in_sweeps=675`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=8192`
- `num_start_chains=8`
- `num_replicas_per_start=1`
- `q_positive_initial_chain_mode=sector`
- `q_top_block_count=32`
- new MCMC seeds: `520000` for `d10`, `1520003` for `d1`

## 实施方式

使用 `production_chunked_scan.py run-chunk` 直接跑指定 chunk。

018 使用 `chunk_size=1`，所以 `disorder_offset == chunk_index == disorder`。
本轮固定 `--disorder-seed` 为 018 原始值，确保复查的是同一个 disorder
realization；MCMC seed 改为 020 新 seed，避免复用 018 链随机数。

远端建议：

- node: `nd-2` 或当前空闲计算节点
- screen: `exp36_020_8start`
- remote root:
  `/home/DATA1/users/yuany/.single_shot/exp36/020_q0p18_L3_d10_8start_reference_20260601`

## 通过标准

mandatory `d10` 通过，当且仅当：

- 8-start 最大 pairwise cold-sector TV 不超过 pooled-bootstrap p99。
- first/second TV 与 start-TV 同量级，没有明显单向 drift。
- `q_top spread` 相比 018 的 `0.038982` 明显下降，且 top sector probabilities
  与 018 在统计误差内相容。

若通过：把 018 的 `q=0.18,L=3,d10` 解释为 4-start、`m=1024` 下的有限测量
边界事件；继续等待 018 的 `q=0.19`，不为 `q=0.18` 追加实验。

sanity `d1` 也必须正常通过；若 `d1` 失败，优先判定为本轮配置/环境/分析流程
异常，先排查而不是立即修改 MCMC 算法。

若 `d10` 失败且 `d1` 正常通过：不能把当前生产配置视为已经在该 disorder 上
热化。下一步新开实验，优先比较同一 disorder 的更长链 `m=16384/32768` 或
更强更新参数，而不是补 disorder average。

## 启动条件

本 plan 已完成 DeepSeek-V4-Pro/Kimi-K2.6 顾问流程和批判性修订，可以进入
远端执行。
