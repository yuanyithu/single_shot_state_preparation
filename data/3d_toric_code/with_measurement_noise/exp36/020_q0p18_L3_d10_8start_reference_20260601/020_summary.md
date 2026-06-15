# 020 q=0.18 L3 d10 targeted 8-start reference summary

更新时间：2026-06-02（本机 Asia/Shanghai；远端日志为 2026-05-31 EDT）

## 目标

018 的 `q=0.18,L=3,disorder=10` 在 4-start、`m=1024` 生产链中留下一个
10000-bootstrap 边界 fail：

- observed start-TV: `0.0195`
- bootstrap p99: `0.0176`
- `q_top=0.964710`
- `q_top spread=0.038982`

020 只复查这个固定 disorder，不补 disorder average，也不改变生产结论口径。
物理问题是：从 8 个 Wilson-loop sector 初态出发，长时间后是否落到同一个
cold-sector 热分布。

## 流程

- 先写入 `plan.md`。
- DeepSeek-V4-Pro: 超时，无有效意见，记录在 `deepseek_plan_review.txt`。
- Kimi-K2.6: 建议加入一个已知通过的 sanity disorder，已采纳。
- 最终跑两个 L3 点：核心 fail `d10` 与 sanity pass `d1`。

## 配置

- `L=3`, `p=0.05`, `q=0.18`
- `num_measurements_per_disorder=8192`
- `num_start_chains=8`
- start labels: `000,100,010,110,001,101,011,111`
- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `effective_num_burn_in_sweeps=675`
- `num_sweeps_between_measurements=6`
- `q_top_block_count=32`

远端结果：

- remote root: `/home/DATA1/users/yuany/.single_shot/exp36/020_q0p18_L3_d10_8start_reference_20260601`
- screen log: `remote_results/exp36_020_nd2.screen.log`
- gate report: `020_sector_gate_boot10000.md`

## 结果

| q | L | disorder | role | q_top | start-TV | boot p99 | TV fail | first/second TV max | top sectors |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.18 | 3 | 1 | sanity pass | 0.999093 | 0.0009 | 0.0011 | 0 | 0.0010 | +++++++:1.000 |
| 0.18 | 3 | 10 | original fail | 0.969400 | 0.0033 | 0.0065 | 0 | 0.0059 | +++++++:0.986, +++----:0.007, -+-+-+-:0.006 |

两个点均通过 pooled-bootstrap p99 sector-histogram gate。

## 与 018 对照

| q | L | disorder | run | starts | m | q_top | q_top spread | start-TV | boot p99 | top sectors |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.18 | 3 | 10 | 018 | 4 | 1024 | 0.964710 | 0.038982 | 0.0195 | 0.0176 | +++++++:0.984, +++----:0.010, -+-+-+-:0.005 |
| 0.18 | 3 | 10 | 020 | 8 | 8192 | 0.969400 | 0.007390 | 0.0033 | 0.0065 | +++++++:0.986, +++----:0.007, -+-+-+-:0.006 |

## 物理结论

020 没有看到 `q=0.18,L=3,d10` 上存在稳定的初态依赖。更长链、8/8 sector 初态
下，常驻 sector 的集合和概率与 018 相容，但不同初态之间的 histogram 差异和
`q_top` spread 明显下降。

因此这个点应解释为 018 的 4-start、`m=1024` 有限测量边界事件，而不是 MCMC
在该 disorder 上稳定卡在不同 Wilson-loop sector 热盆。`q=0.18` 不需要继续
追加 targeted reference。

边界：020 只复查一个 fail 点和一个 sanity 点；它清除的是已知边界异常，不替代
018 全 q-grid 的最终 sector gate。
