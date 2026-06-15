# 019 targeted 8-start sector-reference summary

更新时间：2026-05-31

## 目标

018 的 final gate 大部分通过。用 10000-bootstrap 复核后，稳定保留的
final marginal fail 包括 `q=0.13,L=3,disorder=52` 和
`q=0.23,L=3,disorder=15`。019 复查这些点，
并加入几个 low-cost high-q 对照点，判断这些 fail 是否代表不同初态真的没有
收敛到同一个 cold-sector 热分布。

## 流程

按既定流程先写 `plan.md`，再咨询外部顾问：

- DeepSeek-V4-Pro: 120 秒内无有效输出，记录为 timeout。
- Kimi-K2.6: 建议把核心点链长提高，并提醒 019 通过不能证明全区间完成；
  同时提出 `L=3` exact enumeration。后者未采纳，因为当前 3D `L=3`
  有 `n=81`，本项目 exact 路径枚举 `2^n` 条链，不是低成本 reference。

最终修订为：`8/8` sector 初态、`m=8192`、`q_top_block_count=32`，
只跑 4 个 L3 targeted disorder。

## 配置

- `L=3`
- `p=0.05`
- `num_measurements_per_disorder=8192`
- `num_start_chains=8`
- start labels: `000,100,010,110,001,101,011,111`
- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `effective_num_burn_in_sweeps=675`
- `num_sweeps_between_measurements=6`
- disorder seed 复用 018 对应 chunk，MCMC seed 换为 019 新 seed

远端运行：

- node: `nd-2`
- screen: `exp36_019_8start`
- remote root:
  `/home/DATA1/users/yuany/.single_shot/exp36/019_targeted_8start_sector_reference_20260531`

## Gate 结果

报告：`019_sector_gate_boot10000.md`

| q | L | disorder | q_top | start-TV | boot p99 | TV fail | q_top spread | first/second TV max | top sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.13 | 3 | 52 | 0.880904 | 0.0035 | 0.0128 | 0 | 0.006438 | 0.0078 | +++++++:0.945, +--++--:0.055 |
| 0.22 | 3 | 36 | 0.281263 | 0.0214 | 0.0317 | 0 | 0.009820 | 0.0400 | -+-+-+-:0.451, +++++++:0.403 |
| 0.23 | 3 | 6 | 0.738338 | 0.0071 | 0.0189 | 0 | 0.007679 | 0.0220 | +++++++:0.875, -+-+-+-:0.055 |
| 0.23 | 3 | 15 | 0.888930 | 0.0056 | 0.0125 | 0 | 0.009727 | 0.0081 | +++++++:0.949, +--++--:0.041 |

没有任何 targeted disorder 超过 pooled-bootstrap p99。

## 与 018 对照

| q | L | disorder | run | starts | m | q_top | q_top spread | start-TV | first/second TV max | top sectors |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.13 | 3 | 52 | 018 | 4 | 1024 | 0.865445 | 0.079274 | 0.0400 | 0.0117 | +++++++:0.937, +--++--:0.062 |
| 0.13 | 3 | 52 | 019 | 8 | 8192 | 0.880904 | 0.006438 | 0.0035 | 0.0078 | +++++++:0.945, +--++--:0.055 |
| 0.22 | 3 | 36 | 018 | 4 | 1024 | 0.280845 | 0.027566 | 0.0811 | 0.0527 | -+-+-+-:0.445, +++++++:0.410 |
| 0.22 | 3 | 36 | 019 | 8 | 8192 | 0.281263 | 0.009820 | 0.0214 | 0.0400 | -+-+-+-:0.451, +++++++:0.403 |
| 0.23 | 3 | 6 | 018 | 4 | 1024 | 0.732040 | 0.072726 | 0.0449 | 0.0371 | +++++++:0.872, -+-+-+-:0.059 |
| 0.23 | 3 | 6 | 019 | 8 | 8192 | 0.738338 | 0.007679 | 0.0071 | 0.0220 | +++++++:0.875, -+-+-+-:0.055 |
| 0.23 | 3 | 15 | 018 | 4 | 1024 | 0.892532 | 0.067668 | 0.0332 | 0.0195 | +++++++:0.951, +--++--:0.041 |
| 0.23 | 3 | 15 | 019 | 8 | 8192 | 0.888930 | 0.009727 | 0.0056 | 0.0081 | +++++++:0.949, +--++--:0.041 |

## 物理结论

019 没有看到“不同初态长期驻留在不同 Wilson-loop sector”的信号。
最重要的 `q=0.13,L=3,d52` 在 018 中的 marginal fail，在 8/8 初态、
8 倍 measurement 的 reference 中消失；而且 top sector 概率与 018 相容。
这说明 018 的该 fail 更像有限 measurement 下的 start-chain 统计波动，
不是真实的 metastable sector splitting。

`q=0.23,d15` 的 final marginal fail 也在 reference 中消失。`q=0.22,d36`
和 `q=0.23,d6` 对照点给出同样图像：`q_top` 与 sector
驻留概率和 018 相容，但 8-start 长链的 `q_top spread` 大幅下降。

## 边界

019 只清除了当前已知的 targeted marginal fail，不能证明整个
`L=3,4,5,6; q=0.08..0.23` 全区间已经完成。下一步仍是等待 018 全 q-grid
完成，对所有 final NPZ 做同一 sector-histogram gate。若最终 gate 出现新的
稳定 fail，再做同类 targeted 8-start/长链复查，而不是直接补 disorder。
