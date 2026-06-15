# 017 production-like small matrix summary

## 结论

014/015/016 已经通过 cold-sector 初态无关性 gate。本轮 017 使用生产单初态 `sector` 跑小矩阵：

- `L=3,4,5,6`
- `p=0.05`
- `q=0.08,0.15,0.23`
- 每点 `4` disorder
- `m=1024`

结果没有出现 exp35 那类明显反物理的大尺寸系统性下降。当前配置可以作为正式生产候选；正式扫描需要增加 disorder 数，尤其 `q=0.23` 的小 L 方差较大。

## q_top 曲线

### q = 0.08

| L | mean | std | SEM |
|---|---:|---:|---:|
| 3 | 0.999582 | 0.000836 | 0.000418 |
| 4 | 1.000000 | 0.000000 | 0.000000 |
| 5 | 0.999861 | 0.000279 | 0.000139 |
| 6 | 1.000000 | 0.000000 | 0.000000 |

低 q 饱和区全尺寸 `q_top≈1`，与 016 sector sanity 一致。

### q = 0.15

| L | mean | std | SEM |
|---|---:|---:|---:|
| 3 | 0.984711 | 0.020648 | 0.010324 |
| 4 | 0.990979 | 0.009209 | 0.004605 |
| 5 | 0.996795 | 0.001783 | 0.000892 |
| 6 | 0.993333 | 0.008187 | 0.004094 |

整体趋势仍是大尺寸较高；`L6` 略低于 `L5`，差值约 `0.00346`，小于合并 SEM 量级，不能判定为异常。

### q = 0.23

| L | mean | std | SEM |
|---|---:|---:|---:|
| 3 | 0.952323 | 0.047157 | 0.023579 |
| 4 | 0.974103 | 0.030928 | 0.015464 |
| 5 | 0.995962 | 0.003970 | 0.001985 |
| 6 | 0.995960 | 0.001894 | 0.000947 |

高 q 端没有出现大 L 系统性下降；`L5/L6` 明显高于 `L3/L4`。但 `L3/L4` 的 4-disorder 方差较大，正式生产需要显著增加 disorder 数。

## Block Drift

最大 `|block q_top drift|`：

- `q=0.08`: 最大 `0.004438`
- `q=0.15`: 最大 `0.017630`
- `q=0.23`: 最大 `0.038191`

最大 block range：

- `q=0.08`: 最大 `0.008894`
- `q=0.15`: 最大 `0.038923`
- `q=0.23`: 最大 `0.095860`

较大的 block range 主要出现在 `q=0.23,L=3/4` 的少数 disorder，和这些点本身 disorder/sample 方差较大一致。`L5/L6` 高 q 的 drift/range 明显较小。

## Wall Time

每个 q 的 16 chunks 总 chunk time：

- `q=0.08`: `512.1s`
- `q=0.15`: `547.8s`
- `q=0.23`: `1318.0s`

每个 L 的平均 chunk time：

| q | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|
| 0.08 | 18.1s | 24.7s | 34.6s | 50.6s |
| 0.15 | 18.7s | 26.2s | 37.4s | 54.7s |
| 0.23 | 47.0s | 63.7s | 89.5s | 129.3s |

高 q 是生产成本主导，尤其 `L=6,q=0.23`。

## 判读

这轮结果支持当前配置进入正式生产扫描：

- high-q 和 low-q 的三初态 sector gate 已经通过。
- 小矩阵 `q_top(L,q)` 没有出现明显反物理趋势。
- `m=1024` 对大 L 高 q 看起来已足够稳定；小 L 高 q 主要受 disorder/sample 方差影响。
- 正式生产应优先增加 disorder 数，而不是继续为每个点跑三初态。

## 建议生产配置

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

正式扫描建议先用 `64` disorder/point 作为第一批 production，而不是直接上更大的 disorder 数。若 `q=0.23,L=3/4` 的误差仍过大，再局部补样本。
