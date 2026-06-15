# 012 radical-start high-q probe

日期：2026-05-30

目标：011 的 `q=0.08` radical-start 结果完全饱和为 `q_top=1`，无法充分检验非平凡热态采样。012 在同一候选配置下只把目标点改为 `q=0.23`，重复三初态比较，检查高 q 端是否存在初态依赖或 block drift。

## 设计

共同参数：

- `L=6,p=0.05,q=0.23`
- `K=17,q_hot=0.35,cluster rho=0.15,cold_edge_stride=4`
- `num_measurements_per_disorder=1024`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150,max_effective_num_burn_in_sweeps=750`
- `num_start_chains=4,num_replicas_per_start=1`
- `observable_temperature_mode=cold`
- `q_top_block_count=8`
- `adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`
- 关闭 full sector histogram，只保留 cluster-sector light diagnostics

三条 run 使用同一个 `seed_base=436000`，固定同一个 disorder realization；run 间只改变 `q_positive_initial_chain_mode`。

| run | node | initial mode | purpose |
|---|---|---|---|
| `run01_sector_q023_m1024_seed436000` | `nd-1` | `sector` | zero-syndrome sector representatives |
| `run02_allzero_q023_m1024_seed436000` | `nd-2` | `all_zero` | 极端低权重初态 |
| `run03_randomhigh_q023_m1024_seed436000` | `nd-3` | `random_high_weight` | 极端高权重随机初态 |

## 判据

- 若三种初态的最终 `q_top` 或末 block `q_top` 差异大于 `0.02`，当前候选不能视为高 q 端热化。
- 若任一 run 的 `q_top` block drift/range 呈系统性漂移，当前候选不能进入生产扫描。
- 若三初态仍完全一致，再进入 common-disorder A/B，并把 q=0.08 的饱和性作为判读限制记录。

## 路径

远端 source：

`/home/DATA1/users/yuany/.single_shot/repos/012_radical_start_highq_probe_20260530/source`

远端 run base：

`/home/DATA1/users/yuany/.single_shot/exp36/012_radical_start_highq_probe_20260530`

