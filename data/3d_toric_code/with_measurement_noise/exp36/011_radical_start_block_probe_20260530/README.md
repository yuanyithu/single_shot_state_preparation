# 011 radical-start block probe

日期：2026-05-30

目标：用最小额外机器时间检验 009 候选配置是否只是共同冻结伪装收敛。固定同一个 disorder realization，在 `L=6,p=0.05,q=0.08` 下比较三种 q>0 初态模式的 `q_top` block/window 收敛。

## 设计

共同参数：

- `K=17,q_hot=0.35,cluster rho=0.15,cold_edge_stride=4`
- `num_measurements_per_disorder=1024`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150,max_effective_num_burn_in_sweeps=750`
- `num_start_chains=4,num_replicas_per_start=1`
- `observable_temperature_mode=cold`
- `q_top_block_count=8`
- `adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`
- 关闭 full sector histogram，只保留 cluster-sector light diagnostics

三条 run 使用同一个 `seed_base=435000`，目的是固定 disorder realization；run 间只改变 `q_positive_initial_chain_mode`。

| run | node | initial mode | purpose |
|---|---|---|---|
| `run01_sector_m1024_seed435000` | `nd-1` | `sector` | 现有 start-sector 机制 |
| `run02_allzero_m1024_seed435000` | `nd-2` | `all_zero` | 极端低权重初态 |
| `run03_randomhigh_m1024_seed435000` | `nd-3` | `random_high_weight` | 极端高权重随机初态 |

## 判据

- 若三种初态的最终 `q_top` 或末 block `q_top` 差异大于 `0.02`，当前候选不能视为热化。
- 若任一 run 的 `q_top` block drift/range 呈系统性漂移，当前候选不能进入生产扫描。
- 只有三种初态收敛到同一 `q_top`，且 `Rhat<=1.05, ESS>=100, spread<=0.02`，才进入共同 disorder A/B。

## 操作

远端 source：

`/home/DATA1/users/yuany/.single_shot/repos/011_radical_start_block_probe_20260530/source`

远端 run base：

`/home/DATA1/users/yuany/.single_shot/exp36/011_radical_start_block_probe_20260530`

本轮使用本地 worktree snapshot，同步时排除历史 `data/` 和 `.git/`。`git_commit_sha` 记录当前 HEAD；源码另含未提交的 exp36 block summary/initial-mode 改动。
