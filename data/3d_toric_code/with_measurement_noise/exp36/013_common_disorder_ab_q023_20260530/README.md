# 013 common-disorder A/B q=0.23

日期：2026-05-30

目标：011/012 radical-start 没有发现 `q_top` 初态依赖后，开始小型 common-disorder A/B。只做 `L=6,p=0.05,q=0.23`，用 3 个共同 disorder 比较当前候选、便宜基线和 2x 长链参考，判断当前候选的额外机器时间是否换来更可信的 `q_top`。

## 共同设置

- `L=6,p=0.05,q=0.23`
- `num_disorder_samples_total=3,chunk_size=1,workers=1`
- `num_start_chains=4,num_replicas_per_start=1`
- `num_burn_in_sweeps=150,max_effective_num_burn_in_sweeps=750`
- `num_sweeps_between_measurements=6`
- `observable_temperature_mode=cold`
- `q_positive_initial_chain_mode=sector`
- `q_top_block_count=8`
- 不开 full sector histogram，也不开 cluster-sector light diagnostics；本轮优先比较目标量和 wall time

三条 run 使用同一个 `seed_base=437000`，固定共同 disorder sequence。

## 配置

| run | node | config | measurements | purpose |
|---|---|---|---:|---|
| `run01_candidate_qhot035_rho015_coldedge4_m1024_seed437000` | `nd-1` | `q_hot=0.35,rho=0.15,cold_edge_stride=4` | 1024 | 当前候选 |
| `run02_cheap_qhot032_nocluster_m1024_seed437000` | `nd-2` | `q_hot=0.32,cluster off,cold_edge_stride=1` | 1024 | 便宜基线 |
| `run03_reference_qhot035_rho015_coldedge4_m2048_seed437000` | `nd-3` | 当前候选 2x measurements | 2048 | 近似长链参考 |

## 判据

- 若候选与便宜基线在共同 disorder 上的 `q_top`、block drift、spread/Rhat/ESS 基本一致，优先选择便宜基线。
- 若候选更接近 2x 参考且 block drift 更小，再保留候选。
- 若 2x 参考自身仍有明显 drift，则不应扩大生产扫描，应回到 proposal/kernel 设计。

远端 source：

`/home/DATA1/users/yuany/.single_shot/repos/013_common_disorder_ab_q023_20260530/source`

远端 run base：

`/home/DATA1/users/yuany/.single_shot/exp36/013_common_disorder_ab_q023_20260530`

