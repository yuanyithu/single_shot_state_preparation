# 009 cluster lightdiag target audit summary

目标导向审计：两个 m=2048 lightdiag seed 已同步到本地。

共同参数：`lattice_size=6,p=0.05,q=0.08,q_hot=0.35,num_temperatures=17,num_start_chains=4,cluster_budget_fraction_rho=0.15`。

## 目标指标

| run | q_top | chain q_top | spread | Rhat | ESS | gate | block q_top | block range | last-half-full | wall s | ordinary | swap | observable | cluster |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| run01_coldedge4_m2048_lightdiag_seed433000 | 1.000000 | [1, 1, 1, 1] | 0.000000 | 1.000000 | 2048 | pass | [] |  |  | 98.18 | 67.6 | 1.942 | 5.428 | 10.66 |
| run02_coldedge4_m2048_lightdiag_seed434000 | 1.000000 | [1, 1, 1, 1] | 0.000000 | 1.000000 | 2048 | pass | [] |  |  | 94 | 67.63 | 1.954 | 1.895 | 10.58 |

## 解释指标

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_coldedge4_m2048_lightdiag_seed433000 | 4 | 1 | 6 | 2048 | 1 | 0.135374 | [] | 325 | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 16/8 |
| run02_coldedge4_m2048_lightdiag_seed434000 | 4 | 1 | 6 | 2048 | 1 | 0.119267 | [] | 324 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

结论：

- 两个 009 seed 的最终 q_top、chain q_top、Rhat、ESS 全部呈现完美收敛形态，但这与 006-008 的共同冻结风险一致，不能单独视为热态可信。
- 009 lightdiag wall time 明显低于 full sector diagnostics，后续默认应关闭 full sector histogram，仅保留目标 block summary 和必要 cluster delivery 解释指标。
