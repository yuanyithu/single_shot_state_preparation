# 010 local block summary smoke summary

共同参数：`lattice_size=3,p=0.05,q=0.08,q_hot=0.35,num_temperatures=3,num_start_chains=2,cluster_budget_fraction_rho=0.05`。

## 目标指标

| run | q_top | chain q_top | spread | Rhat | ESS | gate | block q_top | block range | last-half-full | wall s | ordinary | swap | observable | cluster |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| local_block_summary_smoke | 1.000000 | [1, 1] | 0.000000 | 1.000000 | 32 | fail | [1, 1, 1, 1] | 0.000000 | 0.000000 | 0.04125 | 0.0398 | 0.0002027 | 0.0006448 | 0 |

## 解释指标

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| local_block_summary_smoke | 2 | 1 | 1 | 32 | 1 | 0.023810 | [] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
