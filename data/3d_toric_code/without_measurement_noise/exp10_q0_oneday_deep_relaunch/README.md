# exp10_q0_oneday_deep_relaunch

## 实验目的

重开一天内快速 deep 的 `q=0` 基线实验，修正此前错误启动导致的不可用结果。

## 为什么做

`q=0` 是后续 `q>0` threshold 搜索的 calibration。此前 run 被判定为错误后，本轮绕过旧 launcher，直接用不带任何 `--pt-*` 的 q=0 命令重开，并显式使用 `q0_num_start_chains=8` / `num_start_chains=8`。

## 参数

- `L = 3,4,5`
- `p = 0.205,0.210,0.215,0.220,0.225,0.230,0.235,0.240`
- `q = 0`
- `num_disorder_samples_total = 256`
- `chunk_size = 16`
- `q0_num_start_chains = 8`
- `num_burn_in_sweeps = 1000`
- `num_sweeps_between_measurements = 6`
- `num_measurements_per_disorder = 240`
- `common_random_disorder_across_p = true`

## 当前结论

`384/384` chunks 完成。本地 threshold 分析给出共同 interior crossing window：

- `p_min = 0.21457655911521925`
- `p_max = 0.23910998018700172`
- representative `p = 0.22684326965111049`

这次结果可作为当前 one-day deep 的 `q=0` 对照基线。

## 主产物

- `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch.npz`
- `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch.png`
- `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch_sem95.png`
- `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch_gap_crossing.png`
- `threshold_summary.json`
