# exp11_q001_oneday_deep_partial

## 实验目的

回收被主动停止的 `q=0.0100` 一天内快速 deep run 的已有数据，并确认它是否足以给出 threshold 线索。

## 为什么做

`q=0.0100` 的 `L=5` chunk 运行时间异常长。为避免继续消耗服务器时间，本轮停止 nd-3 上的任务，只下载已完成的 chunk 做 partial 分析。

## 参数

- `L = 3,4,5` 为原计划
- 实际完成：仅 `L=3,4`
- `p = 0.170,0.180,0.190,0.200,0.210,0.220,0.230`
- `q = 0.0100`
- `num_disorder_samples_total = 6`
- `chunk_size = 3`
- `num_start_chains = 8`
- `num_replicas_per_start = 2`
- `pt_p_hot = 0.44`
- `pt_num_temperatures = 9`
- `num_burn_in_sweeps = 1000`
- `num_sweeps_between_measurements = 6`
- `num_measurements_per_disorder = 3072`

## 当前结论

只有 `28/42` chunks 完成，其中 `L=3,4` 完成，`L=5` 的 `14` 个正式 chunks 全部缺失。因此本目录只包含 partial L3-L4 分析，不能作为三尺寸 threshold 结论。

L3-L4 gap 在 `p=0.17~0.23` 全部为负，且 CI 很宽：

```text
gap L3-L4 = [-0.19734, -0.23532, -0.33812, -0.30559, -0.20737, -0.05449, -0.00347]
```

同时 `L=4` 高 `p` 端多个点 convergence gate 失败。本轮只说明：当前预算下 `q=0.0100` 的 `L=5` 太慢，且已有 L3-L4 数据不足以定位 threshold。

## 主产物

- `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4.npz`
- `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4.png`
- `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4_sem95.png`
- `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4_gap.png`
- `partial_L3_L4_analysis_summary.json`
