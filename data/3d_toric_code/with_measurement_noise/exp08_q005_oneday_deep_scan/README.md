# exp08_q005_oneday_deep_scan

- 实验目的：一天内快速 deep 定位 `q=0.005` threshold。
- 为什么做：`exp07` 方差太大，需要在关键区加密 `p`，但先把预算压到一天内测试可行性。
- 当前结论：`primary_crossing_window_hit=false`，convergence 仅 `3/21` 通过；不能宣称 threshold，只能作为失败诊断和补样本依据。
- 主看图：[q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random_gap_crossing.png](q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random_gap_crossing.png)
- 摘要：[q_0p0050/threshold_summary.json](q_0p0050/threshold_summary.json)
