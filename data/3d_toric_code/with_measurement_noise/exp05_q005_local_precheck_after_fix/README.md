# exp05_q005_local_precheck_after_fix

- 实验目的：修复后本地轻量预检 `q=0.005` 的 3D `q>0` 生产路径。
- 为什么做：旧 `q>0` 远端 Stage A 结果不可信，先在本地确认 PT/multi-start 路径和诊断字段能正常落盘。
- 当前结论：只作为管线校验和排错依据，不作为正式 threshold 结论。
- 主看图：[q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_gap_crossing.png](q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_gap_crossing.png)
- 主结果 npz：[q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random.npz](q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random.npz)
