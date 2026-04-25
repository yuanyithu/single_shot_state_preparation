# 3D Toric Code Without Measurement Noise

- [`exp01_q0_pipeline_smoke`](exp01_q0_pipeline_smoke/)：生产路径 smoke，只验证可运行性。
- [`exp02_q0_low_p_scout`](exp02_q0_low_p_scout/)：低 `p` 首轮 scout，排除 `p<=0.12` 真实 crossing。
- [`exp03_q0_right_shift_scout`](exp03_q0_right_shift_scout/)：右移到 `p=0.10~0.20`，仍未覆盖 crossing。
- [`exp04_q0_crossing_window_scout`](exp04_q0_crossing_window_scout/)：当前 `q=0` baseline，在 `p≈0.218~0.230` 看到可信 crossing。
- [`exp09_q0_oneday_deep_fixed`](exp09_q0_oneday_deep_fixed/)：一天内快速 deep 修正版，主要作为 one-day 对照。
- [`exp10_q0_oneday_deep_relaunch`](exp10_q0_oneday_deep_relaunch/)：重新启动的 `q=0` one-day deep 修正版，`384/384` chunks 完成；代表 crossing `p≈0.2268`。
