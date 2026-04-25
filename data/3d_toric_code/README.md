# 3D Toric Code Data

3D 实验目录统一使用 `expNN_做什么` 命名。`NN` 表示实验推进顺序，不表示物理优先级。看结果时先读本文件，再进入具体实验目录的 `README.md`。

## No Measurement Noise (`q=0`)

- [`exp01_q0_pipeline_smoke`](without_measurement_noise/exp01_q0_pipeline_smoke/)：生产路径 smoke。目的只是确认 3D `q=0` 管线能跑通，不用于 threshold 结论。
- [`exp02_q0_low_p_scout`](without_measurement_noise/exp02_q0_low_p_scout/)：首轮低 `p` scout。结论是 `p<=0.12` 主要是饱和平台伪 crossing，需要右移窗口。
- [`exp03_q0_right_shift_scout`](without_measurement_noise/exp03_q0_right_shift_scout/)：右移到 `p=0.10~0.20`。结论是仍未覆盖可信 crossing，需要继续右移。
- [`exp04_q0_crossing_window_scout`](without_measurement_noise/exp04_q0_crossing_window_scout/)：`q=0` crossing calibration。结论是在 `p≈0.218~0.230` 首次看到可信 interior crossing；后续 `q=0` 对照优先参考这次。
- [`exp09_q0_oneday_deep_fixed`](without_measurement_noise/exp09_q0_oneday_deep_fixed/)：一天内快速 deep 的 `q=0` 修正版。用于和同批 `q>0` one-day run 对照；需结合 `threshold_summary.json` 判断是否优于 `exp04`。
- [`exp10_q0_oneday_deep_relaunch`](without_measurement_noise/exp10_q0_oneday_deep_relaunch/)：停掉错误 q=0 run 后重开的修正版。`384/384` chunks 完成，给出共同 crossing window `p≈0.2146~0.2391`，代表点 `p≈0.2268`。

## With Measurement Noise (`q>0`)

- [`exp05_q005_local_precheck_after_fix`](with_measurement_noise/exp05_q005_local_precheck_after_fix/)：修复后本地轻量预检。目的为验证 `q>0` PT/multi-start 生产路径能落盘并给出诊断，不作为正式物理结论。
- [`exp06_zero_disorder_quick_scan`](with_measurement_noise/exp06_zero_disorder_quick_scan/)：全零 disorder 单样本快速扫描。目的为快速看截面形状；无 disorder average，不能和正式阈值结果等同。
- [`exp07_q005_broad_scan`](with_measurement_noise/exp07_q005_broad_scan/)：`q=0.005` disorder-averaged PT broad scan。结论是 `17/18` convergence gate 通过，gap 指示 `p≈0.2077`，但只有 `4` disorder，CI 较宽。
- [`exp08_q005_oneday_deep_scan`](with_measurement_noise/exp08_q005_oneday_deep_scan/)：一天内快速 deep 的 `q=0.005` run。结论是 `primary_crossing_window_hit=false`，convergence 仅 `3/21` 通过；不能宣称 threshold，只能作为失败诊断和补样本依据。
- [`exp11_q001_oneday_deep_partial`](with_measurement_noise/exp11_q001_oneday_deep_partial/)：停止过慢 `q=0.0100` one-day deep 后回收的 partial 数据。只有 `L=3,4` 完成，`L=5` 缺失；已生成 L3-L4 partial 图，但不能作为三尺寸 threshold。
