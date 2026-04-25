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
- [`exp12_q005_fine_20260425_nd1`](with_measurement_noise/exp12_q005_fine_20260425_nd1/)：Numba 后的 `q=0.0050` fine run；`528/528` chunks 完成，自动推荐窗口 `p≈0.2164~0.2200`。
- [`exp13_q001_coarse_20260425_nd2`](with_measurement_noise/exp13_q001_coarse_20260425_nd2/)：`q=0.0100` coarse right-side check。结论是右侧大 `p` 已经偏过 crossing 区域。
- [`exp14_q001_fine_20260425_nd3`](with_measurement_noise/exp14_q001_fine_20260425_nd3/)：`q=0.0100` `0.225~0.270` fine scan。支持把后续算力移回 `p≈0.20~0.23`。
- [`exp15_q001_left_denseA_20260425_nd1`](with_measurement_noise/exp15_q001_left_denseA_20260425_nd1/) / [`exp16_q001_left_denseB_20260425_nd2`](with_measurement_noise/exp16_q001_left_denseB_20260425_nd2/)：`q=0.0100` 左侧 dense 独立复本，各 `64` disorder。
- [`exp17_q001_left_fine_20260425_nd3`](with_measurement_noise/exp17_q001_left_fine_20260425_nd3/)：`q=0.0100` 左侧 `0.0025` fine grid。
- [`exp18_q001_left_combined_summary`](with_measurement_noise/exp18_q001_left_combined_summary/)：`q=0.0100` 左侧综合分析；池化 `exp15+exp16` 后 `L3-L4` crossing 约 `p≈0.2233`，`L4-L5` 到 `p=0.230` 仍略负。
- [`exp19_q050_quick_p010_020_20260425_nd1`](with_measurement_noise/exp19_q050_quick_p010_020_20260425_nd1/)：`q=0.0500` 快速摸底；只作方向修正前后的侦察对照，不作为最终 threshold。
- [`exp20a_q050_heavy_p018_022_20260425_nd1`](with_measurement_noise/exp20a_q050_heavy_p018_022_20260425_nd1/) / [`exp20b_q050_heavy_p018_022_20260425_nd2`](with_measurement_noise/exp20b_q050_heavy_p018_022_20260425_nd2/) / [`exp20c_q050_heavy_p018_022_20260425_nd3`](with_measurement_noise/exp20c_q050_heavy_p018_022_20260425_nd3/)：`q=0.0500` 在 `p=0.18~0.22` 的三节点独立 seed 高力度复本，各 `96` disorder。
- [`exp21_q050_heavy_p018_022_combined_summary`](with_measurement_noise/exp21_q050_heavy_p018_022_combined_summary/)：`q=0.0500` 池化综合分析；`288` disorder 后 `L3-L4` crossing 约 `p≈0.193`，`L4-L5` 到 `p=0.22` 仍未 crossing，提示三尺寸共同 threshold 可能在 `p>0.22` 附近。
- [`exp22a_q050_L6_p018_022_20260425_nd1`](with_measurement_noise/exp22a_q050_L6_p018_022_20260425_nd1/) / [`exp22b_q050_L6_p018_022_20260425_nd2`](with_measurement_noise/exp22b_q050_L6_p018_022_20260425_nd2/) / [`exp22c_q050_L6_p018_022_20260425_nd3`](with_measurement_noise/exp22c_q050_L6_p018_022_20260425_nd3/)：`q=0.0500` 的 `L=6` extension，各 `96` disorder。
- [`exp23_q050_L3456_p018_022_combined_summary`](with_measurement_noise/exp23_q050_L3456_p018_022_combined_summary/)：`q=0.0500` 四尺寸综合图；`L=6` 全窗口低于 `L=5`，但 L=6 mixing 诊断较差，因此作为诊断延伸，不作最终 threshold。
