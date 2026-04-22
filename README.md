# single_shot_state_preparation

本仓库当前把实验结果按 `2D/3D` 与 `measurement noise` 是否存在重组到 `data/` 下，Python 源码统一放在 [src/](src/README.md) 下，完整记录见 [实验报告_2D_toric_code.md](实验报告_2D_toric_code.md) 和 [实验报告_3D_toric_code.md](实验报告_3D_toric_code.md)。

## 2D / No Measurement Noise

- `baseline_multisize_local`：最早的 2D `q=0` 多尺寸基线，未见干净 crossing；看 [scan_result_multi_L.png](data/2d_toric_code/without_measurement_noise/baseline_multisize_local/scan_result_multi_L.png)。
- `kernel_mix_local`：随机闭环 proposal 改善了 `q=0` 混合，但 crossing 仍不够尖锐；看 [scan_result_multi_L_kernel_mix_focus.png](data/2d_toric_code/without_measurement_noise/kernel_mix_local/scan_result_multi_L_kernel_mix_focus.png)。
- `q0_geometric_multistart_local`：几何 multistart sampler 明显优于旧 kernel-basis 路径；看 [scan_result_multi_L_q0_geometric_multistart.png](data/2d_toric_code/without_measurement_noise/q0_geometric_multistart_local/scan_result_multi_L_q0_geometric_multistart.png)。
- `q0_threshold_deep_nd3_20260420_221142`：大样本 deep run 把 `q=0` crossing 压到 `p≈0.10~0.106`；看 [scan_result_multi_L_q0_geometric_multistart_threshold_deep_sem95.png](data/2d_toric_code/without_measurement_noise/q0_threshold_deep_nd3_20260420_221142/scan_result_multi_L_q0_geometric_multistart_threshold_deep_sem95.png)。
- `q0_control_extension_nd3_20260421_225303` / `q0_control_summary_20260422`：`L=9,11` 对照补跑支持 `q=0` 有有限 threshold，但 `L7-L9` 还没完全闭合；看 [q0_control_sem95.png](data/2d_toric_code/without_measurement_noise/q0_control_summary_20260422/q0_control_sem95.png)。

## 2D / With Measurement Noise

- `measurement_noise_overnight_nd3_20260421_004035`：先确认 `q=0.01,0.02,0.03` 的 crossing 已整体左移出原窗口；看 [measurement_noise_q_scan_sem95_overview.png](data/2d_toric_code/with_measurement_noise/measurement_noise_overnight_nd3_20260421_004035/measurement_noise_q_scan_sem95_overview.png)。
- `measurement_noise_threshold_search_nd3_20260421_104427`：按 `q` 左移窗口后，仍未出现稳定三尺寸共同 crossing；看 [measurement_noise_threshold_search_gap_summary.png](data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/measurement_noise_threshold_search_gap_summary.png)。
- `no_threshold_final_nd3_20260421_225039`：最终大尺寸主任务目前只完整产出 `q=0.0010` 主结果；主结果在 `data/2d_toric_code/with_measurement_noise/no_threshold_final_nd3_20260421_225039/q_0p0010/`。
- `no_threshold_evidence_nd3_20260422`：综合证据支持“固定非零 `q` 下未观察到稳定有限 threshold”；看 [q_positive_pseudocritical_drift.png](data/2d_toric_code/with_measurement_noise/no_threshold_evidence_nd3_20260422/q_positive_pseudocritical_drift.png)。

## 3D / No Measurement Noise

- `q0_smoke_local_20260421_234202`：3D `q=0` 生产路径 smoke，只验证管线与张量 shape；看 [scan_result_multi_L_3d_toric_q0_smoke.png](data/3d_toric_code/without_measurement_noise/q0_smoke_local_20260421_234202/scan_result_multi_L_3d_toric_q0_smoke.png)。
- `q0_threshold_scout_nd1_20260421_235447`：首轮 scout 证明 `p≤0.12` 只包含饱和平台伪 crossing；看 [scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)。
- `q0_threshold_scout_stageA_nd1_20260422_100557`：窗口右移到 `0.10~0.20` 仍未见 crossing；看 [scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/q0_threshold_scout_stageA_nd1_20260422_100557/local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)。
- `q0_threshold_scout_stageB_nd3_20260422_101018`：在 `p≈0.218~0.230` 首次看到可信 interior crossing；看 [scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/q0_threshold_scout_stageB_nd3_20260422_101018/local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)。

## 3D / With Measurement Noise

- 暂无实验。

更细的数据索引见 [data/README.md](data/README.md)。
