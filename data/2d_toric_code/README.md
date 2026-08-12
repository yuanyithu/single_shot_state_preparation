# 2D Toric Code Legacy Source Index

本目录是历史输入索引，不是当前接手入口，也不直接授予论文证据权限。当前状态、筛选理由和最小缺口以 [`data/2D_final/EVIDENCE.md`](../2D_final/EVIDENCE.md) 为准；旧 run README 中的结论只代表生成时的判断。

## `q=0` 历史输入

- [`baseline_multisize_local`](without_measurement_noise/baseline_multisize_local/README.md)：早期多尺寸基线；`ARCHIVE_ONLY`。
- [`kernel_mix_local`](without_measurement_noise/kernel_mix_local/README.md)：kernel-mix 方法开发；`ARCHIVE_ONLY`。
- [`q0_geometric_multistart_local`](without_measurement_noise/q0_geometric_multistart_local/README.md)：几何四起点原型；`ARCHIVE_ONLY`。
- [`q0_threshold_deep_nd3_20260420_221142`](without_measurement_noise/q0_threshold_deep_nd3_20260420_221142/README.md)：`L=3,5,7` deep run；`RETAIN_FOR_FORMAL_REANALYSIS`。
- [`q0_control_extension_nd3_20260421_225303`](without_measurement_noise/q0_control_extension_nd3_20260421_225303/README.md)：`L=9,11` control extension；`RETAIN_FOR_FORMAL_REANALYSIS`。
- [`q0_control_summary_20260422`](without_measurement_noise/q0_control_summary_20260422/README.md)：旧派生图；`ARCHIVE_ONLY`。

## `q>0` 历史输入

- [`measurement_noise_overnight_nd3_20260421_004035`](with_measurement_noise/measurement_noise_overnight_nd3_20260421_004035/README.md)：被后续选窗覆盖；`ARCHIVE_ONLY`。
- [`measurement_noise_threshold_search_nd3_20260421_104427`](with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/README.md)：六个 `q` 的旧 threshold-search；`LEGACY_VALIDATION_REQUIRED`。
- [`no_threshold_final_nd3_20260421_225039`](with_measurement_noise/no_threshold_final_nd3_20260421_225039/README.md)：`q=0.001` 大尺寸旧数据；`LEGACY_VALIDATION_REQUIRED`。
- [`no_threshold_evidence_nd3_20260422`](with_measurement_noise/no_threshold_evidence_nd3_20260422/README.md)：旧图、表和重复 NPZ；`ARCHIVE_ONLY`。

历史目录名中的 “no threshold” 不是已证明的物理结论。旧机器摘要本身给出 `paper_claim_supported=false`，而正式 `q>0` 权限还受 sampler freezing 与 logical transport 未认证所阻断。
