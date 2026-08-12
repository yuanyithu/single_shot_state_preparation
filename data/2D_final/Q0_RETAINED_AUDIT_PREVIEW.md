# `q=0` retained 数据审计预览

本预览只展示 `EVIDENCE.md` 授权进入正式重分析的两组 `q=0` 数据。它不是最终论文图，也没有使用任何 `q>0` legacy 数据。

## 输入与核验

| Run | Canonical source | NPZ SHA-256 | Manifest |
|---|---|---|---|
| `q0_threshold_deep_nd3_20260420_221142` | `data/2d_toric_code/without_measurement_noise/q0_threshold_deep_nd3_20260420_221142/scan_result_multi_L_q0_geometric_multistart_threshold_deep.npz` | `f3821be7f779119603f1464b9f201ece184106d515f9f19baf5fc2db9a5f4f61` | `720/720 completed` |
| `q0_control_extension_nd3_20260421_225303` | `data/2d_toric_code/without_measurement_noise/q0_control_extension_nd3_20260421_225303/scan_result_multi_L_q0_control_extension.npz` | `06254aa73b3e5c4596bdaf94d076e2c26c7427e43e8ba3b70789b49d199094ee` | `448/448 completed` |

绘图脚本在出图前强制核验 NPZ SHA-256、记录的 source SHA、manifest 完成状态、字段/shape、聚合均值与标准差，以及四起点 max-min spread。任何一项不符都会停止。

## 输出

- [`q0_retained_audit_preview.png`](q0_retained_audit_preview.png)：全部 `L={3,5,7,9,11}` 的 `q_top(p)` 与四起点 spread。
- [`q0_retained_large_L_gap_preview.png`](q0_retained_large_L_gap_preview.png)：`L={7,9,11}` 覆盖范围和相邻尺寸差值诊断。
- [`q0_retained_summary.csv`](q0_retained_summary.csv)：逐 `(L,p)` 的均值、标准差、SEM、95% CI 与四起点 spread。
- [`q0_retained_audit.json`](q0_retained_audit.json)：机器可读 provenance、覆盖缺口和相邻尺寸差值。

PNG 便于快速查看；对应 PDF 是同一脚本生成的矢量交付。误差带为逐点 disorder mean 的正态近似 95% CI。相邻尺寸来自不同 disorder ensemble，差值误差按独立样本传播。

## 当前可读结论

1. 现有 `q=0` 数据支持继续做正式重分析，但不能仅凭曲线视觉相交给出最终 threshold。
2. 在共同网格上，`q_top(L=9)-q_top(L=7)` 的点估计到 `p=0.1100` 仍为正，L7–L9 crossing 尚未被 bracket；置信区间也不支持把边界点解释成精确 crossing。
3. `L=11` 的平均四起点 spread 约为 `0.143–0.150`，明显大于较小尺寸。这不是 mixing 失败的单独证明，但正式拟合前必须保留并解释该诊断。
4. 最直接的 `q=0` 缺口仍是 `L={9,11}`、`p={0.1125,0.1150,0.1175,0.1200,0.1225,0.1250}`。

## 复现

从仓库根目录运行：

```bash
conda run -n 12 python data/2D_final/plot_q0_retained_audit.py
```

脚本只读取 legacy canonical source，并只写入 `data/2D_final/`。
