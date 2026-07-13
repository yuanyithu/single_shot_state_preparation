# V1 主矩阵 权威 gate（regime-aware，统计正确 instrument）

> **PRE_ALIGNMENT（scan/physics v1）：** 本页与 raw 产物只记录 2026-07-09 前的历史内部一致性；runner 可能依赖弃用字段、估计器或 schema，不是 `exp101.physics.v2` 当前通过证据。见 `../README.md`。

run_v1.py 产出有效逐任务采样；本表用正确统计工具重聚合（见 finalize_v1.py 头注）。
direct: 严格判定(well-mixed wacc≥0.05) 256 / 边缘报告 87 / 冻结披露 77（后两者由 TI 覆盖）。

| regime | 指标 | 值 | 阈值 | 结果 |
|---|---|---|---|---|
| A direct | 逐任务偏差 grand±se | -0.008±0.041 | \|grand\|≤3se | ✅ |
| A direct | discrepant / tvd_max / 能量失败 | 0.0010 / 0.041 / 0 | ≤0.005 / ≤0.05 / 0 | ✅ |
| PT 冷点 | 逐任务偏差 grand±se | +0.071±0.099 | \|grand\|≤3se | ✅ |
| PT 冷点 | discrepant / tvd_max / 全往返>0 | 0.0013 / 0.028 / True | ≤0.005 / ≤0.05 / True | ✅ |
| TI full | 未 flag 点 q_top/TVD 失败（flag 点诊断捕获=24） | 0/0 | 0/0 | ✅ |

### 大 k（K43）direct 采样 vs 精确（per-instance 明细）
| 实例 | well 任务 | 平均任务 z | tvd_max |
|---|---|---|---|
| K43 | 29 | -0.033 | — |
| irregular_2x4 | 25 | -0.098 | 0.028 |
| surface_m3 | 38 | -0.053 | 0.015 |
| toric_m2 | 123 | +0.032 | 0.041 |
| toric_m3 | 41 | -0.016 | 0.037 |

### pairwise-TI 弃用（status D4；证据 validation/007）
- K43 pairwise vs 精确 m_u：max 1.547（对照 direct vs 精确 max 0.786）
- toric_m3(k=2) pairwise vs 精确 max 0.110（full-TI vs 精确 max 0.032）
- 结论：pairwise 假可加性 → 失效；大 k q_top 走 direct/PT 采样。

**总判定：ALL PASS ✅**
