# G3.4 V2 解析极限结果

> **PRE_ALIGNMENT（scan/physics v1）：** 标题中的 V2 早于 `exp101.physics.v2`；本页与 raw 产物只记录历史内部一致性，runner 可能依赖弃用字段、估计器或 schema。见 `../README.md`。

| check | 实例/参数 | 关键指标 | 结果 |
|---|---|---|---|
| V2a | toric_m3 | max_abs_z=0.8698, logical_acceptance_all_one=True | ✅ |
| V2a | expander_m2 | max_abs_z=2.226, logical_acceptance_all_one=True | ✅ |
| V2b | expander_m2 | p=0.05, n=100, k=4, max_abs_z=2.417, energy_z=-2.356 | ✅ |
| V2b | expander_m4 | p=0.01, n=400, k=16, max_abs_z=1.71, energy_z=0.0914 | ✅ |
| V2b | expander_m6 | p=0.002, n=900, k=36, max_abs_z=1.996, energy_z=0.6533 | ✅ |
| V2c | — | engine_z_qzero=0.7868, engine_z_qtiny=0.6214, exact_continuity_diff=1.776e-05 | ✅ |
| V2d | — | exact_q_top=0.9947, max_abs_z=0.2683 | ✅ |
| V2e | — | ensembles_identical_at_zero_disorder=True, max_abs_z=1.688 | ✅ |

**总判定：ALL PASS ✅**
墙钟 1s
