# G3.5 V3 Nishimori 恒等式结果

> **PRE_ALIGNMENT（scan/physics v1）：** 本页与 raw 产物只记录旧接线下的历史实验；runner 可能依赖弃用字段、估计器或 schema，不是 `exp101.physics.v2` 当前通过证据。见 `../README.md`。

墙钟 31s

| 级 | 实例 | (p,q) | 关键指标 | 结果 |
|---|---|---|---|---|
| L1 | toric_m2 | (0.08,0.05) | max\|E[m]-E[m²]\|=1.87e-14 | ✅ |
| L1 | toric_m2 | (0.15,0.1) | max\|E[m]-E[m²]\|=6.16e-15 | ✅ |
| L1 | toric_m2 | (0.2,0.15) | max\|E[m]-E[m²]\|=5.72e-15 | ✅ |
| L2 | toric_m3 | (0.1,0.06) | max\|z\|=1.10, max\|diff\|=0.0227 | ✅ |
| L2 | K43 | (0.1,0.06) | max\|z\|=3.53, max\|diff\|=0.0732 | ✅ |
| L3 | expander_m2 | (0.08,0.05) | max\|z\|=2.06, max\|diff\|=0.0439 | ✅ |
| JUDGE | toric_m2_repo_compat | (0.15,0.5) | identity gap=0.250（期望违反） | ✅ |

**总判定：ALL PASS ✅**
