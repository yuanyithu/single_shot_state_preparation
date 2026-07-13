# G3.3 V1c section-frame A/B 结果

> **PRE_ALIGNMENT（scan/physics v1）：** 本页与 raw 产物只记录历史 frame 实验；runner 可能依赖弃用字段、估计器或 schema，不是 `exp101.physics.v2` 当前 section-invariance 证据。见 `../README.md`。

墙钟 18s

| gate | 内容 | 值 | 结果 |
|---|---|---|---|
| G1 | 每 frame 内 枚举=MCMC (max z) | 4.54 (≤5) | ✅ |
| G2 | q=0 相对分布 frame 无关（含 decoder，max rel-TVD） | 1.14e-15 (≤1e-9) | ✅ |
| G3 | q>0 frame 依赖被观测（max A/B rel-TVD） | 0.6025 (>1e-3) | ✅ |
| G4 | 三 frame 指纹互异 | — | ✅ |

**总判定：ALL PASS ✅**
