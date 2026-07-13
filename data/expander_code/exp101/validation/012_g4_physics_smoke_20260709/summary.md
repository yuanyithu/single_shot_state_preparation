# G4.4 mini 端到端物理烟测

> **PRE_ALIGNMENT（scan/physics v1）：** 本页与 raw 产物只记录旧语义下的历史 smoke，不认证修正后的 reduced posterior、estimators 或 schema。见 `../README.md`。

墙钟 67s；文献 p_c(2D RBIM)=0.1094；精确枚举 q=0 true_posterior，CRN disorder×40

| 测试 | 尺寸 | crossing p_c | 相变端行为(低p大码↑/高p大码↓) | 结果 |
|---|---|---|---|---|
| toric_crossing | m2([[8,2,2]]) vs m3([[18,2,3]]) | 0.1329 (文献 0.1094) | ✅ | ✅ |
| surface_crossing | m3([[13,1,3]]) vs m4([[25,1,4]]) | 0.0688 (文献 0.1094) | ✅ | ✅ |
| expander_qtop_monotone | expander m2 q>0 | q_top 0.382→0.114 | 单调↓ | ✅ |

**crossing 括号文献值**：[0.133, 0.069] 包夹 p_c=0.1094。
两尺寸 crossing 对 d≤4 微型码有强有限尺寸效应（toric 高侧/surface 低侧，括号 0.109）；相变端行为（可恢复相/不可恢复相 × 码尺寸标度）干净正确 ⇒ **整条 model→HGP→logicals→observable→物理 链复现 2D 阈值物理**。
精确阈值需更大码 + FSS（用采样器，越枚举界；生产/分析后续）。

**总判定：ALL PASS ✅**
