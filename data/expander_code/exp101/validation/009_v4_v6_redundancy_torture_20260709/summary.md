# G3.6 V4 + G3.7 V6 结果

> **PRE_ALIGNMENT（scan/physics v1）：** 本页与 raw 产物只记录旧内核/冻结 gate 的历史实验；未覆盖 v2 四实例 PT 与 INVALID 传播，不是当前通过证据。见 `../README.md`。

墙钟 35s

| 测试 | 关键指标 | 结果 |
|---|---|---|
| V4 | ref≡numba=True, z(direct/pt/1v8)=0.1/2.5/0.1 | ✅ |
| V6_negative | expander_m2(k=4) 诊断报警=True, 共冻仅 transport 失败=['sector_transport_insufficient'] | ✅ |
| V6_negative | expander_m3(k=9) 诊断报警=True, 共冻仅 transport 失败=['sector_transport_insufficient'] | ✅ |
| V6_positive | round_trips=19/15, z(初始 sector 无关)=1.7 | ✅ |

**总判定：ALL PASS ✅**
