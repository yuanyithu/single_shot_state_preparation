# STATUS.md — exp37 分步进度（执行 agent 接力用）

> 当前阶段 = 下表中第一个状态不是 `PASS` 的阶段，只在该阶段上工作。
> 状态取值：`TODO` / `DOING` / `PASS` / `FAIL`。判据细节见 `detail_plan.md`。
> 全部 PASS 且通过 Definition of Done 后，在本行改写为：`ALL DONE`。

总进度：ALL DONE

| Stage | 名称 | 状态 | 闸门达成数字（贴关键对比值） | 交付目录 | 更新日期 |
|-------|------|------|------------------------------|----------|----------|
| A | 模型对齐与解析锚点（不采样） | PASS | A1 ΔE=0; A2 logZ diff=4.44e-16; A3 max dw=0,max dq=0; A4 roundtrip=0 | `033_stageA_model_anchor_20260603/` | 2026-06-03 |
| B | 金标准参照表（照妖镜 benchmark） | PASS | B1 mid q_top=4/6; q_top=0.250,0.400,0.550,0.700,0.850,0.920; B2 max TV=2.09e-16,max dq=3.33e-16 | `034_stageB_exact_reference_20260603/` | 2026-06-03 |
| C | sector-constrained 采样器 | PASS | C1 violations=0; C2 max abs d_data=0.03075,max abs d_synd=0.03908, exact in 99% block CI for all sectors | `035_stageC_sector_sampler_20260603/` | 2026-06-03 |
| D | 主线估计量 sector-resolved TI | PASS | D1 max TV=0.00496; D2 max abs dq_top=0.00545, CI misses=0; D3 max grid TV=0.00434, max grid abs dq_top=0.00394 | `036_stageD_sector_ti_20260603/accepted_combined/` | 2026-06-03 |
| E | 第二独立方法交叉验证（退火+双向） | PASS | E1 max TV exact=1.26e-14, max abs dq_top=1.61e-14; E2 max TV TI=0.00496, max abs dq_top=0.00545; E3 max bidir gap=8.79e-14, max BAR residual=9.94e-15 | `037_stageE_bidirectional_bridge_20260603/` | 2026-06-03 |
| F | 尺度放大 + 失败地图（L=3,4,5×生产网格） | PASS | Full TI grid repaired with 16 targeted strong TI records; F1 PASS coverage=True, unresolved_tail=0, point statuses=PASS:10/WARN:38/FAIL:0; F2 PASS grid_fail=0; F3 PASS second_checks=3, failures=0, max second dq=0.01034, max TV=0.01286 | `038_stageF_ti_grid_20260603/accepted_repaired_ti_grid_targeted_strong_20260604/` | 2026-06-04 |
| G | 生产曲线 | PASS | G1 PASS max exact-TI abs dq=0.00544999,max TV=0.00495657,CI misses=0; G2 PASS final curve PASS-only, point statuses=PASS:10/WARN:38/FAIL:0, broad crossing claimed=False; G3 PASS q_top reconstruct max diff=0, disorder+TI uncertainty included; unresolved_tail_fail=False | `039_stageG_production_curve_20260604/` | 2026-06-04 |
