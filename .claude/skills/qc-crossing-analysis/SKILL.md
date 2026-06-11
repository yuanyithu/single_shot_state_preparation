---
name: qc-crossing-analysis
description: threshold/相边界 crossing 分析规范(估计量选择、观测量路径、crossing 检测陷阱、bootstrap 约定)。当需要从 sector_ti_results.npz 计算 q_c/p_c、画 crossing 图、或评审已有 crossing 结论时使用。
---

# threshold / crossing 分析规范

参考实现:`exp40_qtop_phase_boundary_20260610/003_boundary_analysis_20260610/analyze_exp40_boundary.py`(多 p 边界)与 `004_p011_highstats_20260611/analyze_p011_highstats.py`(单点高统计 + disorder 块合并)。新分析**复制改造**,不要从零写。

## 估计量(选错 = 结论作废)

- **主估计量 q_top = ⟨m²⟩,交叉验证 w0 = P(真逻辑类) = softmax(−delta_f)[...,0]**。两者的有限尺寸 crossing 都收敛到真阈值(exp39/008 FSS 验证:w0→0.254 最准,q_top→0.268,真值 0.233)。
- **绝不用 Δf-gap crossing**(even-moment、∝L² 畴壁,系统性偏高 → 外推到 0.40 而非 0.233);**绝不用 Binder cumulant**(纯相内恒 2/3,无分辨力)。
- 近 p_c 区(p≳0.17)q_top 的 L 对 crossing 涨落大,**以 w0 为准**。
- 数据必须来自 `exp37_sector_ti.py run --projection-mode linear`(TI,正确观测量);**绝不用 `ais`**(硬编码 decoder_reject = 旧 buggy `x+r(Hx)` 标签)。

## crossing 检测陷阱

- **饱和零差值假 crossing**:低 q 端各 L 曲线完全饱和(q_top≡1),差值恰为 0;天真的 `dvec[i]==0 → crossing` 会把 q_c 假性钉在网格首点(症状:CI 收成 `[q0,q0]`、boot_frac=1.0)。零差值只在两侧异号时才计入(用参考实现里修正版 `cross_q`)。
- **网格必须把 crossing 围在内部**:点估计或 CI 端点贴着网格边缘(如 exp39/007 全部顶在下边缘)= 网格设计错,结论不可用,重设网格重跑;不要靠外推救。
- 多 L 对(L3-L4 / L3-L5 / L4-L5)都算,headline 用 L3-L5;三对应互相一致,严重分歧(尤其近 p_c)要如实标注。

## 统计约定

- disorder bootstrap(成对重采样 disorder 索引,两条 L 曲线用同一组索引),生产 N≥6000;**必须报告 boot_frac**(bootstrap 样本中找到 crossing 的比例),boot_frac<0.9 说明该点统计不足。
- 误差棒 = disorder SEM(`std/sqrt(ndis)`);MCMC 内部误差(blocks/TI stderr)远小于 disorder 涨落,disorder 数是约束(48 太少,256–384 才能把 crossing CI 收到 ~0.008)。
- 多 seed 块合并:同一 (p, q grid) 的不同 seed_base 块沿 disorder 轴 concat(校验 q grid / L list 完全一致);不同 q grid 的源沿 q 轴合并去重(`load_source` 模式)。
- 物理 sanity:q_top/w0 随 q 单调不增;无序侧大 L 更低(扇形展开);p < p_c≈0.227 时 crossing 才有意义,p=0.22 等近临界点只作定性收口。
- 与已知锚点对照:q=0 端点 p_c≈0.227;exp40 平坦带 q_c≈0.03(p∈[0.01,0.20]);exp39/exp40 重叠点(p=0.14: w0≈0.034)。新结果与锚点矛盾时先查流程再谈物理。
