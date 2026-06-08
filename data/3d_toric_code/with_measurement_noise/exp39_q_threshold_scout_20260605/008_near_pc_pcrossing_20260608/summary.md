# exp39/008 — 近 p_c 相边界探测（小 q 的 p-crossing）

**目标**：007 的 Δf 相边界在 p∈[0.02,0.20] 近乎平坦(q_c≈0.05)，向 p_c 的陡降段未解析。本实验固定小 q∈{0.01,0.02,0.03,0.04}、扫 p∈{0.20,0.21,0.22,0.23,0.24,0.25}、L=3,4,5，用 **Δf 的 p-crossing**（有序侧 p<p_c 大 L 更高、无序侧反序）定位 p_c(q)，以补全相边界靠近 q=0 端点 p_c≈0.227 的转弯。

**方法**：6 个 p 共用 `seed_base=840000`（common random number disorder 跨 p，48 disorder），grid129/m8192/burn512，TI / `projection_mode=linear`。p-crossing 用 **paired bootstrap**（一次重采样 disorder index、施加到所有 p，利用 common disorder 降方差）。node↔p：nd-1{0.20,0.23} / nd-2{0.21,0.24} / nd-3{0.22,0.25}，workers=76。run `exp39_boundary_20260608_221720`。

## 结果：整个区域都「有序」，无 crossing

对 **全部** q∈{0.01..0.04} 和 **全部** p∈{0.20..0.25}，Δf gap **L5>L4>L3、且 L-间距基本保持**（深在有序相）。例 q=0.01：L3/L4/L5 在 p=0.20 为 11.3/20.6/32.7，p=0.25 仍为 8.5/15.7/25.0（L5≈3×L3）。**没有 p-crossing**：四个 q 全部 `status=all_ordered(p_c>0.25)`、boot_frac=0。q_top 在此区饱和到 ~0.92–1.0、且噪声大，也给不出干净 crossing。

**核心结论**：**L=3,4,5 的 Δf 有限尺寸「有序区」一直延伸到 p=0.25，越过了渐近阈值 p_c≈0.227**（图中绿圈覆盖 p=0.23/0.24/0.25，在 p_c 竖线右侧）。即**小 L 的 Δf crossing 系统性 OVERESTIMATES 阈值**——与 007 一致（007 的 q_c≈0.05 也高于 q_top 的 ≈0.03）。要看到 p_c≈0.227 处真正的转弯/陡降，需要**有限尺寸标度(FSS)外推**（更大 L，如 L=6,7,8），小 L 的单点 crossing 看不到。

**对 007 边界的含义**：007 的平坦 q_c≈0.05–0.06 是**有限尺寸高估**；真边界更低（q_top 给 ≈0.03，更接近真值但饱和）。边界**形状**（平坦体 + 近 p_c 收口）定性正确，**绝对数值偏高**。Δf 的优点是有序侧不饱和、crossing 干净；代价是它把阈值往高估——这两点要一起讲。

## 产物
- `near_pc_pcrossings.png` — 每个 q 的 Δf-vs-p（L=3,4,5），全程有序、无 crossing
- `near_pc_overview.png` — (p,q) 平面：007 平坦边界 + 008 网格全「Δf-ordered」绿圈 + 渐近 p_c≈0.227 竖线（直观展示有限尺寸有序区越过 p_c）
- `near_pc_summary.json`、`analyze_near_pc.py`（复用 007 的 `launch_boundary_remote.sh`）

## 可选下一步
1. **FSS（推荐，较贵）**：在 p_c 附近(p≈0.20–0.25)+小 q 加跑 L=6（、7），对 crossing 做 1/L 外推得真 p_c(q)；同样可对 007 各 p 加 L=6 把 q_c(p) 往真值修正。
2. **延伸 p（便宜但仍高估）**：把 p 扫到 ~0.30–0.35，找到 L=3,4,5 的有限尺寸 p-crossing 落点（量化高估幅度）。
3. 接受现状：把 007/008 的有限尺寸边界当作真边界的**上界**报告。
