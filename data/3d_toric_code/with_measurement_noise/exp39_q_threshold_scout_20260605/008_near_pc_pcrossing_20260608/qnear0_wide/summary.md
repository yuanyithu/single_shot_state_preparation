# q≈0 大尺寸 FSS + threshold estimator 选择（关键纠正）

**目的**：q≈0（q=0.002 代理，q=0 因 K_q=log((1-q)/q) 发散不被支持）扫 p∈{0.20..0.40}、L=3,4,5,6，用 FSS 判断 Δf-gap crossing 是不是收敛到已知真值 p_c≈0.233（3D RBIM Nishimori），并比较其它 estimator。共用 seed 840000 / scope disorder_index（disorder 跨 L 共用，可直接 FSS）。L=6 run `exp39_boundary_20260609_070709`。

## 结论

**算法正确**（前置验证）：L=2 全枚举 vs MCMC-TI Δf 对到采样误差，见 `../exact_vs_mcmc_L2.md`。问题在 estimator 选择，不在实现。

**各 estimator 的 consecutive-L crossing → 1/L→0 外推**（vs 真值 0.233）：

| estimator | 类型 | 外推 p_c | 判定 |
|---|---|---|---|
| **Δf-gap** `F_(2)-F_(1)` | even（最近竞争扇区） | **0.401** | ✗ 有偏，几乎不漂 |
| q_top `⟨m²⟩` | even | 0.268 | ✓ 收敛（饱和但 crossing 对） |
| signed-mag `⟨m⟩` | odd（sign-aware） | 0.264 | ✓ |
| **w0 = P(真类, sector 0)** | sign-aware | **0.254** | ✓ **最好** |
| Binder `1-⟨m⁴⟩/3⟨m²⟩²` | even | 退化=2/3 | ✗ 无分辨力 |

**物理**：Δf-gap、q_top²、Binder 都是 **even-moment**，分不清「主导逻辑类**对不对**」——pure-correct 与 pure-wrong 的偶矩相同，故它们测的是「**有没有某类主导**」(purity/ordering)，在 full-disorder ~0.40 才换序，**不是**可纠错阈值 0.233。Δf∝L²（membrane 畴壁），其 crossing 尤其钉在 ~0.40。要测可纠错性须 **sign-aware** 量（用 η 定真类）：`w0` 或 `⟨m⟩`，它们随 L 下漂收敛到 0.233（残差~0.02 是 L=3-6 太小）。

**纠正 007**：当时「q_top 饱和 → 改用 Δf-gap」错了。q_top crossing 本就对（外推 0.27），饱和只是看不到有序侧扇出（外观）；Δf-gap 才有偏。**∴ 007 平坦 q_c≈0.05 系统性偏高**，应改用 **w0/q_top crossing**——可用现有 L=3,4,5 数据零成本重算（`m_i=Σ_g w_g(-1)^{bit_i(g)}`，weights=softmax(-delta_f_per_disorder)，真类=sector 0）。

## 产物
- `fss_qnear0.png`、`fss_qnear0.py` — Δf-gap FSS（钉在 0.40 的反面教材）
- `order_param_fss.png`、`order_param_fss.py` — 4 estimator 的 1/L 外推对比（**主图**）
- `order_param_fss_summary.json`、`fss_qnear0_summary.json`
- `../exact_vs_mcmc_L2.md` — 算法正确性验证

## 下一步建议
1. 用 `w0`（或 q_top）crossing **重算 007 q-direction 边界**（现有 L=3,4,5 数据，零新算）→ 修正后的（更低）相边界。
2. 要收紧绝对值需 **更大 L**（L≥6）于关键点 + 正经 data collapse（带 ν 的标度），把残差 0.02 压下去。
