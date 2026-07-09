# exp101 结题报告：expander code 单发制备统计力学 q_top 管线（正确性验证）

2026-07-09。对应 plan.md 的全部 gate（G0–G4 全绿；G5 本报告）。目标：产出**自包含、经多重
ground-truth 交叉验证、后续 exp102+ 可稳健复用的正确程序**。**结论：达成。**

---

## 1. 交付物

自包含 python 包 `exp101/src/`（不依赖主项目 src，文件头标来源；259 单元测试全绿）：

| 模块 | 内容 |
|---|---|
| gf2 / graphs / hgp / expansion / logicals / params / instance / families | 构造层（精确 GF(2)；HGP；expansion 验证；逻辑算符；[[n,k,d]]；家族注册） |
| section / model / observables | 统计力学层：线性/decoder section frame；双系综规范变量；W frame + 三档观测量 |
| prng / reference_mcmc / fast_mcmc | 可移植 PRNG（python/numba 双胞胎，位级一致）+ 参考/numba 双引擎 |
| pt / gates / sector_ti | parallel tempering；收敛诊断 gate；sector-TI 引擎 |
| enumerate_exact / run_scan | 精确枚举（计数表+Gray）；扫描入口（双引擎、多进程并行、兼容 NPZ、断点续采） |

CLI：`cd exp101 && python -m src.run_scan --family expander34 --size-list 2 3 ... --engine direct --num-workers N`。

---

## 2. 验证矩阵结果（全部通过）

| gate | 内容 | 关键证据 |
|---|---|---|
| V0 单测 | spec §10 A–G + MCMC 内核平稳性 | 259 tests；独立 oracle 交叉（bitmask rank、全空间距离、set-并集 expansion） |
| G1 构造 | [[n,k,d]] 已知码对照 | [[8,2,2]]/[[18,2,3]]/[[13,1,3]]/[[25,13,2]]/[[32,2,4]] 全命中；主项目 toric 互证 |
| V1 枚举 vs MCMC vs TI | 5 实例 × (p,q) 网格 × disorder，双系综 | regime-aware ALL PASS：direct 偏差 −0.008±0.041、PT 冷点传输、TI-full flag-aware |
| V1c frame A/B | q=0 frame 无关 / q>0 gauge | q=0 相对分布三 frame 无关 **1e-15**；q>0 依赖被观测 |
| V2 解析极限 | p=0.5 / q=0.5 闭式 / 极限一致 | q=0.5 闭式覆盖 **m=2/4/6（n=100/400/900,k=4/16/36）** z≤2.4 |
| V3 Nishimori | E[m]=E[m²] 三级 | L1 全 4096-disorder 求和 **1.9e-14**；L3 n=100 全 MCMC z=2.06；repo_compat 判别 gap=0.25 |
| V4 冗余 | numba/PT/多起点/RNG | ref≡numba bit 级；各引擎对枚举一致 |
| V6 冻结 torture | 负例必报警 + 正例通过 | expander k=4/k=9 冻结诊断均报警；PT 正例传输 |
| G3.1 枚举互证 | 与主项目 exact_enumeration | logZ 关系 1e-9 + decoder-frame m_u 逐位 |
| G4 服务器 | env/smoke/profile/多节点/清理 | nd-1/2/3 env 11；跨节点 **bit-identical**；物理烟测复现 2D 阈值 |
| G4.4 物理 | 2D toric/surface 阈值 | crossing 0.133/0.069 **包夹文献 RBIM p_c=0.109**，相变端行为正确 |

---

## 3. 关键发现（本实验的核心价值）

1. **【D4，最重要】pairwise-TI 大 k 方法失效**：plan 原定 k>10（即全部 expander 生产家族
   k=m²≤36）的 q_top 方法在 K43(k=13) 上对精确 m_u 偏差达 **1.55/满量程 2**（可加性假设崩溃），
   k=2 亦 0.11（对照 full-TI 0.03）。**非 bug**（exact m_u 被枚举机制与 direct 采样双重佐证）。
   → **弃用；大 k q_top 改用 direct/PT 采样观测量**（已由 V2b k=16/36 + V1 direct K43 验证）。
   证据 `validation/007`。exp101 在烧生产算力前拦截了 dead-end 方法。
2. **主项目 3D 模型 = δ-only 盘度**（validation/001，机器精度）：`exp[−K_p|c⊕η|−K_q|Hc⊕s|]`
   换元后数据项无 η 盘度。exp101 支持 **true_posterior**（双盘度，decoding 正解，Nishimori 成立）
   与 **repo_compat**（δ-only，与 3D 时代对拍）双系综，仅差 (σ_arg,ℓ_ref) 接线。
3. **PT 是纯 python（未 numba）**：direct 引擎 numba 极快（m=6 仅 1.1s/disorder），但 PT
   （冷区 sector 传输方法）大 m 慢（m=6 生产等效≈302s/disorder）。仍可行（远低于既已可行的
   3D L=7 6090s/disorder + disorder 级 240 核并行）。
4. **frame = gauge（修正协议）**：q=0 相对类分布 frame 无关（精确），q>0 依赖 frame；跨 run 比较须同 frame。
5. **section.fingerprint n≥400 崩溃真 bug**（V2 捕获，已修+回归）：会崩所有 m≥4 生产 run。

---

## 4. 已知局限与注记

- **精确阈值未定**：G4.4 是烟测（微型码有限尺寸 crossing 散布），精确 expander 阈值需更大码 + FSS
  （生产/分析后续）。
- **深冷大 k 收敛**：direct 在冷区大 k 冻结（q_top 假值），须 PT；expander 生产深冷区的 PT 收敛
  充分性待 exp102 实测（D4 挂账）。
- **K43(m=1) 特殊**：d=2、k=13，是构造/大 k 强测例，非物理典型（家族从 m=2 起）。
- **枚举界**：full 2^n 要求 n≤28、coset dim≤28（守卫）。
- **expansion 验证**：小 m 空真（γ·n<1），(3,4) 在 spec 示例 (1/10,1/16) 下大 m 不通过属预期。

---

## 5. exp102 生产建议

- **方法**：大 k q_top = **direct + PT 采样观测量**（sampled u，k>10 抽样 64 个）。TI 仅 k≤10 交叉验证。
- **sector/系综**：X 错误/H_Z（|0̄⟩ 制备）；默认 true_posterior（待 D1 确认）。
- **家族**：(3,4)，seed 规则 full_rank 或 full_rank_d3（待 D3；后者 m=2,3 距离更大：seed 12349/12347）。
- **参数网格**：p、q 二维扫；crossing 区加密。每点 disorder ≥ 数十–上百。
- **收敛**：per-u worst-u 冷端 logical 接受率 + PT round-trip 双硬判据（gates.py）；q>0 label 有
  O(1) 局域混合通道（notes/01 §9），crossing 须查随尺寸漂移。
- **算力**：direct 秒级/disorder；PT ~数十–数百 s/disorder（m=6）。disorder 级跨 nd-1/2/3
  （80/80/96 核）`--num-workers` 并行。总成本 ≈ points×disorders×per-disorder，完全可行。
- **生产前 TODO**：①若 PT 成瓶颈 → PT 内循环 numba 化 或 decoder-informed 初始化（起点近 φ(η)）；
  ②launcher 显式传 commit SHA（远端无 .git）；③复制改造 remote-prod-scan 模板 launcher（多节点
  cell 矩阵 + _CELL_SUCCESS + README 恢复手册）；④生产日志确认 workers=N、load≈N。

---

## 6. 待用户决策（非阻塞，已按推荐推进；exp102 前请确认）

- **D1 系综默认**：推荐 true_posterior（decoding 正解）。若要沿用 3D δ-only 请指出。
- **D3 家族 seed 规则**：推荐 full_rank_d3（距离≥3，避免 d=2 脆弱成员）。
- **D4 大 k 方法**：pairwise-TI 已弃用，改 direct/PT 采样（已验证）。如有其它偏好请指出。

---

## 7. 结论

exp101 交付了一个经全面 ground-truth 交叉验证的、正确的、可复用的 expander code 单发制备
q_top 管线。**核心价值**：不仅验证了程序正确性（枚举/解析极限/Nishimori/多引擎/frame/物理阈值
全绿），更在烧生产算力前发现并纠正了 planned 大 k 方法（pairwise-TI）失效、定位了可用方法
（direct/PT 采样），并捕获修复了会崩溃所有大规模生产 run 的真 bug。exp102 可直接复用 `exp101/src`
开展 expander q_top(p,q) 相图的规模化研究。
