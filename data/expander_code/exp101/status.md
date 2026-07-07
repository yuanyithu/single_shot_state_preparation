# exp101 status — 开发进度与检查点（进度唯一真值）

**当前指针**：G1.1（gf2.py + 单元测试）
**循环状态**：运行中（/loop 已启动，2026-07-07）
**最后更新**：2026-07-07（loop 迭代 2：G0.2/G0.3 完成，Phase 0 收官）

---

## 待用户决策

- **D1（不阻塞，按推荐默认推进）系综选择**：G0.1 判决发现（证据 `validation/001_model_semantics_check_20260707/result.json`，5/5 机器精度）：主项目模型的 Gibbs 权重换元后为 `exp[−K_p|u|−K_q|H_Zu⊕δ|]`——**数据错误 η 不进盘度，只有测量噪声 δ**；与标准 decoding posterior `exp[−K_p|c|−K_q|H_Zc⊕s|]`（双盘度）差异巨大（2D toric L=3, p=0.15, q=0.1：q_top 0.82 vs 0.15；q=0 时 repo=clean、true=quenched/RBIM 型）。
  **推荐（当前按此推进）**：exp101 主模型 = `true_posterior`（state-prep decoding 物理正解，Nishimori 恒等式成立），同时保留 `repo_compat`（δ-only）开关用于与 3D 机器对拍——两者共享全部代码，仅差 syndrome 参数接线（s vs δ）与真类标签（sig(η) vs 0），无额外成本。**若你希望 exp101 相图沿用 3D 时代的 δ-only 系综，请指出，我会把生产默认切回。**
- **D2（信息通报，与 exp101 无关，供后续定夺）**：上述发现意味着 **exp40/41 的 3D 相图是 δ-only 系综的相图**；若作为 decoding 阈值引用需重新审视。旁证：memory 中 exp39 "Δf-gap FSS→0.40" 恰≈ clean 3D 对偶 Ising p_c≈0.391（δ-only 系综 q=0 的正解），当时按"真值 0.233=3D RBIM"判为估计量有偏的解读可能需要反转。此项不影响 exp101 进度。

## 已定决策（2026-07-07 用户确认，勿再询问）

1. sector：生产默认 = X 错误 / H_Z checks；管线支持对偶 sector，验证两侧都覆盖小例子。
2. 码族：(d_A, d_B) = (3, 4)，seed 自 12345 系列（满秩 k=m²）。
3. 规模目标：m ≤ 6（n ≤ 900，k ≤ 36），性能对标 3D L=7。
4. 复用方式：自包含移植进 exp101/src/；主项目 src/ 只读（文件头标注来源+commit SHA）。

## 环境快照

- 本地 conda `12`（2026-07-07 已验）：py 3.12.12, numpy 2.4.1, numba 0.65.1, ldpc 2.4.1, scipy 1.17.0, matplotlib 3.10.8, pytest 9.0.3。
- 远端 nd-1/2/3 env `11`：**未验**（挂账：G4.1 前必须 import numba/ldpc + screen 外核数探测，记入 notes/02_env.md）。
- 已复现坑提醒：本地 `conda run` + heredoc 会吞输出——脚本先写 .py 文件再跑。

---

## Gate 状态表

状态取值：`未开始` / `进行中` / `通过` / `失败` / `阻塞`。**通过必须填证据路径**。

### Phase 0 — 侦察与规格

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G0.1 | 主项目接口盘点 → notes/00_interface_recon.md | **通过** | `notes/00_interface_recon.md`；`validation/001_model_semantics_check_20260707/{model_semantics_check.py,result.json}` | 2026-07-07。关键发现：①repo 模型=δ-only 盘度（判决 5/5，含与 repo 枚举逐位互证）→ plan §1.2 双系综；②旧 q_top=全部非零 u 均值（TI 式 (2^kΣw²−1)/(2^k−1)）；③生产路径=exp37 sector-TI（exp41 manifest 溯源）→ 新增 G2.8；④大 k 爆炸点清单与替代 |
| G0.2 | 模型规格推导 → notes/01_model_spec.md | **通过** | `notes/01_model_spec.md` | 2026-07-07。含：双系综权威定义与桥梁恒等式；w_u 湮灭 T⊕S 的三条代数性质（实现断言项）；**Nishimori E[m_u]=E[m_u²] 与 E[w0]=E[purity] 对 true 系综的精确证明**（对 repo 系综显式失败=判别测试）；解析极限公式（E1/E2 分开）；TI 端点条件 ΔF(0)=0 与 pairwise 变体可加性判据；(W_p,W_s,ℓ) 计数表枚举算法（一表通吃全 (p,q) 网格+TI 曲线）；q>0 label 扇区 O(1) 混合的物理解释风险注记（§9）；sector 叙事修正：X/H_Z↔\|0̄⟩ |
| G0.3 | 环境记录 → notes/02_env.md | **通过**（远端挂账） | `notes/02_env.md` | 2026-07-07。本地 12 全套已验；远端 nd-1/2/3 检查清单挂账至 G4.1（plan 允许） |

### Phase 1 — 构造模块（spec 全实现）

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G1.1 | gf2.py + 单测（spec §6/§10D） | 未开始 | — | |
| G1.2 | graphs.py + 确定性构造器 + 单测（§1-2/§10A） | 未开始 | — | cycle_H / repetition_H / complete_bipartite |
| G1.3 | hgp.py（含 hgp_from_H）+ 单测（§3-4/§10B,C） | 未开始 | — | 返回顺序 H_Z, H_X |
| G1.4 | expansion.py 精确验证器 + 单测（§5/§10G） | 未开始 | — | Fraction 精确；witness；小 m 空真记录 |
| G1.5 | logicals.py 配对归一 + 单测（§7/§10E） | 未开始 | — | ⟨x_v,z_u⟩=δ_uv |
| G1.6 | params.py 精确 [[n,k,d]] + 已知码对照（§8/§10F） | 未开始 | — | [[13,1,3]] / 2D toric / K_{4,3} / m=2 / k=0 |
| G1.7 | instance.py + 序列化 + examples/spec_example.py | 未开始 | — | spec 期望交付示例 |
| G1.8 | families.py 官方 m=1..6 seed 注册表 | 未开始 | — | 满秩钉死；JSON 落盘 validation/ |

### Phase 2 — MCMC 管线

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G2.1 | model.py + observables.py（三档记录约定 + 双系综开关）+ 单测 | 未开始 | — | 全量 vs 抽样小 k 交叉 |
| G2.2 | section.py（linear 默认 / BpLsd 可选）+ 防误用断言 | 未开始 | — | 不对观测 s 取 section |
| G2.3 | reference_mcmc.py + 玩具体系精确平稳性 + ΔE fuzz | 未开始 | — | 转移矩阵稳态=Gibbs |
| G2.4 | fast_mcmc.py numba + 与参考实现一致 gate | 未开始 | — | bit 级优先，降级需 notes 说明 |
| G2.5 | pt.py sync_enlarge + swap 单测 + PT vs 单链 | 未开始 | — | cluster update 不移植（记录理由） |
| G2.6 | gates.py 收敛诊断（per-u worst-u 判据） | 未开始 | — | 「共冻≠收敛」内建 |
| G2.7 | run_scan.py + manifest + NPZ schema 兼容 + 续采 | 未开始 | — | manifest 含 ensemble 标签 |
| G2.8 | sector_ti.py TI 引擎移植（泛化 2^k、双系综接线、k>10 basis-pairwise 变体、numba） | 未开始 | — | G0.1 新增；exp37 生产路径泛化 |

### Phase 3 — ground-truth 验证矩阵

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G3.1 | enumerate_exact.py + 与主项目枚举互证 | 未开始 | — | 2D toric L=2 同输入对照 |
| G3.2 | V1 主矩阵：枚举 vs MCMC（5 实例 × 网格 × disorder） | 未开始 | — | z-score/TVD/⟨E⟩/聚合量 gates |
| G3.3 | V1c section-frame A/B | 未开始 | — | 每 frame 下 enum=MCMC |
| G3.4 | V2 解析极限（p=0.5 / q=0.5 闭式 m=2,4,6 / 极限一致） | 未开始 | — | q=0.5 闭式覆盖生产规模 |
| G3.5 | V3 Nishimori 三级（全求和 / 抽样×enum / 全 MCMC n=100） | 未开始 | — | 精确形式以 G0.2 推导为准 |
| G3.6 | V4 实现冗余 A/B（numba/PT/多起点/RNG） | 未开始 | — | |
| G3.7 | V6 冻结扇区 torture（负例必须报警 + 正例通过） | 未开始 | — | per-u 冻结检测 k=4..9 |

### Phase 4 — 服务器规模化验证

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G4.1 | 远端 env 确认 + launcher + 单节点 smoke 全往返 | 未开始 | — | remote-prod-scan checklist |
| G4.2 | 性能 profile m=2..6 + 验收线钉数 | 未开始 | — | 对标 3D L=7 |
| G4.3 | 多节点一致性 + 同 seed 复现 + 续采一致 | 未开始 | — | |
| G4.4 | mini 物理烟测（2D toric 文献对照 + expander sanity） | 未开始 | — | q=0 crossing ∈ [0.09,0.12] |
| G4.5 | 服务器目录规范核查 + 回收校验 + 清 scratch | 未开始 | — | |

### Phase 5 — 毕业

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G5.1 | 全 gate 审计 → report.md | 未开始 | — | |
| G5.2 | 笔记/实验报告.md 增量 | 未开始 | — | |
| G5.3 | CLAUDE.md 增补 + memory | 未开始 | — | |
| G5.4 | git 提交推送 | 未开始 | — | 阶段性提交见各 phase |

---

## validation/ 编号目录索引

（由循环维护：编号 | 对应 gate | 路径 | 一句话结论）

| # | Gate | 路径 | 结论 |
|---|---|---|---|
| 001 | G0.1/G0.2 | `validation/001_model_semantics_check_20260707/` | 判决：repo 模型=δ-only 盘度（T1/T2/T5），true 模型 gauge 恒等式 m(η,δ)=(−1)^{⟨w,η⟩}m(0,s) 精确成立（T3），两模型数值差异巨大（T4）；5/5 通过，机器精度 |

---

## changelog

- 2026-07-07 规划会话：创建 plan/status/prompt；4 项决策经用户确认；本地环境验证完成；exp101 目录初始化。
- 2026-07-07 loop#1（G0.1）：完成主项目全量接口盘点（notes/00_interface_recon.md）；数值判决模型语义（validation/001，δ-only 发现）→ plan §1.2 双系综修订 + 新增 G2.8；考证旧 q_top u 范围=全部非零 u；确认生产路径=sector-TI；写入待用户决策 D1/D2。指针 → G0.2。
- 2026-07-07 loop#2（G0.2+G0.3）：写出模型规格权威文档 notes/01_model_spec.md（Nishimori 恒等式精确证明、w_u 代数性质、TI 端点/pairwise 判据、计数表枚举设计、q>0 label 混合风险注记 §9）；notes/02_env.md 环境记录（远端挂账）；plan §1.2 sector 叙事修正（X/H_Z↔|0̄⟩）。**Phase 0 全绿**，做 phase-0 git 提交。指针 → G1.1。
