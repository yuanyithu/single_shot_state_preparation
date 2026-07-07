# exp101 status — 开发进度与检查点（进度唯一真值）

**当前指针**：G2.1（model.py + observables.py：disorder/能量/观测量三档记录 + 双系综开关）
**循环状态**：运行中（/loop 已启动，2026-07-07）
**最后更新**：2026-07-07（loop 迭代 10：G1.8 完成，**Phase 1 收官**）

---

## 待用户决策

- **D1（不阻塞，按推荐默认推进）系综选择**：G0.1 判决发现（证据 `validation/001_model_semantics_check_20260707/result.json`，5/5 机器精度）：主项目模型的 Gibbs 权重换元后为 `exp[−K_p|u|−K_q|H_Zu⊕δ|]`——**数据错误 η 不进盘度，只有测量噪声 δ**；与标准 decoding posterior `exp[−K_p|c|−K_q|H_Zc⊕s|]`（双盘度）差异巨大（2D toric L=3, p=0.15, q=0.1：q_top 0.82 vs 0.15；q=0 时 repo=clean、true=quenched/RBIM 型）。
  **推荐（当前按此推进）**：exp101 主模型 = `true_posterior`（state-prep decoding 物理正解，Nishimori 恒等式成立），同时保留 `repo_compat`（δ-only）开关用于与 3D 机器对拍——两者共享全部代码，仅差 syndrome 参数接线（s vs δ）与真类标签（sig(η) vs 0），无额外成本。**若你希望 exp101 相图沿用 3D 时代的 δ-only 系综，请指出，我会把生产默认切回。**
- **D2（信息通报，与 exp101 无关，供后续定夺）**：上述发现意味着 **exp40/41 的 3D 相图是 δ-only 系综的相图**；若作为 decoding 阈值引用需重新审视。旁证：memory 中 exp39 "Δf-gap FSS→0.40" 恰≈ clean 3D 对偶 Ising p_c≈0.391（δ-only 系综 q=0 的正解），当时按"真值 0.233=3D RBIM"判为估计量有偏的解读可能需要反转。此项不影响 exp101 进度。
- **D3（不阻塞，双候选已登记）家族 seed 规则是否加距离下限**：G1.8 注册表实测（validation/003）：仅满秩规则下 m=2,3 的 base seed 码距只有 **d=2**（H 重复列）；升级规则「满秩 且 列互异（d≥3）」给出 m=2→seed **12349**（d=4）、m=3→**12347**（d=4），m=4,5,6 两规则重合（seed 12345，d=6,4,8）。**推荐采用 d≥3 规则**（脆弱成员会污染 scaling 解读）；两列都在注册表里，切换零成本。另注：随机家族 d 随 m 非单调（4,4,6,4,8）属正常波动，per-u 观测与注册表 d 记录足以在分析时解读。**当前默认仍为已批准的仅满秩规则，等你表态后 Phase 2+ 的验证实例将按所选规则取成员。**

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
| G1.1 | gf2.py + 单测（spec §6/§10D） | **通过** | `src/gf2.py`；`tests/test_gf2.py`（17 passed）；`validation/002_phase1_module_tests_20260707/pytest_g1_1_gf2.txt` | 2026-07-07。spec §6 全函数 + gf2_solve/gf2_inverse（logicals 备用）；测试含独立 bitmask-oracle 交叉、RREF 结构性质、rank-nullity、quotient 维数/独立性/包含违例、空矩阵边角 |
| G1.2 | graphs.py + 确定性构造器 + 单测（§1-2/§10A） | **通过** | `src/graphs.py`；`tests/test_graphs.py`（26 tests）；`validation/002_.../pytest_g1_2_graphs.txt`（43 passed 全套） | 2026-07-07。spec §10A 全项 + 校验方法坏例检测 + K_{4,3} 唯一性 + 确定性构造器（cycle/repetition/K_{a,b}，秩断言）。**发现并文档化**：config-model 简单图接受率≈exp(−(d_A−1)(d_B−1)/2)，(3,4)≈5% 无碍，(5,6)≈4.5e-5 需加大 max_attempts（已测） |
| G1.3 | hgp.py（含 hgp_from_H）+ 单测（§3-4/§10B,C） | **通过** | `src/hgp.py`；`tests/test_hgp.py`（21 tests）；`validation/002_.../pytest_g1_3_hgp.txt`（64 passed 全套） | 2026-07-07。索引约定用手推 2×3 例锁死；行重 d_A+d_B、分块列重 d_A/d_B（强于 spec 上界）；k 公式对 [[8,2]]/[[18,2]]/[[5,1]]/[[13,1]]/K_{4,3}=[[25,13]] 全对；(3,4)m=2 seed12345 满秩 k=4；非正则 H 支持；与主项目 build_2d_toric_code 秩不变量互证 |
| G1.4 | expansion.py 精确验证器 + 单测（§5/§10G） | **通过** | `src/expansion.py`；`tests/test_expansion.py`（12 tests）；`validation/002_.../pytest_g1_4_expansion.txt`（76 passed 全套） | 2026-07-07。Fraction 精确、双侧、worst-ratio witness、空真显式记录（(3,4)m=2 在 spec 示例 γ=1/10 下双侧空真实测）、float 拒收、子集预算守卫；手工图（匹配/K_{4,3} 边界 δ=1/2/二部环 3/4）+ 随机图 vs 测试内独立 set-oracle 全一致 |
| G1.5 | logicals.py 配对归一 + 单测（§7/§10E） | **通过** | `src/logicals.py`；`tests/test_logicals.py`（20 tests）；`validation/002_.../pytest_g1_5_logicals.txt`（96 passed 全套） | 2026-07-07。商基 + M^{-1} 归一到 δ_ij；verify_logical_pauli_result 全清单（kernel 隶属/非 stabilizer/配对恒等/模 stabilizer 独立/k 公式）；七个码全过（含 K_{4,3} k=13、官方 (3,4)m=2、非正则、k=0 边角）；归一化不破坏逻辑类（随机组合抽查）；非 CSS 输入拒收；蓄意破坏被检出 |
| G1.6 | params.py 精确 [[n,k,d]] + 已知码对照（§8/§10F） | **通过** | `src/params.py`；`tests/test_params.py`（30 tests）；`validation/002_.../pytest_g1_6_params.txt`（126 passed 全套） | 2026-07-07。int-bitmask Gray-code 精确距离 + 维数守卫；[[8,2,2]]/[[18,2,3]]/[[5,1,2]]/[[13,1,3]]/[[25,13,2]]/[[32,2,4]] 全命中；n≤18 与测试内全空间独立 oracle 一致；min_logical 绝非 stabilizer（代码断言+测试）；HGP 经典侧距离定理在 5 小码上与暴力值一致（授权对大 m 记录"定理值"并标注来源）；(3,4)m=2 守卫正确触发、经典侧 d 可得 |
| G1.7 | instance.py + 序列化 + examples/spec_example.py | **通过** | `src/instance.py`；`tests/test_instance.py`（10 tests）；`validation/002_.../pytest_g1_7_instance.txt`（136 passed 全套）；`examples/spec_example_output.txt` + `spec_example_instance.json` | 2026-07-07。spec §9 全签名；距离 provenance 双路径（m=1 暴力 [[25,13,2]]；m=2 守卫→定理路径 d=2 并标注）；JSON 序列化 + 同 seed 重建指纹校验 + 篡改检出；spec 期望示例跑通（m=2: n=100,k=4,满秩,CSS ✓,expansion 空真通过） |
| G1.8 | families.py 官方 m=1..6 seed 注册表 | **通过** | `src/families.py`；`tests/test_families.py`（8 tests，含「首个 seed」逐一复核）；`validation/003_family_registry_20260707/{build_registry.py,family_registry.md,family_registry.json(本地)}`；`validation/002_.../pytest_g1_8_families.txt`（144 passed 全套） | 2026-07-07。双规则注册（呼应 D3）：full_rank（m=2..6 全为 seed 12345；m=2,3 d=2 脆弱）与 full_rank_d3（12349/12347/12345/12345/12345，d=4,4,6,4,8）；m=1=K_{4,3} 作 validation-only 成员登记（rank=1 不满秩）；量子 d 一律标注来源；注册函数确定性可复现 |

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
- 2026-07-07 loop#2（G0.2+G0.3）：写出模型规格权威文档 notes/01_model_spec.md（Nishimori 恒等式精确证明、w_u 代数性质、TI 端点/pairwise 判据、计数表枚举设计、q>0 label 混合风险注记 §9）；notes/02_env.md 环境记录（远端挂账）；plan §1.2 sector 叙事修正（X/H_Z↔|0̄⟩）。**Phase 0 全绿**，phase-0 提交 `73e38cf` 已推送。指针 → G1.1。
- 2026-07-07 loop#3（G1.1）：src/ 包初始化 + gf2.py（spec §6 全函数 + solve/inverse）+ tests/test_gf2.py 17 通过（独立 oracle 交叉验证）。指针 → G1.2。
- 2026-07-07 loop#4（G1.2）：graphs.py（BiregularBipartiteGraph + configuration model 单流可复现 + 校验方法 + cycle/repetition/K_{a,b} 构造器）+ 26 测试全过（累计 43）。文档化 (5,6) 拒绝率坑。指针 → G1.3。
- 2026-07-07 loop#5（G1.3）：hgp.py（spec 公式 + hgp_from_H 任意 H + hgp_expected_parameters 理论值）+ 21 测试全过（累计 64）：索引锁死、CSS 对易、k 公式五个已知码、(3,4)m=2 满秩 k=4、主项目 toric 互证。指针 → G1.4。
- 2026-07-07 loop#6（G1.4）：expansion.py（精确 Fraction、witness、空真记录、预算守卫）+ 12 测试全过（累计 76）：边界情形精确命中、独立 oracle 一致。指针 → G1.5。
- 2026-07-07 loop#7（G1.5）：logicals.py（商基 + 配对归一 + 全清单校验）+ 20 测试全过（累计 96）：七码覆盖含 k=13/k=0，归一化保逻辑类。指针 → G1.6。
- 2026-07-07 loop#8（G1.6）：params.py（bitmask Gray-code 精确距离 + 守卫 + 经典侧距离）+ 30 测试全过（累计 126）：六个已知 [[n,k,d]] 命中、全空间 oracle 一致、经典侧定理交叉验证。指针 → G1.7。
- 2026-07-07 loop#9（G1.7）：instance.py（spec §9 + 指纹/序列化/重建校验 + 距离 provenance）+ examples/spec_example.py 跑通 + 10 测试全过（累计 136）。发现 m=2 seed12345 d=2（重复列）→ 写入 D3（seed 规则加距离下限建议）。指针 → G1.8。
- 2026-07-07 loop#10（G1.8）：families.py 双规则注册表 + 8 测试全过（累计 144）+ validation/003 全表落盘（m=1..6，0.19s）。D3 更新为具体双候选数据。**Phase 1 全绿（G1.1–G1.8，144 tests）**，做 phase-1 git 提交。指针 → G2.1。
