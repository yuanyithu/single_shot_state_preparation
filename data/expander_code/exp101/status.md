# exp101 status — 开发进度与检查点（进度唯一真值）

**当前指针**：G4.3（多节点一致性 + 同 seed 复现 + 续采一致）
**循环状态**：运行中（/loop 已启动，2026-07-07）
**最后更新**：2026-07-09（loop 迭代 31-32：run_scan 并行化 + G4.2 profile 完成）

---

## 待用户决策

- **D1（不阻塞，按推荐默认推进）系综选择**：G0.1 判决发现（证据 `validation/001_model_semantics_check_20260707/result.json`，5/5 机器精度）：主项目模型的 Gibbs 权重换元后为 `exp[−K_p|u|−K_q|H_Zu⊕δ|]`——**数据错误 η 不进盘度，只有测量噪声 δ**；与标准 decoding posterior `exp[−K_p|c|−K_q|H_Zc⊕s|]`（双盘度）差异巨大（2D toric L=3, p=0.15, q=0.1：q_top 0.82 vs 0.15；q=0 时 repo=clean、true=quenched/RBIM 型）。
  **推荐（当前按此推进）**：exp101 主模型 = `true_posterior`（state-prep decoding 物理正解，Nishimori 恒等式成立），同时保留 `repo_compat`（δ-only）开关用于与 3D 机器对拍——两者共享全部代码，仅差 syndrome 参数接线（s vs δ）与真类标签（sig(η) vs 0），无额外成本。**若你希望 exp101 相图沿用 3D 时代的 δ-only 系综，请指出，我会把生产默认切回。**
- **D2（信息通报，与 exp101 无关，供后续定夺）**：上述发现意味着 **exp40/41 的 3D 相图是 δ-only 系综的相图**；若作为 decoding 阈值引用需重新审视。旁证：memory 中 exp39 "Δf-gap FSS→0.40" 恰≈ clean 3D 对偶 Ising p_c≈0.391（δ-only 系综 q=0 的正解），当时按"真值 0.233=3D RBIM"判为估计量有偏的解读可能需要反转。此项不影响 exp101 进度。
- **D4（重要发现，已技术解决，需你知悉/确认生产策略）pairwise-TI 大 k 方法失效**：G3.2 测得 pairwise-TI 估计量（plan §12 原定 k>10 即全部 expander 生产家族 k=m²≤36 的 q_top 方法）在 K43(k=13) 上**完全不近似真 q_top**：max\|m_u^pair − m_u^exact\| 达 0.8–1.45（满量程 2），mean 0.4–0.9。根因 = 扇区自由能的**可加性假设严重失效**（K43 有大量低重逻辑算符，类分布跨 label 位强相关）。**非 bug**：exact m_u 来自 G3.1 机器精度机制，且 K43 direct 引擎采样 m_u 已与该 exact 一致（V1 direct 通过）——pairwise 同时偏离两者。
  **技术解决（已按此推进）**：大 k 生产 q_top 改用 **direct/PT 采样观测量估计**（notes/01 §4 无偏，已由 V2b k=16/k=36 闭式 + V1 direct K43 验证）；pairwise-TI **降级/弃用为 q_top 方法**（仅单个 ΔF_u 单翻转 gap 仍被正确测量，但禁止假可加性合成 q_top）；full-TI 仅 k≤10 有效（小码/交叉验证）。**对生产的含义**：exp102 expander 相图必须走 direct/PT 采样路径 + PT 提供 crossing 区 sector 传输；深冷大 k 区的收敛充分性待 Phase 4 确认。若你希望保留/另设大 k 的 TI 类方法或有其它偏好，请指出；否则按上述执行。
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
| G2.1 | model.py + observables.py（三档记录约定 + 双系综开关）+ 单测 | **通过** | `src/{model,observables,section}.py`；`tests/test_model_observables.py`（21 tests）；`validation/002_.../pytest_g2_1_model_observables.txt`（165 passed 全套） | 2026-07-07。**规范采样变量统一两系综**（π(v)∝exp[−K_p\|v\|−K_q\|Hv⊕σ_arg\|]，系综只差 (σ_arg,ℓ_ref) 接线）；换元等价对旧式 (\|v⊕η\|,s) 枚举逐位锁定；T3 桥梁/T1 η-无关在新接线复核；w_u 三性质独立复核 + φ 分解不变性；三档 u 集（full≤10/sampled+seed）+ 聚合公式对手工类分布精确；CRN disorder 路径；q=0 硬约束接线与 im 校验 |
| G2.2 | section.py（linear 默认 / BpLsd 可选）+ 防误用断言 | **通过** | `src/section.py`（LinearSection + DecoderSection + DecoderObservableFrame + column_priority 备选线性 frame）；`tests/test_section_frames.py`（10 tests）；`validation/002_.../pytest_g2_2_section_frames.txt`（175 passed 全套） | 2026-07-07。BpLsd 后端实测可用（k=13 例含）；缓存有上限（修复主项目无界缓存）；strict 防误用两后端一致；**frame 在 ker 上一致（q=0 frame 无关性实证）**；两 frame 协变性（⊕S 不变/⊕x_u 翻位）；G3.3 的 A/B 接口就绪 |
| G2.3 | reference_mcmc.py + 玩具体系精确平稳性 + ΔE fuzz | **通过** | `src/reference_mcmc.py`；`tests/test_reference_mcmc.py`（11 tests）+ `tests/util_enum.py`；`validation/002_.../pytest_g2_3_reference_mcmc.txt`（186 passed 全套） | 2026-07-07。三 move（bit/S/L）+ per-u logical 接受计数 + RNG 协议文档化；**穷举 ΔE**（全状态×全 move 对全量重算，机器精度）；[[5,1]] 双系综 m_u 对枚举分块 z<5；toric L=2 状态直方图对精确 Gibbs；q=0 coset 采样+跨 sector 起点一致；p=0.5 锚点（L 接受率≡1、⟨w,S⟩=0 ⇒ S-move 永不翻 O_u 的机制被测试记录）；bit 级可复现；修复：初态 auto 按 q_zero 选（q>0 时 strict section 正确拒绝 σ_arg∉im——防误用机制的实战验证） |
| G2.4 | fast_mcmc.py numba + 与参考实现一致 gate | **通过** | `src/{prng,fast_mcmc}.py`；`tests/test_fast_mcmc.py`（12 tests）；`validation/002_.../pytest_g2_4_fast_mcmc.txt`（202 passed 全套） | 2026-07-07。**bit 级一致达成**（未降级）：可移植 PRNG（splitmix64+xorshift128+，python/numba 双胞胎逐位一致）+ RNG 消耗顺序镜像 ⇒ 双引擎 observable_sums/final v/counters/energy trace 全等（5 案例×双系综×q=0 + sampled 档 + sector 起点）；观测奇偶 uint64 words 增量维护；fast vs enum 独立校验；回退路径；提速 ≥5×（m=2）。**排障记录**：①观测集拒绝采样在不可行请求下死循环（k=4 只有 11 个非 basis u 却请求 16）→ 加可行性检查+防御上限+回归测试；②本地 conda run 无 --no-capture-output 吞输出导致排障被拖长 → notes/02 升级为硬规则 |
| G2.5 | pt.py sync_enlarge + swap 单测 + PT vs 单链 | **通过** | `src/pt.py`；`tests/test_pt.py`（11 tests）；`validation/002_.../pytest_g2_5_pt.txt`（214 passed 全套） | 2026-07-08。sync_enlarge/data_only ladder（端点精确、K_p/K_q 比恒定、**耦合空间表述对 p≥0.5 构造性免疫**——exp35 坑属 odds 表述，守卫保留为防御）；swap 公式 50 组 fuzz 对直算机器精度；PT 冷端对枚举 z<5；PT vs 超长单链 z<5；round-trip>0 + replica 守恒 + per-u 冷端接受率记录；q=0 拒收；可复现；all-rungs 模式热端方向性 ✓。cluster update 不移植理由入 docstring |
| G2.6 | gates.py 收敛诊断（per-u worst-u 判据） | **通过** | `src/gates.py`；`tests/test_gates.py`（12 tests）；`validation/002_.../pytest_g2_6_gates.txt`（225 passed 全套） | 2026-07-08。τ/split-R̂/ESS 独立实现（iid/AR(1)ρ=0.9/均值漂移/冻结退化全校准）；多起点编排（sector 轮转）；**方法学发现：q_top spread 符号盲**（不同 sector 共冻给出相同 q_top）→ gate 新增符号敏感 m_u_spread 判据；「共冻≠收敛」：同 sector 共冻在放宽全部统计判据后仍仅因 sector_transport_insufficient 失败（q=0+关 L-move 的严格冻结负例）；PT round-trip 证据可替代局域接受率判据；nan 接受率按冻结处理 |
| G2.7 | run_scan.py + manifest + NPZ schema 兼容 + 续采 | **通过** | `src/run_scan.py`；`tests/test_run_scan.py`（8 tests）；`validation/002_.../pytest_g2_7_run_scan.txt`（246 passed 全套）；CLI `cd exp101 && python -m src.run_scan` | 2026-07-08。TI/direct 双引擎统一入口；per-task seed scope=sha256(family_fp\|sector\|ensemble\|p\|q\|dis\|stream)（系综入 scope 防误合并，已测）；原子 chunk（tmp+rename，无残留）+ 断点续采（reused/computed 计数、损坏 chunk 重算）+ 跨目录确定性；merge 出 sector_ti_results.npz 兼容字段（code_size_list + lattice_size_list 别名、weights 槽 NaN-pad + manifest weights_layout、m_u/ell_ref 新增字段）；manifest 含 commit SHA/版本/ensemble/engine_config |
| G2.8 | sector_ti.py TI 引擎移植（泛化 2^k、双系综接线、k>10 basis-pairwise 变体、numba） | **通过**（numba 挂账→G4.2） | `src/sector_ti.py`；`tests/test_sector_ti.py`（13 tests）；`validation/002_.../pytest_g2_8_sector_ti.txt`（238 passed 全套） | 2026-07-08。sector-preserving proposals（零签名单比特/同签名对/S 行，label 不变性质测试含 k=4 expander）；K_p 网格退火续链 + ΔF trapezoid + block bootstrap + 粗细网格 flags；full 档对枚举：ΔF 逐扇区 z<5、q_top/w0 双系综 ✓；**true 系综 ℓ_ref≠0 的重排精确成立**；q=0 coset TI ✓；**pairwise 档围绕 ℓ_ref 展开**，ΔF̃_u 对精确 −log 比值 z<5、tanh 公式恒等。**numba TI kernel 挂账至 G4.2 性能线**（本版为正确性权威参考） |

### Phase 3 — ground-truth 验证矩阵

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G3.1 | enumerate_exact.py + 与主项目枚举互证 | **通过** | `src/enumerate_exact.py`；`tests/test_enumerate_exact.py`（10 tests）；`validation/002_.../pytest_g3_1_enumerate_exact.txt`（256 passed 全套） | 2026-07-08。(W_p,W_s,ℓ) 计数表 + Gray code（uint64 打包、SWAR popcount、numba+python 双实现逐位一致）；**一表多 (p,q)**：同表 5 组 (p,q) 对独立逐点枚举机器精度（含 q≈0.5）；结构恒等 N_coset≡N_full[:,0,:]；μ_ℓ 对直算（TI 曲线钩子）；K_{4,3} 2^25 枚举 <1s；守卫触发正确；**主项目互证双通道**：logZ 精确关系（Bernoulli 归一差）1e-9 + decoder-frame m_u 逐位（001 T2 固化进套件） |
| G3.2 | V1 主矩阵：枚举 vs MCMC vs TI（5 实例 × 网格 × disorder，双系综） | **通过**（regime-aware） | `validation/004_v1_main_matrix_20260708/{run_v1.py 生成有效采样, finalize_v1.py 权威 gate, results.json, gates_final.json, summary.md}` + `validation/007_pairwise_characterization_20260709/` | 2026-07-09。ALL PASS：direct(well-mixed wacc≥0.05,256 任务) 逐任务偏差 −0.008±0.041/discrepant 0.001/TVD 0.041/能量 0 fail；PT 冷点偏差 +0.071±0.099/discrepant 0.0013/TVD 0.028/全往返>0；TI-full 未 flag 点 0 fail（24 flag 点被自诊断捕获）；**K43 大 k direct 采样 vs 精确验证通过**。**首轮暴露并纠正 4 处 instrument 错配**（详见 plan changelog）+ **D4 关键发现：pairwise-TI 大 k 失效**。方法学教训见 loop#25-26。 |
| G3.3 | V1c section-frame A/B | **通过** | `validation/006_v1c_frame_ab_20260709/{run_v1c.py,results.json,summary.md}` | 2026-07-09（18s）。ALL PASS：G1 每 frame 内 enum=MCMC（z≤4.54）；**G2 q=0 相对分布 frame 无关精确 1.1e-15**（linear-A/B + decoder 三 frame，证实平移不变推导）；G3 q>0 frame 依赖被观测（rel-TVD 0.60，gauge=修正协议）；G4 三 frame 指纹互异。修 2 bug：q=0 走 coset 表、G1 用 z-OR-绝对 |
| G3.4 | V2 解析极限（p=0.5 / q=0.5 闭式 m=2,4,6 / 极限一致） | **通过** | `validation/005_v2_analytic_limits_20260708/{run_v2.py,results.json,summary.md,run_v2.log}` | 2026-07-09。8 检查全绿（1s）：V2a p=0.5 零化+L接受率≡1；**V2b q=0.5 闭式 m_u=(1−2p)^{\|w_u\|} 覆盖 m=2/4/6（n=100/400/900,k=4/16/36）生产规模 z≤2.4**；V2c q→0⁺vsq=0 连续（diff 1.8e-5）；V2d p,q→0 Bayes 极限 q_top=0.995；V2e 零盘度两系综重合。**捕获真生产 bug**：section.fingerprint 用 bytes() 序列化主元列，索引>255（所有 m≥4/n≥400）即崩——已修为定宽 int64 + 回归测试；V2d 修正错误的极限预期（固定 q 时 p→0 不集中于 η） |
| G3.5 | V3 Nishimori 三级（全求和 / 抽样×enum / 全 MCMC n=100） | **通过** | `validation/008_v3_nishimori_20260709/{run_v3.py,results.json,summary.md}` | 2026-07-09（31s）。ALL PASS：**L1 [[8,2,2]] 全 4096-disorder 求和 E[m]=E[m²] 精确 1.9e-14**；L2 toric_m3(z=1.1)/K43(z=3.5) 抽样×枚举；**L3 expander_m2 n=100 越枚举界全 MCMC**（双独立链无偏 m²，z=2.06）；JUDGE repo_compat q=0.5 恒等式违反 gap=0.25（确认恒等式为 true_posterior 特有=系综判别）。修 1 import typo |
| G3.6 | V4 实现冗余 A/B（numba/PT/多起点/RNG） | **通过** | `validation/009_v4_v6_redundancy_torture_20260709/{run_v4_v6.py,results.json,summary.md}` | 2026-07-09。ref≡numba bit 级一致；PT/direct/1-vs-8-start 各自对枚举一致（z 0.1/2.5/0.1<5） |
| G3.7 | V6 冻结扇区 torture（负例必须报警 + 正例通过） | **通过** | 同 009 目录 | 2026-07-09。**负例 expander k=4 与 k=9 诊断均报警**（共冻仅因 sector_transport_insufficient 失败）；正例 PT round_trips 19/15、初始 sector 无关性 z=1.7；per-u 冻结检测在 k=9 生效 |

### Phase 4 — 服务器规模化验证

| Gate | 内容 | 状态 | 证据 | 备注 |
|---|---|---|---|---|
| G4.1 | 远端 env 确认 + launcher + 单节点 smoke 全往返 | **通过** | `validation/010_g4_remote_smoke_20260709/{summary.md,sector_ti_results.npz(本地)}`；notes/02_env.md | 2026-07-09。nd-1/2/3 env 11 确认（核 80/80/96，numba 0.65.1/ldpc 2.3.7）；smoke 全往返（传输→nd-1 运行→sha256 一致→schema 校验 host=nd-1→清 scratch）。**生产前 TODO**：run_scan 加 ProcessPoolExecutor 并行 + launcher 显式传 commit SHA（见 010 summary）|
| G4.2 | 性能 profile m=2..6 + 验收线钉数 | **通过** | `validation/011_g4_profile_20260709/{run_profile.py,profile.json,summary.md}`；`src/run_scan.py`(并行化) | 2026-07-09。**direct(numba) 极快**：m=2..6 = 0.1..1.1s/disorder（8 起点）；**PT 纯 python 是瓶颈**：m=6 生产等效≈302s/disorder。可行性达标（3D L=7 sector-TI 6090s/disorder 既可行，expander 远低于此 + disorder 级 240 核并行）。numba 生效确认。**run_scan 并行化完成**（ProcessPoolExecutor，259 tests）。**新生产 TODO**：PT numba 化或 decoder-init（若大 m PT 成瓶颈）。注：direct-only 大 m 冷点 q_top 是冻结伪值（须 PT，呼应 D4） |
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
- 2026-07-07 loop#10（G1.8）：families.py 双规则注册表 + 8 测试全过（累计 144）+ validation/003 全表落盘（m=1..6，0.19s）。D3 更新为具体双候选数据。**Phase 1 全绿（G1.1–G1.8，144 tests）**，phase-1 提交 `1243c13` 已推送。指针 → G2.1。
- 2026-07-07 loop#11（G2.1）：section.py（线性部分）+ model.py（规范变量双系综接线）+ observables.py（W frame 三性质断言、三档 u 集、聚合）+ 21 测试全过（累计 165）：换元锁定、T3/T1 复核、聚合对第一性精确。指针 → G2.2。
- 2026-07-07 loop#12（G2.2）：section.py 补全（DecoderSection/BpLsd + 上限缓存 + DecoderObservableFrame + column_priority 备选线性 frame）+ 10 测试全过（累计 175）：ker 上 frame 一致实证、协变性双 frame 成立。指针 → G2.3。
- 2026-07-07 loop#13（G2.3）：reference_mcmc.py（三 move + per-u 计数 + debug 不变量）+ 11 测试全过（累计 186）：穷举 ΔE、枚举对拍（双系综/q=0/直方图）、p=0.5 锚点、bit 级复现。初态 auto 修复（strict 防误用实战拦截了错误默认值）。指针 → G2.4。
- 2026-07-07/08 loop#14-18（G2.4）：prng.py（可移植 PRNG 双胞胎）+ fast_mcmc.py（numba kernel、观测奇偶 words 增量）+ 12 测试；排障两轮（faulthandler/逐 case 定位）：修复观测集拒绝采样死循环（+回归测试）与 conda run 捕获吞输出（notes/02 硬规则化）。**bit 级一致 gate 达成**，202 tests 全过（7.85s）。指针 → G2.5。
- 2026-07-08 loop#19（G2.5）：pt.py（sync_enlarge/data_only ladder、swap、round-trip、per-u 冷端计数）+ 11 测试全过（累计 214）。发现：耦合空间表述使 p≥0.5 越界构造性不可能（与主项目 odds 表述的差异记录在案）。指针 → G2.6。
- 2026-07-08 loop#20（G2.6）：gates.py（诊断量+多起点+gate）+ 12 测试全过（累计 225）。**方法学发现：q_top spread 符号盲 → 新增 m_u_spread 符号敏感判据**；负例设计教训：q>0 时 label 可经 single-bit 通道慢泄漏（§9 物理的又一实证），严格冻结负例须用 q=0+关 L-move。指针 → G2.7。
- 2026-07-08 loop#21（G2.8，与 G2.7 顺序互换）：sector_ti.py（TI 引擎泛化：proposals/退火续链/bootstrap/flags/full+pairwise 双档、pairwise 围绕 ℓ_ref）+ 13 测试全过（累计 238），对枚举全绿。numba TI 挂账 G4.2。指针 → G2.7。
- 2026-07-08 loop#22（G2.7）：run_scan.py（双引擎入口、seed scope、原子 chunk 续采、兼容 NPZ+manifest、CLI）+ 8 测试全过（累计 246）。**Phase 2 全绿（G2.1–G2.8，102 个 Phase-2 测试）**，phase-2 提交 `35d0b2a` 已推送。指针 → G3.1。
- 2026-07-08 loop#23（G3.1）：enumerate_exact.py（计数表 + Gray + numba/python 双实现 + 守卫）+ 10 测试全过（累计 256）：一表多 (p,q) 机器精度、K_{4,3} 2^25 <1s、主项目 logZ/decoder-frame 双通道互证。指针 → G3.2。
- 2026-07-09 loop#31-32（run_scan 并行化 + G4.2 profile）：run_scan 加 ProcessPoolExecutor（spawn，确定性=串行，缺失 chunk 鲁棒，+2 回归测试，259 全绿）。G4.2 profile：direct numba 0.1-1.1s/disorder(m=2..6)、PT python 瓶颈 m=6≈302s、可行性达标。发现 PT 未 numba（生产 TODO）+ direct 大 m 冷点 q_top 冻结伪值（须 PT）。Bash classifier 间歇不可用致数次重试。指针 → G4.3。
- 2026-07-09 loop#30（G4.1 通过）：远端 nd-0/1/2/3 连通 + env 11 确认（核 80/80/96）；exp101 src 传输→nd-1 运行 run_scan smoke→sha256 一致回收→schema 校验（host=nd-1 记录）→清 scratch。识别生产前 2 TODO（run_scan 并行化、commit SHA 记录）。指针 → G4.2。
- 2026-07-09 loop#29（G3.6/G3.7 通过 → **Phase 3 全绿 G3.1–G3.7**）：V4 ref≡numba bit 级 + PT/direct/多起点一致；V6 冻结 torture 负例 k=4/k=9 诊断报警、正例 PT 传输+起点无关。Phase 3 收官提交。指针 → Phase 4（G4.1 远端 smoke）。
- 2026-07-09 loop#27-28（G3.3 + G3.5 通过）：frame A/B（q=0 相对分布 frame 无关 1e-15、q>0 gauge 依赖）；Nishimori 三级递进全绿（L1 全求和 1.9e-14、L3 n=100 全 MCMC z=2.06、repo_compat 判别 gap=0.25）。两脚本各修 1-2 个启动 bug。
- 2026-07-09 loop#26（G3.2 通过 + D4 发现）：V1 regime-aware 重跑（3531s）；分析确认 4 处 instrument 错配（逐任务偏差 vs 池化、ergodic 阈值、TI flag-aware、pairwise 比较对象），用 finalize_v1.py 对有效采样数据施正确 instrument → **ALL PASS**。**pairwise-TI 大 k 失效**（validation/007：K43 max dev 1.55、k=2 也 0.11 而 full-TI 0.03）→ 弃用，大 k 走 direct/PT 采样（D4，plan/风险12 已更新）。累计单元测试 258 全绿。
- 2026-07-09 loop#24-25（G3.4 通过 + G3.2 重构）：写 V1/V2 重型验证脚本。**V2 全绿**（含 m=6 生产规模闭式），捕获并修复 section.fingerprint >255 崩溃真 bug（+回归测试，累计 258 tests）+ V2d 极限预期修正。**V1 首轮暴露 instrument 错配**（bare 单链跑 q=0/超冷 + 纯 z-gate → 假红）：分析确认非 bug（q=0 sector 权重单链原理取不到、m_u≈±1 饱和 z 失真、TI raw-ΔF-z 错 instrument）→ 按 regime-aware 重构 run_v1.py（direct 自证遍历区 z-OR-绝对、PT 覆盖冷点、TI 物理量判据），plan changelog 记录。V1 重跑后台进行中。
