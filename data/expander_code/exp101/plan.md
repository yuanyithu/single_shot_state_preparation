# exp101 — Expander code 单发（single-shot）制备统计力学管线：开发与正确性验证计划

版本 2026-07-07（规划会话产出）。执行由 `/loop` + `prompt.md` 驱动；**进度唯一真值在 `status.md`**，本文件是契约：阶段、gate、判据、约定。改动本文件须在文末 changelog 留一行。

---

## 0. 目标与非目标

**背景**：项目从 3D toric code 转向 quantum expander code（随机 biregular 二部图的 hypergraph product，构造规格见 `../expander_code.md`，该文件为权威约定）。物理问题：单发（一次带噪）测量全部 stabilizer 做 state preparation，读出错误率 q 与 Pauli 错误率 p 对逻辑信息可恢复性的影响。方法完全沿用已验证的路线：统计力学映射 + parallel tempering + MCMC 采样 + q_top。

**exp101 目标**：产出一个**自包含、经多重 ground-truth 交叉验证、后续实验（exp102+ 生产相图）可稳健复用**的正确程序，以及完整的验证证据链。

**非目标**：
- 不在本实验产出物理结论相图（Phase 4 只做端到端烟测）；
- 不做 crossing 精修 / FSS（后续用 `qc-crossing-analysis` skill 的规范）；
- 不移植 cluster update（依赖 toric 几何，见 §6.9）；
- 不研究 (5,6) 等其它度数家族。

**已确认决策（2026-07-07 用户拍板）**：
1. **sector**：生产默认 = X 错误 / H_Z checks（与 3D 管线约定逐字一致）；管线参数化支持对偶 sector（交换 H_X↔H_Z），验证阶段两侧都覆盖小例子。
2. **码族**：(d_A, d_B) = (3, 4)，seed 自 12345 系列（n = 25m²，满秩时 k = m²）。
3. **规模目标**：m ≤ 6（n ≤ 900，k ≤ 36），性能验收对标 3D L=7（n=1029）。
4. **复用方式**：自包含移植进 `exp101/src/`；主项目 `src/` 只读参考，被移植文件头部标注来源文件 + commit SHA。

---

## 1. 物理模型与数学约定（权威定义）

### 1.1 码构造
按 `../expander_code.md`：H ∈ F_2^{B×A}（n_B 行 × n_A 列），qubit 顺序 (A×A) 后接 (B×B)，

    H_X = [ I_{n_A}⊗H | Hᵀ⊗I_{n_B} ]，H_Z = [ H⊗I_{n_A} | I_{n_B}⊗Hᵀ ]，H_X H_Zᵀ = 0。

n = n_A² + n_B²；记 r = rank(H)，则 k = (n_A−r)² + (n_B−r)²，ρ_X = ρ_Z = n_A·n_B − (n_A−r)(n_B−r)。

(3,4) 家族：n_A = 4m，n_B = 3m；H 满秩（r = 3m）时 k = m²。**m=1 的唯一简单图是 K_{4,3}**：[[25, 13, d]]（k = 13 = 3²+2²，两项都非零，预期 d=2，以精确计算为准）——是构造模块的强测例，但不代表家族典型物理；官方家族物理成员从 m=2 起，seed 按 G1.8 规则钉死满秩。

### 1.2 disorder 与 Gibbs 模型（X-error sector，生产默认）【2026-07-07 G0.1 修订】
固定 disorder (η, δ)：η ~ Bern(p)^n（X 错误），δ ~ Bern(q)^{m_Z}（读出翻转，m_Z = n_A n_B），观测 syndrome s = H_Z η ⊕ δ。K_p = log((1−p)/p)，K_q = log((1−q)/q)，Nishimori：温标与 disorder 同 (p,q)。

**G0.1 判决发现**（证据 `validation/001_model_semantics_check_20260707/`，机器精度）：主项目模型 `π(v) ∝ exp[−K_p|v⊕η| − K_q|H_Zv⊕s|]` 换元后 ≡ `exp[−K_p|u| − K_q|H_Zu⊕δ|]`——**η 不进 Gibbs 权重，盘度只有 δ**；这与标准 decoding posterior 不同且数值差异巨大。因此 exp101 管线支持**两个系综开关**：

- **`true_posterior`（主模型，生产默认）**：`π*(c) ∝ exp[−K_p|c| − K_q|H_Zc⊕s|]`（等价换元形式 `exp[−K_p|x⊕η|−K_q|H_Zx⊕δ|]`，双盘度；state-prep decoding 的物理正解；Nishimori 恒等式在此系综成立）。真类 = sig(η)，w0 = weights[sig(η)]。
- **`repo_compat`（兼容模式）**：δ-only 盘度，与 3D 时代程序逐位可比，用于对拍与回归。
- 精确桥梁（判决 T3）：`m_u^{true}(η,δ) = (−1)^{⟨w_u,η⟩}·m_u^{true}(0, s)`——同一套采样/TI 机器把 syndrome 参数从 δ 接成 s、真类标签从 0 换成 sig(η) 即得 true 系综，代价≈0。

q=0 时硬约束 H_Z c = s（s ∈ im(H_Z) 自动成立），c 限制在 coset 上；true 系综在 q=0 是 quenched（η 盘度），repo_compat 在 q=0 是 clean。制备叙事对应关系（G0.2 推导修正，见 notes/01 §2）：**主 sector X 错误/H_Z checks ↔ |0̄⟩ 制备**；对偶 sector（Z 错误 / H_X checks ↔ |+̄⟩ 制备）= 把 (H_Z, H_X, z_u, x_u) 全部互换，同一套代码走参数。

### 1.3 L·T·S 分解与 MCMC moves（generic CSS 版）
- **S** = row(H_X)（X-stabilizer 支撑集），dim ρ_X。moves：翻 H_X 单行（权重 d_A+d_B=7）——不动 syndrome、不动逻辑类。
- **L** = span{x_u}（logical X 代表基），dim k。moves：翻 x_u（sector flip）。接受率 ~ (p/(1−p))^{|x_u|}，|x_u| ≥ d ~ m ⇒ 深有序相的冻结与 3D winding 完全同构 ⇒ **冷端 logical-flip 接受率 / sector 流动是收敛硬判据**，CLAUDE.md「物理图像与 L·T·S 分解」一节全文适用；「共冻 ≠ 收敛」。
- **T**：section r 的像，dim ρ_Z。q>0 时 single-bit 翻转采样 T⊕S；q=0 时 T 被钉死，只允许零 syndrome moves（S 行翻转、L 翻转及小组合）。

### 1.4 观测量与 q_top（大 k 适配——本次关键改造点）
z_u：logical Z 代表（ker(H_X)/row(H_Z) 的 quotient basis），构造模块保证配对归一 ⟨x_v, z_u⟩ = δ_uv。

    O_u(c) = (−1)^{⟨z_u, c ⊕ η ⊕ r(H_Z c) ⊕ r(H_Z η)⟩}，m_u = ⟨O_u⟩_π。

r 只作用于 im(H_Z) 的元素（H_Z c 与 H_Z η）；**绝不对观测 s 取 section**（q>0 时 s 不一定 ∈ im(H_Z)，沿用 CLAUDE.md 的坑）。

对一般 u ∈ F_2^k（z_u = ⊕_{i∈u} z_i），类分布纯度与真类概率：

    purity = Σ_ℓ P(ℓ)² = 2^{−k} Σ_{u∈F_2^k} m_u²；  w0 = P(真类) = 2^{−k} Σ_{u} m_u（m_0 ≡ 1）。

**记录约定（每 run 全存，保证任何后续估计量可重算）**：
【G0.1 考证结论】旧 q_top = **全部 2^k−1 个非零 u** 的 m_u² 均值 = 归一化 purity；TI 引擎等价式 `q_top = (2^kΣw_s²−1)/(2^k−1)`（旧代码硬编码 k=3 的 8/7，移植时泛化）。抽样 U_rand 是对同一定义的无偏推广。
- (a) basis 集：k 个基 u 的 m_u 与 m_u²；另存 `q_top_basis = (1/k)Σ_basis m_u²`（新增聚合，两种都记，report 注明差异）。
- (b) k ≤ 10：另存全部 2^k−1 个非零 u 与 2^k 类权重（旧管线同款路径）→ purity/w0 在 MCMC 意义下"全量"。
- (c) k > 10：均匀随机非零 u 抽样集 U_rand（默认 64 个，抽样 seed 入 manifest）→ 无偏估计 purity ≈ 2^{−k} + (1−2^{−k})·mean_{U_rand}(m_u²)，w0 同理用 mean(m_u)。**小 k 上 (b) 与 (c) 交叉一致是 gate 的一部分**（G2.1 / G3.2）。
- section 默认 **linear**（高斯消元；旧管线经 exact-enum 验证过的 "projection-mode linear" 路线）。线性 section r(σ)=Rσ 时 O_u = (−1)^{⟨w_u, c⊕η⟩}，w_u = (I ⊕ R H_Z)ᵀ z_u 可预计算成 mask 并增量维护。BpLsd section 仅作小码 A/B（G3.3）。
- **section-frame 依赖**：q>0 时 m_u 依赖 section 的规范选择（frame）。同一 run 固定 frame 并把 frame 指纹写入 manifest；跨 run 比较必须同 frame。推导见 G0.2，定量表征见 G3.3。

### 1.5 解析基准（V2 用，任意规模均精确）
- **p = 0.5**（K_p=0）：对任意 q、任意 disorder，所有 m_u = 0（L 方向均匀）。
- **q = 0.5**（K_q=0）：c = η⊕e，e ~ Bern(p) iid ⇒ 线性 section 下闭式 m_u = (1−2p)^{|w_u|}（含符号），且 ⟨|c⊕η|⟩ = np、每 check 违反率闭式。**可直接在 m=6 生产规模检验 observable+采样器**。
- **p → 0⁺**：π 集中于 c=η ⇒ q_top → 1（δ=0 disorder 下一阶微扰可写出并对照）。
- **q → 0⁺ vs q=0**：条件 δ=0 的同 disorder 下，q=1e−3 的 q>0 引擎与 q=0 硬约束引擎一致（两条代码路径交叉）。
- **Nishimori gauge 恒等式**：预期 E_{η,δ}[m_u] = E_{η,δ}[m_u²]（单发含测量噪声版的精确形式与适用条件以 G0.2 推导为准；若推导给出修正形式，以推导为 gate 基准）。
- **disorder=0 快测**（η=0, s=0）：CLAUDE.md 快速测试技巧，内部相变扫描路径 + 小码 enum 对照。

---

## 2. 目录结构（干净组织约定）

```
data/expander_code/exp101/
├── plan.md            # 本计划（契约；改动留 changelog）
├── status.md          # 进度唯一真值：gate 状态/证据路径/当前指针/待用户决策
├── prompt.md          # /loop 驱动 prompt
├── report.md          # 毕业时产出
├── notes/             # 推导与侦察：00_interface_recon.md, 01_model_spec.md, 02_env.md, ...
├── src/               # 自包含 Python 包（禁止 import 主项目 src/）
│   ├── __init__.py
│   ├── gf2.py graphs.py hgp.py expansion.py logicals.py params.py
│   ├── instance.py families.py
│   ├── model.py observables.py section.py
│   ├── reference_mcmc.py fast_mcmc.py pt.py gates.py
│   ├── enumerate_exact.py run_scan.py
│   └── launcher/      # 远端提交/监控/回收脚本
├── tests/             # pytest 快速套件（conda run -n 12 python -m pytest tests -q）
├── examples/          # spec 期望示例脚本与输出
└── validation/        # 编号证据目录 001_xxx_YYYYMMDD/ ...（多节点结果放 nd1/ nd2/ nd3/ 子目录）
```

- 命名沿用项目惯例：validation 子目录 `0NN_极简内容_日期`；每个 gate 的证据要么在 tests/ 输出，要么在 validation/ 编号目录。
- 服务器唯一落点：`~/.single_shot/runs/exp101_expander_correctness/<0NN_.../>/nd{1,2,3}/`；结果 tar 回本地并校验后清理服务器 scratch（repos/、cache）。CLAUDE.md 服务器规则全文适用。
- 临时不留档文件用会话 scratchpad，不落 exp101/。

---

## 3. 阶段与验收 gate

通用规则：**gate 通过 = 证据文件落盘 + status.md 登记路径与时间戳**。判据不得为了变绿而放宽；确需放宽 → 写入 status.md「待用户决策」升级处理。每轮 loop 只推进一个有界工作块。

### Phase 0 — 侦察与规格（本地，约 1-2 轮）

- **G0.1 主项目接口盘点** → `notes/00_interface_recon.md`：
  `run_disorder_average_simulation` 的输入输出与调用链；PT（sync_enlarge ladder、swap 权重）、q=0 多起点、收敛 gate、observable/weights 计算、NPZ 写出的逐段语义；`production_chunked_scan.py` 与 launcher 约定中可复用的最小子集；**旧 q_top 的 u 范围考证**（basis k 个还是全部 2^k−1 个）；输出移植清单（每个函数标：原样复制 / 改造 / 重写）。
- **G0.2 模型规格推导** → `notes/01_model_spec.md`：
  §1 全部公式的推导版：单发制备两 sector 的映射推导；section-frame 依赖性的精确刻画；Nishimori 恒等式（含测量噪声）的推导与精确形式；Gray-code 全枚举与 coset 枚举算法设计；大 k 抽样估计量（purity/w0）的无偏性论证。
- **G0.3 环境记录** → `notes/02_env.md`：
  本地 conda 12 已验（2026-07-07：py 3.12.12, numpy 2.4.1, numba 0.65.1, ldpc 2.4.1, scipy 1.17.0, matplotlib 3.10.8, pytest 9.0.3）；远端 nd-1/2/3 env 11 的 numba/ldpc import 与核数确认可延至 G4.1，但必须在 status 挂账。

### Phase 1 — expander code 构造模块（spec 全实现，约 3-6 轮）

对应 `../expander_code.md` 的 §1–§11 与测试要求 §10 A–G，**精确 GF(2)，禁浮点线代**：

- **G1.1** `gf2.py`：spec §6 全部函数 + 单测（rank-nullity、nullspace 全验、rowspace membership、quotient 维数、extend_basis）。
- **G1.2** `graphs.py`：`BiregularBipartiteGraph`（含 seed/rng_description/attempts）+ `random_biregular_graph_from_m`（configuration model，`random.Random(seed)`）+ 三个 check 方法 + **确定性构造器**：`cycle_H(m)`（单圈 circulant）、`repetition_H(m)`（路径）、`complete_bipartite(a,b)`；单测 spec §10A（同 seed 复现、异 seed 不同、度数、简单性、一致性）。
- **G1.3** `hgp.py`：`classical_parity_check_matrix` + `quantum_expander_parity_checks_from_graph`（返回顺序 H_Z, H_X）+ 内部通用 `hgp_from_H(H)`（接受任意 F_2 矩阵，供退化码/irregular 测例）+ `verify_css_commutation`；单测 spec §10B,C（形状、行重 d_A+d_B、列重界、H_X H_Zᵀ=0）。
- **G1.4** `expansion.py`：`verify_vertex_expansion`（精确 Fraction、双侧、failing witness）；单测 spec §10G（手工可验小图、失败 witness）；文档记录小 m 时 γ·n_A<1 导致检查为空真的事实。
- **G1.5** `logicals.py`：`logical_pauli_operators`（quotient basis + pairing 矩阵求逆 + 归一到 δ_ij）+ spec §7 全部一致性检查；单测 spec §10E。
- **G1.6** `params.py`：`code_parameters` 精确 [[n,k,d]]（按重量分层的精确搜索，compute_distance 可选）+ **已知码对照测试**（spec §10F）：
  [[13,1,3]]（repetition 2×3 HGP）；2D toric [[2m²,2,m]]，m=2,3,4（cycle-HGP）；K_{4,3} = [[25,13,·]]（验证 k 公式两项非零，d 精确算出）；(3,4) m=2 官方 seed 记录实际 rank/k（预期 k=4）；k=0 边角例 H=[1] 优雅处理。
- **G1.7** `instance.py`：`build_quantum_expander_code_instance`（spec §9 签名）+ 实例序列化（JSON/NPZ：边表、seed、H、校验和）与复现校验 + `examples/spec_example.py`（spec「Expected final deliverable」示例，m=2, d_A=3, d_B=4, seed=12345, gamma=1/10, delta=1/16）跑通并落盘输出。
- **G1.8** `families.py` 官方家族注册：m=1..6 的 seed 选取规则（自 12345 递增取首个满足：简单图构造成功 且 rank H = 3m 满秩）；注册表 JSON（m, seed, attempts, n, k, rank, H 哈希）落盘 validation/。

### Phase 2 — 统计力学 MCMC 管线（自包含移植，约 6-10 轮）

- **G2.1** `model.py` + `observables.py`：disorder 采样（含 common-random-numbers 路径以支持跨 p 复用底层随机数）、能量项、q=0/q>0 两种状态表示；observable 按 §1.4 三档记录约定实现；单测：O_u(c=η)=+1；线性 section 下 O_u ≡ 预计算 mask ⟨w_u,c⊕η⟩；小 k 上全量(b) vs 抽样(c) 一致。
- **G2.2** `section.py`：linear section（默认）+ BpLsd section（ldpc 可用时）；`H_Z r(σ) = σ` 随机化验证；对"给观测 s 取 section"的防误用断言（q>0 路径显式禁止）。
- **G2.3** `reference_mcmc.py`（纯 numpy、直白、无优化技巧）：moves = single-bit（仅 q>0）+ H_X 行翻转（S）+ logical 翻转（L）；**玩具体系精确平稳性测试**：≤4 qubit 玩具 H 上显式构造转移矩阵，验证稳态 = Gibbs（数值线代对照，非采样）；ΔE 与全量重算的 fuzz 一致性。
- **G2.4** `fast_mcmc.py`：numba CSR kernels，与参考实现同 proposal 协议；gate：同 RNG 流 bit 级一致（若协议做不到 bit 级，则降级为统计等价 gate + 两引擎各自对 exact-enum，并在 notes 说明）；无 numba 环境自动回退参考实现。
- **G2.5** `pt.py`：sync_enlarge (p_k,q_k) ladder（swap 权重同时含 data/syndrome 两项；热端 p_k,q_k<0.5 preflight 校验，沿 exp35 经验）；swap 接受率公式单测（两 replica 显式比值对照）；易参数点 PT vs 超长单链一致。**cluster update 不移植**，在 notes 记录理由与替代（bit+S+L+PT）。
- **G2.6** `gates.py` 收敛诊断移植+扩展：多起点 spread（q=0 默认 8 起点；q>0 sector 初始化 = base ⊕ x_u 轮转覆盖，k 大时轮转子集）；**冷端 logical-flip 接受率 per-u（worst-u 判据）**；PT round-trip 计数；O_u 自相关；能量平稳性；「共冻 ≠ 收敛」内建（多起点一致但 sector 零流动 ⇒ flag 不通过）。
- **G2.7** `run_scan.py` 扫描入口：manifest（repo commit SHA、家族实例哈希、**ensemble 标签（true_posterior/repo_compat）**、sector、section frame 指纹、u-set seed、moves 配置、ladder、burn-in/采样量、RNG 协议、numpy/numba 版本、hostname）；NPZ schema 与 `sector_ti_results.npz` 字段名兼容（新增 code_size_list=m 列表；weights 类数组按 §1.4 的 k≤10 全量 / k>10 basis+sampled 布局，布局写进 manifest；新增真类 sig(η) 字段）；断点续采与 chunk 化（借鉴 production_chunked_scan 的必要最小子集，大 m 优先调度）。
- **G2.8**【G0.1 新增】`sector_ti.py` **sector-TI 引擎移植**（exp37 生产路径的泛化）：线性 projection masks（k×n，泛型）；sector 代表与 sector-preserving proposals（零签名单比特 + 同签名对 + H_X 行）；固定 sector 链 + K_p 网格热力学积分 + 粗/细网格 flags + block bootstrap；`q_top=(2^kΣw²−1)/(2^k−1)` 泛化；**系综接线**：repo_compat 喂 δ、true_posterior 喂 s 且真类=sig(η)；k≤10 全 2^k sector；**k>10 basis-pairwise 变体**（k+1 条链：sector 0 与各 e_u → per-u ΔF_u），其观测量性质在 G3 对全 TI/枚举定量标定。numba fast path 同步移植。单测：swap…（不适用）→ 平稳性借 G2.3 框架、ΔF 在 K_p=0 处=0、2 sector 玩具例与手算一致。

### Phase 3 — ground-truth 交叉验证矩阵（约 8-12 轮，可用 nd 节点加速）

- **G3.1** `enumerate_exact.py`：q>0 全空间 Gray-code 增量枚举（n ≤ 28，超界显式拒绝）+ q=0 coset 枚举（dim ker H_Z ≤ 28）→ 精确类分布 / m_u / ⟨E⟩ / 自由能。**独立性对照**：与主项目 `exact_enumeration.py`（只读运行原脚本，不改动）在 2D toric L=2 同模型输入上数值一致——两套独立实现互证。
- **G3.2 V1 主矩阵（枚举 vs MCMC vs TI）**：实例/参数/disorder 覆盖见 §4；**两个系综（true_posterior / repo_compat）都要过**；gate：逐 (disorder, u) 的 z-score 总体 |mean z| ≤ 0.15、|z|>3 比例 ≤ 1%（计多重比较）；类分布 TVD 阈值；⟨E_data⟩、⟨E_syn⟩ 精确对照；q_top/purity/w0 聚合量对照。direct（参考+numba）与 **sector-TI 引擎（含 k≤10 全量与 basis-pairwise 变体的标定）**都要过。
- **G3.3 V1c section-frame A/B**：两组不同 pivot 的 linear section + BpLsd section；**每个 frame 下 enum 与 MCMC 各自一致**（主 gate）；frame 之间 m_u 差异定量记录并与 G0.2 推导对照（物理注记：frame = 规范选择）。
- **G3.4 V2 解析极限**（§1.5 全部落地）：p=0.5 零化；q=0.5 闭式在 m=2,4,6 三个规模（生产规模验证！）；q→0⁺ vs q=0 引擎；p→0⁺ 微扰；disorder=0 快测路径 + 小码 enum 对照。
- **G3.5 V3 Nishimori 恒等式**（三级递进；**只对 true_posterior 系综成立——对 repo_compat 应显式失败，这本身就是系综判别测试**）：
  (i) [[8,2,2]] **全 disorder 求和精确版**（2^8×2^4=4096 组 disorder × 2^8 枚举，零统计误差）验证模型定义与恒等式本身；
  (ii) K_{4,3} 与 [[18,2,3]] 抽样 disorder（≥200）× 精确枚举版（仅 disorder 抽样误差）；
  (iii) (3,4) m=2（n=100，超出枚举界）**全 MCMC 版**，配对 bootstrap z-gate —— 对 disorder 采样+MCMC+观测量整链在枚举界之外的强校验。
- **G3.6 V4 实现冗余 A/B**：reference vs numba；PT-on vs PT-off（易参数点）；1 起点长链 vs 8 起点；不同 RNG 流。全部统计一致。
- **G3.7 V6 冻结扇区 torture**：
  负例：(3,4) m=2..3、小 p、禁 PT 禁 logical-flip ⇒ **诊断必须报警**（测试诊断的灵敏度，不是测试采样器）；
  正例：同参数开 PT+logical-flip ⇒ round-trip>0、冷端接受率达标、结果与初始 sector 无关；per-u 冻结检测在 k=4..9 实测。

### Phase 4 — 服务器规模化验证（nd-1/2/3 并行，约 4-8 轮 + 墙钟等待）

- **G4.1** 远端 env 11 确认（import numba/ldpc、核数在 screen 外探测并烘焙进 runner）+ launcher 移植（遵循 remote-prod-scan checklist：screen、`conda run --no-capture-output`、显式 `--num-workers N` 并在日志确认 workers=N、load≈N）+ 单节点分钟级 smoke 全往返（提交→健康检查→回收→校验和→清 scratch）。
- **G4.2** 性能 profile：m=2..6 sweep 速率、numba 生效确认、内存；估算 m=6 全 PT 生产单 disorder 成本；**验收线：与 3D L=7 生产点同量级**（profile 后把具体数字钉进 status；超线则先优化，仍超线升级用户决策）。profiler 默认关闭逐环节 sector-signature 诊断（CLAUDE.md 性能坑）。
- **G4.3** 多节点一致性与可复现：同参数跨 nd-1/2/3 统计一致；同 seed 同机重跑 bit 级一致；断点续采（截断-恢复 vs 直跑）一致。
- **G4.4** mini 端到端物理烟测（粗网格低统计，非生产）：
  (a) 2D toric（cycle-HGP，m=3,4,5）q=0：crossing ∈ [0.09, 0.12]（文献 RBIM Nishimori p_c ≈ 0.109）——文献级 end-to-end 校验；
  (b) 2D toric 单轮 q>0：随 m 增大的退化趋势（定性：2D 无 single-shot）；
  (c) expander (3,4) m=2,3(,4)：q=0 crossing 存在性；固定小 p 扫 q 的 q_top 行为 sanity；threshold 方向判读符合 CLAUDE.md 规则（p<p_c 时大码 q_top 更大）；
  (d) 上述所有 run 收敛 gate 全绿（含冷端 logical-flip 判据）。
- **G4.5** 服务器目录规范核查（只用 `~/.single_shot/runs/exp101_*`）+ tar 回本地 validation/ 编号目录 + 校验和 + 清理服务器 scratch。

### Phase 5 — 毕业（约 1-2 轮）

- **G5.1** 全 gate 审计 → `report.md`：验证矩阵结果总表、已知局限（frame 依赖、K_{4,3} 特殊性、枚举界等）、exp102 生产建议（参数网格、disorder 数、预算估计、节点分配）。
- **G5.2** `笔记/实验报告.md` 增量条目（中文、简洁、带时间戳）。
- **G5.3** CLAUDE.md 增补 expander 新坑（按维护规则就地改写优先）；跨会话关键结论存 memory。
- **G5.4** git 提交推送：只加 exp101 代码/文档/小体积证据（JSON/MD/PNG）；大 NPZ 不入库（沿现状：本地 data/ + 服务器 runs/ 备份）。

---

## 4. 验证矩阵总表

### 4.1 精确枚举实例（V1 用；参数为满秩预期值，以构造模块实际输出为准）

| 实例 | 构造 | [[n,k]] | q>0 全枚举 2^n | q=0 coset dim | 备注 |
|---|---|---|---|---|---|
| 2D toric m=2 | cycle-HGP | [[8,2]] | 2^8 | 5 | 兼作 Nishimori 全 disorder 求和 |
| [[13,1,3]] | repetition-HGP | [[13,1]] | 2^13 | 7 | 标准 surface code 参照 |
| 2D toric m=3 | cycle-HGP | [[18,2]] | 2^18 | 10 | |
| 随机 H(2×4) | hgp_from_H | [[20,4]] | 2^20 | 12 | 非正则，测泛化性 |
| K_{4,3} | (3,4) m=1 | [[25,13]] | 2^25 | 19 | 大 k 强测例；官方家族成员 |

### 4.2 参数与 disorder 覆盖（V1 每实例）
- (p,q)：≥12 个组合 ⊂ {0.02, 0.08, 0.15, 0.30} × {0（q=0 引擎）, 0.01, 0.05, 0.15, 0.30}；
- disorder ≥ 12/点：随机 + 强制 (η=0, δ=0) + 手工 plant 的近类边界样本（对 bug 最敏感）；
- 对偶 sector：至少 1 个实例全覆盖走 Z-错误/H_X 路径（对偶正确性）。

### 4.3 手段 × 校验对象

| 手段 | 校验对象 | 规模上限 |
|---|---|---|
| V0 单元测试（spec §10 A–G + MCMC 内核平稳性） | 各模块局部正确性 | — |
| V1 精确枚举（+独立实现互证） | 端到端分布与全部观测量 | n ≤ 28 |
| V1c section-frame A/B | frame 约定与不变性 | n ≤ 28 |
| V2 解析极限（p=0.5, q=0.5 闭式, 极限一致性） | observable + 采样器 | 任意（含 m=6） |
| V3 Nishimori 恒等式（三级递进） | disorder 采样+MCMC+观测量整链 | n=100（MCMC 版） |
| V4 实现冗余 A/B（numba/PT/多起点/RNG） | 工程实现 | 任意 |
| V5 已知码对照（[[n,k,d]] + 2D toric 文献阈值） | 构造模块 + 端到端物理 | m ≤ 5 烟测 |
| V6 冻结扇区 torture（负例必须报警） | 收敛诊断有效性 | m ≤ 3 |
| V7 规模/复现/多节点（Phase 4） | 工程正确性与性能 | m ≤ 6 |

---

## 5. 运行与提交规范

- 本地统一 conda `12`；远端计算节点 env `11` + screen 后台 + `conda run --no-capture-output`；**conda run 禁 heredoc**（本会话已复现该坑），复杂脚本先写 .py 再跑。
- `--num-workers` 显式传值（screen 内 `$(nproc)` 不可信），日志确认 workers=N、load≈N。
- q=0 不传任何 `--pt-*`；PT 仅 q>0；多起点确认 8；sync_enlarge 热端 p_k,q_k < 0.5。
- 所有留档产物在 exp101/ 内（validation/ 编号目录）；scratch 用会话 scratchpad；服务器只用 `~/.single_shot/runs/exp101_*`，回收校验后清理。
- git：每个 phase 全绿后提交一次，message 前缀 `exp101 phase-N:`，只 add 相关文件（禁 `git add .`）；大 NPZ 不入库。
- 主项目 `src/` 与 `data/3d_toric_code/` 一律只读。

---

## 6. 风险与既知坑（exp101 特有；通用坑以 CLAUDE.md 为准）

1. **大 k observable**：k>10 禁全量 2^k 路径（内存/时间保护，程序硬性检查）；purity/w0 用 U_rand 抽样无偏估计，小 k 交叉验证；q_top_basis 与旧 q_top 的 u 范围差异在 G0.1 考证并在 report 注明。
2. **section-frame（q>0）**：m_u 依赖 section 规范；固定 frame 入 manifest；跨 run/跨引擎/跨枚举比较必须同 frame（G3.2 的枚举与 MCMC 必须用同一 section 数据结构）。
3. **冻结扇区 × k 个 logical**：任一 u 都可能单独冻结 ⇒ per-u 判据取 worst-u；sector 初始化轮转覆盖；V6 负例保证诊断灵敏。
4. **(3,4) 随机图 rank 缺陷** ⇒ k≠m²：families 注册表钉死满秩 seed；实例序列化 + 哈希校验，禁止运行时重采图。
5. **expansion 精确验证**：小 m 空真（γ·n_A<1）、大 m 指数爆炸——仅作构造模块正确性测试，不作为物理前提；(3,4) 在 spec 示例 (γ=1/10, δ=1/16) 下大 m 预计不通过，属预期行为，如实记录。
6. **精确枚举规模界**：full 2^n 要求 n≤28、coset 要求 dim≤28，超界显式报错而非静默截断。
7. **K_{4,3} 特殊性**：d=2、k=13，类间壁垒低——是构造/观测量的好测例，但物理烟测以 m≥2 为准。
8. **版本漂移**：manifest 记 numpy/numba/commit；远端先 import 检查再跑（不在生产任务中临时装包）。
9. **cluster update 缺席的混合代价**：expander 无 cluster 加速，深有序相混合更依赖 PT 与 logical-flip；若 Phase 4 发现混合瓶颈，优先调 ladder/moves 配比，不引入未经验证的新算法。
10. **q>0 sector 初始化**：只用零 syndrome sector representatives（base ⊕ x_u），不对观测 syndrome 求 representative（沿用 CLAUDE.md 的坑）。
11. **系综开关（G0.1 新增）**：true_posterior 与 repo_compat 的差别只在 syndrome 参数接线（s vs δ）与真类标签（sig(η) vs 0）——所有 run 的 manifest 必须带 ensemble 标签；跨 run 比较必须同系综；Nishimori 检验只约束 true 系综。生产默认 true_posterior（待用户确认，见 status D1）。
12. **TI 大 k 变体（basis-pairwise）是新估计量** — 【2026-07-09 G3.2 已标定：**判定失效，弃用为 q_top 方法**】。K43(k=13) 上 pairwise m_u 对 exact m_u 的 max 偏差达 0.8–1.45（可加性严重失效），远超任何可用界。**大 k 生产 q_top 改走 direct/PT 采样观测量**（V2b k=16/36 + V1 direct K43 已验）。pairwise 仅保留"正确测量单翻转 ΔF_u"的诊断用途，禁止合成 q_top。full-TI 仅 k≤10 有效。详见 status D4。

---

## 7. 毕业判据（全部满足才标 DONE）

1. G0–G5 全部 gate 绿，status.md 每行有证据路径，可追溯。
2. tests/ 快速套件本地 conda 12 全绿；重型验证均有 validation/ 编号目录证据。
3. V1–V7 无未解释的红/黄项；所有判据妥协均经用户确认并留痕。
4. report.md + 实验报告增量 + CLAUDE.md 增补 + git push 完成。
5. `run_scan.py` 能以一条命令在 nd 节点跑通 m≤6 的合法生产点（exp102 直接复用的最终检验）。

---

## changelog
- 2026-07-07 初版（规划会话；4 项用户决策已确认并写入 §0）。
- 2026-07-08 G2 执行期修订：①G2.8 与 G2.7 顺序互换（先 TI 引擎后扫描入口，使 run_scan 一次覆盖两引擎）；②G2.8 的 numba TI kernel 子项挂账至 G4.2 性能线（python 参考版为正确性权威，正确性判据全绿后不阻塞 Phase 3）；③G2.6 的收敛 gate 增加符号敏感 m_u_spread 判据（q_top spread 被证符号盲——不同 sector 共冻给出相同 q_top）；④pairwise TI 定义细化：链锚定在 {ℓ_ref, ℓ_ref⊕e_u}（true 系综正确锚点，repo_compat 自动退化为 {0,e_u}）。
- 2026-07-09 **G3.2 关键发现：pairwise-TI 大 k 方法失效**。测得 K43(k=13) pairwise m_u vs exact m_u max 偏差 0.8–1.45（可加性失效），planned k>10 方法作废；**大 k q_top 改用 direct/PT 采样观测量**（已由 V2b/V1-direct 验证）。风险 12 更新为"已标定失效"，status D4 记录生产策略含义。此为 exp101 的核心价值兑现：在烧生产算力前发现 planned 方法不可用并定位可用方法。
- 2026-07-09 G3.2 V1 执行期修订（**regime-aware 验证**，首轮 naive 设计暴露方法学问题）：首轮把 bare 单链 direct 引擎跑遍全网格（含 q=0 与超冷点）并用纯 z-gate 判定，结果 A/B/C/D 全红。分析（validation/004 首轮 results.json）确认**非 bug 而是工具/instrument 错配**：(a) **q=0 与超冷点的 sector 权重单链原理上取不到**（q=0 无 single-bit、logical move 冻结 → 只见一个 sector；PT 在 q=0 不可用）——必须由 TI 覆盖；(b) **m_u≈±1 饱和时 block-stderr 塌缩**，纯 z 检验失真（需 z-OR-绝对判据）；(c) **TI 的 raw-ΔF-z 是错误 instrument**（ΔF 大时权重≈0，网格离散误差不进 bootstrap stderr；正确判据是物理量 q_top/weight-TVD——数据佐证：(0.15,0)处 ΔF-z=7 但 weight-TVD=0.05、q_top 全过）。**修正后 V1 设计**：direct 仅在自证遍历区（worst-u 冷端接受率 ≥ 0.02）用 z-OR-绝对判据；新增 **PT-vs-枚举** 覆盖 q>0 冷点（验证 PT 解冻 sector）；TI 用物理量判据（q_top 绝对 + weight-TVD）覆盖含 q=0 全谱、冷点给足配置；每 regime 由其对应工具验证、各工具在其有效域内被检验。判据不放宽，只换到正确 instrument 与正确工具。
- 2026-07-07 G0.1 修订：①判决发现主项目模型为 δ-only 盘度（证据 validation/001），§1.2 引入 true_posterior/repo_compat 双系综开关；②考证旧 q_top = 全部非零 u 均值（=归一化 purity），§1.4 更新；③新增 G2.8 sector-TI 引擎移植（exp37 生产路径泛化 + 大 k basis-pairwise 变体）；④G3.2/G3.5 扩为双系综与 TI 覆盖；⑤§6 新增风险 11/12。详见 notes/00_interface_recon.md。
