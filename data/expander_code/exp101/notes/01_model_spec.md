# notes/01 — 模型规格与推导（G0.2）

日期：2026-07-07。本文是 exp101 管线实现的**数学权威**：所有实现以此为准；与 notes/00（接口盘点）、validation/001（系综判决）互为支撑。

---

## 1. 记号与空间分解

CSS 码 (H_X, H_Z)，n 个 qubit，H_X H_Zᵀ = 0（F_2）。**主 sector：X 错误 / H_Z checks**（m_Z = n_A n_B 行）。

- S := row(H_Xᵀ 的像) = X-stabilizer 支撑集空间 ⊆ ker H_Z，dim ρ_X。
- ker H_Z = S ⊕ L，L = span{x_1..x_k}（logical X 代表），k = n − ρ_X − ρ_Z。
- {z_1..z_k} ⊆ ker H_X（logical Z 代表），配对归一 ⟨x_i, z_j⟩ = δ_ij（构造模块保证）。
- section r: im(H_Z) → F_2^n，H_Z r(σ) = σ。线性 section：r(σ) = Rσ。T-空间（frame 依赖）：T_R := im(R)，F_2^n = T_R ⊕ S ⊕ L。
- **label 泛函**：φ(c) := (⟨z_u, c ⊕ r(H_Z c)⟩)_{u=1..k} ∈ F_2^k。线性 section 下 φ(c)_u = ⟨w_u, c⟩，其中 **w_u := (I ⊕ R H_Z)ᵀ z_u**。
- **w_u 的代数性质**（观测量正确性核心，实现必须断言）：
  (i) ⟨w_u, Rσ⟩ = ⟨z_u, Rσ ⊕ RH_ZRσ⟩ = 0 ∀σ —— **湮灭 T_R**；
  (ii) ⟨w_u, S⟩ = ⟨z_u, S⟩ = 0 —— **湮灭 S**（CSS 对易）；
  (iii) ⟨w_u, x_v⟩ = δ_uv —— **L 坐标读出**。
  ⇒ φ(c) = c 在 frame R 分解下的 L 坐标。

K_p := log((1−p)/p)，K_q := log((1−q)/q)。Nishimori：Gibbs 温标与 disorder 生成同 (p,q)。

## 2. 单发制备 → 统计力学映射（推导）

**|0̄⟩ 制备 ↔ X-error/H_Z sector（主 sector；修正：规划期问答里误写为 |+̄⟩，以本推导为准）**：
初始 |0⟩^n 被全体 Z_j 稳定 ⇒ 全部 H_Z-check 恒 +1、全部 Z̄_u 恒 +1。制备过程中 X 错误 η ~ Bern(p)^n 翻转 Z-check 真值为 (−1)^{(H_Zη)_i}；一次带噪读出得 s = H_Zη ⊕ δ，δ ~ Bern(q)^{m_Z}。按 section/decoder 施加修正 X^{t}，t = r(s)（**frame = 实际修正协议**）。残余 X 错误 = η ⊕ t，其 logical-X 类翻转 Z̄ ⇒ 制备的逻辑失败类 = φ(η⊕r(s)) 所在类。（同时需测 X-checks 定 X-syndrome，其随机初值由 Z-型修正固定，Z̄ 类作用于 |0̄⟩ 平凡、其读出噪声只留下无害的 Z 残余——不进本 sector 的逻辑失败，构成对偶 sector 的同构问题。）
**|+̄⟩ 制备 ↔ Z-error/H_X sector（对偶）**：全部互换 (H_Z↔H_X, x↔z)。管线以参数支持。

信息论最优（MAP-on-classes）版本即后验类分布：给定 s，
`P(c|s) ∝ P(η=c)·P(δ=s⊕H_Zc) = exp[−K_p|c| − K_q|H_Zc⊕s|] =: π*(c)`。

## 3. 两个系综（权威定义）

**E1 true_posterior（生产默认）**：采 (η,δ)，s=H_Zη⊕δ；Gibbs = π*(c) ∝ exp[−K_p|c| − K_q|H_Zc⊕s|]；观测量相对 η。等价换元（c=η⊕x）：exp[−K_p|x⊕η| − K_q|H_Zx⊕δ|]——数据项 η-盘度 + syndrome 项 δ-盘度（文献标准 quenched 形式）。
**E2 repo_compat（兼容）**：主项目模型 exp[−K_p|c⊕η| − K_q|H_Zc⊕s|] ≡（换元）exp[−K_p|u| − K_q|H_Zu⊕δ|]：**只有 δ 盘度**（判决见 validation/001：T1/T2/T5）。
**桥梁恒等式（T3，精确）**：`m_u^{E1}(η,δ) = (−1)^{φ(η)_u}·m_u^{E1}(0, δ:=s)`。
⇒ 实现上两系综共用同一采样/TI/枚举内核，仅差：syndrome 参数接线（E1 喂 s；E2 喂 δ）与真类标签（E1: ℓ_η=φ(η)；E2: 0）。**manifest 必须记 ensemble。**

q=0：E1 = coset 上 exp[−K_p|c|]（quenched，η-盘度真实存在）；E2 = coset 上 exp[−K_p|z|]（clean，与 disorder 无关）。

## 4. 观测量、q_top 与 frame 依赖

对 u ∈ F_2^k（z_u := ⊕_{i∈u} z_i）：
`O_u(c;η) = (−1)^{⟨z_u, c⊕η⊕r(H_Zc)⊕r(H_Zη)⟩} ≡ (−1)^{φ(c)_u ⊕ φ(η)_u}`，m_u := ⟨O_u⟩_{Gibbs}。
类权重 w_ℓ := P(φ(c)=ℓ)，真类 ℓ_η = φ(η)；相对类分布 P̃(ℓ) := w_{ℓ⊕ℓ_η}。

**恒等式（实现与估计量的依据）**：Σ_{u∈F_2^k}(−1)^{⟨u,ℓ⟩} = 2^k δ_{ℓ,0} ⇒
- m_u = Σ_ℓ P̃(ℓ)(−1)^{⟨u,ℓ⟩}；
- purity := Σ_ℓ P̃(ℓ)² = 2^{−k} Σ_{u∈F_2^k} m_u²（含 u=0，m_0=1）；
- **q_top := mean_{u≠0} m_u² = (2^k·purity − 1)/(2^k − 1)**（= 主项目定义的泛化；TI 侧即 (2^kΣw²−1)/(2^k−1)）；
- w0 := P̃(0) = 2^{−k} Σ_u m_u。

**frame 依赖（精确刻画）**：换 section r→r′（Δ(σ)=r⊕r′∈ker H_Z）⇒ φ′(c) = φ(c) ⊕ λ(H_Zc)，λ(σ) := Δ(σ) 的 L 坐标。
- q>0：不同 syndrome 扇区被 λ 重排 ⇒ m_u、w_ℓ、q_top、w0 均 frame 依赖。frame = 修正协议：不同 frame 是不同制备协议的成功率，非 bug；**每 run 固定并记录 frame 指纹，跨 run 同 frame 才可比**（V1c 定量测差异）。
- q=0：所有 c 同 syndrome ⇒ λ(H_Zc)⊕λ(H_Zη) 为常数 0 ⇒ **q=0 一切类量 frame 无关**（V1c 应精确验证此点）。

**估计量（大 k）**：U_rand = 均匀随机非零 u 集（大小 N，seed 入 manifest）。
- q̂_top = mean_{u∈U_rand}(m_u²)：对 u 无偏（条件于精确 m_u）；MCMC 的 m̂_u² 有 +Var(m̂_u) 偏差 ⇒ 用分块（block）内外积差或 bootstrap 去偏，与主项目 q_top block 处理同规格。
- ŵ0 = 2^{−k} + (1−2^{−k})·mean_{U_rand}(m_u)：无偏（m̂_u 线性）。
- k ≤ 10：全 2^k 路径并行输出，用于与抽样路径交叉（G2.1/G3.2 gate）。

## 5. Nishimori 恒等式（E1 精确证明；E2 判别）

设任意固定 frame 的 φ，m(η,δ) := (−1)^{φ(η)_u}·⟨(−1)^{φ(c)_u}⟩_{π*(·|s)}，s=H_Zη⊕δ。
关键观察：P_p(η)P_q(δ) = C·e^{−K_p|η| − K_q|H_Zη⊕s|}（δ = s⊕H_Zη），C=(1−p)^n(1−q)^{m_Z}。对固定 s，对 η 求和：
- Σ_η e^{−K_p|η|−K_q|H_Zη⊕s|}·(−1)^{φ(η)_u} = Z(s)·⟨(−1)^{φ_u}⟩_s；
- Σ_η e^{−K_p|η|−K_q|H_Zη⊕s|} = Z(s)。
⇒ `E[m_u] = C Σ_s Z(s)·⟨(−1)^{φ_u}⟩_s² = E[m_u²]`。∎
同法（用 δ_{ℓℓ′} 版本）：`E[w0] = E[Σ_ℓ w_ℓ²] = E[purity]`；一般地 E[m^{2t−1}] = E[m^{2t}]。
**适用**：任意码、任意 sector、任意固定 section frame、任意 (p,q)（Nishimori 匹配）。
**E2 显式失败**（判别测试）：q=0.5 处 E2 有闭式 E[m_u]=(1−2p)^{|w_u|} ≠ E[m_u²]=(1−2p)^{2|w_u|}。V3 对 E1 过、对 E2 应失败。
（观测 syndrome 分布恒等式 P(s) = C·Z(s)：Nishimori 下 s 的边际正比配分函数——可作额外一致性检查。）

## 6. 解析极限（V2 的权威公式，E1/E2 分别给出）

- **p=0.5**（K_p=0）：两系综所有 m_u = 0（L 方向均匀；证明：Gibbs 权重对 c→c⊕x_v 不变）。
- **q=0.5**（K_q=0）：E2：c=η⊕e，e~Bern(p) iid ⇒ m_u=(1−2p)^{|w_u|}（线性 frame，精确含符号）。E1：π*(c) = Bern(p) iid（syndrome 项消失）⇒ ⟨(−1)^{φ(c)_u}⟩ = (1−2p)^{|w_u|} ⇒ m_u = (−1)^{φ(η)_u}(1−2p)^{|w_u|}；E[m_u] = (1−2p)^{2|w_u|}（对 η 平均后）✓ 与 §5 一致。⟨|c|⟩=np（E1）/⟨|c⊕η|⟩=np（E2）。
- **q→0⁺ vs q=0**：条件 δ=0：E1 的 q>0 引擎（s=H_Zη）与 q=0 硬约束引擎一致；E2 同理。
- **p→0⁺**：E1 集中于 c=η…m_u→(−1)^{φ(η)⊕φ(η)}=+1；E2 集中于 u=0 同样 +1。一阶微扰（δ=0 盘度）可展开对照。
- **disorder=0**（η=0,δ=0 ⇒ s=0）：E1≡E2（同一模型），保留快测路径并作两系综实现互证点。

## 7. sector-TI 理论（含大 k 变体）

固定 label 扇区 ℓ 的受限配分函数 Z_ℓ(K_p) := Σ_{c: φ(c)=ℓ} e^{−K_p|c| − K_q|H_Zc⊕σ_arg|}（E1: σ_arg=s；E2: σ_arg=δ）。
- dF_ℓ/dK_p = ⟨|c|⟩_ℓ =: μ_ℓ(K_p)（E2 时 |c|→|u|）⇒ ΔF_ℓ(K_p*) = ∫_0^{K_p*}[μ_ℓ − μ_0]dK_p（trapezoid，粗/细网格差为 flags）。
- **端点条件**：K_p=0 时 Z_ℓ 与 ℓ 无关（c→c⊕x_v 保 syndrome 项）⇒ ΔF_ℓ(0)=0 ✓ 积分常数固定（实现断言）。
- w_ℓ = softmax(−ΔF_ℓ)；q_top = (2^kΣw²−1)/(2^k−1)；w0 = w_{ℓ_η}（E1）/ w_0（E2）；m_u = Σ_ℓ w_ℓ(−1)^{⟨u,ℓ⟩}（已相对 ℓ_η 移位后）。
- 固定扇区链的提议集 = 零签名单比特 ⊕ 同签名 qubit 对 ⊕ H_X 行（签名 = w-列模式；泛型，exp37 同构）。可选：每逻辑类多代表构成同签名组做组内 heatbath（exp37 even-winding 的泛化）。
- **k>10 basis-pairwise 变体**：只跑 k+1 条链（ℓ ∈ {0, e_1..e_k}），得 ΔF_u := F_{e_u} − F_0，定义
  `m_u^{pair} := tanh(ΔF_u/2)`，`q_top^pair := mean_u (m_u^{pair})²`。
  **精确关系**：若扇区自由能可加（F_ℓ = Σ_u ℓ_u ΔF_u，即类分布按 k 个 label 位因子化），则 m_u^{pair} = m_u 精确成立。非可加性 = 系统偏差来源；小 k 实例上用全 2^k ΔF 直接检验可加性并标定偏差（G3.2 的一部分）。**未经标定不得在生产外推**（plan 风险 12）。

## 8. 精确枚举算法（V1 的设计）

**(W_p, W_s, ℓ) 计数表 + Gray code**（比主项目逐 (p,q) 全扫更强）：
- 全空间版（n ≤ 28）：按 Gray 序遍历 c ∈ F_2^n，每步翻 1 bit：增量维护 W_p ∈ {|c|(E1) 或 |c⊕η|(E2)}、syndrome 向量与 W_s = |H_Zc⊕σ_arg|（列邻接 XOR）、φ(c)（k ≤ 13 时全类标签，w-列位模式 XOR）。累加 int 计数表 N[W_p][W_s][ℓ]（(n+1)×(m_Z+1)×2^k；K_{4,3} 为 26×13×8192 ≈ 22MB ✓）。
- **一表通吃**：任意 (K_p,K_q) 的 Z_ℓ、μ_ℓ、⟨E⟩、m_u、w_ℓ、purity 全部由同一张表精确求值（logsumexp over 网格单元）⇒ 一次枚举覆盖整个 (p,q) 网格 + TI 的 K_p 曲线精确对照（μ_ℓ(K_p) 逐点精确！）。
- coset 版（q=0，dim ker H_Z ≤ 28）：c = r(s) ⊕ Gray(ker 基组合)，表 N[W_p][ℓ]。
- 超界显式报错。numba 加速；纯 python 参考版本互证（G3.1 还与主项目 exact_enumeration.py 只读互证——注意其为 E2 语义 + BpLsd frame，对照时匹配系综与 frame）。

## 9. 物理解释风险注记（不改变 exp101 范围；exp102 与用户须知）

q>0 时 label 扇区间存在 **O(1) 代价的局域混合通道**：任意 j ∈ supp(w_u) 的单比特翻转即改变 φ_u（代价 K_p + K_q·col_w(j)，与系统尺寸无关）⇒ 稀释气体估计给出 ΔF_u ≈ [K_p + K_q·col_w] − log|supp(w_u)| 量级的**上界饱和**，随 n 缓慢下降（−log n）。界面/距离保护（ΔF ~ 随 d 增长）只在 q=0 严格成立。含义：
1. 单轮 q>0 的 q_top/w0 crossing 有随 n 缓慢漂移的可能（crossover 而非渐近相变的风险）；生产必须显式检查 crossing 随尺寸的漂移（qc-crossing 分析时对 (m₁,m₂) 对逐对出值并看趋势）。
2. per-u 的 ΔF_u(n) 标度本身是最干净的观测量（直接量化保护强度），NPZ 全存。
3. 该注记同样适用于 3D 项目的 δ-only 结果（D2 的补充材料）。
此项与"程序正确性"无关——exp101 的 gate 全部是与精确 ground truth 的一致性，不依赖热力学极限解释。

## 10. 实现联动清单

- model.py/observables.py：按 §3/§4 实现 ensemble 开关、φ、w_u（断言 §1(i)-(iii)）、三档记录、去偏 q_top。
- sector_ti.py：按 §7，σ_arg 接线 + ΔF(0)=0 断言 + pairwise 变体。
- enumerate_exact.py：按 §8 计数表设计。
- gates.py：per-u 判据（worst-u）；V3 用 §5 公式。
- run_scan manifest：ensemble、frame 指纹、U_rand seed、ℓ_η。
- plan §1.2 对偶 sector 叙事修正：X/H_Z ↔ |0̄⟩ 制备；Z/H_X ↔ |+̄⟩ 制备（本文 §2）。
