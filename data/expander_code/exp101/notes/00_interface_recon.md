# notes/00 — 主项目接口盘点与移植清单（G0.1）

日期：2026-07-07。基准 commit：`d1bd063`。本文档是 exp101 自包含移植的依据；引用行号以该 commit 为准。

---

## 0. 结论摘要（先读这里）

1. **主项目有两条独立引擎**：
   - **direct-measurement 引擎**（`main.py: run_disorder_average_simulation`）：采样链上直接测 O_u，支持 q=0（零 syndrome moves + 8 起点）与 q>0（single-bit + PT + cluster + winding）。
   - **sector-TI 引擎**（`exp37_sector_ti.py`）：对 2^k 个 logical sector 各跑一条**固定 sector**链，对 K_p 从 0 积分到目标值（热力学积分）得 ΔF_sector → sector weights → q_top。**exp40/41 生产相图全部由此引擎产出**（`sector_ti_results.npz`，manifest: mode=sector_ti, projection_mode=linear）。TI 从 K_p=0（即 p=0.5，各 sector 自由能相等）积分，天然绕开冻结扇区问题——这是它成为生产路径的原因。
2. **模型语义判决（重大发现，机器精度确认）**：主项目两条引擎一致实现的 Gibbs 模型为
   `π(v) ∝ exp[−K_p|v⊕η| − K_q|H_Z v⊕s|]`，换元 u=v⊕η 后 ≡ `exp[−K_p|u| − K_q|H_Z u⊕δ|]`：
   **数据错误 η 不进入 Gibbs 权重（线性 frame 下 m_u 与 η 严格无关），盘度只有测量噪声 δ**。
   标准 decoding posterior 应为 `π*(c) ∝ exp[−K_p|c| − K_q|H_Z c⊕s|]`（换元后 `|x⊕η| + |H_Zx⊕δ|`，双盘度）。两模型数值差异巨大（2D toric L=3, p=0.15, q=0.1：q_top 0.82 vs 0.15）。
   证据：`validation/001_model_semantics_check_20260707/`（T1–T5 全过，含与 repo `compute_exact_logical_observable_means` 的逐位互证）。
   **桥梁恒等式（T3，精确）**：`m_u^{true}(η,δ) = (−1)^{⟨w_u,η⟩} · m_u^{true}(η=0, syndrome-arg=s)`，其中 s=H_Zη⊕δ。即：**同一套机器把 syndrome 参数从 δ 换成 s、真类标签从 0 换成 sig(η)，就精确计算 true 模型**。移植代价≈0。
3. **旧 q_top 的 u 范围考证（G0.1 挂账问题）**：全部 2^k−1 个非零 u 的 m_u² 均值（direct 引擎 masks 含全部组合；TI 引擎 `_q_top_from_weights = (8Σw²−1)/7` = 同一量的 sector-weights 表达 = 归一化 purity `(2^kΣw²−1)/(2^k−1)`）。不是 basis-only。expander 大 k 用随机 u 抽样无偏推广同一定义。
4. **k=m² 的爆炸点**：主项目多处硬编码枚举 2^k（observable masks、sector representatives、TI 的 2^k 条链、q0 起点标签、类权重数组）。k>10 全部需要守卫 + 替代路径（basis+抽样 u；TI 的 sector 子集变体）。
5. 核心数值模块（mcmc/preprocessing/linear_section/exact_enumeration/PT/diagnostics/gate）**全部对任意 H_Z + 逻辑基泛型**，可直接复制改造；lattice 特定的只有 `build_toric_code_examples.py` 的构造器、TI 的 even-winding heatbath 分组、`_build_disorder_uniforms` 的坐标哈希模式、cluster update（不移植）。

---

## 1. 模型语义判决详情

### 1.1 代码事实（逐位确认）
- `mcmc.py:110`：`current_data_term_bits = current_chain_bits ^ disorder_data_error_bits`；能量 = `|data_term|·log_odds_p + |syndrome_term|·log_odds_q`，其中 `syndrome_term = H_Z·chain ⊕ s`（`mcmc.py:117-119`，`main.py:317-340 _compute_total_log_weight`）。
- `exact_enumeration.py:149`：`data_term_bits = chain ^ η`，同一权重形式；MCMC 与枚举在 repo 内互相回归验证过 → 两者是同一模型。
- `exp37_sector_ti.py:1570-1573`：TI 链变量 x 初始化为 sector 代表，`syndrome_term = H_Z x ⊕ measurement_error_bits(δ)`，data 权重 = `|x|` → 与上面同一模型的 u-表述（数据项干净、syndrome 盘度=δ）。η 在 linear 模式下只被记录（`eta_weight`），不进权重。

### 1.2 判决实验（validation/001，2D toric L=2 与 L=3 全枚举）
| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | repo 模型 m_u 对 η 的依赖（linear frame，固定 δ） | 最大偏差 ~1e−15 → **严格无关** |
| T2 | 本脚本 repo-模型复算（bplsd frame） vs repo `compute_exact_logical_observable_means` | ~1e−14 → **代码读法正确** |
| T3 | true 模型恒等式 m(η,δ) = (−1)^{⟨w,η⟩}·m(0,s) | 0 → **精确成立** |
| T4 | repo vs true 差异（H_Zη≠0） | q_top: L2 0.28/0.07，L3 0.82/0.15 → **量级差异** |
| T5 | q=0 限制：repo η-无关（clean）/ true η-依赖（quenched） | 0 vs 0.82-0.87 → **确认** |

### 1.3 物理含义与旁证
- repo 系综 = true 模型在 disorder 切片 (η=0, s=δ) 上的平均：q=0 时是**无盘度 clean 模型**（3D 对偶 Ising，p_c≈0.391），一般 (p,q) 时 p 只作为温度 K_p 进入、不作为盘度。
- 旁证：memory 记录 exp39 "Δf-gap FSS 外推 → 0.40" ≈ clean 3D 值 0.391；被当作"有偏（≠真值 0.233=3D RBIM）"——但 repo 模型在 q=0 本来就不是 RBIM，0.233 的预期可能是误配。exp41 的 q_c(p) 平台 0.033–0.035 ≈ RPGM 类（δ-盘度 gauge 模型）数值——与 δ-only 语义自洽。
- **对 exp101 的决定**（详见 status「待用户决策」D1）：exp101 主模型 = **true posterior 系综**（state-prep 的 decoding 物理所需：η~Bern(p) 与 δ~Bern(q) 双盘度、Nishimori 线上 E[m]=E[m²] 成立）；同时保留 **repo 兼容系综**（δ-only）作开关，用于与 3D 时代结果/机器对拍。两者共享全部代码，仅差 syndrome 参数接线（δ vs s）与真类标签（0 vs sig(η)）。
- **对 3D 项目的 heads-up（不属 exp101 范围）**：exp40/41 相图是 δ-only 系综的相图；若要作为 decoding 阈值引用需重新审视。已记录在 status，供用户定夺。

---

## 2. 两条引擎的接口规格

### 2.1 direct-measurement 引擎（main.py）
- 入口 `run_disorder_average_simulation(parity_check_matrix, dual_logical_z_basis, syndrome_error_probability, data_error_probability, num_disorder_samples, num_burn_in_sweeps, num_sweeps_between_measurements, num_measurements_per_disorder, seed, zero_syndrome_move_data=None, q0_num_start_chains=4, num_start_chains=None, num_replicas_per_start=1, pt_*=..., num_zero_syndrome_sweeps_per_cycle, winding_repeat_factor, winding_plane_heatbath_sweeps, single_bit_proposal_fraction, observable_temperature_mode, q_top_block_count, q_positive_initial_chain_mode="sector", cluster_*, precomputed_*_uniform_values_per_disorder)`（main.py:2842）。**完全泛型**：只吃 H_Z(bool, checks×qubits) + 逻辑 Z 基 (k×n)。
- 每 disorder 编排 `_run_single_disorder_measurement`（main.py:2318）：burn-in → 测量循环（`_run_measurement_update_cycle` = single-bit 扫 + 零 syndrome moves + winding heatbath）→ 每 measurement 用 `_accumulate_logical_observables_fast` 记 O_u（增量 syndrome 缓存 + section）。q=0 时 single-bit 关闭（`_resolve_single_bit_attempts_per_cycle` 返 0），只走零 syndrome moves。
- **zero_syndrome_move_data dict 约定**（family 接口，expander 适配的核心落点）：
  - `contractible_moves` (n_moves×n bool) + `contractible_move_supports` (int32)：S-空间局部闭环（toric=vertex star=H_X 行）。
  - `winding_moves` + `winding_move_supports`：L-空间代表（toric=每方向每平移线各一条，共 dL 条——**多副本提高接受率**，签名相同的副本还构成 even-winding 组）。
  - `start_sector_generators` (g×n bool)：k 个独立 logical 代表，用于 2^g 起点标签（`_build_q0_start_sector_labels` 枚举 2^g——**k>10 需换成子集轮转**）。
  - 缺省时回退 `_build_kernel_basis_from_linear_section`（任意码可用，随机 XOR ≤3 个 ker 基向量作全局提议——已有的泛型逃生门）。
- q=0 多起点：初链 = `r(s) ⊕ 组合(start_sector_generators)`（main.py:1608）；q>0 sector 模式：`0 ⊕ 组合(generators)`（main.py:1656）+ PT 每起点独立跑，spread 进 gate。
- 观测量：`compute_logical_observable_values`（mcmc.py:258）= 修正公式；masks = **全部 2^k−1 个组合**（preprocessing.py:38，`(2^k−1)×n` 稠密数组——**k>10 内存爆炸**）。q_top = mean(m_u²) over 全部非零 u（main.py:1748）。
- direct 引擎 observable 的 section = `build_syndrome_representative_section`（**BpLsd 优先 + 线性回退 + syndrome→chain 结果缓存 dict**，linear_section.py:4-133）。缓存按 packbits(syndrome) 键——大 syndrome 空间时命中率低且内存增长，expander 移植时要设缓存上限。
- 诊断/收敛：`mcmc_diagnostics.analyze_chain_diagnostics/summarize_multi_chain_convergence`（R̂、ESS、spread）；`mcmc_convergence_gate.build_convergence_summary`：阈值 max R̂<1.05、min ESS>200、q_top spread<0.03、**冷端 winding 接受率>1e−4**，PT 时若冷端 winding≈0 且 min swap≈0 → `pt_transport_insufficient`。**泛型，直接移植**；expander 需加 per-u 粒度（worst-u）。

### 2.2 PT（mcmc_parallel_tempering.py）
- swap 公式（:51-135）：`log_ratio = Δlog_odds_p·(W_data_i−W_data_j) + Δlog_odds_q·(W_syn_i−W_syn_j)`，相邻对 even/odd 轮询，交换引用不复制；同一 disorder/s 在全 ladder 共享。
- ladder（mcmc_diagnostics.py:37-99）：`equal_log_odds_ladder`（data-only）；`sync_pt_enlarge_ladder` + `sync_pt_ladders_from_enlarge`（同步放大 K_p、K_q 的 log-odds，硬校验 p_k,q_k<0.5）。自适应 ladder：flow tracker + `adaptive_ladder_from_flow`。**全部泛型，直接移植**。
- PT 仅 q>0（main.py:2924 显式报错）。observable_temperature_mode: all/cold。

### 2.3 sector-TI 引擎（exp37_sector_ti.py）——生产路径
- 每 task = (L, p, q, disorder)（:2068）：
  1. `_build_logical_projection_masks`（:755）：w_u = 线性 section 下 `P_L(x)=⟨z_u, x⊕r(Hx)⟩` 的逐 qubit mask（k×n，**只需 k 个 primitive，不爆炸**）。
  2. `_build_sector_representatives`（:788）：由 `start_sector_generators` 枚举 **2^k** 代表并验签名唯一+零 syndrome——**k>10 爆炸点**。
  3. `_build_sector_preserving_proposals`（:909）：签名平凡的提议集 = 零签名单比特 + 同签名 qubit 对 + contractible moves（kind 0/2；泛型！按 qubit 签名分组配对，签名 = k-bit 模式）。
  4. 每 sector 一条链 `_run_fixed_sector_chain`（:1548）：x 初始化 = sector 代表；对 kp_grid（0→K_p 目标，生产 129 点）逐点 burn-in+测量 `mu=⟨|x|⟩`、`syndrome_mu=⟨|H_Zx⊕δ|⟩`；粗/细网格对照出 `grid_tv/grid_q_top_abs_diff` flags；block bootstrap。
  5. `ΔF = trapezoid(mu, kp_grid) − 第0 sector`；`weights = softmax(−ΔF)`；`q_top = (2^kΣw²−1)/(2^k−1)`（:1735 硬编码 k=3 的 8/7——移植时泛化）。
  - even-winding heatbath（:1131,:1146）：同签名 winding 副本对的组内 heatbath——**toric 特定**（依赖平移副本）；expander 对应物 = 每 logical 类的多个代表（x_u⊕stabilizer 组合）同签名成组，可选实现。
  - numba fast path `_numba_run_fixed_sector_chain` 存在（use_numba）。
- disorder 生成（:100-679 `_build_disorder_uniforms`）：`rng_stream` 模式（泛型，`disorder_seed_scope=disorder_index` 等）+ `coordinate_hash` 模式（splitmix64 按 (seed,kind,type,i,j,k) 哈希——**3D 坐标特定**，跨 L 共用同一 disorder 场；expander 无对应结构，只移植 rng_stream，scope 建议 `(family,m,p,q,disorder_index)`）。
- 精确基准 `_compute_exact_sector_weights_x`（:2333）：x-空间 sector weights 全枚举（其模型即 δ-only；exp101 的 true-模式枚举要喂 s）。
- AIS 路径（`_run_ais` 等）：**废弃不移植**（memory：decoder_reject 标签 bug，exp37/030 失效）。
- 聚合 `_aggregate_results`（:2690）→ `sector_ti_results.npz`（:3105）。

### 2.4 NPZ schema（sector_ti_results.npz，exp40/41 实测字段）
`manifest_json; lattice_size_list; q_values; p_value; q_top_per_disorder (L,q,dis); q_top_stderr_per_disorder; q_top_ci95_per_disorder (…,2); grid_tv_per_disorder; grid_q_top_abs_diff_per_disorder; weights_per_disorder (…,2^k); weights_stderr_per_disorder; delta_f_per_disorder (…,2^k); delta_f_stderr_per_disorder; flags_per_disorder (str); wall_time_seconds_per_disorder; seed_per_disorder; disorder_seed_per_disorder; sample_seed_per_disorder; mean_q_top (L,q); disorder_sem_q_top; mcmc_sem_q_top; total_sem_q_top; pass_fraction`。
manifest 关键键：mode, projection_mode, code_family, disorder_seed_scope, q_values, p_value, lattice_sizes, num_kp_grid_points, git_commit_sha(部分 run)。
**exp101 对应**：`lattice_size_list→code_size_list(m)` 并存同义字段；weights/delta_f 维度在 k≤10 全量、k>10 改为 `(…, k+1+|U_rand|)` 布局（manifest 说明列义）；新增 per-u m_u 数组、真类 sig(η)、ensemble 标签（true/repo_compat）。

### 2.5 production_chunked_scan.py / launcher（只取概念，Phase 2/4 按需再读）
- submit→chunk 任务表（大 L 优先排序 :243）→ 每 chunk 原子写 JSON → merge 校验（`_validate_chunk_payload`）→ manifest 汇总（含 git SHA :269、section 统计合并）。preflight 含 q=0 禁 PT 等参数校验（:4292）。
- 复用要点：原子写、断点续跑（`_mark_existing_chunk_outputs_completed`）、chunk 校验、显式 worker 数。exp101 做最小子集即可。

---

## 3. 逐模块移植清单

| 主项目模块 | 判定 | 说明 |
|---|---|---|
| `mcmc.py` | **复制+微改** | 泛型；observable/disorder/单比特核心。改：section 缓存上限、大 k masks 接口 |
| `preprocessing.py` | **改造** | 邻接表照搬；`build_logical_observable_masks` 改为 basis+U_rand 抽样（k≤10 保留全量路径） |
| `linear_section.py` | **复制** | 泛型；含 BpLsd section + 线性回退。加缓存上限参数 |
| `mcmc_parallel_tempering.py` | **复制+微改** | swap/测量循环泛型；去掉 cluster 依赖或以 no-op 替身 |
| `mcmc_diagnostics.py` | **复制** | ladder/R̂/ESS/flow 全泛型 |
| `mcmc_convergence_gate.py` | **复制+扩展** | 加 per-u worst-u 判据、TI 专用 gate（grid_tv 等已在 TI 内） |
| `exact_enumeration.py` | **改造** | 核心两函数泛型；exp101 版加 true/repo 双系综开关 + q=0 coset 枚举 + Gray-code 加速；并保留"只读运行主项目原函数"作独立互证 |
| `exp37_sector_ti.py` | **提炼重写** | 提取：projection masks、sector reps（k≤10）、sector-preserving proposals、fixed-sector chain+TI 积分、bootstrap、聚合。泛化 8/7→2^k；syndrome 参数按系综接线（δ/s）；大 k sector 子集变体（见 §4）；even-winding 组改为可选的"同类多代表"组 |
| `main.py` | **提炼重写** | run_disorder_average_simulation 及依赖链按需提取；CLI 重写为 exp101/run_scan.py |
| `build_toric_code_examples.py` | **不移植（只读对照）** | expander 侧由 exp101 构造模块提供同约定输出：(H_Z, dual_logical_z_basis, zero_syndrome_move_data) |
| `cluster_update.py` | **不移植** | RREF 依赖 ladder 语义与 toric 使用经验；expander 用 bit+S+L+PT。PT 调用点以 disabled summary 替身 |
| `production_chunked_scan.py` | **概念复用** | 原子写/断点/校验/manifest 模式；代码重写最小版 |
| `profile_3d_q_positive.py` | **不移植** | exp101 写轻量 profile 脚本 |

## 4. 大 k（k=m²）爆炸点与替代

| 位置 | 爆炸 | 替代 |
|---|---|---|
| `build_logical_observable_masks` 2^k−1 masks | 内存/时间 | basis k 个 + U_rand（默认 64）随机非零 u；k≤10 保留全量 |
| `_build_sector_representatives` 2^k | 时间 | k≤10 全量；k>10：{0, e_1..e_k}∪抽样 sector 子集 |
| TI 2^k 条链 | 时间 | k≤10 全 TI；k>10 变体：**basis-pairwise TI**（k+1 条链：sector 0 与各 e_u，输出 per-u ΔF_u 与 pairwise 权重）——须在小 k 上对全 TI/枚举定量验证其作为观测量的性质（exp101 验证矩阵新增项） |
| `_build_q0_start_sector_labels` 2^g 上限 | 起点数 | 轮转子集：起点 i 用 generator (i mod k)（保序可复现） |
| weights/delta_f NPZ 维度 2^k | 存储 | k>10 存 (k+1+|U_rand|) 布局 |
| `_q_top_from_weights` 硬编码 8/7 | 正确性 | 泛化 (2^kΣw²−1)/(2^k−1)；大 k 由 m_u 抽样口径直接给 q_top |

## 5. expander 适配设计要点（Phase 1/2 的输入）

1. family 构造器输出与 toric 同约定三元组：`H_Z (n_An_B×n)`、`dual_logical_z_basis (k×n)`（配对归一后的 z_u）、`zero_syndrome_move_data{contractible=H_X 行(权重 d_A+d_B), winding=x_u 代表(可加低权重变体/多代表), start_sector_generators=x_u 基}`。构造后断言：moves ∈ ker H_Z、winding 签名可逆、H_XH_Z^T=0。
2. 系综开关贯穿两引擎：`ensemble ∈ {true_posterior, repo_compat}`——true：TI/枚举的 syndrome 参数=s、真类=sig(η)、w0=weights[sig(η)]、m_u 含 (−1)^{⟨w,η⟩}；repo_compat：syndrome 参数=δ、真类=0（与 3D 时代逐位可比）。
3. 观测量三档记录（plan §1.4）不变；TI 大 k 变体的输出并入同一 NPZ 口径。
4. section/frame：TI 与 direct 的 linear frame 统一用同一 `build_linear_section` 数据；BpLsd 只作 A/B。

## 6. 其它落地备忘

- `_compute_log_odds(p)` 返回 log(p/(1−p))（负数），权重=|·|·log_odds 直接相加——移植时保持符号约定一致。
- q=0 时 `initial_chain_bits` 必须满足 H_Zc=s（main.py:2390 有断言）。
- PT 使用 `initial_chain_bits_per_temperature`；sector 初始化按温度分配。
- `num_start_chains` 与 `q0_num_start_chains` 的覆盖陷阱（CLAUDE.md 已记）在 exp101 CLI 里避免：只留一个参数。
- 3D burn-in 按 `num_qubits/18` 放大的逻辑不要照搬；exp101 显式传 burn-in 并记录。
- exp37 的 `_build_disorder_uniforms` coordinate_hash 模式不移植；`disorder_seed_scope` 概念保留为 `(family,m,p,q,disorder_index)`。
- `sector_ti_results.npz` 由 exp37_sector_ti.py:3105 写出——exp101 的 run_scan 输出同名文件与兼容字段。
