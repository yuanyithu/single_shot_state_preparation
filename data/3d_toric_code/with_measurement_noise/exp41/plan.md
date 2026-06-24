# exp41 计划：高资源压低误差，给出 3D Toric Code 的 p–q 相图

日期：2026-06-20（优化版，基于 exp40 实测数据修正）

目标：延续 exp40 的 sector-TI 路线（`run --projection-mode linear`），加大计算资源，画出可发表的 3D toric code 在 data error `p` 与 measurement error `q` 平面上的相边界 `q_c(p)`，核心是**压低误差限**。

---

## 0. 从 exp40 学到的三条硬约束（决定本计划全部策略）

这三条来自 exp40 实测，不是猜测，是 exp41 全部资源分配的依据：

1. **加大 L 远比加大 disorder 有效。** exp40/004 实测：p=0.11，L3-L5（384 disorder）headline CI 宽 0.025；仅仅补上 L=6 → L3-L6 CI 收到 **0.010**。`384` disorder 已经把 disorder 噪声打到地板，残余误差由 **crossing 几何太浅**（小 L 曲线在 crossing 处近平行）主导，**不随 √N 收缩**。⇒ exp41 把算力优先投到 **L=7**，而不是把 disorder 堆到 1024。

2. **低统计会把 q_c 系统性压低（"首个变号"估计量在近简并区被噪声拉低）。** 同一 p=0.11：48-dis 给 q_c=0.026，384-dis 修正为 0.043。⇒ 定值 disorder 数**至少 384**；低于此不可信。但高于 384 收益递减（见第 1 条）。

3. **定 threshold 必须用 sign-aware 量；even-moment 量有残余偏差。** exp39/008 FSS 对真值 p_c≈0.233 校准：`w0`=P(真逻辑类)→0.254（最好），signed-mag ⟨m_u⟩→0.264，`q_top`=⟨m_u²⟩→0.268（crossing 方向对、但饱和到 1、偏差略大），`Δf-gap`→0.40（**有偏，禁用**），Binder 退化（禁用）。⇒ **headline 用 sign-aware `w0`（近 p_c 尤其以它为准），`q_W`(=q_top) 作 even-moment 交叉验证。**（见 memory `qtop-saturates-use-deltaf-for-crossing`、`exp36-qpos-frozen-logical-sector`。）

---

## 1. 核心策略修正（与原始 plan 的差异）

| 维度 | 原始 plan | exp41 修正 | 理由 |
|---|---|---|---|
| headline 估计量 | `q_W` 为主，`w0` 仅诊断 | **`w0`（sign-aware）为 headline，`q_W` 作交叉验证**，并报 signed-mag | 硬约束 3 |
| L 与 disorder | L3-7 全部重跑，512→1024 disorder | **复用 exp40/004 的 L3-6×384，仅新跑 L=7**；disorder 维持 384 | 硬约束 1+2 |
| 节点 | 仅 nd-3 | **nd-1/2/3 三节点，按 disorder 分片**（exp40 已验证） | nd-3 单机跑全量约 9 天，不现实 |
| q 网格 | 新网格 0.024–0.058 | **沿用 exp40 的 0.012–0.070 十点网格**（使 L=7 能与 L3-6 直接合并） | 复用前提 + 低端锚点更全 |
| L=7 信任 | 直接生产 | **先过收敛 gate（kp-grid / burn-in A/B）再生产** | L=7 的 burn-in/AIS 偏差是 L 相关的，未验证 |
| 误差目标 | CI ≤0.004 单 pair | **以 1/L 外推 + 全 L 对收敛散点为准**，单 pair 目标 ≤0.005 | crossing 几何限，单 pair 难到 0.004 |

**复用为什么成立**：`w0`、`q_W`、`q_purity`、signed-mag 全部是 `delta_f_per_disorder` 的**事后函数**（`w_g = softmax(-Δf_g)`），不是 run 时固化的标量。exp40/004 的 L=3,4,5,6 × 384-disorder NPZ 已在本地（`exp40/004_p011_highstats/nd{1,2,3}/collected/p0p11/` 及 `nd{1,2,3}_L6/...`），q 网格 `[0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070]`，seed 860000/861000/862000。只要 exp41 的 L=7 用**完全相同的固定参数与 q 网格**，就能沿 L 轴并入，立刻得到全部 L3–L7 crossing 对。

> 备选（仅当审稿要求完全独立复现时）：放弃复用，用新 seed 全量重跑 L=3..7。代价是多花 ~4000+ 核时重产已知正确的 L3-6，不建议作为默认路径。

---

## 2. Observable 定义与命名（统一，消除歧义）

全部从 `delta_f_per_disorder` 重算（`w_g = softmax(-Δf_g)`，`g=0..7` 对应 k=3 的 8 个逻辑类，**sector 0 = 真逻辑类**，由 η 定义）。设 `m_u = Σ_g (-1)^{bit_u(g)} w_g`（u=1,2,3）：

```text
w0      = w_{g=0}                      # sign-aware，P(真类)。【headline】
mbar    = (1/3) Σ_u |m_u|              # sign-aware，signed magnitude 的均值（交叉验证）
q_W     = (1/3) Σ_u m_u^2              # even-moment。== exp40 旧 "q_top"，数值完全一致
q_purity= (8 Σ_g w_g^2 − 1) / 7        # even-moment，sector 纯度归一到 [0,1]（诊断）
```

**关键对应关系，务必写进每个分析脚本头部，不得搞错**（已用本地 smoke 核实）：

```text
weights_per_disorder  ==  softmax(-delta_f_per_disorder)         # 数值完全相同（diff=0）
q_W  =  mean_u m_u^2  从 softmax(-delta_f) 重算  ==  exp40 analyze 脚本发表的 "q_top"   # 复用对齐用这个
w0   =  w_{g=0} 从 softmax(-delta_f) 重算  ==  exp40 w0
q_purity  =  全新量，exp40 未算过，从 weights/delta_f 重算；≠ q_W
```

⚠️ **NPZ 里的标量字段 `q_top_per_disorder` 不是 softmax 重算值**——它是 run 内基于 MCMC 样本/TI 的独立估计，有限统计下与 softmax 重算差异可达 ~0.03（smoke 实测，生产参数下会小很多但不为零）。**绝不能把 L=7 用 stored 字段、L3-6 用 softmax 重算混在一个 crossing 里**（会引入估计量不一致的假信号）。

- **唯一允许的路径**：所有 L、所有量一律从 `delta_f_per_disorder`（或等价的 `weights_per_disorder`）softmax 重算，与 exp40 的 `analyze_p011_L3456.py` 逐位一致。stored 标量字段仅供审计。
- 字段命名统一：`w0_per_disorder`、`mbar_per_disorder`、`q_W_per_disorder`、`q_purity_per_disorder`。
- 不要再用 `qtop` 这个模糊名。

---

## 3. 第一阶段：p=0.11 高资源定点

选 p=0.11 理由不变（平台中部、已有 L3-6×384 参考、crossing 经验充分）。第一阶段 = **把 L=7 加进来，并用最大 L 对把 p=0.11 的 q_c 误差压到极限**。分三步，每步是下一步的前置门。

### 3.0 固定参数（必须与 exp40/004 逐项一致，否则无法合并）

```text
entrypoint                       = src/exp37_sector_ti.py run
code_family                      = 3d_toric
projection_mode                  = linear            # 绝不用 ais / decoder_reject
num_kp_grid_points               = 129               # L=7 待 gate 验证是否需 257
num_burn_in_sweeps               = 512
max_effective_num_burn_in_sweeps = 512               # L=7 待 gate 验证是否需放大
num_measurements                 = 8192
num_sweeps_between_measurements  = 2
block_count                      = 128
num_bootstrap                    = 800               # run 内 block 误差，≠ 下游 crossing bootstrap
winding_heatbath_sweeps          = 1
common_disorder_across_q         = true
disorder_seed_scope              = disorder_index
disorder_realization_mode        = rng_stream
use_numba                        = true
grid_tv_warning / grid_q_top_warning = 0.02
q_grid = 0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070   # = exp40
```

### 3.1 步骤 G1：运维 smoke（确认管线，不看物理）

```text
p=0.11   L=3,4,5,6,7   q=0.040（单点）   disorder=64   node=nd-3
```

通过判据：5 个 L 全部输出 NPZ；日志确认 `workers≈92`、`use_numba=True`、`projection_mode=linear`；记录 L=7 单点 walltime（用于外推全量）。**只验管线，不下物理结论。**

### 3.2 步骤 G2：L=7 收敛 gate（最关键，决定 L=7 能否信）

L=7 的真实风险不是 walltime，而是 **AIS/TI 在更大系统上的偏差**：kp-grid 太粗 → Δf 偏差；每个 kp 节点 burn-in 不足 → 未平衡。注意 `max_effective_num_burn_in_sweeps=512` 会把有效 burn-in 钉在 512，**不随 L 放大**——在 L=7（n=3·343=1029）可能偏少。必须 A/B 验证：

```text
p=0.11   L=7 only   q=0.030,0.034,0.040（跨 crossing 核心）   disorder=64
对比四组：
  (a) kp=129, max_eff_burn=512   ← 生产候选（与 exp40 一致）
  (b) kp=257, max_eff_burn=512
  (c) kp=129, max_eff_burn=2048
  (d) kp=257, max_eff_burn=2048
```

通过判据：`w0` 与 `q_W` 在 (a) 与 (b/c/d) 之间的差 **小于 64-disorder 的 SEM**（即 129/512 已收敛）。
- 若 (a) 已收敛 → L=7 生产用 129/512，与 exp40 L3-6 **完全同参**，合并最干净。
- 若 (a) 未收敛 → 取最小的收敛设置（如 257/512 或 129/2048）作 L=7 生产参数；并补一个 **L=6 在该设置下的对照 shard**（确认 exp40 的 L6 在 129/512 下没有同源偏差，若有则 L=6 也需在新设置下重跑、L3-5 因小系统易收敛可继续复用）。

G2 不过，不进入 3.3。

### 3.3 步骤 P1：L=7 生产（三节点 disorder 分片）

```text
p=0.11   L=7 only   q=exp40 十点网格   disorder=384
分片：nd-1 / nd-2 / nd-3 各 128 disorder
seed_base：910000 / 911000 / 912000   （与 exp40 的 86xxxx 无冲突）
参数：G2 选定的收敛设置（默认 129/512）
```

完成后与 exp40/004 的 L=3,4,5,6 合并（沿 L 轴），得到 **L=3,4,5,6,7 × 384 disorder** 完整数据集。

### 3.4 步骤 P1b：q 网格细化（可选，仅当 crossing 落在粗网格缝里）

exp40 网格在 0.034–0.048 段步长偏粗（0.006/0.008）。若合并后 `w0`/`q_W` 的大 L 对 crossing 恰落在这段缝里、CI 受网格分辨率限制，则**对全部 L=3..7 补 2–3 个 q 点**（如 0.028,0.036,0.044）：
- 因 `common_disorder_across_q + disorder_seed_scope=disorder_index`，disorder 实现与 q 无关——用 exp40 原 seed（860000/861000/862000）重跑 L3-6、用 910000+ 重跑 L7，在**新 q 点**上得到的是**同一批 disorder 实现**，可沿 q 轴直接拼接，不损失 384 统计。
- 仅 2–3 个 q 点 × 全 L，约 1500–2000 核时，按需触发，不默认跑。

### 3.5 disorder 升级（仅在确有必要时，且不是首选）

只有当**合并 + 细化后**大 L 对 crossing 的 CI 仍主要由 disorder 噪声（而非几何）限制（表现为 boot_frac<0.95、bootstrap 分布多峰/贴边、且 SEM 仍随 √N 下降）才追加 disorder 到 512（再加 seed 913000/914000/...）。按硬约束 1，这通常**不是**收紧 CI 的有效杠杆——优先考虑 L=8 或依赖 1/L 外推。

---

## 4. 节点与资源策略

**三节点 nd-1/2/3，按 disorder 分片**（沿用 exp40 launcher：每节点一个 `run` 调用覆盖该分片的全 q 网格，worker pool 把 (L,q,disorder) 任务铺满核）。

- 实测有效核数：nd-1≈76、nd-2≈76、nd-3≈92（探测值 80/80/96 留 headroom）。
- worker 数在 **screen 外**探测并烘焙进 runner；screen 内 `$(nproc)` 在 cgroup 下会误报 1 → 退化串行（历史坑）。日志必须确认 `workers≈76/76/92`、`load≈workers`。
- 线程环境（防 BLAS/numba 嵌套超订）：
  ```bash
  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  ```
- `conda run --no-capture-output -n 11`；`screen` 后台；远端只写 `~/.single_shot/runs/` 与 `~/.single_shot/logs/`。
- 全流程（launch→健康检查→轮询→回收→校验）严格按 skill **`remote-prod-scan`** 的 checklist 执行。

**算力预算（粗估，校准 L³ scaling，exp40 实测 L6≈L3+L4+L5 合计核时）**：

| 步骤 | 规格 | 估算核时 | 三节点墙钟 |
|---|---|---|---|
| G1 smoke | L3-7, q×1, 64dis, nd-3 | ~250 | ~3–5 h |
| G2 gate | L7, q×3, 64dis ×4 设置 | ~1500 | ~6–10 h |
| P1 L=7 生产 | L7, q×10, 384dis | ~6500 | ~24–30 h |
| P1b 细化(可选) | L3-7, q×2–3, 384dis | ~1500–2000 | ~8 h |
| **第一阶段合计** | | **~8000–10000** | **~2 天有效机时** |

（复用 exp40 L3-6 省下约 8000 核时；这正是"加大 L 而非堆 disorder"的体现。）

---

## 5. 分析与产物

### 5.1 合并前校验（每个 shard）

q 网格一致；L 列表一致；`p_value=0.11`；`projection_mode=linear`；`use_numba=True`；`disorder_seed_scope=disorder_index`；固定参数与 exp40/004 一致（或 G2 选定的统一设置）；disorder seed 与 sample seed **无重复**；`_CELL_SUCCESS.json` 存在；crossing 核心区（q≤0.034）pass_fraction≥0.9。

### 5.2 估计量模块（单一来源，杜绝混名）

扩展 exp40/004 的 `analyze_p011_L3456.py` 的 `estimators(delta_f)`，一次返回 `{w0, mbar, q_W, q_purity}`（公式见 §2），脚本头部写明 `q_W ≡ 旧 q_top`。crossing 与 CI 复用 exp40 的 `cross_q`（**零差值仅在两侧异号时计入**，避免饱和段假 crossing）+ disorder-bootstrap。

### 5.3 crossing 表

输出全部 L 对（L3-L5, L3-L6, L3-L7, L4-L6, L4-L7, L5-L7, L6-L7, …），每对对 **w0 / q_W / mbar** 三量各给 `q_c, CI95, boot_frac`，下游 crossing bootstrap `NBOOT=10000`（与 §3.0 的 run 内 `num_bootstrap=800` 是两回事）。`boot_frac<0.95` 不得作 headline。

**1/L 外推**：连续对（用 **L≥4**，丢掉 L3-L4 近简并对——exp40 实测它让外推 CI 发散）做 `q_c` vs `2/(L_a+L_b)` 线性外推到 0，bootstrap 传播 CI；同时给全 L 对的收敛散点图（不止单点）。

### 5.4 图

1. `p011_w0_curves.png` — `w0(q)` 全 L，带 disorder-SEM。【主图】
2. `p011_qW_curves.png` — `q_W(q)` 全 L，带 SEM（even-moment 交叉验证）。
3. `p011_pairwise_diff.png` — 对最大 L 的差值曲线 + bootstrap 带（过零=q_c）。
4. `p011_crossing_fss.png` — `q_c` vs `1/L_mean`（w0 与 q_W 并列）+ 1/L 外推带。
5. `p011_exp40_comparison.png` — 对比 exp40/004（L≤6）量化误差缩小与 crossing 漂移。

### 5.4b（注意）大 L 对的 crossing 通常**向下漂**（exp40：q_W L3-L5=0.043→L5-L6=0.031；w0 L3-L6=0.033→L5-L6=0.032）。L=7 加入后 crossing 可能落到 ~0.028–0.032，**逼近网格低端 0.026/0.030**——这正是沿用 exp40 网格（低端有 0.012/0.018/0.022/0.026/0.030 多点锚定）的价值；若 crossing 贴 0.026 以下，触发 §3.4 在低端补点。

### 5.5 summary.md（每步一份）

run id / 日期 / 远端路径 / 本地路径 / p,q,L,disorder / seed_base / 固定参数（标注是否与 exp40 同参）/ 墙钟 + 核时 / pass_fraction / **w0,q_W,mbar 三量 crossing 表 + 1/L 外推** / 与 exp40 对比 / 是否进入第二阶段的判断。

---

## 6. 第一阶段验收标准

1. G1、G2 通过；L=7 生产 384 disorder 合并成功，得到 L=3..7 完整集。
2. crossing 核心区 pass_fraction≥0.9。
3. **headline `w0` 的最大 L 对（L5-L7 或 L6-L7）crossing CI95 半宽 ≤0.005**（理想 ≤0.004；达不到则以 1/L 外推 + 全 L 对收敛散点为最终呈现，不强行用单 pair）。
4. `w0` 与 `q_W` 的 crossing 不出现物理上矛盾的趋势（方向一致；`q_W` 可整体略高，`w0` 略低，符合 sign-aware vs even-moment 的已知偏差排序）。
5. L 对 crossing 呈现清晰有限尺寸下漂趋势，而非只靠单一小 L 对。
6. 明确回答：
   - 当前规格能否把单点 q_c 画到目标误差？
   - 误差主导是 disorder 统计还是 finite-size geometry？（预期：geometry。）
   - 第二阶段每个 p 点需要跑到哪个 L？（预期：近 p_c 必须 L≥6/7。）
   - L=7 是否需要进入最终相边界 headline？

不达标时的调整顺序（按性价比）：先 §3.4 细化 q 网格 → 改 headline 到更大 L 对 → 加 L=8 → 最后才考虑 §3.5 加 disorder。

---

## 7. 第二阶段：扩展到 p–q 相边界

第一阶段达标后再细化。原则同样是**复用 + 把大 L 投到最有信息量的 p**：

- **平台中段已基本确定**（exp40：q_c(p) 在 p∈[0.02,0.21] 近平、≈0.03–0.04，w0/q_W 一致）。这段**复用 exp40/005（p=0.02,0.05,0.17,0.21 × L3-5×384）+ exp40/004（p=0.11 × L3-6×384）**，只用第一阶段标定好的估计量重算，不重跑。
- **真正缺口在近 p_c≈0.227**：exp40 的 L≤5 在 p=0.20,0.21,0.22 仍给 q_c≈0.042，**看不到向 (0.227,0) 的陡降**（小 L 系统性高估可纠错性）。⇒ 第二阶段算力集中在 **p=0.20,0.21,0.22,0.225 上把 L 加到 6、7**，解析 boundary 的收口段——这是本相图唯一尚未确定、且物理上最重要的部分。
- 主平台若要降误差，挑 1–2 个代表 p（如 p=0.05、0.17）补 L=6/7 即可，不必每个 p 都全量。
- headline 统一用 **`w0` 的最大可用 L 对**（近 p_c 尤其以 w0 为准）；`q_W` 交叉验证。最终边界图必须画 finite-size drift / 1/L 外推，不画单一 crossing 点；近 p_c 段标注"小 L 高估、大 L 收口"。

近 p_c 的 q 网格下移（crossing 随 p→p_c 下移）：

```text
q = 0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.038,0.046
```

---

## 8. 风险与注意事项

### 8.1 L=7 收敛（最大风险）
不是 walltime，而是 AIS/TI 偏差（kp-grid + 每步 burn-in，且有效 burn-in 被 `max_eff=512` 钉死、不随 L 放大）。**必须先过 G2 gate**（§3.2）。L=7 单 shard 若过慢，先交付 L≤6 的复用结果 + 把 L=7 减到 crossing 核心 q（0.026,0.030,0.034,0.040,0.048）。

### 8.2 q 网格贴边
大 L crossing 向下漂可能贴 0.026 低端。处理：沿用 exp40 低端多锚点；若 q_c<0.026 或 CI 贴边，按 §3.4 在低端补点，**不用外推替代缺失 crossing**。

### 8.3 Observable 混淆（高频坑，已在 §2 定死）
**`q_W ≡ 旧 q_top`（mean m²）；`q_purity` 是全新量、≠ q_top。** 严禁把 `q_top_per_disorder` 当成 `q_purity`（原始 plan 此处写反，已修正）。全部从 `delta_f_per_disorder` 重算。

### 8.4 复用一致性
复用 exp40 L3-6 的前提是 L=7 用**完全相同的固定参数**。若 G2 发现 L=7 需要不同 kp/burn-in，按 §3.2 末尾的分支处理（必要时只重跑 L=6，L3-5 继续复用）。seed 不得与 exp40 的 86xxxx 冲突。

### 8.5 服务器目录
远端只用 `~/.single_shot/runs/` 与 `~/.single_shot/logs/`；回收 + 校验后清掉该 run 的 `repos/` 副本与临时 cache。本地 `data/` + git 是 single source of truth；分析在本地做。

---

## 9. 本地与远端目录

本地：
```text
data/3d_toric_code/with_measurement_noise/exp41/
  001_p011_g1_smoke_20260620/
  002_p011_g2_L7_convergence_20260620/
  003_p011_L7_prod_384dis_20260620/      # 内含 nd1/nd2/nd3 + 合并分析
  (004_p011_qgrid_refine_...  按需)
```
远端：
```text
~/.single_shot/runs/exp41_p011_L7_<TS>/{nd1,nd2,nd3}/
~/.single_shot/logs/exp41_p011_L7_<TS>_nd{1,2,3}.log
```

---

## 10. 最小可执行下一步

1. 在 `exp41/001_p011_g1_smoke_20260620/` 基于 `exp40/002_production_boundary_20260610/launch_exp40_boundary.sh`（node↔p 的参考 launcher，机制：每节点一个 `src/exp37_sector_ti.py run` 调用 + worker pool）改一个**按 disorder 分片**的 launcher（node↔disorder-shard：三节点同 p/同 L/同 q，各 128 disorder、不同 seed_base）。
2. 跑 G1 smoke：`p=0.11, L=3..7, q=0.040, disorder=64, nd-3`；确认 workers/numba/NPZ/walltime。
3. 跑 G2 收敛 gate（§3.2 四组），判定 L=7 生产参数。
4. 跑 P1：L=7 × 384 disorder × 三节点（seed 910000/911000/912000）。
5. 与 exp40/004 合并，重算 w0/q_W/mbar 全 L 对 crossing + 1/L 外推，写 summary，对照 §6 验收。
6. 完成后更新 `笔记/实验报告.md`。
