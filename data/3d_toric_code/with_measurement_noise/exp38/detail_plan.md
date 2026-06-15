# detail_plan.md — exp38 生产级放大并出图（执行 agent 唯一权威规范）

> 本文件是 exp38 的唯一权威规范。配套 `prompt.md` 只负责「定位当前阶段 → 推进一步 → 过闸门 → 更新进度」，所有判据细节以本文件为准。
> 进度记录在同目录 `STATUS.md`。每次开工先读 `STATUS.md` 定位当前阶段，再读本文件对应阶段执行。
> **规范基准是仓库根 `AGENTS.md`**（物理图像、L·T·S 分解、运行规则、避坑清单）。本文件不重复，执行前务必先读一遍 `AGENTS.md`。

---

## 0. 目标（一句话）

在 exp37 已验证的 sector-resolved TI 主线之上，把生产网格 `(p=0.05, q, L=3,4,5)` 跑到**统计可分辨**：用「**跨 L 公共 disorder + 配对差分**」把误差棒压到小于 L 间距，给出可信、可证伪、可复现的 `q̄_top(L, q)` 曲线，并明确判定阈值 crossing 是否在统计上被分辨。最终交付两张图 + 验收。

物理图像、energy / observable 定义、L·T·S 分解见 `AGENTS.md`「物理图像与 L·T·S 分解」一节。

---

## 1. 为什么做 exp38（承接 exp37，必须理解）

exp37 已完成并通过验证（A–G 全 PASS）：
- 照妖镜 benchmark 含 `q_top∈[0.2,0.8]` 的中段点（`exp37/034_stageB_exact_reference_*/exact_reference.json`）。
- 主估计量 **sector-resolved TI** 在中段点上对精确解 `TV≤0.005、δq_top≤0.0055`（`exp37/036_stageD_*`）。
- 第二法（退火 + 双向 BAR，**非单步 FEP**）三方互证（`exp37/037_stageE_*`）。
- 032 的「大 L 时 `q_top` 被顶到 1」假象已消失：生产网格大 q 区呈现 `q_top(L=5) < q_top(L=4) ≲ q_top(L=3)`（`exp37/038_stageF_*`）。

**但 exp37 的生产 run 是欠功率的侦察档**，不能直接出图：
1. 48 个点里 38 个 WARN，原因几乎全是 `TI_GRID_*_WARN`（生产档 grid=65/burn=160/m=2048 时 TI 粗/细网格没收敛）+ 每点只有 4 个 disorder 里部分没过。
2. PASS 点误差棒 `SEM≈0.10–0.15`，比 L 间距（0.02–0.05）还大；三条曲线在 PASS 点上重叠，crossing 无法分辨。exp37 Stage G 已诚实写 `broad crossing claimed=False`。

**exp38 的两个根因修复（本计划全程贯彻）：**
1. **档位升级**：全程用 exp37 已验证过的「强档」TI（`grid=129, burn=512, m=8192, stride=2, blocks=128, bootstrap=800`），让每点 TI 网格收敛 → WARN 收敛成 PASS。
2. **方差缩减（最关键）**：三个 L 用**同一组 disorder 种子**（`disorder_seed_scope` 固定到 `disorder_index`、跨 L 公共），主结论看**配对差分** `Δ_{ij}(disorder) = q_top(L_i) − q_top(L_j)` 再对 disorder 平均，而**不是**比独立均值。共享的 disorder 涨落在配对差分里抵消，`Var(Δ)` 远小于两个均值各自的方差 → 用 20–40 个公共 disorder 即可分辨 crossing，而比独立均值要上百个。

---

## 2. 全局铁律（每个阶段都适用，违反任意一条即该阶段判 FAIL）

1. **闸门 = 与外部参照对比的数字**。合法参照只有三类：① 小尺寸精确枚举；② 第二种独立估计量（退火 + 双向 BAR）；③ 解析锚点（`Kp=0 ⇒ w_g=1/8, q_top=0`）。**「内部自洽 / 多初态一致 / 曲线平滑 / 全部 PASS / 看着收敛」一律不算成功。**
2. **主估计量固定为 sector-resolved TI**，**直接复用 exp37 已验证代码路径**（`src/exp37_sector_ti.py` + `exp37/036_*/run_stageD_sector_ti.py`），不另起炉灶、不改估计量语义。**禁止**把单步 FEP / `flip_reweight` 当生产估计量；第二方法只能是**沿路径退火且双向（BAR / 双向 AIS）**，且仅作交叉验证。
3. **红线（最重要）**：若某点除主导扇区外所有 `w_g` 都低于统计分辨率（bootstrap 误差棒覆盖不住、或低于方法统计地板），该点必须标 `UNRESOLVED/FAIL`，输出「`q_top` 下界 + 未分辨」，**绝不允许报成 `q_top≈1`**。「`q_top≈1` 且 `w_sub` 未分辨」永远是失败。
4. **跨 L 公共 disorder + 配对差分是 exp38 的方法核心**：① 同一 `disorder_index` 在 L=3,4,5 上必须对应同一组 `(s, η)` 生成规则（`disorder_seed_scope` 不含 `lattice_size`，使种子跨 L 公共）；② crossing 结论必须由**配对差分 + 配对 disorder bootstrap CI** 给出，单看独立均值重叠不算否证 crossing，也不算证实 crossing。
5. **一次只推进一个阶段**。上一阶段未 PASS、未在 `STATUS.md` 贴出对比数字，不得进入下一阶段。
6. **可信性优先于速度**。宁可某点判 FAIL/UNRESOLVED，也不要为凑 PASS 放宽判据。判据阈值可微调，但必须在交付物里显式记录用了什么阈值。
7. **效率铁律**：先本地 de-risk + 标定 wall-time，再上服务器；q 网格分批（crossing 区密、深有序区稀）；按 L 分节点并行；用 `sleep-until` 阻塞等哨兵而非轮询。数据产物只写本项目 `data/` 内；远端 conda `11` + `screen` + `conda run --no-capture-output`，复杂脚本写 `.py` 文件不用 heredoc，不临时装包；每完成一个阶段增量更新 `笔记/实验报告.md`（中文、简洁）。

---

## 3. 进度状态机：`STATUS.md`

`STATUS.md` 是一张表，每行一个阶段。状态取值：`TODO` / `DOING` / `PASS` / `FAIL`。「当前阶段」= 表中第一个不是 `PASS` 的阶段，只在该阶段上工作。文件不存在时按本节阶段列表（P0–P5）初始化为全 `TODO`。全部 PASS 且过 Definition of Done 后在顶部写 `ALL DONE`。

---

## 4. 阶段定义

> 每个阶段四要素：**目的 / 做什么（高层）/ 成功闸门（外部、数字）/ 交付物**。阈值为推荐默认值，可微调但需在交付物记录。
> 子目录命名：exp38 下从 `001` 起递增带序号，如 `001_p0_regression_anchor_<YYYYMMDD>/`；远端多节点结果放该序号目录下不同 run 子目录。

### Stage P0 — 回归锚点（本地，防代码漂移，便宜先做）

- **目的**：确认 exp38 用的 estimator/代码与 exp37 验证过的完全一致，尺子没歪——再贵的生产 run 也建立在这把尺子上。
- **做什么**：本地 env `12`，复用 exp37 `exact_reference.json` 里若干**中段 `q_top` 点**（如 `q_top≈0.25/0.40/0.55`），用 exp37 的 TI runner 重跑，与精确解对比。
- **成功闸门**（复现 exp37 D1/D2）：
  - P0a：`TV(w_TI, w_exact) ≤ 0.02`。
  - P0b：`|q_top_TI − q_top_exact| ≤ 0.02`，bootstrap CI 覆盖精确值。
  - 任一中段点漂移超阈 → 判 FAIL，停下查代码/环境差异，不进 P1。
- **交付物**：`001_p0_regression_anchor_*/` 验证脚本 + `summary.md`（逐点贴 TV、δq_top、CI）。

### Stage P1 — 本地配对差分 de-risk + 算力标定（本地，关键，决定服务器规模）

- **目的**：烧服务器前，本地证明三件事：① 强档 TI 在生产 L 上 grid 真的收敛；② 跨 L 公共 disorder + 配对差分确实把误差棒压到小于 L 间距；③ 标定单点 wall-time → 给出全网格预算表，使服务器 run 一次到位、不返工。
- **做什么**：本地 env `12`，取 `L∈{3,5}`、crossing 区一个 `q`（如 `q=0.18`）、`N_common=8–12` 个**跨 L 公共** disorder，跑强档 TI（`grid=129, burn=512, m=8192, stride=2, blocks=128, bootstrap=800`；**L=5 显式设 `--max-effective-num-burn-in-sweeps` 防 `num_qubits/18` 自动放大**，见 AGENTS.md）。逐 disorder 算 `Δ=q_top(L5)−q_top(L3)`；对比「unpaired 两均值各自 SEM」与「paired-Δ 的 SEM」；记录 L=3/5 单点 (q,disorder) wall-time。
- **成功闸门**：
  - P1a：强档下每点 `grid TV ≤ 0.02`（确认强档确实收敛，区别于 exp37 侦察档）。
  - P1b：`SEM(paired Δ) < SEM(unpaired mean)` 且有意义地更小（记录缩减比值；目标是配对后 SEM 显著小于典型 L 间距）。
  - P1c：产出 L=3/4/5 单点 wall-time 标定 + 由此推算的**全网格预算表**（含分节点并行后的预计墙钟）。
  - 若 P1a 不过 → 强档仍不够，上调 grid/m 并记录；若 P1b 看不到缩减 → 检查 disorder 是否真的跨 L 公共（铁律 4①），不公共则修正后重测。
- **交付物**：`002_p1_paired_demo_*/` 脚本 + `summary.md`（强档 grid 收敛数字、配对 vs 非配对 SEM 比值、wall-time 标定与预算表）。

### Stage P2 — 服务器生产网格（远端，主算力）

- **目的**：跨 `L=3,4,5` × 聚焦 `q` 网格 × `N_common` 公共 disorder，强档 TI，高效铺满。
- **做什么**：固定 `p=0.05`。**分两批高效铺**：
  - 批 1（crossing 区，高价值，密集 + 多 disorder）：`q∈{0.15,…,0.23}`，`N_common ≥ 24`（按 P1c 预算可上调到 32–40）。
  - 批 2（深有序区，稀疏，本就要 WARN）：`q∈{0.08,0.10,0.12,0.14}`，`N_common` 可较少。
  - 三个 L 用**同一组公共 disorder 种子**（铁律 4①）。复用 exp37 远端脚手架（`launch_stageF_ti_remote.sh / check_stageF_remote_status.py / collect_stageF_ti_remote.py / merge_stageF_ti_shards.py`），按 L 分节点：`nd-1=L3, nd-2=L4, nd-3=L5`，conda `11` + `screen` + `conda run --no-capture-output`，`sleep-until` 阻塞等哨兵。先确认远端 Numba 启用（小 benchmark）。
- **成功闸门**：
  - P2a：覆盖完整，每个 `(L,q,disorder)` 都有 `w_g[8]`、`ΔF_g`、`q_top`、bootstrap 误差、`PASS/WARN/FAIL` flag。
  - P2b（红线）：任何「次主导扇区全部低于统计分辨率」的点标 `UNRESOLVED/FAIL`，`unresolved_tail_fail=0` 时不得有 `q_top≈1` 被当 PASS。
  - P2c：每个 PASS 点 粗/细网格自洽（`grid TV ≤ 0.02` 且 `|Δq_top| ≤ 0.02`）。
- **交付物**：`003_p2_production_grid_*/` 下分节点 run 子目录（各自运行代码 + 数据）+ `merged .../sector_ti_results.npz` + `failure_map.md` 草稿。

### Stage P3 — 第二法抽样交叉验证（远端子集）

- **目的**：crossing 区抽样点上证明 TI 不是单条路径的假象。
- **做什么**：crossing 区每个 `L` 至少取 1 个点（合计 `≥3` 点），跑退火 + 双向 BAR（复用 `exp37/038_*/run_stageF_second_method_subset.py` + `launch_stageF_second_method_remote.sh`），与 TI 点对点比。**严禁退化为单步 FEP。**
- **成功闸门**：
  - P3a：抽样子集 `|q_top_TI − q_top_2nd| ≤ 0.02` 且 `TV ≤ 0.03`。
  - P3b：双向一致性诊断（前向/反向自由能差）在阈值内。
  - 不一致的点降级 WARN/FAIL 并记录（参照 exp37 对 F3 失配点的强档 TI 复跑处理）。
- **交付物**：`004_p3_second_method_subset_*/` 第二法结果 + `summary.md`（TI vs 2nd 点对照）。

### Stage P4 — 验收 + 失败地图 + 配对差分表

- **目的**：逐点定状态、红线落地、算出配对差分量。
- **做什么**：复用/扩展 `exp37/038_*/accept_stageF_ti_grid.py` 做逐点验收；对每对 `(L_i, L_j)`、每个 `q`，用**两 L 都有效（同一 disorder 在两 L 都 PASS）的公共 disorder** 计算 `Δ_{ij}(q) = mean_d[q_top(L_i,d) − q_top(L_j,d)]`，配对 disorder bootstrap 给 CI。
- **成功闸门**：
  - P4a：每点明确 `PASS/WARN/FAIL`（同 P2b 红线 + P2c 网格）。
  - P4b：第二法子集核对通过（同 P3a）。
  - P4c：每个 `q` 的配对差分 `Δ_{ij}` 的**有效配对 disorder 数**被记录（用于判断是否够分辨）；crossing 区至少有若干 `q` 的 `Δ` 配对 CI 不含 0（否则如实记「未分辨」）。
- **交付物**：`005_p4_acceptance_*/` `failure_map.md`（PASS/WARN/FAIL + 原因）+ `paired_difference.csv`（逐 `(L_i,L_j,q)` 的 `Δ`、配对 CI、有效配对数）。

### Stage P5 — 生产曲线 + crossing 判定 + 绘图

- **目的**：产出最终图与可信结论。
- **做什么**：基于 P4 接受的强档网格，复用/扩展 `exp37/039_*/build_stageG_production_curve.py`，画**两张图**：
  - **图 A**：`q̄_top(L, q)`，PASS-only + disorder bootstrap 误差棒；WARN 点以空心标记保留为 context、可区分；不依赖 FAIL/WARN 点下结论。
  - **图 B（crossing 关键图）**：配对差分 `Δ_{ij}(q) = q_top(L_i) − q_top(L_j)` vs `q`，带**配对** bootstrap CI；看 `Δ` 在哪个 `q` 穿过 0 = crossing；只用两 L 都有效的公共 disorder。
- **成功闸门**：
  - G1：小 L 曲线与精确 benchmark 一致（复现 P0 数字）。
  - G2：crossing/趋势结论**只由「配对 CI 不含 0」的点**支撑；明确写出 crossing 是否 **statistically resolved**、在哪个 `q` 区间；不依赖 WARN/FAIL 点。
  - G3：`q_top` 可从保存的 `w_g[8]` 完整重建（max abs diff 0）；误差棒含 disorder bootstrap + TI stderr 两层。
  - 红线：无任何「`q_top≈1` 且尾未分辨」被当成功。
- **交付物**：`006_p5_production_curve_*/` 最终 `npz/csv/md`，**两张 `png`（图 A + 图 B）**，`acceptance.md`（逐条对照 G1–G3、红线、crossing 判定结论）。

---

## 5. 命名与环境约定

- exp38 下每个阶段放**新的带序号子目录**（从 `001` 起递增），命名如 `001_p0_regression_anchor_<YYYYMMDD>/`；多节点分批跑时放该序号目录下不同 run 子目录，每个子目录含该节点运行代码 + 数据。
- 本地 conda 环境 `12`；远端按 `AGENTS.md`：`ssh yuany → ssh nd-x`，conda `11`，`screen` 后台，`conda run --no-capture-output`，复杂脚本写 `.py` 文件再跑（不用 heredoc）。不临时装包；用依赖前先 `conda run -n 11 python -c "import ..."` 验证。
- **L=5 burn 自动放大坑**：默认按 `num_qubits/18` 放大，必须显式设 `--max-effective-num-burn-in-sweeps` 并记录（AGENTS.md）。
- 复用的 exp37 资产（不要重写）：估计器 `src/exp37_sector_ti.py`；精确参照 `exp37/034_stageB_exact_reference_20260603/exact_reference.json`；TI runner `exp37/036_stageD_sector_ti_20260603/run_stageD_sector_ti.py`；第二法 `exp37/037_.../run_stageE_bidirectional_bridge.py` 与 `exp37/038_.../run_stageF_second_method_subset.py`；远端脚手架与验收 `exp37/038_.../{launch,collect,merge,check}_*.{sh,py}`、`accept_stageF_ti_grid.py`；绘图 `exp37/039_.../build_stageG_production_curve.py`。
- 每阶段闸门脚本、对比数字、结论写进该子目录 `summary.md`，在 `STATUS.md` 记一行；完成后增量更新 `笔记/实验报告.md`。

## 6. Definition of Done

P0–P5 全部 `PASS`，且：① 生产曲线（图 A）只由 PASS 点构成；② 小 L 与精确解一致；③ crossing 判定（图 B）由配对差分 + 配对 CI 给出**明确**结论（resolved / 未分辨，含 CI 数字）；④ 没有任何「`q_top≈1` 但次主导扇区未分辨」被当成功；⑤ 所有曲线可从保存的 `w_g[8]` 重建。
