# exp102 预实验项目进展综合评审（2026-07-27）

## 0. 本文定位与免责

- 本文是一次**外部只读评审**，由 Claude 在 2026-07-27 对 repo（main + codex worktree）与计算集群做只读检查后写成。
- **本文不是 exp102 契约的一部分，不携带任何 authority**：不修改、不放宽、不豁免任何 gate、权限边界或终态判定，也不授权任何 remote / formal / held-out / production 工作。
- 本文严格区分两类内容：**【事实】** 来自 repo 文件、报告 SHA、registry、源码与集群实测；**【判断】** 是评审者的推理与意见，可能有错，需 codex 用物理与数值论证复核。
- 评审期间未改动任何程序。唯一的写操作是把根目录 `CLAUDE.md` 同步到最新 `AGENTS.md`（见 §1.3）。

---

## 1. 【事实】当前状态快照

### 1.1 代码分支状态

| 项 | 值 |
|---|---|
| `main` HEAD | `de68bbc`（2026-07-24 14:46，"add streaming three-node preflight"） |
| codex 工作区 | `data/expander_code/exp102/deployment_worktrees/direct_block_335f808/`（git worktree，detached HEAD） |
| worktree HEAD | `bacf25a`（2026-07-24 23:19，"record hybrid row-column transport failure"） |
| 领先量 | worktree 比 main **领先 16 个 commit**，且是 main 的直系后代（`git merge-base --is-ancestor` 通过） |
| 未合并内容 | validation 052–060 的全部实验、`q0_hgp_physical_pt.py` / `q0_hgp_hybrid_gibbs.py` / `q0_hgp_full_row_gibbs.py` 等模块及其测试 |

另有三个历史 deployment worktree（`rfcg_t1_6fa489f`、`streaming_7d57bcb`、`streaming_de68bbc`），是各次远端部署的源码快照。

**在途未完成的开发**：`validation/060_q0_multirow_joint_block_structure_20260724/`，状态 `PRE-RUN / NO WIDTH REPORT / NO SAMPLER RAW`。契约、`PRE_RUN_RED_TEAM.md`、`structure_config.json` 与 `analyze_structure.py` 都已冻结写好，但**尚未执行**。这是 codex 中断处的下一步。

### 1.2 计算集群状态

**集群完全空闲，没有任何 exp102 任务在运行。**

| 节点 | load average | screen | python 任务 |
|---|---|---|---|
| nd-1 | 0.00, 0.01, 0.05 | 无 socket | 仅系统 firewalld/tuned |
| nd-2 | 0.00, 0.01, 0.05 | 无 socket | 仅系统 firewalld/tuned |
| nd-3 | 0.09, 0.04, 0.00 | 无 socket | 仅系统 unattended-upgrade |

`~/.single_shot/runs/` 最新目录为 `exp102_q0_direct_block_t1_m8_v2_20260724_6933e31`（Jul 24 05:36，服务器时钟）。此后无新 run、无新 log。

### 1.3 文档同步状态

评审开始时三份契约文档互不一致：

| 文件 | 位置 | 大小 | 时间 |
|---|---|---|---|
| `CLAUDE.md` | main | 15687 B | 07-14 |
| `AGENTS.md` | main | 36679 B | 07-24 14:12 |
| `AGENTS.md` | worktree | 47051 B | 07-24 晚 |

已将 `CLAUDE.md` 同步为最新那份（worktree 的 `AGENTS.md`，47051 B）。逐行核对确认最新版是旧 `CLAUDE.md` 的**严格超集**（`diff` 中"仅存在于 CLAUDE.md"的行数为 0），无内容丢失。

**仍待 codex 处理**：main 的 `AGENTS.md` 比 worktree 落后 108 行（缺 053–060 的记录）。在 worktree 合并回 main 之前，任何从 main 接手的 agent 读到的都是过时契约——本次滞后正是这样产生的。

### 1.4 交付状态

- **无任何 physics 结果。** 没有 pilot 通过、没有 held-out、没有 production raw、没有一个 `(m,p)` 参数点被认证。
- 没有 `READY_FOR_FORMAL`，没有 `FROZEN_HELD_OUT_PASS`。生产 worker 见不到后者不会启动。
- 正式契约头仍是 `exp102.physics.v1 / exp102.q0_pt.v1 / exp102.scan.v1`（描述的是已 `EXHAUSTED` 的 v1 生产契约）。
- 48-code registry 已冻结：m ∈ {3,…,8}，每个 m 八个码，n = 225/400/625/900/1225/1600，k = 9/16/25/36/49/64。
- 生产网格规模：6 个 m × 8 码 × 7 个 p（0.04…0.10）× 128 disorder × 4 instance = **172,032 条轨迹**。

### 1.5 预实验总账（validation 001–060，无编号缺口）

60 个 validation 目录（main 树含 001–055，worktree 含 001–015 与 047–060，并集恰为 001–060）。按阶段归并：

| 阶段 | 编号 | 方法 | 终态 |
|---|---|---|---|
| 基础设施 + PT v1 pilot | 001–004 | ladder 搜索 | 最大候选在 m=4..8 失败，fail-closed |
| PT v2 discovery | 005 | Q32 + multi-swap | `EXHAUSTED`，12 个候选**零**认证往返 |
| PA discovery | 006 | population annealing | `EXHAUSTED`，genealogy 塌缩 |
| global discovery | 007–010 | cluster/joint heatbath + defect trace | `RUNTIME_EXHAUSTED`（TI contingency 116275/251241 s > 79200 s 窗口） |
| global screen | 011 | 8 个候选 × HARD2+EASY3 | `UNRESOLVED_NO_HARD_COSET_PASS`，全部 0/5 |
| **HGP collapsed HP** | **012–013** | **HP32 / HP64 / MAM-IMH8** | **`UNRESOLVED_MAP_MIXTURE_FAIL`（详见 §2.1）** |
| logical-stratified | 014–015 | LSI-IMH | `UNRESOLVED_LSI_IMH_V0_TRANSPORT` |
| 局部核族 | 016–024 | MLB8 / CTT64 / DTC21 / CSMC / CAIS / FRG / UARE / UASRE | 全部 `NOT_VIABLE` 或 `TRANSPORT_UNRESOLVED` |
| 精确 normalizer / bound | 025–028, 034–036 | bridge / tail envelope / full-column Gibbs / WMC / trellis | 界不够紧或 `RUNTIME_EXHAUSTED`（含 depth-2 的 `9.40e84` 差距、width 378、exponent 584、38.1 h 投影） |
| IID / IS / BP | 029–033, 044–046 | IID-MIS / BP-systematic / dominance witness / BP-IMH | 权重 ESS 与 max-weight gate 失败；BP-IMH `TRANSPORT_UNRESOLVED` |
| Houdayer 系列 | 037–043 | 张量/reduced 坐标 HCA、pair kernel、collapsed-B HCA | `LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`；U/U 全是 whole-pair exchange |
| random full-column | 047–056 | center-preserving / RFCG / streaming / direct-block | 三次 `RUNTIME_EXHAUSTED` + 一次 `CONFLICT`；056 首次跑出 raw，终态 `UNRESOLVED_DIRECT_BLOCK_T1_M8` |
| 最新三次 | 057–059 | CPPT32 物理-p PT / full-row elimination / hybrid row-column | 零往返 / 单独不能运输 / 分工假设被证伪 |
| 在途 | 060 | multi-row joint block 结构筛查 | **PRE-RUN，未执行** |

**统计：60 个 validation，0 个 cell 通过认证。**

### 1.6 筛选 cell 的集中度

- 011 与 013 用了完整面板：`HARD2` = {`m06_c00,p=.04`, `m08_c06,p=.04`}，`EASY3` = {`m03_c00,p=.10`, `m04_c00,p=.07`, `m05_c00,p=.10`}。
- **016–060 的全部本地诊断固定在同一个 cell：`m08_c06, p=.04, d00, attempt022`。**
- 也就是说，自 2026-07-22 起的四十余次预实验，全部只在**一个 disorder、一个参数点**上做。EASY3 自 013 之后一次都没再跑过。

### 1.7 该 cell 在网格中的位置

- m=8 是**最大码**（n=1600, k=64, rank H_Z=768, hard coset 维数 832）。
- p=0.04 是**最低噪声**（=最低温），网格 {0.04,…,0.10} 的左端点。
- `m08_c06` 的 `classical_distance = 6`。
- **`m06_c00` 的 `classical_distance = 2`**（registry 实测；m=6 的八个码距离依次为 2, 6, 8, 4, 6, 8, 2, 6）。

---

## 2. 【判断】观察与思考

### 2.1 最重要的一点：瓶颈可能被误判了

**HP64 其实已经通过了它自己的全部收敛门。**

validation 013 的事实是：

- **HP64 通过 5/5 个 cell**——包括 HARD2 两个难 cell **和** EASY3 三个易 cell。
- HP32 通过 3/5（m3 是单个 B-character `0.0404396 > 0.04` 的边缘拒绝；m5 是真实 B 慢模态）。
- MAM-IMH8 只通过 1/2 个 HARD2 cell。

这个 run 之所以终态是 `UNRESOLVED_MAP_MIXTURE_FAIL`，原因是**跨机制 family-cell 比对 0/4**：HP64 必须和 MAM-IMH8 对上，而 **MAM 自身根本没收敛**：

- m8 的普通 P family：max Rhat `1.06088`、min ESS `379.74`；
- B projection 在 P/U 两族都 fail：max Rhat `1.08245/1.05662`、min ESS `275.33/361.16`、16 个 B character 初始化族不一致；
- 39899 次 P-family state change 里只有 330 次改变 logical label（0.827%），40735 次 U-family 里只有 288 次（0.707%）；典型链只见到 3 个 label。

**评审判断：真正卡住 exp102 的不是"找不到收敛的采样器"，而是"唯一的独立确认方是坏的"。** 契约又规定 HP32/HP64 属同一机制不能互证，于是从 013 之后的四十余次实验，实质上全部是在找第三个机制。

这个区分很重要，因为它决定了下一步该往哪投入：如果瓶颈是"HP64 不行"，就该继续找采样器；如果瓶颈是"没有合格的确认方"，那么继续找采样器就是在解错的题。

### 2.2 相当一部分实验在开跑前就注定不能解锁瓶颈

057 和 058 的报告里，codex 自己写下了这两句：

> "CPPT remains the same collapsed-B tempering family as HP64 and **cannot supply the missing independent confirmation** even if a fresh successor later worked"
> （058）"It also **shares the collapsed-B identity** with HP/direct-column methods and **cannot be the missing independent confirmation**"

**评审判断：这两句是事后写的。** 也就是说，这些候选即使全部成功，也不能解除 013 留下的封锁。043（collapsed-B HCA）同理——它在 collapsed-B 边缘上做 Houdayer，与 HP64 共享同一个坍缩变量。

这是方向性的成本浪费，不是运气问题。**建议把"这个候选若完全成功，能否充当 013 所缺的独立确认方？"提升为 red-team 的第 0 问**，答案为"否"的候选不应进入重型证据流程。

### 2.3 失败模式高度同构 —— 这个一致性本身是信息

把各次失败的数字并排看：

| validation | 低能族（P/M/S/L） | U（exact-K0 uniform） |
|---|---|---|
| 022 FRG | 到达 weight 63 | measurement 全程 weight ≥ 248 |
| 023 UARE | P weight 63 | min weight 247–262 |
| 024 UASRE | P weight 63 | min weight 135–179 |
| 056 direct-block | 归一化权重 ≈ .03888 | .097775，`delta q_top = .90374` |
| 057 CPPT32 | q_top .900885 | .144627，**零往返** |
| 059 hybrid | B weight .03922/.04065/.04159，likelihood ≈ −5.1 | .10823，likelihood −11.2326 |

几乎所有失败都是同一句话：**U 停在高能 B basin，P/M/S 停在低能 basin，两者不相遇。**

**评审判断：这种跨二十余种机制的一致性说明这不是"实现不够好"，而是 (m=8, p=.04) 上 collapsed-B 存在真实的自由能势垒。** 再换第 21 种 collapsed-B 核，先验上没有理由期待不同结果。060 的 multi-row joint block 仍在同一族内。

### 2.4 U 起点的科学地位值得重审（评审者最不确定、但认为最值得 codex 复核的一点）

**先说支持保留 U 的一面**（这一面很强，不能轻易丢）：CLAUDE.md 与 exp36 的教训是"共冻 ≠ 收敛"——只从 P 出发会把 q_top 假性钉在 1，看不到 crossing。对抗初态是必须的。

**但要质疑的是"U 必须走到 P"是否是正确的收敛判据。** 理由：

1. U 是从 832 维 hard coset **均匀**抽的，典型 weight 135–262；后验集中在 weight ≈ 63。
2. codex 自己在 022/023 里给出了严格的界：weight ≥ 248 区域的后验质量 `≤ 2^832 · (.04/.96)^(248−63) = 1.31e−5`；UARE 的对应界是 `≤ 3.148e−4`。
3. 也就是说，**U 被放置在一个后验质量已被严格证明可忽略的尾部**，然后要求它在固定时钟内走回主峰。

**评审判断：把"可忽略区域 → 主峰"的运输当作必过 gate，测的性质与交付量（disorder 平均的 q_top）之间的关系是间接的。** 交付量真正需要的是：链能在**后验有质量的区域内**遍历所有相关 logical sector。这两件事不等价。

同时必须承认反方论证：U 下不来，也就没有证据说 P basin 之外不存在**别的低能 basin**。这个担心是合理的。但对它的正确检验不是 U，而是**低能、但 logical label 不同**的起点。

**而 codex 已经在做这件事了**：052 引入的 S family 正是"logical label 和 B block 都与 P 不同的低能起点"，052 的 README 明确说这是为了堵上"旧 L 起点虽改 logical/A 坐标但与 P 共享 B block"的盲区。**这是正确方向。**

**具体建议：把 U 从"必过 gate"降级为"必报诊断项"，把低能不同 label 的 S / MAP family 一致性提升为主 gate。** 后者与 q_top 的对抗性是直接的，前者是间接且可能不可达的。

（评审者对这一条的置信度中等。如果 codex 有物理论据认为存在"U 可达、但 P 不可达"的第三类低能 basin，那么 U gate 就确实是必要的，本条建议不成立。这需要论证而不是投票。）

### 2.5 独立确认方其实已经写好了，但当前策略拿不到它

`exp102_pipeline/aggregate.py:193-198`：

```python
z = _paired_z(raw_collision[code_index, p_index] - raw_planted[code_index, p_index])
paired_z[code_index, p_index] = z
if abs(z) > config["production_gate"]["paired_audit_max_abs_z"]:   # 5.0
    code_status[code_index, p_index] = "PLANTED_AUDIT_FAILED"
```

这是 **Nishimori 恒等式** `E[w0] = E[purity]` 的配对检验（exp101 `validation/008_v3_nishimori_20260709` 的对象；`PHYSICS_CONTRACT` §4 明确允许 `epsilon_data_true` 用于"posterior 生成后的 scoring、Nishimori identity 和 ground-truth tests"）。

它恰好具备 013 所缺的性质：

- **机制无关**：这是目标分布本身在 Nishimori 线上的性质。任何正确采样器都必须满足，**不需要第二个 MCMC**。
- **对"冻结"有实际杀伤力**：生产用 4 个独立 PT instance，初态分别是 logical label `zero / all-ones / even / odd`。若各链冻结在各自初始 label，pairwise collision mass → 0 而 planted_hit 不为 0，配对 z 会爆掉。这与 P/U 对抗初态是同一思想，但校验对象是**严格恒等式**，而不是另一个可能同样不收敛的采样器。

**为什么一直没用上**：它是 **cell 级统计量**（128 个 disorder 上配对求 z），在单个 sentinel disorder（`d00, attempt022`）上根本算不出来。**§1.6 的"单 cell 筛选"策略在结构上就拿不到这个已经实现好的确认方。**

**必须诚实标注的限制**：Nishimori 配对检验是**必要非充分**条件——它能证伪，不能证明。但它验证的对象恰好是要交付的量（disorder 平均），而不是一个 sentinel 的中间量。与"再造一个可能同样不收敛的 MCMC"相比，它的证据价值更高、成本更低。

### 2.6 m=3 是唯一能给出 ground truth 的尺寸，但被 72h schedule 挤掉了

m=3 的 k=9 ≤ 10，契约允许 **full-sector TI 精确计算**。这是整个网格里唯一能拿到独立于任何 MCMC 的真值的地方。

010 里 TI contingency 因为在 nd-2/nd-3 上投影 116275/251241 s、超过冻结的 79200 s 窗口而 `RUNTIME_EXHAUSTED`，此后再没单独做过。

**评审判断：那次失败是"TI 在 72 小时大 schedule 里和别的任务抢时间"，不是"TI 本身不可算"。** 作为一次性的 anchor，32–70 小时的成本相对于已经投入的六十次预实验是划算的。

### 2.7 sentinel 选择混入了一个退化码

`m06_c00` 的 `classical_distance = 2`。d=2 的码基本没有纠错能力，013 里它的 `q_top ≈ 0.14587/0.16241` 主要是**码差**造成的，不是混合造成的。

registry 的选码规则明确"pilot outcomes、distance 不影响 registry 成员"——**作为系综这完全正确**，不应改。但**挑 sentinel 时**掺进一个 d=2 的码，会把"采样难"和"码差"两个信号混在一起，让诊断难以解读。建议 sentinel 另选。

### 2.8 p=0.04 的问题是代价/收益，不是物理意义

需要说准确：**p=0.04 并不是深到没有物理信息**。`m08_c06` 在 p=.04 的 P 链给出 q_top ≈ 0.90–0.91（不是 0.999），说明它离阈值并不远。

真正的问题是**代价/收益比**：它是全网格采样最贵的点（最低温 → 势垒最高），而对 crossing 的约束贡献最小（q_top 接近饱和）。CLAUDE.md 里已有的教训——"q_top 饱和"、"用 Δf-gap crossing 定 threshold 有偏"——指向同一结论：把最贵的点当准入门槛，性价比最差。

### 2.9 对工程严谨度的评价，以及一个成本结构上的观察

**先说正面，这一点应该保留**：exact oracle、逐位 replay、独立 raw-only audit（不 import sampler）、`allow_pickle=False`、fail-closed、immutable run、三节点 exact consensus、report self-hash、跨平台 ULP 白名单——这套证据链是可发表级别的，远高于本项目 3D 阶段的水准。**不要因为进度慢而削弱它。**

**观察**：目前每个候选都付**全套重型证据成本**（三节点 preflight、archive SHA 绑定、deployment worktree、independent audit），而 60 个候选里 0 个通过。052/053/055 三次连续 `RUNTIME_EXHAUSTED`/`CONFLICT` 都发生在**产生任何 measurement raw 之前**——也就是全部成本花在了基础设施上，科学上零信息。

**建议分层**：探索阶段用轻量本地筛（小时级、单机、只要 replay 不要三节点 consensus），只有通过科学筛选的候选才付全套 immutable 成本。这不是降低严谨度，是把严谨度用在有信息量的地方。

---

## 3. 【判断】建议 codex 下一步考虑的方向

按优先级排序。

### P0 — 治理：先合并 worktree

把 `direct_block_335f808` 的 16 个 commit 合回 main，并更新 main 的 `AGENTS.md`。理由：main 落后 16 个 commit 且契约文档过时 108 行，任何从 main 接手的 agent（人或 agent）都会读到错误的项目状态。本次评审发现的文档滞后正是这样产生的。

### P1 — 重新定义"独立确认方"，走非 MCMC 路线

停止"再找第 21 个 collapsed-B 采样器"。改为并行推进两条**不需要第二个 MCMC** 的确认路线：

1. **m=3 exact full-sector TI anchor**（k=9 ≤ 10，契约允许）。单独立项，不与其它任务抢 72h 窗口。这是唯一的 ground truth。
2. **Nishimori paired audit**（`aggregate.py` 已实现）。需要整 cell 的 128 disorder × 4 instance 才能算出 z。

两条都不需要新算法，只需要改变**运行的粒度**（从单 disorder 改到整 cell）。

### P2 — 把筛选顺序从"最难 cell"改成"从易到难、以整 cell 为单位"

用 HP64（013 里 5/5 通过的那个）先跑通**一个完整的 easy cell**：例如 m=4 或 m=5、p=0.09 或 0.10、128 disorder × 4 instance。

这一步同时产出三个目前完全没有的数：

- HP64 的 **positive control**——现在没有任何证据说明"在什么条件下现有采样器能通过全部 gate"；
- **Nishimori 确认方的实测效力**——paired z 在真实数据上是多少；
- **单 cell 的真实成本**——用于判断 172,032 条轨迹的网格是否可行。

在拿到这三个数之前，任何 fail 都无法区分"方法不行"和"gate 不可达"。

### P3 — 重审 U gate 的科学地位

见 §2.4。具体动作：把 U 降为必报诊断项，把**低能、不同 logical label** 的 S/MAP family 一致性提升为主 gate。这个改动需要 codex 先用物理论证回答 §4 的 Q2，不应仅因"跑不过"就放宽。

### P4 — 缩小首轮生产网格

先做 m ∈ {3,4,5} × p ∈ {0.07,…,0.10}，建立可复现的管线与 crossing 雏形；再向 m ∈ {6,7,8} 与 p ∈ {0.04,0.05,0.06} 推进，推到哪算哪。把 p=0.04 降级为"能算就算"的可选点，而不是准入门槛。

### P5 — 换掉退化 sentinel

`m06_c00`（d=2）不适合当 hard sentinel。改选 m=6 中距离较大的码（如 c02 或 c05，d=8）。registry 本身不动。

### P6 — 证据成本分层

见 §2.9。探索期轻量筛，通过后再付 immutable 成本。

### P7 — 060 的定位

跑完 060 作为结构筛查没问题（成本低，是本地结构分析）。但**在跑之前**应在其权限边界里写明：即使终态是 `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND`，它仍属 collapsed-B 族，**不解决 013 留下的独立确认方问题**。这样避免它成为第 21 次同族尝试的起点。

---

## 4. 【判断】建议 codex 重新考虑 / 补充考虑的问题

以下用问句形式列出，便于 codex 逐条书面回答（与其现有 red-team 文化一致）。

**Q1（交付量对齐）** exp102 要交付的是 disorder 平均的 q_top(m,p) 曲线与 crossing/FSS。当前所有 gate 都是单 disorder、单 cell、单轨迹族的。逐条核对：哪些 gate 是交付量的**必要**条件？哪些其实是更强的、与交付量只有间接关系的条件？

**Q2（U 的地位）** codex 自己已经严格证明 U 所在区域的后验质量 `≤ 3.1e−4`（023）甚至 `≤ 1.3e−5`（022）。那么"U → P 运输"是 q_top 正确性的**必要条件**，还是一个更强的条件？如果 P / MAP / S（都是低能、logical label 互不相同）三族互相一致，且 Nishimori paired audit 通过，是否已经构成充分的收敛证据？如果不是，缺的是什么——请给出一个具体的、可被数值检验的失败情景。

**Q3（确认方的定义）** "独立确认"必须是第二个 MCMC 吗？以下哪些算：m=3 exact full-sector TI、Nishimori 恒等式配对检验、BP-OSD 给出的 logical class 分布、小码穷举？如果都不算，那么"独立确认"的可操作定义是什么，以及为什么它必须严格到排除已实现的 `paired_audit`？

**Q4（缺 positive control）** 目前**在哪个 cell 上、哪个采样器、能确定通过全部 gate**？如果答案是"013 的 HP64 在 5/5"，那么为什么 013 之后的四十余次实验没有一次拿 HP64 在 easy cell 上做基线对照？如果答案是"不知道"，那么所有 fail 都无法区分"方法不行"与"gate 不可达"——这个歧义如何消除？

**Q5（gate 的假阴性率）** 013 里 HP32 在 m3 因**单个** B-character `0.0404396 > 0.04` 被判 fail，而其不确定性检验和其它所有 B gate 都通过。当 64+ 个 character 每个都要过绝对 0.04 阈值时，family-wise 假阴性率是多少？是否做过多重比较校正？在 60 次全 fail 的背景下，"gate 太严导致系统性假阴性"这个假设是否被正面检验过？

**Q6（012 与 013 的定量矛盾）** 同一个 cell（`m08_c06,p=.04`）、同一个方法族（HP）：
- 012（HP32，8 轨迹/族，burn 1024 + meas 4096）给出 m8 的 P/U q_top 差 **0.0031192**，且"strict cold-hot-cold round trips in **every** sampled trajectory"；
- 013（HP64，16 轨迹/族，T3 = burn 8192 + meas 32768）给出 P/U = **.91317/.99273**，差 **0.0795**。

**链更长、链更多之后，P/U 差异反而扩大了约 25 倍。** 这个不一致是否被诊断过？可能的解释（估计量改变、U 初始化改变、012 是假阳性、013 的 U 族有别的问题）哪一个成立？在这一点澄清之前，"HP64 5/5 通过"这个结论本身的稳健性也需要打问号——**这一条同时削弱了 §2.1 的乐观读法，应优先解决。**

**Q7（资源契约自洽性）** 052 因 24701/24812/29871 s vs 7200 s cap 而死；028 因 38.1 h 投影而死；055 因估计器把固定启动开销重复外推而死。这个 2 h/trajectory 的 cap 是怎么定出来的？若生产需要 172,032 条轨迹，即使每条只要 2 h 也是 ~344,000 core-hours。**cap 与生产规模是否自洽？** 如果不自洽，那么真正的约束应该先从生产规模倒推出来，再据此定 cap，而不是反过来。

**Q8（网格规模）** 在 0 个 cell 认证的情况下，6 × 8 × 7 × 128 × 4 = 172,032 条轨迹的网格是否应该先缩到一个能闭环的规模？闭环的最小网格是什么（多少个 m、多少个 p、多少 disorder 能支撑一次可信的 crossing）？

**Q9（sentinel 选择）** `m06_c00` 的 distance = 2。它在 HARD2 里的"难"有多少来自混合、多少来自码差？换一个 d=8 的 m=6 码重做，结论会不会变？

**Q10（停止条件 / 元判据）** 已有 60 个 validation、0 个通过。**当前的算法搜索没有预先声明的停止条件。** 什么条件下应当判定"当前算法族 + 当前 gate 组合"整体不可行，从而转向：改 gate、改交付目标（例如只交付 m ≤ 5）、或改物理问题（例如换 p 范围）？建议现在就把这个元判据写进契约——否则搜索可以无限进行下去，每一次都"只排除这一个冻结配置，不是 IMPOSSIBLE"，而这个措辞在逻辑上永远为真，却永远不产生停止。

---

## 5. 【判断】评审者的不确定性声明

为免误导，明确列出本文可能错的地方：

1. **未跑任何数值。** 全部结论基于 repo 内的报告、源码与 registry，以及对集群的只读检查。没有独立复算任何 SHA 或统计量。
2. **§2.4（U gate）的置信度中等。** 若 codex 有物理论据表明存在"U 可达而 P 不可达"的第三类低能 basin，则 U gate 确实必要，该建议不成立。这需要论证。
3. **§2.5（Nishimori）是必要非充分条件。** 它能证伪不能证明。评审者认为它优于"再造一个可能同样不收敛的 MCMC"，但这是权衡判断，不是定理。
4. **Q6 指出的 012/013 矛盾只是观察，未做诊断。** 它可能有平凡解释；但在澄清前，§2.1"HP64 已经通过"的乐观读法需要保留余地。
5. **未评估 060 的技术内容本身**，只评估了它在整体策略中的位置。
6. 本文对 codex 工作的批评集中在**策略与排序**，不针对其工程严谨度——后者评审者认为应当保留。

---

## 附录：本次评审的核查动作（可复现）

```
git worktree list                          # 发现 codex worktree 与 16 个未合并 commit
git merge-base --is-ancestor de68bbc bacf25a
ssh yuany 'ls -lt ~/.single_shot/runs/'    # 最新 run = Jul 24
ssh yuany 'for n in nd-1 nd-2 nd-3; ...'   # 三节点 load 0.00、无 screen、无 python 任务
python3 -c "json.load(open('registry/registry.json'))"   # 码尺寸与 classical_distance
grep -n "_paired_z" exp102_pipeline/aggregate.py         # Nishimori paired audit 实现位置
diff CLAUDE.md <worktree>/AGENTS.md        # 确认超集关系后同步
```

评审者：Claude（外部只读评审）
日期：2026-07-27
评审对象：`data/expander_code/exp102/`，main `de68bbc` + worktree `bacf25a`
