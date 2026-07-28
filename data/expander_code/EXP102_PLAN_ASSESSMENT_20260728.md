# 对 `EXP102_NEXT_STEP_PLAN_20260727.md` 的评估（2026-07-28）

评估者：Claude（外部只读）。**本文无 authority**，不修改任何 gate、权限边界或终态判定。
所有【核实】条目均由本次独立读取 raw/report/源码得到，命令见附录。

---

## 1. 结论

**这份计划应当采纳，执行顺序建议做三处调整。** 它纠正了我上一份评审中三个实质性错误，
并且这三个纠正我都逐一核实为正确。计划的科学推理比我的评审更严谨。

但计划有一个**结构性缺口**：它的终点门（第 7 节 Stage 6 第 4 条）要求一个 large-k 正交确认方，
而 Stage 0--5 **没有任何一个阶段负责产生它**。按计划原样执行，可以走完全部六个阶段、花完全部预算，
最后仍然终止在与今天完全相同的位置。这一点必须在开工前解决，而不是留到 Stage 6 才发现。

---

## 2. 【核实】计划对我的三处纠正：全部成立

### 2.1 Q6 是我读错了数据，不是矛盾

我把 `.91317/.99273` 当成 HP64 的 P/U，实际是 **HP64 方法均值 vs MAM 方法均值**。
从 `013` terminal report 直接读出：

```text
m08_c06 cell_summaries (method-level q_top)
  HP32     = 0.913557197261285
  HP64     = 0.9131680270339482      <- .91317
  MAM-IMH8 = 0.9927278950353573      <- .99273
```

同一 cell 的族内 P/U 差为：

```text
012 HP32 : |P-U| = 0.0031192311   (P=.9143072238, U=.9111879926)
013 HP32 : |P-U| = 0.0002028227
013 HP64 : |P-U| = 0.0006478062
013 MAM  : |P-U| = 0.0008779218
```

链更长之后 P/U 差**缩小约 15 倍**，方向与我说的相反。计划第 2.1 节的数字与我核实的完全一致。
**我的 Q6 作废，不应据此另开实验。**

### 2.2 m3 full-sector TI 不是 exact ground truth

`exp101/src/sector_ti.py` 的实现是：枚举全部 `2**k` 个 logical sector（这部分是精确的），
但每个 sector 内部用 Metropolis 链沿 33 点 `K_p` 网格退火采样
（`_run_fixed_sector_chain`，`num_burn_in_sweeps=200`、`num_measurements=400`），
再用 `np.trapezoid` 做梯形积分、bootstrap 给不确定度。

因此它是 **sampled plugin estimate + 离散化误差，无 coverage**。我写的"唯一 ground truth"是错的。
计划把它定位为 "sampled orthogonal anchor" 并补上双 replica、正反积分 hysteresis、
grid refinement、split-Rhat/T/2T 是正确的处理。

补充一点计划没写、但值得预注册的风险：q=0 时 sector 内只用 stabilizer 行移动，
**TI 规避的是跨 sector 势垒，不保证规避 sector 内的 B-basin 慢模态**。
从 `K_p=0` 退火能缓解，但这正是 CSMC/CAIS 塌缩的那条路径。
建议 Stage 3 的 TI 门里显式包含 sector 内 B 慢模态诊断，而不只是 energy/weight 的 Rhat。

### 2.3 Nishimori audit 抓不住"互异 label 冻结"

我写"四条链冻结在互异 label 会让 paired z 爆掉"，这是错的。核实估计量定义：

- `collision_mass`（`q0_global.py:620`）是**跨轨迹** U-statistic，`Σ_{i≠j}<Q_i,Q_j>/(T(T-1))`，
  自配对项被排除；
- `posterior_mass_on_planted_class`（`observables.py:344`）= `Q(真类)`。

若 T 条链各自冻结在互异 label 且都没命中真类，则 collision = 0、planted_hit = 0，
**两者相等，audit 精确通过**，而报出的 `q_top ≈ 0` 是错的。k=64 时四个固定初始 label
命中真类的概率可忽略，所以这恰恰是生产中的典型情形，不是构造出来的边缘案例。

计划给的 `P=(.9,.1), Q=(.5,.5)` 反例算术也正确（两者都 = .5，而 `q_top` 分别是 .64 和 0）。

结论：Nishimori 只能是 `MANDATORY_ENSEMBLE_CALIBRATION_AUDIT`，计划第 5 节的定位正确，
其中第 6 条冻结 negative controls（含 four-distinct-label freeze）是必须的。

### 2.4 其它核实为真的点

| 计划中的陈述 | 核实结果 |
|---|---|
| `main=de68bbc` 落后 `origin/main=bacf25a` **17** commits | 属实（我上次写 16，错） |
| origin/main 已含 validation 052--059 | 属实；**不存在"待合并的 worktree"**，我的 P0 表述不准 |
| 根工作区脏，禁止直接 pull/merge | **属实且有具体风险**：4 个 untracked 文件与 incoming diff 重名，其中 `q0_hgp_full_row_gibbs.py` 与 `test_q0_hgp_full_row_gibbs.py` 的本地版本与 `origin/main` 版本**内容不同** |
| m6 上 MAM 自身门通过、却与 HP64 差约 30 SE | 属实：`delta=0.016596, se_delta=0.0005425 -> 30.6 SE`（paired SE，非 unpaired 0.0066）；`method_status` 显示 MAM 1/2 通过，失败的是 m8 |
| 旧 `2h/trajectory` 是 discovery 淘汰线 | 属实：位于 `q0_global.discovery.v1.json` 的 `resource_selection`（`capacity:166, max_trajectory_hours:2.0, safety_factor:2.0`） |

**2.4 中 m6 那条最重要**：它实质性地修正了我 §2.1 的读法。在 m8，MAM 确实坏了；
但在 m6，**两个方法各自内部门全过、仍差 30 paired SE**。这说明问题不止"确认方坏了"，
而是**现有门对某种偏差没有检出力**。计划把 m6 排在 m8 之前（Stage 5）是对的。

---

## 3. 计划的三个缺口

### 缺口 A（严重）—— 没有任何阶段负责产生 large-k 正交确认方

计划第 3 节自己写明 "Orthogonal confirmer：large-k 当前为空缺"，
Stage 6 第 4 条又把它列为 readiness 的必要条件。但：

- Stage 3 的 TI anchor 受 `FULL_SECTOR_TI_MAX_K = 10` 限制，**只能覆盖 m=3**（m=4 已是 k=16）；
- Stage 5 写 "HP64 与**任何** orthogonal candidate 都跑 fresh T/2T"，但这个 candidate 从哪来、
  由哪个阶段产出、什么时候冻结，全文没有定义；
- 第 10.4 条只说"最多两个预注册正交概念"，没说何时预注册。

**后果**：Stages 0--5 全部成功执行完，Stage 6 第 4 条仍然无法满足，终态回到
`UNRESOLVED_WITHIN_ALGORITHM_AND_BUDGET`——与今天相同。

**建议**：在 Stage 1 之后、Stage 4 花钱之前，插入一个 **Stage 3.5「正交概念冻结」**：
把第 10.4 条允许的两个 slot 现在就命名，并对每个 slot 书面论证其主要失败模式为何**不与
collapsed-B 重合**。同时明确回答一个计划没回答的问题：

> **MAM-v2 算不算这两个 slot 之一？**

MAM 是当前唯一已存在的 large-k 正交概念，其 m8 失败是 proposal 供给问题
（最大 measurement log acceptance 仅 `-53.13`），不是时钟不够。换 proposal 属于**新概念**，
还是被第 10.2 条"不再通过加长/改初态排列救援"禁止？这条边界必须先划清，
否则 Stage 5 无对手可比。

### 缺口 B（中等）—— m6 冲突有停止规则，没有裁决机制

Stage 5 规定两法在 m6 仍显著不一致就终止为 `UNRESOLVED_CROSS_METHOD_DISCREPANCY`。
但 m6 的 k=36 > 10，**没有 exact TI 可以裁决谁对**。T/2T 只能测自洽性，测不出偏差方向。

**建议（成本极低、信息量最高的单项改动）**：**把 MAM-IMH8 加进 Stage 3 的 m3 anchor**，
使 Stage 3 变成 **HP64 / MAM / sampled-TI 三方比对**。理由：

- m3 是唯一有第三方 anchor 的尺寸。若 MAM 在 m3 与 TI 一致而 HP64 不一致（或反之），
  就直接得到"哪一方的内部门可信"的证据，m6 的 30 SE 才有解释框架；
- 成本可忽略：013 中 MAM 在 m6（n=900）只用 4148 core-s，m3（n=225）约 1000 core-s 量级，
  相对 Stage 3 的 HP64 + 双 TI replica 是零头；
- 不加这一项，Stage 5 至多产出"两法不一致"，而这已经是今天已知的事实。

### 缺口 C（中等）—— 资源账现在就能算，且会改变排序

计划第 8 节把总账留给 Stage 1。但 013 的 terminal report 里已有实测 `core_seconds`，
现在就能算完（见 §4）。算出来的结论会直接影响 Stage 4 之后的路线，
所以**用户的预算批准点应当前移到 Stage 1 结束、Stage 4 开工之前**，
而不是计划第 8 节写的"正式前"。

---

## 4. 【核实】现在就能给出的资源数

从 013 terminal report 的 per-cell `core_seconds` 反推（HP64、T3 = burn 8192 + meas 32768、
每 cell 2 族 × 16 轨迹 = 32 轨迹）：

```text
m=3  n= 225    264.1 core-s / trajectory
m=4  n= 400    375.7
m=5  n= 625    673.5
m=6  n= 900   1369.5
m=8  n=1600   3739.9
拟合 cost = 0.115 * n^1.383  ->  m=7 (n=1225) 约 2140.7
```

### 4.1 Stage 4（首个 easy full cell）完全付得起

`m03_c00, p=.10, 128 disorders`，沿用同样的 32 轨迹配置：

```text
生成            301 core-hours
含 2x           601 core-hours   (计划第 8 节的 5,976 上限 -> 通过)
166 核墙钟      3.6 小时
```

**Stage 4 不是风险项，可以直接排进去。**

### 4.2 全网格在 diagnostic 配置下不可行

```text
场景                                       生成 core-h    含 2x 天数 @166 核
full grid, 32 traj, T3                       545,622          273.9
full grid,  8 traj, T3                       136,405           68.5
full grid,  8 traj, T1 clocks (1/4)           34,101           17.1
m<=5, 全部 7 个 p, 32 traj, T3                83,678           42.0
m<=5, p>=.07, 32 traj, T3                     47,816           24.0
m<=6, 8 traj, T3                              42,734           21.5
```

（m=7 与 m=8 合计占全网格的 69%。）

**含义**：计划第 9 节"原目标保持 m=3..8, p=.04..10"在**当前 diagnostic 配置下是 274 天**，
不可能实现。但这不是"必须缩范围"的证据——**同一张表显示，把轨迹数从 32 降到 8、
时钟从 T3 降到 T1，全网格降到 17 天，是可行的。**

所以真正的关键问题不是"要不要缩 m 和 p"，而是：

> **Stage 4 的 power 分析能把 production 的轨迹数与时钟压到多低？**

计划已经正确禁止了"机械照搬 P/U 各 16"（第 7 节 Stage 4 第 2 条），
但没有把"必须给出可支撑全网格的最小配置"写成 Stage 4 的**交付项**。建议补上：
Stage 4 的输出除 `EASY_FULL_CELL_CALIBRATION_PASS` 外，还须给出
`(轨迹数, 时钟) -> 估计量 SE` 的曲线，以及据此反推的全网格 core-hour 数。
这是唯一能把范围决策从"感觉太贵"变成"算出来是 N 天"的一步。

---

## 5. 我撤回的意见

| 我原来的意见 | 撤回理由 |
|---|---|
| Q6：012/013 的 P/U 差扩大 25 倍 | 读错列，实际缩小 15 倍（§2.1） |
| m3 TI 是 exact ground truth | 是 sampled plugin estimate（§2.2） |
| Nishimori 对冻结有实际杀伤力 | 互异 label 冻结且不命中真类时 audit 通过（§2.3） |
| P0：把 worktree 合并回 main | 工作已在 `origin/main`；真问题是脏根工作区（§2.4） |
| P3：把 U 降为只报告 | **不再坚持**。HP64 在 013 已通过 U，U 不是 primary 路线的约束；且计划第 4.2 节的论点成立——tail bound 说明 U 当前所在区域质量可忽略，但 U 走不出去仍是最强的非平稳性证伪。保留 U 对 HP64 路线零成本 |
| P4：把 p=.04 降为 optional | 只保留其中的**执行顺序**部分；改正式范围确实需要新科学契约，计划第 9 节的处理更正确 |

我保留的意见：证据成本分层（计划已采纳）、easy-first 顺序（已采纳）、
换 distance-balanced sentinel（已按"新增而不替换"采纳，这个改法比我的更好）、
写元停止条件（已采纳，第 10 节）。

---

## 6. 建议的执行顺序调整

在计划第 11 节的八步基础上，改三处：

1. **第 2 步之后插入「正交概念冻结」**（缺口 A）：命名第 10.4 条的两个 slot，
   书面论证失败模式不与 collapsed-B 重合，并裁定 MAM-v2 是否占用一个 slot。
   未完成此步不进入 Stage 4。
2. **第 6 步的 m3 anchor 扩为三方**（缺口 B）：HP64 / MAM-IMH8 / sampled-TI 同 cell 比对。
   增量成本约 1000 core-s 量级。
3. **060 的 implementation successor 从主线移为 contingency**（新增建议）：
   计划第 7.2 节允许 060 产出一个 exact implementation successor。但计划第 3 节自己规定
   collapsed-B 成功也不能升格为 orthogonal confirmer，而 HP64 已是通过 5/5 的 primary。
   因此该 successor 的边际价值是"第二个同族 primary"，历史上这一步的成本
   （exact conditional + detailed balance + stationarity + replay + 四族 screen，参见 058/059）
   不低。建议：**结构报告照跑**（便宜、能干净关闭该 family），
   **implementation successor 挂起**，仅当 HP64 在 Stage 3 或 Stage 4 失败时才启用。

其余照计划执行。

---

## 7. 一句话总结

计划的科学判断是可靠的，三处对我的纠正全部核实为真，元停止条件（第 10 节）尤其应当保留。
主要风险不在于任何单个技术选择，而在于**终点门要求一个没有任何阶段负责生产的东西**。
把两个正交概念现在就命名冻结、把 MAM 加进 m3 三方 anchor、
并让 Stage 4 交付"最小可行配置"的成本曲线——这三步做完，
这份计划就从"一条可能走回原点的路径"变成"一条能给出确定答案的路径"。

---

## 附录：本次核实使用的命令

```bash
git rev-list --count de68bbc..origin/main                       # 17
git diff --name-only de68bbc..origin/main                       # 与 untracked 比对，4 个重名
# 013 terminal report：method/family q_top、comparisons、method_status、core_seconds
python -c "json.load(open('.../013_.../control/hgp_report.json'))"
# 012 local report：HP32 m8 P/U
python -c "json.load(open('.../012_.../local_report_hp32_b1024_m4096.json'))"
grep -n "def \|trapezoid\|num_measurements" exp101/src/sector_ti.py
sed -n '600,640p' exp102/exp102_pipeline/q0_global.py            # collision_mass 定义
sed -n '317,360p' exp101/src/observables.py                      # planted_class 定义
python -c "json.load(open('exp102/config/q0_global.discovery.v1.json'))['resource_selection']"
```

评估者：Claude（外部只读）
日期：2026-07-28
评估对象：`data/expander_code/EXP102_NEXT_STEP_PLAN_20260727.md`
