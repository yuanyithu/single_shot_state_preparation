# Exp102 下一步综合计划（2026-07-27）

状态：**规划文档 / 无实验 authority / 不授权 remote、formal、held-out 或 production**。

本文基于 [`EXP102_PROGRESS_REVIEW_20260727.md`](EXP102_PROGRESS_REVIEW_20260727.md)、
现有 validation 001--060、validation 013 的 terminal raw/report、当前源码与契约做复核。
本文提出 successor 的工作顺序和停止条件；任何实际实验仍须另立 fresh contract、config、seed
namespace、raw schema 和 immutable source SHA。

## 1. 结论先行

1. **停止无边界地寻找第 21 个 collapsed-B kernel。** 当前最有希望的 primary 是 HP64，
   但它只在 5 个单-disorder diagnostic cells 上通过自身门禁，尚未通过 fresh T/2T、整 cell、
   扩展 panel、formal tuning 或 held-out，不能称为“已找到收敛采样器”。
2. **当前有两个缺口，而不只是一个。** 一是 HP64 缺少跨 disorder、跨长度和 held-out 的
   primary 证据；二是缺少能在 large-k hard cells 上给出同一 `q_top` 或严格区间的正交确认。
   MAM 在 m8 明确不收敛；但在 m6 它自身门禁通过，仍与 HP64 相差约 30 SE，因此也不能把
   全部冲突简单归因于“确认方坏了”。
3. **Nishimori audit 必须加入，但只能是辅助的 ensemble 必要门。** 它不能单独替代独立确认；
   现有 `aggregate.py` 又硬绑定旧 PT-v1 raw/freezer，不能直接套到 HP64。
4. **暂不删除或降级 U gate。** U 从高权重尾部回到主支撑不是估计 `q_top` 的数学必要条件，
   但在没有全局 retained-mass certificate 时，它仍是最强的非平稳性证伪之一。HP64 在 013
   已经通过 U，因此现在放宽 U 既无必要，也会削弱证据。
5. **m3 full-sector TI 是 sampled orthogonal anchor，不是 exact ground truth。** 它枚举全部
   logical sectors，但每个 sector 内仍用 MCMC 和数值积分。应加强它的独立链、正反积分、
   T/2T 和 grid diagnostics 后再与 HP64 比较。
6. **执行顺序改成 easy -> anchor -> discrepancy -> hard，但正式目标不变。** `p=.04`、
   `m=6..8` 和原 HARD2 不能被静默删除；若最终改做 `m<=5` 或 `p>=.07`，必须另立 restricted
   experiment，不能称为原 exp102 成功。
7. **先从总 workload 倒推资源。** 旧 `2h/trajectory` 只是 discovery 淘汰线，不是生产预算。
   在正式设计 sampler replication 之前，不得生成新的 production config。

## 2. 对外部评审的采纳与纠正

| 评审意见 | 本计划判定 | 理由 |
|---|---|---|
| HP64 是当前最强候选 | 采纳，但降权表述 | 013 的 method-specific 5/5 属实；最高权限仍是 diagnostic |
| 真正问题只是 MAM 确认方坏了 | 不完全采纳 | m8 支持此解释；m6 的 MAM 自身有效却与 HP64 约 30 SE 不一致 |
| 012/013 的 HP P/U 差扩大 25 倍 | **纠正为事实误读** | `.91317/.99273` 是 HP64/MAM 方法均值，不是 HP64 P/U |
| Nishimori 可直接充当独立确认 | 修改后采纳 | 纳入 mandatory auxiliary audit，但不作为 sole confirmer |
| m3 TI 是 exact ground truth | **不采纳** | logical sector 全枚举，sector 内仍是 MCMC plugin estimate，无 coverage |
| 将 U 降为只报告 | 当前不采纳 | 尚无低能遗漏 basin 的全局质量上界；S/MAP 也尚未通过 |
| 换掉 `m06_c00` | 修改后采纳 | 保留旧回归 sentinel，另加 distance-balanced m6 code；不改 registry |
| `p=.04` 变成 optional、先缩范围 | 只采纳执行顺序 | 改正式范围需要新科学契约，不能改写原目标 |
| 探索/证据成本分层 | 采纳 | 小时级本地筛先于 immutable 多节点证据链 |
| 给算法搜索写元停止条件 | 采纳 | 见第 10 节 |

### 2.1 Q6 的直接纠正

validation 012 的 m8 HP32 为：

```text
P = .9143072238
U = .9111879926
|P-U| = .0031192311
```

validation 013 的 m8 为：

```text
HP32: P = .9136584419, U = .9134556192, |P-U| = .0002028227
HP64: P = .9128439675, U = .9134917737, |P-U| = .0006478062
MAM:  P = .9922880183, U = .9931659402
```

因此更长的 013 中 HP 的 P/U 差反而更小。`.91317/.99273` 是 HP64 与 MAM 的方法间
结果，见 [`013 README`](../validation/013_q0_hgp_global_screen_20260722/README.md)
及本地冻结 raw `validation/013_q0_hgp_global_screen_20260722/remote_run/exp102_q0_hgp_screen_v2_20260722_4d134ee/control/hgp_report.json`（不纳入 Git）。
不为这个不存在的矛盾另开实验；012 继续只作 provisional design evidence。

### 2.2 HP64 的准确权限

HP64 在 013 的 5/5 表示：对冻结的 HARD2+EASY3 单 disorder panel，P/U、logical/B
characters、Rhat/ESS 和 HP transport 的**内部诊断**均通过。它不表示：

- 五个 `(m,p)` 参数点已经认证；
- 未观测 basin 已被排除；
- HP32 可以作为独立确认；
- MAM 冲突已被解释；
- formal tuning、held-out 或 production 可以启动。

013 自身仍列出 `FRESH_T_AND_2T_HARD2`、扩展 panels、m3 anchor、formal tuning 和 held-out
为缺失阶段。

## 3. 先固定证据角色，后开发算法

任何新候选在写代码前必须回答“完全成功后填哪个缺口”：

| 角色 | 最低要求 | 当前对象 |
|---|---|---|
| Primary sampler | 严格目标、合法支持、fresh T/2T、对抗初态、整 cell 与 held-out | HP64（仅 promising） |
| Orthogonal confirmer | 在同一 cell 估计同一 `q_top`/sector distribution，或给严格覆盖区间；主要错误模式不与 primary 重合 | large-k 当前为空缺 |
| Rigorous certificate | 对遗漏质量、normalizer 或 estimator bias 给可传播到 `q_top` 的严格界 | 当前为空缺；旧 tail/WMC 界不够紧 |
| Auxiliary audit | 能证伪部分错误，但不能单独证明目标量正确 | Nishimori、exact small-HGP、replay、B diagnostics |

答案不属于前三项的候选，只允许小时级结构诊断，不进入三节点 immutable evidence 流程。
collapsed-B 方法即使成功，也不能自动升级为 orthogonal confirmer。

## 4. 初始化与 U：不走两个极端

### 4.1 保留的硬事实

- 非零 syndrome 下物理全零 state 不在 hard coset；不能从非法零态启动。
- 移位坐标的零态就是 P；所有链从它开始会制造共同冻结的表面一致。
- P 可暴露 planted freeze，exact-K0 U 可暴露 broad-support burn failure，truth-free MAP/S
  可暴露低能但不同 logical/B basin 的冻结；三类信息不互相替代。

### 4.2 U 尾界的正确用途

022/023 的 `Pr_pi(|e|>=w)` 上界严格说明当时 U 链仍停在后验可忽略的高权重尾部；
它不能说明低权重区域内不存在另一个高质量 basin。一个 basin 可以在 K=0 下体积极小，
但因低能简并度而有显著 posterior mass，16 条 U 也可能全部错过。

因此当前路线继续要求：

1. U 在预先冻结的 burn clock 退出有严格 negligible-mass 上界的高权重集合；
2. 固定测量时钟下，U 与 P/MAP/S 在 logical、B、weight、likelihood 上满足分布门；
3. P/MAP/S anchors 结果无关、truth-free（P 除外）、logical-signature 与 B coverage 明示；
4. U 不要求“命中 planted state”，只要求不再保留可证伪的初始化记忆。

未来只有在另立 `WARM_START_WITH_CERTIFIED_COVERAGE` 契约、严格证明
`pi(R^c)<=epsilon`，并把其传播为约 `2 epsilon` 的 purity 偏差预算后，才可取消 worst-case U
收敛要求。当前 validation 034 的 tail ratio 上界约 `9.4e84`，离这个条件极远。

## 5. Nishimori：升级为强辅助门，而不是万能确认方

固定 syndrome 后，真实 label 分布为 `P`，错误 sampler 输出为 `Q`。现有 scalar audit 比较：

```text
collision = ||Q||_2^2
planted_hit = <P,Q>.
```

`Q=P` 时相等，但反向不成立。例如 `P=(.9,.1), Q=(.5,.5)` 时两者都为 `.5`，
audit 精确通过，而 normalized `q_top` 分别为 `.64` 和 `0`。共同 P 冻结时两者都可为 1；
四条链冻结在互异 label 且均不命中 truth 时两者又都可为 0。因此“冻结一定让 paired z 爆掉”
不是普遍命题。

新 Nishimori audit 必须：

1. 在当前 `physics.v2`、q=0 hard-coset 与当前 character estimator 下重建 exact small-HGP
   golden；exp101 validation 008 是 `PRE_ALIGNMENT`，不能直接作为当前认证。
2. 为 HP64 新建 raw/schema/analyzer；旧 `aggregate.py` 要求旧 freezer、三节点 PT manifests
   和 6144 raw，只复用数学思想，不复用正式 loader。
3. estimator/scoring chains truth-blind；P-start 只作对抗诊断，不能成为唯一 scoring chain。
4. 对 basis、冻结 nonbasis characters 分别检验
   `E[chi_u(L_true) m_u] = E[m_u^2]`，再做 purity omnibus、weight/B exchangeability 和
   randomized-rank diagnostics。
5. 使用等价门，而不是“未在 5 sigma 拒绝”：

   ```text
   |mean difference| + z_alpha * SE <= delta_N
   SE <= SE_max
   ```

   `delta_N` 必须直接绑定允许的 `q_top` bias。
6. 冻结 negative controls：common-P freeze、four-distinct-label freeze、uniform-label wrong
   sampler、wrong temperature、label permutation 和 wrong posterior。明确记录 uniform `Q`
   等不可消除盲区。
7. 只有每个 disorder 的 sampler gate 已通过后才计算 ensemble identity；不得用 ensemble
   抵消覆盖单 disorder nonconvergence。

Nishimori 最终身份是 `MANDATORY_ENSEMBLE_CALIBRATION_AUDIT`，不是 sole independent confirmer。

## 6. Character gate 的 operating characteristics

HP32 的 m3 唯一失败是 163 个 B characters 中的 `column_08`：`.0404396>.04`，但其
`3SE+.005=.08174`，其它 B gates 均通过。一次只读的同样本量 P/U 随机重分组提示，在同分布
假设下 max-character 超过 `.04` 的概率约 24%；这是 retrospective 诊断，不改变 013 历史终态，
但说明必须先校准 gate。

新校准 validation 使用 exact small-HGP iid posterior draws，并注入已知 `.02/.04/.06` character
shifts，跨 `r`、trajectory 数与 clocks 估计：

- 正确 sampler 的 whole-catalog false-reject；
- 给定真实偏差的检出 power；
- `max |Delta|` 的 simultaneous margin；
- full logical characters 与 B slow-mode probes 各自对 `q_top` bias 的关系。

`.04` 是预注册误差容限，不是逐项 alpha 检验，不能简单事后做 Bonferroni，也不能因一次
边缘 fail 直接放宽。successor 应预注册 `PASS / FAIL / INCONCLUSIVE` 三态和 simultaneous
equivalence rule；013 的原判定保持不变。

## 7. 分阶段执行计划

### Stage 0：治理与证据矩阵（本地，只读优先）

目标：先消除“代码事实”和“科学状态”混乱，不产生 sampler raw。

1. 当前 local `main=de68bbc`，`origin/main=bacf25a`，实际落后 **17 commits**；origin 已含
   validation 052--059。根工作区很脏，禁止直接 pull/merge 覆盖用户文件。
2. 先对根目录 modified/untracked 文件做 path+SHA inventory，并和 `origin/main`、
   `direct_block_335f808` 比较；从 `origin/main` 建一个新的 clean canonical worktree 继续开发。
3. 在确认内容归属前，不删除、移动或覆盖根目录现有 validation 016--055、review 和代码草稿。
4. 生成 evidence matrix，每个 validation 分开记录：
   `method_internal_pass`、`cross_method_pass`、`cell_certified`、`formal_authority`、
   `raw_exists`、`replay/audit` 和 `scientific_role`。禁止再用“60 次全失败”掩盖 HP64 的内部
   5/5，也禁止把内部 5/5 写成参数点认证。
5. validation 060 当前只有未执行草稿；在 clean SHA 提交前不称 immutable frozen。

完成门：clean source、无丢文件的 reconciliation report、证据矩阵三者齐全。

### Stage 1：离线重算、门禁校准与资源倒推

目标：在任何新 remote measurement 前证明“门在测什么、能检出什么、全工作量是否能放下”。

1. 用 013 raw 独立重算 HP32/HP64/MAM 的 family/method 值，固化 Q6 纠正和 m6 约 30 SE
   discrepancy；不改变历史 report。
2. 完成第 6 节 character simultaneous-equivalence 校准。
3. 完成第 5 节 physics.v2 Nishimori exact golden、negative-control power study 和 fresh schema。
4. 为 HP64 建 outcome-blind runtime matrix，至少覆盖 m3/m4/m5 easy、m6 distance-balanced、
   m8 hard 和 p=.04/.07/.10；只读 timing，不看 `q_top` 选资源。
5. 对未来每一 stage 按“生成 + 独立 replay + analyzer + 2x safety”做 LPT wall/core-hour 投影。

硬停止：Nishimori 对预注册的 common-freeze/wrong-target controls 没有预期检出力时，它不能被
写进 readiness 充分条件；gate 校准若显示既高 false-reject 又低 power，先重设计门，不跑整 cell。

### Stage 2：结束 validation 060 支线，而不是开启新搜索树

060 只作为 collapsed-B exact local joint-block family 的最后一次轻量结构筛：

1. 先审计 config/script SHA 与 source cleanliness，再运行一次结构报告。
2. 若 MR2/MR3/MR4/RC1 全部超过冻结 width/memory cap，记录
   `COLLAPSED_B_LOCAL_JOINT_BLOCK_STRUCTURE_EXHAUSTED`，关闭该 family。
3. 若存在 candidate，按预注册顺序只选**一个**最小 block/最低 width/最低 memory survivor。
4. survivor 最多获得一次 exact implementation：n=10/n=13 complete conditional、detailed
   balance/stationarity、PortablePrng replay、目标不含 planted truth，以及 P/U/MAP/S conditional
   self-probability/expected movement screen。
5. 任一 exact 或四族 movement gate 失败即关闭 family；不再试更多 rows、row/column 比例或链长。

即使 survivor 通过，它也只可成为 HP64 的同族 primary 备选，不能填 large-k confirmer 缺口。

### Stage 3：把 m3 anchor 做成可信的 sampled cross-check

首选固定 `m03_c00` 的 fresh `.04/.07/.10` anchors，比较 HP64 与 full-sector TI。

TI successor 必须先补足：

- 每个 logical sector 独立 seed identity，允许 outcome-blind label parallelism；
- 至少两套独立 TI replicas；
- Kp forward/reverse integration hysteresis；
- 49/97 grid refinement或等价预注册离散化检查；
- sector 内 energy/weight 的 split-Rhat、bulk ESS 与 T/2T；
- bootstrap、grid 和 replay 的独立 raw-only audit；
- 输出继续标 `plugin estimate / no confidence coverage`，不得写 exact。

HP64 使用 fresh T 与 2T、合法 P/U/truth-free MAP/B-distinct S；012/013 raw 不复用。
两者对同一 frozen characters、`q_top`、weight 和 sector distribution 做预注册等价比较。

硬停止：任一 anchor 的 HP64 T/2T 不一致、TI forward/reverse/grid/T/2T 不一致，或两方法差异
超过预注册 effect+uncertainty gate，则 HP64 不进入整 cell。不得追加长度补救。

### Stage 4：第一个完整 easy code-p block

在 Stage 3 通过后，以 `m03_c00,p=.10` 为首个 positive-control block：

1. 先跑独立、固定规模的小 cohort，只用于 runtime/schema/power；其 raw 永不并入后续 128。
2. 根据 Stage 1 的 power 和总资源倒推 trajectory 数；不得默认照搬 diagnostic 的 P/U 各16，
   也不得默认复用旧 PT 的 4 instances。
3. 冻结后跑 128 个 fresh disorders，无自适应补样、无 valid-only 聚合、无失败替换。
4. primary estimate 由 truth-blind independent chains/cross-products产生；P-start 保留为单独的
   对抗性诊断，并要求与 truth-blind families 一致。
5. raw 同时保存 character U-statistic、collision diagnostic、planted mass、Nishimori per-character
   moments、weight/B exchangeability、全部初态/末态、timing、seed 与 replay transcript。
6. 输出最高权限为 `EASY_FULL_CELL_CALIBRATION_PASS`，不是 crossing 或 formal result。

这一步必须同时给出：positive control、Nishimori 实测 power/precision、每 code-p block 的真实
生成/回放成本。任一缺失则不扩展。

### Stage 5：先解决 m6 冲突，再上 m8 hard

1. 保留原 `m06_c00,p=.04` 作为已知回归 sentinel；不因 distance=2 删除。
2. fresh panel 追加 `m06_c02` 或 `m06_c05`（distance=8，按结果无关规则冻结），区分
   “旧失败回归”与“distance-balanced representative”。
3. m6 优先于 m8，因为旧 MAM 在 m6 自身门通过却与 HP64 相差约 `.0165`、约 30 SE；这是
   比“m8 MAM 已知不收敛”更有信息量的 confirmer 试验。
4. HP64 与任何 orthogonal candidate 都跑 fresh T/2T、P/U/MAP/S、同一 estimator 和分布门；
   Nishimori 只作附加 ensemble audit。
5. m6 discrepancy 被解释并通过后，才运行原 m8 hard sentinel；同时保留旧 HARD2 作回归，
   新 distance-balanced panel 不得覆盖它。

硬停止：两个各自内部门通过的方法仍在 m6 显著不一致，则 full-range discovery 终止为
`UNRESOLVED_CROSS_METHOD_DISCREPANCY`；没有 large-k orthogonal estimate/bound 时，不得仅凭
HP64+Nishimori 宣布 ready。

### Stage 6：readiness 决策、formal tuning 与 held-out

只有以下项目全部通过，才允许设计新 formal contract：

1. HP64 easy、m3 anchors、m6 discrepancy panel、原 HARD2 与 distance-balanced panel 的
   fresh T/2T 全过；
2. m3 sampled TI 与 HP64 一致，并通过 TI 自身门；
3. physics.v2 Nishimori equivalence audit 通过且 SE 足够小；
4. 至少一个 large-k orthogonal confirmer 或严格 retained-mass/normalizer bound 给出同一
   `q_top` 的一致 estimate/coverage interval；
5. 2x tuning+held-out+production 总投影小于预先批准的 campaign budget。

随后另建新版本（名称待审查，不复用 `exp102.q0_pt.v1`），依次运行 fresh tuning、冻结 sampler、
held-out 和 fail-closed aggregation。旧 PT/PA/global/013/060 raw 一律不得进入正式估计。

若第 4 条在限定预算内无法满足，则原全范围结论保持
`UNRESOLVED_WITHIN_ALGORITHM_AND_BUDGET`。可以向用户提出另立 restricted experiment，
但必须使用 fresh tuning/held-out，且不能包装成原 exp102 成功。

## 8. 资源模型：先算总账

旧 v1 的单位必须分开：

| 单位 | 数量 | 说明 |
|---|---:|---|
| raw/task | 6,144 | 48 codes x 128 disorders；一个 task 内含全部 7 个 p |
| code-p-disorder evaluations | 43,008 | 6 m x 8 codes x 7 p x 128 disorders |
| old PT-v1 instance-runs | 172,032 | 每个 evaluation x 4 instances |

未来 HP64 的 trajectory 数**尚未冻结**。若机械照搬 P/U 各16，将变成
`43,008 x 32 = 1,376,256` 个 HP ensemble trajectories，尚未计 2T/replay，显然不能先跑再问成本。

nd-2/nd-3 contingency capacity 为 166 cores：

```text
72 h capacity = 166 x 72 = 11,952 core-hours
with 2x safety: committed workload <= 5,976 core-hours
```

若套旧 172,032 runs，72h 对应平均 250 s/run，含2x安全系数仅约125 s/run；若每条真用2h，
理想 wall 已约86.4天，含2x约172.7天。故后续资源门必须是：

```text
2 x (generation + replay + analysis + fixed overhead) <= approved stage budget
```

而不是只看 `single trajectory <= 2h`。正式前由用户明确批准 campaign 的 calendar/core-hour
预算；未批准时不得假设“72h discovery window”等于 production budget。所有新 remote 仍只使用
nd-2/nd-3，不使用 nd-1。

## 9. 正式目标与缩范围分支

原目标保持 `m=3..8, p=.04..10, 8 codes/m, 128 disorders`。easy-first 只是降低信息成本：

```text
easy calibration -> m3 anchor -> m6 discrepancy -> m8 hard
                 -> full-range tuning -> held-out -> production
```

若 hard/资源失败，有两个诚实终态：

1. 原范围：`UNRESOLVED_WITHIN_ALGORITHM_AND_BUDGET`；
2. 用户另行批准：创建新名称的 restricted experiment（例如 m=3..5、p>=.07），fresh
   tuning/held-out，并明确它不是 exp102 full-range success。

不得把 `p=.04` 标成 optional、删掉困难 disorder、改 sentinel、只报 valid subset 或从相邻 p
插值来维持原项目“成功”。

## 10. 元停止条件

1. validation 060 最多产生一个 implementation successor；失败后关闭当前 exact local
   collapsed-B joint-block family。
2. 同一失败机制不再通过更多 rows、更多 swaps、更多 populations、加长 T3 或改初态排列救援。
3. 每个 heavy candidate 开始前声明它是 primary、orthogonal confirmer 还是 rigorous bound；
   三者皆非则禁止 heavy run。
4. large-k 独立确认只允许最多两个预注册正交概念，并冻结总 core-hour/calendar budget；预算耗尽
   仍无结果即停止，不再无限追加 validation 编号。
5. easy full cell、m3 anchor、m6 discrepancy、hard sentinel 任一 frozen stage 失败，均不补样、
   不延长、不事后改 gate；根据终态返回上一层科学决策，而不是排列同族 kernel。
6. 任一 stage 的 2x 总工作量超过批准预算，直接 `RUNTIME_EXHAUSTED`。
7. 不得通过共同 P/zero、删除 U、删除 `m06_c00`、把 `.04` 设为 optional、valid-only 聚合、
   合并 exploratory raw 或降低门槛制造通过。
8. 只有 exact target/support、estimator 对齐、独立性、gate power、权限边界和全局资源同时说清，
   才讨论 acceptance、局部 ESS 或 kernel 微优化。

## 11. 建议立即执行的顺序

1. 安全建立 `origin/main@bacf25a` clean worktree并完成 dirty-root reconciliation；不覆盖用户改动。
2. 生成 validation evidence/authority matrix，正式纠正 Q6 与 m3 TI 的定位。
3. 完成 character gate 与 physics.v2 Nishimori negative-control 校准。
4. 完成 HP64 outcome-blind runtime matrix和整 stage 2x成本模型。
5. 本地执行一次 validation 060；按第 7.2 节终止该支线。
6. 冻结 m3 HP64 + sampled-TI anchor contract，先过 exact/replay/runtime，再运行 fresh anchors。
7. anchors 通过后才启动 `m03_c00,p=.10` 的 128-disorder easy full-cell calibration。
8. easy block 通过后，先解决 m6 约30 SE冲突，再决定是否值得投入 m8 hard 和 formal 路线。

这套顺序的核心是：**先验证我们测的是交付量、门有正确 operating characteristics、成功后能
解除哪个封锁，再花服务器时间。**
