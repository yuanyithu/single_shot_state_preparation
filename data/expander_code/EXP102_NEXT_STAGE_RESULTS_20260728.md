# Exp102 下一阶段实验结果与决策（2026-07-28）

状态：**Stage 0--2 已完成；在本地科学门禁处 fail-closed；没有启动远端 measurement。**

本文记录对 [`EXP102_NEXT_STEP_PLAN_20260727.md`](EXP102_NEXT_STEP_PLAN_20260727.md) 和
[`EXP102_PLAN_ASSESSMENT_20260728.md`](EXP102_PLAN_ASSESSMENT_20260728.md) 的实际执行结果。
它不是 sampler、参数点或正式物理结果，也不授权 remote、formal、held-out 或 production。

## 1. 结论

下一阶段并非停在服务器队列中。计划要求在花远端算力前先确认“门禁测到的就是最终交付量”，
这一前置检查已经实际运行，并发现旧 character maximum 门不能支撑它声称的 `q_top` 误差解释。
因此 Stage 3 的 m3 anchors、easy 128-disorder block、m6/m8/HARD2 均按预注册停止规则没有启动。

当前最重要的结论不是 `.9779` 与 `.98` 的边缘数值，而是：

- `max_u |delta m_u| <= .04` 对完整已观测 character catalog 最多只给 `.08` 的
  mean-square/purity 差界；
- 只有完整非零 logical-character catalog 的 mean square 才等于 `q_top`；
- large-k 的有限 sampled characters 不覆盖未观测 character tail；
- 因而只增加 calibration trials、放宽 character 门或直接上服务器，都不会修复 estimator 与交付量错位。

下一项合法工作是 fresh、local-only 的 `q_top/D2 delivery-gate redesign`，而不是 sampler measurement。

## 2. 已执行 validations

| Validation | 实际终态 | 结论与权限 |
|---|---|---|
| 061 governance | `STAGE0_GOVERNANCE_REPORT_COMPLETE` | 完成 dirty-root、worktree 和 validations 001--060 authority inventory；未复制或覆盖用户文件 |
| 062 character calibration | `CHARACTER_GATE_REDESIGN_REQUIRED` | 五个 frozen operating points 全失败；独立 audit 通过对该失败的重算；无 sampler/remote 权限 |
| 063 Nishimori calibration | `NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT` | 30 exact rows 中 14 个 correct-posterior groups 在 `N=2048` 精度不足；旧 auditor 另有 message-taxonomy conflict |
| 064 HP64 evidence/resource | `RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE` | 旧 raw 重算与 package audit 通过，但资源覆盖缺 m7、绝大多数 p 和跨 code/disorder timing；所有 full-grid strict totals 为 `null` |
| 060 joint-block structure | `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND` | MR2 是唯一 survivor（width 25，单表下界 512 MiB）；只保留为 HP64 真正失败后的同族 contingency |
| 065 Nishimori rebind | `CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS` | 完整 payload 有 11 个 MAP-tie 数值 mismatch；独立 verifier 只认证冲突记录正确，不是原 audit PASS |

060--065 都没有生成远端 sampler measurement raw，也没有认证任何 cell、`(m,p)`、
`READY_FOR_FORMAL`、`FROZEN_HELD_OUT_PASS` 或 production 结果。

## 3. 关键数值

### 3.1 Character gate

最大 operating point 为 32 trajectories x 16384 draws。exact logical、exact collapsed-B 和
synthetic logical-511 通过；synthetic B-688 与 logical-4160 的 simultaneous-coverage
Wilson lower 为 `.9779025636 < .98`，所以 selection 和 confirmation 都是 `null`。

062 report SHA 为
`8a36cf41397e6332e9a9c789e5217cfd8d2274e68f85a4b4b591ede5e13d488a`；独立 audit 终态为
`INDEPENDENT_AUDIT_PASS_CHARACTER_GATE_REDESIGN_REQUIRED`。

### 3.2 HP64/MAM 与资源

064 从 validation 013 raw 独立确认：

- m8 的 `.91317/.99273` 是 HP64 与 MAM 的 method-level 差，不是 HP64 P/U；
- HP64 m8 P/U 差约 `.0006478`；
- m6 P-family HP64/MAM 差 `.01659637`，paired SE `.000542538`，即 `30.5903 SE`。

资源输出只能作 scenario proxy。即使较乐观的 8-trajectory T1 same-m proxy，full-grid 仍约
`162495` safety core-hours、75-core ideal wall 约 `2166.6 h`；但这些不是 confidence bound，
不能据此选档或启动 campaign。m3 easy-128 同一代理约 `76.96` safety core-hours，也仍需 future-schema
真实小 cohort 才能冻结。

### 3.3 Nishimori 审计盲区

065 证明 063 除 14 个文本前缀差异外，还有 11 个大于 `2e-13` 的 payload mismatch，最大
`.03400704`。三个 hard coset 具有严格相同的 logical-sector weight enumerator：

| p | syndrome | report/oracle MAP label |
|---:|---:|---:|
| `.04` | `05` | `0 / 15` |
| `.04` | `06` | `0 / 5` |
| `.10` | `03` | `10 / 0` |

浮点 posterior 优势只有约 `2.8e-17`，但不同 `argmax` 合法 tie choice 会改变 MAP-derived
character controls。正确记录必须同时是 `full_payload_match=false` 和
`terminal_gate_invariant=true`；后者不能包装成 audit PASS。以后任何 MAP control 都必须先冻结
基于精确 weight enumerator 的 canonical tie 语义。

## 4. 科学 red-team 后的下一项契约

fresh delivery-gate successor 应直接围绕最终交付量设计：

1. primary equivalence 使用 `|delta q_top_hat| + calibrated uncertainty <= .04`；raw 数值不裁剪；
2. 另设 full-label collision/`D2_norm` 分布门，避免“相同 purity、不同 sector support”通过；
3. 在 exact label distributions 与 outcome-blind synthetic label distributions 上分别校准 null
   false reject，以及真实 `.02/.04/.06` 的 `q_top`/D2 detection power；
4. sampled characters 只可按明确抽样设计给总体 estimand 的概率型 finite-population uncertainty，
   不得声称逐项覆盖未观测 characters；
5. per-character maxima、B/weight、Rhat/ESS、constant-character burn crossing 继续作为独立慢模态与
   transport 诊断，不能被 `q_top/D2` 门替代；
6. common-P freeze 等任何 distribution-only gate 的已知盲区必须显式保留为 negative control，不能
   通过统一初态隐藏；P/U/MAP/B-distinct S 的合法对抗性仍保持。

该 local calibration 即使通过，最高权限也只是允许冻结 Stage-3 m3 anchor contract；它自身不授权远端。

## 5. Large-k 独立确认与停止条件

MR2、HP32 和其它 collapsed-B 同族方法不能充当 HP64 的 large-k orthogonal confirmer；Nishimori
也只是 auxiliary audit。原 MAM-IMH8 机制上较正交，但 m8 已有明确 transport failure，不能通过加长
T3 原地复用。若未来 MAM-v2 占用最多两个 confirmer slots 之一，必须先有 outcome-blind 的
logical-signature/basin proposal coverage、逐 component accepted cross-signature moves，以及同一
`q_top/D2` estimand 的本地 viability contract。第二 slot 尚无成熟对象时应诚实留空。

任何远端工作前必须同时冻结：最多两个 confirmer 概念、各自与 HP64 不共享主要失败模式的论证、
总 core-hour/calendar budget、future-schema runtime matrix 和硬停止条件。缺一项就保持
`UNRESOLVED_WITHIN_ALGORITHM_AND_BUDGET`，不以加链、删 U、共同 P/zero、删 hard disorder 或放宽门补救。
