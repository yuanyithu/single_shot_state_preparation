# exp102 q=0 global sampler 困难点诊断契约 v1

## 1. 目的与权限边界

本契约只回答一个问题：新实现的 hard-coset global sampler 与独立 defect-trace
sampler，能否在已知困难实例和三个容易对照实例上，在固定预算内通过同一套混合与分布门禁。

- 契约版本：`exp102.q0_global.screen_diagnostic.v1`。
- 冻结配置：`config/q0_global.screen_diagnostic.v1.json`，canonical SHA256 为
  `e5fa2ebdc2f22f25342d3d8d5c5ab05027685a4def6aecf8e48e666fa72f468b`。
- 最高终态只能是 `DIAGNOSTIC_SCREEN_PAIR_FOUND`。
- 本契约没有 formal、TI、confirmation、held-out 或 production 权限；不得创建
  `READY_FOR_FORMAL`、`FROZEN_HELD_OUT_PASS`，也不得进入正式 loader/freezer。

本轮是旧 `exp102.q0_global.discovery.v1` 终止后的独立诊断。父 run
`exp102_q0_global_20260721_204b37d` 因必需 TI contingency 超时而
`RUNTIME_EXHAUSTED`；本轮不得复用其 task、raw、seed、control、report 或 marker。
父配置 SHA 和父 source commit 均绑定在新配置中，只用于审计来源，不赋予继承关系。

## 2. 冻结面板与候选方法

`HARD2`：

1. `m06_c00,p=.04,d00,attempt022`
2. `m08_c06,p=.04,d00,attempt022`

`EASY3`：

1. `m03_c00,p=.10,d00,global_fresh_v1`
2. `m04_c00,p=.07,d00,global_fresh_v1`
3. `m05_c00,p=.10,d00,global_fresh_v1`

候选固定为五个 hard-coset 方法 `RC8-QC1/QC4/J08/J12/J16` 和三个
defect-trace 方法 `DT16/DT32/DT64`。每个 method/cell 使用 `P`、`U` 两个对抗性
初始化族，每族 16 条独立轨迹。完整 measurement control 必须恰有
`8 * 5 * 2 * 16 = 1280` 个 task；bias control 必须恰有 `3 * 5 = 15` 个 task。

本轮明确要求八个候选全部进入 screen。若 fresh sampler-only runtime consensus 判定任一
候选在最小允许资源下单轨迹超过 2 小时，整个诊断为 `RUNTIME_EXHAUSTED`，不得删掉该方法
后继续，也不得让旧 full-sector TI timing 参与本轮 runtime 判定。

## 3. 资源、时钟与 schedule

允许的资源档只有：

- `T1 = (burn 2048, measurement 8192)`
- `T2 = (burn 4096, measurement 16384)`
- `T3 = (burn 8192, measurement 32768)`

runtime consensus 只读取 sampler timing，选择三节点均合格的最大共同档。bias tuning 固定为
8 条链、4096 sweeps。诊断 schedule 冻结为 24 小时：preflight 8h、bias 12h、
measurement 22h、analysis 24h。任何 deadline、source/archive/manifest SHA、control 或 ownership
不一致都 fail closed；已有失败 marker 的 deployment 不得原地重跑。

## 4. 隔离 identity、seed 与 raw

本契约拥有独立版本：

- task：`exp102.q0_global.screen_diagnostic.tasks.v1`
- hard raw：`exp102.q0_global.screen_diagnostic.hardcoset.raw.v1`
- defect raw：`exp102.q0_global.screen_diagnostic.defect_trace.raw.v1`
- bias raw：`exp102.q0_global.screen_diagnostic.defect_bias.raw.v1`
- report：`exp102.q0_global.screen_diagnostic.report.v1`
- decision：`exp102.q0_global.screen_diagnostic.decision.v1`

轨迹和 bias seed root 固定为 `q0_global_screen_diagnostic_v1`；character seed namespace 固定为
`q0_global_screen_diagnostic_characters_v1`。困难/容易 cell 的 disorder uniforms 有意保持冻结实例
本身不变，但 sampler stream 与旧 discovery 完全不同。raw 必须使用 `allow_pickle=False`，保存
完整 fixed-clock state/label/weight/counter/hash/timing，并由 analyzer 独立逐位 replay；schema
缺失、额外字段、非有限值、身份或 replay 不一致均为 `CONFLICT`。

defect measurement 必须逐 cell/method/tier 绑定已经 replay 验证的 bias task fingerprint、raw SHA
和 bias SHA。measurement manifest 只能在 15 个 bias raw 全部存在且正确后物化。

## 5. 统计门禁与预先选择规则

每个 cell/method 必须沿用 global discovery 的冻结统计门禁：

- `SE_total(q_top) <= .03`；
- `P/U` 的 `q_top`、character `D2_norm`、normalized mean weight 一致；
- energy、basis characters 与 64 个冻结诊断 characters 满足 `Rhat <= 1.05`、非退化
  `bulk ESS >= 400`，并拒绝共同冻结；
- defect trace 还须满足每轨迹 D0 observations、leave-return excursions、D0 ESS 和
  `Dmax` boundary occupancy 门禁。

候选选择严格分两步且禁止 runner-up rescue：

1. 在所有五个 cell 均通过的方法中，独立选择 hard-coset 方法，冻结排序键为
   `(core_seconds, 1 if QC4 else 0, joint_block_size or 0, method_id)`；因此完全同 core time 时
   顺序为 `QC1,J08,J12,J16,QC4`。
2. 独立选择 aggregate D0 ESS/core-second 最大的 defect-trace 方法；平局按
   `DT16,DT32,DT64`。

只比较这两个预先选中的 primary。它们必须在全部五个 cell 上同时通过 `q_top`、`D2_norm` 和
weight 分布一致性。若该 primary pair 不一致，即使别的较慢 pair 一致，也必须输出
`NO_CROSS_MECHANISM_AGREEMENT`。

## 6. 终态

完整且 replay-valid 的 1280-task report 只有四个统计分支：

- `PAIR_FOUND` -> `DIAGNOSTIC_SCREEN_PAIR_FOUND`
- `NO_HARD_COSET_PASS` -> `UNRESOLVED_NO_HARD_COSET_PASS`
- `NO_DEFECT_TRACE_PASS` -> `UNRESOLVED_NO_DEFECT_TRACE_PASS`
- `NO_CROSS_MECHANISM_AGREEMENT` ->
  `UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT`

terminal decision 必须绑定完整 report 的 canonical SHA 和文件 SHA，并始终保存以下五个正式
阻断：`NO_T_VS_2T`、`NO_FRESH_HARD2_CONFIRMATION`、
`NO_CONF17_RES6_GAP8_SMALL6`、`NO_TI_OR_REVIEWED_INDEPENDENT_ORACLE`、
`NO_HELD_OUT`。因此，即使得到 `DIAGNOSTIC_SCREEN_PAIR_FOUND`，结论也只表示新方法值得另立
正式 discovery 契约，不能解释为任何参数点已经有可信物理结果。
