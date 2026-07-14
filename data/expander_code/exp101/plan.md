# exp101 scan v3：参数点级 Fail-Closed 与 MAP bounds 语义修正

日期：2026-07-14。当前状态：`DONE`。唯一物理权威：
`PHYSICS_CONTRACT.md`（`exp101.physics.v2`）；扫描协议：`exp101.scan.v3`。

本轮不修改 physics v2，不重新解释或迁移 scan v2 结果。014 继续认证 physics v2；
`validation/015_aggregation_safety_20260714/` 与完整测试已通过，scan v3 为 `DONE`。

## 1. 已冻结决策

- 生产固定 `sector=x_error`、`ensemble=true_posterior`；canonical energy、矩阵接线和三类 section
  边界保持 physics v2。
- v1/v2 chunk 永不复用于 v3；v2 NPZ 只作审计，禁止进入 publication 分析。
- 正式聚合零容忍 invalid/missing，不提供隐含阈值或自动重采样。
- 本轮不实现 adaptive retry、checkpoint/续链、crossing 拟合或 FSS/data collapse。
- sampled 的 `map_success_probability` 保持 `None`；只允许从有效估计 purity 生成明确无置信覆盖的
  plug-in estimated bounds。
- 后续若增加 adaptive sampling，必须以 pilot seed 选预算，再用独立 certification seed；不得接受
  pilot 中第一个落回物理区间的估计。

## 2. 工作包与验收条件

| 工作包 | 内容 | 当前状态 | 验收条件 |
|---|---|---|---|
| P0 protocol identity | protocol/task/chunk/manifest 升为 scan v3 | PASS | v1/v2 chunk isolation、fingerprint/schema tests |
| P1 点级聚合 | REPORTABLE/SAMPLING_INSUFFICIENT/INCOMPLETE/FORMAL_ONLY | PASS | all-valid、invalid、missing、legacy、single-disorder tests |
| P2 诊断与比例 | conditional valid-only mean/SEM；planned denominator；删除 pass_fraction | PASS | selection-bias 文档、counts/fractions/schema tests |
| P3 MAP bounds | algebraic 与 estimated 字段分流；五种 kind；no-coverage metadata | PASS | exact/endpoint/TI/sampled/legacy/非法 weights tests |
| P4 publication loader | 统一读取入口与 point mask；v2/legacy/failure/tamper 硬拒绝 | PASS | loader 成功和错误信息 tests |
| P5 文档与认证 | 当前文档、015、完整 conda 12 pytest、实验报告 | PASS | 015 overall PASS，status 已更新为 DONE |

## 3. scan v3 聚合契约

设参数点 planned/present/valid/invalid/missing counts 为 `D/P/V/I/M`：

1. `legacy_delta_only` -> `FORMAL_ONLY`；
2. true posterior 且 `M>0` -> `INCOMPLETE`；
3. true posterior 且 `M=0,I>0` -> `SAMPLING_INSUFFICIENT`；
4. true posterior 且 `M=I=0` -> `REPORTABLE`。

仅 `REPORTABLE` 点填正式 `mean_q_top_estimate`、`disorder_sem_q_top_estimate` 和逐 disorder
crossing input。任何其它状态都令正式 mean/SEM 与整条 crossing input 为 NaN。单 disorder 点可以
reportable，但 SEM 为 NaN。

merge 仍先计算
`conditional_mean_q_top_estimate_valid_only` 与
`conditional_disorder_sem_q_top_estimate_valid_only`，但它们条件化在 gate 通过事件上，仅供诊断，
不得用于 crossing/FSS。`paper_aggregation_fraction` 与 `numerical_pass_fraction` 均以 `D` 为分母；
v3 删除 `pass_fraction`。manifest 的固定 `aggregation_policy` 明示上述规则。

## 4. MAP bounds 与 estimator 契约

- `posterior_statistics()` 在输出 algebraic bounds 前检查 weights 全部有限、非负且
  `abs(sum(weights)-1)<=1e-12`；不归一化、不裁剪，非法输入直接报错。
- exact enumeration -> `exact_posterior_algebraic`；解析 p 端点 ->
  `analytic_endpoint_algebraic`，且 `weights_are_exact_sector_posterior=true`。
- 普通 TI -> `full_sector_ti_plugin_no_coverage`；sampled-valid ->
  `sampled_u_statistic_plugin_no_coverage`，且 `weights_are_exact_sector_posterior=false`。
- sampled-invalid、legacy 或不可得 -> `unavailable`。algebraic 与 estimated 字段永不同时填充。
- 所有逐 disorder `map_success_bound_has_confidence_coverage=false`。plug-in 绘图标签固定为
  `Estimated MAP-purity bounds (plug-in; no confidence coverage)`。
- 无偏 U-statistic 的 finite-sample realization 可负或越界；raw 值必须保留，越界使 disorder
  INVALID，不做 clipping。

## 5. publication loader

公开入口为：

```python
load_publication_q_top(path, point_mask=None) -> PublicationQTopData
```

loader 只接受 `exp101.scan.v3 + true_posterior`，并验证选中点全部 `REPORTABLE`、没有
invalid/missing、正式 mean 与 crossing 数据一致。`point_mask` 用于预先指定分析区域，不是事后
避开失败点的资格推断。错误必须列出 size、q 与 failure reasons；不提供 v2 条件均值 fallback。

## 6. 015 决定性证据

`validation/015_aggregation_safety_20260714/` 至少保存：

- deterministic aggregation/bounds/loader evidence 的 JSON 与 Markdown；
- 负 U-stat raw、点级四状态、planned denominator、整点 crossing fail-closed 证据；
- exact/解析端点/TI/sampled/legacy bounds kind 与非法 weights 拒绝证据；
- loader 的成功、point mask、v2/legacy/failure/tampered schema 拒绝证据；
- v1/v2 chunk isolation、80-character、schema/fingerprint/mixed-engine 回归；
- conda `12` 全部 exp101 pytest 的完整 stdout/stderr、退出码、log SHA256；
- Python/NumPy、Git SHA/dirty、implementation fingerprint 与 evidence inventory。

014 目录和内容原样保留，不用 v3 源码重跑覆盖。015 runner 的 `--skip-pytest` 若存在，只能产生
`INCOMPLETE` 开发证据；不能用于最终认证。

## 7. 文档与交付纪律

- 根 `AGENTS.md`/`CLAUDE.md`、exp101 AGENTS、contract、status、validation index、notes 与报告
  必须一致地区分 physics v2、scan v3、014 physics 证据和 015 aggregation 证据。
- `笔记/实验报告.md` 已在 015 与完整测试通过后用中文更新，并记录实际测试数/fingerprint。
- `exp101修改说明/` 和 `文章.tex` 始终只读，不纳入提交。
- 最终只暂存相关源码、测试、文档与小体积 015 证据；不使用 `git add .`，不混入 exp41 或用户
  未跟踪文件。
- 015 overall 已 PASS，status 已改为
  `DONE — exp101.physics.v2 / exp101.scan.v3`；使用 commit message
  `fix(exp101): make scan v3 aggregation publication-safe` 并 push。
