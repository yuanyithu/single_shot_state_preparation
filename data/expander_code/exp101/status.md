# exp101 status — physics v2 / scan v3 certified

**当前指针：DONE — `exp101.physics.v2` / `exp101.scan.v3` 已认证**

**最后更新：2026-07-14**

物理权威仍是 `PHYSICS_CONTRACT.md`。`validation/014_paper_alignment_20260713/` 继续认证
`exp101.physics.v2`，但其中 scan v2 的 valid-only 聚合只作历史审计，不能支持 publication、
crossing 或 FSS。scan v3 的权威证据必须来自
`validation/015_aggregation_safety_20260714/`。015 已完成 104 项 deterministic assertions；conda
`12` 全套为 365 passed、2 个预期 alias warnings，pytest log SHA256 为
`d26b9c5a1e59fdfb50051c886866eb3c0a8506ab494154c7d80d5baf969622a7`，实现 fingerprint 为
`0e215bb1481310daf44f36f63dee129a838e625f56feb4b6a477fa508e8aa8fe`。

## 固定物理决策

- 生产固定 `sector=x_error`, `H_check=H_Z`, stabilizer=`H_X` rows,
  logical move=`logical_X`, observable=`logical_Z`, prepared state=`|+>_L`。
- 生产固定 `ensemble=true_posterior`；canonical energy 不直接读取
  `epsilon_data_true`。`legacy_delta_only` 仅输出 `formal_*`，点级状态为 `FORMAL_ONLY`。
- exp101 只实现 reduced MLD posterior；完整 preparation/Clifford channel 未实现。
- full TI 仅 `k<=10`；large-k q>0 走四独立 PT，q=0 走 validated 8-start；pairwise 仅 gap diagnostics。

## scan v3 固定决策

- sampled 二阶矩使用独立链 U-statistic。无偏估计在有限样本下可为负或越出物理区间；raw 值
  必须保留且不得裁剪。少于四链、任一 gate 失败或 purity 越界都使该 disorder `INVALID`。
- 参数点只有在所有 planned disorders 均 present 且 valid 时才是 `REPORTABLE`。missing 优先标
  `INCOMPLETE`；无 missing 但有 invalid 标 `SAMPLING_INSUFFICIENT`。两者都必须令正式 mean、SEM
  和整条 crossing input 为 NaN。
- `conditional_mean_q_top_estimate_valid_only` 及其 SEM 只供诊断。它们条件化在 gate 通过事件上，
  存在选择偏差，不得用于 publication/crossing/FSS。
- planned/present/valid/invalid/missing counts 全部保存；`paper_aggregation_fraction` 与
  `numerical_pass_fraction` 都以 planned disorders 为分母。含糊的 `pass_fraction` 在 v3 删除。
- exact posterior 与解析端点只输出 algebraic MAP bounds；普通 TI 与 sampled-valid 只输出
  plug-in estimated bounds，统一声明 `map_success_bound_has_confidence_coverage=false`。sampled
  路径的真正 `map_success_probability` 保持 `None`。
- v1/v2 chunk 与 v2 NPZ 不复用于 v3。正式分析必须通过
  `src.scan_results.load_publication_q_top`；loader 只接受选中区域内全部 `REPORTABLE` 的
  `exp101.scan.v3 + true_posterior` 数据，不提供 v2 fallback。

## 认证状态

| 工作流 | 状态 | 权威证据或剩余条件 |
|---|---|---|
| physics v2 raw/reduced、矩阵接线与 posterior 语义 | PASS | `validation/014_paper_alignment_20260713/` |
| scan v2 valid-only 聚合 | HISTORICAL_ONLY | 014 保留原样；不得作为 publication aggregation |
| scan v3 identity/schema/chunk isolation | PASS | 015 锁定 v1/v2 不复用、80 characters、mixed engine 与 fingerprint |
| 点级 fail-closed 与 planned-denominator fractions | PASS | 015 deterministic aggregation evidence（104 assertions） |
| publication loader 与恶意 schema 拒绝 | PASS | 015 loader evidence + loader tests |
| algebraic/plugin MAP bounds 与非法 weights 拒绝 | PASS | 015 bounds evidence + producer tests |
| conda `12` 全套 exp101 pytest | PASS | 365 passed；2 warnings；exit 0；完整日志与 SHA256 已留档 |

## 交付边界

- `DONE` 表示 reduced-posterior 数值管线及 scan v3 publication aggregation 已按 014+015 认证，
  不表示已经得到 expander threshold 或实现完整 preparation channel。
- v2 产物永久只供审计，不能通过重命名或条件均值迁移为 v3 publication 数据。
- 本轮不实现 adaptive retry、checkpoint/续链、crossing 拟合或 FSS/data collapse。
- 后续若增加自适应采样，必须用 pilot seed 选预算，再用独立 certification seed；不得接受 pilot
  中第一个落回物理区间的估计。
- legacy 3D exp40/41 结果属于 `legacy_delta_only`，不能当作 true-posterior threshold。
- 新实验必须保留 contract/source fingerprint、完整 resolved config、validity mask 与 PT/TI 诊断。

## 历史事实

- 旧 `259 tests + V1–V6` 与 `validation/001`–`013` 均为 `PRE_ALIGNMENT`。
- `validation/007` 暴露 pairwise-TI 在 K43 上 character 偏差最高约 1.55，仍支持永久禁用。
- 旧 scan v1 多节点曾 bit-identical；scan v1/v2 均不得复用于 v3。
- 014 的 synthetic valid=2/invalid=1 点仍发布 valid-only mean/SEM 和部分 crossing，这是 scan v2
  的真实历史行为，也是 scan v3 必须 fail-closed 的直接反例。

逐目录证据边界与 014/015 复现命令见 `validation/README.md`。
