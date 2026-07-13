# exp101 v2 论文语义对齐与数值管线修复计划

日期：2026-07-13。当前状态：`DONE`。唯一物理权威：
`PHYSICS_CONTRACT.md`（`exp101.physics.v2`）；扫描协议：`exp101.scan.v2`。

## 1. 已冻结决策

- 生产固定 `sector=x_error`、`ensemble=true_posterior`；不再询问 true/legacy 二选一。
- x-error convention 是 `H_Z/H_X rows/logical_X/logical_Z/|+>_L`；对偶 z-error 对应 `|0>_L`。
- exp101 实现 reduced MLD posterior，不实现完整 preparation/Clifford channel。
- canonical energy 为 `Kp|e|+Kq|H_check e xor effective_syndrome|`，不直接读取真实错误。
- 正式 ensemble 名是 `true_posterior`、`legacy_delta_only`；两个旧名只作 deprecated alias。
- full TI 只允许 `k<=10`；large-k pairwise 只作 free-energy-gap diagnostics。
- v1 chunk 永不复用；当前验证只认 `validation/014_paper_alignment_20260713/`。

## 2. 工作包与验收条件

| 工作包 | 内容 | 当前状态 | 验收条件 |
|---|---|---|---|
| P0 契约与文档 | 唯一 physics contract、局部 AGENTS、notes、erratum、alignment report、validation index | PASS | 文档无双权威/错误 mapping/旧 PASS 冒充 |
| P1 模型变量 | disorder 正式字段；Gibbs/planted wiring 拆分；canonical alias normalization | PASS | energy independence、true-vs-legacy、q=0、alias tests |
| P2 section/observable | 三类 section 命名；absolute/relative 双输出；boundary-only invariance | PASS | domain guard、Mattis sign、logical-shift 反例 |
| P3 统计量 | 删除 w0；purity/planted/MAP；basis/nonbasis 加权；U-stat/jackknife/FPC | PASS | 人工 posterior/character 表与 Bernoulli 重复实验 |
| P4 engine/gates | auto 三路；TI hard guard；gap-only API；四实例 PT；INVALID 传播 | PASS | routing、large-k rejection、PT failure integration |
| P5 scan v2 | 完整 task fingerprint/cache key；`scan_results.npz`；动态 character 维；valid-only aggregate | PASS | v1 isolation、80 chars、schema、invalid-safe mean |
| P6 exact oracle | raw preparation/reduced canonical 枚举与完整 posterior statistics | PASS | 逐构型、Z、sector weights、q_top、MAP 全相等 |
| P7 validation/交付 | conda 12 全测试、014 证据、report/status/实验报告、scoped commit/push | PASS | 014 权威证据；scoped commit/push 随本次交付完成 |

## 3. 决定性测试矩阵

### A. 物理语义

1. 小 CSS 码遍历 `sigma_prep/measurement_error/epsilon_data_true/a`，验证 raw/reduced 单构型
   weight、partition function、sector weights、q_top、MAP 完全相等。
2. 固定 `effective_syndrome` 时 true energy 对 `epsilon_data_true` 不敏感。
3. shifted-coordinate identity 逐构型成立。
4. `H epsilon_data_true != 0` 时 true 与 legacy 不同；q=0 是 quenched coset vs clean kernel。
5. alias 与 canonical 的 seed、task fingerprint、结果一致，manifest 只存 canonical。
6. x/H_Z=`|+>_L`、z/H_X=`|0>_L` convention 锁死。

### B. logical statistics

1. absolute/relative 每个 character 满足 planted sign；weights 是 sector translation。
2. boundary-only section shift 不变；一般 logical shift 构造可观察反例。
3. `(0.1,0.9)` 锁死 planted mass 0.1、MAP 0.9；purity/MAP bounds 成立。
4. 人工 character 表验证 basis/nonbasis 总体加权。
5. 独立 Bernoulli chains 验证 pooled square 正偏、cross-product 无偏、jackknife 合理。
6. debiased purity 越界保留 raw、标 INVALID、无 success bounds。

### C. engines、gates 与 schema

1. `k>10+ti` pre-task 报错；gap diagnostics keys 不含 `m_u/q_top`。
2. auto 精确解析 small-k TI / large-k q>0 PT / large-k q=0 8-start。
3. PT 任一实例无 round trip、min swap=0 或 cold convergence 失败 -> INVALID。
4. INVALID 不改变 mean/SEM/crossing；valid/invalid/missing count 正确。
5. `k=16,num_random_u=64` 保存全部 80 characters。
6. manifest/NPZ 包含 contract、protocol、fingerprints、git SHA、全 estimator 与 PT diagnostics。
7. family/sector/alias/engine/sampler 任一配置变化都隔离 chunk/cache identity。

## 4. validation/014 产物

目录 `validation/014_paper_alignment_20260713/` 至少保存：

- 完整 pytest 输出与退出状态；
- exact-enumeration JSON + Markdown；
- PT/aggregation integration JSON + Markdown；
- schema manifest 示例与字段审计；
- 全仓错误叙述扫描结果；
- 运行配置、git SHA 与生成脚本。

014 已保存全部上述证据并通过；复现时任一项失败都必须把 `status.md` 降回 `IN_PROGRESS`。

## 5. 文档与历史处置

- `validation/001`–`013` raw 保留，统一标 `PRE_ALIGNMENT`，不修改历史数值。
- report 保留旧性能/数值事实，但删除无条件“物理正确/已毕业/可直接生产”的结论。
- 根 `AGENTS.md`、`CLAUDE.md` 与实验报告明确 v2 已认证，接手仍先读 physics contract。
- `exp101修改说明/` 和 `文章.tex` 始终只读，不纳入本次提交。

## 6. 交付纪律

014 已全绿，status/report/实验报告已按真实证据更新。本次只提交 exp101 修复、相关根文档和
小体积验证证据，使用清晰 commit message 并 push；无关 3D 数据、临时产物和用户提供的修改
说明目录不加入提交。
