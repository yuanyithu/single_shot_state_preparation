# exp101 validation evidence index

当前物理契约：`exp101.physics.v2`。当前扫描契约：`exp101.scan.v3`。

**物理 v2 认证：PASS；scan v3 认证：PASS。** `014_paper_alignment_20260713/` 继续认证
physics v2，但其 scan v2 schema/聚合只作历史审计；scan v3 的 publication aggregation 只认
`015_aggregation_safety_20260714/`。`001`–`013` 全部是 `PRE_ALIGNMENT` 历史证据，其中 runner 可能
依赖已经弃用的 ensemble 名、`eta/delta/sigma_arg/ell_ref/w0` 字段、biased sampled square、
pairwise q_top 或 `exp101.scan.v1` schema。保留 raw 文件用于审计，但不得把旧 summary 的
“PASS/ALL GREEN”提升为 v2 证据，也不得复用旧 chunk。

## 历史证据逐项分级

| # | 目录 | 当前状态 | 仍可参考的内容 | 不能认证的内容 |
|---|---|---|---|---|
| 001 | `001_model_semantics_check_20260707` | `PRE_ALIGNMENT` | 发现旧 3D delta-only 与另一接线数值不同；若干 GF(2) 恒等 | 使用旧 `eta/delta/s` 语义，未证明 raw preparation 与 v2 reduced posterior 等价 |
| 002 | `002_phase1_module_tests_20260707` | `PRE_ALIGNMENT` | GF(2)、图、HGP、逻辑基、距离等结构模块的历史回归日志 | 后半日志基于旧模型字段、observable estimator、TI/scan schema；测试总数不是 v2 认证 |
| 003 | `003_family_registry_20260707` | `PRE_ALIGNMENT` | (3,4) 家族候选 seed、秩和距离的历史结构数据 | v2 task/cache fingerprint 与生产 family rule/seed 隔离尚须 014 重新锁定 |
| 004 | `004_v1_main_matrix_20260708` | `PRE_ALIGNMENT` | 旧实现内 enum/MCMC/TI 的历史数值一致性与 raw 数据 | canonical energy、absolute/relative 输出、无偏二阶矩、v2 gates/schema |
| 005 | `005_v2_analytic_limits_20260708` | `PRE_ALIGNMENT` | p/q 极限及 section fingerprint >255 bug 的历史证据 | 名称中的 “V2” 不是 `exp101.physics.v2`；旧 estimator/字段不可作当前通过 |
| 006 | `006_v1c_frame_ab_20260709` | `PRE_ALIGNMENT` | q=0 boundary-like frame 对拍及 q>0 差异的历史观测 | 一般 section change 的正确边界；不得引用为“任意 frame=gauge” |
| 007 | `007_pairwise_characterization_20260709` | `PRE_ALIGNMENT` | K43 上 pairwise 可加性失败（max character deviation 历史值约 1.55）支持永久禁用 | 任何 pairwise `m_u/q_top` 生产能力；v2 diagnostics API/schema |
| 008 | `008_v3_nishimori_20260709` | `PRE_ALIGNMENT` | 旧接线下 Nishimori 数值现象与 legacy 反例 | v2 raw/reduced golden identity、canonical alias、debiased sampled estimator |
| 009 | `009_v4_v6_redundancy_torture_20260709` | `PRE_ALIGNMENT` | 旧 ref/numba 可复现与冻结负例设计 | 4 个独立 PT 实例、冷端 R-hat/ESS、逐实例 round trips 与 v2 INVALID 传播 |
| 010 | `010_g4_remote_smoke_20260709` | `PRE_ALIGNMENT` | 旧远端 round-trip、文件回收和历史 NPZ 可读性 | 文件是 `sector_ti_results.npz`/scan v1；绝不可被 v2 resume 或 aggregate |
| 011 | `011_g4_profile_20260709` | `PRE_ALIGNMENT` | direct m=6 约 1.1 s/disorder、PT 约 302 s/disorder 的历史性能量级 | v2 正确性、PT 四实例成本、生产预算承诺 |
| 012 | `012_g4_physics_smoke_20260709` | `PRE_ALIGNMENT` | 2D 小码曲线与 expander monotonicity 的历史 smoke | 修正后的论文语义、MAP/planted/purity 统计或 expander threshold |
| 013 | `013_g4_multinode_20260709` | `PRE_ALIGNMENT` | scan v1 的跨节点可复现与旧 scratch 清理记录 | v2 canonical alias、完整 task fingerprint、cache 隔离和 `scan_results.npz` |

## 014：physics v2 认证与 scan v2 历史边界

目标目录：`014_paper_alignment_20260713/`。

014 已完成并继续有效的 physics v2 内容包括：

- conda `12` 下全部 exp101 pytest 的完整 stdout/stderr 与退出码；
- small CSS raw-paper/reduced-canonical exact enumeration JSON 与 Markdown；
- energy independence、true-vs-legacy、q=0 quenched-vs-clean、alias normalization 证据；
- absolute/relative weights/characters、posterior 三统计量、bounds 与 section 反例；
- basis/nonbasis 人工表、U-statistic、jackknife、finite-population 证据；
- engine auto routing、large-k TI rejection、gap-only diagnostics；
- 四实例 PT gate failure -> INVALID 与当时 scan v2 的 valid-only aggregation integration；
- 80-character、manifest/schema、task/cache isolation 与错误叙述扫描结果。

上述 physics 证据仍为 PASS：当时 conda `12` 全套为 343 passed（2 个预期 alias warnings），
exact oracle 的 16,384 个逐构型比较最大误差均为 0，PT integration 为 fresh
`computed=1,reused=0` 且按设计 INVALID。这些测试数、fingerprint 与日志 hash 只描述 014 当时的
实现，不能冒充 scan v3 认证。

014 的 `pt_aggregation_evidence.md` 明确记录 synthetic 点 `valid=2, invalid=1, missing=0` 时仍输出
valid-only mean/SEM=`0.3/0.1`，crossing=`[0.2,0.4,nan]`。这在 scan v2 中是预期行为，但在
scan v3 下必须是 `SAMPLING_INSUFFICIENT`，正式 mean/SEM 和整条 crossing input 全部为 NaN。
因此 014 不再认证 publication aggregation。014 内旧 generic MAP bound 字段与 scan v2 NPZ 也只
供审计；目录必须原样保留，不得用当前 v3 源码重跑并覆盖。

### 014 历史 runner

以下命令只记录 014 当时的复现入口；scan v3 开发中不得运行它覆盖 014：

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp101/validation/014_paper_alignment_20260713/run_alignment_evidence.py
```

runner 当时对真实 PT integration 使用 `force_recompute=True`，并硬断言 `computed=1,reused=0`；
旧的匹配 chunk 也不能替代该次证据。它当时生成：

- `summary.md`：统一 coverage 索引与总状态（不自动修改 `status.md`）；
- `exact_reduction_evidence.{json,md}`：raw/reduced、fixed-y、shifted-coordinate、q=0、alias 与
  posterior 统计量证据；
- `pt_aggregation_evidence.{json,md}`：四实例 PT INVALID、burn-in/measurement round-trip、
  valid-only mean/SEM/crossing 与 task/source fingerprint 证据；
- `pytest_full_output.txt`、`pytest_exit_code.txt`：完整测试 stdout/stderr 与退出码；
- `environment.json`：conda/Python/NumPy、Git SHA/dirty、implementation/task fingerprint、pytest
  log SHA256 与 coverage inventory。

`--skip-pytest` 当时只用于刷新 exact/PT 证据；014 已冻结，不再刷新。

## 015：scan v3 publication aggregation 认证

目标目录：`015_aggregation_safety_20260714/`。deterministic evidence、完整 conda `12` pytest、
退出码、环境信息与 implementation fingerprint 已全部落盘并通过，scan v3 状态为 `PASS`。

015 已认证：

- 全 valid 点为 `REPORTABLE`，正式与 conditional 统计一致；单 disorder 可 reportable 但 SEM 为 NaN；
- 任一 invalid 得到 `SAMPLING_INSUFFICIENT`，任一 missing 得到 `INCOMPLETE`，legacy 得到
  `FORMAL_ONLY`；三者正式 mean/SEM 与整条 crossing input 均关闭；
- planned/present/valid/invalid/missing counts 正确，两个 fraction 都以 planned 为分母，v3 中没有
  `pass_fraction`；
- U-statistic 负值和越界 purity 原样保留，越界 disorder 为 INVALID；conditional valid-only 统计
  明确仅供诊断，不可提升为 publication eligibility；
- publication loader 接受一致的 v3 true-posterior reportable 数据与预先指定 `point_mask`，并对
  v2、legacy、失败点及 tampered schema 给出含 size/q/reason 的明确错误；
- exact posterior、解析端点、普通 TI、sampled-valid、sampled-invalid 与 legacy 的 bounds kind、
  algebraic/estimated 字段互斥、`weights_are_exact_sector_posterior` 和 no-coverage metadata 正确；
- `posterior_statistics()` 拒绝非归一化、负或非有限 weights；
- v1/v2 chunk isolation、80-character、manifest/schema/task/source fingerprint 与 mixed-engine 回归；
- conda `12` 下全部 exp101 tests 的完整 stdout/stderr、退出码和日志 SHA256。

### 015 统一 runner

在项目根目录运行：

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp101/validation/015_aggregation_safety_20260714/run_aggregation_safety_evidence.py
```

runner 应检查 `CONDA_DEFAULT_ENV=12`，以当前解释器执行
`python -m pytest -q data/expander_code/exp101/tests`，并保存：

- deterministic aggregation/bounds evidence 的 JSON 与 Markdown；
- `pytest_full_output.txt`、`pytest_exit_code.txt`；
- `environment.json`：Python/NumPy、两份 contract、Git SHA/dirty、implementation fingerprint、
  evidence test inventory、pytest log SHA256 与 overall status；
- `summary.md`：coverage index 与总状态，但不得自动修改 `status.md`。

`--skip-pytest` 只能生成 `INCOMPLETE` 开发证据，不能用于认证。本次正式证据为 104 项 deterministic
assertions、365 passed、2 个预期 alias warnings、pytest exit 0；implementation fingerprint 为
`0e215bb1481310daf44f36f63dee129a838e625f56feb4b6a477fa508e8aa8fe`，pytest log SHA256 为
`d26b9c5a1e59fdfb50051c886866eb3c0a8506ab494154c7d80d5baf969622a7`。`status.md` 已据此更新为
`DONE — exp101.physics.v2 / exp101.scan.v3`。
