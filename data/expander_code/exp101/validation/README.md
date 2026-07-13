# exp101 validation evidence index

当前物理契约：`exp101.physics.v2`。当前扫描契约：`exp101.scan.v2`。

**当前 v2 认证：PASS。** 权威证据只来自 `014_paper_alignment_20260713/`；`001`–`013` 全部是
`PRE_ALIGNMENT` 历史证据，其中 runner 可能
依赖已经弃用的 ensemble 名、`eta/delta/sigma_arg/ell_ref/w0` 字段、biased sampled square、
pairwise q_top 或 `exp101.scan.v1` schema。保留 raw 文件用于审计，但不得把旧 summary 的
“PASS/ALL GREEN”提升为 v2 证据，也不得复用旧 chunk。

## 历史证据逐项分级

| # | 目录 | v2 状态 | 仍可参考的内容 | 不能认证的内容 |
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

## 当前 v2 认证

目标目录：`014_paper_alignment_20260713/`。

014 当前包含：

- conda `12` 下全部 exp101 pytest 的完整 stdout/stderr 与退出码；
- small CSS raw-paper/reduced-canonical exact enumeration JSON 与 Markdown；
- energy independence、true-vs-legacy、q=0 quenched-vs-clean、alias normalization 证据；
- absolute/relative weights/characters、posterior 三统计量、bounds 与 section 反例；
- basis/nonbasis 人工表、U-statistic、jackknife、finite-population 证据；
- engine auto routing、large-k TI rejection、gap-only diagnostics；
- 四实例 PT gate failure -> INVALID 与 INVALID-safe aggregation integration；
- 80-character、manifest/schema、task/cache isolation 与错误叙述扫描结果。

上述证据均已通过：conda `12` 全套为 343 passed（2 个预期 alias warnings），exact oracle 的
16,384 个逐构型比较最大误差均为 0，PT integration 为 fresh `computed=1,reused=0` 且按设计
INVALID，valid-only 聚合排除了失败样本。`status.md` 与 `report.md` 已据此更新为 v2 `DONE`。

### 014 统一 runner

在项目根目录、conda `12` 中运行：

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp101/validation/014_paper_alignment_20260713/run_alignment_evidence.py
```

runner 对真实 PT integration 使用 `force_recompute=True`，并硬断言 `computed=1,reused=0`；
旧的匹配 chunk 也不能替代本次证据。它生成/刷新：

- `summary.md`：统一 coverage 索引与总状态（不自动修改 `status.md`）；
- `exact_reduction_evidence.{json,md}`：raw/reduced、fixed-y、shifted-coordinate、q=0、alias 与
  posterior 统计量证据；
- `pt_aggregation_evidence.{json,md}`：四实例 PT INVALID、burn-in/measurement round-trip、
  valid-only mean/SEM/crossing 与 task/source fingerprint 证据；
- `pytest_full_output.txt`、`pytest_exit_code.txt`：完整测试 stdout/stderr 与退出码；
- `environment.json`：conda/Python/NumPy、Git SHA/dirty、implementation/task fingerprint、pytest
  log SHA256 与 coverage inventory。

`--skip-pytest` 只用于开发中刷新 exact/PT 证据；此时 `summary.md` 明确保持 `INCOMPLETE`，不能
用于最终认证。
