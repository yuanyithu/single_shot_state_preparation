# exp101 status — v2 论文语义对齐

**当前指针：DONE — `exp101.physics.v2` / `exp101.scan.v2` 已认证**

**最后更新：2026-07-13**

权威证据位于 `validation/014_paper_alignment_20260713/`。旧 2026-07-09 的
`259 tests + V1–V6` 仍统一标为 `PRE_ALIGNMENT`，不得替代 014 或恢复旧字段/估计器。

## 固定物理决策

- 生产固定 `sector=x_error`, `H_check=H_Z`, stabilizer=`H_X` rows,
  logical move=`logical_X`, observable=`logical_Z`, prepared state=`|+>_L`。
- 生产固定 `ensemble=true_posterior`；canonical energy 不直接读取
  `epsilon_data_true`。`legacy_delta_only` 仅输出 `formal_*`，状态为 `FORMAL_ONLY`，不聚合。
- exp101 只实现 reduced MLD posterior；完整 preparation/Clifford channel 未实现。
- full TI 仅 `k<=10`；large-k q>0 走四独立 PT，q=0 走 validated 8-start；pairwise 仅 gap diagnostics。
- sampled 二阶矩使用独立链 U-statistic；少于四链、任一 gate 失败或 purity 越界都不得聚合。

## 认证结果

| 工作流 | 状态 | 权威证据 |
|---|---|---|
| physics contract、局部约束与叙述扫描 | PASS | `PHYSICS_CONTRACT.md`, `AGENTS.md`, `tests/test_contract_text.py` |
| canonical model、alias、CSS move 完备性 | PASS | model/golden tests；坏 move span 在装配时拒绝 |
| section 与 absolute/relative observable | PASS | domain、Mattis sign、boundary shift 与 logical-shift 反例 |
| posterior stats 与 sampled estimator | PASS | artificial posterior、总体加权、U-stat、jackknife、FPC |
| TI/PT/auto/gates | PASS | large-k refusal、独立-sector bootstrap、解析端点、四实例 PT INVALID |
| scan v2/fingerprint/schema/aggregation | PASS | source/task identity、80 characters、valid-only mean/SEM/crossing |
| exact raw/reduced oracle | PASS | 512 disorder contexts、16,384 单构型比较，全部最大误差 0 |
| conda `12` 全套 pytest | PASS | 343 passed；2 个 warning 均为预期 deprecated alias |
| 014 fresh integration | PASS | PT `computed=1,reused=0`；失败样本不进入 aggregate |

014 记录的实现 fingerprint 为
`bc3867f359bdc14e2dee10535e21064c75df21d0ebef5b5102d973bd5d688ae2`。生成证据时
worktree 为 dirty，故同时保存当时 Git SHA/dirty 状态；源码内容身份以 fingerprint 为准。

## 交付边界

- `DONE` 表示 reduced-posterior 数值管线、估计器、路由、gate 与 schema 已按 v2 契约验证，
  不表示已经产生 expander threshold 或完整 preparation-channel 物理结论。
- PT 仍是纯 Python，大码生产成本高；exp102 起量前应按目标 family 做资源预算，不得据 smoke
  推断吞吐。
- legacy 3D exp40/41 结果属于 `legacy_delta_only`，不能当作 true-posterior threshold。
- 新实验必须保留 contract/source fingerprint、完整 resolved config、validity mask 与 PT/TI 诊断。

## PRE_ALIGNMENT 历史事实

- 旧套件曾报告 259 tests 全绿、V1 direct/PT/TI 内部对拍、Nishimori 与冻结 torture。
- `validation/007` 暴露 pairwise-TI 在 K43 上 character 偏差最高约 1.55，仍支持永久禁用。
- 旧 profile 的 direct m=6 约 1.1 s/disorder、PT 约 302 s/disorder只供预算量级参考。
- 旧 scan v1 多节点曾 bit-identical；v2 不复用其 chunk 或 schema。

逐目录历史限制与 014 复现命令见 `validation/README.md`。
