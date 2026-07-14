# exp101 历史报告与 v2 勘误

## AGGREGATION ERRATUM（2026-07-14，优先于 scan v2 聚合结论）

`exp101.physics.v2` 的模型与 posterior 语义不变，仍由
`validation/014_paper_alignment_20260713/` 认证。但 014 同时锁定的 `exp101.scan.v2` 允许在一个
参数点含 invalid disorder 时，对剩余 valid disorders 发布 mean/SEM，并把 crossing input 逐项
mask。该行为会把 gate 通过事件条件化进统计量，存在选择偏差，不能作为 publication aggregation。

当前扫描契约升级为 `exp101.scan.v3`：只有所有 planned disorders 均 present 且 valid 的参数点才
是 `REPORTABLE`；任一 invalid/missing 均关闭整点正式 mean、SEM 与 crossing/FSS 输入。valid-only
条件统计仅作诊断。v1/v2 产物不迁移，正式分析统一经 `load_publication_q_top`。scan v3 已由
`validation/015_aggregation_safety_20260714/` 与完整测试认证，当前状态为 `DONE`。

## PHYSICS ERRATUM（2026-07-13，优先于更早历史结论）

2026-07-09 版报告声称 exp101 已由“259 tests + V1–V6”全面认证并可直接供 exp102 生产。
该结论现已撤回。旧验证只能证明旧实现内部一致，不能认证随后明确的论文语义、无偏估计器、
PT 接入和 scan v2 schema。所有旧证据现标为 `PRE_ALIGNMENT`。

当前唯一物理权威是 `PHYSICS_CONTRACT.md`（`exp101.physics.v2`），当前状态见 `status.md`。
`validation/014_paper_alignment_20260713/` 已通过并继续认证 physics v2；它不再认证 scan v3 的
publication aggregation。以下旧数值仍只作真实历史记录，不代表论文 threshold 结论。

## physics v2 与历史 scan v2 认证结论（2026-07-13）

- conda `12` 全套回归为 **343 passed**；2 个 warning 均来自有意测试 deprecated ensemble alias。
- raw-paper/reduced-canonical oracle 枚举 512 个 disorder contexts、16,384 个候选构型比较；
  pointwise weight、partition function、sector weights、q_top 与 MAP 的最大误差全部为 0。
- exact 证据使用独立 preparation-chain representative；fixed-y 比较实际改变
  `H_check epsilon_data_true`，shifted-coordinate identity 逐构型成立。
- 四实例 PT integration 强制 fresh run（`computed=1,reused=0`），零 measurement round trip 的任务
  正确标为 `INVALID`。014 当时验证的是“invalid 自身不进入 valid-only mean/SEM/crossing”，而非
  scan v3 要求的“一个 invalid 关闭整个参数点”；前者现仅作历史证据。
- task/chunk/NPZ/manifest 记录 source implementation fingerprint 与 Git dirty provenance；v1 chunk、
  浮点 q 路径碰撞和跨 family/sector/config cache 误复用均有回归保护。
- sampled estimator 使用四独立链（q=0 八起点）的 U-statistic，分别保存 pooled raw、debiased、
  jackknife 与 finite-population error；80-character 实例不截断。无偏 U-statistic 的单次实现可为负
  或越出物理区间，必须保存 raw 并使该 disorder INVALID，不得裁剪。
- full TI 限 `k<=10`，独立 sector 链独立 bootstrap；`p=0/0.5` 走解析端点。legacy 数据只进入
  `formal_*`，状态为 `FORMAL_ONLY`，不产生论文 aggregate。exact/解析端点的 MAP-purity bounds 是
  algebraic；普通 TI/sampled 的 bounds 是无置信覆盖的 plug-in estimates，字段不得混用。

因此 exp101 可作为 exp102 的 reduced-posterior 数值基础复用；复用前提是保持 physics v2、
scan v3、生产 `x_error/true_posterior` convention 和所有 validity gates。publication 聚合还必须遵守 scan v3
点级 fail-closed 与统一 loader；015 已全绿并恢复 DONE。本结论不扩展到完整 preparation
channel，也不声称已经得到新的 expander threshold。

## 1. 旧实现确实完成过的工程工作

2026-07-07 至 07-09，exp101 建立了自包含的 GF(2)、图/HGP、logical basis、distance、family、
section、reference/fast MCMC、PT、gates、TI、exact enumeration 与 chunked scan 模块。旧测试对结构
代数、参考/numba 一致性、随机数可复现和若干小码内部对拍提供了有用调试基础。

这些结构模块可以在 v2 复用，但“旧测试通过”不自动证明它们与新字段、fingerprint、engine
routing 或 output schema 的组合正确。

## 2. PRE_ALIGNMENT 数值事实

| 历史实验 | 当时记录的结果 | v2 解释 |
|---|---|---|
| 单元测试总表 | 259 tests 全绿 | 仅旧 API/旧 estimator 的内部回归；不是当前认证 |
| V1 direct | pooled bias 约 `-0.008 +/- 0.041` | 旧 instrument 下的采样对拍；未覆盖四独立链无偏二阶矩 |
| V1 PT 冷点 | bias 约 `+0.071 +/- 0.099`，旧任务有往返 | 未覆盖四个独立 PT 实例、冷端 R-hat/ESS 与 v2 INVALID 传播 |
| V2 解析极限 | q=0.5 闭式覆盖 m=2/4/6，旧 z <= 2.4 | 有用回归，但标题“V2”与 `exp101.physics.v2` 无关 |
| V3 Nishimori | 小码全 disorder 差约 `1.9e-14`；n=100 MCMC z 约 2.06 | 使用旧变量/observable estimator，须以 v2 canonical 与 debiased test 重跑 |
| pairwise characterization | K43(k=13) 对 exact character 最大偏差约 1.55/满量程 2 | 仍是禁用 large-k pairwise q_top 的强历史证据；不能证明新 gap-only API |
| section fingerprint | n >= 400 的旧序列化崩溃被发现并修复 | 工程 bug 修复事实仍成立，v2 fingerprint 覆盖仍需新测 |
| profile | direct m=6 约 1.1 s/disorder；PT 约 302 s/disorder | 只供旧实现性能量级参考；四 PT 实例成本更高 |
| 多节点 | scan v1 曾跨节点 bit-identical | v2 protocol/cache/task identity 改变后必须重证 |
| 2D smoke | 微型 crossing 约0.133/0.069，包围参考0.109 | 仅 smoke；不认证修正后的论文 preparation semantics |

原始文件保留在 `validation/001`–`013`，逐项限制见 `validation/README.md`。

## 3. 必须纠正的旧叙述

### 3.1 状态与矩阵 mapping

生产 `x_error/H_Z` 对应从 `|+>^n` 测 Z checks 制备 `|+>_L`，stabilizer/logical moves 是
X 型，observable 是 dual logical Z。旧报告 §5 写成 `|0>_L` 是错误；现有 H_X/H_Z 矩阵接线
不应交换。Hadamard 对偶 `z_error/H_X` 才对应 `|0>_L`。

### 3.2 canonical posterior

生产 posterior 是

```text
pi(e|effective_syndrome)
  proportional to exp[-Kp|e|-Kq|H_check e xor effective_syndrome|],
effective_syndrome=H_check epsilon_data_true xor measurement_error.
```

真实错误不直接进入能量。旧报告把含真实错误的 shifted-coordinate 表述称为 canonical
“双盘度”并与 repo model 对照，容易造成错误实现。严格 shifted formula 只在换元
`x=e xor epsilon_data_true` 后出现。

正式 legacy 名是 `legacy_delta_only`；`repo_compat` 只是 deprecated alias。3D exp40/41 是
legacy delta-only 历史结果，不能直接作为论文 reduced MLD threshold。

### 3.3 section 与 observable

直接测量 `m_u_absolute`，relative 值只乘 planted Mattis sign。两种 frame 在固定 section 下
拥有相同 q_top/purity/max mass。只对逐 syndrome stabilizer-boundary section shift 有严格不变性；
旧报告“frame=gauge”不能扩展到一般 logical-valued section change。

### 3.4 posterior statistics

旧 `w0` 实际是 planted-class mass，不等于 MAP success。physics v2 分开定义：

```text
posterior_mass_on_planted_class
posterior_purity
map_success_probability=max(weights_absolute)
q_top=(2^k*posterior_purity-1)/(2^k-1).
```

因此旧 3D 报告中把 `w0` 无条件称为“success”或把 q_top 直接称 purity 的句子不能迁移到
exp101 v2。

scan v3 进一步区分 exact/解析端点的
`map_success_algebraic_lower_bound/upper_bound` 与普通 TI/sampled 的
`map_success_estimated_lower_bound/upper_bound`。后者只是 plug-in estimates，固定标注
`Estimated MAP-purity bounds (plug-in; no confidence coverage)`；两组字段不得同时填充。

### 3.5 large-k 方法与 sampled estimator

旧报告正确识别了 pairwise-TI 的可加性失败，但随后称 direct/PT sampled q_top “已验证可生产”
仍然过早：旧路径没有完整锁定 basis/nonbasis 总体权重、四独立链 U-statistic、jackknife/FPC、
四 PT 实例 cold diagnostics 与 disorder-level INVALID 传播。因此当时的生产结论必须等待 014；
这些 estimator/gate 缺口已由 014 补齐。但 014 的 valid-only 点级聚合仍有 publication 选择偏差，
必须由 scan v3 与 015 的 fail-closed 证据补齐。

## 4. physics v2 / scan v3 交付边界

physics v2 只实现论文 preparation 变量约化后的 logical-class MLD posterior，不是完整 Clifford/
preparation channel simulator。raw preparation enumeration 只作为逐构型等价 golden oracle。

预定生产路由为：`k<=10` full TI；`k>10,q>0` PT observable sampling；`k>10,q=0`
validated 8-start。large-k TI 必须硬拒绝；pairwise API 只输出 basis-sector free-energy gaps。

scan v3 不实现自动重采样、checkpoint/续链、crossing 拟合或 FSS。它只保证正式输出在参数点级
fail-closed，并提供拒绝 v2/legacy/失败点和 schema 不一致的 publication loader。

## 5. 当前结论

exp101 v1 提供了有价值的工程骨架和历史反例，尤其提前发现 large-k pairwise-TI 失败，但旧
259-test 数量本身从未证明论文语义。physics v2 的通过只来自契约与 validation/014；scan v2 的
valid-only aggregate 已降为历史。validation/015 与完整测试已经通过，scan v3 的点级 fail-closed、
bounds 语义、publication loader 和 schema 状态为 `DONE`。完整 preparation channel 与
实际 expander threshold 仍在交付边界之外。
