# exp101 论文语义对齐报告

日期：2026-07-13。物理权威：`PHYSICS_CONTRACT.md`（`exp101.physics.v2`）。

本文把用户提供的 `文章.tex`、论文的 reduced canonical formula、代码接口和决定性测试逐项对应。
它是可审计映射，不是第二份物理契约。当前状态为 **DONE**；下表测试均已在
`validation/014_paper_alignment_20260713/` 留下完整证据。

## 1. 对齐结论

1. 论文从 `|+>^n` 测全部 Z checks，生产 sector 必须是
   `x_error/H_Z/rows(H_X)/logical_X/logical_Z/|+>_L`。旧文档把它写成 `|0>_L` 是状态映射错误，
   不是矩阵接线错误。
2. exp101 的数值对象是消去 `sigma_prep` 后的 reduced MLD posterior，不是完整 Clifford 或
   preparation-channel simulator。
3. canonical true posterior 的能量只含候选 `e` 与 `effective_syndrome`；真实
   `epsilon_data_true` 不进入能量。旧实现把“真实错误进入 shifted-coordinate 公式”与
   canonical energy 混写，必须拆开。
4. absolute character 是直接采样量；relative character 只乘 planted Mattis sign。它们在固定
   section 下有同一 `q_top` 和最大 sector mass，但一般 logical-valued section change 不是 gauge。
5. planted-class posterior mass、posterior purity 与 MAP success 是三个不同统计量。旧 `w0` 名称
   掩盖了这一点，v2 删除该公共字段。
6. 旧 large-k pairwise-TI 只能保留为 basis free-energy-gap diagnostics；不能构造 `m_u` 或
   `q_top`。生产 large-k 必须走 PT/q=0 多起点 observable sampling。

## 2. 公式、代码与测试映射

论文锚点按用户提供的 `文章.tex` 原文命名：

| `文章.tex` 锚点 | 原文对象 | exp101 v2 对应 |
|---|---|---|
| `state preparation process` | 从 product `+` state 测 Z stabilizers，得到 preparation-induced X channel | 锁定 `x_error/H_Z/|+>_L` convention；channel 本身不逐步模拟 |
| `Statistical Mechanical Mapping` | `Z(s,sigma_eta)=sum_c q(H_Z c+s)p(c+eta)` 与 logical-class MLD | 枚举原始 `a` 后以 `e=a xor F_total` 双射约化到 canonical posterior |
| `Order Parameter` | `m_u` 是 sector posterior 的 Fourier transform；`q_top=(2^k-1)^-1 sum_{u!=0}m_u^2` | absolute/relative characters 与 normalized purity |
| theorem `Logical Wilson loops and MLD success probability` | Fourier inversion、MAP 与 purity bounds | full weights 输出 `map_success_probability`，并检查 `purity <= MAP <= sqrt(purity)` |
| `correction`, `eq:correct-Ou-nonlinear` / `eq:Ou-full-nonlinear` | 任意 section 下以 `c+r(Hc)` 读 logical class，Mattis sign 单独分离 | `m_absolute` 直接测量，`m_relative=sign*m_absolute`；不恢复旧 factorized shortcut |
| proposition `boundary-shift-invariance` | section 逐 syndrome 加 boundary 时 sector posterior 不变 | 只锁定 boundary-only invariance，并增加一般 logical shift 非不变反例 |

下表把这些公式落实到代码与 014 决定性测试：

| 论文/物理对象 | v2 canonical 表述 | 代码落点 | 决定性 v2 测试 | 修改前行为 |
|---|---|---|---|---|
| 从 `|+>^n` 测 Z checks 制备 logical `+` | `sector=x_error`, `H_check=H_Z`, X stabilizer/logical moves, Z characters | `src/model.py: assemble_sector_model`；manifest `state_prep_protocol` | sector convention test 同时锁定 Hadamard 对偶 | 文档误写成 `|0>_L`；矩阵接线本身正确 |
| preparation variables | `sigma_prep`, `measurement_error`, `s_prep`, `F_total`, `sigma_final` | small-code oracle，不进入生产 state | 枚举所有 `sigma_prep/measurement_error/epsilon_data_true/a` | `eta/delta/s` 混名，原始与 reduced 变量边界不清 |
| reduced observation | `effective_syndrome=H epsilon_data_true xor measurement_error` | `DisorderRealization.effective_syndrome` | 消去恒等式与 shape/domain tests | `observed_syndrome` 容易被误解为 preparation record |
| raw paper weight | `q(Ha xor s_prep)p(a xor F_total)` | `src/enumerate_exact.py` golden oracle | 与下行逐构型、Z、sector weights 完全相等 | 未以独立逐 preparation-variable oracle 锁定 |
| reduced posterior | `p(e)q(He xor effective_syndrome)` | `src/model.py` wiring；reference/fast MCMC；TI/PT | 固定 effective syndrome 改真错，energy bit-identical | 旧说明把 canonical 与含 `epsilon_true` 的 shifted energy混写 |
| paper shifted coordinate | `x=e xor epsilon_true`; `Kp|x xor epsilon_true|+Kq|Hx xor measurement_error|` | 只作 identity helper/test | 逐构型 shifted identity | 曾与 legacy delta-only 接线混称 |
| ensemble identity | canonical: `true_posterior`, `legacy_delta_only`; aliases 先归一化；legacy 只存 `formal_*` | `src/model.py`, `src/run_scan.py` | alias/canonical seed、task fingerprint、formal 结果相同；legacy `FORMAL_ONLY` 且不聚合 | 正式名仍含 `repo_compat`，alias 可污染 seed/schema；legacy 可冒充论文量 |
| CSS move 完备性 | stabilizer rows 与 logical moves 完整张成 `ker(H_check)` | `src/model.py` | 缺 kernel generator 的 malformed model 在装配时拒绝 | q=0 exact 可遍历完整 coset，而 MCMC 只遍历不完整 move span |
| 三类 section | meta decoder / preparation representative / logical-sector section | `src/section.py`; model 字段 `logical_sector_section` | domain guard；q>0 effective syndrome 禁传 sector section | `DecoderSection` 与 `model.section` 名称过宽，易误用 |
| absolute label | `phi_r(e)=[e xor r(He)]` | `src/observables.py` | exact weights 的 Fourier transform逐 character 对拍 | 主要只存 relative character，absolute/relative 未显式并存 |
| planted frame | `m_relative=sign(planted_class)m_absolute` | `src/observables.py`, exact/TI/scan outputs | 每个 u 的符号关系与 weights 平移 | `reference_label/ell_ref` 与 Gibbs syndrome 参数混在一个 wiring 叙述中 |
| section invariance | 只保证逐 syndrome boundary shift | `src/section.py`, `tests/test_section_frames.py` | boundary shift 不变；logical-valued shift 构造反例 | “frame=gauge”容易被读成任意 section 均不变 |
| posterior purity | `sum_l P(l)^2` | exact/TI/stat helpers | Parseval 与 characters 精确对拍 | 曾把 `q_top` 直接称 purity |
| normalized `q_top` | `(2^k purity-1)/(2^k-1)` | `src/observables.py`, `src/sector_ti.py` | weights/characters 两路一致 | sampled 路径曾只平均 nonbasis，且直接平方有正偏 |
| planted mass | `P(planted_logical_class)` | full-weight true-posterior outputs | posterior `(0.1,0.9)` planted=0 -> `0.1` | 公共名 `w0` 被误当 decoder success |
| MAP success | `max_l P(l)` | exact/TI full weights | 上例 MAP=`0.9`；purity bounds | 未作为独立统计量输出 |
| sampled character 总体均值 | `[sum_basis+(N-k)mean_nonbasis]/N` | `src/observables.py` | 人工 character 表精确命中 | nonbasis mean 冒充全部非零 character mean |
| MCMC square 去偏 | 独立链 pair cross-product U-statistic | direct/q0/PT aggregation | Bernoulli 重复实验：raw 正偏、U-stat 无偏、jackknife 合理 | pooled chain mean 的平方保留有限样本正偏 |
| full-sector TI | 仅 `k<=10` | `src/sector_ti.py`, preflight | `k>10 + ti` 建任务前报错 | 入口可在 large-k 落入 pairwise 路径 |
| TI 不确定度 | 独立 sector chain 必须独立 block bootstrap | `src/sector_ti.py` | 人工同序 block 表证明共享 indices 会把 gap stderr 错压为 0 | 把独立链误当 paired samples，制造虚假 covariance |
| TI data-noise 端点 | `p=0.5` uniform、`p=0` class-0 delta 的解析 full-sector 结果 | `src/sector_ti.py`, scan schema | auto 路由端到端 weights/q_top/zero stderr/infinite mask | finite-positive `K_p` guard 与公开概率边界不闭合 |
| pairwise diagnostics | 仅 basis-sector free-energy gaps | 独立 diagnostics API | 输出 key 扫描不得含 `m_u/q_top` | 曾假定 sector free energy 可加并合成 q_top |
| auto routing | small-k full TI；large-k q>0 PT；large-k q=0 8-start | `src/run_scan.py` | 三路参数化测试 | 默认 TI，PT 未进入统一 scan |
| PT validity | 4 个独立实例；冷端 R-hat/ESS；每实例 round trip；min swap 与 worst basis acceptance | `src/pt.py`, `src/gates.py`, scan worker | 人工 round-trip failure -> `INVALID` | 单实例 transport summary 不足以认证二阶矩 |
| scan identity | `exp101.scan.v2`, `scan_results.npz`, full task fingerprint | `src/run_scan.py` | v1 不复用；family/sector/config 隔离；80 characters 不截断 | v1 `sector_ti_results.npz`、cache/task identity 覆盖不足 |
| aggregate | 只用 `valid_for_aggregation=true` | scan merge | INVALID 不改变 mean，valid/invalid/missing 计数正确 | 失败样本与 missing 的聚合语义不完整 |

## 3. 统计量关系的最小反例

对二扇区 posterior `P=(0.1,0.9)`，若 planted class 是 0：

```text
posterior_mass_on_planted_class = 0.1
posterior_purity                = 0.1^2 + 0.9^2 = 0.82
map_success_probability        = 0.9
q_top                           = 2*0.82 - 1 = 0.64.
```

这里 `0.82 <= 0.9 <= sqrt(0.82)`。这个例子同时排除 `w0=success`、`purity=q_top` 两种旧混称。

## 4. preparation-channel 边界

论文的完整物理过程还包含：非物理 syndrome 的 meta-check projection、受读数控制的 Pauli
recovery、preparation-induced channel、额外 data-noise channel 与最终 error correction。exp101 v2
只实现这些步骤约化后的 posterior；以下内容明确不在本次实现范围：

- 完整 Clifford gate/noisy measurement 电路模拟；
- `measurement_error_decoder` 的算法性能或 threshold；
- 逐样本执行 `preparation_chain_representative` recovery channel；
- 对 preparation channel 的 Kraus 表示或 diamond-norm 认证；
- 从 reduced-posterior threshold 自动反推完整硬件电路 threshold。

golden test 枚举原始变量只是证明 reduced formula 与论文求和逐项相等，不把上述边界悄然扩展为
已实现功能。

## 5. 证据状态

- `validation/001`–`013`：`PRE_ALIGNMENT`；可保留真实历史数值，但不认证 v2。
- `validation/014_paper_alignment_20260713/`：v2 权威认证目录；完整 pytest、exact JSON/Markdown、
  PT/aggregation integration 与 reproducibility metadata 均为 PASS。
- `report.md`：历史报告，顶部 erratum 控制解释；不得再引用旧“259 tests + V1–V6”作为当前通过。

最终验收依据不是测试数量，而是本表每条物理等价、统计估计、路由、gate 与 schema 契约都有
独立决定性测试且 014 留档完整；本次两项均已满足。
