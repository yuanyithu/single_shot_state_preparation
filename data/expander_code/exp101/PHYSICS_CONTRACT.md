# exp101 physics contract

- `physics_contract_version`: `exp101.physics.v2`
- `scan_contract_version`: `exp101.scan.v2`
- 生效日期：2026-07-13
- 状态：物理定义已冻结；代码与 `validation/014_paper_alignment_20260713/` 的一致性验证 PASS

本文是 `data/expander_code/exp101/` 唯一的物理权威。`notes/`、`plan.md`、源码注释、
测试和历史报告只能解释或验证本文，不得另立不同公式。若它们与本文冲突，以本文为准。
根目录文档中的 3D toric legacy 模型只描述旧项目，不是 exp101 论文生产模型。

## 1. 研究对象与 sector convention

论文协议为

```text
|+>^n
  -> 一次含 readout error 的全体 Z-check 测量
  -> preparation-induced X-error channel
  -> 独立 stochastic X data noise
  -> 一次完美 final Z-syndrome measurement
  -> logical-class MAP/MLD
```

exp101 的生产配置固定为：

| 配置项 | 值 |
|---|---|
| `sector` | `x_error` |
| `H_check` | `H_Z` |
| stabilizer moves | `rows(H_X)`，即 X stabilizers |
| logical moves | `logical_X` |
| logical observables | dual `logical_Z` characters |
| prepared state | `|+>_L` |
| `state_prep_protocol` | `plus_Zcheck_X` |

现有 `assemble_sector_model` 的矩阵接线正是上述 convention，不得为了修正文档而交换
`H_X/H_Z`。Hadamard 对偶配置是 `z_error/H_X/Z errors/logical_Z moves/logical_X
observables/|0>_L`，但不是 exp101 的生产 sector。

## 2. 实现边界：reduced MLD，不是完整制备电路

exp101 模拟消去 preparation syndrome 后的 **reduced logical-class MLD posterior**。它不逐门
模拟 Clifford 电路，不显式构造完整 preparation channel，也不把 meta-check decoding 或
preparation recovery 当作 MCMC 的一部分。小码 golden test 可以枚举原始 preparation 变量来证明
约化等价，但生产状态空间只有约化后的候选错误。

因此，exp101 能回答的是：给定一次制备读出和最终完美 syndrome 所导出的有效信息，逻辑类
posterior、MAP success 与 `q_top` 如何变化。它本身不认证论文中完整制备 channel 的电路级实现。

## 3. 原始变量与 reduced observation

所有加法均在 GF(2) 上，以下用 `xor` 表示。

| 物理量 | 代码/文档正式名称 | 定义 |
|---|---|---|
| 制备投影的真实 physical syndrome | `sigma_prep` | `sigma_prep in im(H_check)` |
| 制备读出错误 | `measurement_error` | iid Bernoulli(q) check bits |
| 制备轮读数 | `s_prep` | `sigma_prep xor measurement_error` |
| preparation representative | `c_prep(sigma_prep)` | `H_check c_prep(sigma_prep)=sigma_prep` |
| 制备后真实 data error | `epsilon_data_true` | iid Bernoulli(p) qubit bits |
| 最终总错误 | `F_total` | `c_prep(sigma_prep) xor epsilon_data_true` |
| 最终完美 syndrome | `sigma_final` | `H_check F_total` |
| reduced observation | `effective_syndrome` | `s_prep xor sigma_final` |

关键消去式为

```text
effective_syndrome
  = s_prep xor sigma_final
  = H_check epsilon_data_true xor measurement_error.
```

manifest 使用 `syndrome_semantics=effective_y`。历史名 `eta`、`delta`、
`observed_syndrome` 只允许作为发出弃用警告的只读兼容属性；新代码、manifest、seed payload
和文档使用 `epsilon_data_true`、`measurement_error`、`effective_syndrome`。

## 4. 唯一的论文生产 posterior

令

```text
K_p = log((1-p)/p),    K_q = log((1-q)/q).
```

规范 MCMC 状态 `e in F_2^n` 是候选 absolute data error。论文生产 posterior 唯一定义为

```text
pi(e | effective_syndrome)
  proportional to
exp[-K_p |e| - K_q |H_check e xor effective_syndrome|].
```

也就是说，`gibbs_syndrome_argument=effective_syndrome`。真实错误
`epsilon_data_true` **不得**被能量函数、Metropolis 比值、proposal probability、TI sector
free energy 或 PT swap ratio读取。它只允许用于：

1. 生成 `effective_syndrome`；
2. 计算 `planted_logical_class` 与 absolute/relative Mattis frame 的关系；
3. posterior 生成后的 scoring、Nishimori identity 和 ground-truth tests。

固定 `effective_syndrome` 后改变 `epsilon_data_true` 而保持输入 posterior 不变，必须得到逐构型
完全相同的能量。

当 `q=0` 时，第二项成为硬约束

```text
H_check e = effective_syndrome = H_check epsilon_data_true,
```

所以生产模型是 quenched coset，而不是 clean kernel。

### 4.1 与论文原始 preparation 求和的逐构型等价

对论文原始求和变量 `a` 定义

```text
e = a xor F_total.
```

利用 `effective_syndrome=s_prep xor H_check F_total`，逐构型有

```text
q(H_check a xor s_prep) p(a xor F_total)
  = p(e) q(H_check e xor effective_syndrome).
```

因此原始与 reduced 表述的单构型权重、partition function、逐 logical-sector weights、
`q_top` 和 MAP success 必须完全相等。这一恒等式是 v2 golden oracle，不是数值近似。

### 4.2 shifted coordinate 只作等价证明

若定义 `x=e xor epsilon_data_true`，则同一生产 posterior 可写成

```text
E_shifted(x)
  = K_p |x xor epsilon_data_true|
  + K_q |H_check x xor measurement_error|.
```

这才是严格的 paper shifted coordinate。生产 MCMC 仍以 `e` 为规范变量；不得把 shifted
formula 与下面的 legacy 模型混称。

## 5. ensemble 名称与 legacy 边界

正式名称只有：

- `true_posterior`：第 4 节的论文 reduced posterior，生产唯一允许值；
- `legacy_delta_only`：旧 3D 程序的回归模式，不得生成论文相图或阈值结论。

弃用 alias：

- `paper_true_posterior -> true_posterior`；
- `repo_compat -> legacy_delta_only`。

alias 必须在 task seed、chunk identity、cache key、task fingerprint 和 manifest 构造之前归一化；
持久化结果只能出现正式名称。同一配置使用 alias 或正式名必须得到相同 seed 和结果。

旧模型可写成

```text
E_legacy(c)
  = K_p |c xor epsilon_data_true|
  + K_q |H_check c xor effective_syndrome|.
```

换元 `u=c xor epsilon_data_true` 后为

```text
E_legacy(u) = K_p |u| + K_q |H_check u xor measurement_error|,
```

故它只有 measurement-error disorder。`q=0` 时它是 clean kernel，与
`true_posterior` 的 quenched coset 一般不同。legacy 输出中的 sector 数据只是形式上的回归量；
只能保存在显式 `formal_*` 字段（以及 `largest_sector_mass`）中。论文语义的 weights、characters、
`m_u`、`q_top`、posterior、MAP 与 bounds 字段必须为空，且 legacy task 不得进入论文 disorder
average 或 crossing；数值 gate 通过时状态记为 `FORMAL_ONLY`，不得混称论文 `VALID`。

## 6. 三种 section/decoder 不得混用

三个映射的 domain、codomain 与用途不同：

| 名称 | 类型 | 用途 | exp101 状态 |
|---|---|---|---|
| `measurement_error_decoder` | `C_checks -> im(H_check)` 或其 residual map | 将非物理制备读数投影到 physical syndrome/meta-check decoding | reduced production 中不实现 |
| `preparation_chain_representative` | `im(H_check) -> F_2^n` | 为 `sigma_prep` 选 `c_prep` | 仅原始/reduced golden test 需要 |
| `logical_sector_section` | `im(H_check) -> F_2^n` | 给可实现 syndrome 选 qubit chain，并标记 logical sector | 生产 observable/TI 使用 |

`logical_sector_section r` 满足 `H_check r(sigma)=sigma`。历史 `model.section` 和
`DecoderSection` 应分别迁移为清楚的 `logical_sector_section` 与
`QubitChainLogicalSectorSection` 一类完整名称。

一般 `effective_syndrome` 在 `q>0` 时不必属于 `im(H_check)`，因此绝不能把它传给
`logical_sector_section.apply`。q=0 的硬约束是例外：此时它必在像空间中。

## 7. logical sectors 与 absolute/relative observable

固定一个 `logical_sector_section r`，定义 absolute logical label

```text
phi_r(e) = logical class of [e xor r(H_check e)] in F_2^k.
```

对非零 character `u in F_2^k`，直接测量

```text
m_u_absolute = E_pi[(-1)^(u dot phi_r(e))].
```

真实错误只定义 planted reference：

```text
planted_logical_class = phi_r(epsilon_data_true)
planted_mattis_sign_u = (-1)^(u dot planted_logical_class)
m_u_relative = planted_mattis_sign_u * m_u_absolute.
```

相对 sector weights 只是 absolute weights 的平移：

```text
weights_relative[t] = weights_absolute[t xor planted_logical_class].
```

因此，在同一个 section 下必须满足：

- `m_u_relative = planted_mattis_sign_u * m_u_absolute`；
- absolute/relative 的 `q_top` 完全相同；
- absolute/relative 的 `posterior_purity` 与最大 sector mass 完全相同；
- `posterior_mass_on_planted_class = weights_absolute[planted_logical_class]
  = weights_relative[0]`。

### 7.1 section change 的精确边界

若 `r'(sigma)=r(sigma) xor beta(sigma)` 且每个 `beta(sigma)` 都是 X-stabilizer
boundary，则逐 sector posterior 不变。这是必须测试的 boundary-only invariance。

若 section 差含 logical component，该 component 可随 `H_check e` 变化，不保证只是全局平移；
posterior、characters 或 `q_top` 都可能变化。不得再把任意 section change 称为 gauge，也不得把
不同 section fingerprint 的 run 无条件合并。

## 8. `q_top`、purity、planted mass 与 MAP success

令 `P(l)=weights_absolute[l]`，`M=2^k`。定义

```text
posterior_purity = sum_l P(l)^2
q_top = (M * posterior_purity - 1) / (M - 1)
      = mean_{u != 0} m_u_absolute^2
      = mean_{u != 0} m_u_relative^2
posterior_mass_on_planted_class = P(planted_logical_class)
map_success_probability = max_l P(l).
```

`q_top` 是归一化 purity，不等于 `posterior_purity`。planted mass 也不等于 MAP success；例如
posterior `(0.1, 0.9)` 且 planted class 为 0 时，两者分别为 `0.1` 与 `0.9`。

每个 true-posterior full-weight 样本必须验证

```text
posterior_purity
  <= map_success_probability
  <= sqrt(posterior_purity).
```

等价的 `q_top` bounds 为

```text
[1 + (M-1) q_top] / M
  <= map_success_probability
  <= sqrt([1 + (M-1) q_top] / M).
```

公共字段 `w0` 被移除。legacy 模式不输出论文语义的
`posterior_mass_on_planted_class` 或 `map_success_probability`。

## 9. sampled-character estimator

令全部非零 character 数 `N=2^k-1`。sampled 路径总是保存 `k` 个 basis characters，并从
其余 `N-k` 个 nonbasis characters 中无放回均匀抽样。对任意逐 character 量 `f_u`，总体均值的
估计必须是

```text
[sum_basis f_u + (N-k) * mean_sampled_nonbasis f_u] / N.
```

这同样用于一阶 character 与二阶 moment；不得用 nonbasis mean 冒充全部非零 character 均值。
随机 character 的 seed、完整 `u_bitmasks`、实际 count 和 mask 必须保存。数组 character 维按
实际 `k+num_random_u` 分配，例如 `k=16,num_random_u=64` 必须保存 80 行。

### 9.1 独立链去偏

sampled 路径默认运行 `C=4` 条独立链；q=0 路径默认 8 个独立 start chains。对每个 character
保存各链均值 `m[a,u]`，以及：

```text
pooled_square_raw_u = mean_a(m[a,u])^2
m2_debiased_u
  = 2 / [C(C-1)] * sum_{a<b} m[a,u] m[b,u].
```

`m2_debiased_u` 是独立链 U-statistic。必须另存 delete-one-chain jackknife 标准误。随机
nonbasis 抽样误差使用有限总体修正，必须与链/MCMC 误差分开报告。

生产配置少于 4 条独立链不得进入 disorder average。若 debiased purity 或 `q_top` 超出物理
范围，不得静默裁剪：保存 raw 值、标为无效、设置 `valid_for_aggregation=false`，并禁止生成
成功率 bounds。

## 10. engine routing 与方法限制

CLI 默认 `engine=auto`，解析规则固定：

| 条件 | `resolved_engine` |
|---|---|
| `k <= 10` | full-sector thermodynamic integration |
| `k > 10 and q > 0` | parallel-tempering observable sampling |
| `k > 10 and q = 0` | validated 8-start q=0 observable sampling |

显式 `direct`、`pt`、`ti` 仍可请求，并在 manifest 同时记录 requested/resolved 名称；但
full-sector TI 在入口硬限制 `k<=10`，`k>10 + engine=ti` 必须在创建任务前报错。

旧 pairwise-TI 不再是 q_top engine。它只可作为独立的
`basis_sector_free_energy_gap_diagnostics` 返回 basis-sector gap、误差和积分诊断；任何字段名
不得包含 `m_u`、`q_top`、purity 或 success。

full-sector TI 的两个 data-noise 端点走同一 resolved engine 下的解析分支：`p=0.5` 时 data
耦合为零，kernel logical bijection 保证 sector weights 严格均匀；`p=0` 时只剩 absolute class 0，
sector weights 为 delta 分布。后者若同时 `q=0` 且 Gibbs syndrome 非零，则 posterior 无支持，
必须显式报错。解析端点保存 `endpoint_mode`、零不确定度与 infinite-gap mask，不伪造 MCMC/TI
采样轨迹。

## 11. convergence gates 与 validity

所有 gate 使用既有阈值，不因对齐工作放宽：

- full TI：coarse/fine grid TV 与 `q_top` 差任一超阈值即 INVALID；
- direct/q=0：R-hat、ESS、起点 spread、符号敏感 character spread 或 sector transport 任一失败
  即 INVALID；
- PT：运行 4 个独立 PT 实例，把冷端 character trace 送入 R-hat/ESS；每个实例都要达到要求的
  round trip；相邻温度的最小 swap rate 必须非零；pooled worst-basis-logical cold acceptance
  必须达到既有阈值。任一失败即 INVALID。

`INVALID` chunk 保留完整诊断和失败原因，但不得进入 disorder mean、SEM、crossing 或成功率
bounds。`mean_q_top_estimate` 及 SEM 只使用 `valid_for_aggregation=true` 的样本，并同时报告
valid/formal-only/invalid/missing 与 numerically-valid 数量。比例字段分为
`paper_aggregation_fraction` 与 `numerical_pass_fraction`；历史 `pass_fraction` 只作为前者的
deprecated alias，不得解释为数值 gate 通过率。

## 12. v2 scan identity 与最小输出契约

v2 输出文件名为 `scan_results.npz`。任何 `exp101.scan.v1` chunk 均不可复用。

task fingerprint 至少覆盖：physics/scan contract version、canonical ensemble、sector、family
rule/seed、code fingerprint、requested/resolved engine、全部 sampler/estimator 配置。worker model
cache key 还必须包含 sector 与 family rule/seed，避免跨配置误复用。
chunk 复用必须同时核对 outer/inner task fingerprint 与 implementation fingerprint。NPZ 中
`git_worktree_dirty` 使用真正的 bool；无法判定时由独立 `git_worktree_dirty_known=false` 表示，
不得把字符串 `"False"` 当布尔状态保存。

NPZ/manifest 至少保存：

- `state_prep_protocol=plus_Zcheck_X`、`syndrome_semantics=effective_y`；
- physics/scan contract version、canonical ensemble、requested/resolved engine；
- family rule/seed、code/section/observable fingerprints、git SHA、task fingerprint；
- 完整 `u_bitmasks`、random-character seed、character count/mask；
- 所有 absolute/relative per-chain means、raw/debiased per-character moments、两种 frame 的 q_top；
- chain/jackknife 与 finite-population 两类误差；
- 可得时的 absolute/relative weights、`posterior_purity`、
  `posterior_mass_on_planted_class`、`map_success_probability` 与 bounds；
- `q_top_estimate_per_disorder`、`q_top_estimator_name`、validity mask、失败原因；
- 完整 PT ladder、swap、round-trip、cold-acceptance、R-hat/ESS 与 gate diagnostics；
- valid/invalid/missing counts，以及仅由 valid 样本得到的 mean/SEM/crossing inputs。

TI 的主结果来自 full sector weights；sampled 路径的主结果来自 debiased estimator。

## 13. 决定性验证

`validation/014_paper_alignment_20260713/` 是 v2 的唯一当前认证目录。至少需要锁定：

1. 原始 preparation 与 reduced posterior 的逐构型、Z、sector weights、q_top、MAP 完全相等；
2. 固定 `effective_syndrome` 时 true energy 不读取 `epsilon_data_true`；
3. shifted-coordinate 逐构型恒等；
4. `H epsilon_data_true != 0` 时 true 与 legacy 不同，q=0 分别是 quenched/clean；
5. alias 与 canonical 名 seed/结果相同，manifest 只存 canonical；
6. x_error/H_Z 对应 `|+>_L`，对偶 z_error/H_X 对应 `|0>_L`；
7. absolute/relative Mattis 关系、boundary-only invariance 与一般 logical shift 非不变反例；
8. planted mass 与 MAP 的 `(0.1,0.9)` 反例及 purity bounds；
9. basis/nonbasis 加权、独立链 U-statistic、jackknife 与有限总体误差；
10. large-k TI 拒绝、gap diagnostics 无 q_top、auto 三路路由；
11. PT transport 失败导致 INVALID，INVALID 不进入 aggregate；
12. 80-character 不截断、完整 schema/fingerprint/chunk 隔离；
13. exact oracle 输出 full absolute/relative weights、all characters 与全部 posterior statistics；
14. 全仓库错误叙述扫描，以及 conda `12` 下全部 exp101 tests。

旧 `259 tests` 与 `validation/001`–`013` 一律标记为 `PRE_ALIGNMENT`，只能作为历史调试证据，
不能认证本契约。只有 014 全部通过后，`status.md` 才可恢复 `DONE`。
