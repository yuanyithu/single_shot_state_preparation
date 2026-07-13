# notes/00 — exp101 v2 接口对账与迁移清单

日期：2026-07-13。本文是接口导览，不是物理权威；所有公式以
`../PHYSICS_CONTRACT.md`（`exp101.physics.v2`）为准。

## 1. 生产问题与实现边界

生产配置固定为 `sector=x_error`、`ensemble=true_posterior`：

```text
H_check=H_Z
stabilizer moves=rows(H_X)
logical moves=logical_X
logical observables=logical_Z
prepared state=|+>_L
```

Hadamard 对偶 `z_error/H_X/Z errors` 对应 `|0>_L`。`assemble_sector_model` 的现有矩阵接线
正确，迁移只修名称和文档，不交换矩阵。

exp101 只实现消去 preparation syndrome 后的 reduced MLD posterior；完整 Clifford 电路、
meta-check decoder 与 preparation recovery channel 不在生产接口内。

## 2. 公共对象迁移

| v1 名称/行为 | v2 正式接口 | 迁移规则 |
|---|---|---|
| `DisorderRealization.eta` | `epsilon_data_true` | 旧名只读 deprecated property，访问发 warning |
| `DisorderRealization.delta` | `measurement_error` | 同上 |
| `observed_syndrome` | `effective_syndrome` | 语义固定为 `H epsilon_data_true xor measurement_error` |
| `sigma_arg` | `gibbs_syndrome_argument` | 只表示能量里的 syndrome 参数 |
| `reference_label` / `ell_ref` | `planted_logical_class` | 只表示 observable 的 Mattis reference |
| `model.section` | `logical_sector_section` | domain 仅 `im(H_check)` |
| `DecoderSection` | qubit-chain/logical-sector section 的完整名称 | 不得与 meta decoder 混名 |
| `repo_compat` | `legacy_delta_only` | deprecated alias，seed/fingerprint 前归一化 |
| `paper_true_posterior` | `true_posterior` | deprecated alias，持久化只写 canonical 名 |
| `w0` | 删除 | true 模式改为 `posterior_mass_on_planted_class`；另存 MAP/purity |
| `sector_ti_results.npz` | `scan_results.npz` | v1 chunk 永不复用 |

true-posterior wiring 必须是

```text
gibbs_syndrome_argument = effective_syndrome
planted_logical_class   = phi_r(epsilon_data_true).
```

前者进入能量，后者只进入 absolute/relative observable 变换。这个拆分消除了 v1 中最危险的
“Gibbs disorder 与 Mattis reference 共用一个 wiring 叙述”。

legacy 模式的规范换元状态使用 `measurement_error` 作为 Gibbs syndrome argument，但该模式只做
旧 3D 回归，不能输出论文 MLD success。

## 3. section 接口边界

需要在命名和测试中区分：

1. `measurement_error_decoder`：check space 中的 meta-check projection；生产未实现；
2. `preparation_chain_representative`：把 physical preparation syndrome 映成 qubit chain；只用于
   raw/reduced golden oracle；
3. `logical_sector_section`：把 `im(H_check)` 中 syndrome 映成 chain，用于 logical label。

`effective_syndrome` 在 q>0 时一般不属于 `im(H_check)`，禁止调用
`logical_sector_section.apply(effective_syndrome)`。q=0 时它等于 `H epsilon_data_true`，才可作为
硬约束 coset representative 的输入。

## 4. 模块职责

| 模块 | v2 职责 |
|---|---|
| `model.py` | sector assembly；canonical ensemble normalization；disorder 与 Gibbs/planted wiring 分离 |
| `section.py` | linear 或 decoder-backed qubit-chain logical-sector section；strict domain guard；fingerprint |
| `observables.py` | absolute/relative characters；basis+random nonbasis 设计；正确总体加权；U-statistic/jackknife |
| `reference_mcmc.py` / `fast_mcmc.py` | 同一 canonical energy 的参考/加速实现；保存 per-chain cold traces |
| `pt.py` | large-k q>0 的四独立 PT 实例；ladder/swap/round-trip/cold diagnostics |
| `gates.py` | R-hat/ESS、spread、logical acceptance、PT transport 与 INVALID reasons |
| `sector_ti.py` | `k<=10` full-sector TI；large-k 只保留 gap diagnostics，不产生 q_top |
| `enumerate_exact.py` | small-code exact oracle；absolute/relative full weights、all characters 与三个 posterior statistics |
| `run_scan.py` | auto routing；task fingerprint/cache isolation；atomic v2 chunks；valid-only aggregation |

GF(2)、graph/HGP、logical basis、distance、family registry 和 portable PRNG 等构造模块可沿用 v1
结构，但只有当前测试重跑通过后才能称 v2 可复用。

## 5. observable 与输出接口

直接采样 `m_u_absolute`，再用

```text
m_u_relative = (-1)^(u dot planted_logical_class) * m_u_absolute
```

派生 relative frame。两种 frame 的 per-chain means 都保存。full weights 可用时同时保存：

- `weights_absolute` 与平移后的 `weights_relative`；
- `posterior_mass_on_planted_class`；
- `posterior_purity`；
- `map_success_probability=max(weights_absolute)`；
- purity/MAP bounds 与 normalized `q_top`。

sampled 路径 character 维是实际 `k+num_random_u`，另存 count/mask；不能用 `k` 作固定上限。
二阶矩必须保存 pooled-square raw、independent-chain U-statistic、逐 character 值、jackknife chain
error 与 finite-population character-sampling error。

## 6. engine 与 gate 接口

`engine=auto`：

```text
k<=10          -> full-sector TI
k>10 and q>0   -> PT observable sampling
k>10 and q=0   -> validated 8-start q=0 sampling
```

显式 `ti` 在 `k>10` 创建任务前报错。large-k pairwise API 改为
`basis_sector_free_energy_gap_diagnostics`，返回值中不得出现 `m_u/q_top`。

PT sampled 二阶矩要求四个独立实例；每个实例 round-trip、min adjacent swap、pooled worst-basis
logical acceptance、冷端 R-hat/ESS 都属于硬 gate。TI grid 或 direct/q0/PT 任一 gate 失败，chunk
状态为 `INVALID`，只能留档不能聚合。

## 7. scan v2 identity

`exp101.scan.v2` 的 task fingerprint 覆盖：物理/扫描契约、canonical ensemble、sector、family
rule/seed、code fingerprint、resolved engine 与完整 sampler/estimator config。model cache key 同样包含
sector、family rule/seed。alias 在 fingerprint 前归一化。

主结果统一为 `q_top_estimate_per_disorder` + `q_top_estimator_name`：TI 来自 full weights，sampled
路径来自 debiased estimator。mean/SEM/crossing 只用 `valid_for_aggregation=true`，并显式报告
valid/invalid/missing 数量。

## 8. 历史接口处置

`validation/001`–`013`、旧 `259 tests`、`sector_ti_results.npz` 与旧 summary 全部是
`PRE_ALIGNMENT`。结构性发现可用于设计测试，但 runner 可能依赖弃用字段、biased estimator 或
scan v1 schema。当前认证只看 `validation/014_paper_alignment_20260713/`；详见
`../validation/README.md`。
