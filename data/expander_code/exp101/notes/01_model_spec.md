# notes/01 — exp101 v2 模型推导与统计规格

日期：2026-07-13。本文推导 `../PHYSICS_CONTRACT.md`，不另立物理权威。

## 1. CSS 空间与生产 convention

对 CSS code，`H_X H_Z^T=0`。生产 X-error sector 取

```text
H = H_Z,
B = row(H_X),
ker(H) = B plus logical-X classes.
```

选择配对归一的 logical bases `x_i`、`z_j`，满足 `x_i dot z_j=delta_ij`。X stabilizer 和
logical X 作用在 logical `|+>_L` 上平凡，因此从 `|+>^n` 测 Z checks 的 preparation
representative 可在同一 syndrome fiber 中改变；这不把状态 convention 变成 `|0>_L`。

## 2. 从原始 preparation 变量约化

令

```text
s_prep     = sigma_prep xor measurement_error,
F_total    = c_prep(sigma_prep) xor epsilon_data_true,
sigma_final= H F_total,
y_eff      = s_prep xor sigma_final.
```

因为 `H c_prep(sigma_prep)=sigma_prep`，有

```text
y_eff = H epsilon_data_true xor measurement_error.
```

论文原始求和中的候选 correction/error variable 记为 `a`。取 `e=a xor F_total`，则

```text
H a xor s_prep
  = H(e xor F_total) xor s_prep
  = H e xor y_eff,

a xor F_total = e.
```

所以逐构型严格成立

```text
q(Ha xor s_prep) p(a xor F_total)
  = q(He xor y_eff) p(e).
```

变量替换是双射，故 partition function、按同一 logical decomposition 分组的 sector weights、
Fourier characters、`q_top` 与 MAP success 全部相等。

## 3. canonical 与 shifted energy

对独立 Bernoulli 噪声，忽略与 `e` 无关的归一化常数后，生产能量为

```text
E_true(e;y_eff) = K_p |e| + K_q |H e xor y_eff|.
```

固定 `y_eff` 后，它不读取 `epsilon_data_true`。定义 `x=e xor epsilon_data_true` 只得到等价坐标

```text
E_true_shifted(x)
  = K_p |x xor epsilon_data_true|
  + K_q |H x xor measurement_error|.
```

两式不得同时作为 MCMC 状态定义；生产规范状态是 `e`。

legacy 回归模型

```text
E_legacy(c)
  = K_p |c xor epsilon_data_true|
  + K_q |Hc xor y_eff|
```

换元 `u=c xor epsilon_data_true` 后变为

```text
K_p |u| + K_q |Hu xor measurement_error|,
```

即 delta-only。若 `H epsilon_data_true != 0`，它与 true posterior 一般不同；q=0 时 true 固定在
`He=H epsilon_data_true`，legacy 固定在 `Hu=0`。

## 4. 三种 section 与 logical label

meta-check `measurement_error_decoder`、preparation 的 `c_prep` 和 numerical logical-sector
section `r` 是三个映射。只有后者进入 observable：

```text
r: im(H) -> F_2^n,    H r(sigma)=sigma.
```

定义

```text
phi_r(e) = class of e xor r(He) in ker(H)/B.
```

若 r 线性，可写成 basis character mask；若 r 非线性，直接计算 `e xor r(He)`，不能恢复旧
`r^T` factorized shortcut。

对 `r'=r xor beta` 且 `beta(sigma) in B`，logical class 不变。若差包含随 syndrome 变化的
logical component，则 `phi_r` 的分组可变化；不存在“一般 section 都只是 gauge”的定理。

## 5. absolute 与 planted-relative Fourier characters

absolute sector posterior 记为 `P_abs(l)`，则

```text
m_abs(u) = sum_l (-1)^(u dot l) P_abs(l).
```

令 `l_star=phi_r(epsilon_data_true)`。relative posterior 是平移

```text
P_rel(t)=P_abs(t xor l_star),
m_rel(u)=(-1)^(u dot l_star)m_abs(u).
```

这个符号关系发生在固定 section 内；它不授权跨不同 section fingerprint 合并 run。

## 6. Parseval、q_top 与 MAP

令 `M=2^k`，则 Fourier Parseval 给出

```text
purity = sum_l P_abs(l)^2
       = [1 + sum_{u!=0} m_abs(u)^2] / M,

q_top = mean_{u!=0} m_abs(u)^2
      = [M*purity-1]/[M-1].
```

MAP success 与 planted mass 分别是

```text
P_MAP     = max_l P_abs(l),
P_planted = P_abs(l_star).
```

它们不相等；同时有

```text
purity <= P_MAP <= sqrt(purity).
```

只有 true-posterior full weights 才赋予 `P_MAP` 论文 MLD success 语义。legacy 的最大 mass 只能叫
`largest_sector_mass`。

## 7. sampled-character 总体估计

全部非零 character 数 `N=2^k-1`，basis 数 k。若 nonbasis 从 `N-k` 个元素中无放回均匀抽取，
对一阶或二阶逐 character 量 `f_u` 的总体均值估计为

```text
f_bar = [sum_basis f_u + (N-k) mean_sampled_nonbasis(f_u)] / N.
```

非 basis 抽样方差使用有限总体修正。它描述“抽了哪些 u”的误差，不能与 MCMC chain error 混合。

对 C 条独立链的 character mean `m[a,u]`，raw pooled square 有有限样本正偏。无偏二阶估计为

```text
m2_u = 2/[C(C-1)] sum_{a<b} m[a,u]m[b,u].
```

delete-one-chain jackknife 给 chain-level 标准误。生产 sampled 路径 C>=4，q=0 默认 C=8。若聚合
后的 debiased purity/q_top 落出物理区间，保留 raw 数值并标 INVALID，不裁剪。

## 8. 数值方法的合法域

- `k<=10`：full-sector TI 可由全部 weights 精确重建 characters、purity、MAP 与 q_top；
- `k>10,q>0`：四独立 PT 实例采样 cold observable，二阶量用跨实例 U-statistic；
- `k>10,q=0`：8-start validated q=0 sampling；
- pairwise TI：只测 basis-sector free-energy gap，不提供 posterior Fourier transform。

full TI、direct/q0、PT 的任何 convergence gate 失败，都使样本 `valid_for_aggregation=false`。

## 9. exact oracle 的必要输出

小码枚举必须返回：

- `weights_absolute`、`weights_relative`；
- 全部 nonzero characters 的 absolute/relative 值；
- `posterior_purity`、`posterior_mass_on_planted_class`、
  `map_success_probability`、bounds 与 `q_top`；
- raw-paper/reduced-canonical 的逐构型对应与 partition functions。

golden test 要遍历 `sigma_prep`、`measurement_error`、`epsilon_data_true` 和 `a`，而不是只在
已经约化的模型内部让两套共享代码互相对拍。

## 10. 证据解释

旧 validation 证明的是旧字段和旧 estimator 下的内部一致性。特别是：旧 V1–V6、259 tests、
Nishimori 数值、frame A/B、PT torture 与多节点 bit-identical 都不能替代本规格的 raw/reduced
golden test、无偏二阶矩、四实例 PT、canonical alias、v2 schema 和 INVALID-safe aggregation。
当前证据索引见 `../validation/README.md`。
