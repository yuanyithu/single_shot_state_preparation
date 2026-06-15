# plan.md

# 3D Toric Code MCMC 局部极小修复计划

## 0. 总目标

目标不是“让某条 MCMC 链看起来能跑出好曲线”，而是构造一个**可证伪、可复现、可扩展**的计算流程，使得最终输出的曲线

\[
\overline q_{\rm top}(L,p,q)
\]

确实来自每个 disorder 下 8 个逻辑扇区的热权重，而不是来自冷端链被冻结在某个局部极小或某个逻辑扇区后的假象。

本计划用于 Codex 的 `/goal` 多阶段执行。每个阶段都只要求 Codex 完成明确的数学目标、诊断指标和数据产物。具体代码结构未知时，不要求固定实现方式，只要求实现后的结果满足本文件定义的判据。

---

## 1. 核心数学逻辑

### 1.1 状态变量

3D toric code 的 qubit 在边上。把边空间写成

\[
C_1 = \mathbb F_2^n,\qquad n=3L^3.
\]

原始采样变量为 data-error 链

\[
c\in C_1.
\]

固定一个 disorder：

\[
\eta\in C_1,
\qquad
s\in \mathbb F_2^m.
\]

其中 `η` 是真实 data error，`s` 是带测量噪声的 syndrome。令

\[
x=c\oplus \eta,
\]

并定义测量错误图样

\[
m_{\rm err}=s\oplus H_Z\eta.
\]

则能量写成

\[
E(x;K_p,K_q)
=
K_p |x|+K_q |H_Zx\oplus m_{\rm err}|,
\]

其中

\[
K_p=\log\frac{1-p}{p},
\qquad
K_q=\log\frac{1-q}{q}.
\]

采样分布是

\[
\pi(x)\propto \exp[-E(x;K_p,K_q)].
\]

---

### 1.2 逻辑扇区

令

\[
P_L:C_1\to \mathbb F_2^3
\]

为逻辑投影。每个构型的逻辑类为

\[
g(x)=P_Lx\in\mathbb F_2^3.
\]

共有 8 个逻辑扇区：

\[
\Omega_g=
\{x\in C_1: P_Lx=g\},
\qquad
 g\in\mathbb F_2^3.
\]

定义每个扇区的限制配分函数

\[
Z_g(K_p,K_q;m_{\rm err})
=
\sum_{x\in\Omega_g}
\exp[-E(x;K_p,K_q)].
\]

相应的热权重为

\[
w_g=
\frac{Z_g}{\sum_{h\in\mathbb F_2^3}Z_h}.
\]

因此最终真正需要估计的不是整条链在所有构型之间的完整混合，而是 8 个数：

\[
\{w_g\}_{g\in\mathbb F_2^3}.
\]

---

### 1.3 `q_top` 只依赖 8 个扇区权重

对任意非零逻辑方向

\[
u\in\mathbb F_2^3\setminus\{0\},
\]

定义逻辑 observable

\[
O_u(x)=(-1)^{\langle u,g(x)\rangle}.
\]

其热平均为

\[
m_u
=
\sum_g w_g(-1)^{\langle u,g\rangle}.
\]

利用有限群 Fourier / Parseval 恒等式，可得

\[
q_{\rm top}
=
\frac{1}{7}\sum_{u\neq 0}m_u^2
=
\frac{8\sum_g w_g^2-1}{7}.
\]

所以：

- 若某个扇区权重接近 1，则

  \[
  q_{\rm top}\approx 1.
  \]

- 若 8 个扇区接近均匀，

  \[
  w_g\approx \frac18,
  \qquad
  q_{\rm top}\approx 0.
  \]

这意味着曲线是否正确，归根结底取决于 `w_g[8]` 是否正确，而不是某条 MCMC 链是否表面收敛。

---

### 1.4 当前失败模式的数学解释

改变逻辑扇区必须加入一个非平凡 winding chain / logical loop。深有序相中 `x` 通常稀疏，插入长度约为 `L` 的直 winding loop 会使重量上升约 `L`，因此单步接受率近似

\[
\exp(-K_pL)
=
\left(\frac{p}{1-p}\right)^L.
\]

当 `p` 较小时，这个数随 `L` 指数下降。局域 move 试图连续跨越逻辑扇区时，又会遇到随系统尺寸增长的自由能势垒。

因此 landscape 应理解为：

```text
8 个逻辑扇区深势阱
×
每个势阱内部可通过局域/stabilizer/syndrome move 采样
×
势阱之间存在随 L 增长的自由能势垒
```

所以主线修复策略不应依赖冷端链频繁跨越逻辑扇区，而应直接估计每个逻辑扇区的相对自由能。

---

## 2. 正确曲线的定义

一条曲线被认为“正确”，必须满足下面的数学条件。

### 2.1 每个 disorder 下的基本产物

对每个

```text
L, p, q, disorder_seed
```

至少输出：

```text
w_g[8]
DeltaF_g[8]
q_top
stderr_w_g[8]
stderr_DeltaF_g[8]
stderr_q_top
诊断状态 flags
```

其中

\[
\Delta F_g=F_g-F_0,
\qquad
w_g=\frac{e^{-\Delta F_g}}{\sum_h e^{-\Delta F_h}}.
\]

最终曲线是 disorder average：

\[
\overline q_{\rm top}(L,p,q)
=
\mathbb E_{\eta,s}[q_{\rm top}(L,p,q;\eta,s)].
\]

必须同时保存 bootstrap 或 jackknife 误差条。

---

### 2.2 禁止的成功标准

以下现象不能作为成功依据：

```text
1. q_top 曲线看起来平滑。
2. 多个初态得到相同 q_top。
3. 冷端链长时间稳定在某个 sector。
4. winding acceptance 非零但冷端没有 sector round trip。
5. PT swap acceptance 合理但逻辑 sector 没有输运到冷端。
```

原因：多条链可能同时共冻在同一个错误扇区或同一个主导扇区附近，从而给出假性一致结果。

---

## 3. Benchmark 总体原则

所有优化都必须先通过 benchmark，再看生产曲线。

推荐判据顺序：

```text
小 L 精确 benchmark
→ K_p=0 均匀扇区 sanity check
→ sector-resolved free energy 收敛
→ 动力学诊断
→ 大 L 生产曲线
```

---

# Stage 0：建立不可绕过的诊断输出

## 目标

在不改变核心采样算法的前提下，先把当前失败模式量化。Codex 在本阶段不需要优化曲线，只需要保证所有后续阶段都有足够诊断信息。

---

## 必须记录的量

对每个 run 记录：

```text
sector_trace[t] = P_L x_t
sector_histogram[8]
q_top_from_sector_histogram
winding_move_attempts
winding_move_accepts
winding_acceptance_by_temperature
cold_sector_round_trips
logical_observable_tau_int[7]
num_chains_never_flipped_sector
replica_temperature_round_trips     # 若有 PT
per_temperature_sector_histogram    # 若有 PT
```

其中 7 个逻辑 observable 为

\[
O_u=(-1)^{\langle u,g(x)\rangle},
\qquad
u\in\mathbb F_2^3\setminus\{0\}.
\]

---

## 冷端 sector round trip 定义

设冷端测量链的主导扇区为

\[
g_\star=\arg\max_g \widehat w_g.
\]

一次逻辑往返定义为

\[
g_\star\to g\neq g_\star\to g_\star.
\]

建议硬阈值：

```text
cold_sector_round_trips >= 20
```

若为 0，则该采样器在此点直接判为动力学失败。

---

## 逻辑自相关时间

对每个非零 `u` 估计

\[
\tau_{\rm int}(O_u)
=
\frac12+
\sum_{t=1}^{T_{\rm cut}}\rho_u(t).
\]

要求测量长度满足

\[
T_{\rm meas}
>
50\max_{u\neq0}\tau_{\rm int}(O_u).
\]

推荐更严格目标：

\[
T_{\rm meas}
>
100\max_{u\neq0}\tau_{\rm int}(O_u).
\]

---

## Stage 0 退出条件

本阶段完成后，应能回答：

```text
当前坏曲线是因为 q_top 真接近 1，还是因为冷端 sector 共冻？
当前每个 L,p,q 点的 sector round trip 是多少？
当前每个点的逻辑 τ_int 是否可估？
当前 q_top 是否来自完整 sector histogram？
```

若这些问题无法回答，则不得进入生产优化。

---

# Stage 1：小尺寸金标准 benchmark

## 目标

在小系统上建立 ground truth，用来判断任何算法是否真的无偏。

建议尺寸：

```text
L = 2, 3
```

对固定的一组 disorder seeds 和 `(p,q)` 网格，计算精确或高可信参照值：

```text
exact_w_g[8]
exact_DeltaF_g[8]
exact_q_top
```

---

## 金标准计算对象

对每个逻辑扇区：

\[
Z_g
=
\sum_{x:P_Lx=g}
\exp[-K_p|x|-K_q|H_Zx\oplus m_{\rm err}|].
\]

然后

\[
w_g^{\rm exact}
=
\frac{Z_g}{\sum_h Z_h},
\]

\[
q_{\rm top}^{\rm exact}
=
\frac{8\sum_g (w_g^{\rm exact})^2-1}{7}.
\]

---

## 指标 A：sector distribution total variation

\[
{\rm TV}
=
\frac12\sum_g
|\widehat w_g-w_g^{\rm exact}|.
\]

建议合格线：

```text
TV <= 0.02 作为强通过
TV <= 0.05 作为临界附近可接受上限
```

---

## 指标 B：自由能差误差

定义

\[
\Delta F_g^{\rm exact}
=-\log\frac{w_g^{\rm exact}}{w_0^{\rm exact}},
\]

\[
\widehat{\Delta F}_g
=-\log\frac{\widehat w_g}{\widehat w_0}.
\]

对满足

\[
w_g^{\rm exact}>10^{-3}
\]

的扇区，要求

\[
|\widehat{\Delta F}_g-\Delta F_g^{\rm exact}|
<0.1\sim0.2.
\]

---

## 指标 C：`q_top` 误差

\[
\delta q_{\rm top}
=
|\widehat q_{\rm top}-q_{\rm top}^{\rm exact}|.
\]

建议合格线：

```text
delta_q_top <= 0.01 到 0.02
```

---

## Stage 1 退出条件

在 `L=2,3` 的 benchmark 网格上，算法必须满足：

```text
TV(w_hat, w_exact) 合格
DeltaF 误差合格
q_top 误差合格
误差条覆盖 exact 值
```

若失败，优先怀疑：

```text
1. energy 定义错误
2. H_Z x xor m_err 的 convention 错误
3. P_L 逻辑投影错误
4. sector 标号与 logical representative 不一致
5. q_top 公式或归一化错误
6. sampler 尚未在 sector 内热化
```

---

# Stage 2：复现并隔离当前失败模式

## 目标

用现有 sampler 重现坏曲线，并用 Stage 0 的诊断证明问题是否来自逻辑 sector freezing。

---

## 需要完成的对照

对至少以下尺寸运行：

```text
L = 3, 4, 5, 6
```

在同一批 `(p,q)` 和 disorder seeds 上记录：

```text
q_top
sector_histogram[8]
cold_sector_round_trips
logical_tau_int[7]
winding_acceptance_by_temperature
```

---

## 判据

若出现以下组合，则将该点标记为 `FROZEN_SECTOR_FAIL`：

```text
q_top ≈ 1
cold_sector_round_trips = 0
至少一个 logical τ_int 不可估或接近 T_meas
sector_histogram 集中在单一 sector
```

如果小 `L` exact 显示该点应有非零次主导扇区权重，但现有 sampler 给出单 sector 权重，则明确判定为假性有序。

---

## Stage 2 退出条件

形成一张 failure map：

```text
可信区域：baseline sampler 通过 exact / round-trip 诊断
不可信区域：baseline sampler 发生 sector freezing
```

后续优化的目标不是单纯改变曲线，而是把不可信区域转化为通过数学 benchmark 的可信区域。

---

# Stage 3：实现 sector-constrained 采样器

## 目标

构造一个只在固定逻辑扇区内采样的 Markov chain。

对每个

\[
g\in\mathbb F_2^3,
\]

采样空间为

\[
\Omega_g=\{x:P_Lx=g\}.
\]

允许 proposal

\[
x\to x\oplus\delta
\]

必须满足

\[
P_L\delta=0.
\]

因此该 sampler 不需要，也不应该，跨越逻辑扇区。

---

## 数学要求

### 3.1 扇区保持

所有局域 move、stabilizer move、syndrome-carrying move 都必须满足：

\[
P_Lx_{t+1}=P_Lx_t=g.
\]

测试标准：

```text
长时间运行后 sector_trace 只包含指定 g
任何违反 sector invariant 的 move 直接报错
```

---

### 3.2 扇区内 detailed balance

固定 `g` 后，目标分布为

\[
\pi_g(x)
=
\frac{1}{Z_g}
\mathbf 1[P_Lx=g]
\exp[-E(x)].
\]

proposal 和 accept/reject 必须满足

\[
\pi_g(x)T_g(x\to x')
=
\pi_g(x')T_g(x'\to x).
\]

若 proposal 对称，则接受率为

\[
A(x\to x')=
\min\{1,\exp[-(E(x')-E(x))]\}.
\]

若 proposal 非对称，则必须使用 Metropolis-Hastings 比率。

---

### 3.3 扇区内热化

扇区内不要求逻辑翻转，但要求局域 observable 热化。至少监控：

```text
|x|
|H_Z x xor m_err|
E(x)
```

以及它们的 block mean 和自相关时间。

---

## Stage 3 退出条件

对每个 `g=0,...,7`：

```text
sector invariant 永不破坏
energy observable 多初态一致
block bootstrap 误差稳定
L=2,3 下可复现 exact sector 内均值或最终 exact w_g
```

---

# Stage 4：主线算法——sector-resolved thermodynamic integration

## 目标

直接估计每个逻辑扇区的自由能差

\[
\Delta F_g=F_g-F_0,
\]

再由自由能差计算

\[
w_g,
\qquad
q_{\rm top}.
\]

这一步是整个计划的主线。

---

## 4.1 关键锚点：`K_p=0` 时 8 个扇区等权

当

\[
K_p=0,
\]

能量为

\[
E(x)=K_q|H_Zx\oplus m_{\rm err}|.
\]

逻辑代表元 `\ell` 满足

\[
H_Z\ell=0,
\qquad
P_L\ell=h.
\]

于是映射

\[
x\mapsto x\oplus\ell
\]

会把扇区 `g` 双射到 `g\oplus h`，并保持能量不变。因此

\[
Z_0(0,K_q)=Z_1(0,K_q)=\cdots=Z_7(0,K_q).
\]

所以

\[
w_g(K_p=0)=\frac18,
\qquad
q_{\rm top}(K_p=0)=0.
\]

这是自由能积分的天然起点。

---

## 4.2 积分公式

固定目标 `K_q`，沿 `K_p` 从 0 积分到目标值 `K_p^*`。

每个扇区的自由能为

\[
F_g(K_p,K_q)=-\log Z_g(K_p,K_q).
\]

因为

\[
\frac{\partial F_g}{\partial K_p}
=
\langle |x|\rangle_{g,K_p,K_q},
\]

所以

\[
F_g(K_p^*,K_q)-F_g(0,K_q)
=
\int_0^{K_p^*}
\langle |x|\rangle_{g,K_p,K_q}\,dK_p.
\]

由于 `K_p=0` 时所有扇区自由能相同，得到

\[
\Delta F_g(K_p^*,K_q)
=
\int_0^{K_p^*}
\left[
\langle |x|\rangle_{g,K_p,K_q}
-
\langle |x|\rangle_{0,K_p,K_q}
\right]dK_p.
\]

等价地，先定义

\[
I_g=
\int_0^{K_p^*}
\langle |x|\rangle_{g,K_p,K_q}\,dK_p,
\]

再用

\[
\Delta F_g=I_g-I_0.
\]

最后

\[
w_g=
\frac{e^{-\Delta F_g}}{\sum_h e^{-\Delta F_h}},
\]

\[
q_{\rm top}=
\frac{8\sum_gw_g^2-1}{7}.
\]

---

## 4.3 `K_p` 网格原则

不要在 `p` 上均匀积分，而应在直接进入能量的变量 `K_p` 上积分：

\[
K_p=\log\frac{1-p}{p}.
\]

初始网格建议：

```text
K_p grid: 0 到 K_p_target
点数: 32 到 64
```

然后根据收敛情况自适应加密。

优先加密区域：

```text
1. <|x|> 随 K_p 变化快的位置
2. q_top 对网格加密敏感的位置
3. 接近 crossing / transition 的位置
4. 深低 p 区域中自由能差变化陡峭的位置
```

---

## 4.4 数值积分收敛判据

同时运行粗网格和细网格：

```text
coarse: M 个 K_p 点
fine:   2M 或 4M 个 K_p 点
```

要求：

\[
{\rm TV}(w^{\rm coarse},w^{\rm fine})<0.01\sim0.02,
\]

并且

\[
|q_{\rm top}^{\rm coarse}-q_{\rm top}^{\rm fine}|<0.01\sim0.02.
\]

若不满足，则该点标记为 `TI_GRID_FAIL`，需要加密网格或增加采样。

---

## 4.5 统计误差

对每个扇区、每个 `K_p` 网格点，把 MCMC 时间序列切成 block，block 长度应大于局域 observable 的自相关时间。

bootstrap replica 必须重新计算完整流程：

```text
mu_g(K_p)
→ I_g
→ DeltaF_g
→ w_g
→ q_top
```

最终输出：

```text
mean_q_top
stderr_q_top
95% CI
stderr_w_g[8]
stderr_DeltaF_g[8]
```

---

## Stage 4 退出条件

在 `L=2,3` 上：

```text
sector-TI 的 w_g 匹配 exact w_g
sector-TI 的 q_top 匹配 exact q_top
coarse/fine K_p 网格一致
bootstrap CI 覆盖 exact 值
```

在大 `L` 上：

```text
K_p=0 sanity check 通过
TI grid convergence 通过
sector 内 observable 热化通过
所有点都有明确 PASS/FAIL flag
```

---

# Stage 5：独立自由能交叉验证 AIS / bridge / MBAR

## 目标

用与 thermodynamic integration 不同的自由能估计器交叉验证 `w_g`。

这一步不一定是主线，但建议作为生产曲线可信度的重要验证层。

---

## 5.1 AIS 公式

对每个扇区 `g`，定义中间分布

\[
\pi_i^{(g)}(x)
\propto
\mathbf 1[P_Lx=g]
\exp[-K_{p,i}|x|-K_q|H_Zx\oplus m_{\rm err}|],
\]

其中

\[
K_{p,0}=0,
\qquad
K_{p,M}=K_p^*.
\]

AIS 权重为

\[
\log W
=
-\sum_{i=1}^M
(K_{p,i}-K_{p,i-1})|x_{i-1}|.
\]

于是

\[
\widehat{\frac{Z_g(K_p^*,K_q)}{Z_g(0,K_q)}}
=
\frac1R\sum_{r=1}^R W_r^{(g)}.
\]

因为 `K_p=0` 时

\[
Z_g(0,K_q)
\]

对所有 `g` 相同，所以

\[
w_g\propto
\frac1R\sum_{r=1}^R W_r^{(g)}.
\]

---

## 5.2 AIS 诊断

保存：

```text
logW samples per sector
AIS effective sample size
forward/reverse consistency if available
schedule refinement comparison
```

有效样本量：

\[
{\rm ESS}
=
\frac{(\sum_r W_r)^2}{\sum_r W_r^2}.
\]

建议合格线：

```text
ESS >= 100
或 ESS / R >= 0.05
```

---

## 5.3 与 TI 的一致性

要求：

\[
{\rm TV}(w^{\rm TI},w^{\rm AIS})<0.02\sim0.03,
\]

并且

\[
|q_{\rm top}^{\rm TI}-q_{\rm top}^{\rm AIS}|<0.01\sim0.02.
\]

若不一致，优先检查：

```text
1. AIS schedule 是否太粗
2. AIS ESS 是否过低
3. sector-constrained transition 是否混合不足
4. TI grid 是否太粗
5. K_p=0 初始化是否不是正确的 sector 分布
```

---

## Stage 5 退出条件

至少在代表性的 `(L,p,q)` 点上满足：

```text
TI 与 AIS / MBAR 的 w_g 一致
TI 与 AIS / MBAR 的 q_top 一致
两者都匹配小 L exact benchmark
```

---

# Stage 6：可选增量修复——全局 winding proposal

## 目标

如果仍希望保留跨扇区动力学采样，则改造 winding proposal，使其不再只提议固定直 winding loop。

注意：本阶段是辅助，不应替代 Stage 4 的 sector-free-energy 主线，除非它通过全部动力学 benchmark。

---

## 6.1 固定候选 loop 库

预先为每个非零逻辑类 `h` 构造候选集合

\[
\mathcal C_h=\{\ell_{h,1},\ldots,\ell_{h,M_h}\},
\]

满足

\[
P_L\ell_{h,j}=h,
\qquad
H_Z\ell_{h,j}=0.
\]

proposal：

```text
随机选 h != 0
随机选 ell in C_h
x' = x xor ell
```

若选择概率与当前状态无关，则 proposal 对称，接受率为

\[
A(x\to x')
=
\min\{1,e^{-[E(x')-E(x)]}\}.
\]

由于 `H_Z ell=0`，syndrome 项不变，能量差主要为

\[
\Delta E=K_p(|x\oplus\ell|-|x|).
\]

候选库应包含：

```text
1. straight winding loops
2. all translated straight winding loops
3. deformed low-weight nontrivial cycles
4. disorder-adapted low-cost cycles
5. combinations of fundamental logical cycles
```

---

## 6.2 当前构型自适应 proposal

若 proposal 根据当前 `x` 选择低代价 winding loop，则一般是非对称的，必须使用 Metropolis-Hastings：

\[
A(x\to x')
=
\min\left\{
1,
\exp[-(E(x')-E(x))]
\frac{q(x'\to x)}{q(x\to x')}
\right\}.
\]

不得省略 proposal ratio。

---

## 6.3 动力学成功判据

此阶段只有在以下条件下才算成功：

```text
cold_sector_round_trips >= 20
logical τ_int 明显小于测量长度
sector histogram 与 exact / TI 的 w_g 一致
不同初态的 w_g 一致，且不是因为都不翻转
```

若 winding proposal 改善 acceptance 但冷端没有 sector round trip，则仍判失败。

---

# Stage 7：可选增量修复——PT ladder 到 `K_p=0`

## 目标

如果使用 parallel tempering，则温度 ladder 应能把逻辑自由度带到真正无序的端点。

建议固定 `K_q`，沿 `K_p` 建 ladder：

\[
K_p^{(0)}=K_p^*,
\qquad
K_p^{(R)}=0.
\]

热端 `K_p=0` 时 8 个逻辑扇区严格等权，这是最干净的逻辑无序端点。

---

## 7.1 Swap 接受率

相邻 replica 在 `K_p` 上交换时，若 `K_q` 固定，则接受率为

\[
A_{\rm swap}
=
\min\left\{
1,
\exp[-(K_{p,i}-K_{p,j})(|x_j|-|x_i|)]
\right\}.
\]

建议相邻 swap acceptance 处于：

```text
0.15 到 0.5
```

但 swap acceptance 只是辅助指标，不是最终成功标准。

---

## 7.2 PT 硬诊断

必须记录：

```text
replica index round trips
cold-sector round trips
per-temperature sector histogram
hot-end sector histogram
sector excitation transport from hot end to cold end
```

合格线：

```text
热端 g 分布接近 1/8
labeled replica 完成多次 temperature round trip
冷端 sector round trip >= 20
冷端 w_g 与 TI / exact 一致
```

若热端已经均匀但冷端不翻转，说明 transport 失败。

若热端也不均匀，说明 ladder 端点或 proposal 仍不足。

---

# Stage 8：生产曲线生成

## 目标

在通过前面 benchmark 后，生成目标曲线：

\[
\overline q_{\rm top}(L,p,q)
\]

并给出每个点的可信状态。

---

## 8.1 每个点的计算流程

对每个

```text
L, p, q, disorder_seed
```

执行：

```text
1. sector-TI 估计 w_g[8]
2. 计算 DeltaF_g[8]
3. 计算 q_top
4. bootstrap 得到误差条
5. 检查 TI grid convergence
6. 可选：AIS / MBAR 交叉验证
7. 输出 PASS / FAIL / WARNING flag
```

---

## 8.2 Disorder average

对每个 `(L,p,q)`：

\[
\overline q_{\rm top}
=
\frac1N\sum_{a=1}^N q_{\rm top}^{(a)}.
\]

误差条可以使用 disorder bootstrap：

```text
resample disorder seeds
recompute mean q_top
report standard error and 95% CI
```

如果每个 disorder 内部还有 Monte Carlo 误差，则 bootstrap 应把内部误差一并传播，或者至少分别报告：

```text
disorder_error
mcmc_error
ti_grid_error
```

---

## 8.3 曲线接受标准

一条生产曲线只有在满足以下条件时才可接受：

```text
1. 组成该曲线的大多数点为 PASS。
2. crossing 区域的关键点全部为 PASS。
3. 小 L 曲线与 exact benchmark 一致。
4. q=0 或其他已知 sanity case 行为正确。
5. 曲线变化不能来自 FAIL 点的系统性偏差。
6. 误差条包含 disorder bootstrap 和自由能估计误差。
```

对于任何 `WARNING` 或 `FAIL` 点，图中必须能够区分显示，不能混入主曲线后声称成功。

---

# Stage 9：Codex `/goal` 执行模板

下面的模板可以直接作为每个阶段的 `/goal` 起点。

---

## Goal 0：诊断输出

```text
/goal
请在现有项目中加入逻辑扇区诊断。不要改变采样算法的物理含义。

数学要求：
1. 对每个测量样本计算 g=P_L x ∈ F_2^3。
2. 保存 sector_trace、sector_histogram、cold_sector_round_trips。
3. 对 7 个 O_u=(-1)^{<u,g>} 估计积分自相关时间。
4. 如果使用 PT，保存每个温度档的 sector histogram、winding acceptance、replica round trips。
5. q_top 必须从 w_g 或 sector histogram 通过 (8*sum_g w_g^2-1)/7 计算。

完成标准：
每个 run 的输出文件足以判断 cold sector 是否共冻。
```

---

## Goal 1：小 L 金标准

```text
/goal
请建立 L=2,3 的 exact 或高可信 benchmark。

数学目标：
对固定 disorder 和 p,q 网格计算每个逻辑扇区的 Z_g、w_g、DeltaF_g、q_top。

公式：
Z_g=sum_{x:P_Lx=g} exp[-K_p|x|-K_q|H_Zx xor m_err|]
w_g=Z_g/sum_h Z_h
q_top=(8*sum_g w_g^2-1)/7

完成标准：
保存 exact_w_g[8]、exact_DeltaF_g[8]、exact_q_top，并提供比较任意 sampler 输出的 TV、DeltaF error、q_top error。
```

---

## Goal 2：复现失败模式

```text
/goal
请用现有 sampler 在 L=3,4,5,6 上运行诊断，生成 baseline failure map。

数学判断：
如果 q_top≈1 但 cold_sector_round_trips=0，且 sector_histogram 集中在单一 sector，则标记为 FROZEN_SECTOR_FAIL。

完成标准：
输出每个 L,p,q 的 q_top、sector_histogram、cold_sector_round_trips、logical_tau_int、failure_flag。
```

---

## Goal 3：固定扇区采样器

```text
/goal
请实现或整理一个 sector-constrained sampler。

数学要求：
1. 输入目标逻辑扇区 g。
2. 采样空间限制为 Ω_g={x:P_Lx=g}。
3. 所有 proposal δ 必须满足 P_Lδ=0。
4. 采样目标为 π_g(x) ∝ 1[P_Lx=g] exp[-E(x)]。
5. 若 proposal 非对称，必须使用 Metropolis-Hastings ratio。

完成标准：
sector_trace 永远等于指定 g；局域 observable 的 block mean 稳定；L=2,3 上能支持后续 free-energy benchmark。
```

---

## Goal 4：sector-resolved thermodynamic integration

```text
/goal
请实现主线估计器：sector-resolved thermodynamic integration。

数学目标：
固定 K_q，对每个逻辑扇区 g，在 K_p 从 0 到 K_p_target 的网格上估计 μ_g(K_p)=<|x|>_{g,K_p,K_q}。

积分：
I_g=∫_0^{Kp_target} μ_g(K_p)dK_p
DeltaF_g=I_g-I_0
w_g=exp(-DeltaF_g)/sum_h exp(-DeltaF_h)
q_top=(8*sum_g w_g^2-1)/7

关键 sanity check：
K_p=0 时所有 Z_g 相等，因此 w_g=1/8，q_top=0。

完成标准：
L=2,3 上 TV(w_TI,w_exact)、DeltaF error、q_top error 均合格；coarse/fine K_p grid 一致；输出 bootstrap CI。
```

---

## Goal 5：AIS / MBAR 交叉验证

```text
/goal
请实现一个独立的自由能交叉验证器，例如 AIS、bridge sampling 或 MBAR。

数学目标：
在每个 sector g 内估计 Z_g(Kp_target,Kq)/Z_g(0,Kq)。由于 K_p=0 时所有 Z_g(0,Kq) 相等，所以 w_g 正比于该 ratio。

AIS 权重：
log W = -sum_i (Kp_i-Kp_{i-1}) |x_{i-1}|

完成标准：
w_AIS 与 w_TI 的 TV 小于 0.02~0.03；q_top_AIS 与 q_top_TI 差小于 0.01~0.02；AIS ESS 合格。
```

---

## Goal 6：可选 winding proposal 改造

```text
/goal
请改造 winding proposal，但不要让它替代 sector free-energy 主线。

数学要求：
1. 候选 loop ell 必须满足 H_Z ell=0。
2. 它的逻辑类 P_L ell 应为非零 h。
3. 若从固定候选库均匀选择，则 proposal 对称，使用普通 Metropolis。
4. 若根据当前 x 自适应选择低能 loop，则必须使用 Metropolis-Hastings proposal ratio。

完成标准：
cold_sector_round_trips >= 20；logical τ_int 合格；sector histogram 与 exact/TI 的 w_g 一致。
```

---

## Goal 7：PT ladder 到 K_p=0

```text
/goal
请将 parallel tempering ladder 设计为从 K_p_target 连接到 K_p=0，并加入逻辑 sector 输运诊断。

数学要求：
K_p=0 端点处逻辑 sector 应严格等权。

完成标准：
热端 sector histogram 接近均匀；labeled replica 有多次 temperature round trip；冷端 sector round trip 合格；冷端 w_g 与 TI/exact 一致。
```

---

## Goal 8：生产曲线

```text
/goal
请基于已经通过 benchmark 的 sector free-energy estimator 生成生产曲线。

数学要求：
每个 disorder 输出 w_g[8]、DeltaF_g[8]、q_top、误差条和 PASS/FAIL flag。
每个 L,p,q 输出 disorder average q_top 及 CI。

完成标准：
关键 crossing 区域由 PASS 点组成；小 L 与 exact 一致；q_top 曲线可由保存的 w_g[8] 重建；失败点不得混入主曲线声称成功。
```

---

# Stage 10：失败时的决策树

## 10.1 如果 `K_p=0` 下不是 `w_g=1/8`

优先检查：

```text
P_L 定义
logical representative
H_Z ell=0 是否成立
sector 标签 convention
K_p 是否真的为 0
q_top 公式
normalization
```

该问题不解决，不得继续。

---

## 10.2 如果小 L exact 不匹配

优先检查：

```text
energy E(x) 的定义
m_err=s xor H_Z eta
x=c xor eta
H_Z 的边/面 indexing
P_L 与 boundary condition
sector-constrained sampler 的 detailed balance
free-energy 积分符号
softmax(-DeltaF) 的符号
```

---

## 10.3 如果 TI coarse/fine 不一致

处理顺序：

```text
1. 加密 K_p grid
2. 增加每个 grid point 的采样量
3. 检查 <|x|> block error
4. 检查 sector 内混合
5. 使用 AIS/MBAR 交叉验证
```

---

## 10.4 如果 AIS ESS 过低

处理顺序：

```text
1. 加密 annealing schedule
2. 增加中间 transition steps
3. 改善 sector 内 move set
4. 使用 bidirectional / bridge / MBAR
5. 限制 AIS 只作为诊断，不作为主估计器
```

---

## 10.5 如果大 L 曲线仍异常接近 1

不要直接认为物理上就是 1。先检查：

```text
w_g[8] 是否真的有非零尾部
DeltaF_g 是否大到超出统计分辨率
bootstrap 是否能解析次主导 sector
TI grid error 是否被低估
AIS/MBAR 是否一致
small L trend 是否连续外推
```

若所有自由能诊断都通过，则可以接受 `q_top≈1` 是物理结果。否则标记为不可信。

---

# 11. 最终 Definition of Done

整个项目只有在以下条件全部满足时，才认为“能够正确给出想要的曲线”：

```text
1. q_top 由 w_g[8] 计算，而不是只由冻结链的时间平均误判。
2. L=2,3 exact benchmark 通过。
3. K_p=0 均匀 sector sanity check 通过。
4. sector-resolved TI 的 grid convergence 通过。
5. bootstrap / disorder 误差条完整。
6. 关键曲线点有 PASS/FAIL flag。
7. 若使用 PT 或 winding proposal，它们通过 cold-sector round-trip 和 logical τ_int 诊断。
8. 生产曲线的 crossing 或趋势不依赖 FAIL 点。
9. 保存的数据足以重建每个 disorder 的 w_g[8]、DeltaF_g[8]、q_top。
10. 曲线异常时能通过自由能差和 benchmark 定位原因，而不是靠肉眼判断。
```

---

# 12. 推荐执行顺序摘要

```text
Stage 0  诊断输出
Stage 1  L=2,3 exact benchmark
Stage 2  baseline failure map
Stage 3  sector-constrained sampler
Stage 4  sector-resolved thermodynamic integration
Stage 5  AIS / MBAR 交叉验证
Stage 6  可选 winding proposal 改造
Stage 7  可选 PT ladder 到 K_p=0
Stage 8  生产曲线
Stage 9  Codex /goal 持续执行
Stage 10  失败决策树
```

最重要的主线是：

```text
不要要求一条冷端链在 8 个深势阱之间隧穿。
直接估计 8 个逻辑扇区的相对自由能。
再由 w_g 计算 q_top。
```

