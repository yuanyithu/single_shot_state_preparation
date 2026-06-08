# q_top 与 Δf：数学定义、程序对齐、物理图像

本文澄清两件事：(1) 你一直要的 **q_top** 是什么；(2) 我在 `006/deltaf_crossing.png` 里画的 **Δf gap** 是什么、和 q_top 什么关系。所有公式都标了对应的程序位置（`src/exp37_sector_ti.py:行号`），保证与代码一致、无杜撰。

---

## 0. 共识物理图像（出发点）

固定一个 disorder `(s, η)`（`s` 带测量噪声的 syndrome，`η` 真实 data error），edge 上 X-error 构型 `c` 的 Gibbs 态：

$$\pi(c)\ \propto\ \exp\!\big[-K_p\,|c\oplus\eta|\ -\ K_q\,|H_Z c\oplus s|\big],\qquad K_p=\log\tfrac{1-p}{p},\quad K_q=\log\tfrac{1-q}{q}.$$

按 `c = T \oplus L \oplus S` 分解，**逻辑类只由 L 分量决定**，共 $2^k=8$ 类（$k=3$）。对每个逻辑类 $g\in\{0,\dots,7\}$ 定义**受限配分函数**与**扇区权重 / 逻辑类概率**：

$$Z_g=\!\!\sum_{c:\,\text{逻辑类}(c)=g}\!\!e^{-K_p|c\oplus\eta|-K_q|H_Zc\oplus s|},\qquad
\boxed{\,w_g=\frac{Z_g}{\sum_{g'}Z_{g'}}=\pi(\text{逻辑类}=g)\,},\quad \sum_g w_g=1.$$

自由能 $F_g=-\log Z_g$。**q_top 和 Δf 都只是这同一组 $\{w_g\}$（或等价的 $\{F_g\}$）的不同函数。**

程序里这组量的字段（NPZ）：
- `weights_per_disorder[...,8]` $= w_g$
- `delta_f_per_disorder[...,8]` $= F_g-F_0$（见 §2）
- `q_top_per_disorder` $=$ q_top（见 §1）

---

## 1. q_top（你要的量）—— 逻辑类分布的「纯度」，有界、会饱和

**物理定义**（CLAUDE.md）：观测量 $O_u=(-1)^{\langle z_u,\ c+\eta+r(H_Zc)+r(H_Z\eta)\rangle}$ 读出 `c` 相对 `η` 的逻辑类，$m_u=\langle O_u\rangle$，

$$q_\text{top}=\overline{m_u^2}\quad(\text{对 } 2^k-1=7 \text{ 个非平凡逻辑 character } u \text{ 平均}).$$

**用扇区权重表达**（程序实际算的形式）：

$$\boxed{\,q_\text{top}=\frac{2^k\sum_g w_g^2-1}{2^k-1}=\frac{8\sum_g w_g^2-1}{7}\,}$$

> 等价性推导：$m_u=\sum_g w_g\,\chi_u(g)$，$\chi_u(g)=(-1)^{\langle z_u,g\rangle}$。用 $\sum_{u}\chi_u(g)\chi_u(g')=2^k\delta_{gg'}$（含 $u=0$），得 $\sum_{u\neq0}m_u^2=2^k\sum_g w_g^2-1$，除以 $2^k-1$ 即上式。

**含义 / 边界**：$\{w_g\}$ 的纯度（collision probability）。一类独大 $w_g\to1\Rightarrow q_\text{top}\to1$（有序 / 可纠错）；均匀 $w_g=1/8\Rightarrow q_\text{top}=0$（无序）。**有界 $[0,1]$，有序相指数饱和到 1。**

**程序对齐**：`_q_top_from_weights(weights)`，`src/exp37_sector_ti.py:1735`
```python
def _q_top_from_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float((8.0 * np.sum(weights ** 2) - 1.0) / 7.0)
```
（`build_p5` 里用同一式从 `weights` 重建 q_top，reconstruct max abs diff $=0$，证明字段一致。）

---

## 2. Δf（我画的量）—— 扇区自由能 gap，**不是 q_top**，无界、不饱和

**程序里的 `delta_f`**：扇区 $g$ 相对参考扇区 0 的自由能差
$$\texttt{delta\_f}[g]=F_g-F_0.$$
- TI（热力学积分）路径：$F_g(K_p)-F_g(0)=\int_0^{K_p}\langle|c\oplus\eta|\rangle_{g}\,dK_p'$；$K_p=0$ 时逻辑类不进能量、各扇区等同，故 $F_g(0)$ 抵消，得 $\texttt{delta\_f}[g]=F_g-F_0$。
  - 积分：`_integrate_mu`，`:1721` → `np.trapezoid(mu_by_sector, x=kp_grid)`
  - 相减：`delta_f = integrals - integrals[0]`，`:1755`（精确枚举路径等价：`delta_f = -(log_z - log_z[0])`，`:2371`）
- 由 `delta_f` 还原权重：$w_g=\mathrm{softmax}(-\texttt{delta\_f})_g=\dfrac{e^{-\texttt{delta\_f}[g]}}{\sum_{g'}e^{-\texttt{delta\_f}[g']}}=\dfrac{Z_g}{\sum Z_{g'}}$。
  - `_weights_from_delta_f`，`:1725` → `weights = exp(-delta_f); weights /= sum(weights)`

**我画的「Δf gap」**：主导扇区到**最近竞争扇区**的自由能 gap
$$\boxed{\,\Delta f_\text{gap}=F_{(2)}-F_{(1)}=\log\!\frac{w_{(1)}}{w_{(2)}}\,}$$
其中 $w_{(1)}\ge w_{(2)}$ 是最大、第二大的扇区权重（$F_{(1)}\le F_{(2)}$ 是最低两个自由能）。

**程序对齐**（我写的 `006_…/plot_deltaf_crossing.py`）：
```python
df  = d["delta_f_per_disorder"][0]                       # [nq, ndis, 8] = F_g - F_0
gap = np.sort(df, axis=2)[:, :, 1] - np.sort(df, axis=2)[:, :, 0]   # F_(2) - F_(1)
```
排序后 `[1]-[0]` 就是 $F_{(2)}-F_{(1)}$；因 $w_g\propto e^{-F_g}$，$\Delta f_\text{gap}=\log(w_{(1)}/w_{(2)})$。

**含义 / 边界**：翻一个逻辑错的「自由能能垒」/ 对数似然比。**无界 $[0,\infty)$**：有序相 $w_{(1)}\approx1,\ w_{(2)}\approx e^{-cL}\Rightarrow\Delta f_\text{gap}\approx cL$（随 $L$ 线性增长）；无序相 $w_g\approx1/8\Rightarrow\Delta f_\text{gap}\to0$。

---

## 3. 两者关系 —— 为什么 q_top 在有序侧看不出 L 依赖

二者是**同一组 $\{w_g\}$ 的不同函数，单调同向**（"越有序"→ q_top↑ 且 Δf↑），但：

| | q_top | Δf gap |
|---|---|---|
| 表达式 | $(8\sum_g w_g^2-1)/7$ | $\log(w_{(1)}/w_{(2)})$ |
| 取值 | 有界 $[0,1]$ | 无界 $[0,\infty)$ |
| 有序相 $w_{(1)}\to1$ | $\to1$（**饱和**） | $\to\infty$（继续张开 $\sim cL$） |
| 无序相 $w_g\to1/8$ | $\to0$ | $\to0$ |

双扇区主导近似（$w_1$ 大、$w_2$ 小、其余 ≈0）：$q_\text{top}\approx\frac{8(w_1^2+w_2^2)-1}{7}$，$\Delta f=\log(w_1/w_2)$。当 $w_1\to1$：q_top 被 1 卡住、**有序侧不同 $L$ 的劈裂被压到误差棒以下**；Δf 仍随 $L$ 张开。

**结论**：有序侧"大 $L$ → 更受保护"这一 threshold 判据，在 **q_top 上不可见（饱和）**、在 **Δf 上清晰**。这是观测量选择问题，不是物理上没有 threshold。若坚持 q_top，它只能展示无序侧扇出 + 交叉点；有序侧需换不饱和量（Δf）或对 q_top 做 rescale。
