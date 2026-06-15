# 实验 014 计划：sector 直方图收敛性诊断

## 背景与物理图像

**我们要测的物理量（重复对齐）：**
- 对每一个 disorder 抽样一个误差位形 `e`
- 对这个固定 disorder，用 MCMC 采样热态下的物理构型 `c_t`
- 在每个时间片 `c_t`，计算所有 Wilson loop 的期望值（通过时间平均），记作 `<W_u> = m_u`
- 最终 `q_top = average_u m_u^2`
- 这里每个 Wilson loop 符号是 `W_u(c_t) = ±1`，所以 `m_u` 是 `[-1, 1]` 的实数，`q_top` 是 `[0, 1]`

**为什么 `q_top` 相近不等于正确：**
- `q_top` 把 7 个 Wilson loop 平方后求和再平均，是一个单一标量
- 不同的 sector（`W_1..W_7` 的 ±1 组合）可以给出相近的 `q_top`
- 例如 sector `[+,+,+,+,+,+,+]` 和 sector `[-,-,-,-,-,-,-]` 的 `q_top` 都是 1
- 所以不能只看 `q_top` 是否一致

**正确性判定：sector 直方图**
- 记 `sector_t = (W_1(c_t), W_2(c_t), ..., W_N(c_t))`，其中 `N` 是 logical qubit 数（L=6 时 N=7）
- 对 L=6，sector 有 2^7 = 128 种可能
- 如果 MCMC 已经收敛到热态，那么从 **任何** 合理初态出发，足够长时间后，sector 在 128 种可能性上的 **概率分布**（直方图）应当是相同的
- 这才是热态本身的特征，和初态无关

**我们的诊断指标：Total Variation（TV）距离**
- 定义两个直方图的 TV 距离为 `0.5 * Σ |p_i - q_i|`
- TV=0 表示完全相同，TV=1 表示完全正交
- 当 TV 随时间下降并稳定在很小的值（< 0.05），我们判定热态已采样

---

## 诊断方案

### 诊断一：前后期一致性（热化检验）

**目标：** 检验同一初态下，sector 直方图是否随时间演化趋于稳定

**做法：**
- 取 m=2048 个测量时间片
- 将其分成前后两个半段（各 1024 个样本）
- 分别统计两段的 sector 直方图（归一化到概率分布）
- 计算 TV 距离
- 再将整段分成 8 个等块（各 256 个样本），两两计算 TV 距离，观察是否持续下降

**通过标准：** 最后两个块的 TV 距离 < 0.10

### 诊断二：长链一致性（采样充分性）

**目标：** 验证用 2 倍长时间采样，直方图是否稳定

**做法：**
- 对同一 disorder、同一初态，分别运行 m=2048 和 m=4096 的测量
- 计算两者的 sector 直方图 TV 距离

**通过标准：** TV < 0.05（说明采样量足够）

### 诊断三：不同初态收敛到同一热态（收敛性）

**目标：** 验证从 sector/all_zero/random_high_weight 三种初态出发，最终落到同一个 sector 直方图

**做法：**
- 对同一 disorder（L=6, p=0.05, q=0.23）、同一 disorder seed，用三种初态分别跑 m=2048
- 计算每两种初态之间的 sector 直方图 TV 距离
- 同时也看每种初态自身的前后期 TV 距离

**通过标准：** 三种初态两两之间的 TV 距离 < 0.10；各自前后期 TV 距离也 < 0.10

---

## 实施参数

- 平台：L=6, p=0.05, q=0.23（代表性参数）
- Disorder：1 个 fixed disorder seed（如 seed=42），方便对比
- 每种初态：m=2048 测量（burn-in 后），num_start_chains=3（不同 RNG seed）
- 每条链单独记录 `logical_observable_values_per_measurement`（形状 2048 × 7，int8）
- PT 配置：`q_hot=0.35, rho=0.15, cold_edge_stride=4, m=2048`
- Block count 用于 q_top drift 诊断，但不替代 sector histogram

---

## 输出产物

每条链保存：
- `logical_observable_values_per_measurement`：形状 (2048, 7)，int8 数组
- 派生：
  - `sector_histogram_first_half`：前半 1024 的直方图（128 维）
  - `sector_histogram_second_half`：后半 1024 的直方图
  - `sector_histogram_full`：全 2048 的直方图
  - `block_histograms`：8 个 block 的直方图（用于漂移趋势）
  - `tv_distance_first_vs_second_half`
  - `chain_q_top`：从这条链的 m_u 算出的 q_top

综合分析：
- 对每种初态：显示 top-5 sector 及概率
- 对每种初态：显示前后期 TV 距离
- 对三种初态两两：TV 矩阵
- 对诊断二：长链与短链的 TV 距离

---

## 实现

需要修改的代码：
1. `src/main.py`：在 PT 完成后，除了已有的 `logical_observable_values_per_measurement`，增加保存该数组到结果的路径（目前 PT 路径似乎有这个数组，但需确认冷端直方图保存）
2. `src/summarize_exp36_cold_sector.py`（新建）：读取保存的 logical_observable_values，计算 sector 直方图，打印诊断报告

---

## 停止条件

如果三种初态的前后期 TV 距离都 < 0.10，且三种初态两两 TV < 0.10，则：
- 算法已验证收敛到热态，可以进入生产扫描
- 当前 PT 参数组合（q_hot=0.35, rho=0.15）是可用的
- 接下来关注速度优化

如果 TV 距离持续较大（> 0.3），则：
- MCMC 没有充分热化，需要调整 PT 参数（更大的 q_hot？更多温度点？更长 burn-in？）
- 记录失败的配置和 TV 距离，便于后续分析

---

## 外部顾问评估记录

按流程尝试调用 DeepSeek-V4-Pro 和 Kimi-K2.6 评估本计划，但本机 `claude`
CLI 在 `claude --version` 以及两个模型调用中均无输出并挂起，已终止相关进程。
因此本轮不等待外部顾问，避免因工具故障浪费实验时间。

我对原 plan 做如下修正：

- 不把 `TV < 0.05/0.10` 当作绝对数学证明，而是作为本轮工程 gate。
- 先跑 m=2048 的三初态测试；只有 sector 直方图已经接近时，再跑 m=4096 复核。
- 报告必须同时给出 top sectors、前后期 TV、初态间 TV、`q_top` 和 `m_u`，避免 `q_top` 掩盖 sector 分布差异。
- 如果 L=6 结果不清楚，下一步优先用小 L 做 exact/very-long reference，而不是直接加大生产扫描。
