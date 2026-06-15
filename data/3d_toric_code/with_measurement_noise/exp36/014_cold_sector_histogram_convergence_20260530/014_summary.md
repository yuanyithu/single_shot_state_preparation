# 014 cold-sector histogram convergence summary

## 结论

本轮在 `L=6, p=0.05, q=0.23`、同一个 disorder seed `514020` 上做了正确性优先的诊断。

核心结果：三种相差很大的初态 `sector / all_zero / random_high_weight` 最后给出的冷端 sector 驻留分布高度一致；`sector` 初态的 `m=2048` 和 `m=4096` 长链结果也高度一致。

这说明在这个固定 disorder、这个参数点上，当前 PT+cluster 配置没有明显记住初态，冷端 MCMC 样本在 Wilson-loop sector 空间中已经收敛到同一个热态分布。

## 物理图像对应

每个测量时间片都有一个 Wilson loop 符号向量

`sector_t = (W_1(c_t), ..., W_7(c_t))`

本轮没有只看 `q_top`，而是统计了每条链的 `sector_t` 直方图。若 MCMC 没有热化，不同初态可能长期驻留在不同 sector；若已经采到同一热态，不同初态的 sector 直方图应一致。

## 关键数值

三初态 `m=2048`：

- `sector`: `q_top = 0.986653`
- `all_zero`: `q_top = 0.988040`
- `random_high_weight`: `q_top = 0.985546`

前半链 vs 后半链 TV：

- `sector`: mean `0.0039`, max `0.0059`
- `all_zero`: mean `0.0039`, max `0.0078`
- `random_high_weight`: mean `0.0051`, max `0.0068`

三初态之间的 cold-sector TV：

- `sector` vs `all_zero`: `0.0018`
- `sector` vs `random_high_weight`: `0.0012`
- `all_zero` vs `random_high_weight`: `0.0020`

2 倍长链复核：

- `sector, m=4096`: `q_top = 0.986654`
- `sector m=2048` vs `sector m=4096` 的 cold-sector TV: `0.0010`

top sector 分布也一致，主 sector 都是 `+++++++`，概率约 `0.994`；其余低概率 sector 的排序和概率也在采样噪声内一致。

## 这说明了什么

这不是单纯的“`q_top` 接近”。更强的证据是：

- 不同初态最后驻留在同一批 sector。
- 这些 sector 的概率几乎一样。
- 同一初态前半/后半分布稳定。
- 2 倍长链没有改变 sector 分布和 `q_top`。

因此，至少在 `L=6, p=0.05, q=0.23` 的这个 disorder 上，当前配置通过了 sector-space 热化 gate。

## 没有证明什么

本轮仍只检查了一个 disorder、一个最难点附近的 `q=0.23`。它不能单独证明整个 `L=3..6, q=0.08..0.23` 区间的 disorder average 都已经正确。

但它说明之前担心的最危险情形没有出现：即不同初态困在不同 sector、而 `q_top` 因平方平均看起来相近。

## 当前可用配置

- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=2048`
- `num_start_chains=4`
- `q_top_block_count=8`

## 下一步

为了继续省服务器时间，不应立刻全区间生产扫描。下一步建议：

1. 用同一配置在 `q=0.23` 做 `3` 个 common-disorder 的三初态 sector histogram 复核。
2. 若都通过，再只在 `q=0.08` 做一个低 q 饱和点 sanity check。
3. 通过后才进入 `L=3,4,5,6; q=0.08..0.23` 的生产扫描。

完整自动报告见：

`remote_results/014_cold_sector_summary_with_m4096.md`
