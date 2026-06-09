# 算法验证：L=2 精确枚举 vs MCMC-TI（回答「算法是不是错了」）

**问题**：008 发现 L=3,4,5 的 Δf gap 在 p=0.25(>渐近 p_c≈0.23)仍有序、无 crossing，怀疑算法错。

**验证**：在 L=2（24 qubit，可精确枚举所有 2^24 构型）对同一组 disorder（matched by `disorder_seed`，scope `lattice_size,disorder_index`，seed_base 840000）比较**精确 Δf** 与 **MCMC-TI Δf**（`run --projection-mode linear`，grid129/m4096）。q=0.002（q→0 代理，K_q≈6.2；q=0 不被支持因 K_q=log((1-q)/q) 发散）。

| p | exact Δf gap | MCMC Δf gap | \|Δ\|gap | max\|Δf 向量差\| | exact q_top | MCMC q_top |
|---|---|---|---|---|---|---|
| 0.15 | 6.234 | 6.225 | 0.009 | 0.019 | 0.9866 | 0.9866 |
| 0.23 | 4.053 | 4.049 | 0.004 | 0.008 | 0.8863 | 0.8861 |
| 0.30 | 2.396 | 2.398 | 0.004 | 0.011 | 0.5221 | 0.5234 |

（每 p 4/4 disorder 按 seed 匹配。）

**结论：MCMC-TI 精确复现精确枚举 Δf（误差 ~0.01 = 采样噪声），q_top 误差 ~0.001 → 算法实现正确，不是 bug。**

**那 008 的「p=0.25 仍有序」是什么？** 纯**有限尺寸效应**。L=2 的 Δf gap 随 p 单调下降（6.23→4.05→2.40 over p=0.15→0.30）、q_top 也降（0.99→0.89→0.52），即小系统确实随 p 增大而失序——但**有限尺寸 crossing(不同 L 曲线交点)落在远高于渐近 p_c≈0.23 的位置**；小系统在渐近阈值之上仍保有可观逻辑保护(Δf 随 L 增长)。要得到真 p_c 需 **FSS(L→∞ 外推)**，L=3,4,5 单点 crossing 系统性高估。这与 007(Δf q_c≈0.05 vs q_top≈0.03)同源。

证据目录：`exact_L2/{p15,p23,p30}/exact_sector_weights.json`、`mcmc_L2/{p15,p23,p30}/sector_ti_results.npz`、比较脚本 `/tmp/compare_L2.py`(逻辑已记于此)。
