# G4.2 性能 profile（direct + PT，D4 后的大 k 生产方法）

> **PRE_ALIGNMENT（scan/physics v1）：** 本页只提供旧实现的历史性能量级；runner 可能依赖弃用字段、估计器或 schema，不认证 `exp101.physics.v2` 正确性或新 PT 成本。见 `../README.md`。

config: direct 8 起点×(burn 2000+meas 8000); PT 8 温×200 轮；(p=0.05,q=0.03) engine=numba

| m | n | k | direct/disorder(8起点,numba) | PT 生产/disorder(python,外推) | q_top自检 |
|---|---|---|---|---|---|
| 2 | 100 | 4 | 0.1s | 33s | 0.298 |
| 3 | 225 | 9 | 0.3s | 75s | 0.608 |
| 4 | 400 | 16 | 0.5s | 133s | 0.440 |
| 5 | 625 | 25 | 0.8s | 209s | 0.274 |
| 6 | 900 | 36 | 1.1s | 302s | 0.015 |

**结论**：
1. **direct 引擎 numba 极快**：m=6(n=900,k=36) 仅 1.1s/disorder（8 起点）；q_top 采样自检非平凡=真在采样。numba 生效。
2. **PT 是纯 python（未 numba），大 m 慢**：m=6 生产等效 PT≈302s/disorder（外推 10000 轮×8 温）。这是 crossing/冷区 sector 传输的成本驱动。
3. **可行性**：3D L=7 sector-TI 生产为 6090s/disorder（既已可行）。expander direct 远低于此；PT 即便 python 也同量级内，且 disorder 级跨 80/80/96 核 3 节点并行（run_scan --num-workers）⇒ **exp102 生产可行**。

**生产前 TODO（新增）**：若 PT 成为大 m 瓶颈，(a) 把 run_parallel_tempering 内循环 numba 化（对齐 fast_mcmc kernel），或 (b) 用 decoder-informed 初始化（起点近 φ(η) 真类，减少对传输的依赖）。先按 python-PT 起量，瓶颈显现再优化。
