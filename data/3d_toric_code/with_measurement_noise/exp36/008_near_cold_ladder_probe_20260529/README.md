# 008 near-cold ladder probe 20260529

目的：测试 `sync_enlarge` PT ladder 在 cold 附近加密是否能提高中温/热端 sector-changing 构型冷却回 cold 时的保留概率。

物理动机：005 证明 cluster-stage sector change 可以到达 cold，但 cold dwell 和 persistence 很弱；006 的全局降低 swap cadence 会损伤 roundtrip；007 的 cold-edge hold 仍在等待完整结果。008 不改变目标分布，只改变 PT 温度点分布：保持 cold endpoint 和 `q_hot` 不变，把 log heat scale 的 normalized index 从 `x` 改为 `x**power`。`power>1` 会在 cold/near-cold 区间放置更多温度点，在 hot 区间更稀。

程序改动：新增 `pt_ladder_spacing_power` / CLI `--pt-ladder-spacing-power`。

- 默认 `1.0` 保持旧的 uniform log heat-scale spacing。
- 仅支持 `--pt-ladder-mode sync_enlarge`。
- 生产 chunk、manifest 和 final NPZ 保存 `pt_ladder_spacing_power`。

本地验证：

- `python -m py_compile src/mcmc_diagnostics.py src/main.py src/production_chunked_scan.py src/summarize_exp36_probe.py` 通过。
- `python -m unittest discover -s tests` 通过。
- 本地 smoke：`local_ladder_spacing_smoke/`，`pt_ladder_spacing_power=2.0`，final NPZ 中保存 `pt_ladder_spacing_power=2.0`，实际 `q` ladder 为 `[0.08,0.09607933,0.15021112,0.24443533,0.35]`，确认 cold 端 log-odds gap 小于 uniform ladder。

远端计划：先只使用空闲的 nd-1/nd-2，避免挤占 nd-3 上仍在运行的 006 run03 和 007 run03。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0,num_measurements=1024,num_sweeps_between_measurements=6,pt_sector_diagnostic_stride=1,pt_cold_edge_swap_stride=1`。

| run | node | spacing power | seed | purpose |
|---|---|---:|---:|---|
| run01 | nd-1 | 1.5 | 431000 | 轻度 near-cold 加密，观察 min swap、roundtrip、arrival/persistence 是否改善 |
| run02 | nd-2 | 2.0 | 432000 | 更强 near-cold 加密，观察 hot-side 稀疏化是否损伤 transport |

判据：

- 若 `power=1.5/2.0` 增加 cluster changed 后的 cold arrival、diagnostic survived、dwell sum/max 或 cold flips，且 roundtrip/min swap 没有崩溃，则 near-cold ladder 加密是有效方向。
- 若 roundtrip 明显下降或 hot-side bottleneck 变差，同时 cold flips 仍为 0，则说明单纯重排 temperature points 不足以解决 cold persistence。
