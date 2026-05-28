# 007 cold-edge hold probe

目标：测试只降低 cold edge `(k=0,k=1)` 的 PT swap 频率，是否能增加已到达 cold 的 sector-changing replica 的驻留时间和可诊断 persistence。

共同物理参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。

程序改动：新增 `--pt-cold-edge-swap-stride N`。当 alternating swap sweep 本来会尝试 pair `0-1` 时，只每 `N` 次 eligible sweep 尝试一次；其它温度对保持原 cadence。`N=1` 保持旧行为。该调度固定且与状态无关，因此只改变 kernel 调度频率，不改变目标分布。

远端计划：

| run | node | cold-edge stride | measurements | sweeps/meas | diag stride | seed | purpose |
|---|---|---:|---:|---:|---:|---:|---|
| run01 | nd-1 | 1 | 1024 | 6 | 1 | 428000 | 新代码基线，对照 006 run01 |
| run02 | nd-2 | 2 | 1024 | 6 | 1 | 429000 | 轻度增加 cold dwell |
| run03 | nd-3 | 4 | 1024 | 6 | 1 | 430000 | 更强 cold hold，观察 transport 损伤 |

判据：若 stride `2/4` 明显增加 cold dwell sum/max、diagnostic survived 和 cold flips，同时 roundtrip 没有崩溃，则说明 cold-edge dwell 是有效调度方向。若 roundtrip 大幅下降且 cold flips 仍为 0，则说明简单 hold 不能解决 cold persistence。
