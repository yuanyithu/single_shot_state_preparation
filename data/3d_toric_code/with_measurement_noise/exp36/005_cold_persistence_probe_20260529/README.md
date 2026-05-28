# 005 cold-arrival persistence probe 20260529

目的：在 004 的 identity-tracked cluster cold-delivery 基础上，继续追踪 pending cluster-sector change 到达 cold 后的驻留与离开状态，解释为什么 `survived-at-arrival` 没有形成 cold-slot logical-sector flip。

结构：

- `launch_cold_persistence_probe_20260529.sh`: 远端启动脚本。
- `run01_rho015_swap1_m1024_s4_seed422000/`: 细诊断频率复跑，检查 arrival 后是否被下一次 sector diagnostic 采到。
- `run02_rho015_swap2_m1024_s4_seed423000/`: 增加 `pt_swap_sweeps_per_attempt=2`，测试温度扩散对 cold dwell/departure 状态的影响。
- `run03_rho015_swap1_m2048_s8_seed424000/`: 更长生产期，测试 arrival/persistence 统计是否积累到更稳定信号。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。
