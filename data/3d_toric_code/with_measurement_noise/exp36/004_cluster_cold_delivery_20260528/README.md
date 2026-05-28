# 004 cluster cold-delivery diagnostics 20260528

目的：用 identity-tracked 诊断量化中温 cluster-stage logical-sector change 是否被同一个 replica 带到 cold，以及到达 cold 时是 survived、reverted 还是 other。

结构：

- `launch_cluster_cold_delivery_20260528.sh`: 远端启动脚本。
- `run01_rho015_swap1_m512_seed419000/`: 003 run03 的同类复现实验，作为新诊断基线。
- `run02_rho015_swap2_m512_seed420000/`: 增加 `pt_swap_sweeps_per_attempt=2`，测试更快温度扩散是否提高 cold arrival/survival。
- `run03_rho015_swap1_m1024_seed421000/`: 更长生产期，`m=1024,stride=8`，测试 cold arrival 是否只是等待时间不足。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0`。
