# 002 cluster-stage repeats 20260528

目的：复现并分解 `q_hot=0.35 + cluster` 下出现的 cold logical-sector flip。

结构：

- `launch_cluster_stage_repeats_20260528.sh`: 远端启动脚本。
- `run01_qhot035_rho005_seed413000/`: `rho=0.05` 独立 seed 复现。
- `run02_qhot035_rho005_seed414000/`: `rho=0.05` 独立 seed 复现。
- `run03_qhot035_rho010_seed415000/`: `rho=0.10` 检查增加 cluster 预算的影响。

共同参数：`L=6,p=0.05,q=0.08,K=17,m=512,stride=4,num_start_chains=4`。
