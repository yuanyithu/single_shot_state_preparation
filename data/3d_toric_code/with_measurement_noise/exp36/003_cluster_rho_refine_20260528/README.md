# 003 cluster-rho refinement 20260528

目的：验证 `q_hot=0.35, cluster rho=0.10` 的 cold logical-sector flip 是否可复现，并小幅测试 `rho=0.15` 的收益与成本。

结构：

- `launch_cluster_rho_refine_20260528.sh`: 远端启动脚本。
- `run01_qhot035_rho010_seed416000/`: `rho=0.10` 独立 seed 复现。
- `run02_qhot035_rho010_seed417000/`: `rho=0.10` 第二个独立 seed 复现。
- `run03_qhot035_rho015_seed418000/`: `rho=0.15` 检查增加 cluster 预算是否继续提高 sector flips。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,m=512,stride=4,num_start_chains=4,adaptive_pt_rounds=0`。
