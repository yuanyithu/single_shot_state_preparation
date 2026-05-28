# 006 cold-dwell schedule probe 20260529

目的：005 显示 cluster-sector change 可到达 cold，但 cold dwell 很短，多数 arrival 错过 sector diagnostic cadence。006 不改变目标分布，只改变 swap/measurement/diagnostic 调度，区分“cold sector 真的不稳定”和“测量 cadence 抓不到 transient”。

结构：

- `launch_cold_dwell_schedule_probe_20260529.sh`: 远端启动脚本。
- `run01_stride1_m1024_seed425000/`: 基线调度，`stride=1`，尽量捕获所有 measurement 级 cold-sector transient。
- `run02_swap6_m1024_seed426000/`: `pt_swap_attempt_every_num_sweeps=6`，让 cold arrival 后平均多停留一些 local sweeps/measurement。
- `run03_measure1_m2048_seed427000/`: `num_sweeps_between_measurements=1,stride=1,m=2048`，高频测量直接检查 transient 是否进入 measured cold history。

共同物理参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。
