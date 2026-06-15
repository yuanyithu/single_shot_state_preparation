# 006 cold-dwell schedule probe summary

共同参数：`lattice_size=6,p=0.05,q=0.08,q_hot=0.35,num_temperatures=17,num_start_chains=4,cluster_budget_fraction_rho=0.15`。

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_stride1_m1024_seed425000 | 1 | 1 | 6 | 1024 | 1 | 0.117131 | [0, 0, 0, 0] | 151 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run02_swap6_m1024_seed426000 | 1 | 6 | 6 | 1024 | 1 | 0.121739 | [0, 0, 0, 0] | 30 | 4 | 2 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 4/2 |
| run03_measure1_m2048_seed427000 | 1 | 1 | 1 | 2048 | 1 | 0.141708 | [0, 2, 0, 0] | 38 | 6 | 6 | 1 | 4 | 1 | 0 | 1 | 4 | 1 | 12/2 |
