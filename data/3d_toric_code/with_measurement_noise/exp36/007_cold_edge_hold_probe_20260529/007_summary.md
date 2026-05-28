# 007 cold-edge hold probe summary

共同参数：`lattice_size=6,p=0.05,q=0.08,q_hot=0.35,num_temperatures=17,num_start_chains=4,cluster_budget_fraction_rho=0.15`。

| run | cold edge | swap every | sweeps/meas | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01_coldedge1_m1024_seed428000 | 1 | 1 | 6 | 1024 | 1 | 0.142660 | [0, 0, 0, 0] | 159 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run02_coldedge2_m1024_seed429000 | 2 | 1 | 6 | 1024 | 1 | 0.120177 | [0, 0, 0, 0] | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run03_coldedge4_m1024_seed430000 | 4 | 1 | 6 | 1024 | 1 | 0.115535 | [0, 0, 0, 0] | 144 | 58 | 33 | 9 | 11 | 9 | 0 | 9 | 11 | 13 | 400/40 |
