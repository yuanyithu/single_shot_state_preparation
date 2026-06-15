# exp36 sector histogram gate

## Gate Definition

- For each fixed disorder, compare cold-chain sector histograms from different initial states.
- The statistic is the maximum pairwise total-variation distance between start-chain histograms.
- The reference scale is a parametric bootstrap from the pooled sector histogram with the same per-chain sample counts.
- A disorder is flagged when observed max TV is larger than the bootstrap p99 plus the configured epsilon.

- bootstrap replicates: 1000
- TV epsilon: 1e-12

## Initial-State Coverage

| q | L | compared chains | logical-sector start coverage | start labels |
|---:|---:|---:|---:|---|
| 0.140 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.140 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.140 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.140 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.140 | 3 | 64 | 0.951541 | 0.015360 | 0.0055 | 0.0381 | 0.0049 | 0.0791 | 0 | 0.057757 | 0.0879 | 0.2218 | 1.80/4 | +++++++:1.000 (15); +++++++:1.000, -+-+-+-:0.000 (5); +++++++:0.999, +--++--:0.001 (3) |
| 0.140 | 4 | 64 | 0.996210 | 0.001685 | 0.0012 | 0.0098 | 0.0000 | 0.0293 | 0 | 0.019466 | 0.0254 | 0.1458 | 2.95/4 | +++++++:1.000 (36); +++++++:0.998, +--++--:0.002 (3); +++++++:1.000, +--++--:0.000 (2) |
| 0.140 | 5 | 64 | 0.998850 | 0.000216 | 0.0010 | 0.0049 | 0.0020 | 0.0078 | 0 | 0.008905 | 0.0078 | 0.0526 | 2.89/4 | +++++++:1.000 (30); +++++++:1.000, -+-+-+-:0.000 (7); +++++++:1.000, +--++--:0.000 (3) |
| 0.140 | 6 | 64 | 0.997400 | 0.000453 | 0.0019 | 0.0068 | 0.0039 | 0.0137 | 0 | 0.015538 | 0.0137 | 0.0527 | 2.17/4 | +++++++:1.000 (15); +++++++:1.000, +--++--:0.000 (4); +++++++:1.000, +----++:0.000 (3) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.140 | 3 | 0 | 0.964114 | 0.0127 | 0.0176 | 0.0117 | +++++++:0.984, -+-+-+-:0.006, +--++--:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 2 | 0.998327 | 0.0010 | 0.0039 | 0.0020 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 3 | 0.973423 | 0.0059 | 0.0156 | 0.0156 | +++++++:0.988, -+-+-+-:0.006, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 5 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, --++--+:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 6 | 0.865264 | 0.0176 | 0.0332 | 0.0254 | +++++++:0.938, +--++--:0.032, -+-+-+-:0.028 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 7 | 0.967527 | 0.0049 | 0.0176 | 0.0059 | +++++++:0.986, +--++--:0.014, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 8 | 0.946090 | 0.0049 | 0.0205 | 0.0195 | +++++++:0.976, -+-+-+-:0.024 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 9 | 0.907379 | 0.0049 | 0.0274 | 0.0137 | +++++++:0.958, +--++--:0.041, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 10 | 0.975639 | 0.0078 | 0.0146 | 0.0117 | +++++++:0.989, +++----:0.005, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 11 | 0.999442 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 12 | 0.808325 | 0.0117 | 0.0391 | 0.0137 | +++++++:0.908, -+-+-+-:0.091, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 13 | 0.997770 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 14 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 15 | 0.688733 | 0.0225 | 0.0498 | 0.0312 | +++++++:0.845, +--++--:0.113, -+-+-+-:0.024 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 16 | 0.998327 | 0.0010 | 0.0039 | 0.0020 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 17 | 0.997213 | 0.0029 | 0.0049 | 0.0020 | +++++++:0.999, +--++--:0.001, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 19 | 0.511945 | 0.0342 | 0.0654 | 0.0625 | +++++++:0.743, +--++--:0.122, +++----:0.051 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 21 | 0.999442 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 22 | 0.892198 | 0.0127 | 0.0303 | 0.0117 | +++++++:0.950, +++----:0.048, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
| 0.140 | 3 | 23 | 0.881248 | 0.0127 | 0.0312 | 0.0312 | +++++++:0.945, -+-+-+-:0.054, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p14_L3456_d64_m1024_seed518000/run_q0p14_L3456_d64_m1024_seed518000.npz` |
