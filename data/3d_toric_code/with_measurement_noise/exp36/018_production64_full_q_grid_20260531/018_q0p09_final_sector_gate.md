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
| 0.090 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.090 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.090 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.090 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.090 | 3 | 64 | 0.974216 | 0.012258 | 0.0034 | 0.0684 | 0.0000 | 0.0732 | 0 | 0.063677 | 0.0547 | 0.2651 | 2.97/4 | +++++++:1.000 (40); +++++++:0.996, -+-+-+-:0.004 (2); +++++++:0.999, +--++--:0.001, +++----:0.000 (2) |
| 0.090 | 4 | 64 | 0.999138 | 0.000667 | 0.0003 | 0.0078 | 0.0000 | 0.0186 | 0 | 0.017347 | 0.0195 | 0.1032 | 3.86/4 | +++++++:1.000 (58); +++++++:1.000, +++----:0.000 (2); +++++++:0.982, +--++--:0.010, -+-+-+-:0.008 (1) |
| 0.090 | 5 | 64 | 0.999800 | 0.000063 | 0.0003 | 0.0020 | 0.0000 | 0.0049 | 0 | 0.004458 | 0.0039 | 0.0352 | 3.75/4 | +++++++:1.000 (51); +++++++:1.000, +--++--:0.000 (3); +++++++:1.000, -+-+-+-:0.000 (3) |
| 0.090 | 6 | 64 | 0.999260 | 0.000215 | 0.0006 | 0.0049 | 0.0000 | 0.0098 | 0 | 0.011115 | 0.0059 | 0.0526 | 3.33/4 | +++++++:1.000 (42); +++++++:1.000, -+-+-+-:0.000 (5); +++++++:1.000, +--++--:0.000 (4) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.090 | 3 | 0 | 0.988878 | 0.0059 | 0.0098 | 0.0137 | +++++++:0.995, -+-+-+-:0.002, --++--+:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 2 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 3 | 0.999442 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 5 | 0.999442 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 6 | 0.886256 | 0.0234 | 0.0312 | 0.0176 | +++++++:0.948, -+-+-+-:0.030, +--++--:0.020 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 7 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 8 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 9 | 0.991098 | 0.0039 | 0.0088 | 0.0039 | +++++++:0.996, +--++--:0.002, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 10 | 0.997212 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, +++----:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 11 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 12 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 13 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 15 | 0.676029 | 0.0273 | 0.0527 | 0.0449 | +++++++:0.839, +--++--:0.101, --++--+:0.033 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 16 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 17 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 19 | 0.920478 | 0.0156 | 0.0264 | 0.0234 | +++++++:0.964, +--++--:0.018, -+-+-+-:0.009 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 21 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 22 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
| 0.090 | 3 | 23 | 0.995544 | 0.0029 | 0.0059 | 0.0039 | +++++++:0.998, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p09_L3456_d64_m1024_seed518000/run_q0p09_L3456_d64_m1024_seed518000.npz` |
