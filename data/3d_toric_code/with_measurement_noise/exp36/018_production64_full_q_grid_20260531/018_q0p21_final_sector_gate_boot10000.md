# exp36 sector histogram gate

## Gate Definition

- For each fixed disorder, compare cold-chain sector histograms from different initial states.
- The statistic is the maximum pairwise total-variation distance between start-chain histograms.
- The reference scale is a parametric bootstrap from the pooled sector histogram with the same per-chain sample counts.
- A disorder is flagged when observed max TV is larger than the bootstrap p99 plus the configured epsilon.

- bootstrap replicates: 10000
- TV epsilon: 1e-12

## Initial-State Coverage

| q | L | compared chains | logical-sector start coverage | start labels |
|---:|---:|---:|---:|---|
| 0.210 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.210 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.210 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.210 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.210 | 3 | 64 | 0.926721 | 0.016733 | 0.0101 | 0.0596 | 0.0137 | 0.0811 | 0 | 0.066757 | 0.0547 | 0.2606 | 0.89/4 | +++++++:1.000, +--++--:0.000 (4); +++++++:0.999, +++----:0.001, +--++--:0.000 (2); +++++++:1.000, +--++--:0.000, +++----:0.000 (2) |
| 0.210 | 4 | 64 | 0.992422 | 0.002023 | 0.0025 | 0.0137 | 0.0034 | 0.0273 | 0 | 0.028909 | 0.0176 | 0.1494 | 2.44/4 | +++++++:1.000 (19); +++++++:0.999, +++----:0.001, -+-+-+-:0.000 (3); +++++++:1.000, +++----:0.000 (3) |
| 0.210 | 5 | 64 | 0.996903 | 0.000482 | 0.0022 | 0.0078 | 0.0039 | 0.0146 | 0 | 0.015485 | 0.0117 | 0.0699 | 1.91/4 | +++++++:1.000 (9); +++++++:1.000, +--++--:0.000 (4); +++++++:0.999, +--++--:0.000, --++--+:0.000 (3) |
| 0.210 | 6 | 64 | 0.995172 | 0.000544 | 0.0031 | 0.0098 | 0.0059 | 0.0146 | 0 | 0.017683 | 0.0137 | 0.0700 | 1.25/4 | +++++++:0.999, +--++--:0.000, -+-+-+-:0.000 (4); +++++++:0.998, -+-+-+-:0.001, +--++--:0.001 (3); +++++++:0.998, +++----:0.001, -+-+-+-:0.000 (2) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.210 | 3 | 0 | 0.660939 | 0.0127 | 0.0527 | 0.0547 | +++++++:0.832, -+-+-+-:0.079, +--++--:0.061 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 1 | 0.995542 | 0.0049 | 0.0059 | 0.0059 | +++++++:0.998, +--++--:0.001, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 2 | 0.928907 | 0.0127 | 0.0254 | 0.0176 | +++++++:0.968, +++----:0.027, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 3 | 0.786097 | 0.0264 | 0.0449 | 0.0234 | +++++++:0.898, -+-+-+-:0.076, --++--+:0.014 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 4 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 5 | 0.993873 | 0.0020 | 0.0078 | 0.0059 | +++++++:0.997, +++----:0.001, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 6 | 0.831245 | 0.0117 | 0.0381 | 0.0254 | +++++++:0.922, -+-+-+-:0.031, +++----:0.028 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 7 | 0.979525 | 0.0039 | 0.0137 | 0.0137 | +++++++:0.991, +--++--:0.008, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 8 | 0.971797 | 0.0068 | 0.0156 | 0.0137 | +++++++:0.988, -+-+-+-:0.008, +--++--:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 9 | 0.898596 | 0.0176 | 0.0303 | 0.0215 | +++++++:0.954, +--++--:0.037, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 10 | 0.919695 | 0.0166 | 0.0264 | 0.0215 | +++++++:0.964, +++----:0.024, -+-+-+-:0.011 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 11 | 0.977839 | 0.0068 | 0.0137 | 0.0098 | +++++++:0.990, +--++--:0.006, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 12 | 0.951891 | 0.0078 | 0.0205 | 0.0078 | +++++++:0.979, -+-+-+-:0.020, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 13 | 0.891454 | 0.0117 | 0.0312 | 0.0156 | +++++++:0.951, -+-+-+-:0.032, +--++--:0.013 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 15 | 0.872261 | 0.0215 | 0.0332 | 0.0430 | +++++++:0.941, +--++--:0.042, -+-+-+-:0.013 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 16 | 0.992764 | 0.0039 | 0.0078 | 0.0078 | +++++++:0.997, +++----:0.002, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 17 | 0.940473 | 0.0176 | 0.0225 | 0.0176 | +++++++:0.973, -+-+-+-:0.020, +--++--:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 18 | 0.997769 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, +++----:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 19 | 0.474400 | 0.0391 | 0.0674 | 0.0547 | +++++++:0.722, +--++--:0.090, +++----:0.087 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 20 | 0.997770 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, +++----:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 21 | 0.979513 | 0.0098 | 0.0137 | 0.0078 | +++++++:0.991, +++----:0.008, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 22 | 0.899506 | 0.0098 | 0.0293 | 0.0215 | +++++++:0.954, +++----:0.042, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
| 0.210 | 3 | 23 | 0.796305 | 0.0225 | 0.0420 | 0.0273 | +++++++:0.901, -+-+-+-:0.096, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p21_L3456_d64_m1024_seed518000/run_q0p21_L3456_d64_m1024_seed518000.npz` |
