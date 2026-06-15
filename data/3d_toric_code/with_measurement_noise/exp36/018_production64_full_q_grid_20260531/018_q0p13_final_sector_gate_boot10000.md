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
| 0.130 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.130 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.130 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.130 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.130 | 3 | 64 | 0.955150 | 0.015856 | 0.0056 | 0.0459 | 0.0034 | 0.0791 | 1 | 0.079274 | 0.0586 | 0.2733 | 2.31/4 | +++++++:1.000 (26); +++++++:0.999, -+-+-+-:0.001 (3); +++++++:1.000, -+-+-+-:0.000 (3) |
| 0.130 | 4 | 64 | 0.997734 | 0.000961 | 0.0011 | 0.0156 | 0.0000 | 0.0215 | 0 | 0.031845 | 0.0098 | 0.0988 | 3.25/4 | +++++++:1.000 (39); +++++++:1.000, +--++--:0.000 (6); +++++++:0.999, +--++--:0.001 (3) |
| 0.130 | 5 | 64 | 0.998625 | 0.000331 | 0.0011 | 0.0098 | 0.0020 | 0.0137 | 0 | 0.022053 | 0.0098 | 0.0699 | 3.00/4 | +++++++:1.000 (25); +++++++:1.000, -+-+-+-:0.000 (6); +++++++:1.000, +--++--:0.000 (5) |
| 0.130 | 6 | 64 | 0.997150 | 0.000530 | 0.0017 | 0.0088 | 0.0039 | 0.0137 | 0 | 0.019841 | 0.0117 | 0.1201 | 2.23/4 | +++++++:1.000 (16); +++++++:1.000, +--++--:0.000 (6); +++++++:0.999, -+-+-+-:0.001, +--++--:0.000 (3) |

## Flagged Disorders

| q | L | disorder | observed TV | boot p99 | q_top | q_top spread | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.130 | 3 | 52 | 0.0400 | 0.0342 | 0.865445 | 0.079274 | +++++++:0.937, +--++--:0.062, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.130 | 3 | 0 | 0.965205 | 0.0068 | 0.0176 | 0.0215 | +++++++:0.985, +--++--:0.007, --++--+:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 2 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 3 | 0.976178 | 0.0107 | 0.0146 | 0.0156 | +++++++:0.990, -+-+-+-:0.005, +++----:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 5 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 6 | 0.858776 | 0.0234 | 0.0352 | 0.0273 | +++++++:0.935, +--++--:0.035, -+-+-+-:0.029 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 7 | 0.957281 | 0.0068 | 0.0195 | 0.0156 | +++++++:0.981, +--++--:0.019, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 8 | 0.993876 | 0.0020 | 0.0068 | 0.0039 | +++++++:0.997, -+-+-+-:0.002, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 9 | 0.994989 | 0.0020 | 0.0068 | 0.0039 | +++++++:0.998, +--++--:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 10 | 0.975647 | 0.0049 | 0.0146 | 0.0137 | +++++++:0.989, +++----:0.007, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 11 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 12 | 0.960490 | 0.0117 | 0.0186 | 0.0117 | +++++++:0.982, -+-+-+-:0.017, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 13 | 0.998884 | 0.0020 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 15 | 0.646365 | 0.0322 | 0.0547 | 0.0449 | +++++++:0.821, +--++--:0.124, --++--+:0.031 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 16 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 17 | 0.998327 | 0.0020 | 0.0039 | 0.0020 | +++++++:0.999, +--++--:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 19 | 0.480265 | 0.0400 | 0.0674 | 0.0508 | +++++++:0.719, +--++--:0.154, -+-+-+-:0.045 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 21 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 22 | 0.902901 | 0.0078 | 0.0293 | 0.0215 | +++++++:0.956, +++----:0.044, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
| 0.130 | 3 | 23 | 0.794554 | 0.0166 | 0.0410 | 0.0176 | +++++++:0.900, -+-+-+-:0.100 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p13_L3456_d64_m1024_seed518000/run_q0p13_L3456_d64_m1024_seed518000.npz` |
