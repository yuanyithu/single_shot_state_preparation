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
| 0.220 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.220 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.220 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.220 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.220 | 3 | 64 | 0.927366 | 0.015970 | 0.0101 | 0.0811 | 0.0132 | 0.0752 | 1 | 0.065656 | 0.0586 | 0.2381 | 0.88/4 | +++++++:1.000 (3); +++++++:0.999, +++----:0.001 (2); +++++++:0.999, +--++--:0.001 (2) |
| 0.220 | 4 | 64 | 0.991919 | 0.002074 | 0.0026 | 0.0137 | 0.0039 | 0.0264 | 0 | 0.028399 | 0.0254 | 0.1339 | 2.17/4 | +++++++:1.000 (17); +++++++:0.999, +--++--:0.001 (4); +++++++:1.000, +--++--:0.000 (4) |
| 0.220 | 5 | 64 | 0.997084 | 0.000388 | 0.0022 | 0.0068 | 0.0039 | 0.0117 | 0 | 0.015542 | 0.0137 | 0.0698 | 1.89/4 | +++++++:0.999, +--++--:0.000, --++--+:0.000 (6); +++++++:1.000 (6); +++++++:1.000, +++----:0.000 (5) |
| 0.220 | 6 | 64 | 0.994573 | 0.000587 | 0.0031 | 0.0078 | 0.0059 | 0.0146 | 0 | 0.015485 | 0.0137 | 0.0871 | 1.00/4 | +++++++:0.998, +--++--:0.001, +++----:0.000 (3); +++++++:0.996, +--++--:0.002, -+-+-+-:0.001 (2); +++++++:0.997, +--++--:0.001, -+-+-+-:0.001 (2) |

## Flagged Disorders

| q | L | disorder | observed TV | boot p99 | q_top | q_top spread | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.220 | 3 | 36 | 0.0811 | 0.0752 | 0.280845 | 0.027566 | -+-+-+-:0.445, +++++++:0.410, +--++--:0.045 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.220 | 3 | 0 | 0.683426 | 0.0293 | 0.0527 | 0.0352 | +++++++:0.845, -+-+-+-:0.068, +--++--:0.047 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 1 | 0.994429 | 0.0049 | 0.0068 | 0.0059 | +++++++:0.998, +--++--:0.001, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 2 | 0.917220 | 0.0117 | 0.0264 | 0.0195 | +++++++:0.963, +++----:0.029, -+-+-+-:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 3 | 0.678292 | 0.0234 | 0.0527 | 0.0430 | +++++++:0.842, -+-+-+-:0.092, +--++--:0.020 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 5 | 0.989995 | 0.0049 | 0.0088 | 0.0059 | +++++++:0.996, -+-+-+-:0.004, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 6 | 0.779877 | 0.0244 | 0.0439 | 0.0352 | +++++++:0.897, +++----:0.041, -+-+-+-:0.034 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 7 | 0.970750 | 0.0088 | 0.0156 | 0.0176 | +++++++:0.987, +--++--:0.011, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 8 | 0.961438 | 0.0117 | 0.0195 | 0.0137 | +++++++:0.983, -+-+-+-:0.011, +--++--:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 9 | 0.920385 | 0.0146 | 0.0264 | 0.0215 | +++++++:0.964, +--++--:0.028, +++----:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 10 | 0.904953 | 0.0117 | 0.0264 | 0.0273 | +++++++:0.957, +++----:0.026, -+-+-+-:0.016 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 11 | 0.961960 | 0.0059 | 0.0176 | 0.0098 | +++++++:0.983, -+-+-+-:0.008, +--++--:0.008 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 12 | 0.936635 | 0.0137 | 0.0234 | 0.0176 | +++++++:0.972, -+-+-+-:0.017, +--++--:0.010 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 13 | 0.922706 | 0.0156 | 0.0254 | 0.0176 | +++++++:0.965, -+-+-+-:0.019, +--++--:0.011 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 14 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 15 | 0.897506 | 0.0146 | 0.0303 | 0.0371 | +++++++:0.953, +--++--:0.037, -+-+-+-:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 16 | 0.987779 | 0.0039 | 0.0107 | 0.0039 | +++++++:0.995, +++----:0.004, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 17 | 0.901890 | 0.0117 | 0.0312 | 0.0156 | +++++++:0.955, -+-+-+-:0.039, +--++--:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 18 | 0.993874 | 0.0029 | 0.0068 | 0.0039 | +++++++:0.997, +++----:0.002, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 19 | 0.539145 | 0.0400 | 0.0654 | 0.0527 | +++++++:0.764, +++----:0.080, +--++--:0.071 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 20 | 0.997212 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, -+-+-+-:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 21 | 0.993317 | 0.0039 | 0.0078 | 0.0059 | +++++++:0.997, +++----:0.001, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 22 | 0.932095 | 0.0137 | 0.0254 | 0.0117 | +++++++:0.969, +++----:0.026, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
| 0.220 | 3 | 23 | 0.856769 | 0.0205 | 0.0352 | 0.0273 | +++++++:0.933, -+-+-+-:0.067, -+--+-+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p22_L3456_d64_m1024_seed518000/run_q0p22_L3456_d64_m1024_seed518000.npz` |
