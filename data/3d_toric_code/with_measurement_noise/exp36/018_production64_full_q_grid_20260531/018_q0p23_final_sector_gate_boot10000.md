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
| 0.230 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.230 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.230 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.230 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.230 | 3 | 64 | 0.923398 | 0.017068 | 0.0099 | 0.0479 | 0.0132 | 0.0791 | 1 | 0.072726 | 0.0938 | 0.2886 | 0.83/4 | +++++++:1.000 (5); +++++++:0.710, -+-+-+-:0.202, --++--+:0.050 (1); +++++++:0.726, +--++--:0.114, +++----:0.077 (1) |
| 0.230 | 4 | 64 | 0.989885 | 0.003125 | 0.0030 | 0.0195 | 0.0044 | 0.0391 | 0 | 0.036497 | 0.0215 | 0.1756 | 2.09/4 | +++++++:1.000 (12); +++++++:1.000, +--++--:0.000 (5); +++++++:1.000, +++----:0.000 (3) |
| 0.230 | 5 | 64 | 0.997205 | 0.000326 | 0.0021 | 0.0049 | 0.0049 | 0.0098 | 0 | 0.011121 | 0.0078 | 0.0527 | 1.83/4 | +++++++:1.000, +--++--:0.000 (7); +++++++:1.000 (5); +++++++:0.998, +--++--:0.001, -+-+-+-:0.000 (3) |
| 0.230 | 6 | 64 | 0.994312 | 0.000523 | 0.0035 | 0.0078 | 0.0063 | 0.0137 | 0 | 0.017755 | 0.0137 | 0.0872 | 0.94/4 | +++++++:0.999, +--++--:0.000, --++--+:0.000 (3); +++++++:0.997, +--++--:0.001, +++----:0.001 (2); +++++++:0.998, +--++--:0.001, +++----:0.000 (2) |

## Flagged Disorders

| q | L | disorder | observed TV | boot p99 | q_top | q_top spread | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.230 | 3 | 15 | 0.0332 | 0.0312 | 0.892532 | 0.067668 | +++++++:0.951, +--++--:0.041, -+-+-+-:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.230 | 3 | 0 | 0.484821 | 0.0312 | 0.0664 | 0.0586 | +++++++:0.710, -+-+-+-:0.202, --++--+:0.050 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 1 | 0.993317 | 0.0029 | 0.0078 | 0.0039 | +++++++:0.997, +--++--:0.001, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 2 | 0.958130 | 0.0068 | 0.0186 | 0.0098 | +++++++:0.981, +++----:0.008, -+-+-+-:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 3 | 0.699887 | 0.0322 | 0.0527 | 0.0449 | +++++++:0.854, -+-+-+-:0.085, +++----:0.018 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 5 | 0.983348 | 0.0039 | 0.0117 | 0.0078 | +++++++:0.993, -+-+-+-:0.004, +--++--:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 6 | 0.732040 | 0.0449 | 0.0488 | 0.0371 | +++++++:0.872, -+-+-+-:0.059, +++----:0.037 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 7 | 0.982814 | 0.0049 | 0.0117 | 0.0078 | +++++++:0.992, +--++--:0.006, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 8 | 0.951140 | 0.0098 | 0.0205 | 0.0176 | +++++++:0.978, -+-+-+-:0.015, +--++--:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 9 | 0.939352 | 0.0107 | 0.0234 | 0.0176 | +++++++:0.973, +--++--:0.019, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 10 | 0.908626 | 0.0107 | 0.0283 | 0.0273 | +++++++:0.959, +++----:0.027, -+-+-+-:0.012 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 11 | 0.900607 | 0.0078 | 0.0303 | 0.0254 | +++++++:0.955, +--++--:0.036, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 12 | 0.965803 | 0.0107 | 0.0176 | 0.0098 | +++++++:0.985, -+-+-+-:0.010, +--++--:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 13 | 0.929151 | 0.0146 | 0.0254 | 0.0156 | +++++++:0.968, -+-+-+-:0.019, +--++--:0.010 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 15 | 0.892532 | 0.0332 | 0.0312 | 0.0195 | +++++++:0.951, +--++--:0.041, -+-+-+-:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 16 | 0.983358 | 0.0078 | 0.0117 | 0.0059 | +++++++:0.993, +++----:0.006, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 17 | 0.922394 | 0.0078 | 0.0264 | 0.0137 | +++++++:0.965, -+-+-+-:0.026, +--++--:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 18 | 0.994986 | 0.0020 | 0.0068 | 0.0039 | +++++++:0.998, +--++--:0.001, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 19 | 0.482383 | 0.0264 | 0.0684 | 0.0938 | +++++++:0.726, +--++--:0.114, +++----:0.077 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 20 | 0.997212 | 0.0029 | 0.0049 | 0.0039 | +++++++:0.999, +--++--:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 21 | 0.991097 | 0.0049 | 0.0088 | 0.0078 | +++++++:0.996, -+-+-+-:0.002, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 22 | 0.915173 | 0.0049 | 0.0273 | 0.0234 | +++++++:0.962, +++----:0.030, -+-+-+-:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
| 0.230 | 3 | 23 | 0.828545 | 0.0205 | 0.0381 | 0.0410 | +++++++:0.918, -+-+-+-:0.080, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p23_L3456_d64_m1024_seed518000/run_q0p23_L3456_d64_m1024_seed518000.npz` |
