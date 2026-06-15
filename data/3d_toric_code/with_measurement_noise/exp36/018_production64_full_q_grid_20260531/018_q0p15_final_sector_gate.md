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
| 0.150 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.150 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.150 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.150 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.150 | 3 | 64 | 0.953976 | 0.014912 | 0.0064 | 0.0488 | 0.0063 | 0.0791 | 0 | 0.056547 | 0.0586 | 0.2585 | 1.88/4 | +++++++:1.000 (19); +++++++:1.000, +++----:0.000 (5); +++++++:0.999, +++----:0.001 (2) |
| 0.150 | 4 | 64 | 0.995840 | 0.001654 | 0.0014 | 0.0117 | 0.0020 | 0.0273 | 0 | 0.025942 | 0.0195 | 0.1514 | 2.98/4 | +++++++:1.000 (27); +++++++:1.000, +--++--:0.000 (5); +++++++:1.000, -+-+-+-:0.000 (4) |
| 0.150 | 5 | 64 | 0.998633 | 0.000220 | 0.0011 | 0.0039 | 0.0020 | 0.0078 | 0 | 0.008905 | 0.0059 | 0.0353 | 2.70/4 | +++++++:1.000 (23); +++++++:1.000, +--++--:0.000 (6); +++++++:1.000, -+-+-+-:0.000 (5) |
| 0.150 | 6 | 64 | 0.996824 | 0.000400 | 0.0023 | 0.0088 | 0.0039 | 0.0107 | 0 | 0.017718 | 0.0078 | 0.0527 | 1.83/4 | +++++++:1.000 (7); +++++++:1.000, +--++--:0.000 (5); +++++++:1.000, -+-+-+-:0.000 (5) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.150 | 3 | 0 | 0.947277 | 0.0078 | 0.0225 | 0.0117 | +++++++:0.977, +--++--:0.011, --++--+:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 1 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 2 | 0.994430 | 0.0020 | 0.0068 | 0.0059 | +++++++:0.998, +++----:0.002, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 3 | 0.963566 | 0.0068 | 0.0176 | 0.0156 | +++++++:0.984, -+-+-+-:0.008, +++----:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 5 | 0.997769 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, -+-+-+-:0.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 6 | 0.859180 | 0.0205 | 0.0352 | 0.0234 | +++++++:0.935, -+-+-+-:0.033, +--++--:0.030 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 7 | 0.970243 | 0.0068 | 0.0166 | 0.0078 | +++++++:0.987, +--++--:0.012, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 8 | 0.986118 | 0.0049 | 0.0107 | 0.0098 | +++++++:0.994, -+-+-+-:0.005, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 9 | 0.915562 | 0.0107 | 0.0264 | 0.0254 | +++++++:0.962, +--++--:0.036, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 10 | 0.969077 | 0.0059 | 0.0166 | 0.0098 | +++++++:0.986, +++----:0.009, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 11 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 12 | 0.845166 | 0.0107 | 0.0361 | 0.0254 | +++++++:0.927, -+-+-+-:0.072, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 13 | 0.991657 | 0.0068 | 0.0078 | 0.0078 | +++++++:0.996, +--++--:0.003, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 15 | 0.721944 | 0.0303 | 0.0469 | 0.0488 | +++++++:0.865, +--++--:0.092, -+-+-+-:0.023 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 16 | 0.997770 | 0.0029 | 0.0049 | 0.0020 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 17 | 0.982820 | 0.0049 | 0.0117 | 0.0078 | +++++++:0.992, -+-+-+-:0.007, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 19 | 0.550406 | 0.0469 | 0.0615 | 0.0586 | +++++++:0.770, +--++--:0.095, +++----:0.047 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 21 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 22 | 0.889049 | 0.0215 | 0.0312 | 0.0176 | +++++++:0.949, +++----:0.049, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
| 0.150 | 3 | 23 | 0.914159 | 0.0176 | 0.0283 | 0.0156 | +++++++:0.961, -+-+-+-:0.039, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p15_L3456_d64_m1024_seed518000/run_q0p15_L3456_d64_m1024_seed518000.npz` |
