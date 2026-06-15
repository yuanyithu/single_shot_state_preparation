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
| 0.120 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.120 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.120 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.120 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.120 | 3 | 64 | 0.963272 | 0.013036 | 0.0046 | 0.0391 | 0.0010 | 0.0732 | 0 | 0.072558 | 0.0488 | 0.2314 | 2.38/4 | +++++++:1.000 (32); +++++++:1.000, +--++--:0.000, +++----:0.000 (2); +++++++:1.000, -+-+-+-:0.000 (2) |
| 0.120 | 4 | 64 | 0.998318 | 0.000826 | 0.0007 | 0.0098 | 0.0000 | 0.0186 | 0 | 0.021844 | 0.0137 | 0.1182 | 3.38/4 | +++++++:1.000 (45); +++++++:1.000, +--++--:0.000 (5); +++++++:0.999, +--++--:0.001 (3) |
| 0.120 | 5 | 64 | 0.999173 | 0.000192 | 0.0008 | 0.0059 | 0.0000 | 0.0098 | 0 | 0.013327 | 0.0059 | 0.0353 | 3.31/4 | +++++++:1.000 (36); +++++++:1.000, +--++--:0.000 (6); +++++++:0.999, +--++--:0.001 (2) |
| 0.120 | 6 | 64 | 0.997731 | 0.000457 | 0.0015 | 0.0059 | 0.0024 | 0.0127 | 0 | 0.013338 | 0.0078 | 0.0692 | 2.59/4 | +++++++:1.000 (26); +++++++:1.000, +--++--:0.000 (8); +++++++:0.999, +--++--:0.001, -+-+-+-:0.000 (3) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.120 | 3 | 0 | 0.969581 | 0.0059 | 0.0156 | 0.0117 | +++++++:0.987, --++--+:0.006, +--++--:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 2 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 3 | 0.984999 | 0.0029 | 0.0117 | 0.0137 | +++++++:0.993, -+-+-+-:0.003, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 5 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 6 | 0.883125 | 0.0127 | 0.0303 | 0.0254 | +++++++:0.947, +--++--:0.028, -+-+-+-:0.024 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 7 | 0.950780 | 0.0127 | 0.0215 | 0.0195 | +++++++:0.978, +--++--:0.020, --++--+:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 8 | 0.946608 | 0.0088 | 0.0215 | 0.0059 | +++++++:0.976, -+-+-+-:0.024, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 9 | 0.989442 | 0.0020 | 0.0098 | 0.0078 | +++++++:0.995, +--++--:0.004, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 10 | 0.975089 | 0.0098 | 0.0146 | 0.0078 | +++++++:0.989, +++----:0.006, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 11 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 12 | 0.963173 | 0.0078 | 0.0176 | 0.0117 | +++++++:0.984, -+-+-+-:0.015, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 13 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 15 | 0.724733 | 0.0391 | 0.0459 | 0.0449 | +++++++:0.867, +--++--:0.073, --++--+:0.034 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 16 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 17 | 0.998884 | 0.0020 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 19 | 0.507602 | 0.0303 | 0.0625 | 0.0488 | +++++++:0.742, +--++--:0.102, +----++:0.066 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 21 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 22 | 0.967489 | 0.0059 | 0.0166 | 0.0117 | +++++++:0.986, +++----:0.012, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
| 0.120 | 3 | 23 | 0.908025 | 0.0146 | 0.0274 | 0.0234 | +++++++:0.958, -+-+-+-:0.042, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p12_L3456_d64_m1024_seed518000/run_q0p12_L3456_d64_m1024_seed518000.npz` |
