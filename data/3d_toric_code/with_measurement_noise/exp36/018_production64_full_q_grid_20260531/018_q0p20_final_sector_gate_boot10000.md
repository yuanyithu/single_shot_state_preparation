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
| 0.200 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.200 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.200 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.200 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.200 | 3 | 64 | 0.937058 | 0.014996 | 0.0092 | 0.0605 | 0.0098 | 0.0801 | 0 | 0.063601 | 0.0605 | 0.2744 | 1.08/4 | +++++++:1.000 (6); +++++++:0.999, -+-+-+-:0.001, +--++--:0.000 (3); +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 (3) |
| 0.200 | 4 | 64 | 0.991286 | 0.002696 | 0.0024 | 0.0117 | 0.0039 | 0.0342 | 0 | 0.021275 | 0.0234 | 0.2070 | 2.12/4 | +++++++:1.000 (16); +++++++:0.999, +--++--:0.000, +++----:0.000 (4); +++++++:1.000, +--++--:0.000 (4) |
| 0.200 | 5 | 64 | 0.996922 | 0.000561 | 0.0021 | 0.0098 | 0.0039 | 0.0156 | 0 | 0.021951 | 0.0176 | 0.1034 | 1.95/4 | +++++++:1.000, -+-+-+-:0.000 (7); +++++++:1.000 (4); +++++++:1.000, +++----:0.000, -+-+-+-:0.000 (4) |
| 0.200 | 6 | 64 | 0.994447 | 0.000740 | 0.0031 | 0.0068 | 0.0059 | 0.0156 | 0 | 0.013325 | 0.0137 | 0.0700 | 0.88/4 | +++++++:0.999, +--++--:0.000, --++--+:0.000 (4); +++++++:0.999, +--++--:0.000, -+-+-+-:0.000 (3); +++++++:1.000, +--++--:0.000 (3) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.200 | 3 | 0 | 0.669496 | 0.0273 | 0.0518 | 0.0488 | +++++++:0.838, --++--+:0.073, -+-+-+-:0.044 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 1 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 2 | 0.920037 | 0.0127 | 0.0264 | 0.0156 | +++++++:0.964, +++----:0.031, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 3 | 0.845693 | 0.0166 | 0.0381 | 0.0234 | +++++++:0.929, -+-+-+-:0.047, --++--+:0.014 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 5 | 0.995543 | 0.0029 | 0.0068 | 0.0020 | +++++++:0.998, -+-+-+-:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 6 | 0.826464 | 0.0254 | 0.0391 | 0.0332 | +++++++:0.920, -+-+-+-:0.040, +++----:0.022 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 7 | 0.974029 | 0.0078 | 0.0156 | 0.0078 | +++++++:0.989, +--++--:0.010, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 8 | 0.955575 | 0.0068 | 0.0205 | 0.0117 | +++++++:0.980, -+-+-+-:0.017, +--++--:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 9 | 0.963663 | 0.0078 | 0.0176 | 0.0117 | +++++++:0.984, +--++--:0.013, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 10 | 0.906468 | 0.0117 | 0.0283 | 0.0254 | +++++++:0.958, +++----:0.023, -+-+-+-:0.018 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 11 | 0.973489 | 0.0127 | 0.0146 | 0.0137 | +++++++:0.988, +--++--:0.010, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 12 | 0.942309 | 0.0107 | 0.0225 | 0.0098 | +++++++:0.974, -+-+-+-:0.025, -+--+-+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 13 | 0.913823 | 0.0088 | 0.0273 | 0.0234 | +++++++:0.961, -+-+-+-:0.025, +--++--:0.010 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 15 | 0.886230 | 0.0156 | 0.0322 | 0.0117 | +++++++:0.948, +--++--:0.041, -+-+-+-:0.008 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 16 | 0.990545 | 0.0039 | 0.0088 | 0.0059 | +++++++:0.996, +++----:0.003, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 17 | 0.946888 | 0.0049 | 0.0215 | 0.0215 | +++++++:0.976, -+-+-+-:0.017, +--++--:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 18 | 0.996655 | 0.0029 | 0.0059 | 0.0039 | +++++++:0.999, -+-+-+-:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 19 | 0.483700 | 0.0420 | 0.0674 | 0.0605 | +++++++:0.730, +++----:0.080, +--++--:0.077 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 20 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 21 | 0.995542 | 0.0029 | 0.0059 | 0.0039 | +++++++:0.998, +++----:0.001, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 22 | 0.900482 | 0.0098 | 0.0293 | 0.0234 | +++++++:0.955, +++----:0.041, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
| 0.200 | 3 | 23 | 0.785119 | 0.0264 | 0.0420 | 0.0273 | +++++++:0.895, -+-+-+-:0.104, --++--+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p20_L3456_d64_m1024_seed518000/run_q0p20_L3456_d64_m1024_seed518000.npz` |
