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
| 0.130 | 3 | 8 | 8/8 | 000, 100, 010, 110, 001, 101, 011, 111 |
| 0.220 | 3 | 8 | 8/8 | 000, 100, 010, 110, 001, 101, 011, 111 |
| 0.230 | 3 | 8 | 8/8 | 000, 100, 010, 110, 001, 101, 011, 111 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.130 | 3 | 1 | 0.880904 | 0.000000 | 0.0035 | 0.0035 | 0.0128 | 0.0128 | 0 | 0.006438 | 0.0078 | 0.1562 | 0.00/0 | +++++++:0.945, +--++--:0.055, -+-+-+-:0.000 (1) |
| 0.220 | 3 | 1 | 0.281263 | 0.000000 | 0.0214 | 0.0214 | 0.0317 | 0.0317 | 0 | 0.009820 | 0.0400 | 0.0999 | 0.00/0 | -+-+-+-:0.451, +++++++:0.403, +--++--:0.044 (1) |
| 0.230 | 3 | 2 | 0.813634 | 0.075296 | 0.0063 | 0.0071 | 0.0157 | 0.0189 | 0 | 0.009727 | 0.0220 | 0.1779 | 0.00/0 | +++++++:0.875, -+-+-+-:0.055, +++----:0.037 (1); +++++++:0.949, +--++--:0.041, -+-+-+-:0.007 (1) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.130 | 3 | 52 | 0.880904 | 0.0035 | 0.0128 | 0.0078 | +++++++:0.945, +--++--:0.055, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/019_targeted_8start_sector_reference_20260531/remote_results/chunks/q0p13_L03_d052_m8192_8start.npz` |
| 0.220 | 3 | 36 | 0.281263 | 0.0214 | 0.0317 | 0.0400 | -+-+-+-:0.451, +++++++:0.403, +--++--:0.044 | `data/3d_toric_code/with_measurement_noise/exp36/019_targeted_8start_sector_reference_20260531/remote_results/chunks/q0p22_L03_d036_m8192_8start.npz` |
| 0.230 | 3 | 6 | 0.738338 | 0.0071 | 0.0189 | 0.0220 | +++++++:0.875, -+-+-+-:0.055, +++----:0.037 | `data/3d_toric_code/with_measurement_noise/exp36/019_targeted_8start_sector_reference_20260531/remote_results/chunks/q0p23_L03_d006_m8192_8start.npz` |
| 0.230 | 3 | 15 | 0.888930 | 0.0056 | 0.0125 | 0.0081 | +++++++:0.949, +--++--:0.041, -+-+-+-:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/019_targeted_8start_sector_reference_20260531/remote_results/chunks/q0p23_L03_d015_m8192_8start.npz` |
