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
| 0.170 | 3 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.170 | 3 | 64 | 0.948914 | 0.014898 | 0.0073 | 0.0498 | 0.0078 | 0.0791 | 0 | 0.050526 | 0.0781 | 0.2114 | 1.45/4 | +++++++:1.000 (6); +++++++:1.000, +--++--:0.000 (6); +++++++:1.000, +++----:0.000 (3) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.170 | 3 | 0 | 0.868790 | 0.0137 | 0.0342 | 0.0195 | +++++++:0.940, -+-+-+-:0.027, --++--+:0.018 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.170 | 3 | 1 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.170 | 3 | 2 | 0.985562 | 0.0039 | 0.0117 | 0.0059 | +++++++:0.994, +++----:0.004, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.170 | 3 | 3 | 0.951712 | 0.0127 | 0.0205 | 0.0137 | +++++++:0.979, -+-+-+-:0.016, --++--+:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.170 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.170 | 3 | 5 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.170 | 3 | 6 | 0.882585 | 0.0186 | 0.0312 | 0.0195 | +++++++:0.947, -+-+-+-:0.031, +--++--:0.020 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.170 | 3 | 7 | 0.949220 | 0.0166 | 0.0205 | 0.0117 | +++++++:0.977, +--++--:0.021, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.170 | 3 | 8 | 0.990547 | 0.0029 | 0.0088 | 0.0020 | +++++++:0.996, -+-+-+-:0.003, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.170 | 3 | 9 | 0.955021 | 0.0117 | 0.0195 | 0.0059 | +++++++:0.980, +--++--:0.017, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.170 | 3 | 10 | 0.967447 | 0.0088 | 0.0166 | 0.0078 | +++++++:0.986, +++----:0.010, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.170 | 3 | 11 | 0.986668 | 0.0049 | 0.0107 | 0.0137 | +++++++:0.994, +--++--:0.004, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.170 | 3 | 12 | 0.885482 | 0.0107 | 0.0322 | 0.0098 | +++++++:0.947, -+-+-+-:0.050, +--++--:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.170 | 3 | 13 | 0.965250 | 0.0117 | 0.0166 | 0.0156 | +++++++:0.985, +--++--:0.010, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.170 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.170 | 3 | 15 | 0.876479 | 0.0127 | 0.0322 | 0.0312 | +++++++:0.943, +--++--:0.043, -+-+-+-:0.011 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.170 | 3 | 16 | 0.997770 | 0.0029 | 0.0049 | 0.0020 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.170 | 3 | 17 | 0.989442 | 0.0049 | 0.0098 | 0.0059 | +++++++:0.995, -+-+-+-:0.004, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.170 | 3 | 18 | 0.997770 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.170 | 3 | 19 | 0.486254 | 0.0498 | 0.0684 | 0.0352 | +++++++:0.732, +--++--:0.084, +++----:0.065 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.170 | 3 | 20 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.170 | 3 | 21 | 0.996099 | 0.0020 | 0.0059 | 0.0039 | +++++++:0.998, -+-+-+-:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.170 | 3 | 22 | 0.892979 | 0.0137 | 0.0303 | 0.0273 | +++++++:0.951, +++----:0.046, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.170 | 3 | 23 | 0.767315 | 0.0273 | 0.0439 | 0.0312 | +++++++:0.885, -+-+-+-:0.113, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p17_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
