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
| 0.160 | 3 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.160 | 3 | 64 | 0.949977 | 0.016291 | 0.0070 | 0.0605 | 0.0078 | 0.0811 | 0 | 0.057242 | 0.0938 | 0.2345 | 1.62/4 | +++++++:1.000 (16); +++++++:0.999, +--++--:0.000, +++----:0.000 (2); +++++++:1.000, +++----:0.000 (2) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.160 | 3 | 0 | 0.861094 | 0.0273 | 0.0342 | 0.0391 | +++++++:0.937, -+-+-+-:0.028, --++--+:0.019 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.160 | 3 | 1 | 0.997769 | 0.0029 | 0.0049 | 0.0039 | +++++++:0.999, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.160 | 3 | 2 | 0.989442 | 0.0068 | 0.0098 | 0.0039 | +++++++:0.995, +++----:0.004, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.160 | 3 | 3 | 0.964684 | 0.0078 | 0.0176 | 0.0156 | +++++++:0.984, -+-+-+-:0.010, +--++--:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.160 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.160 | 3 | 5 | 0.998327 | 0.0020 | 0.0039 | 0.0020 | +++++++:0.999, -+-+-+-:0.000, --++--+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.160 | 3 | 6 | 0.847112 | 0.0098 | 0.0361 | 0.0195 | +++++++:0.929, -+-+-+-:0.035, +--++--:0.033 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.160 | 3 | 7 | 0.979521 | 0.0059 | 0.0137 | 0.0078 | +++++++:0.991, +--++--:0.008, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.160 | 3 | 8 | 0.989991 | 0.0059 | 0.0098 | 0.0039 | +++++++:0.996, -+-+-+-:0.003, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.160 | 3 | 9 | 0.951294 | 0.0117 | 0.0205 | 0.0156 | +++++++:0.978, +--++--:0.019, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.160 | 3 | 10 | 0.964721 | 0.0068 | 0.0176 | 0.0195 | +++++++:0.984, +++----:0.011, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.160 | 3 | 11 | 0.992214 | 0.0010 | 0.0078 | 0.0020 | +++++++:0.997, +--++--:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.160 | 3 | 12 | 0.880285 | 0.0186 | 0.0312 | 0.0234 | +++++++:0.945, -+-+-+-:0.055, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.160 | 3 | 13 | 0.987226 | 0.0059 | 0.0107 | 0.0059 | +++++++:0.994, +--++--:0.005, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.160 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.160 | 3 | 15 | 0.771328 | 0.0322 | 0.0439 | 0.0312 | +++++++:0.889, +--++--:0.093, -+-+-+-:0.012 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.160 | 3 | 16 | 0.998327 | 0.0020 | 0.0039 | 0.0020 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.160 | 3 | 17 | 0.984466 | 0.0098 | 0.0117 | 0.0137 | +++++++:0.993, -+-+-+-:0.006, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.160 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.160 | 3 | 19 | 0.410750 | 0.0605 | 0.0732 | 0.0703 | +++++++:0.677, +--++--:0.133, +++----:0.073 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.160 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.160 | 3 | 21 | 0.997769 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.160 | 3 | 22 | 0.920242 | 0.0176 | 0.0264 | 0.0137 | +++++++:0.964, +++----:0.034, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.160 | 3 | 23 | 0.825552 | 0.0166 | 0.0381 | 0.0352 | +++++++:0.917, -+-+-+-:0.083 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
