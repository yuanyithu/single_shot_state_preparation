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
| 0.180 | 3 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.180 | 3 | 64 | 0.950092 | 0.013785 | 0.0080 | 0.0508 | 0.0078 | 0.0791 | 1 | 0.059934 | 0.0664 | 0.2190 | 1.36/4 | +++++++:1.000 (9); +++++++:1.000, +--++--:0.000 (3); +++++++:1.000, -+-+-+-:0.000 (2) |

## Flagged Disorders

| q | L | disorder | observed TV | boot p99 | q_top | q_top spread | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.180 | 3 | 10 | 0.0195 | 0.0176 | 0.964710 | 0.038982 | +++++++:0.984, +++----:0.010, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.180 | 3 | 0 | 0.866710 | 0.0254 | 0.0332 | 0.0195 | +++++++:0.939, -+-+-+-:0.025, --++--+:0.020 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.180 | 3 | 1 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.180 | 3 | 2 | 0.985012 | 0.0049 | 0.0117 | 0.0156 | +++++++:0.993, +++----:0.005, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.180 | 3 | 3 | 0.955963 | 0.0078 | 0.0195 | 0.0234 | +++++++:0.980, -+-+-+-:0.011, +++----:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.180 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.180 | 3 | 5 | 0.996098 | 0.0029 | 0.0059 | 0.0059 | +++++++:0.998, -+-+-+-:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.180 | 3 | 6 | 0.878318 | 0.0195 | 0.0332 | 0.0195 | +++++++:0.945, -+-+-+-:0.028, +--++--:0.022 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.180 | 3 | 7 | 0.960426 | 0.0059 | 0.0186 | 0.0117 | +++++++:0.982, +--++--:0.015, +++----:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.180 | 3 | 8 | 0.987214 | 0.0068 | 0.0107 | 0.0059 | +++++++:0.994, -+-+-+-:0.003, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.180 | 3 | 9 | 0.965878 | 0.0098 | 0.0176 | 0.0137 | +++++++:0.985, +--++--:0.014, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.180 | 3 | 10 | 0.964710 | 0.0195 | 0.0176 | 0.0098 | +++++++:0.984, +++----:0.010, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.180 | 3 | 11 | 0.985032 | 0.0020 | 0.0117 | 0.0078 | +++++++:0.993, +--++--:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.180 | 3 | 12 | 0.879519 | 0.0303 | 0.0322 | 0.0137 | +++++++:0.944, -+-+-+-:0.053, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.180 | 3 | 13 | 0.970695 | 0.0107 | 0.0156 | 0.0078 | +++++++:0.987, +--++--:0.008, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.180 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.180 | 3 | 15 | 0.882513 | 0.0186 | 0.0322 | 0.0117 | +++++++:0.946, +--++--:0.047, -+-+-+-:0.004 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.180 | 3 | 16 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.180 | 3 | 17 | 0.981721 | 0.0078 | 0.0127 | 0.0117 | +++++++:0.992, -+-+-+-:0.007, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.180 | 3 | 18 | 0.994431 | 0.0039 | 0.0068 | 0.0078 | +++++++:0.998, +++----:0.002, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.180 | 3 | 19 | 0.487026 | 0.0498 | 0.0684 | 0.0664 | +++++++:0.733, +--++--:0.073, +++----:0.069 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.180 | 3 | 20 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.180 | 3 | 21 | 0.994430 | 0.0049 | 0.0068 | 0.0098 | +++++++:0.998, +++----:0.001, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.180 | 3 | 22 | 0.917042 | 0.0137 | 0.0264 | 0.0195 | +++++++:0.962, +++----:0.034, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.180 | 3 | 23 | 0.793549 | 0.0137 | 0.0420 | 0.0391 | +++++++:0.900, -+-+-+-:0.100, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p18_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
