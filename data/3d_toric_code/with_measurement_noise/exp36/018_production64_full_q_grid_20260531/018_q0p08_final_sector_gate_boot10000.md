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
| 0.080 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.080 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.080 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.080 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.080 | 3 | 64 | 0.986364 | 0.006935 | 0.0022 | 0.0391 | 0.0000 | 0.0537 | 0 | 0.066897 | 0.0684 | 0.2294 | 3.09/4 | +++++++:1.000 (40); +++++++:0.999, -+-+-+-:0.001 (3); +++++++:1.000, +--++--:0.000 (3) |
| 0.080 | 4 | 64 | 0.999835 | 0.000126 | 0.0001 | 0.0049 | 0.0000 | 0.0078 | 0 | 0.011102 | 0.0039 | 0.0526 | 3.92/4 | +++++++:1.000 (61); +++++++:0.997, +++----:0.003, +--++--:0.000 (1); +++++++:0.999, -+-+-+-:0.001, +--++--:0.000 (1) |
| 0.080 | 5 | 64 | 0.999843 | 0.000074 | 0.0002 | 0.0039 | 0.0000 | 0.0059 | 0 | 0.008905 | 0.0039 | 0.0353 | 3.88/4 | +++++++:1.000 (54); +++++++:1.000, +++----:0.000 (4); +++++++:1.000, +--++--:0.000 (3) |
| 0.080 | 6 | 64 | 0.999686 | 0.000112 | 0.0003 | 0.0039 | 0.0000 | 0.0068 | 0 | 0.008894 | 0.0078 | 0.0353 | 3.64/4 | +++++++:1.000 (53); +++++++:0.999, +--++--:0.001 (3); +++++++:0.999, -+-+-+-:0.000, +--++--:0.000 (2) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.080 | 3 | 0 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.080 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.080 | 3 | 2 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.080 | 3 | 3 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.080 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.080 | 3 | 5 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.080 | 3 | 6 | 0.865358 | 0.0186 | 0.0342 | 0.0352 | +++++++:0.938, -+-+-+-:0.040, +--++--:0.018 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.080 | 3 | 7 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.080 | 3 | 8 | 0.997770 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.080 | 3 | 9 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.080 | 3 | 10 | 0.997770 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, -+-+-+-:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.080 | 3 | 11 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.080 | 3 | 12 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.080 | 3 | 13 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.080 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.080 | 3 | 15 | 0.738549 | 0.0283 | 0.0469 | 0.0273 | +++++++:0.873, +--++--:0.094, --++--+:0.020 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.080 | 3 | 16 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.080 | 3 | 17 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.080 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.080 | 3 | 19 | 0.907796 | 0.0137 | 0.0293 | 0.0215 | +++++++:0.958, +--++--:0.022, -+-+-+-:0.009 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.080 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.080 | 3 | 21 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.080 | 3 | 22 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.080 | 3 | 23 | 0.994433 | 0.0039 | 0.0068 | 0.0059 | +++++++:0.998, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p08_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
