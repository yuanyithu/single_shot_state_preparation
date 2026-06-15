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
| 0.190 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.190 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.190 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.190 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.190 | 3 | 64 | 0.943237 | 0.014599 | 0.0081 | 0.0430 | 0.0088 | 0.0791 | 0 | 0.050435 | 0.0859 | 0.2340 | 1.05/4 | +++++++:0.999, -+-+-+-:0.000, +--++--:0.000 (3); +++++++:1.000, +++----:0.000 (3); +++++++:1.000, +--++--:0.000 (3) |
| 0.190 | 4 | 64 | 0.992402 | 0.002784 | 0.0023 | 0.0146 | 0.0029 | 0.0371 | 0 | 0.029258 | 0.0273 | 0.1507 | 2.50/4 | +++++++:1.000 (17); +++++++:1.000, +--++--:0.000 (8); +++++++:1.000, +++----:0.000 (7) |
| 0.190 | 5 | 64 | 0.997371 | 0.000390 | 0.0019 | 0.0078 | 0.0029 | 0.0117 | 0 | 0.017744 | 0.0137 | 0.0699 | 2.06/4 | +++++++:1.000 (11); +++++++:1.000, +--++--:0.000 (3); +++++++:1.000, -+-+-+-:0.000 (3) |
| 0.190 | 6 | 64 | 0.994845 | 0.000654 | 0.0031 | 0.0117 | 0.0059 | 0.0166 | 0 | 0.022042 | 0.0137 | 0.0868 | 1.11/4 | +++++++:1.000, -+-+-+-:0.000 (5); +++++++:0.999, +--++--:0.000, --++--+:0.000 (4); +++++++:0.998, +++----:0.001, +--++--:0.000 (3) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.190 | 3 | 0 | 0.692827 | 0.0273 | 0.0508 | 0.0352 | +++++++:0.851, --++--+:0.054, -+-+-+-:0.049 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.190 | 3 | 1 | 0.998884 | 0.0020 | 0.0029 | 0.0000 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.190 | 3 | 2 | 0.920520 | 0.0156 | 0.0264 | 0.0195 | +++++++:0.964, +++----:0.031, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.190 | 3 | 3 | 0.886046 | 0.0166 | 0.0322 | 0.0176 | +++++++:0.948, -+-+-+-:0.039, --++--+:0.007 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.190 | 3 | 4 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.190 | 3 | 5 | 0.996656 | 0.0020 | 0.0059 | 0.0039 | +++++++:0.999, -+-+-+-:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.190 | 3 | 6 | 0.809614 | 0.0195 | 0.0410 | 0.0371 | +++++++:0.911, -+-+-+-:0.043, +++----:0.027 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.190 | 3 | 7 | 0.978960 | 0.0088 | 0.0137 | 0.0137 | +++++++:0.991, +--++--:0.008, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.190 | 3 | 8 | 0.986110 | 0.0059 | 0.0117 | 0.0059 | +++++++:0.994, -+-+-+-:0.004, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.190 | 3 | 9 | 0.955500 | 0.0088 | 0.0195 | 0.0176 | +++++++:0.980, +--++--:0.014, +++----:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.190 | 3 | 10 | 0.926493 | 0.0117 | 0.0254 | 0.0234 | +++++++:0.967, -+-+-+-:0.017, +++----:0.015 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.190 | 3 | 11 | 0.986123 | 0.0039 | 0.0107 | 0.0059 | +++++++:0.994, +--++--:0.005, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.190 | 3 | 12 | 0.924921 | 0.0059 | 0.0254 | 0.0137 | +++++++:0.966, -+-+-+-:0.032, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.190 | 3 | 13 | 0.974540 | 0.0068 | 0.0146 | 0.0137 | +++++++:0.989, +--++--:0.007, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.190 | 3 | 14 | 0.992214 | 0.0029 | 0.0078 | 0.0098 | +++++++:0.997, +--++--:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.190 | 3 | 15 | 0.884407 | 0.0234 | 0.0322 | 0.0332 | +++++++:0.947, +--++--:0.044, -+-+-+-:0.005 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.190 | 3 | 16 | 0.993320 | 0.0039 | 0.0078 | 0.0059 | +++++++:0.997, +++----:0.002, -+-+-+-:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.190 | 3 | 17 | 0.988339 | 0.0059 | 0.0098 | 0.0059 | +++++++:0.995, -+-+-+-:0.005, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.190 | 3 | 18 | 0.997769 | 0.0020 | 0.0049 | 0.0039 | +++++++:0.999, -+-+-+-:0.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.190 | 3 | 19 | 0.460854 | 0.0322 | 0.0703 | 0.0449 | +++++++:0.715, +--++--:0.085, +++----:0.082 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.190 | 3 | 20 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, -+-+-+-:0.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.190 | 3 | 21 | 0.996656 | 0.0020 | 0.0059 | 0.0020 | +++++++:0.999, +++----:0.001, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.190 | 3 | 22 | 0.910808 | 0.0098 | 0.0273 | 0.0273 | +++++++:0.959, +++----:0.037, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.190 | 3 | 23 | 0.823835 | 0.0195 | 0.0381 | 0.0137 | +++++++:0.916, -+-+-+-:0.082, +++----:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p19_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
