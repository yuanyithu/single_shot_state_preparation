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
| 0.100 | 3 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.100 | 4 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.100 | 5 | 4 | 4/8 | 000, 100, 010, 110 |
| 0.100 | 6 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.100 | 3 | 64 | 0.976843 | 0.008951 | 0.0034 | 0.0342 | 0.0000 | 0.0547 | 0 | 0.062267 | 0.0410 | 0.2072 | 2.75/4 | +++++++:1.000 (35); +++++++:1.000, +--++--:0.000 (3); +++++++:0.998, -+-+-+-:0.002 (2) |
| 0.100 | 4 | 64 | 0.999138 | 0.000592 | 0.0003 | 0.0059 | 0.0000 | 0.0176 | 0 | 0.013009 | 0.0117 | 0.0858 | 3.70/4 | +++++++:1.000 (55); +++++++:1.000, +--++--:0.000 (4); +++++++:0.984, -+-+-+-:0.015, +--++--:0.001 (1) |
| 0.100 | 5 | 64 | 0.999591 | 0.000125 | 0.0004 | 0.0059 | 0.0000 | 0.0078 | 0 | 0.013319 | 0.0078 | 0.0353 | 3.52/4 | +++++++:1.000 (44); +++++++:1.000, +--++--:0.000 (7); +++++++:1.000, -+-+-+-:0.000 (4) |
| 0.100 | 6 | 64 | 0.998775 | 0.000384 | 0.0008 | 0.0088 | 0.0000 | 0.0127 | 0 | 0.017665 | 0.0137 | 0.0699 | 3.09/4 | +++++++:1.000 (36); +++++++:1.000, +--++--:0.000 (6); +++++++:1.000, -+-+-+-:0.000 (5) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.100 | 3 | 0 | 0.983343 | 0.0059 | 0.0117 | 0.0098 | +++++++:0.993, --++--+:0.003, -+-+-+-:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk000.npz` |
| 0.100 | 3 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk001.npz` |
| 0.100 | 3 | 2 | 0.997770 | 0.0029 | 0.0049 | 0.0059 | +++++++:0.999, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk002.npz` |
| 0.100 | 3 | 3 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk003.npz` |
| 0.100 | 3 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk004.npz` |
| 0.100 | 3 | 5 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk005.npz` |
| 0.100 | 3 | 6 | 0.889300 | 0.0225 | 0.0312 | 0.0332 | +++++++:0.950, -+-+-+-:0.025, +--++--:0.024 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk006.npz` |
| 0.100 | 3 | 7 | 0.964835 | 0.0078 | 0.0176 | 0.0098 | +++++++:0.984, +--++--:0.015, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk007.npz` |
| 0.100 | 3 | 8 | 0.996100 | 0.0020 | 0.0059 | 0.0020 | +++++++:0.998, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk008.npz` |
| 0.100 | 3 | 9 | 0.997770 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, +++----:0.001, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk009.npz` |
| 0.100 | 3 | 10 | 0.996656 | 0.0020 | 0.0059 | 0.0039 | +++++++:0.999, -+-+-+-:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk010.npz` |
| 0.100 | 3 | 11 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk011.npz` |
| 0.100 | 3 | 12 | 0.995544 | 0.0039 | 0.0059 | 0.0039 | +++++++:0.998, -+-+-+-:0.002 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk012.npz` |
| 0.100 | 3 | 13 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk013.npz` |
| 0.100 | 3 | 14 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk014.npz` |
| 0.100 | 3 | 15 | 0.699930 | 0.0215 | 0.0488 | 0.0312 | +++++++:0.854, +--++--:0.079, --++--+:0.040 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk015.npz` |
| 0.100 | 3 | 16 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk016.npz` |
| 0.100 | 3 | 17 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk017.npz` |
| 0.100 | 3 | 18 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk018.npz` |
| 0.100 | 3 | 19 | 0.760060 | 0.0342 | 0.0479 | 0.0410 | +++++++:0.885, +--++--:0.085, -+-+-+-:0.018 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk019.npz` |
| 0.100 | 3 | 20 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk020.npz` |
| 0.100 | 3 | 21 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk021.npz` |
| 0.100 | 3 | 22 | 0.992769 | 0.0039 | 0.0078 | 0.0059 | +++++++:0.997, +++----:0.003 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk022.npz` |
| 0.100 | 3 | 23 | 0.986682 | 0.0049 | 0.0107 | 0.0059 | +++++++:0.994, -+-+-+-:0.006, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p10_L3456_d64_m1024_seed518000/chunks/L03_p00_chunk023.npz` |
