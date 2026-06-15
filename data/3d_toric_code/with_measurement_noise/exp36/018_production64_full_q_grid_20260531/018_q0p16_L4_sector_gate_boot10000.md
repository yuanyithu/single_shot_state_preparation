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
| 0.160 | 4 | 4 | 4/8 | 000, 100, 010, 110 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.160 | 4 | 64 | 0.995024 | 0.001783 | 0.0019 | 0.0166 | 0.0020 | 0.0283 | 0 | 0.032091 | 0.0215 | 0.1162 | 2.62/4 | +++++++:1.000 (24); +++++++:1.000, +--++--:0.000 (4); +++++++:1.000, --++--+:0.000 (4) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.160 | 4 | 0 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk000.npz` |
| 0.160 | 4 | 1 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk001.npz` |
| 0.160 | 4 | 2 | 0.998327 | 0.0010 | 0.0039 | 0.0020 | +++++++:0.999, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk002.npz` |
| 0.160 | 4 | 3 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, --++--+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk003.npz` |
| 0.160 | 4 | 4 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk004.npz` |
| 0.160 | 4 | 5 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk005.npz` |
| 0.160 | 4 | 6 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk006.npz` |
| 0.160 | 4 | 7 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk007.npz` |
| 0.160 | 4 | 8 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk008.npz` |
| 0.160 | 4 | 9 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, --++--+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk009.npz` |
| 0.160 | 4 | 10 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk010.npz` |
| 0.160 | 4 | 11 | 0.994984 | 0.0029 | 0.0068 | 0.0039 | +++++++:0.998, +--++--:0.001, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk011.npz` |
| 0.160 | 4 | 12 | 0.999442 | 0.0010 | 0.0020 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk012.npz` |
| 0.160 | 4 | 13 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk013.npz` |
| 0.160 | 4 | 14 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, +--++--:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk014.npz` |
| 0.160 | 4 | 15 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk015.npz` |
| 0.160 | 4 | 16 | 0.998327 | 0.0020 | 0.0039 | 0.0020 | +++++++:0.999, +--++--:0.000, -+--+-+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk016.npz` |
| 0.160 | 4 | 17 | 0.997212 | 0.0020 | 0.0049 | 0.0020 | +++++++:0.999, +--++--:0.000, +++----:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk017.npz` |
| 0.160 | 4 | 18 | 0.998327 | 0.0020 | 0.0039 | 0.0039 | +++++++:0.999, +--++--:0.001 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk018.npz` |
| 0.160 | 4 | 19 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk019.npz` |
| 0.160 | 4 | 20 | 0.998884 | 0.0010 | 0.0029 | 0.0020 | +++++++:1.000, --++--+:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk020.npz` |
| 0.160 | 4 | 21 | 0.998884 | 0.0020 | 0.0029 | 0.0000 | +++++++:1.000, +----++:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk021.npz` |
| 0.160 | 4 | 22 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk022.npz` |
| 0.160 | 4 | 23 | 1.000000 | 0.0000 | 0.0000 | 0.0000 | +++++++:1.000 | `data/3d_toric_code/with_measurement_noise/exp36/018_production64_full_q_grid_20260531/remote_partial/run_q0p16_L3456_d64_m1024_seed518000/chunks/L04_p00_chunk023.npz` |
