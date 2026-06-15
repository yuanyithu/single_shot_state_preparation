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
| 0.180 | 3 | 8 | 8/8 | 000, 100, 010, 110, 001, 101, 011, 111 |

## Summary

| q | L | disorders | q_top mean | q_top SEM | start-TV mean | start-TV max | boot-p99 median | boot-p99 max | TV fails | q_top spread max | first/second TV max | block q_top range max | never-flipped mean/max | dominant sectors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.180 | 3 | 2 | 0.984247 | 0.014847 | 0.0021 | 0.0033 | 0.0038 | 0.0065 | 0 | 0.007390 | 0.0059 | 0.0699 | 0.50/1 | +++++++:0.986, +++----:0.007, -+-+-+-:0.006 (1); +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 (1) |

## Flagged Disorders

No disorder exceeded the bootstrap p99 start-sector TV gate.

## Sample Disorder Rows

| q | L | disorder | q_top | observed TV | boot p99 | first/second TV max | top sectors | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.180 | 3 | 1 | 0.999093 | 0.0009 | 0.0011 | 0.0010 | +++++++:1.000, +--++--:0.000, -+-+-+-:0.000 | `data/3d_toric_code/with_measurement_noise/exp36/020_q0p18_L3_d10_8start_reference_20260601/remote_results/chunks/q0p18_L03_d001_m8192_8start_d1.npz` |
| 0.180 | 3 | 10 | 0.969400 | 0.0033 | 0.0065 | 0.0059 | +++++++:0.986, +++----:0.007, -+-+-+-:0.006 | `data/3d_toric_code/with_measurement_noise/exp36/020_q0p18_L3_d10_8start_reference_20260601/remote_results/chunks/q0p18_L03_d010_m8192_8start_d10.npz` |
