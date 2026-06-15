# Stage F second-method subset summary

Overall subset agreement: FAIL

Estimator: stochastic bidirectional logical-loop bridge with BAR on adjacent lambda intervals.  It uses the same production disorder seeds and sector representatives as the Stage F TI grid, but not the Kp thermodynamic-integration estimator.

Thresholds: TV <= 0.030, |dq_top| <= 0.020.

| L | q | d | TI q_top | bridge q_top | TV | dq_top | bidir gap | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.230 | 1 | 0.492725 | 0.493704 | 0.00266 | 0.00098 | 0.03427 | PASS |
| 4 | 0.210 | 3 | 0.517033 | 0.567893 | 0.03187 | 0.05086 | 0.12543 | FAIL |
| 5 | 0.190 | 0 | 0.497852 | 0.456377 | 0.02871 | 0.04147 | 0.39342 | FAIL |

Artifacts:
- `stageF_second_method_subset.json`
- `stageF_second_method_subset.csv`
