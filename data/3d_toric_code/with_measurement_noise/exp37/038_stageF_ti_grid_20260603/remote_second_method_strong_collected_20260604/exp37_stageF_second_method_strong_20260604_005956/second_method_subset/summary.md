# Stage F second-method subset summary

Overall subset agreement: FAIL

Estimator: stochastic bidirectional logical-loop bridge with BAR on adjacent lambda intervals.  It uses the same production disorder seeds and sector representatives as the Stage F TI grid, but not the Kp thermodynamic-integration estimator.

Thresholds: TV <= 0.030, |dq_top| <= 0.020.

| L | q | d | TI q_top | bridge q_top | TV | dq_top | bidir gap | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.230 | 1 | 0.492725 | 0.484730 | 0.00804 | 0.00800 | 0.00730 | PASS |
| 4 | 0.210 | 3 | 0.517033 | 0.549355 | 0.02042 | 0.03232 | 0.02024 | FAIL |
| 5 | 0.190 | 0 | 0.497852 | 0.459750 | 0.02626 | 0.03810 | 0.11516 | FAIL |

Artifacts:
- `stageF_second_method_subset.json`
- `stageF_second_method_subset.csv`
