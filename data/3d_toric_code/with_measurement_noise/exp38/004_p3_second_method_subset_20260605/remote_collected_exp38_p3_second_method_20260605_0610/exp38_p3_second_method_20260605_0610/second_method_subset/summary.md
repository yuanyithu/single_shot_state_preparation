# exp38 P3 second-method subset summary

Overall: PASS

Estimator: stochastic bidirectional logical-loop bridge with BAR on adjacent lambda intervals, reusing the exp37 validated second-method path.  The only exp38-specific change is that disorder seeds are read from the P2 TI NPZ so the bridge compares against the same disorder realization.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P3a | sampled subset TI vs second method: TV <= 0.03 and |dq_top| <= 0.02 | checks=3, max TV=0.003319, max |dq|=0.005144 | PASS |
| P3b | bidirectional consistency diagnostic within recorded stochastic threshold | max full-path gap=0.046359, max BAR residual=8.185e-12 | PASS |
| Coverage | at least one crossing-region check for each L=3,4,5 | lattice_sizes=[3, 4, 5], num_checks=3 | PASS |

## Point Comparison

| L | q | d | TI q_top | bridge q_top | TV | dq_top | full-path gap | seed | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.220 | 0 | 0.618564 | 0.622115 | 0.002427 | 0.003551 | 0.021625 | 639000 | PASS |
| 4 | 0.220 | 0 | 0.827608 | 0.829683 | 0.001098 | 0.002075 | 0.022287 | 639000 | PASS |
| 5 | 0.220 | 0 | 0.676872 | 0.682016 | 0.003319 | 0.005144 | 0.046359 | 639000 | PASS |

## Artifacts

- `p3_second_method_subset.json`
- `p3_second_method_subset.csv`
- `stageF_second_method_subset.json` (raw exp37 runner output)
- `stageF_second_method_subset.csv` (raw exp37 runner output)
