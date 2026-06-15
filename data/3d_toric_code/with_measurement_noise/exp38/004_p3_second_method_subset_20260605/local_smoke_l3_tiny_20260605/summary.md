# exp38 P3 second-method subset summary

Overall: FAIL

Estimator: stochastic bidirectional logical-loop bridge with BAR on adjacent lambda intervals, reusing the exp37 validated second-method path.  The only exp38-specific change is that disorder seeds are read from the P2 TI NPZ so the bridge compares against the same disorder realization.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P3a | sampled subset TI vs second method: TV <= 0.03 and |dq_top| <= 0.02 | checks=1, max TV=0.198784, max |dq|=0.381312 | FAIL |
| P3b | bidirectional consistency diagnostic within recorded stochastic threshold | max full-path gap=56.902339, max BAR residual=9.481e-13 | FAIL |
| Coverage | at least one crossing-region check for each L=3,4,5 | lattice_sizes=[3], num_checks=1 | FAIL |

## Point Comparison

| L | q | d | TI q_top | bridge q_top | TV | dq_top | full-path gap | seed | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.220 | 0 | 0.618564 | 0.999876 | 0.198784 | 0.381312 | 56.902339 | 639000 | FAIL |

## Artifacts

- `p3_second_method_subset.json`
- `p3_second_method_subset.csv`
- `stageF_second_method_subset.json` (raw exp37 runner output)
- `stageF_second_method_subset.csv` (raw exp37 runner output)
