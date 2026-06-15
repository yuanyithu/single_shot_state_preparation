# exp38 P3 second-method subset summary

Status: `PASS`

Run ID: `exp38_p3_second_method_20260605_0610`

P3 used the exp37 validated stochastic bidirectional logical-loop bridge with BAR, but via `run_p3_second_method_subset.py` so the bridge reconstructs the exact exp38 P2 disorder from `disorder_seed_per_disorder`.

Subset: `3:0.22:0,4:0.22:0,5:0.22:0`.

Config: `num_lambda_points=65`, `burn=512`, `measurements=16384`, `stride=2`, remote conda env `11`, host `nd-3`, Numba available, no Python fallback.

Remote `_SUCCESS.json`: `2026-06-04T22:20:23.812639+00:00`.

The run was monitored with `sleep-until` via `wait_p3_status_20260605_0610.jsonl`; the terminal poll matched all-success at `2026-06-04T22:22:29.096068+00:00`.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P3a | sampled subset TI vs second method: TV <= 0.03 and \|dq_top\| <= 0.02 | checks=3, max TV=0.003319, max \|dq_top\|=0.005144 | PASS |
| P3b | bidirectional consistency diagnostic within recorded stochastic threshold | max full-path gap=0.046359, max BAR residual=8.185e-12 | PASS |
| Coverage | at least one crossing-region check for each L=3,4,5 | lattice_sizes=[3,4,5], num_checks=3 | PASS |

## Point Comparison

| L | q | d | TI q_top | bridge q_top | TV | dq_top | full-path gap | seed | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.220 | 0 | 0.618564 | 0.622115 | 0.002427 | 0.003551 | 0.021625 | 639000 | PASS |
| 4 | 0.220 | 0 | 0.827608 | 0.829683 | 0.001098 | 0.002075 | 0.022287 | 639000 | PASS |
| 5 | 0.220 | 0 | 0.676872 | 0.682016 | 0.003319 | 0.005144 | 0.046359 | 639000 | PASS |

## Artifacts

- Launcher and wrapper: `launch_p3_second_method_remote.sh`, `run_p3_second_method_subset.py`
- Remote manifest: `remote_p3_second_method_manifest.json`
- Wait log: `wait_p3_status_20260605_0610.jsonl`
- Collected run: `remote_collected_exp38_p3_second_method_20260605_0610/exp38_p3_second_method_20260605_0610/`
- P3 result: `remote_collected_exp38_p3_second_method_20260605_0610/exp38_p3_second_method_20260605_0610/second_method_subset/p3_second_method_subset.json`
- P3 summary: `remote_collected_exp38_p3_second_method_20260605_0610/exp38_p3_second_method_20260605_0610/second_method_subset/summary.md`
