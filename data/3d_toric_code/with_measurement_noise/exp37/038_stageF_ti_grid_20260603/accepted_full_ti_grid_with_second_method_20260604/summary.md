# Stage F sector-TI acceptance summary

Overall: DOING/FAIL

Source TI result: `data/3d_toric_code/with_measurement_noise/exp37/038_stageF_ti_grid_20260603/merged_ti_grid_20260604/sector_ti_results.npz`

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| F1 | full grid present and every point has explicit PASS/WARN/FAIL; unresolved high-q_top tails are FAIL | coverage=True, unresolved=0, statuses=FAIL:2, PASS:7, WARN:39 | PASS |
| F2 | every PASS disorder satisfies coarse/fine grid TV and dq <= 0.02 | PASS-disorder grid failures=0 | PASS |
| F3 | sampled subset second method agrees with TI | checks=3, failures=2 | FAIL |

## Artifacts

- `stageF_acceptance.json`
- `stageF_point_status.csv`
- `stageF_disorder_status.csv`
- `failure_map.md`
