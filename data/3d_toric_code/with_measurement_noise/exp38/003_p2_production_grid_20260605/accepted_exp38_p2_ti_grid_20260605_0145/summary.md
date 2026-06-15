# exp38 P2 sector-TI acceptance draft

Overall: PASS

Source TI result: `data/3d_toric_code/with_measurement_noise/exp38/003_p2_production_grid_20260605/merged_exp38_p2_ti_grid_20260605_0145/sector_ti_results.npz`

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P2a | full L x q x disorder coverage with weights, delta_f, q_top, stderr and explicit flags | coverage=True, missing=0, statuses=PASS:20, WARN:19 | PASS |
| P2b | unresolved high-q_top tails are marked FAIL, never PASS | unresolved_tail_fail=0, pass_violations=0 | PASS |
| P2c | every PASS disorder has grid TV and |dq_top| <= 0.02 | PASS-disorder grid failures=0 | PASS |
| Common disorder | disorder_seed_per_disorder identical across L for every (q, disorder) | mismatches=0 | PASS |

## Artifacts

- `p2_acceptance.json`
- `p2_point_status.csv`
- `p2_disorder_status.csv`
- `failure_map.md`
