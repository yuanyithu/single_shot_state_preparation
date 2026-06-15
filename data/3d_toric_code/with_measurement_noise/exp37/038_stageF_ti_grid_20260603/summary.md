# Stage F sector-TI grid progress summary

Overall: PASS

The full sector-TI production grid was run remotely, collected locally, merged,
cross-checked with the Stage E bidirectional logical-loop bridge subset, and
accepted after a targeted strong TI repair of the two F3 mismatch records.

## Completed This Round

- Added and validated remote helpers:
  - `launch_stageF_ti_remote.sh`
  - `check_stageF_remote_status.py`
  - `collect_stageF_ti_remote.py`
  - `merge_stageF_ti_shards.py`
- Launched remote conda `11` + `screen` shards:
  - `nd-1`: `L=3`
  - `nd-2`: `L=4`
  - `nd-3`: `L=5`
- Waited with the `sleep-until` blocking checker until all shard success
  sentinels were present.
- Collected remote outputs into `remote_collected_20260604/`.
- Merged the shards into `merged_ti_grid_20260604/sector_ti_results.npz`.
- Added `run_stageF_second_method_subset.py` and
  `launch_stageF_second_method_remote.sh` for the production-grid
  bidirectional logical-loop bridge BAR subset.
- Ran the stronger F3 subset on `nd-3` with `65` lambda points, burn `512`,
  measurements `16384`, stride `2`; collected it into
  `remote_second_method_strong_collected_20260604/`.
- Diagnosed the two initial F3 mismatches with targeted strong sector-TI:
  `L=4,5`, `q=0.19,0.21`, `4` common disorders, TI grid `129`, burn `512`,
  measurements `8192`, stride `2`.
- Built a repaired grid by replacing only the `16` targeted records
  (`L=4,5 x q=0.19,0.21 x 4 disorders`) with the targeted strong TI results.
- Reran formal Stage F acceptance into
  `accepted_repaired_ti_grid_targeted_strong_20260604/`.

## Production TI Config

- `p=0.05`
- `L=3,4,5`
- `q=0.08..0.23`
- `4` common-disorder samples per `(L,q)`
- Base TI grid `65`, burn `160`, measurements `2048`, stride `2`,
  blocks `64`, bootstrap `600`
- Targeted repair TI grid `129`, burn `512`, measurements `8192`, stride `2`,
  blocks `128`, bootstrap `800`
- linear fixed-sector projection, Numba enabled remotely

## Final Gate Numbers

| Gate | Result | Status |
|---|---:|---|
| F1 | coverage=True, unresolved_tail_fail=0, point statuses=PASS:10/WARN:38/FAIL:0 | PASS |
| F2 | PASS-disorder grid failures=0 | PASS |
| F3 | second-method checks=3, failures=0; max abs(dq_top)=0.01034, max TV=0.01286 | PASS |

## Targeted F3 Diagnosis

The initial F3 failures came from under-sampled original production TI records,
not from a bridge contradiction.  Strong targeted TI agrees with the bridge on
both records.

| L | q | disorder | old TI q_top | strong TI q_top | bridge q_top | abs(strong-bridge) | TV strong-bridge | strong grid TV | strong grid dq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.210 | 3 | 0.517033 | 0.550499 | 0.549355 | 0.001144 | 0.004323 | 0.004544 | 0.006946 |
| 5 | 0.190 | 0 | 0.497852 | 0.449411 | 0.459750 | 0.010340 | 0.012864 | 0.006307 | 0.006621 |

## Final F3 Subset Numbers

| L | q | disorder | TI q_top | bridge q_top | TV | abs(dq_top) | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.230 | 1 | 0.492725 | 0.484730 | 0.00804 | 0.00800 | PASS |
| 4 | 0.210 | 3 | 0.550499 | 0.549355 | 0.00432 | 0.00114 | PASS |
| 5 | 0.190 | 0 | 0.449411 | 0.459750 | 0.01286 | 0.01034 | PASS |

## Key Artifacts

- `remote_runs_manifest.json`
- `wait_stageF_remote_20260604.jsonl`
- `remote_collected_20260604/`
- `merged_ti_grid_20260604/sector_ti_results.npz`
- `remote_second_method_strong_collected_20260604/exp37_stageF_second_method_strong_20260604_005956/second_method_subset/stageF_second_method_subset.json`
- `remote_ti_targeted_strong_manifest.json`
- `wait_stageF_ti_targeted_strong_20260604.jsonl`
- `remote_ti_targeted_strong_collected_20260604/`
- `merged_ti_targeted_strong_20260604/sector_ti_results.npz`
- `targeted_ti_vs_bridge_comparison_20260604/summary.md`
- `repaired_ti_grid_targeted_strong_20260604/sector_ti_results.npz`
- `repaired_ti_grid_targeted_strong_20260604/stageF_second_method_subset_rebased.json`
- `accepted_repaired_ti_grid_targeted_strong_20260604/stageF_acceptance.json`
- `accepted_repaired_ti_grid_targeted_strong_20260604/failure_map.md`

## Next Step

Stage F PASS.  The next `/goal` starts Stage G; do not start Stage G in this
run.
