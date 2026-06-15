# exp38 P2 production grid summary

Status: `PASS`

Run ID: `exp38_p2_ti_grid_20260605_0145`

P2 completed the remote strong sector-TI production grid and produced the merged grid plus P2 failure-map draft. P3 was not started in this stage.

## Configuration

| field | value |
|---|---|
| p | 0.05 |
| q grid | 0.08,0.10,0.12,0.14,0.15..0.23 |
| lattice sizes | 3,4,5 |
| disorder samples | 32 for every q |
| disorder seed scope | disorder_index |
| disorder realization | rng_stream |
| TI grid | 129 |
| burn / cap | 512 / 512 |
| measurements | 8192 |
| stride | 2 |
| blocks / bootstrap | 128 / 800 |
| host split | nd-1:L3, nd-2:L4, nd-3:L5 |

Remote preflight passed on all compute nodes: `nd-1` and `nd-2` reported `nproc=80`, `nd-3` reported `nproc=96`, and all three had `conda` env `11` with `numba_available=True`.

## Remote Run

| L | host | remote status | finished_at UTC |
|---:|---|---|---|
| 3 | nd-1 | success | 2026-06-04T18:42:43.467903+00:00 |
| 4 | nd-2 | success | 2026-06-04T19:49:47.151744+00:00 |
| 5 | nd-3 | success | 2026-06-04T21:26:06.358028+00:00 |

The run was monitored with `sleep-until` via `wait_p2_status_20260605_0145.jsonl`; the terminal poll matched all-success at `2026-06-04T21:29:55.304544+00:00`.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P2a | full L x q x disorder coverage with weights, delta_f, q_top, stderr and explicit flags | coverage=True, missing=0, rows=1248, points=39 | PASS |
| P2b | unresolved high-q_top tails are marked FAIL, never PASS | unresolved_tail_fail=0, pass_violations=0 | PASS |
| P2c | every PASS disorder has grid TV and \|dq_top\| <= 0.02 | PASS-disorder grid failures=0 | PASS |
| Common disorder | `disorder_seed_per_disorder` identical across L for every `(q, disorder)` | mismatches=0 | PASS |

Point statuses: `PASS:20`, `WARN:19`, `FAIL:0`.

Disorder-row statuses: `PASS:1195`, `WARN:53`, `FAIL:0`.

Worst WARN diagnostics: max grid TV `0.091794`, max grid `|dq_top|=0.138340`, both at `L=5,q=0.08,d=7`; these are WARN rows in the failure map, not P2 failures. Max point total SEM is `0.043558` at `L=4,q=0.18`.

## Artifacts

- Remote manifest: `remote_runs_manifest.json`
- Remote preflight: `preflight_summary.json`
- Collected shards: `remote_collected_exp38_p2_ti_grid_20260605_0145/`
- Merged grid: `merged_exp38_p2_ti_grid_20260605_0145/sector_ti_results.npz`
- Merged summary: `merged_exp38_p2_ti_grid_20260605_0145/sector_ti_summary.md`
- P2 acceptance: `accepted_exp38_p2_ti_grid_20260605_0145/p2_acceptance.json`
- Failure map draft: `accepted_exp38_p2_ti_grid_20260605_0145/failure_map.md`
- Point/disorder tables: `accepted_exp38_p2_ti_grid_20260605_0145/p2_point_status.csv`, `accepted_exp38_p2_ti_grid_20260605_0145/p2_disorder_status.csv`
