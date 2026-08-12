# legacy_delta_only nd-3 adaptive pilot

This directory records the completed 2026-08-11/12 `legacy_delta_only`
adaptive pilot on yuany/nd-3.  Frozen evidence under
`data/3d_toric_code/**` remains read-only.

The authorized scope ends after the 48-disorder adaptive pilot.  A strict
bracket may produce a production estimate, but no 384-disorder top-up is
launched without separate authorization.

Operational details and validated run identifiers are recorded here.  Raw
NPZ files, task checkpoints, and logs are intentionally
ignored by Git; small manifests and decision tables are allow-listed exactly.

## Result

- [`validation.json`](validation.json) reproduces the frozen p=.220 L3-L7
  `w0` crossing and validates both 192-disorder raw shards as one complete
  384-disorder baseline.
- [`pilot_decision.json`](pilot_decision.json) and
  [`pilot_decision.csv`](pilot_decision.csv) are the terminal fail-closed
  decision: all three q values remain strictly negative at both `.230` and
  `.240`; all 95% CIs exclude zero and all pass fractions are at least
  `0.979`.  The pilot therefore stops with `no_flip_by_0240` for every q.
- No `.225` or `.235` wave was selected, no strict bracket was formed, and no
  384-disorder production was launched.  The pilot used 12 of the allowed
  maximum 18 new cells.
- Seed range `950000..950383` was checked against shared remote manifests and
  had no overlap before staging.
- nd-3 has 96 logical CPUs.  The launcher measures existing load immediately
  before every wave and uses `min(70, 96 - measured_busy - 16)` workers at
  positive niceness; it never launches a second worker pool for the same wave.

`run_legacy_pilot.py` owns atomic task checkpoints and create/resume semantics;
`analyze_legacy_pilot.py` owns fail-closed raw validation and the adaptive
state machine.  `launch_yuany_nd3.sh` only creates unique shared paths and has
no deletion command.  All nd-3 paths are on the same `/home/DATA1` filesystem;
no node-to-node copy is needed.  `check_remote_wave.py` is a read-only waiter
that accepts success only after the exact wave identity, terminal metadata,
and all declared merged-cell hashes agree.  Both completed waves passed that
check before collection and analysis.

## Remote provenance

- Run: `3dfinal_legacy_pilot_20260811T131837Z_nd3`
- `p0230`: seed base `950000`, 69 workers at niceness 10, 288/288 tasks and
  6/6 merged cells complete.
- `p0240`: seed base `950048`, 69 workers at niceness 10, 288/288 tasks and
  6/6 merged cells complete.
- Both waves used one process pool, one thread per worker, and remained below
  the hard cap of 70 workers.  Existing other-user CPU load was left alone.
- Summed single-thread task wall time was 328.54 hours (`p0230`: 164.08;
  `p0240`: 164.46).  This is an aggregate task-wall-time accounting, not a
  hardware CPU counter.  Stopping before `.235` avoided another comparable
  pilot wave and all unauthorized production work.
- Shared run root:
  `/home/DATA1/users/yuany/.single_shot/runs/3dfinal_legacy_pilot_20260811T131837Z_nd3`
- Exact config hashes, timestamps, and every merged-cell SHA256 are preserved
  in [`source_manifest.json`](source_manifest.json).
