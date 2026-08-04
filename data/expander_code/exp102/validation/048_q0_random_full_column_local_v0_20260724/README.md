# Validation 048: random-scan full-B-column local transport

This is the fresh local-only execution of
`exp102.q0_random_full_column.local.v0`.  The mathematical and scientific
red-team review is in
[`RANDOM_FULL_COLUMN_REVIEW.md`](../../reviews/RANDOM_FULL_COLUMN_REVIEW.md), and the
immutable configuration is
[`q0_random_full_column.local.v0.json`](../../config/q0_random_full_column.local.v0.json).

The experiment first times two outcome-blind exact column conditionals and
applies the frozen T1 factor-two wall gate.  Only a runtime pass permits the
12 fixed P/U/L trajectories.  Each trajectory uses 64 burn updates and 256
measurement updates; this is a short transport screen, not a posterior or
`q_top` result.

Run locally with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/048_q0_random_full_column_local_v0_20260724/run_local_transport.py \
  --config data/expander_code/exp102/config/q0_random_full_column.local.v0.json
```

No raw may be extended, replaced, or reused by a successor contract.

## Terminal infrastructure status

This v0 execution was operator-aborted after one `P_00.npz` raw when the
runner was found to build the `2^24` classical coset-mass table through the
pure-reference path.  The exact conditional itself had already started; the
abort therefore occurred after one raw rather than before raw generation.
Terminal status is `INFRASTRUCTURE_ABORTED_REFERENCE_MASS_PATH_AFTER_ONE_RAW`.

The manifest and lone raw are frozen in `failure_report.json` and forbidden
for every analysis, continuation, merge, and successor.  V0 has no transport,
posterior, or algorithm conclusion.  A successor must use a fresh contract,
seed namespace, manifest, output directory, and the already-tested Numba mass
table implementation; it may not resume the remaining eleven V0 tasks.
