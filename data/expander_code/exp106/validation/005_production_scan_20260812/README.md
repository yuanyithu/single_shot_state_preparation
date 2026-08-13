# Validation 005: production scan, committed replay and aggregation

Status: **`PASS`**. Controlled category: `MEASUREMENT`.

Run root `~/.single_shot/runs/exp106_q001_v1_003` on nd-3, 75 workers, one
decoder thread, inside `screen`, from 2026-08-12T14:25:12Z to
2026-08-12T23:54:10Z.

## Scan

`scan.json`. **13,468 of 13,468 tasks fresh, zero resumed, `PASS`**, in
`30,161 s = 8.38 wall hours`. 108,400 codes, 3,252,000 trials at `q = 0.01` over
ten `p` from 0.005 to 0.07, three trials per code and `p`.

Against a predicted `12.82` wall hours, so the projection was conservative by a
factor of 1.5 -- less than exp105's 3.2, which is what a grid living entirely in
the expensive half of the `p` range looks like: there is less headroom between
the max-over-`p` upper bound and the mean when every point is expensive.

## Replay

`replay.json`. **1,354 of 1,354 committed tasks `PASS`, 331,410 trials bit-exact.**

The subsample was fixed in the Validation 004 preflight, before any production
task ran: ten percent of tasks per `m` plus block 0 of every `m` unconditionally,
from a seed derived from the master seed, the replay namespace, the registry hash
and the frozen `q`. Per `m`: 69, 60, 65, 63, 61, 1036.

Replay is not a rerun of the worker. It builds its own decoder and reconstructs
the logical criterion through `audit_scorer` -- from the RREF pivot rule of `H_Z`
rather than through the exp101 frame the worker used -- and requires agreement on
failure flags, logical labels, readout-match flags, convergence flags, iteration
counts and all four stream digests. A single mismatch invalidates the entire run.

## Aggregation

`overall_status COMPLETE`, `replay_status PASS`,
**1,084,000 of 1,084,000 code-`p` cells `REPORTABLE`**, terminal
`EXP106_NO_CERTIFIED_CROSSING`, aggregate SHA256 `389801a1...4ec8d416`.

## Retrieval

`scan.json`, `replay.json` and `aggregate/ensemble_crossing.npz` were copied into
a staging directory and their SHA256 compared against the remote originals before
being moved into place: `dca99fc2...`, `51a7eb87...`, `389801a1...`, all three
identical. Raw stays on nd-3 -- 13,468 NPZ files, not tracked in Git.

## Two things worth recording about how this run was reached

**The gates blocked four times, and every block was a real defect.** The
deployment builder refused a bundle whose config predated exp106; nd-3 rejected
the cost benchmark because `verify_remote_deployment` hard-coded the production
config filename (wrong at three sites); the resource gate blocked at
`2001.9/1800`; and qualification caught `224` passed against `223` expected,
because a test added during Amendment 1 had not moved its count. None of these
was a false alarm and none was worked around.

**One gate interaction is circular and had to be understood rather than patched.**
A gate report records `config_sha256`, and the report must be committed and
re-deployed before the next gate will read it -- but rebinding the configs to the
new HEAD changes `config_sha256`, so the report stops matching. The rebind was
never required: the deployment builder asks only that the diff between the
config's `source_commit` and HEAD be empty over the **frozen execution paths**,
and `validation/` is deliberately outside them. Configs stay bound to the commit
that froze the package while gate evidence accumulates on top.

## Evidence in this directory

- `scan.json` -- 13,468/13,468 fresh, `PASS`, wall seconds, worker count, and the
  five-way identity tuple
- `replay.json` -- 1,354 results, each with its own raw SHA256 and four stream
  digests

## Authority

Measurement only. The terminal status is certified in Validation 006 through the
loader, and an aggregate the loader will not accept is not a result.
