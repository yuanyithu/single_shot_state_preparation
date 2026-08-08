# Validation 004: production scan and committed replay on nd-3

Status: **`PASS`**. Authorizes Validation 005. No physical result is published
here.

## What ran

Run root `~/.single_shot/runs/exp104_ensemble_v1_002`, nd-3 only, 64 workers, one
decoder thread, inside `screen`, from 2026-08-08T09:48:52Z to
2026-08-08T10:43:00Z. Executed entirely through the verified-archive wrapper,
whose bytecode gate exits 67; the tree was verified clean before and after.

- Deployment manifest SHA256
  `66ed00243e39312c05f2fb5454ec36751464a76aea61c8883bc364de1acf6d9b`, archive
  SHA256 `3832ee2f0f12ed4ad02e91542b89f2a9c962f37eaace9199e59a007159550b38`.
- Config SHA256 `85616f2679a64ffb44c87c7488918385e8e5506d2e8501ecf7f7d4259509db2a`;
  registry SHA256
  `7e40ff18fdf4fd52476894dc21caa516e16a1b97cdfd2a9ad9f803c709f315d4`.

## Scan

`scan.json`: **`PASS`**. All `778` planned tasks, `778` fresh, none resumed, no
unplanned NPZ evidence before or after. Wall time **45.9 minutes** against the
4.09 hours the resource gate projected, which is the expected margin: the
projection takes per-trial cost as the maximum over the benchmarked grid points
and applies it to every point.

12,000 codes, 9 grid points, 4 trials per code and p, **432,000 trials**.

## Replay

`replay.json`: **`PASS`**, scope `committed_subsample`, `83` of `83` expected
tasks, zero non-`PASS` results, **60,120 trials replayed bit for bit**.

The subsample was fixed in Validation 003's resource preflight before the first
production task ran: 2, 3, 6, 11, 21 and 40 tasks for `m = 3..8`, block 0 of every
`m` included unconditionally. Each replayed task reconstructs its codes from
their seeds, builds an independently constructed decoder, and is required to
match on failure flags, logical labels, syndrome match, convergence flags,
iteration counts and all three stream digests. Any single mismatch would have
invalidated the whole run.

## Authority

Technical gate. The aggregate, the certification and the crossing location are
Validation 005. This validation grants no exp102 authority and clears no exp102
blocker.
