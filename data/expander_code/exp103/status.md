# exp103 status

## Current state

**`PENDING_REMOTE_RESOURCE_PREFLIGHT`**

exp103 is a fresh BpLSD decoder-MC line for `q=0` block logical failure. No formal
measurement raw, crossing bracket, asymptotic threshold, or physical result exists
yet. Validation 001 passed and Validation 002 remains the immutable
`BLOCKED_LOCAL_RESOURCE_PREFLIGHT` result. The user has since authorized a
separately versioned, single-node remote execution amendment. Validation 003 has
not started, the remote environment and evidence SHA values are not frozen, and
no formal raw has been launched. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103
cannot change that authority.

## Current gates

1. The historical local gate remains failed: Stage 1 used `104.0546 > 100`
   reserved core-hours; Stage 2 used `1020.9320 > 100` and `64.6832 > 12` hours.
2. The remote amendment freezes one `nd-3` process pool with 64 workers,
   `omp_thread_count=1`, reserve multiplier 2, and per-stage caps of 1200
   core-hours, 24 wall-hours and 128 GiB peak RSS.
3. The exact isolated `exp103_remote_v1_env` prefix and package identity, Linux
   BpLSD binary SHA, remote config
   SHA, source identity and deployment manifest SHA remain
   `TO_BE_FROZEN_BEFORE_FORMAL`.
4. Formal Stage 1 is forbidden until Validation 003 qualifies the exact remote
   identity and reports PASS for both stage resource gates.

The remote profile does not change a scientific parameter, seed, grid, decoder
or statistic. A failed remote gate must be recorded as
`BLOCKED_REMOTE_RESOURCE_PREFLIGHT`; it cannot trigger a host change, code
selection, resampling or grid change. No-crossing and inconclusive remain valid
scientific outcomes after a complete valid run.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: scientific, statistical and original execution contract.
- `REMOTE_EXECUTION_AMENDMENT.md`: authorized remote profile and revised gates.
- `config/decoder_mc.v1.json`: original frozen scientific and local identity;
  the remote config is not yet frozen.
- `validation/INDEX.md`: numbered evidence ledger.
- `raw/`: immutable generated shards; not a source workspace.
- `final_results/`: publication-loader inputs and compact reports when authorized.

## Latest evidence

- Remote execution amendment: authorized but not yet environment-qualified;
  Validation 003 is `NOT_STARTED` and no formal raw exists.
- Validation 002: `BLOCKED_LOCAL_RESOURCE_PREFLIGHT`; both stages are closed.
- Validation 001: `PASS`; 105 exp103 and 75 focused exp101/exp102 regressions passed.
