# exp103 status

## Current state

**`VALIDATION_001_PASS`**

exp103 is a fresh BpLSD decoder-MC line for `q=0` block logical failure. No formal
measurement raw, crossing bracket, asymptotic threshold, or physical result exists
yet. Validation 001 passed on the frozen source/config and authorizes only the
local resource preflight. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103 cannot
change that authority.

## Current gates

1. Validation 002 must pass the separate local resource gate for each three-size stage.
2. Formal raw requires committed and pushed source/config plus exactly eight workers.

If a stage fails its resource gate, record `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` and
do not move it to remote. A no-crossing or inconclusive complete result is valid
science and must not trigger code selection, resampling or a changed grid.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: scientific, statistical and authority contract.
- `config/decoder_mc.v1.json`: frozen grid, panel, decoder, environment and seeds.
- `validation/INDEX.md`: numbered evidence ledger.
- `raw/`: immutable generated shards; not a source workspace.
- `final_results/`: publication-loader inputs and compact reports when authorized.

## Latest evidence

- Validation 001: `PASS`; 105 exp103 and 75 focused exp101/exp102 regressions passed.
- No formal raw or decoder crossing result exists.
