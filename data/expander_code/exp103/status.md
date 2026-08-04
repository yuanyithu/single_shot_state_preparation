# exp103 status

## Current state

**`BLOCKED_LOCAL_RESOURCE_PREFLIGHT`**

exp103 is a fresh BpLSD decoder-MC line for `q=0` block logical failure. No formal
measurement raw, crossing bracket, asymptotic threshold, or physical result exists
yet. Validation 001 passed on the frozen source/config, but Validation 002 blocks
both three-size stages under the preregistered local resource gates. No formal
raw was launched. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103 cannot change
that authority.

## Current gates

1. Stage 1 fails only the core-hour gate: `104.0546 > 100` reserved core-hours.
2. Stage 2 fails core-hour and wall gates: `1020.9320 > 100` and `64.6832 > 12` hours.
3. The contract forbids automatic remote transfer or a formal scan after either failure.

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

- Validation 002: `BLOCKED_LOCAL_RESOURCE_PREFLIGHT`; both stages are closed.
- Validation 001: `PASS`; 105 exp103 and 75 focused exp101/exp102 regressions passed.
- No formal raw, decoder crossing bracket or physical result exists.
