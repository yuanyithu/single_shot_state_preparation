# exp103 remote execution amendment v2

Amendment identity: `exp103.remote_execution.v2`.

This amendment records the user's 2026-08-05 authorization to re-derive the
remote resource caps from the measured Validation 003 evidence and to execute
the unchanged `exp103.decoder_mc.v1` protocol under those caps. It supersedes
only the resource-cap and evidence-path clauses of `exp103.remote_execution.v1`
(`REMOTE_EXECUTION_AMENDMENT.md`); every other v1 clause remains in force
verbatim. Validation 003 and its `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` terminal
status remain immutable evidence and are not reclassified.

## Why the caps changed and why nothing else did

Validation 002 measured the frozen nine-task benchmark on macmini. The v1
remote caps (1200 reserved core-hours, 24 predicted wall-hours per stage)
covered the local-measured Stage 2 reserve (1020.93 core-hours) with about 17%
headroom under the implicit assumption that one nd-3 core matches one macmini
core on this workload. Validation 003 measured the same frozen benchmark on
nd-3 and found a per-core slowdown of about 9.3-9.9x (Stage 1 reserve
104.05 -> 1027.40; Stage 2 reserve 1020.93 -> 9520.39 core-hours), so Stage 2
failed both the core-hour and wall caps. The remote preflight gate exists to
catch exactly this cross-architecture drift before formal compute, and it did.

The scientific protocol is untouched: v2 changes no code, p point, shard or
trial count, decoder field, seed or namespace, statistical or crossing rule,
and no publication mask. Measurement seeds derive only from the frozen
scientific identity and shard key, so the v2 execution identity cannot change
any trial stream; the seed-equality regression against the local canonical
config remains part of the frozen test suite.

## v2 frozen remote execution profile

Identical to v1 except the three starred rows:

| Item | Frozen value |
|---|---|
| Profile / config schema | `exp103.remote_execution.v2` / `exp103.config.remote.v2` |
| Canonical config artifact | `config/decoder_mc.remote.v2.json` |
| Entry route / compute node | `ssh yuany`; exactly one node, `nd-3` (unchanged) |
| Process workers / decoder threads | `64` / `omp_thread_count=1` (unchanged) |
| Stage reserve formula | `2 * (generation + full replay + analysis + fixed overhead)` (unchanged) |
| Per-stage reserved core-hour cap (*) | `10000` core-hours (v1: 1200) |
| Per-stage predicted wall cap (*) | `96` hours (v1: 24) |
| Projected peak RSS cap | `128` GiB (unchanged) |
| Remote environment identity (*) | v1 environment, prefix and frozen BpLSD extension, re-bound under the v2 config SHA |

Ledger derivation (permanent discipline 11): the caps are derived backward
from the frozen nd-3 measurement itself. Validation 003 recorded reserved
totals of 1027.3980 (Stage 1) and 9520.3885 (Stage 2) core-hours and predicted
walls of 9.0109 and 75.3624 hours at 64 workers. The v2 caps admit those
totals with about 5% core-hour and 27% wall margin for benchmark timing noise.
The user-approved reservation is the two-stage total of about 10548 core-hours;
the realistic burn is materially lower (roughly 1300-2000 core-hours) because
the frozen estimator applies each stage's worst per-trial anchor uniformly
across its full grid and then reserves 2x. If a fresh preflight exceeds a v2
cap, the honest terminal state is again `BLOCKED_REMOTE_RESOURCE_PREFLIGHT`;
it is not repaired by re-rolling the benchmark under load or by weakening a
cap without new user authority.

## Gate mechanics under v2

1. Environment qualification and the outcome-blind resource preflight rerun on
   nd-3 under the v2 config identity with the same frozen benchmark panel
   (`m3/m5/m8 x p=.02/.08/.14`, benchmark namespace, 20 trials per task, no
   logical outcome saved or inspected). Selecting a different benchmark panel
   remains forbidden. The Validation 003 evidence is not re-bound to v2; v2
   produces its own qualification and preflight evidence.
2. Both stages must independently pass all three v2 caps before any
   measurement-namespace trial, exactly as in v1.
3. Revised validation numbering: 004 = v2 qualification + remote resource
   gate; 005 = Stage 1 `m=3,4,5` scan, full replay and technical report;
   006 = Stage 2 `m=6,7,8` scan and full replay, unconditional on all Stage 1
   curves exactly as in v1; 007 = full loader-verified crossing and the
   five-validation checkpoint. Committed evidence paths are frozen in the
   package constants at the v2 source commit.
4. Everything else is inherited from v1 unchanged: verified archive
   deployment, clean-source execution with the exit-67 bytecode gate, screen
   with explicit worker count, immutable raw and replay evidence, staged
   SHA-verified retrieval, fail-closed aggregation, the publication loader,
   and the prohibition on asymptotic-threshold, exponent, FSS, `q_top`, MLD or
   preparation-channel claims.

This amendment grants no exp102 authority of any kind and clears no exp102
blocker.
