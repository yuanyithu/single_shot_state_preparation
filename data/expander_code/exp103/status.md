# exp103 status

## Current state

**`BLOCKED_REPLAY_NONDETERMINISM`**

Stage 1 (`m=3,4,5`) generated all 312 code-p tasks and 1248 `VALID` shards on
`nd-3` under the user-authorized `exp103.remote_execution.v2` caps, but the
frozen bit-exact full replay returned `INVALID` on 53 shards. The cause is a
property of the frozen decoder, not of the code, node, or budget: the `ldpc`
2.4.1 `BpLsdDecoder` LSD stage is randomized, so an identical syndrome can
return a different legal correction in a different logical class. Belief
propagation itself is exactly deterministic. The affected region is `m>=4`
with `p>=0.06`, where BP stops converging and LSD runs on nearly every trial;
`m=3` reproduced perfectly.

No aggregate, curve, contrast, crossing or `p_c` exists, and none may be
quoted from the Stage 1 raw. Validations 006 and 007 stay closed. exp102
remains `BLOCKED_BEFORE_REMOTE`; exp103 has cleared none of its blockers.

**Awaiting user decision.** Resuming requires an amendment that either
redefines replay for a randomized decoder or replaces the decoder identity
with a demonstrably deterministic one. Neither may be adopted without explicit
authority, and a failed gate never authorizes weakening itself.

## Current gates

1. Validations 001-004 keep their original terminal states; nothing is
   reclassified. Validation 004 (`PASS`) remains the valid v2 resource gate.
2. Validation 005 is `BLOCKED_REPLAY_NONDETERMINISM`: scan `PASS`
   (1248/1248 `VALID`, none resumed), replay `INVALID` (1195 exact, 53 not).
3. The contract's assumption that the decoder is a deterministic function is
   false. `omp_thread_count` is `NotImplemented` upstream, and pinning
   `OMP_NUM_THREADS`/`MKL`/`OPENBLAS` to 1 does not remove the effect, so no
   frozen knob can restore bit-exactness.
4. Measured across 153,500 paired trials, including a complete re-decode of
   all 53 failing shards, the physical failure flag never disagreed (95% bound
   `2.0e-5` per trial), nor did `syndrome_match`, `bp_converged`,
   `bp_iterations`, or any per-shard error-stream hash. Only the degenerate
   logical-label representative disagrees (37 in 132,500). This is evidence
   about the estimand, not authority to report any number.
5. `ldpc.BpOsdDecoder` is exactly deterministic at the same operating points
   (0 differences in 7,000 decodes) and costs `0.92`-`0.99` times BpLSD per
   trial, so a deterministic decoder is available within the approved ledger.
6. Stage 1 raw stays immutable on the server under
   `~/.single_shot/runs/exp103_remote_v2_001`; it is neither deleted, reused
   as a formal result, nor re-run in place.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: frozen scientific and statistical contract.
- `REMOTE_EXECUTION_AMENDMENT.md` / `_V2.md`: v1 profile and authorized v2 caps.
- `config/decoder_mc.remote.v2.json`: exact qualified v2 remote identity.
- `validation/004_remote_gate_v2_20260805/`: v2 qualification and gate `PASS`.
- `validation/005_stage1_replay_nondeterminism_20260806/`: scan and replay
  evidence, failing shard list, nd-3 and macmini determinism diagnostics, and
  the reproducible local probe.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 005: scan SHA256 `c8a8c529...6835c` `PASS`; replay SHA256
  `fb09b75a...345507` `INVALID`; 53 failing shards, all `m=4`/`m=5`, all
  `p>=0.06`.
- Validation 004: qualification `1e71fb84...ff31fc5` (203/203); preflight
  `fb208777...404c11` `PASS_ALL_STAGES`.
- Validation 003: `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` under v1 caps; unchanged.
