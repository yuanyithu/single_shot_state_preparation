# exp103 decoder amendment v3

Amendment identity: `exp103.decoder_amendment.v3`.
Resulting experiment identity: `exp103.decoder_mc.v2`.

This amendment records the user's 2026-08-06 authorization to replace the
frozen decoder with a deterministic one after Validation 005 proved that the
previous choice cannot satisfy the contract's bit-exact replay gate. It
supersedes the "Frozen decoder and outcome" clause of `EXPERIMENT_CONTRACT.md`
and the decoder identity clauses of `REMOTE_EXECUTION_AMENDMENT.md` and
`REMOTE_EXECUTION_AMENDMENT_V2.md`. Every other clause of the contract and of
amendment v2, including the approved resource caps, remains in force verbatim.

Validations 001-005 keep their original terminal states and remain immutable
evidence. The `exp103.decoder_mc.v1` Stage 1 raw stays on the server as
historical evidence of the defect; the experiment, raw and aggregate schema
bumps below make it structurally impossible to mix that raw with v2 evidence.

## Why the decoder changed

Validation 005 measured, on two platforms and two independent code paths, that
the `ldpc` 2.4.1 `BpLsdDecoder` is not a deterministic function of its input:
its Localised Statistics Decoder stage can return a different legal correction,
in a different logical class, for an identical syndrome. Belief propagation is
exactly deterministic; only the LSD tie-break is not. The rate is about `1e-5`
per trial on the frozen Linux build and about `1.1e-2` on macOS, and it is
concentrated wherever BP fails to converge, which is essentially everywhere at
`m>=4` and `p>=0.06`. No exposed parameter controls it: `omp_thread_count` is
marked `NotImplemented` upstream, and pinning `OMP_NUM_THREADS`, `MKL` and
`OPENBLAS` to one thread does not change the rate.

The contract assumed a deterministic decoder and built a bit-exact replay gate
on that assumption. Rather than weaken the gate, this amendment removes the
assumption's violation at its source.

The change is outcome-blind. No aggregate, curve, contrast, crossing or `p_c`
was ever computed from the v1 Stage 1 raw, so nothing here was selected on a
physical result. The trigger is a documented property of the tool.

## What changes

| Item | Frozen value |
|---|---|
| Decoder class | `ldpc.BpOsdDecoder` (module `ldpc.bposd_decoder._bposd_decoder`) |
| OSD parameters | `osd_method=osd_0`, `osd_order=0` |
| BP parameters | unchanged: `bp_method=product_sum`, `max_iter=n`, `schedule=serial`, natural serial order |
| `omp_thread_count` | still passed as `1`, but treated as a no-op; single-threading is enforced by the pre-existing environment gate requiring `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` |
| Experiment identity | `exp103.decoder_mc.v2` |
| Raw / aggregate / local config schema | `exp103.raw.v2` / `exp103.aggregate.v2` / `exp103.config.v2` |
| Remote config schema / execution profile | `exp103.config.remote.v3` / `exp103.remote_execution.v3` |
| Canonical config artifacts | `config/decoder_mc.v2.json`, `config/decoder_mc.remote.v3.json` |
| Objective token | `bposd_block_logical_failure_crossing_q0` |
| Evidence field name | `bplsd_binary*` renamed to `decoder_binary*` throughout raw, replay, aggregate and qualification evidence, so the schema stays self-describing |

`osd_0` with `osd_order=0` is the standard ordered-statistics baseline for
QLDPC codes and is the natural counterpart of the previous `lsd_order=0`.

### New determinism gate

The frozen regression suite gains a determinism test, and it runs inside remote
qualification like every other frozen test: two freshly constructed decoders,
given an identical syndrome sequence on a code and error rate where BP does not
converge, must return byte-identical corrections. This is the check whose
absence let the v1 defect reach a formal stage. A failure closes the gate.

## What does not change

Value-for-value frozen from the original protocol and amendment v2:

- all 48 registry codes with equal weight, including every `d=2` member;
- `sector=x_error`, `H_check=H_Z`, perfect syndrome, `q=0` Bernoulli-X noise;
- the 13-point `p=0.02,...,0.14` grid, four 2,500-trial shards per code-p, and
  10,000 trials per code-p;
- the master seed, registry-bound seed derivation, and the benchmark,
  measurement, replay and bootstrap namespaces, so every measurement seed is
  bit-identical to the value the v1 config would have derived;
- the residual definition `r = e xor e_hat`, the failure rule, the independent
  GF(2) row-space audit scorer, and the rule that an exception or illegal
  correction invalidates and preserves the whole shard;
- complete independent replay, fail-closed aggregation, the 20,000-draw
  bootstrap family, the crossing classifications, and the publication mask;
- the single-node `nd-3` profile with exactly 64 workers, the verified archive
  deployment, the clean-source exit-67 bytecode gate, immutable raw, and staged
  SHA-verified retrieval;
- the per-stage caps of `10000` reserved core-hours, `96` wall-hours and `128`
  GiB peak RSS. Measured per-trial cost of BP+OSD-0 is `0.92`, `0.93` and
  `0.99` times BpLSD at `n=225`, `625` and `1600`, so the approved ledger
  covers the switch; a fresh outcome-blind preflight still has to pass both
  stages before any measurement, exactly as before;
- the prohibition on asymptotic threshold, critical-exponent, FSS, `q_top`,
  MLD or preparation-channel claims.

## Revised validations and authority

1. Validation 006 freezes the v2 decoder identity and passes the complete local
   regression suite, including the new determinism test.
2. Validation 007 requalifies the exact `nd-3` environment under the v3 config
   and applies the outcome-blind remote resource gate to both stages.
3. Validation 008 runs `m=3,4,5`, then complete independent bit-exact replay,
   and produces the restricted technical evidence.
4. Validation 009 runs `m=6,7,8` whenever Validation 008 is technically
   complete and its replay passes, unconditional on all Stage 1 curves.
5. Validation 010 loads all 48 codes through the publication loader and reports
   the finite-grid crossing classification and the checkpoint.

This amendment authorizes only the exp103 work above. It does not alter exp102
evidence or status, clears no exp102 blocker, and grants no exp102 remote,
formal, held-out, restricted or production authority.
