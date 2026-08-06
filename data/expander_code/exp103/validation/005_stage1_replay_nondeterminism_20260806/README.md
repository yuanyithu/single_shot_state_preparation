# Validation 005: Stage 1 generated, full replay failed on decoder nondeterminism

Status: `BLOCKED_REPLAY_NONDETERMINISM`.

Stage 1 (`m=3,4,5`) generation completed exactly as planned, but the frozen
bit-exact full replay returned `INVALID`, so no Stage 1 aggregate, technical
report, preliminary report, or Stage 2 authorization exists. There is no
physical result. The cause is a property of the frozen decoder itself, not of
the measurement code, the node, or the resource plan: **the `ldpc` 2.4.1
`BpLsdDecoder` is not a deterministic function of its input.** No re-run can
repair this, and the failure is not repaired by relaxing, resampling, or
partially reporting.

## What ran

Deployment commit `8b5d18d813fffafea12fbf4dff3de1ec88ee68e9`, manifest SHA256
`1437108e7a521a3402812fb2233adfd570a4c43701e30f99dbc28ae9449eac0f`, archive
SHA256 `0d83acdbf584a77851a0655848c60be97008cea04387db399ad5c8ef4fa848c3`,
under `exp103.remote_execution.v2` on `nd-3` with 64 workers, run root
`~/.single_shot/runs/exp103_remote_v2_001` (retained). Wall clock
2026-08-05T08:16:08Z to 12:34:42Z.

- `SCAN_STAGE1.json` (SHA256
  `c8a8c52931bb49d05f423be8a5ab67aa64dc71aada5527401924a4b4cea6835c`):
  `PASS`, all `312` code-p tasks and `1248` fresh shards saved `VALID`, none
  resumed, no unplanned NPZ.
- `REPLAY_STAGE1.json` (SHA256
  `fb09b75afe999606af01669565d817c75c9c41696135bc386166ef9a49345507`):
  `INVALID`. `1195/1248` shards replayed bit-exactly; `53` did not, of which
  `50` stopped at a `trial_replay_mismatch` and `3` matched all 2500 trials on
  every compared field yet produced a different `correction_stream_sha256`.

The 53 failing shard identities are in `failing_shards.json`. They are
confined to `m=4` (21) and `m=5` (32) with none at `m=3`, and to `p>=0.06`
with the mode at `p=0.10-0.13`. That is exactly the region where belief
propagation stops converging and the Localised Statistics Decoder does the
work on essentially every trial.

## Root cause, measured two ways

`determinism_diagnostic.json` records a read-only nd-3 re-decode of two
failing shards and two passing controls, each decoded twice more. Independent
of that, `probe_bplsd_determinism.py` reproduces the effect locally on macmini
against a different compiled extension. Both agree:

- Belief propagation is exactly deterministic: `bp_converged` and
  `bp_iterations` never differed in any comparison, on either platform.
- The LSD stage is not. Re-decoding an identical syndrome returns a different
  correction at a strongly platform-dependent rate: order `1e-5` to `1e-4` per
  trial on the frozen Linux build (`53/576` shards in the affected region
  implies about `4e-5`; the two re-decoded failing shards give `2/5000`, biased
  high because they were selected as failures), against about `1.1e-2` per
  trial on the macOS build. Every differing trial had `bp_converged == false`.
- Every alternative correction is legal: it reproduces the same syndrome. The
  alternatives differ by a logical operator, so the recorded logical label
  changes with them.
- The behaviour is not a threading artifact and cannot be switched off. The
  contract freezes `omp_thread_count=1`, but the `ldpc` docstring marks that
  parameter `NotImplemented`; setting `OMP_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`,
  `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` leaves the local rate
  unchanged at `17/1500`. The only seed-like knobs exposed
  (`random_schedule_seed`, `random_serial_schedule`) govern the BP schedule,
  which this contract already fixes to a deterministic serial order and which
  the measurements confirm is deterministic.

## What this does and does not damage

The measured block failure flag was reproducible in every paired comparison
performed:

| Probe | Paired trials | Correction differs | Logical class differs | Failure flag differs |
|---|---:|---:|---:|---:|
| nd-3, two failing + two control shards | 10,000 | 2 | 2 | 0 |
| macmini `m04_c01`, p=0.11 | 3,000 | 29 | 29 | 0 |
| macmini `m04_c01`, p=0.04/0.06/0.08/0.10 | 8,000 | 37 | 36 | 0 |

At `p=0.04` the probe sits at `P_fail = 0.5205` in both passes, that is, in
the unsaturated region where a crossing would live, and the two passes agree
trial by trial on the physical outcome. Mechanistically this is consistent:
the tie that LSD breaks nondeterministically only arises when several
comparable-weight solutions in different logical classes exist, and in that
situation the decode already fails whichever class is chosen. Zero
disagreements in 21,000 paired trials bounds the failure-flag disagreement
rate at `1.4e-4` per trial (95%, rule of three), and for a randomized
decoder such disagreement is part of the estimand rather than an error in it.

### Complete check on the failing set

`failure_flag_reproducibility.json`, produced by
`diagnose_failure_flag_reproducibility.py`, re-decodes **all 53 failing shards**
(132,500 trials) and compares every field against the saved raw. It reports
disagreement counts only, never a rate or an aggregate, so it stays
outcome-blind:

| Compared field | Disagreements over 132,500 trials |
|---|---:|
| `failure_flags` (the primary observable) | **0** |
| `syndrome_match` | 0 |
| `bp_converged` | 0 |
| `bp_iterations` | 0 |
| `error_stream_sha256` exact, per shard | 53 of 53 exact |
| every re-decoded correction reproduces its syndrome | all legal |
| `logical_labels` (degenerate class representative) | 37, in 27 shards |

Across this check and the probes above, 153,500 paired trials produced zero
failure-flag disagreements, a 95% bound of `2.0e-5` per trial. Over the full
6,240,000-trial experiment that bounds any induced shift in a primary rate at
about `2e-5`, two orders of magnitude below the `~5e-3` binomial Monte Carlo
error at 10,000 trials per code-p. The seed derivation, RNG stream, code
construction, BP dynamics and scoring are all exactly reproducible; only the
LSD tie-break is not.

### A deterministic alternative exists at no extra cost

`probe_bposd_determinism.py` and `probe_decoder_cost.py` record that
`ldpc.BpOsdDecoder` (module `ldpc.bposd_decoder._bposd_decoder`) is exactly
deterministic at the same operating points where BpLSD is not: 0 differing
corrections and 0 differing logical classes in 7,000 decodes across
`m04_c01` at p=0.11 and `m05_c01` at p=0.12, for both `osd_0` order 0 and
`osd_cs` order 4, at BP non-convergence rates of 0.997 and 1.000. Per-trial
cost is `0.92`, `0.93` and `0.99` times BpLSD at n=225, 625 and 1600. A
decoder switch would therefore restore bit-exact replay without a material
change to the approved resource ledger.

So the exp103 primary observable appears well defined and reproducible, while
the contract's bit-exact stream comparison is unsatisfiable for `m>=4` at
`p>=0.06`. The contract assumed a deterministic decoder; that assumption is
false. This is a design defect in the frozen verification gate, discovered by
the gate itself, and it is outcome-blind: no aggregate, curve, contrast or
crossing was ever computed, so nothing here was learned from a physical
result.

## Authority

This validation authorizes nothing. Stage 2 remains closed, Validation 006 and
007 remain `NOT_STARTED`, and no crossing, `p_c`, or failure-rate number may be
quoted from the Stage 1 raw. The Stage 1 raw stays immutable on the server as
failed-gate evidence; it is neither deleted nor promoted. Resuming exp103
requires a user-authorized amendment that redefines replay for a randomized
decoder, or that replaces the decoder identity with a demonstrably
deterministic one; neither may be adopted on this validation's own authority.
exp102 remains `BLOCKED_BEFORE_REMOTE`.
