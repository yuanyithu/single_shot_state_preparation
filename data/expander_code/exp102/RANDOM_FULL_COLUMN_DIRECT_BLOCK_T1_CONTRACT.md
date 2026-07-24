# Exp102 q=0 direct-block random-full-column T1 m8 contract

Version: `exp102.q0_random_full_column_direct_block.t1_m8.v1`

This is a fresh diagnostic using `RFCG-C24-DPB12-S1`.  It is not a
continuation of validation 052 and reuses none of its schedule seeds or raw.
Its maximum authority is `DIAGNOSTIC_DIRECT_BLOCK_T1_M8_VIABLE`: a pass may
authorize a separately frozen fresh m6 T1, then fresh T/2T HARD2 work.  It
cannot authorize formal tuning, held-out, production, or a physical q_top.

## Scientific target and implementation boundary

The only cell is `m08_c06,p=.04,d00,attempt022`.  Every state must satisfy

```text
H_Z e = y,
pi(e | y) proportional to (.04/.96)^|e|.
```

The planted error appears only as the adversarial P initial state.  It is
never used in a conditional weight, move selection, energy, or acceptance
rule.  One state-independent PortablePrng clock chooses one of 24 collapsed-B
columns and samples its exact `2^24` conditional.  Direct positive weights are
summed in fixed `2^12` blocks, followed by a second pass through only the
selected block.  Every observation independently redraws `A|B` exactly.

Validation 054 froze the implementation hashes, portable block-subtotal and
trajectory transcripts, and underflow certificate.  The final T1 source must
keep both sampler files byte-identical and must itself pass the same three-node
054 preflight before measurement.  The T1-specific three-node preflight must
also have exact transcript consensus and a replay-budgeted projection no more
than 7200 seconds per trajectory.

## Initial states and the zero-state trap

There are eight independent trajectories in each family:

- `P`: legal planted state, with independent RNG streams;
- `U`: independent exact-K0 uniform hard-coset states;
- `M0/M1`: two truth-free weight-62 MAP anchors with B distance 6;
- `S`: eight truth-free low-energy states with distinct logical labels and B
  blocks, including one same-B/different-logical control.

The physical all-zero state is outside this nonzero hard coset.  Shifted zero
is exactly P.  Starting every chain there would remove the adversarial test and
could turn common collapse into apparent agreement.  MAP/S artifacts are
initialization-only and never enter the kernel.

## Frozen clocks, raw, and statistics

Each trajectory has 2048 burn and 8192 measurement updates.  The 40 task
identities, fresh seeds, fresh logical/B characters, and 14/13/13 node
ownership are frozen before any measurement result.  Four workers per node
and full bit-exact replay are mandatory.  Raw records every selected, old, and
new B column, B trace, packed physical state, label, weight, B weight,
likelihood, eight time blocks, counters, seeds, identities, and timings.

The primary estimator is the character U-statistic q_top using all 64 basis
characters and 4096 frozen nonbasis characters.  Full-label collision is only
a diagnostic.  Each family and every family pair must pass all validation 052
gates without modification:

- q_top SE and absolute/SE-scaled family agreement;
- full-label and B-character D2 upper bounds;
- full/B normalized weight and B log-likelihood agreement;
- all B bit, row, column, and 64 dense characters;
- logical basis and 64 frozen logical diagnostic characters;
- split Rhat, nondegenerate bulk ESS, and constant-character burn crossing;
- per-chain B-column/label movement and bidirectional M0/M1 basin visits.

State or label changes alone are not evidence: exact `A|B` redraws may change
logical labels while the slow B variable remains trapped.  Conversely, a
low-temperature path need not visit every logical direction if the posterior
mass is concentrated; the required conclusion is distributional agreement
under the frozen observables and adversarial starts, not a prescribed visual
trajectory pattern.

## Fail-closed interpretation

Identity, algebra, nonfinite, implementation-hash, portable-reference, or
replay failure is `CONFLICT`.  Resource failure is `RUNTIME_EXHAUSTED`.
Scientific failure is `UNRESOLVED_DIRECT_BLOCK_T1_M8`, not `IMPOSSIBLE`.
No failure may be repaired by adding clocks, starting all chains at P, dropping
U/S/MAP starts, changing characters, weakening gates, or reusing raw.
