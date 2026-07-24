# Exp102 q=0 random-full-column T1 m8 diagnostic contract

Version: `exp102.q0_random_full_column.t1_m8.v0`

This is a fresh diagnostic, not a continuation of validations 048/049 and not
a formal exp102 sampler certification.  Its maximum authority is
`DIAGNOSTIC_RFCG_T1_M8_VIABLE`.  A pass only permits a fresh m6 T1 diagnostic,
then separately frozen T/2T HARD2 work.  It cannot authorize tuning, held-out,
production, or a physical `q_top` result.

## Target and transition

The sole cell is `m08_c06,p=.04,d00,attempt022`.  Every sampled state must obey

```text
H_Z e = y,
pi(e | y) proportional to (.04/.96)^|e|.
```

The planted error is never used in the energy, conditional probability, move
selection, or acceptance.  It is used only as the deliberately adversarial P
initial state.  One fixed clock chooses one of 24 collapsed-B columns with a
state-independent `PortablePrng` draw, enumerates all `2^24` column values, and
draws the exact conditional.  Every measurement clock independently redraws
all A columns exactly from `A|B`.  This is a random-scan mixture of exact
heatbaths and therefore preserves the collapsed hard-coset posterior.

## Why the starts are not all zero

The physical all-zero state is outside the nonzero hard syndrome.  In shifted
coordinates, zero is exactly the already represented planted P start.  Using
only that start would erase the main convergence challenge rather than improve
the Markov kernel.

There are eight independent trajectories in each frozen family:

- `P`: the legal planted state, shared state but independent RNG streams.
- `U`: independent exact-K0 uniform hard-coset states; these expose failure to
  descend from the broad/high-energy part of the support.
- `M0` and `M1`: two truth-free weight-62 MAP artifacts with the same logical
  label but B Hamming distance 6; these expose a known low-energy B bridge.
- `S`: eight deterministic truth-free low-energy catalog states with distinct
  logical labels and distinct B blocks.  One S state deliberately shares the
  M0 B block while having a different logical label, separating A/logical
  redraw activity from real B transport.

The S selection is frozen by
`greedy_maximin_B_distance_then_low_weight_then_catalog_index`, before any T1
raw.  The source artifacts and the resulting state bytes are hashed in the
control artifact.  MAP/catalog artifacts are initialization-only and never
enter the transition.

## Clocks, ownership, and runtime

Each trajectory has 2048 burn updates and 8192 measurement updates.  The 40
tasks, all seeds, character masks, and ownership across nd-1/nd-2/nd-3 are
frozen before preflight.  Four workers per node are fixed in the config.

Each Linux node runs a four-process contention probe.  The factor-two
projection, which budgets the mandatory bit-exact replay, must not exceed 7200
seconds per trajectory on any node.  All three nodes must agree exactly on the
mass-table and portable transcript digests.  Measurement cannot start unless
the aggregate preflight status is `PASS`.

## What must converge

State changes and label changes are diagnostics, not evidence by themselves.
The primary estimate remains the frozen character U-statistic for `q_top`:
all 64 basis characters plus 4096 frozen nonbasis characters.  Label collision
is diagnostic only.

Every initialization family and every pair of families is gated on:

- `SE_total(q_top) <= .03`;
- `|delta q_top| <= .04` and `<= 3 SE_delta + .005`;
- `max(0,D2_norm) + 3 SE_D2 <= .04`;
- normalized full-weight and B-weight agreement;
- collapsed-B log-likelihood agreement;
- all B single-bit, row, column, and 64 dense characters;
- full weight, B weight, B likelihood, all logical basis characters, and 64
  frozen nonbasis logical characters with split `Rhat <= 1.05` and
  nondegenerate bulk ESS `>=400`;
- actual B-column changes and bidirectional visits across the frozen M0/M1
  reaction-coordinate basins.

For any B character that is constant throughout all measurement chains while
the legal initial states contain both signs, every opposite-sign chain must
cross to the common sign during burn.  This prevents common-basin freezing from
being mislabeled as convergence.  Raw values are not clipped.

## Failure interpretation

Any algebra, identity, source, hash, or replay mismatch is `CONFLICT`.  Runtime
failure is `RUNTIME_EXHAUSTED`.  A scientific gate failure is
`UNRESOLVED_RFCG_T1_M8`; it is not `IMPOSSIBLE` and cannot be repaired by
adding clocks, dropping U/S/MAP starts, weakening gates, or reusing raw.
