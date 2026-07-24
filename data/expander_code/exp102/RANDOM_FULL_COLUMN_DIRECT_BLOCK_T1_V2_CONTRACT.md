# Exp102 q=0 direct-block random-full-column T1 m8 v2 contract

Version: `exp102.q0_random_full_column_direct_block.t1_m8.v2`

This is a fresh runtime-corrected successor to terminal validation 055.  It
does not reinterpret validation 055 as PASS and reuses none of its schedule
seeds or measurement raw.  The sampler, T1 clock, adversarial starts,
statistical gates, 7200-second cap, and maximum authority are unchanged.

## Scientific target and implementation boundary

The only cell is `m08_c06,p=.04,d00,attempt022`.  Every sampled state satisfies

```text
H_Z e = y,
pi(e | y) proportional to (.04/.96)^|e|.
```

The planted error appears only as the P initial state.  It is absent from the
direct conditional, energy, update clock, and exact `A|B` observation draw.
The method remains `RFCG-C24-DPB12-S1`: one state-independent PortablePrng
clock selects a collapsed-B column, exact positive weights are accumulated in
fixed `2^12` blocks, and a second pass samples within the selected block.

Both sampler source files and the portable reference remain byte-bound to
validation 054.  The final v2 source must pass the complete validation-054
three-node portable/runtime preflight and the v2 schedule-bound preflight
before measurement.

## Initial-state red team

There are eight independent trajectories in each family:

- `P`: the legal planted state with independent update streams;
- `U`: independent exact-K0 uniform draws from the hard coset;
- `M0/M1`: truth-free weight-62 MAP anchors separated in collapsed B;
- `S`: eight truth-free low-energy states with distinct B blocks and labels.

The physical all-zero state is illegal for this nonzero syndrome.  Zero in the
shifted coordinate is already P.  Starting all chains there would erase the
test of broad-support descent and inter-basin transport.  MAP/S states are
initialization-only and never influence the transition kernel.

## Runtime estimator

Validation 055 is terminally `RUNTIME_EXHAUSTED` because its frozen 10-update
probe linearly repeated fixed initialization and runner overhead.  V2 replaces
only that estimator.  On each node it launches two separate cold four-process
batches that mirror measurement concurrency:

```text
short: 8 burn + 128 measurement updates
long:  16 burn + 256 measurement updates
families: P, M0, S0, U0
```

Every probe performs the actual sampler and full bit-exact replay.  Workspace
allocation remains outside the timer, matching the measurement worker.  For
sampling and replay separately, each family fits

```text
slope = (t_long - t_short) / (272 - 136)
intercept = max(0, t_short - 136 slope)
projection = 2 * [(intercept_sample + 10240 slope_sample)
                + (intercept_replay + 10240 slope_replay)].
```

Both component slopes must be finite and strictly positive.  Otherwise the
node is `RUNTIME_ESTIMATOR_UNSTABLE`; no favorable fallback is allowed.  The
node uses the worst family projection and must remain at or below 7200 seconds.
Mass and all eight discrete transcripts must agree exactly across nd-1/2/3.

The design measures one cold-process startup rather than multiplying startup
by 10,240.  It does not lower the resource cap or use q_top, movement, or any
scientific outcome to select a resource tier.

## Frozen measurement and estimators

Each of the 40 trajectories retains 2048 burn plus 8192 measurement updates,
four workers per node, full replay, fresh task seeds, fresh logical/B
characters, and frozen 14/13/13 ownership.  Raw and analyzer requirements are
identical to validation 055.

Every family and pair must pass the unchanged validation-052/055 gates:
character-U-statistic q_top and SE, full-label and B-character D2, full/B
weight and B likelihood, all B bit/row/column/dense characters, logical
characters, split Rhat, bulk ESS, constant-character burn crossing,
B-column/label movement, and bidirectional M0/M1 basin visits.  Visible state
or label changes cannot substitute for collapsed-B mixing.

## Result permission

The maximum result is `DIAGNOSTIC_DIRECT_BLOCK_T1_M8_VIABLE`, which may only
authorize a separately frozen fresh m6 T1 and then fresh T/2T HARD2 work.  It
is not a physical q_top, formal tuning, held-out, `READY_FOR_FORMAL`, or
production result.  Runtime failure is not sampler failure; statistical
failure is `UNRESOLVED_DIRECT_BLOCK_T1_M8`, never `IMPOSSIBLE`.

No failure may be repaired by changing T1, the cap, the starts, the character
panel, or the gates, or by adding samples after inspection.  Any successor
requires another fresh contract/source/schedule/seeds/raw.
