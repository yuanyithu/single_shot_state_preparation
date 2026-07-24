# Validation 056: runtime-corrected direct-block m8 T1 diagnostic

This directory freezes
`exp102.q0_random_full_column_direct_block.t1_m8.v2`.  It is a fresh successor
to terminal validation 055 and does not reinterpret or reuse that run.

The method remains `RFCG-C24-DPB12-S1`; the only cell remains
`m08_c06,p=.04,d00,attempt022`.  V2 changes only the schedule-bound runtime
estimator: two cold four-process probe lengths time real sampling plus full
replay and fit one startup intercept plus a per-update slope.  T1, the resource
cap, initial families, sampler, statistics, and result permissions are
unchanged.  See `RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_V2_CONTRACT.md` and
`PRE_RUN_RED_TEAM.md`.

The immutable run
`exp102_q0_direct_block_t1_m8_v2_20260724_6933e31` is terminally
**`UNRESOLVED_DIRECT_BLOCK_T1_M8`**.  This is a sampler-convergence failure at
the frozen T1 clock, not a runtime/infrastructure failure, physical q_top, or
proof of impossibility.  It creates no m6, HARD2, formal, held-out, or
production authority.

Frozen local identities:

```text
config SHA256:
70285cf7ae8ecb7d062af7d72980e504edb42313d3e6708ab1e26a3bfbdf899d
control content SHA256:
49665fb9b42d977edfa3ee23218effd7c11563f49715b09a4307aa63edf79c48
control file SHA256:
f5d4cc913595c76d6e435b0e41a292a4655cc2c0f084e529b28313c9f3f83a25
control manifest SHA256:
fd31f5a7febb03959e4857e3336adb9b969797f17f47430cc9c48c43a0df3ce7
logical character SHA256:
347ec3cffdf1d278a13a8afed7b60dc61c98e64f4f5ae171afba0ad015508467
B character SHA256:
4cdbaf99129c81b9f629803c935e74c2958537d16d3f2a926179f6485c7803be
```

The control inherits only the audited 055 fixed H, syndrome, and legal
P/M0/M1/S states, verified by predecessor content/file SHA and fresh algebra.
It regenerates logical and B characters from the v2 config/namespace.  All 40
task seeds in each of the initialization/burn/measurement/observation fields
are unique and disjoint from validation 055 when evaluated at the same source
identity.

A non-authoritative local m8 smoke exercised the real two-length concurrent
sampler and full replay.  All eight component slopes were positive; the worst
factor-two T1 projection was about 1348.4 seconds.  This is implementation
evidence only, not a substitute for Linux timing or cross-node transcript
consensus.  Local conda-12 regressions pass: exp102 `617 passed` and exp101
`366 passed`, with only the existing fork and deprecated-alias warnings.

## Immutable execution and audit

The final source identity is:

```text
source commit:
6933e319b27840976f34e27c0d11313b6973cbe3
archive SHA256:
b62d0e22b7e37f8ca90186cc1d6d5bd9fe6e8d9b2568d9de569fd275ebb13eb5
source manifest SHA256:
135eb089bf1ca60a1009965847fbefef6c9bc238ed3db52258f311845c817e48
schedule SHA256:
ca057fbc2c76de2715dc7318f2f2c5d15567aeef403583df6dc958c28eec58d3
```

The complete validation-054 portable preflight and the fresh two-length v2
preflight both passed exact three-node consensus.  The latter selected the
unchanged T1 tier with a worst replay-inclusive factor-two projection of
`6550.3213s < 7200s`.  Fixed ownership then completed `14/13/13` tasks on
nd-1/nd-2/nd-3.  All 40 raw files were pulled back; no raw was reused.

The frozen primary analyzer returned report SHA
`e1bfb340d13f0053be036f72b7a9be1b567abc6510e269c6672ebeb7ae446015`
and raw-set SHA
`a267ded66e8039dc4d319590b15291545eabc124b7d25406b200d833b8262259`.
The out-of-band `allow_pickle=False` auditor does not call the sampler,
trajectory replay runner, or primary analyzer.  It independently reconstructs
the PortablePrng K0 starts, hard-coset algebra, direct B transcripts, states,
labels, weights, likelihood, q_top/D2, Rhat/ESS, every family/pair gate, MAP
bridge, constant-character rule, and terminal status.  It returned
`INDEPENDENT_RAW_ONLY_AUDIT_PASS`, audit SHA
`ada30d3cca844ede66b29e204f73eb1fe6fe2a297992ff0c28027878aa04b08e`.

## Scientific result

All five families fail the within-family Rhat and ESS gates.  The low-energy
P/M0/M1/S families are mutually close in q_top (`.90378--.92260`) and mean
weight (`.0388708--.0388953`), but their maximum Rhat is
`1.1335--1.3048` and minimum nondegenerate ESS is only `66.86--87.61`.
Their pairwise B-character means, pooled B Rhat, and several B-likelihood
comparisons also fail.  Visible motion is not the issue: every low-energy
trajectory records at least 104 B-column changes and 267 logical-label
changes during measurement, and both MAP families visit the opposite basin in
all eight trajectories.

The exact-K0 U family exposes the decisive global failure.  After the frozen
2048-update burn it remains at normalized state/B weights
`.097775/.101909`, versus about `.03888/.0400` for the low-energy families.
Its q_top is `.0000405` versus `.9038--.9226`, maximum Rhat is infinite, and
minimum ESS is `39.75`.  Every U/low-energy comparison fails all eight
distribution gates; for P/U, `delta q_top=.90374`, the logical D2 upper bound
is `.93903`, the B-character D2 upper bound is `.20827`, and 466 B characters
fail their mean-agreement gate.  U moves more than the other starts (at least
580 B-column and 2406 label changes per trajectory), so acceptance or state
motion cannot repair the conclusion.

The physical zero state remains illegal for this nonzero syndrome; shifted
zero is P.  Replacing U/MAP/S by common P starts would hide exactly the
observed failure.  The correct terminal statement is that this exact
random-scan full-column kernel does not equilibrate the m8 hard sentinel at
T1.  The raw cannot be extended, pooled with a successor, or reinterpreted as
a posterior estimate.

The primary analyzer emitted a `uint8` underflow warning in its
constant-character helper.  No measured B character was globally constant,
so both the corrected independent computation and primary report have zero
freeze failures and the terminal result is unaffected.  Any successor
analyzer must use signed arithmetic and include a tamper regression for this
branch.
