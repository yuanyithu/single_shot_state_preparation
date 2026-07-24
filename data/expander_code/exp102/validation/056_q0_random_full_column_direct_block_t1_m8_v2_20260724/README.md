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

The current state is **pre-schedule and pre-measurement**.  No v2 sampler raw or
q_top exists.  Measurement is authorized only if the final source passes both
the full validation-054 portable/runtime preflight and the v2 three-node
schedule-bound preflight with exact transcript consensus.

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
