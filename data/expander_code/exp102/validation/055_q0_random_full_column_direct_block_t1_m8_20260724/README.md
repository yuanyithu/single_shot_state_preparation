# Validation 055: fresh direct-block m8 T1 diagnostic

This directory freezes the fresh
`exp102.q0_random_full_column_direct_block.t1_m8.v1` diagnostic.  The method is
`RFCG-C24-DPB12-S1`; the sole cell is
`m08_c06,p=.04,d00,attempt022`.

The terminal state is **`RUNTIME_EXHAUSTED` at preflight, with zero measurement
raw**.  This did not test sampler convergence or produce q_top.  See
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_CONTRACT.md` for target, initial states,
statistics, gates, and result permissions.  `PRE_RUN_RED_TEAM.md` records the
target/support/initialization/slow-variable/estimator/common-failure review and
the analyzer blind spots closed before launch.

Frozen control identities:

```text
config SHA256:
19d5f64b59170e60c0dc4727da2d3086e299c48934cb81577a33826ff1f32c71
control content SHA256:
982fb9318fe423a1d642c118c4efccac247e446da07b6bdea4d8a64dab1b8421
control file SHA256:
c84579cb2fcd593b176308610a5c69e0fe47f54136b61b9f70a7fff6d94c4168
control manifest SHA256:
03847ffe8fa95f4d015298e91da7e663e6f9a20312dee57ecac2a2f4ca41ff2e
logical character SHA256:
e395d9752b10d528cc821be0c335fcd2d2711409b7867c3ba3ff73933a52a584
B character SHA256:
a2f0e5e5c052338592c60ef81a2df4414960bd6f273a0bbeb09c2fed3e8c95b8
```

## Terminal remote evidence

Two schedule attempts failed before control creation and are retained only as
infrastructure audit:

```text
exp102_q0_direct_block_t1_m8_20260724_146ef55
  run root was manually created before the fresh-root schedule check
exp102_q0_direct_block_t1_m8_20260724_146ef55_r2
  the schedule marker itself was placed inside and created the fresh run root
```

Neither attempt produced a schedule, preflight, control, or sampler raw.  The
fixed third attempt put schedule markers under `~/.single_shot/logs/` and let
the schedule create its own run root atomically:

```text
run:              exp102_q0_direct_block_t1_m8_20260724_146ef55_r3
source commit:    146ef550591a72435641c47baa8794c338f7a27e
source archive:   b960250283d6b986bc7bb20c1ff4aca3238a9c4ecbae7da88512bc6f591e3c48
source manifest:  8daf8d94ece1adeb52b954d13ea34a7062a2bc8995f3e0043144af3eeac144da
schedule SHA256:  bbc2e268d6e9ed39a2fcae296db3d4dbcb2c49a1f6bf60e6b5678b72b8ee731a
```

The complete validation-054 portable/runtime preflight passed on the final
source.  All nodes agreed exactly, and replay-inclusive T1 projections were
`4216.16/4149.15/4549.57s` on nd-1/nd-2/nd-3, with aggregate SHA
`ae356c9e061ae4aea81b6c7a30baec8a744319bcfd8419483f15f8338cfb35ac`.

Validation 055's separately frozen short runtime probe used only two burn plus
eight measurement updates per worker.  It then linearly scaled the entire
elapsed time, including fixed initialization and runner overhead, to 10,240
updates and multiplied by two.  Its projections were therefore
`9272.13/8779.07/13638.99s`, above the unchanged 7200-second cap.  Aggregate
status is `RUNTIME_EXHAUSTED`, exact consensus remains true, and aggregate SHA
is `7fffcdda598422fab3b33ded26c4acdf77f035932c2cc308c6f8a22a7420f461`.
The workflow correctly stopped before measurement; the only NPZ in the run is
`control/control.npz`.

The independent conda-12 audit verifies the source/control/schedule identities,
both failed schedule attempts, all node and aggregate self-hashes, projection
arithmetic, exact consensus, and raw absence.  It reports
`INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED`, audit SHA
`00622194dc370a66e08a0b94a7108b324aa49322de648fda7656f2c6ed5fc665`.

This is a false-negative resource estimate caused by a frozen estimator, not a
sampler failure, physical result, or mathematical impossibility.  It cannot be
reinterpreted as PASS after the fact.  A successor must use a fresh contract,
source, schedule, seeds, and raw; it should estimate the steady-state per-update
rate with representative replay-inclusive probes while keeping T1, the
7200-second cap, all adversarial initial families, and every convergence gate
unchanged.
