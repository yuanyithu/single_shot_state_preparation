# Validation 055: fresh direct-block m8 T1 diagnostic

This directory freezes the fresh
`exp102.q0_random_full_column_direct_block.t1_m8.v1` diagnostic.  The method is
`RFCG-C24-DPB12-S1`; the sole cell is
`m08_c06,p=.04,d00,attempt022`.

The current state is **pre-measurement**.  No sampler raw or q_top result exists
until the final source passes both the validation-054 portable/runtime
preflight and this workflow's schedule-bound three-node preflight.  See
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
