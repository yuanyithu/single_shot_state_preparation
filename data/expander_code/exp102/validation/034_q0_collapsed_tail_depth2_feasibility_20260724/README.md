# Depth-two collapsed-B tail-envelope feasibility

This frozen local probe asks one narrow question for
`m08_c06 / p=.04 / d00 / attempt022`: can a strict, outward-rounded
depth-two collapsed-B envelope make the unretained B mass no more than one
percent of deterministic retained B marginals?

It uses all unique B marginals from a fixed, non-planted MILP/MAM MAP-anchor
catalog.  The catalog hash, code registry hash, rational `p=1/25`, depth,
memory cap, runtime cap, and tail-fraction target are fixed in
`q0_collapsed_tail_depth2.feasibility.v0.json`.  The probe refuses a changed
or noncanonical config and refuses to overwrite its report.

The preflight is limited to a 6 GiB largest-table cap and a 900-second
envelope-runtime gate.  It contracts the factorized depth-two upper envelope
with directed floating-point rounding, then compares the resulting total
partition upper bound with the retained lower mass.  An upper bound below the
retained lower mass is a conflict, not a zero-tail result.

This is a feasibility test for a possible strict normalizer/tail mechanism.
It does not sample a posterior, estimate purity or `q_top`, validate MCMC
initial states, authorize remote work, or authorize formal, held-out, or
production experiments.  A tight B-tail alone would still need a certified
logical-character or sector-mass treatment before it could support a physical
claim.

## Terminal result

The one frozen run completed with report SHA256
`dffacc4ac340c33b49e8578432ce17a3f8b89a65698d08985677662f3d23f147` and
status `DEPTH2_ENVELOPE_NOT_TIGHT_ENOUGH`.  The width-25 contraction used a
largest 512 MiB table, took `3.141538` seconds, and reached a 2,504,933,376-byte
Darwin peak RSS, so the resource gate passes.  The actual decision is instead
the strict mass gap:

| Quantity | Value |
| --- | ---: |
| Total scaled partition upper bound | `3.110162637896501e-11` |
| Two retained B marginals, summed lower bound | `3.3080527487949514e-96` |
| Tail / retained-mass upper ratio | `9.401792758683984e84` |
| Frozen goal | `<= .01` |

The envelope therefore misses its tightness requirement by roughly 87 decimal
orders.  Fast contraction does not compensate for that missing global-mass
control, and this result must not be converted into a posterior, purity,
`q_top`, or readiness claim.
