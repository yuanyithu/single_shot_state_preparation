# Validation 054: direct-positive block full-column preflight

This outcome-blind implementation/runtime preflight is governed by
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_REVIEW.md`.  It evaluates the same exact
random-scan full-B-column heatbath as validations 052/053, using fixed direct
positive-weight subtotal blocks after an explicit normal-range certificate.

The first phase is local only.  A local PASS may freeze portable subtotal and
trajectory digests into a new clean source; it does not itself authorize a
remote T1, posterior estimate, `q_top`, HARD2, formal, held-out, or production
run.

The first local report from source `f5f2976922ced2276f3bcb890bf24410cbc1db00`
passed its frozen gates, but is superseded for remote authorization because its
runtime seed identity still depended on the config digest.  It is retained as
`superseded_local_preflight_f5f2976.json`; no remote work used it.

Source `a0d4dbf6451240f0c2e07057d45206427ef09db0` then replaced that circular
identity with the frozen `runtime_seed_key` and passed all local gates.  Its
report is retained as `reference_origin_local_preflight_a0d4dbf.json` and was
used, before any remote run, to create `portable_reference.v1.json`.  The
portable artifact freezes all 12 ordered block-subtotal digests and four
sampling/replay transcript digests.  The final clean-source local preflight
must reproduce this artifact before a three-node deployment is allowed.
