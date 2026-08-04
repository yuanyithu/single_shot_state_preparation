# BP-IMH local hard-sentinel viability

This directory is reserved for one immutable local diagnostic run of
`BPIMH-FR64` on `m08_c06,p=.04,d00,attempt022`.  Before raw generation, the
exact target, legal P/U/L starts, fixed 256/2048 clock, full-support proposal,
MH ratio, complete-label `D2_norm` gate, source identity, replay, and authority
boundary are frozen by `reviews/BP_IMH_REVIEW.md`, the canonical config, runner, and
focused tests.

## Terminal outcome

`INFRASTRUCTURE_FAILED_RELATIVE_OUTPUT_PATH`

The frozen v0 run wrote its 24-task manifest and completed enough computation
to begin raw serialization, but failed while creating the first receipt record:
the relative CLI output path was passed to `Path.relative_to()` against an
absolute exp102 root.  The directory contains only the immutable manifest and
one partial-run `P_00.npz`; it has no receipt or report.  This raw is forbidden
for analysis, reuse, extension, or pooling.  See `failure_report.json`.

This is not a sampler failure or a physical result.  A fresh v1 contract must
use a new seed namespace and must regression-test relative CLI path handling.
