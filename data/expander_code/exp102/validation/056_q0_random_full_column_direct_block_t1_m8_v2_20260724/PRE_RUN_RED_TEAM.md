# Validation 056 pre-run scientific red-team

This review is frozen before schedule creation or measurement raw.

## What is the requested deliverable?

The eventual need is an unbiased, converged posterior-purity estimator across
the full exp102 range.  This run is deliberately narrower: it asks whether one
exact hard-coset kernel can pass a single m8 T1 diagnostic without hiding its
known collapsed-B slow variable.  Even a pass is not a physical data point and
cannot replace an independent confirmation method or held-out testing.

## Did runtime work displace the scientific problem?

No sampler parameter changes in v2.  Validation 055 never reached the sampler
test because a 10-update timing probe repeated fixed startup cost in its
projection.  V2 corrects that measurement error only.  It retains 2048+8192
updates, the 7200-second cap, all five initial families, full replay, and all
distributional gates.  Runtime is outcome-blind and reads no q_top or mixing
counter.

## Is zero a legitimate or useful common start?

No.  The physical zero vector violates the nonzero sentinel syndrome.  The
zero vector after shifting by the planted error is exactly P, which is already
present.  Making every chain P would suppress the main adversarial question:
whether broad K0 states and distinct low-energy B basins reach the same target
distribution.  P/U/M0/M1/S x8 therefore remains mandatory.

## Is the kernel targeting the right measure?

Yes.  Every transition remains inside `H_Z e=y` and samples the exact
conditional of `(.04/.96)^|e|`.  The planted error enters only P
initialization.  It is not used in weights, the column clock, or observations.

## Is the measured motion the actual slow motion?

The slow coordinate is collapsed B.  Exact `A|B` redraws can change physical
states and logical labels while B is frozen.  The analyzer therefore retains
B traces, bit/row/column/dense characters, B D2, B weight/likelihood,
constant-character crossing, and MAP-basin transport.  Acceptance, state
changes, or label changes alone cannot pass the run.

## Can timing still produce a misleading PASS?

Short and long probes run in separate cold process pools at the exact
four-worker concurrency.  Sampling and full replay are timed separately; both
slopes must be positive, and the worst P/M0/S/U fit is used with the unchanged
factor-two margin.  This remains an empirical projection, not a proof against
all load variation.  The hard 7200-second cap and fail-closed unstable status
limit its authority.

## What remains a blind spot after a statistical PASS?

Finite characters and five families may share an unobserved basin.  A
single Gibbs mechanism cannot independently validate itself, and low
temperature need not traverse every logical direction if target mass is
concentrated.  The correct requirement is agreement on q_top and frozen
slow-coordinate diagnostics, followed by fresh m6/HARD2 and an independent
method.  V2 cannot authorize formal or production work.
