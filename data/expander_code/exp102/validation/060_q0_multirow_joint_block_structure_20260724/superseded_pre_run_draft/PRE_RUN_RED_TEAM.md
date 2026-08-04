# Validation 060 pre-run scientific red-team

This local-only validation asks whether an exact joint collapsed-B heatbath is
structurally implementable before any sampler or trajectory is written.  It is
frozen before the m8 elimination widths are computed.

## Deliverable and authority

The Exp102 deliverable remains a converged posterior-purity sampler plus a
mechanism-independent confirmer.  Attractive state changes, acceptance or a
small local conditional are not substitutes.  This validation can only name a
joint-block implementation candidate.  It cannot estimate q_top, certify a
cell, authorize remote work, or serve as the independent confirmer.

## Target and support

The target remains the strict collapsed distribution

```text
g_p(B) proportional to b^|B| product_j M_p(Y_j xor B H_j),
b = p/(1-p).
```

No planted error enters a factor.  A t-row block contains all `24*t` entries
in a state-independent set of t B rows.  A row-column cross contains the union
of one complete B row and one complete B column, with the intersection counted
once.  Exact heatbath factors are derived from the full scopes of `B H_j`; no
pairwise approximation, Bethe factorization or sequential one-row surrogate is
allowed.

## Slow coordinate and initial states

Validation 059 shows that sequential exact one-row and one-column updates move
U for roughly one early row sweep and then freeze in a wrong high-energy B
basin.  The candidate therefore has to coordinate variables inside one
conditional draw; merely reporting more changed bits would miss the problem.

This structural stage generates no Markov states.  If a candidate survives,
its conditional movement screen must retain legal P, independent exact-K0 U,
truth-free MAP and B-distinct low-energy S starts.  Physical zero is illegal
for the frozen nonzero syndrome, and shifted zero is P.  Common P/zero starts
are forbidden because they would hide the demonstrated basin split.

## Frozen structures and resource boundary

The deterministic min-fill key is `(missing fill edges, live degree, variable
index)`.  The frozen candidates are:

- all unordered two-row blocks (`MR2`);
- all unordered three-row blocks (`MR3`);
- all unordered four-row blocks (`MR4`); and
- all 24 row-column crosses (`RC1`), whose row index is structurally
  irrelevant but whose selected column is evaluated separately.

For each frozen elimination plan, implementation candidacy requires induced
width `<=25`, largest intermediate factor `<=2^26` float64 entries, largest
initial factor scope `<=26`, and the corresponding single-table byte lower
bound `<=512 MiB`.  These are necessary memory conditions only.  Passing still
requires a fresh implementation benchmark with measured peak memory and a
factor-two fixed-clock projection below the trajectory wall cap.

The scope builder is audited against exact `B H_j` changes for every block
variable.  Width is the actual upper bound of the frozen order, not a claim of
minimum treewidth.  A candidate over the cap is only exhausted for this frozen
factorization/order/resource boundary; it is not mathematically impossible.

## Gates and next boundary

The report must bind the control/H/config/script bytes, pass all scope semantic
audits, and contain finite integer resource quantities.  The terminal status
is `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND` if at least one complete
candidate family passes, otherwise `LOCAL_JOINT_BLOCK_STRUCTURE_EXHAUSTED`.

No trajectory may be generated from this status.  A surviving candidate next
needs an exact implementation with mandatory n=10/n=13 complete conditional,
detailed-balance/stationarity and PortablePrng replay tests, followed by
pre-registered P/U/MAP/S self-probability and expected-movement gates.  Even a
successful collapsed-B sampler remains common-mode with HP/column/row methods
and still needs an orthogonal confirmer or rigorous tail/normalizer evidence.
