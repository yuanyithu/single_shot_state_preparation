# Validation 060 pre-run scientific red-team

This local-only validation asks whether an exact joint collapsed-B heatbath is
structurally implementable before any sampler or trajectory is written.  It is
frozen before the m8 elimination widths are computed.  Its source is not
frozen merely because a draft exists: every validation file must first be
tracked in one clean commit, and the launch guard must pass.

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

This structural stage generates no Markov states.  If a candidate ever reaches
a later conditional-movement screen, that fresh successor must retain legal P,
independent exact-K0 U, truth-free MAP and B-distinct low-energy S starts.
Physical zero is illegal for the frozen nonzero syndrome, and shifted zero is
P.  Common P/zero starts are forbidden because they would hide the demonstrated
basin split.

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

For an MR family, the chosen output-row identities change numerical factor
values but not the deliberately full factor scopes; all subsets of the same
cardinality are graph-isomorphic.  For RC1, the selected output-row identity is
also graph-isomorphic, while all 24 selected input columns are evaluated
separately.  The focused tests check these symmetries on a nontrivial toy H by
direct `B H` perturbation.

The primary scope builder is audited against exact `B H_j` changes for every
block variable.  A separate auditor does not import the primary analyzer: it
reconstructs each factor scope exclusively from direct single-coordinate B
perturbations and independently reimplements the deterministic min-fill order
with integer bitsets.  Width is the actual upper bound of the frozen order, not
a claim of minimum treewidth.  A candidate over the cap is only exhausted for
this frozen factorization/order/resource boundary; it is not mathematically
impossible.  Conversely, a width pass proves no conditional tightness,
movement, mixing, or target-tail coverage.

## Source, input and one-shot integrity

Before the analyzer can inspect a width it must verify:

- the complete Git worktree is clean and every configured validation artifact
  is tracked byte-for-byte at `HEAD`;
- the frozen 056 control file/content/H hashes and the terminal 059 report file,
  self-hash and failure status are unchanged;
- analyzer, preflight, independent auditor, focused test and both documents
  match the SHA256 values in `structure_config.json`; and
- neither the structure report nor independent audit already exists.

The analyzer repeats this guard internally, uses exclusive output creation and
binds the commit plus canonical source-tree SHA.  The result has structural
authority only after the independent reconstruction matches every scope,
order, width, gate and terminal field.  Any non-finite JSON, source mismatch,
algebra mismatch or audit disagreement is `CONFLICT`; it cannot be repaired by
editing or rerunning the same source identity.

## Gates, ranking and next boundary

The terminal status is `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND` if at
least one complete candidate family passes, otherwise
`LOCAL_JOINT_BLOCK_STRUCTURE_EXHAUSTED`.  The report must bind the
control/H/config/source bytes, pass every semantic reconstruction and contain
only finite integer resource quantities.

If several families pass, a single preferred *contingency* candidate is chosen
by the frozen lexicographic key `(block variable count, worst induced width,
worst single-table bytes, candidate order)`.  This ranking is structural and
does not inspect conditional values or sampler outcomes.

No trajectory may be generated from this status.  A surviving candidate does
not automatically receive an implementation: under the 2026-07-28 plan
assessment it is held as a contingency and may proceed only if fresh HP64
Stage 3 or Stage 4 fails.  At most one such successor may then implement the
mandatory n=10/n=13 complete conditional, detailed-balance/stationarity and
PortablePrng replay tests, followed by pre-registered P/U/MAP/S self-probability
and expected-movement gates.  Even a successful collapsed-B sampler remains
common-mode with HP/column/row methods and still needs an orthogonal confirmer
or rigorous tail/normalizer evidence.
