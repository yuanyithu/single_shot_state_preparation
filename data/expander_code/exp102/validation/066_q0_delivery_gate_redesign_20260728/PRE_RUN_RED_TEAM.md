# Validation 066 pre-run scientific red-team

This document is frozen before a delivery-gate report exists.  It records what
the local IID exercise can falsify and what it can never certify.

## 1. Deliverable first

The exp102 deliverable is normalized posterior logical purity (`q_top`).  A
distribution-equivalence gate must also distinguish equal-purity
distributions on different logical supports.  Therefore 066 estimates exactly
two full-label quantities:

```text
q_top(P) = (sum_l P(l)^2 - 2^-k) / (1 - 2^-k)
D2_norm(P,Q) = sum_l (P(l)-Q(l))^2 / (1 - 2^-k).
```

The old maximum-character threshold is not promoted into a purity bound.  It
is retained only because a particular character can reveal a slow mode that a
scalar purity comparison hides.

## 2. Target, support, and input provenance

The small-HGP inputs are not rebuilt from a sampler or planted truth.  The
complete nonzero logical character means already recorded in independently
audited validation 062 are inverted by Walsh transform.  The identity mean is
exactly one.  Reconstructed probabilities must be nonnegative, sum to one,
recover every source character mean, and reproduce source `q_top`.

The k=9/16/36/64 stresses are explicit sparse probability measures on legal
logical labels.  Their supports and probabilities are fixed without observing
calibration output.  k=64 labels and masks stay `uint64`, including bit 63.
No planted error enters an energy, and no physical zero state is invented for
a nonzero syndrome.

## 3. Estimator and uncertainty

Within-trajectory self-products are excluded from each collision U-statistic.
The two sides use independent seeds.  D2 uses all cross-family products.  The
A and B ensembles are independent, but pairwise cross-products share
histograms and are not independent pseudo-replicates.  Trajectories remain the
uncertainty unit.  Delete-one estimates are recomputed after omitting one
trajectory at a time on each side; each side is centered separately before
the two group-wise jackknife variances add.

The signed q_top contrast is used for uncertainty coverage and its absolute
value for equivalence.  D2 is stored without projection to its physical
range.  The gate may use `max(0,D2_hat)` in an upper bound, but neither raw
values nor delete-one values are clipped.  Non-finite values are a conflict.

The D2 null is a degenerate U-statistic, so its delete-one jackknife is only an
asymptotic MCSE diagnostic, not a strict confidence interval.  An independently
seeded outer calibration stage validates joint operating coverage empirically:
for each replicate it takes the maximum error/MCSE over all registered IID
scenarios and both estimands before freezing a quantile.  Selection and
confirmation cannot reuse those draws.  One-sided Wilson intervals, not point
rates alone, classify null and `.02` PASS power, `.06` q_top and D2 FAIL power,
the false-PASS rate of each `.06` row against a `.02` maximum, and joint
interval coverage.  Exact `.04` rows are boundary diagnostics, not
relabelled after seeing their decisions.  PASS uses the original `.95`
operational conjunction.  FAIL uses the preregistered Bonferroni confidence
`1-.05/1390`: 139 hypotheses per stage/point (102 non-boundary base rates, 36
bad-row false-PASS rates, and one simultaneous coverage rate), across five
selection and five potential-confirmation points.  This is an operational
Wilson rule, not a strict familywise coverage theorem.
Point PASS is a conjunction/intersection-union decision and therefore has no
extra PASS multiplicity adjustment.  Point FAIL is the union of atomic
failure claims and therefore uses the full Bonferroni-adjusted evidence.

Bad-q_top power is evaluated on the q_top scalar state, and bad-D2 power on
the D2 scalar state.  Failure of the other scalar cannot substitute for the
registered mechanism's power.  A selected point's fresh confirmation is
terminal: PASS confirms locally, FAIL yields
`SELECTED_POINT_CONFIRMATION_FAILED_REDESIGN_REQUIRED`, and interval overlap
is inconclusive; confirmation never triggers outcome-driven selection of a
larger point or a claim that those larger points failed.

## 4. Character diagnostic is not coverage

For k<=4 every nonzero character is available; for larger k the diagnostic
contains every basis character and a frozen finite set of additional masks.
The maximum observed character difference has no term in either decision
formula.  In particular, it provides no claim about unsampled characters and
cannot veto or rescue a delivery-gate operating point.

## 5. The central blind spot: common wrong convergence

Full labels remove character-catalog tail error only for labels actually
observed.  They do not bound probability mass in a target basin that no chain
visited.  If P and U, or two methods, collapse to the same wrong distribution,
then both `Delta q_top` and D2 can be small while the posterior estimate is
wrong.

Two non-IID controls are therefore frozen as `EXPECTED_KNOWN_BLIND`:

- `common_freeze`: every trajectory on both sides remains at one label;
- `distinct_freeze_same_set`: trajectories remain at different labels, but
  the two sides use the same frozen set.

Their registered target distributions have disjoint support and bad D2.  The
observed distribution-only gate is nevertheless expected to PASS.  The
runner must preserve that label and must never count this as mixing power or a
successful target-posterior estimate.

The collision U-statistic is unbiased only when trajectories within each
family are independent and have the same expected histogram.  Measurement
time blocks cannot substitute for trajectories.  A future design with fixed
stratified starts must analyze each family/stratum explicitly; strata cannot be
pooled and called exchangeable.  The frozen controls are deliberately outside
this IID interpretation.

Consequently future sampler authority still requires legal adversarial P/U,
MAP/B-distinct starts where registered, real B/logical transport, burn
crossing, Rhat/ESS, and an orthogonal confirmation method.  Starting every
chain from P or a common state would conceal exactly the failure these
controls demonstrate.

## 6. Failure modes and stopping rule

- IID multinomial draws are optimistic for autocorrelated Markov clocks.
- Delete-one uncertainty measures between-trajectory sampling variation; it
  is not a deterministic missing-mass or normalizer bound.
- `D2_norm` is L2, not total variation; diffuse or heterogeneous missing tails
  can evade a small L2 comparison.
- Sparse stresses test known constructions, not every large-k distribution.
- A confirmed gate validates its scalar decision calibration only; it does
  not prove a sampler reaches the input distribution.
- A Wilson-certified rate failure may yield
  `DELIVERY_GATE_REDESIGN_REQUIRED`; a requirement interval overlapping its
  target yields `DELIVERY_GATE_CALIBRATION_INCONCLUSIVE`.  Neither permits
  adding trials, changing thresholds, dropping a stress, or selecting after
  confirmation.

The one-shot output calibrates one comparison, not Stage-3 multiplicity over
P/U, T/2T, methods, or cells.  The orthogonal-confirmer portfolio,
future-schema runtime coverage, and campaign budget remain unfrozen or
unapproved.  The output has no remote authority.  There is no legitimate path
from this validation directly to m3 anchors, HARD2, formal tuning, held-out,
or production.
