# 066 q=0 delivery-gate redesign calibration

Status: `SOURCE CONTRACT / LOCAL CALIBRATION ONLY`.  This status does not
expire when generated outputs appear; the current run outcome, if any, must
be read from `delivery_gate_report.json` and `independent_audit.json`.

This fresh, local-only validation calibrates a gate against the exp102
deliverables themselves.  It replaces a maximum-character claim with direct
full-label collision estimates of normalized `q_top` and `D2_norm`.  It does
not run HP64, any Markov chain, or any remote job.

## Frozen estimands

Each trajectory contributes its empirical distribution over complete logical
labels.  Only products between distinct trajectories enter a side's collision
mass.  For logical dimension `k`, the side estimate is

```text
C_hat(P) = mean_{i != j} <P_i, P_j>
q_top_hat(P) = (C_hat(P) - 2^-k) / (1 - 2^-k).
```

For independently seeded sides A and B,

```text
D2_norm_hat = (C_hat(A) + C_hat(B)
               - 2 mean_{i,j}<A_i,B_j>) / (1 - 2^-k).
```

The primary equivalence gate directly tests the absolute difference between
the two normalized `q_top` estimates.  A separate upper gate tests
`D2_norm`.  Omitting each trajectory on each side supplies the two-sample
delete-one jackknife uncertainty.  All computed estimates and delete-one values
remain raw: negative `q_top` or `D2_norm` values are never clipped.  The D2
decision uses `max(0, D2_norm_hat)` only when forming a conservative upper
bound, without changing the saved estimate.

A frozen finite character catalog is also evaluated.  Its maximum observed
mean difference is named and reported only as a slow-mode diagnostic.  It
does not enter the `q_top` interval, D2 interval, calibration multiplier,
selection, confirmation, or terminal status.

## Bound inputs and legal distributions

The runner binds both the completed validation-062 report and its independent
audit by file SHA, internal self-hash, source commit, and terminal status.  It
selects the twelve unique null-shift, complete-logical rows from the first 062
operating point.  Adding the identity character and applying inverse Walsh
reconstructs the complete k=1 and k=4 label distributions.  The runner checks
nonnegativity, normalization, forward-Walsh recovery, and normalized purity
before using them.

Outcome-blind sparse distributions additionally cover k=9, 16, 36, and 64 at
normalized-purity profiles `.05`, `.15`, `.55`, and `.90`.  This spans
near-uniform, m6-like, intermediate, and m8-like regimes instead of tuning one
convenient purity.  The registry contains null pairs, exact normalized-q_top
differences `.02`, `.04`, and `.06`, equal-purity swaps with exact D2 values
`.00`, `.02`, `.04`, and `.06`, and a disjoint-support D2 stress.  All
large-k labels are `uint64`; the k=64 registry and character diagnostics
explicitly exercise bit 63 and never pass through `int64`.

## Selection and confirmation

Each registered operating point freezes one uncertainty multiplier from an
independent calibration seed stage.  For each outer calibration replicate it
takes the maximum studentized error over every registered IID scenario and
both scalar estimands, then freezes the preregistered quantile.  This empirical
simultaneous calibration is not a strict confidence interval, especially for
the degenerate D2-null U-statistic.  Fixed-cost selection and fresh
confirmation use two other seed stages.  A point passes only when one-sided
Wilson bounds meet every frozen requirement:

- null rows pass both delivery gates;
- true `.02` q_top and D2 alternatives still pass both delivery gates;
- rows with true `|Delta q_top| >= .06` fail the q_top scalar gate itself;
- equal-purity rows with true `D2_norm >= .06` fail the D2 scalar gate itself;
- each bad-q_top and bad-D2 row has a false-PASS Wilson upper bound at most
  `.02`, directly limiting the most dangerous erroneous certification rate;
- the joint q_top/D2 interval covers its two true scalar estimands;
- both known-blind controls exhibit their preregistered distribution-gate
  PASS and remain labelled `EXPECTED_KNOWN_BLIND`.

True `.04` rows test boundary behavior but have no required PASS or FAIL rate.
Null, good-alternative, bad-alternative, and known-blind operating power use a
`.95` minimum; simultaneous interval coverage retains the stricter `.98`
minimum.  Thus the redesign does not inherit validation 062's `.90` power
floor merely to make a new gate easier to pass.
Each required rate is classified by one-sided Wilson intervals.  PASS retains
the original `.95` operational conjunction.  FAIL uses a Bonferroni-adjusted
one-sided confidence over the complete frozen protocol: 102 non-boundary
base-rate hypotheses plus 36 bad-row false-PASS hypotheses plus one aggregate
coverage hypothesis, at each of five selection and five potential
confirmation points, for `1390` hypotheses.  Its confidence is
`1-.05/1390`.  Boundary rows contribute no separate rowwise hypothesis but
remain part of the aggregate simultaneous-coverage hypothesis.  For a
minimum-rate requirement, PASS means the unadjusted lower bound reaches the
target and FAIL means the adjusted upper bound is below it.  For a maximum
false-PASS requirement, PASS means the unadjusted upper bound is at most the
limit and FAIL means the adjusted lower bound exceeds it.  All remaining
overlap is INCONCLUSIVE.  These Wilson calculations are an
operational error-control rule, not a strict familywise coverage guarantee.
Point PASS is a conjunction (an intersection-union rule), so it receives no
additional PASS multiplicity adjustment.  Point FAIL is a union over atomic
failure claims, which is why its Wilson evidence uses the full adjustment.
Selection stops at the first PASS point.  Confirmation
cannot change its trajectory count, draw count, multiplier, registry, or
seeds.

If every frozen point is certified below a required rate, the terminal status
is `DELIVERY_GATE_REDESIGN_REQUIRED`.  If no point passes but at least one
requirement remains statistically unresolved, it is
`DELIVERY_GATE_CALIBRATION_INCONCLUSIVE`.  A selection PASS followed by an
inconclusive confirmation is also inconclusive, not a scientific failure.  A
fresh-confirmation FAIL is terminal
`SELECTED_POINT_CONFIRMATION_FAILED_REDESIGN_REQUIRED`; this says only that
the selected point failed confirmation and does not claim larger grid points
failed.  The runner does not try a larger point after seeing confirmation.
Identity, non-finite, self-hash, and audit errors abort as conflicts outside
this statistical taxonomy.

## Known blindness and authority

The U-statistic interpretation requires independent trajectories with the
same expected histogram within each family.  Time blocks are not independent
trajectories.  Future fixed stratified starts must be analyzed by family and
stratum, not pooled as exchangeable replicas.

Direct full-label collision removes the unobserved-character-subsampling tail,
but it does **not** provide a deterministic certificate for target mass in a
basin never visited by the trajectories.  Two controls make this explicit:

1. both families freeze at one common legal label;
2. both families freeze on the same set of distinct legal labels.

In both controls the registered target distributions disagree, yet the
observed q_top/D2 equivalence gate can pass.  That outcome is
`EXPECTED_KNOWN_BLIND`, not evidence of convergence.  A future sampler must
still be rejected when adversarial initial families, B/logical transport,
burn crossing, Rhat/ESS, or an orthogonal confirmation method expose common
collapse.  Common P starts, physical zero, or a shared low-energy basin cannot
replace those checks.

`D2_norm` is an L2 discrepancy, not total variation.  Diffuse missing mass,
trajectory heterogeneity, or a common unvisited tail can remain poorly exposed
even when the saved scalar gates agree.

The maximum status here is
`LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`.  It only permits drafting a
later local anchor contract.  It does not certify MCMC mixing, target-basin
coverage, a cell, an `(m,p)` point, HP64, remote execution, formal tuning,
held-out work, or production.

Even a local PASS leaves four explicit blockers: the large-k orthogonal
confirmer portfolio is unfrozen, future-schema runtime coverage is incomplete,
and the campaign budget is unapproved.  Validation 066 calibrates one
comparison at a time; Stage 3 must separately freeze multiplicity across P/U,
T/2T, methods, and cells.  Future per-character, Rhat/ESS, burn-crossing, and
transport diagnostics retain independent veto authority even though the 066
IID character diagnostic does not enter this scalar gate.
The report therefore persists the independent project-level state
`BLOCKED_BEFORE_REMOTE` regardless of its local calibration status.

Before the one-shot runner is ever invoked, the config and five source
artifacts must be committed in one completely clean worktree.  The runner and
independent auditor reject bytecode, changed hashes, non-finite JSON, repeated
outputs, and source-commit drift.  The auditor does not import the runner and
independently reconstructs distributions, truths, seeds, trial decisions,
Wilson bounds, fixed-order selection, confirmation, and terminal authority.
This v1 uses same-environment bit replay, frozen to NumPy `2.4.1`,
`default_rng`, and `PCG64`.  Replayed raw metrics and every delete-one array
must be bit-identical; no ULP or numerical tolerance is allowed.  The report
does not persist those arrays as raw data.  Instead each scenario commits the
labels, histogram counts, and all metrics with dtype/shape/little-endian
SHA-256 receipts; calibration joint and outer arrays have separate receipts.
The auditor regenerates every committed byte from the seeds.  A receipt is an
audit commitment, not persistent sampler raw and not a substitute for future
raw retention.  The frozen compact-report limit is 10 MiB and the runner and
auditor both reject a larger report.  This makes no cross-version portable-RNG
claim.
