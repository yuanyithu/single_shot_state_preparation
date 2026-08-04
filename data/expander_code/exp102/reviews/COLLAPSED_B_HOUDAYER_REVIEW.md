# Collapsed-B Houdayer red-team review

## Purpose

This document authorizes only a local structural feasibility probe for a new
two-replica move.  It does not authorize MCMC measurement, remote work,
HARD2, a posterior estimate, `q_top`, formal tuning, held-out work, or
production.

The prior physical-coordinate HCA-RHB1 kernel was exact but did not cool two
independent exact-K=0 uniform starts.  Its apparent low-energy L/L
recombinations therefore do not answer the actual bottleneck.  The proposed
counterfactual works directly on the collapsed variable `B`, the slow variable
used by HP64, rather than counting physical-state changes that can disappear
when `A|B` is regenerated.

## Exact target and move

For the HGP equation `H A xor B H = Y`, let `M_p(s)` be the exact classical
coset mass after integrating a column of `A`.  The exact cold marginal is

```text
pi_B(B) proportional to b^|B| product_j M_p(Y[:, j] xor B H[:, j]),
b = p / (1 - p).
```

For two `B` replicas, define a factor hyperedge for each classical column
`j`:

```text
S_j = { (row, column) : H[column, j] = 1 }.
```

Only variables on which the two replicas differ participate.  Connected
components are formed by the induced factor hypergraph.  Swapping every B bit
of one complete component between the replicas leaves each affected factor's
two input syndromes exchanged, and leaves the pair of unary B weights
exchanged.  Hence it preserves `pi_B(B_left) pi_B(B_right)` without an
accept/reject rule.

For a future HP64 composition, the move may operate only at the cold
`lambda=1` rung.  It must inject the changed B masks and recompute the cold
collapsed syndromes; it must not restart, clone, or reinitialize hot rungs.
It must then regenerate `A|B` exactly before recording a physical state.
Treating an old sampled A as if it remained valid after a B-HCA swap would be
wrong.

## Red-team conditions

- The hard target remains `pi(e|y) proportional to (.04/.96)^|e|` subject to
  `H_Z e=y`; planted error may define P and deterministic low-energy L starts
  but never an energy, likelihood, factor, acceptance ratio, or estimator.
- Physical all-zero is outside this sentinel's nonzero-syndrome support.
  Shifted-coordinate zero is P, so it cannot replace P/U/L adversarial
  families.
- The structural probe uses only frozen P-derived distinct-label L states and
  two independently seeded exact-K=0 uniform hard-coset U states.  It has no
  sampler output from which to choose a pair, factor, basis, or threshold.
- A new unordered *physical* pair is insufficient.  The actual sampler, if
  justified later, must separately record whether HCA changes the unordered B
  pair and whether it produces a non-whole-swap logical-label change.  A-only
  noise or stabilizer-only motion is not evidence of transport of the slow
  variable.
- Exact small-HGP enumeration must check component partitioning, factor-pair
  invariance, HCA involution, row sums, detailed balance, and stationarity of
  the complete B-pair move before any real-code probe.
- A structural signal can justify only an exact pair-kernel implementation and
  a fresh local P/U/L test.  It cannot establish finite-time convergence,
  control unobserved target tails, or substitute for a genuinely independent
  confirmation/mass-normalizer method.

## Frozen structural decision

The probe enumerates the first 16 deterministic low-energy, label-distinct
P-derived starts, all 120 unordered L/L pairs, P/L controls, and one frozen
U/U pair.  It reports component geometry and verifies exact factor invariants
for every component subset up to the fixed limit.

- If no L/L pair creates a new unordered B pair, the candidate is
  `COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION` and no hybrid sampler is
  authorized by this review.
- If at least one frozen L/L pair creates a new unordered B pair, the only
  allowed next status is
  `COLLAPSED_B_HCA_LOW_ENERGY_SIGNAL_REQUIRES_EXACT_PAIR_KERNEL`.
- U/U whole-pair behavior is retained as an anti-false-positive diagnostic.
  It neither proves nor, by itself, disproves a future hybrid: that hybrid
  would still have to demonstrate substantive B and logical movement from U
  under a fresh fixed-clock P/U/L experiment.
