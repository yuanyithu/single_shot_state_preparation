# Validation 062 pre-run scientific red-team

This document is frozen before calibration output exists. The validation
measures a diagnostic gate, not the physical quantity requested from exp102.

## 1. Deliverable and authority

The eventual scientific deliverable is posterior purity (`q_top`) on q=0
hard cosets. This validation only calibrates a finite-character equivalence
gate on IID oracle draws. Its maximum status is
`CHARACTER_GATE_COMMON_OPERATING_POINT_CONFIRMED`. It has no sampler, mixing,
cell, remote, formal, held-out, or production authority. A positive result is
only a prerequisite for later sampler work.

## 2. Target distribution and support

For the n=10/k=4 and n=13/k=1 HGP oracles, zero and nonzero syndromes are
handled by complete affine hard-coset enumeration:

```text
pi(e | y) proportional to (p / (1-p))^|e|,  H_Z e = y.
```

The planted error constructs a nonzero syndrome only. It never appears in the
posterior energy. Focused tests independently enumerate all `2^n` physical
bitstrings, filter by syndrome, and compare complete support, state-keyed
weights, logical labels, and collapsed-B labels with the runner. Thus a
syndrome-valid subset cannot masquerade as the full oracle.

The draws are IID multinomials from that exact posterior. There is no initial
state and no Markov transport. Consequently this validation cannot answer
whether P, U, zero, MAP, or stratified starts are suitable for HP64. In
particular, it must not be used to justify replacing adversarial P/U starts by
a common start.

## 3. One deployable operating point

The old failure mode was to satisfy different requirements at incompatible
resource points. Here a candidate is one fixed pair:

```text
(trajectory_count, effective IID draws per trajectory).
```

That pair is shared by every exact catalog and every 15/163/511/688/4160
synthetic stress. At the same pair, one `logical` simultaneous multiplier is
frozen across exact logical, 511, and 4160 rows and both base means; one
`collapsed_B` multiplier is frozen across exact B, 15, 163, and 688 rows and
both base means. The roles may have different multipliers because their
registered multiplicities differ, but a role cannot choose a multiplier by
catalog, base mean, or outcome.

Points are evaluated in fixed independent-draw cost order. The first eligible
point is frozen. Calibration, selection, and confirmation have independent
seed stages. Failure of fresh confirmation ends in
`CHARACTER_GATE_REDESIGN_REQUIRED`; the runner cannot move to another point
after seeing confirmation.

## 4. Three-state rule and finite-trial evidence

For each trial and frozen role multiplier `z`, the candidate rule is

```text
PASS         if max_u (|Delta_hat_u| + z * SE_u) <= .04
FAIL         if max_u (|Delta_hat_u| - z * SE_u) >  .04
INCONCLUSIVE otherwise.
```

The historical rule is retained only as a diagnostic. Eligibility requires,
for every registered category, a one-sided Wilson lower bound of at least
`.90` for null PASS, `.90` for FAIL at a true maximum shift of at least `.06`,
and `.98` for empirical simultaneous coverage. Point estimates alone cannot
select or confirm a point. The empirical multiplier is still a calibrated
Monte Carlo construction, not a distribution-free theorem.

## 5. Relation to purity and q_top

For complete nonzero logical characters,

```text
q_top = mean_{u != 0} m_u^2.
```

Exact logical rows therefore report normalized `q_top` directly. A
collapsed-B character tilt is performed by a valid physical-state
reweighting that preserves the conditional distribution within each selected
B-character sign. Its induced `q_top` is recomputed from the retained logical
sign of every physical state; B-character purity is not relabeled as q_top.
Exact rows retain complete base, shifted, and true-shift vectors.

For any one observed character, `|a^2-b^2| <= 2|a-b|`. Therefore a `.04`
maximum shift gives only a worst-case `.08` purity-change bound averaged over
the frozen observed catalog. This statement has two hard limits:

1. it says nothing about any unobserved character;
2. it is a q_top bound only for a complete logical catalog.

The finite 4160-character large-k catalog is not complete, so passing it
cannot certify full logical purity or missing-basin mass.

## 6. Deliberately nonphysical stress

Synthetic characters are generated as independent binary observables. They
are multiplicity and decision-rule stresses, not a joint HGP label
distribution and not posterior samples. Catalog sizes are frozen at 15, 163,
511, 688, and 4160.

The distributed stress moves every mean from `.80` to `.76`. Its maximum
coordinate shift is `.04`, but its catalog-purity change is

```text
.80^2 - .76^2 = .0624.
```

This is an explicit witness that a max-character tolerance is not itself a
`.04` purity guarantee. Synthetic rows save a deterministic pattern,
nonzero count, extrema, mean square, and full-vector SHA instead of storing a
4160-entry vector in every replication row. They never contain a q_top value.

## 7. Remaining blind spots and stop rule

- IID effective draws are optimistic for autocorrelated MCMC clocks.
- Independent synthetic characters do not cover every dependence structure.
- Finite characters can miss an unobserved basin or high-order logical mode.
- Character agreement does not replace full-label D2, B/weight/likelihood,
  Rhat/ESS, burn crossing, adversarial initialization, or an orthogonal
  confirmation method.
- A conservative multiplier may make null PASS and `.06` detection mutually
  unaffordable. That is a gate-design result, not permission to lower a
  threshold after seeing output.

Only one common selected point passing fresh confirmation can produce the
maximum status. Otherwise no anchor experiment may cite this candidate gate
as calibrated. No output is generated until the config and runner, auditor,
tests, README, and this red-team document are hash-bound to one clean commit.
