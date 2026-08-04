# Collapsed-B tail-envelope feasibility

This local diagnostic asks a deliberately narrower question than MCMC: can a
strict upper bound on the unexpanded collapsed-`B` mass become tight enough to
support a future `q_top` estimator?

For the HGP hard constraint `H A xor B H = Y`, integrating every `A` column
gives

```text
mu(B) = Pr_p(B) product_j Pr_p[H a_j = (Y xor B H)_j].
```

At `p=.04=1/25`, the classical factors are rational.  V0 computes outward
float intervals for every factor using a directed-rounding recurrence.  If the
first `t` rows of `B` are fixed, each remaining factor syndrome is constrained
only on its first `t` bits.  Replacing every factor by its maximum compatible
upper endpoint gives a valid upper bound for that node after summing its
unassigned iid `B` prior.

The probe also contracts the resulting leading-row envelope with a deterministic
min-fill binary factor elimination.  This is an upper bound on the *entire*
collapsed normalizer, not an estimate.  It quantifies whether the factor maxima
are already too loose before attempting an exponentially larger branch tree.

The truth-free `B=0` coordinate is the primary lower-bound anchor.  It is a
valid collapsed coordinate even when the physical all-zero error is outside the
nonzero hard syndrome.  The planted `B` coordinate is recorded only as a
diagnostic and cannot be a future estimator anchor.

## What this cannot prove

Even a tight normalizer envelope does not control `q_top`.  A future
certificate must enumerate or otherwise exactly evaluate retained `B` modes,
sum a valid upper bound over every omitted branch, and propagate that tail
through logical-character or logical-sector intervals.  In particular, an
omitted `B` mode can contribute to an already retained logical sector.  If
retained sector masses are `a_i` and total omitted mass is at most `U`, a safe
purity upper bound is

```text
(sum_i a_i^2 + 2 U max_i a_i + U^2) / (sum_i a_i)^2,
```

unless the omitted part is proved logically sector-disjoint.  Dropping the
cross term is not valid for a `B`-tail bound.

This directory has no MCMC raw, no posterior estimate, no readiness authority,
and no permission to start remote, held-out, or production work.

## V0 result

The report SHA256 is
`4866d10e251be3b943be22b35d49f0703f6ba2f4b6627056f4b9f997d1f7cf82` and the
file SHA256 is
`d847f79a6586c5d19ff4582839fc501e2732402a09ede3284743172b5bd1125e`.
The classical interval recurrence has total lower and upper mass
`.9999999999999851` and `1.0000000000000129`, respectively.  Thus its local
interval arithmetic is behaving as intended on this input.

It is nevertheless far too loose for a tail certificate.  The truth-free
`B=0` lower anchor is only `3.00650946e-315` in the normalized scale.  The
depth-zero envelope is about `10^314.52` times that anchor; after one complete
B row, the exact envelope contraction has induced width 12 but is still about
`10^311.34` times it.  At two rows, the deterministic min-fill width is 25,
above the pre-set cap 18.  The planted-B diagnostic anchor is less extreme,
but its own two-row prefix upper bound remains about `10^86.65` times its
lower weight.

This is terminally `LOCAL_COLLAPSED_B_FACTOR_ENVELOPE_NOT_VIABLE_WITHIN_V0_CAP`.
It rejects the factor-wise-max envelope and this resource cap only; it does not
rule out a materially tighter certified decomposition or any physical result.
