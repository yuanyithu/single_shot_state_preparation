# BP-mixture dominance-witness feasibility

This frozen local structural probe asks a narrow question before anyone tries
to turn the empirically stable BP-systematic proposal into a BP-only posterior
estimator: can a predetermined legal witness already prove that the exact
three-component proposal mixture is too poor to be a finite rejection
envelope?

It does not run MCMC, draw an IID sample, estimate a normalizer, purity, or
`q_top`, or create remote, formal, held-out, or production authority.  Its
pre-run red-team review is `reviews/BP_DOMINANCE_WITNESS_REVIEW.md`.

For the exact target

```text
t(e) = (.04/.96)^|e|,  H_Z e = y,  pi(e) = t(e)/Z,
```

the report scores the exact frozen BP-mixture density `q(e)` on 1,691
deterministic legal states: the planted state, 64 canonical rank-complete
reduced-logical states, and every planted-plus-one-coordinate state from the
forward and reverse systematic hard-coset bases.  It uses only the universal
normalizer fact

```text
Z = Pr_.04(H_Z e=y)/(.96)^1600 <= (.96)^(-1600).
```

so every score is a conservative lower bound on `pi(e)/q(e)`.  Decimal
arithmetic imports the proposal's frozen IEEE probabilities exactly and rounds
the mixture density upward, keeping the lower direction conservative.

## Terminal result

The canonical config SHA256 is
`be78411d1459a6a33f835fc0780f70bd41cd4d0c2f45e9bb659dceb4f3faf180`.
The self-hashed report is
`d36815dce4662c922791409258cf1dbb43492f54465453cce116182e9862e20b`; its
file SHA256 is
`5840e2efc7ff394e85151a3412e4c87368b1f442f8d3c93fb9e452ce48a5e7e1`.

Its terminal status is
`BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE`.  The largest lower
bound is only `5.53e-63` for forward and `2.54e-53` for reverse, far below the
frozen `1e6` expected-proposal-call cap.  This is not a BP success: the loose
universal upper bound on `Z` makes every witness lower bound tiny even when a
proposal assigns a low probability to the state.

Thus the probe does not authorize a fresh BP-only IID run, a rejection sampler,
or a bounded-importance claim.  It instead exposes a circularity that must not
be hidden by a good finite ESS: a useful dominance certificate first requires
a much tighter *global* upper bound on the hard-coset normalizer, which is the
same tail/normalizer problem that remains unresolved.  The result rejects no
posterior or sampler; it only prevents this witness screen from being mistaken
for global coverage evidence.
