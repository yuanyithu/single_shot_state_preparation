# BP proposal-dominance red-team review

## Purpose and authority

This review authorizes one local structural feasibility probe only.  It asks
whether the exact density of a frozen BP-systematic hard-coset proposal could
possibly support a rejection-envelope or bounded-importance certificate for
the m8 hard sentinel.  It does not authorize an IID estimator, MCMC, a
posterior or `q_top` estimate, remote work, HARD2, formal tuning, held-out
work, or production.

This is deliberately not another attempt to make a chain look converged.
The previous BP-IID result had stable finite weights for its two primary
proposals but no evidence about an unobserved target tail.  A valid global
proposal route needs a quantitative bound on that tail, not a common start,
small jackknife SE, or a favorable visible sector distribution.

## Exact target and candidate bound

The target remains

```text
t(e) = b^|e|,  b = .04/.96,  H_Z e = y,
pi(e) = t(e) / Z.
```

The planted error may generate a legal, deterministic witness state, but it
is never used in `t`, a proposal density, a likelihood, or an estimator.  The
physical all-zero state is outside this nonzero-syndrome hard coset; its
shifted-coordinate zero is the already included planted witness.

For each frozen forward/reverse BP-systematic proposal `q`, use its exact full
three-component mixture density, not BP beliefs treated as posterior facts.
Because an iid Bernoulli(.04) error has total probability at most one after
conditioning on any syndrome,

```text
Z = Pr_p(H_Z e = y) / (.96)^n <= (.96)^(-n).
```

Every legal witness therefore gives the rigorous lower bound

```text
sup_e pi(e) / q(e) >= (.96)^n * b^|e_w| / q(e_w).
```

If this lower bound alone exceeds a frozen rejection-envelope cap, no exact
rejection sampler using that proposal can meet that cap.  This is a one-way
negative test: a small witness value does not prove a global upper bound, good
importance coverage, or a posterior result.  A future positive path would
still need a certified global maximization/upper bound and a separate way to
propagate normalizer and logical-sector error into `q_top`.

## Frozen witness panel

All witnesses are deterministic functions of the code, syndrome, canonical
observable frame, and frozen proposal settings.  The probe may not read an
old importance weight, label statistic, or sampler trajectory to add a
witness.

- the legal planted state;
- one state for every rank-increasing signature from the canonical reduced
  single/pair/triple logical catalog, selected by
  `(state_weight, move_weight, signature, packed_move)`;
- the planted state XOR each individual systematic hard-coset coordinate for
  both frozen information-set orders.

Duplicate physical states are scored once but retain all fixed origins.  This
panel is intentionally adversarial to the proposal, not a distributional
sample.  It neither replaces exact-K0 `U` in a Markov-chain test nor claims to
be posterior-representative.

## Decision and blind-spot checks

- The frozen cap applies only to rejection-envelope feasibility.  Exceeding it
  rejects only this BP-mixture envelope route; it does not show the posterior,
  BP proposals, MCMC, or q=0 impossible.
- Passing the witness screen is `INCONCLUSIVE`, never a proposal or physics
  pass.  A witness set cannot establish the missing global maximum.
- The report must store all scores, legal-state replays, proposal identities,
  source hashes, and its self-hash.  Small-HGP exhaustive tests must verify
  proposal normalization, the exact dominance ratio, and the inequality above.
- No `P/U/L` convergence gate is used because this is an iid proposal-bound
  question.  That avoids treating a deliberately broad uniform start as either
  a required posterior sample or a reason to hide initialization memory by
  starting every chain from P.
- A high acceptance rate, BP message stability, raw state changes, or observed
  ESS is not a decision variable.  The only decision statistic is a valid
  lower bound on the proposal-to-posterior dominance constant.
