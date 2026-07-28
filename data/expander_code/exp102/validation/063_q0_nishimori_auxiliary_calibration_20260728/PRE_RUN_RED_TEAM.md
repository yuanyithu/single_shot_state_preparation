# Validation 063 pre-run scientific red-team

This validation asks a deliberately narrow question: can a Nishimori identity
be calibrated well enough to serve as one mandatory **auxiliary** check on a
future q=0 sampler? It does not ask whether HP64 has converged, does not estimate
`q_top`, and cannot authorize the next remote or formal stage.

## Target distribution and coordinates

For each exact small HGP, the physical truth is drawn iid Bernoulli `p` and

```text
y = H_Z epsilon_true,
pi(e | y) proportional to (p/(1-p))^|e|,  H_Z e = y.
```

This is the `exp101.physics.v2`, `true_posterior`, `x_error/H_Z` convention.
The truth generates the syndrome and is used only after sampling for planted
character scoring. It never enters candidate energy, proposal density or
truth-blind scoring-chain initialization. The physical zero state is not
silently substituted for a nonzero hard coset.

The runner enumerates every physical state. The independent oracle separately
constructs the HGP, logicals, sector model and observable frame directly from
exp101 modules; it does not import the runner or exp102 worker. Both enumerate
all syndrome supports, verify hard-coset membership and the `b^Delta-weight`
ratio, and reconstruct each exact logical posterior. This catches a shared
closed-form assumption by checking the actual physical-state table.

## What the identity can and cannot establish

For a truth-blind candidate `Q_y`, true posterior `P_y` and nonzero logical
character `u`, the calibrated difference is

```text
m_Q(u)^2 - chi_u(L_true) m_Q(u),
E[difference | y] = m_Q(u)^2 - m_P(u)m_Q(u).
```

It vanishes for `Q=P`, but the converse is false. Mandatory counterexamples
include:

- uniform logical output, which is blind for every character;
- a common planted freeze, which passes every character by leaking truth;
- a two-label example with equal scalar moments but `q_top=.64` versus `0`;
- four distinct frozen labels, which is blind after scalar/omnibus averaging
  but exposed by basis and nonbasis character maxima.

Therefore an identity tolerance is not a universal `q_top` bias bound. Passing
this audit cannot replace initialization-family agreement, B-slow-variable
diagnostics, tail/normalizer evidence or an independent confirmation method.

## Frozen controls and optimistic power

The exact catalog contains the correct posterior, wrong temperature,
label permutation, truth-blind MAP delta and uniform logical output. Each row is
reported three ways: all-character omnibus mean, maximum basis-character
effect, and maximum nonbasis-character effect. For `k=1`, the nonbasis group is
explicitly non-applicable; it is never filled with a fabricated zero.

Power is frozen at ensemble sizes `128,512,2048`, with 1000 deterministic
replications. It samples exact truth/candidate score populations and contains no
sampler autocorrelation, initialization bias, between-disorder heterogeneity or
MCMC estimation error. It is consequently optimistic. Character groups are
gated separately so a sparse discrepancy cannot be diluted by an omnibus
average.

The catalog and expected outcomes are frozen before the one-shot report. If an
expected detected group misses either the exact effect floor or rejection-rate
gate, that is a scientific calibration failure. The same is true when an
expected equivalent group misses its exact/equivalence gate. Thresholds are not
relaxed after seeing the outcome.

## Terminal status and evidence persistence

The gate determines exactly one report status:

- no failures: at most
  `NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS`;
- one or more frozen scientific failures:
  `NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`.

Both are valid one-shot scientific records and retain the complete exact,
power, control and failure payload. A scientific failure is not thrown away as
`CONFLICT`. Only source/identity/hash mismatch, malformed or nonfinite data,
changed authority, or implementation inconsistency aborts report creation. The
independent auditor recomputes the payload and accepts either terminal status
only when it agrees exactly with the recomputed gate.

## Future raw and ensemble selection

This calibration creates no sampler raw. The future schema requires a manifest
frozen before results, with fresh iid Bernoulli truth and canonical disorder
order. Every record binds code, registry, model/frame/section, schedule,
sampler, analyzer, character set, seeds, initial states and source artifacts.
All arrays are non-pickle, typed, finite where numeric, and covered by a raw
self-hash.

Scoring chains must be truth-blind and retain independent chain identities and
cross-product inputs. P-start chains may remain adversarial diagnostics but may
not enter the truth-blind scoring ensemble. Basis characters are complete;
sampled nonbasis characters use the frozen finite-population weighting rule.
Masks are `uint64`, including bit 63 at `k=64`.

An ensemble audit is computable only when every disorder in the preplanned
fresh-iid manifest is present, in order, and has passed its full sampler gate.
Missing or failed disorders stop the ensemble audit; they cannot be removed,
replaced, averaged away or turned into a pass-only conditional subset.

## Authority and provenance boundary

All report authority booleans are false: no remote, formal, held-out,
production, posterior-estimation or sole-confirmation permission. The maximum
status remains an auxiliary calibration label, not `READY_FOR_FORMAL`.

The config SHA-binds the runner, independent oracle, auditor, raw validator,
schema, tests and these documents. The one-shot runner requires every bound file
and config to be tracked, the entire worktree to be clean, and all Python
bytecode absent. The report binds the commit, config, bound files, source-tree
digest, model/frame/section fingerprints and its own canonical hash. The auditor
permits only that immutable report as a new file before producing its own
one-shot, self-hashed audit.
