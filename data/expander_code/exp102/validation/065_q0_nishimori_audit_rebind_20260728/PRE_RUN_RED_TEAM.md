# Validation 065 pre-run scientific red-team

## Question and authority

The sole question is whether the immutable validation-063 exact report can be
independently audited after replacing a brittle English-message comparison by
numeric, structured failure identities.  Focused pre-run tests found that it
cannot: after the English mismatch, full reconstruction reaches a MAP-tie
semantic mismatch in 11 numeric fields.  The one-shot must persist that
conflict, not convert invariant terminal failures into an audit pass.

Every authority flag is false.  In particular, an audit pass is not
`READY_FOR_FORMAL`, is not a Nishimori calibration pass, is not evidence that
HP64 converges, and cannot select a parameter point or disorder.

## Immutable evidence boundary

The report bytes and file SHA, its canonical self-hash, the 063 config bytes,
the report source commit, the report source-tree digest, and every bound source
blob at that commit are checked before numerical work.  Later edits to the 063
README are irrelevant because the historical blob, rather than the current
working-tree copy, is the authority.  The original report is absent from its
own source commit, so this validation cannot silently treat a regenerated
report as source.

The recorded v2 audit conflict is also byte- and self-hash-bound.  It must say
that no independent audit was created and identify the same immutable report.

## Independent numerical reconstruction

The frozen 063 independent oracle reconstructs the HGP, physics-v2 posterior,
logical distributions, controls, deterministic power simulation, and gate
without importing the 063 runner or exp102 sampler.  Validation 065 compares
all scientific payload fields against this reconstruction.  Numeric tolerance
is limited to the already frozen `2e-13` cross-implementation comparison; all
keys, shapes, booleans, strings, nulls, integer identities, and nonfinite checks
are exact.

The rebind does not accept “same terminal status” as sufficient.  Golden rows,
exact rows, power rows, chain controls, and gate metadata all have to match.
Here the terminal gate is invariant but the full payload is not, so the result
is `CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS`.

The conflict witness enumerates all 11 differing paths and all three tied hard
cosets.  For each coset it records the syndrome, report/oracle labels and
posterior masses, their equal integer weight enumerators, and a canonical
minimum physical state and weight for each selected label.  A tiny floating
posterior delta is not treated as a physical preference when the enumerators
are exactly equal.

## Failure identity and grammar boundary

The decision is rebuilt from numeric fields and frozen thresholds before any
legacy sentence is parsed.  A structured exact-control identity contains:

```text
scope, model_id, p, control, character_group,
expected_outcome, reason_codes, observed values, thresholds, ensemble size
```

For the immutable report, `equivalence gate failed` and
`equivalence power failed` are allowed aliases for the same exact-control key.
No fuzzy matching, substring guessing, reordered path, or unknown prefix is
accepted.  The parsed report and oracle keys must each be unique and equal the
independently rebuilt numeric keys.  Thus language can be normalized without
weakening the gate.

The current 14 failures must arise from
`EQUIVALENCE_RATE_BELOW_MINIMUM`.  If exact effects, detection rates,
applicability, missing rows, or chain controls produce another failure, it is
retained as a distinct structured reason rather than hidden by the known text
conflict.

## Known scientific blind spots

The Nishimori identity remains blind to uniform logical output, common
truth-leaking planted freeze, and the frozen equal-moment counterexamples.
Passing an audit of the calibration cannot provide a universal q_top bias
bound or unobserved-tail control.  The report's null bias-bound fields and
known-blind controls are mandatory and independently checked.

This work deliberately does not respond to a precision failure by increasing
replications, changing `.01`, dropping hard controls, or reporting only passed
groups.  Those would be new calibration designs, not an audit rebind.

## One-shot and verifier separation

The audit runner requires a clean tracked source and no Python bytecode, then
writes one self-hashed conflict JSON file with create-exclusive semantics.  The
verifier allows exactly that untracked file, imports neither the audit runner
nor its helpers, independently repeats the source, numerical, tie, gate, and
grammar checks, and writes a second create-exclusive self-hashed JSON file.

Any source/hash/algebra/nonfinite/identity mismatch is `CONFLICT`; it is never
converted into a scientific calibration failure.  Existing output is never
overwritten or removed in place.
