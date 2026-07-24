# Center-preserving logical XOR structural review

This is a local structural feasibility experiment, not a sampler result.  Its
maximum authority is `LOCAL_CENTER_PRESERVING_STRUCTURE_VIABLE`; it cannot
produce `q_top`, authorize HARD2, or support a formal-readiness claim.

## Scientific target and proposal

The eventual target remains

```text
pi(e | y) proportional to (.04/.96)^|e|,  H_Z e = y.
```

The historical LSI artifact is used only as a frozen, truth-free source of a
syndrome-derived MAP base, code-only logical moves, and decoded absolute
sector representatives.  For every frozen signature, the candidate anchor is
the lower-weight of

```text
base xor codebook_move
decoded_absolute_representative
```

with packed bytes breaking weight ties.  A rank-first then low-weight rule
retains 127 nonzero signatures.  The physical move is `anchor xor base`, hence
it has zero syndrome, the frozen signature, and is self-inverse.  A later
state-independent draw of this XOR move has exact Metropolis ratio

```text
log alpha_untruncated = (|e xor d| - |e|) log(.04/.96).
```

The planted error is unavailable to candidate construction, selection,
probabilities, and acceptance.  It is opened only after the catalog SHA is
fixed, as an adversarial P diagnostic.

## Red-team boundaries

- Low anchor weight and rank 64 do not prove mixing.  The probe separately
  reports usable cross-signature rank and expected real accepts under one
  attempt of every retained move per T3 macrostep.
- Total state changes are irrelevant unless the logical signature changes.
  Every proposal here has a nonzero signature, so accepted moves are genuine
  cross-sector moves.
- A proposal that only collapses L/U starts into one basin can give false
  P/U/L agreement.  Structural success only authorizes a fresh fixed-clock
  P/U/diverse-L sampler, whose full-label D2 and character gates remain
  mandatory.
- Physical zero is not a legal start for this nonzero syndrome.  Shifted zero
  is the P state and cannot replace U or diverse L initializations.
- Full rank among extremely rare moves proves irreducibility, not finite-time
  usefulness.  The report retains the full-rank bottleneck diagnostic even
  though it is not used to claim posterior accuracy.
- The previous LSI, BP-IMH, HP64, and MAM outputs are not reused as samples.
  The old artifact contributes immutable proposal bytes only.

The structural gate requires exact algebra, full catalog rank, at least eight
independent base/P directions with four expected real accepts over a T3
catalog-sweep clock, and an exact selected non-uphill escape route for each of
the eight frozen legal L starts.  These are necessary implementation gates,
not sufficient convergence conditions.
