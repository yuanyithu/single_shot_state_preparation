# Validation 058 pre-run scientific red-team

This local-only validation asks whether an exact full-B-row heatbath is a
mathematically and computationally meaningful independent successor to the
failed full-column Gibbs and CPPT paths.

## Deliverable and authority

The eventual Exp102 deliverable remains replicated posterior-purity sampling
with an independent confirmation, fresh tuning and held-out disorders.  This
validation may only establish row-conditional feasibility and exactness.  It
cannot output q_top, certify m8/HARD2, launch nd-2/nd-3, or authorize formal or
production work.

## Exact target

For collapsed state B and selected matrix row `i`, remove the old row from
each cached A-column syndrome.  For candidate row mask `v`, the conditional is

```text
P(B[i,:]=v | B[-i,:],Y)
  proportional to b^|v| product_j M_p(base_j xor parity(v & h_j) * 2^i),
```

where `h_j` is the support mask of classical column `j` and
`M_p(s)=Pr[H a=s]`.  This is a 24-variable binary factor graph with unary
priors and 32 three-body parity factors for the m8 `(3,4)` matrix.  The planted
error is absent from the conditional.

Small HGP exhaustive enumeration must verify normalization, the production
conditional, detailed balance and full-sweep stationarity.  Reference and
accelerated sampling must consume a frozen PortablePrng stream identically
before any remote successor.  This local feasibility stage implements and
replays only the NumPy reference path; it cannot authorize remote execution.

## Why this addresses the observed slow variable

The failed validation-056 kernel heatbathed one entire B column but left
exact-K0 U far from the low-energy B distribution.  A full row simultaneously
changes one bit in every B column and changes the parity seen by all 32
likelihood factors.  It is therefore an orthogonal B-space block, not an A|B
redraw, logical-label decoration, temperature label, or acceptance counter.

Any later screen must still retain P, exact-K0 U, truth-free B-distinct MAP
starts and low-energy S starts.  Physical zero remains outside the nonzero
syndrome hard coset; shifted zero is P.  Row changes, state changes and label
changes are diagnostics only, not convergence evidence.

## Complexity gate before implementation

The row conditional must not be implemented by assuming a dense `2^24`
enumeration is cheap.  The first gate computes deterministic min-fill and
min-degree elimination orders on the frozen m8 factor scopes.  Development may
continue only if the induced width is at most 18 and the largest binary factor
contains at most `2^19` entries.  The order depends only on H, never on a state,
planted error or observed sampler result.

If the width gate passes, exact variable elimination will be compared against
complete enumeration on both mandatory small HGPs.  Runtime and memory are
measured only after those exact checks.  If it fails, the method stops without
optimizing factor-array kernels.

## Frozen local feasibility gate

Before inspecting any m8 row-conditional values, the local feasibility panel
is frozen to `P/U/M0/S0`.  `P`, `M0` and `S0` are the byte-frozen legal states
from validation 056; `U` is a fresh exact-K0 hard-coset draw whose seed is bound
to this validation and the frozen control SHA.  The 24 rows are always visited
in index order.  No state, row, family or favorable draw may be selected after
seeing a conditional statistic.

The feasibility report passes only if:

- the small-HGP normalization, detailed-balance, stationarity and replay tests
  pass at `1e-13`-scale tolerances;
- every exact entropy, self probability, expected row weight and expected
  Hamming change is finite and within its algebraic range;
- the kernel is nontrivial somewhere on the frozen panel
  (`max expected_hamming_change >= 0.1`), without claiming that this implies
  global transport;
- the largest four-family measured seconds per update, multiplied by 10240
  T1 updates and the frozen factor-two safety margin, is at most 7200 seconds;
- the shared log-mass table is at most 256 MiB and the measured incremental
  Python/NumPy allocation peak for a sweep is at most 1 GiB; and
- repeating every sweep from the same initial state and PortablePrng seed is
  bit-identical for selected rows, old/new masks, final B and cached syndromes.

Passing these gates means only `LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE`.  In
particular, no minimum entropy or movement is imposed separately on P or U:
the true posterior may legitimately make a low-energy row almost deterministic,
while a large one-step U movement can still remain inside the wrong basin.
Those questions require a separately frozen fixed-clock convergence screen.

## Independence and remaining blind spots

This mechanism is independent of replica exchange transport, but it still
shares the collapsed mass identity and A|B reconstruction with HP64.  Exact
full-state enumeration and an independent raw analyzer are therefore still
required.  Even alternating exact row/column heatbaths can share an unobserved
basin, so a finite diagnostic pass would authorize only a replicated screen,
not formal readiness.
