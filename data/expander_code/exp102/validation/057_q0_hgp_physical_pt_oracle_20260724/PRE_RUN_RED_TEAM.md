# Validation 057 pre-run scientific red-team

This review is frozen before any m8 CPPT trajectory or server deployment.

## What is the requested deliverable?

The eventual deliverable is a converged, unbiased q=0 posterior-purity
estimator with a mechanism-independent confirmation path, followed by fresh
tuning and held-out disorders.  CPPT is narrower: it changes the collapsed-B
tempering path from likelihood power at fixed `p=.04` to the physical
posterior family `p=.5 -> .04`.  It can test a possible transport improvement,
but it remains a collapsed-B replica-exchange method and cannot independently
confirm HP64 or authorize formal work.

## Is the target still the requested posterior?

At rung `p_i`, the exact collapsed density is

```text
g_i(B) proportional to (p_i/(1-p_i))^|B|
                      product_j M_{p_i}(Y_j xor (B H)_j),
M_p(s) = Pr_{a iid Bernoulli(p)}[H a=s].
```

The cold endpoint is therefore exactly the required hard-coset posterior at
`p=.04`; the planted error is not in the density.  Adjacent swaps use the four
complete cross-evaluations of `g_i` and `g_{i+1}`.  At `p=.5`, `M_p` and the B
prior are uniform, so one complete eight-bit block sweep refreshes B exactly
when `r<=8` and preserves the uniform endpoint for all supported `r`.

Before any large run, small HGP exhaustive tests must verify the collapsed
density against the full hard-coset posterior, local-sweep stationarity,
adjacent-swap detailed balance, complete local-plus-swap stationarity, the
`p=.5` endpoint, and reference/Numba bit identity through `k=64` bit 63.

## Are the initial states capable of exposing false convergence?

Yes only if the later screen retains P, exact-K0 U, two truth-free B-distinct
MAP starts, and the truth-free low-energy S family.  Physical all-zero is
outside the nonzero-syndrome hard coset; shifted zero is P.  A common P/zero
start would conceal the observed U failure and is forbidden.  HP64 output is
also forbidden as a CPPT warm start because that would make apparent agreement
between the two tempering paths circular.

## Are round trips the requested observable?

No.  Replica round trips and adjacent swap rates are necessary transport
diagnostics, not success criteria.  A later m8 screen must still pass P/U and
all added starts on q_top, full-label D2, B-character D2, normalized full and B
weights, collapsed likelihood, Rhat/ESS, constant-character burn crossing, and
fixed-clock raw replay.  A temperature label can traverse the ladder while the
posterior slow coordinate remains unmixed.

## Did the implementation hide a resource-scale blind spot?

Yes.  For m8, `r=24`, so every physical-p rung needs a `2^24` float64 mass
table.  CPPT32 contains 4 GiB of mass values and CPPT64 contains 8 GiB; the
current implementation also materializes a same-sized log table.  Rebuilding
these arrays inside every trajectory would multiply both setup work and memory
by the worker count.  This is unacceptable even if a one-trajectory smoke
looks fast.  Validation 057 therefore measures table construction separately
and does not authorize an m8 trajectory unless a subsequent contract freezes
one shared immutable artifact, peak memory, construction/replay cost, and an
nd-2/nd-3-only ownership schedule.

## Is CPPT addressing the current highest-value blocker?

Only partially.  HP64 already passed the five-cell P/U diagnostic, while the
formal blocker is the absence of a mechanism-independent confirmation.  CPPT
can expose a path-specific HP error, but agreement would still be same-family
evidence.  Validation 057 has maximum status
`LOCAL_SAME_FAMILY_ORACLE_PASS`; it cannot displace development of an
independent hard-coset mechanism or a rigorous oracle.

## Authority boundary

This validation may certify only mathematical implementation and deterministic
resource facts.  It creates no sampler q_top, no m8/HARD2 result, no
`READY_FOR_FORMAL`, no held-out authority, and no production permission.
