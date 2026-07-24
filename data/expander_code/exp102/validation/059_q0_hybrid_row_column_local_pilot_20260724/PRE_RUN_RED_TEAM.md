# Validation 059 pre-run scientific red-team

This local-only successor tests whether two exact but complementary collapsed-B
Gibbs blocks solve the failure exposed by validations 056 and 058.  It is
frozen before any hybrid trajectory is generated.

## Deliverable and authority

The Exp102 deliverable remains a converged posterior-purity sampler plus a
mechanism-independent confirmer, followed by fresh tuning and held-out
disorders.  This validation can only pass the necessary B-transport pilot as
`LOCAL_HYBRID_B_NECESSARY_GATES_PASS`.  It cannot output q_top, certify m8 or
HARD2, use remote nodes, authorize formal/held-out/production work, or serve as
the independent confirmer.

## Target and support

The state is the collapsed `r x r` binary matrix B.  At physical `p=.04`,

```text
g_p(B) proportional to b^|B| product_j M_p(Y_j xor B H_j),
b = p/(1-p),
M_p(s) = Pr_{a~Bernoulli(p)^n}[H a=s].
```

The planted error is absent from every transition weight.  One macroclock is
frozen to the ordered composition:

1. choose a B column uniformly and apply the exact direct-positive full-column
   heatbath `RFCG-C24-DPB12-S1`;
2. choose a B row uniformly and apply the exact width-12 full-row heatbath
   `RFRG-R24-VE12`;
3. at measurement clocks, draw `A|B,Y` exactly and record the full hard-coset
   state.

Both choices are state-independent PortablePrng draws.  The ordered composition
need not be reversible, but each component preserves `g_p`, so complete
small-HGP transition matrices must verify full-clock stationarity.  The full
state must satisfy `H_Z e=Y` after every observation.

## Initial states and the actual slow coordinate

The frozen panel has four families and four independent trajectories per
family:

- `P`: the legal planted state;
- `U`: four fresh independent exact-K0 uniform hard-coset states;
- `M0`: the frozen truth-free weight-62 MAP anchor;
- `S0`: a frozen truth-free low-energy state with a B block distinct from P
  and M0.

Every trajectory has independent burn, measurement and observation seed
identities.  Physical zero is illegal for this syndrome; shifted zero is P.
Common P/zero starts, HP warm starts, cloned U states and after-result initial
selection are forbidden.

The primary slow coordinate is the B distribution, not total acceptance,
state changes or logical-label changes.  Raw therefore retains every B matrix,
row/column selections and old/new masks, normalized B weight, collapsed
log-likelihood, full state/weight/label, eight time blocks and separate row and
column counters.

## Frozen clocks and necessary gates

The local pilot uses exactly 256 burn plus 1024 measurement macroclocks.  There
is no adaptive stopping, extension, resampling, pooling with validation 056,
or result-driven row/column ratio.  All 16 trajectories must replay exactly.

Using only the last four measurement blocks, every family pair must satisfy:

- absolute normalized mean B-weight difference `<= .015`;
- absolute mean collapsed log-likelihood-per-factor difference `<= .50`;
- mean squared difference of all 576 B-bit means `<= .04`.

Within each family, first-four versus last-four block drift must satisfy the
same B-weight and likelihood thresholds.  Additionally:

- at least three of four U trajectories must end burn with normalized B weight
  `<= .065` and collapsed log-likelihood per factor `>= -6.5`;
- every P/M0/S0 trajectory must have at least one real B-column or B-row change
  in measurement;
- all hard-coset, cache, hash, finite-value and replay checks must pass; and
- the largest replay-inclusive local trajectory time must be `<= 1800s`.

These deliberately loose gates only reject the gross U-versus-low-energy split
seen in validation 056.  Passing does not prove equilibrium, posterior purity,
or unobserved-basin coverage.  It only permits a fresh portable preflight and
then a full `P/U/M0/M1/S x8`, `2048+8192` T1 diagnostic on nd-2/nd-3.

## Common-mode failure and next boundary

The hybrid remains a collapsed-B Gibbs method and reuses the direct-column
conditional identity.  It can make all starts fall into a shared basin while
missing target mass.  A later T1 must retain full B bit/row/column/dense
characters, logical characters, D2, Rhat/ESS, weight/likelihood, burn crossing
and MAP-basin gates.  Even a T1 pass would still require a mechanism-independent
defect-space or rigorous normalizer/tail confirmation before formal readiness.
