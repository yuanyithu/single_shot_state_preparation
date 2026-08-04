# Linear-code trellis WMC structural feasibility

This frozen local probe tests a genuinely different exact representation from
the dense factor-elimination route.  A q=0 hard posterior is a weighted affine
binary code.  For a variable order and parity-check matrix `H_Z`, the
conventional linear-code trellis state exponent at cut `i` is

```text
rank(H_prefix) + rank(H_suffix) - rank(H_Z).
```

This exploits GF(2) linear dependencies that ordinary factor-treewidth
elimination discards.  If the maximum exponent is small enough, a signed
single-copy dynamic program could compute `Z_0` and frozen character
partitions `Z_u` exactly, avoiding the two-copy purity constraint.

The probe measures this state exponent for seven deterministic HGP-aware and
Tanner-min-degree variable orders.  It does not construct a trellis, perform
numeric contraction, or estimate a posterior.  A favorable structural result
would still require a separately implemented affine-trellis recurrence,
outward-rounded signed arithmetic, small-HGP exact-oracle checks, and a
pre-registered character estimator before it can be considered for exp102.

The frozen actionability screen is deliberately conservative: at most exponent
24 and at most 500,000,000 total transition states, with 32 bytes per state
across two working layers.  This probe has no MCMC, remote, held-out, formal,
or production authority.
