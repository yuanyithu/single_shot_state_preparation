# Collapsed SMC V0 local feasibility diagnostic

## Scope

This is a bounded local test of `CSMC64-B8-S1-N128` on only
`m08_c06,p=.04,d00,attempt022`. It cannot calculate `q_top`, estimate the
posterior, authorize a remote HARD2 run, create `READY_FOR_FORMAL`, or modify
the formal exp102 contract. Its sole decision is whether an exact-base
collapsed SMC bridge suffers immediate weight or genealogy collapse.

The manifest is frozen before any raw result is produced. Its SHA and all
source-file hashes bind the cell, code, syndrome, mass table, lambda schedule,
eight population seeds, and raw schema. A local worktree is sufficient only
for this diagnostic; this is not a clean-source deployment.

## Scientific red-team

### Target and support

For the full-row-rank classical seed matrix `H`, write the hard constraint as

```text
H A xor B H = Y.
```

Integrating the `A` columns gives the exact collapsed B marginal

```text
pi_lambda(B) proportional to prior_p(B) * L(B)^lambda,
L(B) = product_j Pr_p[H a_j = (Y xor B H)_j].
```

At `lambda=1`, drawing `A|B` exactly would reconstruct the required q=0
posterior. At `lambda=0`, `B` is exactly iid Bernoulli(`p`). The test uses no
planted error in an energy, importance weight, proposal, resampling decision,
or mutation decision.

The physical all-zero error is outside this cell's nonzero-syndrome support.
`P`, `U`, and `L` are therefore not forced into the population sampler:
they would replace its mathematically exact base distribution with a biased
warm start. This is not a loophole in the initial-condition check. Instead,
four populations use a column-major iid construction and four use a row-major
iid construction. These are distributionally identical but independently
seeded and catch a B-layout or initializer-order mistake. Population roots
are never shared across those eight runs.

### What could look good but still be wrong

- Exact `A|B` redraws, many B-block changes, or a finite final particle cloud
  do not establish posterior mixing.
- A high CESS at one level does not prevent later resampling from erasing all
  independent roots.
- A surviving set of roots does not prove that all cold modes were found.
- Agreement with HP64 would share the collapsed algebra and is not a physical
  independent confirmation.

Accordingly, the V0 test does not use `q_top`, logical-label changes, state
changes, acceptance, or runtime as a pass proxy. It records them only where
they are algebraically required to audit the resample-move path.

### Exactness checks before finite-budget claims

The test suite enumerates the `n=10` and `n=13` HGP hard cosets at
`p=.04,.10,.25` and verifies that the collapsed cold B mass equals the
enumerated full posterior mass. It also checks the fixed bridge schedule,
systematic parent map, root ancestry, and exact reference/Numba transcript
identity. The raw-only analyzer independently recomputes every stored B
likelihood, incremental normalized weight, CESS, parent vector, root vector,
root-family summary, and finite-value condition with `allow_pickle=False`.
A separate full seed replay reruns every population before the raw-only audit.

## Frozen mechanism

- Method: `CSMC64-B8-S1-N128`.
- Bridge: the existing HP64 quadratic schedule
  `lambda_i=i^2/63^2`, with its frozen SHA
  `9aa5269ce0eee77473f7d0375ea9d007aa31cf6daf1e47d0cb4af23224be45c0`.
  Reusing this schedule avoids an unblinded schedule search; it does not reuse
  HP trajectories, raw files, seeds, or estimates.
- At every nonzero lambda stage: compute the exact incremental likelihood
  weight, systematic-resample unconditionally, then give every child one full
  exact 8-bit collapsed B-block heatbath sweep at that lambda.
- Each resampled output slot receives a new PortablePrng substream. A cloned
  parent never clones an RNG state.
- Eight independent populations: `4 column_major + 4 row_major`, each with
  128 particles. There is no adaptation, restart, extra mutation sweep,
  altered population size, or choice based on a result.

## Feasibility gate

Every one of the eight frozen populations must meet all of these values;
otherwise this configuration is terminally
`LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`.

| quantity | required value |
|---|---:|
| minimum per-stage CESS / N | `>= .50` |
| maximum per-stage normalized incremental weight | `<= .10` |
| final root-family ESS | `>= 16` |
| final distinct roots | `>= 32` |
| final largest root fraction | `<= .20` |

A pass has the deliberately narrow status
`LOCAL_COLLAPSED_SMC_WEIGHT_GENEALOGY_VIABLE`. It means only that a complete,
fresh, independently reviewed sampler design may be worth developing. It is
not evidence of q=0 posterior convergence, a valid `q_top`, a HARD2 pass, or
a formal experiment result. A failure rejects only this fixed SMC
configuration and resource envelope; it is not a mathematical impossibility
claim and must not be "rescued" by extending, pooling, or retuning its raw.

## Terminal result

The frozen local run completed all eight populations, then passed a full
deterministic seed replay and a separate raw-only audit. Its terminal status
is `LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`.

- Manifest SHA256:
  `ee3496f1d08e3e78db306f91b921a96d402c80a225b8c7e214978590e615f979`.
- Runner replay SHA256:
  `4f59ea1766432dece1b4d5bac263d906ba426cdef42c196e6fac0b016650b0f8`.
- Report SHA256:
  `4bea937e5b6ae60dc4971d516b1c068da8f6cc1602d75947afd9903549311b70`.
- Independent raw-only audit SHA256:
  `73aff5e55eda314b8382813bd6a1feb3c64a25d3eda6bc11071f9161db224a23`.

All eight populations fail the root gate: final distinct roots range from one
to five, final root-family ESS from `1.00` to `2.74`, and largest-root mass
from `.49` to `1.00`. The decisive diagnosis is more specific than a blanket
SMC failure. At stage 15 the median root ESS is still `57.49/128`, but by
stage 31 it is only `1.22/128`; after that the median remains near one. At the
same selected stages the incremental CESS is often about `.9N` and the
per-stage maximum normalized weight about `.01`. Thus unconditional systematic
resampling at every one of 63 stages repeatedly coalesces roots even when a
single step looks benign. It is an algorithmic failure of this frozen
always-resample configuration, not evidence that the collapsed marginal is
ill-defined or that all non-resampling annealed methods fail.

The raw cannot be extended, pooled, or used for `q_top`. Any successor must
be a fresh contract with fresh seeds and raw; it must address this explicitly
identified resampling mechanism rather than hiding it by dropping ancestry
diagnostics or lowering the gate.
