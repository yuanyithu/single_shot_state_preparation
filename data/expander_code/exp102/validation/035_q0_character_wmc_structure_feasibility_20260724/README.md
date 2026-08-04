# Single-copy character-WMC structural feasibility

This frozen local probe tests a structural alternative to the earlier direct
two-copy purity WMC.  For a logical character `u`, a one-copy signed partition
function is

```text
Z_u = sum_{H_Z e=y} b^|e| (-1)^<w_u,e>,    m_u = Z_u / Z_0.
```

For `k=64`, the experiment contract already estimates `q_top` from a frozen
finite population of characters.  Therefore an exact one-copy calculation of
`Z_0` and selected `Z_u` would avoid the earlier two-copy equal-logical-label
constraint.  This probe does not calculate any `Z_u`; it asks whether that
route has a realistic exact factor-elimination structure on the m8 hard
sentinel.

It compares two exact encodings of the same single-copy hard constraint:

- raw `H_Z` parity-check scopes;
- the previous ternary-XOR-chain encoding.

For each encoding it runs deterministic min-degree and exact greedy min-fill
order selection, with a frozen 120-second structural cap per order and an
adjacency-edge safety cap.  Unary weights and character signs do not change
factor scopes, so a favorable one-copy structural result would apply equally
to `Z_0` and every frozen character insertion.  Conversely, a poor width only
rejects these encodings/orders; it does not prove exact character WMC or q=0
impossible.

The probe has no posterior, q_top, MCMC, remote, held-out, or production
authority.  A completed structural result would still require a separate
signed, outward-rounded numerical contraction and exact-oracle tests before it
could be used as a sampler or an estimator.
