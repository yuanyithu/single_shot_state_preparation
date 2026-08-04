# Fixed-sector free-energy bridge feasibility

This directory tests a possible alternative to directly forcing one q=0 MCMC
trajectory through every logical sector.  It is intentionally local-only and
has no `q_top`, posterior, remote, formal, held-out, or production authority.

For a fixed logical sector and a logical kernel vector `d`, let `d_t` toggle
the support of `d` one physical bit at a time.  The endpoint map
`e -> e xor d` bijects two logical sectors, while every intermediate ensemble

```text
pi_t(e) proportional to (p/(1-p))^|e xor d_t|
```

lives on the original fixed stabilizer coset.  For the next bit `i`, the exact
identity is

```text
Z_(t+1)/Z_t = E_pi_t[(p/(1-p))^(1-2*(e xor d_t)_i)].
```

The exact small-HGP oracle exercises zero/nonzero syndromes at `.04`, `.10`,
and `.25`; `test_q0_sector_bridge.py` checks the identity by full enumeration.
On m8, the 768 elementary auxiliary updates are separately verified to span
the full 768-dimensional stabilizer subgroup and to leave the logical label
unchanged.  That makes them algebraically suitable for fixed-sector sampling,
but says nothing by itself about mixing or sector-tail coverage.

## V0 lesson

The retained `sector_bridge_probe.json` is V0.  It used one-sided marginals
only.  P and a deterministic same-sector S-tail joined the same weight-62
basin within four exact heatbath sweeps, but all six forward path bits were
zero in their twelve fixed observations.  The apparent product `24^-6` is
therefore *not* a legitimate sector-ratio conclusion: an unseen forward rare
bit can have a material probability.

## V1 arithmetic invalidation

`sector_bridge_probe_v1.json` completed with SHA256
`61c0b551ff49234b84e97ce203c6bebdc052c9a554b266d38054db26ef74116c`, but
its reverse-side estimator reversed the bit exponent as well as the partition
ratio.  The result is retained for audit only and makes no overlap, ratio, or
sector-mass claim.

V2 writes `sector_bridge_probe_v2.json` under fresh seeds.  It records every
path bit at both adjacent bridge ensembles, and its reverse estimate is exactly
`1 / E_(t+1)[(p/(1-p))^(1-2*x_i)]`.  The small-HGP exact oracle now verifies
both directions before the local probe is used.  Even a V2 agreement is only a
short-clock overlap signal; it would not bound unvisited logical sectors or
certify `q_top`.

## V2 result

V2 completed with report SHA256
`436f8bd9f68e786562b8bdd56a620d227ca650d8ef62436409754b76f509bdd6` and
file SHA256
`b967596ef7514a6800677a9885e0d0544c40d7bc818c549d243d917fb380fa9a`.  The
six forward bridge-bit means are zero for both P and S.  At the next ensemble,
all but one mean are one; P's second step is `10/12`.  Consequently P's
forward and reverse products are respectively
`5.232780885631004e-9` and `6.277157494182837e-9`, whereas S gives the former
value in both directions.  Twelve clocks and one trajectory per family do not
distinguish finite-sample noise from a failure to mix within a bridge ensemble.

V2 is therefore `LOCAL_FIXED_SECTOR_BRIDGE_OVERLAP_UNRESOLVED`.  It has no
sector-mass, purity, `q_top`, readiness, remote, held-out, or production claim.
