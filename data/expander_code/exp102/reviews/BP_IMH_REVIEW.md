# BP-systematic independence-MH red-team review

## Purpose and authority

This review authorizes one fresh local viability experiment on
`m08_c06,p=.04,d00,attempt022`.  It asks whether an exact, full-support
independence Metropolis-Hastings kernel built from the frozen forward and
reverse BP-systematic proposal constructions can remove adversarial-start
memory at a fixed clock.

The old BP-IID weights motivate the candidate only.  No old sample, estimate,
seed, gate, or pass status is reused.  A local pass can authorize only a fresh
HARD2 diagnostic contract.  It is not `READY_FOR_FORMAL`, a physical `q_top`,
held-out evidence, or production authority.

## Exact target and transition

The target is

```text
pi(e | y) proportional to b^|e|,  b=.04/.96,  H_Z e=y.
```

The planted error is used only as one legal initialization and to construct
result-independent legal logical starts.  It never enters the target,
proposal density, or acceptance ratio.

For each systematic order, freeze the exact three-component proposal

```text
q_order = .90 q_BP + .09 q_prior + .01 q_uniform.
```

The transition proposal is

```text
q_FR = .5 q_forward + .5 q_reverse.
```

Every component has full support in an exact affine hard-coset coordinate
bijection.  Given a proposed state `e'`, accept with

```text
min(1, b^(|e'|-|e|) q_FR(e) / q_FR(e')).
```

The current proposal density is cached only as an exact state function; it is
never approximated from BP messages or samples.  Small-HGP exhaustive tests
must verify proposal normalization, complete transition row sums, detailed
balance, and stationarity before the real-code run.

## Initialization and fixed clock

There are three legal adversarial families, each with eight independent RNG
streams:

- `P`: the planted hard-coset state;
- `U`: independent exact-K0 uniform hard-coset states;
- `L`: eight distinct deterministic low-energy logical states from the first
  eight entries of the canonical reduced single/pair/triple catalog.

The L family deliberately does not clone one favorable logical state.  The
physical all-zero state remains illegal for this nonzero syndrome; shifted
zero is P and cannot be a fourth independent start.

Every trajectory uses exactly 256 burn steps and 2048 fixed measurement
steps.  There is no adaptation, restart, extension, thinning selected from a
trace, or running until a sample quota is met.  Raw stores every proposed and
current state, proposal source/component, acceptance uniform and decision,
label, weight, initial/burn/final state, identity, hashes, and counters.

## Predeclared viability gates

Algebra, deterministic seed replay, raw identity, and all acceptance decisions
must pass exactly.  Statistical gates then require:

- every U trajectory has a real accepted state change during burn and ends burn below normalized
  physical weight `.15`;
- every trajectory has at least 16 accepted measurement state changes and a state-change rate at
  least `.01`;
- every family has diagnostic `q_top` delete-one-trajectory SE at most `.03`;
- P/U/L pairwise diagnostic `q_top` differences are at most `.04` and at most
  `3 SE + .005`;
- normalized physical-weight differences are at most `.01` and at most
  `3 SE + 1/n`; normalized B-weight differences are at most `.02` and at most
  `3 SE + 1/576`;
- all 64 basis-character mean differences are at most `.04`, and their mean
  squared difference is at most `.04`;
- each P/U/L pair passes the full 64-bit label-collision distribution gate
  `max(0,D2_norm)+3 SE_D2 <= .04`; this prevents equal purity and equal
  single-character marginals from hiding disjoint logical-sector support;
- physical weight, B weight, and nonconstant basis characters have split
  `Rhat <= 1.05` and aggregate bulk ESS at least 400;
- if a character is constant after burn while an initial chain had the
  opposite sign, every such opposite chain must reach the common sign during
  burn;
- early/late family `q_top` and normalized-weight diagnostics pass the same
  fixed tolerances.

Observed total acceptance, state changes, or BP component counts cannot rescue
a failed distribution gate.  Conversely, rank-64 logical motion is not a
necessary gate when the actual low-temperature posterior may be concentrated.

## Decision boundary

`LOCAL_BP_IMH_TRANSPORT_VIABLE_FOR_HARD2` means only that this exact kernel
deserves a fresh HARD2 T/2T comparison and an independent mechanism.  It does
not resolve the unobserved-tail problem by itself.  Any failed family,
comparison, replay, or algebra gate terminates this configuration as
`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`; its raw cannot be extended or pooled.
