# DTC V0 local D=0 transport preflight

This directory contains a bounded local diagnostic for the independent
defect-tempered conditional mechanism. It is not a posterior experiment and
cannot produce `q_top`, `READY_FOR_FORMAL`, or a production authorization.

## Frozen mechanism

The finite-rung target is

```text
pi_j(e) proportional to exp[-K_p |e| - Kq_j D(e)]
D(e) = |H_Z e xor y|
```

At every finite `Kq_j`, conditioning a fixed-clock state on `D=0` is exactly
the q=0 hard-coset posterior `exp[-K_p |e|]` subject to `H_Z e=y`. The 21-rung
ladder descends from `Kq=4` to an exact iid Bernoulli(`p`) `Kq=0` endpoint;
neighbor swaps use `(Kq_i-Kq_j)(D_i-D_j)`. This establishes the conditional
target identity, not finite-budget mixing.

- Cell: `m08_c06`, `p=.04`, `d00`, `attempt022`.
- Method: `DTC21-S1`, one sweep per rung and round.
- Resources: 256 burn plus 2048 fixed measurement rounds.
- Starts: eight trajectories each from planted `P`, exact-K0 uniform hard-coset
  `U`, and deterministic legal low-energy `L`.
- Manifest: `local_m8_d0_transport_v0/MANIFEST.json`, SHA256
  `751f76bec3831fd8fad39ee96972bd2a5e54a3da4a2e87a90ba202554decb337`.

Physical all-zero is not a legal start because this cell has nonzero syndrome;
in shifted coordinates zero is already `P`. The `L` start was frozen before
sampling as the deterministic minimum-energy nontrivial reduced logical
single/pair/triple candidate (weight 67, versus `P` weight 63).

## Decision rule

Every P/U/L family separately needs all of the following:

- at least 256 `D=0` clocks and eight cold `D=0` returns in every trajectory;
- at least 64 D=0 label changes, with six of eight trajectories contributing
  at least eight changes;
- rank at least 16 in D=0 label deltas;
- D=0 leave-return coverage for at least 16 basis and 16 frozen nonbasis
  characters.

This is deliberately only a local transport signal gate. Defect returns,
swap acceptance, ordinary state changes, and even many label changes can all
occur in a small logical subgroup and cannot substitute for the gate.

## Terminal result

All 24 raw files completed and passed the frozen full seed replay. The report
SHA256 is `58f1dbb227d748edeb266fe42fefd74768dc2384d3bcf2dfc850b6339000e49c`.
The separate `independent_raw_audit.py` does not invoke the sampler and
rebuilds the code, syndrome, P/U/L starts, labels, defects, D=0 masks, raw
digests, counters, and gate directly from NPZ with `allow_pickle=False`; it
passed with SHA256
`6990ea671153446e65592b29f4d1a3ad08c954abb9767476e1ff193e4df8cb2f`.

The terminal status is `LOCAL_D0_TRANSPORT_NOT_VIABLE`.

Verification passes: the dedicated defect-tempered suite reports `17 passed`,
and the exp102 suite plus exp101 HGP/logical/exact regressions report
`593 passed` (two upstream multiprocessing deprecation warnings). The complete
second regression log and explicit exit code are retained as
`full_regression.log` and `full_regression.exit`.

| family | D=0 label changes | delta rank | basis LR | nonbasis LR | decisive failures |
|:--|--:|--:|--:|--:|:--|
| P | 166 | 1 | 15/64 | 31/64 | 1/8 chains with >=8 changes; rank |
| U | 61 | 2 | 8/64 | 52/64 | low D=0 count, total changes, chain count, rank |
| L | 201 | 3 | 19/64 | 55/64 | 1/8 chains with >=8 changes; rank |

All three families make many D=0 leave-return excursions, but their
cross-label changes are concentrated in one or two trajectories and cover
only a few logical directions. This is direct evidence against global
hard-coset transport for this exact mechanism and budget. The raw is terminal:
do not extend, pool, or interpret it as a posterior estimate, and do not
deploy this kernel to HARD2 or any formal exp102 stage. It does not establish
mathematical impossibility.
