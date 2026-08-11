# Validation 007: Track B — the fast path, and why the anchor is not certified

Status: **`ANCHOR_NOT_CERTIFIABLE_TI_GATE_FAILS_ON_THE_INFORMATIVE_DISORDERS`**.

Two separable outcomes. The engineering succeeded: the fixed-label sector chain
now has a bit-exact numba fast path worth about a factor 1,200 at `m = 3`. The
measurement did not: the certified full-sector TI instrument cannot deliver a
`q_top` anchor at `q = 0.05` under the contract's fail-closed rule, and the
reason is measured rather than asserted.

This changes nothing about Track A. Track B is a preregistered secondary and
could not, by contract, change Track A's terminal status in either direction.

## The fast path

`exp105_pipeline/sector_ti_fast.py`. It replaces the inner loop of
`exp101/src/sector_ti.py::_run_fixed_sector_chain` and nothing else. Integration,
bootstrap, the coarse/fine grid gates, the sector weights and `q_top` all remain
the certified exp101 code, reached through a scoped `fast_chain_installed()`
substitution that restores the original afterwards.

**It is bit-exact with the certified reference, not merely compatible.** I had
told the user otherwise, on the assumption that `sector_ti` drew from numpy's
`default_rng`; it does not. `_run_label_integrations` constructs
`PortablePrng(seed)` — the same portable xorshift128+ twin that makes
`fast_mcmc.py` bit-identical to `reference_mcmc.py`. The kernel threads the
identical state through the identical sequence of draws, including the
`random()` that the reference consumes unconditionally on every attempt even
when the move is accepted on energy alone. Verified: identical `mu`,
`syndrome_mu`, `block_mu` and `acceptance` arrays, and identical RNG state on
return.

The speedup comes from hoisting one identity out of the attempt loop. Whether a
check's syndrome parity flips under a proposal depends only on `H` and the
support, never on the state, so the reference's per-attempt
`np.unique(np.concatenate(...))` and parity loop recompute a constant. Both that
and the qubit supports are flattened to CSR once.

| m | k | sectors | full TI per disorder, reference | with the fast path |
|---|---:|---:|---|---|
| 2 | 4 | 16 | 8.8 min | ~6 s |
| 3 | 9 | 512 | ~20 h (projected) | ~60 s |

## Why the anchor is not certified

`configuration_agreement_probe.json` (SHA256 `063180fd...086bfc32`), 60 disorders
at `m = 2` on tuning-namespace seeds disjoint from the anchor's frozen ones, each
run at the certified 33-point `K_p` grid and again at a refined 129-point grid.

| p | valid at 33 points | valid at 129 points | both valid | max &#124;Δq_top&#124; when both pass |
|---|---:|---:|---:|---:|
| 0.001 | 16/20 | 18/20 | 16 | 0.01397 |
| 0.010 | 8/20 | 15/20 | 8 | 0.01984 |
| 0.040 | 1/20 | 3/20 | 0 | — |

Three things follow, and each is fatal on its own.

**The gate pass rate collapses exactly where the physics is.** Failures
concentrate on disorders whose posterior is *not* concentrated — the ones
carrying information about `q_top`. At `p = 0.001`, where the posterior is
nearly a delta, 16 of 20 pass; at `p = 0.04`, where sectors compete, 1 of 20
does. A representative failure at `p = 0.001` had `q_top = 0.704` with
`grid_tv = 0.737`, against passing disorders whose `grid_tv` was below `0.0006`.

**Fail-closed therefore voids every point, by arithmetic.** The frozen anchor
cell is 8 codes by 8 disorders, and the contract's rule tolerates zero invalid
disorders. Even at the most favourable grid point, an 80 percent per-disorder
pass rate gives `0.8^64 ≈ 6e-7` for a reportable cell. Running the full anchor
would spend hours to produce `NaN` at every point, which is why it was not run.

**Valid-only averaging is not the escape, it is the trap.** The surviving
disorders are the concentrated ones, so their mean `q_top` is biased upward by
construction. This is precisely the conditioning-on-the-gate selection bias that
exp101's scan v3 fail-closed rule exists to prevent, and `PHYSICS_CONTRACT.md`
§11 forbids using such conditional means for publication.

**And the gate has a demonstrated false positive.** Two of the 60 disorders pass
*both* configurations while the two configurations disagree about `q_top` by
`0.014` and `0.020` — larger than the gate's own `grid_q_top_warning` of `0.01`.
Passing does not imply accuracy to the gate's own tolerance. A third case at
`m = 2` is sharper still: one disorder passed at the 129-point grid reporting
`q_top = 0.976`, while a twelve-times heavier configuration reported `0.927` and
failed.

At `m = 3` the same pattern appears: 3 of 8 disorders valid at `p = 0.001`, with
single-disorder `q_top` estimates moving between `0.56` and `0.99` across
configurations.

## What this does and does not say

It says the **instrument** is not adequate here, not that `q_top` is unknowable.
Full-sector TI integrates each sector's free energy with an independent chain;
when several sectors compete, the free-energy differences are `O(1)` and both the
`K_p` quadrature and the per-sector chain equilibration have to be good
simultaneously. At `q = 0.05` they are not, on this family, at the certified
configuration or at the refinements probed.

It does **not** weaken Track A, which never depended on it. It does mean the
certified bound `E[q_top] >= (1 - P_fail)^2` reported in Validation 006 stays
**uncalibrated**: we know it holds, and we do not know how tight it is.

Making it certifiable would mean a better estimator or a better gate for
full-sector TI — a fresh contract with its own red team, not an anchor. Per
permanent discipline 13, this family of attempt is not extended by adding
budget to the same instrument.

## An operational hazard worth writing down

exp101's `prng.py` and `fast_mcmc.py` compile cached numba kernels under
whichever module name imported them: `src.prng` when exp101's own suite runs,
`exp101_certified_src.prng` when the bridge loads it. Numba's on-disk cache
cannot serve both, so a shared cache directory makes whichever suite runs second
fail with `ModuleNotFoundError: No module named 'src.prng'` — in either order,
and looking exactly like a broken package rather than a cache collision. It cost
a diagnosis here, and it would have looked like exp105 breaking exp101.

`exp105/anchor/__init__.py` therefore points `NUMBA_CACHE_DIR` at its own
directory via `setdefault`, which leaves the nd-3 runner's explicit
per-deployment cache in place. Verified: both suites pass in either order.
This is a property of the `exp101_bridge` pattern, not of Track B, so exp102
and exp104 carry it too — they simply never exercised exp101's numba kernels.

## Evidence in this directory

- `configuration_agreement_probe.json`

## Reproduction

```bash
NUMBA_CACHE_DIR=$(mktemp -d) conda run -n 12 --no-capture-output python -m pytest \
  data/expander_code/exp105/tests/test_sector_ti_fast.py -q
```
