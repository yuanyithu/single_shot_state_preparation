# Validation 001: contract freeze, red team, ensemble census, gate infeasibility

Status: **`PASS`**. Authorizes Validation 002. No remote transfer, no production
compute, no physical result.

## What was frozen

`EXPERIMENT_CONTRACT.md` (`exp105.noisy_syndrome_mc.v1`) and
`config/noisy_mc.pilot.v1.json`, bound to
`config/ensemble_registry.pilot.v1.npz` (registry SHA256
`83b4602ad453e466e687d44bbfe594b1a33bc755ff1c3e6b413efe243bdb85e8`, 400 codes,
200 each for `m = 3, 8`), source commit `3a1ce1b`.

The production plan is **deliberately not frozen**. `config.P_TOKENS`,
`CODES_PER_M`, `TRIALS_PER_CODE_P` and `CODES_PER_TASK` are `None`, and every
production entry point raises `ProductionPlanNotFrozen` until Validation 003
evaluates the contract's section 6 rules on pilot measurements. That is asserted
by `tests/test_contract_and_config.py::test_the_production_plan_is_not_frozen_yet`,
so "not yet frozen" is a state the code enforces rather than a note in a
document.

The decoder identity is the frozen `exp103.decoder_mc.v2` / exp104 identity in
every field except `max_iter`, which follows the block length actually being
decoded and is therefore `n + n_c` on the augmented matrix. The installed
backend is the same one exp104 qualified: Python 3.12.12, numpy 2.4.1, scipy
1.17.0, ldpc 2.4.1, decoder extension SHA256
`944a96a657a89fbd04c127edb2eba1033f56de0161ddcd2ba7e57dee76777ccc`.

## Why the requested observable was replaced, measured rather than argued

The request was `q_top` at `q = 0.05`. `pt_gate_infeasibility.json` (SHA256
`c4e2f282fc8a0ecaa16ac5c840f4f3763c79e3142918a1bdf132e4b48d95d52b`), produced by
`measure_pt_gate_infeasibility.py`, measures the inputs to that decision from the
certified exp101 source. It runs no Markov chain.

| m | k | resolved engine at `q = 0.05` | `logical_X` basis weight min/med/max | log10 acceptance bound at `p = 0.05` |
|---|---:|---|---|---:|
| 3 | 9 | `full_sector_ti` | 2/6/24 | -27.6 |
| 4 | 16 | `parallel_tempering_observable_sampling` | 8/22/30 | -34.5 |
| 5 | 25 | `parallel_tempering_observable_sampling` | 4/10/50 | -57.5 |
| 6 | 36 | `parallel_tempering_observable_sampling` | 8/30/48 | -55.2 |
| 7 | 49 | `parallel_tempering_observable_sampling` | 8/20/36 | -41.4 |
| 8 | 64 | `parallel_tempering_observable_sampling` | 12/45/70 | -80.6 |

The gate threshold is `min_cold_logical_acceptance = 1e-4`, that is `-4` in the
same units. A cold L-move flips a whole logical basis operator and, because
logicals lie in `ker(H_check)`, the syndrome term is unchanged and only
`dE = K_p * d|v|` survives; the bound above is therefore an *upper* bound on the
acceptance, not a pessimistic estimate. The shortfall is thirty to seventy-six
orders of magnitude, and even a distance-optimal basis (`d = 6..8` in this
family) leaves about `1e-7`.

`full_sector_ti` is available only at `k <= 10`, that is `m <= 3`. So at
`m >= 4` every disorder would be `INVALID` and scan v3's fail-closed rule would
set every mean, SEM and crossing input to `NaN`. This is arithmetic, not a
tuning problem, and it is the same wall exp102 has been at for 66 validations.

**What exp105 delivers towards the request instead** is a certified one-sided
bound, `E[q_top] >= (1 - P_fail)^2` for large `k` (contract section 7), plus a
transport-free `q_top` anchor at `m = 2, 3` where full-sector TI applies.

## Ensemble composition census

`composition_census_primary.json` (SHA256
`9686f84f3d8620f6038150df6ea7ef2d395ea9ba108a2bfeec24c9a4d7d72c49`), 20,000
accepted codes per `m`, drawn under exp105's own fresh master seed and its own
ensemble namespace.

| m | acceptance | d=2 | d=4 | d=6 | d=8 | d=10 |
|---|---|---|---|---|---|---|
| 2 | 0.3776 | **0.1587** | 0.8413 | - | - | - |
| 3 | 0.7205 | **0.2237** | 0.5865 | 0.1899 | - | - |
| 4 | 0.8623 | **0.1918** | 0.3984 | 0.4017 | 0.0082 | - |
| 5 | 0.9307 | **0.1600** | 0.2810 | 0.4917 | 0.0674 | - |
| 6 | 0.9630 | **0.1379** | 0.2065 | 0.4557 | 0.1996 | 0.0005 |
| 7 | 0.9793 | **0.1156** | 0.1611 | 0.3736 | 0.3378 | 0.0120 |
| 8 | 0.9891 | **0.1009** | 0.1304 | 0.2969 | 0.4061 | 0.0658 |

This is an **independent reproduction of exp104's census**: exp105 uses a
different master seed and a different seed namespace, and every acceptance rate
agrees with exp104's to within 0.0025 and every distance-2 fraction to within
0.006. The ensemble rule is therefore implemented identically, which is what
makes exp104's `q = 0` cells comparable evidence. `m = 2` is new and is used only
by the Track B anchor.

## Scientific red team (permanent discipline 12)

**Target distribution and support.** The product of the i.i.d. bit-flip channel
on `n = 25m²` qubits at rate `p`, the i.i.d. readout channel on `n_c = 12m²`
Z-checks at rate `q = 0.05`, and the uniform distribution over accepted codes.
Support is the whole product space. There is no conditioning, no importance
weighting and no rejection after the fact.

**Coordinates and initial states.** Not applicable to Track A: it draws i.i.d.
samples and runs no Markov chain, so permanent discipline 4 does not bind. For
Track B it does bind, and the answer is that full-sector thermodynamic
integration computes each logical sector's free energy separately under a
sector-preserving chain at fixed label. No trajectory is ever required to cross a
logical or collapsed-B barrier; the barrier appears as a free-energy difference
between independently computed sectors rather than as a transport requirement.
That is precisely why Track B is available at `m = 2, 3` while the PT route is
not available anywhere.

**Estimand and deliverable.** `P_fail(m, p; q = 0.05)`, the probability that a
code drawn from the ensemble is assigned the wrong logical class by the frozen
decoder. The deliverable is a certified bracket for the crossing of
`Delta38(p)`, or the honest absence of one.

**Gate false positives.** The replay gate could pass on a corrupt run if the
corruption were deterministic and shared between the worker and the replayer.
Three things are in place against that: the replayer constructs its own decoder
rather than reusing the worker's, it scores through
`audit_scorer.trivial_class_generators` and `independent_label_map`, which
reconstruct the logical criterion from the section's pivot rule without touching
the exp101 frame the worker uses, and the four stream digests bind the actual
byte sequences. A shared bug would have to survive two independent scorers and
four digests.

**Gate false negatives.** The determinism regression could pass vacuously if it
ran on instances where belief propagation converged, since the ordered-statistics
path is where nondeterminism would live.
`test_augmented_decoder_reaches_the_non_convergent_path` asserts that
`max_iter` is actually exhausted before the determinism assertions run.

**Common failure mode.** Both scorers and both decoders share the same compiled
`ldpc` extension. If that binary were nondeterministic, everything here would
agree and everything would be wrong. This is why permanent discipline 15 requires
the determinism gate to be *measured* in the environment that will run the
production scan, which Validation 004 does again on nd-3, and why exp105 never
mixes artifacts across platforms.

**Selection effects.** The acceptance rate is size dependent (0.378 at `m = 2`
rising to 0.989 at `m = 8`), so the family being studied is *full-rank* random
(3,4)-biregular expander codes. That is part of the definition, not a correction
applied afterwards. Classical distance is recorded for every code because
recording is not selecting; no code is dropped, reweighted or replaced after its
failure rate is known.

**The pilot as a selection effect.** The locating pilot chooses the production
grid and the per-`m` allocation, so pilot codes must not also be production
codes. They are not: the pilot draws from `NAMESPACES["pilot"]` and its own
registry file, and `test_the_pilot_ensemble_namespace_draws_different_codes`
asserts the two draws share no parity-check matrix.

**Authority boundary.** Local only. Nothing here authorizes a pilot run, a
remote transfer, or any production compute.

**What would complete success unlock?** Nothing that is currently blocked.
exp102 remains `BLOCKED_BEFORE_REMOTE` with all four blockers open. exp105
asserts no asymptotic threshold, no exponent, no finite-size-scaling collapse, no
`q_top` estimate at `m >= 4`, and no preparation-channel claim. It delivers a
finite-grid decoder-dependent crossing, a one-sided bound, and a small anchor.

## Evidence in this directory

- `measure_pt_gate_infeasibility.py`, `pt_gate_infeasibility.json`
- `composition_census_primary.json`

## Reproduction

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp105/validation/001_contract_and_redteam_20260811/measure_pt_gate_infeasibility.py

conda run -n 12 --no-capture-output python -m \
  data.expander_code.exp105.exp105_pipeline.ensemble census \
  data/expander_code/exp105/validation/001_contract_and_redteam_20260811/composition_census_primary.json \
  --accepted-per-m 20000 --m-values 2 3 4 5 6 7 8
```
