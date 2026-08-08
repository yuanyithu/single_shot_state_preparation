# Validation 001: contract freeze, red team, ensemble census, local gates

Status: **`PASS`**. Authorizes Validation 002. No remote transfer, no production
compute, no physical result.

## What was frozen

`EXPERIMENT_CONTRACT.md` and `config/ensemble_mc.v1.json` /
`config/ensemble_mc.remote.v1.json`, bound to
`config/ensemble_registry.v1.json` (registry SHA256
`7e40ff18fdf4fd52476894dc21caa516e16a1b97cdfd2a9ad9f803c709f315d4`, 12,000
codes, 2000 per `m` for `m = 3..8`).

The decoder identity is byte for byte the frozen `exp103.decoder_mc.v2` identity,
so exp103's 624 published cells remain comparable evidence and any difference
between the two experiments is attributable to the ensemble and the estimator
rather than to the decoder.

## Ensemble composition census

Codes are generated from seeds and accepted on two algebraic criteria only:
`rank(H) = 3m` over GF(2), and no repeated parity-check matrix at that `m`.
Nothing is selected on distance, expansion or decoder behaviour.

Measured on 20,000 accepted codes per `m`, and repeated with an independent
master seed (`composition_census_primary.json`,
`composition_census_independent.json`):

| m | acceptance | d=2 | d=4 | d=6 | d=8 | d=10 |
|---|---|---|---|---|---|---|
| 3 | 0.7212 | **0.2291** | 0.5903 | 0.1807 | - | - |
| 4 | 0.8632 | **0.1945** | 0.3960 | 0.4023 | 0.0072 | - |
| 5 | 0.9332 | **0.1541** | 0.2837 | 0.4951 | 0.0670 | - |
| 6 | 0.9618 | **0.1331** | 0.2114 | 0.4562 | 0.1988 | 0.0005 |
| 7 | 0.9793 | **0.1135** | 0.1653 | 0.3784 | 0.3319 | 0.0109 |
| 8 | 0.9900 | **0.1042** | 0.1221 | 0.2992 | 0.4079 | 0.0665 |

The independent seed reproduces every distance-2 fraction to within 0.0006 and
every acceptance rate to within 0.0022, so these are measurements rather than
draws.

**The distance-2 fraction falls monotonically with size, and it is the reason
exp103's primary contrast was positive everywhere.** exp103's eight-code panels
happened to contain `0, 3, 2, 2, 0, 1` distance-2 codes for `m = 3..8`. The
`m = 3` panel drew none where about 23 percent were due, so `P_fail(m=3)` was
biased low and `Delta38 = P_fail(m8) - P_fail(m3)` was pushed positive at every
grid point. Reweighting exp103's own measured per-distance rates by the census
composition puts a negative-to-positive reversal of `Delta38` between `p = 0.05`
and `p = 0.06`.

That reconstruction is exp103 evidence viewed through a composition measurement.
It is not an exp104 result, it is not certified, and exp104's terminal decision
does not consult it.

## Scientific red team (permanent discipline 12)

**Target distribution and support.** The i.i.d. bit-flip channel on `n = 25m²`
qubits at rate `p`, and the uniform distribution over accepted codes. Support is
the whole product space; there is no conditioning, no importance weighting and no
rejection after the fact.

**Coordinates and initial states.** Not applicable in the sense discipline 4
intends. exp104 draws independent samples directly and runs no Markov chain, so
there is no slow variable, no collapsed-B basin, no self-loop and no barrier to
cross. Every trial is an independent draw scored once.

**Estimand and deliverable.** The pooled failure fraction per `(m, p)`; the
contrast `Delta38`; a certified bracket; and, only if a bracket is certified,
an interpolated crossing location with a percentile bootstrap interval. Nothing
asymptotic is delivered.

**Gate false positives.** A cluster bootstrap that under-covers would certify a
crossing that is not there. Mitigations: codes are the resampling unit and carry
their whole curve, so between-code variation is inside the interval by
construction rather than assumed away; the band is simultaneous over the nine
grid points of the primary contrast; and the loader recomputes the band and the
decision from the stored per-code counts, so a published band that cannot be
re-derived is rejected.

**Gate false negatives.** A band too wide to certify anything, which is exactly
what happened to exp103. Mitigation: the simultaneous scope is the primary
contrast's grid only, not six curves and five contrasts at once, and the bracket
does not require adjacent grid points.

**Common-mode failure.** The band and the point estimate come from the same
trials, so a systematic error in the decoder or the scorer would move both
together and no interval would notice. Mitigations: two independent scorers are
required to agree exhaustively; the replay path constructs its decoder
independently of the worker path and both are required to agree trial by trial;
and Validation 002 requires the exp104 code path to reproduce frozen exp103 raw
shards bit for bit, which ties this pipeline to an already published measurement.

**Selection effects.** The acceptance rate is size dependent, 0.721 at `m = 3`
against 0.990 at `m = 8`, so the filter bites harder at small `m`. It is
nonetheless a definitional criterion: it is algebraic, it is applied identically
at every `m`, it never refers to any measured outcome, and it is what makes
`k = m²` hold exactly across the panel. The family under study is *full-rank*
random (3,4)-biregular expander codes and the contract says so. A different
choice would be a different family, not a correction to this one.

**Authority boundary.** Finite grid, one decoder, one ensemble, `q = 0` code
capacity. No asymptotic threshold, no exponent, no finite-size scaling, no
`q_top`, no MLD, no preparation channel.

**"What would complete success unlock?"** Nothing that is currently blocked.
exp102 stays `BLOCKED_BEFORE_REMOTE` whatever exp104 returns; exp104 authorizes
no exp102 stage and its contract never claimed it would. What complete success
delivers is a certified crossing location for this decoder on this ensemble,
and a measured explanation of why exp103 could not see one.

## Local gates

- 131 tests pass in `data/expander_code/exp104/tests`, including
  `test_decoder_determinism.py`, which asserts that belief propagation actually
  fails to converge in the tested regime before asserting that the decoder is a
  deterministic function of its input. Permanent discipline 15 requires this to
  be resident, and the 10 percent replay policy makes it load-bearing.
- The certified subsets exp103 qualified against still pass unchanged: 58 in
  exp101, 17 in exp102.
- `local_resource_preflight.json`: `PASS`. Generation upper bound 16.33
  core-hours on macmini, committed replay 1.71, reserved 40.08 against a 100
  core-hour cap, projected peak RSS 7.95 GiB against 12.
- Every `m` is benchmarked directly rather than extrapolated from an anchor.
  exp103 projected `m = 6` and `m = 7` from its `m = 8` anchor, which inflated
  them by more than a factor two; a gate carrying that much slop cannot
  discriminate.

## Design decisions recorded here

**Four trials per code is optimal, not a compromise.** With a measured per-code
construction cost of about ten trials at `m = 8` (0.96 s against 0.10 s per
trial, dominated by `logical_pauli_operators`) and a between-code standard
deviation near 0.15 in the crossing region, minimising the variance of the
ensemble mean at fixed budget gives three to four trials per code.

**The grid stops at `p = 0.10`.** Above it every code fails essentially always,
the between-code standard deviation drops below 0.001, and belief propagation
always exhausts `max_iter = n`, which is the most expensive regime. exp103 spent
roughly 60 percent of its compute there.

**`power_check.py`** records the pre-registration power analysis: synthetic
per-code rates drawn from exp103's measured per-distance rates and assigned by
the exp104 registry's actual composition, pushed through the real
`cluster_bootstrap`, `classify_crossing` and `crossing_location`. It returns a
simultaneous half-width near 0.022 against exp103's 0.2601. It is a design check
on the machinery and the band width, not a prediction of the outcome, and its
point estimates are model output.

## Authority

Local implementation gate only. Authorizes Validation 002. Grants no remote
authority, publishes no physical result, and clears no exp102 blocker.
