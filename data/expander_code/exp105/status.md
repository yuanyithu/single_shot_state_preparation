# exp105 status

## Current state

**`EXP105_NO_CERTIFIED_CROSSING`** — Track A complete, published, closed.
Track B closed as `ANCHOR_NOT_CERTIFIABLE_TI_GATE_FAILS_ON_THE_INFORMATIVE_DISORDERS`.

At `q = 0.05`, over `p` from `0.001` to `0.07`, the ensemble-mean block logical
failure rate of the frozen BP+OSD-0 decoder is higher for the larger code at
**every** grid point. `Delta38` is certified **positive at all 10 points** and
negative at none, with a simultaneous band half-width of `0.010486`. There is no
crossing, and that absence is certified rather than unresolved.

3,314/3,314 tasks fresh on nd-3 in 2.20 wall hours; 17,617 codes; 1,057,020
trials; replay 337/337 with 110,160 trials bit-exact; 176,170/176,170 cells
`REPORTABLE`; re-derived on macmini through the loader.

Towards the originally requested observable, the certified one-sided bound
`E[q_top] >= (1 - P_fail)^2` reaches `0.97190` at `m = 3, p = 0.001`.

The pilot found `Delta38 > 0` at all 14 grid points from `p = 0.001` to `0.07`.
At `q = 0.05` this family is already above threshold from readout noise alone:
with the data error set identically to zero, an `m = 8` code still fails about 3
percent of the time while `m = 3` fails under 2 percent, and the gap grows with
`m` because a larger code carries proportionally more checks to misread. The
frozen grid rule therefore resolves to its fallback branch, and the production
run it plans would certify `EXP105_NO_CERTIFIED_CROSSING`.

exp105 (`exp105.noisy_syndrome_mc.v1`) measures the ensemble block logical
failure rate of the frozen exp103/exp104 BP+OSD-0 decoder at readout error rate
`q = 0.05`, over randomly generated expander codes, and asks where it crosses as
the code grows.

## Why the observable is not q_top

The request was `q_top` at `q = 0.05`. It is not measurable at `m >= 4` with the
frozen certified instrument, and Validation 001 measures rather than argues this:
the family has `k = m²`, so engine routing sends every `m >= 4` to parallel
tempering, whose validity gate requires cold logical acceptance `>= 1e-4`, while
the measured `logical_X` basis weights (8 to 70) put the acceptance between
`1e-34` and `1e-81` at `p = 0.05`. Every disorder would be `INVALID` and every
published statistic `NaN`.

exp105 measures the decoder-MAP failure rate of the same exp101 posterior
instead. That yields a **certified one-sided bound** `E[q_top] >= (1 - P_fail)²`
for large `k`, informative exactly on the ordered side where the sampling route
is blocked, plus a **transport-free `q_top` anchor** at `m = 2, 3` (Track B).

## Current gates

1. The production plan is **not frozen**. `config.P_TOKENS`, `CODES_PER_M`,
   `TRIALS_PER_CODE_P` and `CODES_PER_TASK` are `None` and every production
   entry point raises `ProductionPlanNotFrozen` until Validation 003 evaluates
   the contract's section 6 rules on pilot measurements.
2. The locating pilot draws from its own ensemble namespace and its own registry,
   so no code that helps choose the frozen grid is later measured on it.
3. The compiled decoder is not bit-portable across platforms (exp104 Validation
   002). Generation, replay and aggregation all happen on nd-3 against the pinned
   nd-3 binary; artifacts are never mixed across platforms.
4. No further exp105 Track A compute is authorized. `report.py` was corrected
   after the measurement, so the live source tree no longer matches the frozen
   configs; the configs are deliberately left bound to the freeze that produced
   the published aggregate, and any new compute needs a fresh freeze.
5. Track B is closed without an anchor. The numba fast path was built and is
   bit-exact with the certified reference (about 1,200x at `m = 3`), so cost is
   not the obstacle. The obstacle is that the TI grid gate fails preferentially
   on the disorders whose posterior is not concentrated -- 16/20, 8/20 and 1/20
   valid at `p = 0.001`, `0.01`, `0.04` -- and has a demonstrated false positive.
   Fail-closed therefore voids every point and valid-only averaging would bias
   `q_top` upward. The Track A bound stays **uncalibrated**: known to hold, not
   known to be tight. Making TI certifiable here needs a fresh contract.
6. `exp105_pipeline` is identity-frozen to the measurement. Track B and the
   corrected report generator live in `anchor/` and `publication/` for that
   reason; `exp105_pipeline/report.py` is exactly as it was when the scan ran.

## Authority and limits

exp105 asserts no asymptotic threshold, no critical exponent, no
finite-size-scaling collapse, no `q_top` **estimate** at `m >= 4`, and no
preparation-channel claim. **Complete success clears no exp102 blocker**; exp102
remains `BLOCKED_BEFORE_REMOTE` with all four blockers open. At `q = 0.05` the
absence of a crossing is a legitimate terminal state, not a failure.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract, its freezing rules and
  its primary-only terminal rule.
- `config/noisy_mc.pilot.v1.json`, `config/ensemble_registry.pilot.v1.npz`.
- `validation/001_...`: contract freeze, scientific red team, ensemble census,
  measured PT-gate infeasibility.
- `validation/002_...`: local implementation gate.
- `validation/003_...`: locating pilot, cost benchmark, the production plan, and
  the bracket it opens: `q_c` lies strictly inside `(0, 0.05)`.
- `validation/004_...`: nd-3 qualification and resource gate.
- `validation/005_...`: production scan, committed replay and aggregation.
- `validation/006_...`: loader-verified publication and the terminal status.
- `validation/007_...`: Track B, the fast path and the anchor's negative.
- `final_results/`: published aggregate, report, curves and plots.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 007: Track B closed. Fast path bit-exact, ~1,200x at `m = 3`; anchor
  not certifiable because the TI gate fails where the physics is and has a
  demonstrated false positive.
- Validation 006: `EXP105_NO_CERTIFIED_CROSSING`, aggregate SHA256 `ff73fd9c...`,
  simultaneous half-width `0.010486`, 10/10 points certified positive.
- Validation 005: scan `PASS` 3,314/3,314 in 2.20 h; replay `PASS` 337/337 with
  110,160 trials bit-exact.
- Validation 004: `PASS`. 166/58/131/17 tests on nd-3, nothing skipped; reserved
  644.7/800 core-hours, wall 7.01/14 h. The first qualification and the first
  resource projection both failed, and between them found four real defects --
  including that the allocation rule had been evaluated with macmini costs for a
  run that happens on nd-3, where a trial at `m = 8` is eight times slower.
- Validation 003: `PILOT_COMPLETE_NO_SIGN_CHANGE`. 44/44 pilot tasks VALID;
  `Delta38` from `+0.064` at `p = 0.001` to `+0.541` at `p = 0.03`, never
  negative. Cost benchmark: a trial at `m = 8` costs 70 times one at `m = 3`.
  Between-code spread below the pilot's resolution, the opposite of exp104, so
  the binding variance term at `q = 0.05` is shot noise rather than code
  diversity.
- Validation 002: local implementation gate, exp105 / exp104 / exp101 suites.
- Validation 001: `PASS`. Census reproduces exp104's composition independently
  (acceptance within 0.0025, distance-2 fraction within 0.006 at every shared
  `m`); PT gate shortfall measured at 30 to 76 orders of magnitude.
