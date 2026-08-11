# exp105 status

## Current state

**`PILOT_COMPLETE_NO_SIGN_CHANGE`** — contract frozen, pipeline implemented and
locally gated, locating pilot run. No remote transfer, no production compute, no
published result.

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
4. No nd-3 compute is authorized. The production plan is evaluated and recorded
   in Validation 003 but deliberately **not applied**: writing the constants into
   `config.py` is the freeze, and the freeze happens when the run is authorized.

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
- `validation/003_...`: locating pilot, cost benchmark, evaluated (not applied)
  production plan, and the bracket it opens: `q_c` lies strictly inside `(0, 0.05)`.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

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
