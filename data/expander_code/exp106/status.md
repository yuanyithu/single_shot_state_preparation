# exp106 status

## Current state

**`IMPLEMENTATION_COMPLETE_PLAN_NOT_FROZEN`** — the package and both equality
gates are built and green locally; the production plan does not exist yet and
every production entry point refuses.

exp106 (`exp106.noisy_syndrome_mc.q001.v1`) measures the ensemble block logical
failure rate of the frozen exp103/exp104 BP+OSD-0 decoder at readout error rate
`q = 0.01`, over randomly generated expander codes at `m = 3..8`, and asks whether
it still crosses as the code grows.

## Why exp106 exists

exp104 and exp105 bracketed the question without answering it: a certified
crossing at `p_cross = 0.05512` when `q = 0`, and a certified *absence* of one
anywhere in `p in [0.001, 0.07]` when `q = 0.05`. The readout threshold of this
decoder on this family therefore lies strictly inside `(0, 0.05)`. exp106
measures one interior point.

**Both terminal states are real.** At `p = 0.04`, moving `q` from 0 to 0.05 shifts
`Delta38` by `+0.49`; the `q = 0` dip it would have to erase is only `0.053` deep.
A readout penalty linear in `q` contributes `+0.098` at `q = 0.01` and erases it;
one going as `q²` contributes `+0.020` and the crossing survives. That is why the
band matters more here than in either predecessor.

## Current gates

1. The production plan is **not frozen**. `config.P_TOKENS`, `CODES_PER_M`,
   `TRIALS_PER_CODE_P` and `CODES_PER_TASK` are `None` and every production entry
   point raises `ProductionPlanNotFrozen` until Validation 003 evaluates the
   contract's section 6 rules. 29 tests skip for this reason, and that skip is
   load-bearing: nd-3 qualification requires zero skipped tests, so an unfrozen
   plan cannot reach the machine.
2. `QUALIFICATION_EXPECTED_PASSES` is unset for the exp105 and exp106 groups and
   `require_expected_pass_counts()` refuses to qualify until Validation 002
   measures them.
3. The locating pilot draws from its own ensemble namespace and its own registry,
   so no code that helps choose the frozen grid is later measured on it.
4. The compiled decoder is not bit-portable across platforms (exp104 Validation
   002). Generation, replay and aggregation all happen on nd-3 against the pinned
   nd-3 binary; artifacts are never mixed across platforms.
5. No nd-3 time is authorized beyond the outcome-blind cost benchmark until the
   Validation 004 resource gate passes.

## What differs from exp105, and why

- **`q = 0.01`**, a fresh master seed and fresh namespaces, so no exp104 or exp105
  code enters an exp106 panel by construction.
- **A different pilot grid and a different fallback grid.** exp105's were
  log-spaced for a low-`p` regime; exp106's are dense across `p in [0.02, 0.06]`,
  the window exp104 measured negative at `q = 0`.
- **Costs are measured on nd-3 before the allocation rule is evaluated**, through
  a new plan-independent `remote_cli cost-benchmark` and a `pilot_remote` config
  phase. This breaks the circularity that made exp105 evaluate its rule on macmini
  numbers and blocked its first resource gate at 5,367.8 core-hours against a cap
  of 800.
- **The allocation rule is preregistered in its `s`-form**,
  `s = sqrt(sigma_c² + sigma_w²/T)`, which exp105 had to substitute mid-flight
  when `sigma_c` collapsed below pilot resolution.
- **A second equality gate at `q = 0.05`**, requiring exp106 to reproduce exp105
  bit for bit on exp105's own registry. The inherited exp104 gate runs at `q = 0`
  and cannot reach the augmented matrix, the mixed channel, the readout draw or
  the `q > 0` criterion; this one does.
- **`report.py` ships corrected.** exp105's in-package copy carries three defects
  it could not fix without orphaning its published aggregate, one of which killed
  its remote aggregate stage after the NPZ had been written.
- **72 workers**, and caps of 1800 reserved core-hours / 20 wall hours from a
  frozen `G = 800` core-hour generation budget.
- **No Track B.** exp105 established that full-sector TI cannot certify a `q_top`
  anchor at `q > 0`; permanent discipline 13 forbids extending that attempt. The
  certified bound `E[q_top] >= (1 - P_fail)²` is still reported, and stays
  uncalibrated.

## Authority and limits

exp106 asserts no asymptotic threshold, no critical exponent, no
finite-size-scaling collapse, no `q_top` **estimate** at `m >= 4`, no MLD
statement and no preparation-channel claim. It locates no threshold *curve*: two
points and a bracket are not a curve. **Complete success clears no exp102
blocker**; exp102 remains `BLOCKED_BEFORE_REMOTE` with all four blockers open.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract, its freezing rules and
  its primary-only terminal rule.
- `config/noisy_mc.pilot.v1.json`, `config/noisy_mc.pilot.remote.v1.json`,
  `config/ensemble_registry.pilot.v1.npz`.
- `validation/INDEX.md`: numbered evidence ledger.
- `raw/README.md`, `final_results/README.md`: where evidence lives and why.

## Latest evidence

- Package ported and green locally: 153 passed, 29 skipped (all pre-freeze
  markers). The exp105 `q = 0.05` equality gate passes bit for bit at
  `m = 3, 4, 5, 8` across `p = 0.001, 0.01, 0.04`.
- Pilot registry built: 200 codes each at `m = 3, 8` plus two each at
  `m = 4..7` for the cost benchmark. Acceptance rate `0.738` at `m = 3`, against
  exp104's independently measured `0.723`.
