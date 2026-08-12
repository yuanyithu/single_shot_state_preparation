# exp106 status

## Current state

**`AMENDED_AWAITING_ND3_RESOURCE_GATE`** — the pilot is complete, the plan is
frozen, and the first resource gate blocked. Two user-authorized amendments have
been applied and the gate is being re-run. No production compute has run.

exp106 (`exp106.noisy_syndrome_mc.q001.v1`) measures the ensemble block logical
failure rate of the frozen exp103/exp104 BP+OSD-0 decoder at readout error rate
`q = 0.01`, over randomly generated expander codes at `m = 3..8`, and asks
whether it still crosses as the code grows.

## Why exp106 exists

exp104 and exp105 bracketed the question without answering it: a certified
crossing at `p_cross = 0.05512` when `q = 0`, and a certified *absence* of one
anywhere in `p in [0.001, 0.07]` when `q = 0.05`. The readout threshold of this
decoder on this family therefore lies strictly inside `(0, 0.05)`. exp106
measures one interior point.

## What the pilot found

`Delta38` is **positive at all fourteen pilot points**, from `+0.035` at
`p = 0.005` to `+0.263` at `p = 0.06`. There is no negative-to-positive sign
change to bracket, so the grid rule returns its fallback branch.

The `q = 0` dip that `q = 0.01` would have to erase is `0.053` deep at worst. At
the shared grid points the readout channel already contributes `+0.118` to
`+0.300` — two to six times what erasing it requires. On the two points where the
`q = 0.05` curve has not saturated, a fivefold reduction in `q` buys a fourfold
reduction in the penalty: close to linear, `alpha ≈ 0.865`, which is the branch
under which no crossing survives.

**None of this is certified.** The pilot's pointwise `SD(Delta38)` is about
`0.024`, so its smallest point is 1.5 standard deviations from zero. The
production run exists to settle the small-`p` end, where a dip that had *moved*
rather than vanished would hide.

## The frozen plan

| m | codes | trials per (code,p) | codes per task |
|---|---:|---:|---:|
| 3 | 76,162 | 3 | 113 |
| 4 | 13,068 | 3 | 22 |
| 5 | 5,176 | 3 | 8 |
| 6 | 2,464 | 3 | 4 |
| 7 | 1,186 | 3 | 2 |
| 8 | 10,344 | 3 | 1 |

Grid `{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.055, 0.07}`.
**108,400 codes, 3,252,000 trials, 13,468 tasks, 799.8 of 800 budgeted
generation core-hours**, predicted pointwise `SD(Delta38) = 0.00305` and a
simultaneous half-width near `0.0078`.

The rule landed in the **opposite regime from exp105**: trials at the floor of
three rather than the cap of six, because at `q = 0.01` the between-code spread
is recoverable (`sigma_c ≈ 0.12` at both primary sizes) and the budget is better
spent on codes. That is exp104's regime returning, and it is why the `s`-form of
the allocation rule had to be preregistered rather than chosen after the fact.

## Current gates

1. The production plan is **frozen**. All 29 previously skipped tests now run;
   nd-3 qualification requires zero skips, which is what enforced the ordering.
2. `QUALIFICATION_EXPECTED_PASSES["exp106"]` is set from the post-freeze count and
   `require_expected_pass_counts()` refuses to qualify until every group's count
   is measured.
3. The compiled decoder is not bit-portable across platforms (exp104 Validation
   002). Generation, replay and aggregation all happen on **one** host against the
   pinned binary; artifacts are never mixed across platforms. The nd-3 cost
   benchmark is the one nd-3 artifact in the chain, and it is a *cost*
   measurement, not an outcome -- it is superseded by the nd-2 re-measurement
   rather than mixed with it.
4. **The nd-3 resource gate blocked** at `2001.95` reserved core-hours against a
   cap of `1800` (wall and RSS both passed). The cause was measurement scatter on
   a contended host: the same six per-`m` costs re-measured minutes apart moved by
   factors of `0.69` to `1.58`, in both directions. Contract section 6 stopped the
   run rather than shrinking the panel, and a failed gate does not authorize its
   own relaxation.
5. **Amendment 1, user-authorized: 72 -> 75 workers.** The move to nd-2 was also
   authorized, attempted, and is **impossible**: nd-1 and nd-2 run CentOS 7 with
   glibc 2.17, nd-3 runs Ubuntu 24.04 with glibc 2.39, and the frozen decoder
   extension requires `GLIBC_2.29` — it raises `ImportError` on nd-2. Rebuilding
   it there would change the binary hash, and a byte-identical decoder is the
   only reason exp106 is comparable to exp104 and exp105; exp104 Validation 002
   measured that this decoder is not bit-portable across builds. **nd-3 is the
   only host this experiment can run on.** The 75 workers carry over and shorten
   wall time, but core-hours are work rather than parallelism, so they do not
   clear the gate.
6. **Amendment 2, user-authorized: reserved core-hour cap 1800 -> 2200.** The
   original 1800 was `2 x (800+80+1+1) = 1764` rounded up — two percent of margin
   against a rule that spends the entire budget by construction, so the
   projection began at the ceiling. 2200 sits about nine percent above the
   observed `2001.95`. Wall (`20`) and RSS (`128 GiB`) are unchanged and were
   never binding. The generation budget stays at 800, so **the panel does not
   move** and neither does the precision.
7. No production compute is authorized until a Validation 004 resource gate
   passes.

## What differs from exp105, and why

- **`q = 0.01`**, a fresh master seed and fresh namespaces, so no exp104 or
  exp105 code enters an exp106 panel by construction.
- **A different pilot grid and fallback grid**, dense across `p in [0.02, 0.06]`,
  the window exp104 measured negative at `q = 0`.
- **Costs measured on the compute host before the allocation rule is evaluated**,
  through a new plan-independent `remote_cli cost-benchmark` and a `pilot_remote`
  config phase. This breaks the circularity that made exp105 evaluate its rule on
  macmini numbers and blocked its first resource gate at 5,367.8 core-hours. The
  compute host is now a single constant, `config.COMPUTE_HOST`, read by the
  execution profile, the expected remote environment and the allocation rule's
  refusal — it was spelled inline at six sites before the run moved.
- **The allocation rule preregistered in its `s`-form**, which exp105 had to
  substitute mid-flight.
- **A second equality gate at `q = 0.05`**, requiring exp106 to reproduce exp105
  bit for bit on exp105's own registry.
- **`report.py` ships corrected and tested.** exp105's copy had an undefined name
  that killed its remote aggregate stage after the NPZ had been written.
- **nd-3 at 75 workers**, caps of 2200 reserved core-hours / 20 wall hours.
- **No Track B.** exp105 established that full-sector TI cannot certify a `q_top`
  anchor at `q > 0`; permanent discipline 13 forbids extending that attempt. The
  certified bound `E[q_top] >= (1 - P_fail)²` is still reported, uncalibrated.

## Authority and limits

exp106 asserts no asymptotic threshold, no critical exponent, no
finite-size-scaling collapse, no `q_top` **estimate** at `m >= 4`, no MLD
statement and no preparation-channel claim. It locates no threshold *curve*: two
points and a bracket are not a curve. **Complete success clears no exp102
blocker**; exp102 remains `BLOCKED_BEFORE_REMOTE` with all four blockers open.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract and its freezing rules.
- `config/`: pilot, pilot-remote, production and production-remote configs, plus
  the pilot and production registries.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 003: `PILOT_COMPLETE_NO_SIGN_CHANGE`. 44/44 pilot tasks `VALID`;
  `Delta38` positive at all 14 points; nd-3 costs measured outcome-blind; plan
  frozen at 108,400 codes and 3,252,000 trials. Two pilot runs were discarded
  before this one — the second because orphaned pool workers kept writing into a
  recreated raw directory after their parent was killed.
- Validation 002: `PASS`. 194 exp106 + 166 exp105 + 131 exp104 + 58 exp101 +
  17 exp102 tests, bytecode clean; both equality gates green.
- Validation 001: `PASS`. Census reproduces exp105's composition independently to
  within 0.0036 in acceptance and 0.0042 in the distance-2 fraction; zero codes
  shared with either predecessor.
