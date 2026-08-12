# Validation 003: locating pilot, nd-3 cost benchmark, and the frozen plan

Status: **`PILOT_COMPLETE_NO_SIGN_CHANGE`**. Controlled category: `PLANNING_GATE`.

The plan is now frozen and applied. This directory records the measurements that
determined it and nothing else: no physical result, no parameter-point
certification, and no authorization beyond Validation 004's gates.

## The pilot

44/44 tasks `VALID`, `m = 3, 8`, 200 codes each, 14 grid points, 4 trials per
(code, `p`) -- 22,400 trials, run locally on 8 workers under thread pinning. Own
ensemble namespace, own registry, never merged into production.

`Delta38` is **positive at all fourteen points**:

| p | `P_fail(3)` | `P_fail(8)` | `Delta38` |
|---|---:|---:|---:|
| 0.005 | 0.03375 | 0.06875 | +0.03500 |
| 0.0075 | 0.04625 | 0.10875 | +0.06250 |
| 0.01 | 0.07875 | 0.14250 | +0.06375 |
| 0.015 | 0.08625 | 0.20000 | +0.11375 |
| 0.02 | 0.14875 | 0.25750 | +0.10875 |
| 0.025 | 0.18875 | 0.32500 | +0.13625 |
| 0.03 | 0.26500 | 0.38125 | +0.11625 |
| 0.035 | 0.29375 | 0.45500 | +0.16125 |
| 0.04 | 0.38750 | 0.55125 | +0.16375 |
| 0.045 | 0.42125 | 0.60375 | +0.18250 |
| 0.05 | 0.45500 | 0.71375 | +0.25875 |
| 0.055 | 0.55500 | 0.76875 | +0.21375 |
| 0.06 | 0.58375 | 0.84625 | +0.26250 |
| 0.07 | 0.73625 | 0.94625 | +0.21000 |

So the grid rule returns its fallback branch, `grid_rule_reason` =
`no_sign_change_fallback_grid`, and the production grid is the frozen fallback
`{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.055, 0.07}`.

**This is a pilot, and its precision says so.** With 800 trials per `(m, p)` the
pointwise standard deviation of `Delta38` is about `0.024`, so the smallest
point, `+0.035` at `p = 0.005`, is barely 1.5 standard deviations from zero on
its own. Fourteen consecutive positive points is a different matter, but the
pilot is not the measurement and nothing here is certified. The production run
exists precisely to settle the small-`p` end, where a dip that had *moved* rather
than vanished would hide.

## What it says about the physics, against the two predecessors

The three experiments now share grid points, and the comparison is the reason
exp106 exists. `Delta38` at each `q`:

| p | q = 0 (exp104) | q = 0.01 (exp106 pilot) | q = 0.05 (exp105 pilot) |
|---|---:|---:|---:|
| 0.02 | −0.00962 | **+0.10875** | +0.47000 |
| 0.03 | −0.02562 | **+0.11625** | +0.54125 |
| 0.04 | −0.05312 | **+0.16375** | +0.44750 |
| 0.05 | −0.04075 | **+0.25875** | +0.32875 |
| 0.06 | +0.03887 | **+0.26250** | +0.21125 |
| 0.07 | +0.14225 | **+0.21000** | +0.13625 |

The `q = 0` dip is `0.053` deep at its worst. At `q = 0.01` the readout channel
already contributes between `+0.118` and `+0.300` at those same points -- two to
six times what it would need to erase the dip.

**How the penalty scales with `q`.** The contract named two candidate scalings in
advance and said the answer sat between them. On the two points where the
`q = 0.05` curve has not saturated (`P_fail(m=8)` of 0.72 and 0.93 at
`p = 0.02, 0.03`), the readout penalty relative to `q = 0` is:

| p | shift at q=0.01 | shift at q=0.05 | ratio | implied `q^alpha` |
|---|---:|---:|---:|---:|
| 0.02 | +0.11838 | +0.47962 | 0.247 | `alpha = 0.869` |
| 0.03 | +0.14187 | +0.56688 | 0.250 | `alpha = 0.861` |

A fivefold reduction in `q` buys a fourfold reduction in the penalty:
**close to linear**, `alpha ≈ 0.865`. That is the branch under which the
expectation of no crossing holds, and it is what the pilot sees.

Above `p = 0.03` the ratio is meaningless because `q = 0.05` has saturated
(`P_fail(m=8)` reaches 0.966, 0.994, 0.999, 1.000 at `p = 0.04..0.07`), so both
contrasts are being squeezed toward zero from above rather than reflecting the
channel. Those points are excluded here rather than averaged in. This is a
two-point estimate from pilot-grade data and is a diagnostic, not a claim.

## The nd-3 cost benchmark

`cost_benchmark.json`, `report_sha256` `18304495...19ec56b`, measured **on nd-3**
through the verified-source chain under the pilot remote config: 36 outcome-blind
tasks, six sizes by two codes by the first, middle and last pilot grid points
(`0.005, 0.035, 0.07`), six trials per timed batch, one worker.

| m | `kappa_m` (s/code) | `c_m` (s/trial) | exp105's nd-3 `c_m` at q=0.05 |
|---|---:|---:|---:|
| 3 | 0.1665 | 0.0825 | 0.0968 |
| 4 | 0.5292 | 0.4226 | 0.3075 |
| 5 | 1.6585 | 1.0564 | 0.7496 |
| 6 | 3.3069 | 2.2251 | 1.5488 |
| 7 | **17.6447** | 4.2668 | 2.8761 |
| 8 | 16.9278 | 5.8403 | 4.8821 |

Two things to record honestly.

**Trials cost more at `q = 0.01` than at `q = 0.05`**, by about 20 percent at
`m = 8`. That is the right direction: with a weaker readout channel the decoder
can no longer explain a heavy syndrome as misreads, so belief propagation grinds
against `max_iter` more often and OSD does more work.

**The `m = 7` per-code cost is anomalous.** `kappa_7 = 17.64 s` exceeds
`kappa_8 = 16.93 s`, which is impossible for real work -- building an `m = 8`
frame is strictly larger. It is an upper bound over six samples, and nd-3 is
shared with another user holding ten cores continuously, so one contended sample
inflated it. Its effect is bounded and conservative: `kappa` is 6 percent of
`u_7`, so `m = 7` receives about 9 percent fewer codes than an uncontended
measurement would have given it. `m = 7` is a diagnostic size and is not in the
primary contrast. **The measurement is left as measured.** Substituting a
"corrected" cost after seeing it would be exactly the post-hoc adjustment the
contract's freezing rules exist to prevent.

## The frozen plan

`allocation_plan.json`, `report_sha256` `cd945203...74d2e3bc5`, status
`EVALUATED_NOT_APPLIED` -- evaluating the rule is not applying it; writing the
constants into `config.py` is a separate, deliberate act.

| m | codes | trials per (code,p) | codes per task | `u_m` (s) | `s_eff` |
|---|---:|---:|---:|---:|---:|
| 3 | 76,162 | 3 | 113 | 2.64 | 0.25536 |
| 4 | 13,068 | 3 | 22 | 13.21 | 0.29560 |
| 5 | 5,176 | 3 | 8 | 33.35 | 0.29560 |
| 6 | 2,464 | 3 | 4 | 70.06 | 0.29560 |
| 7 | 1,186 | 3 | 2 | 145.65 | 0.29560 |
| 8 | 10,344 | 3 | 1 | 192.14 | 0.29560 |

**108,400 codes, 3,252,000 trials, 13,468 tasks, 799.8 of 800 budgeted
generation core-hours.** Predicted pointwise `SD(Delta38) = 0.003050`, so a
simultaneous half-width near `0.0078`.

**The rule landed in the opposite regime from exp105, and that is the whole point
of preregistering the `s`-form.** exp105's pilot measured `sigma_c` at or below
its own resolution, because at `q = 0.05` failure is driven by a readout channel
common to every code; its rule pushed trials to the cap of six and bought few
codes. At `q = 0.01` the between-code spread is recoverable -- `sigma_c ≈ 0.12`
at both primary sizes, from `s_eff` and the anchor's within-code variance -- so
the same rule pushes trials to the **floor of three** and spends the budget on
codes instead. That is exp104's regime returning. Had the rule been left in
exp105's raw-`sigma_c` form it would have had to be patched here in flight, after
seeing the pilot, which is precisely the decision the contract removed.

The anchor point is `p = 0.025`, the pilot point nearest the geometric centre of
the fallback grid, chosen by the frozen rule rather than by inspection.

## What was frozen, and the registry

`config.py` now carries `PRODUCTION_PLAN_FROZEN = True` with the four constants
above. The production registry `config/ensemble_registry.v1.npz` was built to
match: 108,400 codes, `registry_sha256`
`16da2268d8d9a69065bc23c6c302491682d2509a427873cafc28b0977c6dcfb2`. Acceptance
rates on the production draw run from 0.718 at `m = 3` to 0.990 at `m = 8`,
matching the Validation 001 census.

## Process notes worth keeping

**The first two pilot runs were discarded, not resumed.** The first was bound to
a config that was superseded before it finished. The second was killed to replace
it -- but `pkill` on the parent left the `ProcessPoolExecutor` workers running,
because spawned workers do not carry the parent's command line, and they kept
writing into the recreated raw directory. The replacement run then reported
`EXISTS` for those tasks and skipped them, which would have silently mixed raw
from two configs into one pilot. Caught by noticing that the log had 20 lines
against 38 files. The third run is the one recorded here: 44 tasks, 44 `VALID`,
zero `EXISTS`.

**The nd-3 cost benchmark failed on its first attempt** with "remote deployment
does not contain its canonical config": `verify_remote_deployment` looked up
`noisy_mc.remote.v1.json` by name, which silently assumed the production remote
config. exp106 has two remote schemas on purpose, since the cost benchmark must
run before the production plan exists. Fixed at all three sites that shared the
assumption, with tests; see Validation 002.

## Evidence in this directory

- `pilot_plan.json` -- pooled rates, `sigma_c`, `sigma_c_raw`, `sigma_w2`,
  `Delta38`, the grid rule's output and its reason
- `cost_benchmark.json` -- the nd-3 measurement, outcome-blind, 36 tasks
- `allocation_plan.json` -- the rule evaluated on the two above

## Reproduction

```bash
conda run -n 12 --no-capture-output python -m \
  data.expander_code.exp106.exp106_pipeline.pilot measure \
  --config data/expander_code/exp106/config/noisy_mc.pilot.v1.json \
  --raw-root data/expander_code/exp106/raw/pilot_v1 --output <pilot_plan.json>

conda run -n 12 --no-capture-output python -m \
  data.expander_code.exp106.exp106_pipeline.pilot allocate \
  --pilot-plan <pilot_plan.json> --cost-benchmark <cost_benchmark.json> \
  --output <allocation_plan.json>
```
