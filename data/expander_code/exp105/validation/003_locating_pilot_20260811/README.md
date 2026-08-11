# Validation 003: locating pilot, cost benchmark, and the evaluated production plan

Status: **`PILOT_COMPLETE_NO_SIGN_CHANGE`**. The frozen grid rule resolves to its
fallback branch. The production plan is **evaluated and recorded, not applied**:
writing the constants into `config.py` is the freeze, and the freeze happens when
the production run is authorized. No remote transfer, no production compute.

## The pilot

44/44 tasks `VALID`, `m = 3, 8`, 200 codes each, 14 grid points from `p = 0.001`
to `0.07`, four trials per (code, `p`), `q = 0.05`. Independent seed namespace
`exp105.pilot.v1` and an independent ensemble namespace, so no code here is a
production code. Run locally on 8 workers with threads pinned to one.

| p | `P_fail(m=3)` | `P_fail(m=8)` | `Delta38` |
|---|---:|---:|---:|
| 0.001 | 0.0125 | 0.0762 | **+0.0638** |
| 0.002 | 0.0300 | 0.1388 | +0.1088 |
| 0.003 | 0.0338 | 0.1750 | +0.1412 |
| 0.005 | 0.0600 | 0.2587 | +0.1987 |
| 0.0075 | 0.0737 | 0.3725 | +0.2988 |
| 0.01 | 0.1062 | 0.4650 | +0.3588 |
| 0.015 | 0.1737 | 0.6700 | +0.4963 |
| 0.02 | 0.2500 | 0.7200 | +0.4700 |
| 0.025 | 0.3300 | 0.8450 | +0.5150 |
| 0.03 | 0.3887 | 0.9300 | **+0.5413** |
| 0.04 | 0.5188 | 0.9663 | +0.4475 |
| 0.05 | 0.6650 | 0.9938 | +0.3287 |
| 0.06 | 0.7875 | 0.9988 | +0.2113 |
| 0.07 | 0.8638 | 1.0000 | +0.1362 |

**`Delta38` is positive at every point.** The pilot's own pointwise standard
deviation is about `0.016`, so the smallest value, `+0.0638` at `p = 0.001`, is
already four standard deviations from zero, and every other point is ten to
thirty. There is no sign change to bracket, so the grid rule returns its frozen
log-spaced fallback `{0.001, 0.0015, 0.0025, 0.004, 0.006, 0.01, 0.016, 0.025,
0.04, 0.07}`.

## What that means physically

At `q = 0.05` this family is **already above threshold from readout noise alone**,
before any data noise is added. Growing the code adds proportionally more checks
(`n_c = 12m²`), and each check carries an independent 5 percent chance of being
misread, so a larger code has more opportunities for a cluster of readout errors
to imitate a low-weight data error. The larger code is therefore worse at every
`p`, and `Delta38` never turns negative.

The `p -> 0` limit makes this concrete and is worth stating separately, because
it is the part that does not depend on the data channel at all. At `p = 0.001`
an `m = 8` code carries about 1.6 expected data errors against 38.4 expected
readout errors, and it already fails 7.6 percent of the time, against 1.25
percent for `m = 3`.

## Control: is this the decoder or the physics?

A "larger codes are always worse" signature is exactly what a mis-wired
augmented channel would produce, so it was checked directly. With the **data
error set identically to zero** and only readout noise present, the decoder
recovers the readout pattern exactly and returns a zero data estimate in 59 of 60
trials at `m = 3` and 51 of 60 at `m = 8`, with 1 and 2 logical failures
respectively. The decoder is doing the right thing; the size dependence is real.

## Cost benchmark

`cost_benchmark.json` (SHA256 `5be8e5c7...c358194`), outcome blind: it times the
real worker path and records only seconds. Codes 0 and 1 of the production
ensemble namespace, twelve trials at the first, middle and last grid points, the
maximum taken as the upper bound.

| m | `kappa` (s/code) | `c` (s/trial) |
|---|---:|---:|
| 3 | 0.026 | 0.0087 |
| 4 | 0.068 | 0.0281 |
| 5 | 0.186 | 0.0720 |
| 6 | 0.439 | 0.1639 |
| 7 | 0.964 | 0.3255 |
| 8 | 2.164 | 0.6078 |

A trial at `m = 8` costs 70 times a trial at `m = 3`, which is what makes unequal
panels worth the trouble.

## The variance barrel moved

exp104 measured a between-code standard deviation of 0.15 to 0.32 at `q = 0` and
concluded that shot noise never binds, which is why it bought codes rather than
trials. **At `q = 0.05` the opposite holds.** The pilot measures `sigma_c` at or
below its own resolution at almost every grid point: failure here is driven by
the readout channel, which is common to all codes, so codes are far more alike
than they are at `q = 0`. The binding term is shot noise, and the rule responds
by pushing trials per code to its cap of six at every `m`.

That also exposes a gap in the frozen rule, which is written in terms of
`sigma_c` alone and is degenerate when `sigma_c -> 0`. It is resolved
structurally, not from the data: the quantity that actually enters the variance
of a per-code rate is `s = sqrt(sigma_c^2 + sigma_w^2 / T)`, which reduces to
`sigma_c` when `sigma_c` dominates, so the primary split uses `s`. The anchor
grid point is fixed data-independently as the pilot point nearest the geometric
centre of the fallback grid, `p = 0.0075`.

## The evaluated plan

`allocation_plan.json` (SHA256 `bb171821...02eed5`), status
`EVALUATED_NOT_APPLIED`.

| m | codes | trials per code-p | core hours |
|---|---:|---:|---:|
| 3 | 89,225 | 6 | 13.5 |
| 4 | 35,745 | 6 | 17.4 |
| 5 | 13,900 | 6 | 17.4 |
| 6 | 6,096 | 6 | 17.4 |
| 7 | 3,056 | 6 | 17.4 |
| 8 | 19,279 | 6 | 206.9 |

Generation 290.0 core-hours, about 4.98 wall hours at 64 workers including
replay. Predicted pointwise standard deviation of `Delta38`: **0.00147**, against
the pilot's 0.0159.

## The honest question this raises

Running that plan would certify `EXP105_NO_CERTIFIED_CROSSING` at a resolution of
0.0015, for a contrast the 200-code pilot already resolves at four to thirty
standard deviations. That is a legitimate terminal state and the contract names it
one, but it is five hours of nd-3 time spent sharpening a conclusion that is not
in doubt.

The pilot has instead produced a **new bracket that is in doubt**: at `q = 0` the
crossing is at `p = 0.05512` (exp104), and at `q = 0.05` there is no crossing at
any `p`. The readout threshold of this decoder on this family therefore lies
strictly inside `(0, 0.05)`. Locating it is a different experiment and needs its
own contract and authorization; it is not something exp105 may quietly become.

Both options are put to the user before any nd-3 time is spent.

## Evidence in this directory

- `measure_costs.py`, `cost_benchmark.json`
- `pilot_plan.json` (SHA256 `c5c42f7e...704e79`)
- `allocation_plan.json`
- pilot raw: `data/expander_code/exp105/raw/pilot_v1/` (44 tasks, not tracked in Git)

## Authority end

The pilot is not evidence about physics that may be published. Its raw is never
merged into production and never enters a published statistic. What it authorizes
is the freezing of the production plan, and nothing else.
