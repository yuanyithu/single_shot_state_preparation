# Validation 006: loader-verified publication

Status: **`EXP106_NO_CERTIFIED_CROSSING`** — complete, published, closed.

## What was verified

`loader_verification.json`: the published aggregate re-derived independently on
macmini through `exp106_pipeline.loader.load_exp106_crossing`, which recomputes
from the stored per-code counts rather than trusting the summary fields --
per-code rates, Wilson intervals, pooled means, cluster standard errors, the
distance strata table, the 20,000-replicate cluster bootstrap and its simultaneous
band, the terminal decision, the crossing location and the `q_top` bound. An
aggregate the loader will not accept is not a result.

- aggregate SHA256 `389801a1...4ec8d416`
- `overall_status` `COMPLETE`, `replay_status` `PASS`
- **1,084,000 of 1,084,000** code-`p` cells `REPORTABLE`
- simultaneous band half-width **`0.007892`**, against `0.0078` predicted by the
  section 6 allocation rule before the run
- **10 of 10** grid points certified positive, **0** certified negative

## The result

At `q = 0.01`, over `p` from `0.005` to `0.07`, the ensemble-mean block logical
failure rate of the frozen BP+OSD-0 decoder is **higher for the larger code at
every grid point**, with the simultaneous 95 percent band excluding zero from
below everywhere. **There is no crossing, and that absence is certified rather
than unresolved.**

| p | `Delta38` | simultaneous band |
|---|---:|---|
| 0.005 | +0.04151 | `[+0.03361, +0.04940]` |
| 0.010 | +0.08224 | `[+0.07435, +0.09013]` |
| 0.015 | +0.09823 | `[+0.09034, +0.10613]` |
| 0.020 | +0.11916 | `[+0.11127, +0.12705]` |
| 0.025 | +0.12951 | `[+0.12162, +0.13740]` |
| 0.030 | +0.13940 | `[+0.13151, +0.14730]` |
| 0.035 | +0.15845 | `[+0.15056, +0.16635]` |
| 0.045 | +0.20359 | `[+0.19570, +0.21149]` |
| 0.055 | +0.24447 | `[+0.23657, +0.25236]` |
| 0.070 | +0.24177 | `[+0.23388, +0.24967]` |

The per-`m` curves are ordered by size at every point:

| p | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 |
|---|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.02629 | 0.03163 | 0.03973 | 0.04897 | 0.05396 | 0.06780 |
| 0.020 | 0.14164 | 0.16570 | 0.19629 | 0.22173 | 0.23215 | 0.26079 |
| 0.055 | 0.52867 | 0.61835 | 0.67485 | 0.71821 | 0.74311 | 0.77314 |
| 0.070 | 0.70644 | 0.81290 | 0.87036 | 0.90828 | 0.93227 | 0.94822 |

**The small-`p` end is the part that mattered and it is settled.** A dip that had
*moved* left rather than vanished would have hidden there, and the pilot could not
have seen it: its pointwise SD was about `0.024` against a smallest point of
`+0.035`. The production band at `p = 0.005` is `[+0.0336, +0.0494]`, so the
larger code is worse there by at least `0.034` with 95 percent simultaneous
confidence. There is no room left for a residual crossing anywhere in the window.

## What this closes

exp104 certified a crossing at `p_cross = 0.05512` when `q = 0`. exp105 certified
no crossing at any `p` when `q = 0.05`, which put the readout threshold of this
decoder on this family strictly inside `(0, 0.05)`. exp106 now certifies no
crossing at `q = 0.01`, so

```text
q_c in (0, 0.01).
```

The interval has narrowed by a factor of five. It is still open, and locating it
would be a further experiment with its own contract; exp106 may not become one.

**What the mechanism looks like.** At `q = 0` the advantage of the larger code was
worth at most `0.053` in `Delta38`. At `q = 0.01` the readout channel already
costs the larger code `+0.118` to `+0.300` at the same grid points -- two to six
times what erasing that advantage requires. The reason is structural: a code at
`m = 8` carries `n_c = 12m² = 768` checks against `m = 3`'s 108, so seven times
as many independent opportunities to misread, and one round of readout is not
protected by any meta-check in `H_aug = [H_Z | I]`.

## The certified bound on the requested observable

The original request was `q_top`. Section 7 gives a certified one-sided bound from
what was measured: per disorder `map_success <= sqrt(purity)`, no decoder beats
MAP success at its own observation, and Jensen gives
`E[q_top] >= (M S² - 1)/(M - 1)` with `S = 1 - P_fail` and `M = 2^k`.

| p | `E[q_top] >=` at m=3 | at m=8 |
|---|---:|---:|
| 0.005 | **0.94800** | 0.86900 |
| 0.010 | 0.88787 | 0.73985 |
| 0.015 | 0.81356 | 0.64633 |
| 0.020 | 0.73627 | 0.54642 |
| 0.025 | 0.65538 | 0.46302 |
| 0.030 | 0.57504 | 0.38373 |
| 0.035 | 0.49921 | 0.30117 |
| 0.045 | 0.34786 | 0.14998 |
| 0.055 | 0.22063 | 0.05147 |
| 0.070 | 0.08439 | 0.00268 |

This is a bound, never an estimate, informative only where the success rate is
large -- which is the ordered side, exactly where the sampling route to `q_top` is
blocked. It remains **uncalibrated**: exp105 Validation 007 established that
full-sector TI cannot certify an anchor at `q > 0`, so we know the bound holds and
do not know how tight it is.

## A note on the variance regime

The allocation rule put trials at the **floor** of three and spent the budget on
codes, the opposite of exp105's cap of six. The published curves show why that was
right: `between_code_std` runs from `0.094` to `0.295`, and the cluster standard
error exceeds the pooled binomial one at every point (for instance `0.002851`
against `0.002647` at `m = 8, p = 0.025`). Code diversity is a real component of
the variance at `q = 0.01`, as it was at `q = 0` and was not at `q = 0.05`. Had
the rule been left in exp105's raw-`sigma_c` form it would have had to be patched
after seeing the pilot; the preregistered `s`-form handled both regimes without a
decision being made.

## Authority and limits

exp106 reports a finite-grid, decoder-dependent result for one frozen BP+OSD-0
decoder on one randomly generated expander-code ensemble at `q = 0.01`, plus a
certified one-sided lower bound on the disorder-averaged `q_top` of the exp101
posterior.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no `q_top` **estimate** at `m >= 4`, no maximum-likelihood-decoding
statement and no preparation-channel claim. It locates no threshold *curve*: three
points and a bracket are not a curve.

**Complete success clears no exp102 blocker.** exp102 remains
`BLOCKED_BEFORE_REMOTE` with all four blockers open, as exp106's contract said
from the start that it would.

## Evidence in this directory

- `loader_verification.json` — the independent macmini re-derivation

## Reproduction

```bash
conda run -n 12 --no-capture-output python -c "
from data.expander_code.exp106.exp106_pipeline.config import load_config
from data.expander_code.exp106.exp106_pipeline.loader import load_exp106_crossing
config = load_config('data/expander_code/exp106/config/noisy_mc.remote.v1.json')
payload = load_exp106_crossing('data/expander_code/exp106/final_results/ensemble_crossing.npz', config)
print(payload['terminal_status'], payload['bootstrap_half_width'])
"
```
