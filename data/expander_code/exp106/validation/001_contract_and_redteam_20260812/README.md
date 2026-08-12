# Validation 001: contract freeze, scientific red team, ensemble census

Status: **`PASS`**. Controlled category: `IMPLEMENTATION_GATE`.

Local only. Authorizes Validation 002. No remote transfer, no physical result, no
exp102 authority.

## What was frozen

`EXPERIMENT_CONTRACT.md`, identity `exp106.noisy_syndrome_mc.q001.v1`. Thirteen
sections, all fixed before any measurement: the estimand, the ensemble rule, the
physics and scoring, the section 6 grid and allocation rules, the estimators and
the `q_top` bound, the fail-closed rules and the three gates, the execution caps,
and the terminal decision.

`config.Q_TOKEN = "0.01"`, `MASTER_SEED_HEX` fresh, production plan **absent**:
`P_TOKENS`, `CODES_PER_M`, `TRIALS_PER_CODE_P` and `CODES_PER_TASK` are `None`
behind `ProductionPlanNotFrozen`, and stay that way until Validation 003 evaluates
the section 6 rules.

## The ensemble census

`ensemble_census.json`, SHA256 `3c380727...0451aa`. 20,000 accepted codes per `m`
for `m = 2..8` under exp106's own master seed and namespace -- an independent
draw, not a re-read of exp105's.

The ensemble rule should be a property of the rule, not of the seed, so the
composition must reproduce. It does, at every size:

| m | acceptance (exp105) | acceptance (exp106) | Δ | distance-2 fraction (exp105) | (exp106) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.37763 | 0.37889 | 0.00126 | 0.15870 | 0.16005 | 0.00135 |
| 3 | 0.72046 | 0.71793 | 0.00253 | 0.22370 | 0.22590 | 0.00220 |
| 4 | 0.86233 | 0.86584 | 0.00351 | 0.19180 | 0.18770 | 0.00410 |
| 5 | 0.93071 | 0.93106 | 0.00035 | 0.15995 | 0.15915 | 0.00080 |
| 6 | 0.96302 | 0.96432 | 0.00130 | 0.13785 | 0.13365 | 0.00420 |
| 7 | 0.97929 | 0.98020 | 0.00091 | 0.11560 | 0.11620 | 0.00060 |
| 8 | 0.98912 | 0.98966 | 0.00054 | 0.10090 | 0.10105 | 0.00015 |

Every acceptance rate agrees to within `0.0036` and every distance-2 fraction to
within `0.0042`, against a binomial standard error of about `0.003` at 20,000
samples. exp105's own census reproduced exp104's to the same tolerance, so this is
now the third independent draw of the same ensemble.

The distance-2 fraction falling monotonically with `m` -- `0.226` at `m = 3` down
to `0.101` at `m = 8` -- is the mechanism exp104 identified behind exp103's
anomaly, and it is why exp106 samples thousands of codes per size rather than
freezing a small panel.

## Disjointness

`disjointness.json`, produced by `measure_disjointness.py`. Status **`DISJOINT`**.

Disjointness from exp104 and exp105 is meant to hold by construction -- the
candidate seed is `sha256(master_seed : namespace : m : candidate_index)` and
exp106's master seed is neither of theirs -- but "by construction" is an argument,
and a validation directory holds measurements. Comparing accepted parity-check
matrices directly: 408 exp106 codes against 12,000 exp104 codes and 17,617 exp105
codes, **zero shared**.

This matters because both equality gates read a predecessor's registry on purpose.
Reading it is required; drawing from it would not be.

## Scientific red team (permanent discipline 12)

**Target distribution and support.** The estimand is a probability over a random
code and a random error, sampled directly by i.i.d. draws. There is no support
question: every `(eps, mu)` in the product Bernoulli measure is reachable and is
drawn with its own weight.

**Coordinates and initial states.** No chain, no initial state. Permanent
discipline 4 does not apply here for the same reason it did not apply to exp105
Track A: exp106 runs no Markov chain, so there is no slow variable, no basin and
no barrier to cross. Discipline 2's demand for adversarial initial states is
likewise vacuous.

**Slow variables and self-loops.** None. The only sequential dependence is the
seeded RNG stream inside one `(code, p)` cell, which is a stream and not a chain.

**Estimator and deliverable.** The deliverable is `Delta38` and its simultaneous
band. The estimator is a pooled failure fraction with equal weights, no trimming
and no reweighting; the band is a cluster bootstrap over codes, which is the right
resampling unit because codes -- not trials -- are the independent draws from the
ensemble.

**Gate false positives and false negatives, and common-mode failure.**
The replay gate compares two independently constructed decoders and two
independently reconstructed logical criteria, so a bug shared by both is the
common-mode risk. That is exactly what the two equality gates address, and they
are independent of the replay gate: the exp104 gate compares against a package
frozen at `q = 0`, the exp105 gate against one frozen at `q = 0.05`. A defect that
survived all three would have to be present, identically, in three separately
frozen packages.

The one gate with a real false-negative risk is the resource projection, which is
an upper bound and was 3.2 times conservative for exp105. That costs headroom, not
correctness.

**exact / independent confirmation.** The exp105 equality gate is exact: the two
packages are required to agree bit for bit, not statistically. The exp104 gate is
an algebraic identity at `q = 0`. The census is an independent reproduction of a
measured composition. There is no place where exp106 relies on agreement in
distribution where agreement in bits was available.

**Authority boundary.** exp106 runs on nd-3 only, generates/replays/aggregates on
one platform, and its production entry points are inert until Validation 003.

**"What would complete success unlock?"** It closes one of the two open sides of
the bracket exp105 left: a certified crossing puts `q_c in (0.01, 0.05)`, a
certified absence puts it in `(0, 0.01)`. It unlocks **no exp102 blocker**, and
the contract says so in section 12. If the honest answer to this question had been
"nothing", the experiment would not be worth nd-3 time; the answer is "one
bracket, halved", which is worth about nine wall hours and no more.

## What could still go wrong, stated in advance

**The band may be too wide to certify either terminal.** The `q = 0` dip is only
`0.053` deep, and any residual at `q = 0.01` is shallower. The budget is sized for
a simultaneous half-width near `0.0072`, but that projection rests on a per-trial
cost ratio to exp105 that has not yet been measured. If the band lands too wide,
that is a failure of the measurement rather than a statement about the physics,
and section 11 requires reporting it as such rather than dressing it up as a
result.

**The grid rule may bracket in the wrong place.** The pilot has 200 codes per `m`
and 4 trials, so its `Delta38` carries a pointwise SD near `0.016`. A true dip of
depth `0.01` would be invisible to it and the rule would fall back. The fallback
grid is therefore chosen to cover the whole window exp104 measured negative,
precisely so that the fallback branch still supports the claim a no-crossing
terminal has to make.

## Evidence in this directory

- `ensemble_census.json` -- 20,000 accepted codes per `m`, `m = 2..8`
- `disjointness.json` -- zero shared codes with exp104 or exp105
- `measure_disjointness.py` -- the script that produced it

## Reproduction

```bash
conda run -n 12 --no-capture-output python -m \
  data.expander_code.exp106.exp106_pipeline.ensemble \
  census <output> --accepted-per-m 20000

conda run -n 12 --no-capture-output python \
  data/expander_code/exp106/validation/001_contract_and_redteam_20260812/measure_disjointness.py
```
