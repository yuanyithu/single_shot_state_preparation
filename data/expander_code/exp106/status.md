# exp106 status

## Current state

**`EXP106_NO_CERTIFIED_CROSSING`** — complete, published, closed.

At `q = 0.01`, over `p` from `0.005` to `0.07`, the ensemble-mean block logical
failure rate of the frozen BP+OSD-0 decoder is higher for the larger code at
**every** grid point. `Delta38` is certified **positive at all 10 points** and
negative at none, with a simultaneous band half-width of `0.007892`. There is no
crossing, and that absence is certified rather than unresolved.

13,468/13,468 tasks fresh on nd-3 in 8.38 wall hours at 75 workers; 108,400
codes; 3,252,000 trials; replay 1,354/1,354 with 331,410 trials bit-exact;
1,084,000/1,084,000 cells `REPORTABLE`; re-derived on macmini through the loader.

Towards the originally requested observable, the certified one-sided bound
`E[q_top] >= (1 - P_fail)^2` reaches `0.94800` at `m = 3, p = 0.005`. It is
**uncalibrated**: exp105 Validation 007 established that full-sector TI cannot
certify an anchor at `q > 0`.

## What it closes

exp104: crossing at `p_cross = 0.05512` when `q = 0`. exp105: no crossing at any
`p` when `q = 0.05`. exp106: no crossing at any `p` when `q = 0.01`. So the
readout threshold of this decoder on this family satisfies

```text
q_c in (0, 0.01)
```

a fivefold narrowing of the interval exp105 opened. Locating it is a further
experiment and needs its own contract; exp106 may not become one.

The mechanism is structural. At `q = 0` the larger code's advantage was worth at
most `0.053` in `Delta38`. At `q = 0.01` the readout channel already costs the
larger code `+0.118` to `+0.300` at the same grid points, because a code at
`m = 8` carries `n_c = 12m^2 = 768` checks against `m = 3`'s 108 -- seven times as
many independent chances to misread -- and one round of readout is protected by no
meta-check in `H_aug = [H_Z | I]`.

## The frozen plan

| m | codes | trials per (code,p) | codes per task |
|---|---:|---:|---:|
| 3 | 76,162 | 3 | 113 |
| 4 | 13,068 | 3 | 22 |
| 5 | 5,176 | 3 | 8 |
| 6 | 2,464 | 3 | 4 |
| 7 | 1,186 | 3 | 2 |
| 8 | 10,344 | 3 | 1 |

Grid `{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.055, 0.07}`,
frozen by the section 6 rules on pilot statistics and nd-3 costs before any
production task ran. Predicted pointwise `SD(Delta38) = 0.00305`; achieved
simultaneous half-width `0.007892` against `0.0078` predicted.

Trials landed at the **floor** of three, the opposite of exp105's cap of six,
because at `q = 0.01` the between-code spread is real (`between_code_std` from
`0.094` to `0.295`, cluster SE above pooled binomial SE at every point). That is
exp104's regime returning, and the preregistered `s`-form of the allocation rule
handled both regimes without a decision being made after the fact.

## Amendments

1. **72 -> 75 workers** (user-authorized). The move to nd-2 was also authorized
   and is impossible: nd-1 and nd-2 run CentOS 7 with glibc 2.17, nd-3 runs
   Ubuntu 24.04 with glibc 2.39, and the frozen decoder extension requires
   `GLIBC_2.29`. Rebuilding it would break the byte-identical decoder identity
   that makes exp106 comparable to exp104 and exp105.
2. **Reserved core-hour cap 1800 -> 2200** (user-authorized), after the first
   resource gate blocked at `2001.95`. The original ceiling was
   `2 x (800+80+1+1) = 1764` rounded up -- two percent of margin against a rule
   that spends the whole budget by construction. The generation budget stayed at
   800, so the panel and the precision did not move. Recorded honestly: the
   passing projection came in at `1626.7`, which would have cleared 1800 as well;
   the raised ceiling is what makes the gate robust rather than a coin flip.

Neither amendment touched the estimand, the ensemble, the grid, the physics, the
estimators, the bands, the fail-closed rules, the replay gate, the two equality
gates or the terminal decision.

## Authority and limits

exp106 asserts no asymptotic threshold, no critical exponent, no
finite-size-scaling collapse, no `q_top` **estimate** at `m >= 4`, no MLD
statement and no preparation-channel claim. It locates no threshold *curve*:
three points and a bracket are not a curve. **Complete success clears no exp102
blocker**; exp102 remains `BLOCKED_BEFORE_REMOTE` with all four blockers open.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract, its freezing rules, its
  primary-only terminal rule and both amendments.
- `config/`: four configs and two registries, all identity-bound.
- `validation/001_...`: contract freeze, red team, independent census, disjointness.
- `validation/002_...`: local implementation gate and the two equality gates.
- `validation/003_...`: locating pilot, nd-3 cost benchmark, the frozen plan.
- `validation/004_...`: nd-3 qualification and the resource gate, including the
  blocked first projection.
- `validation/005_...`: production scan, committed replay, aggregation.
- `validation/006_...`: loader-verified publication and the terminal status.
- `final_results/`: published aggregate, report, curves and plots.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 006: `EXP106_NO_CERTIFIED_CROSSING`, aggregate SHA256 `389801a1...`,
  simultaneous half-width `0.007892`, 10/10 points certified positive, 0
  negative. Certified bound `E[q_top] >= 0.94800` at `m = 3, p = 0.005`.
- Validation 005: scan `PASS` 13,468/13,468 in 8.38 h; replay `PASS` 1,354/1,354
  with 331,410 trials bit-exact; 1,084,000/1,084,000 cells `REPORTABLE`.
- Validation 004: qualification `PASS` at 224/166/131/58/17 on the second
  attempt; resource gate `PASS` at `1626.7/2200` on the second attempt. The
  blocked first projection is retained. Between them the gates caught four real
  defects and no false alarms.
