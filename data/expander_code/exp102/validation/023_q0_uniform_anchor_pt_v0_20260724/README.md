# Uniform-anchored full-energy replica exchange V0

This directory freezes a local-only adversarial-initialization screen for a
new q=0 global transition. It is not a posterior result, a formal pilot,
`READY_FOR_FORMAL`, held-out evidence, or a production authorization.

## Why this is a new mechanism

Writing the exact collapsed marginal as

```text
S(B) = log(p/(1-p)) |B| + sum_j log M(Y_j xor (B H)_j),
```

the rung target is `pi_lambda(B) proportional to exp(lambda S(B))`. At
`lambda=0`, every B bit is exactly uniform and is globally refreshed. Every
positive rung performs one exact full-row variable-elimination heatbath; every
adjacent exchange uses the complete-energy ratio

```text
(lambda_i-lambda_j) [S(B_j)-S(B_i)].
```

The cold `lambda=1` B state is paired with a fresh exact conditional A draw,
so each fixed-clock observation has the required hard-coset posterior as its
stationary marginal.

This is not the old PT/HP/CTT route under a new name:

- Historical q=0 PT had no collapsed-B exact row conditional nor an exact
  all-B uniform refresh.
- HP/CTT temper only the integrated A likelihood and retain
  `B~Bernoulli(.04)` at their hot endpoint.
- FRG has the row conditional but no global-temperature endpoint or exchange.

Small HGP enumeration verifies the endpoint target, complete-energy swap
detailed balance, stationarity of the two-rung composite transition, and
reference/Numba bit identity. The implementation must not be used for a raw
screen until those tests pass.

## Frozen Screen

- Cell: `m08_c06,p=.04,d00,attempt022` with its nonzero hard syndrome.
- Starts: planted legal `P`, exact-K=0 hard-coset-uniform `U`, and a fixed
  legal low-energy logical-tail `L`. Physical all-zero is illegal here;
  shifted-coordinate zero is already `P`.
- Candidates: `UARE32-R1` and `UARE64-R1`, each using the same cosine ladder,
  one state-independent full-row update per positive rung per clock, and an
  exact uniform hot refresh per clock.
- Resources: eight independent trajectories for every `(method, P/U/L)`;
  fixed `burn=256` and `measurement=2048` clocks. There is no resampling,
  clone, adaptive ladder, result-dependent extension, or `q_top` selection.
- Runtime: the pre-raw profile is in `runtime_probe.json`. Its slowest
  candidate takes `.200786808389239` seconds per round, so the frozen V0
  trajectory has a factor-two projection of about `925.23` seconds, within the
  1200-second hard cap. This is the sole basis for choosing the V0 clock.

## What The Gate Asks

The gate is designed to test the deliverable--a cold posterior distribution--
rather than demand rank-64 label movement when a low-temperature posterior may
legitimately concentrate.

For all three initialization families and every pair of families, the analyzer
uses independent-trajectory means of fixed-clock data to compare normalized
physical weight, normalized B weight, collapsed score per A factor, 64 logical
basis characters, 64 frozen nonbasis characters, and 64 frozen dense B masks.
Each comparison must satisfy both its absolute bound and `3 SE + .01`. At
least four independent complete trajectories per family are required. B-mask
variation is retained as a diagnostic rather than a mandatory movement gate:
a genuinely concentrated posterior may make a mask constant, whereas unequal
P/U/L distributions still fail the direct cross-family comparison. The
analyzer never calculates or reads `q_top` to choose a method.

It additionally applies the target-support red team. If a U trajectory stays
above a measurement weight `w` for which, relative to a known legal weight
`w0`,

```text
Pr_pi(|e| >= w) <= 2^dim(hard coset) (p/(1-p))^(w-w0) <= .001,
```

then it is trapped in demonstrably negligible target support and the method
fails even if it makes many local B or label changes. This avoids treating the
FRG-style U high-energy wandering as convergence. Conversely, constant
characters alone are not failure: a concentrated posterior is allowed if the
opposing starts reach the same fixed-clock distribution.

The selected method is the only passing candidate; if both pass the frozen
tie-break is smaller `UARE32-R1`. If neither passes, status is
`LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`. Either result remains local
diagnostic evidence only. A later independent mechanism, fresh HARD2,
confirmation, and formal tuning/held-out campaign would still be required.

## Result

The frozen local screen completed with
`LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`. The audited failure, the
post-replay reporting defect, and the separate bit-exact V2 replay evidence
are recorded in [RESULT.md](RESULT.md). This remains a rejection of two fixed
local configurations, not a formal q=0 result or a claim of impossibility.
