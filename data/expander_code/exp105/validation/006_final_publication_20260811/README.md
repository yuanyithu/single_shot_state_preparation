# Validation 006: loader-verified publication

Status: **`EXP105_NO_CERTIFIED_CROSSING`** — complete, published, closed.

## What was verified

`loader_verification.json`: the published aggregate re-derived independently on
macmini through `exp105_pipeline.loader.load_exp105_crossing`, which recomputes
from the stored per-code counts rather than trusting the summary fields: per-code
rates, Wilson intervals, pooled means, cluster standard errors, the distance
strata table, the cluster bootstrap and its simultaneous band, the terminal
decision, the crossing location, and the `q_top` lower bound. An aggregate the
loader will not accept is not a result.

- aggregate SHA256 `ff73fd9c...`
- `overall_status` `COMPLETE`, `replay_status` `PASS`
- **176,170 of 176,170** code-`p` cells `REPORTABLE`
- simultaneous band half-width `0.010486`
- **10 of 10** grid points certified positive, **0** certified negative

## The result

At `q = 0.05`, over `p` from `0.001` to `0.07`, the ensemble-mean block logical
failure rate of the frozen BP+OSD-0 decoder is **higher for the larger code at
every grid point**, with the simultaneous 95 percent band excluding zero from
below everywhere. There is no crossing, and that absence is certified rather than
unresolved.

`Delta38` runs from `+0.0668 [+0.0563, +0.0773]` at `p = 0.001` to a maximum of
`+0.5193 [+0.5088, +0.5298]` at `p = 0.025`, falling back to `+0.1384` at
`p = 0.07` only because both sizes saturate.

## The certified bound on the requested observable

The request was `q_top`. Section 7 of the contract gives a certified one-sided
bound from what was measured: per disorder `map_success <= sqrt(purity)`, no
decoder beats MAP success at its own observation, and Jensen gives
`E[q_top] >= (M S^2 - 1)/(M - 1)` with `S = 1 - P_fail` and `M = 2^k`.

| p | `E[q_top] >=` at m=3 | at m=8 |
|---|---:|---:|
| 0.0010 | **0.97190** | 0.84471 |
| 0.0015 | 0.96198 | 0.79797 |
| 0.0025 | 0.93850 | 0.71409 |
| 0.0040 | 0.90803 | 0.59716 |
| 0.0060 | 0.86788 | 0.47546 |
| 0.0100 | 0.77873 | 0.27753 |
| 0.0160 | 0.64003 | 0.11760 |
| 0.0250 | 0.45746 | 0.02491 |
| 0.0400 | 0.21121 | 0.00056 |
| 0.0700 | 0.01722 | 0.00000 |

This is a bound, never an estimate, and it is informative only where the success
rate is large — which is the ordered side, exactly where the sampling route to
`q_top` is blocked. It says nothing about how tight it is; the `m = 2, 3`
transport-free anchor is what would calibrate that, and it is not yet run.

## A note on the source tree

`exp105_pipeline/report.py` was corrected **after** the measurement: it carried
three defects from the exp104 port that only surface at report time — an
undefined name, a stale field name, and two-decimal `p` formatting that collapses
`0.001`, `0.0015` and `0.0025` to the same string on this grid.

The frozen configs are deliberately **not** rebound to the corrected tree. They
are bound to the source freeze that produced the measurement, and the published
aggregate carries that `config_sha256`; rebinding would orphan it. The
consequence is that the live tree's `source_tree_sha256` no longer matches the
configs, so **any further exp105 compute requires a fresh source freeze and fresh
configs**. Report generation does not, because it reads the published aggregate
and never touches raw or the decoder.

## Authority and limits

exp105 reports a finite-grid, decoder-dependent result for one frozen decoder on
one randomly generated expander-code ensemble at `q = 0.05`, plus a certified
one-sided lower bound on the disorder-averaged `q_top` of the exp101 posterior.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no `q_top` **estimate** at `m >= 4`, no maximum-likelihood-decoding
statement and no preparation-channel claim.

**Complete success clears no exp102 blocker.** exp102 remains
`BLOCKED_BEFORE_REMOTE` with all four blockers open, as exp105's contract said
from the start that it would.

## What it brackets, for whoever comes next

exp104: crossing at `p = 0.05512` when `q = 0`. exp105: no crossing at any `p`
when `q = 0.05`. So the readout threshold of this decoder on this family lies
strictly inside `(0, 0.05)`. That is a new, open question with a bracket around
it, and it needs its own contract.
