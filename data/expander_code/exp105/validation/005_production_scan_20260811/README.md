# Validation 005: production scan, committed replay and aggregation on nd-3

Status: **`PASS`**. The measurement is complete and its terminal status is
`EXP105_NO_CERTIFIED_CROSSING`. Publication verification is Validation 006.

## Scan

`scan.json`: **`PASS`**, 3,314 of 3,314 tasks, all fresh, none resumed, in
**7,928 seconds (2.20 wall hours)** on nd-3 with 64 workers. Against the frozen
projection of 7.01 predicted wall hours, that is a factor 3.2 of headroom, in the
same direction as exp103's and exp104's.

17,617 codes and 1,057,020 trials at `q = 0.05`, over the ten frozen grid points
from `p = 0.001` to `p = 0.07`, six trials per code and `p`.

## Committed replay

`replay.json`: **`PASS`**, 337 of 337 committed tasks, **110,160 trials
reproduced bit for bit**.

Replay is not a rerun of the worker. It builds its own decoder, and it scores
through `audit_scorer.independent_label_map` and
`audit_scorer.trivial_class_generators`, which reconstruct the logical criterion
from the RREF pivot rule of `H_Z` without touching the exp101 frame the worker
used. It then requires agreement on failure flags, logical labels, readout-match
flags, convergence flags, iteration counts and all four stream digests. A bug
shared between the two scorers would have to survive two independent
implementations and four digests.

The subsample was fixed in Validation 004, before any production task ran.

## Aggregation

`overall_status` **`COMPLETE`**, `replay_status` **`PASS`**,
**176,170 of 176,170** code-`p` cells `REPORTABLE`, zero unexpected raw errors.

Terminal status **`EXP105_NO_CERTIFIED_CROSSING`**.

| p | `P_fail(m=3)` | `P_fail(m=8)` | `Delta38` | simultaneous band |
|---|---:|---:|---:|---|
| 0.0010 | 0.01412 | 0.08092 | +0.06679 | [+0.05631, +0.07728] |
| 0.0015 | 0.01916 | 0.10671 | +0.08755 | [+0.07707, +0.09804] |
| 0.0025 | 0.03118 | 0.15496 | +0.12379 | [+0.11330, +0.13427] |
| 0.0040 | 0.04700 | 0.22724 | +0.18024 | [+0.16975, +0.19072] |
| 0.0060 | 0.06826 | 0.31047 | +0.24220 | [+0.23172, +0.25269] |
| 0.0100 | 0.11730 | 0.47319 | +0.35589 | [+0.34540, +0.36637] |
| 0.0160 | 0.19954 | 0.65707 | +0.45753 | [+0.44704, +0.46801] |
| 0.0250 | 0.32286 | 0.84218 | +0.51932 | [+0.50884, +0.52981] |
| 0.0400 | 0.53875 | 0.97625 | +0.43749 | [+0.42701, +0.44798] |
| 0.0700 | 0.86165 | 1.00000 | +0.13835 | [+0.12787, +0.14884] |

Simultaneous band half-width **0.01049**, against exp104's 0.2601 at `q = 0` with
its 2,000-code panels and exp103's 0.2601.

## This is a certified absence, not a failure to resolve

The primary contrast is **certified positive at all ten grid points**: the
simultaneous 95 percent band excludes zero from below everywhere. There is no
negative point, so there is no bracket, so the terminal is
`EXP105_NO_CERTIFIED_CROSSING` — but the reason is that the larger code is
certifiably worse than the smaller one at every `p` in the window, not that the
experiment could not tell.

Physically, at `q = 0.05` this family is already above threshold from readout
noise alone. A code at `m = 8` carries `n_c = 12m² = 768` checks, each with an
independent 5 percent chance of being misread, against `n = 1600` data qubits
carrying as few as 1.6 expected errors at `p = 0.001`. Validation 003's control,
with the data error set identically to zero, showed the same size dependence.

The `Delta38` curve peaks at `p = 0.025` and falls afterwards only because both
sizes saturate: `P_fail(m=8)` reaches 1.00000 at `p = 0.07`.

## What this brackets

exp104 puts the crossing at `p = 0.05512` when `q = 0`. exp105 finds no crossing
at any `p` when `q = 0.05`. The readout threshold of this decoder on this family
therefore lies strictly inside `(0, 0.05)`. Locating it is a different experiment
and needs its own contract; exp105 may not become it.

## Evidence in this directory

- `scan.json`, `replay.json`
- The aggregate, report, curves and plots: `../../final_results/`
- Raw: `~/.single_shot/runs/exp105_noisy_v1_004/raw/` on nd-3, 3,314 files, not
  tracked in Git.

## Authority end

A finite-grid, decoder-dependent result for one frozen decoder on one randomly
generated ensemble at one readout rate. No asymptotic threshold, no exponent, no
finite-size scaling, no `q_top` estimate at `m >= 4`, no MLD claim, no
preparation-channel claim. **Clears no exp102 blocker.**
