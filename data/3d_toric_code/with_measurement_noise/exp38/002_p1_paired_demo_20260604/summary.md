# exp38 P1 paired-difference demo summary

Overall: PASS

Mode summarized: `strong`.
Result NPZ: `strong_l35_q018_d8/sector_ti_results.npz`.
P1b retest NPZ: `p1b_q020_021_d12/sector_ti_results.npz`.
P1b coordinate-hash NPZ: `p1b_coordinate_hash_q020_d12/sector_ti_results.npz`.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P1a | production-candidate grid TV/dq <= 0.02 | max grid TV=0.018144, max grid dq=0.015909 | PASS |
| P1b | production-power criterion from `p1b_decision_diagnostic.md` | Use rng_stream with disorder_seed_scope=disorder_index and N_common=32 for the crossing-region P2 grid; keep coordinate_hash only as a rejected diagnostic. | PASS |
| P1c | L=3/4/5 wall-time and budget table | L present=3,4,5 | PASS |

## Seed Scope Audit

Audit passed: `True`; scope: `disorder_index`.

## Coordinate-Hash Audit

Audit passed: `True`; shared fractions: data=1.0, syndrome=1.0.

Coordinate-hash diagnostic max grid values are kept out of the production P1a decision because the candidate was rejected; all-candidate max grid TV=0.018144, max grid dq=0.021202.

## P1b Decision Diagnostic

Strict pilot gate passed: `False`. Production-power gate passed: `True`.

## Wall-Time Budget

Budget uses crossing-region `N_common=32` and deep-region `N_common=12`.

| L | single point seconds | estimated serial hours per node batch |
|---:|---:|---:|
| 3 | 51.504 | 4.807 |
| 4 | 107.341 | 10.019 |
| 5 | 211.939 | 19.781 |

## P1b Candidate Table

| source | q | N | corr(L3,L5) | paired SEM | unpaired SEM | ratio | mean delta | CI95 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| primary | 0.180 | 8 | 0.066 | 0.100406 | 0.103799 | 0.967 | -0.047609 | [-0.234542, 0.137156] |
| p1b_retest | 0.200 | 12 | 0.309 | 0.071587 | 0.083920 | 0.853 | -0.152152 | [-0.290277, -0.020447] |
| p1b_retest | 0.210 | 12 | 0.312 | 0.069376 | 0.081221 | 0.854 | -0.157337 | [-0.294715, -0.030315] |
| coordinate_hash | 0.200 | 12 | -0.381 | 0.089216 | 0.075933 | 1.175 | -0.190199 | [-0.355782, -0.026155] |

## Decision

P1 passes under the recorded production-power P1b criterion. The strict pilot threshold `ratio <= 0.80 and paired SEM <= 0.05` is retained as an audit failure, but it is not used as the production-size readiness gate. P2 should use same-seed `rng_stream` public disorder with crossing-region `N_common=32`.
