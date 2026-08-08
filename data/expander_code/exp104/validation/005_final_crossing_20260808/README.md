# Validation 005: certified crossing and its location

Terminal status: **`EXP104_CERTIFIED_CROSSING`**.
Certified finite-grid bracket: **`[0.05, 0.06]`**.
Crossing location: **`p_cross = 0.05512`, 95% bootstrap interval
`[0.05327, 0.05699]`**.

The complete panel was aggregated fail-closed and published: `108,000` of
`108,000` code-p cells `REPORTABLE`, 12,000 codes, 432,000 trials, replay `PASS`
on the committed subsample. The aggregate was re-loaded independently on macmini
through `load_exp104_crossing`, which recomputes rates, Wilson intervals, pooled
means, cluster standard errors, the strata table, the cluster bootstrap, the
simultaneous band, the terminal decision and the crossing location from the
stored per-code counts. It accepted the aggregate and reproduced this
classification.

## Frozen identity

- Aggregate `ensemble_crossing.npz` SHA256
  `dcca50dd8a8f1c684e7262ca15597025aced5205af73db5015f6bd56ec8da130`.
- Config SHA256
  `85616f2679a64ffb44c87c7488918385e8e5506d2e8501ecf7f7d4259509db2a`; registry
  SHA256 `7e40ff18fdf4fd52476894dc21caa516e16a1b97cdfd2a9ad9f803c709f315d4`;
  experiment `exp104.ensemble_mc.v1`.
- CSV, PNG and NPZ artifacts sit beside this README and are not tracked in Git;
  their bytes are pinned by the `file_sha256` manifest inside the committed
  `report.json`, and the verification is recorded in `local_verification.json`.

## Primary result

Ensemble-mean block logical failure rate, 2000 random codes per `m`, four trials
per code and grid point. The band is the 95% simultaneous cluster bootstrap over
the nine grid points of `Delta38`.

| p | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 | Delta38 | band | |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.02 | 0.0813 | 0.0791 | 0.0749 | 0.0736 | 0.0686 | 0.0716 | -0.0096 | [-0.0307, +0.0115] | |
| 0.03 | 0.1420 | 0.1514 | 0.1348 | 0.1260 | 0.1184 | 0.1164 | -0.0256 | [-0.0467, -0.0045] | certified negative |
| 0.04 | 0.2286 | 0.2334 | 0.2236 | 0.2059 | 0.1961 | 0.1755 | -0.0531 | [-0.0742, -0.0320] | certified negative |
| 0.05 | 0.3351 | 0.3530 | 0.3494 | 0.3324 | 0.3118 | 0.2944 | -0.0408 | [-0.0619, -0.0196] | certified negative |
| 0.06 | 0.4550 | 0.5120 | 0.5186 | 0.5032 | 0.5131 | 0.4939 | +0.0389 | [+0.0178, +0.0600] | certified positive |
| 0.07 | 0.5895 | 0.6573 | 0.7027 | 0.7160 | 0.7244 | 0.7318 | +0.1422 | [+0.1211, +0.1634] | certified positive |
| 0.08 | 0.7069 | 0.7960 | 0.8508 | 0.8770 | 0.9056 | 0.9153 | +0.2084 | [+0.1873, +0.2295] | certified positive |
| 0.09 | 0.7943 | 0.8966 | 0.9490 | 0.9714 | 0.9822 | 0.9902 | +0.1960 | [+0.1749, +0.2171] | certified positive |
| 0.10 | 0.8752 | 0.9561 | 0.9862 | 0.9959 | 0.9972 | 0.9994 | +0.1241 | [+0.1030, +0.1452] | certified positive |

`Delta38` is certified negative at `p = 0.03, 0.04, 0.05` and certified positive
from `p = 0.06` onward, so the reversal is certified and the tightest bracket is
`[0.05, 0.06]` — adjacent grid points, though the rule did not require that.

**`p = 0.02` is not certified.** `Delta38 = -0.0096` there against a half-width of
0.0211, so the band contains zero and the point is reported as uncertified rather
than as a negative. Below `p = 0.03` the curves are too close together for this
sample to separate them.

## The band

The simultaneous half-width is **`0.0211`**, against exp103's `0.2601` on the same
question: **12.3 times narrower**. Two changes produce that, and neither is extra
sampling:

- The resampling unit is the code, and there are 2000 per `m` instead of 8. The
  largest cluster standard error over the whole panel is `0.00652` against
  exp103's largest of `0.115`.
- The simultaneous scope is the primary contrast's nine grid points, not six
  curves and five adjacent contrasts at once.

exp104 used **432,000 trials against exp103's 6,240,000** — 14.4 times fewer — and
certified a crossing exp103 could not see.

## Why exp103 saw the opposite sign

The composition actually drawn, as fractions of the 2000 codes at each `m`:

| m | d=2 | d=4 | d=6 | d=8 | d=10 |
|---|---:|---:|---:|---:|---:|
| 3 | 0.2285 | 0.5960 | 0.1755 | - | - |
| 4 | 0.1925 | 0.3995 | 0.4015 | 0.0065 | - |
| 5 | 0.1535 | 0.2705 | 0.5065 | 0.0695 | - |
| 6 | 0.1280 | 0.2210 | 0.4515 | 0.1995 | - |
| 7 | 0.1170 | 0.1685 | 0.3760 | 0.3280 | 0.0105 |
| 8 | 0.1040 | 0.1215 | 0.3130 | 0.3950 | 0.0665 |

This reproduces the Validation 001 census to within sampling error. exp103's
eight-code panels drew `0, 3, 2, 2, 0, 1` distance-2 codes for `m = 3..8`; its
`m = 3` panel drew none where about 23 percent were due, which biased
`P_fail(m=3)` low and pushed `Delta38` positive at every grid point. Nothing was
wrong with exp103's decoder, seeds or scoring — Validation 002 shows the two code
paths are the same function — and nothing was wrong with its trial count. Its
panels simply were not comparable across `m`.

## Preregistered secondaries

Diagnostics. They cannot change the terminal status and none is a published
result.

Distance-stratified failure rate at `m = 8`:

| p | d=2 | d=4 | d=6 | d=8 | d=10 |
|---|---:|---:|---:|---:|---:|
| 0.02 | 0.5084 | 0.0648 | 0.0192 | 0.0117 | 0.0038 |
| 0.04 | 0.7728 | 0.2315 | 0.1042 | 0.0769 | 0.0602 |
| 0.06 | 0.9087 | 0.6029 | 0.4688 | 0.3927 | 0.3647 |
| 0.08 | 0.9964 | 0.9475 | 0.9062 | 0.8978 | 0.8759 |
| 0.10 | 1.0000 | 1.0000 | 1.0000 | 0.9987 | 0.9981 |

Failure falls monotonically with distance at every grid point, and the
distance-2 stratum carries a floor of 0.51 already at `p = 0.02` that does not
shrink with size. That stratum is 22.9 percent of the ensemble at `m = 3` and
10.4 percent at `m = 8`, which is why the ensemble mean improves with size at low
`p` at all.

The adjacent contrast family is now coherent rather than alternating: `Delta45`,
`Delta56`, `Delta67` and `Delta78` are all negative or near zero below the
crossing and positive above it, and `Delta34` turns positive earliest. In exp103
these alternated in sign because the panels differed in composition; here they do
not.

## Authority

This validation reports a finite-grid, decoder-dependent, code-capacity result
for one frozen BP+OSD-0 decoder on one randomly generated full-rank
(3,4)-biregular expander-code ensemble at `q = 0`. `p_cross` is where the
ensemble-mean failure curves for `m = 3` and `m = 8` cross under this decoder on
this grid.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no `q_top`, no maximum-likelihood-decoding statement and no
preparation-channel claim. A different decoder, a different ensemble or a
different acceptance rule would give a different number.

It clears no exp102 blocker and authorizes no exp102 stage. exp102 remains
`BLOCKED_BEFORE_REMOTE`.
