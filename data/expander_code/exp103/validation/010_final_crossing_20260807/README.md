# Validation 010: final loader-verified crossing classification

Terminal status: **`EXP103_NO_CORRECT_CROSSING_IN_WINDOW`**. Certified
finite-grid bracket: **none**.

The complete 48-code panel was aggregated fail-closed and published through the
frozen loader. All `624` code-p cells are `REPORTABLE`, `6,240,000` trials, and
the combined replay gate is `PASS` with scope `final_combined`. The preregistered
primary contrast `Delta38 = P_fail(m8) - P_fail(m3)` never reverses from
negative to positive across `p = 0.02 ... 0.14`, so the frozen decision rule
returns no crossing and no bracket.

The publication was independently re-loaded on macmini through
`load_exp103_crossing`, which rechecks hashes, counts, rates, Wilson intervals,
equal-weight means, fail-closed statuses, contrasts and the terminal decision;
it accepted the aggregate and reproduced this classification.

## Frozen identity

- Aggregate `decoder_crossing.npz` SHA256
  `460b3868c3f903d44366d52c3eccb7d589f7534733601f3a6030cd85e74ec7cf`;
  `report.json` SHA256
  `fe3549335551eb9c8976dea2427df1afc6efc8cc3b21037dbe262da51e32f535`.
- Config SHA256
  `f35bf575b1260c6dcfc83865a19c815fef36e8d5a6d03d9dff8dfbb601af3449`;
  experiment `exp103.decoder_mc.v2`; aggregate schema `exp103.aggregate.v2`.
- CSV, PNG and NPZ artifacts live beside this README but are not tracked in
  Git; their bytes are pinned by the `file_sha256` manifest inside the
  committed `report.json`.

## Primary result

`Delta38` point estimates are positive at every grid point, with a minimum of
`0.0584` at `p=0.05` and a maximum of `0.2470` at `p=0.08`, falling back to
`0.0145` at `p=0.14` where every panel saturates at one. A negative-to-positive
reversal is therefore absent, which is the preregistered
`EXP103_NO_CORRECT_CROSSING_IN_WINDOW` outcome rather than an anomaly.

The frozen 20,000-draw simultaneous band has half-width `0.2601` on the failure
scale and is applied to every primary curve and every declared contrast. No
bracket in this experiment could have been certified at that width, whatever
the point estimates had done.

## Two adjacent-size contrasts do reverse, and the decision does not use them

The frozen classifier keys the terminal status entirely on `Delta38`. In
`crossing.classify_final_crossing`, a `Delta38` with no correct-direction
reversal returns `EXP103_NO_CORRECT_CROSSING_IN_WINDOW` immediately, before the
adjacent contrasts are examined at all. That is the rule Validation 001 froze
and tested, and it is the rule that ran here.

The declared adjacent family nevertheless contains reversals:

| p | Delta34 | Delta45 | Delta56 | Delta67 | Delta78 |
|---|---:|---:|---:|---:|---:|
| 0.02 | 0.0914 | -0.0248 | 0.0170 | -0.1021 | 0.1116 |
| 0.04 | 0.1352 | -0.0426 | 0.0142 | -0.1594 | 0.1129 |
| 0.06 | 0.1525 | -0.0172 | 0.0102 | -0.1017 | 0.0613 |
| 0.07 | 0.1603 | 0.0123 | 0.0262 | -0.0427 | 0.0426 |
| 0.08 | 0.1492 | 0.0336 | 0.0357 | 0.0013 | 0.0273 |
| 0.10 | 0.0983 | 0.0251 | 0.0093 | 0.0034 | 0.0012 |

`Delta45` is negative through `p=0.06` and positive from `p=0.07`, and
`Delta67` is negative through `p=0.07` and positive from `p=0.08`. Both are
correct-direction, threshold-like reversals, bracketed at `[0.06, 0.07]` and
`[0.07, 0.08]`. `Delta34`, `Delta56` and `Delta78` stay positive throughout.

This is disclosed, not relabelled. The contract's prose for this status reads
"complete valid data contain no negative-to-positive point-estimate reversal",
which taken literally is false here, while the frozen implementation means
"no such reversal in the primary contrast `Delta38`", which is true. The gap
between the prose and the code is a defect of the contract text, and a
successor contract must state the primary-only scope explicitly. The terminal
status is not changed after seeing the data.

## Why the primary cannot see a threshold, and what the secondaries show

These are secondary, plug-in, uncertified observations. Under the contract they
cannot change the primary status, and none of them is a published result.

The panel contains eight frozen classical-distance-2 codes. Their mean block
failure is `0.4051` already at `p=0.02` and `0.7449` at `p=0.05`. That floor
comes from the distance, not the size, so it does not shrink as `m` grows; it
adds a large, roughly size-independent offset to the low-`p` end of every
equal-weight mean, and because the eight codes are spread unevenly over the six
`m` panels it also makes the primary means non-monotone in `m` at low `p`. An
equal-weight mean over an ensemble that mixes distance-2 codes with
distance-10 codes is not a threshold probe.

Grouping the same trials by classical distance shows the expected signature:

| p | d=2 (8 codes) | d=4 (9) | d=6 (21) | d=8 (8) | d=10 (2) |
|---|---:|---:|---:|---:|---:|
| 0.02 | 0.4051 | 0.0404 | 0.0136 | 0.0123 | 0.0051 |
| 0.05 | 0.7449 | 0.3077 | 0.2237 | 0.2099 | 0.1441 |
| 0.07 | 0.8999 | 0.6259 | 0.6193 | 0.6520 | 0.6100 |
| 0.08 | 0.9488 | 0.7653 | 0.8086 | 0.8577 | 0.8650 |
| 0.10 | 0.9902 | 0.9194 | 0.9660 | 0.9893 | 0.9987 |

Among the `d>=4` strata the ordering is monotonically decreasing in distance up
to `p=0.06`, flat at `p=0.07`, and monotonically increasing from `p=0.08`. The
reversal is bracketed between `p=0.07` and `p=0.08`.

The per-`m` median over the eight codes, which suppresses the distance-2 tail
without deleting any code, reverses between `p=0.05` and `p=0.06`: the `m=8`
median is below the `m=3` median at `p<=0.05` (`0.2095` against `0.2579`) and
above it from `p=0.06` (`0.4200` against `0.3815`).

Four independent views of the same trials therefore locate a threshold-like
reversal in a narrow region: the per-`m` median at `[0.05, 0.06]`, `Delta45` at
`[0.06, 0.07]`, the distance-stratified means at `[0.07, 0.08]`, and `Delta67`
at `[0.07, 0.08]`. Taken together they are consistent with a decoder threshold
near `p ~ 0.06-0.08` for this decoder on this ensemble. **None of this is
certified.** Every simultaneous band in the frozen family contains zero at
every grid point, so the experiment certifies no location, and these agreeing
point estimates are exactly the kind of evidence that a successor experiment
must be designed to test rather than to confirm.

The pattern of which contrasts reverse is itself explained by ensemble
composition rather than physics. The primary means at `p=0.02` alternate with
`m`: `0.0307`, `0.1221`, `0.0972`, `0.1142`, `0.0121`, `0.1237` for `m=3..8`.
Panels differ in how many distance-2 codes they carry, so adjacent-size
contrasts alternate in sign, and the pairs that fail to reverse are the pairs
whose composition difference outweighs the size difference. Comparing `m`
panels of unequal distance composition is not a size comparison.

## Methodological finding for any successor experiment

Shot noise is not the limiting quantity anywhere in this panel. The largest
fixed-panel Monte Carlo standard error over all 624 cells is `0.0018`, while
the largest between-code standard deviation is `0.3245`, and it is the latter
that the simultaneous max-absolute-deviation band inflates into `+/-0.2601`.
Adding trials per cell would change nothing. What binds is the definition of
the estimand over a heterogeneous ensemble and the choice of simultaneous
band; a successor contract should address those, not the sample size.

## Known defect in the generated artifacts

`report.md` and the primary plot title name the superseded `BpLSD` decoder.
That string was missed by the rename in the v3 amendment, and regenerating the
artifacts would change the frozen package tree and invalidate the identity that
this aggregate is bound to, so it is recorded here instead of patched. The
machine-readable authority field is correct
(`finite_grid_bposd_decoder_crossing_only`), and the config SHA, decoder binary
SHA and experiment identity carried by the aggregate all identify BP+OSD-0
unambiguously. The source string is to be corrected in the next freeze.

## Authority

This validation reports a finite-grid, decoder-dependent, code-capacity result
for one frozen decoder on one frozen 48-code ensemble. It asserts no asymptotic
threshold, no critical exponent, no finite-size-scaling collapse, no `q_top`,
no MLD statement and no preparation-channel claim. It clears no exp102 blocker
and authorizes no exp102 stage. The secondary distance-strata and median
observations above are diagnostics; certifying them requires a separately
contracted experiment with its own preregistered primary and band.
