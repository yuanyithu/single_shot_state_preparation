# exp104 status

## Current state

**`EXP104_CERTIFIED_CROSSING`** — complete, published, closed.

exp104 measured the `q = 0` code-capacity block logical failure rate of the
frozen `exp103.decoder_mc.v2` BP+OSD-0 decoder over a randomly generated
expander-code ensemble: 2000 codes per `m` for `m = 3..8`, four trials per code
and grid point, nine points from 0.02 to 0.10, 432,000 trials. Everything ran on
nd-3 with 64 workers; the scan took 45.9 minutes.

- Certified bracket **`[0.05, 0.06]`**.
- Crossing location **`p_cross = 0.05512`**, 95% bootstrap interval
  **`[0.05327, 0.05699]`**.
- Simultaneous band half-width **`0.0211`**, against exp103's `0.2601`.
- `108,000` of `108,000` code-p cells `REPORTABLE`; committed replay `PASS` with
  `60,120` trials bit-exact.

`Delta38` is certified negative at `p = 0.03, 0.04, 0.05` and certified positive
from `p = 0.06` onward. `p = 0.02` is **not** certified: the contrast is `-0.0096`
against a half-width of `0.0211`, so the band contains zero there.

No asymptotic threshold, exponent, FSS, `q_top`, MLD or preparation-channel claim
is made. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp104 cleared none of its
blockers, as its contract always stated it could not.

## What this settles about exp103

exp103 returned no certified crossing not because there is none but because its
eight-code panels were not comparable across `m`. The measured composition
confirms it: the distance-2 fraction of this family falls from `0.229` at `m = 3`
to `0.104` at `m = 8`, and exp103's panels drew `0, 3, 2, 2, 0, 1` such codes. Its
`m = 3` panel drew none where about 23 percent were due, which biased
`P_fail(m=3)` low and pushed `Delta38` positive at every grid point.

Nothing was wrong with exp103's decoder, seeds or scoring: Validation 002 shows
the two code paths are bit-identical functions over 60,000 trials. Nothing was
wrong with its trial count either. exp104 used 14.4 times fewer trials and
certified what exp103 could not, by spending the budget on codes instead.

## Current gates

1. Validations 001-005 keep their terminal states; nothing is reclassified.
2. Publication is loader-verified on macmini: `108,000/108,000` `REPORTABLE`,
   `overall_status` `COMPLETE`, `replay_status` `PASS`, terminal status
   `EXP104_CERTIFIED_CROSSING`, `p_cross` and the band re-derived from the stored
   per-code counts.
3. The compiled decoder is not bit-portable across platforms (Validation 002).
   exp104 generated, replayed and aggregated entirely on nd-3 against the pinned
   nd-3 binary SHA256, and never mixes artifacts across platforms.
4. No further exp104 compute is authorized. Restarting requires a new contract.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract and its primary-only
  terminal rule.
- `config/ensemble_mc.v1.json`, `config/ensemble_mc.remote.v1.json`,
  `config/ensemble_registry.v1.json`.
- `validation/001_...`: red team, ensemble census, local gates.
- `validation/002_...`: implementation equality with exp103; platform finding.
- `validation/003_...`: nd-3 qualification and resource gate.
- `validation/004_...`, `005_...`: the measurement and the published result.
- `validation/INDEX.md`: numbered evidence ledger.
- `final_results/`: published aggregate, report, curves and plots.

## Latest evidence

- Validation 005: `EXP104_CERTIFIED_CROSSING`, aggregate SHA256
  `dcca50dd...8da130`, bracket `[0.05, 0.06]`, `p_cross 0.05512`.
- Validation 004: scan `PASS` 778/778 in 45.9 min; replay `PASS` 83/83.
- Validation 003: nd-3 qualification and resource gate `PASS`.
