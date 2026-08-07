# exp103 status

## Current state

**`EXP103_NO_CORRECT_CROSSING_IN_WINDOW`** — complete, published, closed.

exp103 measured the `q=0` code-capacity block logical failure rate of one
frozen decoder over the frozen 48-code expander ensemble. Under
`exp103.decoder_mc.v2` with the deterministic `ldpc.BpOsdDecoder`
(`osd_method=osd_0`, `osd_order=0`), both stages ran on `nd-3`, every one of
the `2496` shards replayed bit for bit, and the complete panel of `624`
code-p cells and `6,240,000` trials was aggregated fail-closed and published
through the frozen loader.

The preregistered primary contrast `Delta38` is positive at every grid point,
so there is no negative-to-positive reversal and no certified bracket. The
frozen simultaneous band has half-width `0.2601`, which would have prevented
certifying any bracket in any case.

No asymptotic threshold, exponent, FSS, `q_top`, MLD or preparation-channel
claim is made. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103 cleared none of
its blockers, as its contract always stated it could not.

## What the result means, and what it does not

The primary equal-weight mean cannot resolve a threshold on this ensemble, and
Validation 010 identifies why: eight frozen classical-distance-2 codes fail
`0.4051` of the time already at `p=0.02`, a floor set by distance rather than
size, spread unevenly over the six `m` panels. Comparing `m` panels of unequal
distance composition is not a size comparison, which is also why adjacent-size
contrasts alternate in sign.

Four uncertified views of the same trials locate a threshold-like reversal in a
narrow region: the per-`m` median at `[0.05, 0.06]`, `Delta45` at `[0.06,
0.07]`, the distance-stratified means at `[0.07, 0.08]`, and `Delta67` at
`[0.07, 0.08]`. Under the contract these are diagnostics and cannot change the
primary status, and every simultaneous band contains zero everywhere, so no
location is certified.

The frozen classifier keys the terminal status on `Delta38` alone and returns
before examining the adjacent family. The contract's prose for this status
instead says "complete valid data contain no negative-to-positive point-estimate
reversal", which is literally false here because `Delta45` and `Delta67` do
reverse. The gap is a defect of the contract text, disclosed in Validation 010
and not resolved by relabelling a frozen decision after seeing data; a successor
contract must state the primary-only scope explicitly.

Shot noise never binds: the largest fixed-panel Monte Carlo standard error over
all 624 cells is `0.0018` against a largest between-code standard deviation of
`0.3245`. A successor experiment should change the estimand over a
heterogeneous ensemble and the simultaneous band, not the sample size. Any such
experiment needs its own contract and explicit user authorization.

## Current gates

1. Validations 001-010 keep their terminal states; nothing is reclassified.
   The `exp103.decoder_mc.v1` BpLSD raw stays on the server as immutable
   evidence of the randomized-decoder defect and is never promoted or reused.
2. Publication is loader-verified: `624/624` `REPORTABLE`, `overall_status`
   `COMPLETE`, `replay_status` `PASS` with scope `final_combined`, terminal
   status `EXP103_NO_CORRECT_CROSSING_IN_WINDOW`.
3. Known artifact defect: `report.md` and the primary plot title name the
   superseded `BpLSD` decoder. The machine-readable authority field, config
   SHA, decoder binary SHA and experiment identity are all correct. Recorded in
   Validation 010 rather than patched, because regenerating would invalidate
   the frozen identity the aggregate is bound to. To be fixed in the next
   freeze, together with the stale `BpLSD` strings in pipeline error messages.
4. No further exp103 compute is authorized. Restarting requires a new contract.

## Evidence map

- `EXPERIMENT_CONTRACT.md`, `REMOTE_EXECUTION_AMENDMENT.md` / `_V2.md`,
  `DECODER_AMENDMENT_V3.md`: contract and the three authorized amendments.
- `config/decoder_mc.v2.json`, `config/decoder_mc.remote.v3.json`.
- `validation/005_stage1_replay_nondeterminism_20260806/`: the randomized
  decoder defect, its root cause and reproducible probes.
- `validation/008_...`, `009_...`: bit-exact stage evidence.
- `validation/010_final_crossing_20260807/`: the published result, its
  secondary diagnostics and its known defect.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 010: `EXP103_NO_CORRECT_CROSSING_IN_WINDOW`, aggregate SHA256
  `460b3868...4ec7cf`, report SHA256 `fe354933...2f535`, no bracket.
- Validation 009: Stage 2 scan and replay `PASS`, `1248/1248`, 33h33m.
- Validation 008: Stage 1 scan and replay `PASS`, `1248/1248`, 4h26m.
