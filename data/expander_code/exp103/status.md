# exp103 status

## Current state

**`STAGE1_MEASUREMENT_RUNNING` under `exp103.decoder_mc.v2`**

exp103 is a decoder Monte Carlo line for `q=0` code-capacity block logical
failure. After Validation 005 proved the original BpLSD choice cannot satisfy
the contract's bit-exact replay gate, the user authorized amendment
`exp103.decoder_amendment.v3` on 2026-08-06: the frozen decoder becomes the
deterministic `ldpc.BpOsdDecoder` with `osd_method=osd_0`, `osd_order=0`,
every BP parameter unchanged, and the experiment identity moves to
`exp103.decoder_mc.v2` so v1 evidence can never mix with v2 evidence.

Validation 006 froze that identity locally and Validation 007 requalified
`nd-3` at `206/206` and passed both outcome-blind resource gates. Validation
008 (`m=3,4,5` scan, complete bit-exact replay, technical report) started at
2026-08-06T08:43:56Z. No aggregate, curve, contrast, crossing or `p_c` exists
yet. exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103 has cleared none of its
blockers.

## Current gates

1. Validations 001-005 keep their original terminal states; nothing is
   reclassified. The `exp103.decoder_mc.v1` Stage 1 raw stays on the server as
   immutable evidence of the defect and is never promoted or reused.
2. v2 identity: experiment `exp103.decoder_mc.v2`, remote config
   `decoder_mc.remote.v3.json` SHA
   `f35bf575b1260c6dcfc83865a19c815fef36e8d5a6d03d9dff8dfbb601af3449`, source
   commit `6baee24bf59f8486966842dd6699a58fafecf33d`, package tree
   `5583e1e964ecc8036a805873d765fa645d29a394df03eca00da8d8646d69c722`, Linux
   OSD extension `3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079`.
3. Validation 007: qualification `206/206`, preflight `PASS_ALL_STAGES` with
   Stage 1 `1012.53 <= 10000` core-hours and `8.89 <= 96` wall-hours, Stage 2
   `9499.00 <= 10000` and `75.20 <= 96`, RSS `19.78`/`25.68` of `128` GiB.
4. The frozen suite now contains a determinism gate: two freshly constructed
   decoders must return byte-identical corrections on an identical syndrome
   sequence where BP does not converge. It has passed on the measurement host.
5. Stage 2 launches only after Stage 1 is technically complete with a passing
   bit-exact replay and a committed technical report, and that decision stays
   unconditional on all Stage 1 curves.
6. Run root `~/.single_shot/runs/exp103_remote_v3_001` on `nd-3`, 64 workers.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: frozen scientific and statistical contract.
- `REMOTE_EXECUTION_AMENDMENT.md` / `_V2.md`: execution profile and caps.
- `DECODER_AMENDMENT_V3.md`: authorized deterministic decoder and its gate.
- `config/decoder_mc.v2.json`, `config/decoder_mc.remote.v3.json`.
- `validation/005_stage1_replay_nondeterminism_20260806/`: the defect, its root
  cause, and the reproducible probes.
- `validation/007_remote_gate_v3_20260806/`: v3 qualification and gate `PASS`.
- `validation/INDEX.md`: numbered evidence ledger.

## Latest evidence

- Validation 008: running; no shard retrieved or reported yet.
- Validation 007: qualification `faa9042f...a08c56e`, preflight
  `a034d2c1...fe824d`, both `PASS`; the deterministic decoder costs 1.3% and
  0.2% less than the randomized one it replaces.
- Validation 005: `BLOCKED_REPLAY_NONDETERMINISM`; unchanged immutable evidence
  and the reason the decoder changed.
