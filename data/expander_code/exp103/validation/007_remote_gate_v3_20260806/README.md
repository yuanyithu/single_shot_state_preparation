# Validation 007: v3 remote gate for the deterministic decoder (nd-3)

Status: `PASS` (both stages).

Under user-authorized amendment `exp103.decoder_amendment.v3`
(`DECODER_AMENDMENT_V3.md`), the exact `nd-3` environment requalified against
the new deterministic decoder identity and the same frozen outcome-blind
benchmark panel was re-measured and re-gated. Caps are inherited unchanged from
amendment v2: 10000 reserved core-hours, 96 predicted wall-hours, 128 GiB RSS
per stage.

## Frozen v3 identity

- Experiment `exp103.decoder_mc.v2`; decoder `ldpc.BpOsdDecoder` with
  `osd_method=osd_0`, `osd_order=0` and unchanged BP parameters.
- Config `config/decoder_mc.remote.v3.json`, SHA256
  `f35bf575b1260c6dcfc83865a19c815fef36e8d5a6d03d9dff8dfbb601af3449`.
- Source commit `6baee24bf59f8486966842dd6699a58fafecf33d`; package tree SHA256
  `5583e1e964ecc8036a805873d765fa645d29a394df03eca00da8d8646d69c722`.
- Linux OSD extension `_bposd_decoder.cpython-312-x86_64-linux-gnu.so`, SHA256
  `3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079`.
- Qualification deployment commit `0206e2d29b341604a156a2f6978116cee4125eba`,
  manifest SHA256
  `470e8b338a144197d17f78eba59ad509569edd2d4def87119bd110e18e574fad`.
- Preflight deployment commit `e4d4a4976c2acab46eb2001b0c336a8f5bcce2e2`,
  manifest SHA256
  `66855c24d1c08cf7e087724c17483f7324e920ea52f2cfb0eca2faff5b26082f`.
- Host `nd-3`, 64 workers, one decoder thread; run root
  `~/.single_shot/runs/exp103_remote_v3_001`.

## Qualification result

`environment_qualification.json` has SHA256
`faa9042fa975dd7d4050b737844e7147e9569720359778dfb0e0caf73a08c56e`:
`206/206` passes (exp103 131 + exp101 58 + exp102 17) with zero skipped,
xfailed, xpassed or deselected tests, and clean bytecode before and after.

This run is the first to execute the new determinism gate on the measurement
host itself. Two freshly constructed decoders, from both the worker and the
replay construction paths, returned byte-identical corrections over an
identical syndrome sequence on `m04_c01` at `p=0.11`, where belief propagation
fails to converge on more than half the trials. BP+OSD-0 bit-exactness is
therefore attested on the Linux build under contract, not only by the macmini
probes recorded in Validation 005.

## Outcome-blind resource result

`remote_resource_preflight.json` has SHA256
`a034d2c18d55f629850880c2671a7b3cc5c077bf1ad8bcb88b85c98efffe824d`, from the
frozen nine `(m3,m5,m8) x (.02,.08,.14)` benchmark tasks; `outcome_blind=true`,
`logical_outcomes_saved=false`.

| Stage | Reserved core-hours | Predicted wall-hours | Peak RSS GiB | Result |
|---|---:|---:|---:|---|
| Stage 1, m3-m5 | 1012.53 / 10000 | 8.89 / 96 | 19.78 / 128 | `PASS` |
| Stage 2, m6-m8 | 9499.00 / 10000 | 75.20 / 96 | 25.68 / 128 | `PASS` |

The deterministic decoder is marginally cheaper than the randomized one it
replaces: against Validation 004's BpLSD measurement the Stage 1 reserve falls
from 1026.29 to 1012.53 core-hours and Stage 2 from 9521.82 to 9499.00, that
is by 1.3% and 0.2%. This confirms in situ the 0.92-0.99 cost ratio measured
locally, so the switch consumes no additional approved budget.

## Authority

Both v3 stage gates pass, so formal measurement opens in order: Validation 008
(Stage 1 `m=3,4,5` scan, complete bit-exact replay, technical report), then
Validation 009 (Stage 2 `m=6,7,8`, unconditional on all Stage 1 curves), then
Validation 010 (publication loader, crossing classification, checkpoint). This
validation grants no exp102 authority and clears no exp102 blocker.
