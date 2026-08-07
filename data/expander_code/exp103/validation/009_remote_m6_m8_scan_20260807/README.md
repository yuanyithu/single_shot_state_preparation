# Validation 009: Stage 2 (m=6,7,8) scan and bit-exact full replay

Status: `PASS`.

Stage 2 was launched on the authority of Validation 008's technical pass alone,
with no dependence on any Stage 1 curve, exactly as the contract requires. It
generated every planned shard and its complete bit-exact replay passed with
zero exceptions.

## What ran

Under `exp103.decoder_mc.v2` with the deterministic `BpOsdDecoder`, on `nd-3`
with 64 workers and one decoder thread, run root
`~/.single_shot/runs/exp103_remote_v3_001`, from 2026-08-06T13:21:08Z to
2026-08-07T22:54:17Z, that is 33 hours 33 minutes against the frozen 75.20-hour
predicted wall and the 96-hour cap.

- Deployment commit `dc1ad42c441d8425a60eff4b14514f5693ad96e5`, manifest SHA256
  `e210a1769a899690ac7a0f21619cf2eb8da809c5c636b06b8a2d48c6cfcbfdc6`, archive
  SHA256 `c80b035be771fa5a50f107f6989d24b8f990e870bf624c8979e5cb19d71690d2`.
- Config SHA256
  `f35bf575b1260c6dcfc83865a19c815fef36e8d5a6d03d9dff8dfbb601af3449`; package
  tree `5583e1e964ecc8036a805873d765fa645d29a394df03eca00da8d8646d69c722`.

An earlier launch at 13:18:55Z aborted after 14 seconds because the committed
Stage 1 technical report had been filed under a filename the authorization gate
does not accept. It failed before scheduling any task, produced no shard and no
artifact, and the corrected relaunch is the run recorded here.

## Results

`SCAN_STAGE2.json` (SHA256
`6d8b478da3f29740977f17faf0a25ddf212a44e5bbf6f7843699338ddfd18c09`): `PASS`,
all `312` planned code-p tasks, `1248` fresh `VALID` shards, none resumed, no
unplanned NPZ.

`REPLAY_STAGE2.json` (SHA256
`78905e8f93a1d408a78e96337ad350c3a57c4af1dd4d5eee8c0da84846d2083a`): `PASS`,
scope `stage2`, `1248` of `1248` expected shards, zero non-`PASS` results.

Across both stages the deterministic decoder reproduced `2496` of `2496`
shards bit for bit.

## Authority

Stage 2 is technically complete, so the final aggregate and the publication
loader run as Validation 010. No physical result is published here. This
validation grants no exp102 authority and clears no exp102 blocker.
