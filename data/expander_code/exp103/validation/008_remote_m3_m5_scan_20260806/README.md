# Validation 008: Stage 1 (m=3,4,5) scan and bit-exact full replay

Status: `TECHNICAL_PASS`.

Stage 1 generated every planned shard and, unlike the `exp103.decoder_mc.v1`
attempt recorded in Validation 005, the complete bit-exact replay passed on
all 1248 shards with zero exceptions. This validation reports technical
completeness only. It quotes no rate, curve, contrast or crossing, and the
Stage 2 decision is unconditional on the Stage 1 curves.

## What ran

Under `exp103.decoder_mc.v2` with the deterministic `BpOsdDecoder`
(`osd_method=osd_0`, `osd_order=0`), on `nd-3` with 64 workers and one decoder
thread, run root `~/.single_shot/runs/exp103_remote_v3_001`, from
2026-08-06T08:43:54Z to 13:10:05Z, that is 4 hours 26 minutes against the
frozen 8.89-hour predicted wall and the 96-hour cap.

- Deployment commit `141d7804237394c06681510c9c2219a398a07dda`, manifest SHA256
  `dae0ea93167bc707951da9b65d1b1e4830049a8f10169b0c99a46bc71553a1d4`, archive
  SHA256 `067df3ad609ba6c582e8d13017be516e34ecf259e765ecfcec5283ee283e74cb`.
- Config SHA256
  `f35bf575b1260c6dcfc83865a19c815fef36e8d5a6d03d9dff8dfbb601af3449`; package
  tree `5583e1e964ecc8036a805873d765fa645d29a394df03eca00da8d8646d69c722`;
  Linux OSD extension
  `3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079`.

## Results

`SCAN_STAGE1.json` (SHA256
`a0e4192cf2cc94643bc5185319d60ba2bb5290cbdca70079abfdfbd13eb6fd5b`): `PASS`.
All `312` planned code-p tasks produced `1248` fresh `VALID` shards, none
resumed, and no unplanned NPZ appeared in the stage root.

`REPLAY_STAGE1.json` (SHA256
`391c05a56f2fb396f07f4ada6b95f861d5890fd5de3bb7700ad957e29c9301be`): `PASS`,
scope `stage1`, `1248` of `1248` expected shards, **zero non-`PASS` results**.
Every trial's error was regenerated from its frozen seed, re-decoded by audit
code that does not import the worker scorer, and compared field by field along
with all three stream hashes. The comparison that failed on 53 shards under the
randomized decoder now succeeds on every shard.

`technical_report.json`, the run-root `stage1_technical_report.json` verbatim (SHA256
`d1cc26b4c77558409bb41c4cbda6f9c414abda5abc36a46537cc261a292f88bc`):
`TECHNICAL_PASS`, `reportable_code_p = 312`, `measurement_shards = 1248`,
`replay_status = PASS`, `outcome_blind_stage2_authorization = true`, bound to
aggregate SHA256
`6c0c0df08db048379fb190d641e190f06049636518a7c504e78f858304ac7fc0`, replay
report SHA256
`569d972c9fe4b0a5ab1fd410eb0e46e462e5f11a94d3e246e452653fc046a7b7` and raw
manifest SHA256
`68c2779eaa626333de66d6f06a9caacc4f40c76debe0d139b5ef4d7f22f09c07`.

The Stage 1 aggregate is `EXP103_INCOMPLETE` by construction: `m=6,7,8` have no
raw yet, so fail-closed aggregation refuses a complete panel, exactly as the
contract requires. `verify-stage stage1` revalidated every live artifact and
returned `PASS`.

## Authority

Stage 1 is technically complete with a passing bit-exact replay, so Validation
009 (`m=6,7,8`) is authorized and its launch does not depend on any Stage 1
curve. No physical result is published here; the finite-grid crossing
classification belongs to Validation 010 through the publication loader. This
validation grants no exp102 authority and clears no exp102 blocker.
