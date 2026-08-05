# Validation 004: v2 remote resource gate (nd-3)

Status: `PASS` (both stages).

Under user-authorized amendment `exp103.remote_execution.v2`
(`REMOTE_EXECUTION_AMENDMENT_V2.md`; caps re-derived from the frozen
Validation 003 measurement: per-stage reserved cap 10000 core-hours,
predicted wall cap 96 hours, RSS cap 128 GiB unchanged), the exact nd-3
environment requalified and the same frozen outcome-blind benchmark panel
was re-measured and re-gated. No scientific field changed; measurement seeds
are unchanged by construction and the frozen seed-equality regression ran
inside qualification.

## Frozen v2 identity

- Scientific protocol: `exp103.decoder_mc.v1`, value-for-value unchanged.
- Config: `config/decoder_mc.remote.v2.json`, SHA256
  `497b9299db065c2b55668a11c2bf40cecbc8a226b13eb924f563f571e4d9794e`.
- v2 source commit `e6a0881552d6b8da42442bbfcb3b674cb9e56c27`; package tree
  SHA256 `912dea91e7f72b0d20cc5782c0c5f49ae5330e670317f9f89f5530168102210f`.
- Qualification deployment: commit
  `6715cad6aff827dbb77133bdb07cceff7721207e`, manifest SHA256
  `5b5d0d97697999636b3a0d0e6b893110b36520c62ad8f65fe42583101ead0508`,
  archive SHA256
  `433f6bc5ced49105cb5ca71ac3a7636f94bc9659f31ab9b8ed7d7e25be852a52`.
- Preflight deployment (contains the committed qualification byte-for-byte):
  commit `cc9675a0e08a609b1de2693da247bcf27823157d`, manifest SHA256
  `9c1e0d00d33b7f325ae97438c769c4d287ac094525f74f96160f8f76fc239fd1`,
  archive SHA256
  `4c4db7ebb96108a0b9cf2a5692ef3464ca0068338234708ed78d3dd74fb8179d`.
- Host `nd-3`, 64 workers, one decoder thread; run root
  `~/.single_shot/runs/exp103_remote_v2_001`.

## Qualification result

`environment_qualification.json` has SHA256
`1e71fb840669865b9261eaf6ffc0098e9a8af2bfc0360fb1cea4d4430ff31fc5`.
It records `203/203` passes (exp103 128 + exp101 58 + exp102 17) with zero
skipped, xfailed, xpassed, or deselected tests, and revalidates the official
ldpc source archive, compiled extension SHA, package versions, clean-source
wrapper, and host identity in the exact frozen environment.

## Outcome-blind resource result

`remote_resource_preflight.json` has SHA256
`fb20877760142423fe5418efe1c3d92ab21ce0efb813aff076e0d3d461404c11`, from the
frozen nine `(m3,m5,m8) x (.02,.08,.14)` benchmark tasks;
`outcome_blind=true`, `logical_outcomes_saved=false`.

| Stage | Reserved core-hours | Predicted wall-hours | Peak RSS GiB | Result |
|---|---:|---:|---:|---|
| Stage 1, m3-m5 | 1026.2918 / 10000 | 9.0023 / 96 | 19.4312 / 128 | `PASS` |
| Stage 2, m6-m8 | 9521.8178 / 10000 | 75.3736 / 96 | 25.6985 / 128 | `PASS` |

The fresh measurement reproduces Validation 003 to 0.11% (Stage 1 reserve
1027.3980 -> 1026.2918) and 0.015% (Stage 2 reserve 9520.3885 -> 9521.8178),
confirming both the 003 evidence and the v2 backward ledger derivation.

## Authority

Both v2 stage gates pass, so formal measurement opens in order: Validation
005 (Stage 1 `m=3,4,5` scan, full replay, technical report), then Validation
006 (Stage 2 `m=6,7,8`, unconditional on all Stage 1 curves), then
Validation 007 (publication loader, crossing classification, checkpoint).
This validation grants no exp102 authority and clears no exp102 blocker.
