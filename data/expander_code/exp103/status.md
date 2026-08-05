# exp103 status

## Current state

**`REMOTE_GATE_V2_PASSED; STAGE1_MEASUREMENT_OPEN`**

exp103 remains a BpLSD decoder-MC line for `q=0` block logical failure with
scientific protocol `exp103.decoder_mc.v1` unchanged. After the v1 remote
resource gate blocked Stage 2 (Validation 003), the user authorized
amendment `exp103.remote_execution.v2` on 2026-08-05: caps re-derived from
the frozen 003 measurement (per-stage reserved 10000 core-hours, wall 96
hours, RSS 128 GiB), all other execution and scientific clauses inherited
verbatim. Validation 004 requalified nd-3 (203/203) and passed both stage
gates on a fresh outcome-blind run of the same frozen benchmark. Formal
measurement proceeds as Validation 005 (m3-m5), then 006 (m6-m8,
unconditional on Stage 1 curves), then 007 (crossing + checkpoint). At this
commit no measurement shard exists yet; Stage 1 launch follows immediately.
exp102 remains `BLOCKED_BEFORE_REMOTE`; exp103 clears none of its blockers.

## Current gates

1. Validations 001/002/003 remain immutable evidence with their original
   terminal states; nothing is reclassified.
2. Validation 004: `PASS` both stages. Stage 1 reserve `1026.2918 <= 10000`
   core-hours, wall `9.0023 <= 96`, RSS `19.4312 <= 128`; Stage 2 reserve
   `9521.8178 <= 10000`, wall `75.3736 <= 96`, RSS `25.6985 <= 128`.
   Reproduces 003 within 0.11%.
3. v2 identity: config `decoder_mc.remote.v2.json` SHA
   `497b9299db065c2b55668a11c2bf40cecbc8a226b13eb924f563f571e4d9794e`,
   source commit `e6a0881552d6b8da42442bbfcb3b674cb9e56c27`, package tree
   `912dea91e7f72b0d20cc5782c0c5f49ae5330e670317f9f89f5530168102210f`,
   host nd-3 with 64 workers, run root
   `~/.single_shot/runs/exp103_remote_v2_001`.
4. Stage 2 launches only after Stage 1 is technically complete with a
   passing full replay and the committed technical report; the decision is
   unconditional on all Stage 1 curves.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: frozen scientific and statistical contract.
- `REMOTE_EXECUTION_AMENDMENT.md`: v1 profile (superseded caps; historical).
- `REMOTE_EXECUTION_AMENDMENT_V2.md`: authorized v2 caps and gate mechanics.
- `config/decoder_mc.remote.v2.json`: exact qualified v2 remote identity.
- `validation/004_remote_gate_v2_20260805/`: v2 qualification + gate PASS.
- `validation/INDEX.md`: numbered evidence ledger.
- `raw/`: no formal exp103 shard at this commit; Stage 1 raw arrives under
  `raw/stage1/` after retrieval and SHA verification.

## Latest evidence

- Validation 004: qualification SHA256 `1e71fb84...ff31fc5` (203/203);
  preflight SHA256 `fb208777...404c11`, `PASS_ALL_STAGES`, outcome-blind.
- Validation 003: `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` under v1 caps;
  unchanged immutable evidence.
- Validation 002: `BLOCKED_LOCAL_RESOURCE_PREFLIGHT`; unchanged.
