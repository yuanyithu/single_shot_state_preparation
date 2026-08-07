# exp103 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_oracles_20260804/) | Contract, schema, decoder identity and exact oracles | `PASS` | IMPLEMENTATION_GATE | Validation 002 local preflight only |
| [002](./002_local_resource_preflight_20260804/) | Local timing, RSS and stage budgets | `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` | RESOURCE_GATE | No local scan and no remote transfer |
| [003](./003_remote_resource_preflight_20260804/) | Exact nd-3 environment, deployment identity and remote stage budgets | `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` | REMOTE_RESOURCE_GATE | No formal scan; Stage 2 exceeded the v1 core-hour and wall caps |
| [004](./004_remote_gate_v2_20260805/) | Requalification and re-gate under user-authorized v2 caps (10000/96/128) | `PASS` | REMOTE_RESOURCE_GATE | Both stages pass; reproduces 003 within 0.11%; opens formal measurement |
| [005](./005_stage1_replay_nondeterminism_20260806/) | Frozen m3-m5 remote scan and bit-exact full replay | `BLOCKED_REPLAY_NONDETERMINISM` | MEASUREMENT_GATE | Scan PASS (1248 VALID shards); replay INVALID on 53. BpLSD's LSD stage is randomized, so bit-exact replay is unsatisfiable for m>=4 at p>=0.06. No aggregate, no physical result |
| [006](../DECODER_AMENDMENT_V3.md) | Freeze the deterministic BP+OSD-0 decoder identity and pass the local suite | `PASS` | IMPLEMENTATION_GATE | `exp103.decoder_mc.v2`; 131 local tests including the new determinism gate; amendment recorded in `DECODER_AMENDMENT_V3.md` |
| [007](./007_remote_gate_v3_20260806/) | nd-3 requalification and remote resource gate under the v3 decoder identity | `PASS` | REMOTE_RESOURCE_GATE | 206/206 including the determinism gate on the Linux build; both stages pass; opens formal measurement |
| [008](./008_remote_m3_m5_scan_20260806/) | Frozen m3-m5 remote scan and bit-exact full replay | `TECHNICAL_PASS` | PRELIMINARY_MEASUREMENT | Scan PASS (1248 fresh VALID); replay PASS 1248/1248 with zero exceptions; authorizes Validation 009 with no dependence on Stage 1 curves |
| [009](./009_remote_m6_m8_scan_20260807/) | Unconditional m6-m8 remote extension after technical PASS | `PASS` | MEASUREMENT | Scan PASS (1248 fresh VALID); replay PASS 1248/1248; 2496/2496 shards bit-exact across both stages |
| [010](./010_final_crossing_20260807/) | Full loader-verified crossing classification and checkpoint | `EXP103_NO_CORRECT_CROSSING_IN_WINDOW` | FINAL_ANALYSIS | 624/624 REPORTABLE, 6.24M trials, no certified bracket; simultaneous half-width 0.2601; eight frozen d=2 codes dominate the low-p mean; secondary distance strata reverse between p=0.07 and 0.08. No exp102 authority |
