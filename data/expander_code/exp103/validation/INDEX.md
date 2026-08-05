# exp103 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_oracles_20260804/) | Contract, schema, decoder identity and exact oracles | `PASS` | IMPLEMENTATION_GATE | Validation 002 local preflight only |
| [002](./002_local_resource_preflight_20260804/) | Local timing, RSS and stage budgets | `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` | RESOURCE_GATE | No local scan and no remote transfer |
| [003](./003_remote_resource_preflight_20260804/) | Exact nd-3 environment, deployment identity and remote stage budgets | `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` | REMOTE_RESOURCE_GATE | No formal scan; Stage 2 exceeded the v1 core-hour and wall caps |
| [004](./004_remote_gate_v2_20260805/) | Requalification and re-gate under user-authorized v2 caps (10000/96/128) | `PASS` | REMOTE_RESOURCE_GATE | Both stages pass; reproduces 003 within 0.11%; opens formal measurement |
| 005 | Frozen m3-m5 remote scan and full replay | `IN_PROGRESS` | PRELIMINARY_MEASUREMENT | Opened by Validation 004 |
| 006 | Unconditional m6-m8 remote extension after technical PASS | `NOT_STARTED` | MEASUREMENT | Awaits Validation 005 technical PASS + replay |
| 007 | Full loader-verified crossing and checkpoint | `NOT_STARTED` | FINAL_ANALYSIS | Awaits Validation 006; no exp102 authority |
