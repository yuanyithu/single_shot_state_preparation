# exp103 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_oracles_20260804/) | Contract, schema, decoder identity and exact oracles | `PASS` | IMPLEMENTATION_GATE | Validation 002 local preflight only |
| [002](./002_local_resource_preflight_20260804/) | Local timing, RSS and stage budgets | `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` | RESOURCE_GATE | No local scan and no remote transfer |
| [003](./003_remote_resource_preflight_20260804/) | Exact nd-3 environment, deployment identity and remote stage budgets | `NOT_STARTED` | REMOTE_RESOURCE_GATE | No formal measurement until both stages PASS |
| 004 | Frozen m3-m5 remote scan and full replay | `NOT_STARTED` | PRELIMINARY_MEASUREMENT | Restricted preliminary only |
| 005 | Unconditional m6-m8 remote extension after technical PASS | `NOT_STARTED` | MEASUREMENT | Requires Validation 003 Stage 2 PASS |
| 006 | Full loader-verified crossing and checkpoint | `NOT_STARTED` | FINAL_ANALYSIS | No exp102 authority |
