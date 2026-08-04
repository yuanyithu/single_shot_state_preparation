# Validation 002: local resource preflight

Status: `BLOCKED_LOCAL_RESOURCE_PREFLIGHT`.

The fixed nine benchmark tasks use only the benchmark seed namespace and record
timing/RSS. Logical outcomes are neither saved nor inspected. The report evaluates
the frozen resource formula separately for Stage 1 (`m=3,4,5`) and Stage 2
(`m=6,7,8`); a failed stage is not transferred to remote.

The immutable canonical report is `resource_preflight.json` (SHA256
`455b23c95340da9676a1622ff5f24e8e158f668dcc6da92689496661e1c31d39`).
Stage 1 is blocked because its reserved total is `104.0546` core-hours, above
the frozen `100` core-hour cap; its `7.3784` hour wall and `2.6532` GiB RSS
checks pass. Stage 2 is blocked at `1020.9320` reserved core-hours and `64.6832`
wall-hours; its `4.7054` GiB RSS check passes. No formal shard was launched.
