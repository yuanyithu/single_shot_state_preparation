# Validation 002: local resource preflight

Status: `PENDING`.

The fixed nine benchmark tasks use only the benchmark seed namespace and record
timing/RSS. Logical outcomes are neither saved nor inspected. The report evaluates
the frozen resource formula separately for Stage 1 (`m=3,4,5`) and Stage 2
(`m=6,7,8`); a failed stage is not transferred to remote.
