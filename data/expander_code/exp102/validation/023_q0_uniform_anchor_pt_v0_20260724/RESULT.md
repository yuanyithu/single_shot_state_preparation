# UARE V0 local result

## Terminal status

`LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`

This is a local adversarial-initialization diagnostic only. It is not a
posterior estimate, `q_top` result, HARD2 authorization, formal pilot,
held-out result, or production authorization.

## Frozen evidence

- Frozen cell: `m08_c06,p=.04,d00,attempt022`, with its nonzero hard syndrome.
- Manifest SHA256: `9098102f1612cb70630d936fb86b949e9a19baa428c187238741d6dbd2f1b560`.
- `RUN_COMPLETE.json`: all 48 immutable raw files, SHA256
  `322a23b72f1fb443e435f95ce64088f7a524437a3005b3e1979e7bb2ff507761`.
- The independent raw-only audit is V2 because V1 had a dictionary-indexing
  defect in its time-half summary before it could write a result. V2 records
  its source hash, reads pickle-free raw only, and reports SHA256
  `76e5233dba8a0a24618199f0f397552f9d8d01dd12bc5701016ea6f200d5290f`.
- `replay_validate_v2.py` leaves the manifest-bound V1 runner unmodified and
  calls its raw validator and deterministic trajectory replay for all 48
  tasks. It reports `all_bit_identical=true`, matches every audit raw hash,
  and has replay SHA256
  `f2c84bb8334d7b1ac6c7c56799ca9e4296c07a24274066e2d5983df2e0d767d4`.

The frozen V1 runner has the same post-replay time-half dictionary-indexing
defect, so it emitted no `REPORT.json`. It was not edited: changing a
manifest-bound source file after raw creation would invalidate the evidence.
The V2 replay artifact is deliberately separate and does not overwrite the
frozen runner's namespace.

## Why both candidates fail

Both `UARE32-R1` and `UARE64-R1` fail the predeclared target-support gate and
P/U/L distribution comparisons. The planted P and legal low-energy L families
agree, but the exact-K=0 uniform hard-coset U family does not agree with either
one. Its minimum measurement weights are 247--255 for UARE32 and 247--262 for
UARE64, whereas the known legal planted state has weight 63.

For this hard coset, dimension 832 is already a deliberately loose
multiplicity bound. At `p=.04`, every U trajectory's observed region has

```text
Pr_pi(|e| >= w) <= 2^832 * (.04/.96)^(w-63) <= 3.148385600959564e-4,
```

and therefore falls below the frozen `.001` support threshold. U also fails
the fixed-clock early/late stability comparisons. Thus ordinary B movement,
swaps, or label changes cannot be interpreted as convergence to the cold
hard-coset posterior.

The result rejects only these two fixed UARE configurations at this fixed
local budget. The raw must not be extended, pooled, reweighted, used for
`q_top`, sent to HARD2, or used to authorize `READY_FOR_FORMAL`, tuning,
held-out, or production. It does not establish that q=0, the posterior, or
all uniform-anchored replica-exchange algorithms are mathematically
impossible.
