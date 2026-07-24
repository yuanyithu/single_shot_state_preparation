# 058 q=0 exact full-row elimination feasibility

Status: **`LOCAL CONDITIONAL FEASIBLE / STANDALONE LOW-ENERGY TRANSPORT NOT
VIABLE`**.

This local-only validation tests whether a state-independent exact variable
elimination order can make a full 24-bit B-row heatbath practical.  It targets
the B slow coordinate exposed by validation 056 and is mechanistically
independent of CPPT/HP replica transport.  It has no q_top, remote, formal,
held-out or production authority; see `PRE_RUN_RED_TEAM.md`.

## Exact and resource result

The conditional target is

```text
P(B[i,:]=v | B[-i,:],Y)
  proportional to (p/(1-p))^|v|
               product_j M_p(base_j xor parity(v & h_j) * 2^i).
```

Deterministic min-fill gives induced width 12, a largest factor of 8192
entries and order SHA `43aa25dd...5ac`.  Complete zero/nonzero-syndrome tests
on the mandatory n=10 and n=13 HGPs compare the elimination distribution and
normalizer to row enumeration, check single-row detailed balance and complete
row-sweep stationarity, and replay the PortablePrng draw and cached syndromes;
all 20 tests pass.  The final exp101+exp102 regression is `1020 passed` with
four pre-existing deprecation/fork warnings.

On frozen `m08_c06,p=.04,d00`, the 128 MiB mass table builds in `0.316s`.
The conservative row-update time is `0.01291s`, the incremental sweep peak is
about 17.0 MiB, and the frozen factor-two T1 projection is `264.39s < 7200s`.
The local feasibility report is `LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE`, SHA
`0f99bba4...172da`.  This status deliberately certifies only exactness and
computability, not transport.

## Scientific stop

The outcome-blind `P/U/M0/S0` panel exposes the decisive failure:

| family | median conditional entropy (bits) | median expected changed bits | minimum self probability | sampled 24-row sweep |
|---|---:|---:|---:|---:|
| P | 0 | `1.59e-21` | `.9999999939` | 0 rows |
| M0 | 0 | `1.19e-21` | `.9999999930` | 0 rows |
| S0 | 0 | `1.85e-21` | `.9999999926` | 0 rows |
| U | `2.619` | `11.645` | `3.41e-51` | 24 rows / 294 bits |

An independent target-only implementation, which does not import the row
sampler or its statistic code, reproduces expected Hamming changes within
`7.8e-13` and self log probabilities within `1.9e-13`.  Its conservative
union bounds for seeing even one row change in 10240 cyclic updates are
`9.33e-6`, `9.70e-6` and `9.89e-6` from P, M0 and S0.  Audit status is
`INDEPENDENT_TARGET_AUDIT_PASS_LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE`, SHA
`3845759b...bd1`.

The full-row move therefore collapses a high-energy U state aggressively but
is effectively the identity on all three frozen low-energy basins.  A
standalone fixed-clock screen would only reconfirm this exact bound and is not
run.  The implementation may be reused only as a component of a fresh,
pre-registered mixed kernel that separately supplies low-energy basin
transport.  Such a mixed collapsed-B method would still share HP/direct-column
failure modes and could not serve as the required mechanism-independent
confirmation.
