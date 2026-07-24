# 059 q=0 hybrid row-column local pilot

Status: **`LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`**.

This validation freezes one exact direct-positive full-B-column heatbath plus
one exact variable-elimination full-B-row heatbath per macroclock.  Its first
stage is a 16-trajectory local B-transport necessary-condition pilot on frozen
P/U/M0/S0 starts.  Maximum authority is
`LOCAL_HYBRID_B_NECESSARY_GATES_PASS`; see `PRE_RUN_RED_TEAM.md`.

The frozen source is `1e9097644dbed0ccb6cd61da1dc80d57413ce4bb`.
Complete small-HGP transition matrices verify that the ordered exact-column
then exact-row macroclock preserves the collapsed target; trajectory replay and
hard-coset observations also pass (`11` focused hybrid tests, `99` related
row/column/hybrid tests).  The post-run focused regression is `107 passed`; the
complete exp101+exp102 regression is `1033 passed, 4 existing warnings`.

All 16 frozen P/U/M0/S0 trajectories complete 256 burn plus 1024 measurement
macroclocks and exact replay.  The terminal report has SHA
`2f25aa7c...873ba`, raw-set SHA `db6a303e...cd88`.  A separate raw-only auditor
does not call the sampler, row/column kernels or primary analyzer; it rebuilds
all B transitions, cached syndromes, hard-coset states, labels, weights,
likelihoods, counters, seeds, summaries and gates.  It returns
`INDEPENDENT_RAW_AUDIT_PASS_LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`, SHA
`443d461d...b7c`.

The loose necessary gates fail decisively:

| family | late B weight / 576 | late likelihood / factor | burn row changes | measurement row changes |
|---|---:|---:|---:|---:|
| P | `.03922` | `-5.2297` | 0 | 0 |
| M0 | `.04065` | `-5.1555` | 0 | 0 |
| S0 | `.04159` | `-5.0977` | 0 | 0 |
| U | `.10823` | `-11.2326` | 21--25 | 1--3 |

Every low-energy pair passes, while every U/low-energy comparison fails:
U/P differs by `.06901` in normalized B weight, `6.0030` in likelihood per
factor and `.04992` in B-bit mean-square distance.  All four U burn endpoints
miss the pre-registered `.065/-6.5` collapse gate, and U first/last likelihood
drift `.5695` also fails.  The row block changes roughly one sweep of rows
during U burn, then becomes almost frozen in the wrong high-energy basin; the
column block does not repair it.

This falsifies the proposed division of labor.  No remote/T1 job is launched,
and this raw cannot be extended, pooled with validation 056, reported as q_top
or used for formal/held-out/production work.  A successor must introduce a move
that coordinates multiple rows/columns or otherwise crosses the demonstrated
collapsed-B basin barrier; merely alternating the two exact one-block kernels
is exhausted.
