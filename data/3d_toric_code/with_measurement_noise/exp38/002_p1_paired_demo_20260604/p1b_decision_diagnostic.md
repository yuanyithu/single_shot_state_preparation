# exp38 P1b decision diagnostic

Target paired SEM: `0.05`.
Planned production N_common: `32`.

## Decision

Strict pilot gate passed: `False`.
Production-power criterion passed: `True`.
Selected candidate: `p1b_retest` q=0.210, N=12, paired SEM=0.069376, unpaired SEM=0.081221, ratio=0.854, N_needed=24.

Recommendation: use `disorder_seed_scope=disorder_index`, `disorder_realization_mode=rng_stream`, and crossing-region `N_common=32` in P2.

## Candidate Table

| source | q | N | corr | paired SEM | unpaired SEM | ratio | N needed | projected SEM @24 | @32 | @40 | CI excludes 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| primary | 0.180 | 8 | 0.066 | 0.100406 | 0.103799 | 0.967 | 33 | 0.057970 | 0.050203 | 0.044903 | False |
| p1b_retest | 0.200 | 12 | 0.309 | 0.071587 | 0.083920 | 0.853 | 25 | 0.050620 | 0.043838 | 0.039210 | True |
| p1b_retest | 0.210 | 12 | 0.312 | 0.069376 | 0.081221 | 0.854 | 24 | 0.049056 | 0.042484 | 0.037999 | True |
| coordinate_hash | 0.200 | 12 | -0.381 | 0.089216 | 0.075933 | 1.175 | 39 | 0.063085 | 0.054633 | 0.048866 | True |
