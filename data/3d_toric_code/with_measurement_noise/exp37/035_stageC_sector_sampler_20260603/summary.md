# Stage C sector sampler summary

Overall: PASS

Sampler: production decoder-reject fixed-sector sweep from `src/exp37_sector_ti.py`.
Exact reference: L=2 full enumeration at p=0.28, q=0.305, zero eta and zero measurement error.
MCMC config: burn=3000, measurements=24000, stride=2, blocks=48, bootstrap=4000, CI=[0.005,0.995].

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| C1 | sector_trace constant for all sectors | violations=0 | PASS |
| C2 | exact means inside block-bootstrap CI | max abs d_data=0.03075, max abs d_synd=0.03908 | PASS |

## Sector Mean Comparison

| sector | data exact | data MCMC | data 99% CI | syndrome exact | syndrome MCMC | syndrome 99% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.410585 | 1.426250 | [1.357625, 1.492501] | 2.290452 | 2.310333 | [2.227500, 2.393917] |
| 1 | 4.811502 | 4.814792 | [4.768875, 4.860875] | 5.105466 | 5.120583 | [5.061912, 5.179750] |
| 2 | 4.811502 | 4.842250 | [4.786830, 4.900127] | 5.105466 | 5.095583 | [5.019163, 5.173169] |
| 3 | 6.842094 | 6.858917 | [6.805582, 6.915085] | 6.671668 | 6.632583 | [6.583750, 6.683917] |
| 4 | 4.811502 | 4.800542 | [4.747917, 4.850961] | 5.105466 | 5.106667 | [5.043081, 5.173418] |
| 5 | 6.842094 | 6.823042 | [6.770122, 6.877000] | 6.671668 | 6.639833 | [6.591916, 6.690008] |
| 6 | 6.842094 | 6.855208 | [6.808040, 6.899213] | 6.671668 | 6.665833 | [6.613750, 6.719751] |
| 7 | 7.921701 | 7.934167 | [7.878540, 7.991084] | 7.556727 | 7.544583 | [7.502499, 7.586251] |

Artifacts:
- `stageC_results.json`
- `sector_mean_comparison.csv`
- `sector_trace_sample.csv`
