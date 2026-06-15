# exp38 P4 acceptance and paired differences

Status: `PASS`

## Inputs

- P2 acceptance: `/Users/jarvis/Desktop/sync/project D/data/3d_toric_code/with_measurement_noise/exp38/003_p2_production_grid_20260605/accepted_exp38_p2_ti_grid_20260605_0145/p2_acceptance.json`
- P2 TI NPZ: `/Users/jarvis/Desktop/sync/project D/data/3d_toric_code/with_measurement_noise/exp38/003_p2_production_grid_20260605/merged_exp38_p2_ti_grid_20260605_0145/sector_ti_results.npz`
- P3 second method: `/Users/jarvis/Desktop/sync/project D/data/3d_toric_code/with_measurement_noise/exp38/004_p3_second_method_subset_20260605/remote_collected_exp38_p3_second_method_20260605_0610/exp38_p3_second_method_20260605_0610/second_method_subset/p3_second_method_subset.json`

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| P4a | P2 acceptance gates passed, common disorder verified, no FAIL point/disorder rows | P2 gates={'P2a': True, 'P2b': True, 'P2c': True, 'common_disorder': True}, point statuses={'PASS': 20, 'WARN': 19}, disorder statuses={'PASS': 1195, 'WARN': 53} | PASS |
| P4b | P3 second-method subset gates passed | checks=3, max TV=0.003319, max |dq_top|=0.005144, max full-path gap=0.046359 | PASS |
| P4c | PASS-only paired differences recorded for every q/L-pair; crossing-region CIs include some nonzero separations | rows=39/39, min paired N=21, CI excludes zero rows=5 at q=[0.2, 0.21, 0.22, 0.23] | PASS |

## Pair Resolution Context

| L pair | min paired N | crossing-region rows | crossing-region CI excludes zero | q values | resolution |
|---|---:|---:|---:|---|---|
| L3-L4 | 25 | 9 | 1 | [0.22] | resolved at listed q |
| L3-L5 | 26 | 9 | 4 | [0.2, 0.21, 0.22, 0.23] | resolved at listed q |
| L4-L5 | 21 | 9 | 0 | [] | unresolved by paired CI |

## Paired Difference Highlights

| L pair | q | delta mean | paired SEM | 95% paired bootstrap CI | N paired |
|---|---:|---:|---:|---:|---:|
| L3-L5 | 0.200 | 0.093636 | 0.046340 | [0.004285, 0.183186] | 28 |
| L3-L5 | 0.210 | 0.126523 | 0.048923 | [0.034266, 0.220022] | 27 |
| L3-L4 | 0.220 | 0.119672 | 0.048277 | [0.029338, 0.213907] | 32 |
| L3-L5 | 0.220 | 0.151243 | 0.043315 | [0.069371, 0.235345] | 32 |
| L3-L5 | 0.230 | 0.120005 | 0.044547 | [0.034386, 0.207773] | 29 |

## Artifacts

- `p4_acceptance.json`
- `failure_map.md`
- `paired_difference.csv`
- `p4_point_status.csv`
