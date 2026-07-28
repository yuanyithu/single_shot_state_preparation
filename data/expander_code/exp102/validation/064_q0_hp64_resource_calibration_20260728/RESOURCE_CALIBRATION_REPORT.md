# HP64 resource calibration report

Status: `RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE`.

No option is selected. Every numerical total below is a planning proxy, not a confidence bound.
Strict empirical totals remain `null` because validation 013 lacks m7, most p values, and multi-code/multi-disorder timing distributions.

## Grid sizes

| Stage | Evaluations |
|---|---:|
| `m3_easy_block_128` | 128 |
| `calibration_grid_3p` | 18432 |
| `formal_grid_7p` | 43008 |

## T3/2T, 32-trajectory proxy examples

| Stage | Scenario | Clock | Safety core-hours | Ideal 166-core hours | Ideal 75-core hours |
|---|---|---:|---:|---:|---:|
| `m3_easy_block_128` | `same_m_proxy_with_m7_global_max` | `T3` | 1210.490 | 7.292 | 16.140 |
| `m3_easy_block_128` | `same_m_proxy_with_m7_global_max` | `2T` | 2420.980 | 14.584 | 32.280 |
| `m3_easy_block_128` | `global_observed_max_proxy` | `T3` | 17029.233 | 102.586 | 227.056 |
| `m3_easy_block_128` | `global_observed_max_proxy` | `2T` | 34058.465 | 205.171 | 454.113 |
| `calibration_grid_3p` | `same_m_proxy_with_m7_global_max` | `T3` | 1111247.434 | 6694.262 | 14816.632 |
| `calibration_grid_3p` | `same_m_proxy_with_m7_global_max` | `2T` | 2222494.868 | 13388.523 | 29633.265 |
| `calibration_grid_3p` | `global_observed_max_proxy` | `T3` | 2452209.501 | 14772.346 | 32696.127 |
| `calibration_grid_3p` | `global_observed_max_proxy` | `2T` | 4904419.002 | 29544.693 | 65392.253 |
| `formal_grid_7p` | `same_m_proxy_with_m7_global_max` | `T3` | 2592910.680 | 15619.944 | 34572.142 |
| `formal_grid_7p` | `same_m_proxy_with_m7_global_max` | `2T` | 5185821.360 | 31239.888 | 69144.285 |
| `formal_grid_7p` | `global_observed_max_proxy` | `T3` | 5721822.169 | 34468.808 | 76290.962 |
| `formal_grid_7p` | `global_observed_max_proxy` | `2T` | 11443644.338 | 68937.616 | 152581.925 |

The full 72-row option matrix is in `resource_scenarios.csv`; the 18-row empirical coverage matrix is in `timing_coverage.csv`.
Ideal wall times omit scheduling imbalance, serial stages, filesystem contention, current load, and failures.
