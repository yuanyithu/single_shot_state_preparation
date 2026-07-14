# Deterministic aggregation and bounds evidence

Status: **PASS**

This evidence uses the real scan-v3 merge and producer functions;
no long stochastic sampling is part of the certification fixture.

## Parameter-point aggregation

| case | status | planned/present/valid/invalid/missing | official mean | conditional mean | crossing finite |
|---|---|---:|---:|---:|---:|
| reportable | REPORTABLE | 2/2/2/0/0 | 0.4 | 0.4 | True |
| invalid | SAMPLING_INSUFFICIENT | 3/3/2/1/0 | NaN | 0.4 | False |
| missing | INCOMPLETE | 3/2/2/0/1 | NaN | 0.4 | False |
| legacy | FORMAL_ONLY | 2/2/0/0/0 | NaN | NaN | False |
| single | REPORTABLE | 1/1/1/0/0 | 0.3 | 0.3 | True |

Invalid and missing points retain raw and valid-only diagnostics, but their official mean, SEM, and entire crossing row are NaN. The valid-only statistics are diagnostics-only because conditioning on gate success can introduce selection bias.

## Publication loader

| case | outcome |
|---|---|
| reportable | accepted |
| invalid | PublicationPointNotReportableError |
| missing | PublicationPointNotReportableError |
| legacy | NonPublicationEnsembleError |
| scan_v2 | UnsupportedScanContractError |

## MAP-purity bounds

| producer | kind | exact weights | algebraic | estimated |
|---|---|---:|---:|---:|
| exact | exact_posterior_algebraic | True | True | False |
| analytic_endpoint | analytic_endpoint_algebraic | True | True | False |
| ordinary_ti | full_sector_ti_plugin_no_coverage | False | False | True |
| sampled_valid | sampled_u_statistic_plugin_no_coverage | False | False | True |
| sampled_invalid | unavailable | False | False | False |
| legacy | unavailable | False | False | False |
| posterior_statistics | exact_posterior_algebraic | None | True | False |

Plug-in plot label: `Estimated MAP-purity bounds (plug-in; no confidence coverage)`.

Assertions passed: **104**.
