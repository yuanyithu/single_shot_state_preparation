# exp106 final results

Published artifacts land here after Validation 006 re-derives the aggregate
independently on macmini through `exp106_pipeline.loader.load_exp106_crossing`.
An aggregate the loader will not accept is not a result.

Expected contents once the run completes:

| file | what it is |
|---|---|
| `ensemble_crossing.npz` | the published aggregate, the only source of truth for every number below |
| `report.json`, `report.md` | terminal status, the `q_top` lower bound table, scope |
| `primary_curves.csv` | per `(m, p)` mean, pointwise band, cluster SE, failures, trials |
| `crossing_contrasts.csv` | `Delta38`, its simultaneous band, the per-point certification, adjacent contrasts |
| `distance_strata.csv`, `ensemble_composition.csv` | preregistered diagnostics |
| `primary_crossing.png`, `distance_strata.png` | plots |

`code_diagnostics.csv` is regenerated on demand and deliberately not tracked: it
carries one row per code and `p`, and is fully derivable from the published NPZ.
