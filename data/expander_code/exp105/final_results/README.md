# exp105 published results

Written only by the final aggregation stage, from raw evidence that passed the
fail-closed gates and the committed bit-exact replay. If any `(m, p)` cell is
not `REPORTABLE`, or the replay gate is not `PASS`, every published statistic
here is `NaN` and the run is `INCOMPLETE`.

Every file is re-derivable from the stored per-code counts through
`exp105_pipeline.loader.load_exp105_crossing`, which recomputes the rates, the
pooled means, the cluster standard errors, the strata table, the simultaneous
band, the terminal decision, the crossing location and the certified `q_top`
lower bound rather than trusting the summary fields. An aggregate the loader
will not accept is not a result.
