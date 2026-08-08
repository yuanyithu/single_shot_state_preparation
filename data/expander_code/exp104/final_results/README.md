# exp104 final results

Published artifacts land here only after the aggregate has been re-verified
locally through `loader.load_exp104_crossing`, which recomputes rates, Wilson
intervals, pooled means, cluster standard errors, the distance-strata table, the
simultaneous band, the terminal decision and the crossing location from the
stored per-code counts.

Nothing is published while `overall_status` is `INCOMPLETE` or the replay gate is
not `PASS`.
