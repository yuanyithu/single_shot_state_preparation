# Validation 053: exact streaming full-column preflight

This outcome-blind implementation/runtime preflight compares the historical
dense full-column CDF with the fresh single-buffer Numba streaming CDF.  It
does not run a T1 chain or estimate `q_top`.

The frozen gates and permissions are in
`RANDOM_FULL_COLUMN_STREAMING_REVIEW.md`.  Node work runs only through a
verified clean-source archive and immutable stage markers.  The aggregate
requires identical source identities, complete m8 CDF digest catalogs, and
portable sampling/replay transcript catalogs on nd-1/nd-2/nd-3.  A local pass
can only authorize this fresh three-node preflight; only aggregate `PASS` can
authorize a separately frozen T1 successor.
