# Validation 053: exact streaming full-column preflight

This outcome-blind implementation/runtime preflight compares the historical
dense full-column CDF with the fresh single-buffer Numba streaming CDF.  It
does not run a T1 chain or estimate `q_top`.

The frozen gates and permissions are in
`RANDOM_FULL_COLUMN_STREAMING_REVIEW.md`.  No report exists until the source is
committed, tested, packaged from a clean worktree, and run with the exact
source commit.  A local pass can only authorize a fresh three-node preflight.
