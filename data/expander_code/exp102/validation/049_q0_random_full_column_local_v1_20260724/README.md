# Validation 049: fresh random-scan full-B-column local transport v1

V1 changes only infrastructure from the aborted validation 048: the large
classical coset-mass table is built by the exact Numba implementation already
covered by reference/Numba tests.  The cell, 64+256 clock, P/U/L allocation,
all transport gates, and maximum authority are unchanged.  V1 uses a fresh
contract, seed namespace, manifest, raw directory, and report; it reuses no V0
seed or raw.

Run with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/049_q0_random_full_column_local_v1_20260724/run_local_transport.py \
  --config data/expander_code/exp102/config/q0_random_full_column.local.v1.json
```

As in V0, a pass is only a local short-clock transport signal.  It is not a
posterior, `q_top`, HARD2, formal, held-out, or production result.
