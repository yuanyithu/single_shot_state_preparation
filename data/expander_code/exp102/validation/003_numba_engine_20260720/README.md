# Full-round Numba q=0 PT benchmark

This diagnostic compares the Python reference oracle with the production
full-round Numba kernel using exactly the same seed. It excludes the first JIT
compile from the timed samples and refuses to report a speedup unless every
trajectory and diagnostic field is bit-identical.

Run from the project root in the local development environment:

```bash
conda run --no-capture-output -n 12 python \
  data/expander_code/exp102/validation/003_numba_engine_20260720/benchmark_q0_pt.py
```

Use `--code-id m08_c00` to exercise the `k=64` boundary.
