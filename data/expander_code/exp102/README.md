# exp102 random-HGP q=0 scan

This directory contains the independent exp102 pipeline. It does not modify or claim exp101 scan-v3
certification. Read `EXPERIMENT_CONTRACT.md` before running anything.

Stable module entry points (run from the project root with conda environment `12`):

```bash
python -m data.expander_code.exp102.exp102_pipeline.registry data/expander_code/exp102/registry
python -m data.expander_code.exp102.exp102_pipeline.pilot merge-select PILOT_RAW REGISTRY CONFIG PILOT_REPORT
python -m data.expander_code.exp102.exp102_pipeline.pilot freeze PILOT_REPORT FROZEN --registry REGISTRY --config CONFIG
python -m data.expander_code.exp102.exp102_pipeline.tasks REGISTRY CONFIG TASK_PLAN --frozen FROZEN
python -m data.expander_code.exp102.exp102_pipeline.aggregate RAW REGISTRY CONFIG FROZEN FINAL_RESULTS
python -m data.expander_code.exp102.exp102_pipeline.plot FINAL_RESULTS/exp102_results.npz FINAL_RESULTS
```

The exhausted PT-v2 design search has a separate entry point and config. `DISCOVERY_RAW` must
include the remote `control/` evidence plus every node's raw manifest, status, SUCCESS marker, and
exact NPZ set:

```bash
python -m data.expander_code.exp102.exp102_pipeline.discovery analyze \
  DISCOVERY_RAW data/expander_code/exp102/registry/registry.json \
  data/expander_code/exp102/config/discovery.v2.json SOURCE_COMMIT DISCOVERY_REPORT
```

Discovery output cannot be passed to `pilot merge-select` or `pilot freeze`.
The frozen search stopped at transport with zero certified round trips, so it did not produce a
formal v2 configuration; see `validation/005_pt_v2_discovery_20260720/README.md`.

Production is intentionally blocked until pilot/held-out validation writes a configuration whose
status is `FROZEN_HELD_OUT_PASS`. Do not hand-edit that status to bypass pilot gates.
Production workers are launched only through the report-verifying production stage runner; direct
worker invocation intentionally lacks the verified freezer/source environment and is rejected.
