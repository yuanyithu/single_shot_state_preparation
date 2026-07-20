# exp102 random-HGP q=0 scan

This directory contains the independent exp102 pipeline. It does not modify or claim exp101 scan-v3
certification. Read `EXPERIMENT_CONTRACT.md` before running anything.

Stable module entry points (run from the project root with conda environment `12`):

```bash
python -m data.expander_code.exp102.exp102_pipeline.registry data/expander_code/exp102/registry
python -m data.expander_code.exp102.exp102_pipeline.pilot merge-select PILOT_RAW REGISTRY CONFIG PILOT_REPORT
python -m data.expander_code.exp102.exp102_pipeline.pilot freeze PILOT_REPORT FROZEN --registry REGISTRY --config CONFIG
python -m data.expander_code.exp102.exp102_pipeline.tasks REGISTRY CONFIG TASK_PLAN
python -m data.expander_code.exp102.exp102_pipeline.worker REGISTRY CONFIG FROZEN CODE_ID DISORDER OUTPUT
python -m data.expander_code.exp102.exp102_pipeline.aggregate RAW REGISTRY CONFIG FROZEN FINAL_RESULTS
python -m data.expander_code.exp102.exp102_pipeline.plot FINAL_RESULTS/exp102_results.npz FINAL_RESULTS
```

Production is intentionally blocked until pilot/held-out validation writes a configuration whose
status is `FROZEN_HELD_OUT_PASS`. Do not hand-edit that status to bypass pilot gates.
