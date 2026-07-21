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

The reviewed successor search is the separately versioned PA discovery. Read
`PA_DISCOVERY_CONTRACT.md` before using these entry points:

```bash
python -m data.expander_code.exp102.exp102_pipeline.pa_discovery make-config \
  data/expander_code/exp102/registry/registry.json \
  data/expander_code/exp102/config/q0_pa.discovery.v1.json
python -m data.expander_code.exp102.exp102_pipeline.pa_discovery plan \
  REGISTRY PA_CONFIG SOURCE_COMMIT hard_screen METHODS_JSON TASK_MANIFEST
python -m data.expander_code.exp102.exp102_pipeline.pa_discovery analyze-hard \
  HARD_RAW TASK_MANIFEST REGISTRY PA_CONFIG HARD_REPORT
```

Only a `READY_FOR_CONFIRMATION` hard report may freeze the two blinded manifests through
`freeze-confirmation`; only complete confirmation plus resolution may return `READY_FOR_FORMAL`.
PA raw is intentionally incompatible with the formal PT pilot/freezer. The transport autopsy has
its own `transport_autopsy` module and can only explain the old PT failure.

Production is intentionally blocked until pilot/held-out validation writes a configuration whose
status is `FROZEN_HELD_OUT_PASS`. Do not hand-edit that status to bypass pilot gates.
Production workers are launched only through the report-verifying production stage runner; direct
worker invocation intentionally lacks the verified freezer/source environment and is rejected.
