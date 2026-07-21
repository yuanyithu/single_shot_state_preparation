# exp102 q=0 preflight runtime-gate separation

Status: `REPAIR IMPLEMENTED / FRESH PREFLIGHT PENDING`.  This is
infrastructure evidence only; no sampler task ran and no physics result was
produced.

## Failed immutable attempt

- Run: `exp102_q0_global_20260721_c6c26b9`
- Source: `c6c26b9f4972c067ca77c9d6d790847bb62f48c0`
- Archive SHA256:
  `27a65384026bf19b359c2c76aafbb7df7d3f25f3d7196f3b437c28de5083af2e`
- Manifest SHA256:
  `af2dcc1d959520c72d2f04d9f1578787e49eabd3f381d9bdb795412379bf80ea`
- Schedule file SHA256:
  `42f09874c9e00391c2dac0b7850313d11a64877ed46eca83720dd76b37967e37`
- Schedule identity SHA256:
  `f34a253789d8658f7161f591093c4e212c7bcec5e1dd400a8584efb2ae8c1a52`

The nd-2 regression suite returned two failures because its live runtime
fixture classified a transient TI timing as `RUNTIME_EXHAUSTED`.  The
regression then stopped before the dedicated, persisted three-node runtime
gate.  All non-runtime tests passed and the run is permanently FAILED.

## Diagnosis

A fresh, verified-source postmortem benchmark on the same nd-2 and same source
returned `PASS` with T3 selected.  Every hard/defect method was eligible; the
factor-two full discovery projection was about 1.32 hours.  The volatile edge
was the TI anchor projection: 77,982 seconds against a 79,200-second stage
window.  A small timing fluctuation can legitimately change that machine
qualification, but it is not a deterministic code regression.

The regression now still executes the real benchmark and verifies that its
PASS/`RUNTIME_EXHAUSTED` status, tier selection, methods, and projections are
self-consistent.  Consensus unit tests use a deterministic PASS-shaped report.
Only `run_global_preflight_node.py` plus `combine_runtime_reports` may make the
actual three-node resource decision.  No runtime threshold, TI configuration,
resource tier, safety factor, or cumulative deadline was changed.

The frozen schedule, compressed logs, FAILED markers, and postmortem runtime
JSON are in `failed_run_evidence/`.  A new source commit, deployment, run ID,
and 72-hour schedule are mandatory.

Local validation after the separation repair: exp102 `224 passed`; the two
runtime/consensus regressions pass with a cold cache; compileall, shell syntax,
and `git diff --check` pass.  Exp101 and root physics code are unchanged from
the preceding `366 passed` and `16 passed` runs.
