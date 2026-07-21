# exp102 q=0 global-sampling discovery

Status: `IMPLEMENTED / FRESH PREFLIGHT PENDING`. The first immutable remote
attempts failed before sampler work and are permanently closed; see
`../008_q0_global_preflight_portability_20260721/` and
`../009_q0_global_runtime_gate_separation_20260721/`. This directory does not
contain a three-node discovery result and does not establish
`READY_FOR_FORMAL`.

## Frozen scope

- Contract: `exp102.q0_global.discovery.v1`
- Config SHA256:
  `1d0a453f2bf8445ad6587c612c2eabb3049e76e2d73b59c230b8b1358b06e565`
- Hard raw: `exp102.q0_hardcoset.raw.v1`
- Defect raw: `exp102.q0_defect_trace.raw.v1`
- Bias raw: `exp102.q0_defect_bias.raw.v1`
- TI raw: `exp102.q0_global.ti_anchor.raw.v1`
- Registry SHA256:
  `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`

The complete protocol, panels, gates, resources, and stopping rules are in
`../../GLOBAL_DISCOVERY_CONTRACT.md`. PT-v2 and PA raw, controls, seeds, and
freezers are incompatible with this workflow.

## Files

- `benchmark_global.py`: per-node candidate and TI timing plus three-node
  worst-case runtime consensus.
- `cross_node_global.py`: reference/Numba and three-node canonical digest.
- `wmc_feasibility.py`: width-bounded exact WMC feasibility for SMALL6; an
  exceeded width or timeout is diagnostic only.
- `run_global_preflight_node.py`: complete exp102+exp101 Linux tests, digest,
  runtime, and nd-1 WMC.
- `orchestrate_global_preflight.py`: immutable three-node preflight launcher and
  consensus builder.
- `run_global_stage.py`: one frozen owner-node task executor.
- `run_global_wrapper.sh`: exclusive RUNNING/SUCCESS/FAILED marker wrapper.
- `orchestrate_global.py`: load check, ownership freeze, remaining-wall gate,
  verified-source launch, and marker wait for one stage.

Every compute-node Python command is entered through the archived
`run_verified_source.sh`. Direct execution inside `repos/<run>/source` is
forbidden because it changes the verified tree. Preflight and stage workers
also reverify the complete archive identity before writing SUCCESS.

## Local validation

Use conda environment `12`:

```bash
conda run -n 12 --no-capture-output pytest -q -p no:cacheprovider \
  data/expander_code/exp102/tests
conda run -n 12 --no-capture-output pytest -q -p no:cacheprovider \
  data/expander_code/exp101/tests
PYTHONPATH=src conda run -n 12 --no-capture-output pytest -q \
  -p no:cacheprovider tests
```

The global-specific suite covers both exact HGP oracles, zero/nonzero syndromes,
all frozen p values, full transition matrices, reference/Numba transcripts,
k=64 masks, WMC, raw replay/tampering, remote evidence, schedule/control
freezes, parallel analyzer replay, TI comparison, and readiness.

## Remote decision tree

1. Build a source package from a clean committed HEAD with the existing
   `002_numba_smoke_20260719/build_source_package.py`.
2. Copy that deployment to `~/.single_shot/repos/<run>/` on nd-0, freeze
   `GLOBAL_72H_SCHEDULE.json` immediately, and never replace it.
3. Run `orchestrate_global_preflight.py`; require all three Linux test suites,
   runtime consensus, digest consensus, and the bounded WMC report.
4. Run every runtime-eligible bias and measurement candidate on
   `HARD2+EASY3`, verify remote evidence, and analyze all raw.
5. Before hour 20, freeze selection, postselection plan, bias/TI controls, and
   templates. No result-driven method or resource changes are legal afterward.
6. Run selected hard and defect methods at T and 2T on HARD2. Stop the full
   range unless both mechanisms and every comparison pass.
7. If HARD2 passes, run CONF17+GAP8+SMALL6 at 2T, RES6 at fresh T, and the m3
   full-sector TI anchors.
8. Retrieve and hash all evidence, then run the local analyzer with an explicit
   `--num-workers N` and the complete `--run-root/--ownership/--deployment-root/
   --schedule` tuple. Finalize only from reports bound to the frozen controls.

Failure cannot be repaired in place: do not delete FAILED markers, add samples,
extend a tier, change bias tuning, weaken a gate, or reuse a partial deployment.
The fail-closed full-range result is
`UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`, not `IMPOSSIBLE`.
