# exp102 validation index

Current status is pre-pilot, not scientific certification.

- `001_local_implementation_20260719/`: registry cardinality/dimensions, task-plan identity,
  exp102 unit tests, and selected exp101 regression tests (83 combined PASS after the Numba update).
- `002_numba_smoke_20260719/`: local nonzero-syndrome pilot-cell smoke; diagnostic only and not a
  pilot pass. The deliberately tiny round budget fails mixing gates as expected.
- `003_numba_engine_20260720/`: full-round Numba/reference bit-identity and performance benchmark,
  including the `m=8,k=64` boundary. Local speedup is about 177x--196x.
- `004_pilot_ladder_20260720/`: clean-SHA three-node preflight and the complete configured ladder
  search. The maximum candidate fails m=4..8, so the pilot is fail-closed before gamma/held-out.
- `005_pt_v2_discovery_20260720/`: isolated Q32-ladder/multi-swap discovery implementation,
  cross-node digest runner, immutable ownership launcher, and discovery-only raw validation. The
  three-node screen completed, but all 12 transport candidates produced zero certified round
  trips; the frozen route is `EXHAUSTED` before S128/confirmation. Report-v3 also binds every raw
  file to its source/control/ownership/marker evidence.
- `006_pa_discovery_20260721/`: reviewed successor search using fixed-schedule q=0 population
  annealing plus a no-extra-randomness replay of 16 old PT trajectories. The implementation,
  frozen Q32 schedules/panels, exact oracles, reference/Numba identity, raw analyzer, immutable
  node ownership, marker verification, and runtime tools are certified. The clean-source Linux
  runtime gate and all 4 autopsy plus 64 hard-screen tasks completed. Autopsy is four times
  `INCONCLUSIVE` because conditioned attempts are insufficient; all four PA methods fail both hard
  cells through catastrophic genealogy collapse. The frozen zero-pass branch is `EXHAUSTED`, so
  rescue and blinded confirmation were not run and this is not `READY_FOR_FORMAL`.
- `007_q0_global_discovery_20260721/`: isolated global-sampling successor implementation. It
  includes deterministic logical catalogs, hard-coset cluster/joint heatbath kernels, independent
  defect trace, exact HGP/WMC oracles, no-pickle raw replay, character/D2 gates, m3 full-sector TI
  anchors, three-node preflight/runtime/digest consensus, immutable stage ownership, 72-hour
  schedule, postselection/control freezes, and fail-closed readiness. Local implementation tests
  pass; three immutable preflight attempts are audited below. The final attempt stopped at the
  frozen runtime gate before sampler work. No discovery stage completed, so this directory
  contains no physics result.
- `008_q0_global_preflight_portability_20260721/`: immutable evidence from the failed first global
  preflight plus its repair. It fixes archive git provenance, deterministic legacy-section testing,
  source-tree writes by the spec example test, cold-JIT TI projection, and post-worker source
  reverification. A fresh commit/deployment/run/schedule is required; this is not a sampler result.
- `009_q0_global_runtime_gate_separation_20260721/`: immutable evidence from the failed second
  preflight. A live performance fixture fluctuated at the TI wall boundary and incorrectly failed
  the deterministic regression suite before the persisted three-node runtime gate. Tests now
  validate live-report self-consistency while the unchanged dedicated consensus alone decides
  machine eligibility. The same-node postmortem passed T3; no sampler task ran.
- `010_q0_global_runtime_exhausted_20260721/`: immutable evidence from the third clean preflight.
  All node workers and canonical digests passed and all hard/defect methods fit T3, but the
  required full-sector-TI contingency exceeded its frozen 79,200-second window on nd-2 and nd-3.
  The discovery therefore closes before screen as
  `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`; no sampler raw or physics result exists. The audit
  repair persists this legal aggregate terminal status without weakening any downstream gate.
- `011_q0_global_screen_diagnostic_20260721/`: isolated HARD2+EASY3 screen authorized after the
  full discovery runtime stop. The first immutable run passed preflight and all 15 bias tasks but
  closed before all 1280 measurement trajectories because platform fractional powers generated
  two gamma bit patterns. Its metadata-only conflict evidence is frozen; the repaired source uses
  exact `3/5` Decimal arithmetic, a self-checked full-schedule SHA, and digest v2. A fresh remote
  run is required, and the strongest possible result remains diagnostic rather than formal.
- Held-out and production evidence do not exist. Their absence is an active production blocker.
