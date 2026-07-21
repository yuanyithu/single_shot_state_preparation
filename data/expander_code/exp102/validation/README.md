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
- Held-out and production evidence do not exist. Their absence is an active production blocker.
