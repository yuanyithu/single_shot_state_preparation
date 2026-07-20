# exp102 validation index

Current status is pre-pilot, not scientific certification.

- `001_local_implementation_20260719/`: registry cardinality/dimensions, task-plan identity,
  exp102 unit tests, and selected exp101 regression tests (83 combined PASS after the Numba update).
- `002_numba_smoke_20260719/`: local nonzero-syndrome pilot-cell smoke; diagnostic only and not a
  pilot pass. The deliberately tiny round budget fails mixing gates as expected.
- `003_numba_engine_20260720/`: full-round Numba/reference bit-identity and performance benchmark,
  including the `m=8,k=64` boundary. Local speedup is about 177x--196x.
- The old-source cross-node digest passed, but pilot tuning must restart and cross-node determinism
  must be repeated for the new clean SHA. Held-out and production evidence do not yet exist.
