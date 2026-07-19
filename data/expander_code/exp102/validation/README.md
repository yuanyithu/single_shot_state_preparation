# exp102 validation index

Current status is pre-pilot, not scientific certification.

- `001_local_implementation_20260719/`: registry cardinality/dimensions, task-plan identity,
  exp102 unit tests, and selected exp101 regression tests (83 combined PASS after the Numba update).
- `002_numba_smoke_20260719/`: local nonzero-syndrome pilot-cell smoke; diagnostic only and not a
  pilot pass. The deliberately tiny round budget fails mixing gates as expected.
- Pilot tuning, held-out certification, cross-node determinism, and production evidence do not yet
  exist. Their absence is an intentional production blocker.
