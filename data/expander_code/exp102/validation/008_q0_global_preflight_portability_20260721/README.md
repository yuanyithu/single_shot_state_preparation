# exp102 q=0 global preflight portability repair

Status: `FIRST FAILURE AUDIT / REPAIR COMPLETE`.  The next immutable attempt
exposed a separate runtime-test separation issue documented in validation 009.
This validation is infrastructure evidence only and contains no sampler
discovery result.

## Failed immutable attempt

- Run: `exp102_q0_global_20260721_6f26fd5`
- Source commit: `6f26fd507a38670711b27eb1c8a8177224488885`
- Archive SHA256:
  `c282453b9c1d149ae98a315189b3a31e1fceae6a3d71d2dcb05c332c21e3ba69`
- Source manifest SHA256:
  `21a1f9354c6dc1d33b206c31a6ce5431385a637c9235ac3f618505e3e2cc4908`
- Frozen schedule file SHA256:
  `441c4b0585072f9524f9c9860c48219d24e2385adb027332c6bee6269234e344`
- Frozen schedule identity SHA256:
  `42576f2c8db6aed251d72737eebc3ce0ffee685f3913d111e0bcea7ae77b4edf`

The three-node preflight failed before runtime/digest/WMC evidence was
accepted.  Its markers remain permanently `FAILED`; it must never be resumed
or repaired in place.

## Causes and repairs

1. Exp101 scan provenance returned `git_worktree_dirty=null` in a verified
   archive without `.git`.  It now binds the verified source commit marker and
   manifest to a known-clean `false` value.
2. A legacy cross-project observable test created two optional BP-LSD decoder
   instances.  Linux ldpc 2.3.7 could choose different valid representatives;
   the test now compares two independent deterministic linear sections.
3. The spec example test wrote two ignored files into the source tree.  Tests
   now pass a pytest temporary output directory, and preflight/stage workers
   reverify the complete source identity after execution.
4. The TI timing probe multiplied one-time cold JIT startup by the full sweep
   ratio.  It now warms once, times steady-state work, and charges startup once
   in the projection.
5. The post-repair local archive test passed all 590 tests but caught Numba
   cache files under `source/` when the verifier received a relative deployment
   path.  The verifier now canonicalizes its root before deriving cache paths;
   that deployment was never uploaded or assigned a schedule.

The captured schedule, gzip-compressed node pytest/orchestrator logs, and
immutable FAILED markers are in `failed_run_evidence/`.  A fresh clean commit,
deployment, run ID, and 72-hour schedule are required after local and
clean-archive regression passes.

## Local repair validation

- exp102: `224 passed`
- exp101: `366 passed, 2 expected deprecation warnings`
- root legacy suite with `PYTHONPATH=src`: `16 passed`
- cold-cache runtime selection/consensus regression: `2 passed`
- compileall, shell syntax, and `git diff --check`: pass
