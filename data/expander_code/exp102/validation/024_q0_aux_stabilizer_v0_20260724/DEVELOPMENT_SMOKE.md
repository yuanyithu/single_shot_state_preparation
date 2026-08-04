# UASRE Development Smoke Record

These directories are development diagnostics only.  None is an immutable
local viability result, may be pooled, or may be used for posterior estimates,
remote deployment, or formal authorization.

- `development_manifest_smoke_001` checked initial manifest construction before
  the complete dependency binding was added.  It contains no trajectory raw.
- `development_manifest_smoke_002` produced one UASRE32 exact-K0-U trajectory.
  Its subsequent raw validator exposed a missing complete-score import in the
  runner.  That raw is retained only as a defect-reproduction artifact.
- `development_manifest_smoke_003` was generated after the import fix.  One
  UASRE32 exact-K0-U trajectory passed the runner's full deterministic replay
  and the raw-only audit's algebra, trace, score, and counter checks.  It was
  generated before the source binding was expanded to its complete transitive
  module set, so it is also diagnostic only.

The final `local_hard_viability/` manifest may be prepared only after the
source/config/audit set is stable.  Its 48 raw trajectories require a fresh
run, complete replay, and independent audit.
