# 063 q=0 Nishimori auxiliary-audit calibration

Status: `NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT /
CONFLICT_INDEPENDENT_AUDIT_MESSAGE_TAXONOMY_MISMATCH / NO SAMPLER RAW`.

This local-only validation calibrates a possible Nishimori identity check as an
**auxiliary diagnostic**. It does not run HP64, estimate any exp102 parameter
point, or repair the unresolved global sampler problem.

The one-shot exact report will independently enumerate the `n=10,k=4` and
`n=13,k=1` HGPs at `p=.04,.10,.25`. The runner and a separately implemented
oracle both rebuild `exp101.physics.v2`, every physical truth state, syndrome
mass, hard-coset posterior, logical distribution, control metric and optimistic
power row. The independent auditor imports the oracle, not the runner or the
exp102 worker.

Calibration is fail-closed. A passing frozen control catalog may receive at
most
`NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS`. A scientific
power or control failure is still persisted, with all failures, as
`NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`; it is not relabeled as an
infrastructure conflict. Identity, hash, nonfinite or implementation conflicts
abort instead of producing scientific evidence.

Even the maximum status has no remote, formal, held-out, production, posterior
estimation or sole-confirmer authority. The identity has known all-character
blind controls and supplies no universal bound on `q_top` error. Future use is
permitted only on a preplanned complete fresh-iid disorder ensemble after every
per-disorder sampler gate passes; a pass-only subset is not auditable.

The config binds the runner, independent oracle, auditor, future-raw validator,
schema, tests and both documents by SHA-256. The one-shot runner also requires a
clean tracked worktree with no Python bytecode. See `PRE_RUN_RED_TEAM.md` for the
scientific contract and failure semantics.

## One-shot result and audit conflict

The exact report from source `26bed0bce1fac040919be8a79103b3e25550bdd1`
contains 30 exact control rows and 14 frozen calibration failures. All 14 are
correct-posterior equivalence failures at `N=2048`; the pre-registered
wrong-temperature, label-permutation and MAP-delta detection controls do not
fail. Its terminal status is therefore
`NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`. This is an auxiliary-gate
precision failure, not a sampler or parameter-point result.

The v2 independent auditor did not complete. It stopped at the first failure
because the report used the prefix `equivalence gate failed` while the
independent oracle used `equivalence power failed` for the same frozen failure
identity. `independent_audit_conflict.json` preserves the exact exception,
hashes and zero-authority boundary. The report remains byte-for-byte unchanged,
and no `independent_audit.json` exists. A fresh successor auditor may rebind and
independently recompute this immutable report, but must not rerun or edit it.
