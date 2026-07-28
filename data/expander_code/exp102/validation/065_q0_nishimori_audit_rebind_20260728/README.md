# 065 q=0 Nishimori audit rebind

Status before the one-shot run: `SOURCE FROZEN / CONFLICT PERSISTENCE NOT RUN`.

This validation repairs only the evidence interface exposed by validation 063.
It does **not** edit or rerun the immutable exact report.  The first mismatch
observed by the old fail-fast auditor was between two English failure prefixes;
that auditor stopped before establishing full numerical agreement.  Validation
065 is the fresh test of that full agreement, not a retrospective assumption
that it already passed.  Its focused development test exposed a second,
numerical mismatch which the old fail-fast auditor never reached.

The mismatch is confined to the truth-blind MAP-delta control.  Three hard
cosets have mathematically tied logical-sector weight enumerators, while the
runner and oracle floating-point paths select different tied labels with
`argmax`.  This changes 11 reported fields (maximum absolute difference
`.0340070400000001`) even though both choices are legal and the 14 terminal
failure identities are unchanged.

`audit_rebind.py` verifies the exact input bytes, report self-hash, original
source commit, source-tree binding, and every source blob named by the original
config.  It then invokes the frozen 063 independent oracle to rebuild all six
golden rows, 30 exact-control rows, 30 power rows, three chain controls, and the
terminal gate.  Failure decisions are rebuilt directly from the numeric gates
as structured identities.  The two frozen legacy prefixes
`equivalence gate failed` and `equivalence power failed` are aliases only at
this final translation boundary; neither prefix determines the decision.

The one-shot result is therefore expected to be
`CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS`, not an audit
pass.  It persists all 11 paths and the three syndrome/label/tie witnesses,
with `terminal_gate_invariant=true` and `full_payload_match=false`.

`verify_audit_rebind.py` is a separate one-shot verifier.  It does not import
the rebind runner.  It rechecks source and input identities, reruns the frozen
independent oracle, independently reproduces the conflict and structured
terminal failures, and writes its own self-hashed verification record.

The underlying scientific result remains
`NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`: 14 correct-posterior groups at
the frozen `N=2048` point miss the `.01` simultaneous-equivalence precision
target.  The invariant terminal gate does not cure the full-payload conflict
and cannot be reported as an audit pass.  Validation 065 does not upgrade the
identity to a q_top error bound and grants no sampler, remote, formal, held-out,
production, posterior-estimation, or confirmation authority.

The intended sequence from a clean committed source is:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/065_q0_nishimori_audit_rebind_20260728/audit_rebind.py
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/065_q0_nishimori_audit_rebind_20260728/verify_audit_rebind.py
```

Both commands are create-once.  Existing output is a hard stop; old output is
never deleted to obtain a second attempt.
