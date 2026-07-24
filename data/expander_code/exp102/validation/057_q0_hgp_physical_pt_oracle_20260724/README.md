# 057 q=0 collapsed physical-p PT oracle and resource audit

Status: `LOCAL_SMOKE_RUNTIME_ELIGIBLE / T1 PAIR PENDING`.

This local-only validation checks the mathematical CPPT implementation and the
previously missed m8 mass-table scale before deciding whether the method is
worth a separate sampling contract.  See `PRE_RUN_RED_TEAM.md` for the target,
initialization requirements, diagnostic limitations, resource gate, and
authority boundary.

The maximum possible status is `LOCAL_SAME_FAMILY_ORACLE_PASS`.  CPPT remains
a collapsed-B tempering method and cannot serve as the mechanism-independent
confirmation required for formal Exp102 readiness.

The frozen CPPT32 m8 smoke from source `8ffb48f540285f4000cd7307d2f5b8adfb406c91`
passed only its runtime gate.  Its 32-rung read-only log-mass artifact was built
once in `10.38s`; two 40-round trajectories project to a worst T1 time of
`474.99s`, below the `7200s` cap.  The short P trajectory had mean cold weight
`62.28`, while U remained at `207.09`; neither completed a round trip.  Those
short-clock values are deliberately not a mixing decision.  Report SHA256 is
`27e3341ca0c1d9bed64dc9e64a4874bd1455e97c62e1be9b64b7ac4bdf926346`.

The next frozen step is one local full-T1 P/U necessary-condition pair.  It
must pass the existing per-trajectory HP swap/round-trip gates plus deliberately
loose single-chain P/U distribution diagnostics before any replicated or
remote screen is designed.  A pass can authorize only a fresh replicated
diagnostic; a failure terminates CPPT without using nd-2/nd-3.
