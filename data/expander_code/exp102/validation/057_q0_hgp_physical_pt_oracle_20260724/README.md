# 057 q=0 collapsed physical-p PT oracle and resource audit

Status: `LOCAL_T1_PAIR_UNRESOLVED / DO NOT DEPLOY`.

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

The frozen full-T1 pair from source
`a90d3f01641f4ce1432f739d7a76cf6f9128885a` completed locally without using
nd-2/nd-3.  Both raw files pass hard-coset, state/B, label, weight, identity,
counter and character replay.  The primary report SHA is
`287d62b5373f50bc867d874dd15511a3e0418f1bf25b5e7c5e60ca2e072de1c1`;
P/U raw SHAs are `e771084a...6a27` and `dada68f8...7f`.

The pair fails both frozen necessary gates.  P/U plug-in q_top diagnostics are
`.900885/.144627` (difference `.756258`), logical-character D2 is `.346827`,
B-character D2 is `.093028`, normalized full-weight difference is `.022927`,
and collapsed log-likelihood differs by `2.50668` per factor.  Both trajectories
complete zero cold-hot-cold round trips; P/U worst adjacent swap rates are
`.00547/.03945`, and only `14/32` origins ever visit the cold rung.  These are
large initialization memories, not a marginal threshold miss.

`audit_t1_pair.py` does not call the CPPT sampler, trajectory runner or primary
analyzer.  It independently validates the two `allow_pickle=False` raws,
recomputes the hard-coset algebra, B/state/label/weight traces, cold likelihood,
all characters, transport counters, gates and terminal decision.  Its status
is `INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED`, audit SHA
`1dd1260d80469ac12f1061a289af540041cffb3a0d3857073299a2effbae0bf0`.

This frozen CPPT32 route stops here: no replicated screen, m6/HARD2, nd-2/nd-3
deployment, q_top estimate, formal, held-out or production work is authorized.
The result does not prove CPPT64 or all replica exchange impossible, but a
CPPT64 successor would require a new reviewed contract and still could not be
the mechanism-independent confirmation missing from HP64.  The next priority
is therefore an orthogonal hard-coset mechanism that directly attacks the B
slow coordinate, not another longer or denser version of this ladder.
