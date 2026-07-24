# Random-full-column T1 m8 diagnostic

Terminal status: `RUNTIME_EXHAUSTED` before measurement.

The immutable remote run `exp102_q0_rfcg_t1_m8_20260724_6fa489f` used source
`6fa489f838dffea15b07e1ef3b3fbee3951dd3c0`.  All three Linux nodes agreed
exactly on the mass table and four portable transition transcripts, so the
aggregate preflight has `exact_consensus=true`.  The replay-inclusive
factor-two projections were `24701.46884727478`, `24812.0581073761`, and
`29871.419471740723` seconds per trajectory on nd-1/nd-2/nd-3, all above the
frozen 7200-second cap.  The workflow therefore created no measurement raw.

The independent local audit verifies the canonical JSON and every self-hash,
the frozen control content, 40-task/five-family schedule and 14/13/13
ownership, seed uniqueness, stage markers, node projections, cross-node
consensus, aggregate decision, and absence of measurement files.  It records
`INDEPENDENT_PREFLIGHT_AUDIT_PASS_RUNTIME_EXHAUSTED_CONFIRMED` with audit SHA
`817425dbaa6a9e5d90d03d34efe16f957beb7424eddd27dcde7cf12d60d75c6d` and
evidence-package SHA
`1d4ec020e65a654aba21ecbe910f424b41401f4f30821cc2604310a022de0506`.

This is a runtime failure of this frozen implementation/configuration, not a
sampler convergence result, a physical parameter-point result, or an
impossibility claim.  It grants no m6, HARD2, formal, held-out, or production
authority.  Changing the clock, concurrency, cap, or implementation requires
a separately reviewed fresh contract rather than continuation of this run.

This directory freezes the outcome-free initialization geometry, control
artifact, schedule/preflight/measurement workflow, and local analyzer for
`exp102.q0_random_full_column.t1_m8.v0`.

No T1 measurement raw exists.  The fail-closed two-hour trajectory gate
stopped the run after exact three-node preflight.  See
`RANDOM_FULL_COLUMN_T1_CONTRACT.md` for the scientific contract and result
permissions.

Frozen control identities:

```text
config SHA256:
952c65491883423b21e4c51015d167b56489f33b99b369ff0dfdebd2db5c0a85
control content SHA256:
b99fb047e787fd999cde113bd3c64a1e9ef0e41e805d79a3d6d5f7995b6b8df6
control file SHA256:
a43865186be0865ba8f1eac35ec22354ebe92ea6528091ce32e6f6dcaa118a41
control manifest SHA256:
336d3e24a0f65970d4fcaa24de7f292798aedd89a6dcb548a06a37c73afb33cc
logical character SHA256:
b1114958ca49faff773d9794abc7ff4bcb9ef695958fa4b741444bf57aca3518
B character SHA256:
d0494bba1d04527c54b9b5222d5c985ed5650dee0b9c6fafc1b6f8c45604a411
```

The local four-worker contention smoke projected above the remote gate.  This
does not authorize or reject the method: the frozen contract assigns runtime
authority only to the nd-1/nd-2/nd-3 aggregate preflight.
