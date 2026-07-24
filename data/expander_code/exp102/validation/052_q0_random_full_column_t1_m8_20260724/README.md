# Random-full-column T1 m8 diagnostic

This directory freezes the outcome-free initialization geometry, control
artifact, schedule/preflight/measurement workflow, and local analyzer for
`exp102.q0_random_full_column.t1_m8.v0`.

No T1 measurement raw exists yet.  Measurement is fail-closed behind exact
three-node digest consensus and the replay-inclusive two-hour trajectory
runtime gate.  See `RANDOM_FULL_COLUMN_T1_CONTRACT.md` for the scientific
contract and result permissions.

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
