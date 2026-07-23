# Logical-signature V0 transport screen

This directory contains the deployment wrapper for the isolated
`exp102.q0_logical_stratified.v0.v1` diagnostic.  It has no formal experimental
authority.  The screen generates a complete immutable BpLSD candidate
transcript on `nd-1`, validates the resulting artifacts on all Linux nodes,
then runs only `m08_c06, p=.04, d00` under the two frozen LSI-IMH proposal
temperatures and `P/U/L` starts.

Every command is executed from `run_verified_source.sh`; this wrapper only
adds exclusive stage markers and logs.  A V0 report can say at most
`LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN`, which is evidence for a later
fresh HARD2 comparison, not convergence or authorization for exp102.

The frozen manifest is always `control/V0_MANIFEST.json` and locates the
immutable artifacts through `../artifacts`, never an absolute host path.
Consequently a pulled run tree can be replayed unchanged on macmini; an
altered layout or artifact digest is rejected before any raw is accepted.

## Execution result: artifact portability conflict (2026-07-23)

The immutable clean-source attempt
`exp102_q0_lsi_v0_20260723_b9a08a4` stopped after its successful artifact
stage and before `control/V0_MANIFEST.json`, Linux preflight, or any V0
trajectory. The run used source
`b9a08a4905e4c8e999e0c9e5b3408f20e83c4436`, archive SHA256
`a53515a6af914077303b040caa6d3b5046af0054cf8bc3683c10289e1548ae53`,
and source-manifest SHA256
`f151754e619f233e8abd544ea4a5d1bb6ec58cfc6c7f999866bd08680e0712a0`.

Its required macmini-to-nd-1 artifact equality gate failed. The macmini
artifact manifest SHA is
`f90fc8d23be45e7b5122424e96fe5d6769aa73cf20339dcc0e6da814db67e64f`, while
nd-1's is
`6171de3b81a6f84ba070ba62fb7c52620687284c860d0a0bc9513b8a51d74b98`.
The immutable code, codebook, syndrome and V0 scalars agree, but the builder
environments differ: macmini used `ldpc/numpy/scipy=2.4.1/2.4.1/1.17.0` and
nd-1 used `2.3.7/2.3.4/1.16.3`. Both decoded all 113566 candidates, but their
base anchors and 112866 decoded states differ, producing different catalogs,
tail schedules and proposal identities. The pulled nd-1 artifact passes the
local exact GF(2) checks on its retained states, so the outcome is a strict
cross-environment identity conflict rather than corrupt raw or MCMC evidence.

This attempt has terminal status `CONFLICT_CROSS_ENV_ARTIFACT_IDENTITY`.
No sampler output exists and nothing from this run can support a transport,
convergence, physics, readiness, held-out or production claim. Evidence is
kept in the ignored `remote_run/exp102_q0_lsi_v0_20260723_b9a08a4/` tree. Do
not retry it in place. Any successor must first review and freeze a portable
artifact policy with a pinned builder stack or a single authoritative builder
plus algebraic-only cross-platform verification.
