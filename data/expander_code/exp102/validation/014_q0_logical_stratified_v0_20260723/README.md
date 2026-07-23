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
