# Logical-signature V0v2 transport screen

This directory freezes the successor to the terminal V0v1 artifact-portability
conflict.  The diagnostic contract is
`exp102.q0_logical_stratified.v0.v2`; its only configuration is
[`q0_logical_stratified.v0.v2.json`](../../config/q0_logical_stratified.v0.v2.json).
It is a narrow test of whether a frozen label-stratified independence-MH
proposal visibly transports every logical direction on one hard sentinel:

```text
m08_c06, p=.04, d00, attempt022
tau=.5 and 1.0; P/U/L starts; 8 trajectories per family
burn=512, measurement=4096
```

It estimates neither `q_top` nor any physical result.  Its only possible
positive status is `LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN`; that status is
permission to design a fresh, independent HARD2 confirmation, not
`READY_FOR_FORMAL`, tuning, held-out, or production authorization.

## Why V0v2 exists

V0v1 stopped before a sampler trajectory when macmini and `nd-1` independently
constructed different, internally valid BpLSD/MILP proposal artifacts.  Solver
optimality is not part of the MH target or Hastings ratio, so V0v2 freezes one
valid proposal instead of treating cross-version solver output as an invariant.

`nd-1` is the sole artifact producer.  It creates the complete decoder
transcript, rank-complete catalog, tail schedule, and both proposal artifacts
exactly once.  `nd-2`, `nd-3`, and macmini never regenerate them.  They load
the producer bytes and independently verify their hashes, hard-coset algebra,
logical labels, transcript selection, proposal normalization, and artifact
identity.  A regenerated artifact, altered authority, altered transcript, or
any changed discrete sampler decision is `CONFLICT`.

Floating proposal densities may differ by a few ULP across supported NumPy/libm
stacks.  Portable comparison therefore permits bounded float replay only after
requiring the complete discrete state/decision trace to agree exactly.  The
preflight digest excludes floats, while any acceptance, label, state, seed,
identity, or transcript difference remains fatal.

## Initialization and gate

`P` starts at the planted legal hard-coset state.  `U` starts from an exact
K=0 uniform hard-coset draw.  `L` starts from a deterministic legal decoded
tail state outside the retained catalog.  The physical all-zero string is not
an admissible replacement: this sentinel has nonzero syndrome; in shifted
coordinates it is already the planted `P` start.  Collapsing to a common start
would hide initialization memory rather than establish convergence.

For each `(tau, P/U/L)` family, the measurement phase alone must show all of:

| Requirement | Frozen threshold |
|---|---:|
| Accepted cross-label changes | at least 128 |
| Chains with at least 8 measurement cross-label changes | at least 6 of 8 |
| Distinct accepted catalog anchors | at least 16 |
| GF(2) rank of accepted measurement label deltas | exactly 64 |
| Basis characters with a measurement leave-return | all 64 |
| Frozen nonbasis characters with a measurement leave-return | all 64 |

Burn-only moves, total acceptance, same-sector state changes, catalog rank and
proposal importance-sampling ESS are recorded diagnostics only.  They do not
substitute for the gate.  The full-rank and character checks deliberately
reject a proposal that only moves in a small logical subgroup.

## Immutable execution order

Every remote command must run through `run_verified_source.sh`, use conda
environment `11`, and be launched in `screen`.  Do not run Python directly in
`repos/<run>/source/`; do not retry a failed stage in place.

1. Create a fresh deployment and fresh `runs/<run_id>` tree.  On `nd-1`, run
   `prepare-artifacts`, then `build-manifest`, through `run_v0_stage.sh`.
2. Run `audit-artifacts` on `nd-1`, `nd-2`, and `nd-3`; pull the immutable
   run tree to macmini and run the same audit locally under conda `12`.
   All four `artifact_audit_sha256` values must match exactly.
3. Run the fixed probe `preflight` on the same four hosts.  All
   `portable_discrete_raw_sha256` rows must match exactly; each result binds
   the matching artifact-audit SHA.
4. Only after those checks, run the 24 fixed tasks on `nd-2` and the other 24
   on `nd-3`.  The manifest alternates trajectory indices so each `(tau,
   family)` contributes four trajectories to each node; node ownership and
   resource tier are immutable.
5. Replay every raw on its producing Linux host, pull the complete run tree,
   and run local portable analysis with `--portable`.  No extra trajectory,
   longer chain, alternate start, or weaker gate is legal after seeing data.

The `audit` command prints canonical JSON but creates no raw data.  The stage
wrapper stores its exact log hash in its immutable `SUCCESS` marker.  It
accepts only the V0v2 module/action pair and chains each stage to the required
prior `SUCCESS` marker.
