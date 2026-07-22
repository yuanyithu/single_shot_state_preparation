# 013 q=0 HGP global sampler screen

## Current status

`PRE-RUN / DIAGNOSTIC TEST AUTHORIZED`

This directory is the evidence root for the fresh
`exp102.q0_hgp_global.screen.v1` server screen. At creation time no 013 remote
raw, control, preflight consensus, report or terminal decision exists. The
source must first be completed, tested, committed and packaged from a clean Git
archive; this README must not be read as evidence that a server job is already
running or that a sampler has passed.

One infrastructure-only launch from source
`7654bcced23688705f396695370661199b81648a` stopped in the outer launcher
because `nd-0` has no `screen`. It created no run root, orchestrator log,
stage marker or raw and launched no preflight/sampler. That source is not
retried in place; the guarded `nohup` + `setsid` launcher below requires a
fresh deployment/run.

Authority is limited to `DIAGNOSTIC_HARD_PAIR_FOUND`. This run cannot produce
`READY_FOR_FORMAL`, `FROZEN_HELD_OUT_PASS`, publication data or any of the 6144
production tasks.

## Frozen inputs

- Contract: `exp102.q0_hgp_global.screen.v1`.
- Contract document: `../../HGP_GLOBAL_SCREEN_CONTRACT.md`.
- Config: `../../config/q0_hgp_global.screen.v1.json`.
- Canonical config SHA256:
  `3c65ef96ce231b4aea4499b5a6030f1cc82475117c5ee5ecc7633d972ef8edc9`.
- Registry SHA256:
  `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`.
- Candidates: `HP32`, `HP64`, `MAM-IMH8`.
- HP panel: frozen `HARD2+EASY3`, five cells in contract order.
- MAP/IS panel: frozen `HARD2` only.
- Replication: `P/U` each 16 trajectories per scheduled method/cell.
- Measurement control size: 384 fresh tasks (320 HP, 64 MAP).
- Resource choices: only `T1/T2/T3`; one common tier selected from runtime and
  the frozen execution topology without reading physics results. This contract
  has no formal peak-memory selection gate; read-only inspection before launch
  found roughly 251/251/503 GiB on nd-1/nd-2/nd-3.
- Preflight: `nd-1`, `nd-2`, `nd-3`.
- Execution after frozen ownership: `nd-2`, `nd-3` only.
- Final full replay: frozen on `nd-3` with 91 workers.

The five disorder-uniform seeds are intentionally the earlier frozen values so
the physical cells are unchanged. All sampler, anchor, character and auxiliary
IS streams use new `q0_hgp_global_screen_*_v1` namespaces. No raw or seed from
005--012 is reusable.

## What counts as a pass

One HP method must pass all five cells from both initial families, with every
individual trajectory satisfying its swap and cold-hot-cold transport gate.
`MAM-IMH8` must independently pass both `HARD2` cells, with every individual
trajectory accepting during burn and satisfying its measurement acceptance
gate. The selected HP and MAM-IMH8 distributions must then agree on `HARD2`
under the frozen character `q_top`, `D2_norm` and normalized-weight gates.
`EASY3` is an HP runtime/false-negative control and has no independent-method
certification claim.

Both mechanisms also face the same collapsed-`B` slow-variable gate. Raw binds
all `r^2` B-bit characters, row/column parities and 64 deterministic dense
masks. Every character gets split-Rhat; `L(B)`, `|B|`, row/column parities and
nondegenerate dense masks get ESS, at least 48 dense masks must be informative,
and constant `L(B)`/`|B|` chains fail. P/U and cross-mechanism comparisons also
cover B-character mean-square difference, every individual B-character mean
difference, normalized B weight and likelihood. Each character must meet both
the absolute `.04` and `3 SE + .005` limits, so one frozen bit cannot disappear
inside a catalog average. For a constant B character, every initially opposite
chain must reach the common sign by the burn endpoint.

The 50000-draw importance-sampling calculation is proposal-overlap diagnostics
only. It cannot pass a method, select HP32 versus HP64, replace MCMC samples or
enter a physical estimate.

## Prior feasibility evidence (not reusable)

The local 012 HP32 probe was the first result to remove the planted/uniform
initialization split on both known hard cells and to show strict replica round
trips. It used only 8 trajectories per family, a provisional source identity
and a shorter schedule, so it justifies this screen but does not satisfy it.
The MAP-mixture overlap probe likewise suggested viable independent-MH
acceptance but is not a frozen-chain convergence result.

Follow-up local probes showed that extending MAP anchors on m3 was the wrong
optimization target: deterministic weight-23/25 near-MAP shells still gave the
planted chain zero accepted T1 moves, both weight-24 solves hit 180 seconds, and
the m5 primary solve also exceeded 180 seconds. Those shells are excluded. The
server run does not spend 96 additional MAP trajectories on EASY3 tasks already
known to be unsuitable.

The physical all-zero bit string is not a generic alternative initialization:
for nonzero syndrome it lies outside the q=0 hard coset. `P` and exact-uniform
`U` are intentionally opposing valid starts used to expose initialization
memory. The pass criteria are finite-budget diagnostics, not a proof of mixing;
one sentinel disorder cannot certify or rule out an entire `(m,p)` point.

## Required evidence sequence

1. Build a clean source archive and bind its commit/archive/manifest SHAs.
2. Run `launch_hgp_orchestrator.sh ... preflight` on `nd-0`. Because `nd-0`
   has no `screen`, this outer control process uses fixed `nohup` + `setsid`
   with an atomic phase guard and PID metadata. It still launches every
   schedule/artifact/preflight job on `nd-1`/`nd-2`/`nd-3` in an independent
   `screen` through the verified archive and immutable stage wrapper.
3. Fail closed on any source, digest, solver, transcript or reference/Numba
   difference; otherwise select the largest common resource tier that fits.
4. Stop at the aggregate preflight boundary, pull the frozen artifacts and
   preflight evidence to the macmini, and replay them under conda `12`. The
   stored remote solver identity is provenance; local replay checks exact
   GF(2)/hash/proposal content without rerunning MILP under the local version.
   `local_preflight_audit.py` writes a self-hashed attestation; an exact
   mismatch is fail-closed and does not invent a float tolerance.
5. Only after that audit, run `launch_hgp_orchestrator.sh ... measurement`
   from the same verified source before the control-freeze deadline, passing
   the canonical, regular (non-symlink) server copy of
   `HGP_LOCAL_PREFLIGHT_ATTESTATION.json` and its file SHA. The detached
   orchestrator revalidates that attestation, its phase-bound launch guard, all
   prior SUCCESS markers, aggregate `PASS`, selected tier, source identity and
   deadlines before materializing the immutable 384-task control.
6. Run fresh nd-2/nd-3 screen tasks without migration, resampling or in-place
   retry, then perform the frozen 91-worker full replay on nd-3.
7. Pull all raw and terminal evidence locally, verify file/canonical hashes and
   independently replay state, label, weight, residual, proposal and transition
   records before accepting the terminal package.

Any failed marker or failed/duplicate `nd-0` launch guard requires a fresh
deployment/run ID. The retained guard is not deleted for an in-place retry.
This directory will be updated with the actual clean source commit, run ID,
selected tier, evidence hashes and terminal result only after those artifacts
exist.
