# 013 q=0 HGP global sampler screen

## Current status

`V1 TERMINAL CONFLICT / V2 PRE-RUN / DIAGNOSTIC TEST AUTHORIZED`

This directory records the failed v1 preflight boundary and is the evidence
root for the fresh `exp102.q0_hgp_global.screen.v2` screen. V2 has not been
launched: no v2 remote preflight, control, measurement raw, report or terminal
decision exists. Implementation and local tests are not evidence that a server
job is running or that either sampler has passed.

The terminal v1 run was
`exp102_q0_hgp_screen_20260722_2e6ba2a`, source
`2e6ba2a864d7db6ae04e79867d1678dbcfe42580`, archive SHA256
`6c5aae08c43a196426c27c41d58e6ad5f6a6f94cc8e494519641794ccf99c5e4`
and source-manifest SHA256
`42ceb8b8619cf5de1d9dc0de16fac31a4f88165d1f6cbeaa038d63accf1046cb`.
Its nd-1/nd-2/nd-3 preflight workers and aggregate report all reached `PASS`,
selecting T3. The mandatory macOS audit first exposed a verifier bug that
treated stored remote solver provenance as a local NumPy/SciPy version
requirement. After that provenance issue was isolated, continued audit found a
one-ULP MAM `log_q`/acceptance difference and full-digest drift in both
50000-draw IS probes. Those drifts, not the solver-provenance verifier bug,
make v1 terminal `CONFLICT`. It produced no
measurement control, measurement raw or result and must not be resumed in
place.

Two still earlier attempts remain infrastructure-only history. Source
`7654bcced23688705f396695370661199b81648a` stopped before a run root because
`nd-0` has no `screen`. Source
`df97fb5a7d38543beb515444b5692427dd28cc41` persisted the guarded nd-0
orchestrator but hit the RHEL Bash 4.2 empty-array rule before a run root.
Neither produced preflight or sampler raw. The later v1 run above did reach
preflight PASS; its local `CONFLICT` supersedes the earlier statement that 013
had no preflight evidence.

The schedule remains `exp102.q0_hgp_global.screen.schedule.v2`: nd-0 freezes a
single `CLOCK_BOOTTIME + boot_id` authority because compute-node epochs are
unsynchronized. Measurement must reuse the preflight anchor. Before every
stage launch and SUCCESS acceptance, equality with a deadline or a boot-ID
change fails closed.

Authority is limited to `DIAGNOSTIC_HARD_PAIR_FOUND`. Even a complete v2 pass
cannot produce `READY_FOR_FORMAL`, `FROZEN_HELD_OUT_PASS`, publication data or
any of the 6144 production tasks.

## Frozen inputs

- Contract: `exp102.q0_hgp_global.screen.v2`.
- Contract document: `../../HGP_GLOBAL_SCREEN_CONTRACT.md`.
- Config: `../../config/q0_hgp_global.screen.v2.json`.
- Canonical config SHA256:
  `38092ec030f6c283f163c0ddb3eed612aa850c76ce34f130520522646fa883dc`.
- Registry SHA256:
  `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`.
- Candidates: `HP32`, `HP64`, `MAM-IMH8`.
- HP panel: frozen `HARD2+EASY3`, five cells in contract order.
- MAP/IS panel: frozen `HARD2` only.
- MAP anchor catalog: `exp102.q0_map_mixture.anchors.v3`.
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
the physical cells are unchanged. All v2 sampler and auxiliary draws use fresh
`q0_hgp_global_screen_*_v2` namespaces; no v1 trajectory, preflight probe, IS
draw or sampler seed stream is reusable. Unchanged character catalogs and
deterministic anchor tie-breaks identify frozen mathematical objects, not
reused sampler draws, and are rebound through the v2 config and anchor-v3
hashes. Preflight digest, runtime warmup/timed and preflight/runtime IS streams
remain disjoint from all 384 measurement tasks and the two frozen screen IS
streams. The complete trajectory/IS 63-bit seed enumeration allows only the
intentional reference/Numba reuse of one identical tiny-oracle stream and is
repeated with the final clean commit/archive/manifest identities before launch.

V2 preflight additionally freezes four MAM portability probes: both `HARD2`
cells from both `P` and `U`, each with `burn=256` and `measurement=2048`.
Across Linux they belong to the complete full transcript. The local portable
projection exactly checks their proposal identities, uniforms, acceptance and
state-change decisions, and resulting states. No ULP, absolute, relative or
rounding tolerance is permitted.

HP does not expose every internal categorical uniform and decision. Its
portable claim is therefore limited to exact fixed-clock discrete outputs and
transport counters. Neither this README nor a future report may upgrade that
claim to bitwise replay of every hidden HP decision.

## What counts as a pass

One HP method must pass all five cells from both initial families, with every
individual trajectory satisfying its swap and cold-hot-cold transport gate.
`MAM-IMH8` must independently pass both `HARD2` cells, with every individual
trajectory changing state during burn and satisfying its measurement
state-change count/rate gates. Accepted self-proposals remain recorded as MH
diagnostics but do not count as transport. The selected HP and MAM-IMH8
distributions must then agree on `HARD2`
under the frozen character `q_top`, `D2_norm` and normalized-weight gates.
`EASY3` is an HP runtime/false-negative control and has no independent-method
certification claim.

All `Delta q_top` uncertainty calculations are paired on the common frozen
character masks. The character finite-population SE is taken from the
per-character contrast itself, and the two independent trajectory ensembles
contribute side-specific delete-one jackknife variances. Marginal character
SEs are not added as if the masks had been sampled twice.

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

These gates do not make finite diagnostics into a proof. The frozen logical/B
characters may miss a higher-order mode, 16 `U` starts may miss a basin with
tiny K=0 volume but material posterior mass, and HP/MAM may share that failure
mode. B is a targeted collapsed bottleneck, not an assumption that no other
slow variable exists; the full logical, energy, weight, transport and
cross-method gates remain mandatory. This residual risk is one reason the
maximum authority stays diagnostic-only.

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
   The 6/8/22/24-hour deadlines derive only from the schedule-v2 nd-0
   `CLOCK_BOOTTIME` anchor; node-local epoch values cannot grant extra time or
   cause a false expiry.
3. Require full-Linux exact consensus: all three nodes must agree byte-for-byte
   on the complete full and portable payloads, including the frozen floating
   transcript. Fail closed on any source, artifact, field-manifest, digest,
   discrete transcript or reference/Numba difference. Only then select the
   largest common resource tier that fits safeguarded generation from the
   worst-case hour-8 control origin by hour 22 and safeguarded analysis by
   hour 24.
4. Stop at the aggregate preflight boundary, pull the frozen artifacts and
   preflight evidence to the macmini, and replay them under conda `12`. The
   stored remote solver identity is provenance; local replay checks exact
   GF(2)/hash/proposal content without rerunning MILP under the local version.
   `local_preflight_audit.py` rebuilds both projections and the four MAM
   acceptance-decision digests. It may attest `PASS_EXACT` if the full payload
   also agrees, or `PORTABLE_PASS` if every declared portable field and MAM
   decision agrees after remote full consensus. A full mismatch remains
   recorded only at paths exhaustively declared nonportable. There is no ULP
   or other numerical tolerance, and a portable mismatch is `CONFLICT`.
5. Only after that audit, run `launch_hgp_orchestrator.sh ... measurement`
   from the same verified source before the control-freeze deadline, passing
   the canonical, regular (non-symlink) server copy of
   `HGP_LOCAL_PREFLIGHT_ATTESTATION.json` and its file SHA. The detached
   orchestrator revalidates that attestation, its phase-bound launch guard, all
   prior SUCCESS markers, aggregate `PASS`, selected tier, source identity and
   deadlines before materializing the immutable 384-task control.
6. Run fresh nd-2/nd-3 screen tasks without migration, resampling or in-place
   retry, then perform the frozen 91-worker full replay on nd-3.
7. Pull all raw and terminal evidence locally. Run the independent analyzer and
   `local_terminal_audit.py` to verify the joint nd-0 terminal, file/canonical
   hashes, algebra and stored full/portable/decision evidence for every raw.
   The terminal package and measurement acceptance manifest are a joint
   terminal: validation must reconstruct the exact 384-trajectory plus two-IS
   raw/claim set from the frozen control and reject any missing, extra,
   symlinked or hash-mismatched file.

Any failed marker or failed/duplicate `nd-0` launch guard requires a fresh
deployment/run ID. The retained guard is not deleted for an in-place retry.
This directory will be updated with the actual clean source commit, run ID,
selected tier, evidence hashes and terminal result only after those artifacts
exist.
