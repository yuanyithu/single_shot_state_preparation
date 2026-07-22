# 013 q=0 HGP global sampler screen

## Current status

`V2 TERMINAL UNRESOLVED_MAP_MIXTURE_FAIL / PRE-PILOT`

The fresh immutable run is
`exp102_q0_hgp_screen_v2_20260722_4d134ee`, source
`4d134ee7ca25125d341eb11cbfa34d6856514101`, archive SHA256
`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`
and source-manifest SHA256
`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`.
All three Linux preflight workers reached exact consensus and selected
`T3=(8192 burn,32768 measurement)`. The macmini conda-12 audit returned
`PORTABLE_PASS`; it exactly reproduced every declared portable field and all
four MAM acceptance-decision probes. The only full-payload differences were
the 12 preregistered nonportable floating fields.

Measurement completed 384/384 frozen sampler tasks and 2/2 IS diagnostics on
the fixed nd-2/nd-3 ownership without migration, retry, replacement or
resampling. The nd-3 full analyzer and local terminal audit validated all
386 raw files. The terminal package status is
`UNRESOLVED_MAP_MIXTURE_FAIL`: HP64 passed 5/5 cells, HP32 passed 3/5, and
MAM-IMH8 passed only 1/2 HARD2 cells. Both HP/MAM pairs passed 0/4
family-cell agreement comparisons, so `selected_pair=null`.

The central scientific discrepancy is not hidden by a runtime or
infrastructure error. HP32 and HP64 agree on the hard cells, while MAM is
systematically higher: on m6 HP64/MAM give `q_top=0.14587/0.16241`; on m8
they give `0.91317/0.99273`. The m6 difference is below the absolute `.04`
limit but fails the paired uncertainty inequality; the m8 difference also
exceeds `.04`. MAM's m8 P family has maximum Rhat `1.06088` and minimum ESS
`379.74`; its B projections fail in both P and U, with maximum Rhat
`1.08245/1.05662`, minimum ESS `275.33/361.16` and 16 inconsistent B
characters.

Ordinary transport counters were misleading on that same m8 cell. Every MAM
trajectory recorded at least 520 burn and 1947 measurement state changes, yet
raw replay shows that only 330 of 39899 P-family state changes and 288 of
40735 U-family changes altered the logical label. A typical chain visited only
three labels. Both distinct m8 minimum-weight MAP anchors have the same
all-zero 64-bit logical coordinate. Within each family's 524288 attempts, the
`theta_logical=.08,.25,.5` components accepted no proposals. The `.02`
component had 11 accepted proposals per family, but only 4 P and 7 U actual
same-sector state changes and no logical-label change. The proposal-overlap IS
estimate and aggregate acceptance therefore mostly measured within-sector
motion, not the global quantity needed for `q_top`.

HP32's two EASY3 failures have different meanings. The m3 failure is a
borderline conservative rejection: one column character was `0.0404396`
against `.04`, while its uncertainty test and every other B gate passed. The
m5 failure is clear slow-mode evidence: U has maximum B Rhat `1.1552`, minimum
B ESS `327.0`, and pooled P/U Rhat `1.1172`. HP64 passes both controls, so it is
a promising candidate, but HP32/HP64 are two settings of one mechanism and
cannot independently confirm each other.

The complete evidence identities are:

- measurement acceptance manifest:
  `42e12338dac640b725728f25c46b4d853a23e02392b2c9f2471f519ffcf5bba1`;
- joint terminal:
  `7e9bd8d7efb657649c4a0b4f0d146b72063d4584b291479a49c805e6834ab4f1`;
- terminal package:
  `233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`;
- report self-SHA:
  `bb2b8ef99dfbb1ba008bfddf1a64bc0ad9fccabc350bf2e0bd28b48d19dca062`;
- local terminal attestation:
  `386e8a0eeadb5c24b376014b522dec36322456abf3b0d636c1ad16cc7681c755`.

The predecessor v1 run
`exp102_q0_hgp_screen_20260722_2e6ba2a` remains `CONFLICT`: after its Linux
preflight PASS, local audit found one-ULP MAM floating drift and two IS full
digest drifts. It produced no measurement control or raw and was not resumed.
Two earlier attempts stopped before a run root on nd-0 infrastructure issues.

This completed screen has diagnostic authority only. It certifies no `(m,p)`
point and cannot produce `READY_FOR_FORMAL`, `FROZEN_HELD_OUT_PASS`,
publication data or any of the 6144 production tasks.

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
The v2 sequence above is complete and terminal; its raw remains diagnostic
evidence and cannot be reused by a later sampler contract.
