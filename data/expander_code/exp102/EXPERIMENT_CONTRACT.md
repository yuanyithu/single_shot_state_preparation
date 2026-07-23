# exp102 experiment contract

- `physics_contract_version`: `exp102.physics.v1`
- `pt_contract_version`: `exp102.q0_pt.v1`
- `scan_contract_version`: `exp102.scan.v1`
- Parent physics: `exp101.physics.v2`, `true_posterior`, `x_error/H_Z`, `|+>_L`.

At `q=0`, exp102 samples `pi(e|y) proportional to exp(-K_p |e|)` subject to
`H_Z e=y=H_Z epsilon_true`. Every PT rung has this same hard coset. The ladder is linear in
coupling after the configured power transform, and swaps use
`(K_i-K_j)(|e_i|-|e_j|)`. Single-bit and syndrome-changing moves are forbidden.

Each disorder uses four independently seeded PT instances initialized in logical labels zero,
all-ones, even bits, and odd bits. Labels are `uint64`; `k>64` is rejected. The primary disorder
estimator is the average of six pairwise empirical label collisions, normalized by the uniform
mass. Raw estimates are never clipped. Planted hits are an audit statistic, not a per-disorder
name for qtop.

Publication aggregation is fail-closed at disorder, code, and m levels. A code cell needs 128/128
present and valid disorders plus the paired planted audit. An m cell needs 8/8 reportable codes.
The plotted error bar is exactly `std(code_means,ddof=1)/sqrt(8)`.

Registry selection accepts the first eight simple, full-row-rank, distinct classical matrices for
each m. Pilot outcomes, distance, and MCMC difficulty never affect registry membership. Production
workers reconstruct only frozen registry entries and refuse an un-certified PT configuration.

The ordered ladder schedule is contract data, not a Cartesian product. It tries `R=8,12,16,24,32,48,64`
at `p_hot=0.45`, then the same sequence at `0.475`, then `R=8,12,16,24,32,48,64,96,128`
at `0.49`. The `R=96,128` tail was explicitly approved after the original schedule failed its swap
gate on 2026-07-20. Every new configuration hash requires a clean-source pilot; raw validity from an
older configuration or source commit is never reused.

## PT v2 discovery boundary

`exp102.discovery.v2` is an isolated design search for a future `exp102.q0_pt.v2` contract. Its
Q32 ladders, multi-swap trajectories, raw files, reports, and namespaces are not formal pilot or
production evidence. A v2 production config may be generated only after discovery confirms both a
primary and a backup with different `ladder_id` values. Until that happens, the version header above
continues to describe the exhausted v1 production contract and no freezer may be created.

The frozen 2026-07-20 search is itself `EXHAUSTED`: D0/D2/D3/D4 passed the short ladder screen,
but all `S=4,16,64` transport candidates produced zero certified round trips on the two hard cells.
Because every S64 candidate also had an instance with no hot-updated visit, the conditional S128
branch was not legal and the confirmation panel was not run. Resuming requires a newly reviewed
algorithm/contract rather than more rounds, S128, weaker gates, or reuse of this discovery raw.

## PA discovery boundary

`PA_DISCOVERY_CONTRACT.md` freezes the reviewed successor algorithm search under
`exp102.q0_pa.discovery.v1`, with an independent `exp102.q0_pa.raw.v1` schema and seed namespace.
It also freezes a no-extra-randomness replay of 16 old PT trajectories under
`exp102.transport_autopsy.raw.v1`. These are discovery evidence only: they do not modify the
formal version header above and cannot be passed to the PT pilot, freezer, scan-v1 merge, or
production worker.

The PA route uses exactly uniform hard-coset populations at K=0, fixed theta/Q32 annealing,
systematic resampling, and coordinate or exact block4 heatbath mutation. Its four-method hard
screen, one legal rescue branch, blinded 17-cell confirmation, six-cell N256/N512 resolution
panel, genealogy/ESS gates, U-statistic, jackknife MCSE, runtime budget, and fail-closed stopping
rules are contract data. Success can produce only `READY_FOR_FORMAL`; a later clean contract,
tuning, and held-out campaign are still required before any `FROZEN_HELD_OUT_PASS` can exist.

## Global-sampling discovery boundary

`GLOBAL_DISCOVERY_CONTRACT.md` freezes the next isolated search under
`exp102.q0_global.discovery.v1`. It combines one hard-coset global mechanism (cluster Gibbs or
joint stabilizer-logical heatbath) with the independently implemented fixed-clock defect-trace
mechanism. It uses new raw versions, seed namespaces, controls, schedule, analyzer, and readiness
combiner; no PT/PA discovery raw or formal-v1 raw can be reused.

The frozen HARD2/EASY3/CONF17/RES6/GAP8/SMALL6 panels, two 16-trajectory initialization families,
T/2T resources, character and distribution gates, three-node runtime/digest consensus, m3
full-sector TI anchors, and 72-hour fail-closed decision tree are contract data. The implementation
exists but the remote discovery has not run. Its strongest possible success is
`READY_FOR_FORMAL`, which still requires a later clean `exp102.q0_global.v1` tuning and held-out
campaign before production can be considered.

## Logical-signature V0 boundary

`exp102.q0_logical_stratified.v0.v1` is a deliberately narrower diagnostic for a new
label-first independence-MH proposal.  It is not a continuation of the exhausted HGP/PT/PA
screens and cannot reuse their raw, seeds, artifacts, estimates, or terminal statuses.  The only
scheduled cell is `m08_c06, p=.04, d00, attempt022`; the two pre-registered proposal temperatures
are `.5` and `1.0`, each with eight independent `P`, exact-K0 `U`, and legal low-energy
signature-stratified `L` starts.

Before a V0 trajectory can run, an immutable artifact must bind the code matrix, hard syndrome,
full BpLSD decoded-candidate transcript, deterministic rank-first catalog selection, affine
coordinates, proposal, source/config/registry/cell identity, and tail-start schedule.  Every raw
stores both burn and measurement proposal/decision transcripts and is replayed before write.
The V0 analyzer gates actual accepted cross-label changes, source-anchor diversity, character
leave-return excursions, and `P/U/L` starts; total acceptance, local state changes, and proposal
IS ESS are diagnostics only.

The sole possible V0 success string is `LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN`.  It means
only that the proposal merits a fresh HARD2 comparison with an independent confirmer.  It is not
`READY_FOR_FORMAL`, does not certify convergence, and cannot authorize tuning, held-out, or any
production task.

## Logical-signature V0v2 boundary

V0v1 is terminally `CONFLICT_CROSS_ENV_ARTIFACT_IDENTITY`; its source and
artifact bytes must not be retried or repurposed.  The successor
`exp102.q0_logical_stratified.v0.v2` is an equally narrow, fresh diagnostic
whose config and procedures are frozen in
`validation/015_q0_logical_stratified_v0b_20260723/`.  It fixes exactly one
proposal producer (`nd-1`) under `single_producer_algebraic_audit.v1`.
Other hosts audit the identical frozen bytes without rerunning BpLSD/MILP;
their static artifact-audit SHA and fixed-probe discrete trace SHA must agree.

V0v2 retains planted `P`, exact-K0 `U`, and legal decoded-tail `L` starts.
It cannot replace these with physical all-zero starts because this hard
syndrome is nonzero; shifted-coordinate zero is already `P`.  Its transport
gate is measurement-only and requires, separately for each proposal
temperature and each P/U/L family, 128 accepted cross-label moves, six chains
with at least eight such moves, 16 catalog sources, rank-64 accepted label
deltas, and leave-return coverage for every basis and frozen nonbasis
character.  Burn-only movement, total acceptance, and same-sector changes are
diagnostics, not success evidence.  A pass remains only
`LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN` and has no formal authority.
