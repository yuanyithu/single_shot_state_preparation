# exp102 q=0 HGP global sampler screen contract v2

## 1. Purpose and authority boundary

This contract authorizes one fresh diagnostic screen of two exact-target HGP
sampler mechanisms. It asks whether a collapsed likelihood-power sampler mixes
on all five frozen controls and whether the independent multi-anchor
independence-MH sampler confirms it on the two known hard cells within a fixed
resource envelope.

- Contract: `exp102.q0_hgp_global.screen.v2`.
- Config: `config/q0_hgp_global.screen.v2.json`.
- Canonical config SHA256:
  `38092ec030f6c283f163c0ddb3eed612aa850c76ce34f130520522646fa883dc`.
- Maximum success status: `DIAGNOSTIC_HARD_PAIR_FOUND`.
- Formal tuning, confirmation, resolution, held-out, publication loading and
  production are outside this contract.

The screen cannot create `READY_FOR_FORMAL` or `FROZEN_HELD_OUT_PASS` and
cannot launch any of the 6144 production disorder tasks. In particular, MAP
agreement on `HARD2` does not independently certify `EASY3`. A successful
screen only permits a separately reviewed discovery/formal-sampler contract.

This contract has now been executed once, without changing any frozen input or
gate. Run `exp102_q0_hgp_screen_v2_20260722_4d134ee` terminated as
`UNRESOLVED_MAP_MIXTURE_FAIL`; Section 8 records the outcome. This historical
execution note does not retroactively alter the preregistered methods,
statistics, branch order or authority boundary.

The predecessor v1 run
`exp102_q0_hgp_screen_20260722_2e6ba2a` used source
`2e6ba2a864d7db6ae04e79867d1678dbcfe42580`, archive SHA256
`6c5aae08c43a196426c27c41d58e6ad5f6a6f94cc8e494519641794ccf99c5e4`
and source-manifest SHA256
`42ceb8b8619cf5de1d9dc0de16fac31a4f88165d1f6cbeaa038d63accf1046cb`.
Its nd-1/nd-2/nd-3 Linux preflight reached aggregate `PASS` and selected T3.
The required macOS conda-12 replay first exposed a verifier defect: stored
remote solver provenance was incorrectly treated as a requirement on the
local NumPy/SciPy versions. Continuing the audit after isolating that defect
then found one-ULP MAM `log_q`/acceptance drift and full-digest drift in both
50000-draw IS probes. Those transcript/digest drifts, not the solver-provenance
verifier defect, make v1 terminal `CONFLICT`. It created no
measurement control, measurement raw or result and grants no authority to
resume in place. No v1 sampler/preflight raw, auxiliary draws or sampler seed
streams may enter v2.

The local evidence in `validation/012_hgp_collapsed_power_pt_20260722/` is
feasibility evidence only. Its source identity predates the final clean source,
so none of its raw, seeds, reports or decisions may enter this screen.
PT/PA/QC/JB/defect-trace raw and all controls from validation 005--011 are also
forbidden.

## 2. Frozen target and mechanisms

Every method targets exactly

```text
pi(e | y) proportional to (p/(1-p))^|e| 1[H_Z e = y].
```

The planted error selects initialization family `P` but never enters an
energy, likelihood, proposal, anchor, acceptance ratio or adaptive decision.

### 2.1 HP32 and HP64

For the full-row-rank classical HGP seed matrix `H`, write the hard constraint
as

```text
H A xor B H = Y.
```

After integrating out `A`, the cold replica samples the exact collapsed
posterior of `B`; `A|B` is then drawn exactly. The power ladder is frozen as

```text
lambda_i = i^2 / (R-1)^2,  i=0,...,R-1,
```

where `R=32` for `HP32` and `R=64` for `HP64`. Only the collapsed syndrome
log-likelihood is powered. The `lambda=0` endpoint refreshes the prior and the
`lambda=1` endpoint is the target posterior. Each rung uses fixed eight-bit
collapsed heatbath blocks and adjacent replicas use the exact Metropolis ratio.
The ladder arrays and their SHAs are part of the canonical config; no ladder or
block may be changed after observing a result.

### 2.2 MAM-IMH8

`MAM-IMH8` constructs at most eight distinct minimum-weight anchors using only
`(H_Z,y,p)` and frozen deterministic tie-break seeds. The solver configuration,
solver identity, exact GF(2) primal replay, anchor states and anchor SHA are
bound before trajectories start. Planted errors are not an allowed input.
This method, its anchor artifacts and its IS diagnostics are scheduled only on
`HARD2`. V2 uses `exp102.q0_map_mixture.anchors.v3`; the catalog digest now
binds the recorded solver identity as generation provenance, so changing that
identity cannot leave the catalog SHA unchanged.

The solver identity is generation provenance, not a requirement that every
later verifier install the same NumPy/SciPy/HiGHS versions. All three Linux
preflight nodes must agree on the stored remote identity and immutable artifact
bytes. Loading an already frozen artifact for sampling or the independent
conda-12 replay validates that stored identity is present and known, then
replays the file/content hashes, ordered anchors, exact GF(2) constraints,
weights, tie-break/objective hashes, affine coordinates and proposal SHA. It
does not rerun MILP or replace the stored identity with the verifier's local
solver identity. Portable execution is private to this verified-artifact path;
a generated or unbound in-memory catalog cannot opt into it with a boolean
flag. Raw continues to record the stored generation identity.

The proposal is the normalized mixture of the frozen product-Bernoulli
components in the config. Every component probability lies strictly inside
`(0,1)` and the defensive `theta=.5` component has positive weight, so the
proposal has full support. Every transition uses the full Hastings factor

```text
min(1, b^(|e'|-|e|) q(e)/q(e')),  b=p/(1-p).
```

Anchor frequency is not a physical mode weight. There is no proposal tuning,
adaptive restart, state-dependent attempt count or planted-error dressing.

The frozen 50000-draw self-normalized importance-sampling calculation is an
auxiliary proposal-overlap diagnostic. Its ESS, maximum weight, estimated
normalization and stationary acceptance estimate cannot pass or fail a sampler,
select a method or resource tier, replace IMH samples, or enter `q_top`.
Its field manifest separates portable proposal identities, coordinates, packed
states, weights and component indices from nonportable `log_q`, log-importance
and derived floating diagnostics. Linux full evidence freezes both partitions;
local portable evidence compares only the declared portable partition exactly.
This is an evidence projection, not a numerical tolerance.

### 2.3 Initialization and scientific red-team boundary

The physical all-zero bit string is not a generic q=0 initialization: it has
syndrome zero and is outside the target affine space whenever `y != 0`.
Projecting it through an arbitrary section would create a different,
section-dependent initialization and is not evidence of faster mixing. `P` and
`U` are deliberately adversarial rather than claimed optimal: `P` is a valid
planted posterior draw under the generative model, while `U` samples the entire
hard coset at `K=0`. A method must forget both. A later production workflow may
use a canonical or MAP warm start only after this initialization-robustness gate
has passed.

Exact stationarity and convergence evidence are kept separate. Small-code
enumeration and reference/Numba transcript identity check the endpoint target,
collapsed marginal, conditional `A|B` reconstruction and swap ratio. P/U,
transport, Rhat, ESS, character and weight gates then test finite-budget
mixing; they cannot repair an incorrect kernel. Conversely, failure of a
transport or acceptance threshold only means unresolved within this frozen
algorithm and budget, not that the posterior is ill-defined or impossible to
sample.

Equal `q_top` alone is insufficient because different logical distributions
can have the same purity; this is why the family `D2_norm`, basis-character and
weight gates remain mandatory. Even all gates are finite diagnostics rather
than a mathematical proof of mixing. One disorder per sentinel cannot certify
an `(m,p)` point, interpolate neighboring p values, or replace fresh formal
tuning and held-out disorders.

Residual common-mode risk remains explicit. A finite logical/B-character
catalog cannot mathematically exclude an unsampled higher-order Fourier mode,
and 16 exact-uniform `U` starts can miss a basin with tiny K=0 volume but
material target-posterior mass. HP and MAM could then agree inside the same
missed decomposition. The B projections specifically target HP's collapsed
bottleneck; B is not assumed to be the only slow variable of either method,
so the full logical-character, energy, weight, transport and cross-mechanism
gates cannot be replaced by B diagnostics. These limitations are why a pass
retains diagnostic-only authority.

## 3. Frozen cells and independent trajectories

The ordered panel is unchanged from the previous frozen global screen.

`HARD2`:

1. `m06_c00,p=.04,d00,attempt022`
2. `m08_c06,p=.04,d00,attempt022`

`EASY3`:

1. `m03_c00,p=.10,d00,global_fresh_v1`
2. `m04_c00,p=.07,d00,global_fresh_v1`
3. `m05_c00,p=.10,d00,global_fresh_v1`

The panel SHA, five cell fingerprints and five disorder-uniform seeds remain
the already frozen values recorded in the config. This preserves the physical
instances, not any sampler stream. The v2 root, HP/MAM measurement, preflight,
runtime and IS namespaces are fresh `q0_hgp_global_screen_*_v2` streams and may
not reuse v1 draws. The unchanged character catalogs and deterministic anchor
tie-break namespace identify unchanged mathematical objects rather than sampler
trajectories; their bytes are rebound by the v2 config and anchor-v3 hashes.

Every scheduled method/cell has two adversarial initialization families:

- `P`: start from the planted hard-coset error;
- `U`: start from an exact `K=0` uniform hard-coset draw.

Each family contains 16 independent trajectories. The immutable measurement
control therefore contains exactly

```text
2 HP methods * 5 cells * 2 families * 16 trajectories = 320 tasks
1 MAP method * 2 cells * 2 families * 16 trajectories = 64 tasks
total = 384 tasks.
```

There is no resampling, cloning,
replacement of a failed trajectory, early stopping after enough samples, or
post-result addition of tasks.

This restricted MAP scope was frozen before any server measurement. Local
feasibility showed that the unique m3 minimum anchor and deterministic
near-MAP shells left the planted chain at zero accepted moves, while some shell
MILPs reached the 180-second cap; the m5 primary solve also exceeded 180
seconds. Those probes are design evidence only. Near-MAP shells are not part of
this contract, and known-failing EASY3 MAP tasks are not run merely to increase
the task count.

## 4. Source, seeds and raw isolation

The screen uses a clean Git archive and a verified, bytecode-free source tree.
Every task identity binds source commit, archive/manifest SHA, registry SHA,
config SHA, cell fingerprint, method, resource tier, initialization family,
trajectory index, stage and the new seed namespace.

Measurement HP trajectories, measurement MAM trajectories and frozen screen IS
diagnostics use three disjoint namespaces. Preflight digest, runtime warmup,
runtime timed, preflight IS and runtime IS each use a further dedicated
namespace. The only
intentional seed reuse is the reference/Numba execution of the same tiny-oracle
case, whose complete identities and transcripts must be bit-identical. A
regression enumerates every frozen trajectory and IS 63-bit stream and rejects
any other numeric collision, including any auxiliary/measurement overlap. The
same enumeration is repeated with the eventual clean deployment's actual
commit/archive/manifest identities before remote launch; the unit-test fixture
identity alone is not deployment evidence.

Fresh versions are:

- tasks: `exp102.q0_hgp_global.screen.tasks.v2`;
- HP raw: `exp102.q0_hgp_power.raw.v3`;
- MAP sampler transcript: `exp102.q0_map_mixture.raw.v2`;
- MAP screen raw envelope: `exp102.q0_map_mixture.raw.v4`;
- MAP artifact: `exp102.q0_hgp_global.screen.map_artifact.v2` with anchor
  catalog `exp102.q0_map_mixture.anchors.v3`;
- IS raw: `exp102.q0_hgp_global.screen.is_diagnostic.v2`;
- report: `exp102.q0_hgp_global.screen.report.v2`;
- decision: `exp102.q0_hgp_global.screen.decision.v2`.

NPZ loading always uses `allow_pickle=False`. Raw numerical estimates are not
clipped. A missing/extra field, nonfinite value, algebra failure, SHA/identity
mismatch, seed mismatch or replay mismatch is `CONFLICT`, not a statistical
failure.

Every raw stores the initial, burn-end and final state plus every fixed-clock
measurement state in packed form, labels/signatures, physical weights, hard
residual weights, eight time blocks, all seeds/hashes, timing and counters. The
analyzer independently reconstructs residuals, weights and labels.

Each sampler raw also binds an exhaustive field manifest and separate full,
portable and nonportable-float digests. The full projection is byte-exact
Linux evidence. The portable projection contains only explicitly declared
cross-platform discrete fields and remains byte-exact; no field is rounded and
no ULP allowance exists. For MAM, proposal states, uniforms, acceptance flags,
state-change flags and fixed-clock states form an additional exact acceptance-
decision digest. Nonportable log-q/log-acceptance arrays remain preserved and
full-replayed on Linux but cannot silently enter the local portable claim.

The HP-v3/MAP-v4 screen envelopes also bind a deterministic packed
`B`-character catalog and its SHA. For a classical matrix with `r` rows,
catalog order is all
`r^2` row-major single bits, all row parities, all column parities, then 64
PortablePrng-derived mutually distinct dense masks of weight between one third
and two thirds of `r^2`. The analyzer extracts `B` from the already packed full
initial, burn-end and fixed-clock measurement states and independently
recomputes `|B|`,
`L(B)=sum_j log M_p[(Y xor B H)_j]`, and every catalog character. Thus no
sampler-supplied derived B statistic can pass its own replay.

HP raw additionally binds the classical mass table SHA, exact lambda array and
SHA, per-rung local attempts/changes, per-edge swap attempts/accepts, origin hot
and cold visits, strict cold-hot-cold round trips, cold likelihood trace and
final origin permutation. MAM-IMH8 raw binds solver identity, anchor catalog
and component SHAs, coordinate SHA, proposal SHA and parameters, every proposed
state/log-q, every acceptance uniform/log-ratio/decision, and stage-separated
attempt/accept counts. It also records, at every burn and measurement clock,
whether an accepted proposal actually differs from the pre-step state, plus
stage-separated state-change counts. An accepted proposal equal to the current
state is an MH self-loop, not transport. HP does not expose the internal
uniform and individual decision transcript needed to claim per-decision
cross-platform replay. Its portable claim is deliberately limited to exact
agreement of fixed-clock discrete outputs and transport counters; it must never
be summarized as bit identity of every hidden HP decision.

Anchor/proposal artifacts are constructed once per `HARD2` cell, replayed on
all three nodes and bound by SHA to all 32 MAM-IMH8 trajectories for that cell.
Sharing a read-only deterministic artifact is allowed; sharing trajectory RNG
state is not.

## 5. Three-node preflight and runtime selection

Preflight runs independently on `nd-1`, `nd-2` and `nd-3` under conda `11` and
must pass before controls are materialized. It includes:

- all relevant exp102 and exp101 HGP/logical/exact/loader regressions;
- tiny exact normalization, stationarity and detailed-balance oracles;
- exact K0 initializer and hard-residual replay;
- reference/Numba bit identity, including `k=64` and bit 63;
- canonical H, mass, lambda, character, anchor, coordinate and proposal SHAs;
- exact full-Linux transcript digests and separately sealed portable
  projections;
- four fixed MAM portability probes: both `HARD2` cells, each from `P` and `U`,
  with `burn=256` and `measurement=2048`, including exact proposal draws,
  uniforms, acceptance decisions and resulting states;
- source/archive/config/registry/seed tamper rejection;
- per-method runtime measurement.

This screen does not claim a formal per-task peak-memory benchmark or use
memory observations to select a tier. Before launch, read-only node inspection
confirmed approximately 251/251/503 GiB on nd-1/nd-2/nd-3; the frozen worker
counts remain 75/91 on nd-2/nd-3. Any actual allocation failure is an
infrastructure failure and cannot be repaired by silently reducing ownership
or changing the selected tier in place.

Execution is intentionally split at the aggregate preflight boundary. The
first verified orchestrator phase stops after the three node reports are
combined and exposes the immutable schedule, artifact manifest/files and
aggregate preflight hashes. Before the control-freeze deadline, those files are
pulled to the macmini and replayed under conda `12`; only an explicit second
verified orchestrator invocation carrying the exact local-attestation file SHA
may freeze the 384-task control.

All three Linux nodes must agree byte-for-byte on the complete full payload,
including declared nonportable floats, and on the portable payload. This is
the full-Linux exact consensus. The local audit then independently rebuilds
both projections from the same verified artifact bytes; its authorization
claim is the portable-local exact projection. It emits `PASS_EXACT`
when the full payload also matches, or `PORTABLE_PASS` only when the portable
payload and the four MAM acceptance-decision digests match exactly after remote
full consensus. A local full-payload mismatch is diagnostic and may occur only
in fields excluded by the frozen exhaustive field manifest; no runtime field
can be moved between partitions. There is no ULP, absolute, relative or
rounding tolerance. Any portable state, coordinate, proposal identity,
uniform, acceptance decision, counter, hash or IS discrete-transcript mismatch
is `CONFLICT`. A local solver-version mismatch alone is not a conflict under
the stored-provenance rule above.

For HP, `acceptance_decision_sha256` is explicitly the portable fixed-clock
output/transit-counter digest because the kernel does not expose every internal
uniform and decision. This field name does not authorize a stronger scientific
claim. No measurement node may be launched merely because three remote worker
processes exited successfully; the aggregate report must be `PASS`, name one
frozen common tier, and the local attestation must be a fully validated
`PASS_EXACT` or `PORTABLE_PASS`.

The storage node `nd-0` has no `screen` executable. Its verified outer
orchestrator therefore runs under fixed `/usr/bin/nohup` plus
`/usr/bin/setsid`, with an atomic per-run/per-phase launch-guard directory,
identity metadata and separate bootstrap/orchestrator PID records. The guard
is never reused: a duplicate or failed detached launch requires a fresh
deployment/run as otherwise specified by this contract. The Python
orchestrator verifies the dedicated persistence token, canonical guard
identity and that it is a session leader before it can create a stage. This
change applies only to the `nd-0` control process. Every scientific preflight,
control, sampler and analysis stage on `nd-1`/`nd-2`/`nd-3` still runs in its
own immutable `screen` through the verified wrapper.

The compute nodes run RHEL Bash 4.2. The verified stage wrapper must therefore
remain compatible with Bash 4.2 `set -u` semantics: the sole zero-prerequisite
stage, `build-schedule`, may not expand an empty array. An executable regression
must reach a `build-schedule` SUCCESS marker with an exactly empty
`prerequisite_success_sha256` list before a clean source may be deployed.

The frozen schedule schema is `exp102.q0_hgp_global.screen.schedule.v2`.
Observed node epochs are not synchronized (approximately nd-1 `+204 s`, nd-2
`+298 s`, and nd-3 `+2 s` relative to nd-0 during the infrastructure audit), so
no compute-node or macOS epoch grants deadline authority. The preflight
orchestrator captures one nd-0 `CLOCK_BOOTTIME` interval of at most one second,
the nd-0 Linux boot ID and a diagnostic Unix timestamp. Exact boottime
deadlines are start plus 6/8/22/24 hours. Measurement must reconstruct the
schedule stage from that same frozen anchor; it may not capture a new one.
The nd-0 orchestrator checks the boot ID and boottime before every stage launch
and again at the instant every SUCCESS marker is accepted. Equality with a
deadline is failure, and a reboot invalidates the run. Node-local Unix times
are finite diagnostic fields only; monotonic elapsed time records local
duration and cross-node epoch ordering is forbidden.

Each accepted stage has an exclusive, self-hashed nd-0 record that binds the
exact SUCCESS bytes, stage fingerprint, prerequisite graph, boot ID, observed
boottime and applicable deadline. Ordered preflight and measurement acceptance
manifests bind the complete record sets, phase launch metadata and output file
hashes. A terminal result is the conjunction of the terminal package and the
measurement acceptance manifest, never either file alone. Before publishing
that manifest and again after local transfer, validation reconstructs the exact
384-trajectory plus two-IS raw/claim set from the frozen control and verifies
every regular non-symlink path, identity and file hash; missing, extra or
tampered raw is `CONFLICT` even if analysis previously completed.

Cross-node Linux full scientific arrays and transcripts must be bit identical.
Across Linux and macOS, the declared portable projection must be bit identical.
No ULP whitelist applies to either projection. A platform `log/exp`, NumPy,
SciPy or HiGHS difference in a declared nonportable float remains visible in
the full evidence and local mismatch paths; a difference in any portable digest
or MAM acceptance decision is `CONFLICT`.

Only these common resource tiers exist:

| tier | burn | measurement |
|---|---:|---:|
| `T1` | 2048 | 8192 |
| `T2` | 4096 | 16384 |
| `T3` | 8192 | 32768 |

For HP the units are rounds; for MAM-IMH8 they are IMH steps. Using only
preflight timing, select the largest single tier that all three
methods can run. Runtime projection mirrors the actual sequential stages:

1. generation takes the larger of the frozen nd-2 and nd-3 ownership LPT
   makespans at capacities 75 and 91, including each node's owned IS work;
2. final analysis is frozen to nd-3 with 91 workers, replays both IS files
   serially, and replays all 384 trajectories with a separate 91-lane LPT;
3. one-time serial artifact construction is added once.

The three wall times are summed and only then multiplied by the independent
factor-two safety margin; pretending that final replay follows generation
ownership or dividing aggregate work by aggregate capacity is forbidden. Tier
selection assumes the latest permitted control origin at hour 8: safeguarded
generation must complete by hour 22 and safeguarded analysis by hour 24. The
pre-existing 79200-second complete-screen budget remains an additional cap,
not a replacement for those stricter stage deadlines. Node epochs and observed
physics outcomes cannot change the projection. A predicted trajectory above
7200 seconds removes the entire screen with `RUNTIME_EXHAUSTED`; candidates may
not be silently deleted. If `T1` does not fit, the outcome is
`RUNTIME_EXHAUSTED`.

Resource selection must not read `q_top`, labels, weights, acceptance outcomes,
IS estimates or any pass/fail physics statistic. After controls are frozen,
execution is only on `nd-2` and `nd-3`; ownership is immutable and a task cannot
migrate within a stage. A failed marker requires a fresh run/deployment rather
than an in-place rerun.

## 6. Statistical estimators and common gates

`q_top` uses the independently debiased character U-statistic. For `k<=10`, use
all nonzero characters. For `k>10`, use all basis characters and the same 4096
frozen, uniformly distinct nonbasis characters. All masks and signatures are
`uint64`; `k=64` and bit 63 must never pass through a signed `int64` boundary.
Raw-label collision is diagnostic only.

The family-distribution statistic is

```text
D2_norm = mean_{u != 0} (m_P(u)-m_U(u))^2
        = ||P_P-P_U||_2^2 / (1-2^-k).
```

Finite-sample negative U-statistics remain negative. Total MCSE conservatively
combines trajectory delete-one jackknife and character finite-population/batch
SE in quadrature. Every P/U or cross-method `Delta q_top` uses the shared
frozen character sample as a paired design: character SE is computed directly
from the per-character squared-mean differences, while independent left/right
delete-one trajectory contrasts contribute separate jackknife variances. It is
forbidden to add the two marginal character SEs as though their common masks
were independently sampled.

Every method/cell must pass all common gates:

```text
SE_total(q_top) <= .03
|Delta q_top| <= .04
|Delta q_top| <= 3 SE_delta + .005
max(0,D2_hat) + 3 SE_D2 <= .04
normalized mean-weight difference <= .01
normalized mean-weight difference <= 3 SE + 1/n
split Rhat <= 1.05
nondegenerate bulk ESS >= 400
```

Trace gates cover energy, every basis character and 64 frozen diagnostic
nonbasis characters. If a measured character is constant while initial chains
contain both signs, every opposite-sign chain must reach the common sign during
burn; otherwise the method fails common-freezing detection.

The collapsed slow variable has a separate, stricter gate for every
method/cell and each 16-chain initialization family:

```text
split Rhat <= 1.05 for L(B), |B|, and every frozen B character
bulk ESS >= 400 for L(B), |B|, all row/column parities, and all
  nondegenerate dense characters
at least 48 of the 64 dense characters are nondegenerate
no individual trajectory has constant L(B) or constant |B|
```

All `r^2` single-bit characters remain frozen, raw-bound and subject to the
vectorized split-Rhat and P/U distribution gates. By reviewed design they do
not each receive a separate FFT ESS calculation: at `r=24` that would repeat
576 highly redundant long FFTs per family. ESS is instead mandatory on the
row/column and dense multibit projections above. This explicit compromise does
not waive the Rhat, constant-trace, dense-coverage or distribution tests.

For any B character constant at the measurement clock, every trajectory whose
initial sign is opposite the common sign must have the common sign at the burn
endpoint. This is a conservative sufficient witness that each such chain
crossed during burn; checking only whether any one chain crossed is forbidden.
Within each method, and again between HP and MAM on matching HARD2 families,
the frozen B-character mean-square difference has upper bound `.04`. Every
individual B-character mean difference must also be at most `.04` and
`3 SE + .005`; this prevents one frozen bit from being diluted by the catalog
average. Normalized mean `|B|` differs by at most `.01` and `3 SE + 1/r^2`, and
mean `L(B)` per A factor differs by at most `.01` and `3 SE + 1/n`. Pooled B
Rhat must also pass.

HP also fails unless every one of its 32 independent trajectories for the cell
has, over its full fixed burn-plus-measurement clock:

```text
every adjacent-edge swap rate >= .05
every adjacent edge has at least 20 accepted swaps
at least half of all replica origins visit the cold endpoint
sum of strict cold-hot-cold round trips over origins >= 4.
```

No aggregate across trajectories may hide one failed transport path.

MAM-IMH8 also fails unless every one of its 32 independent trajectories for
the cell has the following stage-specific records:

```text
at least one actual state change during burn
measurement state-change rate >= .05
measurement state changes >= 400.
```

Acceptance decisions and acceptance rates remain in raw as MH diagnostics, but
cannot satisfy these transition gates because an accepted proposal may be an
exact self-loop. The state-change gates are not substitutes for Rhat, ESS, P/U
agreement or the distribution gate.

## 7. Method selection, agreement and terminal states

`HP32` and `HP64` both run on all five cells. Among HP methods that pass every
gate on every cell, choose one primary by

```text
(total runtime core-seconds, number of replicas, method_id).
```

No `q_top` value is part of this ordering. `MAM-IMH8` is the single independent
`HARD2` primary. The selected HP and MAM-IMH8 must pass the same `q_top`,
`D2_norm` and normalized-weight agreement inequalities for corresponding `P`
families and corresponding `U` families on both `HARD2` cells. `EASY3` is an HP
runtime/false-negative control and has no cross-mechanism claim. If the
selected HP disagrees on `HARD2`, the slower HP cannot rescue it.

After a complete replay-valid report, the statistical branches are:

- five-cell HP plus HARD2 pair pass -> `DIAGNOSTIC_HARD_PAIR_FOUND`;
- neither HP passes all cells -> `UNRESOLVED_NO_HP_PASS`;
- MAM-IMH8 fails either hard cell -> `UNRESOLVED_MAP_MIXTURE_FAIL`;
- both primaries pass individually but disagree ->
  `UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT`.

Infrastructure, identity or replay errors instead produce `CONFLICT`; schedule
or runtime infeasibility produces `RUNTIME_EXHAUSTED`. No branch means a physical
parameter is mathematically impossible.

Even `DIAGNOSTIC_HARD_PAIR_FOUND` retains at least these formal blockers:

```text
NO_FRESH_T_VS_2T
NO_CONF17_RES6_GAP8_SMALL6
NO_M3_EXACT_OR_FULL_SECTOR_TI_CONFIRMATION
NO_FORMAL_96_CELL_TUNING
NO_448_CELL_HELD_OUT
NO_FROZEN_HELD_OUT_PASS
```

Thus this contract can identify a promising hard-cell sampler pair and an HP
candidate worth broader testing, but it cannot report the full
`p=.04..10,m=3..8` range and cannot start production.

## 8. Completed execution record

The sole v2 execution used source
`4d134ee7ca25125d341eb11cbfa34d6856514101`, archive SHA256
`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`
and source-manifest SHA256
`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`.
All three Linux preflight reports reached exact consensus, the common tier was
T3, and the mandatory macmini audit returned `PORTABLE_PASS`. Fixed ownership
then produced all 384 trajectory raw and both IS raw. Full nd-3 replay and the
independent local audit validated the exact control-derived 386-file set.

The terminal method counts were:

```text
HP32       3/5 cells pass
HP64       5/5 cells pass
MAM-IMH8   1/2 HARD2 cells pass
HP32/MAM   0/4 family-cell comparisons pass
HP64/MAM   0/4 family-cell comparisons pass
```

HP64 is therefore a promising same-mechanism candidate, but no independent
pair passed. MAM failed `m08_c06,p=.04` on family/B Rhat and ESS, B-character
agreement, and HP/MAM `q_top` agreement. Its ordinary state-change gate did
pass: the minimum trajectory had 520 burn changes, 1947 measurement changes
and measurement change rate `.0594`. This confirms the contract's statement
that generic transition counts are not substitutes for slow-variable and
distribution gates.

Post-run inspection of the already frozen raw identified the structural
failure mode. The two distinct minimum-weight m8 anchors have identical
all-zero logical coordinates. The 16 P trajectories made 39899 measurement
state changes but only 330 logical-label changes; U made 40735 and 288. Thus
the proposal's apparently adequate acceptance was overwhelmingly
within-sector. Components with `theta_logical=.08,.25,.5` accepted no
proposals. The `.02` component had 22 accepted proposals across both families,
but only 11 actual same-sector state changes (P/U: 4/7) and no logical change.
This forensics did not change any gate or terminal branch.

The result also validates the conservative purpose of cross-mechanism
comparison. On m6, both methods individually passed and both P/U families
agreed internally, yet HP64 and MAM estimated `q_top=0.14587` and `0.16241`;
their absolute difference passed `.04` but failed the paired uncertainty gate.
On m8 they estimated `0.91317` and `0.99273`, failing both inequalities.
Neither aggregate acceptance, global IS ESS, normalized weight nor aggregate
D2 alone would establish the required logical distribution agreement.

The terminal-package identity is
`233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`,
the joint-terminal SHA is
`7e9bd8d7efb657649c4a0b4f0d146b72063d4584b291479a49c805e6834ab4f1`,
and the local terminal attestation is
`386e8a0eeadb5c24b376014b522dec36322456abf3b0d636c1ad16cc7681c755`.
Formal and production authorization are both false.

A successor must not simply lengthen this MAM configuration. It first needs a
separately reviewed, result-independent construction with deterministic
logical-signature coverage and a viability gate on accepted cross-signature
transport. P/U remain necessary legal adversarial starts; additional starts
should be legal and signature-stratified. The physical all-zero string remains
outside all five nonzero-syndrome hard cosets and is not a valid replacement.
