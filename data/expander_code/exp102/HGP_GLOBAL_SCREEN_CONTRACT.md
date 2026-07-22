# exp102 q=0 HGP global sampler screen contract v1

## 1. Purpose and authority boundary

This contract authorizes one fresh diagnostic screen of two exact-target HGP
sampler mechanisms. It asks whether a collapsed likelihood-power sampler mixes
on all five frozen controls and whether the independent multi-anchor
independence-MH sampler confirms it on the two known hard cells within a fixed
resource envelope.

- Contract: `exp102.q0_hgp_global.screen.v1`.
- Config: `config/q0_hgp_global.screen.v1.json`.
- Canonical config SHA256:
  `3c65ef96ce231b4aea4499b5a6030f1cc82475117c5ee5ecc7633d972ef8edc9`.
- Maximum success status: `DIAGNOSTIC_HARD_PAIR_FOUND`.
- Formal tuning, confirmation, resolution, held-out, publication loading and
  production are outside this contract.

The screen cannot create `READY_FOR_FORMAL` or `FROZEN_HELD_OUT_PASS` and
cannot launch any of the 6144 production disorder tasks. In particular, MAP
agreement on `HARD2` does not independently certify `EASY3`. A successful
screen only permits a separately reviewed discovery/formal-sampler contract.

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
`HARD2`.

The solver identity is generation provenance, not a requirement that every
later verifier install the same NumPy/SciPy/HiGHS versions. All three Linux
preflight nodes must agree on the stored remote identity and immutable artifact
bytes. Loading an already frozen artifact for sampling or the independent
conda-12 replay validates that stored identity is present and known, then
replays the file/content hashes, ordered anchors, exact GF(2) constraints,
weights, tie-break/objective hashes, affine coordinates and proposal SHA. It
does not rerun MILP or replace the stored identity with the verifier's local
solver identity. Raw continues to record the stored generation identity.

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
instances, not any sampler stream. All sampler, character, anchor and IS seed
namespaces are new `q0_hgp_global_screen_*_v1` namespaces.

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

Fresh versions are:

- tasks: `exp102.q0_hgp_global.screen.tasks.v1`;
- HP raw: `exp102.q0_hgp_power.raw.v2`;
- MAP raw: `exp102.q0_map_mixture.raw.v2`;
- report: `exp102.q0_hgp_global.screen.report.v1`;
- decision: `exp102.q0_hgp_global.screen.decision.v1`.

NPZ loading always uses `allow_pickle=False`. Raw numerical estimates are not
clipped. A missing/extra field, nonfinite value, algebra failure, SHA/identity
mismatch, seed mismatch or replay mismatch is `CONFLICT`, not a statistical
failure.

Every raw stores the initial, burn-end and final state plus every fixed-clock
measurement state in packed form, labels/signatures, physical weights, hard
residual weights, eight time blocks, all seeds/hashes, timing and counters. The
analyzer independently reconstructs residuals, weights and labels.

The v2 screen envelope also binds a deterministic packed `B`-character
catalog and its SHA. For a classical matrix with `r` rows, catalog order is all
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
attempt/accept counts.

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
- exact transcript digests for HP categorical/swap decisions and MAP proposals/
  acceptances;
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
may freeze the 384-task control. The normal attestation status is `PASS` and
requires byte-identical complete canonical digests. If an actual mismatch is
observed, the automatic audit emits
`MISMATCH_REQUIRES_PORTABILITY_REVIEW`, which grants no launch authority. This
contract and source accept only an exact `PASS`; `PORTABLE_PASS` is not a
status in this contract and cannot be introduced by continuing the same run.
Any future portability exception requires a different reviewed contract,
source commit and fresh run. No ULP whitelist is preregistered speculatively.
A local solver-version mismatch alone is not a conflict under the stored-provenance
rule above, while any discrete state/anchor/proposal/transcript/hash mismatch
is `CONFLICT`. No measurement node may be launched merely because three remote
worker processes exited successfully; the aggregate report must be `PASS` and
name one frozen common tier.

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

Cross-node scientific arrays and decision transcripts must be bit identical.
No ULP whitelist applies to categorical weights, states, labels, anchors,
proposals or acceptance decisions. A platform `log/exp`, NumPy, SciPy or HiGHS
difference that changes a frozen digest is `CONFLICT`.

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
ownership or dividing aggregate work by aggregate capacity is forbidden. The
resulting complete 384-task schedule must fit the 79200-second screen window. A
predicted trajectory above 7200 seconds removes the entire screen with
`RUNTIME_EXHAUSTED`; candidates may not be silently deleted. If `T1` does not
fit, the outcome is `RUNTIME_EXHAUSTED`.

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
SE in quadrature.

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
at least one proposal is accepted during burn
acceptance rate >= .05
accepted proposals >= 400.
```

These are transition gates, not substitutes for Rhat, ESS, P/U agreement or
the distribution gate.

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
