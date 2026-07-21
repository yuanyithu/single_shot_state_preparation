# exp102 q=0 global-sampling discovery contract

- Discovery contract: `exp102.q0_global.discovery.v1`
- Hard-coset raw: `exp102.q0_hardcoset.raw.v1`
- Defect-trace raw: `exp102.q0_defect_trace.raw.v1`
- Bias-tuning raw: `exp102.q0_defect_bias.raw.v1`
- Full-sector TI raw/report: `exp102.q0_global.ti_anchor.raw.v1` /
  `exp102.q0_global.ti_anchor.report.v1`
- Schedule/postselection/control freeze: `exp102.q0_global.schedule.v1` /
  `exp102.q0_global.postselection_plan.v1` /
  `exp102.q0_global.control_freeze.v1`
- Frozen config SHA256:
  `1d0a453f2bf8445ad6587c612c2eabb3049e76e2d73b59c230b8b1358b06e565`
- Parent physics: `exp102.physics.v1`, `true_posterior`, `x_error/H_Z`, `q=0`
- Current status: implementation and exact-oracle validation; no discovery result yet

This is the reviewed successor search after both PT-v2 transport and PA genealogy
exhausted. It does not modify the historical formal versions
`exp102.q0_pt.v1 / exp102.scan.v1`. Nothing produced here may enter the old
pilot, freezer, scan-v1 merge, or production worker. Even complete success can
produce only `READY_FOR_FORMAL`; a later clean formal contract, tuning campaign,
and held-out campaign are still mandatory before `FROZEN_HELD_OUT_PASS`.

## 1. Posterior and non-negotiable isolation

Every sampler targets

```text
pi(e | y) proportional to exp(-K_p |e|),  H_Z e = y,
K_p = log((1-p)/p),  y = H_Z epsilon_data_true.
```

The energy is `|e|`, never `|e xor epsilon_data_true|`. The planted error is an
initial state and audit value only. PT/PA raw, seeds, task identities, schedules,
freezers, and loaders are rejected. New seed identities bind source, config,
registry, cell, method, resource tier, initialization family, trajectory, stage,
and role.

There are two adversarial initialization families, each with 16 independent
trajectories and no cloning or resampling:

- `P`: start at the planted error;
- `U`: draw exactly uniformly from the affine hard coset at `K=0` using the
  independent stabilizer plus logical coordinate bijection.

## 2. Frozen logical proposal catalog

The catalog is a deterministic function of the code matrix and observable frame.
Starting from the canonical logical move basis, repeatedly apply the ordered row
operation `row_i ^= row_j` giving the greatest strict reduction of total Hamming
weight; ties use `(i,j)`. Stop when no strict decrease exists.

Enumerate every reduced single, pair, and triple. Each proposal stores its support
and the explicit `uint64` signature `W d`; proposal position is not a logical bit.
Sort and deduplicate by `(weight, signature, little-endian packed support)`. Greedily
retain a signature-rank-`k` subset, then fill in sorted order to `min(8k,512)`.
Every proposal must satisfy `H_Z d=0`, the stored signature must equal `W d`, and
the frozen catalog SHA is raw identity.

Random restarts, outcome-driven dressing, state-dependent catalog selection, and
changing the catalog after observing a trajectory are forbidden.

## 3. Hard-coset candidates

Every macrostep first performs one random-permutation stabilizer heatbath sweep and
one random-permutation catalog heatbath sweep. A toggle with weight change
`Delta w` is chosen with probability `1/(1+exp(K_p Delta w))`.

### 3.1 Pin-and-kernel cluster Gibbs

For one cluster update, set `b=p/(1-p)`:

```text
every current 1 bit is free;
every current 0 bit is pinned independently with probability 1-b;
d is uniform in ker(H_Z[:, free]);
e <- e xor d.
```

The restricted kernel is sampled by deterministic packed RREF plus independent
PortablePrng free coefficients. This is a rejection-free augmented Gibbs step.
There is no adaptive controller, cost-based skip, or outcome-driven attempt count.

- `RC8-QC1`: one cluster per macrostep;
- `RC8-QC4`: four clusters per macrostep.

### 3.2 Joint stabilizer-logical heatbath

For every reduced logical direction, construct one fixed block containing that
direction followed by stabilizers ordered by decreasing support overlap and then
row index. Greedily keep only generators that increase GF(2) rank.

Each macrostep selects one frozen block uniformly and state-independently. It
enumerates the `2^b` actual XOR states with a Gray walk and samples their exact
categorical posterior. Overlapping supports are evaluated on the actual state;
summing individual deltas is forbidden.

- `RC8-J08`, `RC8-J12`, `RC8-J16` use block sizes 8, 12, and 16;
- J16 runs only if its Linux runtime gate passes;
- a joint method and reduced-only variant cannot count as independent mechanisms.

## 4. Independent defect-trace confirmation

Defect trace samples the extended target

```text
mu(e) proportional to exp[-K_p |e| + bias(D(e))],
D(e) = |H_Z e xor y|,  0 <= D <= Dmax.
```

It uses random-permutation single-bit exact heatbath sweeps with `K_q=0`.
Observations occur at every fixed sweep clock. Only observations whose stored
`D=0` enter the hard-coset estimator. First-return sampling, running until a
sample quota is met, catalog moves, and hard-coset cluster moves are forbidden.

Candidates are `DT16`, `DT32`, and `DT64`. For each cell/method/resource identity,
bias tuning uses eight independent uniform-hard-coset chains and exactly 4096
sweeps. The target histogram is `rho_0=.25` and `rho_d=.75/Dmax` for `d>0`:

```text
bias_d <- bias_d + gamma_t (rho_d - observed_fraction_d)
gamma_t = min(.1, .5/(t+10)^.6)
```

Tuning, burn, and measurement have independent seed roles. The complete bias raw
is validated, frozen by SHA, and bound into every measurement task before that
task exists.

## 5. Resources and fixed clock

Only these base tiers exist:

| Tier | Burn sweeps | Measurement sweeps |
|---|---:|---:|
| `T1` | 2048 | 8192 |
| `T2` | 4096 | 16384 |
| `T3` | 8192 | 32768 |

Strict confirmation uses `2T`, exactly doubling both counts with a fresh namespace.
Linux timing on all three nodes chooses the largest tier for which the complete
preregistered schedule, divided by the nd-2/nd-3 166-core contingency capacity
and multiplied by a factor-two safety margin, is at most 58 hours. Any projected
trajectory above two hours eliminates its method. If T1 does not fit, the method
is `RUNTIME_EXHAUSTED`.

Resource choice reads timing only. It may not read a label, character, weight
estimate, `q_top`, or pass/fail physics outcome.

## 6. Raw evidence and characters

Every trajectory raw stores its full identity; initial, burn-end, and final state;
every fixed measurement state in little-endian bit-packed form; burn labels;
measurement labels, weights, residual or defect counts; eight time blocks;
kernel counters; catalog, joint-block, bias, character, source, config, registry,
and seed hashes; and timing. There are no parents, ancestry, resampling, or clones.

The analyzer opens NPZ with `allow_pickle=False`, rebuilds the frozen code and
disorder, reconstructs catalog/characters, reruns the trajectory from its seed,
and independently recomputes every label, weight, residual/defect, counter, and
digest. A missing/extra field or file, non-finite value, identity/hash mismatch,
algebra failure, or replay difference is `CONFLICT`.

The primary estimator is a character U-statistic:

- `k<=10`: all `2^k-1` nonzero characters;
- `k>10`: all basis characters plus 4096 frozen, uniformly drawn, distinct
  nonbasis `uint64` masks;
- `k=64`: masks and signatures remain `uint64`; bit 63 and the `2^64` boundary
  never pass through `np.int64`.

For each character, square its mean with cross-products of independent
trajectories. Combine delete-one-trajectory jackknife SE and character
finite-population SE in quadrature. Raw estimates are never clipped.

Initialization, resource, and method comparisons also estimate

```text
D2_norm = mean_{u != 0} (m_A(u)-m_B(u))^2
        = ||P_A-P_B||_2^2 / (1-2^-k).
```

Its within-family squares are independently debiased. Negative finite-sample
values remain negative.

An independent-trajectory raw-label collision estimate is retained only as a
diagnostic. It does not replace the character estimator and is not used to pass
any gate.

## 7. Statistical gates

Every family and comparison is fail-closed:

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

Trace gates cover energy, all basis characters, and 64 frozen diagnostic
nonbasis characters. If a measured character is constant while initial chains
include both signs, at least one opposite-sign chain must visit the common sign
during burn; otherwise this is common freezing, not convergence.

Defect trace additionally requires, per trajectory, at least 200 fixed-clock
`D=0` observations and 50 complete leave-return excursions. Median per-chain
conditional ESS must be at least 50, aggregate family conditional ESS at least
800, and `Dmax` boundary occupancy at most .10.

Initialization (`P/U`), `T/2T`, and the selected hard/defect mechanisms must all
pass the same distribution, weight, and `q_top` comparison gates.

## 8. Frozen panels

The canonical config stores ordered cells, uniform seeds, and their SHAs before
any result:

- `HARD2`: `m06_c00/.04/d00/attempt022`,
  `m08_c06/.04/d00/attempt022`;
- `EASY3`: `m03_c00/.10/d00`, `m04_c00/.07/d00`,
  `m05_c00/.10/d00`, all `global_fresh_v1`;
- `CONF17`: historical ordered SHA
  `8f2c1a6d60f346ecc5bf703f7e5d0d17d068462f978c78dd937ace0fb98b41be`;
- `RES6`: historical ordered SHA
  `03f9b16dbc0cc52ee18313cdf57fd25ea4db50f44687971bedac53662b275c22`;
- `GAP8`: m06_c00 and m08_c06 at `.05,.06,.08,.09`, d00,
  `global_fresh_v1`, ordered SHA
  `25c14dd7b5ddfc1725a6fdcd6629a70319ef97f020eaa583ae67d78a598b8aae`;
- `SMALL6`: m03_c00 and m04_c00 at `.04,.07,.10`, d00,
  `global_fresh_v1`, ordered SHA
  `018a52aa41153b36d9fc869d2f7f7308fa00258166b43f6404b713b117efe484`.

Fresh seeds are common across p for the same code and disorder index. They may
not be replaced after seeing a result.

## 9. Exact and portability gates

The mandatory small-code oracles use HGP codes from `H=[1,1,1]` (`n=10,k=4`)
and `H=[[1,1,0],[0,1,1]]` (`n=13,k=1`) at zero/nonzero syndrome and
`p=.04,.10,.25`. Tests cover affine uniformity, catalog laws, complete cluster
stationarity/detailed balance, joint-block stationarity/detailed balance,
extended worm stationarity and conditional posterior, character/collision/D2
statistics, jackknife/finite-population SE, reference/Numba transcript identity,
and the k=64 boundary.

All three Linux nodes must produce one canonical digest. Source, config,
registry, task, ownership, marker, catalog, bias, schedule, and seed tampering
must be rejected. Clean-source execution goes only through `run_verified_source.sh`.

The Linux preflight runs the complete exp102 and exp101 suites on nd-1/2/3,
then requires a common eligible-method set, the same selected resource tier,
and the same canonical transcript digest on all three nodes. Resource selection
uses the worst of the three timing reports, never a local macOS timing.

Weighted-model-counting on actual m3/m4 is feasibility evidence only and has a
two-hour solver limit per cell. Only exact values, rigorous interval width at
most .02, or explicit `(epsilon,delta)` bounds count. BP/Bethe/OSD and unbounded
tensor truncation are diagnostics, not certification. Fresh m3 `.04/.07/.10`
anchors must also agree with the already certified full-sector TI route.

## 10. Frozen 72-hour decision tree

1. Hours 0-8: three-node tests/digest, candidate runtime, and WMC feasibility.
   Any exact, transcript, or digest conflict stops as `CONFLICT`; a projected
   trajectory above two hours eliminates that method.
2. Hours 8-20: screen all hard and defect candidates on `HARD2+EASY3`. Select
   the fastest fully passing hard method and the passing defect method with the
   highest `D0 ESS/core-second`; exact ties prefer fewer attempts/smaller block
   or Dmax.
3. At hour 20: atomically freeze methods, tier, biases, and all later manifests.
   No new method, length, bias rule, disorder, or weaker gate may be added.
4. Hours 20-44: selected hard and defect methods run fresh T and 2T on HARD2.
   Both mechanisms and their mutual comparison must pass or full-range discovery
   stops.
5. Hours 44-66: both run CONF17+GAP8+SMALL6 at 2T and RES6 at fresh T; m3 also
   runs certified full-sector TI anchors.
6. Hours 66-72: retrieve, hash, independently replay/analyze, and report. There
   is no supplemental sampling, extension, or compressed confirmation.

Before each stage, load is rechecked and node ownership is frozen. Tasks never
migrate within a stage. If the factor-two remaining-wall projection exceeds the
remaining schedule, stop `RUNTIME_EXHAUSTED`. An early failure may spend at most
12 hours on the preregistered GAP8 diagnostic boundary, labeled
`DIAGNOSTIC_BOUNDARY`; it cannot certify a reduced range.

The hour-20 freeze is a three-part chain: method selection, a postselection plan
fixing every method/tier and ordered panel, and a control index fixing all bias
and TI manifests. Measurement manifests may be materialized only after their
frozen bias raw exists; they must bind both the postselection-plan SHA and the
control-freeze SHA. Final readiness re-hashes every frozen control, requires
those bindings on all three measurement reports, and checks the TI report
against the frozen TI-manifest SHA. Analyzer replay may use an explicit local
`--num-workers N`; every worker still reconstructs and replays every assigned
trajectory, and the default remains one worker.

## 11. Final interpretation

Only one hard-coset global method and one defect-trace method passing HARD2,
CONF17, RES6, GAP8, SMALL6, all mutual/resource/initialization gates, and the m3
TI anchors yields `READY_FOR_FORMAL`. This is not a physical result.

Hard-cell failure is `UNRESOLVED_WITHIN_72H` for those exact disorders and makes
the corresponding `(m,p)` fail-closed. A GAP8 failure applies only to that
sentinel; it is not extrapolated to neighboring p. Passing `.05-.10` sentinels
only justifies later formal tuning/held-out work.

If no complete pair succeeds, the full-range outcome is
`UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`, never `IMPOSSIBLE`. Any later
reduced-p or reduced-m study needs a fresh scientific contract, tuning, and
held-out panel; difficult disorders cannot be deleted or relabeled as success.
