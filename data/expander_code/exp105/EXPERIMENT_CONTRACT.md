# exp105 experiment contract

Experiment identity `exp105.noisy_syndrome_mc.v1`. Frozen before any production
task runs. Everything below is preregistered; nothing in it may be changed after
outcomes are seen.

## 1. Question

At readout error rate `q = 0.05`, does the ensemble-average block logical failure
rate of one frozen BP+OSD-0 decoder cross as the hypergraph-product code grows,
and where?

The estimand is a property of a *random* code, not of any particular code:

> `P_fail(m, p; q)` is the probability that a code drawn uniformly from the
> exp105 ensemble at parameter `m` is assigned the wrong logical class by the
> frozen decoder, when one round of Z-check readout at error rate `q` is followed
> by i.i.d. data noise at rate `p` and one perfect final round.

It is estimated by the pooled misclassification fraction over all trials at
`(m, p)`.

## 2. Why exp105 exists, and what it is not

The originally requested observable was `q_top`, the normalized logical-sector
purity of the exp101 reduced-MLD posterior (`exp101/PHYSICS_CONTRACT.md` §8). For
this code family that observable is **not measurable at `m >= 4` with the frozen
certified instrument**, for a reason that is algebraic rather than a matter of
tuning:

- The family has `n = 25m²` and `k = m²`, so engine routing (`exp101/src/run_scan.py`,
  `resolve_engine`) gives full-sector thermodynamic integration only for `k <= 10`,
  that is `m <= 3`. Every `m >= 4` is routed to four-instance parallel tempering.
- The PT validity gate (`exp101/src/gates.py`, `evaluate_pt_convergence_gate`)
  requires pooled worst-basis cold logical acceptance `>= 1e-4`. A cold L-move is
  a bare flip of one logical basis operator with `dE = K_p * d|v|`
  (`exp101/src/reference_mcmc.py`), so its acceptance is
  `~ exp(-K_p (1 - 2p) w)` in the basis weight `w`.
- The measured `logical_X` basis weights of this family are, as min/median/max:
  `m=3` 2/6/24, `m=4` 8/22/30, `m=5` 4/10/50, `m=6` 8/30/48, `m=7` 8/20/36,
  `m=8` 12/45/70. At `p = 0.05`, `K_p = 2.944`, so the heaviest basis direction at
  `m = 6` has acceptance about `e^-79`. Even a distance-optimal basis
  (`d = 6..8` here) gives about `1e-7`, against a gate of `1e-4`.

Therefore every `m >= 4` disorder would be `INVALID`, and scan v3 fail-closed
aggregation would set every point's mean, SEM and crossing input to `NaN`. This
is the same wall exp102 has been at for 66 validations. exp105 does not attempt
to move it.

exp105 instead measures the decoder-MAP failure rate of the *same* posterior. The
connection to the requested observable is a one-sided bound, not a substitute
estimate, and it is stated in section 7.

## 3. Ensemble definition

Identical in rule and in seed derivation to exp104, extended to `m = 2`.

A candidate is a random (3,4)-biregular simple bipartite graph built by the
configuration model from a seed derived from `(master_seed, m, candidate_index)`;
multi-edges are rejected and the construction retried within one continuous RNG
stream.

A candidate is **accepted** if and only if:

1. `rank(H) = 3m` over GF(2), and
2. its parity-check matrix has not already been accepted at this `m`.

Both criteria are algebraic. **No criterion refers to distance, expansion,
decoder behaviour or any measured outcome, and no code is removed, reweighted or
replaced after its failure rate is known.** Candidates are scanned in index order
and the first `C_m` accepted codes per `m` form the panel. The family is
*full-rank* random (3,4)-biregular expander codes, for which `n = 25m²` and
`k = m²` exactly at every `m`.

Codes are reproduced from their seeds rather than stored. The registry records
provenance and structure only, and every stored hash is rechecked whenever a code
is rebuilt. Because exp105 uses of order 10^5 codes, the registry is a compact
columnar file rather than JSON, carrying one `registry_sha256` over its canonical
byte serialization.

`master_seed_hex` is fixed in `config/noisy_mc.v1.json` and is **not** exp104's
master seed: exp105 draws its own ensemble so that no exp104 code enters exp105's
panels by construction. The exp104 registry is still used, unmodified and
read-only, by the cross-validation gate of section 9.

## 4. Physics and scoring

One trial at `(m, code, p)` with `q = 0.05`:

1. `eps ~ Bernoulli(p)^n` data error, `mu ~ Bernoulli(q)^{n_c}` readout error on
   the `n_c` Z-checks, drawn from one continuous seeded stream.
2. `y_eff = H_Z eps xor mu`. This is exactly `effective_syndrome` as defined in
   `exp101/PHYSICS_CONTRACT.md` §3, with `H_check = H_Z`, `sector = x_error`,
   prepared state `|+>_L`.
3. Decode `H_aug = [H_Z | I_{n_c}]` with `error_channel = [p]*n + [q]*n_c`,
   `bp_method = product_sum`, `schedule = serial`, natural serial order,
   `osd_method = osd_0`, `osd_order = 0`, `max_iter = n + n_c`, one OpenMP thread.
   Take `eps_hat = c_hat[:n]`.
4. Label with the exp101 absolute logical label
   `phi_r(e) = logical class of [e xor r(H_Z e)]` (`PHYSICS_CONTRACT.md` §7),
   where `r` is the production `LinearSection` built by
   `exp101/src/model.py::assemble_sector_model`. Because `r` is linear, `phi_r`
   is the GF(2) map `e -> A e` for a `k x n` matrix `A` built once per code, and
   `A` is taken from exp101's `build_observable_frame`, which self-verifies that
   it annihilates the stabilizers and `im(r)` and pairs with the logical moves.
5. The trial **fails** iff `A (eps_hat xor eps) != 0`.

**What changes at `q > 0` is the criterion, not the label map.** For this family
`A` equals `logical_Z` exactly: `r` places values only on the RREF pivot columns
of `H_Z`, and exp101's `logical_Z` basis is supported entirely off those columns,
so the section term vanishes. That is a measured structural property, asserted in
`tests/test_label_map.py`, and exp105 computes `A` through the certified frame
rather than assuming it.

The real change is that exp104 also required the residual to have zero syndrome.
At `q > 0` the residual syndrome is `H_Z (eps_hat xor eps) = mu_hat xor mu`,
which need not vanish. Requiring it to vanish would be wrong here: the protocol
ends in a perfect final round that measures the residual syndrome exactly and
removes it, so a residual with nonzero syndrome but trivial class is a success,
and counting it as a failure would charge for the readout channel twice. Only the
logical class survives, which is exactly what `PHYSICS_CONTRACT.md` §8 defines
MAP success to be. At `q = 0` the syndrome always matches and the two criteria
coincide, which is what makes the section 9 equality gate meaningful.

`max_iter = n + n_c` is the only decoder-spec difference from exp104 and it is
the augmented block length, the same rule exp104 applied to its own block length.
The `q = 0` comparison of section 9 runs the unaugmented exp104 path, so exp104
comparability is unaffected.

The decoder is deterministic in the sense required by permanent discipline 15 only
if measured. `tests/test_decoder_determinism.py` is a resident regression gate
that first asserts the augmented decoder actually exhausts `max_iter` without
converging at the production operating point, then asserts bit-exact repetition;
it runs in the local suite and again during nd-3 qualification.

## 5. Measurement plan

| Quantity | Value |
|---|---|
| `q` | `0.05`, fixed |
| `m` | 3, 4, 5, 6, 7, 8 (Track A); 2, 3 (Track B anchor) |
| `p` grid | ten points, frozen by the section 6 rule before any production task |
| codes per `m` | frozen by the section 6 rule; unequal across `m` by design |
| trials per (code, `p`) | frozen by the section 6 rule, clipped to `[3, 6]` |

One task covers a contiguous block of codes at one `m` across the whole `p` grid,
so each code's logical frame and label matrix are built once and amortized over
the grid.

## 6. Locating pilot and the freezing rules

A pilot runs under the independent seed namespace `exp105.pilot.v1`, drawing its
codes from the independent ensemble namespace `exp105.noisy_syndrome_mc.pilot.v1`
and its own registry file, at `m = 3, 8` only, over

```text
p in {0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015,
      0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07}
```

with 200 codes per `m` and four trials per (code, `p`). **Pilot raw is never
merged into production and never enters any published statistic, and no pilot
code is ever a production code.** Its sole function is to evaluate the two rules
below, which are frozen here, before the pilot runs.

The grid reaches down to `0.001` on an argument fixed before anything ran. exp104
put the `q = 0` crossing at `p = 0.05512`, and readout noise can only move it
down. If the threshold curve is even roughly linear near the axes and the two
axes have comparable scales, `q = 0.05` leaves a `p` budget of order `0.005`. A
pilot that cannot bracket the crossing wastes the production run it exists to
plan, so the low end is cheap insurance rather than a guess at the answer.

**Grid rule.** Let `[p_lo, p_hi]` be the innermost pair of pilot grid points at
which the pilot point estimate of `Delta38 = P_fail(8) - P_fail(3)` changes sign
from negative to positive. The production grid is ten uniformly spaced points
over `[p_lo - 2h, p_hi + 2h]`, where `h` is the pilot grid spacing at `p_lo`,
rounded to four decimals, deduplicated, and clipped to `[0.0005, 0.10]`. If the
pilot estimate shows no negative-to-positive sign change anywhere, the production
grid is the frozen log-spaced fallback
`{0.001, 0.0015, 0.0025, 0.004, 0.006, 0.01, 0.016, 0.025, 0.04, 0.07}`, chosen
that way because if the crossing is not where the pilot looked then its order of
magnitude is what is uncertain, not its third decimal.

**Allocation rule.** Let `c_m` be the measured per-trial cost, `kappa_m` the
measured per-code frame cost in seconds, `sigma_c(m)` the between-code standard
deviation and `sigma_w²(m)` the mean within-code trial variance, each taken at the
pilot grid point nearest the bracket. With generation budget `G = 290` core-hours
and `P = 10`:

- `T_m = clip(round(sqrt(kappa_m * sigma_w²(m) / (P * c_m * sigma_c²(m)))), 3, 6)`.
- Each diagnostic size `m in {4,5,6,7}` receives `0.06 G`.
- The primary pair `m in {3,8}` splits the remaining `0.76 G` in the ratio
  `C_3 / C_8 = (sigma_c(3) / sigma_c(8)) * sqrt(u_8 / u_3)`, where
  `u_m = kappa_m + P T_m c_m` is the per-code cost, which minimizes the variance
  of `Delta38` at fixed cost.
- `C_m = floor(share_m / u_m)` rounded down to a multiple of the frozen block size.

Unequal `C_m` is deliberate. The variance of the primary contrast is
`sigma_c²(3)/C_3 + sigma_c²(8)/C_8` while a code at `m = 8` costs about seventy
times a code at `m = 3`, so equal panel sizes spend most of the budget on the
smaller of the two variance terms.

If the pilot's measured `c_m` imply that the allocation rule cannot be satisfied
within the section 10 caps, the experiment stops before production and reports
that, rather than reducing the panel silently.

## 7. Estimators, bands and the bound on `q_top`

- **Primary**: `P_fail(m, p)`, the pooled failure fraction over all trials at
  `(m, p)`. Equal weight, no reweighting, no trimming.
- **Primary contrast**: `Delta38(p) = P_fail(8, p) - P_fail(3, p)`.
- **Band**: cluster bootstrap resampling **codes** within each `m`, each code
  carrying its whole ten-point curve and all of its trials, 20,000 replicates at
  95 percent. The band is **simultaneous across the ten grid points of `Delta38`
  only**.
- All per-`m` curves and all adjacent contrasts receive **pointwise** intervals
  and are labelled diagnostics.

**Bound on the requested observable.** For the exact posterior at one disorder,
`map_success_probability <= sqrt(posterior_purity)` (`PHYSICS_CONTRACT.md` §8),
and no decoder can exceed MAP success at its own observation. Writing
`S = 1 - P_fail` for the ensemble-mean decoder success rate and `M = 2^k`,

```text
E[q_top] >= (M * S² - 1) / (M - 1),
```

by Jensen's inequality on `purity >= map_success²`. For `k = m² >= 9` the
right-hand side is `S²` to within `2^-9`. This is a **certified lower bound on
the disorder-averaged `q_top`**, not an estimate of it, and it is informative only
where `S` is large. exp105 reports it as such and makes no upper-bound claim.

## 8. Preregistered secondaries

Diagnostics. They cannot change the terminal status and are never published as
results: distance-stratified curves; the adjacent contrast family; the ensemble
composition census `f_d(m)` including `m = 2`; per-`(m, p)` belief-propagation
convergence rate and mean iteration count; the split of failures into those with
`mu_hat = mu` and those without; per-code failure counts.

Classical distance is recorded for every code because recording is not selecting.

**Track B, the `m = 2, 3` `q_top` anchor**, is a preregistered secondary. It uses
exp101 `run_sector_ti` at `k <= 10`, which integrates each logical sector's free
energy separately and therefore requires no logical transport, so the section 2
gate does not bind it. It carries its own `exp105.anchor.raw.v1` schema, its own
fail-closed aggregation with zero tolerance for invalid or missing disorders, and
its own loader. It never produces `scan_results.npz`, never claims
`exp101.scan.v3`, and is never passed to `src.scan_results.load_publication_q_top`.
It asserts no threshold and performs no finite-size scaling. **Track B cannot
change Track A's terminal status in either direction.**

## 9. Fail-closed rules and gates

- A task is `VALID` only if every planned code in it completed. Any other outcome
  is stored as `INVALID` evidence and is never rerun in place.
- An `(m, p)` cell with any `INVALID` task is `SAMPLING_INSUFFICIENT`; with any
  missing task it is `INCOMPLETE`. Either makes the run `INCOMPLETE` and every
  published statistic `NaN`.
- Unplanned or duplicate raw evidence aborts aggregation.
- Raw evidence is immutable: writing over an existing task is an error.

**Replay gate.** A preregistered random 10 percent of tasks per `m`, plus block 0
of every `m` unconditionally, fixed by a seed derived before any production task
runs and recorded in the resource preflight. Replay reruns each selected task
from its seeds through an independently constructed decoder and requires bit-exact
agreement on failure flags, logical labels, convergence flags, iteration counts
and all four stream digests. **Any single mismatch invalidates the entire run.**
The subsample is never narrowed after the fact.

**exp104 equality gate.** Before any exp105 production task runs, exp105 and
exp104 must be shown to be the same function at `q = 0`. exp104's production raw
lives on nd-3 and is not tracked in Git, so this is a package-to-package
comparison on shared codes and shared seeds, the same form exp104's own
Validation 002 used against exp103: for codes drawn from exp104's frozen
registry, both packages build the model, both decode `H_Z` with the exp104
decoder identity, and both score the residual. The gate requires byte-identical
`H_Z`, `H_X` and logical frames, identical corrections and iteration counts, and
identical verdicts and labels.

That the two scoring criteria agree at `q = 0` is an identity, not a
coincidence: with the readout channel off the correction reproduces the syndrome
exactly, the residual lies in `ker(H_Z)`, the section term in `phi_r` drops out
and exp104's residual pairing is exactly `phi_r`. A disagreement is therefore a
bug, and the gate is a real one. exp104's own evidence is read-only and is
never rerun.

## 10. Execution

Entry `ssh yuany`, compute host **nd-3 only**, conda environment as frozen in the
remote config, 64 workers passed explicitly, long runs under `screen`, run root
`~/.single_shot/runs/`. The clean-source tree runs only through
`run_verified_source.sh`, whose bytecode gate exits 67. A failed run is not rerun
in place.

The compiled decoder is not bit-portable across platforms (exp104 Validation 002).
exp105 generates, replays and aggregates entirely on nd-3 against the pinned nd-3
binary SHA256, and never mixes artifacts across platforms.

Resource caps, preregistered, with reserve multiplier 2 applied to generation plus
replay plus analysis plus fixed overhead, per permanent discipline 11:

| quantity | cap |
|---|---:|
| reserved core hours | 800 |
| predicted wall hours | 14 |
| projected peak RSS | 128 GiB |

## 11. Terminal decision

`EXP105_CERTIFIED_CROSSING` if and only if there exist grid points `p_a < p_b`
such that the simultaneous band lies entirely below zero at `p_a` and entirely
above zero at `p_b`. The reported bracket is `[a*, b*]` where `b*` is the smallest
certified-positive point preceded by any certified-negative point and `a*` is the
largest certified-negative point below `b*`. The bracket does not require adjacent
grid points.

Otherwise `EXP105_NO_CERTIFIED_CROSSING`. **At `q = 0.05` the absence of a
crossing is a real physical possibility and a legitimate terminal state**, not a
failure of the experiment.

**The decision depends on `Delta38` and its simultaneous band alone.** No other
contrast, curve, stratum, Track B result or bound can change it, in either
direction. This sentence is the contract text and `crossing.classify_crossing` is
its implementation.

If and only if a bracket is certified, the crossing location `p_cross` is
reported: the first negative-to-positive linear interpolation of `Delta38` inside
the frozen bracket, with a percentile bootstrap interval from the same replicates.
Replicates showing no sign change inside the frozen sub-grid are counted as
undefined, and if fewer than 95 percent are defined the location is reported as
`NaN` with that reason attached.

## 12. Authority and limits

exp105 reports a finite-grid, decoder-dependent result for one frozen decoder on
one randomly generated expander-code ensemble at `q = 0.05`, plus a certified
one-sided lower bound on the disorder-averaged `q_top` of the exp101 posterior,
plus a transport-free `q_top` anchor at `m = 2, 3`.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no maximum-likelihood-decoding statement, no `q_top` **estimate** at
`m >= 4`, and no preparation-channel claim.

**Complete success clears no exp102 blocker.** exp102 remains
`BLOCKED_BEFORE_REMOTE` with all four blockers open, and nothing measured here
authorizes any exp102 stage.

## 13. Scientific red team

Recorded per permanent discipline 12 in
`validation/001_contract_and_redteam_20260811/README.md`.

Permanent discipline 4 does not apply to Track A: it draws i.i.d. samples directly
and runs no Markov chain, so there is no slow variable, no basin and no barrier to
cross. It does apply to Track B, and Track B's answer is recorded in Validation
001: full-sector TI integrates each logical sector's free energy under a
sector-preserving chain at fixed label, so no trajectory is ever required to cross
a logical or collapsed-B barrier; the barrier appears as a free-energy difference
between independently computed sectors rather than as a transport requirement.
