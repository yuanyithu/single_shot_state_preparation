# exp106 experiment contract

Identity: `exp106.noisy_syndrome_mc.q001.v1`.

Preregistered. Every rule below is fixed before any measurement runs, and the
sections that a measurement is allowed to fill in say so explicitly.

## 1. Question

At readout error rate `q = 0.01`, does the ensemble-average block logical failure
rate of one frozen BP+OSD-0 decoder still cross as the hypergraph-product code
grows, and if so where?

The estimand is a property of a *random* code, not of any particular one:

> `P_fail(m, p; q)` is the probability that a code drawn uniformly from the
> exp106 ensemble at parameter `m` is assigned the wrong logical class by the
> frozen decoder, when one round of Z-check readout at error rate `q` is followed
> by i.i.d. data noise at rate `p` and one perfect final round.

The primary contrast is `Delta38(p) = P_fail(8, p) - P_fail(3, p)`.

## 2. Why exp106 exists

exp104 and exp105 bracketed this question without answering it.

| | q | result |
|---|---|---|
| exp104 | 0 | `EXP104_CERTIFIED_CROSSING`, `p_cross = 0.05512`, CI `[0.05327, 0.05699]` |
| exp105 | 0.05 | `EXP105_NO_CERTIFIED_CROSSING`; `Delta38` certified **positive at all 10 points**, negative at none |

So the readout threshold of this decoder on this family lies strictly inside
`(0, 0.05)`. exp105 recorded that in four places and said each time that locating
it "is a different experiment and needs its own contract; exp105 may not become
it." exp106 is that contract, at one interior point.

**The answer is genuinely open, and the design follows from that.** At `p = 0.04`,
moving `q` from `0` to `0.05` shifts `Delta38` by `+0.49`. If the readout penalty
were linear in `q`, `q = 0.01` contributes `+0.098` and erases the q=0 dip, whose
full depth is only `0.053`. If it goes as `q²` -- plausible, since an isolated
misread is cheap for the decoder to explain and a logical failure needs misreads
to conspire -- it contributes `+0.020` and the crossing survives, shallower and
shifted left. `q^1.5` gives `+0.044`, almost exactly the boundary. **Both terminal
states are real physical possibilities and neither is an experimental failure.**

### What exp106 is not

The originally requested observable, `q_top`, is **not measurable at `m >= 4`**
with the frozen certified instrument, for algebraic reasons exp105 measured
rather than argued (its Validation 001): the family has `k = m²`, so engine
routing sends every `m >= 4` to parallel tempering, whose validity gate requires
pooled worst-basis cold logical acceptance `>= 1e-4`, while the measured
`logical_X` basis weights put that acceptance between `1e-34` and `1e-81`. Every
disorder would be `INVALID` and every published statistic `NaN`.

exp106 measures the decoder-MAP failure rate of the same exp101 posterior
instead, which yields a **certified one-sided bound** `E[q_top] >= (1 - P_fail)²`
for large `k` (section 7). There is **no Track B anchor**: exp105 established that
full-sector TI cannot certify one at `q > 0`, and permanent discipline 13 forbids
adding budget to the same instrument.

## 3. Ensemble definition

Identical in rule and in seed derivation to exp104 and exp105.

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
and the first `C_m` accepted codes per `m` form the panel. `n = 25m²` and
`k = m²` exactly at every `m`, with `n_c = 12m²` Z-checks.

Codes are reproduced from their seeds rather than stored; the registry records
provenance and structure only, carries one `registry_sha256` over its canonical
byte serialization, and every stored hash is rechecked whenever a code is rebuilt.

`master_seed_hex` is fixed in the configs and is **neither exp104's nor
exp105's**: exp106 draws its own ensemble so that no code from either enters an
exp106 panel by construction. Both registries are still used, unmodified and
read-only, by the two equality gates of section 9.

The **locating pilot** draws from a further separate namespace and its own
registry file, so that no code which helps choose the frozen grid is later
measured on it. That registry carries all six sizes even though the pilot panel
is the primary pair only; the extra rows exist solely so the nd-3 cost benchmark
can time every size, and they are never scanned, aggregated or published.

## 4. Physics and scoring

One trial at `(m, code, p)` with `q = 0.01`:

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
   taken from exp101's `build_observable_frame`, which self-verifies that it
   annihilates the stabilizers and `im(r)` and pairs with the logical moves.
5. The trial **fails** iff `A (eps_hat xor eps) != 0`.

**What changes at `q > 0` is the criterion, not the label map.** For this family
`A` equals `logical_Z` exactly, because `r` places values only on the RREF pivot
columns of `H_Z` while exp101's `logical_Z` basis is supported entirely off those
columns. That is a measured structural property, asserted in
`tests/test_label_map.py`, and exp106 computes `A` through the certified frame
rather than assuming it.

The real change from exp104 is that exp104 also required the residual to have
zero syndrome. At `q > 0` the residual syndrome is `mu_hat xor mu`, which need not
vanish. Requiring it to vanish would be wrong: the protocol ends in a perfect
final round that measures the residual syndrome exactly and removes it, so a
residual with nonzero syndrome but trivial class is a success, and counting it as
a failure would charge for the readout channel twice.

The decoder is deterministic in the sense required by permanent discipline 15
**only if measured**. `tests/test_decoder_determinism.py` is a resident
regression gate that first asserts the augmented decoder actually exhausts
`max_iter` without converging at the production operating point, then asserts
bit-exact repetition; it runs in the local suite and again during nd-3
qualification.

## 5. Measurement plan

| Quantity | Value |
|---|---|
| `q` | `0.01`, fixed |
| `m` | 3, 4, 5, 6, 7, 8 |
| `p` grid | ten points, frozen by the section 6 rule before any production task |
| codes per `m` | frozen by the section 6 rule; unequal across `m` by design |
| trials per (code, `p`) | frozen by the section 6 rule, clipped to `[3, 6]` |

One task covers a contiguous block of codes at one `m` across the whole `p` grid,
so each code's logical frame and label matrix are built once and amortized over
the grid.

## 6. Locating pilot and the freezing rules

A pilot runs under the independent seed namespace `exp106.noisy_syndrome_mc.pilot.v1`
and its own registry file, at `m = 3, 8` only, over

```text
p in {0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03,
      0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07}
```

with 200 codes per `m` and four trials per (code, `p`). **Pilot raw is never
merged into production and never enters any published statistic, and no pilot
code is ever a production code.** Its sole function is to evaluate the rules
below, which are frozen here, before the pilot runs.

The grid is dense across `p in [0.02, 0.06]` on an argument fixed before anything
ran, and it is deliberately **not** exp105's. exp104 measured `Delta38` negative
exactly on `p in [0.02, 0.055]` at `q = 0`, with a crossing at `0.05512`; readout
noise can only move that window down and make it shallower. exp105's pilot grid
was log-spaced for a low-`p` regime and put seven of its fourteen points below
`0.016`, which is the wrong place to look for a residual dip at `q = 0.01`.

**Grid rule.** Let `[p_lo, p_hi]` be the innermost pair of pilot grid points at
which the pilot point estimate of `Delta38` changes sign from negative to
positive. The production grid is ten uniformly spaced points over
`[p_lo - 2h, p_hi + 2h]`, where `h` is the pilot grid spacing at `p_lo`, rounded
to four decimals, deduplicated, and clipped to `[0.0005, 0.10]`. If the pilot
estimate shows no negative-to-positive sign change anywhere, the production grid
is the frozen fallback

```text
{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.055, 0.07}
```

which covers the window exp104 measured negative. exp105's fallback reached down
to `0.001` on the argument that if the crossing is not where the pilot looked
then its order of magnitude is uncertain. That argument does not transfer: here
the crossing's location at `q = 0` is known to three significant figures, and the
only question is whether `q = 0.01` has erased it. A no-crossing terminal must
therefore mean "certified positive across the whole window where `q = 0` was
negative", which the fallback above can support and a low-`p` log grid cannot.

**Cost measurement.** `c_m`, the per-trial cost, and `kappa_m`, the per-code frame
cost, are measured **on nd-3** by `remote_cli cost-benchmark`, which is
outcome-blind and independent of the production plan. This is not a detail. exp105
evaluated the same allocation rule on costs measured on the macmini, and its nd-3
resource gate blocked at 5,367.8 reserved core-hours against a cap of 800, because
a trial at `m = 8` costs about eight times as much on nd-3. The rule spends a
budget of core-hours on the machine that runs it. `allocate` refuses any cost
report whose `device` is not `nd-3`.

**Allocation rule.** Let `sigma_c(m)` be the between-code standard deviation and
`sigma_w²(m)` the mean within-code trial variance, each taken at the pilot grid
point nearest the bracket -- or, when the grid rule falls back, at the pilot point
nearest the geometric centre of the production grid. Write

```text
s_m = sqrt(sigma_c(m)² + sigma_w²(m) / T_m),
```

the standard deviation of one code's *observed* rate, which is what actually
enters the variance of a panel mean. With generation budget `G = 800` core-hours
and `P` the number of production grid points:

- `T_m = clip(round(sqrt(kappa_m * sigma_w²(m) / (P * c_m * sigma_c(m)²))), 3, 6)`,
  taking the `T -> 6` limit when `sigma_c(m)` is zero.
- Each diagnostic size `m in {4,5,6,7}` receives `0.06 G`.
- The primary pair `m in {3,8}` splits the remaining `0.76 G` in the ratio
  `C_3 / C_8 = (s_3 / s_8) * sqrt(u_8 / u_3)`, where `u_m = kappa_m + P T_m c_m`
  is the per-code cost. This minimizes `Var(Delta38) = s_3²/C_3 + s_8²/C_8` at
  fixed cost.
- `C_m = floor(share_m / u_m)` rounded down to a multiple of the block size, which
  is itself the rule `codes_per_task[m] = max(1, floor(300 s / u_m))`.

**The `s`-form is preregistered here, not chosen later.** exp105's rule was
written in terms of raw `sigma_c` and degenerated into `0/0` when the pilot
measured `sigma_c` at or below its own resolution -- which is what happened at
`q = 0.05`, where failure is driven by a readout channel common to every code.
exp105 had to substitute this same form mid-flight. At `q = 0.01` the channel is
five times weaker and `sigma_c` may well be recoverable, but which regime applies
must not be a decision made after seeing the pilot. `s_m` reduces to `sigma_c(m)`
when the between-code spread dominates and to the shot-noise term when it does
not, so one formula covers both.

Unequal `C_m` is deliberate. A code at `m = 8` costs about fifty times a code at
`m = 3` on nd-3, so equal panel sizes would spend most of the budget on the
smaller of the two variance terms.

If the measured costs imply that the allocation rule cannot be satisfied within
the section 10 caps, the experiment stops before production and reports that,
rather than reducing the panel silently.

## 7. Estimators, bands and the bound on `q_top`

- **Primary**: `P_fail(m, p)`, the pooled failure fraction over all trials at
  `(m, p)`. Equal weight, no reweighting, no trimming.
- **Primary contrast**: `Delta38(p) = P_fail(8, p) - P_fail(3, p)`.
- **Band**: cluster bootstrap resampling **codes** within each `m`, each code
  carrying its whole curve and all of its trials, 20,000 replicates at 95
  percent. The band is **simultaneous across the grid points of `Delta38` only**.
- All per-`m` curves and all adjacent contrasts receive **pointwise** intervals
  and are labelled diagnostics.

**Bound on the requested observable.** For the exact posterior at one disorder,
`map_success_probability <= sqrt(posterior_purity)` (`PHYSICS_CONTRACT.md` §8),
and no decoder can exceed MAP success at its own observation. Writing
`S = 1 - P_fail` and `M = 2^k`,

```text
E[q_top] >= (M * S² - 1) / (M - 1),
```

by Jensen's inequality on `purity >= map_success²`. For `k = m² >= 9` the
right-hand side is `S²` to within `2^-9`. This is a **certified lower bound on
the disorder-averaged `q_top`**, not an estimate of it, and it is informative only
where `S` is large. exp106 reports it as such and makes no upper-bound claim. It
remains **uncalibrated**, for the reason recorded in exp105 Validation 007: the
instrument that would calibrate it cannot be certified at `q > 0`.

## 8. Preregistered secondaries

Diagnostics. They cannot change the terminal status and are never published as
results: distance-stratified curves; the adjacent contrast family; the ensemble
composition census `f_d(m)` including `m = 2`; per-`(m, p)` belief-propagation
convergence rate and mean iteration count; the split of failures into those with
`mu_hat = mu` and those without; per-code failure counts.

Classical distance is recorded for every code because recording is not selecting.

## 9. Fail-closed rules and gates

- A task is `VALID` only if every planned code in it completed. Any other outcome
  is stored as `INVALID` evidence and is never rerun in place.
- An `(m, p)` cell with any `INVALID` task is `SAMPLING_INSUFFICIENT`; with any
  missing task it is `INCOMPLETE`. Either makes the run `INCOMPLETE` and every
  published statistic `NaN`.
- Unplanned or duplicate raw evidence aborts aggregation.
- Raw evidence is immutable: writing over an existing task is an error.
- Raw carrying a foreign `q` is an unexpected error and poisons the run. **No
  filename anywhere encodes `q`** -- exp104, exp105 and exp106 raw files, configs
  and aggregates are name-identical -- so this check and the run-root naming are
  the only things keeping three experiments' evidence apart.

**Replay gate.** A preregistered random 10 percent of tasks per `m`, plus block 0
of every `m` unconditionally, fixed by a seed derived before any production task
runs and recorded in the resource preflight. Replay reruns each selected task
from its seeds through an independently constructed decoder and an independently
reconstructed logical criterion, and requires bit-exact agreement on failure
flags, logical labels, readout-match flags, convergence flags, iteration counts
and all four stream digests. **Any single mismatch invalidates the entire run.**
The subsample is never narrowed after the fact.

**exp104 equality gate, at `q = 0`.** Before any exp106 production task runs,
exp106 and exp104 must be shown to be the same function with the readout channel
off: for codes drawn from exp104's frozen registry, both packages build the model,
both decode `H_Z` with the exp104 decoder identity, and both score the residual.
Byte-identical `H_Z`, `H_X` and logical frames, identical corrections and
iteration counts, identical verdicts and labels. That the two scoring criteria
agree at `q = 0` is an identity, not a coincidence, so a disagreement is a bug.

**exp105 equality gate, at `q = 0.05`.** The stronger of the two, and the one that
actually covers the port. The exp104 comparison cannot reach the augmented matrix,
the mixed error channel, the readout draw or the `q > 0` failure criterion,
because at `q = 0` none of them is exercised. exp105 ran all of it for 1,057,020
trials, so exp106 must reproduce it **bit for bit** on exp105's own frozen
production registry and its own `q`: identical corrections, iteration counts,
logical labels, readout-match flags and verdicts. Here the two packages are meant
to be the same function outright, not to agree in a limit.

Both experiments' production raw lives on nd-3 and is not tracked in Git, so both
gates are package-to-package comparisons rather than replays of stored files --
the same form exp104's own Validation 002 used against exp103. Neither predecessor
is rerun; their evidence is read-only.

## 10. Execution

Entry `ssh yuany`, compute host **nd-3 only**, conda environment as frozen in the
remote config, **72 workers** passed explicitly, long runs under `screen`, run
root `~/.single_shot/runs/`. The clean-source tree runs only through
`run_verified_source.sh`, whose bytecode gate exits 67. A failed run is not rerun
in place.

72 rather than the 64 exp103/104/105 inherited: nd-3 has 96 logical CPUs on 48
physical cores, and another user has held ten cores continuously, leaving 76 free
hyperthreads.

The compiled decoder is not bit-portable across platforms (exp104 Validation 002).
exp106 generates, replays and aggregates entirely on nd-3 against the pinned nd-3
binary SHA256, and never mixes artifacts across platforms. The locating pilot's
*statistics* run locally, which is sound because they choose a grid rather than
report a result; its *costs* do not, per section 6.

Resource caps, preregistered, with reserve multiplier 2 applied to generation plus
replay plus analysis plus fixed overhead, per permanent discipline 11:

| quantity | cap |
|---|---:|
| reserved core hours | 1800 |
| predicted wall hours | 20 |
| projected peak RSS | 128 GiB |

These follow from the frozen `G = 800` core-hour generation budget:
`2 x (800 + 80 + 1 + 1) = 1764` and `(800 + 80) / 72 + 2 = 14.2`. The expected
production scan is about 9 wall hours.

## 11. Terminal decision

`EXP106_CERTIFIED_CROSSING` if and only if there exist grid points `p_a < p_b`
such that the simultaneous band lies entirely below zero at `p_a` and entirely
above zero at `p_b`. The reported bracket is `[a*, b*]` where `b*` is the smallest
certified-positive point preceded by any certified-negative point and `a*` is the
largest certified-negative point below `b*`. The bracket does not require adjacent
grid points.

Otherwise `EXP106_NO_CERTIFIED_CROSSING`.

**The decision depends on `Delta38` and its simultaneous band alone.** No other
contrast, curve, stratum or bound can change it, in either direction. This
sentence is the contract text and `crossing.classify_crossing` is its
implementation.

If and only if a bracket is certified, the crossing location `p_cross` is
reported: the first negative-to-positive linear interpolation of `Delta38` inside
the frozen bracket, with a percentile bootstrap interval from the same replicates.
Replicates showing no sign change inside the frozen sub-grid are counted as
undefined, and if fewer than 95 percent are defined the location is reported as
`NaN` with that reason attached.

**Both terminals are results.** A certified crossing narrows the readout threshold
to `q_c in (0.01, 0.05)` and gives a second point on the threshold curve; a
certified absence narrows it to `q_c in (0, 0.01)`. A band too wide to certify
either is the one outcome that is a failure of the measurement rather than a
statement about the physics, and it is reported as such rather than dressed up.

## 12. Authority and limits

exp106 reports a finite-grid, decoder-dependent result for one frozen decoder on
one randomly generated expander-code ensemble at `q = 0.01`, plus a certified
one-sided lower bound on the disorder-averaged `q_top` of the exp101 posterior.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no maximum-likelihood-decoding statement, no `q_top` **estimate** at
`m >= 4`, and no preparation-channel claim. It locates no threshold *curve*: two
points and a bracket are not a curve.

**Complete success clears no exp102 blocker.** exp102 remains
`BLOCKED_BEFORE_REMOTE` with all four blockers open, and nothing measured here
authorizes any exp102 stage.

## 13. Scientific red team

Recorded per permanent discipline 12 in
`validation/001_contract_and_redteam_20260812/README.md`.

Permanent discipline 4 does not apply: exp106 draws i.i.d. samples directly and
runs no Markov chain, so there is no slow variable, no basin and no barrier to
cross. Permanent discipline 13 is why there is no Track B.
