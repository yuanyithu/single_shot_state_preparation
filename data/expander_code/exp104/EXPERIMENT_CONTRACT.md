# exp104 experiment contract

Experiment identity `exp104.ensemble_mc.v1`. Frozen before any production task
runs. Everything below is preregistered; nothing in it may be changed after
outcomes are seen.

## 1. Question

At `q = 0` code capacity, does the ensemble-average block logical failure rate
of one frozen BP+OSD-0 decoder cross as the hypergraph-product code grows, and
where?

The estimand is a property of a *random* code, not of any particular code:

> `P_fail(m, p)` is the probability that a code drawn uniformly from the exp104
> ensemble at parameter `m`, decoded by the frozen decoder, fails to correct a
> single i.i.d. bit-flip error pattern at physical rate `p`.

It is estimated by the pooled failure fraction over all trials at `(m, p)`.

## 2. Why exp104 exists

exp103 measured the same decoder on 48 frozen codes, eight per `m`, with 10,000
trials per code-p, and returned `EXP103_NO_CORRECT_CROSSING_IN_WINDOW` with a
simultaneous band of half-width 0.2601. Its own evidence explains why:

- The largest Monte Carlo standard error over its 624 cells was 0.0018 against a
  largest between-code standard deviation of 0.3245. Shot noise never bound.
- Eight of its 48 codes had classical distance 2 and failed 0.4051 of the time
  already at `p = 0.02`, a floor set by distance rather than by size.
- Those eight codes were spread unevenly over the six `m` panels by chance, so
  the panels were not comparable and the primary contrast was not a size
  comparison.

exp104 therefore spends its budget on codes instead of on trials per code. It
does not change the decoder, the noise model, the objective or the definition of
failure.

## 3. Ensemble definition

A candidate is a random (3,4)-biregular simple bipartite graph built by the
configuration model from a seed derived from `(master_seed, m, candidate_index)`;
multi-edges are rejected and the construction retried within one continuous RNG
stream, exactly as in exp101 and exp102.

A candidate is **accepted** if and only if:

1. `rank(H) = 3m` over GF(2), and
2. its parity-check matrix has not already been accepted at this `m`.

Both criteria are algebraic. **No criterion refers to distance, expansion,
decoder behaviour or any measured outcome, and no code is removed, reweighted or
replaced after its failure rate is known.** Candidates are scanned in index order
and the first `2000` accepted codes per `m` form the panel.

The acceptance rate is size dependent, measured at 0.723 for `m = 3` rising to
0.991 for `m = 8`, because low-`m` graphs are more often rank deficient. This is
part of the definition of the family being studied, not a correction applied
afterwards. The family is *full-rank* random (3,4)-biregular expander codes, for
which `n = 25m²` and `k = m²` exactly at every `m`.

Codes are reproduced from their seeds rather than stored. The registry records
provenance and structure only, and every stored hash is rechecked whenever a code
is rebuilt.

## 4. Measurement plan

| Quantity | Value |
|---|---|
| `m` | 3, 4, 5, 6, 7, 8 |
| `p` | 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10 |
| codes per `m` | 2000 (12,000 total) |
| trials per (code, `p`) | 4 |
| total trials | 432,000 |
| decoder | `ldpc.BpOsdDecoder`, `product_sum`, `serial`, natural order, `max_iter = n`, `osd_method = osd_0`, `osd_order = 0`, one OpenMP thread |

The decoder identity is byte for byte the frozen `exp103.decoder_mc.v2`
identity, so exp103's 624 cells remain comparable evidence. `max_iter = n` is the
dominant cost but changing it would change the physics.

The grid stops at `p = 0.10`. Above it every code fails essentially always, the
between-code standard deviation falls below 0.001, and belief propagation always
exhausts `max_iter`; exp103 spent roughly 60 percent of its compute there.

Four trials per code is the cost-optimal split, not a compromise: with a measured
per-code construction cost of about ten trials at `m = 8` and a between-code
standard deviation near 0.15 in the crossing region, the variance of the ensemble
mean at fixed budget is minimised at three to four trials per code.

One task covers a contiguous block of codes at one `m` across the whole `p` grid,
so each code's logical frame is built once. Block sizes are frozen per `m`.

## 5. Estimators and bands

- **Primary**: `P_fail(m, p)`, the pooled failure fraction over all `2000 x 4`
  trials at `(m, p)`. Equal weight, no reweighting, no trimming.
- **Primary contrast**: `Delta38(p) = P_fail(8, p) - P_fail(3, p)`.
- **Band**: cluster bootstrap resampling **codes**, each carrying its whole
  nine-point curve and all of its trials, 20,000 replicates at 95 percent. The
  band is **simultaneous across the nine grid points of `Delta38` only**. exp103
  took a maximum absolute deviation simultaneously over six curves, thirteen
  points and five adjacent contrasts, which is the main reason its half-width was
  0.2601.
- All six per-`m` curves and all five adjacent contrasts receive **pointwise**
  intervals and are labelled diagnostics.

## 6. Terminal decision

`EXP104_CERTIFIED_CROSSING` if and only if there exist grid points `p_a < p_b`
such that the simultaneous band lies entirely below zero at `p_a` and entirely
above zero at `p_b`. The reported bracket is `[a*, b*]` where `b*` is the
smallest certified-positive point preceded by any certified-negative point and
`a*` is the largest certified-negative point below `b*`.

Otherwise `EXP104_NO_CERTIFIED_CROSSING`.

**The bracket does not require adjacent grid points.** Requiring adjacency, as
exp103 did, forces the experiment to certify a contrast that vanishes at the
crossing, which no finite sample can do.

**The decision depends on `Delta38` and its simultaneous band alone.** No other
contrast, curve or stratum can change it, in either direction. This sentence is
the contract text and `crossing.classify_crossing` is its implementation; exp103
disclosed a gap between its prose and its code, and this contract states the
primary-only scope explicitly so that no such gap exists.

If and only if a bracket is certified, the crossing location `p_cross` is
reported: the first negative-to-positive linear interpolation of `Delta38` inside
the frozen bracket, with a percentile bootstrap interval from the same
replicates. Replicates showing no sign change inside the frozen sub-grid are
counted as undefined, and if fewer than 95 percent are defined the location is
reported as `NaN` with that reason attached.

## 7. Preregistered secondaries

Diagnostics. They cannot change the terminal status and are never published as
results: distance-stratified curves; the adjacent contrast family; the ensemble
composition census `f_d(m)`; per-`(m, p)` belief-propagation convergence rate and
mean iteration count; per-code failure counts.

Classical distance is recorded for every code because recording is not selecting.
It costs a `2^m` codeword enumeration and it is what makes the primary result
interpretable.

## 8. Fail-closed rules

- A task is `VALID` only if every planned code in it completed. Any other outcome
  is stored as `INVALID` evidence and is never rerun in place.
- An `(m, p)` cell with any `INVALID` task is `SAMPLING_INSUFFICIENT`; with any
  missing task it is `INCOMPLETE`. Either makes the run `INCOMPLETE` and every
  published statistic `NaN`.
- Unplanned or duplicate raw evidence aborts aggregation.
- Raw evidence is immutable: writing over an existing task is an error.

## 9. Replay gate

A preregistered random 10 percent of tasks per `m`, plus block 0 of every `m`
unconditionally, fixed by a seed derived before any production task runs and
recorded in the resource preflight. Because one task spans the whole `p` grid,
sampling tasks covers every `(m, p)` combination.

Replay reruns each selected task from its seeds through an independently
constructed decoder and requires bit-exact agreement on failure flags, logical
labels, syndrome match, convergence flags, iteration counts and all three stream
digests. **Any single mismatch invalidates the entire run.** The subsample is
never narrowed after the fact.

Sampling below full replay is admissible here only because determinism is
measured rather than assumed, as permanent discipline 15 requires:
`tests/test_decoder_determinism.py` is a resident regression gate that asserts
non-convergence is actually reached before asserting determinism inside it, it
runs in the local suite and again during nd-3 qualification, exp103 reproduced
2496 of 2496 shards bit for bit with this decoder identity on this node, and
Validation 002 replays frozen exp103 shards through this package.

## 10. Execution

Entry `ssh yuany`, compute host **nd-3 only**, conda environment as frozen in the
remote config, 64 workers passed explicitly, long runs under `screen`, run root
`~/.single_shot/runs/`. The clean-source tree runs only through
`run_verified_source.sh`, whose bytecode gate exits 67. A failed run is not
rerun in place.

Resource caps, preregistered: reserved core hours `<= 900`, predicted wall hours
`<= 16`, projected peak RSS `<= 128 GiB`, with reserve multiplier 2 applied to
generation plus replay plus analysis plus fixed overhead, per permanent
discipline 11. Projections are upper bounds built from the maximum over the
benchmarked grid points and the next anchor at or above each `m`; exp103's
equivalent bound over-predicted wall time by a factor 2.24, and the caps leave
room for that.

## 11. Authority and limits

exp104 reports a finite-grid, decoder-dependent, code-capacity result for one
frozen decoder on one randomly generated expander-code ensemble at `q = 0`.

It asserts no asymptotic threshold, no critical exponent, no finite-size-scaling
collapse, no `q_top`, no maximum-likelihood-decoding statement and no
preparation-channel claim. A certified crossing is a property of this decoder on
this ensemble over this finite grid.

**Complete success clears no exp102 blocker.** exp102 remains
`BLOCKED_BEFORE_REMOTE`, and nothing measured here authorizes any exp102 stage.

## 12. Scientific red team

Recorded per permanent discipline 12 in
`validation/001_contract_and_census_20260808/README.md`, including the target
distribution and its support, the estimand and deliverable, gate false-positive
and false-negative modes and their common failure, the size-dependent acceptance
rate as a potential selection effect, and the answer to "what would complete
success unlock" (nothing; see section 11).

Permanent discipline 4 does not apply: exp104 draws i.i.d. samples directly and
runs no Markov chain, so there is no slow variable, no basin and no barrier to
cross.
