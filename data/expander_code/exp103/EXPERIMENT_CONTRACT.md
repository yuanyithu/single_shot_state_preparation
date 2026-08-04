# exp103 BpLSD decoder-MC experiment contract

Contract identity: `exp103.decoder_mc.v1`. This experiment is independent of the
exp102 posterior-sampler program. It estimates only the finite-size block logical
failure rate of one frozen BpLSD decoder under code-capacity X noise.

## Scientific target and authority

- Noise is `sector=x_error`, `H_check=H_Z`, perfect syndrome, `q=0`, with each
  physical error bit drawn independently from `Bernoulli(p)`.
- The panel is all 48 codes in exp102 registry SHA
  `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`.
  Each `m=3,...,8` panel contains eight equal-weight codes. No member, including
  any of the eight frozen `d=2` codes, may be removed or downweighted.
- The grid is exactly `p=0.02,0.03,...,0.14`. Each code-p has four immutable
  shards of 2,500 fresh trials, hence 10,000 trials per code-p and 6,240,000
  trials in the complete experiment.
- The primary is the equal-weight mean of the eight code-level block failure
  rates at each m-p. Medians, code curves, distance strata, BP diagnostics and
  between-code dispersion are secondary and cannot change the primary status.
- Version 1 may report a certified finite-grid crossing bracket. It must not fit
  an asymptotic threshold, critical exponent, FSS collapse, `q_top`, MLD, or a
  preparation-channel claim.

## Frozen decoder and outcome

The worker imports `ldpc.BpLsdDecoder` directly and provides `error_rate=p`,
`bp_method=product_sum`, `max_iter=n`, `schedule=serial`, the natural serial
order, `lsd_method=LSD_CS`, `lsd_order=0`, `bits_per_step=1`,
`always_run_lsd=false`, and `omp_thread_count=1`. There is no section decoder,
linear fallback, alternate decoder, or trial redraw.

For `y=H_Z e`, the decoder returns `e_hat` and the residual is
`r=e xor e_hat`. A trial fails when either `H_Z r != 0` or
`logical_Z @ r != 0`. `decoder.converge=false` remains a valid decoder outcome.
A legal correction with mismatched syndrome is a block failure and a separate
diagnostic. An exception or illegal correction shape, dtype, or value invalidates
the complete shard and is saved; it is never dropped or replaced.

The canonical environment is macmini (runtime hostname `ymini.local`), active
conda environment `12`, Python 3.12.12, NumPy 2.4.1, SciPy 1.17.0 and ldpc 2.4.1.
Runtime validation also requires the active conda prefix to be the prefix from
which Python is running. The config additionally freezes the BpLSD extension
SHA, source commit, source-tree SHA, config SHA and registry SHA. Formal
statistics run only in that identity. Raw and replay evidence attest the device,
hostname and conda identity that executed them.

## Seeds, raw data and replay

Benchmark, measurement, replay-control and bootstrap namespaces are distinct.
Measurement seeds are SHA256-derived from the fresh master seed, measurement
namespace, registry SHA, code ID, normalized p token and shard index. Task order,
worker count and resume order therefore cannot change a trial stream.

`exp103.raw.v1` stores identity fields, per-trial failure, logical label,
syndrome-match, BP convergence and BP iteration fields, plus SHA256 digests of
the error, correction and label streams. Errors need not be stored because the
frozen seed and NumPy identity reconstruct them exactly. Failed raw is immutable;
formal output is never overwritten in place.

Full replay regenerates every measurement error, invokes BpLSD directly again,
and uses audit code that does not import the worker scorer. It compares all trial
fields and all three stream hashes. Replay is invoked separately on `raw/stage1`
and `raw/stage2`; the final aggregate binds both immutable stage manifests rather
than decoding Stage 1 a third time. Validation also compares the logical-pairing
scorer with an independently implemented GF(2) row-space scorer.

## Fail-closed aggregation and publication

The aggregator ignores raw summary claims and recomputes counts from per-trial
fields. Missing planned shards make a code-p `INCOMPLETE`; an invalid shard,
conflicting duplicate, identity drift or malformed trial field makes it
`INVALID`. A nonreportable code-p makes the corresponding m-p primary mean,
interval and crossing input NaN. There is no valid-only official aggregation.

Each reportable code-p receives a 95% Wilson interval. Each m-p reports primary
mean, median, fixed-panel binomial MC SE, between-code standard deviation and SEM.
The frozen 20,000-draw two-level bootstrap resamples whole code curves within m,
then draws parametric binomial shot noise for each selected code-p. One maximum
absolute-deviation quantile simultaneously bands all primary curves, the endpoint
contrast and every adjacent-size contrast in the declared family.

The publication loader accepts only `exp103.aggregate.v1`, the complete axes and
the preregistered full p mask. It rechecks hashes, counts, rates, Wilson intervals,
equal-weight means, fail-closed statuses, contrasts and the terminal decision.
It rejects exp101/exp102 schemas and post-hoc point masks.

## Crossing decision

For larger minus smaller failure rates, the correct direction is negative at a
lower p and positive at a higher p. A bracket uses two adjacent certified grid
points; interpolation is plot-only. The final primary contrast is
`Delta38=P_fail(m8)-P_fail(m3)`. All consecutive triples `(3,4,5)` through
`(6,7,8)` are preregistered.

- `EXP103_DECODER_CROSSING_RESOLVED`: the simultaneous band for Delta38 is fully
  negative then fully positive across its unique bracket, and one triple has two
  certified adjacent-size brackets with overlapping intervals.
- `EXP103_PAIRWISE_BRACKET_ONLY`: Delta38 has that certified bracket but no triple
  supplies the required multi-size consistency.
- `EXP103_NO_CORRECT_CROSSING_IN_WINDOW`: complete valid data contain no
  negative-to-positive point-estimate reversal.
- `EXP103_DECODER_CROSSING_INCONCLUSIVE`: a reversal exists but bands, multiple
  reversals, or size conflict prevent a unique certified bracket.
- `EXP103_INCOMPLETE` and `EXP103_INVALID`: formal crossing is not published.

Stage 1 may report restricted Delta35 and the `(3,4,5)` triple, but this has no
authority over whether Stage 2 runs. Stage 2 depends only on Stage 1 technical
completeness/replay and its own resource gate.

## Resource and execution gates

Validation 001 freezes the contract/config/schema and passes tiny CSS exhaustive
oracles, analytic endpoints, scorer agreement, decoder identity, seed isolation,
fail-closed aggregation and synthetic crossing cases. Validation 002 benchmarks
fixed m3/m5/m8 and p=.02/.08/.14 tasks with benchmark seeds. It records timing
and RSS only; no logical outcome is saved or inspected.

Each three-size stage requires
`2 * (generation + full replay + analysis + fixed overhead) <= 100 core-hours`,
predicted local wall time at eight workers at most 12 hours, and projected peak
RSS at most 12 GiB. A failed stage becomes `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` and
does not move to remote. Formal measurement uses exactly eight processes and one
OpenMP thread per decoder.

The source, config and contract must be committed and pushed before measurement.
The contract bytes must still equal the version at the config's frozen source
commit; a later committed contract edit closes the execution gate. Stage 1 is
Validation 003, Stage 2 is Validation 004, and the full aggregate and
five-validation checkpoint are Validation 005.

## Scientific red-team

- Target/support: Bernoulli physical errors and perfect syndromes are sampled
  directly; there is no posterior chain, translated support, or planted truth in
  a sampling ratio.
- Coordinates/outcome: HGP convention, residual, stabilizer row space and logical
  pairing are explicit and independently checked. A nonzero syndrome residual can
  never be called success.
- Slow variables/self-loops: this is decoder Monte Carlo, not an MCMC convergence
  claim. BP non-convergence is measured but cannot delete a valid decoder output.
- Estimand/delivery: block failure of the specified decoder is distinct from MLD,
  posterior purity and `q_top`; code heterogeneity is displayed, not selected away.
- False positive/negative gates: simultaneous bands control the frozen comparison
  family; a no-crossing or inconclusive outcome is acceptable and no window is
  chosen after seeing curves.
- Exact/independent checks: tiny CSS enumeration, `p=0`, the `p=0.5` uniform-error
  relation, analytic stabilizer/logical residuals, and independent pairing versus
  row-space scorers replace the invalid idea of treating m3 TI as exact MLD.
- Permissions: even a completely successful exp103 only motivates a separately
  contracted larger decoder scan or exp104 comparison. It clears none of exp102's
  four blockers and authorizes no remote, formal, held-out or production exp102 work.
