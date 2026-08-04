# Local iid-MIS hard-sentinel diagnostic

This is a new, local-only feasibility test for
`m08_c06, p=.04, d00, attempt022`.  Its maximum authority is
`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_ONLY`: an explicit PASS or UNRESOLVED
terminal status cannot create a posterior result, `READY_FOR_FORMAL`, held-out
authority, a remote launch, or production work.

## Frozen target and schedule

The target is exactly the q=0 hard-coset posterior

```text
pi(e | y) proportional to (p / (1-p))^|e|,  H_Z e = y.
```

There is no MCMC chain and therefore no P/U/L initialization family.  The
nonzero-syndrome physical all-zero state is neither used nor legal.  Each of
the 16 independent blocks has 1,024 iid draws from all three frozen proposal
families, in the fixed order MAM-IMH8, LSI-IMH-T05, LSI-IMH-T10.  This is
49,152 total fresh draws with a unique PortablePrng stream per
`(block, proposal)`, no cloning, resampling, extension, or stopping based on
the result.

The selected source density is the equal-weight mixture of all three proposal
densities.  Equal weights are mandatory because every block allocates an equal
number of draws to each source.  The small-code exact test caught and now
rejects the otherwise subtle mismatch between unequal source allocation and an
unequal mixture density.

MAM is rebuilt only from `(H_Z, y, p)` and its frozen deterministic MILP rules;
it never receives the planted error.  The two LSI artifacts are only loaded as
fixed, algebraically revalidated proposal objects.  No old raw samples, old
random streams, MCMC states, or old estimates enter the test.  Keeping both
T05 and T10 in the schedule prevents choosing the more favorable historical
temperature after seeing this test; T10 is a mandatory stress diagnostic, not
a pass gate.

## Preflight evidence

`../030_q0_iid_is_preflight_20260724/runtime_probe.json` verifies legal
hard-coset output and positive uniform-coordinate defensive components without
reading an importance weight or a sector result.  It found roughly 1.1--1.3 ms
per own-proposal draw.  `../030_q0_iid_is_preflight_20260724/cross_density_runtime_probe.json`
measures the actually required three-proposal cross-density calculation and
projects about 107.5 seconds for the frozen 49,152-draw schedule.  These are
runtime/algebra observations only.

## Estimators and gates

The raw stores every packed state, logical label, physical weight, source,
block, source density and equal-mixture density.  The analyzer uses
`allow_pickle=False`, reconstructs every state-derived field, recomputes every
proposal density and verifies all draws against the hard coset before it reads
an estimator.

For each proposal separately and for the equal-mixture sample, it estimates
sector purity through ordered cross-block products.  Same-draw collisions are
not used.  It reports delete-one-block jackknife uncertainty, per-block
importance ESS and maximum normalized weight.  It also estimates the normalized
sector-distribution L2 difference between the independently drawn MAM and
LSI-T05 estimates.  The frozen primary gate requires stable MAM and T05
weights, agreement in `q_top`, a small distribution-distance upper diagnostic,
and a stable, precise equal-mixture estimate.

Passing those gates still cannot prove tail coverage: each proposal has full
support, but its globally uniform defensive component is exponentially weak in
the 832-dimensional hard coset.  A finite iid sample can therefore miss a
remote target mode even with an attractive ESS and cross-proposal agreement.
This diagnostic is useful precisely because it eliminates initial-state and
MCMC-transport artifacts, not because it supplies a rigorous weighted-count
certificate.  Any successor must retain an independent tail/normalizer check
or fresh held-out confirmation rather than upgrading this result directly.

## Terminal result

The frozen schedule completed once and is terminally
`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`.  The raw-only analyzer passed
without pickle loading and a current-source replay recomputed the same raw
fields, densities, collision diagnostics, and gates.  The config SHA256 is
`ae57a1e84251c3e643513ab65025c2c050f0e661516ed388ea63c76f1adecc42`; raw
SHA256 is `6cc3c19710725ef5ab714e010d636c7a0a0e7928db71e68059b1029009382071`;
and the immutable report SHA256 is
`d7dd5521b7292f68c01f0202d623da34968d060d3ae3422cd6f117e837a36e0a`.

| View | Collision diagnostic | Jackknife SE | Minimum block ESS | Maximum block weight |
| --- | ---: | ---: | ---: | ---: |
| Equal MIS mixture | .992081 | .001402 | 28.78 | .1574 |
| MAM-IMH8 | .980449 | .010140 | 22.09 | .1522 |
| LSI-IMH-T05 | .990156 | .001192 | 28.91 | .1629 |
| LSI-IMH-T10 stress | .993232 | .001992 | 27.40 | .1476 |

The frozen primary requirements were block ESS at least `50` and maximum
normalized block weight at most `.10`.  MAM, LSI-T05, and the mixture all fail
both weight-stability requirements.  MAM/T05 happen to pass their predeclared
agreement and distribution-distance checks, and the mixture passes its
precision check, but those successes cannot compensate for unstable importance
weights.  None of the values in the table is a reportable `q_top`.

There is no MCMC start in this diagnostic: a physical all-zero state would be
illegal for this nonzero syndrome and was not substituted for a hard-coset
draw.  This outcome therefore does not support changing every MCMC chain to a
common zero/P start; it only removes that particular source of uncertainty from
this test.  It also does not make rank-64 logical transport a mathematical
requirement for purity.  A concentrated posterior can have negligible
equilibrium sector changes, whereas a stable-looking finite IID estimate can
still miss a remote target mode.

The raw records the source proposal, state, label, physical weight, and stored
source/mixture densities; replay reconstructs proposal coordinates from each
state.  It does not record the internal MAM/LSI anchor and component IDs.  That
omission does not invalidate the raw replay or the failed gate, but it prevents
component-level diagnosis of the tail weights.  Do not patch or extend this
frozen raw.  A successor needs a new contract with fresh artifacts, seeds, and
raw schema that stores component provenance, plus an independently justified
tail/normalizer bound or confirmation method.
