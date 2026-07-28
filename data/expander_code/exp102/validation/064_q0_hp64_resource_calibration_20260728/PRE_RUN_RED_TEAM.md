# Validation 064 scientific and operational red-team

Status before execution: `LOCAL READ-ONLY EVIDENCE AUDIT / NO SAMPLER RAW`.

## Deliverable and authority

This validation has two deliberately separated deliverables:

1. independently recompute a small set of validation 013 `q_top` facts from
   immutable raw, solely to settle the HP64/MAM interpretation; and
2. construct outcome-blind HP64 resource scenarios from timing fields.

It cannot select a sampler length, certify convergence, authorize a remote
measurement, create `READY_FOR_FORMAL`, or turn validation 013 raw into formal
evidence.  Its maximum status is
`RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE`.

## Target, support, and estimator

No Markov transition is executed here.  The scientific audit reads the frozen
logical labels and frozen character masks from validation 013 with
`allow_pickle=False`.  It independently reconstructs per-trajectory character
means, the cross-trajectory U-statistic, finite-character correction, and the
two-sided delete-one trajectory jackknife.  It does not reinterpret label
collision as the primary estimator.

This says nothing new about target support or convergence.  In particular,
HP64's internal P/U agreement on five single-disorder cells remains diagnostic,
and HP64/HP32 remain one mechanism.

## Initial states and slow variables

The audit preserves the historical P and exact-K0 U identities.  It neither
removes U nor substitutes physical zero.  Physical zero is illegal for the
nonzero hard syndromes and shifted zero is P.  Resource arithmetic never reads
the P/U values, B diagnostics, acceptance, ESS, or any other scientific
outcome.

## Separation against outcome-driven resource selection

The program builds the resource payload first from only:

- task identity;
- `core_seconds` and `wall_seconds`;
- frozen clock/trajectory counts; and
- validation 013's stored analyzer timing benchmarks.

The resource function has no `q_top` argument.  The discrepancy audit is built
later and serialized to a separate JSON file.  Every clock and trajectory
option is emitted; this validation chooses none of them.

## Coverage and extrapolation traps

Validation 013 provides HP64 T3 timing for only five cells:

- `m03_c00,p=.10`;
- `m04_c00,p=.07`;
- `m05_c00,p=.10`;
- `m06_c00,p=.04`; and
- `m08_c06,p=.04`.

There is no empirical m7 timing, no multi-disorder timing distribution, no
eight-code timing distribution, and most `(m,p)` anchors are absent.  The
strict full-grid estimate is therefore `null`.

For planning arithmetic only, two visibly named scenarios are emitted:

- a same-m single-cell proxy, with m7 assigned the largest observed proxy; and
- the largest observed per-trajectory cost assigned to every evaluation.

Neither is a confidence bound.  Fixed-clock work is expected to be close to
linear, but code geometry, setup, cache, node contention, future raw schema,
and p-dependent control flow have not been ruled out.  The analyzer proxy only
covers the stored validation 013 B-family/B-comparison benchmark.  New-schema
analysis must be benchmarked again before freezing a campaign.

## Resource accounting

Every scenario reports generation, one complete replay, analyzer proxy, their
sum, and then a factor-two safety multiplier.  Ideal wall times at 166 and 75
cores are lower bounds, not schedules: they omit LPT imbalance, serial control,
artifact generation, filesystem contention, failures, and current live load.

The formal seven-p grid has 43,008 code-p-disorder evaluations.  The three-p
calibration grid has 18,432.  These are evaluation counts, not the historical
6,144 outer disorder tasks; no unproven sharing across p is credited.

## Integrity and stopping conditions

The config binds the runner, independent auditor, focused tests, README, and
this red-team by tracked repository-relative path and SHA-256.  The runner
requires a wholly clean Git worktree, no bytecode/cache artifacts, and records
the current calibration source commit and tree SHA.  These fields are named
separately from `validation013_source_commit`; conflating the historical raw
source with the calibration implementation source is a conflict.

The run fails closed if source/config authority, report self-hash, task/raw
identity, raw count, character masks, finite timing, report-versus-raw timing
sum, or raw-versus-report scientific recomputation disagrees.  Canonical JSON
forbids NaN and infinities.  Historical reports are never modified.

All outputs are installed exclusively.  The runner refuses any pre-existing
expected output and never uses overwrite/replace semantics.  The independent
auditor verifies exact versions, statuses, authority, config/source provenance,
tables, artifacts, and headline values before exclusively creating
`independent_package_audit.json`.  A partial package is terminal evidence for
that source/config identity, not permission to delete files and retry.

If the audit succeeds, the next legitimate action is to collect the missing
outcome-blind runtime matrix and calibrate the new analyzer schema.  It is not
to choose the cheapest row in the scenario table or launch the easy block.
