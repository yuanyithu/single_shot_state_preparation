# Exact full-B-column Gibbs runtime gate

This isolated local diagnostic tests a new exact collapsed-HGP move: it
heatbaths every one of the `r=24` bits in a selected B column jointly.  The
conditional enumerates all `2^24` candidates, so it is target-exact but much
larger than the prior eight-bit B-column blocks or full-row conditionals.

It is not a posterior experiment.  The runtime probe does not redraw a full
state and does not construct labels, characters, `q_top`, or initial/observed
weights.  It therefore cannot be used as evidence for a physical parameter
point, a P/U/L comparison, remote work, tuning, held-out, or production.

## Correctness scope

`test_q0_hgp_full_column_gibbs.py` has 26 local passes.  It exhaustively checks
the `n=10` and `n=13` HGP examples for zero/nonzero syndrome and
`p=.04,.10,.25`, including direct conditional probabilities, per-column
detailed balance, full-sweep stationarity, cached-syndrome preservation, and
deterministic replay.  A public sampler clock is one random-permutation sweep
over all B columns, rather than one column update, so its resource accounting
is comparable to the rest of the q=0 contract.

## Frozen runtime result

The single outcome-blind m8 timing run is bound to
`runtime_probe.json` (report SHA256
`847b2abe1bfc1f91364a9a944d59ad30ca7ba84979e282ddce6f451506a63a80`).  It
uses `m08_c06,p=.04,d00,attempt022`, exactly one warm-up and exactly two timed
column conditionals.  The schedule and decision rule were fixed before the
timing run:

```text
T1 trajectory updates = (2048 burn + 8192 measurement sweeps) * 24 columns
projected wall         = 2 * (setup + timed_seconds_per_column * T1 updates)
pass only if projected wall <= 7200 seconds
```

The run builds 16,777,216 candidates, peaks at 1,197,178,880 bytes RSS, takes
`.442808` seconds to construct the tables, and takes `.278952` seconds per
full-column conditional.  The conservative projected T1 trajectory wall time
is `137111.403` seconds (about 38.1 hours), rather than at most two hours.
Its terminal status is therefore `RUNTIME_EXHAUSTED`.

No P/U/L local screen was started after this outcome.  Such a screen would not
make an approximately nineteen-times-over-budget kernel suitable for the
formal schedule, and continuing it would turn a timing failure into an
irrelevant local-motion optimization.

This rejects only the complete `2^24` full-column enumeration at this resource
gate.  It does not imply that the q=0 posterior, a differently factorized
exact block conditional, a certified collapsed-B tail bound, or another
global estimator is impossible.
