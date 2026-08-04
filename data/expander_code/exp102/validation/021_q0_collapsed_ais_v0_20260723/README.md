# Collapsed AIS V0 local path-weight diagnostic

## Scope

This is a fresh local-only test of `CAIS64-B8-S1-N128` on
`m08_c06,p=.04,d00,attempt022`. It is a response to the specific,
audited failure of the previous always-resample SMC V0: it uses no resampling
or cloning whatsoever. It cannot calculate `q_top`, estimate a formal
posterior result, authorize HARD2, create `READY_FOR_FORMAL`, or start a
remote/formal exp102 task.

The manifest is frozen before raw creation and binds the cell, code, syndrome,
mass table, 64-level bridge, source-file hashes, eight independent population
seeds, and raw schema. This is local dirty-worktree diagnostic evidence only,
not a clean-source deployment.

## Mathematical red-team

With `H A xor B H = Y`, integrating out independent A columns gives

```text
pi_lambda(B) proportional to prior_p(B) * L(B)^lambda,
L(B) = product_j Pr_p[H a_j = (Y xor B H)_j].
```

The cold endpoint (`lambda=1`) is exactly the collapsed q=0 posterior. The
base endpoint (`lambda=0`) is exactly iid `B~Bernoulli(p)`, so P/U/L and the
physical all-zero vector are deliberately not used as warm starts. The latter
is outside this cell's nonzero-syndrome hard coset; injecting any of these
states would invalidate the known base measure rather than test convergence.
Four independent populations use a column-major iid construction and four use
a row-major one, which are mathematically equivalent but expose layout errors.

At level `t`, before mutation, the path weight is updated by

```text
log W_t = log W_(t-1) + (lambda_t-lambda_(t-1)) log L(B_(t-1)).
```

The subsequent mutation is a uniform random choice of a B block followed by
its exact categorical heatbath. That one-step mixture is reversible with
respect to `pi_lambda`; a fixed number of applications remains reversible.
The AIS formula therefore uses the full accumulated path weight, not an SMC
normalizer shortcut. The exact tests enumerate the n=10/n=13 HGP collapsed
state spaces at `.04/.10/.25`, verify detailed balance for the random-block
kernel, verify the multi-level AIS weighted measure equals the cold target,
and require reference/Numba transcript identity including the actual m8
64-level path.

This still does **not** prove finite-particle coverage of all cold modes. A
high importance ESS can coexist with a shared collapsed-algebra blind spot,
and even a V0 pass would need a new full sampler/independent-confirmation
contract. Conversely, a failure means only that this frozen AIS budget has
unusable weight concentration, not that the posterior or all AIS schedules
are impossible.

## Frozen V0

- Method: `CAIS64-B8-S1-N128`.
- Bridge: the pre-existing HP64 quadratic schedule `lambda_i=i^2/63^2`, SHA
  `9aa5269ce0eee77473f7d0375ea9d007aa31cf6daf1e47d0cb4af23224be45c0`.
  It reuses no HP raw, seed, state, estimate, or decision.
- Mutation: one reversible random-scan 8-bit B-block heatbath sweep at each
  nonzero level; every particle and level has its own PortablePrng substream.
- Population: eight independent N=128 paths (`4 column_major + 4 row_major`).
  There is no resampling, clone, adaptive restart, result-driven extra sweep,
  or q_top-based choice.

Every population must meet all frozen V0 gates:

| quantity | required value |
|---|---:|
| final full-path importance ESS / N | `>= .25` |
| final normalized importance weight | `<= .10` |
| largest single incremental normalized weight | `<= .10` |

The only positive V0 status is `LOCAL_COLLAPSED_AIS_PATH_WEIGHT_VIABLE`; it
means merely that a later, fresh, fully audited sampler design may be worth
considering. The negative status is `LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`.
Neither status is a q_top, posterior, HARD2, formal, held-out, or production
result. Raw is immutable: it may not be extended, pooled, or retuned.

## Terminal result

All eight frozen paths completed. The focused exact/reference/Numba test
suite passed (`10 passed`), and the subsequent exp101+exp102 regression suite
passed `922` tests with zero failures/errors. All raw paths then passed
deterministic seed replay and an independent raw-only audit. The terminal result is nevertheless
`LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`, because each population fails
every frozen path-weight gate.

The frozen runner retains `RUNNING.json` as a historical start marker. It is
not a liveness signal after raw exists: the terminal evidence is the matching
`RUN_COMPLETE.json`, `REPLAY.json`, `REPORT.json`, and `SUCCESS.json` files.
The runner is source-bound by this raw and is not modified retroactively.

- Manifest SHA256:
  `c3dc27a3e0d7a233ac66027c61f7e642e2cb343b5b01bc8120dd3e0211965ba6`.
- Replay SHA256:
  `5e6ae5e47ca67e17692f12051fd71a65a400664e9633ead4d20f15558e662ac7`.
- Report SHA256:
  `2f6c298324ce7f647cceec7ddd7f377a9dc2a2391ca77d0d8bf49dc2ab0f9324`.
- Independent raw-only audit SHA256:
  `c211911b2ceaaf6e2b033950b8eef32d6ac4c9623e68ae2b5f4cdd6ce5317321`.

Final full-path ESS/N is only `.0078125--.0100431` (required `>=.25`), while
the final dominant normalized weight is `.872760--1.000000` (required `<=.10`)
and the largest one-stage normalized weight is `.122396--.214436` (required
`<=.10`). The median cumulative ESS is `85.93/128` at stage 15, `1.22/128`
at stage 31, and `1.000002/128` at the cold endpoint. Therefore removing the
previous SMC resampling mechanism did prevent genealogy collapse, but did not
solve the relevant path-weight problem: almost all full AIS mass concentrates
on one particle late in the bridge.

`independent_raw_audit.py` reads only saved NPZ raw with `allow_pickle=False`.
It does not import the AIS engine or invoke a sampler; it independently
reconstructs the HGP syndrome, iid base population, coset mass dynamic
program, B-derived A syndromes, likelihoods, incremental/cumulative weights,
mutation-counter constraints, final target, gates, replay/report identities,
and source binding. It agrees exactly with this terminal negative result.

This does not disprove the collapsed posterior, a different bridge, or all
AIS. It closes only `CAIS64-B8-S1-N128`: no extension, pooling, adaptive
rescue, q_top estimate, HARD2 deployment, or formal authorization can use
this raw. A new candidate would require a fresh contract and independent
confirmation rather than merely more particles or more steps here.
