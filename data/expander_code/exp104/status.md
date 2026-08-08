# exp104 status

## Current state

**Contract frozen, local gates passed. No production compute has run.**

exp104 measures the `q = 0` code-capacity block logical failure rate of the
frozen `exp103.decoder_mc.v2` BP+OSD-0 decoder over a large, randomly generated
expander-code ensemble: 2000 codes per `m` for `m = 3..8`, four trials per code
and `p`, nine grid points from 0.02 to 0.10, 432,000 trials in total.

The estimand is the failure probability of a *random* code from the ensemble, so
the budget goes into codes rather than into trials per code. exp103 established
that shot noise never bound (largest cell standard error 0.0018 against a largest
between-code standard deviation of 0.3245); exp104 uses 14 times fewer trials
than exp103 and expects roughly an 18-fold improvement in the precision of the
quantity being certified.

## What is settled, and what is not

Settled by measurement, recorded in Validation 001:

- The ensemble composition. The classical-distance-2 fraction falls monotonically
  with size: `0.229, 0.195, 0.154, 0.133, 0.114, 0.104` for `m = 3..8`, measured
  on 20,000 accepted codes per `m` and reproduced by an independent master seed
  to within 0.0006. Acceptance rate rises from 0.721 to 0.990 over the same
  range.
- This is why exp103's primary contrast was positive everywhere. Its eight-code
  panels drew `0, 3, 2, 2, 0, 1` distance-2 codes for `m = 3..8`; the `m = 3`
  panel drew none where about 23 percent were due, so its curve sat too low and
  pushed `Delta38` positive at every grid point.
- Reweighting exp103's measured per-distance rates by the true composition puts a
  negative-to-positive reversal of `Delta38` between `p = 0.05` and `p = 0.06`.

Not settled, and not claimed: nothing about exp104's own outcome. The reweighted
reconstruction above is exp103 evidence viewed through a composition measurement,
not an exp104 result, and it is exactly the kind of agreeing point estimate a
successor experiment must be designed to test rather than to confirm.

## Current gates

1. Validation 001 is `PASS`: contract frozen, 131 local tests green including the
   decoder-determinism regression gate, ensemble census measured twice, local
   resource preflight `PASS`.
2. No remote compute is authorized until Validation 002 (cross-validation against
   frozen exp103 raw) and Validation 003 (nd-3 qualification and remote resource
   gate) both pass.
3. Replay is a preregistered 10 percent subsample fixed before production by a
   frozen seed. Any single bit-exact mismatch invalidates the whole run; the
   subsample is never narrowed afterwards.
4. exp102 remains `BLOCKED_BEFORE_REMOTE`. exp104 clears none of its blockers and
   authorizes none of its stages, as its contract states.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: the preregistered contract, including the terminal
  decision rule and its explicit primary-only scope.
- `config/ensemble_mc.v1.json`, `config/ensemble_mc.remote.v1.json`,
  `config/ensemble_registry.v1.json`.
- `validation/INDEX.md`: numbered evidence ledger.
- `validation/001_contract_and_census_20260808/`: red team, census, local gates.

## Latest evidence

- Validation 001: `PASS`. Registry SHA256 `7e40ff18...c709f315d4`, 12,000 codes.
  Census `f_2(m)` monotone decreasing, two seeds agreeing to 0.0006. Local
  preflight `PASS` at 16.33 generation core-hours upper bound on macmini.
