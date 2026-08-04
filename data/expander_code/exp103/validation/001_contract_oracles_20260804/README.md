# Validation 001: contract and oracles

Status: `PASS`.

This validation freezes `exp103.decoder_mc.v1` and must cover tiny CSS exhaustive
enumeration, analytic endpoints, residual golden cases, two independent GF(2)
scorers, exact BpLSD kwargs/backend identity, seed isolation, registry retention,
fail-closed raw aggregation, publication-loader rejection and synthetic crossing
decisions. The frozen package has source commit
`6ab3558402c24571f4580da11d13b22d0888d5d9` and source-tree SHA
`65bd2b869430fe9b72160f9cc06cfbb1ca5fb2a0a661240dc210fb28f1647d9b`.

All 105 exp103 tests, 58 focused exp101 algebra/HGP/logical regressions and 17
focused exp102 registry/loader/source regressions passed in the canonical
macmini conda-12 identity. The machine-readable authority is `report.json`.
This PASS authorizes only Validation 002 local resource preflight; it grants no
measurement result and changes no exp102 blocker.
