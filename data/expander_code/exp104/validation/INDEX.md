# exp104 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_and_census_20260808/) | Contract freeze, scientific red team, ensemble composition census, local oracles and resource preflight | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes Validation 002. No remote transfer, no physical result |
| [002](./002_exp103_cross_validation_20260808/) | Does the exp104 code path equal the frozen exp103 code path? | `PASS` | CROSS_VALIDATION | 60,000 trials bit-identical across both packages on one machine; authorizes Validation 003. Records, without gating, that the compiled decoder is not bit-portable across platforms |
| [003](./003_remote_gate_20260808/) | nd-3 environment qualification and remote resource gate | `PASS` | REMOTE_RESOURCE_GATE | 131/58/17 tests on the Linux build, decoder binary identical to exp103's; reserved 270.9/900 core-h, wall 4.09/16 h, RSS 24.3/128 GiB. Opens formal measurement. The failed first qualification attempt is retained |
