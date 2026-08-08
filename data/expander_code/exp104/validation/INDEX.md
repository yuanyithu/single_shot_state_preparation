# exp104 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_and_census_20260808/) | Contract freeze, scientific red team, ensemble composition census, local oracles and resource preflight | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes Validation 002. No remote transfer, no physical result |
| [002](./002_exp103_cross_validation_20260808/) | Does the exp104 code path equal the frozen exp103 code path? | `PASS` | CROSS_VALIDATION | 60,000 trials bit-identical across both packages on one machine; authorizes Validation 003. Records, without gating, that the compiled decoder is not bit-portable across platforms |
| [003](./003_remote_gate_20260808/) | nd-3 environment qualification and remote resource gate | `PASS` | REMOTE_RESOURCE_GATE | 131/58/17 tests on the Linux build, decoder binary identical to exp103's; reserved 270.9/900 core-h, wall 4.09/16 h, RSS 24.3/128 GiB. Opens formal measurement. The failed first qualification attempt is retained |
| [004](./004_remote_scan_20260808/) | Production scan and committed bit-exact replay on nd-3 | `PASS` | MEASUREMENT | 778/778 tasks fresh in 45.9 min, 432,000 trials; replay 83/83 with 60,120 trials bit-exact and zero exceptions |
| [005](./005_final_crossing_20260808/) | Full loader-verified crossing certification and location | `EXP104_CERTIFIED_CROSSING` | FINAL_ANALYSIS | 108,000/108,000 REPORTABLE; simultaneous half-width 0.0211; certified bracket [0.05, 0.06]; p_cross 0.05512 with 95% CI [0.05327, 0.05699]. No exp102 authority |
