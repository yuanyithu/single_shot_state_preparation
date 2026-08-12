# exp106 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_and_redteam_20260812/) | Contract freeze, scientific red team, independent ensemble census and disjointness from exp104 and exp105 | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes Validation 002. No remote transfer, no physical result. Records that the census reproduces exp105's composition under an independent master seed to within 0.0036 in acceptance and 0.0042 in the distance-2 fraction at every `m`, and that zero of 408 exp106 codes appear in exp104's 12,000 or exp105's 17,617 |
| [002](./002_local_implementation_gate_20260812/) | Does the exp106 package hold up locally: determinism, the `phi_r` scoring map, and cross-package equality against both predecessors? | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes the locating pilot. 192 exp106 + 166 exp105 + 131 exp104 + 58 exp101 + 17 exp102 tests, bytecode clean, `source_tree_sha256` identical to the value bound into both pilot configs. The **exp105 equality gate reproduces exp105 bit for bit at `q = 0.05`**, which is the only check that reaches the augmented matrix, the mixed channel, the readout draw and the `q > 0` criterion. The 29 skips are the unfrozen production plan and are load-bearing: nd-3 qualification allows none, so an unfrozen plan cannot reach the machine. Authorizes no remote transfer and no production compute |

Terminal statuses are the original strings recorded in each directory. No entry
here may be upgraded into a parameter-point certification, a production
authorization, or any exp102 authority.
