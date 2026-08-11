# exp105 validation evidence index

| Number | Question | Terminal status | Controlled category | Authority |
|---:|---|---|---|---|
| [001](./001_contract_and_redteam_20260811/) | Contract freeze, scientific red team, ensemble composition census, and the measured infeasibility of the requested `q_top` observable at `m >= 4` | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes Validation 002. No remote transfer, no physical result. Records that the PT cold-logical-acceptance gate falls 30 to 76 orders of magnitude short, and that the census reproduces exp104's composition under an independent master seed |
| [002](./002_local_implementation_gate_20260811/) | Does the exp105 package hold up locally: determinism, the `phi_r` scoring map, cross-package equality with exp104, and the full suite? | `PASS` | IMPLEMENTATION_GATE | Local only; authorizes the locating pilot. 137 exp105 + 131 exp104 + 366 exp101 tests, nothing skipped. Records that the label map coincides with `logical_Z` for this family and that the q>0 change is the failure criterion, and that `test_remote_execution.py` is deferred to Validation 004. Authorizes no remote transfer and no production compute |

Terminal statuses are the original strings recorded in each directory. No entry
here may be upgraded into a parameter-point certification, a production
authorization, or any exp102 authority.
