# exp105 ensemble crossing report

Terminal status: `EXP105_NO_CERTIFIED_CROSSING`

- Decoder: BP+OSD-0, identity frozen byte for byte with `exp103.decoder_mc.v2`.
- Ensemble: {3: 9500, 4: 3300, 5: 1360, 6: 656, 7: 352, 8: 2449} randomly generated codes per m, m = 3..8, no post-selection.
- {3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 6} trials per code and p over 10 grid points.
- Simultaneous band half-width on the primary contrast: `0.0105`.

No certified bracket and no crossing location.

The primary contrast is **certified positive at all 10 grid points**: the simultaneous band excludes zero from below everywhere, so the larger code is worse than the smaller one at every p in the window. This is a certified absence of a crossing, not a failure to resolve one.

## Bound on q_top

Per disorder the exact posterior satisfies `map_success <= sqrt(purity)`, and no decoder beats MAP success at its own observation, so with `S = 1 - P_fail` and `M = 2^k` Jensen gives `E[q_top] >= (M S^2 - 1)/(M - 1)`. This is a certified one-sided bound, never an estimate, and it is informative only where `S` is large. Its strongest value here is `0.97190` at `m = 3, p = 0.001`. Full table in `report.json`.

## Scope

Finite-grid, decoder-dependent result for one frozen BP+OSD-0 decoder on one randomly generated expander-code ensemble at q = 0.05. No asymptotic threshold, no critical exponent, no finite-size scaling, no q_top *estimate* at m >= 4, no MLD and no preparation-channel claim. Clears no exp102 blocker.

