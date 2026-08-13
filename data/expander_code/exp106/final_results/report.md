# exp106 ensemble crossing report

Terminal status: `EXP106_NO_CERTIFIED_CROSSING`

- Decoder: BP+OSD-0, identity frozen byte for byte with `exp103.decoder_mc.v2`.
- Ensemble: {3: 76162, 4: 13068, 5: 5176, 6: 2464, 7: 1186, 8: 10344} randomly generated codes per m, m = 3..8, no post-selection.
- {3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3} trials per code and p over 10 grid points.
- Simultaneous band half-width on the primary contrast: `0.0079`.

No certified bracket and no crossing location.

The primary contrast is **certified positive at all 10 grid points**: the simultaneous band excludes zero from below everywhere, so the larger code is worse than the smaller one at every p in the window. This is a certified absence of a crossing, not a failure to resolve one.

## Bound on q_top

Per disorder the exact posterior satisfies `map_success <= sqrt(purity)`, and no decoder beats MAP success at its own observation, so with `S = 1 - P_fail` and `M = 2^k` Jensen gives `E[q_top] >= (M S^2 - 1)/(M - 1)`. This is a certified one-sided bound, never an estimate, and it is informative only where `S` is large. Its strongest value here is `0.94800` at `m = 3, p = 0.005`. Full table in `report.json`.

## Scope

Finite-grid, decoder-dependent result for one frozen BP+OSD-0 decoder on one randomly generated expander-code ensemble at q = 0.01. No asymptotic threshold, no critical exponent, no finite-size scaling, no q_top *estimate* at m >= 4, no MLD and no preparation-channel claim. Clears no exp102 blocker.

