# exp104 ensemble crossing report

Terminal status: `EXP104_CERTIFIED_CROSSING`

- Decoder: BP+OSD-0, identity frozen byte for byte with `exp103.decoder_mc.v2`.
- Ensemble: 2000 randomly generated codes per m, m = 3..8, no post-selection.
- 4 trials per code and p over 9 grid points.
- Simultaneous band half-width on the primary contrast: `0.0211`.

Certified bracket: `[0.05, 0.06]`.

Crossing location: `p_cross = 0.05512` with 95% bootstrap interval `[0.05327, 0.05699]` (defined in 1.000 of replicates).

## Scope

Finite-grid, decoder-dependent, code-capacity result for one frozen BP+OSD-0 decoder on one randomly generated expander-code ensemble at q=0. No asymptotic threshold, no critical exponent, no finite-size scaling, no q_top, no MLD and no preparation-channel claim. Clears no exp102 blocker.

