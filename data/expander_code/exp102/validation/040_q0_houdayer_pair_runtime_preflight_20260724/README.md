# Reduced-coordinate Houdayer pair runtime preflight

This local preflight does not test posterior observables or convergence.  It
only asks whether the reference `HCA-RHB1` pair clock can fit a fixed local
adversarial-start screen before such a screen is allowed to exist.

The fixed target is the q=0 hard-coset posterior for each replica and their
product for the pair.  Each clock contains one random-scan sweep (832 updates
per replica) followed by one complete-component Houdayer move in the canonical
reduced-logical coordinate system.  Timing uses the legal section state at
coordinate zero against the legal all-one coordinate state.  This is fixed
before timing and deliberately does not select a low-energy pair, read
weights, labels, characters, `q_top`, or a sampler outcome.

The only pass condition is the frozen, factor-two-safe projection for
`128 + 1024` clocks being at most two hours per pair trajectory.  A pass does
not establish accuracy or launch authority; it merely permits a fresh local
P/P, U/U, L/L, and P/L fixed-clock diagnostic under this unchanged schedule.
