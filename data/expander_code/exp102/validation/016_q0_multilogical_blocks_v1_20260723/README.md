# Multi-logical block heatbath V1

This isolated validation records the successor candidate `MLB8-J16`.
Each fixed block contains eight independent reduced logical directions and
eight independent stabilizers.  A block update enumerates all `2^16` XOR
states and samples its exact conditional hard-coset Boltzmann distribution.

This directory currently contains only local algebra and coverage evidence:

- exact detailed-balance/stationarity tests on the n=10/n=13 HGP oracles;
- reference/Numba trajectory identity on real m3;
- deterministic construction and rank coverage across the frozen 48-code
  registry;
- a warmed, three-repeat m8 hard-sentinel runtime diagnostic in
  `m8_runtime.json`: the catalog-free exact J16 path measured 6.89 ms per
  sweep locally, projecting to about 70.8 seconds at T1 and 283.3 seconds at
  T3 per trajectory.  This is a
  feasibility screen only, not sampling evidence;
- 125 q=0 global/logical-stratified/MLB tests with zero failures or errors in
  `q0_suite.xml`.

There is an important independence limitation to preserve in any future
screen.  The current runner now freezes an empty logical catalog and rejects
any nonempty or tampered replacement: it uses only the regular stabilizer
heatbath plus the multi-logical exact block.  That removes the old reduced
single/pair/triple logical catalog as a possible hidden source of transport.
Future raw must still record the component counters and confirm zero catalog
attempts; total label changes remain insufficient as a convergence claim.

It has no sampler raw, no posterior estimate, and no authorization for the
formal exp102 scan.  A future remote HARD2 screen must use a fresh immutable
contract, source archive, seed namespace, and result-blind gates.
