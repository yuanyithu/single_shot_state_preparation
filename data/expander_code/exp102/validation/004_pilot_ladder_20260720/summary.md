# exp102 configured ladder search

Source `bbe72da` passed preflight on nd-1, nd-2, and nd-3 with NumPy 2.3.4, Numba 0.65.1,
14/14 exp102 tests per node, and canonical digest
`9af48083dff55741b662aed24815a88b82a2bf3d484e8231064f0b2dee753827`.

The ordered ladder search ran 10,752/10,752 cells on nd-2/nd-3. Raw evidence is retained under
run `exp102_pilot_20260720_bbe72da`. The original merge-select report SHA256 was
`c343cf402d2c4f722b8a8e2d5d6e87c931cae9ecbc30e40eb37b019f546b963e`.

A post-run stage audit found that the source incorrectly applied character trace convergence in
ladder/gamma, although the frozen selection policy reserves it for rounds. The implementation is
fixed and the raw counters were separately reclassified using only the intended ladder gates.

- m=3 first passes all 96 tuning cells at `p_hot=0.45,R=64`.
- m=4..8 have no passing ladder pair.
- At the maximum `p_hot=0.49,R=64`, swap/hot/residual-valid cells are respectively
  `93/96, 89/96, 85/96, 84/96, 87/96`.
- Every failed maximum-candidate cell has at least one sub-0.15 swap edge. Character constants are
  diagnostic at this stage and no longer contribute to ladder validity.

The corrected classification still leaves every m=4..8 below 96/96, so the frozen policy requires
a stop. Gamma, rounds, held-out, freezing, the 6144-task plan, and production were not started. A
new clean-SHA run is required after any approved expansion of the pilot schedule.
