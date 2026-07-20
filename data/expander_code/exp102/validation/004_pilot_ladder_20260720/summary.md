# exp102 configured ladder search

Source `bbe72da` passed preflight on nd-1, nd-2, and nd-3 with NumPy 2.3.4, Numba 0.65.1,
14/14 exp102 tests per node, and canonical digest
`9af48083dff55741b662aed24815a88b82a2bf3d484e8231064f0b2dee753827`.

The ordered ladder search ran 10,752/10,752 cells on nd-2/nd-3. Raw evidence is retained under
run `exp102_pilot_20260720_bbe72da`; the local merge-select report SHA256 is
`c343cf402d2c4f722b8a8e2d5d6e87c931cae9ecbc30e40eb37b019f546b963e`.

- m=3 first passes all 96 tuning cells at `p_hot=0.45,R=64`.
- m=4..8 have no passing ladder pair.
- At the maximum `p_hot=0.49,R=64`, valid cells are respectively
  `90/96, 88/96, 80/96, 72/96, 37/96`.
- Failures include sub-0.15 swap edges and untrusted constant characters, especially for large m.

The frozen policy requires a stop when the maximum candidate fails. Gamma, rounds, held-out,
freezing, the 6144-task plan, and production were therefore not started.
