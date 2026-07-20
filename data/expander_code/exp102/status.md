# exp102 status

**PILOT RESTART REQUIRED — production not started**

The independent registry, bit-identical reference/Numba hard-coset q=0 PT, net-transport
diagnostics, task identity/resume, fail-closed aggregation, publication loader, and pilot cell
runner are implemented. The first fake-Numba `R=8` ladder completed but failed all 576 cells; its
partial `R=12` successor was stopped after the full-round Numba replacement made that source SHA
obsolete. Pilot tuning must restart from the new clean SHA. Held-out certification and the 6144
production tasks have not run, so no threshold curve or scientific result is claimed.

Production requires `engine=numba`; the reference engine is an oracle only. The full-round Numba
kernel is bit-identical through the `k=64` boundary and gives about 177x--196x speedup in local
benchmarks. The combined exp102/exp101 regression is 107/107 PASS. Cross-node preflight against the
new source SHA remains required before restarted tuning evidence can be accepted.
