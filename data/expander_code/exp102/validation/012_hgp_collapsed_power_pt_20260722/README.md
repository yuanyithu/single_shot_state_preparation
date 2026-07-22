# 012 collapsed HGP local feasibility

This directory contains local conda-12 design evidence for the HGP collapsed
likelihood-power sampler. It is not a frozen server run and cannot be reused by
validation 013, formal tuning, held-out, or production.

The retained script ran HP32 on the two previously frozen HARD2 disorders with
8 planted and 8 exact-uniform trajectories per cell at burn 1024 and
measurement 4096. The provisional report found:

- m6 P/U `q_top` difference `0.0001597`;
- m8 P/U `q_top` difference `0.0031192`;
- m8 minimum adjacent swap rates about `0.139--0.154`;
- strict cold-hot-cold round trips in every sampled trajectory.

These observations motivated the larger immutable screen but do not satisfy
its 16-trajectory, T1/T2/T3, exact-replay, cross-mechanism, or server preflight
requirements. Local NPZ files are ignored by Git; only the script and compact
report are retained as design evidence.
