# exp102 experiment contract

- `physics_contract_version`: `exp102.physics.v1`
- `pt_contract_version`: `exp102.q0_pt.v1`
- `scan_contract_version`: `exp102.scan.v1`
- Parent physics: `exp101.physics.v2`, `true_posterior`, `x_error/H_Z`, `|+>_L`.

At `q=0`, exp102 samples `pi(e|y) proportional to exp(-K_p |e|)` subject to
`H_Z e=y=H_Z epsilon_true`. Every PT rung has this same hard coset. The ladder is linear in
coupling after the configured power transform, and swaps use
`(K_i-K_j)(|e_i|-|e_j|)`. Single-bit and syndrome-changing moves are forbidden.

Each disorder uses four independently seeded PT instances initialized in logical labels zero,
all-ones, even bits, and odd bits. Labels are `uint64`; `k>64` is rejected. The primary disorder
estimator is the average of six pairwise empirical label collisions, normalized by the uniform
mass. Raw estimates are never clipped. Planted hits are an audit statistic, not a per-disorder
name for qtop.

Publication aggregation is fail-closed at disorder, code, and m levels. A code cell needs 128/128
present and valid disorders plus the paired planted audit. An m cell needs 8/8 reportable codes.
The plotted error bar is exactly `std(code_means,ddof=1)/sqrt(8)`.

Registry selection accepts the first eight simple, full-row-rank, distinct classical matrices for
each m. Pilot outcomes, distance, and MCMC difficulty never affect registry membership. Production
workers reconstruct only frozen registry entries and refuse an un-certified PT configuration.

The ordered ladder schedule is contract data, not a Cartesian product. It tries `R=8,12,16,24,32,48,64`
at `p_hot=0.45`, then the same sequence at `0.475`, then `R=8,12,16,24,32,48,64,96,128`
at `0.49`. The `R=96,128` tail was explicitly approved after the original schedule failed its swap
gate on 2026-07-20. Every new configuration hash requires a clean-source pilot; raw validity from an
older configuration or source commit is never reused.

## PT v2 discovery boundary

`exp102.discovery.v2` is an isolated design search for a future `exp102.q0_pt.v2` contract. Its
Q32 ladders, multi-swap trajectories, raw files, reports, and namespaces are not formal pilot or
production evidence. A v2 production config may be generated only after discovery confirms both a
primary and a backup with different `ladder_id` values. Until that happens, the version header above
continues to describe the exhausted v1 production contract and no freezer may be created.

The frozen 2026-07-20 search is itself `EXHAUSTED`: D0/D2/D3/D4 passed the short ladder screen,
but all `S=4,16,64` transport candidates produced zero certified round trips on the two hard cells.
Because every S64 candidate also had an instance with no hot-updated visit, the conditional S128
branch was not legal and the confirmation panel was not run. Resuming requires a newly reviewed
algorithm/contract rather than more rounds, S128, weaker gates, or reuse of this discovery raw.
