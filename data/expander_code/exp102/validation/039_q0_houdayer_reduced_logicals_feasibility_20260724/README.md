# Canonical-reduced Houdayer component feasibility

The preceding tensor-complement coordinate probe found only whole-replica
exchanges for every frozen real logical pair.  That result must not silently
be generalized to every exact Houdayer coordinate system: component geometry
depends on the kernel basis.

This one additional local probe fixes the only natural alternative already
specified by the global-discovery contract: independent H_X stabilizers plus
the deterministic canonical reduced logical basis.  It uses exactly the same
hard disorder, P-derived candidate catalog, pair schedule, component-subset
cap, pair-energy checks, and definition of a meaningful result as `038`.
There is no adaptive basis search, restart, chain-result feedback, or choice
after seeing the tensor-basis result.

The runner reuses the already tested frozen pair-schedule evaluator from
`038`, but binds both its source hash and this wrapper's source hash in the
new report.  The only injected dependency is the explicit reduced-logical
coordinate-basis constructor.  This makes the one-basis counterfactual easy
to audit without mutating the earlier immutable probe.

As with `038`, this is not MCMC and grants no posterior, `q_top`, remote,
held-out, formal, or production authority.  If this fixed coordinate basis
also produces only whole-pair exchanges, no further Houdayer basis tuning is
authorized without a separately justified contract.
