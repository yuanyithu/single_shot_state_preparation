# Real-low-energy Houdayer component feasibility

This local, deterministic structural probe follows the earlier coordinate-only
Houdayer check.  It asks a narrower question: when the two replicas are
actual legal low-energy logical states derived from the frozen hard disorder,
does a component swap create a new *unordered replica pair*, or does it only
exchange the two entire replicas?

The candidate starts are generated before inspecting any chain output.  The
probe reduces the canonical logical basis using the existing deterministic
rule, enumerates its single/pair/triple XORs, and retains the minimum-weight
P-derived state for each nonzero logical label.  It then freezes two views of
that finite catalog: the first 16 label-distinct low-energy states, and a
64-state greedy rank-complete subset.  The fixed pair schedule contains P
against each low-energy state, every low-energy pair, and P against each
rank-complete state.

For every pair, the report records the coordinate components, their logical
rank, whether a component is the entire disagreement set, and the number of
distinct unordered physical replica pairs obtained from all component subsets
when there are at most 12 components.  Above that fixed cap it checks only
the original pair plus each one-component swap and labels the result as
non-exhaustive.  It also verifies hard-coset membership and exact preservation
of the pair energy for every evaluated subset.

This is not MCMC.  It creates no posterior estimate, `q_top`, convergence
claim, remote job, held-out result, or production authority.  A nontrivial
result would only justify implementing and exhaustively validating a local
two-replica sampler; a whole-pair exchange is explicitly not counted as
global mixing.
