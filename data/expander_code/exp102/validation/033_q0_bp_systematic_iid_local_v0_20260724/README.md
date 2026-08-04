# BP-systematic IID-MIS hard-sentinel diagnostic

This is a fresh local feasibility test for `m08_c06,p=.04,d00,attempt022`.
The target is exactly `pi(e|y) proportional to (.04/.96)^|e|` on `H_Z e=y`;
planted error is used only to generate the frozen disorder, never in energy or
proposal scoring.  It uses no MCMC trajectory or initialization family.

The three equal-allocation sources are two BP-guided systematic hard-coset
proposals with independently frozen forward/reverse information sets, and a
freshly rebuilt MAM proposal.  Each has a BP/prior/uniform or anchor/component
mixture with exact state density.  Every raw record stores its source,
coordinate, anchor index when present, and component index, so later analysis
can expose a component that dominates the importance tail.

The schedule is fixed before target weights are read: 16 independent blocks,
1,024 draws from every source in every block, for 49,152 fresh draws total.
The preceding outcome-blind preflight is bound by SHA
`dfb643d4e70bdf6d6198eefc4eca061e895cb9f79a034b83c384cb96f5975e59` and
verified the two BP proposals' hard-coset algebra and runtime.  This run has
no resampling, cloning, adaptive extension, source selection, or reuse of old
samples/seeds.

The primary gate requires both BP proposals and the equal mixture to meet
block-weight stability, BP-forward/reverse agreement in both purity-derived
`q_top` diagnostics and sector-distribution distance, mixture precision, and
minimum provenance coverage of every internal BP component.  MAM is a mandatory
reported stress comparison but cannot rescue a failed BP gate.  Even a PASS is
only empirical feasibility: full proposal support and stable finite weights do
not constitute a rigorous bound on a completely unobserved target mode, so it
cannot authorize remote, formal, held-out, or production work.

## Terminal result

The frozen schedule completed once and is terminally
`LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED`. The runner regenerated every
draw from rebuilt proposal objects before writing raw; its deterministic
generation and hard-coset algebra replays both pass. The separate
`allow_pickle=False` raw-only analyzer also passes. The config SHA256 is
`117a8c36b511469cd052c7185cdd9d7553fa9fcbe8c91ef3cd08703f04ac3f86`, raw
SHA256 is `fd662ae5a30ce0e0aa70ebf6253882da91c7cf479db9669400affe972a1625da`,
and the report's internal SHA256 is
`2a62ddf1d7bfc49b06e2a80e4d6d45f2d7558970bed4f7de28faedd0f25705fb`.

| View | Collision diagnostic | Jackknife SE | Minimum block ESS | Maximum block weight |
| --- | ---: | ---: | ---: | ---: |
| BP-SYS-F64 | .994531 | .002250 | 730.10 | .00750 |
| BP-SYS-R64 | 1.000000 | < .000001 | 457.00 | .00219 |
| Equal three-source mixture | .992881 | .002047 | 23.48 | .14695 |
| MAM-IMH8 stress source | .995101 | .001537 | 23.00 | .14765 |

The two BP sources pass their individual frozen weight and component-coverage
gates. Their diagnostic difference is `.0054695` with jackknife SE `.0022496`,
and their normalized sector-distance diagnostic is `1.37e-5 +/- 1.15e-5`;
those predeclared pairwise gates pass. The all-source mixture nevertheless
fails its separately required weight-stability gate: several blocks are
dominated by MAM-source samples. Because the frozen estimator and its gate
explicitly use the equal mixture of all three sources, this rejects the exact
three-source schedule. It does not retrospectively establish or reject a
different BP-only estimator; removing MAM after seeing this result would
require a fresh contract, fresh seeds, and fresh raw.

The displayed collision values are not reportable posterior purity or
`q_top`. In particular, both BP proposals derive essentially all observed
normalized target mass from their BP component, while their defensive prior
and uniform components provide no observed target mass. This is compatible
with good local proposal overlap but supplies no bound on a remote unobserved
mode. The fixed BP messages were also still oscillatory in the outcome-blind
preflight; that does not invalidate the exact proposal density, but it is an
additional reason not to interpret the finite diagnostic as a posterior
certificate. No remote task, `READY_FOR_FORMAL`, held-out run, or production
authorization exists.
