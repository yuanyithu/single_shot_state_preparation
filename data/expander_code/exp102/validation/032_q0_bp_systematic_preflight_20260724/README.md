# BP-systematic proposal preflight

This outcome-blind local preflight tests a new proposal mechanism, not a
posterior estimator.  It parameterizes a q=0 hard coset by an exact physical
information set: selected free physical bits are sampled from a three-component
mixture (loopy-BP marginals, the prior, and uniform coordinates), then pivot
bits are solved exactly over GF(2).  Thus every draw is on `H_Z e=y` and the
proposal density is exact even though BP itself is only an approximation used
to choose proposal probabilities.

The frozen preflight panel is the nonzero-syndrome m8 hard sentinel
`m08_c06,p=.04,d00,attempt022`, with both forward and reverse column orders,
64 fixed BP iterations, and 128 direct draws per order.  It records only
identity, hard-coset replay, BP numerical summaries, and runtime.  It does not
compute a target importance weight, sector label, purity, or `q_top`; no output
from this stage can choose a favorable order or authorize server work.

If it passes algebra and the prospective runtime budget, a separate frozen
IID-MIS contract must include both orders, store internal component provenance,
use fresh seeds/raw, and require independent tail/weight diagnostics.  A
successful proposal preflight alone cannot establish that finite draws cover a
remote target mode.
