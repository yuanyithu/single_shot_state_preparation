# 062 q=0 character-gate calibration

Status: `PRE-RUN / NO CALIBRATION REPORT / NO SAMPLER RAW`.

This local-only validation asks whether one deployable character-equivalence
operating point can meet the frozen false-reject, detection-power, and
simultaneous-coverage requirements. It uses exact small-HGP IID posterior
draws and explicitly nonphysical independent-character stresses. It does not
run HP64 or any MCMC chain.

## Frozen decision

Every candidate point has one trajectory count and one draw count shared by
all exact and synthetic rows. Two simultaneous multipliers are calibrated at
that same point:

- the `logical` multiplier is shared by exact logical catalogs and the frozen
  511/4160-character logical stresses, including both base means;
- the `collapsed_B` multiplier is shared by exact collapsed-B catalogs and
  the frozen 15/163/688-character B stresses, including both base means.

Points are tried in fixed independent-draw cost order. Selection and fresh
confirmation use disjoint seed stages. A terminal
`CHARACTER_GATE_COMMON_OPERATING_POINT_CONFIRMED` requires the first eligible
selection point, without changing either multiplier, to pass fresh
confirmation. Rates at both stages are gated by one-sided Wilson lower bounds,
not point estimates.

## Character and q_top accounting

The exact logical catalog contains every nonzero logical character, so its
mean squared character value is normalized `q_top` directly. For an exact
collapsed-B tilt, the runner reweights physical hard-coset states and then
recomputes induced `q_top` from the retained conditional logical signs. Exact
rows save every true character shift (at most 15 entries).

Synthetic rows cover catalog sizes 15, 163, 511, 688, and 4160. They save the
frozen shift pattern, nonzero count, minimum, maximum, mean square, and a hash
of the complete vector instead of repeating thousands of values. The
distributed `0.80 -> 0.76` stress is deliberately nonphysical: every observed
character moves by only `.04`, while catalog purity moves by `.0624`.

For a frozen observed catalog, `max |Delta m_u| <= .04` implies only the
deterministic worst-case bound `|Delta mean_u(m_u^2)| <= .08`. It gives no
coverage for unobserved characters. It bounds normalized `q_top` only when the
observed catalog is the complete nonzero logical catalog.

## Authority and execution

The maximum authority is calibration of this character gate. It cannot
certify mixing, HP64, a cell, an `(m,p)` point, remote work, formal tuning,
held-out work, or production.

Before the one-shot runner is invoked, all five source artifacts and the
config must be committed at one completely clean worktree. The runner rejects
bytecode and existing outputs. Local execution must use conda environment
`12` with `PYTHONDONTWRITEBYTECODE=1`. The independent auditor reconstructs
all rates, Wilson bounds, point summaries, selection, confirmation, authority,
and exact source-commit provenance without importing the runner.
