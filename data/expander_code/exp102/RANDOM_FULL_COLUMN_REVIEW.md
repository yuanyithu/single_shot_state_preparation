# Random-scan full-B-column Gibbs local review

This experiment asks whether one expensive but genuinely global collapsed-B
conditional is a useful clock.  It does not estimate a publishable `q_top` and
cannot authorize remote HARD2 or formal work.

The exact collapsed target is

```text
P(B) proportional to p^|B| (1-p)^(r^2-|B|)
                  product_j Pr_{a~Bernoulli(p)}[H a = y_j xor (B H)_j].
```

At each clock the kernel selects one of the `r=24` B columns with a
state-independent `PortablePrng` draw, enumerates all `2^24` column values,
and heatbaths from the exact conditional.  It then redraws every A column
exactly when recording an observation.  A random-scan mixture of exact
conditionals leaves the target invariant; no planted state, decoder energy,
or approximate proposal density occurs in the transition.

The earlier full-column feasibility run defined one public sweep as all 24
column conditionals.  That projected a T1 trajectory to about 38 hours and
was correctly rejected under that contract.  The present contract changes the
scientific clock before seeing new raw: one clock is one random full-column
conditional.  At the measured `.278952` seconds per update, a future
`2048+8192` T1 random-scan trajectory projects to about 48 minutes, or about
95 minutes with the required factor-two margin.  This is a new schedule, not
a reinterpretation or continuation of the old raw.

## Red-team boundaries

- A full-column update can change many B bits but still remain in one basin.
  Gates therefore use P/U/L B-weight, B-likelihood, full-weight, label-D2,
  Rhat/ESS, and actual label changes; changed-bit counts alone cannot pass.
- Exact redraws of `A|B` can make full-state labels look noisy while B remains
  frozen.  B diagnostics remain mandatory and are evaluated separately.
- P and the low-energy L starts have the same B mask on this sentinel.  L is
  retained to expose logical/A initialization memory, but it is not falsely
  presented as an independent B start.  Exact-K0 U supplies the adversarial B
  initialization.
- The physical zero state is illegal for this nonzero syndrome.  P, independent
  exact-K0 U, and distinct legal L states remain the only frozen families.
- The short 64+256 update clock is a transport screen.  Passing only permits a
  separately frozen larger T/2T diagnostic; failing terminates this schedule.

All candidate settings, gates, seeds, and clocks are frozen before raw.  There
is no result-dependent extension or replacement trajectory.
