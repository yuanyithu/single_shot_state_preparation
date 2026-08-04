# Direct purity WMC feasibility

This is a bounded local feasibility test, not a posterior result or a formal
sampling stage.  It addresses a specific blind spot in the former MCMC gate:
`q_top` is a logical-sector purity, so a correct calculation need not make a
single low-temperature Markov chain visibly traverse all 64 logical bits.

For the q=0 hard posterior and `b=p/(1-p)`, define

```text
Z = sum_{H e = y} b^|e|
C = sum_{H e1 = H e2 = y, W(e1 xor e2)=0} b^(|e1|+|e2|).
```

Then `posterior_purity=C/Z^2` and

```text
q_top = (2^k * posterior_purity - 1) / (2^k - 1).
```

This script asks only whether the existing exact factor-elimination WMC engine
can reach either count on the m8 hard sentinel.  It encodes every parity as a
ternary XOR chain, uses the engine's deterministic min-degree order, and stops
before allocating a factor wider than a fixed cap.  It does not emit a numeric
count, approximate `q_top`, or use any MCMC raw.

## Result

`wmc_width_probe.json` reports SHA256
`970df49959eb380d785a4eb030bfd97e2eb1486865f63715212d7a0b0fd541a2`.

- The single-copy `Z` encoding has 5,440 binary variables and 6,208 ternary
  or unary factors.  It exceeds width 64 at width 67 while 1,153 variables
  remain.
- The two-copy equal-logical-label `C` encoding has 14,832 variables and
  16,432 factors.  It exceeds width 64 at width 66 while 2,494 variables
  remain.

Thus this exact ternary-XOR elimination encoding is not a practical direct
purity counter for this m8 sentinel at the tested widths.  This does not prove
that a different exact contraction, a certified collapsed-B branch-and-bound,
or the physical posterior is impossible.  It only redirects the next design
step toward a genuinely global B-tail bound rather than another unvalidated
cross-sector MCMC variant.
