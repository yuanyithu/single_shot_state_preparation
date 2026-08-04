# Exp102 q=0 streaming full-column review

Version: `exp102.q0_random_full_column_streaming.preflight.v1`

## Question

Can the exact random-scan full-B-column heatbath from validation 052 be made
fast enough to run its unchanged 2048+8192 T1 diagnostic without changing the
target, clock, initialization panel, or convergence gates?

The old implementation materializes canonical masks, popcounts, log priors,
XOR indices, log weights, and scratch arrays for all `2^24` candidates.  Four
concurrent workers repeatedly stream roughly 528 MiB of private candidate data
per update.  The successor computes the same log weight in the same operation
order inside one Numba loop and overwrites a single 128 MiB array with the CDF.
Candidate order and the portable uniform variate are unchanged.

This is an implementation experiment, not a new sampler.  Its target remains

```text
H_Z e = y,
pi(e | y) proportional to (.04/.96)^|e|.
```

The planted error is not used in any weight or transition.  The optimization
does not cache outcome-selected basins, truncate candidates, approximate the
mass table, change floating precision, or alter the number of Markov clocks.

## Pre-registered gates

Before any T1 raw may exist:

1. Small HGP zero/nonzero-syndrome tests at `p=.04,.10,.25` must reproduce the
   direct conditional and preserve detailed balance/stationarity to `2e-13`.
2. Reference and Numba streaming CDFs must be byte-identical on the small
   oracle.  Streaming trajectories must match the legacy discrete and float
   raw fields and replay bit-exactly.
3. On m8, the complete 16,777,216-entry legacy and streaming CDFs must be
   byte-identical for the frozen P/M0/S0/U0 states and columns 0/11/17.  This
   is a transition-equivalence check, not a posterior diagnostic.
4. The single-process median conditional speedup must be at least `4.2x`.
5. Four concurrent P/M0/S0/U0 probes each run 8 burn plus 128 measurement
   updates, followed by complete bit replay.  The worst projected 2048+8192
   replay-inclusive trajectory time, with an additional `1.2` safety factor,
   must not exceed 7200 seconds.

The Numba specialization is compiled once in the parent before forking.  This
is part of the proposed production implementation, not discarded timing: all
workers inherit the same compiled code, while sampling and replay setup costs
remain inside each timed probe.  Separating compilation from steady-state
work closes the previous blind spot where a very short probe could multiply a
one-time JIT cost by every planned update.

Any CDF mismatch is `CONFLICT`.  A speedup or projection failure is
`RUNTIME_EXHAUSTED`.  Passing only authorizes a fresh clean-source three-node
preflight with the same gates.  It does not authorize T1 measurement, m6,
HARD2, formal tuning, held-out, production, or a physical result.

If the three-node preflight passes, a separately frozen successor must retain
all P/U/M0/M1/S starts and all q_top/D2/B/logical diagnostics from validation
052.  It may not shorten the clock, weaken the cap, or reuse validation 052
seeds or raw.
