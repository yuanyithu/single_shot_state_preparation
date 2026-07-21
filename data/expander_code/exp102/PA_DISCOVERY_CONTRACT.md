# exp102 q=0 population-annealing discovery contract

- Discovery contract: `exp102.q0_pa.discovery.v1`
- Raw schema: `exp102.q0_pa.raw.v1`
- Transport autopsy raw schema: `exp102.transport_autopsy.raw.v1`
- Parent physics: `exp102.physics.v1`, `true_posterior`, `x_error/H_Z`, `q=0`
- Execution result: `EXHAUSTED` at the complete 64-population hard screen; no rescue or confirmation

This document freezes the algorithm search authorized after PT-v2 exhausted its
transport route. It does not modify the historical formal versions
`exp102.q0_pt.v1 / exp102.scan.v1`. Neither PT-v2 raw nor the raw produced by
this PA discovery may enter a formal pilot, held-out freezer, or production
merge. A successful discovery may say only `READY_FOR_FORMAL`; it cannot create
`FROZEN_HELD_OUT_PASS`.

## 1. Hard-coset population

One task is one independent population. Each cell uses eight tasks, never eight
subsamples from one task. At `K=0`, every particle is initialized as

```text
section(y) xor random independent-stabilizer coefficients
           xor random logical coefficients.
```

The worker must first prove that every stabilizer row is independent and that

```text
rank(H_X) + k = n - rank(H_Z).
```

It also checks that stabilizer plus logical rows span the complete hard-coset
kernel. Failure is `CONFLICT`, not a sampling failure. This makes the binary
coordinate map a bijection and the `K=0` initializer exactly uniform.

The target at the final stage is

```text
pi(e | y) proportional to exp(-K_p |e|),  H_Z e = y.
```

No syndrome-changing move is permitted.

## 2. Frozen Q32 schedule

For `G` anneal steps, the schedule has `G+1` points:

```text
theta_t = pi/4 + (t/G) (asin(sqrt(p)) - pi/4)
p_t     = sin(theta_t)^2
K_t     = log((1-p_t)/p_t).
```

The runtime consumes the pre-expanded, strictly increasing Q32 integers for
`K_t/K_G`, including exact endpoints `0` and `2^32`, and verifies their SHA256.
The config contains schedules for every formal p value
`.04,.05,.06,.07,.08,.09,.10` and `G=96,192,384`. No worker may insert,
remove, or adapt a stage.

Every stage performs, in this order:

1. incremental reweighting of the old particles;
2. CESS, ordinary ESS, and maximum normalized-weight recording;
3. systematic resampling iff `ESS < 0.5 N`, followed by equal weights;
4. the configured number of mutation sweeps at the new `K_t`.

Mutation occurs at every stage. Mutation sweeps are Markov moves, not extra
posterior samples.

## 3. Mutation kernels and randomness

`coordinate` uses a random permutation of independent stabilizer coordinates,
then a random permutation of logical coordinates. A proposed coordinate toggle
is selected with heatbath probability

```text
1 / (1 + exp(K_t delta_weight)).
```

`block4` uses the same stabilizer sweep. It randomly permutes the logical
coordinates, groups them four at a time, enumerates all at most 16 actual XOR
states, and samples their exact categorical distribution. Logical supports can
overlap; adding single-move deltas is forbidden. The implementation uses a Gray
walk only to evaluate the actual XOR state weights efficiently.

All exponential tables are computed in Python and passed to Numba. Reference
and Numba consume identical PortablePrng streams. Each mutation stream is
derived from the explicit tuple

```text
(source commit, trajectory config, cell, population,
 anneal stage, sweep, output slot).
```

Thus resampled clones receive independent output-slot streams; a parent's RNG
state is never cloned.

## 4. Raw evidence and analyzer

Every population raw stores the complete identity, source/config/registry and
schedule hashes, seed identity, final bit-packed states, final weights and
labels, every stage's energies and pre/post-decision weights, CESS/ESS,
resampling flags and offsets, parents and offspring counts, root ancestry,
mutation counters, logical flow, log-normalizer increments, log-Z, genealogy,
planted-class mass, unique-state diagnostics, and timing.

The analyzer always opens NPZ with `allow_pickle=False`. It independently
rebuilds the code and disorder, recomputes final residuals, weights, energies,
labels, planted label, every resampling decision and parent vector, ancestry,
genealogy, estimators, and all identity/digest checks. Remote evidence also
binds the clean source archive, immutable task manifest, canonical LPT node
ownership, raw hashes, status file, and exclusive SUCCESS marker. A duplicate,
unexpected file, seed/source/config/hash mismatch, non-finite value, or algebraic
failure is `CONFLICT`.

Discrete state, labels, parents, offspring, ancestry, resampling decisions,
counters, identities, and hashes remain exact. Derived float64 replay permits at
most 8 ULP for the coupling/probability ladder, 64 ULP for non-cumulative
derived floats, and `32*G` ULP for cumulative log-Z. The stage-scaled log-Z
bound covers roundoff accumulation and cancellation while remaining below
`1e-13` absolute error in the observed hard-screen worst case. This portability
erratum is required because NumPy 2.3 and 2.4 can differ by a few ULP in `exp`;
it does not alter a sampling gate or permit a discrete decision to differ.

## 5. Frozen screen and conditional rescue

The hard cells are, in order:

```text
m06_c00, p=.04, d00, attempt022
m08_c06, p=.04, d00, attempt022
```

Each base method uses `M=8`, `N=256` populations on both cells:

| Method | Kernel | G | sweeps/stage |
|---|---|---:|---:|
| `C192-2` | coordinate | 192 | 2 |
| `B96-1` | block4 | 96 | 1 |
| `B192-1` | block4 | 192 | 1 |
| `B96-2` | block4 | 96 | 2 |

This is 64 population tasks. If zero methods pass, discovery is immediately
`EXHAUSTED`. If exactly one passes, and only then, run `B384-2` with block4,
`G=384`, two sweeps, `M=8,N=512` on the two hard cells (16 tasks). If at least
two base methods pass, rescue is forbidden. Rescue is also forbidden when two
passing methods exist but no pair is statistically consistent.

Eligible methods are ranked by total hard-cell core time; exact ties use smaller
`N*G*s`, smaller G, smaller s, coordinate before block4, then method ID. The
first consistent distinct-method pair becomes primary and backup.

## 6. Population, cell, and consistency gates

Every population must have zero residual, finite normalized weights, and an
exactly replayable identity/transcript. At every anneal step it must satisfy:

```text
conditional ESS / N >= 0.70
maximum pre-resampling normalized weight <= 0.10
post-decision ESS / N >= 0.50.
```

Final genealogy requires family ESS at least 4, at least 8 distinct initial
families, and maximum family mass at most 0.50. Across the eight populations of
a cell, median family ESS must be at least 8 and median distinct families at
least 16. There is deliberately no minimum unique-label, logical-flow,
acceptance, PT round-trip, or R-hat gate.

Each population produces a weighted logical-label distribution `P_r`. For
eight populations, the estimator averages the 28 cross-population collisions:

```text
q_top = (mean_{r<s} <P_r,P_s> - 2^-k) / (1 - 2^-k).
```

It is not clipped. MCSE is the delete-one-population jackknife; particles and
the 28 pairs are not treated as independent samples. Hard cells require
`SE(q_top) <= .05`. A method pair must satisfy on both cells

```text
abs(delta q_top) <= .06
abs(delta q_top) <= 3 sqrt(SE_a^2 + SE_b^2) + .005.
```

Planted mass, log-Z, unique states/labels, and final logical flow are diagnostics
only. Planted mass is never named q_top.

## 7. Blinded confirmation and resolution

After selecting primary/backup and before opening any fresh result, the worker
must atomically freeze both task manifests and the freeze record. The 17-cell
confirmation panel has ordered SHA256
`8f2c1a6d60f346ecc5bf703f7e5d0d17d068462f978c78dd937ace0fb98b41be`.
Both methods rerun all 17 cells with `M=8,N=512`, including the hard cells, for
272 tasks. Cell identity is the complete tuple
`(code_id,p,disorder_index,disorder_source)`.

The six-cell resolution panel has ordered SHA256
`03f9b16dbc0cc52ee18313cdf57fd25ea4db50f44687971bedac53662b275c22`.
Both methods rerun it under an independent `resolution` trajectory namespace
with `M=8,N=256`, for 96 tasks.

Every N512 confirmation cell requires `SE(q_top)<=.03`. Primary versus backup
on all 17 cells, and N256 versus N512 within each method on all six resolution
cells, must satisfy both

```text
abs(delta q_top) <= .04
abs(delta q_top) <= 3 sqrt(SE_1^2 + SE_2^2) + .005.
```

Any complete numerical failure is `EXHAUSTED`; missing non-conflicting evidence
is `INCOMPLETE`. Fresh results cannot cause extra methods, particles, stages,
sweeps, disorders, or weaker gates to be added.

## 8. Runtime and stop rules

Linux benchmarking warms each m6/m8 and coordinate/block4 path, then runs two
populations at `N128,G96` and two at `N256,G192`. A differential slope is used
for projection. Hard screen may start only if all of these pass:

```text
m8 slowest kernel <= 200 us / particle-sweep
startup <= 120 s
every population task <= 20 min predicted
full worst-case schedule, with factor-2 safety, <= 180 min predicted.
```

The worst-case projection uses the frozen nd-2/nd-3 contingency capacity, so
it remains valid when a busy nd-1 is excluded. The stage launcher accepts only
a clean-source Linux runtime report whose archive and manifest hashes match the
deployment. The hard-screen launcher has an independent two-hour timeout.

The remote wall limit is four hours. If two hard-pass candidates do not exist
by two hours, stop without compressing confirmation. Capacity defaults are
`nd-1=75`, `nd-2=75`, `nd-3=91`; if nd-1 is busy, ownership is frozen on
nd-2/nd-3 before the stage. Tasks never migrate inside a stage.

## 9. PT transport autopsy

The autopsy replays exactly four old D0/D4, S64 cell-config tasks (16 instance
trajectories), with burn 2000 and measurement 8000. Its config binds each old
raw SHA, old task fingerprint, old uniform and instance seeds, and parent source
`da69528b43f4a9d1635083c21d713ba63ccec4ab`. It records round permutations,
replica weights/labels/phases, within-round rung extrema, phase/direction edge
counters, endpoint events, first passages, churn, and post-hot return lags.
Parent paths retain the canonical `transport/5480511a57d1/` control-hash layer;
only the local read-only cache resolver may accept the flattened retrieval layout.
Instrumentation consumes no randomness and must reproduce every parent label,
swap/logical counter, transport total, and residual bit-for-bit.

Classification is:

- conditional edge rate below .05 while aggregate is at least .05:
  `CONDITIONAL_EDGE_BOTTLENECK`;
- no conditional narrow edge but no hot relaxation/frontier arrival:
  `GLOBAL_DIFFUSION_OR_RELAXATION_LIMITED`;
- certified hot update followed by a return-conditioned rate below .05:
  `POST_HOT_HYSTERESIS`;
- fewer than 200 attempts for the condition needed by that branch:
  `INCONCLUSIVE`.

Autopsy explains the old failure only. It neither certifies PA nor opens a new
PT parameter search.

## 10. Boundary after success

Only two fully passing, mutually consistent methods can produce
`READY_FOR_FORMAL`. A later, separate clean commit must define
`exp102.q0_pa.v1 / exp102.scan.v2`, run each m's 96-cell tuning and 448-cell
held-out panels, and pass all six m values before creating
`FROZEN_HELD_OUT_PASS`. Until then, no production task plan or production run
may exist.
