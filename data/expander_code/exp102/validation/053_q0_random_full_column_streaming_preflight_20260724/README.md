# Validation 053: exact streaming full-column preflight

Terminal status: **`CONFLICT` (and independently runtime-exhausted); no T1 or
measurement raw exists.**

This outcome-blind implementation/runtime preflight compares the historical
dense full-column CDF with the fresh single-buffer Numba streaming CDF.  It
does not run a T1 chain or estimate `q_top`.

The frozen gates and permissions are in
`reviews/RANDOM_FULL_COLUMN_STREAMING_REVIEW.md`.  Node work runs only through a
verified clean-source archive and immutable stage markers.  The aggregate
requires identical source identities, complete m8 CDF digest catalogs, and
portable sampling/replay transcript catalogs on nd-1/nd-2/nd-3.  A local pass
can only authorize this fresh three-node preflight; only aggregate `PASS` can
authorize a separately frozen T1 successor.

## Frozen deployment

- Run: `exp102_q0_streaming_preflight_20260724_de68bbc`
- Source: `de68bbc06aa729063b24c1f40ba23cc404a44c9c`
- Archive SHA256:
  `e8f14f856cad43d8bf787d7954990054a989a3b49e41efdfa3209ee279986586`
- Manifest SHA256:
  `08ddeff372f05d8f296893fce37815e97aabc4dd2678495ce4b3079e28460271`
- Config SHA256:
  `6dbd28a893ae6c1532c044ef98645dfa67fec852e19fff50379da4b3eb81a899`

The authoritative macmini report from this source passed all local gates.  All
12 complete `2^24` CDF comparisons were byte-identical, the median speedup was
`4.939112634x`, and the worst four-worker replay-inclusive T1 projection was
`2432.393551s`.  `superseded_local_preflight_7d57bcb.json` is retained only as
historical evidence from the earlier implementation commit; it does not
authorize this deployment.

## Three-node result

All node stages and the aggregate stage completed successfully as programs,
but each node report correctly failed closed as `CONFLICT`.  On every Linux
node exactly one of the 12 legacy/streaming comparisons failed:
`U0,column=11`.  The streaming CDF SHA for that entry was nevertheless the
same on macmini and all three nodes,
`394e6b11aa0f39219bf079d4acd05e256292fd258c7a5a3f07b19878b05e7f7a`.
The complete streaming CDF digest catalog and all four portable sampling plus
replay transcript hashes also agree across all four machines.  Thus the
observed byte mismatch is in the Linux legacy dense floating reference versus
the portable streaming result, not a cross-node disagreement of the proposed
streaming sampler.  The preregistered contract still makes any such mismatch
terminal `CONFLICT`; it is not waived after seeing the result.

The implementation also failed its independent resource gates, so resolving
the reference comparison alone would not authorize T1:

| node | median speedup | worst projected replay-inclusive T1 (s) |
|---|---:|---:|
| nd-1 | 2.591095774 | 8797.829313 |
| nd-2 | 2.537205025 | 9144.887582 |
| nd-3 | 1.382317098 | 17760.299181 |

The frozen requirements were speedup at least `4.2x` and projection at most
`7200s`.  Aggregate SHA256 is
`628f67a807f1d7d30eeca22efe773048aef85866f2928f1538a325c127e959cd`.
The conda-12 independent audit verifies canonical JSON, all report and
aggregate self-hashes, source/config bindings, stage markers, log equality,
the sole mismatch identity, runtime arithmetic, cross-machine CDF catalogs,
and portable transcript catalogs.  Its status is
`INDEPENDENT_AUDIT_PASS_CONFLICT_AND_RUNTIME_EXHAUSTION_CONFIRMED`, audit
SHA256
`6426a1a01c01747f474d587a10cdb6db9e53db09112193499a8f9307adb7640f`.

No convergence claim, physical result, m6/HARD2 authorization,
`READY_FOR_FORMAL`, held-out pass, or production authority is created.  A
successor must use a fresh implementation contract and source.  It may not
reinterpret this run as PASS, shorten the frozen T1 clock, drop replay, remove
P/U/MAP/S adversarial starts, or weaken the convergence gates.
