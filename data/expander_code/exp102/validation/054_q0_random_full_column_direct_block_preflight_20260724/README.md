# Validation 054: direct-positive block full-column preflight

Terminal status: **`PASS`; this authorizes only a fresh direct-block m8 T1
diagnostic, not a convergence or physical claim.**

This outcome-blind implementation/runtime preflight is governed by
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_REVIEW.md`.  It evaluates the same exact
random-scan full-B-column heatbath as validations 052/053, using fixed direct
positive-weight subtotal blocks after an explicit normal-range certificate.

The first phase is local only.  A local PASS may freeze portable subtotal and
trajectory digests into a new clean source; it does not itself authorize a
remote T1, posterior estimate, `q_top`, HARD2, formal, held-out, or production
run.

The first local report from source `f5f2976922ced2276f3bcb890bf24410cbc1db00`
passed its frozen gates, but is superseded for remote authorization because its
runtime seed identity still depended on the config digest.  It is retained as
`superseded_local_preflight_f5f2976.json`; no remote work used it.

Source `a0d4dbf6451240f0c2e07057d45206427ef09db0` then replaced that circular
identity with the frozen `runtime_seed_key` and passed all local gates.  Its
report is retained as `reference_origin_local_preflight_a0d4dbf.json` and was
used, before any remote run, to create `portable_reference.v1.json`.  The
portable artifact freezes all 12 ordered block-subtotal digests and four
sampling/replay transcript digests.  The final clean-source local preflight
must reproduce this artifact before a three-node deployment is allowed.

## Final local preflight

The final clean source is
`61d605a5e27db0970457736c72d1c45d72a12b10`.  Its macmini report reproduced
the frozen portable artifact and passed every complete-m8 weight, underflow,
runtime, and focused-test gate.  The largest scaled absolute weight error was
`2.020606e-14` (gate `5e-13`), largest relative error `7.290711e-14` (gate
`5e-12`), largest total variation `4.148991e-15` (gate `2e-12`), and the
candidate log-weight lower bound was `-221.658`, far above the binary64
normal-underflow margin.  Direct-block was `2.62269x` faster than streaming;
the worst replay-inclusive T1 projection was `1282.48s`.

## Three-node result

- Run: `exp102_q0_direct_block_preflight_20260724_61d605a`
- Archive SHA256:
  `61bb87e70320f7371504ea99c320e49baf1140b4ac9d3050fc9a3b742d5a7bec`
- Manifest SHA256:
  `a6be723a7aa59b7d1305e518b859fdbef50f6b0f881ca08d04088ebe2dcdb49f`
- Aggregate status/SHA256: `PASS` /
  `27f6d276a219545bb45e48e827e06eb5dd45a328bb4523786c645a00010612bc`

All three Linux nodes exactly reproduced the 12 frozen block-subtotal digests
and four P/M0/S0/U0 PortablePrng sampling-plus-replay transcripts.  All
complete `2^24` direct-positive versus legacy-log weight checks passed.  The
runtime results were:

| node | speedup over streaming | worst replay-inclusive T1 projection (s) |
|---|---:|---:|
| nd-1 | 2.9866x | 4144.85 |
| nd-2 | 3.0900x | 4139.52 |
| nd-3 | 5.5412x | 5454.14 |

The worst projection is below the frozen `7200s` per-trajectory cap.  The
conda-12 independent audit does not call the preflight combiner: it verifies
canonical serialization, all self-hashes, source/config/reference bindings,
the frozen catalogs, every numerical/runtime gate and arithmetic derivation,
cross-machine consensus, stage markers, and log/report equality.  It reports
`INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED`, audit SHA256
`9646c6f92070024680728bf377e802e647b48a2b66ca6210c89c436fbd70f539`.

This PASS proves implementation identity, exact conditional-weight agreement,
portable replay, and frozen-resource feasibility.  It does **not** prove that
the Markov chain mixes, that `q_top` is correct, or that a cell is reportable.
The only successor authorized here is a fresh m8 T1 diagnostic with new
contract/seeds/raw and the full P/U/M0/M1/S adversarial initialization panel.
It may not reuse validation 052 seeds/raw, initialize every chain at P/zero,
weaken the B/global convergence gates, or proceed directly to m6, HARD2,
formal tuning, held-out, or production.
