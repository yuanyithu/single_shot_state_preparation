# UASRE V0 local result

## Terminal status

`LOCAL_AUXILIARY_STABILIZER_TRANSPORT_UNRESOLVED`

This is a local adversarial-initialization diagnostic only. It is not a
posterior estimate, `q_top` result, HARD2 authorization, formal pilot,
held-out result, or production authorization.

## Frozen evidence

- Frozen cell: `m08_c06,p=.04,d00,attempt022`, with its nonzero hard
  syndrome.
- Config SHA256: `2154b05d8a21ca7eaa6f653eca62356bf8f91c5a73fee7790b2303f68122ceea`.
- Manifest SHA256: `1c5b931117a35b859c33a1a1abe348d0f8e547784395812e2ccb3884b2271c29`.
- Source-binding SHA256: `ddd773fe5b995f6f70812a91ccdc9c37d0e68a62a73c2cb115caabea110ca415`.
- `RUN_COMPLETE.json`: all 48 immutable raw files, run SHA256
  `c262bc5f9b6320d22fb066a3d70a61783fce5f1479fee437c50f1c4d23e9261f`.
- The manifest-bound runner validates the raw without replay and reports
  SHA256 `dd42401222d64ab22b01c361d14bab096eb8291f45254edc834be0f8e6bf7aba`.
- A separate six-worker validator repeats every trajectory bit-for-bit;
  `all_bit_identical=true` for all 48 tasks, with replay SHA256
  `d99d0b27d8edb13c3b58bce4d05b15974befa281146b0ca19c71e02f5591b669`.
- The pickle-free independent audit does not import the sampler or runner.
  It rebuilds the hard-coset algebra, starts, scores, packed traces, and
  pre-registered gates, and reports SHA256
  `646c0ee7f40bac604adbd5c206c7bc25164b5fcc9c291d21e4baa8af5e09becf`.
- The crosscheck confirms the manifest, raw, replay, and audit identities;
  all pre-registered gate summaries agree, with SHA256
  `485e1cd4f6f2bc01902bb3c8a2342c80a2a23d3b377ce16b57f9eb242f8d2966`.

`RUNNING.json` is a stale runner marker: `RUN_COMPLETE.json`, `SUCCESS.json`,
the replay, audit, and crosscheck prove that the run is complete. Here
`SUCCESS.json` means that terminal analysis completed, not that either sampler
passed.

## Why both candidates fail

Both frozen configurations, `UASRE32-R1-A1` and `UASRE64-R1-A1`, use the same
fixed `(burn, measurement)=(256,2048)` schedule and eight independent
trajectories in each of three legal initialization families:

- `P`: planted error;
- `U`: exact K=0 uniform hard-coset draw;
- `L`: a legal low-energy state with a different logical label.

The P/L pairwise comparisons agree, which is deliberately not treated as a
mixing proof. U remains strongly separated from both P and L. For both
replica counts, P/U and U/L fail the predeclared comparisons of normalized
weight, normalized B weight, complete score, all 128 logical characters, and
most B-mask means. U also fails fixed-clock early/late stability. Some P/L
B-mask time-block checks fail as well, so neither configuration passes every
frozen gate.

The observed U minimum measurement weights are 135--174 for UASRE32 and
163--179 for UASRE64, while a known legal P state has weight 63. Unlike the
earlier UARE and full-row-Gibbs failures, the deliberately conservative
target-support bound is inconclusive at these weights: it is capped at 1 for
every U trajectory. This result therefore does not claim that U is trapped in
a proven negligible-posterior region. The fail-closed conclusion follows from
the independent cross-family and fixed-clock distribution disagreements.

All 64 B-mask diagnostics are nonconstant for every method/family. State
movement, label movement, or local B movement is consequently not enough to
establish global equilibration. Replacing U and L by a common P-like or
physical-zero start would hide the tested failure: physical zero is outside
this nonzero-syndrome hard coset, and shifted-coordinate zero is already P.

The result rejects only these two frozen UASRE configurations at this fixed
local budget. The raw must not be extended, pooled, reweighted, used for
`q_top`, sent to HARD2, or used to authorize remote work,
`READY_FOR_FORMAL`, tuning, held-out, or production. It does not establish
that q=0, the posterior, or auxiliary-stabilizer replica exchange is
mathematically impossible. A successor must use a materially new reviewed
mechanism, fresh manifest and seeds, and an independent confirmation route.
