# exp102 q=0 global preflight runtime exhaustion

Status: `RUNTIME_EXHAUSTED / DISCOVERY CLOSED BEFORE SCREEN`.

This is a fail-closed resource result, not a sampler or physics result. No bias,
screen, HARD2-fresh, confirmation, resolution, TI-anchor, held-out, or
production task ran. The full-range discovery outcome is
`UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`, never `IMPOSSIBLE`.

## Immutable attempt

- Run: `exp102_q0_global_20260721_204b37d`
- Source: `204b37d8e00e7d11ffa2b6766b90d947892e179d`
- Archive SHA256:
  `1583dce6b8bb81ad7780f323d21300b158ad435d710f3c0226b7b3028b8eb7f7`
- Source-manifest SHA256:
  `b69290798a11a3bf548483c6e223f96a64e0d9c7be0e48b89fa6e54a28a57ea3`
- Schedule-file SHA256:
  `7874a0d967ba866d8834cf380b408947af614bdf3bec7b50c0f30fb4a332465c`
- Schedule identity SHA256:
  `35e08b457f6a96eea252bc8d6653950aecb231ac85b6ac66c129f799ca0d02c1`
- Stage fingerprint:
  `0ae347a314c23ecba2d7239af6d6203f6e4e8f91dd5111810bdb3a1fb89ab538`

The local clean archive passed the combined exp102+exp101 suite (`590 passed`)
and a post-test full-tree verifier. On Linux, all three verified-source workers
completed with exclusive `SUCCESS` markers. nd-1 and nd-2 reported `587 passed,
3 skipped`; nd-3 reported `590 passed`. The three canonical digests were
identical:

`a3730d7380575976f88e35f5490b24a9b6949e3817b2fb3880775736cf2ad364`.

The bounded WMC feasibility check returned six `INCONCLUSIVE_WIDTH` records.
It supplies no exact or rigorous-width certification and does not change the
resource decision.

## Runtime decision

Every hard-coset and defect-trace candidate remained eligible at T3. The
factor-two complete-schedule projections were far below the frozen 58-hour
limit:

| Node | T3 schedule projection | TI contingency projection | TI gate |
|---|---:|---:|---|
| nd-1 | 1.3074 h | 78,705.4 s | PASS |
| nd-2 | 2.4408 h | 116,274.9 s | `RUNTIME_EXHAUSTED` |
| nd-3 | 2.0548 h | 251,240.7 s | `RUNTIME_EXHAUSTED` |

The required m3 full-sector TI anchor has a frozen 79,200-second confirmation
window. Consensus uses the worst of all three node reports, so nd-2 and nd-3
close the preflight even though the new samplers themselves fit. Skipping TI,
selecting only nd-1, repeating until a favorable timing appears, changing the
safety factor, or extending the window would violate the frozen contract.

The source used for this run correctly persisted each node's terminal runtime
status, but its combiner accepted only node-level `PASS` and therefore raised
before writing aggregate `runtime_consensus.json` and `preflight_report.json`.
The subsequent audit repair accepts and verifies both legal node statuses,
persists aggregate `RUNTIME_EXHAUSTED`, and leaves every downstream stage gated
on aggregate `PASS`. This run was not retried and its evidence was not changed.

## Evidence audit

`remote_run/` is the complete 80 KiB run tree and `remote_logs/` contains the
three worker logs. `EVIDENCE_SHA256SUMS` binds every copied file. After this
verification, the 25 MiB reconstructible remote deployment/Numba cache was
removed; the server `runs/` tree and logs remain intact. Verify locally with:

```bash
cd data/expander_code/exp102/validation/010_q0_global_runtime_exhausted_20260721
shasum -a 256 -c EVIDENCE_SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python verify_evidence.py
```

The independent verifier checks source/config/registry/schedule identities,
exclusive markers, pytest-log hashes, all three digest/runtime reports, WMC,
and the absence of sampler raw. Its terminal status is
`VERIFIED_RUNTIME_EXHAUSTED`.
