# Completed diagnostic-screen evidence

This metadata-only bundle verifies completed run
`exp102_q0_screen_diagnostic_20260721_342dd5b` and its terminal status
`UNRESOLVED_NO_HARD_COSET_PASS`.

The 15 bias NPZs and 1280 measurement NPZs are intentionally excluded from
Git. `RAW_SHA256SUMS` records all 1295 raw hashes, while the four node raw
manifests bind those hashes to the frozen controls, ownership, source,
schedule, and seven exclusive SUCCESS markers. The complete raw tree remains
under `data/expander_code/exp102/raw/screen_diagnostic/` and in the remote
`runs/` backup.

`remote_evidence/` is the immutable server metadata. `independent_replay/`
was produced locally in conda `12` by running `analyze_screen` through the
same verified source archive with eight workers. Both analyzers replay every
raw state, label, weight, counter, and bias transcript exactly and produce the
same gates and terminal status. Their reports have exactly 80 audited derived
float differences: 62 `core_seconds` and 18
`min_nondegenerate_bulk_ess` values, all within 4 ULP and with maximum
absolute difference `1.8189894035458565e-12`. Each report, decision, and
package retains its own valid canonical self-hash and file links.

Verify the closed evidence set with:

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp102/validation/011_q0_global_screen_diagnostic_20260721/completed_run_evidence/verify_evidence.py
```

Expected status:

```text
VERIFIED_UNRESOLVED_NO_HARD_COSET_PASS
```

The verifier rejects optimized Python, missing or extra evidence files,
tampered source/control/marker identities, raw-manifest disagreement, any
authorization flag, any scientific-field disagreement, and float differences
outside the two-field 4-ULP whitelist. `EVIDENCE_SHA256SUMS` closes the bundle.
