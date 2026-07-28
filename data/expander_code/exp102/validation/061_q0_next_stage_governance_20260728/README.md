# 061 Exp102 next-stage governance

Status: `STAGE0_GOVERNANCE_REPORT_COMPLETE`.

This local-only Stage-0 validation inventories the dirty root without changing
it and produces an evidence/authority matrix before any new sampler work. Its
maximum authority is `STAGE0_GOVERNANCE_REPORT_COMPLETE`.

The inventory records every modified or untracked file outside deployment
worktrees, hashes its current bytes, and compares an overlapping path against:

- `origin/main@bacf25a53870d04dc538d1caaee3336c8fe1be7a`; and
- the existing `direct_block_335f808` draft worktree.

Registered worktrees are listed separately with collapsed porcelain status.
See `PRE_RUN_RED_TEAM.md` for the mutation, authority, and interpretation
boundaries.

Run from the bacf25a successor worktree with local conda environment `12`:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/061_q0_next_stage_governance_20260728/reconcile_governance.py \
  --dirty-root '/Users/jarvis/Desktop/sync/project D' \
  --direct-block '/Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/direct_block_335f808'
```

Expected generated evidence:

- `dirty_root_inventory.json` and `.csv`: full path-level SHA inventory;
- `worktree_status.json`: registered worktree identities and collapsed status;
- `RECONCILIATION_REPORT.md`: concise comparison and unresolved ownership;
- `evidence_authority_matrix.json` and `.md`: validation-level authority;
- `stage0_manifest.json`: SHA-256 and size of every generated report.

This stage never copies or chooses a dirty file. A later implementation stage
must make every reconciliation choice explicitly from this evidence.

## Completed result

The conda-12 run bound both the canonical worktree and `origin/main` to
`bacf25a53870d04dc538d1caaee3336c8fe1be7a`. It inventoried 1,472 dirty-root
files outside deployment worktrees: 5 tracked modifications and 1,467
untracked files. Twenty paths overlap `origin/main` and the direct-block
draft; seven are byte-different and remain unresolved ownership decisions.

The evidence matrix contains one row for each validation 001--060 plus
method-level rows for validation 013. Validation 013 remains
`UNRESOLVED_MAP_MIXTURE_FAIL`; validation 060 remains an untracked
`PRE-RUN / NO WIDTH REPORT / NO SAMPLER RAW` draft. No cell, formal, held-out,
or production authority was created.
