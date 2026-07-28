# 060 q=0 joint collapsed-B block structure

Status: `PRE-RUN / NO WIDTH REPORT / NO SAMPLER RAW`.

This local-only validation freezes exact multi-row and row-column-cross factor
scopes plus a deterministic min-fill order before inspecting the m8 widths.
It is a structural resource screen, not a sampling experiment.  Maximum
authority is `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND`; see
`PRE_RUN_RED_TEAM.md`.

Validation 056 shows that exact one-column updates do not equilibrate the m8
hard sentinel, validation 058 shows that exact one-row updates are essentially
the identity in the low-energy basins, and validation 059 shows that their
sequential hybrid freezes U in a wrong basin.  Validation 060 therefore asks
only whether one *joint* conditional has a manageable frozen factor graph.  A
small width is not evidence of movement, convergence, posterior mass coverage,
or an independent confirmation of HP64.

## Frozen source protocol

The analyzer is deliberately unusable from an untracked or dirty draft.  Its
launch guard requires every configured source, documentation and input artifact
to be tracked byte-for-byte at `HEAD`, requires the complete worktree to be
clean, and refuses to run if any validation-060 output already exists.  The
report binds the commit and a canonical source-tree SHA.  This prevents the
historical draft failure mode in which an untracked script could write an
unrelated `source_commit` into a report.

After the draft and its guard/auditor/tests have been reviewed and committed
from a clean canonical worktree, the one-shot local sequence is:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python -m pytest -q -p no:cacheprovider \
  data/expander_code/exp102/validation/060_q0_multirow_joint_block_structure_20260724/test_structure_logic.py

PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/060_q0_multirow_joint_block_structure_20260724/preflight_structure.py

PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/060_q0_multirow_joint_block_structure_20260724/analyze_structure.py

PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/060_q0_multirow_joint_block_structure_20260724/audit_structure.py
```

The terminal structural result is trusted only when
`independent_structure_audit.json` says
`INDEPENDENT_STRUCTURE_AUDIT_PASS`.  The auditor does not import the analyzer:
it rebuilds scopes from direct one-coordinate `B H` perturbations and
reimplements min-fill with integer bitsets.  Any disagreement is `CONFLICT`,
not permission to edit or rerun the same source identity.

## Result boundary

`MR2/MR3/MR4` represent all unordered output-row subsets; their full-scope
graphs are isomorphic under output-row relabeling, so one representative graph
per row count is a conservative complete-family width check.  `RC1` evaluates
all 24 selected B columns; its selected output row is likewise structurally
isomorphic.  Focused tests verify both claims by direct perturbation.

If no family passes, this frozen exact local joint-block family closes within
the stated factorization/order/memory boundary.  If a family passes, the report
names one outcome-blind preferred contingency candidate by block variables,
worst width, worst single-table bytes and frozen candidate order.  Per
`EXP102_PLAN_ASSESSMENT_20260728.md`, implementation remains suspended unless
fresh HP64 Stage 3 or Stage 4 fails.  Even then it receives at most one fresh
exact implementation successor and cannot serve as the required large-k
orthogonal confirmer.
