# Exp102 Stage-0 reconciliation report

Generated: `2026-07-28T12:38:22.419997+00:00` under conda environment `12`.

This report is read-only evidence. It does not reconcile, copy, delete, merge, commit, or launch anything.
The complete path-level inventory is in `dirty_root_inventory.json` and `.csv`.

## Source identities

- Dirty root HEAD: `de68bbc06aa729063b24c1f40ba23cc404a44c9c`; branch `main`.
- Canonical worktree HEAD: `bacf25a53870d04dc538d1caaee3336c8fe1be7a`; expected `bacf25a53870d04dc538d1caaee3336c8fe1be7a`.
- Baseline `origin/main`: `bacf25a53870d04dc538d1caaee3336c8fe1be7a`.
- Dirty root versus baseline: ahead `0`, behind `17`.
- Direct-block draft HEAD: `bacf25a53870d04dc538d1caaee3336c8fe1be7a`.
- Deployment subtree excluded from dirty-root enumeration: `data/expander_code/exp102/deployment_worktrees`.

## Inventory summary

- Dirty modified/untracked file records outside deployment worktrees: **1472**.
- Status counts: `{' M': 5, '??': 1467}`.
- Paths also present in origin/main: **20**; byte-different: **7**.
- Paths also present in direct-block draft: **20**; byte-different: **7**.
- SHA values are SHA-256 of regular-file bytes; symlinks hash their link-target bytes.

## Tracked dirty paths

| XY | Path | Current SHA-256 | Same as origin/main | Same as direct-block |
|---|---|---|---:|---:|
|  M | CLAUDE.md | 9cd8ca8d75847f2e31d6da1c92c5fd467e28573b251ab683f548dd460ef66fc3 | False | False |
|  M | data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed.py | a383dc6f076f92ca63c09e8f4c5c0be3dbb61eea8bd16d1ecd7405c49412439c | False | False |
|  M | data/expander_code/exp102/exp102_pipeline/q0_logical_stratified.py | 6fe9693b1b7f08069d8a6da75a5a44e6f552ae2cfa6e51bd632d90a1b86c9b3b | False | False |
|  M | data/expander_code/exp102/tests/test_q0_logical_stratified.py | 9b71a96d4a76e190cfb5ea6d968541bf364450af5cfafbe5578b0b0caf140638 | False | False |
|  M | data/expander_code/exp102/validation/015_q0_logical_stratified_v0b_20260723/README.md | ef7bf50a888d1019f2302c2c637e630a5efe66925ae5cc4553de5958727129a9 | False | False |

## Overlap differences requiring an ownership decision

These rows exist in at least one comparison source but are not byte-identical to it. No choice is made here.

| Path | XY | Same as origin/main | Same as direct-block |
|---|---|---:|---:|
| CLAUDE.md |  M | False | False |
| data/expander_code/exp102/exp102_pipeline/q0_hgp_collapsed.py |  M | False | False |
| data/expander_code/exp102/exp102_pipeline/q0_hgp_full_row_gibbs.py | ?? | False | False |
| data/expander_code/exp102/exp102_pipeline/q0_logical_stratified.py |  M | False | False |
| data/expander_code/exp102/tests/test_q0_hgp_full_row_gibbs.py | ?? | False | False |
| data/expander_code/exp102/tests/test_q0_logical_stratified.py |  M | False | False |
| data/expander_code/exp102/validation/015_q0_logical_stratified_v0b_20260723/README.md |  M | False | False |

## Registered worktrees

Each worktree was queried with `GIT_OPTIONAL_LOCKS=0` and untracked directories collapsed; no deployment tree was recursively inventoried.

| Worktree | HEAD | Branch/state | Dirty | Status entries |
|---|---|---|---:|---:|
| /Users/jarvis/Desktop/sync/project D | `de68bbc06aa729063b24c1f40ba23cc404a44c9c` | refs/heads/main | True | 126 |
| /Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/direct_block_335f808 | `bacf25a53870d04dc538d1caaee3336c8fe1be7a` | detached | True | 5 |
| /Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/next_stage_20260728_bacf25a | `bacf25a53870d04dc538d1caaee3336c8fe1be7a` | refs/heads/exp102-next-stage-20260728 | True | 6 |
| /Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/rfcg_t1_6fa489f | `6fa489f838dffea15b07e1ef3b3fbee3951dd3c0` | detached | False | 0 |
| /Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/streaming_7d57bcb | `7d57bcbbf439eec8a4570c9769ba7f29ddd3aef0` | detached | False | 0 |
| /Users/jarvis/Desktop/sync/project D/data/expander_code/exp102/deployment_worktrees/streaming_de68bbc | `335f808e7d30242561ba74c20c21aac3fd31955e` | detached | False | 0 |

## Governance decision

- The dirty root is not a safe development or merge target.
- No root path was changed; byte comparisons are evidence for a later human-owned reconciliation.
- Canonical follow-up work must remain on the bacf25a-based successor branch and explicitly select any dirty-root draft it needs.
- Validation 060 is present as an untracked draft, remains `PRE-RUN`, and has no immutable or sampler authority.
- No remote, sampler, formal, held-out, or production stage is authorized by this report.
