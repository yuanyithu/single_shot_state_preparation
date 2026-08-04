# exp102 status

## Current state

**`BLOCKED_BEFORE_REMOTE`**

exp102 remains pre-pilot. There is no certified cell or `(m,p)`, no `READY_FOR_FORMAL`, no `FROZEN_HELD_OUT_PASS`, and no production result. Diagnostic, calibration, implementation, runtime, or audit passes do not raise this authority.

The full-range conclusion remains `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`. It is not `IMPOSSIBLE` and is not a physical failure of any parameter point.

No remote sampler, formal tuning, held-out work, production, raw extension, or post-hoc gate change is authorized by the current evidence. Any successor requires a fresh user-approved scientific contract, fresh seeds/raw, and the red-team required by root `AGENTS.md`.

## Open blockers

All four blockers remain open:

1. `LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN`
2. `FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE`
3. `CAMPAIGN_BUDGET_UNAPPROVED`
4. `STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN`

Worker-level `SUCCESS`, a local calibration PASS, or a same-family method PASS cannot clear these blockers. Downstream stages require an explicit aggregate PASS and the authority stated in their contract.

## Recent evidence

### Validation 066: delivery gate calibration

Terminal status: `LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`.

The local full-label `q_top`/`D2_norm` gate has a confirmed operating point in its frozen same-environment calibration. The independent audit reproduces that result. Deliberately common-wrong controls also pass, confirming a real common-failure blind spot. Therefore 066 calibrates only the scalar comparison gate; it does not prove mixing, transport, target-basin coverage, unvisited-tail coverage, or posterior correctness, and it authorizes no sampler stage.

Authority and complete evidence: `validation/066_q0_delivery_gate_redesign_20260728/README.md` and its report/audit files.

### Validations 063 and 065: Nishimori auxiliary audit

Validation 063 remains `NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`. Validation 065 remains `CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS`; its verifier confirms that the conflict was recorded correctly, not that the numerical audit passed. Terminal gate invariance does not upgrade a payload mismatch to an audit pass. Nishimori remains an auxiliary diagnostic without a universal `q_top`-error or independent-confirmation guarantee.

Authority and complete evidence: `validation/063_q0_nishimori_auxiliary_calibration_20260728/README.md` and `validation/065_q0_nishimori_audit_rebind_20260728/README.md`.

### Validations 060 and 064: structure and resources

Validation 064 remains `RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE`: its timing matrix is scenario arithmetic, not complete campaign coverage or budget approval. Validation 060 is `LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND`, with MR2 only as a suspended same-family structural contingency; it is not an orthogonal confirmer and authorizes no remote run.

Authority and complete evidence: `validation/060_q0_multirow_joint_block_structure_20260724/structure_report.json`, `validation/060_q0_multirow_joint_block_structure_20260724/independent_structure_audit.json`, `validation/064_q0_hp64_resource_calibration_20260728/resource_calibration_report.json`, and `validation/064_q0_hp64_resource_calibration_20260728/independent_package_audit.json`. The two README files are source-bound pre-run documents, not terminal summaries.

## Evidence map

- `validation/INDEX.md`: one-line ledger for validations 001--066; read this first.
- `HISTORY.md`: archived pre-cleanup context and full chronological status; not live authority and not a default read.
- `validation/<NNN_...>/`: per-validation evidence; use its README when that file states the terminal result, otherwise use the final report/audit in the directory. Enter through `validation/INDEX.md`.
- `EXPERIMENT_CONTRACT.md`: frozen exp102 experiment contract.
- `GLOBAL_DISCOVERY_CONTRACT.md`, `GLOBAL_SCREEN_DIAGNOSTIC_CONTRACT.md`, `HGP_GLOBAL_SCREEN_CONTRACT.md`: method-specific historical contracts.
- `reviews/`: strategy reviews and the approved context/worktree cleanup rationale; reviews do not change physics authority.

When a summary conflicts with a frozen contract or validation evidence, preserve the narrower permission and stop for review. Historical run-level hashes, measurements, and prohibitions are intentionally kept out of this live status; retrieve them from the validation README/report or `HISTORY.md` only when the task needs them.
