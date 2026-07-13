# exp101.physics.v2 validation 014 evidence index

Overall certification evidence: `PASS`. This file does not itself change `status.md`.

## Reproducibility

- Contracts: `exp101.physics.v2` / `exp101.scan.v2`.
- Python/NumPy: `3.12.12` / `2.4.1`.
- Conda environment: `12`.
- Git SHA / dirty: `eb3403a0b3c5b6d453929d18d119f6a4dd851338` / `True`.
- Implementation fingerprint: `bc3867f359bdc14e2dee10535e21064c75df21d0ebef5b5102d973bd5d688ae2`.
- Full pytest exit/log SHA256: `0` / `64fb7379524732a441944f00533e920be7e82a49ca77149bfaf7c4a6329aef33`.

## Decisive evidence

- Exact raw-paper/reduced enumeration: `PASS`; see `exact_reduction_evidence.json` and `.md`.
  This artifact also records fixed-y truth independence off the kernel, shifted-coordinate equality, q=0 true/legacy separation, alias routing, absolute/relative characters, and posterior bounds.
- Fresh PT/aggregation integration: `PASS`; see `pt_aggregation_evidence.json` and `.md`.
- Complete exp101 pytest suite: `PASS`; see `pytest_full_output.txt` and `pytest_exit_code.txt`.

## Coverage map

| Contract area | Decisive test | Suite status |
|---|---|---|
| CSS move completeness prevents exact/MCMC q=0 support mismatch | `tests/test_physics_golden.py::test_incomplete_q_zero_move_set_is_rejected_at_model_assembly` | PASS |
| raw/reduced, shifted-coordinate, fixed-y, and x/z convention | `tests/test_paper_reduction_golden.py::test_fixed_effective_syndrome_is_truth_independent_off_kernel` | PASS |
| canonical aliases, posterior statistics, and unphysical bounds | `tests/test_model_observables.py::test_unphysical_debiased_purity_is_retained_without_success_bounds` | PASS |
| boundary-only invariance and logical-shift counterexample | `tests/test_section_frames.py::test_logical_section_shift_is_not_claimed_as_gauge` | PASS |
| population weighting, U-statistic, jackknife, and FPC | `tests/test_model_observables.py::test_delete_one_jackknife_tracks_repeated_sampling_error` | PASS |
| large-k TI rejection and gap-only diagnostics | `tests/test_sector_ti.py::test_diagnostics_match_exact_gaps_and_expose_no_purity` | PASS |
| independent-sector TI bootstrap uncertainty | `tests/test_sector_ti.py::test_independent_sector_resampling_avoids_false_zero_gap_error` | PASS |
| analytic p=0 and p=0.5 full-sector TI endpoints | `tests/test_run_scan.py::test_auto_full_ti_endpoints_are_analytic_end_to_end` | PASS |
| three-way auto routing and large-k preflight refusal | `tests/test_run_scan.py::test_auto_routes_all_three_production_paths` | PASS |
| actual k=16 observable construction preserves all 80 characters | `tests/test_run_scan.py::test_actual_k16_observable_set_keeps_16_plus_64_characters` | PASS |
| manifest/NPZ schema and source/config/cache identity isolation | `tests/test_run_scan.py::test_source_fingerprint_participates_in_chunk_identity` | PASS |
| chunk outer/inner task identity tamper rejection | `tests/test_run_scan.py::test_mismatched_inner_task_fingerprint_recomputes_chunk` | PASS |
| Git dirty provenance uses boolean value plus known marker | `tests/test_scan_estimators.py::test_unknown_git_dirty_state_uses_known_marker` | PASS |
| deprecated aliases normalize before seed/result/manifest storage | `tests/test_run_scan.py::test_alias_is_canonical_before_seed_and_manifest` | PASS |
| legacy sampled output is formal-only and excluded from aggregation | `tests/test_run_scan.py::test_legacy_sampled_scan_is_formal_only_end_to_end` | PASS |
| INVALID-safe mean/SEM/crossing aggregation | `tests/test_scan_estimators.py::test_sem_uses_two_valid_samples_and_excludes_invalid` | PASS |
| PT endpoint state machine and fresh-phase round trips | `tests/test_pt.py::test_new_phase_does_not_inherit_partial_transit` | PASS |
| current narrative and PRE_ALIGNMENT overwrite guards | `tests/test_contract_text.py::test_historical_runners_cannot_overwrite_alignment_warning` | PASS |

Machine-readable inventory and hashes are in `environment.json`.
