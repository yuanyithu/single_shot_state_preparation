"""G2.7 单元测试：扫描入口（chunk 原子性/续采/merge schema/确定性）。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.run_scan import (
    ENGINE_FULL_TI,
    ENGINE_PT,
    ENGINE_Q0,
    PROTOCOL_VERSION,
    _build_specs,
    _model_cache_key,
    _resolved_engine_config,
    build_arg_parser,
    build_code,
    merge,
    resolve_engine,
    scan,
    state_prep_protocol_for_sector,
    task_seed,
)
from src.model import assemble_sector_model
from src.observables import build_observable_frame, build_observable_set

FAST_TI = dict(num_kp_grid_points=9, num_burn_in_sweeps=40,
               num_measurements=120, block_count=6, num_bootstrap=60)
FAST_DIRECT = dict(num_burn_in_sweeps=150, num_measurements=1200,
                   num_starts=4)


class TestTaskSeeds:
    def test_scope_and_determinism(self):
        a = task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 3, "s")
        b = task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 3, "s")
        assert a == b
        assert a != task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 4, "s")
        assert a != task_seed("fp", "x_error", "true_posterior", 0.1, 0.06, 3, "s")
        assert a != task_seed(
            "fp", "x_error", "legacy_delta_only", 0.1, 0.05, 3, "s"
        )
        assert task_seed(
            "fp", "x_error", "repo_compat", 0.1, 0.05, 3, "s"
        ) == task_seed(
            "fp", "x_error", "legacy_delta_only", 0.1, 0.05, 3, "s"
        )
        assert task_seed(
            "fp", "x_error", "paper_true_posterior", 0.1, 0.05, 3, "s"
        ) == a
        assert a != task_seed("fp2", "x_error", "true_posterior", 0.1, 0.05, 3, "s")


class TestBuildCode:
    def test_known_families(self):
        for family, size, expected_nk in [
            ("surface", 2, (5, 1)), ("toric", 2, (8, 2)), ("k43", 1, (25, 13)),
            ("expander34", 2, (100, 4)),
        ]:
            H_Z, H_X, logicals, meta = build_code(family, size)
            assert (H_X.shape[1], logicals.k) == expected_nk
            assert "classical_sha" in meta


class TestScanEndToEnd:
    def test_ti_scan_resume_and_schema(self, tmp_path):
        out = tmp_path / "scan_ti"
        npz_path, report = scan(
            out, "surface", [2], 0.12, [0.08, 0.15], 2,
            engine="ti", engine_config=FAST_TI,
        )
        assert (report["reused"], report["computed"], report["total"]) == (0, 4, 4)
        assert report["failed"] == []
        # 续采：全部 chunk 复用，结果不变
        with np.load(npz_path, allow_pickle=True) as data:
            q_top_first = data["q_top_estimate_per_disorder"].copy()
        npz_path2, report2 = scan(
            out, "surface", [2], 0.12, [0.08, 0.15], 2,
            engine="ti", engine_config=FAST_TI,
        )
        assert (report2["reused"], report2["computed"], report2["total"]) == (4, 0, 4)
        with np.load(npz_path2, allow_pickle=True) as data:
            assert np.array_equal(
                data["q_top_estimate_per_disorder"], q_top_first
            )
        # schema 字段
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            required_manifest_fields = {
                "protocol", "physics_contract_version",
                "scan_contract_version", "state_prep_protocol",
                "syndrome_semantics", "canonical_ensemble", "sector",
                "family_rule", "family_seed", "requested_engine",
                "resolved_engines", "resolved_engine_configs",
                "implementation_fingerprint", "git_commit_sha",
                "git_worktree_dirty", "per_size_code_fingerprint",
                "per_size_section_fingerprint",
                "per_size_observable_frame_fingerprint",
                "aggregation_policy",
                "planned_disorder_count", "present_disorder_count",
                "valid_disorder_count", "numerically_valid_disorder_count",
                "invalid_disorder_count", "missing_disorder_count",
                "aggregation_status_per_point",
                "aggregation_failure_reasons_per_point",
                "reportable_for_crossing_fss",
                "fraction_semantics",
            }
            assert required_manifest_fields <= set(manifest)
            required_npz_fields = {
                "u_bitmasks_per_disorder", "u_rand_seed_per_disorder",
                "character_count_per_disorder",
                "character_mask_per_disorder",
                "chain_character_means_absolute_per_disorder",
                "chain_character_means_relative_per_disorder",
                "m2_u_pooled_square_raw_per_disorder",
                "m2_u_debiased_per_disorder",
                "m2_u_debiased_jackknife_se_per_disorder",
                "q_top_estimate_per_disorder",
                "q_top_crossing_input_per_disorder",
                "q_top_estimator_name_per_disorder",
                "formal_q_top_per_disorder",
                "formal_q_top_estimator_name_per_disorder",
                "weights_absolute_per_disorder",
                "weights_relative_per_disorder",
                "formal_sector_weights_absolute_per_disorder",
                "formal_sector_weights_relative_per_disorder",
                "formal_sector_characters_absolute_per_disorder",
                "formal_sector_characters_relative_per_disorder",
                "posterior_purity_per_disorder",
                "posterior_mass_on_planted_class_per_disorder",
                "map_success_probability_per_disorder",
                "map_success_algebraic_lower_bound_per_disorder",
                "map_success_algebraic_upper_bound_per_disorder",
                "map_success_estimated_lower_bound_per_disorder",
                "map_success_estimated_upper_bound_per_disorder",
                "map_success_bound_kind_per_disorder",
                "map_success_bound_has_confidence_coverage_per_disorder",
                "mean_q_top_estimate", "disorder_sem_q_top_estimate",
                "conditional_mean_q_top_estimate_valid_only",
                "conditional_disorder_sem_q_top_estimate_valid_only",
                "aggregation_status_per_point",
                "aggregation_failure_reasons_per_point",
                "reportable_for_crossing_fss",
                "planned_disorder_count", "present_disorder_count",
                "valid_for_aggregation", "numerically_valid", "formal_only",
                "failure_reasons_per_disorder",
                "implementation_fingerprint_per_disorder",
                "git_worktree_dirty", "git_worktree_dirty_known",
                "task_fingerprint_per_disorder",
                "code_fingerprint_per_disorder",
                "section_fingerprint_per_disorder",
                "observable_frame_fingerprint_per_disorder",
                "observable_set_fingerprint_per_disorder",
                "pt_ladder_p_per_disorder", "pt_ladder_q_per_disorder",
                "pt_swap_rates_per_disorder",
                "pt_burn_in_round_trips_per_disorder",
                "pt_measurement_round_trips_per_disorder",
                "gate_diagnostics_json_per_disorder",
                "ti_endpoint_mode_per_disorder",
                "paper_aggregation_fraction", "numerical_pass_fraction",
            }
            assert required_npz_fields <= set(data.files)
            assert manifest["protocol"] == PROTOCOL_VERSION
            assert manifest["canonical_ensemble"] == "true_posterior"
            assert manifest["git_commit_sha"]
            assert isinstance(manifest["git_worktree_dirty"], bool)
            assert data["git_worktree_dirty"].dtype == np.dtype(bool)
            assert data["git_worktree_dirty_known"].dtype == np.dtype(bool)
            assert bool(data["git_worktree_dirty_known"])
            assert bool(data["git_worktree_dirty"]) \
                is manifest["git_worktree_dirty"]
            assert len(manifest["implementation_fingerprint"]) == 64
            assert manifest["state_prep_protocol"] == "plus_Zcheck_X"
            assert manifest["syndrome_semantics"] == "effective_y"
            assert manifest["aggregation_policy"] == {
                "point_eligibility": "all_planned_disorders_valid",
                "fraction_denominator": "planned_disorders",
                "maximum_invalid_disorders": 0,
                "maximum_missing_disorders": 0,
                "conditional_statistics_purpose": "diagnostics_only",
                "conditional_statistics_are_publication_eligible": False,
                "crossing_input_policy": (
                    "whole_point_nan_unless_reportable"
                ),
            }
            resolved_config = manifest["resolved_engine_configs"][
                ENGINE_FULL_TI
            ][0]
            assert resolved_config["num_measurements"] \
                == FAST_TI["num_measurements"]
            assert manifest["per_size_k"]["2"] == 1
            assert data["q_top_estimate_per_disorder"].shape == (1, 2, 2)
            assert data["weights_absolute_per_disorder"].shape[3] == 2
            assert data["delta_f_stderr_per_disorder"].shape[3] == 2
            assert np.isfinite(data["ti_grid_tv_per_disorder"]).all()
            assert np.all(
                np.char.str_len(
                    data["ti_diagnostics_json_per_disorder"].astype(str)
                ) > 0
            )
            assert data["character_means_absolute_per_disorder"].shape \
                == (1, 2, 2, 1)
            assert np.all(data["character_count_per_disorder"] == 1)
            assert np.all(data["character_mask_per_disorder"])
            assert data["lattice_size_list"].tolist() == [2]  # 兼容别名
            assert np.all(np.char.find(
                data["flags_per_disorder"].astype(str), "") >= -1)
            assert data["mean_q_top_estimate"].shape == (1, 2)
            assert str(data["implementation_fingerprint"]) \
                == manifest["implementation_fingerprint"]
            assert not data[
                "weights_are_exact_sector_posterior_per_disorder"
            ].any()
            assert data["weights_cover_all_sectors_per_disorder"].all()
            assert np.isnan(data[
                "map_success_algebraic_lower_bound_per_disorder"
            ]).all()
            estimated_finite = np.isfinite(data[
                "map_success_estimated_lower_bound_per_disorder"
            ])
            assert np.array_equal(
                estimated_finite, data["valid_for_aggregation"]
            )
            expected_kinds = np.where(
                data["valid_for_aggregation"],
                "full_sector_ti_plugin_no_coverage",
                "unavailable",
            )
            assert np.array_equal(
                data["map_success_bound_kind_per_disorder"],
                expected_kinds,
            )
            assert not data[
                "map_success_bound_has_confidence_coverage_per_disorder"
            ].any()
            assert "pass_fraction" not in data.files
            assert "map_success_lower_bound_per_disorder" not in data.files
            assert "map_success_upper_bound_per_disorder" not in data.files
        # 无 .tmp 残留（原子写）
        assert not list((out / "chunks").glob("*.tmp"))

    def test_ti_grid_gate_failure_is_invalid_and_has_no_success_bounds(
        self, tmp_path
    ):
        config = {
            **FAST_TI,
            # Any finite TV/q_top discrepancy exceeds these thresholds.
            "grid_tv_warning": -1.0,
            "grid_q_top_warning": -1.0,
        }
        npz_path, report = scan(
            tmp_path / "ti_invalid", "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=config,
        )
        assert report["computed"] == 1 and report["failed"] == []
        with np.load(npz_path) as data:
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert "ti_grid_tv_exceeded" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )
            assert np.isnan(data["mean_q_top_estimate"][0, 0])
            for name in (
                "map_success_algebraic_lower_bound_per_disorder",
                "map_success_algebraic_upper_bound_per_disorder",
                "map_success_estimated_lower_bound_per_disorder",
                "map_success_estimated_upper_bound_per_disorder",
            ):
                assert np.isnan(data[name][0, 0, 0])
            assert str(data[
                "map_success_bound_kind_per_disorder"
            ][0, 0, 0]) == "unavailable"

    @pytest.mark.parametrize(
        ("p_value", "endpoint_mode", "expected_q_top", "expected_weights"),
        [
            (0.0, "p_zero_delta", 1.0, [1.0, 0.0]),
            (0.5, "p_half_uniform", 0.0, [0.5, 0.5]),
        ],
    )
    def test_auto_full_ti_endpoints_are_analytic_end_to_end(
        self, tmp_path, p_value, endpoint_mode, expected_q_top,
        expected_weights,
    ):
        npz_path, report = scan(
            tmp_path / endpoint_mode, "surface", [2], p_value, [0.08], 1,
            engine="auto", engine_config={"ti": FAST_TI},
        )
        assert report["computed"] == 1 and report["failed"] == []
        with np.load(npz_path) as data:
            assert str(data["resolved_engine_per_disorder"][0, 0, 0]) \
                == ENGINE_FULL_TI
            assert str(data["ti_endpoint_mode_per_disorder"][0, 0, 0]) \
                == endpoint_mode
            assert data["q_top_estimate_per_disorder"][0, 0, 0] \
                == expected_q_top
            assert np.allclose(
                data["weights_absolute_per_disorder"][0, 0, 0],
                expected_weights,
            )
            assert data["q_top_stderr_per_disorder"][0, 0, 0] == 0.0
            assert data["ti_grid_tv_per_disorder"][0, 0, 0] == 0.0
            assert data[
                "weights_are_exact_sector_posterior_per_disorder"
            ][0, 0, 0]
            assert str(data[
                "map_success_bound_kind_per_disorder"
            ][0, 0, 0]) == "analytic_endpoint_algebraic"
            assert np.isfinite(data[
                "map_success_algebraic_lower_bound_per_disorder"
            ][0, 0, 0])
            assert np.isnan(data[
                "map_success_estimated_lower_bound_per_disorder"
            ][0, 0, 0])
            infinite_mask = data[
                "delta_f_infinite_mask_per_disorder"
            ][0, 0, 0]
            if p_value == 0.0:
                assert infinite_mask.tolist() == [False, True]
            else:
                assert not infinite_mask.any()

    def test_deterministic_across_fresh_runs(self, tmp_path):
        r1, _ = scan(tmp_path / "a", "surface", [2], 0.12, [0.08], 2,
                     engine="ti", engine_config=FAST_TI)
        r2, _ = scan(tmp_path / "b", "surface", [2], 0.12, [0.08], 2,
                     engine="ti", engine_config=FAST_TI)
        with np.load(r1, allow_pickle=True) as d1, \
                np.load(r2, allow_pickle=True) as d2:
            assert np.array_equal(d1["q_top_per_disorder"],
                                  d2["q_top_per_disorder"])
            assert np.array_equal(d1["disorder_seed_per_disorder"],
                                  d2["disorder_seed_per_disorder"])

    def test_corrupted_chunk_recomputed(self, tmp_path):
        out = tmp_path / "scan_corrupt"
        scan(out, "surface", [2], 0.12, [0.08], 1, engine="ti",
             engine_config=FAST_TI)
        chunk = next((out / "chunks").glob("task_*.json"))
        chunk.write_text("{broken", encoding="utf-8")
        _, report = scan(out, "surface", [2], 0.12, [0.08], 1, engine="ti",
                         engine_config=FAST_TI)
        assert report["computed"] == 1  # 损坏被重算

    def test_mismatched_inner_task_fingerprint_recomputes_chunk(self, tmp_path):
        out = tmp_path / "inner_fingerprint"
        scan(
            out, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        chunk = next((out / "chunks").glob("task_*.json"))
        payload = json.loads(chunk.read_text(encoding="utf-8"))
        outer_fingerprint = payload["task_fingerprint"]
        payload["result"]["task_fingerprint"] = "wrong-inner-fingerprint"
        payload["result"]["q_top_estimate"] = 999.0
        chunk.write_text(json.dumps(payload), encoding="utf-8")

        npz_path, report = scan(
            out, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        assert report["computed"] == 1 and report["reused"] == 0
        repaired = json.loads(chunk.read_text(encoding="utf-8"))
        assert repaired["result"]["task_fingerprint"] == outer_fingerprint
        with np.load(npz_path) as data:
            assert data["q_top_estimate_per_disorder"][0, 0, 0] != 999.0

    def test_direct_engine_scan(self, tmp_path):
        npz_path, report = scan(
            tmp_path / "scan_direct", "surface", [2], 0.12, [0.08], 1,
            engine="direct", engine_config={
                **FAST_DIRECT, "record_observable_trajectory": False,
            },
        )
        assert report["computed"] == 1
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["requested_engine"] == "direct"
            assert manifest["resolved_engine_configs"][
                "direct_observable_sampling"
            ][0]["record_observable_trajectory"] is True
            q_top = data["q_top_estimate_per_disorder"]
            assert np.isfinite(q_top).all()
            assert data["chain_character_means_relative_per_disorder"].shape[3] == 4
            assert str(data["q_top_estimator_name_per_disorder"][0, 0, 0]) \
                == "independent_chain_u_statistic"

    def test_legacy_sampled_scan_is_formal_only_end_to_end(self, tmp_path):
        config = {
            **FAST_DIRECT,
            "gate_thresholds": {
                "max_r_hat": 99.0,
                "min_ess": 1.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 0.0,
            },
        }
        npz_path, report = scan(
            tmp_path / "legacy_direct", "surface", [2], 0.12, [0.08], 1,
            engine="direct", engine_config=config,
            ensemble="legacy_delta_only",
        )
        assert report["computed"] == 1 and report["failed"] == []
        with np.load(npz_path) as data:
            assert data["formal_only"][0, 0, 0]
            assert data["numerically_valid"][0, 0, 0]
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert np.isnan(data["q_top_estimate_per_disorder"]).all()
            assert np.isnan(data["q_top_crossing_input_per_disorder"]).all()
            assert np.isnan(data["character_means_absolute_per_disorder"]).all()
            assert np.isnan(data["weights_absolute_per_disorder"]).all()
            assert np.isfinite(data["formal_q_top_per_disorder"]).all()
            assert np.isfinite(
                data["formal_sector_characters_absolute_per_disorder"]
            ).all()
            assert np.isfinite(
                data["formal_sector_weights_absolute_per_disorder"]
            ).all()
            assert np.isnan(data["posterior_purity_per_disorder"]).all()
            assert np.isnan(data["map_success_probability_per_disorder"]).all()
            assert data["numerically_valid_disorder_count"][0, 0] == 1
            assert data["formal_only_disorder_count"][0, 0] == 1
            assert data["invalid_disorder_count"][0, 0] == 0
            assert data["paper_aggregation_fraction"][0, 0] == 0.0
            assert data["numerical_pass_fraction"][0, 0] == 1.0
            assert "pass_fraction" not in data.files
            assert str(data["aggregation_status_per_point"][0, 0]) \
                == "FORMAL_ONLY"
            assert not data["reportable_for_crossing_fss"][0, 0]
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["numerically_valid_disorder_count"] == [[1]]
            assert "pass_fraction" not in manifest["fraction_semantics"]
        chunk = next((tmp_path / "legacy_direct" / "chunks").glob("*.json"))
        task = json.loads(chunk.read_text(encoding="utf-8"))["result"]
        assert task["task_status"] == "FORMAL_ONLY"
        assert task["q_top_estimate"] is None
        assert task["weights_absolute"] is None
        assert task["character_means_absolute"] is None
        assert task["formal_q_top"] is not None
        assert task["formal_sector_weights_absolute"] is not None

    def test_fewer_than_four_direct_chains_is_invalid_end_to_end(
        self, tmp_path
    ):
        config = {
            "num_burn_in_sweeps": 5,
            "num_measurements": 32,
            "num_starts": 3,
            "gate_thresholds": {
                "max_r_hat": 1e9,
                "min_ess": 0.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 0.0,
            },
        }
        npz_path, _ = scan(
            tmp_path / "three_chains", "surface", [2], 0.12, [0.08], 1,
            engine="direct", engine_config=config,
        )
        with np.load(npz_path) as data:
            assert data["independent_chain_count_per_disorder"][0, 0, 0] == 3
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert "independent_chain_count<4" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )
            assert np.isnan(data["mean_q_top_estimate"][0, 0])

    def test_out_of_range_debiased_purity_invalidates_task_end_to_end(
        self, tmp_path, monkeypatch
    ):
        import src.run_scan as run_scan_module

        def fake_starts(
            model, frame, observable_set, wiring, config, base_seed,
            num_starts=8, sector_bitmasks=None, engine="fast",
        ):
            starts = []
            for value in (1.0, -1.0, 1.0, -1.0):
                trajectory = np.full(
                    (16, observable_set.num_u), value, dtype=np.int8
                )
                starts.append({
                    "m_u_relative": np.full(observable_set.num_u, value),
                    "m_u_absolute": np.full(observable_set.num_u, value),
                    "observable_trajectory": trajectory,
                    "aggregates": {
                        "q_top_all": 1.0, "q_top_basis": 1.0,
                    },
                    "sector_bitmask": 0,
                    "acceptance": {
                        "logical_per_u": np.ones(model.k),
                    },
                })
            return starts

        monkeypatch.setattr(run_scan_module, "run_multi_start", fake_starts)
        config = {
            "num_burn_in_sweeps": 1,
            "num_measurements": 16,
            "num_starts": 4,
            "gate_thresholds": {
                "max_r_hat": 1e9,
                "min_ess": 0.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 0.0,
            },
        }
        npz_path, _ = scan(
            tmp_path / "purity_invalid", "surface", [2], 0.12, [0.08], 1,
            engine="direct", engine_config=config,
        )
        with np.load(npz_path) as data:
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert "debiased_posterior_purity_out_of_range" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )
            for name in (
                "map_success_algebraic_lower_bound_per_disorder",
                "map_success_algebraic_upper_bound_per_disorder",
                "map_success_estimated_lower_bound_per_disorder",
                "map_success_estimated_upper_bound_per_disorder",
            ):
                assert np.isnan(data[name][0, 0, 0])
            assert str(data[
                "map_success_bound_kind_per_disorder"
            ][0, 0, 0]) == "unavailable"

    def test_direct_gate_failure_marks_worker_result_invalid(
        self, tmp_path
    ):
        config = {
            "num_burn_in_sweeps": 2,
            "num_measurements": 16,
            "num_starts": 4,
            "logical_move_repeat": 0,
            "gate_thresholds": {
                "max_r_hat": 1e9,
                "min_ess": 0.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 1e-4,
            },
        }
        npz_path, _ = scan(
            tmp_path / "direct_gate_invalid", "surface", [2], 0.12,
            [0.08], 1, engine="direct", engine_config=config,
        )
        with np.load(npz_path) as data:
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert "sector_transport_insufficient" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )
            assert np.isnan(data["mean_q_top_estimate"][0, 0])

    def test_alias_is_canonical_before_seed_and_manifest(self, tmp_path):
        alias_config = {
            **FAST_TI, "grid_tv_warning": 1.0,
            "grid_q_top_warning": 1.0,
        }
        r_true, _ = scan(tmp_path / "e_true", "surface", [2], 0.12, [0.08], 1,
                         engine="ti", engine_config=alias_config,
                         ensemble="true_posterior")
        r_paper, _ = scan(
            tmp_path / "e_paper", "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=alias_config,
            ensemble="paper_true_posterior",
        )
        r_repo, _ = scan(tmp_path / "e_repo", "surface", [2], 0.12, [0.08], 1,
                         engine="ti", engine_config=alias_config,
                         ensemble="repo_compat")
        r_legacy, _ = scan(
            tmp_path / "e_legacy", "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=alias_config,
            ensemble="legacy_delta_only",
        )
        with np.load(r_true, allow_pickle=True) as dt, \
                np.load(r_paper, allow_pickle=True) as dp, \
                np.load(r_repo, allow_pickle=True) as dr, \
                np.load(r_legacy, allow_pickle=True) as dl:
            assert json.loads(str(dt["manifest_json"]))["canonical_ensemble"] \
                == "true_posterior"
            assert json.loads(str(dr["manifest_json"]))["canonical_ensemble"] \
                == "legacy_delta_only"
            assert json.loads(str(dp["manifest_json"]))["canonical_ensemble"] \
                == "true_posterior"
            assert np.array_equal(
                dp["disorder_seed_per_disorder"],
                dt["disorder_seed_per_disorder"],
            )
            assert np.array_equal(
                dp["q_top_estimate_per_disorder"],
                dt["q_top_estimate_per_disorder"],
            )
            assert np.array_equal(
                dr["disorder_seed_per_disorder"],
                dl["disorder_seed_per_disorder"],
            )
            assert np.array_equal(
                dr["q_top_estimate_per_disorder"],
                dl["q_top_estimate_per_disorder"], equal_nan=True,
            )
            assert np.array_equal(
                dr["formal_q_top_per_disorder"],
                dl["formal_q_top_per_disorder"], equal_nan=True,
            )
            assert np.array_equal(
                dr["formal_sector_weights_absolute_per_disorder"],
                dl["formal_sector_weights_absolute_per_disorder"],
                equal_nan=True,
            )
            assert np.isnan(dr["q_top_estimate_per_disorder"]).all()
            assert np.isnan(dr["q_top_crossing_input_per_disorder"]).all()
            assert np.isnan(dr["weights_absolute_per_disorder"]).all()
            assert np.isnan(dr["character_means_absolute_per_disorder"]).all()
            assert np.isfinite(dr["formal_q_top_per_disorder"]).all()
            assert np.isfinite(
                dr["formal_sector_weights_absolute_per_disorder"]
            ).all()
            assert dr["formal_only"].all()
            assert not dr["valid_for_aggregation"].any()
            # 系综影响 seed scope（防误合并）
            assert not np.array_equal(dt["disorder_seed_per_disorder"],
                                      dr["disorder_seed_per_disorder"])

    def test_toric_multi_size_padding(self, tmp_path):
        """不同 m（k=2 恒定）多尺寸合并；槽位 pad 语义。"""
        npz_path, _ = scan(
            tmp_path / "scan_ms", "toric", [2, 3], 0.12, [0.10], 1,
            engine="ti", engine_config=FAST_TI,
        )
        with np.load(npz_path, allow_pickle=True) as data:
            assert data["q_top_estimate_per_disorder"].shape == (2, 1, 1)
            assert np.isfinite(data["q_top_estimate_per_disorder"]).all()
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["per_size_k"] == {"2": 2, "3": 2}


class TestParallelism:
    def test_parallel_matches_serial_bit_identical(self, tmp_path):
        """确定性：num_workers=4 与 =1 必须逐位一致（seed scope 与 worker 数无关）。"""
        r1, rep1 = scan(tmp_path / "serial", "surface", [2], 0.12,
                        [0.08, 0.15], 2, engine="ti", engine_config=FAST_TI,
                        num_workers=1)
        r4, rep4 = scan(tmp_path / "par", "surface", [2], 0.12,
                        [0.08, 0.15], 2, engine="ti", engine_config=FAST_TI,
                        num_workers=4)
        assert rep4["num_workers"] == 4 and rep4["computed"] == 4
        assert rep4["failed"] == []
        with np.load(r1, allow_pickle=True) as d1, \
                np.load(r4, allow_pickle=True) as d4:
            assert np.array_equal(d1["q_top_estimate_per_disorder"],
                                  d4["q_top_estimate_per_disorder"])
            assert np.array_equal(d1["disorder_seed_per_disorder"],
                                  d4["disorder_seed_per_disorder"])
            assert np.array_equal(d1["character_means_relative_per_disorder"],
                                  d4["character_means_relative_per_disorder"])

    def test_merge_handles_missing_chunk(self, tmp_path):
        out = tmp_path / "miss"
        scan(out, "surface", [2], 0.12, [0.08, 0.15], 2, engine="ti",
             engine_config=FAST_TI, num_workers=1)
        chunk = sorted((out / "chunks").glob("task_*.json"))[0]
        chunk.unlink()   # 模拟失败/缺失 cell
        npz_path = merge(out, "surface", [2], 0.12, [0.08, 0.15], 2,
                         "x_error", "true_posterior", "ti", FAST_TI,
                         "full_rank")
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["missing_chunks"] == 1
            flags = data["flags_per_disorder"].astype(str)
            assert (flags == "MISSING").sum() == 1
            assert np.isfinite(data["q_top_estimate_per_disorder"]).sum() == 3
            incomplete = data["missing_disorder_count"] > 0
            assert incomplete.sum() == 1
            assert np.all(data["planned_disorder_count"] == 2)
            assert data["present_disorder_count"][incomplete].item() == 1
            assert str(data["aggregation_status_per_point"][
                incomplete
            ].item()) == "INCOMPLETE"
            assert not data["reportable_for_crossing_fss"][
                incomplete
            ].item()
            assert np.isnan(data["mean_q_top_estimate"][incomplete]).all()
            assert np.isnan(data[
                "q_top_crossing_input_per_disorder"
            ][incomplete]).all()
            valid_count = data["valid_disorder_count"][incomplete].item()
            conditional = data[
                "conditional_mean_q_top_estimate_valid_only"
            ][incomplete].item()
            assert np.isfinite(conditional) == (valid_count > 0)
            assert data["paper_aggregation_fraction"][incomplete].item() \
                == valid_count / 2
            numerical_count = data[
                "numerically_valid_disorder_count"
            ][incomplete].item()
            assert data["numerical_pass_fraction"][incomplete].item() \
                == numerical_count / 2

    def test_merge_all_missing_still_emits_incomplete_audit_result(
        self, tmp_path
    ):
        out = tmp_path / "all_missing"
        scan(
            out, "surface", [2], 0.12, [0.08], 2,
            engine="ti", engine_config=FAST_TI,
        )
        for chunk in (out / "chunks").glob("task_*.json"):
            chunk.unlink()
        npz_path = merge(
            out, "surface", [2], 0.12, [0.08], 2, "x_error",
            "true_posterior", "ti", FAST_TI, "full_rank",
        )
        with np.load(npz_path) as data:
            assert str(data["aggregation_status_per_point"][0, 0]) \
                == "INCOMPLETE"
            assert data["planned_disorder_count"][0, 0] == 2
            assert data["present_disorder_count"][0, 0] == 0
            assert data["missing_disorder_count"][0, 0] == 2
            assert data["paper_aggregation_fraction"][0, 0] == 0.0
            assert data["numerical_pass_fraction"][0, 0] == 0.0
            assert np.isnan(data["mean_q_top_estimate"][0, 0])
            assert np.isnan(
                data["q_top_crossing_input_per_disorder"][0, 0]
            ).all()


class TestV3RoutingAndIdentity:
    def test_auto_routes_all_three_production_paths(self):
        assert state_prep_protocol_for_sector("x_error") \
            == "plus_Zcheck_X"
        assert state_prep_protocol_for_sector("z_error") \
            == "zero_Xcheck_Z"
        assert resolve_engine("auto", 10, 0.1) == ENGINE_FULL_TI
        assert resolve_engine("auto", 11, 0.1) == ENGINE_PT
        assert resolve_engine("auto", 11, 0.0) == ENGINE_Q0
        with pytest.raises(ValueError, match="k>10"):
            resolve_engine("ti", 11, 0.1)
        with pytest.raises(ValueError, match="q>0"):
            resolve_engine("pt", 11, 0.0)

    def test_large_k_ti_rejected_before_chunk_creation(self, tmp_path):
        output = tmp_path / "large_ti"
        with pytest.raises(ValueError, match="k>10"):
            scan(
                output, "k43", [1], 0.1, [0.05], 1,
                engine="ti", engine_config=FAST_TI,
            )
        assert not (output / "chunks").exists()

    def test_config_change_invalidates_chunk_identity(self, tmp_path):
        output = tmp_path / "identity"
        first, report = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        assert report["computed"] == 1
        with np.load(first) as data:
            fingerprint_first = str(data["task_fingerprint_per_disorder"][0, 0, 0])
        changed = dict(FAST_TI, num_measurements=FAST_TI["num_measurements"] + 1)
        second, report = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=changed,
        )
        assert report["computed"] == 1 and report["reused"] == 0
        with np.load(second) as data:
            fingerprint_second = str(data["task_fingerprint_per_disorder"][0, 0, 0])
        assert fingerprint_first != fingerprint_second

    def test_source_fingerprint_participates_in_chunk_identity(
        self, tmp_path, monkeypatch
    ):
        import src.run_scan as run_scan_module

        monkeypatch.setattr(
            run_scan_module, "implementation_fingerprint", lambda: "a" * 64
        )
        first = _build_specs(
            tmp_path, "surface", [2], 0.12, [0.08], 1, "x_error",
            "true_posterior", "ti", FAST_TI, "full_rank", None, 64,
        )[0]
        monkeypatch.setattr(
            run_scan_module, "implementation_fingerprint", lambda: "b" * 64
        )
        second = _build_specs(
            tmp_path, "surface", [2], 0.12, [0.08], 1, "x_error",
            "true_posterior", "ti", FAST_TI, "full_rank", None, 64,
        )[0]
        assert first["task_fingerprint"] != second["task_fingerprint"]
        assert first["chunk_path"] != second["chunk_path"]

    def test_nearby_q_values_have_unique_exact_chunk_paths(self, tmp_path):
        specs = _build_specs(
            tmp_path, "surface", [2], 0.12,
            [0.10000000000001, 0.10000000000002], 2, "x_error",
            "true_posterior", "ti", FAST_TI, "full_rank", None, 64,
        )
        paths = [spec["chunk_path"] for spec in specs]
        assert len(paths) == len(set(paths))

    def test_forced_trajectory_is_normalized_before_fingerprinting(self):
        from src.run_scan import ENGINE_DIRECT

        for resolved in (ENGINE_PT, ENGINE_Q0, ENGINE_DIRECT):
            config = _resolved_engine_config(
                "auto", resolved, {"record_observable_trajectory": False}
            )
            assert config["record_observable_trajectory"] is True

    def test_q0_auto_defaults_to_eight_starts(self):
        config = _resolved_engine_config("auto", ENGINE_Q0, {})
        assert config["num_starts"] == 8
        assert config["record_observable_trajectory"] is True

    def test_q0_auto_executes_eight_independent_starts(self, tmp_path):
        config = {"q0": {
            "num_burn_in_sweeps": 2,
            "num_measurements": 16,
            "logical_move_repeat": 0,
            "gate_thresholds": {
                "max_r_hat": 1e9,
                "min_ess": 0.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 1e-4,
            },
        }}
        npz_path, report = scan(
            tmp_path / "q0_auto", "k43", [1], 0.1, [0.0], 1,
            engine="auto", engine_config=config, u_rand_count=4,
        )
        assert report["computed"] == 1 and report["failed"] == []
        with np.load(npz_path) as data:
            assert str(data["resolved_engine_per_disorder"][0, 0, 0]) \
                == ENGINE_Q0
            assert data[
                "independent_chain_count_per_disorder"
            ][0, 0, 0] == 8
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert "sector_transport_insufficient" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )

    def test_actual_k16_observable_set_keeps_16_plus_64_characters(self):
        H_Z, H_X, logicals, _ = build_code("expander34", 4)
        model = assemble_sector_model(
            H_X, H_Z, logicals, sector="x_error"
        )
        frame = build_observable_frame(model)
        observable_set = build_observable_set(
            frame, u_rand_seed=123, num_random_u=64
        )
        assert model.k == 16
        assert observable_set.tier == "sampled"
        assert observable_set.num_u == 80
        assert np.unique(observable_set.u_bitmasks).size == 80

    def test_cli_accepts_both_deprecated_ensemble_aliases(self, tmp_path):
        common = [
            "--output-dir", str(tmp_path), "--size-list", "2",
            "--p-value", "0.1", "--q-values", "0.05",
            "--num-disorders", "1",
        ]
        parser = build_arg_parser()
        assert parser.parse_args(
            common + ["--ensemble", "paper_true_posterior"]
        ).ensemble == "paper_true_posterior"
        assert parser.parse_args(
            common + ["--ensemble", "repo_compat"]
        ).ensemble == "repo_compat"

    def test_force_recompute_overwrites_matching_current_chunk(self, tmp_path):
        output = tmp_path / "force"
        _, first = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        _, second = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI, force_recompute=True,
        )
        assert first["computed"] == 1
        assert second["computed"] == 1
        assert second["reused"] == 0

    def test_v1_chunk_is_never_reused(self, tmp_path):
        output = tmp_path / "v1_isolation"
        specs = _build_specs(
            output, "surface", [2], 0.12, [0.08], 1, "x_error",
            "true_posterior", "ti", FAST_TI, "full_rank", None, 64,
        )
        chunk = Path(specs[0]["chunk_path"])
        chunk.parent.mkdir(parents=True)
        chunk.write_text(json.dumps({
            "protocol": "exp101.scan.v1",
            "task_fingerprint": specs[0]["task_fingerprint"],
            "result": {"q_top_estimate": 999.0},
        }), encoding="utf-8")
        _, report = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        assert report["computed"] == 1 and report["reused"] == 0
        assert json.loads(chunk.read_text(encoding="utf-8"))["protocol"] \
            == PROTOCOL_VERSION

    def test_v2_chunk_with_matching_fingerprints_is_never_reused(
        self, tmp_path
    ):
        output = tmp_path / "v2_isolation"
        scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        chunk = next((output / "chunks").glob("task_*.json"))
        payload = json.loads(chunk.read_text(encoding="utf-8"))
        payload["protocol"] = "exp101.scan.v2"
        payload["result"]["scan_contract_version"] = "exp101.scan.v2"
        payload["result"]["q_top_estimate"] = 999.0
        chunk.write_text(json.dumps(payload), encoding="utf-8")

        npz_path, report = scan(
            output, "surface", [2], 0.12, [0.08], 1,
            engine="ti", engine_config=FAST_TI,
        )
        assert report["computed"] == 1 and report["reused"] == 0
        repaired = json.loads(chunk.read_text(encoding="utf-8"))
        assert repaired["protocol"] == PROTOCOL_VERSION
        assert repaired["result"]["scan_contract_version"] \
            == PROTOCOL_VERSION
        with np.load(npz_path) as data:
            assert data["q_top_estimate_per_disorder"][0, 0, 0] != 999.0

    def test_model_cache_key_isolates_all_family_configuration(self):
        baseline = _model_cache_key(
            "expander34", 2, "x_error", "full_rank", 11
        )
        assert baseline != _model_cache_key(
            "expander34", 2, "z_error", "full_rank", 11
        )
        assert baseline != _model_cache_key(
            "expander34", 2, "x_error", "full_rank_d3", 11
        )
        assert baseline != _model_cache_key(
            "expander34", 2, "x_error", "full_rank", 12
        )

    def test_pt_gate_failure_marks_chunk_invalid_and_excludes_mean(
        self, tmp_path, monkeypatch
    ):
        from src.pt import PtResult
        from src.reference_mcmc import MoveCounters
        import src.run_scan as run_scan_module

        call_index = 0

        def fake_pt(model, frame, observable_set, wiring, pt_config, seed,
                    sector_bitmask_per_replica=None):
            nonlocal call_index
            rng = np.random.default_rng(100 + call_index)
            trajectory = rng.choice(
                [-1, 1], size=(128, observable_set.num_u)
            ).astype(np.int8)
            relative = trajectory.mean(axis=0)
            counters = MoveCounters(
                logical_attempts_per_u=np.full(model.k, 100),
                logical_accepts_per_u=np.full(model.k, 10),
            )
            result = PtResult(
                m_u_cold=relative,
                observable_sums_cold=trajectory.sum(axis=0),
                num_measurements=trajectory.shape[0],
                ladder_p=np.asarray([0.1, 0.2]),
                ladder_q=np.asarray([0.05, 0.2]),
                swap_attempts=np.asarray([100]),
                swap_accepts=np.asarray([50]),
                counters_per_rung=[counters, counters],
                round_trips=0 if call_index == 0 else 2,
                replica_id_per_rung=np.asarray([0, 1]),
                observable_trajectory_cold=trajectory,
                m_u_cold_relative=relative,
                m_u_cold_absolute=relative,
                observable_trajectory_cold_relative=trajectory,
                observable_trajectory_cold_absolute=trajectory,
            )
            call_index += 1
            return result

        monkeypatch.setattr(
            run_scan_module, "run_parallel_tempering", fake_pt
        )
        config = {"pt": {
            "num_temperatures": 2,
            "q_hot": 0.3,
            "num_burn_in_rounds": 1,
            "num_measurement_rounds": 128,
            "num_instances": 4,
            "gate_thresholds": {
                "max_r_hat": 99.0,
                "min_ess": 1.0,
                "max_q_top_spread": 2.0,
                "max_m_u_spread": 2.0,
                "min_cold_logical_acceptance": 1e-4,
                "min_round_trips": 1,
            },
        }}
        npz_path, report = scan(
            tmp_path / "pt_invalid", "k43", [1], 0.1, [0.05], 1,
            engine="auto", engine_config=config,
        )
        assert report["computed"] == 1 and report["failed"] == []
        with np.load(npz_path) as data:
            assert not data["valid_for_aggregation"][0, 0, 0]
            assert data["invalid_disorder_count"][0, 0] == 1
            assert np.isnan(data["mean_q_top_estimate"][0, 0])
            assert "pt_instance_round_trips_insufficient" in str(
                data["failure_reasons_per_disorder"][0, 0, 0]
            )
            for name in (
                "map_success_algebraic_lower_bound_per_disorder",
                "map_success_algebraic_upper_bound_per_disorder",
                "map_success_estimated_lower_bound_per_disorder",
                "map_success_estimated_upper_bound_per_disorder",
            ):
                assert np.isnan(data[name][0, 0, 0])
            assert str(data[
                "map_success_bound_kind_per_disorder"
            ][0, 0, 0]) == "unavailable"
            assert np.isnan(
                data["q_top_crossing_input_per_disorder"][0, 0, 0]
            )
        chunk = next((tmp_path / "pt_invalid" / "chunks").glob("*.json"))
        payload = json.loads(chunk.read_text(encoding="utf-8"))
        assert payload["result"]["task_status"] == "INVALID"
