"""Focused v2 scan-estimator, validity, and flexible-schema tests."""

import json

import numpy as np

from src.gates import GateThresholds, evaluate_pt_convergence_gate
from src.observables import (
    ObservableSet,
    independent_chain_squared_character_estimates,
    sampled_nonzero_character_mean,
)
from src.pt import PtResult
from src.reference_mcmc import MoveCounters
from src.run_scan import (
    PROTOCOL_VERSION,
    _build_specs,
    _sampled_estimator_result,
    merge,
)


def observable_set(k, masks, basis_positions, tier="sampled"):
    return ObservableSet(
        tier=tier,
        k=k,
        u_bitmasks=np.asarray(masks, dtype=np.int64),
        W_rows=np.zeros((len(masks), 1), dtype=np.uint8),
        basis_positions=np.asarray(basis_positions, dtype=np.int64),
        num_random_u=max(0, len(masks) - k),
    )


class TestIndependentChainEstimator:
    def test_cross_product_removes_finite_chain_square_bias(self):
        rng = np.random.default_rng(123)
        true_mean = 0.2
        num_repeats, num_chains, samples_per_chain = 12000, 4, 20
        plus_counts = rng.binomial(
            samples_per_chain, (1 + true_mean) / 2,
            size=(num_repeats, num_chains),
        )
        means = 2 * plus_counts / samples_per_chain - 1
        raw = means.mean(axis=1) ** 2
        debiased = np.asarray([
            independent_chain_squared_character_estimates(row[:, None])[
                "m2_u_debiased"
            ][0]
            for row in means
        ])
        assert raw.mean() > true_mean**2 + 0.008
        assert abs(debiased.mean() - true_mean**2) < 0.004

    def test_basis_and_sampled_nonbasis_use_population_weights(self):
        obs = observable_set(4, [1, 2, 4, 8, 3, 5], [0, 1, 2, 3])
        values = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 14.0])
        estimate, sampling_se = sampled_nonzero_character_mean(obs, values)
        # N=15: all four basis entries are exact; two sampled entries represent
        # the remaining N-k=11 nonbasis characters.
        assert np.isclose(estimate, (10.0 + 11.0 * 12.0) / 15.0)
        expected_se = (11 / 15) * np.sqrt((1 - 2 / 11) * 8.0 / 2)
        assert np.isclose(sampling_se, expected_se)

    def test_out_of_range_debiased_purity_is_unclipped_and_invalid(self):
        obs = observable_set(1, [1], [0], tier="full")
        means = np.asarray([[1.0], [-1.0], [1.0], [-1.0]])
        result = _sampled_estimator_result(
            obs, means, means, "true_posterior", planted_class=0,
            minimum_chains=4,
        )
        assert np.isclose(result["q_top_estimate"], -1 / 3)
        assert np.isclose(result["posterior_purity"], 1 / 3)
        assert "debiased_posterior_purity_out_of_range" in \
            result["estimator_failure_reasons"]
        assert result["map_success_probability"] is None
        assert result["map_success_lower_bound"] is None
        assert result["map_success_upper_bound"] is None

    def test_fewer_than_four_independent_chains_is_not_aggregatable(self):
        obs = observable_set(1, [1], [0], tier="full")
        means = np.asarray([[0.2], [0.3], [0.4]])
        result = _sampled_estimator_result(
            obs, means, means, "true_posterior", planted_class=0,
            minimum_chains=4,
        )
        assert result["num_independent_chains"] == 3
        assert "independent_chain_count<4" in \
            result["estimator_failure_reasons"]

    def test_sampled_full_character_inversion_is_not_labelled_map_success(self):
        obs = observable_set(2, [1, 2, 3], [0, 1], tier="full")
        means = np.asarray([
            [0.20, 0.10, 0.05],
            [0.18, 0.12, 0.04],
            [0.22, 0.08, 0.06],
            [0.20, 0.10, 0.05],
        ])
        result = _sampled_estimator_result(
            obs, means, means, "true_posterior", planted_class=0,
            minimum_chains=4,
        )
        assert result["weights_absolute"] is not None
        assert result["weights_estimator_name"] \
            == "character_inversion_of_sample_means"
        assert result["weights_are_exact_sector_posterior"] is False
        assert result["largest_sector_mass"] is not None
        assert result["map_success_probability"] is None

    def test_legacy_sampled_estimator_exposes_formal_fields_only(self):
        obs = observable_set(2, [1, 2, 3], [0, 1], tier="full")
        means = np.asarray([
            [0.20, 0.10, 0.05],
            [0.18, 0.12, 0.04],
            [0.22, 0.08, 0.06],
            [0.20, 0.10, 0.05],
        ])
        result = _sampled_estimator_result(
            obs, means, means, "legacy_delta_only", planted_class=0,
            minimum_chains=4,
        )
        for name in (
            "character_means_absolute", "character_means_relative",
            "chain_character_means_absolute",
            "chain_character_means_relative", "m2_u_debiased",
            "q_top_estimate", "q_top_absolute", "q_top_relative",
            "posterior_purity", "posterior_mass_on_planted_class",
            "map_success_probability", "map_success_lower_bound",
            "map_success_upper_bound", "weights_absolute",
            "weights_relative",
        ):
            assert result[name] is None
        assert result["formal_q_top"] is not None
        assert result["formal_sector_weights_absolute"] is not None
        assert result["formal_sector_characters_absolute"] is not None
        assert result["largest_sector_mass"] is not None


def _write_result(spec, result):
    path = spec["chunk_path"]
    payload = {
        "protocol": PROTOCOL_VERSION,
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "result": result,
    }
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_result(spec, *, q_top, valid, character_count=1, k=1):
    values = np.linspace(0.1, 0.2, character_count).tolist()
    return {
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "git_commit_sha": "synthetic",
        "git_worktree_dirty": False,
        "family": {"family": "surface", "size": 2},
        "k": k,
        "code_fingerprint": "code",
        "section_fingerprint": "section",
        "observable_frame_fingerprint": "frame",
        "resolved_engine": spec["resolved_engine"],
        "resolved_engine_config": spec["engine_config"],
        "character_count": character_count,
        "u_bitmasks": list(range(1, character_count + 1)),
        "character_means_absolute": values,
        "character_means_relative": values,
        "m2_u_pooled_square_raw": values,
        "m2_u_debiased": values,
        "m2_u_debiased_jackknife_se": [0.01] * character_count,
        "q_top_estimate": q_top,
        "q_top_absolute": q_top,
        "q_top_relative": q_top,
        "q_top_estimator_name": "synthetic",
        "valid_for_aggregation": valid,
        "failure_reasons": [] if valid else ["synthetic_invalid"],
        "flags": "PASS" if valid else "INVALID:synthetic_invalid",
    }


class TestMergeValidityAndShape:
    def test_invalid_disorder_does_not_enter_mean(self, tmp_path):
        specs = _build_specs(
            tmp_path, "surface", [2], 0.1, [0.05], 2, "x_error",
            "true_posterior", "direct", {}, "full_rank", None, 64,
        )
        _write_result(specs[0], _synthetic_result(
            specs[0], q_top=0.2, valid=True
        ))
        _write_result(specs[1], _synthetic_result(
            specs[1], q_top=0.9, valid=False
        ))
        path = merge(
            tmp_path, "surface", [2], 0.1, [0.05], 2, "x_error",
            "true_posterior", "direct", {}, "full_rank",
            expected_specs=specs,
        )
        with np.load(path) as data:
            assert np.isclose(data["mean_q_top_estimate"][0, 0], 0.2)
            assert data["valid_disorder_count"][0, 0] == 1
            assert data["invalid_disorder_count"][0, 0] == 1
            assert data["missing_disorder_count"][0, 0] == 0
            assert data["paper_aggregation_fraction"][0, 0] == 0.5
            assert data["numerical_pass_fraction"][0, 0] == 0.5
            assert data["pass_fraction"][0, 0] == 0.5
            assert np.isclose(
                data["q_top_crossing_input_per_disorder"][0, 0, 0], 0.2
            )
            assert np.isnan(
                data["q_top_crossing_input_per_disorder"][0, 0, 1]
            )

    def test_sem_uses_two_valid_samples_and_excludes_invalid(self, tmp_path):
        specs = _build_specs(
            tmp_path, "surface", [2], 0.1, [0.05], 3, "x_error",
            "true_posterior", "direct", {}, "full_rank", None, 64,
        )
        for spec, value, valid in zip(
            specs, (0.2, 0.6, 0.99), (True, True, False)
        ):
            _write_result(
                spec, _synthetic_result(
                    spec, q_top=value, valid=valid
                )
            )
        path = merge(
            tmp_path, "surface", [2], 0.1, [0.05], 3, "x_error",
            "true_posterior", "direct", {}, "full_rank",
            expected_specs=specs,
        )
        with np.load(path) as data:
            assert np.isclose(data["mean_q_top_estimate"][0, 0], 0.4)
            assert np.isclose(data["disorder_sem_q_top_estimate"][0, 0], 0.2)
            assert data["valid_disorder_count"][0, 0] == 2
            assert data["invalid_disorder_count"][0, 0] == 1
            assert np.isclose(
                data["paper_aggregation_fraction"][0, 0], 2 / 3
            )
            assert np.isclose(
                data["numerical_pass_fraction"][0, 0], 2 / 3
            )

    def test_unknown_git_dirty_state_uses_known_marker(self, tmp_path):
        specs = _build_specs(
            tmp_path, "surface", [2], 0.1, [0.05], 1, "x_error",
            "true_posterior", "direct", {}, "full_rank", None, 64,
        )
        result = _synthetic_result(specs[0], q_top=0.2, valid=True)
        result["git_worktree_dirty"] = None
        _write_result(specs[0], result)
        path = merge(
            tmp_path, "surface", [2], 0.1, [0.05], 1, "x_error",
            "true_posterior", "direct", {}, "full_rank",
            expected_specs=specs,
        )
        with np.load(path) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["git_worktree_dirty"] is None
            assert data["git_worktree_dirty"].dtype == np.dtype(bool)
            assert not bool(data["git_worktree_dirty"])
            assert not bool(data["git_worktree_dirty_known"])

    def test_character_axis_uses_actual_count_not_k(self, tmp_path):
        specs = _build_specs(
            tmp_path, "surface", [2], 0.1, [0.05], 1, "x_error",
            "true_posterior", "direct", {}, "full_rank", None, 64,
        )
        # Synthetic k=16 sampled task: 16 basis + 64 random nonbasis rows.
        result = _synthetic_result(
            specs[0], q_top=0.3, valid=True, character_count=80, k=16
        )
        _write_result(specs[0], result)
        path = merge(
            tmp_path, "surface", [2], 0.1, [0.05], 1, "x_error",
            "true_posterior", "direct", {}, "full_rank",
            expected_specs=specs,
        )
        with np.load(path) as data:
            assert data["character_means_absolute_per_disorder"].shape[-1] == 80
            assert data["character_count_per_disorder"][0, 0, 0] == 80
            assert data["character_mask_per_disorder"][0, 0, 0].sum() == 80


def _fake_pt_result(seed, round_trips):
    rng = np.random.default_rng(seed)
    trajectory = rng.choice([-1, 1], size=(1000, 1)).astype(np.int8)
    counters = MoveCounters(
        logical_attempts_per_u=np.asarray([1000]),
        logical_accepts_per_u=np.asarray([100]),
    )
    return PtResult(
        m_u_cold=trajectory.mean(axis=0),
        observable_sums_cold=trajectory.sum(axis=0),
        num_measurements=trajectory.shape[0],
        ladder_p=np.asarray([0.1, 0.2]),
        ladder_q=np.asarray([0.05, 0.2]),
        swap_attempts=np.asarray([100]),
        swap_accepts=np.asarray([50]),
        counters_per_rung=[counters, counters],
        round_trips=round_trips,
        replica_id_per_rung=np.asarray([0, 1]),
        observable_trajectory_cold=trajectory,
    )


class TestPtProductionGate:
    def test_one_instance_without_round_trip_invalidates_pt(self):
        obs = observable_set(1, [1], [0], tier="full")
        results = [
            _fake_pt_result(10 + index, 0 if index == 0 else 2)
            for index in range(4)
        ]
        report = evaluate_pt_convergence_gate(
            results, obs,
            thresholds=GateThresholds(
                max_r_hat=2.0, min_ess=1.0, max_q_top_spread=1.0,
                max_m_u_spread=1.0, min_cold_logical_acceptance=1e-4,
                min_round_trips=1,
            ),
        )
        assert not report.passed
        assert "pt_instance_round_trips_insufficient" in report.failed_checks
        assert report.metrics["pt_round_trips_per_instance"].tolist() \
            == [0, 2, 2, 2]

    def test_q_top_spread_uses_weighted_all_character_population(self):
        obs = observable_set(3, [1, 2, 4, 3], [0, 1, 2])
        results = [_fake_pt_result(20 + index, 2) for index in range(4)]
        for index, result in enumerate(results):
            trajectory = np.ones((1000, 4), dtype=np.int8)
            # Basis characters are identical.  Only the sampled nonbasis
            # character differs, so a basis-only spread would be zero.
            if index >= 2:
                trajectory[:, 3] = np.tile([-1, 1], 500)
            result.observable_trajectory_cold = trajectory
            result.m_u_cold = trajectory.mean(axis=0)
            result.observable_sums_cold = trajectory.sum(axis=0)
            result.counters_per_rung[0] = MoveCounters(
                logical_attempts_per_u=np.full(3, 1000),
                logical_accepts_per_u=np.full(3, 100),
            )
        report = evaluate_pt_convergence_gate(
            results, obs,
            thresholds=GateThresholds(
                max_r_hat=1e9, min_ess=0.0, max_q_top_spread=0.1,
                max_m_u_spread=2.0, min_cold_logical_acceptance=0.0,
                min_round_trips=1,
            ),
        )
        assert report.metrics["q_top_spread"] > 0.1
        assert any(
            check.startswith("q_top_spread>")
            for check in report.failed_checks
        )
