"""G2.6 单元测试：诊断量合成序列校准 + gate 正例/负例（含「共冻≠收敛」）。"""

import numpy as np
import pytest

from src.gates import (
    GateThresholds,
    default_sector_bitmasks,
    effective_sample_size,
    evaluate_convergence_gate,
    integrated_autocorrelation_time,
    run_multi_start,
    split_r_hat,
)
from src.graphs import cycle_parity_check_matrix, repetition_parity_check_matrix
from src.hgp import hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model, draw_disorder, wire_ensemble
from src.observables import build_observable_frame, build_observable_set
from src.pt import PtConfig, run_parallel_tempering
from src.reference_mcmc import ReferenceMcmcConfig


def build_setup(classical):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    obs_set = build_observable_set(frame)
    return model, frame, obs_set


def make_wiring(model, frame, p, q, seed):
    disorder = draw_disorder(model, p, q, np.random.default_rng(seed))
    return wire_ensemble(model, disorder, "true_posterior", frame)


class TestSyntheticDiagnostics:
    def test_iid_series(self):
        rng = np.random.default_rng(1)
        series = rng.choice([-1.0, 1.0], size=20000)
        tau = integrated_autocorrelation_time(series)
        assert 0.5 < tau < 1.5
        chains = series.reshape(4, 5000)
        assert abs(split_r_hat(chains) - 1.0) < 0.02
        ess = effective_sample_size(chains)
        assert ess > 10000

    def test_ar1_tau(self):
        rng = np.random.default_rng(2)
        rho, n = 0.9, 200000
        noise = rng.normal(size=n)
        series = np.empty(n)
        series[0] = noise[0]
        for i in range(1, n):
            series[i] = rho * series[i - 1] + noise[i]
        tau = integrated_autocorrelation_time(series)
        expected = (1 + rho) / (1 - rho)  # = 19
        assert expected / 2 < tau < expected * 2

    def test_r_hat_detects_mean_shift(self):
        rng = np.random.default_rng(3)
        chain_a = rng.normal(0.0, 1.0, size=5000)
        chain_b = rng.normal(1.5, 1.0, size=5000)
        assert split_r_hat(np.stack([chain_a, chain_b])) > 1.2

    def test_frozen_chain_degeneracies(self):
        frozen = np.ones((2, 1000))
        assert split_r_hat(frozen) == 1.0
        assert effective_sample_size(frozen) == 0.0
        mixed = np.stack([np.ones(1000), -np.ones(1000)])
        assert split_r_hat(mixed) == float("inf")

    def test_default_sector_bitmasks(self):
        assert default_sector_bitmasks(3, 8) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert default_sector_bitmasks(1, 4) == [0, 1, 0, 1]
        assert default_sector_bitmasks(0, 3) == [0, 0, 0]


class TestGatePositive:
    def test_easy_parameters_pass(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=5)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=300, num_measurements=6000,
            record_observable_trajectory=True,
        )
        starts = run_multi_start(model, frame, obs_set, wiring, config,
                                 base_seed=100, num_starts=4)
        report = evaluate_convergence_gate(starts)
        assert report.passed, (report.failed_checks, report.metrics)
        assert report.metrics["local_transport_ok"]
        assert report.metrics["max_r_hat"] < 1.05

    def test_trajectory_required(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=5)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=10, num_measurements=50)
        with pytest.raises(ValueError, match="trajectory"):
            run_multi_start(model, frame, obs_set, wiring, config, base_seed=1,
                            num_starts=2)


class TestGateNegativeFrozen:
    """冻结负例。干净的冻结场景用 **q=0**：硬约束禁止 single-bit ⇒ 关闭
    L-move（repeat=0）后 label **严格守恒**；类内 S-move 混合快、噪声小。
    （q>0 时 label 仍可经 single-bit 通道指数慢泄漏——notes/01 §9 的物理，
    不适合做确定性的负例。）"""

    def _frozen_starts(self, sector_bitmasks):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.0, seed=7)
        assert wiring.q_zero
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=300, num_measurements=4000,
            logical_move_repeat=0,               # 冻结 L 通道（q=0 下严格）
            record_observable_trajectory=True,
        )
        starts = run_multi_start(model, frame, obs_set, wiring, config,
                                 base_seed=200,
                                 sector_bitmasks=sector_bitmasks)
        return starts

    def test_different_sector_starts_detected_by_m_u_spread(self):
        """符号敏感判据的必要性：q_top=mean m_u² 符号盲——不同 sector 共冻
        的 q_top 一致；m_u spread 才能看到 ± 翻转（≈2|m|）。"""
        starts = self._frozen_starts([0, 1, 2, 3])
        report = evaluate_convergence_gate(starts)
        assert not report.passed
        assert "sector_transport_insufficient" in report.failed_checks
        assert report.metrics["m_u_spread"] > 1.0     # 符号翻转（2|m|，|m| 高）
        assert any("m_u_spread" in c for c in report.failed_checks)
        # q_top spread 保持小（符号盲的直接证据）
        assert report.metrics["q_top_spread"] < 0.2

    def test_same_sector_cofrozen_still_fails(self):
        """「共冻 ≠ 收敛」核心：同 sector 起点下放宽全部统计判据，
        gate 仍必须仅因 sector transport 失败。"""
        starts = self._frozen_starts([0, 0, 0, 0])
        # q=0 + 无 L-move ⇒ O_u 逐链严格常数 ⇒ ESS=0：连 ESS 检查也放空
        loose = GateThresholds(
            max_r_hat=1e6, min_ess=0.0,
            max_q_top_spread=0.5, max_m_u_spread=0.5,
        )
        report = evaluate_convergence_gate(starts, thresholds=loose)
        assert report.metrics["m_u_spread"] < 0.3     # 无符号翻转（仅小噪声）
        assert not report.passed
        assert report.failed_checks == ["sector_transport_insufficient"]
        assert report.metrics["worst_u_cold_logical_acceptance"] == 0.0
        assert any("共冻" in note for note in report.notes)

    def test_pt_round_trips_rescue_transport(self):
        """同参数 + PT 有往返 ⇒ transport 判据被 PT 证据满足。"""
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.02, 0.01, seed=7)
        pt_result = run_parallel_tempering(
            model, frame, obs_set, wiring,
            PtConfig(num_temperatures=6, q_hot=0.45,
                     num_burn_in_rounds=200, num_measurement_rounds=4000),
            seed=9,
        )
        assert pt_result.round_trips >= 1
        starts = self._frozen_starts([0, 0, 0, 0])
        report = evaluate_convergence_gate(starts, pt_result=pt_result)
        assert report.metrics["pt_transport_ok"]
        assert "sector_transport_insufficient" not in report.failed_checks

    def test_worst_u_semantics(self):
        """acceptance 有一个 u 为 0 ⇒ worst-u 取到它（nan 也按 0）。"""
        fake = {
            "observable_trajectory": np.random.default_rng(0).choice(
                [-1, 1], size=(1000, 3)),
            "aggregates": {"q_top_all": 0.5, "q_top_basis": 0.5},
            "acceptance": {"logical_per_u": np.array([0.5, 0.0, np.nan])},
        }
        report = evaluate_convergence_gate(
            [fake, fake], thresholds=GateThresholds(min_ess=1.0,
                                                    max_q_top_spread=1.0)
        )
        assert report.metrics["worst_u_cold_logical_acceptance"] == 0.0
        assert "sector_transport_insufficient" in report.failed_checks
