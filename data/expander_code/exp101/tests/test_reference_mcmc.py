"""G2.3 单元测试：参考 MCMC。

层级：
  A. 穷举 ΔE 正确性（所有状态 × 所有 move，对全量能量重算，机器精度）
  B. 长链经验分布 / m_u 对精确枚举的统计一致（固定 seed，含 q=0 coset）
  C. 解析锚点：p=0.5 时 L-move 接受率恒 1、m_u≈0
  D. 不变量、可复现性、多起点一致
"""

import numpy as np
import pytest

from src.gf2 import gf2_matmul
from src.graphs import cycle_parity_check_matrix, repetition_parity_check_matrix
from src.hgp import hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model, draw_disorder, wire_ensemble
from src.observables import build_observable_frame, build_observable_set
from src.reference_mcmc import (
    ReferenceMcmcConfig,
    make_initial_state,
    run_reference_mcmc,
    single_bit_log_acceptance,
    support_move_log_acceptance,
)
from tests.util_enum import enum_m_u


def build_setup(classical, sector="x_error"):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector=sector)
    frame = build_observable_frame(model)
    obs_set = build_observable_set(frame)
    return model, frame, obs_set


def make_wiring(model, frame, p, q, seed, ensemble="true_posterior"):
    disorder = draw_disorder(model, p, q, np.random.default_rng(seed))
    return disorder, wire_ensemble(model, disorder, ensemble, frame)


def total_energy_of(model, wiring, v):
    syndrome_term = gf2_matmul(model.H_check, v[:, None])[:, 0] ^ wiring.sigma_arg
    weight_s = float(syndrome_term.sum())
    if wiring.q_zero:
        assert weight_s == 0
        return wiring.K_p * float(v.sum())
    return wiring.K_p * float(v.sum()) + wiring.K_q * weight_s


class TestExhaustiveDeltaE:
    """[[5,1]]（n=5，32 状态）上穷举所有 (状态, move) 的 ΔE。"""

    def setup_method(self):
        self.model, self.frame, self.obs_set = build_setup(
            repetition_parity_check_matrix(2)
        )

    def _all_states(self):
        n = self.model.num_qubits
        for value in range(1 << n):
            yield np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)

    def test_single_bit_delta_matches_full_recompute(self):
        from src.reference_mcmc import McmcState

        _, wiring = make_wiring(self.model, self.frame, 0.12, 0.08, seed=1)
        for v in self._all_states():
            syndrome_term = (
                gf2_matmul(self.model.H_check, v[:, None])[:, 0] ^ wiring.sigma_arg
            ).astype(np.uint8)
            state = McmcState(
                v=v.copy(), syndrome_term=syndrome_term,
                data_weight=int(v.sum()),
                syndrome_weight=int(syndrome_term.sum()),
            )
            for j in range(self.model.num_qubits):
                log_acc = single_bit_log_acceptance(self.model, wiring, state, j)
                flipped = v.copy()
                flipped[j] ^= 1
                expected = -(
                    total_energy_of(self.model, wiring, flipped)
                    - total_energy_of(self.model, wiring, v)
                )
                assert np.isclose(log_acc, expected, atol=1e-12)

    def test_support_moves_delta_matches_full_recompute(self):
        from src.reference_mcmc import McmcState

        _, wiring = make_wiring(self.model, self.frame, 0.12, 0.08, seed=2)
        supports = [np.flatnonzero(r).astype(np.int64)
                    for r in self.model.stabilizer_rows] + [
            np.flatnonzero(r).astype(np.int64)
            for r in self.model.logical_move_basis
        ]
        for v in self._all_states():
            state = McmcState(
                v=v.copy(),
                syndrome_term=np.zeros(self.model.num_checks, dtype=np.uint8),
                data_weight=int(v.sum()), syndrome_weight=0,
            )
            for support in supports:
                log_acc = support_move_log_acceptance(wiring, state, support)
                flipped = v.copy()
                flipped[support] ^= 1
                expected = -wiring.K_p * float(
                    int(flipped.sum()) - int(v.sum())
                )
                assert np.isclose(log_acc, expected, atol=1e-12)
                # syndrome 确实不变
                s1 = gf2_matmul(self.model.H_check, v[:, None])[:, 0]
                s2 = gf2_matmul(self.model.H_check, flipped[:, None])[:, 0]
                assert np.array_equal(s1, s2)


class TestStationarityAgainstEnum:
    @pytest.mark.parametrize("ensemble", ["true_posterior", "repo_compat"])
    def test_surface5_qpos_m_u_matches_enum(self, ensemble):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.12, 0.08, seed=3,
                                       ensemble=ensemble)
        exact_m, _ = enum_m_u(model, obs_set, wiring)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=500, num_measurements=20000,
            num_sweeps_between_measurements=1,
            record_observable_trajectory=True, debug_invariants=False,
        )
        result = run_reference_mcmc(model, frame, obs_set, wiring, config,
                                    seed=42)
        traj = result["observable_trajectory"].astype(np.float64)
        blocks = np.array_split(traj, 20, axis=0)
        block_means = np.array([b.mean(axis=0) for b in blocks])
        stderr = block_means.std(axis=0, ddof=1) / np.sqrt(len(blocks))
        z = (result["m_u"] - exact_m) / np.maximum(stderr, 1e-3)
        assert np.max(np.abs(z)) < 5.0, (result["m_u"], exact_m, z)

    def test_toric2_qpos_state_histogram_matches_gibbs(self):
        """2D toric L=2（256 状态）：状态直方图 vs 精确 Gibbs（分箱 z 检验）。"""
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.15, 0.12, seed=5)
        exact_m, probs = enum_m_u(model, obs_set, wiring)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=500, num_measurements=30000,
            record_state_trajectory=True,
        )
        result = run_reference_mcmc(model, frame, obs_set, wiring, config,
                                    seed=7)
        states = result["state_trajectory"]
        packed = states @ (1 << np.arange(model.num_qubits, dtype=np.int64))
        counts = np.bincount(packed, minlength=1 << model.num_qubits)
        total = counts.sum()
        # 聚合到概率 ≥ 1e-3 的状态箱（其余合并），做粗 z 检验（样本相关 ⇒ 容差放宽）
        for state_value, probability in probs.items():
            if probability < 5e-3:
                continue
            empirical = counts[state_value] / total
            sd = np.sqrt(probability * (1 - probability) / total)
            assert abs(empirical - probability) < 12 * sd, (
                state_value, empirical, probability
            )
        # m_u 一致（更敏感的整体量）
        traj_m = result["m_u"]
        assert np.max(np.abs(traj_m - exact_m)) < 0.03

    def test_q_zero_coset_sampling_matches_enum(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.18, 0.0, seed=9)
        assert wiring.q_zero
        exact_m, probs = enum_m_u(model, obs_set, wiring)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=300, num_measurements=20000,
            debug_invariants=True,
        )
        result = run_reference_mcmc(model, frame, obs_set, wiring, config,
                                    seed=11, initial_mode="section")
        assert np.max(np.abs(result["m_u"] - exact_m)) < 0.03
        # 从不同 sector 起点也一致（q=0 遍历性：S+L moves 融通 coset）
        result_shift = run_reference_mcmc(
            model, frame, obs_set, wiring, config, seed=12,
            initial_mode="section", sector_bitmask=0b11,
        )
        assert np.max(np.abs(result_shift["m_u"] - exact_m)) < 0.03


class TestAnalyticAnchors:
    def test_p_half_logical_acceptance_is_one_and_m_u_zero(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.5, 0.1, seed=13)
        assert wiring.K_p == 0.0
        config = ReferenceMcmcConfig(num_burn_in_sweeps=200,
                                     num_measurements=16000,
                                     record_observable_trajectory=True)
        result = run_reference_mcmc(model, frame, obs_set, wiring, config,
                                    seed=15)
        rates = result["acceptance"]
        assert np.all(rates["logical_per_u"] == 1.0)  # ΔE=0 ⇒ 恒接受
        assert rates["stabilizer"] == 1.0
        # 精确值 0；组合 u 的 O_u 只被 single-bit 缓慢去相关（L 成对翻转抵消、
        # ⟨w_u,S⟩=0 ⇒ stabilizer 不翻）——用分块 z 检验自校准误差
        traj = result["observable_trajectory"].astype(np.float64)
        block_means = np.array(
            [b.mean(axis=0) for b in np.array_split(traj, 20, axis=0)]
        )
        stderr = block_means.std(axis=0, ddof=1) / np.sqrt(20)
        z = result["m_u"] / np.maximum(stderr, 1e-3)
        assert np.max(np.abs(z)) < 5.0, (result["m_u"], stderr, z)

    def test_per_u_counters_populated(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.1, 0.05, seed=17)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=10, num_measurements=50,
                                     logical_move_repeat=2)
        result = run_reference_mcmc(model, frame, obs_set, wiring, config,
                                    seed=19)
        counters = result["counters"]
        expected_attempts = (10 + 50) * 2  # (burnin+measure sweeps)×repeat
        assert np.all(counters.logical_attempts_per_u == expected_attempts)


class TestReproducibilityAndInvariants:
    def test_same_seed_identical_trajectory(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.1, 0.07, seed=21)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=50, num_measurements=500)
        r1 = run_reference_mcmc(model, frame, obs_set, wiring, config, seed=23)
        r2 = run_reference_mcmc(model, frame, obs_set, wiring, config, seed=23)
        assert np.array_equal(r1["observable_sums"], r2["observable_sums"])
        assert np.array_equal(r1["final_state"].v, r2["final_state"].v)

    def test_debug_invariants_pass_long_run(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        disorder, wiring = make_wiring(model, frame, 0.2, 0.15, seed=25)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=100,
                                     num_measurements=500,
                                     debug_invariants=True)
        run_reference_mcmc(model, frame, obs_set, wiring, config, seed=27)

    def test_q_zero_bad_initial_rejected(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        rng = np.random.default_rng(29)
        disorder = draw_disorder(model, 0.2, 0.0, rng)
        # 强制非零 syndrome（若 η 恰为 0 则翻一位）
        if not disorder.observed_syndrome.any():
            disorder.eta[0] ^= 1
            disorder.observed_syndrome = gf2_matmul(
                model.H_check, disorder.eta[:, None]
            )[:, 0]
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        with pytest.raises(ValueError, match="hard constraint"):
            make_initial_state(model, wiring, mode="zero")
