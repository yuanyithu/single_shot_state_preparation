"""G2.5 单元测试：PT ladder / swap 公式 / PT 端到端对枚举与单链。"""

import numpy as np
import pytest

from src.graphs import cycle_parity_check_matrix, repetition_parity_check_matrix
from src.hgp import hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import (
    assemble_sector_model,
    coupling_from_probability,
    draw_disorder,
    wire_ensemble,
)
from src.observables import build_observable_frame, build_observable_set
from src.pt import (
    PtConfig,
    _RoundTripCounter,
    data_only_ladder,
    probability_from_coupling,
    run_parallel_tempering,
    swap_log_ratio,
    sync_enlarge_ladder,
)
from src.reference_mcmc import (
    McmcState,
    ReferenceMcmcConfig,
    run_reference_mcmc,
)
from tests.util_enum import enum_m_u


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


class TestLadders:
    def test_sync_enlarge_endpoints_and_monotonic(self):
        p_ladder, q_ladder = sync_enlarge_ladder(0.05, 0.03, 0.40, 8)
        assert np.isclose(p_ladder[0], 0.05) and np.isclose(q_ladder[0], 0.03)
        assert np.isclose(q_ladder[-1], 0.40)
        assert np.all(np.diff(p_ladder) > 0) and np.all(np.diff(q_ladder) > 0)
        assert np.all(p_ladder < 0.5) and np.all(q_ladder < 0.5)
        # 同步性：每 rung 的 K_p/K_q 之比恒定
        ratios = [
            coupling_from_probability(p) / coupling_from_probability(q)
            for p, q in zip(p_ladder, q_ladder)
        ]
        assert np.allclose(ratios, ratios[0])

    def test_sync_enlarge_safe_by_construction(self):
        """耦合空间表述的安全性质：K_k=K/enlarge>0 ⇒ p_k=1/(1+e^{K_k})<0.5 恒成立。

        （exp35 的 p 越界坑属于主项目的 odds 缩放表述；本表述构造性免疫，
        运行时守卫保留为防御性死代码。）极端参数下逼近但不越界：
        """
        p_ladder, q_ladder = sync_enlarge_ladder(0.30, 0.02, 0.45, 8)
        assert p_ladder[-1] > 0.45  # 强放大下 p_hot 逼近 0.5
        assert np.all(p_ladder < 0.5) and np.all(q_ladder < 0.5)
        assert np.isclose(q_ladder[-1], 0.45)

    def test_sync_enlarge_input_validation(self):
        with pytest.raises(ValueError):
            sync_enlarge_ladder(0.05, 0.10, 0.08, 8)  # q_hot < q_cold
        with pytest.raises(ValueError):
            sync_enlarge_ladder(0.05, 0.03, 0.40, 1)

    def test_data_only_ladder(self):
        p_ladder, q_ladder = data_only_ladder(0.05, 0.30, 0.07, 5)
        assert np.isclose(p_ladder[0], 0.05) and np.isclose(p_ladder[-1], 0.30)
        assert np.all(q_ladder == 0.07)

    def test_probability_coupling_roundtrip(self):
        for p in (0.01, 0.1, 0.3, 0.49):
            assert np.isclose(
                probability_from_coupling(coupling_from_probability(p)), p
            )


class TestSwapFormula:
    def test_swap_log_ratio_matches_direct_weight_ratio(self):
        """fuzz：随机构型对，swap 公式 vs 直接能量重算的比值。"""
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        wiring_cold = make_wiring(model, frame, 0.08, 0.05, seed=1)
        p_ladder, q_ladder = sync_enlarge_ladder(0.08, 0.05, 0.40, 4)
        from src.model import EnsembleWiring

        wirings = [
            EnsembleWiring(
                ensemble="true_posterior",
                gibbs_syndrome_argument=(
                    wiring_cold.gibbs_syndrome_argument
                ),
                planted_logical_class=wiring_cold.planted_logical_class,
                K_p=coupling_from_probability(p_ladder[r]),
                K_q=coupling_from_probability(q_ladder[r]), q_zero=False,
            )
            for r in range(4)
        ]
        rng = np.random.default_rng(2)
        from src.gf2 import gf2_matmul

        def state_of(v):
            syndrome = (
                gf2_matmul(model.H_check, v[:, None])[:, 0]
                ^ wiring_cold.gibbs_syndrome_argument
            ).astype(np.uint8)
            return McmcState(v=v, syndrome_term=syndrome,
                             data_weight=int(v.sum()),
                             syndrome_weight=int(syndrome.sum()))

        for _ in range(50):
            i = int(rng.integers(0, 3))
            v_i = (rng.random(model.num_qubits) < 0.4).astype(np.uint8)
            v_j = (rng.random(model.num_qubits) < 0.4).astype(np.uint8)
            s_i, s_j = state_of(v_i), state_of(v_j)
            got = swap_log_ratio(wirings[i], wirings[i + 1], s_i, s_j)

            def log_weight(wr, st):
                return -wr.K_p * st.data_weight - wr.K_q * st.syndrome_weight

            expected = (
                log_weight(wirings[i], s_j) + log_weight(wirings[i + 1], s_i)
                - log_weight(wirings[i], s_i) - log_weight(wirings[i + 1], s_j)
            )
            assert np.isclose(got, expected, atol=1e-12)


class TestRoundTripCounter:
    def test_initial_hot_to_cold_is_not_a_round_trip(self):
        counter = _RoundTripCounter(num_replicas=2, cold_replica=0)
        # Replica 1 starts hot.  Its first arrival at cold has not completed a
        # cold->hot->cold path and must not increment the counter.
        counter.observe_endpoints(cold_replica=1, hot_replica=0)
        assert counter.total == 0
        assert counter.counts.tolist() == [0, 0]
        # Replica 0 did start cold and has now visited hot; returning it to
        # cold completes exactly one valid path.
        counter.observe_endpoints(cold_replica=0, hot_replica=1)
        assert counter.total == 1
        assert counter.counts.tolist() == [1, 0]

    def test_new_phase_does_not_inherit_partial_transit(self):
        burn = _RoundTripCounter(num_replicas=2, cold_replica=0)
        burn.observe_endpoints(cold_replica=1, hot_replica=0)
        assert burn.total == 0
        measurement = _RoundTripCounter(num_replicas=2, cold_replica=1)
        measurement.observe_endpoints(cold_replica=0, hot_replica=1)
        assert measurement.total == 0


class TestPtEndToEnd:
    def test_pt_cold_m_u_matches_enum_surface5(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.10, 0.06, seed=3)
        exact_m, _ = enum_m_u(model, obs_set, wiring)
        config = PtConfig(
            num_temperatures=4, q_hot=0.40, num_burn_in_rounds=300,
            num_measurement_rounds=12000,
            record_observable_trajectory=True,
        )
        result = run_parallel_tempering(model, frame, obs_set, wiring, config,
                                        seed=5)
        traj = result.observable_trajectory_cold.astype(np.float64)
        blocks = np.array_split(traj, 20, axis=0)
        block_means = np.array([b.mean(axis=0) for b in blocks])
        stderr = block_means.std(axis=0, ddof=1) / np.sqrt(20)
        z = (result.m_u_cold - exact_m) / np.maximum(stderr, 1e-3)
        assert np.max(np.abs(z)) < 5.0, (result.m_u_cold, exact_m, z)
        # ladder 端点 = 冷参数（rung0 断言在实现内，附带验证 swap 有活动）
        assert result.swap_attempts.sum() > 0
        assert result.swap_rates().min() > 0.01

    def test_pt_agrees_with_long_single_chain_toric2(self):
        model, frame, obs_set = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=7)
        pt_result = run_parallel_tempering(
            model, frame, obs_set, wiring,
            PtConfig(num_temperatures=4, q_hot=0.40,
                     num_burn_in_rounds=300, num_measurement_rounds=10000,
                     record_observable_trajectory=True),
            seed=9,
        )
        single = run_reference_mcmc(
            model, frame, obs_set, wiring,
            ReferenceMcmcConfig(num_burn_in_sweeps=1000,
                                num_measurements=30000,
                                record_observable_trajectory=True),
            seed=11,
        )
        def stderr_of(traj):
            blocks = np.array_split(traj.astype(np.float64), 20, axis=0)
            means = np.array([b.mean(axis=0) for b in blocks])
            return means.std(axis=0, ddof=1) / np.sqrt(20)

        combined = np.sqrt(
            stderr_of(pt_result.observable_trajectory_cold) ** 2
            + stderr_of(single["observable_trajectory"]) ** 2
        )
        z = (pt_result.m_u_cold - single["m_u"]) / np.maximum(combined, 1e-3)
        assert np.max(np.abs(z)) < 5.0

    def test_round_trips_and_replica_conservation(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.10, seed=13)
        result = run_parallel_tempering(
            model, frame, obs_set, wiring,
            PtConfig(num_temperatures=4, q_hot=0.42,
                     num_burn_in_rounds=100, num_measurement_rounds=3000),
            seed=15,
        )
        assert result.round_trips > 0  # 易参数下必须有替身往返
        assert result.round_trips == result.measurement_round_trips
        assert result.burn_in_round_trips >= 0
        assert result.burn_in_round_trips_per_replica.shape == (4,)
        assert result.measurement_round_trips_per_replica.shape == (4,)
        assert sorted(result.replica_id_per_rung.tolist()) == [0, 1, 2, 3]
        # per-u 冷端 logical 接受率被记录且有限
        cold_rates = result.cold_logical_acceptance_per_u()
        assert cold_rates.shape == (model.k,)
        assert np.all(np.isfinite(cold_rates))

    def test_q_zero_rejected(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.10, 0.0, seed=17)
        with pytest.raises(ValueError, match="q>0"):
            run_parallel_tempering(model, frame, obs_set, wiring,
                                   PtConfig(num_temperatures=4, q_hot=0.4),
                                   seed=19)

    def test_reproducible(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.10, 0.06, seed=21)
        config = PtConfig(num_temperatures=3, q_hot=0.4,
                          num_burn_in_rounds=50, num_measurement_rounds=500)
        r1 = run_parallel_tempering(model, frame, obs_set, wiring, config,
                                    seed=23)
        r2 = run_parallel_tempering(model, frame, obs_set, wiring, config,
                                    seed=23)
        assert np.array_equal(r1.observable_sums_cold, r2.observable_sums_cold)
        assert r1.round_trips == r2.round_trips

    def test_observable_all_rungs_mode(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.10, 0.06, seed=25)
        result = run_parallel_tempering(
            model, frame, obs_set, wiring,
            PtConfig(num_temperatures=3, q_hot=0.4, num_burn_in_rounds=50,
                     num_measurement_rounds=500, observable_rungs="all"),
            seed=27,
        )
        assert result.m_u_per_rung.shape == (3, obs_set.num_u)
        assert np.allclose(result.m_u_per_rung[0], result.m_u_cold)
        # 热端更无序：|m_u| 单调趋弱的方向性（宽松检查）
        assert np.abs(result.m_u_per_rung[-1]).mean() <= (
            np.abs(result.m_u_per_rung[0]).mean() + 0.1
        )
