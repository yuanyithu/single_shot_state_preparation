"""G2.1 单元测试：section（线性部分）/ model / observables。

关键锁定：
  - w_u 三性质（独立复核，不只靠构造内断言）
  - 规范形式（|v|, σ_arg）与旧形式（|v⊕η|, s, 标签相对 η）的换元等价（数值逐位）
  - true 系综桥梁恒等式 T3 在新接线下成立
  - full / sampled 两档估计量一致；聚合公式对手工类分布精确
"""

import numpy as np
import pytest

from src.gf2 import gf2_matmul
from src.graphs import cycle_parity_check_matrix, random_biregular_graph_from_m
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import (
    assemble_sector_model,
    coupling_from_probability,
    disorder_from_uniforms,
    draw_disorder,
    wire_ensemble,
)
from src.observables import (
    aggregate_observables,
    build_observable_frame,
    build_observable_set,
    observable_values,
)
from src.section import build_linear_section


def build_models(classical, sectors=("x_error", "z_error")):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    return [assemble_sector_model(H_X, H_Z, logicals, sector=s) for s in sectors]


def enum_m_u_canonical(model, frame, obs_set, wiring):
    """规范形式全枚举：权重 exp[−K_p|v|−K_q|Hv⊕σ_arg|]（q=0 → 硬约束）。"""
    n = model.num_qubits
    total = np.zeros(obs_set.num_u, dtype=np.float64)
    norm = 0.0
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        syndrome_term = gf2_matmul(model.H_check, v[:, None])[:, 0] ^ wiring.sigma_arg
        weight_s = int(syndrome_term.sum())
        if wiring.q_zero:
            if weight_s:
                continue
            log_weight = -wiring.K_p * float(v.sum())
        else:
            log_weight = -wiring.K_p * float(v.sum()) - wiring.K_q * float(weight_s)
        weight = np.exp(log_weight)
        total += weight * observable_values(obs_set, wiring, v).astype(np.float64)
        norm += weight
    return total / norm


def enum_m_u_legacy_repo_form(model, frame, obs_set, disorder):
    """旧形式（主项目约定）：权重 exp[−K_p|v⊕η|−K_q|Hv⊕s|]，标签相对 η。

    不用新 wiring 机制，直接按公式（作为独立参照）。
    """
    n = model.num_qubits
    K_p = coupling_from_probability(disorder.p)
    K_q = coupling_from_probability(disorder.q)
    total = np.zeros(obs_set.num_u, dtype=np.float64)
    norm = 0.0
    W = obs_set.W_rows
    label_eta = gf2_matmul(W, disorder.eta[:, None])[:, 0]
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        data_w = int(np.count_nonzero(v ^ disorder.eta))
        syndrome_w = int(
            (gf2_matmul(model.H_check, v[:, None])[:, 0]
             ^ disorder.observed_syndrome).sum()
        )
        weight = np.exp(-K_p * data_w - K_q * syndrome_w)
        parity = gf2_matmul(W, v[:, None])[:, 0] ^ label_eta
        total += weight * (1.0 - 2.0 * parity.astype(np.float64))
        norm += weight
    return total / norm


@pytest.fixture(scope="module")
def toric2_setup():
    classical = cycle_parity_check_matrix(2)
    model = build_models(classical, sectors=("x_error",))[0]
    frame = build_observable_frame(model)
    obs_set = build_observable_set(frame)
    return model, frame, obs_set


class TestSectionLinear:
    @pytest.mark.parametrize("classical_builder", [
        lambda: cycle_parity_check_matrix(3),
        lambda: np.ones((3, 4), dtype=np.uint8),
        lambda: classical_parity_check_matrix(
            random_biregular_graph_from_m(2, 3, 4, seed=12345)),
    ])
    def test_section_property_and_linearity(self, classical_builder):
        H_Z, _ = hgp_from_H(classical_builder())
        section = build_linear_section(H_Z)
        rng = np.random.default_rng(3)
        sigmas = []
        for _ in range(10):
            x = (rng.random(H_Z.shape[1]) < 0.4).astype(np.uint8)
            sigma = gf2_matmul(H_Z, x[:, None])[:, 0]
            sigmas.append(sigma)
            r = section.apply(sigma)
            assert np.array_equal(gf2_matmul(H_Z, r[:, None])[:, 0], sigma)
            assert section.in_image(sigma)
        # 线性：r(σ1⊕σ2) = r(σ1)⊕r(σ2)
        r12 = section.apply(sigmas[0] ^ sigmas[1])
        assert np.array_equal(
            r12, section.apply(sigmas[0]) ^ section.apply(sigmas[1])
        )

    def test_misuse_rejected(self):
        H_Z, _ = hgp_from_H(cycle_parity_check_matrix(2))  # rank 亏 1 ⇒ im 真子集
        section = build_linear_section(H_Z)
        bad = np.zeros(H_Z.shape[0], dtype=np.uint8)
        bad[0] = 1  # 单 check 违反 ∉ im（toric 约束：违反数偶）
        assert not section.in_image(bad)
        with pytest.raises(ValueError, match="im\\(H\\)"):
            section.apply(bad)

    def test_section_after_H_columns(self):
        H_Z, _ = hgp_from_H(cycle_parity_check_matrix(3))
        section = build_linear_section(H_Z)
        RH = section.section_after_H(H_Z)
        for j in (0, 5, 11):
            e = np.zeros(H_Z.shape[1], dtype=np.uint8)
            e[j] = 1
            sigma = gf2_matmul(H_Z, e[:, None])[:, 0]
            assert np.array_equal(RH[:, j], section.apply(sigma))


class TestObservableFrame:
    @pytest.mark.parametrize("classical_builder,check_sectors", [
        (lambda: cycle_parity_check_matrix(2), ("x_error", "z_error")),
        (lambda: cycle_parity_check_matrix(3), ("x_error", "z_error")),
        (lambda: np.ones((3, 4), dtype=np.uint8), ("x_error",)),
        (lambda: classical_parity_check_matrix(
            random_biregular_graph_from_m(2, 3, 4, seed=12345)), ("x_error", "z_error")),
    ])
    def test_w_three_properties_independent_recheck(self, classical_builder,
                                                    check_sectors):
        for model in build_models(classical_builder(), sectors=check_sectors):
            frame = build_observable_frame(model)  # 内部断言已跑
            W = frame.W_basis
            # 独立复核 (i)：对随机 im 元素的 section 像正交
            rng = np.random.default_rng(11)
            for _ in range(5):
                x = (rng.random(model.num_qubits) < 0.4).astype(np.uint8)
                sigma = gf2_matmul(model.H_check, x[:, None])[:, 0]
                t_vec = model.section.apply(sigma)
                assert not gf2_matmul(W, t_vec[:, None]).any()
            # (ii) stabilizer 行随机组合
            coeff = (rng.random(model.stabilizer_rows.shape[0]) < 0.5).astype(np.uint8)
            s_vec = gf2_matmul(coeff[None, :], model.stabilizer_rows)[0]
            assert not gf2_matmul(W, s_vec[:, None]).any()
            # (iii) 配对
            pairing = gf2_matmul(W, model.logical_move_basis.T)
            assert np.array_equal(pairing, np.eye(model.k, dtype=np.uint8))

    def test_label_decomposition_invariance(self, toric2_setup):
        """φ(v ⊕ stabilizer ⊕ T-元素) = φ(v)；φ(v ⊕ x_u) = φ(v)⊕e_u。"""
        model, frame, _ = toric2_setup
        rng = np.random.default_rng(5)
        v = (rng.random(model.num_qubits) < 0.5).astype(np.uint8)
        base = frame.label_of(v)
        stab = model.stabilizer_rows[0]
        assert np.array_equal(frame.label_of(v ^ stab), base)
        x0 = model.logical_move_basis[0]
        expected = base.copy()
        expected[0] ^= 1
        assert np.array_equal(frame.label_of(v ^ x0), expected)


class TestEnsembleWiringAndEquivalence:
    def _disorder(self, model, p, q, seed, force_nontrivial_eta=True):
        rng = np.random.default_rng(seed)
        disorder = draw_disorder(model, p, q, rng)
        if force_nontrivial_eta and not gf2_matmul(
                model.H_check, disorder.eta[:, None]).any():
            disorder.eta[0] ^= 1
            disorder.observed_syndrome = (
                gf2_matmul(model.H_check, disorder.eta[:, None])[:, 0]
                ^ disorder.delta
            )
            disorder.eta_weight = int(disorder.eta.sum())
        return disorder

    def test_repo_compat_canonical_equals_legacy_form(self, toric2_setup):
        """换元锁定：规范 (|v|, δ, ℓ_ref=0) ≡ 旧式 (|v⊕η|, s, 标签相对 η)。"""
        model, frame, obs_set = toric2_setup
        for seed in (1, 2, 3):
            disorder = self._disorder(model, 0.15, 0.1, seed)
            wiring = wire_ensemble(model, disorder, "repo_compat", frame)
            canonical = enum_m_u_canonical(model, frame, obs_set, wiring)
            legacy = enum_m_u_legacy_repo_form(model, frame, obs_set, disorder)
            assert np.allclose(canonical, legacy, atol=1e-12)

    def test_true_posterior_bridge_identity_T3(self, toric2_setup):
        """m_true(η,δ) = (−1)^{⟨u,φ(η)⟩} 因子已含于 wiring；验证与 (η=0, δ:=s) 等价。"""
        model, frame, obs_set = toric2_setup
        disorder = self._disorder(model, 0.15, 0.1, seed=7)
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        m_direct = enum_m_u_canonical(model, frame, obs_set, wiring)
        # 构造 η=0、δ=s 的 disorder：σ_arg 相同、ℓ_ref=0
        from src.model import DisorderRealization

        shifted = DisorderRealization(
            eta=np.zeros(model.num_qubits, dtype=np.uint8),
            delta=disorder.observed_syndrome.copy(),
            observed_syndrome=disorder.observed_syndrome.copy(),
            p=disorder.p, q=disorder.q,
        )
        wiring_shifted = wire_ensemble(model, shifted, "true_posterior", frame)
        m_shifted = enum_m_u_canonical(model, frame, obs_set, wiring_shifted)
        # 手工乘符号 (−1)^{⟨u, φ(η)⟩}
        label_eta = frame.label_of(disorder.eta)
        mask = sum(1 << b for b, bit in enumerate(label_eta) if bit)
        signs = np.array(
            [1.0 - 2.0 * (int(u & mask).bit_count() & 1)
             for u in obs_set.u_bitmasks]
        )
        assert np.allclose(m_direct, signs * m_shifted, atol=1e-12)

    def test_repo_compat_eta_independence_T1(self, toric2_setup):
        model, frame, obs_set = toric2_setup
        base = self._disorder(model, 0.15, 0.1, seed=11)
        from src.model import DisorderRealization

        results = []
        for eta_seed in (0, 1, 2):
            rng = np.random.default_rng(eta_seed + 100)
            eta = (rng.random(model.num_qubits) < 0.3).astype(np.uint8)
            disorder = DisorderRealization(
                eta=eta,
                delta=base.delta.copy(),
                observed_syndrome=(
                    gf2_matmul(model.H_check, eta[:, None])[:, 0] ^ base.delta
                ),
                p=0.15, q=0.1,
            )
            wiring = wire_ensemble(model, disorder, "repo_compat", frame)
            results.append(enum_m_u_canonical(model, frame, obs_set, wiring))
        assert np.allclose(results[0], results[1], atol=1e-12)
        assert np.allclose(results[0], results[2], atol=1e-12)

    def test_observable_at_truth_is_plus_one(self, toric2_setup):
        """true 系综：v = η（= 候选即真实错误）⇒ 所有 O_u = +1。"""
        model, frame, obs_set = toric2_setup
        disorder = self._disorder(model, 0.15, 0.1, seed=13)
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        values = observable_values(obs_set, wiring, disorder.eta)
        assert np.all(values == 1)

    def test_q_zero_wiring(self, toric2_setup):
        model, frame, _ = toric2_setup
        disorder = self._disorder(model, 0.15, 0.0, seed=17)
        assert disorder.delta_weight == 0
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        assert wiring.q_zero
        # 硬约束满足的 v（=η）能量有限；违反者抛错
        assert np.isfinite(wiring.total_energy(model, disorder.eta))
        bad = disorder.eta.copy()
        bad[0] ^= 1
        with pytest.raises(ValueError, match="hard constraint"):
            wiring.total_energy(model, bad)


class TestObservableSetTiers:
    def test_full_vs_sampled_consistency(self):
        """k=4 码：强制 sampled 档（full_max_k=2），逐 u 与全量档一致。"""
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        frame = build_observable_frame(model)
        full_set = build_observable_set(frame, full_max_k=10)
        sampled_set = build_observable_set(
            frame, full_max_k=2, num_random_u=8, u_rand_seed=42
        )
        assert full_set.tier == "full" and sampled_set.tier == "sampled"
        # sampled 的每个 u 行必须与 full 中同 bitmask 的行一致
        full_index = {int(u): i for i, u in enumerate(full_set.u_bitmasks)}
        for i, u in enumerate(sampled_set.u_bitmasks):
            j = full_index[int(u)]
            assert np.array_equal(sampled_set.W_rows[i], full_set.W_rows[j])
        # basis 行位置正确：u = e_i
        for i, pos in enumerate(sampled_set.basis_positions):
            assert int(sampled_set.u_bitmasks[pos]) == 1 << i

    def test_sampled_requires_seed(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        frame = build_observable_frame(model)
        with pytest.raises(ValueError, match="u_rand_seed"):
            build_observable_set(frame, full_max_k=2, num_random_u=4)

    def test_sampled_infeasible_request_raises(self):
        """回归（G2.4 排障）：请求的随机 u 数超过可用非 basis u 总数必须报错，
        绝不允许拒绝采样死循环。k=4 ⇒ 可用 2^4−1−4=11。"""
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        frame = build_observable_frame(model)
        with pytest.raises(ValueError, match="exceeds available"):
            build_observable_set(frame, full_max_k=2, num_random_u=16,
                                 u_rand_seed=5)
        # 恰好取满 11 个可行
        obs = build_observable_set(frame, full_max_k=2, num_random_u=11,
                                   u_rand_seed=5)
        assert obs.num_u == 4 + 11


class TestAggregates:
    def test_aggregates_match_hand_class_distribution(self, toric2_setup):
        """k=2：手工类分布 P(ℓ) → m_u → 聚合公式 vs 第一性直算。"""
        model, frame, obs_set = toric2_setup
        assert obs_set.tier == "full" and obs_set.k == 2
        P = {0: 0.6, 1: 0.25, 2: 0.1, 3: 0.05}  # 相对类分布 P̃(ℓ)
        m_u = []
        for u in obs_set.u_bitmasks:
            m = sum(P[l] * (1 - 2 * (int(u & l).bit_count() & 1)) for l in P)
            m_u.append(m)
        agg = aggregate_observables(obs_set, np.array(m_u))
        purity_direct = sum(p * p for p in P.values())
        assert np.isclose(agg["purity"], purity_direct, atol=1e-12)
        assert np.isclose(agg["w0"], P[0], atol=1e-12)
        assert np.isclose(
            agg["q_top_all"], (4 * purity_direct - 1) / 3, atol=1e-12
        )
        assert np.isclose(agg["q_top_basis"], np.mean(np.array(m_u)[
            obs_set.basis_positions.tolist()] ** 2), atol=1e-12)

    def test_sampled_aggregate_unbiasedness_structure(self):
        """sampled 档聚合对「全部非零 u」平均的复现：用全集充当抽样集验证公式。"""
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        frame = build_observable_frame(model)
        full_set = build_observable_set(frame, full_max_k=10)
        rng = np.random.default_rng(9)
        m_u = rng.uniform(-1, 1, size=full_set.num_u)
        agg_full = aggregate_observables(full_set, m_u)
        # 构造一个 sampled 集 = 全部非零 u（basis + 其余全部作为“随机”）
        sampled = build_observable_set(
            frame, full_max_k=2, num_random_u=full_set.num_u - frame.k,
            u_rand_seed=1,
        )
        # 把 m_u 重排到 sampled 的 u 顺序
        index = {int(u): i for i, u in enumerate(full_set.u_bitmasks)}
        m_sampled = np.array([m_u[index[int(u)]] for u in sampled.u_bitmasks])
        agg_sampled = aggregate_observables(sampled, m_sampled)
        # 抽到全体非零 u 时，“无偏估计”与全量精确值只差 basis 行不进平均的口径；
        # 此处验证公式一致性：用去除 basis 的均值重算 full 口径
        random_mask = np.ones(sampled.num_u, dtype=bool)
        random_mask[sampled.basis_positions] = False
        assert np.isclose(
            agg_sampled["q_top_all"], float(np.mean(m_sampled[random_mask] ** 2))
        )


class TestDisorderDraw:
    def test_reproducible_and_crn(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        d1 = draw_disorder(model, 0.1, 0.05, np.random.default_rng(7))
        d2 = draw_disorder(model, 0.1, 0.05, np.random.default_rng(7))
        assert np.array_equal(d1.eta, d2.eta) and np.array_equal(d1.delta, d2.delta)
        # CRN：同一批 uniforms，p 增大 ⇒ η 单调增长（集合包含）
        rng = np.random.default_rng(8)
        du = rng.random(model.num_qubits)
        su = rng.random(model.num_checks)
        low = disorder_from_uniforms(model, 0.05, 0.02, du, su)
        high = disorder_from_uniforms(model, 0.2, 0.02, du, su)
        assert np.all(low.eta <= high.eta)
        # s = Hη ⊕ δ 自洽
        expected = gf2_matmul(model.H_check, high.eta[:, None])[:, 0] ^ high.delta
        assert np.array_equal(high.observed_syndrome, expected)

    def test_coupling_edge_values(self):
        assert coupling_from_probability(0.5) == 0.0
        assert np.isinf(coupling_from_probability(0.0))
        with pytest.raises(ValueError):
            coupling_from_probability(0.7)
