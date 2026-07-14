"""Canonical disorder, absolute/relative characters, and estimators.

The old repository-energy equivalence below is deliberately limited to the
``legacy_delta_only`` regression mode; it is not a coordinate description of
the paper's ``true_posterior`` ensemble.
"""

import numpy as np
import pytest

from src.gf2 import gf2_matmul, gf2_rank
from src.graphs import cycle_parity_check_matrix, random_biregular_graph_from_m
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import (
    DisorderRealization,
    EnsembleWiring,
    assemble_sector_model,
    coupling_from_probability,
    disorder_from_uniforms,
    draw_disorder,
    normalize_ensemble,
    wire_ensemble,
)
from src.observables import (
    absolute_observable_values,
    aggregate_observables,
    build_observable_frame,
    build_observable_set,
    independent_chain_squared_character_estimates,
    observable_values,
    posterior_statistics,
    relative_observable_values,
    sampled_nonzero_character_mean,
)
from src.section import build_linear_section


def build_models(classical, sectors=("x_error", "z_error")):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    return [assemble_sector_model(H_X, H_Z, logicals, sector=s) for s in sectors]


def enum_m_u_canonical(model, frame, obs_set, wiring):
    """Enumerate the reduced weight in its absolute candidate-error state."""
    n = model.num_qubits
    total = np.zeros(obs_set.num_u, dtype=np.float64)
    norm = 0.0
    for value in range(1 << n):
        e = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        syndrome_term = (
            gf2_matmul(model.H_check, e[:, None])[:, 0]
            ^ wiring.gibbs_syndrome_argument
        )
        weight_s = int(syndrome_term.sum())
        if wiring.q_zero:
            if weight_s:
                continue
            log_weight = -wiring.K_p * float(e.sum())
        else:
            log_weight = (
                -wiring.K_p * float(e.sum())
                - wiring.K_q * float(weight_s)
            )
        weight = np.exp(log_weight)
        total += weight * observable_values(
            obs_set, wiring, e
        ).astype(np.float64)
        norm += weight
    return total / norm


def enum_m_u_old_repository_form(model, frame, obs_set, disorder):
    """Independently evaluate the historical delta-only repository formula.

    This is regression evidence only and is never used as the paper posterior.
    """
    n = model.num_qubits
    K_p = coupling_from_probability(disorder.p)
    K_q = coupling_from_probability(disorder.q)
    total = np.zeros(obs_set.num_u, dtype=np.float64)
    norm = 0.0
    W = obs_set.W_rows
    label_eta = gf2_matmul(W, disorder.epsilon_data_true[:, None])[:, 0]
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        data_w = int(np.count_nonzero(v ^ disorder.epsilon_data_true))
        syndrome_w = int(
            (gf2_matmul(model.H_check, v[:, None])[:, 0]
             ^ disorder.effective_syndrome).sum()
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
            kernel_generators = np.vstack((
                model.stabilizer_rows, model.logical_move_basis
            ))
            assert gf2_rank(kernel_generators) == (
                model.num_qubits - gf2_rank(model.H_check)
            )
            assert not gf2_matmul(
                model.stabilizer_rows, model.logical_obs_basis.T
            ).any()
            # 独立复核 (i)：对随机 im 元素的 section 像正交
            rng = np.random.default_rng(11)
            for _ in range(5):
                x = (rng.random(model.num_qubits) < 0.4).astype(np.uint8)
                sigma = gf2_matmul(model.H_check, x[:, None])[:, 0]
                t_vec = model.logical_sector_section.apply(sigma)
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
                model.H_check, disorder.epsilon_data_true[:, None]).any():
            disorder.epsilon_data_true[0] ^= 1
            disorder.effective_syndrome = (
                gf2_matmul(
                    model.H_check, disorder.epsilon_data_true[:, None]
                )[:, 0]
                ^ disorder.measurement_error
            )
            disorder.epsilon_data_weight = int(
                disorder.epsilon_data_true.sum()
            )
        return disorder

    def test_legacy_delta_only_equals_old_repository_form(self, toric2_setup):
        """The legacy canonical variable reproduces only the old repository model."""
        model, frame, obs_set = toric2_setup
        for seed in (1, 2, 3):
            disorder = self._disorder(model, 0.15, 0.1, seed)
            wiring = wire_ensemble(
                model, disorder, "legacy_delta_only", frame
            )
            canonical = enum_m_u_canonical(model, frame, obs_set, wiring)
            legacy = enum_m_u_old_repository_form(
                model, frame, obs_set, disorder
            )
            assert np.allclose(canonical, legacy, atol=1e-12)

    def test_fixed_effective_syndrome_absolute_and_relative_relation(
        self, toric2_setup
    ):
        """At fixed effective syndrome, ground truth changes only Mattis signs."""
        model, frame, obs_set = toric2_setup
        disorder = self._disorder(model, 0.15, 0.1, seed=7)
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        m_direct = enum_m_u_canonical(model, frame, obs_set, wiring)
        # A zero planted class with the same fixed Gibbs argument isolates the
        # absolute character means.
        shifted = DisorderRealization(
            epsilon_data_true=np.zeros(model.num_qubits, dtype=np.uint8),
            measurement_error=disorder.effective_syndrome.copy(),
            effective_syndrome=disorder.effective_syndrome.copy(),
            p=disorder.p, q=disorder.q,
        )
        wiring_shifted = wire_ensemble(model, shifted, "true_posterior", frame)
        m_shifted = enum_m_u_canonical(model, frame, obs_set, wiring_shifted)
        # Apply the planted Mattis sign explicitly.
        label_eta = frame.label_of(disorder.epsilon_data_true)
        mask = sum(1 << b for b, bit in enumerate(label_eta) if bit)
        signs = np.array(
            [1.0 - 2.0 * (int(u & mask).bit_count() & 1)
             for u in obs_set.u_bitmasks]
        )
        assert np.allclose(m_direct, signs * m_shifted, atol=1e-12)

    def test_legacy_delta_only_ground_truth_independence(self, toric2_setup):
        model, frame, obs_set = toric2_setup
        base = self._disorder(model, 0.15, 0.1, seed=11)
        results = []
        for eta_seed in (0, 1, 2):
            rng = np.random.default_rng(eta_seed + 100)
            eta = (rng.random(model.num_qubits) < 0.3).astype(np.uint8)
            disorder = DisorderRealization(
                epsilon_data_true=eta,
                measurement_error=base.measurement_error.copy(),
                effective_syndrome=(
                    gf2_matmul(model.H_check, eta[:, None])[:, 0]
                    ^ base.measurement_error
                ),
                p=0.15, q=0.1,
            )
            wiring = wire_ensemble(
                model, disorder, "legacy_delta_only", frame
            )
            results.append(enum_m_u_canonical(model, frame, obs_set, wiring))
        assert np.allclose(results[0], results[1], atol=1e-12)
        assert np.allclose(results[0], results[2], atol=1e-12)

    def test_observable_at_truth_is_plus_one(self, toric2_setup):
        """The planted-relative character is +1 at the true candidate error."""
        model, frame, obs_set = toric2_setup
        disorder = self._disorder(model, 0.15, 0.1, seed=13)
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        values = observable_values(
            obs_set, wiring, disorder.epsilon_data_true
        )
        assert np.all(values == 1)

    def test_q_zero_wiring(self, toric2_setup):
        model, frame, _ = toric2_setup
        disorder = self._disorder(model, 0.15, 0.0, seed=17)
        assert disorder.measurement_error_weight == 0
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        assert wiring.q_zero
        # The true error lies in the quenched coset; a violating state is rejected.
        assert np.isfinite(
            wiring.total_energy(model, disorder.epsilon_data_true)
        )
        bad = disorder.epsilon_data_true.copy()
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
        assert np.isclose(
            agg["posterior_mass_on_planted_class"], P[0], atol=1e-12
        )
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
        # 抽到全部 nonbasis 时必须精确复现全体非零 character 平均。
        assert np.isclose(
            agg_sampled["q_top"], agg_full["q_top"]
        )


class TestDisorderDraw:
    def test_reproducible_and_crn(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(classical_parity_check_matrix(graph),
                             sectors=("x_error",))[0]
        d1 = draw_disorder(model, 0.1, 0.05, np.random.default_rng(7))
        d2 = draw_disorder(model, 0.1, 0.05, np.random.default_rng(7))
        assert np.array_equal(
            d1.epsilon_data_true, d2.epsilon_data_true
        ) and np.array_equal(d1.measurement_error, d2.measurement_error)
        # With common uniforms, increasing p grows the data-error support.
        rng = np.random.default_rng(8)
        du = rng.random(model.num_qubits)
        su = rng.random(model.num_checks)
        low = disorder_from_uniforms(model, 0.05, 0.02, du, su)
        high = disorder_from_uniforms(model, 0.2, 0.02, du, su)
        assert np.all(low.epsilon_data_true <= high.epsilon_data_true)
        # effective_syndrome = H epsilon_data_true xor measurement_error.
        expected = (
            gf2_matmul(
                model.H_check, high.epsilon_data_true[:, None]
            )[:, 0]
            ^ high.measurement_error
        )
        assert np.array_equal(high.effective_syndrome, expected)

    def test_coupling_edge_values(self):
        assert coupling_from_probability(0.5) == 0.0
        assert np.isinf(coupling_from_probability(0.0))
        with pytest.raises(ValueError):
            coupling_from_probability(0.7)

    def test_disorder_dimensions_are_validated_before_wiring(self):
        model = build_models(
            cycle_parity_check_matrix(2), sectors=("x_error",)
        )[0]
        with pytest.raises(ValueError, match="measurement_error.*length"):
            DisorderRealization(
                epsilon_data_true=np.zeros(model.num_qubits, dtype=np.uint8),
                measurement_error=np.zeros(model.num_checks, dtype=np.uint8),
                effective_syndrome=np.zeros(
                    model.num_checks + 1, dtype=np.uint8
                ),
                p=0.1,
                q=0.05,
            )
        disorder = DisorderRealization(
            epsilon_data_true=np.zeros(model.num_qubits + 1, dtype=np.uint8),
            measurement_error=np.zeros(model.num_checks, dtype=np.uint8),
            effective_syndrome=np.zeros(model.num_checks, dtype=np.uint8),
            p=0.1,
            q=0.05,
        )
        with pytest.raises(ValueError, match="epsilon_data_true length"):
            wire_ensemble(
                model,
                disorder,
                "true_posterior",
                build_observable_frame(model),
            )


class TestV2PhysicsSemantics:
    def test_true_energy_uses_fixed_effective_syndrome_not_truth(
        self, toric2_setup
    ):
        model, frame, _ = toric2_setup
        base = draw_disorder(model, 0.15, 0.1, np.random.default_rng(91))
        supported_qubit = int(np.flatnonzero(model.H_check.any(axis=0))[0])
        shifted_truth = base.epsilon_data_true.copy()
        shifted_truth[supported_qubit] ^= 1
        shifted_data_syndrome = gf2_matmul(
            model.H_check, shifted_truth[:, None]
        )[:, 0]
        original_data_syndrome = gf2_matmul(
            model.H_check, base.epsilon_data_true[:, None]
        )[:, 0]
        assert not np.array_equal(
            shifted_data_syndrome, original_data_syndrome
        )
        shifted_measurement = base.effective_syndrome ^ shifted_data_syndrome
        alternative = DisorderRealization(
            epsilon_data_true=shifted_truth,
            measurement_error=shifted_measurement,
            effective_syndrome=base.effective_syndrome,
            p=base.p,
            q=base.q,
        )
        wiring_a = wire_ensemble(model, base, "true_posterior", frame)
        wiring_b = wire_ensemble(model, alternative, "true_posterior", frame)
        assert np.array_equal(
            wiring_a.gibbs_syndrome_argument,
            wiring_b.gibbs_syndrome_argument,
        )
        rng = np.random.default_rng(92)
        for _ in range(10):
            e = (rng.random(model.num_qubits) < 0.5).astype(np.uint8)
            assert wiring_a.total_energy(model, e) == wiring_b.total_energy(
                model, e
            )

    def test_true_and_legacy_differ_for_nontrivial_data_syndrome(
        self, toric2_setup
    ):
        model, frame, _ = toric2_setup
        epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
        epsilon[0] = 1
        measurement = np.zeros(model.num_checks, dtype=np.uint8)
        effective = gf2_matmul(model.H_check, epsilon[:, None])[:, 0]
        assert effective.any()
        disorder = DisorderRealization(
            epsilon_data_true=epsilon,
            measurement_error=measurement,
            effective_syndrome=effective,
            p=0.15,
            q=0.1,
        )
        true = wire_ensemble(model, disorder, "true_posterior", frame)
        legacy = wire_ensemble(
            model, disorder, "legacy_delta_only", frame
        )
        assert not np.array_equal(
            true.gibbs_syndrome_argument,
            legacy.gibbs_syndrome_argument,
        )
        e = np.zeros(model.num_qubits, dtype=np.uint8)
        assert true.total_energy(model, e) != legacy.total_energy(model, e)

    def test_deprecated_ensemble_aliases_normalize_before_storage(
        self, toric2_setup
    ):
        model, frame, _ = toric2_setup
        disorder = draw_disorder(model, 0.1, 0.05, np.random.default_rng(93))
        with pytest.warns(DeprecationWarning):
            true = wire_ensemble(
                model, disorder, "paper_true_posterior", frame
            )
        with pytest.warns(DeprecationWarning):
            legacy = wire_ensemble(model, disorder, "repo_compat", frame)
        assert true.ensemble == "true_posterior"
        assert legacy.ensemble == "legacy_delta_only"
        assert normalize_ensemble("true_posterior") == "true_posterior"

    def test_legacy_disorder_aliases_are_warned_read_only(self, toric2_setup):
        model, _, _ = toric2_setup
        disorder = draw_disorder(model, 0.1, 0.05, np.random.default_rng(94))
        for alias, canonical in (
            ("eta", "epsilon_data_true"),
            ("delta", "measurement_error"),
            ("observed_syndrome", "effective_syndrome"),
        ):
            with pytest.warns(DeprecationWarning, match=alias):
                legacy_value = getattr(disorder, alias)
            assert np.array_equal(legacy_value, getattr(disorder, canonical))
            with pytest.raises(ValueError, match="read-only"):
                legacy_value[0] ^= 1
        with pytest.warns(DeprecationWarning, match="eta_weight"):
            assert disorder.eta_weight == disorder.epsilon_data_weight
        with pytest.warns(DeprecationWarning, match="delta_weight"):
            assert disorder.delta_weight == disorder.measurement_error_weight
        with pytest.raises(AttributeError):
            disorder.observed_syndrome = np.zeros(model.num_checks, dtype=np.uint8)

    def test_legacy_constructor_aliases_and_conflicts(self, toric2_setup):
        model, _, _ = toric2_setup
        data = np.zeros(model.num_qubits, dtype=np.uint8)
        syndrome = np.zeros(model.num_checks, dtype=np.uint8)
        with pytest.warns(DeprecationWarning) as warnings_seen:
            disorder = DisorderRealization(
                eta=data,
                delta=syndrome,
                observed_syndrome=syndrome,
                eta_weight=0,
                delta_weight=0,
                p=0.1,
                q=0.05,
            )
        assert len(warnings_seen) == 5
        disorder.validate_for_model(model)
        required = {
            "epsilon_data_true": data,
            "measurement_error": syndrome,
            "effective_syndrome": syndrome,
        }
        for old_name, canonical_name, value in (
            ("eta", "epsilon_data_true", data),
            ("delta", "measurement_error", syndrome),
            ("observed_syndrome", "effective_syndrome", syndrome),
            ("eta_weight", "epsilon_data_weight", 0),
            ("delta_weight", "measurement_error_weight", 0),
        ):
            canonical = {**required, canonical_name: value}
            with pytest.raises(TypeError, match="cannot pass both"):
                DisorderRealization(
                    **canonical,
                    **{old_name: value},
                    p=0.1,
                    q=0.05,
                )

    def test_legacy_wiring_aliases_are_read_only_and_conflicts_rejected(
        self, toric2_setup
    ):
        model, _, _ = toric2_setup
        syndrome = np.zeros(model.num_checks, dtype=np.uint8)
        logical_class = np.zeros(model.k, dtype=np.uint8)
        with pytest.warns(DeprecationWarning):
            wiring = EnsembleWiring(
                ensemble="true_posterior",
                sigma_arg=syndrome,
                ell_ref=logical_class,
                K_p=1.0,
                K_q=1.0,
            )
        for alias in ("sigma_arg", "reference_label", "ell_ref"):
            with pytest.warns(DeprecationWarning, match=alias):
                view = getattr(wiring, alias)
            if view.size:
                with pytest.raises(ValueError, match="read-only"):
                    view[0] ^= 1
        with pytest.raises(TypeError, match="cannot pass both"):
            EnsembleWiring(
                ensemble="true_posterior",
                gibbs_syndrome_argument=syndrome,
                sigma_arg=syndrome,
                planted_logical_class=logical_class,
            )
        for alias in ("reference_label", "ell_ref"):
            with pytest.raises(TypeError, match="cannot pass both"):
                EnsembleWiring(
                    ensemble="true_posterior",
                    gibbs_syndrome_argument=syndrome,
                    planted_logical_class=logical_class,
                    **{alias: logical_class},
                )

    def test_absolute_relative_mattis_relation(self, toric2_setup):
        model, frame, obs_set = toric2_setup
        disorder = draw_disorder(model, 0.2, 0.1, np.random.default_rng(95))
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        e = np.arange(model.num_qubits, dtype=np.uint8) & 1
        absolute = absolute_observable_values(obs_set, e)
        relative = relative_observable_values(obs_set, wiring, e)
        signs = relative_observable_values(
            obs_set,
            wiring,
            np.zeros(model.num_qubits, dtype=np.uint8),
        )
        assert np.array_equal(relative, signs * absolute)
        assert np.array_equal(relative**2, absolute**2)
        assert len(frame.fingerprint()) == 64
        assert frame.fingerprint() == build_observable_frame(model).fingerprint()

    def test_planted_mass_is_not_map_success(self):
        stats = posterior_statistics(np.array([0.1, 0.9]), planted_class=0)
        assert stats["posterior_mass_on_planted_class"] == 0.1
        assert stats["map_success_probability"] == 0.9
        assert stats["posterior_purity"] <= stats["map_success_probability"]
        assert stats["map_success_probability"] <= stats[
            "map_success_algebraic_upper_bound"
        ]
        assert stats["map_success_algebraic_lower_bound"] == stats[
            "posterior_purity"
        ]
        assert stats["map_success_estimated_lower_bound"] is None
        assert stats["map_success_estimated_upper_bound"] is None
        assert stats["map_success_bound_kind"] \
            == "exact_posterior_algebraic"
        assert stats["map_success_bound_has_confidence_coverage"] is False
        assert "map_success_lower_bound" not in stats
        assert "map_success_upper_bound" not in stats

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            ([-0.1, 1.1], "nonnegative"),
            ([0.1, 0.8], "sum to 1"),
            ([0.5, 0.5 + 2e-12], "sum to 1"),
            ([np.nan, 1.0], "finite"),
            ([np.inf, 0.0], "finite"),
        ],
    )
    def test_posterior_statistics_rejects_invalid_weights(
        self, weights, message
    ):
        with pytest.raises(ValueError, match=message):
            posterior_statistics(np.asarray(weights, dtype=np.float64))


class TestSampledCharacterEstimators:
    def _sampled_k4(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        model = build_models(
            classical_parity_check_matrix(graph), sectors=("x_error",)
        )[0]
        frame = build_observable_frame(model)
        return build_observable_set(
            frame, full_max_k=2, num_random_u=2, u_rand_seed=101
        )

    def test_basis_and_nonbasis_population_weighting(self):
        obs = self._sampled_k4()
        values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0])
        estimate, _ = sampled_nonzero_character_mean(obs, values)
        N = (1 << 4) - 1
        expected = (1 + 2 + 3 + 4 + (N - 4) * 15.0) / N
        assert estimate == expected
        assert estimate != np.mean(values[4:])

    def test_cross_chain_square_is_unbiased_while_pooled_square_is_not(self):
        rng = np.random.default_rng(102)
        true_mean = 0.3
        samples_per_chain = 50
        probability_plus = (1 + true_mean) / 2
        # Treat independent repetitions as separate characters so the helper
        # vectorizes the Monte Carlo verification.
        counts = rng.binomial(
            samples_per_chain, probability_plus, size=(4, 5000)
        )
        chain_means = 2.0 * counts / samples_per_chain - 1.0
        result = independent_chain_squared_character_estimates(chain_means)
        truth = true_mean**2
        assert np.mean(result["m2_u_pooled_square_raw"]) > truth + 0.002
        assert abs(np.mean(result["m2_u_debiased"]) - truth) < 0.003
        assert np.all(np.isfinite(result["m2_u_debiased_jackknife_se"]))

    def test_delete_one_jackknife_tracks_repeated_sampling_error(self):
        rng = np.random.default_rng(103)
        true_mean = 0.3
        samples_per_chain = 100
        repetitions = 6000
        counts = rng.binomial(
            samples_per_chain,
            (1.0 + true_mean) / 2.0,
            size=(4, repetitions),
        )
        chain_means = 2.0 * counts / samples_per_chain - 1.0
        result = independent_chain_squared_character_estimates(chain_means)
        empirical_se = float(np.std(result["m2_u_debiased"], ddof=1))
        jackknife_rms = float(np.sqrt(np.mean(
            result["m2_u_debiased_jackknife_se"] ** 2
        )))
        assert 0.85 < jackknife_rms / empirical_se < 1.15

    def test_unphysical_debiased_purity_is_retained_without_success_bounds(
        self, toric2_setup
    ):
        _, _, obs = toric2_setup
        result = aggregate_observables(
            obs,
            np.zeros(obs.num_u),
            m2_u_values=np.full(obs.num_u, -0.5),
        )
        assert result["posterior_purity"] < 1.0 / (1 << obs.k)
        assert not result["posterior_purity_within_physical_bounds"]
        assert result["map_success_algebraic_lower_bound"] is None
        assert result["map_success_algebraic_upper_bound"] is None
        assert result["map_success_estimated_lower_bound"] is None
        assert result["map_success_estimated_upper_bound"] is None
        assert result["map_success_bound_kind"] == "unavailable"
        assert result["map_success_bound_has_confidence_coverage"] is False

    def test_physical_u_statistic_uses_estimated_bounds(self, toric2_setup):
        _, _, obs = toric2_setup
        character_means = np.linspace(0.1, 0.3, obs.num_u)
        result = aggregate_observables(
            obs,
            character_means,
            m2_u_values=character_means**2,
        )
        assert result["map_success_algebraic_lower_bound"] is None
        assert result["map_success_algebraic_upper_bound"] is None
        assert result["map_success_estimated_lower_bound"] == result[
            "posterior_purity"
        ]
        assert np.isclose(
            result["map_success_estimated_upper_bound"],
            np.sqrt(result["posterior_purity"]),
        )
        assert result["map_success_bound_kind"] \
            == "sampled_u_statistic_plugin_no_coverage"
        assert result["map_success_bound_has_confidence_coverage"] is False
