"""G3.1 单元测试：计数表枚举（对独立逐点枚举、结构恒等、主项目互证）。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.enumerate_exact as enumerate_exact_module

from src.enumerate_exact import (
    NUMBA_AVAILABLE,
    build_coset_table,
    build_full_table,
    evaluate_table,
    exact_reference,
)
from src.graphs import (
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model, draw_disorder, wire_ensemble
from src.observables import build_observable_frame, build_observable_set
from tests.util_enum import enum_class_weights, enum_m_u

MAIN_SRC = Path(__file__).resolve().parents[4] / "src"


def build_setup(classical):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    return model, frame


def make_wiring(model, frame, p, q, seed, ensemble="true_posterior"):
    disorder = draw_disorder(model, p, q, np.random.default_rng(seed))
    return disorder, wire_ensemble(model, disorder, ensemble, frame)


PQ_GRID = [(0.05, 0.02), (0.12, 0.08), (0.25, 0.15), (0.4, 0.3), (0.25, 0.49)]


class TestAgainstPerPointEnum:
    @pytest.mark.parametrize("classical_builder", [
        lambda: repetition_parity_check_matrix(2),
        lambda: cycle_parity_check_matrix(2),
    ])
    def test_full_table_matches_util_enum_many_pq(self, classical_builder):
        """一表多 (p,q)：同一张表在 5 组 (p,q) 下 vs 独立逐点枚举，机器精度。"""
        model, frame = build_setup(classical_builder())
        obs_set = build_observable_set(frame)
        disorder, wiring0 = make_wiring(model, frame, 0.12, 0.08, seed=1)
        table = build_full_table(
            model, frame, wiring0.gibbs_syndrome_argument
        )
        from src.model import EnsembleWiring, coupling_from_probability

        for p, q in PQ_GRID:
            wiring = EnsembleWiring(
                ensemble=wiring0.ensemble,
                gibbs_syndrome_argument=wiring0.gibbs_syndrome_argument,
                planted_logical_class=wiring0.planted_logical_class,
                K_p=coupling_from_probability(p),
                K_q=coupling_from_probability(q), q_zero=False,
            )
            expected_m, _ = enum_m_u(model, obs_set, wiring)
            planted_bits = sum(
                1 << b
                for b, v in enumerate(wiring.planted_logical_class)
                if v
            )
            got = evaluate_table(table, wiring.K_p, wiring.K_q,
                                 planted_class_bits=planted_bits)
            # m_u：util_enum 的观测集含组合 u；basis 位于前 k 个位置
            basis_m = expected_m[obs_set.basis_positions.tolist()]
            assert np.allclose(
                got["m_u_basis_relative"], basis_m, atol=1e-10
            )
            expected_weights = enum_class_weights(model, frame, wiring)
            assert np.allclose(got["weights_absolute"], expected_weights,
                               atol=1e-10)

    def test_q_zero_coset_matches_util_enum(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        obs_set = build_observable_set(frame)
        disorder, wiring = make_wiring(model, frame, 0.15, 0.0, seed=3)
        expected_m, _ = enum_m_u(model, obs_set, wiring)
        result = exact_reference(model, frame, wiring)
        basis_m = expected_m[obs_set.basis_positions.tolist()]
        assert np.allclose(
            result["m_u_basis_relative"], basis_m, atol=1e-10
        )


class TestStructuralIdentities:
    def test_coset_equals_full_ws0_slice(self):
        """For an image Gibbs argument, the coset table equals the w_s=0 slice."""
        model, frame = build_setup(cycle_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.15, 0.0, seed=5)
        full = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        coset = build_coset_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        assert np.array_equal(full.table[:, 0, :], coset.table)

    def test_python_and_numba_identical(self):
        assert NUMBA_AVAILABLE
        model, frame = build_setup(repetition_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.12, 0.08, seed=7)
        t_nb = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        t_py = build_full_table(model, frame, wiring.gibbs_syndrome_argument,
                                force_python=True)
        assert np.array_equal(t_nb.table, t_py.table)
        _, wiring0 = make_wiring(model, frame, 0.15, 0.0, seed=9)
        c_nb = build_coset_table(
            model, frame, wiring0.gibbs_syndrome_argument
        )
        c_py = build_coset_table(
            model, frame, wiring0.gibbs_syndrome_argument,
                                 force_python=True)
        assert np.array_equal(c_nb.table, c_py.table)

    def test_mu_per_label_matches_direct(self):
        """μ_ℓ = ⟨w_p⟩_ℓ（TI 曲线交叉钩子）对直算。"""
        model, frame = build_setup(cycle_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.2, 0.1, seed=11)
        table = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        got = evaluate_table(table, wiring.K_p, wiring.K_q)
        # 直算：逐状态加权平均 |v|（按 label 分组）
        from src.gf2 import gf2_matmul

        n = model.num_qubits
        sums = np.zeros(1 << model.k)
        wsum = np.zeros(1 << model.k)
        for value in range(1 << n):
            v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
            syndrome = gf2_matmul(model.H_check, v[:, None])[:, 0] \
                ^ wiring.gibbs_syndrome_argument
            weight = np.exp(-wiring.K_p * v.sum()
                            - wiring.K_q * syndrome.sum())
            bits = 0
            for b, val in enumerate(frame.label_of(v)):
                if val:
                    bits |= 1 << b
            sums[bits] += weight * v.sum()
            wsum[bits] += weight
        expected_mu = sums / wsum
        assert np.allclose(got["mu_per_label"], expected_mu, atol=1e-10)

    def test_complete_absolute_and_relative_character_oracle(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.17, 0.09, seed=121)
        result = exact_reference(model, frame, wiring)
        N = (1 << model.k) - 1
        assert np.array_equal(result["u_bitmasks"], np.arange(1, N + 1))
        assert result["characters_absolute"].shape == (N,)
        assert result["characters_relative"].shape == (N,)
        for index, u in enumerate(result["u_bitmasks"]):
            expected_absolute = sum(
                probability
                * (1 - 2 * ((int(u) & logical_class).bit_count() & 1))
                for logical_class, probability in enumerate(
                    result["weights_absolute"]
                )
            )
            assert np.isclose(
                result["characters_absolute"][index], expected_absolute
            )
        planted_bits = sum(
            1 << bit
            for bit, value in enumerate(wiring.planted_logical_class)
            if value
        )
        signs = np.array([
            1 - 2 * ((int(u) & planted_bits).bit_count() & 1)
            for u in result["u_bitmasks"]
        ])
        assert np.allclose(
            result["characters_relative"],
            signs * result["characters_absolute"],
        )
        assert np.isclose(
            result["q_top"], np.mean(result["characters_absolute"] ** 2)
        )


class TestGuards:
    def test_large_n_rejected(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12349)
        model, frame = build_setup(classical_parity_check_matrix(graph))
        _, wiring = make_wiring(model, frame, 0.1, 0.05, seed=13)
        with pytest.raises(ValueError, match="guard"):
            build_full_table(model, frame, wiring.gibbs_syndrome_argument)

    def test_q_zero_coset_rejects_large_logical_dimension_before_allocation(self):
        oversized = SimpleNamespace(
            num_qubits=1,
            num_checks=1,
            k=enumerate_exact_module.MAX_K + 1,
        )
        with pytest.raises(ValueError, match="coset guard: k="):
            build_coset_table(
                oversized,
                frame=None,
                gibbs_syndrome_argument=np.zeros(1, dtype=np.uint8),
            )

    def test_q_zero_coset_applies_table_byte_guard(self, monkeypatch):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.1, 0.0, seed=14)
        monkeypatch.setattr(enumerate_exact_module, "MAX_TABLE_BYTES", 1)
        with pytest.raises(ValueError, match="coset guard: table too large"):
            build_coset_table(
                model, frame, wiring.gibbs_syndrome_argument
            )

    def test_k43_within_guards_smoke(self):
        """K_{4,3}（n=25, k=13）：守卫内最大实例的构建烟测 + 总数守恒。"""
        model, frame = build_setup(np.ones((3, 4), dtype=np.uint8))
        _, wiring = make_wiring(model, frame, 0.1, 0.05, seed=15)
        table = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        assert table.total_states == 1 << 25
        got = evaluate_table(table, wiring.K_p, wiring.K_q)
        assert np.isclose(got["weights_absolute"].sum(), 1.0)
        assert got["q_top"] is not None


class TestLegacyExactSchema:
    def test_legacy_values_are_exposed_only_under_formal_names(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        _, wiring = make_wiring(
            model,
            frame,
            p=0.13,
            q=0.09,
            seed=16,
            ensemble="legacy_delta_only",
        )
        result = exact_reference(model, frame, wiring)
        for name in (
            "weights_absolute",
            "weights_relative",
            "characters_absolute",
            "characters_relative",
            "m_u_absolute",
            "m_u_relative",
            "q_top",
            "q_top_absolute",
            "q_top_relative",
            "posterior_purity",
            "posterior_mass_on_planted_class",
            "map_success_probability",
            "map_success_algebraic_lower_bound",
            "map_success_algebraic_upper_bound",
            "map_success_estimated_lower_bound",
            "map_success_estimated_upper_bound",
            "posterior_purity_within_physical_bounds",
        ):
            assert result[name] is None
        assert result["map_success_bound_kind"] == "unavailable"
        assert result["map_success_bound_has_confidence_coverage"] is False
        assert result["weights_are_exact_sector_posterior"] is False
        assert result["formal_weights_are_exact_sector_posterior"] is True
        assert "map_success_lower_bound" not in result
        assert "map_success_upper_bound" not in result
        weights = result["formal_sector_weights_absolute"]
        assert np.isclose(weights.sum(), 1.0)
        assert np.isclose(result["formal_sector_purity"], np.sum(weights**2))
        assert result["largest_sector_mass"] == np.max(weights)
        assert np.isclose(
            result["formal_q_top"],
            np.mean(result["formal_sector_characters_absolute"] ** 2),
        )

    def test_exact_table_legacy_argument_alias_is_read_only(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        _, wiring = make_wiring(model, frame, 0.13, 0.09, seed=18)
        table = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )
        with pytest.warns(DeprecationWarning, match="sigma_arg"):
            alias = table.sigma_arg
        with pytest.raises(ValueError, match="read-only"):
            alias[0] ^= 1


class TestMainProjectCrossCheck:
    """与主项目 exact_enumeration 只读互证（两套独立实现）。"""

    def _main_module(self):
        if str(MAIN_SRC) not in sys.path:
            sys.path.append(str(MAIN_SRC))
        from exact_enumeration import compute_exact_logical_observable_means
        from preprocessing import build_logical_observable_masks

        return compute_exact_logical_observable_means, \
            build_logical_observable_masks

    def test_log_partition_relation_machine_precision(self):
        """The legacy-only table reproduces the historical 3D repository logZ.

        This is explicitly a regression check, not evidence for the paper's
        true-posterior semantics.
        """
        compute_main, build_masks = self._main_module()
        model, frame = build_setup(cycle_parity_check_matrix(2))
        p, q = 0.13, 0.09
        disorder, wiring = make_wiring(model, frame, p, q, seed=17,
                                       ensemble="legacy_delta_only")
        # The historical API still calls these observed syndrome and eta.
        H_main = model.H_check.astype(bool)
        from src.logicals import logical_pauli_operators as _lp

        masks = build_masks(
            H_main, model.logical_obs_basis.astype(bool), None
        )
        main_result = compute_main(
            parity_check_matrix=H_main,
            observed_syndrome_bits=disorder.effective_syndrome.astype(bool),
            disorder_data_error_bits=disorder.epsilon_data_true.astype(bool),
            syndrome_error_probability=q,
            data_error_probability=p,
            logical_observable_masks=masks,
        )
        table = build_full_table(
            model, frame, wiring.gibbs_syndrome_argument
        )  # legacy Gibbs argument = measurement error
        got = evaluate_table(table, wiring.K_p, wiring.K_q)
        n, m_c = model.num_qubits, model.num_checks
        expected_log_Z = (
            got["log_Z"] + n * np.log(1 - p) + m_c * np.log(1 - q)
        )
        assert np.isclose(main_result["log_partition_function"],
                          expected_log_Z, atol=1e-9)

    def test_m_u_matches_main_in_decoder_frame(self):
        """Legacy-only characters match the historical API in its decoder frame."""
        compute_main, build_masks = self._main_module()
        model, frame = build_setup(cycle_parity_check_matrix(2))
        p, q = 0.13, 0.09
        disorder, _ = make_wiring(model, frame, p, q, seed=19)
        H_main = model.H_check.astype(bool)
        masks = build_masks(H_main, model.logical_obs_basis.astype(bool), None)
        main_result = compute_main(
            parity_check_matrix=H_main,
            observed_syndrome_bits=disorder.effective_syndrome.astype(bool),
            disorder_data_error_bits=disorder.epsilon_data_true.astype(bool),
            syndrome_error_probability=q,
            data_error_probability=p,
            logical_observable_masks=masks,
        )
        # 本地小 n 直接枚举（decoder frame、legacy 权重）
        from src.section import (
            DecoderObservableFrame,
            LogicalSectorQubitChainSection,
        )
        from src.gf2 import gf2_matmul
        from src.model import coupling_from_probability

        dec_frame = DecoderObservableFrame(
            model.H_check, model.logical_obs_basis,
            LogicalSectorQubitChainSection(model.H_check),
        )
        K_p = coupling_from_probability(p)
        K_q = coupling_from_probability(q)
        n = model.num_qubits
        num_masks = masks.shape[0]
        total = np.zeros(num_masks)
        norm = 0.0
        label_eta = dec_frame.label_of(disorder.epsilon_data_true)
        masks_u8 = masks.astype(np.uint8)
        obs_basis = model.logical_obs_basis
        for value in range(1 << n):
            v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
            data_w = int(
                np.count_nonzero(v ^ disorder.epsilon_data_true)
            )
            syn = gf2_matmul(model.H_check, v[:, None])[:, 0] \
                ^ disorder.effective_syndrome
            weight = np.exp(-K_p * data_w - K_q * int(syn.sum()))
            label_v = dec_frame.label_of(v)
            rel = label_v ^ label_eta
            # masks 行 = z 基的全部非零组合（顺序与主项目一致：bit 序）
            values = np.empty(num_masks)
            for mask_index in range(num_masks):
                combo = mask_index + 1
                parity = 0
                for b in range(model.k):
                    if (combo >> b) & 1:
                        parity ^= int(rel[b])
                values[mask_index] = 1.0 - 2.0 * parity
            total += weight * values
            norm += weight
        local_m = total / norm
        assert np.allclose(local_m, main_result["m_u_values"], atol=1e-9)
