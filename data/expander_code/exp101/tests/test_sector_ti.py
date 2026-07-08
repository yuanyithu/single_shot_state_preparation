"""G2.8 单元测试：sector-TI 引擎对精确枚举（权重/w0/q_top/m_u/pairwise 机制）。"""

import numpy as np
import pytest

from src.graphs import (
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model, draw_disorder, wire_ensemble
from src.observables import build_observable_frame
from src.sector_ti import (
    SectorTiConfig,
    build_sector_preserving_proposals,
    run_sector_ti,
    sector_representative,
)
from tests.util_enum import enum_class_weights


def build_setup(classical):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    return model, frame


def make_wiring(model, frame, p, q, seed, ensemble="true_posterior",
                force_nontrivial_label=False):
    rng = np.random.default_rng(seed)
    while True:
        disorder = draw_disorder(model, p, q, rng)
        wiring = wire_ensemble(model, disorder, ensemble, frame)
        if not force_nontrivial_label or wiring.reference_label.any():
            return wiring


FAST_CONFIG = SectorTiConfig(
    num_kp_grid_points=13, num_burn_in_sweeps=80, num_measurements=240,
    block_count=8, num_bootstrap=120,
)


class TestProposals:
    @pytest.mark.parametrize("classical_builder", [
        lambda: cycle_parity_check_matrix(2),
        lambda: repetition_parity_check_matrix(2),
        lambda: classical_parity_check_matrix(
            random_biregular_graph_from_m(2, 3, 4, seed=12349)),
    ])
    def test_all_proposals_preserve_label(self, classical_builder):
        model, frame = build_setup(classical_builder())
        proposals = build_sector_preserving_proposals(model, frame)
        rng = np.random.default_rng(4)
        for _ in range(5):
            v = (rng.random(model.num_qubits) < 0.5).astype(np.uint8)
            base = frame.label_of(v)
            for support in proposals["supports"]:
                flipped = v.copy()
                flipped[support] ^= 1
                assert np.array_equal(frame.label_of(flipped), base)

    def test_proposal_inventory_nonempty(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        proposals = build_sector_preserving_proposals(model, frame)
        assert proposals["num_stab"] == model.stabilizer_rows.shape[0]
        assert (proposals["num_single"] + proposals["num_pairs"]) > 0


class TestSectorRepresentatives:
    def test_labels_correct_qpos_and_qzero(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring_pos = make_wiring(model, frame, 0.12, 0.08, seed=1)
        for label in range(1 << model.k):
            v = sector_representative(model, wiring_pos, label)
            bits = 0
            for b, val in enumerate(frame.label_of(v)):
                if val:
                    bits |= 1 << b
            assert bits == label
        wiring_zero = make_wiring(model, frame, 0.15, 0.0, seed=2)
        v0 = sector_representative(model, wiring_zero, 0b10)
        from src.gf2 import gf2_matmul

        syndrome = gf2_matmul(model.H_check, v0[:, None])[:, 0]
        assert np.array_equal(syndrome, wiring_zero.sigma_arg)  # 约束满足


class TestFullTierAgainstEnum:
    @pytest.mark.parametrize("ensemble", ["true_posterior", "repo_compat"])
    def test_surface5_weights_and_qtop(self, ensemble):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=3,
                             ensemble=ensemble)
        exact_weights = enum_class_weights(model, frame, wiring)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=11)
        assert result["tier"] == "full"
        # 权重逐扇区 z 检验（ΔF bootstrap stderr 传播到权重的粗略界：用 ΔF z）
        exact_delta = -np.log(exact_weights / exact_weights[0])
        z = (result["delta_f"] - exact_delta) / np.maximum(
            result["delta_f_stderr"], 1e-3
        )
        assert np.max(np.abs(z)) < 5.0, (result["delta_f"], exact_delta, z)
        # q_top 与 w0
        k = model.k
        exact_qtop = ((1 << k) * np.sum(exact_weights ** 2) - 1) / ((1 << k) - 1)
        assert abs(result["q_top"] - exact_qtop) < max(
            5 * result["q_top_stderr"], 0.02
        )
        ell_ref = result["ell_ref"]
        assert np.isclose(
            result["w0"], exact_weights[ell_ref], atol=0.03
        )

    def test_toric2_true_ensemble_nontrivial_reference(self):
        """真类 ℓ_ref ≠ 0 的重排正确性（true 系综核心语义）。"""
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.10, seed=5,
                             force_nontrivial_label=True)
        assert wiring.reference_label.any()
        exact_weights = enum_class_weights(model, frame, wiring)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=13)
        ell_ref = result["ell_ref"]
        assert ell_ref != 0
        # 相对权重 = 绝对权重按 ℓ⊕ℓ_ref 重排
        for t in range(1 << model.k):
            assert np.isclose(
                result["weights_relative"][t],
                result["weights_absolute"][result["labels"].index(
                    t ^ ell_ref)],
            )
        # m_u 与枚举一致（相对真类）
        exact_relative = np.array(
            [exact_weights[t ^ ell_ref] for t in range(1 << model.k)]
        )
        exact_m = np.array([
            sum(exact_relative[t] * (1 - 2 * ((t >> u) & 1))
                for t in range(1 << model.k))
            for u in range(model.k)
        ])
        assert np.max(np.abs(result["m_u_basis"] - exact_m)) < 0.06

    def test_q_zero_coset_ti(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.0, seed=7)
        exact_weights = enum_class_weights(model, frame, wiring)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=15)
        exact_delta = -np.log(exact_weights / exact_weights[0])
        z = (result["delta_f"] - exact_delta) / np.maximum(
            result["delta_f_stderr"], 1e-3
        )
        assert np.max(np.abs(z)) < 5.0


class TestPairwiseMechanism:
    def test_pairwise_matches_exact_gaps(self):
        """pairwise 档机制验证：ΔF̃_u 对精确 −log(P(ℓ_ref⊕e_u)/P(ℓ_ref))。"""
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.10, seed=9,
                             force_nontrivial_label=True)
        exact_weights = enum_class_weights(model, frame, wiring)
        config = SectorTiConfig(
            num_kp_grid_points=13, num_burn_in_sweeps=80,
            num_measurements=240, block_count=8, num_bootstrap=120,
            full_max_k=1,   # 强制 pairwise（k=2 > 1）
        )
        result = run_sector_ti(model, frame, wiring, config, seed=17)
        assert result["tier"] == "pairwise"
        ell_ref = result["ell_ref"]
        exact_gaps = np.array([
            -np.log(exact_weights[ell_ref ^ (1 << u)]
                    / exact_weights[ell_ref])
            for u in range(model.k)
        ])
        z = (result["delta_f_per_u"] - exact_gaps) / np.maximum(
            result["delta_f_stderr"][1:], 1e-3
        )
        assert np.max(np.abs(z)) < 5.0
        assert np.allclose(
            result["m_u_pairwise"], np.tanh(result["delta_f_per_u"] / 2)
        )


class TestStructure:
    def test_anchor_delta_zero_and_flags(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=19)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=21)
        assert result["delta_f"][0] == 0.0
        assert "flags" in result and "grid_tv" in result
        assert result["proposal_summary"]["num_stab"] > 0

    def test_reproducible(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=23)
        r1 = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=25)
        r2 = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=25)
        assert np.array_equal(r1["delta_f"], r2["delta_f"])

    def test_k_zero_rejected(self):
        H_Z, H_X = hgp_from_H(np.array([[1]], dtype=np.uint8))
        logicals = logical_pauli_operators(H_X, H_Z)
        model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
        frame = build_observable_frame(model)
        wiring = make_wiring(model, frame, 0.1, 0.05, seed=27)
        with pytest.raises(ValueError, match="k >= 1"):
            run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=29)
