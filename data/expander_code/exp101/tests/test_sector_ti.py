"""Full-sector TI and diagnostic-only basis-gap tests."""

import numpy as np
import pytest

from src.graphs import (
    complete_bipartite_graph,
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import (
    EnsembleWiring,
    assemble_sector_model,
    draw_disorder,
    wire_ensemble,
)
from src.observables import build_observable_frame
from src.sector_ti import (
    FULL_SECTOR_TI_MAX_K,
    SectorTiConfig,
    _bootstrap_delta_f,
    basis_sector_free_energy_gap_diagnostics,
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
        if not force_nontrivial_label or wiring.planted_logical_class.any():
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
        assert np.array_equal(
            syndrome, wiring_zero.gibbs_syndrome_argument
        )


class TestFullTierAgainstEnum:
    @pytest.mark.parametrize(
        "ensemble", ["true_posterior", "legacy_delta_only"]
    )
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
        # Paper q_top exists only for the true posterior.  Legacy retains the
        # same numerical regression data under explicit formal-only names.
        k = model.k
        exact_qtop = ((1 << k) * np.sum(exact_weights ** 2) - 1) / ((1 << k) - 1)
        planted = result["planted_logical_class_bitmask"]
        if ensemble == "true_posterior":
            assert abs(result["q_top"] - exact_qtop) < max(
                5 * result["q_top_stderr"], 0.02
            )
            assert np.isclose(
                result["posterior_mass_on_planted_class"],
                exact_weights[planted], atol=0.03,
            )
        else:
            assert result["weights_absolute"] is None
            assert result["characters_absolute"] is None
            assert result["m_u_absolute"] is None
            assert result["q_top"] is None
            assert abs(result["formal_q_top"] - exact_qtop) < max(
                5 * result["formal_q_top_stderr"], 0.02
            )
            assert np.allclose(
                result["formal_sector_weights_absolute"],
                exact_weights,
                atol=0.03,
            )
            assert result["posterior_mass_on_planted_class"] is None
            assert result["map_success_probability"] is None

    def test_toric2_true_ensemble_nontrivial_reference(self):
        """真类 ℓ_ref ≠ 0 的重排正确性（true 系综核心语义）。"""
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.10, seed=5,
                             force_nontrivial_label=True)
        assert wiring.planted_logical_class.any()
        exact_weights = enum_class_weights(model, frame, wiring)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=13)
        planted = result["planted_logical_class_bitmask"]
        assert planted != 0
        # 相对权重 = 绝对权重按 ℓ⊕ℓ_ref 重排
        for t in range(1 << model.k):
            assert np.isclose(
                result["weights_relative"][t],
                result["weights_absolute"][result["labels"].index(
                    t ^ planted)],
            )
        # m_u 与枚举一致（相对真类）
        exact_relative = np.array(
            [exact_weights[t ^ planted] for t in range(1 << model.k)]
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


class TestBasisGapDiagnostics:
    def test_diagnostics_match_exact_gaps_and_expose_no_purity(self):
        model, frame = build_setup(cycle_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.15, 0.10, seed=9,
                             force_nontrivial_label=True)
        exact_weights = enum_class_weights(model, frame, wiring)
        config = SectorTiConfig(
            num_kp_grid_points=13, num_burn_in_sweeps=80,
            num_measurements=240, block_count=8, num_bootstrap=120,
        )
        result = basis_sector_free_energy_gap_diagnostics(
            model, frame, wiring, config, seed=17
        )
        planted = result["planted_logical_class_bitmask"]
        exact_gaps = np.array([
            -np.log(exact_weights[planted ^ (1 << u)]
                    / exact_weights[planted])
            for u in range(model.k)
        ])
        z = (result["basis_sector_free_energy_gaps"] - exact_gaps) / np.maximum(
            result["basis_sector_free_energy_gap_stderr"], 1e-3
        )
        assert np.max(np.abs(z)) < 5.0
        assert not any("q_top" in key or "m_u" in key for key in result)

    def test_large_k_full_ti_rejected_before_sampling(self):
        classical = classical_parity_check_matrix(
            complete_bipartite_graph(4, 3)
        )
        model, frame = build_setup(classical)
        assert model.k > FULL_SECTOR_TI_MAX_K
        wiring = make_wiring(model, frame, 0.1, 0.05, seed=10)
        with pytest.raises(ValueError, match="k<=10"):
            run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=17)


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


class TestBootstrapIndependence:
    def test_independent_sector_resampling_avoids_false_zero_gap_error(self):
        kp_grid = np.asarray([0.0, 1.0])
        per_block = np.asarray([0.0, 1.0, 2.0, 3.0])
        one_chain = np.stack([per_block, per_block])
        # Identically ordered block values are a sharp regression fixture:
        # shared bootstrap indices cancel every fluctuation in the gap.
        block_arrays = np.stack([one_chain, one_chain])
        seed = 123
        independent = _bootstrap_delta_f(
            block_arrays, kp_grid, num_bootstrap=200, seed=seed
        )
        assert np.std(independent[:, 1], ddof=1) > 0.1

        rng = np.random.default_rng(seed ^ 0x5EED)
        shared = []
        for _ in range(200):
            indices = rng.integers(0, 4, size=4)
            mu = block_arrays[:, :, indices].mean(axis=2)
            integrals = np.trapezoid(mu, x=kp_grid, axis=1)
            shared.append(integrals - integrals[0])
        assert np.std(np.asarray(shared)[:, 1], ddof=1) == 0.0


class TestAnalyticEndpoints:
    def test_p_half_is_exact_uniform_under_full_ti(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.5, 0.08, seed=31)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=33)
        assert result["endpoint_mode"] == "p_half_uniform"
        assert np.allclose(result["weights_absolute"], [0.5, 0.5])
        assert np.allclose(result["characters_absolute"], [0.0])
        assert result["q_top"] == 0.0
        assert result["q_top_stderr"] == 0.0
        assert result["grid_tv"] == 0.0

    def test_p_zero_is_exact_absolute_class_zero_delta(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.0, 0.08, seed=35)
        result = run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=37)
        assert result["endpoint_mode"] == "p_zero_delta"
        assert np.allclose(result["weights_absolute"], [1.0, 0.0])
        assert np.allclose(result["characters_absolute"], [1.0])
        assert result["q_top"] == 1.0
        assert result["q_top_stderr"] == 0.0

    def test_p_zero_q_zero_nonzero_syndrome_has_no_support(self):
        model, frame = build_setup(repetition_parity_check_matrix(2))
        nonzero_syndrome = model.H_check[:, 0].copy()
        assert nonzero_syndrome.any()
        wiring = EnsembleWiring(
            ensemble="true_posterior",
            gibbs_syndrome_argument=nonzero_syndrome,
            planted_logical_class=np.zeros(model.k, dtype=np.uint8),
            K_p=np.inf,
            K_q=np.inf,
            q_zero=True,
        )
        with pytest.raises(ValueError, match="zero support"):
            run_sector_ti(model, frame, wiring, FAST_CONFIG, seed=39)
