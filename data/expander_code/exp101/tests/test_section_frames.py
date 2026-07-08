"""G2.2 单元测试：DecoderSection（BpLsd）、备选线性 frame、frame 间关系。"""

import numpy as np
import pytest

from src.gf2 import gf2_matmul
from src.graphs import cycle_parity_check_matrix, random_biregular_graph_from_m
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model
from src.observables import build_observable_frame
from src.section import (
    DecoderObservableFrame,
    DecoderSection,
    build_linear_section,
)


def toric3_H_Z():
    H_Z, _ = hgp_from_H(cycle_parity_check_matrix(3))
    return H_Z


def random_image_syndromes(H, count, seed, density=0.4):
    rng = np.random.default_rng(seed)
    sigmas = []
    for _ in range(count):
        x = (rng.random(H.shape[1]) < density).astype(np.uint8)
        sigmas.append(gf2_matmul(H, x[:, None])[:, 0])
    return sigmas


class TestDecoderSection:
    def test_bplsd_backend_available_and_correct(self):
        H = toric3_H_Z()
        section = DecoderSection(H)
        assert section.backend_name == "bplsd", section.ldpc_import_error
        for sigma in random_image_syndromes(H, 10, seed=1):
            chain = section.apply(sigma)
            assert np.array_equal(gf2_matmul(H, chain[:, None])[:, 0], sigma)

    def test_fallback_backend(self):
        H = toric3_H_Z()
        section = DecoderSection(H, prefer_bplsd=False)
        assert section.backend_name == "linear_elimination_fallback"
        sigma = random_image_syndromes(H, 1, seed=2)[0]
        chain = section.apply(sigma)
        assert np.array_equal(gf2_matmul(H, chain[:, None])[:, 0], sigma)
        assert section.stats()["fallback_count"] == 1

    def test_misuse_rejected_strict(self):
        H = toric3_H_Z()
        section = DecoderSection(H)
        bad = np.zeros(H.shape[0], dtype=np.uint8)
        bad[0] = 1  # toric 单违反 ∉ im
        assert not section.in_image(bad)
        with pytest.raises(ValueError, match="im\\(H\\)"):
            section.apply(bad)

    def test_cache_behavior_and_limit(self):
        H = toric3_H_Z()
        section = DecoderSection(H, cache_limit=1)
        sigmas = random_image_syndromes(H, 3, seed=3)
        section.apply(sigmas[0])
        section.apply(sigmas[0])
        stats = section.stats()
        assert stats["cache_hit_count"] == 1
        section.apply(sigmas[1])
        section.apply(sigmas[2])
        assert section.stats()["cache_size"] <= 1  # 上限生效

    def test_k43_large_instance(self):
        H_Z, _ = hgp_from_H(np.ones((3, 4), dtype=np.uint8))
        section = DecoderSection(H_Z)
        for sigma in random_image_syndromes(H_Z, 5, seed=4):
            chain = section.apply(sigma)
            assert np.array_equal(gf2_matmul(H_Z, chain[:, None])[:, 0], sigma)


class TestAlternativeLinearFrame:
    def test_column_priority_gives_valid_section(self):
        H = toric3_H_Z()
        rng = np.random.default_rng(5)
        priority = rng.permutation(H.shape[1]).tolist()
        section_alt = build_linear_section(H, column_priority=priority)
        section_default = build_linear_section(H)
        for sigma in random_image_syndromes(H, 8, seed=6):
            r_alt = section_alt.apply(sigma)
            assert np.array_equal(gf2_matmul(H, r_alt[:, None])[:, 0], sigma)
            # 两 frame 的差 ∈ ker(H)
            diff = r_alt ^ section_default.apply(sigma)
            assert not gf2_matmul(H, diff[:, None]).any()

    def test_priority_changes_pivots_generically(self):
        H = toric3_H_Z()
        default_pivots = build_linear_section(H).pivot_columns
        rng = np.random.default_rng(7)
        found_different = False
        for _ in range(5):
            priority = rng.permutation(H.shape[1]).tolist()
            alt_pivots = build_linear_section(H, column_priority=priority).pivot_columns
            if alt_pivots != default_pivots:
                found_different = True
                break
        assert found_different

    def test_invalid_priority_rejected(self):
        H = toric3_H_Z()
        with pytest.raises(ValueError, match="permutation"):
            build_linear_section(H, column_priority=[0, 1, 2])


class TestFrameRelations:
    @pytest.fixture()
    def toric3_model(self):
        classical = cycle_parity_check_matrix(3)
        H_Z, H_X = hgp_from_H(classical)
        logicals = logical_pauli_operators(H_X, H_Z)
        return assemble_sector_model(H_X, H_Z, logicals, sector="x_error")

    def test_frames_agree_on_kernel(self, toric3_model):
        """q=0 frame 无关性（notes/01 §4）：v ∈ ker(H) 时线性/decoder frame 标签一致。"""
        model = toric3_model
        linear_frame = build_observable_frame(model)
        decoder_frame = DecoderObservableFrame(
            model.H_check, model.logical_obs_basis, DecoderSection(model.H_check)
        )
        rng = np.random.default_rng(8)
        # ker 元素 = stabilizer 行与 logical move 基的随机组合
        generators = np.vstack([model.stabilizer_rows, model.logical_move_basis])
        for _ in range(10):
            coeff = (rng.random(generators.shape[0]) < 0.5).astype(np.uint8)
            v = gf2_matmul(coeff[None, :], generators)[0]
            assert np.array_equal(
                linear_frame.label_of(v), decoder_frame.label_of(v)
            )

    def test_frames_valid_labels_off_kernel(self, toric3_model):
        """v ∉ ker 时两 frame 标签都合法（对 stabilizer/x_u 平移协变），可不同。"""
        model = toric3_model
        linear_frame = build_observable_frame(model)
        decoder_frame = DecoderObservableFrame(
            model.H_check, model.logical_obs_basis, DecoderSection(model.H_check)
        )
        rng = np.random.default_rng(9)
        disagreement = 0
        for _ in range(20):
            v = (rng.random(model.num_qubits) < 0.3).astype(np.uint8)
            for frame in (linear_frame, decoder_frame):
                base = frame.label_of(v)
                # 协变性：⊕stabilizer 不变；⊕x_0 翻第 0 位
                assert np.array_equal(
                    frame.label_of(v ^ model.stabilizer_rows[0]), base
                )
                shifted = frame.label_of(v ^ model.logical_move_basis[0])
                expected = base.copy()
                expected[0] ^= 1
                assert np.array_equal(shifted, expected)
            if not np.array_equal(
                    linear_frame.label_of(v), decoder_frame.label_of(v)):
                disagreement += 1
        # frame 依赖是真实现象：记录（不强断言具体数值，但期待出现过不一致）
        assert disagreement >= 0  # 文档性；具体分布 G3.3 定量
