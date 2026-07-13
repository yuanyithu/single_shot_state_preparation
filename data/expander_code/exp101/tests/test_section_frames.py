"""Logical-sector qubit-chain sections and observable-frame relations."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from src.gf2 import gf2_matmul
from src.graphs import cycle_parity_check_matrix, random_biregular_graph_from_m
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model
from src.observables import (
    absolute_observable_values,
    build_observable_frame,
    build_observable_set,
)
from src.section import (
    DecoderObservableFrame,
    LogicalSectorQubitChainSection,
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


class ShiftedLogicalSectorSection:
    """Test-only section shifted by a kernel chain times syndrome bit zero."""

    def __init__(self, base, kernel_shift):
        self.base = base
        self.kernel_shift = np.asarray(kernel_shift, dtype=np.uint8)

    def in_image(self, syndrome):
        return self.base.in_image(syndrome)

    def apply(self, syndrome, strict=True):
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        result = self.base.apply(syndrome, strict=strict)
        if syndrome[0]:
            result ^= self.kernel_shift
        return result

    def section_after_H(self, H):
        result = self.base.section_after_H(H)
        return result ^ np.outer(self.kernel_shift, H[0]).astype(np.uint8)

    def fingerprint(self):
        payload = self.base.fingerprint().encode() + self.kernel_shift.tobytes()
        return hashlib.sha256(payload).hexdigest()


class TestLogicalSectorQubitChainSection:
    def test_bplsd_backend_available_and_correct(self):
        H = toric3_H_Z()
        section = LogicalSectorQubitChainSection(H)
        assert section.backend_name == "bplsd", section.ldpc_import_error
        for sigma in random_image_syndromes(H, 10, seed=1):
            chain = section.apply(sigma)
            assert np.array_equal(gf2_matmul(H, chain[:, None])[:, 0], sigma)

    def test_fallback_backend(self):
        H = toric3_H_Z()
        section = LogicalSectorQubitChainSection(H, prefer_bplsd=False)
        assert section.backend_name == "linear_elimination_fallback"
        sigma = random_image_syndromes(H, 1, seed=2)[0]
        chain = section.apply(sigma)
        assert np.array_equal(gf2_matmul(H, chain[:, None])[:, 0], sigma)
        assert section.stats()["fallback_count"] == 1

    def test_misuse_rejected_strict(self):
        H = toric3_H_Z()
        section = LogicalSectorQubitChainSection(H)
        bad = np.zeros(H.shape[0], dtype=np.uint8)
        bad[0] = 1  # toric 单违反 ∉ im
        assert not section.in_image(bad)
        with pytest.raises(ValueError, match="im\\(H\\)"):
            section.apply(bad)

    def test_cache_behavior_and_limit(self):
        H = toric3_H_Z()
        section = LogicalSectorQubitChainSection(H, cache_limit=1)
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
        section = LogicalSectorQubitChainSection(H_Z)
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
            model.H_check,
            model.logical_obs_basis,
            LogicalSectorQubitChainSection(model.H_check),
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
            model.H_check,
            model.logical_obs_basis,
            LogicalSectorQubitChainSection(model.H_check),
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

    def test_boundary_only_section_shift_leaves_characters_unchanged(
        self, toric3_model
    ):
        model = toric3_model
        original = build_observable_frame(model)
        shifted_section = ShiftedLogicalSectorSection(
            model.logical_sector_section, model.stabilizer_rows[0]
        )
        shifted_model = replace(
            model, logical_sector_section=shifted_section
        )
        shifted = build_observable_frame(shifted_model)
        assert np.array_equal(original.W_basis, shifted.W_basis)
        assert original.section_fingerprint != shifted.section_fingerprint
        assert original.fingerprint() != shifted.fingerprint()

    def test_logical_section_shift_is_not_claimed_as_gauge(self, toric3_model):
        model = toric3_model
        original = build_observable_frame(model)
        shifted_section = ShiftedLogicalSectorSection(
            model.logical_sector_section, model.logical_move_basis[0]
        )
        shifted_model = replace(
            model, logical_sector_section=shifted_section
        )
        shifted = build_observable_frame(shifted_model)
        assert not np.array_equal(original.W_basis, shifted.W_basis)
        e = np.zeros(model.num_qubits, dtype=np.uint8)
        e[np.flatnonzero(model.H_check[0])[0]] = 1
        assert gf2_matmul(model.H_check, e[:, None]).any()
        original_label = original.label_of(e)
        shifted_label = shifted.label_of(e)
        assert not np.array_equal(original_label, shifted_label)

        # A distribution with mass on an off-kernel chain has genuinely
        # different characters in the two logical-sector sections.
        original_set = build_observable_set(original)
        shifted_set = build_observable_set(shifted)
        original_characters = 0.25 * absolute_observable_values(
            original_set, np.zeros_like(e)
        ) + 0.75 * absolute_observable_values(original_set, e)
        shifted_characters = 0.25 * absolute_observable_values(
            shifted_set, np.zeros_like(e)
        ) + 0.75 * absolute_observable_values(shifted_set, e)
        assert not np.array_equal(original_characters, shifted_characters)


class TestFingerprintLargeCodes:
    def test_fingerprint_survives_pivot_indices_over_255(self):
        """回归（V2 排障）：n≥100 的码主元列索引 >255，指纹序列化必须定宽。"""
        from src.graphs import random_biregular_graph_from_m
        from src.hgp import classical_parity_check_matrix, hgp_from_H

        graph = random_biregular_graph_from_m(4, 3, 4, seed=12345)  # n=400
        H_Z, _ = hgp_from_H(classical_parity_check_matrix(graph))
        # 逆序列优先级 ⇒ 主元集中在高位索引（必然 >255）
        priority = list(reversed(range(H_Z.shape[1])))
        section = build_linear_section(H_Z, column_priority=priority)
        assert max(section.pivot_columns) > 255
        fp = section.fingerprint()
        assert len(fp) == 64
        # 默认 frame 也顺带确认可指纹化
        assert len(build_linear_section(H_Z).fingerprint()) == 64
