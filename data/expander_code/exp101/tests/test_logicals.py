"""G1.5 单元测试：逻辑 Pauli 算符（spec §10 E）。"""

import numpy as np
import pytest

from src.gf2 import gf2_in_rowspace, gf2_matmul, gf2_rank, gf2_rowspace_basis
from src.graphs import (
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import (
    classical_parity_check_matrix,
    hgp_from_H,
)
from src.logicals import logical_pauli_operators, verify_logical_pauli_result


def known_codes():
    """(名称, classical H, 期望 k)。"""
    cases = [
        ("toric_m2", cycle_parity_check_matrix(2), 2),
        ("toric_m3", cycle_parity_check_matrix(3), 2),
        ("surface_m2", repetition_parity_check_matrix(2), 1),
        ("surface_m3", repetition_parity_check_matrix(3), 1),
        ("K43", np.ones((3, 4), dtype=np.uint8), 13),
        ("irregular_2x4", np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8), 4),
    ]
    graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)  # 官方家族 m=2（满秩）
    cases.append(("expander_34_m2", classical_parity_check_matrix(graph), 4))
    return cases


@pytest.mark.parametrize("name,classical,expected_k", known_codes())
class TestLogicalPauliOnKnownCodes:
    def test_full_checklist(self, name, classical, expected_k):
        H_Z, H_X = hgp_from_H(classical)
        result = logical_pauli_operators(H_X, H_Z)
        assert result.k == expected_k
        assert result.logical_X.shape == (expected_k, H_X.shape[1])
        assert result.logical_Z.shape == (expected_k, H_X.shape[1])
        # spec §7 全部检查（strict 抛错即失败）
        checks = verify_logical_pauli_result(H_X, H_Z, result, strict=True)
        assert all(checks.values())
        # 代表元全非零
        if expected_k:
            assert np.all(result.logical_X.sum(axis=1) > 0)
            assert np.all(result.logical_Z.sum(axis=1) > 0)
        # 归一前配对可逆（能走到归一化本身即证明），归一后为单位阵
        assert np.array_equal(
            result.pairing_matrix, np.eye(expected_k, dtype=np.uint8)
        )

    def test_normalized_X_stays_logical(self, name, classical, expected_k):
        """归一化（M^{-1} 混类）不得把代表推回 stabilizer 或推出 ker。"""
        H_Z, H_X = hgp_from_H(classical)
        result = logical_pauli_operators(H_X, H_Z)
        if expected_k == 0:
            return
        assert not gf2_matmul(H_Z, result.logical_X.T).any()
        rowspace_H_X = gf2_rowspace_basis(H_X)
        for x in result.logical_X:
            assert not gf2_in_rowspace(x, rowspace_H_X)
        # 任意非零组合仍非 stabilizer（商基独立性的强断言，抽查 8 个随机组合）
        rng = np.random.default_rng(7)
        for _ in range(8):
            coefficients = rng.integers(0, 2, size=expected_k).astype(np.uint8)
            if not coefficients.any():
                coefficients[0] = 1
            combo = gf2_matmul(coefficients[None, :], result.logical_X)[0]
            assert not gf2_in_rowspace(combo, rowspace_H_X)


class TestEdgeAndErrorPaths:
    def test_k_zero_trivial_code(self):
        """H=[1] 的 HGP：n=2，k=0，优雅返回空结果。"""
        H_Z, H_X = hgp_from_H(np.array([[1]], dtype=np.uint8))
        result = logical_pauli_operators(H_X, H_Z)
        assert result.k == 0
        assert result.logical_X.shape == (0, 2)
        checks = verify_logical_pauli_result(H_X, H_Z, result, strict=True)
        assert all(checks.values())

    def test_non_css_pair_rejected(self):
        H_X = np.array([[1, 1, 0]], dtype=np.uint8)
        H_Z = np.array([[1, 0, 0]], dtype=np.uint8)  # H_X H_Z^T = 1 ≠ 0
        with pytest.raises(ValueError, match="CSS"):
            logical_pauli_operators(H_X, H_Z)

    def test_column_mismatch_rejected(self):
        with pytest.raises(ValueError):
            logical_pauli_operators(
                np.zeros((1, 3), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8)
            )

    def test_verify_detects_corruption(self):
        H_Z, H_X = hgp_from_H(cycle_parity_check_matrix(3))
        result = logical_pauli_operators(H_X, H_Z)
        # 把一个 logical X 换成 stabilizer 行 → 检查必须报错
        corrupted_X = result.logical_X.copy()
        corrupted_X[0] = gf2_rowspace_basis(H_X)[0]
        from dataclasses import replace

        corrupted = replace(result, logical_X=corrupted_X)
        with pytest.raises(AssertionError, match="checks failed"):
            verify_logical_pauli_result(H_X, H_Z, corrupted, strict=True)
        loose = verify_logical_pauli_result(H_X, H_Z, corrupted, strict=False)
        assert not loose["logical_X_not_stabilizer"] or not loose["pairing_is_identity"]


class TestPairingStructure:
    def test_k43_large_k_pairing(self):
        """k=13 的大 k 情形：配对求逆与归一全程可行。"""
        H_Z, H_X = hgp_from_H(np.ones((3, 4), dtype=np.uint8))
        result = logical_pauli_operators(H_X, H_Z)
        assert result.k == 13
        assert gf2_rank(result.pre_normalization_pairing) == 13
        assert np.array_equal(result.pairing_matrix, np.eye(13, dtype=np.uint8))

    def test_deterministic_output(self):
        H_Z, H_X = hgp_from_H(cycle_parity_check_matrix(3))
        result_1 = logical_pauli_operators(H_X, H_Z)
        result_2 = logical_pauli_operators(H_X, H_Z)
        assert np.array_equal(result_1.logical_X, result_2.logical_X)
        assert np.array_equal(result_1.logical_Z, result_2.logical_Z)
