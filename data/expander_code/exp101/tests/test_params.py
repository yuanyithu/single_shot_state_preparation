"""G1.6 单元测试：精确 [[n,k,d]]（spec §10 F）。

oracle：n ≤ 18 时用测试内独立的全空间 2^n 枚举（不共享 kernel/Gray 逻辑）。
"""

import numpy as np
import pytest

from src.gf2 import gf2_in_rowspace, gf2_matmul, gf2_rowspace_basis
from src.graphs import (
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.params import (
    classical_code_distance,
    code_parameters,
    hgp_classical_side_distances,
)


def oracle_distance_full_space(H_check, H_stab, num_qubits):
    """min |v|: H_check v=0, v ∉ row(H_stab)；全空间枚举，n≤18。"""
    stab_basis = gf2_rowspace_basis(H_stab)
    best = None
    for value in range(1, 1 << num_qubits):
        vector = np.array(
            [(value >> j) & 1 for j in range(num_qubits)], dtype=np.uint8
        )
        if best is not None and int(vector.sum()) >= best:
            continue
        if gf2_matmul(H_check, vector[None, :].T).any():
            continue
        if gf2_in_rowspace(vector, stab_basis):
            continue
        best = int(vector.sum())
    return best


KNOWN_CODES = [
    # (name, classical H, [[n,k,d]] 期望)
    ("toric_m2", cycle_parity_check_matrix(2), (8, 2, 2)),
    ("toric_m3", cycle_parity_check_matrix(3), (18, 2, 3)),
    ("surface_m2", repetition_parity_check_matrix(2), (5, 1, 2)),
    ("surface_m3", repetition_parity_check_matrix(3), (13, 1, 3)),
    ("K43", np.ones((3, 4), dtype=np.uint8), (25, 13, 2)),
]


class TestKnownCodeParameters:
    @pytest.mark.parametrize("name,classical,nkd", KNOWN_CODES)
    def test_exact_nkd(self, name, classical, nkd):
        H_Z, H_X = hgp_from_H(classical)
        params = code_parameters(H_X, H_Z, compute_distance=True)
        assert (params.n, params.k, params.d) == nkd
        # 最小逻辑算符自洽：权重=距离、在 ker、非 stabilizer
        assert int(params.min_logical_X.sum()) == params.d_X
        assert int(params.min_logical_Z.sum()) == params.d_Z
        assert not gf2_matmul(H_Z, params.min_logical_X[None, :].T).any()
        assert not gf2_matmul(H_X, params.min_logical_Z[None, :].T).any()
        assert not gf2_in_rowspace(params.min_logical_X, gf2_rowspace_basis(H_X))
        assert not gf2_in_rowspace(params.min_logical_Z, gf2_rowspace_basis(H_Z))

    @pytest.mark.parametrize(
        "name,classical,nkd",
        [c for c in KNOWN_CODES if c[2][0] <= 18],
    )
    def test_distance_matches_full_space_oracle(self, name, classical, nkd):
        H_Z, H_X = hgp_from_H(classical)
        params = code_parameters(H_X, H_Z, compute_distance=True)
        assert params.d_X == oracle_distance_full_space(H_Z, H_X, params.n)
        assert params.d_Z == oracle_distance_full_space(H_X, H_Z, params.n)

    def test_toric_m4_distance_4(self):
        """[[32,2,4]]：全空间 oracle 太大，用已知构造事实断言。"""
        H_Z, H_X = hgp_from_H(cycle_parity_check_matrix(4))
        params = code_parameters(H_X, H_Z, compute_distance=True)
        assert (params.n, params.k, params.d_X, params.d_Z, params.d) == (
            32, 2, 4, 4, 4,
        )

    @pytest.mark.parametrize("name,classical,nkd", KNOWN_CODES)
    def test_k_agrees_with_logical_pairs(self, name, classical, nkd):
        H_Z, H_X = hgp_from_H(classical)
        params = code_parameters(H_X, H_Z, compute_distance=False)
        logicals = logical_pauli_operators(H_X, H_Z)
        assert params.k == logicals.k
        assert not params.distance_computed and params.d is None

    @pytest.mark.parametrize("name,classical,nkd", KNOWN_CODES)
    def test_d_lower_bounds_quotient_representatives(self, name, classical, nkd):
        """d ≤ 商基代表的最小权重（代表未必最短，但不能更短）。"""
        H_Z, H_X = hgp_from_H(classical)
        params = code_parameters(H_X, H_Z, compute_distance=True)
        logicals = logical_pauli_operators(H_X, H_Z)
        assert params.d_X <= int(logicals.logical_X.sum(axis=1).min())
        assert params.d_Z <= int(logicals.logical_Z.sum(axis=1).min())


class TestClassicalSideTheoremCrossCheck:
    """HGP 量子 d 与经典侧 d 的关系：小码上精确交叉验证。"""

    @pytest.mark.parametrize("name,classical,nkd", KNOWN_CODES)
    def test_theorem_min_matches_bruteforce(self, name, classical, nkd):
        sides = hgp_classical_side_distances(classical)
        H_Z, H_X = hgp_from_H(classical)
        params = code_parameters(H_X, H_Z, compute_distance=True)
        assert sides["theorem_min"] == params.d

    def test_classical_distances_known_values(self):
        assert classical_code_distance(cycle_parity_check_matrix(5)) == 5
        assert classical_code_distance(repetition_parity_check_matrix(4)) == 4
        assert classical_code_distance(np.ones((3, 4), dtype=np.uint8)) == 2
        # 满秩转置侧 k=0 → None
        assert classical_code_distance(repetition_parity_check_matrix(4).T) is None

    def test_official_family_m2_sides(self):
        """(3,4) m=2：量子暴力不可行（守卫），经典侧精确可得并记录。"""
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        classical = classical_parity_check_matrix(graph)
        sides = hgp_classical_side_distances(classical)
        assert sides["d_ker_HT"] is None  # 满秩 ⇒ 转置侧无码字
        assert isinstance(sides["d_ker_H"], int) and 1 <= sides["d_ker_H"] <= 8
        assert sides["theorem_min"] == sides["d_ker_H"]


class TestGuardsAndEdgeCases:
    def test_k0_returns_none_distance(self):
        H_Z, H_X = hgp_from_H(np.array([[1]], dtype=np.uint8))
        params = code_parameters(H_X, H_Z, compute_distance=True)
        assert params.k == 0
        assert params.d is None and params.d_X is None and params.d_Z is None
        assert params.distance_computed

    def test_kernel_dim_guard_raises_for_m2_family(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        H_Z, H_X = hgp_from_H(classical_parity_check_matrix(graph))
        with pytest.raises(ValueError, match="max_kernel_dim"):
            code_parameters(H_X, H_Z, compute_distance=True)
        # 不算距离则一切正常
        params = code_parameters(H_X, H_Z, compute_distance=False)
        assert (params.n, params.k) == (100, 4)

    def test_column_mismatch_rejected(self):
        with pytest.raises(ValueError):
            code_parameters(
                np.zeros((1, 3), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8)
            )
