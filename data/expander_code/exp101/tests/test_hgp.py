"""G1.3 单元测试：hypergraph product（spec §10 B, C + 索引约定锁死 + k 公式）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

from src.gf2 import gf2_rank
from src.graphs import (
    complete_bipartite_graph,
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import (
    classical_parity_check_matrix,
    hgp_expected_parameters,
    hgp_from_H,
    quantum_expander_parity_checks_from_graph,
    verify_css_commutation,
)

MAIN_SRC = Path("/Users/jarvis/Desktop/sync/project D/src")


class TestClassicalMatrix:
    """spec §10 B。"""

    @pytest.mark.parametrize("m,seed", [(1, 12345), (2, 12345), (3, 777)])
    def test_shape_and_weights_34(self, m, seed):
        graph = random_biregular_graph_from_m(m, 3, 4, seed)
        H = classical_parity_check_matrix(graph)
        assert H.shape == (graph.n_B, graph.n_A)
        assert np.all(H.sum(axis=1) == graph.d_B)  # 行重 = d_B
        assert np.all(H.sum(axis=0) == graph.d_A)  # 列重 = d_A

    def test_entry_convention(self):
        graph = complete_bipartite_graph(4, 3)
        H = classical_parity_check_matrix(graph)
        assert H.shape == (3, 4)
        assert np.all(H == 1)


class TestQuantumMatrices:
    """spec §10 C + 显式索引锁死。"""

    @pytest.mark.parametrize(
        "m,d_A,d_B,seed",
        [(1, 3, 4, 12345), (2, 3, 4, 12345), (2, 2, 2, 5), (3, 2, 3, 99)],
    )
    def test_shapes_weights_commutation(self, m, d_A, d_B, seed):
        graph = random_biregular_graph_from_m(m, d_A, d_B, seed)
        H_Z, H_X = quantum_expander_parity_checks_from_graph(graph)
        n_A, n_B = graph.n_A, graph.n_B
        expected_shape = (n_A * n_B, n_A**2 + n_B**2)
        assert H_X.shape == expected_shape and H_Z.shape == expected_shape
        assert verify_css_commutation(H_X, H_Z)
        # 行重 = d_A + d_B
        assert np.all(H_X.sum(axis=1) == d_A + d_B)
        assert np.all(H_Z.sum(axis=1) == d_A + d_B)
        # 列重：A×A 块各 d_A，B×B 块各 d_B（强于 spec 的 2·max 上界）
        for M in (H_X, H_Z):
            column_weights = M.sum(axis=0)
            assert np.all(column_weights[: n_A**2] == d_A)
            assert np.all(column_weights[n_A**2 :] == d_B)
            assert np.all(column_weights <= 2 * max(d_A, d_B))

    def test_explicit_index_convention(self):
        """手推 2×3 例锁死 Kronecker 约定（防静默转置）。

        H = [[1,0,1],[0,1,1]]，n_B=2，n_A=3。
        """
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        H_Z, H_X = hgp_from_H(H)
        n_A, n_B = 3, 2
        # --- X-check (a=1, b=0)，行 = 1·2+0 = 2 ---
        #   A×A: (1, a2) with H[0,a2]=1 → a2∈{0,2} → 列 3, 5
        #   B×B: (b1, 0) with H[b1,1]=1 → b1=1 → 列 9 + 1·2+0 = 11
        expected_x_row = {1 * n_A + 0, 1 * n_A + 2, n_A**2 + 1 * n_B + 0}
        assert set(np.flatnonzero(H_X[2]).tolist()) == expected_x_row
        # --- Z-check (b=1, a=0)，行 = 1·3+0 = 3 ---
        #   A×A: (a1, 0) with H[1,a1]=1 → a1∈{1,2} → 列 1·3+0=3, 2·3+0=6
        #   B×B: (1, b2) with H[b2,0]=1 → b2=0 → 列 9 + 1·2+0 = 11
        expected_z_row = {1 * n_A + 0, 2 * n_A + 0, n_A**2 + 1 * n_B + 0}
        assert set(np.flatnonzero(H_Z[3]).tolist()) == expected_z_row

    def test_graph_vs_matrix_paths_agree(self):
        graph = complete_bipartite_graph(4, 3)
        from_graph = quantum_expander_parity_checks_from_graph(graph)
        from_matrix = hgp_from_H(np.ones((3, 4), dtype=np.uint8))
        assert np.array_equal(from_graph[0], from_matrix[0])
        assert np.array_equal(from_graph[1], from_matrix[1])

    def test_commutation_detects_corruption(self):
        H_Z, H_X = hgp_from_H(cycle_parity_check_matrix(3))
        assert verify_css_commutation(H_X, H_Z)
        corrupted = H_Z.copy()
        corrupted[0, 0] ^= 1
        assert not verify_css_commutation(H_X, corrupted)


class TestKFormulaOnKnownCodes:
    """k = (n_A−r)² + (n_B−r)²；rank H_X = rank H_Z = n_A n_B − (n_A−r)(n_B−r)。"""

    @pytest.mark.parametrize(
        "classical,expected_n,expected_k",
        [
            (cycle_parity_check_matrix(2), 8, 2),        # 2D toric m=2
            (cycle_parity_check_matrix(3), 18, 2),       # 2D toric m=3
            (repetition_parity_check_matrix(2), 5, 1),   # 最小 surface 型
            (repetition_parity_check_matrix(3), 13, 1),  # [[13,1,3]]
            (np.ones((3, 4), dtype=np.uint8), 25, 13),   # K_{4,3}: 3²+2²
        ],
    )
    def test_known_parameters(self, classical, expected_n, expected_k):
        H_Z, H_X = hgp_from_H(classical)
        n = H_X.shape[1]
        rank_HX = gf2_rank(H_X)
        rank_HZ = gf2_rank(H_Z)
        k = n - rank_HX - rank_HZ
        assert n == expected_n
        assert k == expected_k
        expected = hgp_expected_parameters(classical, gf2_rank(classical))
        assert expected["n"] == n
        assert expected["k"] == k
        assert rank_HX == expected["rank_HX"]
        assert rank_HZ == expected["rank_HZ"]

    def test_official_family_m2_rank_and_k(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        H = classical_parity_check_matrix(graph)
        H_Z, H_X = quantum_expander_parity_checks_from_graph(graph)
        r = gf2_rank(H)
        expected = hgp_expected_parameters(H, r)
        n = H_X.shape[1]
        k = n - gf2_rank(H_X) - gf2_rank(H_Z)
        assert n == 100
        assert k == expected["k"]
        # 满秩时 k=m²=4；若该 seed 非满秩，此断言指引 families 注册表换 seed
        if r == graph.n_B:
            assert k == 4

    def test_irregular_H_supported(self):
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)  # 2×4 非正则
        H_Z, H_X = hgp_from_H(H)
        assert verify_css_commutation(H_X, H_Z)
        n = H_X.shape[1]
        k = n - gf2_rank(H_X) - gf2_rank(H_Z)
        expected = hgp_expected_parameters(H, gf2_rank(H))
        assert (n, k) == (20, expected["k"]) and k == 4


class TestCrossCheckMainProjectToric:
    """cycle-HGP 与主项目 build_2d_toric_code 的不变量互证（只读 import）。"""

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_rank_invariants_match(self, L):
        if str(MAIN_SRC) not in sys.path:
            sys.path.append(str(MAIN_SRC))
        from build_toric_code_examples import build_2d_toric_code  # noqa: PLC0415

        main_H_Z, main_logicals = build_2d_toric_code(L)
        H_Z, H_X = hgp_from_H(cycle_parity_check_matrix(L))
        # 同一码族的不变量：n、rank H_Z、k
        assert H_Z.shape[1] == main_H_Z.shape[1] == 2 * L * L
        assert gf2_rank(H_Z) == gf2_rank(main_H_Z.astype(np.uint8)) == L * L - 1
        k = H_Z.shape[1] - gf2_rank(H_X) - gf2_rank(H_Z)
        assert k == main_logicals.shape[0] == 2
