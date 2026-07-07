"""G1.2 单元测试：二部图构造（spec §10 A + 确定性构造器）。"""

import numpy as np
import pytest

from src.gf2 import gf2_rank
from src.graphs import (
    BiregularBipartiteGraph,
    complete_bipartite_graph,
    cycle_parity_check_matrix,
    family_sizes_from_m,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)


FAMILY_CASES = [
    # (m, d_A, d_B, seed)
    (1, 3, 4, 12345),
    (2, 3, 4, 12345),
    (3, 3, 4, 12345),
    (2, 3, 4, 777),
    (2, 2, 2, 5),
    (4, 2, 3, 99),
]


class TestFamilySizes:
    def test_34_family_table(self):
        for m in (1, 2, 3, 6):
            n_A, n_B = family_sizes_from_m(m, 3, 4)
            assert (n_A, n_B) == (4 * m, 3 * m)
            assert n_A * 3 == n_B * 4

    def test_gcd_reduction(self):
        # d_A=2,d_B=2: g=2,a=b=1 → n_A=n_B=m
        assert family_sizes_from_m(5, 2, 2) == (5, 5)
        # d_A=4,d_B=6: g=2,a=2,b=3 → n_A=3m,n_B=2m
        assert family_sizes_from_m(2, 4, 6) == (6, 4)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            family_sizes_from_m(0, 3, 4)
        with pytest.raises(ValueError):
            family_sizes_from_m(2, 0, 4)


class TestRandomBiregularGraph:
    @pytest.mark.parametrize("m,d_A,d_B,seed", FAMILY_CASES)
    def test_sizes_degrees_simple_consistent(self, m, d_A, d_B, seed):
        graph = random_biregular_graph_from_m(m, d_A, d_B, seed)
        n_A, n_B = family_sizes_from_m(m, d_A, d_B)
        assert (graph.n_A, graph.n_B) == (n_A, n_B)
        graph.check_degrees()
        graph.check_simple()
        graph.check_consistency()
        # 边数守恒 + 无重边（set 语义下逐点重建校验）
        edges = graph.edge_set()
        assert len(edges) == n_A * d_A
        # 元数据
        assert graph.seed == seed
        assert graph.construction_attempts >= 1
        assert "random.Random" in graph.rng_description
        assert graph.extra_metadata["m"] == m

    @pytest.mark.parametrize("m,d_A,d_B,seed", FAMILY_CASES)
    def test_same_seed_reproduces_exactly(self, m, d_A, d_B, seed):
        graph_1 = random_biregular_graph_from_m(m, d_A, d_B, seed)
        graph_2 = random_biregular_graph_from_m(m, d_A, d_B, seed)
        assert graph_1.edge_set() == graph_2.edge_set()
        assert graph_1.construction_attempts == graph_2.construction_attempts

    def test_different_seeds_usually_differ(self):
        edge_sets = {
            random_biregular_graph_from_m(3, 3, 4, seed).edge_set()
            for seed in range(20)
        }
        # 20 个 seed 至少产生 15 种不同图（“usually different”）
        assert len(edge_sets) >= 15

    def test_m1_34_is_complete_bipartite(self):
        graph = random_biregular_graph_from_m(1, 3, 4, seed=12345)
        assert graph.n_A == 4 and graph.n_B == 3
        for neighbors in graph.A_to_B:
            assert neighbors == set(range(3))

    def test_impossible_parameters_raise(self):
        # (d_A,d_B)=(4,2), m=1: g=2,a=2,b=1 → n_A=1,n_B=2，d_A=4 > n_B=2 不可能
        with pytest.raises(ValueError):
            random_biregular_graph_from_m(1, 4, 2, seed=0)
        # 对称分支：d_B=4 > n_A=2
        with pytest.raises(ValueError):
            random_biregular_graph_from_m(1, 2, 4, seed=0)

    def test_max_attempts_exhaustion_raises(self):
        with pytest.raises(RuntimeError):
            random_biregular_graph_from_m(1, 3, 4, seed=0, max_attempts=0)

    def test_high_degree_needs_more_attempts(self):
        # (5,6): 接受率 ≈ e^{-10} ≈ 4.5e-5 ⇒ 默认 1e4 attempts 大概率不够（文档化事实），
        # 加大 max_attempts 后可构造，且同 seed 复现（含 attempts 数）。
        graph_1 = random_biregular_graph_from_m(2, 5, 6, seed=12345, max_attempts=500000)
        graph_2 = random_biregular_graph_from_m(2, 5, 6, seed=12345, max_attempts=500000)
        graph_1.validate()
        assert graph_1.construction_attempts > 100  # 远超低度数情形
        assert graph_1.edge_set() == graph_2.edge_set()
        assert graph_1.construction_attempts == graph_2.construction_attempts


class TestValidationMethodsDetectCorruption:
    def _base_graph(self):
        return random_biregular_graph_from_m(2, 3, 4, seed=12345)

    def test_degree_corruption_detected(self):
        graph = self._base_graph()
        victim = next(iter(graph.A_to_B[0]))
        graph.A_to_B[0] = graph.A_to_B[0] - {victim}
        with pytest.raises(ValueError):
            graph.check_degrees()

    def test_consistency_corruption_detected(self):
        graph = self._base_graph()
        b_old = next(iter(graph.A_to_B[0]))
        b_new = next(b for b in range(graph.n_B) if b not in graph.A_to_B[0])
        graph.A_to_B[0] = (graph.A_to_B[0] - {b_old}) | {b_new}
        with pytest.raises(ValueError):
            graph.check_consistency()

    def test_range_corruption_detected(self):
        graph = self._base_graph()
        graph.A_to_B[0] = (graph.A_to_B[0] - {next(iter(graph.A_to_B[0]))}) | {999}
        with pytest.raises(ValueError):
            graph.check_simple()


class TestDeterministicConstructors:
    def test_complete_bipartite(self):
        graph = complete_bipartite_graph(4, 3)
        graph.validate()
        assert graph.d_A == 3 and graph.d_B == 4
        assert len(graph.edge_set()) == 12

    def test_cycle_matrix(self):
        for m in (2, 3, 5, 7):
            matrix = cycle_parity_check_matrix(m)
            assert matrix.shape == (m, m)
            assert np.all(matrix.sum(axis=0) == 2) and np.all(matrix.sum(axis=1) == 2)
            assert gf2_rank(matrix) == m - 1  # 单圈 ⇒ rank m−1，k(H)=k(Hᵀ)=1
        with pytest.raises(ValueError):
            cycle_parity_check_matrix(1)

    def test_repetition_matrix(self):
        for m in (2, 3, 5):
            matrix = repetition_parity_check_matrix(m)
            assert matrix.shape == (m - 1, m)
            assert gf2_rank(matrix) == m - 1  # 满秩 ⇒ k(H)=1, k(Hᵀ)=0
        with pytest.raises(ValueError):
            repetition_parity_check_matrix(1)
