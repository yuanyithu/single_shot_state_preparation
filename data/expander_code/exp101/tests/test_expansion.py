"""G1.4 单元测试：精确 vertex-expansion 验证器（spec §10 G）。

含测试内独立 oracle（直接 set 运算逐子集复核，不共享位掩码实现）。
"""

from fractions import Fraction
from itertools import combinations

import pytest

from src.expansion import verify_vertex_expansion
from src.graphs import (
    BiregularBipartiteGraph,
    complete_bipartite_graph,
    random_biregular_graph_from_m,
)


def make_graph(n_A, n_B, d_A, d_B, edges):
    A_to_B = [set() for _ in range(n_A)]
    B_to_A = [set() for _ in range(n_B)]
    for a, b in edges:
        A_to_B[a].add(b)
        B_to_A[b].add(a)
    graph = BiregularBipartiteGraph(
        n_A=n_A, n_B=n_B, d_A=d_A, d_B=d_B,
        A_to_B=A_to_B, B_to_A=B_to_A,
        seed=-1, rng_description="manual", construction_attempts=1,
    )
    graph.validate()
    return graph


def matching_graph(m):
    """完美匹配：d_A=d_B=1，|Γ(S)|=|S|，比值恒 1。"""
    return make_graph(m, m, 1, 1, [(i, i) for i in range(m)])


def bipartite_cycle_graph(m):
    """C_{2m}：A_i ~ {B_i, B_{(i+1)%m}}，d_A=d_B=2。"""
    edges = []
    for i in range(m):
        edges.append((i, i))
        edges.append((i, (i + 1) % m))
    return make_graph(m, m, 2, 2, edges)


def oracle_check(graph, gamma, delta, side):
    """独立 oracle：set 并集逐子集验证；返回 (ok, worst_ratio)。"""
    if side == "left":
        adjacency, degree, num_vertices = graph.A_to_B, graph.d_A, graph.n_A
    else:
        adjacency, degree, num_vertices = graph.B_to_A, graph.d_B, graph.n_B
    ok = True
    worst = None
    for size in range(1, num_vertices + 1):
        if Fraction(size) > gamma * num_vertices:
            break
        for subset in combinations(range(num_vertices), size):
            neighborhood = set()
            for vertex in subset:
                neighborhood |= adjacency[vertex]
            ratio = Fraction(len(neighborhood), degree * size)
            worst = ratio if worst is None else min(worst, ratio)
            if Fraction(len(neighborhood)) < (1 - delta) * degree * size:
                ok = False
    return ok, worst


class TestManualGraphs:
    def test_matching_passes_everything(self):
        graph = matching_graph(6)
        result = verify_vertex_expansion(graph, Fraction(1), Fraction(0))
        assert result.passed
        assert result.worst_left_ratio == 1 and result.worst_right_ratio == 1
        assert result.max_subset_size_left == 6
        assert not result.vacuous_left

    def test_complete_bipartite_boundary_case(self):
        """K_{4,3} 左侧：|Γ(S)|=3 恒定；|S|=2 时需 (1−δ)·3·2 ≤ 3 ⇔ δ ≥ 1/2。"""
        graph = complete_bipartite_graph(4, 3)
        # γ=1/2 ⇒ |S|≤2；δ=1/2 恰好在边界（≥ 判据）⇒ 通过
        result_boundary = verify_vertex_expansion(
            graph, Fraction(1, 2), Fraction(1, 2), sides="left"
        )
        assert result_boundary.passed
        assert result_boundary.worst_left_ratio == Fraction(1, 2)
        # δ=49/100 略紧 ⇒ 失败，witness 是大小 2 的子集
        result_fail = verify_vertex_expansion(
            graph, Fraction(1, 2), Fraction(49, 100), sides="left"
        )
        assert not result_fail.passed
        assert result_fail.failing_side == "left"
        assert len(result_fail.failing_subset) == 2
        assert result_fail.failing_neighborhood_size == 3
        assert result_fail.required_neighborhood_size == Fraction(51, 100) * 3 * 2
        # witness 自洽：|Γ(witness)| 与报告一致且确实违例
        neighborhood = set()
        for a in result_fail.failing_subset:
            neighborhood |= graph.A_to_B[a]
        assert len(neighborhood) == result_fail.failing_neighborhood_size
        assert Fraction(len(neighborhood)) < result_fail.required_neighborhood_size

    def test_cycle_adjacent_pair_case(self):
        """C_12（m=6）：相邻 A 对 |Γ|=3，ratio=3/4。δ=1/4 边界过、δ=1/5 失败。"""
        graph = bipartite_cycle_graph(6)
        result_pass = verify_vertex_expansion(
            graph, Fraction(1, 3), Fraction(1, 4), sides="left"
        )
        assert result_pass.passed
        assert result_pass.worst_left_ratio == Fraction(3, 4)
        assert result_pass.max_subset_size_left == 2
        result_fail = verify_vertex_expansion(
            graph, Fraction(1, 3), Fraction(1, 5), sides="left"
        )
        assert not result_fail.passed
        assert result_fail.failing_neighborhood_size == 3
        assert result_fail.required_neighborhood_size == Fraction(4, 5) * 2 * 2


class TestVacuousAndSides:
    def test_vacuous_when_gamma_too_small(self):
        """(3,4) m=2：n_A=8, γ=1/10 ⇒ γn_A<1 ⇒ 空真（spec 示例参数的实情）。"""
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12345)
        result = verify_vertex_expansion(graph, "1/10", "1/16")
        assert result.passed
        assert result.vacuous_left and result.vacuous_right
        assert result.worst_left_ratio is None
        assert result.num_subsets_checked_left == 0
        assert any("vacuous" in note for note in result.notes)

    def test_sides_selection(self):
        graph = complete_bipartite_graph(4, 3)
        left_only = verify_vertex_expansion(graph, "1/2", "1/2", sides="left")
        assert left_only.checked_left and not left_only.checked_right
        assert left_only.worst_right_ratio is None
        right_only = verify_vertex_expansion(graph, "1/2", "1/2", sides="right")
        assert right_only.checked_right and not right_only.checked_left

    def test_string_and_int_inputs(self):
        graph = matching_graph(4)
        assert verify_vertex_expansion(graph, "1/2", "0").passed
        assert verify_vertex_expansion(graph, 1, 0).passed

    def test_float_rejected(self):
        graph = matching_graph(4)
        with pytest.raises(TypeError):
            verify_vertex_expansion(graph, 0.1, "1/16")
        with pytest.raises(TypeError):
            verify_vertex_expansion(graph, "1/10", 0.0625)

    def test_out_of_range_rejected(self):
        graph = matching_graph(4)
        with pytest.raises(ValueError):
            verify_vertex_expansion(graph, "3/2", "1/16")
        with pytest.raises(ValueError):
            verify_vertex_expansion(graph, "1/2", "-1/16")

    def test_subset_budget_guard(self):
        graph = random_biregular_graph_from_m(3, 3, 4, seed=12345)  # n_A=12
        with pytest.raises(ValueError):
            verify_vertex_expansion(graph, Fraction(1), Fraction(0), max_subsets=10)


class TestAgainstOracle:
    @pytest.mark.parametrize("seed", [12345, 777, 99])
    def test_random_34_graphs_match_oracle(self, seed):
        graph = random_biregular_graph_from_m(3, 3, 4, seed=seed)  # n_A=12, n_B=9
        for gamma, delta in [
            (Fraction(1, 4), Fraction(1, 16)),
            (Fraction(1, 4), Fraction(1, 3)),
            (Fraction(1, 3), Fraction(1, 2)),
        ]:
            result = verify_vertex_expansion(graph, gamma, delta)
            ok_left, worst_left = oracle_check(graph, gamma, delta, "left")
            ok_right, worst_right = oracle_check(graph, gamma, delta, "right")
            assert result.passed == (ok_left and ok_right)
            assert result.worst_left_ratio == worst_left
            assert result.worst_right_ratio == worst_right
