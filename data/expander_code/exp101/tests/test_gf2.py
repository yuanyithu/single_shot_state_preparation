"""G1.1 单元测试：GF(2) 工具包（spec §10 D + 扩展性质测试）。

oracle：本文件内独立实现的 python-int bitmask 高斯消元 rank，与被测代码
无共享逻辑，用于交叉验证。
"""

import numpy as np
import pytest

from src.gf2 import (
    as_gf2_matrix,
    gf2_extend_basis,
    gf2_in_rowspace,
    gf2_inverse,
    gf2_matmul,
    gf2_nullspace,
    gf2_quotient_basis,
    gf2_rank,
    gf2_row_echelon,
    gf2_rowspace_basis,
    gf2_solve,
)


def oracle_rank(matrix):
    """独立 oracle：行的 bitmask 消元。"""
    matrix = np.asarray(matrix).astype(np.uint8) % 2
    rows = [int("".join(str(b) for b in row), 2) if row.size else 0 for row in matrix]
    rank = 0
    for column_bit in range(matrix.shape[1] - 1, -1, -1) if matrix.size else []:
        mask = 1 << column_bit
        pivot = next((i for i in range(rank, len(rows)) if rows[i] & mask), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] & mask):
                rows[i] ^= rows[rank]
        rank += 1
    return rank


RANDOM_SHAPES = [(1, 1), (3, 7), (7, 3), (8, 8), (12, 20), (20, 12), (5, 5)]


def random_matrix(rng, shape, density=0.5):
    return (rng.random(shape) < density).astype(np.uint8)


class TestRankAndEchelon:
    def test_fixed_cases(self):
        assert gf2_rank(np.eye(4, dtype=np.uint8)) == 4
        assert gf2_rank(np.zeros((3, 5), dtype=np.uint8)) == 0
        assert gf2_rank([[1, 1], [1, 1]]) == 1
        assert gf2_rank(np.ones((3, 4), dtype=np.uint8)) == 1

    def test_empty_shapes(self):
        assert gf2_rank(np.zeros((0, 5), dtype=np.uint8)) == 0
        assert gf2_rank(np.zeros((5, 0), dtype=np.uint8)) == 0
        assert gf2_nullspace(np.zeros((0, 5), dtype=np.uint8)).shape == (5, 5)
        assert gf2_nullspace(np.zeros((5, 0), dtype=np.uint8)).shape == (0, 0)

    def test_rank_matches_oracle_random(self):
        rng = np.random.default_rng(11)
        for shape in RANDOM_SHAPES:
            for density in (0.2, 0.5, 0.8):
                for _ in range(5):
                    matrix = random_matrix(rng, shape, density)
                    assert gf2_rank(matrix) == oracle_rank(matrix)

    def test_rref_properties(self):
        rng = np.random.default_rng(12)
        for shape in RANDOM_SHAPES:
            matrix = random_matrix(rng, shape)
            rref, pivots = gf2_row_echelon(matrix)
            # 主元列结构：主元位置为 1，其列其余为 0；主元列严格递增
            assert pivots == sorted(pivots)
            for pivot_index, pivot_column in enumerate(pivots):
                column = rref[:, pivot_column]
                assert column[pivot_index] == 1
                assert np.count_nonzero(column) == 1
            # row space 保持：rref 的每行 ∈ rowspace(matrix)，反之亦然（秩相等即可）
            stacked = np.vstack([matrix, rref])
            assert oracle_rank(stacked) == oracle_rank(matrix)
            # 输入不被修改
            assert matrix.dtype == np.uint8

    def test_input_not_mutated(self):
        matrix = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
        copy = matrix.copy()
        gf2_row_echelon(matrix)
        gf2_nullspace(matrix)
        gf2_rank(matrix)
        assert np.array_equal(matrix, copy)


class TestNullspace:
    def test_rank_nullity_and_membership(self):
        rng = np.random.default_rng(13)
        for shape in RANDOM_SHAPES:
            matrix = random_matrix(rng, shape)
            nullspace = gf2_nullspace(matrix)
            rank = gf2_rank(matrix)
            assert rank + nullspace.shape[0] == shape[1]
            if nullspace.shape[0]:
                # 每个基向量以及随机组合都要满足 M v = 0
                assert not gf2_matmul(matrix, nullspace.T).any()
                combo_coefficients = random_matrix(rng, (10, nullspace.shape[0]))
                combos = gf2_matmul(combo_coefficients, nullspace)
                assert not gf2_matmul(matrix, combos.T).any()
                # 基自身独立
                assert gf2_rank(nullspace) == nullspace.shape[0]


class TestRowspaceMembership:
    def test_membership_equivalence_with_rank(self):
        rng = np.random.default_rng(14)
        for shape in RANDOM_SHAPES:
            matrix = random_matrix(rng, shape)
            basis = gf2_rowspace_basis(matrix)
            for _ in range(10):
                vector = random_matrix(rng, (1, shape[1]))[0]
                expected = oracle_rank(np.vstack([matrix, vector])) == oracle_rank(matrix)
                assert gf2_in_rowspace(vector, basis) == expected
                assert gf2_in_rowspace(vector, matrix) == expected  # 非 RREF 基也可用

    def test_row_combinations_are_members(self):
        rng = np.random.default_rng(15)
        matrix = random_matrix(rng, (6, 11))
        for _ in range(10):
            coefficients = random_matrix(rng, (1, 6))[0]
            combo = gf2_matmul(coefficients[None, :], matrix)[0]
            assert gf2_in_rowspace(combo, matrix)


class TestExtendBasis:
    def test_extension_properties(self):
        rng = np.random.default_rng(16)
        for _ in range(10):
            base_matrix = random_matrix(rng, (4, 12))
            existing = gf2_rowspace_basis(base_matrix)
            candidates = random_matrix(rng, (8, 12))
            extended, added_indices = gf2_extend_basis(existing, candidates)
            # 扩展后行独立
            assert gf2_rank(extended) == extended.shape[0]
            # 秩 = span(existing ∪ candidates) 的秩
            assert extended.shape[0] == oracle_rank(np.vstack([existing, candidates]))
            # 被选中的候选原样进入扩展基
            for offset, candidate_index in enumerate(added_indices):
                assert np.array_equal(
                    extended[existing.shape[0] + offset], candidates[candidate_index]
                )
            # 每个候选都落在扩展基的 span 内
            for candidate in candidates:
                assert gf2_in_rowspace(candidate, extended)

    def test_dependent_existing_basis_rejected(self):
        dependent = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError):
            gf2_extend_basis(dependent, np.zeros((1, 3), dtype=np.uint8))


class TestQuotientBasis:
    def test_quotient_properties(self):
        rng = np.random.default_rng(17)
        for _ in range(10):
            matrix = random_matrix(rng, (6, 14))
            kernel = gf2_nullspace(matrix)
            if kernel.shape[0] < 2:
                continue
            # 子空间 = kernel 基的随机组合的 span（保证包含关系）
            coefficients = random_matrix(rng, (3, kernel.shape[0]))
            subspace = gf2_matmul(coefficients, kernel)
            quotient = gf2_quotient_basis(kernel, subspace)
            expected_dim = gf2_rank(kernel) - gf2_rank(subspace)
            assert quotient.shape[0] == expected_dim
            for representative in quotient:
                # 代表元 ∈ span(kernel)
                assert gf2_in_rowspace(representative, kernel)
                # 代表元 ∉ span(subspace)（非平凡类）
                assert not gf2_in_rowspace(representative, subspace)
            # 模子空间独立：rank(subspace ⊕ quotient) = rank(subspace) + dim(quotient)
            stacked = np.vstack([subspace, quotient]) if quotient.size else subspace
            assert oracle_rank(stacked) == oracle_rank(subspace) + expected_dim

    def test_containment_violation_raises(self):
        kernel = np.array([[1, 0, 0]], dtype=np.uint8)
        subspace = np.array([[0, 1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError):
            gf2_quotient_basis(kernel, subspace)

    def test_full_quotient_when_subspace_empty(self):
        kernel = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        quotient = gf2_quotient_basis(kernel, np.zeros((0, 3), dtype=np.uint8))
        assert quotient.shape[0] == 2


class TestSolveAndInverse:
    def test_solve_random_consistent_systems(self):
        rng = np.random.default_rng(18)
        for shape in RANDOM_SHAPES:
            matrix = random_matrix(rng, shape)
            secret = random_matrix(rng, (1, shape[1]))[0]
            target = gf2_matmul(matrix, secret[None, :].T).T[0]
            solution = gf2_solve(matrix, target)
            assert solution is not None
            assert np.array_equal(
                gf2_matmul(matrix, solution[None, :].T).T[0], target
            )

    def test_solve_detects_inconsistency(self):
        matrix = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        target = np.array([1, 0], dtype=np.uint8)
        assert gf2_solve(matrix, target) is None

    def test_inverse_roundtrip_and_singular(self):
        rng = np.random.default_rng(19)
        found = 0
        while found < 5:
            matrix = random_matrix(rng, (6, 6))
            if oracle_rank(matrix) < 6:
                with pytest.raises(ValueError):
                    gf2_inverse(matrix)
                continue
            inverse = gf2_inverse(matrix)
            assert np.array_equal(gf2_matmul(matrix, inverse), np.eye(6, dtype=np.uint8))
            assert np.array_equal(gf2_matmul(inverse, matrix), np.eye(6, dtype=np.uint8))
            found += 1

    def test_as_gf2_matrix_normalizes_values(self):
        matrix = as_gf2_matrix(np.array([[2, 3], [4, 5]]))
        assert np.array_equal(matrix, np.array([[0, 1], [0, 1]], dtype=np.uint8))
