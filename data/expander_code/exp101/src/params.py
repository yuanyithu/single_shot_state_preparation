"""精确 [[n,k,d]] 码参数（expander_code.md 规格 §8）。

  n = 列数；k = n − rank(H_X) − rank(H_Z)
  d_X = min{|x| : x ∈ ker(H_Z) \\ row(H_X)}   （X 型逻辑算符最小权重）
  d_Z = min{|z| : z ∈ ker(H_X) \\ row(H_Z)}
  d = min(d_X, d_Z)

距离搜索是精确的（禁启发式）：对 ker 的 2^dim 个元素做 Gray-code 遍历
（python-int bitmask，XOR 一个基向量/步，popcount 取权重），只在权重刷新
当前最优时才做「∉ rowspace」检查（RREF-int 归约）。dim 超过 max_kernel_dim
时显式 ValueError（不静默截断；(3,4) m≥2 的 ker 维数 ~n/2 超出暴力范围，
家族层面另用经典侧距离与上界记录，见 hgp_classical_side_distances）。

k=0 时无逻辑算符，d_X/d_Z/d 返回 None。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import (
    as_gf2_matrix,
    gf2_matmul,
    gf2_nullspace,
    gf2_rank,
    gf2_rowspace_basis,
)


def _rows_to_ints(matrix):
    """每行 → python int（bit j = 列 j）。"""
    return [
        int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        for row in matrix
    ]


def _int_to_vector(value, num_columns):
    vector = np.zeros(num_columns, dtype=np.uint8)
    for j in range(num_columns):
        if (value >> j) & 1:
            vector[j] = 1
    return vector


class _IntEchelon:
    """int-bitmask 版增量 RREF（列 = bit 位；pivot 取最低位）。"""

    def __init__(self, row_ints):
        self.rows = []
        self.pivots = []
        for row in row_ints:
            self.insert(row)

    def reduce(self, value):
        for row, pivot in zip(self.rows, self.pivots):
            if (value >> pivot) & 1:
                value ^= row
        return value

    def insert(self, value):
        residue = self.reduce(value)
        if residue == 0:
            return False
        pivot = (residue & -residue).bit_length() - 1
        for index, row in enumerate(self.rows):
            if (row >> pivot) & 1:
                self.rows[index] = row ^ residue
        self.rows.append(residue)
        self.pivots.append(pivot)
        return True

    def contains(self, value):
        return self.reduce(value) == 0


def _min_weight_outside_rowspace(kernel_basis, rowspace_basis, max_kernel_dim):
    """min{|v| : v ∈ span(kernel_basis), v ∉ span(rowspace_basis)} 精确搜索。

    返回 (weight, vector) 或 (None, None)（span 内全是 rowspace 成员，即 k=0 侧）。
    """
    kernel_basis = as_gf2_matrix(kernel_basis)
    dim, num_columns = kernel_basis.shape
    if dim > max_kernel_dim:
        raise ValueError(
            f"kernel dimension {dim} exceeds max_kernel_dim={max_kernel_dim}; "
            "exact brute-force distance is infeasible here (raise the limit "
            "explicitly only if you accept 2^dim enumeration)"
        )
    if dim == 0:
        return None, None
    basis_ints = _rows_to_ints(kernel_basis)
    membership = _IntEchelon(_rows_to_ints(as_gf2_matrix(rowspace_basis)))

    best_weight = None
    best_vector_int = None
    current = 0
    for step in range(1, 1 << dim):
        flip_index = (step & -step).bit_length() - 1  # 标准二进制反射 Gray code
        current ^= basis_ints[flip_index]
        weight = current.bit_count()
        if best_weight is not None and weight >= best_weight:
            continue
        if membership.contains(current):
            continue
        best_weight = weight
        best_vector_int = current
        if best_weight == 1:
            break
    if best_weight is None:
        return None, None
    return best_weight, _int_to_vector(best_vector_int, num_columns)


@dataclass
class CodeParameters:
    n: int
    k: int
    rank_H_X: int
    rank_H_Z: int
    d_X: object = None            # int | None
    d_Z: object = None
    d: object = None
    min_logical_X: object = None  # np.ndarray | None
    min_logical_Z: object = None
    distance_computed: bool = False


def code_parameters(H_X, H_Z, compute_distance=True, max_kernel_dim=24):
    """spec §8 主函数。距离精确（暴力/Gray-code），可选。"""
    H_X = as_gf2_matrix(H_X)
    H_Z = as_gf2_matrix(H_Z)
    if H_X.shape[1] != H_Z.shape[1]:
        raise ValueError("H_X and H_Z must have the same number of columns")
    num_qubits = H_X.shape[1]
    rank_H_X = gf2_rank(H_X)
    rank_H_Z = gf2_rank(H_Z)
    k = num_qubits - rank_H_X - rank_H_Z
    result = CodeParameters(
        n=num_qubits, k=k, rank_H_X=rank_H_X, rank_H_Z=rank_H_Z
    )
    if not compute_distance:
        return result

    result.distance_computed = True
    if k == 0:
        return result

    # d_X：ker(H_Z) 中的非 row(H_X) 最小权重
    d_X, min_x = _min_weight_outside_rowspace(
        gf2_nullspace(H_Z), gf2_rowspace_basis(H_X), max_kernel_dim
    )
    # d_Z：ker(H_X) 中的非 row(H_Z) 最小权重
    d_Z, min_z = _min_weight_outside_rowspace(
        gf2_nullspace(H_X), gf2_rowspace_basis(H_Z), max_kernel_dim
    )
    if d_X is None or d_Z is None:
        raise AssertionError("k>0 but a distance side found no logical vector")
    # 防御断言：返回值必须是真逻辑算符（spec §10F：绝不返回 stabilizer）
    if gf2_matmul(H_Z, min_x[None, :].T).any():
        raise AssertionError("min_logical_X is not in ker(H_Z)")
    if gf2_matmul(H_X, min_z[None, :].T).any():
        raise AssertionError("min_logical_Z is not in ker(H_X)")
    result.d_X = int(d_X)
    result.d_Z = int(d_Z)
    result.d = int(min(d_X, d_Z))
    result.min_logical_X = min_x
    result.min_logical_Z = min_z
    return result


def classical_code_distance(parity_check_matrix, max_kernel_dim=24):
    """经典码 ker(H) 的精确最小距离（k=0 时 None）。"""
    parity_check_matrix = as_gf2_matrix(parity_check_matrix)
    kernel = gf2_nullspace(parity_check_matrix)
    if kernel.shape[0] == 0:
        return None
    empty_rowspace = np.zeros((0, parity_check_matrix.shape[1]), dtype=np.uint8)
    weight, _ = _min_weight_outside_rowspace(kernel, empty_rowspace, max_kernel_dim)
    return int(weight)


def hgp_classical_side_distances(classical_matrix, max_kernel_dim=24):
    """HGP 两个经典侧的精确距离（供与量子 d 的定理关系交叉验证/家族记录）。

    返回 {"d_ker_H": int|None, "d_ker_HT": int|None, "theorem_min": int|None}。
    theorem_min = 非 None 侧的最小值；对 HGP(H,H) 的量子 d 的理论预言
    （在小码上与暴力精确值交叉验证后方可用于大码记录，且必须标注来源）。
    """
    classical_matrix = as_gf2_matrix(classical_matrix)
    d_ker_H = classical_code_distance(classical_matrix, max_kernel_dim)
    d_ker_HT = classical_code_distance(classical_matrix.T, max_kernel_dim)
    candidates = [d for d in (d_ker_H, d_ker_HT) if d is not None]
    return {
        "d_ker_H": d_ker_H,
        "d_ker_HT": d_ker_HT,
        "theorem_min": min(candidates) if candidates else None,
    }
