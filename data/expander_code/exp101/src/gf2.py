"""GF(2) 精确线性代数工具包（expander_code.md 规格 §6）。

约定：
  - 矩阵/向量一律解释为 F_2 元素；输入接受 bool / 任意整型（内部 %2 转 uint8）。
  - 消元是确定性的：按列从左到右找主元、行交换，输出 reduced row echelon form
    （RREF；主元列上下都清零）。不做列置换。
  - 空矩阵约定：shape (0, n) rank 0，nullspace = I_n；shape (m, 0) rank 0，
    nullspace 为 shape (0, 0)。
  - 所有函数无副作用（输入不被原地修改）、无随机性。
"""

import numpy as np


def as_gf2_matrix(matrix):
    """规范化为 2D uint8 (0/1) 矩阵。"""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got ndim={matrix.ndim}")
    return (matrix.astype(np.int64) % 2).astype(np.uint8)


def as_gf2_vector(vector):
    """规范化为 1D uint8 (0/1) 向量。"""
    vector = np.asarray(vector)
    if vector.ndim != 1:
        raise ValueError(f"expected 1D vector, got ndim={vector.ndim}")
    return (vector.astype(np.int64) % 2).astype(np.uint8)


def gf2_matmul(left, right):
    """(left @ right) mod 2，uint8。"""
    left = np.asarray(left).astype(np.uint8)
    right = np.asarray(right).astype(np.uint8)
    return (left.astype(np.int64) @ right.astype(np.int64) % 2).astype(np.uint8)


def gf2_row_echelon(matrix):
    """确定性 RREF。

    返回 (rref_matrix, pivot_columns)：
      rref_matrix: uint8，与输入同形状（副本）
      pivot_columns: list[int]，第 i 个主元位于 (i, pivot_columns[i])
    """
    rref = as_gf2_matrix(matrix).copy()
    num_rows, num_columns = rref.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(num_columns):
        if pivot_row >= num_rows:
            break
        candidate_rows = np.flatnonzero(rref[pivot_row:, column])
        if candidate_rows.size == 0:
            continue
        source_row = pivot_row + int(candidate_rows[0])
        if source_row != pivot_row:
            rref[[pivot_row, source_row]] = rref[[source_row, pivot_row]]
        other_rows = np.flatnonzero(rref[:, column])
        for row in other_rows:
            if row != pivot_row:
                rref[row] ^= rref[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    return rref, pivot_columns


def gf2_rank(matrix):
    _, pivot_columns = gf2_row_echelon(matrix)
    return len(pivot_columns)


def gf2_nullspace(matrix):
    """ker(matrix) 的一组基，shape (nullity, num_columns)。

    满足 matrix @ v = 0 (mod 2) 对每个基向量 v；nullity = n − rank。
    """
    matrix = as_gf2_matrix(matrix)
    num_rows, num_columns = matrix.shape
    rref, pivot_columns = gf2_row_echelon(matrix)
    pivot_set = set(pivot_columns)
    free_columns = [c for c in range(num_columns) if c not in pivot_set]
    basis = np.zeros((len(free_columns), num_columns), dtype=np.uint8)
    for basis_index, free_column in enumerate(free_columns):
        basis[basis_index, free_column] = 1
        for pivot_index, pivot_column in enumerate(pivot_columns):
            basis[basis_index, pivot_column] = rref[pivot_index, free_column]
    return basis


def gf2_rowspace_basis(matrix):
    """row space 的 RREF 基，shape (rank, num_columns)。"""
    rref, pivot_columns = gf2_row_echelon(matrix)
    return rref[: len(pivot_columns)].copy()


def _reduce_vector(vector, echelon_rows, echelon_pivots):
    """把 vector 对 (RREF 行, 主元列) 消元；返回残差（副本）。"""
    residue = vector.copy()
    for row, pivot in zip(echelon_rows, echelon_pivots):
        if residue[pivot]:
            residue ^= row
    return residue


class _EchelonWorkspace:
    """增量 RREF 工作区：支持逐向量插入与归约。"""

    def __init__(self, num_columns):
        self.num_columns = int(num_columns)
        self.rows = []
        self.pivots = []

    @classmethod
    def from_matrix(cls, matrix):
        matrix = as_gf2_matrix(matrix)
        workspace = cls(matrix.shape[1])
        for row in matrix:
            workspace.insert(row)
        return workspace

    def reduce(self, vector):
        return _reduce_vector(as_gf2_vector(vector), self.rows, self.pivots)

    def insert(self, vector):
        """归约后若残差非零则加入工作区；返回 (加入与否, 残差)。"""
        residue = self.reduce(vector)
        nonzero = np.flatnonzero(residue)
        if nonzero.size == 0:
            return False, residue
        pivot = int(nonzero[0])
        for existing_row in self.rows:
            if existing_row[pivot]:
                existing_row ^= residue
        self.rows.append(residue)
        self.pivots.append(pivot)
        return True, residue

    def contains(self, vector):
        return not self.reduce(vector).any()

    @property
    def rank(self):
        return len(self.rows)


def gf2_in_rowspace(vector, rowspace_basis):
    """vector ∈ rowspace(rowspace_basis)？（基不要求已成 RREF。）"""
    vector = as_gf2_vector(vector)
    workspace = _EchelonWorkspace.from_matrix(rowspace_basis)
    if vector.shape[0] != workspace.num_columns:
        raise ValueError("vector length does not match basis columns")
    return workspace.contains(vector)


def gf2_extend_basis(existing_basis, candidate_vectors):
    """贪心扩基。

    返回 (extended_basis, added_candidate_indices)：
      extended_basis: uint8 (r+t, n) = 原基行 + 被选中的候选行（原样保留）
      added_candidate_indices: list[int]，被选中候选在输入中的下标
    要求 existing_basis 行线性独立（否则 ValueError）。
    """
    existing_basis = as_gf2_matrix(existing_basis)
    candidate_vectors = as_gf2_matrix(candidate_vectors)
    if existing_basis.shape[0] and candidate_vectors.shape[0]:
        if existing_basis.shape[1] != candidate_vectors.shape[1]:
            raise ValueError("existing_basis and candidates column mismatch")
    workspace = _EchelonWorkspace.from_matrix(existing_basis)
    if workspace.rank != existing_basis.shape[0]:
        raise ValueError("existing_basis rows are not linearly independent")
    added_rows = []
    added_candidate_indices = []
    for candidate_index, candidate in enumerate(candidate_vectors):
        inserted, _ = workspace.insert(candidate)
        if inserted:
            added_rows.append(candidate.copy())
            added_candidate_indices.append(candidate_index)
    if added_rows:
        extended_basis = np.vstack([existing_basis, np.array(added_rows, dtype=np.uint8)])
    else:
        extended_basis = existing_basis.copy()
    return extended_basis, added_candidate_indices


def gf2_quotient_basis(kernel_basis, subspace_basis):
    """span(kernel_basis) / span(subspace_basis) 的代表元。

    要求 span(subspace_basis) ⊆ span(kernel_basis)（否则 ValueError）。
    返回 uint8 (dim_kernel − dim_subspace, n)；每个代表元 ∈ span(kernel_basis)，
    且诸代表元模 span(subspace_basis) 线性独立。代表元取「模子空间消元后的残差」，
    确定性。
    """
    kernel_basis = as_gf2_matrix(kernel_basis)
    subspace_basis = as_gf2_matrix(subspace_basis)
    if subspace_basis.shape[0] and kernel_basis.shape[0]:
        if subspace_basis.shape[1] != kernel_basis.shape[1]:
            raise ValueError("kernel/subspace column mismatch")
    kernel_workspace = _EchelonWorkspace.from_matrix(kernel_basis)
    for row_index, subspace_row in enumerate(subspace_basis):
        if not kernel_workspace.contains(subspace_row):
            raise ValueError(
                f"subspace_basis row {row_index} is not inside span(kernel_basis)"
            )
    combined_workspace = _EchelonWorkspace.from_matrix(subspace_basis)
    representatives = []
    for kernel_row in kernel_basis:
        inserted, residue = combined_workspace.insert(kernel_row)
        if inserted:
            representatives.append(residue.copy())
    expected_dimension = gf2_rank(kernel_basis) - gf2_rank(subspace_basis)
    if len(representatives) != expected_dimension:
        raise AssertionError(
            "quotient dimension mismatch: "
            f"got {len(representatives)}, expected {expected_dimension}"
        )
    num_columns = kernel_basis.shape[1] if kernel_basis.ndim == 2 else 0
    if representatives:
        return np.array(representatives, dtype=np.uint8)
    return np.zeros((0, num_columns), dtype=np.uint8)


def gf2_solve(matrix, target):
    """解 matrix @ x = target (mod 2)。有解返回某个特解 x（uint8），无解返回 None。"""
    matrix = as_gf2_matrix(matrix)
    target = as_gf2_vector(target)
    num_rows, num_columns = matrix.shape
    if target.shape[0] != num_rows:
        raise ValueError("target length does not match matrix rows")
    augmented = np.concatenate([matrix, target[:, None]], axis=1)
    rref, pivot_columns = gf2_row_echelon(augmented)
    if num_columns in pivot_columns:
        return None
    solution = np.zeros(num_columns, dtype=np.uint8)
    for pivot_index, pivot_column in enumerate(pivot_columns):
        solution[pivot_column] = rref[pivot_index, num_columns]
    return solution


def gf2_inverse(matrix):
    """方阵求逆；奇异则 ValueError。"""
    matrix = as_gf2_matrix(matrix)
    size = matrix.shape[0]
    if matrix.shape[1] != size:
        raise ValueError("matrix must be square")
    augmented = np.concatenate(
        [matrix, np.eye(size, dtype=np.uint8)], axis=1
    )
    rref, pivot_columns = gf2_row_echelon(augmented)
    if pivot_columns[:size] != list(range(size)) or len(pivot_columns) < size:
        raise ValueError("matrix is singular over GF(2)")
    return rref[:, size:].copy()
