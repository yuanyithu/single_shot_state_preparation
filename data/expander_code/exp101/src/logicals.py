"""逻辑 Pauli 算符（expander_code.md 规格 §7，标准 CSS 约定）。

  logical Z = ker(H_X) / row(H_Z) 的商基代表 z_1..z_k
  logical X = ker(H_Z) / row(H_X) 的商基代表 x_1..x_k
  配对矩阵 M[i,j] = x_i · z_j (mod 2) 必须可逆；用 X ← M^{-1} X 归一化到
  x_i · z_j = δ_ij。

说明：
  - 同型 Pauli（X-X、Z-Z）在 CSS 下自动对易（symplectic 形式恒 0），无需数值检查；
    verify_logical_pauli_result 检查的是非平凡条件（kernel 隶属、非 stabilizer、
    配对恒等、模 stabilizer 独立性）。
  - 归一化混合的是逻辑类的非零组合，仍在 ker(H_Z) 内、仍非 stabilizer
    （商基独立性保证），tests 显式验证。
  - k=0 时优雅返回空结果（0×n 数组、0×0 配对）。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import (
    as_gf2_matrix,
    gf2_in_rowspace,
    gf2_inverse,
    gf2_matmul,
    gf2_nullspace,
    gf2_quotient_basis,
    gf2_rank,
    gf2_rowspace_basis,
)
from .hgp import verify_css_commutation


@dataclass
class LogicalPauliResult:
    logical_X: np.ndarray            # (k, n) uint8，已配对归一
    logical_Z: np.ndarray            # (k, n) uint8
    pairing_matrix: np.ndarray       # 归一后 = I_k
    k: int
    pre_normalization_pairing: np.ndarray  # 归一前 M
    rank_H_X: int
    rank_H_Z: int


def logical_pauli_operators(H_X, H_Z):
    """spec §7 主函数。要求 H_X H_Zᵀ = 0（否则 ValueError）。"""
    H_X = as_gf2_matrix(H_X)
    H_Z = as_gf2_matrix(H_Z)
    if H_X.shape[1] != H_Z.shape[1]:
        raise ValueError("H_X and H_Z must have the same number of columns")
    if not verify_css_commutation(H_X, H_Z):
        raise ValueError("CSS commutation H_X H_Z^T = 0 fails; not a CSS pair")

    num_qubits = H_X.shape[1]
    rank_H_X = gf2_rank(H_X)
    rank_H_Z = gf2_rank(H_Z)
    expected_k = num_qubits - rank_H_X - rank_H_Z

    kernel_H_X = gf2_nullspace(H_X)
    kernel_H_Z = gf2_nullspace(H_Z)
    rowspace_H_Z = gf2_rowspace_basis(H_Z)
    rowspace_H_X = gf2_rowspace_basis(H_X)

    logical_Z = gf2_quotient_basis(kernel_H_X, rowspace_H_Z)
    logical_X = gf2_quotient_basis(kernel_H_Z, rowspace_H_X)
    if logical_Z.shape[0] != expected_k or logical_X.shape[0] != expected_k:
        raise AssertionError(
            f"logical dimension mismatch: dim(Z)={logical_Z.shape[0]}, "
            f"dim(X)={logical_X.shape[0]}, expected k={expected_k}"
        )

    if expected_k == 0:
        identity = np.zeros((0, 0), dtype=np.uint8)
        return LogicalPauliResult(
            logical_X=np.zeros((0, num_qubits), dtype=np.uint8),
            logical_Z=np.zeros((0, num_qubits), dtype=np.uint8),
            pairing_matrix=identity,
            k=0,
            pre_normalization_pairing=identity,
            rank_H_X=rank_H_X,
            rank_H_Z=rank_H_Z,
        )

    pairing = gf2_matmul(logical_X, logical_Z.T)
    try:
        pairing_inverse = gf2_inverse(pairing)
    except ValueError as error:
        raise ValueError(
            "pairing matrix x_i·z_j is singular over GF(2); "
            "quotient bases do not form dual pairs"
        ) from error
    logical_X_normalized = gf2_matmul(pairing_inverse, logical_X)
    final_pairing = gf2_matmul(logical_X_normalized, logical_Z.T)
    if not np.array_equal(final_pairing, np.eye(expected_k, dtype=np.uint8)):
        raise AssertionError("pairing normalization failed to reach identity")

    return LogicalPauliResult(
        logical_X=logical_X_normalized,
        logical_Z=logical_Z,
        pairing_matrix=final_pairing,
        k=expected_k,
        pre_normalization_pairing=pairing,
        rank_H_X=rank_H_X,
        rank_H_Z=rank_H_Z,
    )


def verify_logical_pauli_result(H_X, H_Z, result, strict=True):
    """spec §7 一致性检查清单；strict=True 时任何失败抛 AssertionError。

    返回 dict[str, bool]（全部检查的结果，便于留档）。
    """
    H_X = as_gf2_matrix(H_X)
    H_Z = as_gf2_matrix(H_Z)
    rowspace_H_X = gf2_rowspace_basis(H_X)
    rowspace_H_Z = gf2_rowspace_basis(H_Z)
    k = result.k
    checks = {}

    checks["logical_X_in_ker_H_Z"] = (
        not gf2_matmul(H_Z, result.logical_X.T).any() if k else True
    )
    checks["logical_Z_in_ker_H_X"] = (
        not gf2_matmul(H_X, result.logical_Z.T).any() if k else True
    )
    checks["logical_X_not_stabilizer"] = all(
        not gf2_in_rowspace(x, rowspace_H_X) for x in result.logical_X
    )
    checks["logical_Z_not_stabilizer"] = all(
        not gf2_in_rowspace(z, rowspace_H_Z) for z in result.logical_Z
    )
    checks["pairing_is_identity"] = np.array_equal(
        gf2_matmul(result.logical_X, result.logical_Z.T),
        np.eye(k, dtype=np.uint8),
    )
    # 模 stabilizer 独立：rank(row(H_X) ⊕ X) = rank_H_X + k（Z 侧同理）
    if k:
        stacked_X = np.vstack([rowspace_H_X, result.logical_X])
        stacked_Z = np.vstack([rowspace_H_Z, result.logical_Z])
        checks["logical_X_independent_mod_stabilizers"] = (
            gf2_rank(stacked_X) == result.rank_H_X + k
        )
        checks["logical_Z_independent_mod_stabilizers"] = (
            gf2_rank(stacked_Z) == result.rank_H_Z + k
        )
    else:
        checks["logical_X_independent_mod_stabilizers"] = True
        checks["logical_Z_independent_mod_stabilizers"] = True
    checks["k_matches_rank_formula"] = (
        k == H_X.shape[1] - result.rank_H_X - result.rank_H_Z
    )
    # 同型对易：CSS 下自动成立（文档性检查，恒 True）
    checks["same_type_commutation_automatic"] = True

    if strict:
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"logical Pauli checks failed: {failed}")
    return checks
