"""Hypergraph product 构造（expander_code.md 规格 §3–§4，公式为权威）。

数学约定（与 spec 逐字一致）：
    H ∈ F_2^{B×A}，shape (n_B, n_A)，H[b,a] = 1 iff (a,b) ∈ E。
    qubit 顺序：先 A×A 全部，再 B×B 全部。
        A×A qubit (a1, a2) → 列 a1·n_A + a2
        B×B qubit (b1, b2) → 列 n_A² + b1·n_B + b2
    check 顺序（两类 check 同一索引方式，行 = A×B 对）：
        H_X 行 (a, b) → a·n_B + b     [X-type stabilizer 生成元]
        H_Z 行 (b, a) → b·n_A + a     [Z-type stabilizer 生成元]
    权威公式：
        H_X = [ I_{n_A}⊗H  |  Hᵀ⊗I_{n_B} ]
        H_Z = [ H⊗I_{n_A}  |  I_{n_B}⊗Hᵀ ]
    两者 shape 均为 (n_A·n_B, n_A²+n_B²)；H_X H_Zᵀ = 0 over F_2
    （代数上 = Hᵀ⊗H + Hᵀ⊗H ≡ 0 mod 2）。

展开成显式条目（tests 用它锁死约定，防静默转置）：
    X-check (a,b) 触及：A×A qubit (a, a2) ∀a2: H[b,a2]=1；B×B qubit (b1, b) ∀b1: H[b1,a]=1
    Z-check (b,a) 触及：A×A qubit (a1, a) ∀a1: H[b,a1]=1；B×B qubit (b, b2) ∀b2: H[b2,a]=1
biregular 时行重 = d_A + d_B；列重：A×A 列在 H_X 与 H_Z 中各为 d_A，B×B 列各为 d_B。

k 公式（HGP 一般理论，r = rank H）：
    k = (n_A − r)² + (n_B − r)²，rank H_X = rank H_Z = n_A·n_B − (n_A−r)(n_B−r)。
"""

import numpy as np

from .gf2 import as_gf2_matrix, gf2_matmul


def classical_parity_check_matrix(graph):
    """由二部图构造 H ∈ F_2^{B×A}（spec §3）：H[b,a] = 1 iff b~a。"""
    matrix = np.zeros((graph.n_B, graph.n_A), dtype=np.uint8)
    for a, neighbors in enumerate(graph.A_to_B):
        for b in neighbors:
            matrix[b, a] = 1
    return matrix


def hgp_from_H(classical_matrix):
    """任意经典 H 的 hypergraph product。

    返回 (H_Z, H_X)（与 spec §4 的返回顺序一致）。
    """
    classical_matrix = as_gf2_matrix(classical_matrix)
    n_B, n_A = classical_matrix.shape
    identity_A = np.eye(n_A, dtype=np.uint8)
    identity_B = np.eye(n_B, dtype=np.uint8)
    H_X = np.concatenate(
        [
            np.kron(identity_A, classical_matrix),
            np.kron(classical_matrix.T, identity_B),
        ],
        axis=1,
    ).astype(np.uint8)
    H_Z = np.concatenate(
        [
            np.kron(classical_matrix, identity_A),
            np.kron(identity_B, classical_matrix.T),
        ],
        axis=1,
    ).astype(np.uint8)
    expected_shape = (n_A * n_B, n_A * n_A + n_B * n_B)
    assert H_X.shape == expected_shape and H_Z.shape == expected_shape
    return H_Z, H_X


def quantum_expander_parity_checks_from_graph(graph):
    """spec §4 接口：返回 (H_Z, H_X)。"""
    return hgp_from_H(classical_parity_check_matrix(graph))


def verify_css_commutation(H_X, H_Z):
    """精确验证 H_X H_Zᵀ = 0 over F_2（spec §4）。"""
    product = gf2_matmul(as_gf2_matrix(H_X), as_gf2_matrix(H_Z).T)
    return not product.any()


def hgp_expected_parameters(classical_matrix, classical_rank):
    """HGP 参数理论值（供构造后断言/对照）：n、k、rank H_X = rank H_Z。"""
    classical_matrix = as_gf2_matrix(classical_matrix)
    n_B, n_A = classical_matrix.shape
    k_H = n_A - classical_rank
    k_HT = n_B - classical_rank
    return {
        "n": n_A * n_A + n_B * n_B,
        "k": k_H * k_H + k_HT * k_HT,
        "rank_HX": n_A * n_B - k_H * k_HT,
        "rank_HZ": n_A * n_B - k_H * k_HT,
    }
