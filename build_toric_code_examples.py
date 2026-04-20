"""
生成 2D / 3D toric code 的 (parity_check_matrix, dual_logical_z_basis)，
作为 run_disorder_average_simulation 的输入示例。

约定（两个版本共用）：
  - data qubit 放在 1-cell（边）上
  - Z stabilizer 放在 2-cell（面）上：每个 stabilizer = 面上的 4 条边
  - X stabilizer 放在 0-cell（顶点）上：每个 stabilizer = 顶点上的若干条入边
    （2D 是 4 条，3D 是 6 条）
  - 所有方向用周期边界 (torus)：2D 给出 k=2 个逻辑 qubit，3D 给出 k=3 个
  - 逻辑 Z = 非平凡 1-闭链，沿某个方向的轴线

依据这个约定，H_Z 每行是一个 plaquette 的 4 条边；逻辑 Z 的每一条代表链是一圈
沿某个坐标轴缠绕 torus 的边集合。这些代表链显然在 ker H_X 中（作为 1-闭链）。
"""

import numpy as np


# =========================================================
# 2D toric code  (L × L torus)
# =========================================================
#
# qubit 编号：
#   水平边 horizontal_edge[row_index][column_index]：
#       从 (row_index, column_index) 到 (row_index, column_index + 1 mod L)
#       qubit_index = row_index * L + column_index                       ∈ [0, L²)
#   竖直边 vertical_edge[row_index][column_index]：
#       从 (row_index, column_index) 到 (row_index + 1 mod L, column_index)
#       qubit_index = L² + row_index * L + column_index                  ∈ [L², 2L²)
#
# plaquette 编号：
#   plaquette[row_index][column_index]：左上角为 (row_index, column_index) 的单元面
#       check_index = row_index * L + column_index                       ∈ [0, L²)
#       包含四条边：
#         上: horizontal_edge[row_index][column_index]
#         下: horizontal_edge[(row_index + 1) mod L][column_index]
#         左: vertical_edge[row_index][column_index]
#         右: vertical_edge[row_index][(column_index + 1) mod L]
#
# 逻辑 Z：
#   logical_z_horizontal: 所有 horizontal_edge[0][·]     —— 沿着 column 方向一圈
#   logical_z_vertical:   所有 vertical_edge[·][0]       —— 沿着 row 方向一圈

def build_2d_toric_code(lattice_size):
    """
    输入：
      lattice_size: int, L，晶格边长（建议 >= 2）
    返回：
      parity_check_matrix:  shape (L², 2L²),  dtype=bool
      dual_logical_z_basis: shape (2,  2L²),  dtype=bool
    """
    num_horizontal_edges = lattice_size * lattice_size
    num_vertical_edges = lattice_size * lattice_size
    num_qubits = num_horizontal_edges + num_vertical_edges
    num_plaquettes = lattice_size * lattice_size

    def horizontal_edge_qubit(row_index, column_index):
        return row_index * lattice_size + column_index

    def vertical_edge_qubit(row_index, column_index):
        return num_horizontal_edges + row_index * lattice_size + column_index

    # --- 构造 H_Z：每个 plaquette 一行，占 4 个 1 ---
    parity_check_matrix = np.zeros((num_plaquettes, num_qubits), dtype=bool)
    for row_index in range(lattice_size):
        for column_index in range(lattice_size):
            plaquette_index = row_index * lattice_size + column_index
            row_next = (row_index + 1) % lattice_size
            column_next = (column_index + 1) % lattice_size

            parity_check_matrix[plaquette_index, horizontal_edge_qubit(row_index, column_index)] = True
            parity_check_matrix[plaquette_index, horizontal_edge_qubit(row_next, column_index)] = True
            parity_check_matrix[plaquette_index, vertical_edge_qubit(row_index, column_index)] = True
            parity_check_matrix[plaquette_index, vertical_edge_qubit(row_index, column_next)] = True

    # --- 构造逻辑 Z 基：两条非平凡 1-闭链 ---
    dual_logical_z_basis = np.zeros((2, num_qubits), dtype=bool)
    # 第 0 行：所有 horizontal_edge[0][·]，沿 column 方向缠绕
    for column_index in range(lattice_size):
        dual_logical_z_basis[0, horizontal_edge_qubit(0, column_index)] = True
    # 第 1 行：所有 vertical_edge[·][0]，沿 row 方向缠绕
    for row_index in range(lattice_size):
        dual_logical_z_basis[1, vertical_edge_qubit(row_index, 0)] = True

    return parity_check_matrix, dual_logical_z_basis


def build_2d_toric_zero_syndrome_move_data(lattice_size):
    """
    为 2D toric code 构造 q=0 采样所需的几何闭环 move。

    输出：
      zero_syndrome_move_data: dict
        "contractible_moves":
            np.ndarray，shape (L^2, 2L^2)，dtype=bool
            每行是一个 vertex-star 型局部闭环，weight 固定为 4
        "winding_moves":
            np.ndarray，shape (2L, 2L^2)，dtype=bool
            前 L 行是固定 column 的 horizontal winding loop，
            后 L 行是固定 row 的 vertical winding loop，weight 固定为 L
        "start_sector_generators":
            np.ndarray，shape (2, 2L^2)，dtype=bool
            两条独立的非平凡 kernel loop，用于生成 4 个 q=0 合法初态
    """
    parity_check_matrix, _ = build_2d_toric_code(lattice_size)

    num_horizontal_edges = lattice_size * lattice_size
    num_qubits = 2 * num_horizontal_edges
    num_vertices = lattice_size * lattice_size

    def horizontal_edge_qubit(row_index, column_index):
        return row_index * lattice_size + column_index

    def vertical_edge_qubit(row_index, column_index):
        return num_horizontal_edges + row_index * lattice_size + column_index

    contractible_moves = np.zeros((num_vertices, num_qubits), dtype=bool)
    for row_index in range(lattice_size):
        for column_index in range(lattice_size):
            vertex_index = row_index * lattice_size + column_index
            row_prev = (row_index - 1) % lattice_size
            column_prev = (column_index - 1) % lattice_size

            contractible_moves[
                vertex_index,
                horizontal_edge_qubit(row_index, column_index),
            ] = True
            contractible_moves[
                vertex_index,
                horizontal_edge_qubit(row_index, column_prev),
            ] = True
            contractible_moves[
                vertex_index,
                vertical_edge_qubit(row_index, column_index),
            ] = True
            contractible_moves[
                vertex_index,
                vertical_edge_qubit(row_prev, column_index),
            ] = True

    winding_moves = np.zeros(
        (2 * lattice_size, num_qubits),
        dtype=bool,
    )
    for column_index in range(lattice_size):
        for row_index in range(lattice_size):
            winding_moves[
                column_index,
                horizontal_edge_qubit(row_index, column_index),
            ] = True

    for row_index in range(lattice_size):
        winding_row_index = lattice_size + row_index
        for column_index in range(lattice_size):
            winding_moves[
                winding_row_index,
                vertical_edge_qubit(row_index, column_index),
            ] = True

    start_sector_generators = np.stack(
        (winding_moves[0], winding_moves[lattice_size]),
        axis=0,
    )

    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    contractible_syndrome_bits = (
        parity_check_matrix_uint8 @ contractible_moves.T.astype(np.uint8)
    ) % 2
    winding_syndrome_bits = (
        parity_check_matrix_uint8 @ winding_moves.T.astype(np.uint8)
    ) % 2
    assert not np.any(contractible_syndrome_bits), (
        "contractible q=0 moves must lie in ker(H_Z)"
    )
    assert not np.any(winding_syndrome_bits), (
        "winding q=0 moves must lie in ker(H_Z)"
    )
    assert np.all(contractible_moves.sum(axis=1) == 4), (
        "contractible q=0 moves must have weight 4"
    )
    assert np.all(winding_moves.sum(axis=1) == lattice_size), (
        "winding q=0 moves must have weight L"
    )

    return {
        "contractible_moves": contractible_moves,
        "winding_moves": winding_moves,
        "start_sector_generators": start_sector_generators,
    }


# =========================================================
# 3D toric code  (L × L × L torus)
# =========================================================
#
# qubit 编号（直接按 edge 方向 + 三个坐标 (i, j, k) ∈ [0, L)³）：
#   x_edge[i][j][k]: 从 (i, j, k) 到 ((i+1) mod L, j, k)
#       qubit_index = 0 * L³ + (i * L + j) * L + k
#   y_edge[i][j][k]: 从 (i, j, k) 到 (i, (j+1) mod L, k)
#       qubit_index = 1 * L³ + (i * L + j) * L + k
#   z_edge[i][j][k]: 从 (i, j, k) 到 (i, j, (k+1) mod L)
#       qubit_index = 2 * L³ + (i * L + j) * L + k
#
# plaquette 编号（按法向分三类，每类用 (i, j, k) 定位）：
#   xy_plaquette[i][j][k]: 法向 +z，左下角 (i, j, k)，4 条边为
#       x_edge[i][j][k], x_edge[i][(j+1) mod L][k],
#       y_edge[i][j][k], y_edge[(i+1) mod L][j][k]
#       check_index = 0 * L³ + (i * L + j) * L + k
#   xz_plaquette[i][j][k]: 法向 +y，4 条边为
#       x_edge[i][j][k], x_edge[i][j][(k+1) mod L],
#       z_edge[i][j][k], z_edge[(i+1) mod L][j][k]
#       check_index = 1 * L³ + (i * L + j) * L + k
#   yz_plaquette[i][j][k]: 法向 +x，4 条边为
#       y_edge[i][j][k], y_edge[i][j][(k+1) mod L],
#       z_edge[i][j][k], z_edge[i][(j+1) mod L][k]
#       check_index = 2 * L³ + (i * L + j) * L + k
#
# 逻辑 Z：沿三个坐标轴的非平凡闭链
#   logical_z_x: 所有 x_edge[i][0][0]
#   logical_z_y: 所有 y_edge[0][j][0]
#   logical_z_z: 所有 z_edge[0][0][k]

def build_3d_toric_code(lattice_size):
    """
    输入：
      lattice_size: int, L，晶格边长（建议 >= 2）
    返回：
      parity_check_matrix:  shape (3L³, 3L³), dtype=bool
      dual_logical_z_basis: shape (3,   3L³), dtype=bool
    """
    L = lattice_size
    L_cubed = L * L * L
    num_qubits = 3 * L_cubed
    num_plaquettes = 3 * L_cubed

    def edge_qubit(edge_type_index, i, j, k):
        # edge_type_index: 0 = x_edge, 1 = y_edge, 2 = z_edge
        return edge_type_index * L_cubed + (i * L + j) * L + k

    def plaquette_check(plaquette_type_index, i, j, k):
        # plaquette_type_index: 0 = xy, 1 = xz, 2 = yz
        return plaquette_type_index * L_cubed + (i * L + j) * L + k

    parity_check_matrix = np.zeros((num_plaquettes, num_qubits), dtype=bool)

    for i in range(L):
        for j in range(L):
            for k in range(L):
                i_next = (i + 1) % L
                j_next = (j + 1) % L
                k_next = (k + 1) % L

                # xy 面（法向 +z）：含两条 x_edge 和两条 y_edge
                check_xy = plaquette_check(0, i, j, k)
                parity_check_matrix[check_xy, edge_qubit(0, i,      j,      k)] = True
                parity_check_matrix[check_xy, edge_qubit(0, i,      j_next, k)] = True
                parity_check_matrix[check_xy, edge_qubit(1, i,      j,      k)] = True
                parity_check_matrix[check_xy, edge_qubit(1, i_next, j,      k)] = True

                # xz 面（法向 +y）：含两条 x_edge 和两条 z_edge
                check_xz = plaquette_check(1, i, j, k)
                parity_check_matrix[check_xz, edge_qubit(0, i,      j, k     )] = True
                parity_check_matrix[check_xz, edge_qubit(0, i,      j, k_next)] = True
                parity_check_matrix[check_xz, edge_qubit(2, i,      j, k     )] = True
                parity_check_matrix[check_xz, edge_qubit(2, i_next, j, k     )] = True

                # yz 面（法向 +x）：含两条 y_edge 和两条 z_edge
                check_yz = plaquette_check(2, i, j, k)
                parity_check_matrix[check_yz, edge_qubit(1, i, j,      k     )] = True
                parity_check_matrix[check_yz, edge_qubit(1, i, j,      k_next)] = True
                parity_check_matrix[check_yz, edge_qubit(2, i, j,      k     )] = True
                parity_check_matrix[check_yz, edge_qubit(2, i, j_next, k     )] = True

    # --- 构造逻辑 Z 基：三条沿坐标轴的非平凡闭链 ---
    dual_logical_z_basis = np.zeros((3, num_qubits), dtype=bool)
    for i in range(L):
        dual_logical_z_basis[0, edge_qubit(0, i, 0, 0)] = True
    for j in range(L):
        dual_logical_z_basis[1, edge_qubit(1, 0, j, 0)] = True
    for k in range(L):
        dual_logical_z_basis[2, edge_qubit(2, 0, 0, k)] = True

    return parity_check_matrix, dual_logical_z_basis


# =========================================================
# 辅助：也构造 H_X 用于验证 dual_logical_z_basis 在 ker H_X 中
# （H_X 不进接口，只用于 sanity check）
# =========================================================

def build_2d_toric_x_check_matrix(lattice_size):
    """2D toric code 的 H_X：每个 vertex 一行，4 条入边为 1。"""
    num_horizontal_edges = lattice_size * lattice_size
    num_vertical_edges = lattice_size * lattice_size
    num_qubits = num_horizontal_edges + num_vertical_edges
    num_vertices = lattice_size * lattice_size

    def horizontal_edge_qubit(row_index, column_index):
        return row_index * lattice_size + column_index

    def vertical_edge_qubit(row_index, column_index):
        return num_horizontal_edges + row_index * lattice_size + column_index

    x_check_matrix = np.zeros((num_vertices, num_qubits), dtype=bool)
    for row_index in range(lattice_size):
        for column_index in range(lattice_size):
            vertex_index = row_index * lattice_size + column_index
            row_prev = (row_index - 1) % lattice_size
            column_prev = (column_index - 1) % lattice_size
            # 顶点 (row_index, column_index) 入射的四条边
            x_check_matrix[vertex_index, horizontal_edge_qubit(row_index, column_index)] = True
            x_check_matrix[vertex_index, horizontal_edge_qubit(row_index, column_prev)] = True
            x_check_matrix[vertex_index, vertical_edge_qubit(row_index, column_index)] = True
            x_check_matrix[vertex_index, vertical_edge_qubit(row_prev, column_index)] = True
    return x_check_matrix


def build_3d_toric_x_check_matrix(lattice_size):
    """3D toric code 的 H_X：每个 vertex 一行，6 条入边为 1。"""
    L = lattice_size
    L_cubed = L * L * L
    num_qubits = 3 * L_cubed
    num_vertices = L_cubed

    def edge_qubit(edge_type_index, i, j, k):
        return edge_type_index * L_cubed + (i * L + j) * L + k

    x_check_matrix = np.zeros((num_vertices, num_qubits), dtype=bool)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                vertex_index = (i * L + j) * L + k
                i_prev = (i - 1) % L
                j_prev = (j - 1) % L
                k_prev = (k - 1) % L
                # 6 条入射边
                x_check_matrix[vertex_index, edge_qubit(0, i,      j, k)] = True
                x_check_matrix[vertex_index, edge_qubit(0, i_prev, j, k)] = True
                x_check_matrix[vertex_index, edge_qubit(1, i, j,      k)] = True
                x_check_matrix[vertex_index, edge_qubit(1, i, j_prev, k)] = True
                x_check_matrix[vertex_index, edge_qubit(2, i, j, k     )] = True
                x_check_matrix[vertex_index, edge_qubit(2, i, j, k_prev)] = True
    return x_check_matrix


# =========================================================
# 小工具：GF(2) 上的 rank
# =========================================================

def gf2_rank(boolean_matrix):
    """GF(2) 上的高斯消元算 rank。只用来 sanity check，不追求效率。"""
    working_matrix = boolean_matrix.copy().astype(bool)
    num_rows, num_columns = working_matrix.shape
    current_pivot_row = 0
    for column_index in range(num_columns):
        if current_pivot_row >= num_rows:
            break
        pivot_row_index = -1
        for row_index in range(current_pivot_row, num_rows):
            if working_matrix[row_index, column_index]:
                pivot_row_index = row_index
                break
        if pivot_row_index == -1:
            continue
        if pivot_row_index != current_pivot_row:
            working_matrix[[current_pivot_row, pivot_row_index]] = \
                working_matrix[[pivot_row_index, current_pivot_row]]
        for row_index in range(num_rows):
            if row_index != current_pivot_row and working_matrix[row_index, column_index]:
                working_matrix[row_index] ^= working_matrix[current_pivot_row]
        current_pivot_row += 1
    return current_pivot_row


# =========================================================
# 验证与打印
# =========================================================

def verify_and_report(name, parity_check_matrix, dual_logical_z_basis,
                      x_check_matrix, expected_num_logical_qubits):
    """打印 code 的基本参数，并检查关键条件。"""
    num_checks, num_qubits = parity_check_matrix.shape
    num_logical_qubits = dual_logical_z_basis.shape[0]

    rank_h_z = gf2_rank(parity_check_matrix)
    rank_h_x = gf2_rank(x_check_matrix)
    num_z_redundancies = num_checks - rank_h_z
    num_x_redundancies = x_check_matrix.shape[0] - rank_h_x

    # k = dim ker(H_X) - rank(H_Z) = (n - rank H_X) - rank H_Z
    k_computed = (num_qubits - rank_h_x) - rank_h_z

    # 检查 dual_logical_z_basis 每一行落在 ker H_X 中
    # i.e. 对每一行 z_bar, H_X @ z_bar = 0 (mod 2)
    product = (x_check_matrix.astype(np.int8) @ dual_logical_z_basis.T.astype(np.int8)) % 2
    all_in_kernel = bool((product == 0).all())

    # 每行 / 每列的 1 的个数（抽样打印）
    row_weights = parity_check_matrix.sum(axis=1)
    column_weights = parity_check_matrix.sum(axis=0)

    print(f"========== {name} ==========")
    print(f"  num_qubits (n)          = {num_qubits}")
    print(f"  num_checks (Z)          = {num_checks}")
    print(f"  rank(H_Z)               = {rank_h_z}")
    print(f"  Z stabilizer 冗余个数    = {num_z_redundancies}")
    print(f"  num_checks (X)          = {x_check_matrix.shape[0]}")
    print(f"  rank(H_X)               = {rank_h_x}")
    print(f"  X stabilizer 冗余个数    = {num_x_redundancies}")
    print(f"  k (按公式计算)           = {k_computed}")
    print(f"  k (由 dual_logical_z_basis 给出) = {num_logical_qubits}")
    print(f"  与预期 k = {expected_num_logical_qubits} 一致：{k_computed == expected_num_logical_qubits == num_logical_qubits}")
    print(f"  每行 H_Z 的 weight（应为 4）：min={row_weights.min()} max={row_weights.max()}")
    print(f"  每列 H_Z 的 weight：min={column_weights.min()} max={column_weights.max()}")
    print(f"  dual_logical_z_basis 各行 weight = {dual_logical_z_basis.sum(axis=1).tolist()}")
    print(f"  dual_logical_z_basis 每行是否都在 ker H_X 中：{all_in_kernel}")
    print()


if __name__ == "__main__":
    # 2D 例子：L = 4 足够看到 rank/冗余结构
    L_2d = 4
    h_z_2d, logical_z_2d = build_2d_toric_code(L_2d)
    h_x_2d = build_2d_toric_x_check_matrix(L_2d)
    verify_and_report(f"2D toric code, L={L_2d}",
                      h_z_2d, logical_z_2d, h_x_2d,
                      expected_num_logical_qubits=2)

    # 3D 例子：L = 3 足够看到 cube 冗余（每 cube 贡献一个 Z 冗余）
    L_3d = 3
    h_z_3d, logical_z_3d = build_3d_toric_code(L_3d)
    h_x_3d = build_3d_toric_x_check_matrix(L_3d)
    verify_and_report(f"3D toric code, L={L_3d}",
                      h_z_3d, logical_z_3d, h_x_3d,
                      expected_num_logical_qubits=3)

    # 也看一下 L=2 的 3D（最小 non-trivial 例子）
    L_3d_tiny = 2
    h_z_3d_tiny, logical_z_3d_tiny = build_3d_toric_code(L_3d_tiny)
    h_x_3d_tiny = build_3d_toric_x_check_matrix(L_3d_tiny)
    verify_and_report(f"3D toric code, L={L_3d_tiny}",
                      h_z_3d_tiny, logical_z_3d_tiny, h_x_3d_tiny,
                      expected_num_logical_qubits=3)
