import numpy as np

from build_toric_code_examples import (
    build_2d_toric_code,
    build_3d_toric_code,
)
from linear_section import (
    apply_linear_section,
    build_linear_section,
    verify_linear_section,
)


def build_checks_touching_each_qubit(parity_check_matrix):
    """
    把 H_Z 按列展开成邻接表。

    输入：
      parity_check_matrix: np.ndarray，shape (num_checks, num_qubits)，dtype=bool

    输出：
      checks_touching_each_qubit: list[np.ndarray]
        长度为 num_qubits
        第 i 个元素 shape (num_checks_touching_qubit_i,)，dtype=int32
    """
    num_qubits = parity_check_matrix.shape[1]
    checks_touching_each_qubit = []

    for qubit_index in range(num_qubits):
        touching_check_indices = np.flatnonzero(
            parity_check_matrix[:, qubit_index]
        ).astype(np.int32)
        checks_touching_each_qubit.append(touching_check_indices)

    return checks_touching_each_qubit


def build_logical_observable_masks(
        parity_check_matrix,
        dual_logical_z_basis,
        linear_section_data):
    """
    构造所有非零 u 对应的 dual logical representative \\bar z_u。

    历史上这个函数返回线性化 observable mask。r 修正为任意 section
    之后，真正的 observable 必须在采样时按
    c + eta + r(H_Z c) + r(H_Z eta) 计算。为减少调用面变更，
    函数名暂时保留，但返回值语义已经是 \\bar z_u 向量集合。

    输入：
      parity_check_matrix: np.ndarray，shape (num_checks, num_qubits)，dtype=bool
      dual_logical_z_basis: np.ndarray，shape (num_logical_qubits, num_qubits)，dtype=bool
      linear_section_data: dict，保留兼容参数；不再用于构造 mask

    输出：
      logical_observable_masks: np.ndarray，
        shape (2**num_logical_qubits - 1, num_qubits)，dtype=bool
    """
    del parity_check_matrix, linear_section_data
    num_logical_qubits, num_qubits = dual_logical_z_basis.shape
    num_masks = (1 << num_logical_qubits) - 1

    logical_observable_masks = np.zeros((num_masks, num_qubits), dtype=bool)

    # 对每个非零 u，按位异或组合 primitive dual logical vectors。
    for mask_index in range(1, num_masks + 1):
        combined_mask = np.zeros(num_qubits, dtype=bool)
        for logical_qubit_index in range(num_logical_qubits):
            if (mask_index >> logical_qubit_index) & 1:
                combined_mask ^= dual_logical_z_basis[logical_qubit_index]
        logical_observable_masks[mask_index - 1] = combined_mask

    return logical_observable_masks


def _run_preprocessing_self_check(
        code_family_name,
        parity_check_matrix,
        dual_logical_z_basis,
        lattice_size,
        seed):
    random_number_generator = np.random.default_rng(seed)
    linear_section_data = build_linear_section(parity_check_matrix)
    verify_linear_section(
        parity_check_matrix,
        linear_section_data,
        num_random_tests=20,
        rng=np.random.default_rng(seed + 1),
    )

    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix,
        dual_logical_z_basis,
        linear_section_data,
    )

    num_qubits = parity_check_matrix.shape[1]
    for qubit_index in range(num_qubits):
        expected_touching_checks = np.flatnonzero(
            parity_check_matrix[:, qubit_index]
        ).astype(np.int32)
        assert np.array_equal(
            checks_touching_each_qubit[qubit_index],
            expected_touching_checks,
        ), f"{code_family_name} 邻接表错误: qubit_index={qubit_index}"

    num_logical_qubits = dual_logical_z_basis.shape[0]
    num_masks = logical_observable_masks.shape[0]
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)

    z_bar_vectors = np.zeros_like(logical_observable_masks)
    for mask_index in range(1, num_masks + 1):
        combined_logical_z_vector = np.zeros(num_qubits, dtype=bool)
        for logical_qubit_index in range(num_logical_qubits):
            if (mask_index >> logical_qubit_index) & 1:
                combined_logical_z_vector ^= dual_logical_z_basis[
                    logical_qubit_index
                ]
        z_bar_vectors[mask_index - 1] = combined_logical_z_vector

    for _ in range(20):
        random_chain_bits = random_number_generator.integers(
            0,
            2,
            size=num_qubits,
        ).astype(bool)
        random_syndrome_bits = (
            parity_check_matrix_uint8 @ random_chain_bits.astype(np.uint8)
        ) % 2
        recovered_chain_bits = apply_linear_section(
            random_syndrome_bits.astype(bool),
            linear_section_data,
        )
        gauge_representative_bits = random_chain_bits ^ recovered_chain_bits

        for mask_index in range(num_masks):
            logical_observable_parity = bool(
                np.bitwise_xor.reduce(
                    logical_observable_masks[mask_index]
                    & gauge_representative_bits
                )
            )
            logical_z_parity = bool(
                np.bitwise_xor.reduce(
                    z_bar_vectors[mask_index] & gauge_representative_bits
                )
            )
            assert logical_observable_parity == logical_z_parity, (
                f"{code_family_name} 逻辑观测量掩码 gauge 等价性失败: "
                f"mask_index={mask_index}"
            )

    mask_hamming_weights = logical_observable_masks.astype(np.int32).sum(axis=1)
    print(
        f"{code_family_name} L={lattice_size}: "
        f"num_logical_qubits={num_logical_qubits}, "
        f"logical_observable_masks.shape={logical_observable_masks.shape}, "
        f"mask_hamming_weights={mask_hamming_weights}"
    )


if __name__ == "__main__":
    parity_check_matrix_2d, dual_logical_z_basis_2d = build_2d_toric_code(
        lattice_size=3
    )
    parity_check_matrix_3d_l2, dual_logical_z_basis_3d_l2 = (
        build_3d_toric_code(lattice_size=2)
    )
    parity_check_matrix_3d_l3, dual_logical_z_basis_3d_l3 = (
        build_3d_toric_code(lattice_size=3)
    )
    _run_preprocessing_self_check(
        code_family_name="2D toric",
        parity_check_matrix=parity_check_matrix_2d,
        dual_logical_z_basis=dual_logical_z_basis_2d,
        lattice_size=3,
        seed=0,
    )
    _run_preprocessing_self_check(
        code_family_name="3D toric",
        parity_check_matrix=parity_check_matrix_3d_l2,
        dual_logical_z_basis=dual_logical_z_basis_3d_l2,
        lattice_size=2,
        seed=10,
    )
    _run_preprocessing_self_check(
        code_family_name="3D toric",
        parity_check_matrix=parity_check_matrix_3d_l3,
        dual_logical_z_basis=dual_logical_z_basis_3d_l3,
        lattice_size=3,
        seed=20,
    )
