import numpy as np


class SyndromeRepresentativeSection:
    """
    Syndrome-to-chain representative map r: im(H_Z) -> C^1.

    BP-LSD is used when available.  The Gaussian-elimination section remains a
    strict fallback and is also used to validate or repair failed decoder output.
    """

    def __init__(self, parity_check_matrix, prefer_bplsd=True):
        self.parity_check_matrix = np.asarray(parity_check_matrix, dtype=bool)
        self.parity_check_matrix_uint8 = self.parity_check_matrix.astype(
            np.uint8
        )
        self.num_checks, self.num_qubits = self.parity_check_matrix.shape
        self.fallback_linear_section_data = build_linear_section(
            self.parity_check_matrix
        )
        self.decoder = None
        self.backend_name = "linear_elimination_fallback"
        self.fallback_count = 0
        self.decoder_failure_count = 0
        self.ldpc_import_error = None
        self.apply_count = 0
        self.cache_hit_count = 0
        self.cache = {}

        if prefer_bplsd:
            self._try_build_bplsd_decoder()

    def _try_build_bplsd_decoder(self):
        try:
            try:
                from ldpc import BpLsdDecoder
            except ImportError:
                from ldpc.bplsd_decoder import BpLsdDecoder
        except Exception as exc:  # pragma: no cover - optional dependency
            self.ldpc_import_error = repr(exc)
            return

        decoder_kwargs_candidates = [
            {
                "error_rate": 0.05,
                "bp_method": "product_sum",
                "max_iter": max(10, self.num_qubits),
                "schedule": "serial",
                "lsd_method": "lsd_cs",
                "lsd_order": 0,
            },
            {
                "error_rate": 0.05,
                "bp_method": "ms",
                "max_iter": max(10, self.num_qubits),
                "lsd_method": "lsd_cs",
                "lsd_order": 0,
            },
            {},
        ]
        for decoder_kwargs in decoder_kwargs_candidates:
            try:
                self.decoder = BpLsdDecoder(
                    self.parity_check_matrix.astype(np.uint8),
                    **decoder_kwargs,
                )
                self.backend_name = "bplsd"
                return
            except Exception as exc:  # pragma: no cover - version dependent
                self.ldpc_import_error = repr(exc)
        self.decoder = None

    def _validate_chain(self, syndrome_bits, chain_bits):
        recovered_syndrome = (
            self.parity_check_matrix_uint8 @ chain_bits.astype(np.uint8)
        ) % 2
        return np.array_equal(recovered_syndrome.astype(bool), syndrome_bits)

    def _fallback(self, syndrome_bits):
        self.fallback_count += 1
        chain_bits = apply_linear_section(
            syndrome_bits,
            self.fallback_linear_section_data,
        )
        if not self._validate_chain(syndrome_bits, chain_bits):
            raise ValueError("section fallback failed: H_Z r(s) != s")
        return chain_bits

    def apply(self, syndrome_bits):
        syndrome_bits = np.asarray(syndrome_bits, dtype=bool)
        if syndrome_bits.shape != (self.num_checks,):
            raise ValueError(
                f"syndrome_bits must have shape ({self.num_checks},)"
            )
        self.apply_count += 1
        cache_key = np.packbits(syndrome_bits.astype(np.uint8)).tobytes()
        cached_chain_bits = self.cache.get(cache_key)
        if cached_chain_bits is not None:
            self.cache_hit_count += 1
            return cached_chain_bits.copy()

        if self.decoder is None:
            chain_bits = self._fallback(syndrome_bits)
            self.cache[cache_key] = chain_bits.copy()
            return chain_bits

        try:
            decoded = self.decoder.decode(syndrome_bits.astype(np.uint8))
            chain_bits = np.asarray(decoded, dtype=np.uint8).reshape(-1) % 2
            if chain_bits.shape != (self.num_qubits,):
                raise ValueError("BpLsdDecoder returned wrong vector length")
            chain_bits = chain_bits.astype(bool)
            if self._validate_chain(syndrome_bits, chain_bits):
                self.cache[cache_key] = chain_bits.copy()
                return chain_bits
        except Exception:
            pass

        self.decoder_failure_count += 1
        chain_bits = self._fallback(syndrome_bits)
        self.cache[cache_key] = chain_bits.copy()
        return chain_bits

    def stats(self):
        return {
            "backend_name": self.backend_name,
            "apply_count": int(self.apply_count),
            "cache_hit_count": int(self.cache_hit_count),
            "cache_size": int(len(self.cache)),
            "fallback_count": int(self.fallback_count),
            "decoder_failure_count": int(self.decoder_failure_count),
            "ldpc_import_error": self.ldpc_import_error,
        }


def build_syndrome_representative_section(parity_check_matrix,
                                          prefer_bplsd=True):
    return SyndromeRepresentativeSection(
        parity_check_matrix=parity_check_matrix,
        prefer_bplsd=prefer_bplsd,
    )


def apply_section(syndrome_bits, section_data):
    if isinstance(section_data, SyndromeRepresentativeSection):
        return section_data.apply(syndrome_bits)
    return apply_linear_section(syndrome_bits, section_data)


def build_linear_section(parity_check_matrix):
    """
    对 H_Z 做 GF(2) 高斯消元 + 列置换，得到 E · H_Z · Π = [[I_ρ, A], [0, 0]]。
    以 XOR 脚本形式记录 E，以排列数组形式记录 Π。

    输入：
      parity_check_matrix: np.ndarray，shape (num_checks, num_qubits)，dtype=bool

    输出：
      linear_section_data: dict
        - row_xor_steps: list[tuple]，长度为若干步
        - column_permutation: np.ndarray，shape (num_qubits,)，dtype=int32
        - rank: int
        - num_checks: int
        - num_qubits: int
    """
    num_checks, num_qubits = parity_check_matrix.shape

    # 工作矩阵会被原地消元，避免改动输入
    working_matrix = parity_check_matrix.copy().astype(bool)
    column_permutation = np.arange(num_qubits, dtype=np.int32)
    row_xor_steps = []

    current_pivot = 0
    while current_pivot < min(num_checks, num_qubits):
        pivot_row_index = -1
        pivot_col_index = -1

        # 在右下子矩阵中寻找一个主元 1
        for row_index in range(current_pivot, num_checks):
            nonzero_columns = np.flatnonzero(
                working_matrix[row_index, current_pivot:]
            )
            if nonzero_columns.size > 0:
                pivot_row_index = row_index
                pivot_col_index = current_pivot + int(nonzero_columns[0])
                break

        if pivot_row_index == -1:
            break

        # 先交换行，并把操作记入 E 的脚本
        if pivot_row_index != current_pivot:
            working_matrix[[current_pivot, pivot_row_index]] = working_matrix[
                [pivot_row_index, current_pivot]
            ]
            row_xor_steps.append(("swap_rows", current_pivot, pivot_row_index))

        # 再交换列，只更新矩阵和列排列
        if pivot_col_index != current_pivot:
            working_matrix[:, [current_pivot, pivot_col_index]] = working_matrix[
                :, [pivot_col_index, current_pivot]
            ]
            column_permutation[[current_pivot, pivot_col_index]] = (
                column_permutation[[pivot_col_index, current_pivot]]
            )

        # 把主元列其余位置全部消成 0
        for other_row_index in range(num_checks):
            if other_row_index == current_pivot:
                continue
            if working_matrix[other_row_index, current_pivot]:
                working_matrix[other_row_index] ^= working_matrix[current_pivot]
                row_xor_steps.append(
                    ("xor_rows", other_row_index, current_pivot)
                )

        current_pivot += 1

    linear_section_data = {
        "row_xor_steps": row_xor_steps,
        "column_permutation": column_permutation,
        "rank": current_pivot,
        "num_checks": num_checks,
        "num_qubits": num_qubits,
    }
    return linear_section_data


def apply_linear_section(syndrome_bits, linear_section_data):
    """
    计算 r(σ) = Π · [σ_hat; 0]，其中 σ_hat 是 (E · σ) 的前 ρ 位。

    输入：
      syndrome_bits: np.ndarray，shape (num_checks,)，dtype=bool
      linear_section_data: dict

    输出：
      chain_bits: np.ndarray，shape (num_qubits,)，dtype=bool
    """
    rank = linear_section_data["rank"]
    num_qubits = linear_section_data["num_qubits"]
    row_xor_steps = linear_section_data["row_xor_steps"]
    column_permutation = linear_section_data["column_permutation"]

    # 依次应用 E 的行操作，得到 E · syndrome_bits
    transformed_syndrome_bits = syndrome_bits.copy().astype(bool)
    for operation_name, first_index, second_index in row_xor_steps:
        if operation_name == "swap_rows":
            temporary_value = transformed_syndrome_bits[first_index]
            transformed_syndrome_bits[first_index] = transformed_syndrome_bits[
                second_index
            ]
            transformed_syndrome_bits[second_index] = temporary_value
        else:
            transformed_syndrome_bits[first_index] ^= transformed_syndrome_bits[
                second_index
            ]

    # 取前 rank 位并补零到长度 num_qubits
    padded_chain_bits = np.zeros(num_qubits, dtype=bool)
    padded_chain_bits[:rank] = transformed_syndrome_bits[:rank]

    # 按 Π 的约定恢复到原始列顺序
    chain_bits = np.zeros(num_qubits, dtype=bool)
    chain_bits[column_permutation] = padded_chain_bits
    return chain_bits


def apply_linear_section_transpose(chain_like_vector, linear_section_data):
    """
    计算 r^T · v。

    输入：
      chain_like_vector: np.ndarray，shape (num_qubits,)，dtype=bool
      linear_section_data: dict

    输出：
      syndrome_like_vector: np.ndarray，shape (num_checks,)，dtype=bool
    """
    rank = linear_section_data["rank"]
    num_checks = linear_section_data["num_checks"]
    row_xor_steps = linear_section_data["row_xor_steps"]
    column_permutation = linear_section_data["column_permutation"]

    # 先施加 Π^T，把向量变到消元后的列坐标
    intermediate_chain_bits = chain_like_vector[column_permutation]

    # 再施加嵌入矩阵的转置，只保留前 rank 位
    intermediate_syndrome_bits = np.zeros(num_checks, dtype=bool)
    intermediate_syndrome_bits[:rank] = intermediate_chain_bits[:rank]

    # 最后逆序施加 E^T
    syndrome_like_vector = intermediate_syndrome_bits
    for operation_name, first_index, second_index in reversed(row_xor_steps):
        if operation_name == "swap_rows":
            temporary_value = syndrome_like_vector[first_index]
            syndrome_like_vector[first_index] = syndrome_like_vector[second_index]
            syndrome_like_vector[second_index] = temporary_value
        else:
            syndrome_like_vector[second_index] ^= syndrome_like_vector[first_index]

    return syndrome_like_vector


def verify_linear_section(parity_check_matrix, linear_section_data,
                          num_random_tests=20, rng=None):
    """
    随机检查 section 与其转置是否正确。

    输入：
      parity_check_matrix: np.ndarray，shape (num_checks, num_qubits)，dtype=bool
      linear_section_data: dict
      num_random_tests: int
      rng: np.random.Generator 或 None

    输出：
      无返回值；若发现错误则抛出 AssertionError
    """
    if rng is None:
        rng = np.random.default_rng(0)

    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    num_checks, num_qubits = parity_check_matrix.shape

    for _ in range(num_random_tests):
        # 先造一个保证在 im(H_Z) 里的 syndrome
        random_chain_bits = rng.integers(0, 2, size=num_qubits).astype(bool)
        syndrome_in_image = (
            parity_check_matrix_uint8 @ random_chain_bits.astype(np.uint8)
        ) % 2
        syndrome_in_image = syndrome_in_image.astype(bool)

        recovered_chain = apply_linear_section(
            syndrome_in_image,
            linear_section_data,
        )
        recovered_syndrome = (
            parity_check_matrix_uint8 @ recovered_chain.astype(np.uint8)
        ) % 2
        assert np.array_equal(
            syndrome_in_image,
            recovered_syndrome.astype(bool),
        ), "section 破坏: H_Z r(σ) ≠ σ"

        # 再检查转置配对关系
        random_chain_like_vector = rng.integers(0, 2, size=num_qubits).astype(bool)
        left_side = bool(
            np.bitwise_xor.reduce(random_chain_like_vector & recovered_chain)
        )
        recovered_syndrome_like_vector = apply_linear_section_transpose(
            random_chain_like_vector,
            linear_section_data,
        )
        right_side = bool(
            np.bitwise_xor.reduce(
                recovered_syndrome_like_vector & syndrome_in_image
            )
        )
        assert left_side == right_side, "转置关系破坏: <v, r(σ)> ≠ <r^T v, σ>"


if __name__ == "__main__":
    random_number_generator = np.random.default_rng(20260420)

    # 用固定 seed 重复采样，直到拿到一个 GF(2) 满秩的 (8, 12) 矩阵
    while True:
        parity_check_matrix = random_number_generator.integers(
            0,
            2,
            size=(8, 12),
            dtype=np.int64,
        ).astype(bool)
        linear_section_data = build_linear_section(parity_check_matrix)
        if linear_section_data["rank"] == 8:
            break

    verify_linear_section(
        parity_check_matrix,
        linear_section_data,
        num_random_tests=20,
        rng=np.random.default_rng(0),
    )

    print(f"rank: {linear_section_data['rank']}")
    print(f"row_xor_steps length: {len(linear_section_data['row_xor_steps'])}")
    print(
        "column_permutation:",
        linear_section_data["column_permutation"],
    )
