"""Syndrome section r: im(H) → F_2^n（G2.1 线性部分；G2.2 增补 BpLsd 与防误用）。

线性 section 构造（确定性）：
  取 H 的 RREF 主元列集 P（|P| = r = rank H）；M := H[:, P] 满列秩且
  im(M) = im(H)。对 [M | I] 做 RREF 得顶部 r 行 [I_r | T]，则对 σ ∈ im(H)：
      r(σ) := embed_P(T σ)   （支撑在主元列上的唯一解）
  底部 (m_c − r) 行 [0 | N] 给出隶属检验：σ ∈ im(H) ⇔ N σ = 0。

性质（tests 验证）：H r(σ) = σ ∀σ ∈ im(H)；r 线性；r(H e_j) 可批量算出
（supply W = Z ⊕ Z·(r∘H) 的构造）。

⚠ 防误用（plan §6 风险 10）：q>0 时观测到的 s 不一定 ∈ im(H)——apply 前
必须 in_image 检查（strict 模式自动检查并抛错）。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import as_gf2_matrix, as_gf2_vector, gf2_matmul, gf2_row_echelon


@dataclass
class LinearSection:
    num_checks: int
    num_qubits: int
    rank: int
    pivot_columns: list        # H 的主元列（长度 r）
    solve_matrix: np.ndarray   # T (r × m_c)：x_pivots = T σ
    membership_matrix: np.ndarray  # N ((m_c−r) × m_c)：σ∈im ⇔ Nσ=0

    def in_image(self, syndrome):
        syndrome = as_gf2_vector(syndrome)
        if self.membership_matrix.shape[0] == 0:
            return True
        return not gf2_matmul(self.membership_matrix, syndrome[:, None]).any()

    def apply(self, syndrome, strict=True):
        """r(σ)。strict=True 时 σ ∉ im(H) 抛 ValueError（防对观测 s 误用）。"""
        syndrome = as_gf2_vector(syndrome)
        if syndrome.shape[0] != self.num_checks:
            raise ValueError("syndrome length mismatch")
        if strict and not self.in_image(syndrome):
            raise ValueError(
                "syndrome is not in im(H); refusing to apply section "
                "(likely misuse: taking a section of a noisy observed syndrome)"
            )
        chain = np.zeros(self.num_qubits, dtype=np.uint8)
        pivot_values = gf2_matmul(self.solve_matrix, syndrome[:, None])[:, 0]
        chain[self.pivot_columns] = pivot_values
        return chain

    def section_after_H(self, parity_check_matrix):
        """批量 r(H e_j)：返回 n×n 矩阵，其第 j 列 = r(H e_j)。"""
        parity_check_matrix = as_gf2_matrix(parity_check_matrix)
        pivot_values = gf2_matmul(self.solve_matrix, parity_check_matrix)  # r × n
        result = np.zeros(
            (self.num_qubits, parity_check_matrix.shape[1]), dtype=np.uint8
        )
        result[self.pivot_columns, :] = pivot_values
        return result

    def fingerprint(self):
        import hashlib

        # 注意：主元列索引可 >255（n≥100 的码），必须用定宽整型序列化
        payload = (
            np.asarray(self.pivot_columns, dtype=np.int64).tobytes()
            + np.ascontiguousarray(self.solve_matrix).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()


def build_linear_section(parity_check_matrix, column_priority=None):
    """线性 section。column_priority（列的置换）改变主元列选择 ⇒ 得到不同的
    线性 frame（G3.3 frame A/B 用）；None 时按自然列序（默认 frame）。"""
    parity_check_matrix = as_gf2_matrix(parity_check_matrix)
    num_checks, num_qubits = parity_check_matrix.shape
    if column_priority is None:
        _, pivot_columns = gf2_row_echelon(parity_check_matrix)
    else:
        column_priority = list(column_priority)
        if sorted(column_priority) != list(range(num_qubits)):
            raise ValueError("column_priority must be a permutation of columns")
        _, permuted_pivots = gf2_row_echelon(
            parity_check_matrix[:, column_priority]
        )
        pivot_columns = sorted(column_priority[p] for p in permuted_pivots)
    rank = len(pivot_columns)
    pivot_block = parity_check_matrix[:, pivot_columns]  # m_c × r，满列秩
    augmented = np.concatenate(
        [pivot_block, np.eye(num_checks, dtype=np.uint8)], axis=1
    )
    rref, aug_pivots = gf2_row_echelon(augmented)
    if aug_pivots[:rank] != list(range(rank)):
        raise AssertionError("pivot block lost full column rank during RREF")
    solve_matrix = rref[:rank, rank:].copy()
    membership_matrix = rref[rank:, rank:].copy()
    section = LinearSection(
        num_checks=num_checks,
        num_qubits=num_qubits,
        rank=rank,
        pivot_columns=list(pivot_columns),
        solve_matrix=solve_matrix,
        membership_matrix=membership_matrix,
    )
    return section


class DecoderSection:
    """BpLsd 解码器 section（非线性 frame；G2.2）。

    仅用于 frame A/B 对照（G3.3）与低权重代表需求——生产观测路径是线性 frame
    （W mask），本类不进采样热循环。行为对齐主项目 SyndromeRepresentativeSection，
    但修复两点（notes/00 §3）：缓存有上限；strict 防误用（σ∉im 拒绝）用精确
    隶属矩阵而非试错。
    """

    def __init__(self, parity_check_matrix, prefer_bplsd=True, cache_limit=100000):
        self.parity_check_matrix = as_gf2_matrix(parity_check_matrix)
        self.num_checks, self.num_qubits = self.parity_check_matrix.shape
        self.linear_fallback = build_linear_section(self.parity_check_matrix)
        self.decoder = None
        self.backend_name = "linear_elimination_fallback"
        self.ldpc_import_error = None
        self.cache = {}
        self.cache_limit = int(cache_limit)
        self.apply_count = 0
        self.cache_hit_count = 0
        self.fallback_count = 0
        self.decoder_failure_count = 0
        if prefer_bplsd:
            self._try_build_bplsd()

    def _try_build_bplsd(self):
        try:
            try:
                from ldpc import BpLsdDecoder
            except ImportError:
                from ldpc.bplsd_decoder import BpLsdDecoder
        except Exception as error:  # pragma: no cover - optional dependency
            self.ldpc_import_error = repr(error)
            return
        for kwargs in (
            {
                "error_rate": 0.05,
                "bp_method": "product_sum",
                "max_iter": max(10, self.num_qubits),
                "schedule": "serial",
                "lsd_method": "lsd_cs",
                "lsd_order": 0,
            },
            {"error_rate": 0.05, "bp_method": "ms",
             "max_iter": max(10, self.num_qubits),
             "lsd_method": "lsd_cs", "lsd_order": 0},
            {},
        ):
            try:
                self.decoder = BpLsdDecoder(
                    self.parity_check_matrix.astype(np.uint8), **kwargs
                )
                self.backend_name = "bplsd"
                return
            except Exception as error:  # pragma: no cover - version dependent
                self.ldpc_import_error = repr(error)
        self.decoder = None

    def in_image(self, syndrome):
        return self.linear_fallback.in_image(syndrome)

    def _validate(self, syndrome, chain):
        recovered = gf2_matmul(self.parity_check_matrix, chain[:, None])[:, 0]
        return np.array_equal(recovered, syndrome)

    def apply(self, syndrome, strict=True):
        syndrome = as_gf2_vector(syndrome)
        if syndrome.shape[0] != self.num_checks:
            raise ValueError("syndrome length mismatch")
        if strict and not self.in_image(syndrome):
            raise ValueError(
                "syndrome is not in im(H); refusing to apply section "
                "(likely misuse: taking a section of a noisy observed syndrome)"
            )
        self.apply_count += 1
        key = np.packbits(syndrome).tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hit_count += 1
            return cached.copy()

        chain = None
        if self.decoder is not None:
            try:
                decoded = self.decoder.decode(syndrome.astype(np.uint8))
                candidate = (
                    np.asarray(decoded, dtype=np.uint8).reshape(-1) % 2
                )
                if candidate.shape == (self.num_qubits,) and self._validate(
                        syndrome, candidate):
                    chain = candidate
                else:
                    self.decoder_failure_count += 1
            except Exception:
                self.decoder_failure_count += 1
        if chain is None:
            self.fallback_count += 1
            chain = self.linear_fallback.apply(syndrome, strict=False)
            if not self._validate(syndrome, chain):
                raise AssertionError("linear fallback violated H r(σ) = σ")
        if len(self.cache) < self.cache_limit:
            self.cache[key] = chain.copy()
        return chain

    def stats(self):
        return {
            "backend_name": self.backend_name,
            "apply_count": self.apply_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_size": len(self.cache),
            "cache_limit": self.cache_limit,
            "fallback_count": self.fallback_count,
            "decoder_failure_count": self.decoder_failure_count,
            "ldpc_import_error": self.ldpc_import_error,
        }

    def fingerprint(self):
        import hashlib

        payload = (
            f"decoder_section:{self.backend_name}".encode()
            + np.ascontiguousarray(self.parity_check_matrix).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()


class DecoderObservableFrame:
    """decoder frame 的标签器：φ_dec(v) = ⟨z_u, v ⊕ r_dec(H v)⟩（逐样本解码）。

    仅供 G3.3 frame A/B 与小码对照；接口与 ObservableFrame.label_of 兼容。
    """

    def __init__(self, H_check, logical_obs_basis, decoder_section):
        self.H_check = as_gf2_matrix(H_check)
        self.Z = as_gf2_matrix(logical_obs_basis)
        self.k = self.Z.shape[0]
        self.section = decoder_section

    def label_of(self, v):
        v = as_gf2_vector(v)
        syndrome = gf2_matmul(self.H_check, v[:, None])[:, 0]
        representative = self.section.apply(syndrome, strict=False)
        closed = v ^ representative
        if self.k == 0:
            return np.zeros(0, dtype=np.uint8)
        return gf2_matmul(self.Z, closed[:, None])[:, 0]
