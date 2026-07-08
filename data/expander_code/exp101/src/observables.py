"""观测量层（G2.1）：线性 frame W、三档 u 集、m_u 与聚合量。

数学权威 notes/01 §1/§4：
    w_u = (I ⊕ R H)ᵀ z_u；φ(v)_u = ⟨w_u, v⟩；O_u = (−1)^{⟨u,ℓ_ref⟩+⟨w_u,v⟩}。
    w_u 三性质（构造后硬断言）：
      (i) 湮灭 T = im(R)   (ii) 湮灭 S = row(stabilizer)   (iii) ⟨w_u, x_v⟩=δ_uv。
三档 u 记录（plan §1.4）：basis k 个；k ≤ full_max_k 时全 2^k−1；否则
U_rand 均匀随机非零 u（seed 记录）。聚合：
    q_top_all = mean_{u≠0} m_u²（全量精确 / 抽样无偏）
    q_top_basis = mean_{basis} m_u²
    purity = 2^{−k}(1 + Σ_{u≠0} m_u²)（抽样时 2^{−k} + (1−2^{−k})·mean）
    w0 = 2^{−k}(1 + Σ_{u≠0} m_u)（同上）
"""

from dataclasses import dataclass, field

import numpy as np

from .gf2 import as_gf2_matrix, as_gf2_vector, gf2_matmul

DEFAULT_FULL_MAX_K = 10
DEFAULT_NUM_RANDOM_U = 64


@dataclass
class ObservableFrame:
    """线性 section frame 下的 W 基（k×n）与指纹。"""

    W_basis: np.ndarray
    k: int
    num_qubits: int
    section_fingerprint: str

    def label_of(self, v):
        """φ(v) ∈ F_2^k。"""
        v = as_gf2_vector(v)
        if self.k == 0:
            return np.zeros(0, dtype=np.uint8)
        return gf2_matmul(self.W_basis, v[:, None])[:, 0]


def build_observable_frame(model):
    """由 SectorModel 构造 W 并验证三性质（失败即 AssertionError）。"""
    Z = as_gf2_matrix(model.logical_obs_basis)
    k = Z.shape[0]
    if k == 0:
        return ObservableFrame(
            W_basis=np.zeros((0, model.num_qubits), dtype=np.uint8),
            k=0,
            num_qubits=model.num_qubits,
            section_fingerprint=model.section.fingerprint(),
        )
    RH = model.section.section_after_H(model.H_check)  # n×n：第 j 列 = r(H e_j)
    W = (Z ^ gf2_matmul(Z, RH)).astype(np.uint8)

    # (i) 湮灭 T = im(R)：⟨w_u, r(σ)⟩ = 0。im(R) 由 r(基 im 元素) 张成；
    #     用 r(H e_j) 的列（张满 im(R)，因 H e_j 张满 im(H)）检验。
    if gf2_matmul(W, RH).any():
        raise AssertionError("w_u fails to annihilate T = im(R)")
    # (ii) 湮灭 S = row(stabilizer)
    if gf2_matmul(W, as_gf2_matrix(model.stabilizer_rows).T).any():
        raise AssertionError("w_u fails to annihilate stabilizer rows")
    # (iii) L 坐标：⟨w_u, x_v⟩ = δ_uv
    pairing = gf2_matmul(W, as_gf2_matrix(model.logical_move_basis).T)
    if not np.array_equal(pairing, np.eye(k, dtype=np.uint8)):
        raise AssertionError("w_u fails L-coordinate pairing with move basis")

    return ObservableFrame(
        W_basis=W,
        k=k,
        num_qubits=model.num_qubits,
        section_fingerprint=model.section.fingerprint(),
    )


@dataclass
class ObservableSet:
    """观测的 u 集合：每行一个 u（bitmask 表示基组合）与对应 w_u 行。"""

    tier: str                 # "full" | "sampled" | "basis_only"
    k: int
    u_bitmasks: np.ndarray    # (num_u,) int64，非零
    W_rows: np.ndarray        # (num_u, n) uint8
    basis_positions: np.ndarray  # basis u=e_i 在列表中的下标（长度 k）
    u_rand_seed: object = None
    num_random_u: int = 0

    @property
    def num_u(self):
        return int(self.u_bitmasks.shape[0])


def build_observable_set(frame, full_max_k=DEFAULT_FULL_MAX_K,
                         num_random_u=DEFAULT_NUM_RANDOM_U, u_rand_seed=None):
    """三档 u 集。k ≤ full_max_k → 全部 2^k−1；否则 basis + U_rand（去重、非零、
    不与 basis 重复）。k=0 返回空集。"""
    k = frame.k
    if k == 0:
        return ObservableSet(
            tier="basis_only", k=0,
            u_bitmasks=np.zeros(0, dtype=np.int64),
            W_rows=np.zeros((0, frame.num_qubits), dtype=np.uint8),
            basis_positions=np.zeros(0, dtype=np.int64),
        )

    def rows_for(bitmasks):
        rows = np.zeros((len(bitmasks), frame.num_qubits), dtype=np.uint8)
        for row_index, bitmask in enumerate(bitmasks):
            for bit in range(k):
                if (bitmask >> bit) & 1:
                    rows[row_index] ^= frame.W_basis[bit]
        return rows

    basis_masks = [1 << i for i in range(k)]
    if k <= full_max_k:
        u_list = list(range(1, 1 << k))
        tier = "full"
        seed_used = None
        num_rand = 0
    else:
        if u_rand_seed is None:
            raise ValueError("k > full_max_k requires explicit u_rand_seed")
        num_random_u = int(num_random_u)
        available = (1 << k) - 1 - k  # 非零且非 basis 的 u 总数
        if num_random_u > available:
            raise ValueError(
                f"num_random_u={num_random_u} exceeds available distinct "
                f"non-basis u count {available} for k={k}; lower num_random_u "
                "or use the full tier"
            )
        rng = np.random.default_rng(u_rand_seed)
        chosen = set()
        basis_set = set(basis_masks)
        max_draws = 1000 * max(num_random_u, 1)  # 防御上限（可行时远够）
        draws = 0
        while len(chosen) < num_random_u:
            draws += 1
            if draws > max_draws:
                raise RuntimeError(
                    "rejection sampling for U_rand exceeded defensive draw "
                    "limit; this should be unreachable for feasible requests"
                )
            candidate = int(rng.integers(1, 1 << k))
            if candidate in basis_set or candidate in chosen:
                continue
            chosen.add(candidate)
        u_list = basis_masks + sorted(chosen)
        tier = "sampled"
        seed_used = int(u_rand_seed)
        num_rand = int(num_random_u)

    u_bitmasks = np.asarray(u_list, dtype=np.int64)
    basis_positions = np.asarray(
        [u_list.index(mask) for mask in basis_masks], dtype=np.int64
    )
    return ObservableSet(
        tier=tier, k=k, u_bitmasks=u_bitmasks, W_rows=rows_for(u_list),
        basis_positions=basis_positions, u_rand_seed=seed_used,
        num_random_u=num_rand,
    )


def observable_values(observable_set, wiring, v):
    """单个构型 v 的 O_u ∈ {+1,−1}^{num_u}（含 ℓ_ref 符号）。"""
    v = as_gf2_vector(v)
    parities = gf2_matmul(observable_set.W_rows, v[:, None])[:, 0]
    reference = _reference_signs(observable_set, wiring)
    return ((1 - 2 * parities.astype(np.int8)) * reference).astype(np.int8)


def _reference_signs(observable_set, wiring):
    """(−1)^{⟨u, ℓ_ref⟩} 每 u 一个。"""
    label = np.asarray(wiring.reference_label, dtype=np.int64)
    label_mask = 0
    for bit, value in enumerate(label):
        if value:
            label_mask |= 1 << bit
    overlaps = np.bitwise_and(observable_set.u_bitmasks, np.int64(label_mask))
    parity = np.zeros(observable_set.num_u, dtype=np.int8)
    for index, overlap in enumerate(overlaps):
        parity[index] = int(overlap).bit_count() & 1
    return (1 - 2 * parity).astype(np.int8)


def aggregate_observables(observable_set, m_u_values):
    """由 m_u（与 u_bitmasks 对齐）计算聚合量（notes/01 §4 公式）。"""
    m_u_values = np.asarray(m_u_values, dtype=np.float64)
    if m_u_values.shape[0] != observable_set.num_u:
        raise ValueError("m_u length mismatch")
    k = observable_set.k
    if k == 0:
        return {"q_top_all": None, "q_top_basis": None, "purity": None, "w0": None}
    basis_m = m_u_values[observable_set.basis_positions]
    result = {"q_top_basis": float(np.mean(basis_m**2))}
    if observable_set.tier == "full":
        q_top_all = float(np.mean(m_u_values**2))
        sum_m = float(np.sum(m_u_values))
        sum_m2 = float(np.sum(m_u_values**2))
        result["q_top_all"] = q_top_all
        result["purity"] = (1.0 + sum_m2) / (1 << k)
        result["w0"] = (1.0 + sum_m) / (1 << k)
    else:
        random_mask = np.ones(observable_set.num_u, dtype=bool)
        random_mask[observable_set.basis_positions] = False
        random_m = m_u_values[random_mask]
        # 均匀非零 u 的无偏估计（basis 行不属均匀样本，不进平均）
        mean_m2 = float(np.mean(random_m**2)) if random_m.size else float("nan")
        mean_m = float(np.mean(random_m)) if random_m.size else float("nan")
        scale = 1.0 - 1.0 / (1 << k)
        result["q_top_all"] = mean_m2
        result["purity"] = 1.0 / (1 << k) + scale * mean_m2
        result["w0"] = 1.0 / (1 << k) + scale * mean_m
    return result
