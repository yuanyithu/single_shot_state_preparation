"""统计力学模型层（G2.1）：sector 装配、disorder、双系综接线、能量约定。

数学权威：notes/01_model_spec.md。

规范采样变量（canonical frame，两系综统一）：
    π(v) ∝ exp[ −K_p·|v| − K_q·|H v ⊕ σ_arg| ]，v ∈ F_2^n
    K_p = log((1−p)/p) ≥ 0（p<0.5），K_q 同理；q=0 时为硬约束 H v = σ_arg。
系综只差接线（validation/001 T1–T3 与 notes/01 §3 证明等价性）：
    - true_posterior：σ_arg = s = Hη⊕δ，参考标签 ℓ_ref = φ(η)
      （此时 v 就是候选错误 c，π = 标准 decoding posterior）
    - repo_compat  ：σ_arg = δ，       ℓ_ref = 0
      （等价于主项目模型 exp[−K_p|c⊕η|−K_q|Hc⊕s|] 的换元 u=c⊕η）
观测量：m_u = (−1)^{⟨u, ℓ_ref⟩} · ⟨(−1)^{⟨w_u, v⟩}⟩_π（线性 frame，见 observables.py）。

sector 约定：
    - "x_error"（生产默认，|0̄⟩ 制备）：H_check=H_Z；观测配对基=logical_Z；
      L-moves/sector 基=logical_X。
    - "z_error"（对偶，|+̄⟩ 制备）：H_check=H_X；观测配对基=logical_X；
      L-moves 基=logical_Z。
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .gf2 import as_gf2_matrix, gf2_matmul
from .section import LinearSection, build_linear_section

ENSEMBLES = ("true_posterior", "repo_compat")
SECTORS = ("x_error", "z_error")


def coupling_from_probability(probability):
    """K = log((1−p)/p)；p=0 → inf（调用方需走硬约束路径）；p=0.5 → 0。"""
    probability = float(probability)
    if not (0.0 <= probability < 0.5 or probability == 0.5):
        raise ValueError("probability must be in [0, 0.5]")
    if probability == 0.0:
        return float("inf")
    return float(np.log((1.0 - probability) / probability))


@dataclass
class SectorModel:
    """一个 Pauli sector 的完整模型输入（自洽校验过）。"""

    sector: str
    H_check: np.ndarray                # (m_c, n)
    stabilizer_rows: np.ndarray        # (m_s, n) 对偶 check 矩阵行 = S-moves
    logical_obs_basis: np.ndarray      # z-型观测配对基 (k, n)
    logical_move_basis: np.ndarray     # x-型 L-move/sector 基 (k, n)
    section: LinearSection
    k: int
    num_checks: int
    num_qubits: int
    checks_touching_each_qubit: list = field(default_factory=list)

    def fingerprint(self):
        payload = (
            self.sector.encode()
            + np.ascontiguousarray(self.H_check).tobytes()
            + np.ascontiguousarray(self.logical_obs_basis).tobytes()
            + np.ascontiguousarray(self.logical_move_basis).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()


def build_checks_touching_each_qubit(parity_check_matrix):
    parity_check_matrix = as_gf2_matrix(parity_check_matrix)
    return [
        np.flatnonzero(parity_check_matrix[:, j]).astype(np.int32)
        for j in range(parity_check_matrix.shape[1])
    ]


def assemble_sector_model(H_X, H_Z, logicals, sector="x_error"):
    """由 CSS 矩阵与配对归一逻辑基装配 SectorModel（含自洽断言）。

    logicals: LogicalPauliResult（x_i·z_j=δ_ij 已归一）。
    """
    if sector not in SECTORS:
        raise ValueError(f"sector must be one of {SECTORS}")
    H_X = as_gf2_matrix(H_X)
    H_Z = as_gf2_matrix(H_Z)
    if sector == "x_error":
        H_check, stabilizer_rows = H_Z, H_X
        obs_basis, move_basis = logicals.logical_Z, logicals.logical_X
    else:
        H_check, stabilizer_rows = H_X, H_Z
        obs_basis, move_basis = logicals.logical_X, logicals.logical_Z

    obs_basis = as_gf2_matrix(obs_basis)
    move_basis = as_gf2_matrix(move_basis)
    k = obs_basis.shape[0]
    # 自洽断言：move 基 ∈ ker(H_check)；stabilizer 行 ∈ ker(H_check)；配对 = I
    if k and gf2_matmul(H_check, move_basis.T).any():
        raise AssertionError("logical move basis not in ker(H_check)")
    if gf2_matmul(H_check, stabilizer_rows.T).any():
        raise AssertionError("stabilizer rows not in ker(H_check) (CSS violated)")
    if k:
        pairing = gf2_matmul(move_basis, obs_basis.T)
        if not np.array_equal(pairing, np.eye(k, dtype=np.uint8)):
            raise AssertionError("move/obs bases are not pairing-normalized")

    section = build_linear_section(H_check)
    return SectorModel(
        sector=sector,
        H_check=H_check,
        stabilizer_rows=stabilizer_rows,
        logical_obs_basis=obs_basis,
        logical_move_basis=move_basis,
        section=section,
        k=k,
        num_checks=H_check.shape[0],
        num_qubits=H_check.shape[1],
        checks_touching_each_qubit=build_checks_touching_each_qubit(H_check),
    )


@dataclass
class DisorderRealization:
    eta: np.ndarray              # (n,) uint8 数据错误
    delta: np.ndarray            # (m_c,) uint8 测量翻转
    observed_syndrome: np.ndarray  # s = Hη ⊕ δ
    p: float
    q: float
    eta_weight: int = 0
    delta_weight: int = 0

    def syndrome_argument(self, ensemble):
        if ensemble == "true_posterior":
            return self.observed_syndrome
        if ensemble == "repo_compat":
            return self.delta
        raise ValueError(f"ensemble must be one of {ENSEMBLES}")


def draw_disorder(model, p, q, rng):
    """η ~ Bern(p)^n，δ ~ Bern(q)^{m_c}，s = Hη⊕δ。"""
    return disorder_from_uniforms(
        model, p, q,
        data_uniforms=rng.random(model.num_qubits),
        syndrome_uniforms=rng.random(model.num_checks),
    )


def disorder_from_uniforms(model, p, q, data_uniforms, syndrome_uniforms):
    """CRN 路径：同一批 Uniform[0,1) 可在不同 (p,q) 间复用以降差分方差。"""
    data_uniforms = np.asarray(data_uniforms, dtype=np.float64)
    syndrome_uniforms = np.asarray(syndrome_uniforms, dtype=np.float64)
    if data_uniforms.shape != (model.num_qubits,):
        raise ValueError("data_uniforms shape mismatch")
    if syndrome_uniforms.shape != (model.num_checks,):
        raise ValueError("syndrome_uniforms shape mismatch")
    eta = (data_uniforms < float(p)).astype(np.uint8)
    delta = (syndrome_uniforms < float(q)).astype(np.uint8)
    observed = (gf2_matmul(model.H_check, eta[:, None])[:, 0] ^ delta).astype(
        np.uint8
    )
    return DisorderRealization(
        eta=eta, delta=delta, observed_syndrome=observed,
        p=float(p), q=float(q),
        eta_weight=int(eta.sum()), delta_weight=int(delta.sum()),
    )


@dataclass
class EnsembleWiring:
    """一个 (model, disorder, ensemble) 的采样问题定义。"""

    ensemble: str
    sigma_arg: np.ndarray        # 权重里的 syndrome 参数
    reference_label: np.ndarray  # ℓ_ref ∈ F_2^k（true: φ(η)；repo: 0）
    K_p: float
    K_q: float
    q_zero: bool

    def total_energy(self, model, v):
        """K_p|v| + K_q|Hv⊕σ_arg|（诊断/对拍用全量重算）。q=0 时第二项须为 0。"""
        v = np.asarray(v, dtype=np.uint8)
        syndrome_term = (
            gf2_matmul(model.H_check, v[:, None])[:, 0] ^ self.sigma_arg
        )
        weight_s = int(syndrome_term.sum())
        if self.q_zero:
            if weight_s:
                raise ValueError("q=0 hard constraint violated: H v != sigma_arg")
            return self.K_p * float(v.sum())
        return self.K_p * float(v.sum()) + self.K_q * float(weight_s)


def wire_ensemble(model, disorder, ensemble, observable_frame=None):
    """装配 EnsembleWiring。true_posterior 需要 observable_frame 算 ℓ_ref=φ(η)。"""
    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}")
    sigma_arg = disorder.syndrome_argument(ensemble).copy()
    q_zero = disorder.q == 0.0
    if q_zero:
        # true: s = Hη ∈ im ✓；repo: δ=0 ∈ im ✓——但仍显式校验，杜绝错误接线
        if not model.section.in_image(sigma_arg):
            raise AssertionError("q=0 sigma_arg must lie in im(H_check)")
    if ensemble == "true_posterior":
        if observable_frame is None:
            raise ValueError(
                "true_posterior requires observable_frame to compute ℓ_ref=φ(η)"
            )
        reference_label = observable_frame.label_of(disorder.eta)
    else:
        reference_label = np.zeros(model.k, dtype=np.uint8)
    return EnsembleWiring(
        ensemble=ensemble,
        sigma_arg=sigma_arg,
        reference_label=reference_label,
        K_p=coupling_from_probability(disorder.p),
        K_q=coupling_from_probability(disorder.q) if not q_zero else float("inf"),
        q_zero=q_zero,
    )
