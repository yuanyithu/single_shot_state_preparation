"""Reference MCMC for the canonical reduced posterior.

The sampled energy is ``K_p |e| + K_q |H e xor y_eff|``.  Ground-truth
data errors never enter an acceptance ratio.  Character observations are
returned in both the absolute logical-sector frame and the planted-relative
(Mattis) frame; the historical ``m_u`` keys remain relative aliases.
move 集（notes/01 §1.3 的 L·T·S 对应）：
  - single-bit Metropolis（仅 q>0）：采样 T⊕S；ΔE = ±K_p + K_q·Δ|syndrome_term|
  - stabilizer-row 翻转（S-move）：syndrome 不变；ΔE = K_p·Δ|v|
  - logical 翻转（L-move，sector flip）：syndrome 不变；ΔE = K_p·Δ|v|；
    **per-u 接受计数**（冷端 logical 接受率 = 收敛硬判据的数据源）
q=0：single-bit 关闭，e 初始必须满足 H e=effective_syndrome
（用 logical-sector section 代表 + sector 平移）。

RNG 协议（reproducibility + 与 fast 引擎 bit 级一致的基础）：单个
PortablePrng(seed)（src/prng.py，python/numba 双胞胎逐位一致）；每个 sweep 依次
  (1) single-bit：qubit 顺序 = rng.permutation(n)（q=0 跳过，不消耗随机数）
  (2) S-move：行顺序 = rng.permutation(num_stab_rows)
  (3) L-move：u 顺序 = rng.permutation(k)
  每次 Metropolis 判定消耗一个 rng.random()（log_acceptance >= 0 时也消耗，
  保证轨迹与判定分支无关地可复现）。permutation 内部 Fisher–Yates 消耗
  n−1 次 randbelow——fast 引擎必须逐一镜像。
"""

from dataclasses import dataclass, field

import numpy as np

from .gf2 import gf2_matmul
from .observables import character_signs_for_label, observable_values
from .prng import PortablePrng


@dataclass
class ReferenceMcmcConfig:
    num_burn_in_sweeps: int = 200
    num_measurements: int = 1000
    num_sweeps_between_measurements: int = 1
    logical_move_repeat: int = 1        # 每 sweep 每个 u 的 L-move 次数
    record_state_trajectory: bool = False
    record_observable_trajectory: bool = False
    debug_invariants: bool = False      # 每 sweep 后全量重算对拍（慢，测试用）


@dataclass
class McmcState:
    v: np.ndarray                  # (n,) uint8
    syndrome_term: np.ndarray      # (m_c,) uint8 = H e xor effective_syndrome
    data_weight: int
    syndrome_weight: int


def make_initial_state(model, wiring, mode="auto", sector_bitmask=0):
    """初态构造。

    mode:
      "auto"        q=0 → "section"；q>0 → "zero"（y_eff 此时一般 ∉ im(H)，
                    严格 section 会正确拒绝——这正是防误用设计）
      "zero"        v = 0（q>0 任意；q=0 仅当 y_eff=0）
      "section"     v = r(y_eff)（要求 y_eff ∈ im(H)，q=0 恒成立）
    sector_bitmask：再 ⊕ 相应 logical move 基组合（sector 平移，q=0/q>0 通用）。
    """
    if mode == "auto":
        mode = "section" if wiring.q_zero else "zero"
    n = model.num_qubits
    if mode == "zero":
        v = np.zeros(n, dtype=np.uint8)
    elif mode == "section":
        v = model.logical_sector_section.apply(
            wiring.gibbs_syndrome_argument, strict=True
        ).copy()
    else:
        raise ValueError("mode must be auto|zero|section")
    for bit in range(model.k):
        if (int(sector_bitmask) >> bit) & 1:
            v ^= model.logical_move_basis[bit]
    syndrome_term = (
        gf2_matmul(model.H_check, v[:, None])[:, 0]
        ^ wiring.gibbs_syndrome_argument
    ).astype(np.uint8)
    if wiring.q_zero and syndrome_term.any():
        raise ValueError(
            "q=0 initial state violates hard constraint H e = y_eff "
            "(use mode='section')"
        )
    return McmcState(
        v=v,
        syndrome_term=syndrome_term,
        data_weight=int(v.sum()),
        syndrome_weight=int(syndrome_term.sum()),
    )


def single_bit_log_acceptance(model, wiring, state, qubit_index):
    """翻转 qubit_index 的 Metropolis log 接受率（q>0）。"""
    delta_data = -1 if state.v[qubit_index] else +1
    touched = model.checks_touching_each_qubit[qubit_index]
    flipped = state.syndrome_term[touched]
    delta_syndrome = int(flipped.size) - 2 * int(flipped.sum())
    return -wiring.K_p * delta_data - wiring.K_q * delta_syndrome


def support_move_log_acceptance(wiring, state, move_support_row):
    """按位掩码行（S/L move，syndrome 不变）的 log 接受率。"""
    overlap = int(state.v[move_support_row].sum())
    weight = int(move_support_row.shape[0])
    delta_data = weight - 2 * overlap
    return 0.0 if delta_data == 0 else -wiring.K_p * delta_data


@dataclass
class MoveCounters:
    single_bit_attempts: int = 0
    single_bit_accepts: int = 0
    stabilizer_attempts: int = 0
    stabilizer_accepts: int = 0
    logical_attempts_per_u: np.ndarray = None
    logical_accepts_per_u: np.ndarray = None

    def rates(self):
        def rate(a, b):
            return float(a) / float(b) if b else float("nan")

        per_u = None
        if self.logical_attempts_per_u is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                per_u = np.where(
                    self.logical_attempts_per_u > 0,
                    self.logical_accepts_per_u
                    / np.maximum(self.logical_attempts_per_u, 1),
                    np.nan,
                )
        return {
            "single_bit": rate(self.single_bit_accepts, self.single_bit_attempts),
            "stabilizer": rate(self.stabilizer_accepts, self.stabilizer_attempts),
            "logical_per_u": per_u,
            "logical_overall": rate(
                int(self.logical_accepts_per_u.sum()) if per_u is not None else 0,
                int(self.logical_attempts_per_u.sum()) if per_u is not None else 0,
            ),
        }


def _stab_supports(model):
    return [np.flatnonzero(row).astype(np.int64) for row in model.stabilizer_rows]


def _logical_supports(model):
    return [np.flatnonzero(row).astype(np.int64)
            for row in model.logical_move_basis]


def run_sweep(model, wiring, state, rng, counters, stab_supports,
              logical_supports, logical_move_repeat=1):
    """一个完整 sweep（RNG 协议见模块 docstring）。"""
    if not wiring.q_zero:
        for qubit_index in rng.permutation(model.num_qubits):
            counters.single_bit_attempts += 1
            log_acc = single_bit_log_acceptance(model, wiring, state, qubit_index)
            u = rng.random()
            if log_acc >= 0.0 or u < np.exp(log_acc):
                counters.single_bit_accepts += 1
                delta_data = -1 if state.v[qubit_index] else +1
                state.v[qubit_index] ^= 1
                touched = model.checks_touching_each_qubit[qubit_index]
                flipped = state.syndrome_term[touched]
                state.syndrome_weight += int(flipped.size) - 2 * int(flipped.sum())
                state.syndrome_term[touched] ^= 1
                state.data_weight += delta_data
    for row_index in rng.permutation(len(stab_supports)):
        counters.stabilizer_attempts += 1
        support = stab_supports[row_index]
        log_acc = support_move_log_acceptance(wiring, state, support)
        u = rng.random()
        if log_acc >= 0.0 or u < np.exp(log_acc):
            counters.stabilizer_accepts += 1
            overlap = int(state.v[support].sum())
            state.data_weight += int(support.shape[0]) - 2 * overlap
            state.v[support] ^= 1
    for _ in range(int(logical_move_repeat)):
        for u_index in rng.permutation(model.k) if model.k else []:
            counters.logical_attempts_per_u[u_index] += 1
            support = logical_supports[u_index]
            log_acc = support_move_log_acceptance(wiring, state, support)
            u = rng.random()
            if log_acc >= 0.0 or u < np.exp(log_acc):
                counters.logical_accepts_per_u[u_index] += 1
                overlap = int(state.v[support].sum())
                state.data_weight += int(support.shape[0]) - 2 * overlap
                state.v[support] ^= 1


def _check_invariants(model, wiring, state):
    v = state.v
    syndrome = (
        gf2_matmul(model.H_check, v[:, None])[:, 0]
        ^ wiring.gibbs_syndrome_argument
    ).astype(np.uint8)
    assert np.array_equal(syndrome, state.syndrome_term), "syndrome cache broken"
    assert int(v.sum()) == state.data_weight, "data weight cache broken"
    assert int(syndrome.sum()) == state.syndrome_weight, "syndrome weight cache broken"
    if wiring.q_zero:
        assert state.syndrome_weight == 0, "q=0 constraint violated"


def _weighted_energy(coupling, weight):
    """Evaluate ``coupling * weight`` without the hard-limit ``inf * 0``."""
    return 0.0 if int(weight) == 0 else float(coupling) * int(weight)


def run_reference_mcmc(model, frame, observable_set, wiring, config, seed,
                       initial_mode="auto", sector_bitmask=0):
    """完整参考采样。返回 dict：m_u、聚合前原始和、接受率（含 per-u logical）、
    能量迹、可选轨迹。"""
    rng = PortablePrng(seed)
    state = make_initial_state(model, wiring, initial_mode, sector_bitmask)
    counters = MoveCounters(
        logical_attempts_per_u=np.zeros(model.k, dtype=np.int64),
        logical_accepts_per_u=np.zeros(model.k, dtype=np.int64),
    )
    stab_supports = _stab_supports(model)
    logical_supports = _logical_supports(model)

    for _ in range(int(config.num_burn_in_sweeps)):
        run_sweep(model, wiring, state, rng, counters, stab_supports,
                  logical_supports, config.logical_move_repeat)
        if config.debug_invariants:
            _check_invariants(model, wiring, state)

    num_u = observable_set.num_u
    observable_sums = np.zeros(num_u, dtype=np.int64)
    observable_traj = (
        np.zeros((config.num_measurements, num_u), dtype=np.int8)
        if config.record_observable_trajectory else None
    )
    state_traj = (
        np.zeros((config.num_measurements, model.num_qubits), dtype=np.uint8)
        if config.record_state_trajectory else None
    )
    energy_trace = np.zeros(config.num_measurements, dtype=np.float64)

    for measurement_index in range(int(config.num_measurements)):
        for _ in range(int(config.num_sweeps_between_measurements)):
            run_sweep(model, wiring, state, rng, counters, stab_supports,
                      logical_supports, config.logical_move_repeat)
            if config.debug_invariants:
                _check_invariants(model, wiring, state)
        values = observable_values(observable_set, wiring, state.v)
        observable_sums += values.astype(np.int64)
        if observable_traj is not None:
            observable_traj[measurement_index] = values
        if state_traj is not None:
            state_traj[measurement_index] = state.v
        energy_trace[measurement_index] = _weighted_energy(
            wiring.K_p, state.data_weight
        ) + (
            0.0
            if wiring.q_zero
            else _weighted_energy(wiring.K_q, state.syndrome_weight)
        )

    m_u_relative = (
        observable_sums.astype(np.float64) / float(config.num_measurements)
    )
    signs = character_signs_for_label(
        observable_set, wiring.planted_logical_class
    ).astype(np.int64)
    m_u_absolute = m_u_relative * signs
    observable_sums_absolute = observable_sums * signs
    observable_trajectory_absolute = (
        observable_traj * signs[None, :]
        if observable_traj is not None else None
    )
    return {
        "m_u": m_u_relative,
        "m_u_relative": m_u_relative,
        "m_u_absolute": m_u_absolute,
        "observable_sums": observable_sums,
        "observable_sums_relative": observable_sums,
        "observable_sums_absolute": observable_sums_absolute,
        "num_measurements": int(config.num_measurements),
        "acceptance": counters.rates(),
        "counters": counters,
        "energy_trace": energy_trace,
        "observable_trajectory": observable_traj,
        "observable_trajectory_relative": observable_traj,
        "observable_trajectory_absolute": observable_trajectory_absolute,
        "state_trajectory": state_traj,
        "final_state": state,
    }
