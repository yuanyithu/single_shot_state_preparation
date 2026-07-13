"""Parallel tempering（G2.5，仅 q>0；exp37/主项目 sync_enlarge 语义的规范变量版）。

ladder（notes/00 §2.2）：
  - "sync_enlarge"：enlarge_k 由 q 的耦合比几何插值（spacing_power 塑形），
    K_p,k = K_p/enlarge_k，K_q,k = K_q/enlarge_k（同步放大），换回概率后硬校验
    p_k, q_k < 0.5（超界 ValueError——对应 CLAUDE.md exp35 的坑）。
  - "data_only"：只放大 K_p（等 log-odds 插值），K_q 固定。
swap（规范变量推导）：相邻 rung i,j 交换构型 v_i, v_j：
  log_ratio = (K_p,i−K_p,j)(W_p(v_i)−W_p(v_j)) + (K_q,i−K_q,j)(W_s(v_i)−W_s(v_j))
  与主项目 log-odds 形式等价（K = −log_odds）。even/odd 交替奇偶尝试。
All rungs share one canonical Gibbs syndrome and planted logical reference;
only state-object references are swapped.
replica 往返（round trip）只计完整的冷→热→冷路径；burn-in 与 measurement
分别计数，生产 gate 只使用 measurement-phase 往返。

cluster update 不移植（决策记录）：主项目 cluster_update 的 RREF 受限子空间
采样是为 toric 几何/ladder 语义调过的加速器，非正确性必需；expander 用
bit+S+L+PT 组合，混合不足时优先调 ladder/往返数而非引入未经验证的新算法
（plan §6 风险 9）。

性能注：本模块用参考引擎逐 rung 扫（纯 python）——G2/G3 验证规模足够；
生产直采路径若需 numba PT 在 Phase 4 评估（TI 引擎才是生产主力）。
"""

from dataclasses import dataclass, field

import numpy as np

from .model import EnsembleWiring, coupling_from_probability
from .observables import character_signs_for_label, observable_values
from .prng import PortablePrng
from .reference_mcmc import (
    MoveCounters,
    _logical_supports,
    _stab_supports,
    make_initial_state,
    run_sweep,
)


def probability_from_coupling(coupling):
    """K = log((1−p)/p) 的逆：p = 1/(1+e^K)。"""
    return 1.0 / (1.0 + float(np.exp(coupling)))


def sync_enlarge_ladder(p_cold, q_cold, q_hot, num_temperatures,
                        spacing_power=1.0):
    """返回 (p_ladder, q_ladder)，rung0=冷（原参数），末端 q=q_hot。"""
    num_temperatures = int(num_temperatures)
    if num_temperatures < 2:
        raise ValueError("num_temperatures must be >= 2")
    if not (0.0 < q_cold < q_hot < 0.5):
        raise ValueError("need 0 < q_cold < q_hot < 0.5")
    spacing_power = float(spacing_power)
    if not (np.isfinite(spacing_power) and spacing_power > 0):
        raise ValueError("spacing_power must be positive")
    K_p = coupling_from_probability(p_cold)
    K_q = coupling_from_probability(q_cold)
    K_q_hot = coupling_from_probability(q_hot)
    hot_enlarge = K_q / K_q_hot
    if hot_enlarge <= 1.0:
        raise ValueError("q_hot must be hotter than q_cold")
    shaped = np.linspace(0.0, 1.0, num_temperatures) ** spacing_power
    enlarge = np.exp(shaped * np.log(hot_enlarge))
    p_ladder = np.array(
        [probability_from_coupling(K_p / e) for e in enlarge], dtype=np.float64
    )
    q_ladder = np.array(
        [probability_from_coupling(K_q / e) for e in enlarge], dtype=np.float64
    )
    if np.any(p_ladder >= 0.5) or np.any(q_ladder >= 0.5):
        raise ValueError(
            "sync_enlarge ladder pushes p_k or q_k to >= 0.5; "
            "reduce q_hot or p_cold (known exp35-style constraint)"
        )
    return p_ladder, q_ladder


def data_only_ladder(p_cold, p_hot, q_cold, num_temperatures):
    num_temperatures = int(num_temperatures)
    if num_temperatures < 2:
        raise ValueError("num_temperatures must be >= 2")
    if not (0.0 < p_cold < p_hot < 0.5):
        raise ValueError("need 0 < p_cold < p_hot < 0.5")
    K_cold = coupling_from_probability(p_cold)
    K_hot = coupling_from_probability(p_hot)
    couplings = np.linspace(K_cold, K_hot, num_temperatures)
    p_ladder = np.array(
        [probability_from_coupling(K) for K in couplings], dtype=np.float64
    )
    q_ladder = np.full(num_temperatures, float(q_cold), dtype=np.float64)
    return p_ladder, q_ladder


def swap_log_ratio(wiring_i, wiring_j, state_i, state_j):
    """相邻 rung 交换的 Metropolis log 比（规范变量精确式）。"""
    delta_Kp = wiring_i.K_p - wiring_j.K_p
    delta_Kq = wiring_i.K_q - wiring_j.K_q
    return (
        delta_Kp * (state_i.data_weight - state_j.data_weight)
        + delta_Kq * (state_i.syndrome_weight - state_j.syndrome_weight)
    )


@dataclass
class PtConfig:
    num_temperatures: int = 8
    q_hot: float = 0.40
    ladder_mode: str = "sync_enlarge"   # sync_enlarge | data_only
    p_hot: float = None                  # data_only 模式用
    spacing_power: float = 1.0
    num_burn_in_rounds: int = 200
    num_measurement_rounds: int = 1000
    sweeps_per_round: int = 1
    logical_move_repeat: int = 1
    observable_rungs: str = "cold"       # cold | all
    record_observable_trajectory: bool = False


@dataclass
class PtResult:
    m_u_cold: np.ndarray
    observable_sums_cold: np.ndarray
    num_measurements: int
    ladder_p: np.ndarray
    ladder_q: np.ndarray
    swap_attempts: np.ndarray            # (K−1,)
    swap_accepts: np.ndarray
    counters_per_rung: list              # MoveCounters
    round_trips: int
    replica_id_per_rung: np.ndarray
    m_u_per_rung: object = None          # observable_rungs="all" 时 (K, num_u)
    observable_trajectory_cold: object = None
    energy_trace_cold: object = None
    m_u_cold_absolute: object = None
    m_u_cold_relative: object = None
    m_u_per_rung_absolute: object = None
    observable_trajectory_cold_absolute: object = None
    observable_trajectory_cold_relative: object = None
    burn_in_round_trips: object = None
    measurement_round_trips: object = None
    burn_in_round_trips_per_replica: object = None
    measurement_round_trips_per_replica: object = None

    def swap_rates(self):
        with np.errstate(invalid="ignore"):
            return self.swap_accepts / np.maximum(self.swap_attempts, 1)

    def cold_logical_acceptance_per_u(self):
        return self.counters_per_rung[0].rates()["logical_per_u"]


class _RoundTripCounter:
    """Count complete cold->hot->cold replica paths within one phase."""

    _UNARMED = 0
    _SEEN_COLD = 1
    _SEEN_HOT_AFTER_COLD = 2

    def __init__(self, num_replicas, cold_replica):
        self.state = np.full(
            int(num_replicas), self._UNARMED, dtype=np.int8
        )
        self.counts = np.zeros(int(num_replicas), dtype=np.int64)
        self.state[int(cold_replica)] = self._SEEN_COLD

    def observe_endpoints(self, cold_replica, hot_replica):
        cold_replica = int(cold_replica)
        hot_replica = int(hot_replica)
        if self.state[cold_replica] == self._SEEN_HOT_AFTER_COLD:
            self.counts[cold_replica] += 1
            self.state[cold_replica] = self._SEEN_COLD
        elif self.state[cold_replica] == self._UNARMED:
            self.state[cold_replica] = self._SEEN_COLD
        if self.state[hot_replica] == self._SEEN_COLD:
            self.state[hot_replica] = self._SEEN_HOT_AFTER_COLD

    @property
    def total(self):
        return int(self.counts.sum())


def run_parallel_tempering(model, frame, observable_set, wiring, pt_config,
                           seed, sector_bitmask_per_replica=None):
    """PT 主循环（仅 q>0）。wiring = 冷端 EnsembleWiring（rung0 精确等于它）。"""
    if wiring.q_zero:
        raise ValueError("parallel tempering is only supported for q>0")
    p_cold = probability_from_coupling(wiring.K_p)
    q_cold = probability_from_coupling(wiring.K_q)
    if pt_config.ladder_mode == "sync_enlarge":
        ladder_p, ladder_q = sync_enlarge_ladder(
            p_cold, q_cold, pt_config.q_hot, pt_config.num_temperatures,
            pt_config.spacing_power,
        )
    elif pt_config.ladder_mode == "data_only":
        if pt_config.p_hot is None:
            raise ValueError("data_only ladder requires p_hot")
        ladder_p, ladder_q = data_only_ladder(
            p_cold, pt_config.p_hot, q_cold, pt_config.num_temperatures,
        )
    else:
        raise ValueError("ladder_mode must be sync_enlarge|data_only")

    num_rungs = pt_config.num_temperatures
    wirings = []
    for rung in range(num_rungs):
        wirings.append(EnsembleWiring(
            ensemble=wiring.ensemble,
            gibbs_syndrome_argument=wiring.gibbs_syndrome_argument,
            planted_logical_class=wiring.planted_logical_class,
            K_p=coupling_from_probability(ladder_p[rung]),
            K_q=coupling_from_probability(ladder_q[rung]),
            q_zero=False,
        ))
    # rung0 与输入 wiring 完全一致（防 ladder 端点漂移）
    assert abs(wirings[0].K_p - wiring.K_p) < 1e-12
    assert abs(wirings[0].K_q - wiring.K_q) < 1e-12

    rng = PortablePrng(seed)
    if sector_bitmask_per_replica is None:
        sector_bitmask_per_replica = [0] * num_rungs
    states = [
        make_initial_state(model, wirings[rung], "zero",
                           sector_bitmask_per_replica[rung])
        for rung in range(num_rungs)
    ]
    counters = [
        MoveCounters(
            logical_attempts_per_u=np.zeros(model.k, dtype=np.int64),
            logical_accepts_per_u=np.zeros(model.k, dtype=np.int64),
        )
        for _ in range(num_rungs)
    ]
    stab_supports = _stab_supports(model)
    logical_supports = _logical_supports(model)

    swap_attempts = np.zeros(num_rungs - 1, dtype=np.int64)
    swap_accepts = np.zeros(num_rungs - 1, dtype=np.int64)
    replica_id = np.arange(num_rungs, dtype=np.int64)
    burn_in_round_trips = _RoundTripCounter(num_rungs, replica_id[0])
    active_round_trip_counter = burn_in_round_trips

    num_u = observable_set.num_u
    sums_cold = np.zeros(num_u, dtype=np.int64)
    sums_per_rung = (
        np.zeros((num_rungs, num_u), dtype=np.int64)
        if pt_config.observable_rungs == "all" else None
    )
    traj_cold = (
        np.zeros((pt_config.num_measurement_rounds, num_u), dtype=np.int8)
        if pt_config.record_observable_trajectory else None
    )
    energy_cold = np.zeros(pt_config.num_measurement_rounds, dtype=np.float64)

    def one_round(parity):
        for rung in range(num_rungs):
            for _ in range(pt_config.sweeps_per_round):
                run_sweep(model, wirings[rung], states[rung], rng,
                          counters[rung], stab_supports, logical_supports,
                          pt_config.logical_move_repeat)
        for i in range(parity, num_rungs - 1, 2):
            swap_attempts[i] += 1
            log_ratio = swap_log_ratio(
                wirings[i], wirings[i + 1], states[i], states[i + 1]
            )
            u = rng.random()
            if log_ratio >= 0.0 or u < np.exp(log_ratio):
                swap_accepts[i] += 1
                states[i], states[i + 1] = states[i + 1], states[i]
                replica_id[i], replica_id[i + 1] = (
                    replica_id[i + 1], replica_id[i]
                )
        active_round_trip_counter.observe_endpoints(
            replica_id[0], replica_id[-1]
        )

    parity = 0
    for _ in range(int(pt_config.num_burn_in_rounds)):
        one_round(parity)
        parity ^= 1
    # Do not carry a partial burn-in transit into production diagnostics.
    measurement_round_trips = _RoundTripCounter(num_rungs, replica_id[0])
    active_round_trip_counter = measurement_round_trips
    for measurement_index in range(int(pt_config.num_measurement_rounds)):
        one_round(parity)
        parity ^= 1
        values_cold = observable_values(observable_set, wirings[0],
                                        states[0].v)
        sums_cold += values_cold.astype(np.int64)
        if traj_cold is not None:
            traj_cold[measurement_index] = values_cold
        if sums_per_rung is not None:
            for rung in range(1, num_rungs):
                sums_per_rung[rung] += observable_values(
                    observable_set, wirings[rung], states[rung].v
                ).astype(np.int64)
            sums_per_rung[0] += values_cold.astype(np.int64)
        energy_cold[measurement_index] = (
            wirings[0].K_p * states[0].data_weight
            + wirings[0].K_q * states[0].syndrome_weight
        )

    num_meas = int(pt_config.num_measurement_rounds)
    m_u_relative = sums_cold.astype(np.float64) / num_meas
    character_signs = character_signs_for_label(
        observable_set, wiring.planted_logical_class
    ).astype(np.int64)
    m_u_absolute = m_u_relative * character_signs
    m_u_per_rung_relative = (
        sums_per_rung.astype(np.float64) / num_meas
        if sums_per_rung is not None else None
    )
    trajectory_absolute = (
        traj_cold * character_signs[None, :]
        if traj_cold is not None else None
    )
    return PtResult(
        m_u_cold=m_u_relative,
        observable_sums_cold=sums_cold,
        num_measurements=num_meas,
        ladder_p=ladder_p,
        ladder_q=ladder_q,
        swap_attempts=swap_attempts,
        swap_accepts=swap_accepts,
        counters_per_rung=counters,
        round_trips=measurement_round_trips.total,
        replica_id_per_rung=replica_id,
        m_u_per_rung=m_u_per_rung_relative,
        observable_trajectory_cold=traj_cold,
        energy_trace_cold=energy_cold,
        m_u_cold_absolute=m_u_absolute,
        m_u_cold_relative=m_u_relative,
        m_u_per_rung_absolute=(
            m_u_per_rung_relative * character_signs[None, :]
            if m_u_per_rung_relative is not None else None
        ),
        observable_trajectory_cold_absolute=trajectory_absolute,
        observable_trajectory_cold_relative=traj_cold,
        burn_in_round_trips=burn_in_round_trips.total,
        measurement_round_trips=measurement_round_trips.total,
        burn_in_round_trips_per_replica=(
            burn_in_round_trips.counts.copy()
        ),
        measurement_round_trips_per_replica=(
            measurement_round_trips.counts.copy()
        ),
    )
