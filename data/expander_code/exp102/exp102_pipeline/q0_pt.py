"""Hard-coset replica exchange using only stabilizer and logical moves."""

from dataclasses import dataclass

import numpy as np

from .exp101_bridge import load_exp101
from .labels import bits_to_uint64

try:
    from numba import njit
except ImportError:  # pragma: no cover - production preflight requires Numba
    njit = None


def _support_delta_reference(vector, support):
    return int(support.size) - 2 * int(vector[support].sum())


if njit is not None:
    @njit(cache=True)
    def _support_delta_numba(vector, support):
        ones = 0
        for index in support:
            ones += int(vector[index])
        return support.size - 2 * ones
else:
    _support_delta_numba = None


def coupling(p):
    if not 0.0 < float(p) < 0.5:
        raise ValueError("probability must lie in (0,0.5)")
    return float(np.log((1.0 - p) / p))


def coupling_ladder(p_cold, p_hot, num_temperatures, gamma):
    if num_temperatures < 2 or not p_cold < p_hot < 0.5 or gamma <= 0:
        raise ValueError("invalid q=0 PT ladder")
    x = np.linspace(0.0, 1.0, int(num_temperatures)) ** float(gamma)
    K = coupling(p_cold) + (coupling(p_hot) - coupling(p_cold)) * x
    return K, 1.0 / (1.0 + np.exp(K))


def swap_log_acceptance(K_i, K_j, weight_i, weight_j):
    return float(K_i - K_j) * int(weight_i - weight_j)


@dataclass(frozen=True)
class Q0PtConfig:
    p_hot: float
    num_temperatures: int
    gamma: float
    burn_rounds: int
    measurement_rounds: int
    sweeps_per_round: int = 1
    logical_move_repeat: int = 1


def run_q0_pt_instance(model, frame, syndrome, p_cold, config, seed, initial_label,
                       engine="reference"):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_matmul
    from exp101_certified_src.prng import PortablePrng
    from exp101_certified_src.reference_mcmc import MoveCounters, _logical_supports, _stab_supports

    if engine not in {"reference", "numba"}:
        raise ValueError("engine must be reference or numba")
    if engine == "numba" and _support_delta_numba is None:
        raise RuntimeError("Numba engine requested but numba is unavailable")
    if model.k > 64:
        raise ValueError("k>64 is unsupported")
    K, probabilities = coupling_ladder(p_cold, config.p_hot, config.num_temperatures, config.gamma)
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = []
    for _ in range(config.num_temperatures):
        vector = base.copy()
        for bit in range(model.k):
            if (int(initial_label) >> bit) & 1:
                vector ^= model.logical_move_basis[bit]
        states.append({"v": vector, "weight": int(vector.sum())})
    rng = PortablePrng(seed)
    stabilizers, logicals = _stab_supports(model), _logical_supports(model)
    logical_attempts = np.zeros((config.num_temperatures, model.k), dtype=np.int64)
    logical_accepts = np.zeros_like(logical_attempts)
    swap_attempts = np.zeros(config.num_temperatures - 1, dtype=np.int64)
    swap_accepts = np.zeros_like(swap_attempts)
    replica_at = np.arange(config.num_temperatures, dtype=np.int64)
    phase = np.zeros(config.num_temperatures, dtype=np.int8)
    arrival_hot_label = np.zeros(config.num_temperatures, dtype=np.uint64)
    left_hot_label = np.zeros(config.num_temperatures, dtype=np.uint64)
    round_trips = 0
    changing_round_trips = 0
    max_residual = 0

    def label(state):
        return bits_to_uint64(frame.label_of(state["v"]))

    delta_of = _support_delta_numba if engine == "numba" else _support_delta_reference

    def sweep(rung):
        state = states[rung]
        for row in rng.permutation(len(stabilizers)):
            support = stabilizers[row]
            delta = int(delta_of(state["v"], support))
            u = rng.random()
            if -K[rung] * delta >= 0 or u < np.exp(-K[rung] * delta):
                state["v"][support] ^= 1
                state["weight"] += delta
        for _ in range(config.logical_move_repeat):
            for bit in rng.permutation(model.k):
                support = logicals[bit]
                delta = int(delta_of(state["v"], support))
                logical_attempts[rung, bit] += 1
                u = rng.random()
                if -K[rung] * delta >= 0 or u < np.exp(-K[rung] * delta):
                    logical_accepts[rung, bit] += 1
                    state["v"][support] ^= 1
                    state["weight"] += delta

    def one_round(parity, measure):
        nonlocal round_trips, changing_round_trips, max_residual
        for rung in range(config.num_temperatures):
            for _ in range(config.sweeps_per_round):
                sweep(rung)
            residual = gf2_matmul(model.H_check, states[rung]["v"][:, None])[:, 0] ^ syndrome
            max_residual = max(max_residual, int(residual.sum()))
        for edge in range(parity, config.num_temperatures - 1, 2):
            swap_attempts[edge] += 1
            log_a = swap_log_acceptance(K[edge], K[edge + 1], states[edge]["weight"], states[edge + 1]["weight"])
            u = rng.random()
            if log_a >= 0 or u < np.exp(log_a):
                swap_accepts[edge] += 1
                states[edge], states[edge + 1] = states[edge + 1], states[edge]
                replica_at[edge], replica_at[edge + 1] = replica_at[edge + 1], replica_at[edge]
        if measure:
            cold, hot = int(replica_at[0]), int(replica_at[-1])
            if phase[hot] == 1:
                phase[hot] = 2
                arrival_hot_label[hot] = label(states[-1])
            elif phase[hot] == 2:
                left_hot_label[hot] = label(states[-1])
            if phase[cold] == 2:
                round_trips += 1
                # Count net logical transport, not the existence of transient flips at hot.
                changing_round_trips += int(label(states[0]) != arrival_hot_label[cold])
                phase[cold] = 1
            elif phase[cold] == 0:
                phase[cold] = 1

    parity = 0
    for _ in range(config.burn_rounds):
        one_round(parity, False); parity ^= 1
    phase[:] = 0
    phase[int(replica_at[0])] = 1
    labels = np.zeros(config.measurement_rounds, dtype=np.uint64)
    for index in range(config.measurement_rounds):
        one_round(parity, True); parity ^= 1
        labels[index] = label(states[0])
    return {
        "labels": labels, "ladder_K": K, "ladder_p": probabilities,
        "swap_attempts": swap_attempts, "swap_accepts": swap_accepts,
        "logical_attempts": logical_attempts, "logical_accepts": logical_accepts,
        "round_trips": round_trips, "sector_changing_round_trips": changing_round_trips,
        "max_hard_coset_residual": max_residual,
        "engine": engine, "numba_enabled": engine == "numba",
        "hot_arrival_labels": arrival_hot_label, "hot_departure_labels": left_hot_label,
    }
