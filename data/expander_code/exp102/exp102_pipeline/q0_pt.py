"""Hard-coset replica exchange using only stabilizer and logical moves.

The reference implementation deliberately stays in Python and is the oracle.
The production implementation puts complete sweeps, swaps, transport tracking,
and label recording in one Numba kernel.  States are stored by replica and a
rung-to-replica map is swapped, so an accepted exchange never copies an
``O(n)`` state vector.
"""

from dataclasses import dataclass
from functools import lru_cache

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
    @njit(cache=True, inline="always")
    def _nb_next_uint64(state):
        """Mirror exp101.prng.nb_next_uint64 without a dynamic import."""
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << np.uint64(23))
        x = x ^ (x >> np.uint64(17))
        x = x ^ y ^ (y >> np.uint64(26))
        state[1] = x
        return x + y


    @njit(cache=True, inline="always")
    def _nb_random(state):
        return float(_nb_next_uint64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


    @njit(cache=True, inline="always")
    def _nb_fill_permutation(state, buffer):
        for index in range(buffer.size):
            buffer[index] = index
        for index in range(buffer.size - 1, 0, -1):
            selected = int(_nb_next_uint64(state) % np.uint64(index + 1))
            temporary = buffer[index]
            buffer[index] = buffer[selected]
            buffer[selected] = temporary


    @njit(cache=True)
    def _run_q0_pt_numba_core(
        states,
        weights,
        state_labels,
        stabilizer_indices,
        stabilizer_offsets,
        logical_indices,
        logical_offsets,
        check_indices,
        check_offsets,
        syndrome,
        move_acceptance,
        swap_acceptance,
        rng_state,
        burn_rounds,
        measurement_rounds,
        sweeps_per_round,
        logical_move_repeat,
    ):
        """Run the full q=0 PT trajectory in nopython mode.

        ``move_acceptance[rung, delta]`` and
        ``swap_acceptance[edge, weight_deficit]`` are computed with NumPy on
        the Python side.  This avoids cross-libm differences at Metropolis
        boundaries while keeping all hot-loop decisions inside the kernel.
        """
        num_temperatures = states.shape[0]
        num_stabilizers = stabilizer_offsets.size - 1
        num_logicals = logical_offsets.size - 1

        logical_attempts = np.zeros((num_temperatures, num_logicals), dtype=np.int64)
        logical_accepts = np.zeros((num_temperatures, num_logicals), dtype=np.int64)
        swap_attempts = np.zeros(num_temperatures - 1, dtype=np.int64)
        swap_accepts = np.zeros(num_temperatures - 1, dtype=np.int64)
        labels = np.zeros(measurement_rounds, dtype=np.uint64)
        replica_at = np.arange(num_temperatures, dtype=np.int64)
        phase = np.zeros(num_temperatures, dtype=np.int8)
        arrival_hot_label = np.zeros(num_temperatures, dtype=np.uint64)
        left_hot_label = np.zeros(num_temperatures, dtype=np.uint64)
        stabilizer_order = np.empty(num_stabilizers, dtype=np.int64)
        logical_order = np.empty(num_logicals, dtype=np.int64)
        round_trips = 0
        changing_round_trips = 0
        parity = 0
        if burn_rounds == 0:
            phase[replica_at[0]] = 1

        total_rounds = burn_rounds + measurement_rounds
        for round_index in range(total_rounds):
            for rung in range(num_temperatures):
                replica = replica_at[rung]
                for _ in range(sweeps_per_round):
                    _nb_fill_permutation(rng_state, stabilizer_order)
                    for order_index in range(num_stabilizers):
                        move = stabilizer_order[order_index]
                        start = stabilizer_offsets[move]
                        stop = stabilizer_offsets[move + 1]
                        ones = 0
                        for position in range(start, stop):
                            ones += int(states[replica, stabilizer_indices[position]])
                        delta = (stop - start) - 2 * ones
                        uniform = _nb_random(rng_state)
                        if delta <= 0 or uniform < move_acceptance[rung, delta]:
                            for position in range(start, stop):
                                states[replica, stabilizer_indices[position]] ^= np.uint8(1)
                            weights[replica] += delta

                    for _ in range(logical_move_repeat):
                        _nb_fill_permutation(rng_state, logical_order)
                        for order_index in range(num_logicals):
                            bit = logical_order[order_index]
                            start = logical_offsets[bit]
                            stop = logical_offsets[bit + 1]
                            ones = 0
                            for position in range(start, stop):
                                ones += int(states[replica, logical_indices[position]])
                            delta = (stop - start) - 2 * ones
                            logical_attempts[rung, bit] += 1
                            uniform = _nb_random(rng_state)
                            if delta <= 0 or uniform < move_acceptance[rung, delta]:
                                logical_accepts[rung, bit] += 1
                                for position in range(start, stop):
                                    states[replica, logical_indices[position]] ^= np.uint8(1)
                                weights[replica] += delta
                                state_labels[replica] ^= np.uint64(1) << np.uint64(bit)

            for edge in range(parity, num_temperatures - 1, 2):
                swap_attempts[edge] += 1
                left_replica = replica_at[edge]
                right_replica = replica_at[edge + 1]
                weight_difference = weights[left_replica] - weights[right_replica]
                uniform = _nb_random(rng_state)
                if (
                    weight_difference >= 0
                    or uniform < swap_acceptance[edge, -weight_difference]
                ):
                    swap_accepts[edge] += 1
                    replica_at[edge] = right_replica
                    replica_at[edge + 1] = left_replica

            if round_index >= burn_rounds:
                cold_replica = replica_at[0]
                hot_replica = replica_at[-1]
                if phase[hot_replica] == 1:
                    phase[hot_replica] = 2
                    arrival_hot_label[hot_replica] = state_labels[hot_replica]
                elif phase[hot_replica] == 2:
                    left_hot_label[hot_replica] = state_labels[hot_replica]
                if phase[cold_replica] == 2:
                    round_trips += 1
                    changing_round_trips += int(
                        state_labels[cold_replica] != arrival_hot_label[cold_replica]
                    )
                    phase[cold_replica] = 1
                elif phase[cold_replica] == 0:
                    phase[cold_replica] = 1
                labels[round_index - burn_rounds] = state_labels[cold_replica]
            elif round_index + 1 == burn_rounds:
                # The oracle resets transport state after the final burn round.
                phase[:] = 0
                phase[replica_at[0]] = 1
            parity ^= 1

        # The validated move set preserves the hard coset algebraically.  A
        # sparse final-state check catches implementation/indexing corruption
        # without restoring the old dense residual calculation every round.
        max_residual = 0
        for replica in range(num_temperatures):
            residual_weight = 0
            for check in range(check_offsets.size - 1):
                parity_bit = syndrome[check]
                for position in range(check_offsets[check], check_offsets[check + 1]):
                    parity_bit ^= states[replica, check_indices[position]]
                residual_weight += int(parity_bit)
            if residual_weight > max_residual:
                max_residual = residual_weight

        return (
            labels,
            swap_attempts,
            swap_accepts,
            logical_attempts,
            logical_accepts,
            round_trips,
            changing_round_trips,
            max_residual,
            arrival_hot_label,
            left_hot_label,
        )
else:  # pragma: no cover
    _run_q0_pt_numba_core = None


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


def _supports_to_csr(supports):
    offsets = np.zeros(len(supports) + 1, dtype=np.int64)
    for index, support in enumerate(supports):
        offsets[index + 1] = offsets[index] + int(support.size)
    indices = np.empty(int(offsets[-1]), dtype=np.int64)
    for index, support in enumerate(supports):
        indices[offsets[index]:offsets[index + 1]] = support
    return np.ascontiguousarray(indices), offsets


@lru_cache(maxsize=8)
def _acceptance_tables(K_tuple, num_qubits, max_move_weight):
    """Cache immutable Metropolis thresholds shared by the four instances."""
    K = np.asarray(K_tuple, dtype=np.float64)
    deltas = np.arange(int(max_move_weight) + 1, dtype=np.float64)
    move = np.exp(-K[:, None] * deltas[None, :])
    deficits = np.arange(int(num_qubits) + 1, dtype=np.float64)
    swap = np.empty((K.size - 1, deficits.size), dtype=np.float64)
    for edge in range(K.size - 1):
        swap[edge] = np.exp(-float(K[edge] - K[edge + 1]) * deficits)
    return np.ascontiguousarray(move), np.ascontiguousarray(swap)


def _run_reference_instance(model, frame, syndrome, K, config, seed, initial_label,
                            stabilizers, logicals, base):
    from exp101_certified_src.gf2 import gf2_matmul
    from exp101_certified_src.prng import PortablePrng

    states = []
    for _ in range(config.num_temperatures):
        vector = base.copy()
        for bit in range(model.k):
            if (int(initial_label) >> bit) & 1:
                vector ^= model.logical_move_basis[bit]
        states.append({"v": vector, "weight": int(vector.sum())})
    rng = PortablePrng(seed)
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

    def sweep(rung):
        state = states[rung]
        for row in rng.permutation(len(stabilizers)):
            support = stabilizers[row]
            delta = _support_delta_reference(state["v"], support)
            uniform = rng.random()
            if -K[rung] * delta >= 0 or uniform < np.exp(-K[rung] * delta):
                state["v"][support] ^= 1
                state["weight"] += delta
        for _ in range(config.logical_move_repeat):
            for bit in rng.permutation(model.k):
                support = logicals[bit]
                delta = _support_delta_reference(state["v"], support)
                logical_attempts[rung, bit] += 1
                uniform = rng.random()
                if -K[rung] * delta >= 0 or uniform < np.exp(-K[rung] * delta):
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
            log_a = swap_log_acceptance(
                K[edge], K[edge + 1], states[edge]["weight"], states[edge + 1]["weight"]
            )
            uniform = rng.random()
            if log_a >= 0 or uniform < np.exp(log_a):
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
                changing_round_trips += int(label(states[0]) != arrival_hot_label[cold])
                phase[cold] = 1
            elif phase[cold] == 0:
                phase[cold] = 1

    parity = 0
    for _ in range(config.burn_rounds):
        one_round(parity, False)
        parity ^= 1
    phase[:] = 0
    phase[int(replica_at[0])] = 1
    labels = np.zeros(config.measurement_rounds, dtype=np.uint64)
    for index in range(config.measurement_rounds):
        one_round(parity, True)
        parity ^= 1
        labels[index] = label(states[0])
    return {
        "labels": labels,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "logical_attempts": logical_attempts,
        "logical_accepts": logical_accepts,
        "round_trips": round_trips,
        "sector_changing_round_trips": changing_round_trips,
        "max_hard_coset_residual": max_residual,
        "hot_arrival_labels": arrival_hot_label,
        "hot_departure_labels": left_hot_label,
    }


def _run_numba_instance(model, frame, syndrome, K, config, seed, initial_label,
                        stabilizers, logicals, base, support_csr):
    from exp101_certified_src.prng import PortablePrng

    vector = base.copy()
    for bit in range(model.k):
        if (int(initial_label) >> bit) & 1:
            vector ^= model.logical_move_basis[bit]
    states = np.repeat(vector[None, :], config.num_temperatures, axis=0)
    states = np.ascontiguousarray(states, dtype=np.uint8)
    weights = np.full(config.num_temperatures, int(vector.sum()), dtype=np.int64)
    initial_state_label = bits_to_uint64(frame.label_of(vector))
    state_labels = np.full(config.num_temperatures, initial_state_label, dtype=np.uint64)
    (
        stabilizer_indices, stabilizer_offsets,
        logical_indices, logical_offsets,
        check_indices, check_offsets,
    ) = support_csr
    max_move_weight = max(
        max((int(support.size) for support in stabilizers), default=0),
        max((int(support.size) for support in logicals), default=0),
    )
    move_acceptance, swap_acceptance = _acceptance_tables(
        tuple(float(value) for value in K), model.num_qubits, max_move_weight
    )
    rng_state = PortablePrng(seed).state_array()

    outputs = _run_q0_pt_numba_core(
        states,
        weights,
        state_labels,
        stabilizer_indices,
        stabilizer_offsets,
        logical_indices,
        logical_offsets,
        check_indices,
        check_offsets,
        syndrome,
        move_acceptance,
        swap_acceptance,
        rng_state,
        int(config.burn_rounds),
        int(config.measurement_rounds),
        int(config.sweeps_per_round),
        int(config.logical_move_repeat),
    )
    (
        labels,
        swap_attempts,
        swap_accepts,
        logical_attempts,
        logical_accepts,
        round_trips,
        changing_round_trips,
        max_residual,
        arrival_hot_label,
        left_hot_label,
    ) = outputs
    return {
        "labels": labels,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "logical_attempts": logical_attempts,
        "logical_accepts": logical_accepts,
        "round_trips": int(round_trips),
        "sector_changing_round_trips": int(changing_round_trips),
        "max_hard_coset_residual": int(max_residual),
        "hot_arrival_labels": arrival_hot_label,
        "hot_departure_labels": left_hot_label,
    }


def run_q0_pt_instance(model, frame, syndrome, p_cold, config, seed, initial_label,
                       engine="reference"):
    load_exp101()
    from exp101_certified_src.reference_mcmc import _logical_supports, _stab_supports

    if engine not in {"reference", "numba"}:
        raise ValueError("engine must be reference or numba")
    if engine == "numba" and _run_q0_pt_numba_core is None:
        raise RuntimeError("Numba engine requested but numba is unavailable")
    if model.k > 64:
        raise ValueError("k>64 is unsupported")
    K, probabilities = coupling_ladder(
        p_cold, config.p_hot, config.num_temperatures, config.gamma
    )
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    base = model.logical_sector_section.apply(syndrome, strict=True)
    support_cache = getattr(model, "_exp102_q0_support_cache", None)
    if support_cache is None:
        stabilizers, logicals = _stab_supports(model), _logical_supports(model)
        check_supports = [
            np.flatnonzero(row).astype(np.int64) for row in model.H_check
        ]
        support_cache = (
            stabilizers,
            logicals,
            (*_supports_to_csr(stabilizers),
             *_supports_to_csr(logicals),
             *_supports_to_csr(check_supports)),
        )
        model._exp102_q0_support_cache = support_cache
    stabilizers, logicals, support_csr = support_cache
    if engine == "reference":
        result = _run_reference_instance(
            model, frame, syndrome, K, config, seed, initial_label,
            stabilizers, logicals, base,
        )
    else:
        result = _run_numba_instance(
            model, frame, syndrome, K, config, seed, initial_label,
            stabilizers, logicals, base, support_csr,
        )
    result.update({
        "ladder_K": K,
        "ladder_p": probabilities,
        "engine": engine,
        "numba_enabled": engine == "numba",
    })
    return result
