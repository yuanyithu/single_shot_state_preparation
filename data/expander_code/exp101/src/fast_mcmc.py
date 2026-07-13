"""numba fast path（G2.4）：与 reference_mcmc **bit 级一致**的快引擎。

一致性设计：
  - 共用 src/prng.py 可移植 RNG（python/numba 双胞胎逐位一致）；
  - sweep 结构与 RNG 消耗顺序逐一镜像 reference（permutation 的 Fisher–Yates
    也消耗同样的 randbelow 序列）；
  - 接受判定同式：log_acc >= 0 or u < exp(log_acc)，float64 同一算式；
  - 观测量：⟨w_u, v⟩ 奇偶用 uint64 words 增量维护（每 qubit / 每 logical row
    预打包翻转掩码；stabilizer row 因 ⟨w_u,S⟩=0 不改奇偶——构造时断言后跳过）。
无 numba 时 run_fast_mcmc 自动回退 reference（结果 dict 带 engine 标签）。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import gf2_matmul
from .observables import character_signs_for_label
from .prng import (
    NUMBA_AVAILABLE,
    PortablePrng,
    nb_fill_permutation,
    nb_random,
)
from .reference_mcmc import (
    MoveCounters,
    McmcState,
    make_initial_state,
    run_reference_mcmc,
)

if NUMBA_AVAILABLE:
    from numba import njit, uint64

    @njit(cache=True)
    def _assert_invariants_nb(v, syndrome_term, weights_io, check_flat,
                              check_off, gibbs_syndrome_argument, q_zero):
        data_weight = 0
        for j in range(v.shape[0]):
            data_weight += int(v[j])
        if data_weight != weights_io[0]:
            raise AssertionError("data weight cache broken")

        recomputed = gibbs_syndrome_argument.copy()
        for j in range(v.shape[0]):
            if v[j]:
                for index in range(check_off[j], check_off[j + 1]):
                    recomputed[check_flat[index]] ^= 1
        syndrome_weight = 0
        for check in range(syndrome_term.shape[0]):
            if recomputed[check] != syndrome_term[check]:
                raise AssertionError("syndrome cache broken")
            syndrome_weight += int(syndrome_term[check])
        if syndrome_weight != weights_io[1]:
            raise AssertionError("syndrome weight cache broken")
        if q_zero and syndrome_weight != 0:
            raise AssertionError("q=0 constraint violated")
else:  # pragma: no cover
    njit = None


def _csr(rows_of_indices):
    offsets = np.zeros(len(rows_of_indices) + 1, dtype=np.int64)
    for i, idx in enumerate(rows_of_indices):
        offsets[i + 1] = offsets[i] + len(idx)
    flat = np.concatenate(
        [np.asarray(idx, dtype=np.int64) for idx in rows_of_indices]
    ) if rows_of_indices else np.zeros(0, dtype=np.int64)
    if flat.size == 0:
        flat = np.zeros(0, dtype=np.int64)
    return flat, offsets


@dataclass
class FastChainData:
    check_flat: np.ndarray
    check_off: np.ndarray
    stab_flat: np.ndarray
    stab_off: np.ndarray
    log_flat: np.ndarray
    log_off: np.ndarray
    qubit_obs_words: np.ndarray   # (n, W64) uint64
    log_obs_words: np.ndarray     # (k, W64) uint64
    num_u: int
    num_words: int


def build_fast_chain_data(model, observable_set):
    check_flat, check_off = _csr(model.checks_touching_each_qubit)
    stab_supports = [np.flatnonzero(r) for r in model.stabilizer_rows]
    log_supports = [np.flatnonzero(r) for r in model.logical_move_basis]
    stab_flat, stab_off = _csr(stab_supports)
    log_flat, log_off = _csr(log_supports)

    num_u = observable_set.num_u
    num_words = max(1, (num_u + 63) // 64)
    W = observable_set.W_rows
    qubit_obs_words = np.zeros((model.num_qubits, num_words), dtype=np.uint64)
    for j in range(model.num_qubits):
        for u_idx in range(num_u):
            if W[u_idx, j]:
                qubit_obs_words[j, u_idx >> 6] |= np.uint64(1) << np.uint64(
                    u_idx & 63
                )
    log_obs_words = np.zeros((model.k, num_words), dtype=np.uint64)
    for i in range(model.k):
        flips = gf2_matmul(W, model.logical_move_basis[i][:, None])[:, 0]
        for u_idx in range(num_u):
            if flips[u_idx]:
                log_obs_words[i, u_idx >> 6] |= np.uint64(1) << np.uint64(
                    u_idx & 63
                )
    # stabilizer row 不改任何 ⟨w_u,·⟩ 奇偶（w_u 湮灭 S）——断言后 kernel 跳过
    for row in model.stabilizer_rows:
        if gf2_matmul(W, row[:, None]).any():
            raise AssertionError("stabilizer row changes observable parity")
    return FastChainData(
        check_flat=check_flat, check_off=check_off,
        stab_flat=stab_flat, stab_off=stab_off,
        log_flat=log_flat, log_off=log_off,
        qubit_obs_words=qubit_obs_words, log_obs_words=log_obs_words,
        num_u=num_u, num_words=num_words,
    )


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _one_sweep(rng_state, v, syndrome_term, weights_io, obs_parity,
                   check_flat, check_off, qubit_obs_words,
                   stab_flat, stab_off, log_flat, log_off, log_obs_words,
                   K_p, K_q, q_zero, logical_repeat,
                   counters, log_att, log_acc_counts,
                   qubit_perm, stab_perm, log_perm, num_words):
        n = v.shape[0]
        num_stab = stab_off.shape[0] - 1
        k = log_off.shape[0] - 1
        if not q_zero:
            nb_fill_permutation(rng_state, qubit_perm)
            for perm_index in range(n):
                j = qubit_perm[perm_index]
                counters[0] += 1
                delta_data = -1 if v[j] else 1
                delta_syn = 0
                for idx in range(check_off[j], check_off[j + 1]):
                    delta_syn += 1 - 2 * int(syndrome_term[check_flat[idx]])
                log_acc = -K_p * delta_data - K_q * delta_syn
                u = nb_random(rng_state)
                if log_acc >= 0.0 or u < np.exp(log_acc):
                    counters[1] += 1
                    v[j] ^= 1
                    for idx in range(check_off[j], check_off[j + 1]):
                        c = check_flat[idx]
                        weights_io[1] += 1 - 2 * int(syndrome_term[c])
                        syndrome_term[c] ^= 1
                    weights_io[0] += delta_data
                    for w in range(num_words):
                        obs_parity[w] ^= qubit_obs_words[j, w]
        nb_fill_permutation(rng_state, stab_perm)
        for perm_index in range(num_stab):
            r = stab_perm[perm_index]
            counters[2] += 1
            overlap = 0
            for idx in range(stab_off[r], stab_off[r + 1]):
                overlap += v[stab_flat[idx]]
            delta = (stab_off[r + 1] - stab_off[r]) - 2 * overlap
            log_acc = 0.0 if delta == 0 else -K_p * delta
            u = nb_random(rng_state)
            if log_acc >= 0.0 or u < np.exp(log_acc):
                counters[3] += 1
                for idx in range(stab_off[r], stab_off[r + 1]):
                    v[stab_flat[idx]] ^= 1
                weights_io[0] += delta
        for _ in range(logical_repeat):
            if k > 0:
                nb_fill_permutation(rng_state, log_perm)
                for perm_index in range(k):
                    i = log_perm[perm_index]
                    log_att[i] += 1
                    overlap = 0
                    for idx in range(log_off[i], log_off[i + 1]):
                        overlap += v[log_flat[idx]]
                    delta = (log_off[i + 1] - log_off[i]) - 2 * overlap
                    log_acc = 0.0 if delta == 0 else -K_p * delta
                    u = nb_random(rng_state)
                    if log_acc >= 0.0 or u < np.exp(log_acc):
                        log_acc_counts[i] += 1
                        for idx in range(log_off[i], log_off[i + 1]):
                            v[log_flat[idx]] ^= 1
                        weights_io[0] += delta
                        for w in range(num_words):
                            obs_parity[w] ^= log_obs_words[i, w]

    @njit(cache=True)
    def _run_chain_kernel(rng_state, v, syndrome_term, weights_io, obs_parity,
                          check_flat, check_off, qubit_obs_words,
                          stab_flat, stab_off, log_flat, log_off,
                          log_obs_words, gibbs_syndrome_argument,
                          K_p, K_q, q_zero,
                          num_burn, num_meas, sweeps_between, logical_repeat,
                          ref_signs, obs_sums, energy_trace,
                          counters, log_att, log_acc_counts,
                          record_traj, obs_traj, record_state, state_traj,
                          debug_invariants, num_words):
        n = v.shape[0]
        qubit_perm = np.empty(n, dtype=np.int64)
        stab_perm = np.empty(stab_off.shape[0] - 1, dtype=np.int64)
        log_perm = np.empty(max(log_off.shape[0] - 1, 1), dtype=np.int64)
        num_u = obs_sums.shape[0]
        for _ in range(num_burn):
            _one_sweep(rng_state, v, syndrome_term, weights_io, obs_parity,
                       check_flat, check_off, qubit_obs_words,
                       stab_flat, stab_off, log_flat, log_off, log_obs_words,
                       K_p, K_q, q_zero, logical_repeat,
                       counters, log_att, log_acc_counts,
                       qubit_perm, stab_perm, log_perm, num_words)
            if debug_invariants:
                _assert_invariants_nb(
                    v, syndrome_term, weights_io, check_flat, check_off,
                    gibbs_syndrome_argument, q_zero,
                )
        for m in range(num_meas):
            for _ in range(sweeps_between):
                _one_sweep(rng_state, v, syndrome_term, weights_io, obs_parity,
                           check_flat, check_off, qubit_obs_words,
                           stab_flat, stab_off, log_flat, log_off,
                           log_obs_words,
                           K_p, K_q, q_zero, logical_repeat,
                           counters, log_att, log_acc_counts,
                           qubit_perm, stab_perm, log_perm, num_words)
                if debug_invariants:
                    _assert_invariants_nb(
                        v, syndrome_term, weights_io, check_flat, check_off,
                        gibbs_syndrome_argument, q_zero,
                    )
            for u_idx in range(num_u):
                bit = (obs_parity[u_idx >> 6] >> uint64(u_idx & 63)) & uint64(1)
                value = ref_signs[u_idx] * (1 - 2 * int(bit))
                obs_sums[u_idx] += value
                if record_traj:
                    obs_traj[m, u_idx] = value
            if record_state:
                for j in range(n):
                    state_traj[m, j] = v[j]
            if q_zero:
                energy_trace[m] = (
                    0.0 if weights_io[0] == 0 else K_p * weights_io[0]
                )
            else:
                data_energy = (
                    0.0 if weights_io[0] == 0 else K_p * weights_io[0]
                )
                syndrome_energy = (
                    0.0 if weights_io[1] == 0 else K_q * weights_io[1]
                )
                energy_trace[m] = data_energy + syndrome_energy


def run_fast_mcmc(model, frame, observable_set, wiring, config, seed,
                  initial_mode="auto", sector_bitmask=0, chain_data=None,
                  force_reference=False):
    """与 run_reference_mcmc 同 API/同随机流的快引擎；无 numba 自动回退。"""
    if force_reference or not NUMBA_AVAILABLE:
        result = run_reference_mcmc(model, frame, observable_set, wiring,
                                    config, seed, initial_mode, sector_bitmask)
        result["engine"] = "reference_fallback"
        return result

    if chain_data is None:
        chain_data = build_fast_chain_data(model, observable_set)
    state = make_initial_state(model, wiring, initial_mode, sector_bitmask)
    rng_state = PortablePrng(seed).state_array()

    v = state.v.copy()
    syndrome_term = state.syndrome_term.copy()
    weights_io = np.array(
        [state.data_weight, state.syndrome_weight], dtype=np.int64
    )
    parities = gf2_matmul(observable_set.W_rows, v[:, None])[:, 0] \
        if observable_set.num_u else np.zeros(0, dtype=np.uint8)
    obs_parity = np.zeros(chain_data.num_words, dtype=np.uint64)
    for u_idx in range(observable_set.num_u):
        if parities[u_idx]:
            obs_parity[u_idx >> 6] |= np.uint64(1) << np.uint64(u_idx & 63)

    ref_signs = character_signs_for_label(
        observable_set, wiring.planted_logical_class
    ).astype(np.int64)
    num_u = observable_set.num_u
    obs_sums = np.zeros(num_u, dtype=np.int64)
    energy_trace = np.zeros(config.num_measurements, dtype=np.float64)
    counters_arr = np.zeros(4, dtype=np.int64)
    log_att = np.zeros(model.k, dtype=np.int64)
    log_acc_counts = np.zeros(model.k, dtype=np.int64)
    record_traj = bool(config.record_observable_trajectory)
    obs_traj = np.zeros(
        (config.num_measurements if record_traj else 1,
         num_u if record_traj else 1),
        dtype=np.int64,
    )
    record_state = bool(config.record_state_trajectory)
    state_traj = np.zeros(
        (config.num_measurements if record_state else 1,
         model.num_qubits if record_state else 1),
        dtype=np.uint8,
    )
    K_q_eff = 0.0 if wiring.q_zero else float(wiring.K_q)

    _run_chain_kernel(
        rng_state, v, syndrome_term, weights_io, obs_parity,
        chain_data.check_flat, chain_data.check_off,
        chain_data.qubit_obs_words,
        chain_data.stab_flat, chain_data.stab_off,
        chain_data.log_flat, chain_data.log_off, chain_data.log_obs_words,
        np.asarray(wiring.gibbs_syndrome_argument, dtype=np.uint8),
        float(wiring.K_p), K_q_eff, bool(wiring.q_zero),
        int(config.num_burn_in_sweeps), int(config.num_measurements),
        int(config.num_sweeps_between_measurements),
        int(config.logical_move_repeat),
        ref_signs, obs_sums, energy_trace,
        counters_arr, log_att, log_acc_counts,
        record_traj, obs_traj, record_state, state_traj,
        bool(config.debug_invariants), chain_data.num_words,
    )

    counters = MoveCounters(
        single_bit_attempts=int(counters_arr[0]),
        single_bit_accepts=int(counters_arr[1]),
        stabilizer_attempts=int(counters_arr[2]),
        stabilizer_accepts=int(counters_arr[3]),
        logical_attempts_per_u=log_att,
        logical_accepts_per_u=log_acc_counts,
    )
    final_state = McmcState(
        v=v, syndrome_term=syndrome_term,
        data_weight=int(weights_io[0]), syndrome_weight=int(weights_io[1]),
    )
    m_u_relative = (
        obs_sums.astype(np.float64) / float(config.num_measurements)
    )
    obs_sums_absolute = obs_sums * ref_signs
    relative_trajectory = obs_traj.astype(np.int8) if record_traj else None
    absolute_trajectory = (
        relative_trajectory * ref_signs[None, :]
        if relative_trajectory is not None else None
    )
    return {
        "engine": "numba",
        "m_u": m_u_relative,
        "m_u_relative": m_u_relative,
        "m_u_absolute": m_u_relative * ref_signs,
        "observable_sums": obs_sums,
        "observable_sums_relative": obs_sums,
        "observable_sums_absolute": obs_sums_absolute,
        "num_measurements": int(config.num_measurements),
        "acceptance": counters.rates(),
        "counters": counters,
        "energy_trace": energy_trace,
        "observable_trajectory": relative_trajectory,
        "observable_trajectory_relative": relative_trajectory,
        "observable_trajectory_absolute": absolute_trajectory,
        "state_trajectory": state_traj if record_state else None,
        "final_state": final_state,
    }
