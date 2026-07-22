"""Collapsed Gibbs sampler for the q=0 hypergraph-product posterior.

For a full-row-rank classical matrix ``H`` with shape ``(r, n)``, write an
X-error as matrices ``(A, B)`` on the ``A x A`` and ``B x B`` qubits.  The
hard syndrome equation is

    H A + B H = Y.

After fixing ``B``, the ``n`` columns of ``A`` are independent classical
cosets of ``H`` and contain only ``2**(n-r)`` states.  We integrate those
columns out, update blocks of a whole column of ``B`` by exact heatbath, and
then draw every column of ``A`` exactly from its conditional distribution.
The resulting transition leaves ``p**|e| (1-p)**(N-|e|)`` conditioned on the
hard syndrome invariant without proposing a macroscopic logical string.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame

try:  # pragma: no cover - exercised in the remote preflight
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


COLLAPSED_VERSION = "exp102.q0_hgp_collapsed.v1"
COLLAPSED_RAW_VERSION = "exp102.q0_hgp_collapsed.raw.v1"
COLLAPSED_METHODS = ("HC08", "HC12", "HC16", "HM08", "HM12", "HM16")
POWER_PT_METHODS = ("HP16", "HP32", "HP64")
COUNTER_NAMES = (
    "block_attempts",
    "block_changes",
    "block_changed_bits",
    "a_column_draws",
    "a_column_changes",
)


class CollapsedConflictError(ValueError):
    pass


@dataclass(frozen=True)
class CollapsedConfig:
    method_id: str
    p: float
    burn_sweeps: int
    measurement_sweeps: int

    def __post_init__(self):
        if self.method_id not in COLLAPSED_METHODS:
            raise ValueError("unknown collapsed-HGP method")
        if not 0.0 < float(self.p) < 0.5:
            raise ValueError("collapsed-HGP p must lie in (0, 0.5)")
        if int(self.burn_sweeps) <= 0 or int(self.measurement_sweeps) <= 0:
            raise ValueError("collapsed-HGP sweep counts must be positive")
        if int(self.measurement_sweeps) % 8:
            raise ValueError("measurement_sweeps must divide into eight time blocks")

    @property
    def block_size(self):
        return int(self.method_id[2:])

    @property
    def mixed_orientation(self):
        return self.method_id.startswith("HM")

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_sweeps": int(self.burn_sweeps),
            "measurement_sweeps": int(self.measurement_sweeps),
            "block_size": self.block_size,
            "mixed_orientation": self.mixed_orientation,
        }


@dataclass(frozen=True)
class CollapsedPowerPtConfig:
    method_id: str
    p: float
    burn_rounds: int
    measurement_rounds: int
    block_size: int = 8

    def __post_init__(self):
        if self.method_id not in POWER_PT_METHODS:
            raise ValueError("unknown collapsed power-PT method")
        if not 0.0 < float(self.p) < 0.5:
            raise ValueError("collapsed power-PT p must lie in (0, 0.5)")
        if int(self.burn_rounds) <= 0 or int(self.measurement_rounds) <= 0:
            raise ValueError("collapsed power-PT round counts must be positive")
        if int(self.measurement_rounds) % 8:
            raise ValueError("measurement_rounds must divide into eight time blocks")
        if int(self.block_size) != 8:
            raise ValueError("collapsed power-PT freezes eight-bit B blocks")

    @property
    def num_replicas(self):
        return int(self.method_id[2:])

    @property
    def lambda_values(self):
        denominator = (self.num_replicas - 1) ** 2
        values = np.asarray(
            [(index * index) / denominator for index in range(self.num_replicas)],
            dtype=np.float64,
        )
        values[0], values[-1] = 0.0, 1.0
        return values

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_rounds": int(self.burn_rounds),
            "measurement_rounds": int(self.measurement_rounds),
            "block_size": int(self.block_size),
            "num_replicas": self.num_replicas,
            "lambda_values": self.lambda_values.tolist(),
            "tempered_term": "collapsed_syndrome_log_likelihood_only",
        }


def _bits_to_mask(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size > 32:
        raise ValueError("collapsed-HGP masks support at most 32 bits")
    value = 0
    for bit, entry in enumerate(bits):
        value |= int(entry) << bit
    return np.uint32(value)


def _mask_to_bits(mask, size):
    return np.asarray([(int(mask) >> bit) & 1 for bit in range(int(size))], dtype=np.uint8)


def _classical_column_masks(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    return np.asarray([_bits_to_mask(H[:, column]) for column in range(H.shape[1])], dtype=np.uint32)


def _classical_row_neighbors(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    degree = int(H.sum(axis=1).max(initial=0))
    neighbors = np.full((H.shape[0], degree), -1, dtype=np.int32)
    counts = np.zeros(H.shape[0], dtype=np.int32)
    for row in range(H.shape[0]):
        values = np.flatnonzero(H[row]).astype(np.int32)
        neighbors[row, :values.size] = values
        counts[row] = values.size
    return neighbors, counts


def _section_and_kernel_masks(H):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_nullspace, gf2_rank
    from exp101_certified_src.section import build_linear_section

    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    if r > 32 or n > 32 or gf2_rank(H) != r:
        raise ValueError("collapsed-HGP requires a full-row-rank H with dimensions at most 32")
    section = build_linear_section(H)
    section_masks = np.zeros(n, dtype=np.uint32)
    for pivot_index, qubit in enumerate(section.pivot_columns):
        section_masks[qubit] = _bits_to_mask(section.solve_matrix[pivot_index])
    kernel = gf2_nullspace(H)
    basis_masks = np.asarray([_bits_to_mask(row) for row in kernel], dtype=np.uint32)
    combinations = np.zeros(1 << kernel.shape[0], dtype=np.uint32)
    for value in range(1, combinations.size):
        low = value & -value
        bit = low.bit_length() - 1
        combinations[value] = combinations[value ^ low] ^ basis_masks[bit]
    return section_masks, combinations


def validate_hgp_wiring(H, model):
    """Reject a transpose or sector mismatch before using the tensor factorization."""
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank
    from exp101_certified_src.hgp import hgp_from_H

    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    expected, _ = hgp_from_H(H)
    if model.sector != "x_error" or not np.array_equal(model.H_check, expected):
        raise CollapsedConflictError("collapsed-HGP model wiring does not match H_Z")
    if gf2_rank(H) != r or model.num_qubits != n * n + r * r:
        raise CollapsedConflictError("collapsed-HGP dimensions or classical rank changed")
    if model.k != (n - r) ** 2:
        raise CollapsedConflictError("collapsed-HGP logical dimension changed")
    return True


def split_hgp_state(state, H):
    H = np.asarray(H, dtype=np.uint8)
    r, n = H.shape
    state = np.asarray(state, dtype=np.uint8)
    if state.shape != (n * n + r * r,):
        raise ValueError("HGP state length mismatch")
    return state[:n * n].reshape(n, n).copy(), state[n * n:].reshape(r, r).copy()


def join_hgp_state(A, B):
    A = np.ascontiguousarray(A, dtype=np.uint8)
    B = np.ascontiguousarray(B, dtype=np.uint8)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("HGP state blocks must be square matrices")
    return np.concatenate((A.reshape(-1), B.reshape(-1))).astype(np.uint8, copy=False)


def hgp_syndrome_matrix(A, B, H):
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)
    H = np.asarray(H, dtype=np.uint8)
    return ((H.astype(np.int64) @ A.astype(np.int64)
             + B.astype(np.int64) @ H.astype(np.int64)) % 2).astype(np.uint8)


def _build_coset_mass_reference(column_masks, r, p):
    size = 1 << int(r)
    current = np.zeros(size, dtype=np.float64)
    scratch = np.empty(size, dtype=np.float64)
    current[0] = 1.0
    keep = np.float64(1.0 - float(p))
    flip = np.float64(p)
    for mask in np.asarray(column_masks, dtype=np.uint32):
        integer_mask = int(mask)
        for syndrome in range(size):
            left = keep * current[syndrome]
            right = flip * current[syndrome ^ integer_mask]
            scratch[syndrome] = left + right
        current, scratch = scratch, current
    return current


if njit is not None:
    @njit(cache=True)
    def _build_coset_mass_numba(column_masks, r, p):
        size = 1 << r
        current = np.zeros(size, dtype=np.float64)
        scratch = np.empty(size, dtype=np.float64)
        current[0] = 1.0
        keep = 1.0 - p
        for column in range(column_masks.size):
            mask = int(column_masks[column])
            for syndrome in range(size):
                left = keep * current[syndrome]
                right = p * current[syndrome ^ mask]
                scratch[syndrome] = left + right
            temporary = current
            current = scratch
            scratch = temporary
        return current
else:  # pragma: no cover
    _build_coset_mass_numba = None


def build_classical_coset_mass(H, p, *, engine="numba"):
    """Return ``Pr[H x=s]`` for every syndrome under iid Bernoulli(p) bits."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    if r > 24:
        raise ValueError("collapsed-HGP mass table is capped at classical rank 24")
    section_masks, combinations = _section_and_kernel_masks(H)
    del section_masks
    if combinations.size != 1 << (n - r):
        raise CollapsedConflictError("classical kernel dimension changed")
    columns = _classical_column_masks(H)
    if engine == "reference":
        result = _build_coset_mass_reference(columns, r, float(p))
    elif engine == "numba":
        if _build_coset_mass_numba is None:
            raise RuntimeError("Numba is required for the collapsed-HGP engine")
        result = _build_coset_mass_numba(columns, r, float(p))
    else:
        raise ValueError("collapsed-HGP mass engine must be reference or numba")
    if result.shape != (1 << r,) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise CollapsedConflictError("classical coset mass table is invalid")
    if abs(float(result.sum()) - 1.0) > 5e-13:
        raise CollapsedConflictError("classical coset mass table is not normalized")
    return np.ascontiguousarray(result)


def _qubit_signatures(frame):
    if int(frame.k) > 64:
        raise ValueError("collapsed-HGP labels require k <= 64")
    result = np.zeros(frame.num_qubits, dtype=np.uint64)
    for qubit in range(frame.num_qubits):
        value = np.uint64(0)
        for bit in np.flatnonzero(frame.W_basis[:, qubit]):
            value |= np.uint64(1) << np.uint64(bit)
        result[qubit] = value
    return result


def _initial_collapsed_masks(initial_state, syndrome, H):
    A, B = split_hgp_state(initial_state, H)
    r, n = H.shape
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    if not np.array_equal(hgp_syndrome_matrix(A, B, H), Y):
        raise CollapsedConflictError("collapsed-HGP initial state is outside the hard coset")
    b_columns = np.asarray([_bits_to_mask(B[:, column]) for column in range(r)], dtype=np.uint32)
    y_columns = np.asarray([_bits_to_mask(Y[:, column]) for column in range(n)], dtype=np.uint32)
    h_columns = _classical_column_masks(H)
    a_syndromes = y_columns.copy()
    for column in range(n):
        for row in np.flatnonzero(H[:, column]):
            a_syndromes[column] ^= b_columns[int(row)]
    for column in range(n):
        if _bits_to_mask((H.astype(np.int64) @ A[:, column].astype(np.int64) % 2).astype(np.uint8)) != a_syndromes[column]:
            raise CollapsedConflictError("collapsed-HGP matrix factorization is inconsistent")
    return b_columns, a_syndromes, h_columns


def _reference_advance_transport(transport_phase, round_trips,
                                 hot_origin, cold_origin):
    """Record one endpoint observation using strict cold-hot-cold cycles."""
    if transport_phase[hot_origin] == 1:
        transport_phase[hot_origin] = 2
    if transport_phase[cold_origin] == 0:
        transport_phase[cold_origin] = 1
    elif transport_phase[cold_origin] == 2:
        round_trips[cold_origin] += 1
        transport_phase[cold_origin] = 1


if njit is not None:
    @njit(cache=True, inline="always")
    def _hp_advance_transport(transport_phase, round_trips,
                              hot_origin, cold_origin):
        if transport_phase[hot_origin] == np.uint8(1):
            transport_phase[hot_origin] = np.uint8(2)
        if transport_phase[cold_origin] == np.uint8(0):
            transport_phase[cold_origin] = np.uint8(1)
        elif transport_phase[cold_origin] == np.uint8(2):
            round_trips[cold_origin] += 1
            transport_phase[cold_origin] = np.uint8(1)


    @njit(cache=True, inline="always")
    def _hc_next(state):
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << np.uint64(23))
        x = x ^ (x >> np.uint64(17))
        x = x ^ y ^ (y >> np.uint64(26))
        state[1] = x
        return x + y


    @njit(cache=True, inline="always")
    def _hc_random(state):
        return float(_hc_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


    @njit(cache=True, inline="always")
    def _hc_randbelow(state, n):
        return int(_hc_next(state) % np.uint64(n))


    @njit(cache=True, inline="always")
    def _hc_permutation(state, values):
        for index in range(values.size):
            values[index] = index
        for index in range(values.size - 1, 0, -1):
            selected = _hc_randbelow(state, index + 1)
            temporary = values[index]
            values[index] = values[selected]
            values[selected] = temporary


    @njit(cache=True, inline="always")
    def _hc_popcount(value):
        count = 0
        while value:
            count += 1
            value &= value - np.uint32(1)
        return count


    @njit(cache=True, inline="always")
    def _hc_parity(value):
        return _hc_popcount(value) & 1


    @njit(cache=True, inline="always")
    def _hc_pack(state, output):
        for byte in range(output.size):
            value = 0
            start = byte * 8
            stop = min(start + 8, state.size)
            for bit in range(start, stop):
                value |= int(state[bit]) << (bit - start)
            output[byte] = np.uint8(value)


    @njit(cache=True, inline="always")
    def _hc_update_seen(seen, label):
        for bit in range(seen.shape[0]):
            seen[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = np.uint8(1)


    @njit(cache=True)
    def _hc_build_candidate_masks(row_order, start, stop, candidate_masks, candidate_counts):
        width = stop - start
        categories = 1 << width
        candidate_masks[0] = np.uint32(0)
        candidate_counts[0] = np.uint8(0)
        for category in range(1, categories):
            low = category & -category
            bit = 0
            temporary = low
            while temporary > 1:
                bit += 1
                temporary >>= 1
            previous = category ^ low
            candidate_masks[category] = (
                candidate_masks[previous]
                | (np.uint32(1) << np.uint32(row_order[start + bit]))
            )
            candidate_counts[category] = candidate_counts[previous] + np.uint8(1)
        return categories


    @njit(cache=True)
    def _hc_update_b_sweep(b_columns, a_syndromes, rng_state, block_size,
                           neighbors, neighbor_counts, mass, odds_powers,
                           column_order, row_order, candidate_masks,
                           candidate_counts, candidate_weights, counters):
        r = b_columns.size
        _hc_permutation(rng_state, column_order)
        for column_slot in range(r):
            b_column = column_order[column_slot]
            _hc_permutation(rng_state, row_order)
            for start in range(0, r, block_size):
                stop = min(start + block_size, r)
                categories = _hc_build_candidate_masks(
                    row_order, start, stop, candidate_masks, candidate_counts,
                )
                selected_positions = np.uint32(0)
                for position in range(start, stop):
                    selected_positions |= np.uint32(1) << np.uint32(row_order[position])
                old_selected = b_columns[b_column] & selected_positions
                total = 0.0
                for category in range(categories):
                    candidate = candidate_masks[category]
                    probability = odds_powers[int(candidate_counts[category])]
                    for neighbor_slot in range(neighbor_counts[b_column]):
                        factor = neighbors[b_column, neighbor_slot]
                        base = a_syndromes[factor] ^ old_selected
                        probability *= mass[int(base ^ candidate)]
                    candidate_weights[category] = probability
                    total += probability
                if not (total > 0.0):
                    return False
                threshold = _hc_random(rng_state) * total
                cumulative = 0.0
                selected = categories - 1
                for category in range(categories):
                    cumulative += candidate_weights[category]
                    if threshold < cumulative:
                        selected = category
                        break
                new_selected = candidate_masks[selected]
                delta = old_selected ^ new_selected
                counters[0] += 1
                if delta:
                    counters[1] += 1
                    counters[2] += _hc_popcount(delta)
                    b_columns[b_column] ^= delta
                    for neighbor_slot in range(neighbor_counts[b_column]):
                        factor = neighbors[b_column, neighbor_slot]
                        a_syndromes[factor] ^= delta
        return True


    @njit(cache=True)
    def _hp_update_b_sweep(b_columns, a_syndromes, rng_state, block_size,
                           power, neighbors, neighbor_counts, log_mass,
                           log_odds, column_order, row_order,
                           candidate_masks, candidate_counts,
                           candidate_weights):
        r = b_columns.size
        _hc_permutation(rng_state, column_order)
        changes = 0
        for column_slot in range(r):
            b_column = column_order[column_slot]
            _hc_permutation(rng_state, row_order)
            for start in range(0, r, block_size):
                stop = min(start + block_size, r)
                categories = _hc_build_candidate_masks(
                    row_order, start, stop, candidate_masks, candidate_counts,
                )
                selected_positions = np.uint32(0)
                for position in range(start, stop):
                    selected_positions |= np.uint32(1) << np.uint32(row_order[position])
                old_selected = b_columns[b_column] & selected_positions
                maximum = -np.inf
                for category in range(categories):
                    candidate = candidate_masks[category]
                    log_probability = float(candidate_counts[category]) * log_odds
                    if power != 0.0:
                        likelihood = 0.0
                        for neighbor_slot in range(neighbor_counts[b_column]):
                            factor = neighbors[b_column, neighbor_slot]
                            base = a_syndromes[factor] ^ old_selected
                            likelihood += log_mass[int(base ^ candidate)]
                        log_probability += power * likelihood
                    candidate_weights[category] = log_probability
                    if log_probability > maximum:
                        maximum = log_probability
                total = 0.0
                for category in range(categories):
                    probability = np.exp(candidate_weights[category] - maximum)
                    candidate_weights[category] = probability
                    total += probability
                if not (total > 0.0):
                    return -1
                threshold = _hc_random(rng_state) * total
                cumulative = 0.0
                selected = categories - 1
                for category in range(categories):
                    cumulative += candidate_weights[category]
                    if threshold < cumulative:
                        selected = category
                        break
                new_selected = candidate_masks[selected]
                delta = old_selected ^ new_selected
                if delta:
                    changes += 1
                    b_columns[b_column] ^= delta
                    for neighbor_slot in range(neighbor_counts[b_column]):
                        factor = neighbors[b_column, neighbor_slot]
                        a_syndromes[factor] ^= delta
        return changes


    @njit(cache=True, inline="always")
    def _hp_log_likelihood(a_syndromes, log_mass):
        result = 0.0
        for factor in range(a_syndromes.size):
            result += log_mass[int(a_syndromes[factor])]
        return result


    @njit(cache=True)
    def _run_power_pt_core(initial_state, initial_b, initial_syndromes,
                           burn_rng_states, measurement_rng_states,
                           burn_observation_rng, measurement_observation_rng,
                           lambdas, burn_rounds,
                           measurement_rounds, block_size, n, r,
                           section_masks, kernel_combinations, neighbors,
                           neighbor_counts, log_mass, log_odds, odds_powers,
                           qubit_signatures):
        replicas = lambdas.size
        b_states = np.empty((replicas, r), dtype=np.uint32)
        syndrome_states = np.empty((replicas, n), dtype=np.uint32)
        origins = np.empty(replicas, dtype=np.int32)
        for rung in range(replicas):
            b_states[rung] = initial_b
            syndrome_states[rung] = initial_syndromes
            origins[rung] = rung
        total_rounds = burn_rounds + measurement_rounds
        labels = np.empty(measurement_rounds, dtype=np.uint64)
        weights = np.empty(measurement_rounds, dtype=np.int32)
        packed = np.empty(
            (measurement_rounds, (initial_state.size + 7) // 8), dtype=np.uint8,
        )
        burn_labels = np.empty(burn_rounds, dtype=np.uint64)
        local_attempts = np.zeros(replicas, dtype=np.int64)
        local_changes = np.zeros(replicas, dtype=np.int64)
        swap_attempts = np.zeros(replicas - 1, dtype=np.int64)
        swap_accepts = np.zeros(replicas - 1, dtype=np.int64)
        hot_visits = np.zeros(replicas, dtype=np.int64)
        cold_visits = np.zeros(replicas, dtype=np.int64)
        # A round trip is cold -> hot -> cold.  Origins that start away from
        # the cold endpoint must first establish a cold visit.
        transport_phase = np.zeros(replicas, dtype=np.uint8)
        transport_phase[origins[replicas - 1]] = np.uint8(1)
        round_trips = np.zeros(replicas, dtype=np.int64)
        column_order = np.empty(r, dtype=np.int32)
        row_order = np.empty(r, dtype=np.int32)
        categories = 1 << min(block_size, r)
        candidate_masks = np.empty(categories, dtype=np.uint32)
        candidate_counts = np.empty(categories, dtype=np.uint8)
        candidate_weights = np.empty(max(categories, kernel_combinations.size), dtype=np.float64)
        likelihoods = np.empty(replicas, dtype=np.float64)
        state = initial_state.copy()
        previous_state = initial_state.copy()
        burn_endpoint = initial_state.copy()
        cold_burn_weight = np.empty(burn_rounds, dtype=np.int32)
        cold_likelihood = np.empty(total_rounds, dtype=np.float64)
        for round_index in range(total_rounds):
            stage_rng_states = (
                burn_rng_states if round_index < burn_rounds
                else measurement_rng_states
            )
            stage_observation_rng = (
                burn_observation_rng if round_index < burn_rounds
                else measurement_observation_rng
            )
            for rung in range(replicas):
                changed = _hp_update_b_sweep(
                    b_states[rung], syndrome_states[rung], stage_rng_states[rung],
                    block_size, lambdas[rung], neighbors, neighbor_counts,
                    log_mass, log_odds, column_order, row_order,
                    candidate_masks, candidate_counts, candidate_weights,
                )
                if changed < 0:
                    return (state, burn_endpoint, packed, burn_labels, labels,
                            weights, local_attempts, local_changes, swap_attempts,
                            swap_accepts, hot_visits, cold_visits, round_trips,
                            cold_burn_weight, cold_likelihood, origins, False)
                local_attempts[rung] += (r * ((r + block_size - 1) // block_size))
                local_changes[rung] += changed
                likelihoods[rung] = _hp_log_likelihood(syndrome_states[rung], log_mass)
            parity = round_index & 1
            for lower in range(parity, replicas - 1, 2):
                upper = lower + 1
                delta = ((lambdas[lower] - lambdas[upper])
                         * (likelihoods[upper] - likelihoods[lower]))
                swap_attempts[lower] += 1
                accept = delta >= 0.0
                if not accept:
                    accept = _hc_random(stage_rng_states[lower]) < np.exp(delta)
                if accept:
                    swap_accepts[lower] += 1
                    temporary_b = b_states[lower].copy()
                    b_states[lower] = b_states[upper]
                    b_states[upper] = temporary_b
                    temporary_s = syndrome_states[lower].copy()
                    syndrome_states[lower] = syndrome_states[upper]
                    syndrome_states[upper] = temporary_s
                    temporary_likelihood = likelihoods[lower]
                    likelihoods[lower] = likelihoods[upper]
                    likelihoods[upper] = temporary_likelihood
                    temporary_origin = origins[lower]
                    origins[lower] = origins[upper]
                    origins[upper] = temporary_origin
            hot_origin = origins[0]
            cold_origin = origins[replicas - 1]
            hot_visits[hot_origin] += 1
            cold_visits[cold_origin] += 1
            _hp_advance_transport(
                transport_phase, round_trips, hot_origin, cold_origin,
            )
            cold_likelihood[round_index] = likelihoods[replicas - 1]
            previous_state[:] = state
            label, weight = _hc_sample_full_state(
                state, previous_state, b_states[replicas - 1],
                syndrome_states[replicas - 1], stage_observation_rng, n, r,
                section_masks, kernel_combinations, odds_powers,
                qubit_signatures, candidate_weights,
                np.zeros(5, dtype=np.int64),
            )
            if round_index < burn_rounds:
                burn_labels[round_index] = label
                cold_burn_weight[round_index] = weight
                if round_index + 1 == burn_rounds:
                    burn_endpoint[:] = state
            else:
                measurement = round_index - burn_rounds
                labels[measurement] = label
                weights[measurement] = weight
                _hc_pack(state, packed[measurement])
        return (state, burn_endpoint, packed, burn_labels, labels, weights,
                local_attempts, local_changes, swap_attempts, swap_accepts,
                hot_visits, cold_visits, round_trips, cold_burn_weight,
                cold_likelihood, origins, True)


    @njit(cache=True)
    def _hc_update_b_row_sweep(b_columns, a_syndromes, rng_state, block_size,
                               neighbors, neighbor_counts, h_columns, log_mass,
                               log_odds, column_order, row_order,
                               candidate_masks, candidate_counts,
                               candidate_weights, counters):
        r = b_columns.size
        n = h_columns.size
        _hc_permutation(rng_state, row_order)
        for row_slot in range(r):
            matrix_row = row_order[row_slot]
            row_bit = np.uint32(1) << np.uint32(matrix_row)
            _hc_permutation(rng_state, column_order)
            for start in range(0, r, block_size):
                stop = min(start + block_size, r)
                categories = _hc_build_candidate_masks(
                    column_order, start, stop, candidate_masks, candidate_counts,
                )
                selected_positions = np.uint32(0)
                old_selected = np.uint32(0)
                for position in range(start, stop):
                    matrix_column = column_order[position]
                    column_bit = np.uint32(1) << np.uint32(matrix_column)
                    selected_positions |= column_bit
                    if b_columns[matrix_column] & row_bit:
                        old_selected |= column_bit
                maximum = -np.inf
                for category in range(categories):
                    candidate = candidate_masks[category]
                    log_probability = float(candidate_counts[category]) * log_odds
                    for factor in range(n):
                        touched = selected_positions & h_columns[factor]
                        if touched:
                            base = a_syndromes[factor]
                            if _hc_parity(old_selected & h_columns[factor]):
                                base ^= row_bit
                            if _hc_parity(candidate & h_columns[factor]):
                                base ^= row_bit
                            log_probability += log_mass[int(base)]
                    candidate_weights[category] = log_probability
                    if log_probability > maximum:
                        maximum = log_probability
                total = 0.0
                for category in range(categories):
                    probability = np.exp(candidate_weights[category] - maximum)
                    candidate_weights[category] = probability
                    total += probability
                if not (total > 0.0):
                    return False
                threshold = _hc_random(rng_state) * total
                cumulative = 0.0
                selected = categories - 1
                for category in range(categories):
                    cumulative += candidate_weights[category]
                    if threshold < cumulative:
                        selected = category
                        break
                new_selected = candidate_masks[selected]
                delta = old_selected ^ new_selected
                counters[0] += 1
                if delta:
                    counters[1] += 1
                    counters[2] += _hc_popcount(delta)
                    for matrix_column in range(r):
                        column_bit = np.uint32(1) << np.uint32(matrix_column)
                        if delta & column_bit:
                            b_columns[matrix_column] ^= row_bit
                            for neighbor_slot in range(neighbor_counts[matrix_column]):
                                factor = neighbors[matrix_column, neighbor_slot]
                                a_syndromes[factor] ^= row_bit
        return True


    @njit(cache=True)
    def _hc_sample_full_state(state, previous_state, b_columns, a_syndromes,
                              rng_state, n, r, section_masks,
                              kernel_combinations, odds_powers,
                              qubit_signatures, candidate_weights, counters):
        for qubit in range(state.size):
            state[qubit] = np.uint8(0)
        label = np.uint64(0)
        weight = 0
        for column in range(n):
            syndrome = a_syndromes[column]
            section_state = np.uint32(0)
            for row in range(n):
                if _hc_parity(section_masks[row] & syndrome):
                    section_state |= np.uint32(1) << np.uint32(row)
            total = 0.0
            for category in range(kernel_combinations.size):
                candidate = section_state ^ kernel_combinations[category]
                probability = odds_powers[_hc_popcount(candidate)]
                candidate_weights[category] = probability
                total += probability
            threshold = _hc_random(rng_state) * total
            cumulative = 0.0
            selected = kernel_combinations.size - 1
            for category in range(kernel_combinations.size):
                cumulative += candidate_weights[category]
                if threshold < cumulative:
                    selected = category
                    break
            chosen = section_state ^ kernel_combinations[selected]
            counters[3] += 1
            changed = False
            for row in range(n):
                qubit = row * n + column
                bit = np.uint8((chosen >> np.uint32(row)) & np.uint32(1))
                state[qubit] = bit
                if bit:
                    weight += 1
                    label ^= qubit_signatures[qubit]
                if bit != previous_state[qubit]:
                    changed = True
            if changed:
                counters[4] += 1
        offset = n * n
        for column in range(r):
            value = b_columns[column]
            for row in range(r):
                qubit = offset + row * r + column
                bit = np.uint8((value >> np.uint32(row)) & np.uint32(1))
                state[qubit] = bit
                if bit:
                    weight += 1
                    label ^= qubit_signatures[qubit]
        return label, weight


    @njit(cache=True)
    def _run_collapsed_stage_numba_core(initial_state, b_columns,
                                        a_syndromes, rng_state, sweeps,
                                        record_states, block_size, n, r,
                                        section_masks, kernel_combinations,
                                        neighbors, neighbor_counts, h_columns,
                                        mass, log_mass, log_odds,
                                        mixed_orientation,
                                        odds_powers, qubit_signatures):
        bytes_per_state = (initial_state.size + 7) // 8
        packed_count = sweeps if record_states else 0
        packed = np.empty((packed_count, bytes_per_state), dtype=np.uint8)
        labels = np.empty(sweeps, dtype=np.uint64)
        weights = np.empty(sweeps, dtype=np.int32)
        counters = np.zeros(5, dtype=np.int64)
        state = initial_state.copy()
        previous_state = initial_state.copy()
        column_order = np.empty(r, dtype=np.int32)
        row_order = np.empty(r, dtype=np.int32)
        categories = 1 << min(block_size, r)
        candidate_masks = np.empty(categories, dtype=np.uint32)
        candidate_counts = np.empty(categories, dtype=np.uint8)
        candidate_weights = np.empty(max(categories, kernel_combinations.size), dtype=np.float64)
        for sweep in range(sweeps):
            ok = _hc_update_b_sweep(
                b_columns, a_syndromes, rng_state, block_size, neighbors,
                neighbor_counts, mass, odds_powers, column_order, row_order,
                candidate_masks, candidate_counts, candidate_weights, counters,
            )
            if not ok:
                return state, b_columns, a_syndromes, packed, labels, weights, counters, False
            if mixed_orientation:
                ok = _hc_update_b_row_sweep(
                    b_columns, a_syndromes, rng_state, block_size, neighbors,
                    neighbor_counts, h_columns, log_mass, log_odds,
                    column_order, row_order, candidate_masks, candidate_counts,
                    candidate_weights, counters,
                )
                if not ok:
                    return state, b_columns, a_syndromes, packed, labels, weights, counters, False
            previous_state[:] = state
            label, weight = _hc_sample_full_state(
                state, previous_state, b_columns, a_syndromes, rng_state, n, r,
                section_masks, kernel_combinations, odds_powers,
                qubit_signatures, candidate_weights, counters,
            )
            labels[sweep] = label
            weights[sweep] = weight
            if record_states:
                _hc_pack(state, packed[sweep])
        return state, b_columns, a_syndromes, packed, labels, weights, counters, True
else:  # pragma: no cover
    _run_collapsed_stage_numba_core = None
    _run_power_pt_core = None


def _reference_candidate_masks(order, start, stop):
    width = int(stop) - int(start)
    masks = np.zeros(1 << width, dtype=np.uint32)
    counts = np.zeros(1 << width, dtype=np.uint8)
    for category in range(1, masks.size):
        low = category & -category
        bit = low.bit_length() - 1
        previous = category ^ low
        masks[category] = masks[previous] | (
            np.uint32(1) << np.uint32(order[start + bit])
        )
        counts[category] = counts[previous] + np.uint8(1)
    return masks, counts


def _reference_categorical_draw(weights, rng):
    total = float(np.sum(weights, dtype=np.float64))
    if not total > 0.0:
        raise CollapsedConflictError("reference categorical weights vanished")
    threshold = rng.random() * total
    cumulative = 0.0
    selected = len(weights) - 1
    for category, probability in enumerate(weights):
        cumulative += float(probability)
        if threshold < cumulative:
            selected = category
            break
    return selected


def _reference_power_b_sweep(b_columns, a_syndromes, rng, block_size,
                             power, neighbors, neighbor_counts, log_mass,
                             log_odds, *, categorical_draw=None):
    if categorical_draw is None:
        categorical_draw = _reference_categorical_draw
    r = b_columns.size
    changes = 0
    for b_column in rng.permutation(r):
        b_column = int(b_column)
        row_order = rng.permutation(r)
        for start in range(0, r, block_size):
            stop = min(start + block_size, r)
            masks, counts = _reference_candidate_masks(row_order, start, stop)
            selected_positions = np.uint32(0)
            for position in range(start, stop):
                selected_positions |= np.uint32(1) << np.uint32(row_order[position])
            old_selected = b_columns[b_column] & selected_positions
            weights = np.empty(masks.size, dtype=np.float64)
            maximum = -np.inf
            for category, candidate in enumerate(masks):
                log_probability = float(counts[category]) * float(log_odds)
                if power != 0.0:
                    likelihood = 0.0
                    for neighbor_slot in range(int(neighbor_counts[b_column])):
                        factor = int(neighbors[b_column, neighbor_slot])
                        base = a_syndromes[factor] ^ old_selected
                        likelihood += log_mass[int(base ^ candidate)]
                    log_probability += float(power) * likelihood
                weights[category] = log_probability
                if log_probability > maximum:
                    maximum = log_probability
            for category in range(weights.size):
                weights[category] = np.exp(weights[category] - maximum)
            selected = categorical_draw(weights, rng)
            delta = old_selected ^ masks[selected]
            if delta:
                changes += 1
                b_columns[b_column] ^= delta
                for neighbor_slot in range(int(neighbor_counts[b_column])):
                    factor = int(neighbors[b_column, neighbor_slot])
                    a_syndromes[factor] ^= delta
    return changes


def _reference_sample_full_state(previous_state, b_columns, a_syndromes, rng,
                                 n, r, section_masks, kernel_combinations,
                                 odds_powers, qubit_signatures, *,
                                 categorical_draw=None):
    if categorical_draw is None:
        categorical_draw = _reference_categorical_draw
    state = np.zeros_like(previous_state)
    label = np.uint64(0)
    weight = 0
    for column in range(n):
        syndrome = a_syndromes[column]
        section_state = np.uint32(0)
        for row in range(n):
            if int(section_masks[row] & syndrome).bit_count() & 1:
                section_state |= np.uint32(1) << np.uint32(row)
        probabilities = np.empty(kernel_combinations.size, dtype=np.float64)
        for category, combination in enumerate(kernel_combinations):
            candidate = section_state ^ combination
            probability = odds_powers[int(candidate).bit_count()]
            probabilities[category] = probability
        selected = categorical_draw(probabilities, rng)
        chosen = section_state ^ kernel_combinations[selected]
        for row in range(n):
            qubit = row * n + column
            bit = np.uint8((chosen >> np.uint32(row)) & np.uint32(1))
            state[qubit] = bit
            if bit:
                weight += 1
                label ^= qubit_signatures[qubit]
    offset = n * n
    for column in range(r):
        value = b_columns[column]
        for row in range(r):
            qubit = offset + row * r + column
            bit = np.uint8((value >> np.uint32(row)) & np.uint32(1))
            state[qubit] = bit
            if bit:
                weight += 1
                label ^= qubit_signatures[qubit]
    return state, label, weight


def _run_power_pt_reference_core(initial_state, initial_b, initial_syndromes,
                                 burn_rngs, measurement_rngs,
                                 burn_observation_rng, measurement_observation_rng,
                                 lambdas, burn_rounds,
                                 measurement_rounds, block_size, n, r,
                                 section_masks, kernel_combinations, neighbors,
                                 neighbor_counts, log_mass, log_odds,
                                 odds_powers, qubit_signatures):
    replicas = lambdas.size
    b_states = np.repeat(initial_b[None, :], replicas, axis=0)
    syndrome_states = np.repeat(initial_syndromes[None, :], replicas, axis=0)
    origins = np.arange(replicas, dtype=np.int32)
    total_rounds = burn_rounds + measurement_rounds
    labels = np.empty(measurement_rounds, dtype=np.uint64)
    weights = np.empty(measurement_rounds, dtype=np.int32)
    packed = np.empty(
        (measurement_rounds, (initial_state.size + 7) // 8), dtype=np.uint8,
    )
    burn_labels = np.empty(burn_rounds, dtype=np.uint64)
    local_attempts = np.zeros(replicas, dtype=np.int64)
    local_changes = np.zeros(replicas, dtype=np.int64)
    swap_attempts = np.zeros(replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(replicas - 1, dtype=np.int64)
    hot_visits = np.zeros(replicas, dtype=np.int64)
    cold_visits = np.zeros(replicas, dtype=np.int64)
    transport_phase = np.zeros(replicas, dtype=np.uint8)
    transport_phase[origins[-1]] = 1
    round_trips = np.zeros(replicas, dtype=np.int64)
    state = initial_state.copy()
    burn_endpoint = initial_state.copy()
    cold_burn_weight = np.empty(burn_rounds, dtype=np.int32)
    cold_likelihood = np.empty(total_rounds, dtype=np.float64)
    likelihoods = np.empty(replicas, dtype=np.float64)
    attempts_per_sweep = r * ((r + block_size - 1) // block_size)
    for round_index in range(total_rounds):
        stage_rngs = burn_rngs if round_index < burn_rounds else measurement_rngs
        stage_observation_rng = (
            burn_observation_rng if round_index < burn_rounds
            else measurement_observation_rng
        )
        for rung in range(replicas):
            local_changes[rung] += _reference_power_b_sweep(
                b_states[rung], syndrome_states[rung], stage_rngs[rung], block_size,
                lambdas[rung], neighbors, neighbor_counts, log_mass, log_odds,
            )
            local_attempts[rung] += attempts_per_sweep
            likelihood = 0.0
            for factor in range(n):
                likelihood += log_mass[int(syndrome_states[rung, factor])]
            likelihoods[rung] = likelihood
        for lower in range(round_index & 1, replicas - 1, 2):
            upper = lower + 1
            delta = ((lambdas[lower] - lambdas[upper])
                     * (likelihoods[upper] - likelihoods[lower]))
            swap_attempts[lower] += 1
            accept = delta >= 0.0
            if not accept:
                accept = stage_rngs[lower].random() < np.exp(delta)
            if accept:
                swap_accepts[lower] += 1
                b_states[[lower, upper]] = b_states[[upper, lower]]
                syndrome_states[[lower, upper]] = syndrome_states[[upper, lower]]
                likelihoods[[lower, upper]] = likelihoods[[upper, lower]]
                origins[[lower, upper]] = origins[[upper, lower]]
        hot_origin, cold_origin = int(origins[0]), int(origins[-1])
        hot_visits[hot_origin] += 1
        cold_visits[cold_origin] += 1
        _reference_advance_transport(
            transport_phase, round_trips, hot_origin, cold_origin,
        )
        cold_likelihood[round_index] = likelihoods[-1]
        state, label, weight = _reference_sample_full_state(
            state, b_states[-1], syndrome_states[-1], stage_observation_rng, n, r,
            section_masks, kernel_combinations, odds_powers, qubit_signatures,
        )
        if round_index < burn_rounds:
            burn_labels[round_index] = label
            cold_burn_weight[round_index] = weight
            if round_index + 1 == burn_rounds:
                burn_endpoint = state.copy()
        else:
            measurement = round_index - burn_rounds
            labels[measurement] = label
            weights[measurement] = weight
            packed[measurement] = _pack_state(state)
    return (state, burn_endpoint, packed, burn_labels, labels, weights,
            local_attempts, local_changes, swap_attempts, swap_accepts,
            hot_visits, cold_visits, round_trips, cold_burn_weight,
            cold_likelihood, origins, True)


def _state_label(frame, state):
    bits = frame.label_of(np.asarray(state, dtype=np.uint8))
    value = np.uint64(0)
    for bit, entry in enumerate(bits):
        if entry:
            value |= np.uint64(1) << np.uint64(bit)
    return value


def _basis_seen(labels, k):
    result = np.zeros((int(k), 2), dtype=np.uint8)
    for label in np.asarray(labels, dtype=np.uint64):
        for bit in range(int(k)):
            result[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = 1
    return result


def _pack_state(state):
    return np.packbits(np.asarray(state, dtype=np.uint8), bitorder="little")


def _run_stage_numba(state, b_columns, a_syndromes, config, sweeps, record,
                     rng_state, H, section_masks, kernel_combinations,
                     neighbors, neighbor_counts, mass, odds_powers,
                     qubit_signatures):
    if _run_collapsed_stage_numba_core is None:
        raise RuntimeError("Numba is required for the collapsed-HGP engine")
    r, n = H.shape
    return _run_collapsed_stage_numba_core(
        np.ascontiguousarray(state, dtype=np.uint8),
        np.ascontiguousarray(b_columns, dtype=np.uint32),
        np.ascontiguousarray(a_syndromes, dtype=np.uint32),
        np.ascontiguousarray(rng_state, dtype=np.uint64), int(sweeps), bool(record),
        int(config.block_size), int(n), int(r), section_masks,
        kernel_combinations, neighbors, neighbor_counts,
        _classical_column_masks(H), mass, np.ascontiguousarray(np.log(mass)),
        float(np.log(float(config.p) / (1.0 - float(config.p)))),
        bool(config.mixed_orientation), odds_powers,
        qubit_signatures,
    )


def run_collapsed_trajectory(model, frame, H, syndrome, config,
                             seed_identity, initial_state, *, engine="numba",
                             mass=None):
    """Run one fresh collapsed-HGP trajectory from an arbitrary hard-coset state."""
    if engine != "numba":
        raise ValueError("the first collapsed-HGP production prototype supports engine='numba'")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise CollapsedConflictError("collapsed-HGP observable frame mismatch") from exc
    H = np.ascontiguousarray(H, dtype=np.uint8)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,):
        raise ValueError("collapsed-HGP syndrome shape mismatch")
    if config.method_id != seed_identity.method_id:
        raise CollapsedConflictError("collapsed-HGP config/seed method mismatch")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    expected_mass = build_classical_coset_mass(H, config.p, engine="numba")
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if mass.shape != (1 << H.shape[0],) or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise CollapsedConflictError("collapsed-HGP supplied mass table is invalid")
    if not np.array_equal(mass, expected_mass):
        raise CollapsedConflictError("collapsed-HGP supplied mass table does not match H and p")
    odds = float(config.p) / (1.0 - float(config.p))
    odds_powers = np.ones(max(H.shape) + 1, dtype=np.float64)
    for value in range(1, odds_powers.size):
        odds_powers[value] = odds_powers[value - 1] * odds
    signatures = _qubit_signatures(frame)
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    initial = state.copy()
    burn = _run_stage_numba(
        state, b_columns, a_syndromes, config, config.burn_sweeps, False,
        PortablePrng(seed_identity.seed("burn")).state_array(), H,
        section_masks, kernel_combinations, neighbors, neighbor_counts, mass,
        odds_powers, signatures,
    )
    if not burn[-1]:
        raise CollapsedConflictError("collapsed-HGP burn categorical weights vanished")
    state, b_columns, a_syndromes = burn[0], burn[1], burn[2]
    burn_endpoint = state.copy()
    measured = _run_stage_numba(
        state, b_columns, a_syndromes, config, config.measurement_sweeps, True,
        PortablePrng(seed_identity.seed("measurement")).state_array(), H,
        section_masks, kernel_combinations, neighbors, neighbor_counts, mass,
        odds_powers, signatures,
    )
    if not measured[-1]:
        raise CollapsedConflictError("collapsed-HGP measurement categorical weights vanished")
    state, packed, labels, weights = measured[0], measured[3], measured[4], measured[5]
    unpacked = np.unpackbits(packed, axis=1, count=model.num_qubits, bitorder="little").astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    if residuals.any():
        raise CollapsedConflictError("collapsed-HGP emitted a state outside the hard coset")
    replay_labels = np.asarray([_state_label(frame, row) for row in unpacked], dtype=np.uint64)
    if not np.array_equal(labels, replay_labels) or not np.array_equal(weights, unpacked.sum(axis=1)):
        raise CollapsedConflictError("collapsed-HGP cached labels or weights drifted")
    mass_digest = hashlib.sha256(np.asarray(mass, dtype=">f8").tobytes()).hexdigest()
    return {
        "raw_version": COLLAPSED_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "initial_state_packed": _pack_state(initial),
        "burn_state_packed": _pack_state(burn_endpoint),
        "final_state_packed": _pack_state(state),
        "measurement_states_packed": packed,
        "burn_labels": burn[4],
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_sweeps // 8,
        ),
        "burn_counters": burn[6],
        "measurement_counters": measured[6],
        "burn_basis_seen": _basis_seen(burn[4], model.k),
        "initial_label": _state_label(frame, initial),
        "burn_label": _state_label(frame, burn_endpoint),
        "final_label": _state_label(frame, state),
        "mass_sha256": mass_digest,
        "engine": engine,
    }


def run_collapsed_power_pt_trajectory(model, frame, H, syndrome, config,
                                      seed_identity, initial_state, *,
                                      engine="numba", mass=None):
    """Run likelihood-power replica exchange on the exact collapsed marginal."""
    if engine not in ("reference", "numba"):
        raise ValueError("collapsed power PT engine must be reference or numba")
    if engine == "numba" and _run_power_pt_core is None:
        raise RuntimeError("Numba is required for accelerated collapsed power PT")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise CollapsedConflictError("collapsed power-PT observable frame mismatch") from exc
    H = np.ascontiguousarray(H, dtype=np.uint8)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,):
        raise ValueError("collapsed power-PT syndrome shape mismatch")
    if config.method_id != seed_identity.method_id:
        raise CollapsedConflictError("collapsed power-PT config/seed method mismatch")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    mass_engine = "numba" if _build_coset_mass_numba is not None else "reference"
    expected_mass = build_classical_coset_mass(H, config.p, engine=mass_engine)
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if mass.shape != (1 << H.shape[0],) or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise CollapsedConflictError("collapsed power-PT supplied mass table is invalid")
    if not np.array_equal(mass, expected_mass):
        raise CollapsedConflictError("collapsed power-PT supplied mass table does not match H and p")
    log_mass = np.ascontiguousarray(np.log(mass))
    odds = float(config.p) / (1.0 - float(config.p))
    odds_powers = np.ones(max(H.shape) + 1, dtype=np.float64)
    for value in range(1, odds_powers.size):
        odds_powers[value] = odds_powers[value - 1] * odds
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    if engine == "numba":
        burn_rng_states = np.asarray([
            PortablePrng(seed_identity.seed("burn", "replica", rung)).state_array()
            for rung in range(config.num_replicas)
        ], dtype=np.uint64)
        measurement_rng_states = np.asarray([
            PortablePrng(seed_identity.seed("measurement", "replica", rung)).state_array()
            for rung in range(config.num_replicas)
        ], dtype=np.uint64)
        result = _run_power_pt_core(
            state, b_columns, a_syndromes, burn_rng_states,
            measurement_rng_states,
            PortablePrng(seed_identity.seed("burn", "observation")).state_array(),
            PortablePrng(seed_identity.seed("measurement", "observation")).state_array(),
            np.ascontiguousarray(config.lambda_values), int(config.burn_rounds),
            int(config.measurement_rounds), int(config.block_size), int(H.shape[1]),
            int(H.shape[0]), section_masks, kernel_combinations, neighbors,
            neighbor_counts, log_mass, float(np.log(odds)), odds_powers,
            _qubit_signatures(frame),
        )
    else:
        burn_rngs = [
            PortablePrng(seed_identity.seed("burn", "replica", rung))
            for rung in range(config.num_replicas)
        ]
        measurement_rngs = [
            PortablePrng(seed_identity.seed("measurement", "replica", rung))
            for rung in range(config.num_replicas)
        ]
        result = _run_power_pt_reference_core(
            state, b_columns, a_syndromes, burn_rngs, measurement_rngs,
            PortablePrng(seed_identity.seed("burn", "observation")),
            PortablePrng(seed_identity.seed("measurement", "observation")),
            config.lambda_values, int(config.burn_rounds),
            int(config.measurement_rounds), int(config.block_size), int(H.shape[1]),
            int(H.shape[0]), section_masks, kernel_combinations, neighbors,
            neighbor_counts, log_mass, float(np.log(odds)), odds_powers,
            _qubit_signatures(frame),
        )
    if not result[-1]:
        raise CollapsedConflictError("collapsed power-PT categorical weights vanished")
    final_state, burn_endpoint, packed = result[0], result[1], result[2]
    burn_labels, labels, weights = result[3], result[4], result[5]
    unpacked = np.unpackbits(
        packed, axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    replay_labels = np.asarray(
        [_state_label(frame, row) for row in unpacked], dtype=np.uint64,
    )
    if (residuals.any() or not np.array_equal(labels, replay_labels)
            or not np.array_equal(weights, unpacked.sum(axis=1))):
        raise CollapsedConflictError("collapsed power-PT raw replay failed")
    lambdas = config.lambda_values
    ladder_digest = hashlib.sha256(np.asarray(lambdas, dtype=">f8").tobytes()).hexdigest()
    mass_digest = hashlib.sha256(np.asarray(mass, dtype=">f8").tobytes()).hexdigest()
    return {
        "raw_version": COLLAPSED_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "initial_state_packed": _pack_state(state),
        "burn_state_packed": _pack_state(burn_endpoint),
        "final_state_packed": _pack_state(final_state),
        "measurement_states_packed": packed,
        "burn_labels": burn_labels,
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_rounds // 8,
        ),
        "burn_basis_seen": _basis_seen(burn_labels, model.k),
        "initial_label": _state_label(frame, state),
        "burn_label": _state_label(frame, burn_endpoint),
        "final_label": _state_label(frame, final_state),
        "local_attempts_by_rung": result[6],
        "local_changes_by_rung": result[7],
        "swap_attempts": result[8],
        "swap_accepts": result[9],
        "hot_visits_by_origin": result[10],
        "cold_visits_by_origin": result[11],
        "round_trips_by_origin": result[12],
        "cold_burn_weights": result[13],
        "cold_log_likelihood": result[14],
        "final_origins_by_rung": result[15],
        "lambda_values": lambdas,
        "lambda_sha256": ladder_digest,
        "mass_sha256": mass_digest,
        "round_trip_definition": "cold_hot_cold_after_established_cold_visit",
        "engine": engine,
    }
