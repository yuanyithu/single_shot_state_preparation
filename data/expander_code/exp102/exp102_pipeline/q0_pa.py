"""Fixed-schedule q=0 population annealing for exp102 discovery.

This module is deliberately independent of the formal PT-v1 pipeline.  A
population starts exactly uniformly on the requested hard coset at ``K=0``,
then performs fixed resample-move SMC steps on a frozen Q32 schedule.  The
Python implementation is the oracle; Numba accelerates only the mutation
kernel and consumes the same pre-derived PortablePrng substreams.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .labels import bits_to_uint64
from .q0_pt import Q32_ONE, coupling, ladder_x_q32_sha256
from .seeds import derive_seed

try:
    from numba import njit
except ImportError:  # pragma: no cover - remote preflight requires Numba
    njit = None


PA_CONTRACT_VERSION = "exp102.q0_pa.discovery.v1"
PA_SCHEDULE_VERSION = "exp102.q0_pa.theta_q32.v1"
PA_RESAMPLE_ESS_FRACTION = 0.5
PA_LOGICAL_KERNELS = ("coordinate", "block4")


class PaConflictError(ValueError):
    """An identity, algebraic, or transcript conflict in PA evidence."""


def theta_schedule_q32(p_target, num_anneal_steps):
    """Construct the contract Q32 array for ``K_t/K_G``.

    The floating-point construction is used only while freezing config data.
    Production runs consume the stored integers and verify their SHA256.
    """
    p_target = float(p_target)
    steps = int(num_anneal_steps)
    if not 0.0 < p_target < 0.5 or steps <= 0:
        raise ValueError("invalid PA target probability or anneal step count")
    theta_final = math.asin(math.sqrt(p_target))
    final_K = coupling(p_target)
    values = []
    for stage in range(steps + 1):
        theta = math.pi / 4.0 + (stage / steps) * (theta_final - math.pi / 4.0)
        p_stage = math.sin(theta) ** 2
        K_stage = 0.0 if stage == 0 else math.log((1.0 - p_stage) / p_stage)
        value = int(math.floor((K_stage / final_K) * Q32_ONE + 0.5))
        values.append(value)
    values[0] = 0
    values[-1] = Q32_ONE
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Q32 PA schedule is not strictly increasing")
    return tuple(values)


def validate_pa_schedule_q32(schedule_q32, p_target, num_anneal_steps):
    if not isinstance(schedule_q32, (tuple, list)):
        raise ValueError("PA Q32 schedule must be a tuple or list")
    values = []
    for value in schedule_q32:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("PA Q32 schedule entries must be integers")
        values.append(int(value))
    if len(values) != int(num_anneal_steps) + 1:
        raise ValueError("PA Q32 schedule length must be G+1")
    if values[0] != 0 or values[-1] != Q32_ONE:
        raise ValueError("PA Q32 schedule endpoints must be 0 and 2**32")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("PA Q32 schedule must be strictly increasing")
    expected = theta_schedule_q32(p_target, num_anneal_steps)
    if tuple(values) != expected:
        raise ValueError("PA Q32 schedule differs from the frozen theta formula")
    return tuple(values)


@dataclass(frozen=True)
class Q0PaConfig:
    p_target: float
    num_particles: int
    num_anneal_steps: int
    rejuvenation_sweeps: int
    logical_kernel: str
    schedule_q32: tuple[int, ...]
    schedule_sha256: str
    resample_ess_fraction: float = PA_RESAMPLE_ESS_FRACTION
    schedule_version: str = PA_SCHEDULE_VERSION

    def __post_init__(self):
        integer_fields = {
            "num_particles": self.num_particles,
            "num_anneal_steps": self.num_anneal_steps,
            "rejuvenation_sweeps": self.rejuvenation_sweeps,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if not 0.0 < float(self.p_target) < 0.5:
            raise ValueError("p_target must lie in (0,0.5)")
        if int(self.num_particles) <= 0 or int(self.num_anneal_steps) <= 0:
            raise ValueError("particle and anneal counts must be positive")
        if int(self.rejuvenation_sweeps) < 0:
            raise ValueError("rejuvenation_sweeps must be nonnegative")
        if self.logical_kernel not in PA_LOGICAL_KERNELS:
            raise ValueError("unknown PA logical kernel")
        if float(self.resample_ess_fraction) != PA_RESAMPLE_ESS_FRACTION:
            raise ValueError("PA resampling threshold is frozen at 0.5")
        if self.schedule_version != PA_SCHEDULE_VERSION:
            raise ValueError("PA schedule version mismatch")
        schedule = validate_pa_schedule_q32(
            self.schedule_q32, self.p_target, self.num_anneal_steps,
        )
        digest = ladder_x_q32_sha256(schedule)
        if str(self.schedule_sha256) != digest:
            raise ValueError("PA Q32 schedule SHA256 mismatch")
        object.__setattr__(self, "schedule_q32", schedule)

    def as_dict(self):
        return {
            "p_target": float(self.p_target),
            "num_particles": int(self.num_particles),
            "num_anneal_steps": int(self.num_anneal_steps),
            "rejuvenation_sweeps": int(self.rejuvenation_sweeps),
            "logical_kernel": self.logical_kernel,
            "resample_ess_fraction": float(self.resample_ess_fraction),
            "schedule_version": self.schedule_version,
            "schedule_q32": list(self.schedule_q32),
            "schedule_sha256": self.schedule_sha256,
        }


@dataclass(frozen=True)
class PaSeedIdentity:
    source_commit: str
    config_sha256: str
    cell_fingerprint: str
    population_index: int
    trajectory_namespace: str

    def __post_init__(self):
        if len(self.source_commit) != 40 or any(c not in "0123456789abcdef" for c in self.source_commit):
            raise ValueError("PA source commit must be a full lowercase Git SHA")
        for name, value in (
                ("config_sha256", self.config_sha256),
                ("cell_fingerprint", self.cell_fingerprint)):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"PA {name} must be a lowercase SHA256")
        if isinstance(self.population_index, bool) or int(self.population_index) < 0:
            raise ValueError("PA population index is invalid")
        if not self.trajectory_namespace:
            raise ValueError("PA trajectory namespace is empty")

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "population_index": int(self.population_index),
            "trajectory_namespace": self.trajectory_namespace,
        }

    def seed(self, role, stage, sweep, output_slot):
        """Derive the mandatory clone-safe PortablePrng substream."""
        return derive_seed(
            "q0_pa_discovery_v1_substream",
            self.source_commit,
            self.config_sha256,
            self.cell_fingerprint,
            int(self.population_index),
            self.trajectory_namespace,
            str(role),
            int(stage),
            int(sweep),
            int(output_slot),
        )

    @property
    def population_seed(self):
        return self.seed("population", 0, 0, 0)


def pa_coupling_schedule(config):
    fractions = np.asarray(config.schedule_q32, dtype=np.float64) / float(Q32_ONE)
    return np.ascontiguousarray(coupling(config.p_target) * fractions)


def validate_hard_coset_basis(model):
    """Return the affine dimension or fail closed on a non-bijective basis."""
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    stabilizers = np.ascontiguousarray(model.stabilizer_rows, dtype=np.uint8)
    rank_stabilizers = int(gf2_rank(stabilizers))
    if rank_stabilizers != stabilizers.shape[0]:
        raise PaConflictError("stabilizer rows are not independent")
    rank_check = int(gf2_rank(model.H_check))
    affine_dimension = model.num_qubits - rank_check
    if rank_stabilizers + int(model.k) != affine_dimension:
        raise PaConflictError(
            "rank(H_X)+k does not equal n-rank(H_Z) for the hard coset"
        )
    combined = np.vstack((stabilizers, model.logical_move_basis))
    if int(gf2_rank(combined)) != affine_dimension:
        raise PaConflictError("stabilizer and logical coordinates are not a coset basis")
    return affine_dimension


def _supports_to_csr(rows):
    supports = [np.flatnonzero(row).astype(np.int64) for row in rows]
    offsets = np.zeros(len(supports) + 1, dtype=np.int64)
    for index, support in enumerate(supports):
        offsets[index + 1] = offsets[index] + support.size
    indices = np.empty(int(offsets[-1]), dtype=np.int64)
    for index, support in enumerate(supports):
        indices[offsets[index]:offsets[index + 1]] = support
    return supports, np.ascontiguousarray(indices), offsets


def _initial_population(model, frame, syndrome, config, seed_identity):
    from exp101_certified_src.prng import PortablePrng

    particles = int(config.num_particles)
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], particles, axis=0).astype(np.uint8, copy=False)
    labels = np.empty(particles, dtype=np.uint64)
    for slot in range(particles):
        rng = PortablePrng(seed_identity.seed("initialize", 0, 0, slot))
        for row in model.stabilizer_rows:
            if rng.randbelow(2):
                states[slot] ^= row
        for bit, row in enumerate(model.logical_move_basis):
            if rng.randbelow(2):
                states[slot] ^= row
        labels[slot] = bits_to_uint64(frame.label_of(states[slot]))
    return np.ascontiguousarray(states), labels


def _probability_tables(K, num_qubits):
    deltas = np.arange(-num_qubits, num_qubits + 1, dtype=np.int64)
    heatbath = np.empty((K.size, deltas.size), dtype=np.float64)
    for stage, value in enumerate(K):
        x = value * deltas.astype(np.float64)
        positive = x >= 0.0
        exp_negative = np.exp(-np.abs(x))
        heatbath[stage, positive] = exp_negative[positive] / (1.0 + exp_negative[positive])
        heatbath[stage, ~positive] = 1.0 / (1.0 + exp_negative[~positive])
    energy_delta = np.arange(num_qubits + 1, dtype=np.float64)
    boltzmann_delta = np.exp(-K[:, None] * energy_delta[None, :])
    return np.ascontiguousarray(heatbath), np.ascontiguousarray(boltzmann_delta)


def systematic_resampling(weights, offset):
    """Return canonical systematic parents for an offset in ``[0,1/N)``."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("systematic resampling requires finite one-dimensional weights")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("systematic resampling weights must be normalized")
    particles = weights.size
    offset = float(offset)
    if not 0.0 <= offset < 1.0 / particles:
        raise ValueError("systematic resampling offset is outside [0,1/N)")
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = offset + np.arange(particles, dtype=np.float64) / particles
    parents = np.searchsorted(cumulative, positions, side="right")
    return np.minimum(parents, particles - 1).astype(np.int64)


def _mutation_rng_states(seed_identity, stage, sweeps, particles):
    from exp101_certified_src.prng import PortablePrng

    states = np.empty((sweeps, particles, 2), dtype=np.uint64)
    for sweep in range(sweeps):
        for slot in range(particles):
            seed = seed_identity.seed("mutation", stage, sweep, slot)
            states[sweep, slot] = PortablePrng(seed).state_array()
    return states


def _toggle_reference(state, support):
    delta = int(support.size) - 2 * int(state[support].sum())
    state[support] ^= 1
    return delta


def _mutate_stage_reference(states, energies, labels, stage, config, seed_identity,
                            stabilizer_supports, logical_supports,
                            heatbath, boltzmann_delta):
    from exp101_certified_src.prng import PortablePrng

    counters = np.zeros(4, dtype=np.int64)
    logical_bit_flips = np.zeros(len(logical_supports), dtype=np.int64)
    for sweep in range(config.rejuvenation_sweeps):
        for slot in range(config.num_particles):
            rng = PortablePrng(seed_identity.seed("mutation", stage, sweep, slot))
            state = states[slot]
            for coordinate in rng.permutation(len(stabilizer_supports)):
                support = stabilizer_supports[int(coordinate)]
                delta = int(support.size) - 2 * int(state[support].sum())
                counters[0] += 1
                if rng.random() < heatbath[stage, delta + state.size]:
                    state[support] ^= 1
                    energies[slot] += delta
                    counters[1] += 1

            logical_order = rng.permutation(len(logical_supports))
            if config.logical_kernel == "coordinate":
                for bit_value in logical_order:
                    bit = int(bit_value)
                    support = logical_supports[bit]
                    delta = int(support.size) - 2 * int(state[support].sum())
                    counters[2] += 1
                    if rng.random() < heatbath[stage, delta + state.size]:
                        state[support] ^= 1
                        energies[slot] += delta
                        labels[slot] ^= np.uint64(1) << np.uint64(bit)
                        logical_bit_flips[bit] += 1
                        counters[3] += 1
            else:
                for block_start in range(0, len(logical_supports), 4):
                    block = logical_order[block_start:block_start + 4]
                    block_size = block.size
                    categories = 1 << block_size
                    candidate_energies = np.empty(categories, dtype=np.int64)
                    candidate_energies[0] = energies[slot]
                    previous_gray = 0
                    scratch_energy = int(energies[slot])
                    for enumeration in range(1, categories):
                        gray = enumeration ^ (enumeration >> 1)
                        changed = gray ^ previous_gray
                        changed_position = (changed & -changed).bit_length() - 1
                        bit = int(block[changed_position])
                        scratch_energy += _toggle_reference(state, logical_supports[bit])
                        candidate_energies[gray] = scratch_energy
                        previous_gray = gray
                    for position in range(block_size):
                        if (previous_gray >> position) & 1:
                            state[logical_supports[int(block[position])]] ^= 1
                    minimum = int(candidate_energies.min())
                    probabilities = boltzmann_delta[
                        stage, candidate_energies - minimum
                    ]
                    threshold = rng.random() * float(probabilities.sum())
                    cumulative = 0.0
                    selected = categories - 1
                    for category in range(categories):
                        cumulative += float(probabilities[category])
                        if threshold < cumulative:
                            selected = category
                            break
                    counters[2] += 1
                    if selected:
                        counters[3] += 1
                        for position in range(block_size):
                            if (selected >> position) & 1:
                                bit = int(block[position])
                                state[logical_supports[bit]] ^= 1
                                labels[slot] ^= np.uint64(1) << np.uint64(bit)
                                logical_bit_flips[bit] += 1
                    energies[slot] = candidate_energies[selected]
    return counters, logical_bit_flips


if njit is not None:
    @njit(cache=True, inline="always")
    def _nb_next_uint64(state):
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


    @njit(cache=True, inline="always")
    def _nb_support_delta(state, indices, start, stop):
        ones = 0
        for position in range(start, stop):
            ones += int(state[indices[position]])
        return (stop - start) - 2 * ones


    @njit(cache=True, inline="always")
    def _nb_toggle_support(state, indices, start, stop):
        for position in range(start, stop):
            state[indices[position]] ^= np.uint8(1)


    @njit(cache=True)
    def _mutate_stage_numba_core(
        states,
        energies,
        labels,
        stage,
        kernel_code,
        stabilizer_indices,
        stabilizer_offsets,
        logical_indices,
        logical_offsets,
        heatbath,
        boltzmann_delta,
        rng_states,
    ):
        particles = states.shape[0]
        num_qubits = states.shape[1]
        num_stabilizers = stabilizer_offsets.size - 1
        num_logicals = logical_offsets.size - 1
        counters = np.zeros(4, dtype=np.int64)
        logical_bit_flips = np.zeros(num_logicals, dtype=np.int64)
        stabilizer_order = np.empty(num_stabilizers, dtype=np.int64)
        logical_order = np.empty(num_logicals, dtype=np.int64)
        candidate_energies = np.empty(16, dtype=np.int64)

        for sweep in range(rng_states.shape[0]):
            for slot in range(particles):
                rng_state = rng_states[sweep, slot]
                state = states[slot]
                _nb_fill_permutation(rng_state, stabilizer_order)
                for order_index in range(num_stabilizers):
                    coordinate = stabilizer_order[order_index]
                    start = stabilizer_offsets[coordinate]
                    stop = stabilizer_offsets[coordinate + 1]
                    delta = _nb_support_delta(state, stabilizer_indices, start, stop)
                    counters[0] += 1
                    if _nb_random(rng_state) < heatbath[stage, delta + num_qubits]:
                        _nb_toggle_support(state, stabilizer_indices, start, stop)
                        energies[slot] += delta
                        counters[1] += 1

                _nb_fill_permutation(rng_state, logical_order)
                if kernel_code == 0:
                    for order_index in range(num_logicals):
                        bit = logical_order[order_index]
                        start = logical_offsets[bit]
                        stop = logical_offsets[bit + 1]
                        delta = _nb_support_delta(state, logical_indices, start, stop)
                        counters[2] += 1
                        if _nb_random(rng_state) < heatbath[stage, delta + num_qubits]:
                            _nb_toggle_support(state, logical_indices, start, stop)
                            energies[slot] += delta
                            labels[slot] ^= np.uint64(1) << np.uint64(bit)
                            logical_bit_flips[bit] += 1
                            counters[3] += 1
                else:
                    for block_start in range(0, num_logicals, 4):
                        block_size = min(4, num_logicals - block_start)
                        categories = 1 << block_size
                        candidate_energies[0] = energies[slot]
                        previous_gray = 0
                        scratch_energy = energies[slot]
                        for enumeration in range(1, categories):
                            gray = enumeration ^ (enumeration >> 1)
                            changed = gray ^ previous_gray
                            changed_position = 0
                            while ((changed >> changed_position) & 1) == 0:
                                changed_position += 1
                            bit = logical_order[block_start + changed_position]
                            start = logical_offsets[bit]
                            stop = logical_offsets[bit + 1]
                            delta = _nb_support_delta(state, logical_indices, start, stop)
                            _nb_toggle_support(state, logical_indices, start, stop)
                            scratch_energy += delta
                            candidate_energies[gray] = scratch_energy
                            previous_gray = gray
                        for position in range(block_size):
                            if ((previous_gray >> position) & 1) != 0:
                                bit = logical_order[block_start + position]
                                _nb_toggle_support(
                                    state, logical_indices,
                                    logical_offsets[bit], logical_offsets[bit + 1],
                                )
                        minimum = candidate_energies[0]
                        for category in range(1, categories):
                            if candidate_energies[category] < minimum:
                                minimum = candidate_energies[category]
                        total = 0.0
                        for category in range(categories):
                            total += boltzmann_delta[
                                stage, candidate_energies[category] - minimum
                            ]
                        threshold = _nb_random(rng_state) * total
                        cumulative = 0.0
                        selected = categories - 1
                        for category in range(categories):
                            cumulative += boltzmann_delta[
                                stage, candidate_energies[category] - minimum
                            ]
                            if threshold < cumulative:
                                selected = category
                                break
                        counters[2] += 1
                        if selected != 0:
                            counters[3] += 1
                            for position in range(block_size):
                                if ((selected >> position) & 1) != 0:
                                    bit = logical_order[block_start + position]
                                    _nb_toggle_support(
                                        state, logical_indices,
                                        logical_offsets[bit], logical_offsets[bit + 1],
                                    )
                                    labels[slot] ^= np.uint64(1) << np.uint64(bit)
                                    logical_bit_flips[bit] += 1
                        energies[slot] = candidate_energies[selected]
        return counters, logical_bit_flips
else:  # pragma: no cover
    _mutate_stage_numba_core = None


def _family_statistics(root_ancestry, weights, num_particles):
    masses = np.bincount(
        np.asarray(root_ancestry, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        minlength=int(num_particles),
    ).astype(np.float64)
    positive = masses > 0.0
    return {
        "family_masses": masses,
        "family_ess": float(1.0 / np.sum(masses * masses)),
        "distinct_initial_families": int(np.count_nonzero(positive)),
        "max_family_mass": float(masses.max(initial=0.0)),
    }


def run_q0_pa_population(model, frame, syndrome, config, seed_identity,
                         engine="reference", *, initial_states=None,
                         resampling_enabled=True):
    """Run one independently seeded population and return its full transcript."""
    load_exp101()
    from exp101_certified_src.gf2 import gf2_matmul
    from exp101_certified_src.prng import PortablePrng

    if engine not in {"reference", "numba"}:
        raise ValueError("PA engine must be reference or numba")
    if engine == "numba" and _mutate_stage_numba_core is None:
        raise RuntimeError("Numba PA engine requested but Numba is unavailable")
    if not isinstance(config, Q0PaConfig) or not isinstance(seed_identity, PaSeedIdentity):
        raise TypeError("PA config and seed identity must use their canonical dataclasses")
    if model.k > 64:
        raise PaConflictError("PA uint64 labels do not support k>64")
    affine_dimension = validate_hard_coset_basis(model)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    if syndrome.shape != (model.num_checks,):
        raise ValueError("PA syndrome shape mismatch")

    if initial_states is None:
        states, labels = _initial_population(model, frame, syndrome, config, seed_identity)
    else:
        states = np.ascontiguousarray(initial_states, dtype=np.uint8)
        if states.shape != (config.num_particles, model.num_qubits):
            raise ValueError("oracle initial_states shape mismatch")
        residual = gf2_matmul(model.H_check, states.T).T ^ syndrome[None, :]
        if residual.any():
            raise ValueError("oracle initial_states are outside the requested hard coset")
        labels = np.asarray(
            [bits_to_uint64(frame.label_of(state)) for state in states],
            dtype=np.uint64,
        )
    energies = states.sum(axis=1, dtype=np.int64)
    particles = int(config.num_particles)
    steps = int(config.num_anneal_steps)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    roots = np.arange(particles, dtype=np.int64)
    K = pa_coupling_schedule(config)
    heatbath, boltzmann_delta = _probability_tables(K, model.num_qubits)
    stabilizer_supports, stabilizer_indices, stabilizer_offsets = _supports_to_csr(
        model.stabilizer_rows
    )
    logical_supports, logical_indices, logical_offsets = _supports_to_csr(
        model.logical_move_basis
    )

    stage_energies = np.empty((steps + 1, particles), dtype=np.int64)
    stage_energies[0] = energies
    stage_pre_weights = np.empty((steps, particles), dtype=np.float64)
    stage_post_weights = np.empty_like(stage_pre_weights)
    conditional_ess = np.empty(steps, dtype=np.float64)
    ess_before_decision = np.empty(steps, dtype=np.float64)
    ess_after_decision = np.empty(steps, dtype=np.float64)
    max_pre_weight = np.empty(steps, dtype=np.float64)
    resampled = np.zeros(steps, dtype=np.bool_)
    resampling_offsets = np.full(steps, -1.0, dtype=np.float64)
    parents = np.empty((steps, particles), dtype=np.int64)
    offspring_counts = np.empty_like(parents)
    root_history = np.empty((steps + 1, particles), dtype=np.int64)
    root_history[0] = roots
    mutation_counters = np.zeros((steps, 4), dtype=np.int64)
    logical_bit_flips = np.zeros((steps, model.k), dtype=np.int64)
    log_normalizer_increments = np.empty(steps, dtype=np.float64)
    cumulative_log_z = np.empty(steps + 1, dtype=np.float64)
    cumulative_log_z[0] = affine_dimension * math.log(2.0)

    for stage in range(1, steps + 1):
        delta_K = float(K[stage] - K[stage - 1])
        log_incremental = -delta_K * energies.astype(np.float64)
        maximum = float(log_incremental.max())
        incremental = np.exp(log_incremental - maximum)
        mean_factor = float(np.dot(weights, incremental))
        second_factor = float(np.dot(weights, incremental * incremental))
        if not np.isfinite(mean_factor) or mean_factor <= 0.0 or second_factor <= 0.0:
            raise PaConflictError("non-finite PA incremental normalization")
        conditional_ess[stage - 1] = particles * mean_factor * mean_factor / second_factor
        log_normalizer_increments[stage - 1] = maximum + math.log(mean_factor)
        cumulative_log_z[stage] = (
            cumulative_log_z[stage - 1] + log_normalizer_increments[stage - 1]
        )
        unnormalized = weights * incremental
        normalization = float(unnormalized.sum())
        pre_weights = unnormalized / normalization
        stage_pre_weights[stage - 1] = pre_weights
        ess = float(1.0 / np.dot(pre_weights, pre_weights))
        ess_before_decision[stage - 1] = ess
        max_pre_weight[stage - 1] = float(pre_weights.max())

        if resampling_enabled and ess < config.resample_ess_fraction * particles:
            rng = PortablePrng(seed_identity.seed("resample", stage, 0, 0))
            offset = rng.random() / particles
            selected = systematic_resampling(pre_weights, offset)
            states = np.ascontiguousarray(states[selected])
            energies = np.ascontiguousarray(energies[selected])
            labels = np.ascontiguousarray(labels[selected])
            roots = np.ascontiguousarray(roots[selected])
            weights = np.full(particles, 1.0 / particles, dtype=np.float64)
            resampled[stage - 1] = True
            resampling_offsets[stage - 1] = offset
            parents[stage - 1] = selected
        else:
            weights = pre_weights.copy()
            parents[stage - 1] = np.arange(particles, dtype=np.int64)
        offspring_counts[stage - 1] = np.bincount(
            parents[stage - 1], minlength=particles,
        )
        ess_after_decision[stage - 1] = float(1.0 / np.dot(weights, weights))

        if config.rejuvenation_sweeps:
            if engine == "reference":
                stage_counters, stage_flips = _mutate_stage_reference(
                    states, energies, labels, stage, config, seed_identity,
                    stabilizer_supports, logical_supports, heatbath, boltzmann_delta,
                )
            else:
                rng_states = _mutation_rng_states(
                    seed_identity, stage, config.rejuvenation_sweeps, particles,
                )
                stage_counters, stage_flips = _mutate_stage_numba_core(
                    states, energies, labels, stage,
                    0 if config.logical_kernel == "coordinate" else 1,
                    stabilizer_indices, stabilizer_offsets,
                    logical_indices, logical_offsets,
                    heatbath, boltzmann_delta, rng_states,
                )
            mutation_counters[stage - 1] = stage_counters
            logical_bit_flips[stage - 1] = stage_flips
        stage_post_weights[stage - 1] = weights
        stage_energies[stage] = energies
        root_history[stage] = roots

    residual = gf2_matmul(model.H_check, states.T).T ^ syndrome[None, :]
    max_residual = int(residual.sum(axis=1).max(initial=0))
    recomputed_energies = states.sum(axis=1, dtype=np.int64)
    if not np.array_equal(energies, recomputed_energies):
        raise PaConflictError("PA mutation energy counters disagree with final states")
    recomputed_labels = np.asarray(
        [bits_to_uint64(frame.label_of(state)) for state in states], dtype=np.uint64,
    )
    if not np.array_equal(labels, recomputed_labels):
        raise PaConflictError("PA logical labels disagree with final states")
    family = _family_statistics(roots, weights, particles)
    return {
        "final_states": states,
        "final_weights": weights,
        "final_labels": labels,
        "final_energies": energies,
        "stage_energies": stage_energies,
        "stage_pre_weights": stage_pre_weights,
        "stage_post_weights": stage_post_weights,
        "conditional_ess": conditional_ess,
        "ess_before_decision": ess_before_decision,
        "ess_after_decision": ess_after_decision,
        "max_pre_weight": max_pre_weight,
        "resampled": resampled,
        "resampling_offsets": resampling_offsets,
        "parents": parents,
        "offspring_counts": offspring_counts,
        "root_ancestry": root_history,
        "mutation_counters": mutation_counters,
        "logical_bit_flips": logical_bit_flips,
        "log_normalizer_increments": log_normalizer_increments,
        "log_z": cumulative_log_z,
        "ladder_K": K,
        "ladder_p": 1.0 / (1.0 + np.exp(K)),
        "max_hard_coset_residual": max_residual,
        "affine_dimension": affine_dimension,
        "population_seed": seed_identity.population_seed,
        "engine": engine,
        **family,
    }


def weighted_label_distribution(labels, weights):
    labels = np.asarray(labels, dtype=np.uint64)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.ndim != 1 or weights.shape != labels.shape or labels.size == 0:
        raise ValueError("weighted label distribution shape mismatch")
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-13)):
        raise ValueError("weighted label distribution requires normalized finite weights")
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    unique, first = np.unique(sorted_labels, return_index=True)
    masses = np.add.reduceat(sorted_weights, first)
    masses /= masses.sum()
    return unique.astype(np.uint64), masses.astype(np.float64)


def label_distribution_collision(distribution_a, distribution_b):
    labels_a, masses_a = distribution_a
    labels_b, masses_b = distribution_b
    left = right = 0
    collision = 0.0
    while left < labels_a.size and right < labels_b.size:
        if labels_a[left] == labels_b[right]:
            collision += float(masses_a[left] * masses_b[right])
            left += 1
            right += 1
        elif labels_a[left] < labels_b[right]:
            left += 1
        else:
            right += 1
    return collision


def population_qtop_jackknife(distributions, k):
    """Cross-population collision U-statistic and delete-one-population MCSE."""
    distributions = list(distributions)
    count = len(distributions)
    if count < 3:
        raise ValueError("population collision estimator requires at least three populations")
    if not 0 <= int(k) <= 64:
        raise ValueError("invalid logical dimension")

    def estimate(indices):
        collisions = []
        for position, first in enumerate(indices):
            for second in indices[position + 1:]:
                collisions.append(label_distribution_collision(
                    distributions[first], distributions[second],
                ))
        collision = float(np.mean(collisions))
        uniform = 2.0 ** (-int(k))
        return collision, (collision - uniform) / (1.0 - uniform)

    collision, qtop = estimate(list(range(count)))
    delete_one = np.asarray([
        estimate([index for index in range(count) if index != omitted])[1]
        for omitted in range(count)
    ], dtype=np.float64)
    centre = float(delete_one.mean())
    variance = (count - 1.0) / count * float(np.sum((delete_one - centre) ** 2))
    return {
        "collision_mass": collision,
        "q_top": qtop,
        "q_top_mcse": math.sqrt(max(variance, 0.0)),
        "delete_one_q_top": delete_one,
        "pair_count": count * (count - 1) // 2,
    }


def pa_population_gate(result):
    """Recompute all per-population numerical gates without clipping."""
    failures = []
    if int(result["max_hard_coset_residual"]) != 0:
        failures.append("hard_coset_residual")
    if np.any(np.asarray(result["conditional_ess"]) / result["final_weights"].size < 0.70):
        failures.append("conditional_ess_fraction")
    if np.any(np.asarray(result["max_pre_weight"]) > 0.10):
        failures.append("max_normalized_particle_weight")
    threshold = PA_RESAMPLE_ESS_FRACTION * result["final_weights"].size
    if np.any(np.asarray(result["ess_after_decision"]) + 1e-12 < threshold):
        failures.append("post_decision_ess")
    if float(result["family_ess"]) < 4.0:
        failures.append("final_family_ess")
    if int(result["distinct_initial_families"]) < 8:
        failures.append("distinct_initial_families")
    if float(result["max_family_mass"]) > 0.50:
        failures.append("max_family_mass")
    return not failures, failures


def canonical_population_digest(result):
    """Cross-platform digest over all stochastic and transcript arrays."""
    digest = hashlib.sha256()
    fields = (
        "final_states", "final_weights", "final_labels", "final_energies",
        "stage_energies", "stage_pre_weights", "stage_post_weights",
        "conditional_ess", "ess_before_decision", "ess_after_decision",
        "max_pre_weight", "resampled", "resampling_offsets", "parents",
        "offspring_counts", "root_ancestry", "mutation_counters",
        "logical_bit_flips", "log_normalizer_increments", "log_z",
    )
    for field in fields:
        value = np.ascontiguousarray(result[field])
        digest.update(field.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        if value.dtype.byteorder in {"=", "<"} and value.dtype.itemsize > 1:
            value = value.astype(value.dtype.newbyteorder(">"), copy=False)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def config_from_record(record, p_target, num_particles=None):
    """Build a validated dataclass from one frozen method/config record."""
    particles = record["num_particles"] if num_particles is None else num_particles
    key = f"p={float(p_target):.2f},G={int(record['num_anneal_steps'])}"
    schedule = record["schedules"][key]
    return Q0PaConfig(
        p_target=float(p_target),
        num_particles=int(particles),
        num_anneal_steps=int(record["num_anneal_steps"]),
        rejuvenation_sweeps=int(record["rejuvenation_sweeps"]),
        logical_kernel=str(record["logical_kernel"]),
        schedule_q32=tuple(schedule["q32"]),
        schedule_sha256=str(schedule["sha256"]),
    )


def pa_config_fingerprint(config):
    if not isinstance(config, Q0PaConfig):
        raise TypeError("expected Q0PaConfig")
    return sha256_json({"contract": PA_CONTRACT_VERSION, **config.as_dict()})


def seed_identity_json(seed_identity):
    if not isinstance(seed_identity, PaSeedIdentity):
        raise TypeError("expected PaSeedIdentity")
    return canonical_json(seed_identity.as_dict())
