"""Physical-p replica exchange for the exact collapsed q=0 HGP marginal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_hgp_collapsed import (
    CollapsedConflictError,
    _basis_seen,
    _classical_row_neighbors,
    _hc_pack,
    _hc_popcount,
    _hc_random,
    _hc_sample_full_state,
    _hp_advance_transport,
    _hp_update_b_sweep,
    _initial_collapsed_masks,
    _pack_state,
    _qubit_signatures,
    _reference_advance_transport,
    _reference_power_b_sweep,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .q0_global import GlobalConflictError, validate_observable_frame
from .seeds import derive_seed

try:  # pragma: no cover - exercised by reference-only environments
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


PHYSICAL_PT_VERSION = "exp102.q0_hgp_collapsed_physical_pt.v0"
PHYSICAL_PT_RAW_VERSION = "exp102.q0_hgp_collapsed_physical_pt.raw.v0"
PHYSICAL_PT_METHODS = ("CPPT32", "CPPT64")
PHYSICAL_PT_SEED_ROOT = "exp102_q0_hgp_collapsed_physical_pt_v0"


@dataclass(frozen=True)
class CollapsedPhysicalPtConfig:
    method_id: str
    p_cold: float
    burn_rounds: int
    measurement_rounds: int
    block_size: int = 8
    beta_exponent: int = 2

    def __post_init__(self):
        if self.method_id not in PHYSICAL_PT_METHODS:
            raise ValueError("unknown collapsed physical-p PT method")
        if not 0.0 < float(self.p_cold) < 0.5:
            raise ValueError("collapsed physical-p PT cold p must lie in (0,.5)")
        if (
            isinstance(self.burn_rounds, bool)
            or isinstance(self.measurement_rounds, bool)
            or int(self.burn_rounds) <= 0
            or int(self.measurement_rounds) <= 0
        ):
            raise ValueError("collapsed physical-p PT clocks must be positive")
        if int(self.measurement_rounds) % 8:
            raise ValueError("measurement rounds must divide into eight blocks")
        if int(self.block_size) != 8 or int(self.beta_exponent) != 2:
            raise ValueError("collapsed physical-p PT freezes block 8 and beta exponent 2")

    @property
    def num_replicas(self):
        return int(self.method_id[4:])

    @property
    def beta_values(self):
        denominator = (self.num_replicas - 1) ** self.beta_exponent
        values = np.asarray([
            (index ** self.beta_exponent) / denominator
            for index in range(self.num_replicas)
        ], dtype=np.float64)
        values[0], values[-1] = 0.0, 1.0
        return values

    @property
    def p_values(self):
        coupling = math.log((1.0 - float(self.p_cold)) / float(self.p_cold))
        values = 1.0 / (1.0 + np.exp(self.beta_values * coupling))
        values[0], values[-1] = 0.5, float(self.p_cold)
        return values

    def as_dict(self):
        return {
            "beta_exponent": int(self.beta_exponent),
            "beta_values": self.beta_values.tolist(),
            "block_size": int(self.block_size),
            "burn_rounds": int(self.burn_rounds),
            "measurement_rounds": int(self.measurement_rounds),
            "method_id": self.method_id,
            "num_replicas": self.num_replicas,
            "p_cold": float(self.p_cold),
            "p_hot": 0.5,
            "p_values": self.p_values.tolist(),
            "tempered_terms": "B_prior_and_A_coset_mass",
        }


@dataclass(frozen=True)
class PhysicalPtSeedIdentity:
    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    resource_tier: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str

    def __post_init__(self):
        if (
            len(self.source_commit) != 40
            or any(c not in "0123456789abcdef" for c in self.source_commit)
        ):
            raise ValueError("physical-p PT source commit must be a full lowercase Git SHA")
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            value = getattr(self, name)
            if (
                len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"physical-p PT {name} must be a lowercase SHA256")
        if self.method_id not in PHYSICAL_PT_METHODS:
            raise ValueError("invalid physical-p PT method identity")
        if not self.resource_tier or not self.init_family or not self.trajectory_namespace:
            raise ValueError("physical-p PT seed identity string is empty")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("physical-p PT trajectory index is invalid")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            PHYSICAL_PT_SEED_ROOT, self.source_commit, self.config_sha256,
            self.registry_sha256, self.cell_fingerprint, self.method_id,
            self.resource_tier, self.init_family, int(self.trajectory_index),
            self.trajectory_namespace, str(stage), str(role), int(index),
        )

    def as_dict(self):
        return {
            "cell_fingerprint": self.cell_fingerprint,
            "config_sha256": self.config_sha256,
            "init_family": self.init_family,
            "method_id": self.method_id,
            "registry_sha256": self.registry_sha256,
            "resource_tier": self.resource_tier,
            "source_commit": self.source_commit,
            "trajectory_index": int(self.trajectory_index),
            "trajectory_namespace": self.trajectory_namespace,
        }


def physical_pt_resource_requirements(H, config):
    """Return deterministic table sizes before allocating a physical-p ladder."""
    H = np.asarray(H, dtype=np.uint8)
    if H.ndim != 2:
        raise ValueError("physical-p PT H must be a matrix")
    rows = int(H.shape[0])
    entries = int(config.num_replicas) * (1 << rows)
    table_bytes = entries * np.dtype(np.float64).itemsize
    return {
        "log_mass_table_bytes": table_bytes,
        "mass_table_bytes": table_bytes,
        "num_mass_entries": entries,
        "num_replicas": int(config.num_replicas),
        "rank_rows": rows,
    }


def _sha256_float64_be(values, chunk_entries=1 << 20):
    """Hash float64 values in canonical big-endian chunks without a full copy."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.size, int(chunk_entries)):
        stop = min(flat.size, start + int(chunk_entries))
        digest.update(np.asarray(flat[start:stop], dtype=">f8").tobytes())
    return digest.hexdigest()


def _matrix_sha256(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(H.shape, dtype=">u8").tobytes())
    digest.update(H.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalPtMassArtifact:
    h_sha256: str
    p_values_sha256: str
    log_mass_tables: np.ndarray
    log_mass_tables_sha256: str = ""

    def __post_init__(self):
        tables = np.asarray(self.log_mass_tables, dtype=np.float64)
        if tables.ndim != 2 or not tables.flags.c_contiguous:
            raise ValueError("physical-p PT log-mass tables must be a contiguous matrix")
        if not np.all(np.isfinite(tables)):
            raise ValueError("physical-p PT log-mass tables are non-finite")
        for name in ("h_sha256", "p_values_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"physical-p PT artifact {name} is invalid")
        actual = _sha256_float64_be(tables)
        if self.log_mass_tables_sha256 and self.log_mass_tables_sha256 != actual:
            raise CollapsedConflictError("physical-p PT log-mass artifact SHA changed")
        object.__setattr__(self, "log_mass_tables", tables)
        object.__setattr__(self, "log_mass_tables_sha256", actual)
        tables.setflags(write=False)

    def validate_binding(self, H, p_values):
        p_values = np.ascontiguousarray(p_values, dtype=np.float64)
        expected_shape = (p_values.size, 1 << int(np.asarray(H).shape[0]))
        if (
            self.h_sha256 != _matrix_sha256(H)
            or self.p_values_sha256 != _sha256_float64_be(p_values)
            or self.log_mass_tables.shape != expected_shape
        ):
            raise CollapsedConflictError("physical-p PT log-mass artifact binding changed")
        return True

    def as_dict(self):
        return {
            "h_sha256": self.h_sha256,
            "log_mass_tables_sha256": self.log_mass_tables_sha256,
            "p_values_sha256": self.p_values_sha256,
            "shape": [int(value) for value in self.log_mass_tables.shape],
        }


if njit is not None:

    @njit(cache=True, inline="always")
    def _cppt_log_target(b_columns, a_syndromes, log_mass, log_odds):
        value = 0.0
        for column in range(b_columns.size):
            value += _hc_popcount(b_columns[column]) * log_odds
        for factor in range(a_syndromes.size):
            value += log_mass[int(a_syndromes[factor])]
        return value


    @njit(cache=True)
    def _run_physical_pt_core(
            initial_state, initial_b, initial_syndromes,
            burn_rng_states, measurement_rng_states,
            burn_observation_rng, measurement_observation_rng,
            p_values, burn_rounds, measurement_rounds, block_size, n, r,
            section_masks, kernel_combinations, neighbors, neighbor_counts,
            log_masses, log_odds, cold_odds_powers, qubit_signatures):
        replicas = p_values.size
        b_states = np.empty((replicas, r), dtype=np.uint32)
        syndrome_states = np.empty((replicas, n), dtype=np.uint32)
        origins = np.arange(replicas, dtype=np.int32)
        for rung in range(replicas):
            b_states[rung] = initial_b
            syndrome_states[rung] = initial_syndromes
        total_rounds = burn_rounds + measurement_rounds
        labels = np.empty(measurement_rounds, dtype=np.uint64)
        weights = np.empty(measurement_rounds, dtype=np.int32)
        packed = np.empty(
            (measurement_rounds, (initial_state.size + 7) // 8),
            dtype=np.uint8,
        )
        cold_b_columns = np.empty((measurement_rounds, r), dtype=np.uint32)
        burn_labels = np.empty(burn_rounds, dtype=np.uint64)
        burn_cold_weights = np.empty(burn_rounds, dtype=np.int32)
        burn_cold_b_weights = np.empty(burn_rounds, dtype=np.int32)
        measurement_cold_b_weights = np.empty(measurement_rounds, dtype=np.int32)
        local_attempts = np.zeros(replicas, dtype=np.int64)
        local_changes = np.zeros(replicas, dtype=np.int64)
        swap_attempts = np.zeros(replicas - 1, dtype=np.int64)
        swap_accepts = np.zeros(replicas - 1, dtype=np.int64)
        hot_visits = np.zeros(replicas, dtype=np.int64)
        cold_visits = np.zeros(replicas, dtype=np.int64)
        transport_phase = np.zeros(replicas, dtype=np.uint8)
        transport_phase[origins[-1]] = np.uint8(1)
        round_trips = np.zeros(replicas, dtype=np.int64)
        column_order = np.empty(r, dtype=np.int32)
        row_order = np.empty(r, dtype=np.int32)
        categories = 1 << min(block_size, r)
        candidate_masks = np.empty(categories, dtype=np.uint32)
        candidate_counts = np.empty(categories, dtype=np.uint8)
        candidate_weights = np.empty(
            max(categories, kernel_combinations.size), dtype=np.float64,
        )
        state = initial_state.copy()
        previous_state = initial_state.copy()
        burn_endpoint = initial_state.copy()
        attempts_per_sweep = r * ((r + block_size - 1) // block_size)
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
                    b_states[rung], syndrome_states[rung],
                    stage_rng_states[rung], block_size, 1.0, neighbors,
                    neighbor_counts, log_masses[rung], log_odds[rung],
                    column_order, row_order, candidate_masks,
                    candidate_counts, candidate_weights,
                )
                if changed < 0:
                    return (state, burn_endpoint, packed, burn_labels, labels,
                            weights, cold_b_columns, burn_cold_weights,
                            burn_cold_b_weights, measurement_cold_b_weights,
                            local_attempts, local_changes, swap_attempts,
                            swap_accepts, hot_visits, cold_visits, round_trips,
                            origins, False)
                local_attempts[rung] += attempts_per_sweep
                local_changes[rung] += changed
            parity = round_index & 1
            for lower in range(parity, replicas - 1, 2):
                upper = lower + 1
                self_score = _cppt_log_target(
                    b_states[lower], syndrome_states[lower],
                    log_masses[lower], log_odds[lower],
                ) + _cppt_log_target(
                    b_states[upper], syndrome_states[upper],
                    log_masses[upper], log_odds[upper],
                )
                cross_score = _cppt_log_target(
                    b_states[upper], syndrome_states[upper],
                    log_masses[lower], log_odds[lower],
                ) + _cppt_log_target(
                    b_states[lower], syndrome_states[lower],
                    log_masses[upper], log_odds[upper],
                )
                delta = cross_score - self_score
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
                    temporary_origin = origins[lower]
                    origins[lower] = origins[upper]
                    origins[upper] = temporary_origin
            hot_origin = origins[0]
            cold_origin = origins[-1]
            hot_visits[hot_origin] += 1
            cold_visits[cold_origin] += 1
            _hp_advance_transport(
                transport_phase, round_trips, hot_origin, cold_origin,
            )
            previous_state[:] = state
            label, weight = _hc_sample_full_state(
                state, previous_state, b_states[-1], syndrome_states[-1],
                stage_observation_rng, n, r, section_masks,
                kernel_combinations, cold_odds_powers, qubit_signatures,
                candidate_weights, np.zeros(5, dtype=np.int64),
            )
            b_weight = 0
            for column in range(r):
                b_weight += _hc_popcount(b_states[-1, column])
            if round_index < burn_rounds:
                burn_labels[round_index] = label
                burn_cold_weights[round_index] = weight
                burn_cold_b_weights[round_index] = b_weight
                if round_index + 1 == burn_rounds:
                    burn_endpoint[:] = state
            else:
                measurement = round_index - burn_rounds
                labels[measurement] = label
                weights[measurement] = weight
                cold_b_columns[measurement] = b_states[-1]
                measurement_cold_b_weights[measurement] = b_weight
                _hc_pack(state, packed[measurement])
        return (state, burn_endpoint, packed, burn_labels, labels, weights,
                cold_b_columns, burn_cold_weights, burn_cold_b_weights,
                measurement_cold_b_weights, local_attempts, local_changes,
                swap_attempts, swap_accepts, hot_visits, cold_visits,
                round_trips, origins, True)
else:  # pragma: no cover
    _run_physical_pt_core = None


def _reference_log_target(b_columns, a_syndromes, log_mass, log_odds):
    return (
        sum(int(value).bit_count() for value in b_columns) * float(log_odds)
        + sum(float(log_mass[int(value)]) for value in a_syndromes)
    )


def _run_physical_pt_reference_core(
        initial_state, initial_b, initial_syndromes, burn_rngs,
        measurement_rngs, burn_observation_rng, measurement_observation_rng,
        p_values, burn_rounds, measurement_rounds, block_size, n, r,
        section_masks, kernel_combinations, neighbors, neighbor_counts,
        log_masses, log_odds, cold_odds_powers, qubit_signatures):
    replicas = p_values.size
    b_states = np.repeat(initial_b[None, :], replicas, axis=0)
    syndrome_states = np.repeat(initial_syndromes[None, :], replicas, axis=0)
    origins = np.arange(replicas, dtype=np.int32)
    total_rounds = burn_rounds + measurement_rounds
    labels = np.empty(measurement_rounds, dtype=np.uint64)
    weights = np.empty(measurement_rounds, dtype=np.int32)
    packed = np.empty(
        (measurement_rounds, (initial_state.size + 7) // 8), dtype=np.uint8,
    )
    cold_b_columns = np.empty((measurement_rounds, r), dtype=np.uint32)
    burn_labels = np.empty(burn_rounds, dtype=np.uint64)
    burn_cold_weights = np.empty(burn_rounds, dtype=np.int32)
    burn_cold_b_weights = np.empty(burn_rounds, dtype=np.int32)
    measurement_cold_b_weights = np.empty(measurement_rounds, dtype=np.int32)
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
    attempts_per_sweep = r * ((r + block_size - 1) // block_size)
    for round_index in range(total_rounds):
        stage_rngs = burn_rngs if round_index < burn_rounds else measurement_rngs
        observation_rng = (
            burn_observation_rng if round_index < burn_rounds
            else measurement_observation_rng
        )
        for rung in range(replicas):
            local_changes[rung] += _reference_power_b_sweep(
                b_states[rung], syndrome_states[rung], stage_rngs[rung],
                block_size, 1.0, neighbors, neighbor_counts,
                log_masses[rung], log_odds[rung],
            )
            local_attempts[rung] += attempts_per_sweep
        for lower in range(round_index & 1, replicas - 1, 2):
            upper = lower + 1
            self_score = _reference_log_target(
                b_states[lower], syndrome_states[lower], log_masses[lower],
                log_odds[lower],
            ) + _reference_log_target(
                b_states[upper], syndrome_states[upper], log_masses[upper],
                log_odds[upper],
            )
            cross_score = _reference_log_target(
                b_states[upper], syndrome_states[upper], log_masses[lower],
                log_odds[lower],
            ) + _reference_log_target(
                b_states[lower], syndrome_states[lower], log_masses[upper],
                log_odds[upper],
            )
            delta = cross_score - self_score
            swap_attempts[lower] += 1
            if delta >= 0.0 or stage_rngs[lower].random() < math.exp(delta):
                swap_accepts[lower] += 1
                b_states[[lower, upper]] = b_states[[upper, lower]]
                syndrome_states[[lower, upper]] = syndrome_states[[upper, lower]]
                origins[[lower, upper]] = origins[[upper, lower]]
        hot_origin, cold_origin = int(origins[0]), int(origins[-1])
        hot_visits[hot_origin] += 1
        cold_visits[cold_origin] += 1
        _reference_advance_transport(
            transport_phase, round_trips, hot_origin, cold_origin,
        )
        state, label, weight = _reference_sample_full_state(
            state, b_states[-1], syndrome_states[-1], observation_rng, n, r,
            section_masks, kernel_combinations, cold_odds_powers,
            qubit_signatures,
        )
        b_weight = sum(int(value).bit_count() for value in b_states[-1])
        if round_index < burn_rounds:
            burn_labels[round_index] = label
            burn_cold_weights[round_index] = weight
            burn_cold_b_weights[round_index] = b_weight
            if round_index + 1 == burn_rounds:
                burn_endpoint = state.copy()
        else:
            measurement = round_index - burn_rounds
            labels[measurement] = label
            weights[measurement] = weight
            cold_b_columns[measurement] = b_states[-1]
            measurement_cold_b_weights[measurement] = b_weight
            packed[measurement] = _pack_state(state)
    return (state, burn_endpoint, packed, burn_labels, labels, weights,
            cold_b_columns, burn_cold_weights, burn_cold_b_weights,
            measurement_cold_b_weights, local_attempts, local_changes,
            swap_attempts, swap_accepts, hot_visits, cold_visits,
            round_trips, origins, True)


def build_physical_pt_mass_artifact(H, p_values, engine):
    """Build each mass table once and retain only the read-only log tables."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    p_values = np.ascontiguousarray(p_values, dtype=np.float64)
    mass_engine = "numba" if engine == "numba" else "reference"
    tables = np.empty((p_values.size, 1 << H.shape[0]), dtype=np.float64)
    for index, p in enumerate(p_values):
        mass = build_classical_coset_mass(H, float(p), engine=mass_engine)
        np.log(mass, out=tables[index])
    return PhysicalPtMassArtifact(
        h_sha256=_matrix_sha256(H),
        p_values_sha256=_sha256_float64_be(p_values),
        log_mass_tables=tables,
    )


def run_collapsed_physical_pt_trajectory(
        model, frame, H, syndrome, config, seed_identity, initial_state, *,
        engine="numba", mass_artifact=None):
    """Run exact replica exchange along the physical-p collapsed marginals."""
    if engine not in ("reference", "numba"):
        raise ValueError("physical-p PT engine must be reference or numba")
    if engine == "numba" and _run_physical_pt_core is None:
        raise RuntimeError("Numba is required for accelerated physical-p PT")
    if config.method_id != seed_identity.method_id:
        raise CollapsedConflictError("physical-p PT config/seed method mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise CollapsedConflictError("physical-p PT observable frame mismatch") from exc
    H = np.ascontiguousarray(H, dtype=np.uint8)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,):
        raise ValueError("physical-p PT syndrome shape mismatch")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        state, syndrome, H,
    )
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    p_values = np.ascontiguousarray(config.p_values)
    if mass_artifact is None:
        mass_artifact = build_physical_pt_mass_artifact(H, p_values, engine)
    if not isinstance(mass_artifact, PhysicalPtMassArtifact):
        raise TypeError("physical-p PT requires a verified mass artifact")
    mass_artifact.validate_binding(H, p_values)
    log_masses = mass_artifact.log_mass_tables
    log_odds = np.ascontiguousarray(np.log(p_values / (1.0 - p_values)))
    cold_odds = float(config.p_cold) / (1.0 - float(config.p_cold))
    cold_odds_powers = np.ones(max(H.shape) + 1, dtype=np.float64)
    for value in range(1, cold_odds_powers.size):
        cold_odds_powers[value] = cold_odds_powers[value - 1] * cold_odds
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    burn_rngs = [
        PortablePrng(seed_identity.seed("burn", "replica", rung))
        for rung in range(config.num_replicas)
    ]
    measurement_rngs = [
        PortablePrng(seed_identity.seed("measurement", "replica", rung))
        for rung in range(config.num_replicas)
    ]
    burn_observation = PortablePrng(seed_identity.seed("burn", "observation"))
    measurement_observation = PortablePrng(
        seed_identity.seed("measurement", "observation")
    )
    if engine == "numba":
        result = _run_physical_pt_core(
            state, b_columns, a_syndromes,
            np.asarray([rng.state_array() for rng in burn_rngs], dtype=np.uint64),
            np.asarray([
                rng.state_array() for rng in measurement_rngs
            ], dtype=np.uint64),
            burn_observation.state_array(), measurement_observation.state_array(),
            p_values, int(config.burn_rounds), int(config.measurement_rounds),
            int(config.block_size), int(H.shape[1]), int(H.shape[0]),
            section_masks, kernel_combinations, neighbors, neighbor_counts,
            log_masses, log_odds, cold_odds_powers, _qubit_signatures(frame),
        )
    else:
        result = _run_physical_pt_reference_core(
            state, b_columns, a_syndromes, burn_rngs, measurement_rngs,
            burn_observation, measurement_observation, p_values,
            int(config.burn_rounds), int(config.measurement_rounds),
            int(config.block_size), int(H.shape[1]), int(H.shape[0]),
            section_masks, kernel_combinations, neighbors, neighbor_counts,
            log_masses, log_odds, cold_odds_powers, _qubit_signatures(frame),
        )
    if not result[-1]:
        raise CollapsedConflictError("physical-p PT categorical weights vanished")
    unpacked = np.unpackbits(
        result[2], axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    replay_labels = np.asarray([
        _state_label(frame, row) for row in unpacked
    ], dtype=np.uint64)
    if (
        residuals.any()
        or not np.array_equal(result[4], replay_labels)
        or not np.array_equal(result[5], unpacked.sum(axis=1))
    ):
        raise CollapsedConflictError("physical-p PT raw replay failed")
    ladder_sha = _sha256_float64_be(p_values)
    return {
        "beta_values": config.beta_values,
        "burn_cold_b_weights": result[8],
        "burn_cold_weights": result[7],
        "burn_label": _state_label(frame, result[1]),
        "burn_labels": result[3],
        "burn_state_packed": _pack_state(result[1]),
        "cold_b_columns": result[6],
        "engine": engine,
        "final_label": _state_label(frame, result[0]),
        "final_origins_by_rung": result[17],
        "final_state_packed": _pack_state(result[0]),
        "hot_visits_by_origin": result[14],
        "initial_label": _state_label(frame, state),
        "initial_state_packed": _pack_state(state),
        "local_attempts_by_rung": result[10],
        "local_changes_by_rung": result[11],
        "log_mass_tables_sha256": mass_artifact.log_mass_tables_sha256,
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_rounds // 8,
        ),
        "measurement_cold_b_weights": result[9],
        "measurement_labels": result[4],
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_states_packed": result[2],
        "measurement_weights": result[5],
        "method_id": config.method_id,
        "p_values": p_values,
        "p_values_sha256": ladder_sha,
        "raw_version": PHYSICAL_PT_RAW_VERSION,
        "round_trip_definition": "cold_hot_cold_after_established_cold_visit",
        "round_trips_by_origin": result[16],
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "swap_accepts": result[13],
        "swap_attempts": result[12],
        "version": PHYSICAL_PT_VERSION,
        "cold_visits_by_origin": result[15],
        "burn_basis_seen": _basis_seen(result[3], model.k),
    }
