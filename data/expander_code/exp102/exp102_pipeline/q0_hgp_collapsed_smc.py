"""Fixed-schedule SMC feasibility engine for the collapsed q=0 HGP marginal.

This is deliberately a population diagnostic, not an MCMC replacement.  The
collapsed B marginal has the exact bridge

    pi_lambda(B) proportional to prior_p(B) * L(B)**lambda,

where ``L(B)`` is the product of the classical A-column coset masses.  At
``lambda=0`` every B bit is iid Bernoulli(p), so the initializer is exact and
does not need a planted, uniform-hard-coset, or all-zero warm start.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    CollapsedConflictError,
    _bits_to_mask,
    _classical_row_neighbors,
    _hp_update_b_sweep,
    _reference_power_b_sweep,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .seeds import derive_seed


COLLAPSED_SMC_VERSION = "exp102.q0_hgp_collapsed_smc.v0"
COLLAPSED_SMC_RAW_VERSION = "exp102.q0_hgp_collapsed_smc.raw.v0"
COLLAPSED_SMC_KERNEL = "systematic_resample_exact_collapsed_block_heatbath.v1"
BASE_FAMILIES = ("column_major", "row_major")


class CollapsedSmcConflictError(ValueError):
    """A collapsed-SMC identity, algebra, or replay invariant failed."""


def _strict_sha256(value, name):
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _strict_commit(value):
    value = str(value)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_commit must be a lowercase full Git SHA")
    return value


def _float64_big_endian_sha256(values):
    return hashlib.sha256(
        np.asarray(values, dtype=">f8").tobytes(order="C")
    ).hexdigest()


@dataclass(frozen=True)
class CollapsedSmcConfig:
    """Frozen bridge and fixed resample-move kernel parameters."""

    p: float
    num_particles: int
    lambda_values: tuple[float, ...]
    mutation_sweeps: int = 1
    block_size: int = 8
    method_id: str = ""

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("collapsed SMC p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in ("num_particles", "mutation_sweeps", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.num_particles < 2:
            raise ValueError("collapsed SMC requires at least two particles")
        if not 1 <= self.mutation_sweeps <= 16:
            raise ValueError("collapsed SMC mutation sweeps must lie in [1, 16]")
        if self.block_size != 8:
            raise ValueError("collapsed SMC freezes eight-bit B blocks")
        values = tuple(float(value) for value in self.lambda_values)
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("collapsed SMC lambda schedule is invalid")
        if values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("collapsed SMC lambda schedule must start at zero and end at one")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("collapsed SMC lambda schedule must be strictly increasing")
        object.__setattr__(self, "lambda_values", values)
        expected = (
            f"CSMC{len(values):02d}-B{self.block_size}-S{self.mutation_sweeps}"
            f"-N{self.num_particles}"
        )
        if self.method_id and self.method_id != expected:
            raise ValueError("collapsed SMC method ID does not match the frozen parameters")
        object.__setattr__(self, "method_id", expected)

    @property
    def num_levels(self):
        return len(self.lambda_values)

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": self.p,
            "num_particles": self.num_particles,
            "lambda_values": list(self.lambda_values),
            "mutation_sweeps": self.mutation_sweeps,
            "block_size": self.block_size,
            "resampling": "systematic_every_nonzero_lambda_stage",
            "kernel": COLLAPSED_SMC_KERNEL,
            "prior_endpoint": "exact_iid_bernoulli_B",
        }


@dataclass(frozen=True)
class CollapsedSmcSeedIdentity:
    """Clone-safe seed namespace for one independent SMC population."""

    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    base_family: str
    population_index: int
    trajectory_namespace: str

    def __post_init__(self):
        object.__setattr__(self, "source_commit", _strict_commit(self.source_commit))
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            object.__setattr__(self, name, _strict_sha256(getattr(self, name), name))
        if not isinstance(self.method_id, str) or not self.method_id.startswith("CSMC"):
            raise ValueError("collapsed SMC method ID is invalid")
        if self.base_family not in BASE_FAMILIES:
            raise ValueError("collapsed SMC base family is invalid")
        if isinstance(self.population_index, bool) or int(self.population_index) < 0:
            raise ValueError("collapsed SMC population index is invalid")
        object.__setattr__(self, "population_index", int(self.population_index))
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("collapsed SMC trajectory namespace is empty")

    def seed(self, role, stage=0, sweep=0, output_slot=0):
        return derive_seed(
            COLLAPSED_SMC_VERSION,
            self.trajectory_namespace,
            self.source_commit,
            self.config_sha256,
            self.registry_sha256,
            self.cell_fingerprint,
            self.method_id,
            self.base_family,
            self.population_index,
            str(role),
            int(stage),
            int(sweep),
            int(output_slot),
        )

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": self.method_id,
            "base_family": self.base_family,
            "population_index": self.population_index,
            "trajectory_namespace": self.trajectory_namespace,
        }


def collapsed_lambda_schedule(num_levels):
    """Return the HP-compatible quadratic schedule used by the V0 pilot."""
    levels = int(num_levels)
    if levels < 2:
        raise ValueError("collapsed SMC requires at least two lambda levels")
    denominator = float((levels - 1) ** 2)
    values = tuple((index * index) / denominator for index in range(levels))
    return (0.0, *values[1:-1], 1.0)


def _y_columns(syndrome, H):
    r, n = H.shape
    matrix = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    return np.asarray([_bits_to_mask(matrix[:, column]) for column in range(n)], dtype=np.uint32)


def a_syndromes_from_b(syndrome, H, b_columns):
    """Return the A-column coset syndromes ``Y xor B H`` for every particle."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    if b_columns.ndim != 2 or b_columns.shape[1] != H.shape[0]:
        raise ValueError("collapsed SMC B-column array has the wrong shape")
    y_columns = _y_columns(syndrome, H)
    result = np.repeat(y_columns[None, :], b_columns.shape[0], axis=0)
    for column in range(H.shape[1]):
        for row in np.flatnonzero(H[:, column]):
            result[:, column] ^= b_columns[:, int(row)]
    return np.ascontiguousarray(result, dtype=np.uint32)


def collapsed_log_likelihood(a_syndromes, log_mass):
    """Evaluate ``log L(B)`` for each particle from its A coset syndromes."""
    values = np.asarray(a_syndromes, dtype=np.uint32)
    log_mass = np.asarray(log_mass, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.all(values < log_mass.size):
        raise ValueError("collapsed SMC A syndromes are invalid")
    result = np.zeros(values.shape[0], dtype=np.float64)
    for factor in range(values.shape[1]):
        result += log_mass[values[:, factor]]
    return result


def collapsed_log_target(b_columns, a_syndromes, p, log_mass):
    """Unnormalized log density of the exact collapsed cold B marginal."""
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    if b_columns.ndim == 1:
        b_columns = b_columns[None, :]
    bit_weights = np.asarray(
        [sum(int(value).bit_count() for value in row) for row in b_columns],
        dtype=np.float64,
    )
    return bit_weights * math.log(float(p) / (1.0 - float(p))) + collapsed_log_likelihood(
        a_syndromes, log_mass,
    )


def _initial_b_columns(config, H, seed_identity):
    """Draw exact iid prior B states in either independently audited order."""
    from exp101_certified_src.prng import PortablePrng

    r = int(H.shape[0])
    result = np.zeros((config.num_particles, r), dtype=np.uint32)
    if seed_identity.base_family == "column_major":
        positions = [(column, row) for column in range(r) for row in range(r)]
    else:
        positions = [(column, row) for row in range(r) for column in range(r)]
    for output_slot in range(config.num_particles):
        rng = PortablePrng(seed_identity.seed("initialize", output_slot=output_slot))
        for column, row in positions:
            if rng.random() < config.p:
                result[output_slot, column] |= np.uint32(1) << np.uint32(row)
    return result


def _normalized_incremental_weights(log_likelihood, delta_lambda):
    log_increment = float(delta_lambda) * np.asarray(log_likelihood, dtype=np.float64)
    if log_increment.ndim != 1 or not np.all(np.isfinite(log_increment)):
        raise CollapsedSmcConflictError("collapsed SMC incremental log weights are non-finite")
    maximum = float(log_increment.max())
    unnormalized = np.exp(log_increment - maximum)
    total = float(unnormalized.sum(dtype=np.float64))
    if not total > 0.0 or not math.isfinite(total):
        raise CollapsedSmcConflictError("collapsed SMC incremental weights vanished")
    weights = np.ascontiguousarray(unnormalized / total)
    cess = 1.0 / float(np.square(weights).sum(dtype=np.float64))
    return weights, cess, float(weights.max()), maximum + math.log(total / weights.size)


def systematic_resampling(weights, offset):
    """Return deterministic systematic parents for one offset in ``[0, 1/N)``."""
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 1 or weights.size < 2 or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)):
        raise ValueError("collapsed SMC systematic resampling weights are invalid")
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("collapsed SMC systematic resampling weights are not normalized")
    particles = int(weights.size)
    offset = float(offset)
    if not 0.0 <= offset < 1.0 / particles:
        raise ValueError("collapsed SMC systematic resampling offset is invalid")
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative[-1] = 1.0
    parents = np.empty(particles, dtype=np.int32)
    parent = 0
    for child in range(particles):
        position = offset + child / particles
        while parent < particles - 1 and position >= cumulative[parent]:
            parent += 1
        parents[child] = parent
    return parents


def _mutate_population(b_columns, a_syndromes, config, seed_identity, stage,
                       neighbors, neighbor_counts, log_mass, log_odds, engine):
    """Apply clone-safe exact pi_lambda block heatbath sweeps to every child."""
    from exp101_certified_src.prng import PortablePrng

    if engine not in ("reference", "numba"):
        raise ValueError("collapsed SMC engine must be reference or numba")
    if engine == "numba" and _hp_update_b_sweep is None:
        raise RuntimeError("Numba is required for the collapsed SMC numba engine")
    particles, r = b_columns.shape
    categories = 1 << min(config.block_size, r)
    column_order = np.empty(r, dtype=np.int32)
    row_order = np.empty(r, dtype=np.int32)
    candidate_masks = np.empty(categories, dtype=np.uint32)
    candidate_counts = np.empty(categories, dtype=np.uint8)
    candidate_weights = np.empty(categories, dtype=np.float64)
    changes = np.zeros(particles, dtype=np.int32)
    power = config.lambda_values[stage]
    for output_slot in range(particles):
        for sweep in range(config.mutation_sweeps):
            seed = seed_identity.seed("mutation", stage, sweep, output_slot)
            if engine == "reference":
                changed = _reference_power_b_sweep(
                    b_columns[output_slot], a_syndromes[output_slot],
                    PortablePrng(seed), config.block_size, power, neighbors,
                    neighbor_counts, log_mass, log_odds,
                )
            else:
                changed = _hp_update_b_sweep(
                    b_columns[output_slot], a_syndromes[output_slot],
                    PortablePrng(seed).state_array(), config.block_size, power,
                    neighbors, neighbor_counts, log_mass, log_odds, column_order,
                    row_order, candidate_masks, candidate_counts, candidate_weights,
                )
                if changed < 0:
                    raise CollapsedSmcConflictError("collapsed SMC block heatbath vanished")
            changes[output_slot] += int(changed)
    return changes


def run_collapsed_smc_population(model, frame, H, syndrome, config, seed_identity,
                                 *, engine="numba", mass=None):
    """Run one fixed-schedule resample-move population on the collapsed target.

    Every nonzero lambda stage reweights the exact previous bridge, applies
    systematic resampling unconditionally, and gives each output slot fresh
    mutation randomness.  The function records enough B-level state to audit
    weights, ancestry, and hard-coset algebra without computing q_top.
    """
    if not isinstance(config, CollapsedSmcConfig):
        raise TypeError("collapsed SMC config has the wrong type")
    if getattr(seed_identity, "method_id", None) != config.method_id:
        raise CollapsedSmcConflictError("collapsed SMC config/seed method mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise CollapsedSmcConflictError("collapsed SMC observable frame mismatch") from exc
    H = np.ascontiguousarray(H, dtype=np.uint8)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    if syndrome.shape != (model.num_checks,):
        raise ValueError("collapsed SMC syndrome shape mismatch")
    r, _ = H.shape
    mass_engine = "numba" if engine == "numba" else "reference"
    expected_mass = build_classical_coset_mass(H, config.p, engine=mass_engine)
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if (mass.shape != expected_mass.shape or not np.all(np.isfinite(mass))
            or np.any(mass <= 0.0) or not np.array_equal(mass, expected_mass)):
        raise CollapsedSmcConflictError("collapsed SMC mass table does not match H and p")
    log_mass = np.ascontiguousarray(np.log(mass))
    log_odds = math.log(config.p / (1.0 - config.p))
    neighbors, neighbor_counts = _classical_row_neighbors(H)

    b_columns = _initial_b_columns(config, H, seed_identity)
    a_syndromes = a_syndromes_from_b(syndrome, H, b_columns)
    log_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
    if not np.all(np.isfinite(log_likelihood)):
        raise CollapsedSmcConflictError("collapsed SMC initial likelihood is non-finite")

    levels = config.num_levels
    particles = config.num_particles
    b_by_stage = np.empty((levels, particles, r), dtype=np.uint32)
    log_likelihood_by_stage = np.empty((levels, particles), dtype=np.float64)
    roots_by_stage = np.empty((levels, particles), dtype=np.int32)
    normalized_incremental_weights = np.empty((levels - 1, particles), dtype=np.float64)
    parent_indices = np.empty((levels - 1, particles), dtype=np.int32)
    resample_offsets = np.empty(levels - 1, dtype=np.float64)
    conditional_ess = np.empty(levels - 1, dtype=np.float64)
    max_normalized_weight = np.empty(levels - 1, dtype=np.float64)
    log_normalizer_increments = np.empty(levels - 1, dtype=np.float64)
    mutation_block_changes = np.empty((levels - 1, particles), dtype=np.int32)
    roots = np.arange(particles, dtype=np.int32)
    b_by_stage[0] = b_columns
    log_likelihood_by_stage[0] = log_likelihood
    roots_by_stage[0] = roots

    from exp101_certified_src.prng import PortablePrng

    for stage in range(1, levels):
        weights, cess, maximum, log_increment = _normalized_incremental_weights(
            log_likelihood, config.lambda_values[stage] - config.lambda_values[stage - 1],
        )
        offset = PortablePrng(seed_identity.seed("resample", stage)).random() / particles
        parents = systematic_resampling(weights, offset)
        normalized_incremental_weights[stage - 1] = weights
        conditional_ess[stage - 1] = cess
        max_normalized_weight[stage - 1] = maximum
        log_normalizer_increments[stage - 1] = log_increment
        resample_offsets[stage - 1] = offset
        parent_indices[stage - 1] = parents
        b_columns = np.ascontiguousarray(b_columns[parents].copy())
        a_syndromes = np.ascontiguousarray(a_syndromes[parents].copy())
        roots = np.ascontiguousarray(roots[parents].copy())
        mutation_block_changes[stage - 1] = _mutate_population(
            b_columns, a_syndromes, config, seed_identity, stage, neighbors,
            neighbor_counts, log_mass, log_odds, engine,
        )
        expected_a_syndromes = a_syndromes_from_b(syndrome, H, b_columns)
        if not np.array_equal(a_syndromes, expected_a_syndromes):
            raise CollapsedSmcConflictError("collapsed SMC mutation broke B/A algebra")
        log_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
        if not np.all(np.isfinite(log_likelihood)):
            raise CollapsedSmcConflictError("collapsed SMC mutation likelihood is non-finite")
        b_by_stage[stage] = b_columns
        log_likelihood_by_stage[stage] = log_likelihood
        roots_by_stage[stage] = roots

    root_counts = np.bincount(roots, minlength=particles).astype(np.int32)
    root_fractions = root_counts.astype(np.float64) / particles
    root_family_ess = 1.0 / float(np.square(root_fractions).sum(dtype=np.float64))
    bit_weights = np.asarray(
        [sum(int(value).bit_count() for value in row) for row in b_columns],
        dtype=np.int16,
    )
    attempts_per_particle = (
        config.mutation_sweeps * r * ((r + config.block_size - 1) // config.block_size)
    )
    return {
        "raw_version": COLLAPSED_SMC_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "b_columns_by_stage": b_by_stage,
        "log_likelihood_by_stage": log_likelihood_by_stage,
        "roots_by_stage": roots_by_stage,
        "normalized_incremental_weights": normalized_incremental_weights,
        "parent_indices": parent_indices,
        "resample_offsets": resample_offsets,
        "conditional_ess": conditional_ess,
        "max_normalized_weight": max_normalized_weight,
        "log_normalizer_increments": log_normalizer_increments,
        "mutation_block_changes": mutation_block_changes,
        "mutation_block_attempts_per_particle": np.int32(attempts_per_particle),
        "final_root_counts": root_counts,
        "final_root_family_ess": np.float64(root_family_ess),
        "final_distinct_roots": np.int32(np.count_nonzero(root_counts)),
        "final_max_root_fraction": np.float64(root_fractions.max()),
        "final_b_bit_weights": bit_weights,
        "final_collapsed_log_target": collapsed_log_target(
            b_columns, a_syndromes, config.p, log_mass,
        ),
        "lambda_values": np.asarray(config.lambda_values, dtype=np.float64),
        "lambda_sha256": _float64_big_endian_sha256(config.lambda_values),
        "mass_sha256": _float64_big_endian_sha256(mass),
        "kernel": COLLAPSED_SMC_KERNEL,
        "engine": engine,
    }
