"""No-resampling annealed importance sampling for the collapsed q=0 HGP target.

The implementation is a local feasibility engine.  It keeps every exact-prior
particle lineage alive, accumulates the full AIS path weight, and applies a
reversible random-scan B-block heatbath at each bridge level.  It intentionally
does not calculate q_top or claim posterior convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    _bits_to_mask,
    _classical_row_neighbors,
    _ctt_nb_reversible_transition,
    _ctt_reference_reversible_transition,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .seeds import derive_seed


COLLAPSED_AIS_VERSION = "exp102.q0_hgp_collapsed_ais.v0"
COLLAPSED_AIS_RAW_VERSION = "exp102.q0_hgp_collapsed_ais.raw.v0"
COLLAPSED_AIS_KERNEL = "reversible_random_block_heatbath_ais.v1"
BASE_FAMILIES = ("column_major", "row_major")


class CollapsedAisConflictError(ValueError):
    """A collapsed-AIS identity, algebra, or numerical invariant failed."""


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


def float64_big_endian_sha256(values):
    return hashlib.sha256(
        np.asarray(values, dtype=">f8").tobytes(order="C")
    ).hexdigest()


def quadratic_lambda_schedule(num_levels):
    """Return the HP64-compatible bridge without reading any simulation output."""
    levels = int(num_levels)
    if levels < 2:
        raise ValueError("collapsed AIS requires at least two lambda levels")
    denominator = float((levels - 1) ** 2)
    values = tuple((index * index) / denominator for index in range(levels))
    return (0.0, *values[1:-1], 1.0)


@dataclass(frozen=True)
class CollapsedAisConfig:
    """Frozen bridge, particle count, and reversible mutation parameters."""

    p: float
    num_particles: int
    lambda_values: tuple[float, ...]
    mutation_sweeps: int = 1
    block_size: int = 8
    method_id: str = ""

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("collapsed AIS p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in ("num_particles", "mutation_sweeps", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.num_particles < 2:
            raise ValueError("collapsed AIS requires at least two particles")
        if not 1 <= self.mutation_sweeps <= 16:
            raise ValueError("collapsed AIS mutation sweeps must lie in [1, 16]")
        if self.block_size != 8:
            raise ValueError("collapsed AIS freezes eight-bit B blocks")
        values = tuple(float(value) for value in self.lambda_values)
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("collapsed AIS lambda schedule is invalid")
        if values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("collapsed AIS lambda schedule must start at zero and end at one")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("collapsed AIS lambda schedule must be strictly increasing")
        object.__setattr__(self, "lambda_values", values)
        expected = (
            f"CAIS{len(values):02d}-B{self.block_size}-S{self.mutation_sweeps}"
            f"-N{self.num_particles}"
        )
        if self.method_id and self.method_id != expected:
            raise ValueError("collapsed AIS method ID does not match the frozen parameters")
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
            "resampling": "none",
            "kernel": COLLAPSED_AIS_KERNEL,
            "prior_endpoint": "exact_iid_bernoulli_B",
        }


@dataclass(frozen=True)
class CollapsedAisSeedIdentity:
    """One independent exact-base AIS population with clone-free substreams."""

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
        if not isinstance(self.method_id, str) or not self.method_id.startswith("CAIS"):
            raise ValueError("collapsed AIS method ID is invalid")
        if self.base_family not in BASE_FAMILIES:
            raise ValueError("collapsed AIS base family is invalid")
        if isinstance(self.population_index, bool) or int(self.population_index) < 0:
            raise ValueError("collapsed AIS population index is invalid")
        object.__setattr__(self, "population_index", int(self.population_index))
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("collapsed AIS trajectory namespace is empty")

    def seed(self, role, stage=0, sweep=0, output_slot=0):
        return derive_seed(
            COLLAPSED_AIS_VERSION,
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


@dataclass(frozen=True)
class _ReversibleMutationConfig:
    block_size: int
    reversible_sweeps_per_level: int


def _y_columns(syndrome, H):
    rows, columns = H.shape
    matrix = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    return np.asarray([_bits_to_mask(matrix[:, column]) for column in range(columns)], dtype=np.uint32)


def a_syndromes_from_b(syndrome, H, b_columns):
    """Return all A coset syndromes for a batch of collapsed B states."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    if b_columns.ndim != 2 or b_columns.shape[1] != H.shape[0]:
        raise ValueError("collapsed AIS B-column array has the wrong shape")
    result = np.repeat(_y_columns(syndrome, H)[None, :], b_columns.shape[0], axis=0)
    for column in range(H.shape[1]):
        for row in np.flatnonzero(H[:, column]):
            result[:, column] ^= b_columns[:, int(row)]
    return np.ascontiguousarray(result, dtype=np.uint32)


def collapsed_log_likelihood(a_syndromes, log_mass):
    values = np.asarray(a_syndromes, dtype=np.uint32)
    if values.ndim == 1:
        values = values[None, :]
    log_mass = np.asarray(log_mass, dtype=np.float64)
    if values.ndim != 2 or not np.all(values < log_mass.size):
        raise ValueError("collapsed AIS A syndromes are invalid")
    result = np.zeros(values.shape[0], dtype=np.float64)
    for factor in range(values.shape[1]):
        result += log_mass[values[:, factor]]
    return result


def collapsed_log_target(b_columns, a_syndromes, p, log_mass, power=1.0):
    """Unnormalized log density of the exact collapsed bridge at ``power``."""
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    if b_columns.ndim == 1:
        b_columns = b_columns[None, :]
    bit_weights = np.asarray(
        [sum(int(value).bit_count() for value in row) for row in b_columns],
        dtype=np.float64,
    )
    return bit_weights * math.log(float(p) / (1.0 - float(p))) + float(power) * collapsed_log_likelihood(
        a_syndromes, log_mass,
    )


def normalized_log_weights(log_weights):
    """Return normalized weights, ESS, maximum weight, and log mean weight."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if log_weights.ndim != 1 or not np.all(np.isfinite(log_weights)):
        raise CollapsedAisConflictError("collapsed AIS log weights are non-finite")
    maximum = float(log_weights.max())
    values = np.exp(log_weights - maximum)
    total = float(values.sum(dtype=np.float64))
    if not total > 0.0 or not math.isfinite(total):
        raise CollapsedAisConflictError("collapsed AIS weights vanished")
    normalized = np.ascontiguousarray(values / total)
    ess = 1.0 / float(np.square(normalized).sum(dtype=np.float64))
    return normalized, ess, float(normalized.max()), maximum + math.log(total / normalized.size)


def _initial_b_columns(config, H, seed_identity):
    from exp101_certified_src.prng import PortablePrng

    rows = int(H.shape[0])
    result = np.zeros((config.num_particles, rows), dtype=np.uint32)
    if seed_identity.base_family == "column_major":
        positions = [(column, row) for column in range(rows) for row in range(rows)]
    else:
        positions = [(column, row) for row in range(rows) for column in range(rows)]
    for output_slot in range(config.num_particles):
        rng = PortablePrng(seed_identity.seed("initialize", output_slot=output_slot))
        for column, row in positions:
            if rng.random() < config.p:
                result[output_slot, column] |= np.uint32(1) << np.uint32(row)
    return result


def _mutate_population(b_columns, a_syndromes, config, seed_identity, stage,
                       neighbors, neighbor_counts, log_mass, log_odds, engine):
    """Apply a reversible random-scan block-heatbath power at one bridge level."""
    from exp101_certified_src.prng import PortablePrng

    if engine not in ("reference", "numba"):
        raise ValueError("collapsed AIS engine must be reference or numba")
    if engine == "numba" and _ctt_nb_reversible_transition is None:
        raise RuntimeError("Numba is required for the collapsed AIS numba engine")
    particles, rows = b_columns.shape
    mutation_config = _ReversibleMutationConfig(
        block_size=config.block_size,
        reversible_sweeps_per_level=config.mutation_sweeps,
    )
    candidate_weights = np.empty(1 << min(config.block_size, rows), dtype=np.float64)
    attempts = np.empty(particles, dtype=np.int32)
    changes = np.empty(particles, dtype=np.int32)
    power = config.lambda_values[stage]
    for output_slot in range(particles):
        seed = seed_identity.seed("mutation", stage, 0, output_slot)
        if engine == "reference":
            count, changed = _ctt_reference_reversible_transition(
                b_columns[output_slot], a_syndromes[output_slot], PortablePrng(seed),
                mutation_config, power, neighbors, neighbor_counts, log_mass, log_odds,
            )
        else:
            count, changed = _ctt_nb_reversible_transition(
                b_columns[output_slot], a_syndromes[output_slot],
                PortablePrng(seed).state_array(), config.block_size, config.mutation_sweeps,
                power, neighbors, neighbor_counts, log_mass, log_odds, candidate_weights,
            )
            if count < 0:
                raise CollapsedAisConflictError("collapsed AIS reversible mutation vanished")
        attempts[output_slot] = int(count)
        changes[output_slot] = int(changed)
    return attempts, changes


def run_collapsed_ais_population(model, frame, H, syndrome, config, seed_identity,
                                 *, engine="numba", mass=None):
    """Run one clone-free exact-base AIS population and retain full path weights.

    The incremental AIS factor is evaluated before the reversible transition:
    ``log w_t += (lambda_t-lambda_{t-1}) log L(B_{t-1})``.  Since the
    random-scan heatbath kernel is reversible with respect to each bridge,
    this is the standard AIS extended-path weight rather than an SMC
    resampling estimator.
    """
    if not isinstance(config, CollapsedAisConfig):
        raise TypeError("collapsed AIS config has the wrong type")
    if getattr(seed_identity, "method_id", None) != config.method_id:
        raise CollapsedAisConflictError("collapsed AIS config/seed method mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise CollapsedAisConflictError("collapsed AIS observable frame mismatch") from exc
    H = np.ascontiguousarray(H, dtype=np.uint8)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    if syndrome.shape != (model.num_checks,):
        raise ValueError("collapsed AIS syndrome shape mismatch")
    mass_engine = "numba" if engine == "numba" else "reference"
    expected_mass = build_classical_coset_mass(H, config.p, engine=mass_engine)
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if (mass.shape != expected_mass.shape or not np.all(np.isfinite(mass))
            or np.any(mass <= 0.0) or not np.array_equal(mass, expected_mass)):
        raise CollapsedAisConflictError("collapsed AIS mass table does not match H and p")
    log_mass = np.ascontiguousarray(np.log(mass))
    log_odds = math.log(config.p / (1.0 - config.p))
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    b_columns = _initial_b_columns(config, H, seed_identity)
    a_syndromes = a_syndromes_from_b(syndrome, H, b_columns)
    log_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
    if not np.all(np.isfinite(log_likelihood)):
        raise CollapsedAisConflictError("collapsed AIS initial likelihood is non-finite")

    levels = config.num_levels
    particles = config.num_particles
    rows = H.shape[0]
    b_by_stage = np.empty((levels, particles, rows), dtype=np.uint32)
    log_likelihood_by_stage = np.empty((levels, particles), dtype=np.float64)
    cumulative_log_weights = np.zeros((levels, particles), dtype=np.float64)
    log_weight_increments = np.empty((levels - 1, particles), dtype=np.float64)
    incremental_ess = np.empty(levels - 1, dtype=np.float64)
    incremental_max_weight = np.empty(levels - 1, dtype=np.float64)
    cumulative_ess = np.empty(levels - 1, dtype=np.float64)
    cumulative_max_weight = np.empty(levels - 1, dtype=np.float64)
    mutation_attempts = np.empty((levels - 1, particles), dtype=np.int32)
    mutation_changes = np.empty((levels - 1, particles), dtype=np.int32)
    b_by_stage[0] = b_columns
    log_likelihood_by_stage[0] = log_likelihood

    for stage in range(1, levels):
        increment = (
            config.lambda_values[stage] - config.lambda_values[stage - 1]
        ) * log_likelihood
        log_weight_increments[stage - 1] = increment
        cumulative_log_weights[stage] = cumulative_log_weights[stage - 1] + increment
        _, incremental_ess[stage - 1], incremental_max_weight[stage - 1], _ = normalized_log_weights(increment)
        _, cumulative_ess[stage - 1], cumulative_max_weight[stage - 1], _ = normalized_log_weights(
            cumulative_log_weights[stage],
        )
        attempts, changes = _mutate_population(
            b_columns, a_syndromes, config, seed_identity, stage, neighbors,
            neighbor_counts, log_mass, log_odds, engine,
        )
        mutation_attempts[stage - 1] = attempts
        mutation_changes[stage - 1] = changes
        expected_a_syndromes = a_syndromes_from_b(syndrome, H, b_columns)
        if not np.array_equal(a_syndromes, expected_a_syndromes):
            raise CollapsedAisConflictError("collapsed AIS mutation broke B/A algebra")
        log_likelihood = collapsed_log_likelihood(a_syndromes, log_mass)
        if not np.all(np.isfinite(log_likelihood)):
            raise CollapsedAisConflictError("collapsed AIS mutation likelihood is non-finite")
        b_by_stage[stage] = b_columns
        log_likelihood_by_stage[stage] = log_likelihood

    final_weights, final_ess, final_max_weight, log_mean_weight = normalized_log_weights(
        cumulative_log_weights[-1],
    )
    bit_weights = np.asarray(
        [sum(int(value).bit_count() for value in row) for row in b_columns],
        dtype=np.int16,
    )
    return {
        "raw_version": COLLAPSED_AIS_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "b_columns_by_stage": b_by_stage,
        "log_likelihood_by_stage": log_likelihood_by_stage,
        "log_weight_increments": log_weight_increments,
        "cumulative_log_weights": cumulative_log_weights,
        "incremental_ess": incremental_ess,
        "incremental_max_weight": incremental_max_weight,
        "cumulative_ess": cumulative_ess,
        "cumulative_max_weight": cumulative_max_weight,
        "mutation_attempts": mutation_attempts,
        "mutation_changes": mutation_changes,
        "final_normalized_weights": final_weights,
        "final_importance_ess": np.float64(final_ess),
        "final_max_normalized_weight": np.float64(final_max_weight),
        "final_log_mean_weight": np.float64(log_mean_weight),
        "final_b_bit_weights": bit_weights,
        "final_collapsed_log_target": collapsed_log_target(
            b_columns, a_syndromes, config.p, log_mass,
        ),
        "lambda_values": np.asarray(config.lambda_values, dtype=np.float64),
        "lambda_sha256": float64_big_endian_sha256(config.lambda_values),
        "mass_sha256": float64_big_endian_sha256(mass),
        "kernel": COLLAPSED_AIS_KERNEL,
        "engine": engine,
    }
