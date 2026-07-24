"""Exact full-B-column Gibbs updates for the collapsed q=0 HGP posterior.

Existing collapsed kernels update a small block inside one B column or an
entire B row.  This kernel instead samples all ``r`` bits of one B column from
its exact conditional.  It is expensive (``2**r`` candidates), but it changes
every B row coherently and is therefore a genuinely different global move.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - the optimized successor requires Numba.
    njit = None

from .exp101_bridge import load_exp101
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _qubit_signatures,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    validate_hgp_wiring,
)


FULL_COLUMN_GIBBS_VERSION = "exp102.q0_hgp_full_column_gibbs.v0"
FULL_COLUMN_GIBBS_KERNEL = "exact_collapsed_full_B_column_heatbath.v1"
FULL_COLUMN_GIBBS_METHOD_ID = "FCG-C24"
FULL_COLUMN_STREAMING_VERSION = "exp102.q0_hgp_full_column_streaming.v1"
FULL_COLUMN_STREAMING_KERNEL = "exact_collapsed_full_B_column_heatbath_streaming.v1"
FULL_COLUMN_DIRECT_BLOCK_VERSION = "exp102.q0_hgp_full_column_direct_block.v1"
FULL_COLUMN_DIRECT_BLOCK_KERNEL = (
    "exact_collapsed_full_B_column_heatbath_direct_positive_block.v1"
)
FULL_COLUMN_DIRECT_BLOCK_BITS = 12
FULL_COLUMN_DIRECT_MIN_NORMAL_MARGIN_LOG = math.log(float(1 << 20))
FULL_COLUMN_COUNTER_NAMES = (
    "column_updates",
    "column_changes",
    "column_changed_bits",
    "a_conditional_draws",
)


class FullColumnGibbsConflictError(ValueError):
    """Raised when a full-column conditional loses its target identity."""


@dataclass(frozen=True)
class FullColumnCandidateCache:
    """State-independent candidates for an exact B-column conditional."""

    rows: int
    p: float
    masks: np.ndarray
    popcounts: np.ndarray
    log_prior: np.ndarray

    def __post_init__(self):
        if not 1 <= int(self.rows) <= 24:
            raise ValueError("full-column Gibbs supports 1 <= rows <= 24")
        expected = 1 << int(self.rows)
        if (self.masks.shape != (expected,) or self.masks.dtype != np.uint32
                or self.popcounts.shape != (expected,) or self.popcounts.dtype != np.uint8
                or self.log_prior.shape != (expected,) or self.log_prior.dtype != np.float64):
            raise ValueError("full-column candidate cache has the wrong shape or dtype")
        if not np.array_equal(self.masks, np.arange(expected, dtype=np.uint32)):
            raise ValueError("full-column candidate masks are not canonical")
        if not np.all(np.isfinite(self.log_prior)):
            raise ValueError("full-column candidate log prior is non-finite")


@dataclass
class FullColumnWorkspace:
    """Reusable dense arrays; avoiding per-update allocations controls memory."""

    xor_indices: np.ndarray
    log_weights: np.ndarray
    scratch: np.ndarray

    def __post_init__(self):
        expected = self.log_weights.shape
        if (self.xor_indices.shape != expected or self.xor_indices.dtype != np.uint32
                or self.log_weights.dtype != np.float64 or self.scratch.shape != expected
                or self.scratch.dtype != np.float64):
            raise ValueError("full-column workspace has the wrong shape or dtype")


@dataclass(frozen=True)
class FullColumnStreamingCache:
    """Small state-independent cache for the memory-streaming conditional."""

    rows: int
    p: float
    log_odds: float
    popcount16: np.ndarray

    def __post_init__(self):
        if not 1 <= int(self.rows) <= 24:
            raise ValueError("streaming full-column Gibbs supports 1 <= rows <= 24")
        if (not math.isfinite(float(self.p)) or not 0.0 < float(self.p) < 0.5
                or not math.isfinite(float(self.log_odds))):
            raise ValueError("streaming full-column cache has invalid probability data")
        if (self.popcount16.shape != (1 << 16,)
                or self.popcount16.dtype != np.uint8):
            raise ValueError("streaming full-column popcount table is invalid")


@dataclass
class FullColumnStreamingWorkspace:
    """One dense CDF, replacing the legacy six-array candidate workspace."""

    cdf: np.ndarray

    def __post_init__(self):
        if self.cdf.ndim != 1 or self.cdf.dtype != np.float64:
            raise ValueError("streaming full-column workspace is invalid")


@dataclass(frozen=True)
class FullColumnDirectBlockCache:
    """State-independent data for direct positive-weight block sampling."""

    rows: int
    p: float
    block_bits: int
    mass: np.ndarray
    mass_min: float
    popcount16: np.ndarray
    odds_powers: np.ndarray

    def __post_init__(self):
        if not 1 <= int(self.rows) <= 24:
            raise ValueError("direct-block full-column rows are invalid")
        if int(self.block_bits) != min(FULL_COLUMN_DIRECT_BLOCK_BITS, int(self.rows)):
            raise ValueError("direct-block partition changed")
        if (not math.isfinite(float(self.p)) or not 0.0 < float(self.p) < 0.5
                or self.popcount16.shape != (1 << 16,)
                or self.popcount16.dtype != np.uint8
                or self.mass.shape != (1 << int(self.rows),)
                or self.mass.dtype != np.float64
                or not math.isfinite(float(self.mass_min))
                or not float(self.mass_min) > 0.0
                or self.odds_powers.shape != (int(self.rows) + 1,)
                or self.odds_powers.dtype != np.float64
                or not np.all(np.isfinite(self.odds_powers))
                or not np.all(self.odds_powers > 0.0)):
            raise ValueError("direct-block full-column cache is invalid")


@dataclass
class FullColumnDirectBlockWorkspace:
    """Only one subtotal per fixed candidate block is materialized."""

    block_sums: np.ndarray

    def __post_init__(self):
        if (self.block_sums.ndim != 1 or self.block_sums.dtype != np.float64
                or self.block_sums.size <= 0):
            raise ValueError("direct-block full-column workspace is invalid")


@dataclass(frozen=True)
class FullColumnGibbsConfig:
    """Fixed-clock local diagnostic configuration for the full-column kernel.

    A sweep visits every B column once in a fresh random permutation.  Keeping
    the public clock in sweeps, rather than individual block updates, prevents
    this unusually expensive kernel from being compared to other samplers on
    an ambiguous time scale.
    """

    p: float
    burn_sweeps: int
    measurement_sweeps: int
    method_id: str = FULL_COLUMN_GIBBS_METHOD_ID
    column_schedule: str = "random_permutation_each_sweep"

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("full-column Gibbs p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_sweeps", "measurement_sweeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.measurement_sweeps % 8:
            raise ValueError("measurement sweeps must divide into eight time blocks")
        if self.method_id != FULL_COLUMN_GIBBS_METHOD_ID:
            raise ValueError("full-column Gibbs method ID is not frozen")
        if self.column_schedule != "random_permutation_each_sweep":
            raise ValueError("full-column Gibbs column schedule is not frozen")

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": self.p,
            "burn_sweeps": self.burn_sweeps,
            "measurement_sweeps": self.measurement_sweeps,
            "column_schedule": self.column_schedule,
            "kernel": FULL_COLUMN_GIBBS_KERNEL,
            "a_update": "exact_column_conditional_draw_after_each_B_update",
        }


def _as_binary_matrix(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    if H.ndim != 2 or H.shape[0] <= 0 or H.shape[1] <= 0:
        raise ValueError("full-column Gibbs H must have positive dimensions")
    if H.shape[0] > 24 or H.shape[1] > 32 or not np.all((H == 0) | (H == 1)):
        raise ValueError("full-column Gibbs H is outside the supported binary shape")
    if not np.all(H.sum(axis=0) > 0) or not np.all(H.sum(axis=1) > 0):
        raise ValueError("full-column Gibbs H cannot contain an empty row or column")
    return H


def _popcount_table_16():
    table = np.zeros(1 << 16, dtype=np.uint8)
    for value in range(1, table.size):
        table[value] = table[value >> 1] + np.uint8(value & 1)
    return table


def build_full_column_candidate_cache(rows, p):
    """Build the canonical 2**r full-column candidate list once per p."""
    rows = int(rows)
    p = float(p)
    if not 1 <= rows <= 24 or not math.isfinite(p) or not 0.0 < p < 0.5:
        raise ValueError("invalid full-column cache parameters")
    size = 1 << rows
    masks = np.arange(size, dtype=np.uint32)
    table = _popcount_table_16()
    popcounts = np.asarray(
        table[masks & np.uint32(0xFFFF)] + table[masks >> np.uint32(16)],
        dtype=np.uint8,
    )
    log_prior = np.ascontiguousarray(
        popcounts.astype(np.float64) * math.log(p / (1.0 - p)), dtype=np.float64,
    )
    return FullColumnCandidateCache(rows, p, masks, popcounts, log_prior)


def build_full_column_workspace(cache):
    if not isinstance(cache, FullColumnCandidateCache):
        raise TypeError("full-column workspace needs a candidate cache")
    size = cache.masks.size
    return FullColumnWorkspace(
        xor_indices=np.empty(size, dtype=np.uint32),
        log_weights=np.empty(size, dtype=np.float64),
        scratch=np.empty(size, dtype=np.float64),
    )


def build_full_column_streaming_cache(rows, p):
    """Build the O(2**16) cache used by the streaming exact conditional."""
    rows = int(rows)
    p = float(p)
    if not 1 <= rows <= 24 or not math.isfinite(p) or not 0.0 < p < 0.5:
        raise ValueError("invalid streaming full-column cache parameters")
    return FullColumnStreamingCache(
        rows=rows,
        p=p,
        log_odds=math.log(p / (1.0 - p)),
        popcount16=_popcount_table_16(),
    )


def build_full_column_streaming_workspace(cache):
    if not isinstance(cache, FullColumnStreamingCache):
        raise TypeError("streaming full-column workspace needs a streaming cache")
    return FullColumnStreamingWorkspace(
        cdf=np.empty(1 << cache.rows, dtype=np.float64),
    )


def build_full_column_direct_block_cache(rows, p, mass):
    """Build the fixed 2**12 block partition and direct prior weights."""
    rows = int(rows)
    p = float(p)
    if not 1 <= rows <= 24 or not math.isfinite(p) or not 0.0 < p < 0.5:
        raise ValueError("invalid direct-block full-column cache parameters")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    if (mass.shape != (1 << rows,) or not np.all(np.isfinite(mass))
            or not np.all(mass > 0.0)):
        raise ValueError("invalid direct-block full-column mass table")
    odds = p / (1.0 - p)
    return FullColumnDirectBlockCache(
        rows=rows,
        p=p,
        block_bits=min(FULL_COLUMN_DIRECT_BLOCK_BITS, rows),
        mass=mass,
        mass_min=float(np.min(mass)),
        popcount16=_popcount_table_16(),
        odds_powers=np.asarray(
            [odds ** weight for weight in range(rows + 1)], dtype=np.float64,
        ),
    )


def build_full_column_direct_block_workspace(cache):
    if not isinstance(cache, FullColumnDirectBlockCache):
        raise TypeError("direct-block workspace needs a direct-block cache")
    block_count = 1 << (cache.rows - cache.block_bits)
    return FullColumnDirectBlockWorkspace(
        block_sums=np.empty(block_count, dtype=np.float64),
    )


def collapsed_a_syndromes(H, syndrome, b_columns):
    """Rebuild packed A-column syndromes ``Y xor B H`` from a B state."""
    H = _as_binary_matrix(H)
    rows, columns = H.shape
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    if syndrome.shape != (rows, columns) or b_columns.shape != (rows,):
        raise ValueError("full-column collapsed state has the wrong shape")
    result = np.zeros(columns, dtype=np.uint32)
    for column in range(columns):
        value = 0
        for row in range(rows):
            value |= int(syndrome[row, column]) << row
        for b_column in np.flatnonzero(H[:, column]):
            value ^= int(b_columns[int(b_column)])
        result[column] = np.uint32(value)
    return result


def _validate_collapsed_state(H, syndrome, b_columns, a_syndromes):
    H = _as_binary_matrix(H)
    rows, columns = H.shape
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    a_syndromes = np.asarray(a_syndromes, dtype=np.uint32)
    if b_columns.shape != (rows,) or a_syndromes.shape != (columns,):
        raise ValueError("full-column collapsed state has the wrong shape")
    expected = collapsed_a_syndromes(H, syndrome, b_columns)
    if not np.array_equal(a_syndromes, expected):
        raise FullColumnGibbsConflictError("cached A syndromes do not match B H")
    return H


def _column_log_weights_unchecked(H, b_columns, a_syndromes, column_index,
                                  log_mass, cache, workspace):
    rows, _ = H.shape
    column_index = int(column_index)
    if not 0 <= column_index < rows:
        raise ValueError("full-column index is invalid")
    log_mass = np.asarray(log_mass, dtype=np.float64)
    if log_mass.shape != (1 << rows,) or not np.all(np.isfinite(log_mass)):
        raise ValueError("full-column log mass table is invalid")
    if cache.rows != rows or cache.masks.size != log_mass.size:
        raise FullColumnGibbsConflictError("full-column cache does not match H")
    if workspace.log_weights.size != cache.masks.size:
        raise FullColumnGibbsConflictError("full-column workspace does not match cache")
    old = np.uint32(b_columns[column_index])
    neighbors = np.flatnonzero(H[column_index]).astype(np.int32)
    if neighbors.size == 0:
        raise FullColumnGibbsConflictError("full-column update has no likelihood factors")
    first = int(neighbors[0])
    base = np.uint32(a_syndromes[first]) ^ old
    np.bitwise_xor(cache.masks, base, out=workspace.xor_indices)
    np.take(log_mass, workspace.xor_indices, out=workspace.log_weights)
    np.add(workspace.log_weights, cache.log_prior, out=workspace.log_weights)
    for factor in neighbors[1:]:
        base = np.uint32(a_syndromes[int(factor)]) ^ old
        np.bitwise_xor(cache.masks, base, out=workspace.xor_indices)
        np.take(log_mass, workspace.xor_indices, out=workspace.scratch)
        np.add(workspace.log_weights, workspace.scratch, out=workspace.log_weights)
    return neighbors


def _streaming_cdf_reference(log_mass, bases, log_odds, popcount16, cdf):
    """Reference arithmetic order for the exact streaming CDF."""
    maximum = -math.inf
    for candidate in range(cdf.size):
        value = float(log_mass[candidate ^ int(bases[0])])
        value += (
            int(popcount16[candidate & 0xFFFF])
            + int(popcount16[candidate >> 16])
        ) * float(log_odds)
        for factor in range(1, bases.size):
            value += float(log_mass[candidate ^ int(bases[factor])])
        cdf[candidate] = value
        if value > maximum:
            maximum = value
    total = 0.0
    for candidate in range(cdf.size):
        total += math.exp(float(cdf[candidate]) - maximum)
        cdf[candidate] = total
    return total


if njit is not None:
    @njit(cache=True)
    def _streaming_cdf_numba(log_mass, bases, log_odds, popcount16, cdf):
        maximum = -math.inf
        for candidate in range(cdf.size):
            value = log_mass[candidate ^ int(bases[0])]
            value += (
                int(popcount16[candidate & 0xFFFF])
                + int(popcount16[candidate >> 16])
            ) * log_odds
            for factor in range(1, bases.size):
                value += log_mass[candidate ^ int(bases[factor])]
            cdf[candidate] = value
            if value > maximum:
                maximum = value
        total = 0.0
        for candidate in range(cdf.size):
            total += math.exp(cdf[candidate] - maximum)
            cdf[candidate] = total
        return total
else:  # pragma: no cover - exercised by the explicit engine guard.
    _streaming_cdf_numba = None


def _direct_candidate_weight_reference(candidate, mass, bases, odds_powers,
                                       popcount16):
    popcount = (
        int(popcount16[candidate & 0xFFFF])
        + int(popcount16[candidate >> 16])
    )
    weight = float(odds_powers[popcount])
    for factor in range(bases.size):
        weight *= float(mass[candidate ^ int(bases[factor])])
    return weight


def _direct_block_sums_reference(mass, bases, odds_powers, popcount16,
                                 block_bits, block_sums):
    block_size = 1 << int(block_bits)
    total = 0.0
    for block in range(block_sums.size):
        subtotal = 0.0
        start = block * block_size
        for candidate in range(start, start + block_size):
            subtotal += _direct_candidate_weight_reference(
                candidate, mass, bases, odds_powers, popcount16,
            )
        block_sums[block] = subtotal
        total += subtotal
    return total


def _direct_block_select_reference(mass, bases, odds_powers, popcount16,
                                   block_bits, block_sums, threshold):
    prefix = 0.0
    selected_block = block_sums.size - 1
    for block in range(block_sums.size):
        next_prefix = prefix + float(block_sums[block])
        if float(threshold) < next_prefix:
            selected_block = block
            break
        prefix = next_prefix
    block_size = 1 << int(block_bits)
    start = selected_block * block_size
    cumulative = prefix
    for candidate in range(start, start + block_size):
        cumulative += _direct_candidate_weight_reference(
            candidate, mass, bases, odds_powers, popcount16,
        )
        if float(threshold) < cumulative:
            return candidate
    return start + block_size - 1


if njit is not None:
    @njit(cache=True, inline="always")
    def _direct_candidate_weight_numba(candidate, mass, bases, odds_powers,
                                       popcount16):
        popcount = (
            int(popcount16[candidate & 0xFFFF])
            + int(popcount16[candidate >> 16])
        )
        weight = odds_powers[popcount]
        for factor in range(bases.size):
            weight *= mass[candidate ^ int(bases[factor])]
        return weight

    @njit(cache=True)
    def _direct_block_sums_numba(mass, bases, odds_powers, popcount16,
                                 block_bits, block_sums):
        block_size = 1 << int(block_bits)
        total = 0.0
        for block in range(block_sums.size):
            subtotal = 0.0
            start = block * block_size
            for candidate in range(start, start + block_size):
                subtotal += _direct_candidate_weight_numba(
                    candidate, mass, bases, odds_powers, popcount16,
                )
            block_sums[block] = subtotal
            total += subtotal
        return total

    @njit(cache=True)
    def _direct_block_select_numba(mass, bases, odds_powers, popcount16,
                                   block_bits, block_sums, threshold):
        prefix = 0.0
        selected_block = block_sums.size - 1
        for block in range(block_sums.size):
            next_prefix = prefix + block_sums[block]
            if threshold < next_prefix:
                selected_block = block
                break
            prefix = next_prefix
        block_size = 1 << int(block_bits)
        start = selected_block * block_size
        cumulative = prefix
        for candidate in range(start, start + block_size):
            cumulative += _direct_candidate_weight_numba(
                candidate, mass, bases, odds_powers, popcount16,
            )
            if threshold < cumulative:
                return candidate
        return start + block_size - 1
else:  # pragma: no cover - exercised by the explicit engine guard.
    _direct_block_sums_numba = None
    _direct_block_select_numba = None


def _direct_block_underflow_certificate(cache, factor_count):
    log_lower_bound = (
        math.log(float(cache.odds_powers[-1]))
        + int(factor_count) * math.log(float(cache.mass_min))
    )
    required = math.log(float(np.finfo(np.float64).tiny)) + (
        FULL_COLUMN_DIRECT_MIN_NORMAL_MARGIN_LOG
    )
    if not math.isfinite(log_lower_bound) or not log_lower_bound > required:
        raise FullColumnGibbsConflictError(
            "direct-block candidate weights lack the frozen normal-range margin",
        )
    return log_lower_bound


def _column_direct_block_sums_unchecked(
        H, b_columns, a_syndromes, column_index, mass, cache, workspace, *,
        engine):
    rows, _ = H.shape
    column_index = int(column_index)
    if not 0 <= column_index < rows:
        raise ValueError("direct-block full-column index is invalid")
    if (not isinstance(cache, FullColumnDirectBlockCache)
            or not isinstance(workspace, FullColumnDirectBlockWorkspace)
            or cache.rows != rows
            or workspace.block_sums.shape
            != (1 << (rows - cache.block_bits),)):
        raise FullColumnGibbsConflictError("direct-block full-column cache mismatch")
    if mass is not cache.mass:
        raise FullColumnGibbsConflictError(
            "direct-block full-column mass is not bound to its cache",
        )
    old = np.uint32(b_columns[column_index])
    neighbors = np.flatnonzero(H[column_index]).astype(np.int32)
    if neighbors.size == 0:
        raise FullColumnGibbsConflictError(
            "direct-block full-column update has no likelihood factors",
        )
    log_lower_bound = _direct_block_underflow_certificate(cache, neighbors.size)
    bases = np.ascontiguousarray(a_syndromes[neighbors] ^ old, dtype=np.uint32)
    arguments = (
        mass, bases, cache.odds_powers, cache.popcount16, cache.block_bits,
        workspace.block_sums,
    )
    if engine == "reference":
        total = _direct_block_sums_reference(*arguments)
    elif engine == "numba":
        if _direct_block_sums_numba is None:
            raise FullColumnGibbsConflictError("Numba direct-block engine is unavailable")
        total = _direct_block_sums_numba(*arguments)
    else:
        raise ValueError("direct-block full-column engine must be reference or numba")
    if (not math.isfinite(float(total)) or not float(total) > 0.0
            or not np.all(np.isfinite(workspace.block_sums))
            or not np.all(workspace.block_sums > 0.0)):
        raise FullColumnGibbsConflictError("direct-block full-column weights vanished")
    return neighbors, bases, float(total), log_lower_bound


def full_column_direct_block_subtotals(
        H, syndrome, b_columns, a_syndromes, column_index, mass, *, cache=None,
        workspace=None, engine="numba"):
    """Expose frozen subtotals for portability and runtime preflights."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    if cache is None:
        raise ValueError("direct-block subtotals require a p-bound cache")
    workspace = (
        build_full_column_direct_block_workspace(cache)
        if workspace is None else workspace
    )
    _, _, total, log_lower_bound = _column_direct_block_sums_unchecked(
        H, b_columns, a_syndromes, column_index, mass, cache, workspace,
        engine=engine,
    )
    return workspace.block_sums.copy(), total, log_lower_bound


def full_column_direct_block_conditional_probabilities(
        H, syndrome, b_columns, a_syndromes, column_index, mass, *, cache=None):
    """Return direct positive probabilities for exact small-code checks."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    if cache is None:
        raise ValueError("direct-block conditional requires a p-bound cache")
    rows, _ = H.shape
    mass = np.asarray(mass, dtype=np.float64)
    old = np.uint32(b_columns[int(column_index)])
    neighbors = np.flatnonzero(H[int(column_index)]).astype(np.int32)
    if mass is not cache.mass:
        raise FullColumnGibbsConflictError(
            "direct-block conditional mass is not bound to its cache",
        )
    _direct_block_underflow_certificate(cache, neighbors.size)
    bases = np.ascontiguousarray(a_syndromes[neighbors] ^ old, dtype=np.uint32)
    probabilities = np.empty(1 << rows, dtype=np.float64)
    total = 0.0
    for candidate in range(probabilities.size):
        weight = _direct_candidate_weight_reference(
            candidate, mass, bases, cache.odds_powers, cache.popcount16,
        )
        probabilities[candidate] = weight
        total += weight
    if not math.isfinite(total) or not total > 0.0:
        raise FullColumnGibbsConflictError("direct-block probabilities vanished")
    probabilities /= total
    return probabilities


def full_column_direct_block_gibbs_update(
        b_columns, a_syndromes, H, syndrome, column_index, mass, cache,
        workspace, rng, *, engine="numba"):
    """Heatbath one B column using fixed direct-positive subtotal blocks."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    neighbors, bases, total, _ = _column_direct_block_sums_unchecked(
        H, b_columns, a_syndromes, column_index, mass, cache, workspace,
        engine=engine,
    )
    threshold = float(rng.random()) * total
    arguments = (
        np.asarray(mass, dtype=np.float64), bases, cache.odds_powers,
        cache.popcount16, cache.block_bits, workspace.block_sums, threshold,
    )
    if engine == "reference":
        selected = int(_direct_block_select_reference(*arguments))
    elif engine == "numba":
        if _direct_block_select_numba is None:
            raise FullColumnGibbsConflictError("Numba direct-block engine is unavailable")
        selected = int(_direct_block_select_numba(*arguments))
    else:
        raise ValueError("direct-block full-column engine must be reference or numba")
    column_index = int(column_index)
    old = np.uint32(b_columns[column_index])
    new = np.uint32(selected)
    delta = old ^ new
    b_columns[column_index] = new
    if delta:
        for factor in neighbors:
            a_syndromes[int(factor)] ^= delta
    return bool(delta), int(int(delta).bit_count())


def _column_streaming_cdf_unchecked(H, b_columns, a_syndromes, column_index,
                                    log_mass, cache, workspace, *, engine):
    rows, _ = H.shape
    column_index = int(column_index)
    if not 0 <= column_index < rows:
        raise ValueError("streaming full-column index is invalid")
    if (not isinstance(cache, FullColumnStreamingCache)
            or not isinstance(workspace, FullColumnStreamingWorkspace)
            or cache.rows != rows
            or workspace.cdf.shape != (1 << rows,)):
        raise FullColumnGibbsConflictError("streaming full-column cache mismatch")
    log_mass = np.asarray(log_mass, dtype=np.float64)
    if log_mass.shape != (1 << rows,):
        raise ValueError("streaming full-column log mass table is invalid")
    old = np.uint32(b_columns[column_index])
    neighbors = np.flatnonzero(H[column_index]).astype(np.int32)
    if neighbors.size == 0:
        raise FullColumnGibbsConflictError(
            "streaming full-column update has no likelihood factors",
        )
    bases = np.ascontiguousarray(a_syndromes[neighbors] ^ old, dtype=np.uint32)
    if engine == "reference":
        total = _streaming_cdf_reference(
            log_mass, bases, cache.log_odds, cache.popcount16, workspace.cdf,
        )
    elif engine == "numba":
        if _streaming_cdf_numba is None:
            raise FullColumnGibbsConflictError("Numba streaming engine is unavailable")
        total = _streaming_cdf_numba(
            log_mass, bases, cache.log_odds, cache.popcount16, workspace.cdf,
        )
    else:
        raise ValueError("streaming full-column engine must be reference or numba")
    if not math.isfinite(float(total)) or not float(total) > 0.0:
        raise FullColumnGibbsConflictError("streaming full-column weights vanished")
    return neighbors, float(total)


def full_column_streaming_conditional_probabilities(
        H, syndrome, b_columns, a_syndromes, column_index, log_mass, *,
        cache=None, workspace=None, engine="numba"):
    """Return the streaming conditional for exact small-code comparisons."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    if cache is None:
        raise ValueError("streaming full-column conditional requires a p-bound cache")
    workspace = (
        build_full_column_streaming_workspace(cache)
        if workspace is None else workspace
    )
    _, total = _column_streaming_cdf_unchecked(
        H, b_columns, a_syndromes, column_index, log_mass, cache, workspace,
        engine=engine,
    )
    probabilities = np.empty_like(workspace.cdf)
    probabilities[0] = workspace.cdf[0]
    np.subtract(workspace.cdf[1:], workspace.cdf[:-1], out=probabilities[1:])
    probabilities /= total
    return probabilities


def full_column_streaming_gibbs_update(
        b_columns, a_syndromes, H, syndrome, column_index, log_mass, cache,
        workspace, rng, *, engine="numba"):
    """Heatbath a full B column with one O(2**r) streaming CDF buffer."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    neighbors, total = _column_streaming_cdf_unchecked(
        H, b_columns, a_syndromes, column_index, log_mass, cache, workspace,
        engine=engine,
    )
    threshold = float(rng.random()) * total
    selected = int(np.searchsorted(workspace.cdf, threshold, side="right"))
    if selected >= workspace.cdf.size:
        selected = workspace.cdf.size - 1
    column_index = int(column_index)
    old = np.uint32(b_columns[column_index])
    new = np.uint32(selected)
    delta = old ^ new
    b_columns[column_index] = new
    if delta:
        for factor in neighbors:
            a_syndromes[int(factor)] ^= delta
    return bool(delta), int(int(delta).bit_count())


def full_column_conditional_probabilities(H, syndrome, b_columns, a_syndromes,
                                          column_index, log_mass, cache=None,
                                          workspace=None):
    """Return an exact full-column conditional for small-code oracle checks."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    if cache is None:
        raise ValueError("full-column conditional requires a p-bound candidate cache")
    rows, _ = H.shape
    if cache.rows != rows:
        raise FullColumnGibbsConflictError("full-column cache does not match H")
    workspace = build_full_column_workspace(cache) if workspace is None else workspace
    _column_log_weights_unchecked(
        H, b_columns, a_syndromes, column_index, log_mass, cache, workspace,
    )
    maximum = float(np.max(workspace.log_weights))
    np.subtract(workspace.log_weights, maximum, out=workspace.log_weights)
    np.exp(workspace.log_weights, out=workspace.log_weights)
    total = float(workspace.log_weights.sum(dtype=np.float64))
    if not math.isfinite(total) or not total > 0.0:
        raise FullColumnGibbsConflictError("full-column conditional weights vanished")
    probabilities = np.ascontiguousarray(workspace.log_weights / total, dtype=np.float64)
    return probabilities


def full_column_gibbs_update(b_columns, a_syndromes, H, syndrome, column_index,
                             log_mass, cache, workspace, rng):
    """Heatbath one full B column and update every affected A syndrome exactly."""
    H = _validate_collapsed_state(H, syndrome, b_columns, a_syndromes)
    neighbors = _column_log_weights_unchecked(
        H, b_columns, a_syndromes, column_index, log_mass, cache, workspace,
    )
    maximum = float(np.max(workspace.log_weights))
    np.subtract(workspace.log_weights, maximum, out=workspace.log_weights)
    np.exp(workspace.log_weights, out=workspace.log_weights)
    np.cumsum(workspace.log_weights, out=workspace.log_weights)
    total = float(workspace.log_weights[-1])
    if not math.isfinite(total) or not total > 0.0:
        raise FullColumnGibbsConflictError("full-column conditional weights vanished")
    threshold = float(rng.random()) * total
    selected = int(np.searchsorted(workspace.log_weights, threshold, side="right"))
    if selected >= cache.masks.size:
        selected = cache.masks.size - 1
    column_index = int(column_index)
    old = np.uint32(b_columns[column_index])
    new = np.uint32(cache.masks[selected])
    delta = old ^ new
    b_columns[column_index] = new
    if delta:
        for factor in neighbors:
            a_syndromes[int(factor)] ^= delta
    return bool(delta), int(int(delta).bit_count())


def _new_portable_rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    return PortablePrng(int(seed))


def run_full_column_gibbs_trajectory(model, frame, H, syndrome, config,
                                     initial_state, update_seed, observation_seed,
                                     *, cache=None, workspace=None):
    """Run a fixed-clock exact B-column Gibbs diagnostic from a legal state."""
    if not isinstance(config, FullColumnGibbsConfig):
        raise TypeError("full-column Gibbs config has the wrong type")
    matrix = _as_binary_matrix(H)
    validate_hgp_wiring(matrix, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise FullColumnGibbsConflictError("full-column observable frame mismatch") from exc
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    if syndrome.shape != (model.H_check.shape[0],):
        raise ValueError("full-column syndrome has the wrong shape")
    initial_state = np.asarray(initial_state, dtype=np.uint8)
    if initial_state.shape != (model.num_qubits,):
        raise ValueError("full-column initial state has the wrong shape")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(initial_state, syndrome, matrix)
    mass = build_classical_coset_mass(matrix, config.p, engine="reference")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    cache = build_full_column_candidate_cache(matrix.shape[0], config.p) if cache is None else cache
    workspace = build_full_column_workspace(cache) if workspace is None else workspace
    if cache.rows != matrix.shape[0] or cache.p != config.p:
        raise FullColumnGibbsConflictError("full-column cache does not match configuration")
    update_rng = _new_portable_rng(update_seed)
    observation_rng = _new_portable_rng(observation_seed)
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    odds = config.p / (1.0 - config.p)
    odds_powers = np.asarray([odds ** weight for weight in range(matrix.shape[1] + 1)], dtype=np.float64)
    signatures = _qubit_signatures(frame)
    state = initial_state.copy()
    rows, columns = matrix.shape
    counters = np.zeros(4, dtype=np.int64)

    def one_sweep():
        for column_index in update_rng.permutation(rows):
            changed, changed_bits = full_column_gibbs_update(
                b_columns, a_syndromes, matrix, syndrome.reshape(rows, columns),
                int(column_index), log_mass, cache, workspace, update_rng,
            )
            counters[0] += 1
            if changed:
                counters[1] += 1
                counters[2] += changed_bits

    for _ in range(config.burn_sweeps):
        one_sweep()
    burn_b_columns = b_columns.copy()
    labels = np.empty(config.measurement_sweeps, dtype=np.uint64)
    weights = np.empty(config.measurement_sweeps, dtype=np.int32)
    b_trace = np.empty((config.measurement_sweeps, rows), dtype=np.uint32)
    blocks = np.empty(config.measurement_sweeps, dtype=np.int8)
    for sweep in range(config.measurement_sweeps):
        one_sweep()
        previous = state.copy()
        state, label, weight = _reference_sample_full_state(
            previous, b_columns, a_syndromes, observation_rng, columns, rows,
            section_masks, kernel_combinations, odds_powers, signatures,
        )
        counters[3] += 1
        residual = (
            model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
        ).astype(np.uint8)
        if not np.array_equal(residual, syndrome):
            raise FullColumnGibbsConflictError("full-column Gibbs left the hard coset")
        if _state_label(frame, state) != label or int(state.sum()) != int(weight):
            raise FullColumnGibbsConflictError("full-column cached state statistics drifted")
        labels[sweep] = label
        weights[sweep] = weight
        b_trace[sweep] = b_columns
        blocks[sweep] = min(7, 8 * sweep // config.measurement_sweeps)
    return {
        "initial_b_columns": np.asarray(_initial_collapsed_masks(initial_state, syndrome, matrix)[0], dtype=np.uint32),
        "burn_b_columns": burn_b_columns,
        "final_b_columns": b_columns.copy(),
        "measurement_b_columns": b_trace,
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_blocks": blocks,
        "counters": counters,
        "mass": mass,
        "cache_rows": cache.rows,
    }
